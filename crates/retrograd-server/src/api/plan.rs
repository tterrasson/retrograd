//! `POST /v1/plan`, and the body every planning route shares.
//!
//! One body, three forms: a recipe, a whole configuration, or a raw TOML
//! document. All three go through the same guard, the same resolver and the same
//! budget check. `/v1/plan` never creates anything; `POST /v1/runs` uses the same
//! resolution and then hands it to the runtime (`super::runs`).

use axum::Json;
use axum::extract::{Query, State};
use http::header;
use serde::Deserialize;
use serde_json::Value;

use crate::dto::{self, PlanForm};
use crate::error::{ApiError, ApiResult, ErrorCode};
use crate::guard;
use crate::resolve;
use crate::state::AppState;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanQuery {
    /// Accept a configuration whose estimate is over budget. Only meaningful for
    /// the forms that skip the resolver's phases: the caller chose the numbers,
    /// so the caller may take responsibility for them.
    #[serde(default)]
    pub force: bool,
    /// Measure the candidate instead of only estimating it.
    ///
    /// Off by default here and on by default for `POST /v1/runs`: a plan is
    /// meant to be a fast, side-effect-free question, and measuring loads the
    /// model onto the device behind the run queue.
    #[serde(default)]
    pub calibrate: Option<bool>,
}

pub async fn plan(
    State(state): State<AppState>,
    Query(query): Query<PlanQuery>,
    body: PlanBody,
) -> ApiResult<Json<dto::PlanResponse>> {
    let resolved =
        resolve_body(&state, body, query.force, query.calibrate.unwrap_or(false)).await?;
    Ok(Json(resolved.response))
}

pub(crate) async fn resolve_body(
    state: &AppState,
    body: PlanBody,
    force: bool,
    calibrate: bool,
) -> ApiResult<resolve::Resolved> {
    let request = body.0;
    let mut params = request.parameters();
    let name = request.name.clone();

    // A fork resolves to one thing - a resume path - and enters as an override.
    // Everything downstream then treats it as the client having pinned
    // `checkpoint.resume_from`, which is exactly what it is.
    let target = match &request.fork_from {
        Some(fork) => {
            if params
                .pointer("/checkpoint/resume_from")
                .is_some_and(|value| !value.is_null())
            {
                return Err(ApiError::invalid(
                    "fork_from and checkpoint.resume_from say the same thing; send one",
                )
                .with_field(
                    "/params/checkpoint/resume_from",
                    ErrorCode::OverrideConflict,
                    "implied by fork_from",
                ));
            }
            let target = resolve::fork::target(state, fork).await?;
            retrograd_plan::merge::deep_merge(&mut params, &target.as_override()).map_err(
                |error| {
                    ApiError::invalid(error.message).with_field(
                        "/fork_from",
                        ErrorCode::OverrideConflict,
                        "cannot be applied over these params",
                    )
                },
            )?;
            Some(target)
        }
        None => None,
    };

    let resolved = match request.form().map_err(ApiError::invalid)? {
        PlanForm::Recipe(recipe) => {
            resolve::plan_recipe(state, recipe, &params, name, calibrate).await
        }
        PlanForm::Config(document) => {
            let reward_id = target
                .as_ref()
                .and_then(|target| target.reward_id.as_deref());
            resolve::plan_config(state, document, &params, name, force, calibrate, reward_id).await
        }
        // `{"fork_from": {"run": "…"}}` and nothing else: the parent's own
        // configuration is the base, so this is form (b) with a document the
        // server supplied rather than the client.
        PlanForm::Fork(_) => {
            let target = target
                .as_ref()
                .expect("a fork form implies a resolved fork target");
            let document = target.document.clone().ok_or_else(|| {
                ApiError::invalid(
                    "a checkpoint path carries a manifest and no configuration; send \
                     fork_from.run to reuse a parent's, or a recipe or config alongside \
                     fork_from.path",
                )
                .with_field(
                    "/fork_from/path",
                    ErrorCode::MissingField,
                    "needs a configuration",
                )
            })?;
            resolve::plan_config(
                state,
                &document,
                &params,
                name,
                force,
                calibrate,
                target.reward_id.as_deref(),
            )
            .await
        }
    }?;

    // After the resolution, because what has to be compared is the configuration
    // that would actually run - the parent's, or the recipe's, with every
    // parameter already merged in.
    if let Some(target) = &target {
        resolve::fork::check_compatible(target, &resolved.config, &params)?;
    }
    Ok(resolved)
}

/// The request body, in whichever of the three forms arrived.
///
/// Extracting it by hand rather than with `Json<PlanRequest>` for two reasons:
/// the guard must see the raw tree before serde drops anything into a typed
/// field, and form (c) is a TOML document, which serde's JSON extractor
/// would reject with the wrong error.
pub struct PlanBody(pub dto::PlanRequest);

impl<S> axum::extract::FromRequest<S> for PlanBody
where
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(
        request: axum::extract::Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        let is_toml = request
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .is_some_and(|value| {
                value.starts_with("application/toml") || value.starts_with("text/toml")
            });

        let bytes = axum::body::Bytes::from_request(request, state)
            .await
            .map_err(|rejection| {
                crate::problem_from_rejection(rejection.status(), rejection.body_text())
            })?;

        let tree: Value = if is_toml {
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| ApiError::invalid("the TOML body is not valid UTF-8"))?;
            toml_body(text)?
        } else {
            serde_json::from_slice(&bytes)
                .map_err(|error| ApiError::invalid(format!("invalid JSON body: {error}")))?
        };

        // The contract, at any depth, before anything is typed: a client never supplies a
        // command, an endpoint or a credential.
        guard::reject_server_declared(&tree, "")?;

        let request: dto::PlanRequest = crate::extract::from_json_value(tree)?;
        Ok(Self(request))
    }
}

/// Fields a *complete* configuration cannot omit. They are what tells form (c)
/// apart from a `params` tree posted as TOML.
///
/// Distinguishing the two by content rather than by a second content type or a
/// second route is a deliberate trade: a client that already has
/// `examples/smoke_tiny_grpo.toml` can post it whole *or* amputated to the same
/// endpoint and get the obvious answer either way. The cost is that a partial
/// document which happens to carry both of these is read as a complete one - and
/// then refused by `config::build`, naming the field it is missing, which is the
/// same answer form (c) would have given.
const REQUIRED_BY_A_COMPLETE_DOCUMENT: [&str; 2] = ["/run/algorithm", "/model/path"];

/// Turns a TOML body into the JSON request the rest of the pipeline handles.
///
/// A complete document is form (c) and goes to `config`, parsed by the very
/// function `config::load` uses so the CLI and the API cannot disagree about what
/// a document means. A partial one goes to `params`, deep-merged over what the
/// resolver derives - which is what lets a client send the half of a recipe it
/// has an opinion about.
fn toml_body(text: &str) -> Result<Value, ApiError> {
    let tree: Value = toml::from_str(text)
        .map_err(|error| ApiError::invalid(format!("invalid TOML body: {error}")))?;
    let is_complete = REQUIRED_BY_A_COMPLETE_DOCUMENT
        .iter()
        .all(|pointer| tree.pointer(pointer).is_some());
    if !is_complete {
        return Ok(serde_json::json!({"params": tree}));
    }
    let document = retrograd_config::parse_toml(text, "request body")?;
    Ok(serde_json::json!({
        "config": serde_json::to_value(&document).map_err(unrenderable_document)?
    }))
}

/// A document `parse_toml` accepted and `serde_json` cannot render is a server
/// defect, not something the client sent: a 500, never a 422.
fn unrenderable_document(error: serde_json::Error) -> ApiError {
    ApiError::internal(format!(
        "the parsed document could not be rendered: {error}"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unrenderable_document_is_a_server_defect() {
        let error = serde_json::from_str::<Value>("{").expect_err("truncated JSON");
        let problem = unrenderable_document(error);
        assert!(matches!(problem.kind, crate::error::ProblemKind::Internal));
        assert!(
            problem
                .detail
                .starts_with("the parsed document could not be rendered: ")
        );
    }

    #[test]
    fn a_partial_toml_body_becomes_params_and_a_complete_one_becomes_a_config() {
        let partial = toml_body("[lora]\nrank = 8\n[training]\nctx = 1024\n").expect("parses");
        assert_eq!(partial["params"]["lora"]["rank"], 8);
        assert_eq!(partial["params"]["training"]["ctx"], 1024);
        assert!(partial.get("config").is_none());

        let complete = toml_body(
            r#"
[run]
algorithm = "sft"
[model]
path = "m.gguf"
[lora]
output = "a.gguf"
[sft]
data = "d.jsonl"
"#,
        )
        .expect("parses");
        assert!(complete.get("params").is_none());
        assert_eq!(complete["config"]["run"]["algorithm"], "sft");
    }

    /// A body naming an algorithm but no model is *not* a complete document, so
    /// it is read as params - where an unknown `run` section is then refused by
    /// the schema rather than accepted as a configuration missing its model.
    #[test]
    fn half_of_the_required_fields_is_not_a_complete_document() {
        let tree = toml_body("[run]\nalgorithm = \"sft\"\n").expect("parses");
        assert!(tree["params"]["run"]["algorithm"] == "sft");
    }

    #[test]
    fn a_toml_body_that_does_not_parse_is_a_422_and_not_a_panic() {
        let error = toml_body("[training\nctx = 1").expect_err("must be refused");
        assert!(
            error.detail.contains("invalid TOML body"),
            "{}",
            error.detail
        );
    }
}
