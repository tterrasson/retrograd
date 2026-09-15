//! `GET /v1/openapi.json`.
//!
//! The document is assembled from two halves, and the split is on purpose.
//!
//! **The schemas are derived.** Every DTO already carries `utoipa::ToSchema`
//! behind the `openapi` feature, so the component section is generated from the
//! very types the handlers serialize. A hand-written schema would be a second
//! description of the wire format, and the failure mode of a second description
//! is that it is wrong six months later and nobody notices.
//!
//! **The paths are a table in this file.** The alternative is a
//! `#[utoipa::path]` attribute on every handler, which puts the route, its
//! parameters and its responses in three places (the router, the attribute,
//! the handler) instead of two. The table below sits next to nothing else and
//! is checked by a test that walks the router: a route the router serves and
//! the table omits fails the build's tests, which is the property that
//! matters.

use axum::Json;
use axum::response::IntoResponse;
use serde_json::{Value, json};

/// One query parameter of one operation.
struct QueryParam {
    name: &'static str,
    /// `"string"` | `"integer"` | `"boolean"`.
    kind: &'static str,
    description: &'static str,
}

const fn query(name: &'static str, kind: &'static str, description: &'static str) -> QueryParam {
    QueryParam {
        name,
        kind,
        description,
    }
}

/// One operation: method, path, what it does, and the schema of what it answers.
struct Operation {
    method: &'static str,
    path: &'static str,
    summary: &'static str,
    /// Component name of the request body, when the operation takes a JSON one.
    request: Option<&'static str>,
    /// Media types of a body that is *not* JSON - a dataset upload, whose body
    /// is the file itself and therefore has no component to point at.
    raw_request: &'static [&'static str],
    query: &'static [QueryParam],
    /// Component name of the success response, or `None` for `204`.
    response: Option<&'static str>,
    /// Answers `201` on success. With [`Operation::or_ok`], `200` as well.
    created: bool,
    /// What a `200` means for an operation that normally answers `201`,
    /// nothing was made, because it already existed.
    repeat: Option<&'static str>,
}

const fn op(
    method: &'static str,
    path: &'static str,
    summary: &'static str,
    request: Option<&'static str>,
    response: Option<&'static str>,
) -> Operation {
    Operation {
        method,
        path,
        summary,
        request,
        raw_request: &[],
        query: &[],
        response,
        created: false,
        repeat: None,
    }
}

impl Operation {
    /// The operation answers `201`.
    const fn created(mut self) -> Self {
        self.created = true;
        self
    }

    /// …and `200` when there was nothing new to create.
    const fn or_ok(mut self, description: &'static str) -> Self {
        self.repeat = Some(description);
        self
    }

    const fn accepts(mut self, media_types: &'static [&'static str]) -> Self {
        self.raw_request = media_types;
        self
    }

    const fn with_query(mut self, params: &'static [QueryParam]) -> Self {
        self.query = params;
        self
    }
}

/// Every operation of the V1 surface, in the order the contract introduces them.
const OPERATIONS: &[Operation] = &[
    op(
        "get",
        "/v1/health",
        "Liveness, version and compiled backends",
        None,
        Some("Health"),
    ),
    op(
        "get",
        "/v1/capabilities",
        "Devices, memory and effective budgets",
        None,
        Some("Capabilities"),
    ),
    op(
        "get",
        "/v1/defaults",
        "Derivation rules and the settings applied by default",
        None,
        Some("Defaults"),
    ),
    op(
        "get",
        "/v1/rewards",
        "Rewards the operator declared",
        None,
        Some("Rewards"),
    ),
    op(
        "get",
        "/v1/judges",
        "Judges the operator declared",
        None,
        Some("Judges"),
    ),
    op(
        "get",
        "/v1/mcp-servers",
        "MCP servers and the tools they really expose",
        None,
        Some("McpServers"),
    ),
    op(
        "get",
        "/v1/environments",
        "Environments the operator declared",
        None,
        Some("Environments"),
    ),
    op("get", "/v1/openapi.json", "This document", None, None),
    op(
        "post",
        "/v1/preflight",
        "Build the training graph and report on it",
        Some("PreflightRequest"),
        Some("PreflightResponse"),
    ),
    op(
        "post",
        "/v1/plan",
        "Resolve a recipe without creating anything",
        Some("PlanRequest"),
        Some("PlanResponse"),
    ),
    op(
        "post",
        "/v1/runs",
        "Create a run",
        Some("PlanRequest"),
        Some("RunView"),
    )
    .created()
    .or_ok("a dry run, or a repeat of an Idempotency-Key: no run was started"),
    op(
        "get",
        "/v1/runs",
        "List runs, newest first",
        None,
        Some("RunListing"),
    ),
    op(
        "get",
        "/v1/runs/{id}",
        "One run: state, progress, effective config, plan",
        None,
        Some("RunView"),
    ),
    op(
        "patch",
        "/v1/runs/{id}",
        "Adjust a schedule on a live run",
        Some("PatchRequest"),
        Some("CommandAccepted"),
    ),
    op(
        "delete",
        "/v1/runs/{id}",
        "Forget a finished run",
        None,
        None,
    ),
    op(
        "post",
        "/v1/runs/{id}/pause",
        "Pause at the next progress callback",
        None,
        Some("CommandAccepted"),
    ),
    op(
        "post",
        "/v1/runs/{id}/resume",
        "Resume a paused run",
        None,
        Some("CommandAccepted"),
    ),
    op(
        "post",
        "/v1/runs/{id}/cancel",
        "Stop at a boundary, or now",
        Some("CancelRequest"),
        Some("CommandAccepted"),
    ),
    op(
        "post",
        "/v1/runs/{id}/checkpoints",
        "Ask for a checkpoint at the next boundary",
        None,
        Some("CommandAccepted"),
    ),
    op(
        "get",
        "/v1/runs/{id}/checkpoints",
        "Checkpoints on disk",
        None,
        Some("CheckpointListing"),
    ),
    op(
        "post",
        "/v1/runs/{id}/evaluate",
        "Run the configured evaluation now",
        Some("EvaluateRequest"),
        Some("EvaluationResult"),
    ),
    op(
        "post",
        "/v1/runs/{id}/generate",
        "Sample against the live adapter",
        Some("GenerateRequest"),
        Some("GenerationResult"),
    ),
    op(
        "get",
        "/v1/runs/{id}/artifacts",
        "What this run wrote",
        None,
        Some("ArtifactListing"),
    ),
    op(
        "get",
        "/v1/runs/{id}/artifacts/{name}",
        "Download one artifact",
        None,
        None,
    ),
    op(
        "get",
        "/v1/runs/{id}/metrics",
        "Metric samples, pull form",
        None,
        Some("MetricsPage"),
    ),
    op(
        "get",
        "/v1/runs/{id}/events",
        "Server-sent events for one run",
        None,
        None,
    ),
    op(
        "get",
        "/v1/events",
        "Server-sent events across every run",
        None,
        None,
    ),
    op(
        "post",
        "/v1/datasets",
        "Upload a dataset, content-addressed",
        None,
        Some("DatasetView"),
    )
    .created()
    .or_ok("the same bytes were already stored; nothing was written twice")
    .accepts(&["application/x-ndjson", "text/plain", "application/json"])
    .with_query(&[
        query("name", "string", "A label for the dataset's own card."),
        query(
            "format",
            "string",
            "auto (default) | text | jsonl. Absent or auto determines the \
             format from the content.",
        ),
    ]),
    op(
        "get",
        "/v1/datasets",
        "List stored datasets, newest first",
        None,
        Some("DatasetListing"),
    )
    .with_query(&[
        query("limit", "integer", "Page size, 1 to 500. Default 50."),
        query(
            "cursor",
            "string",
            "The id of the last item of the previous page.",
        ),
    ]),
    op(
        "get",
        "/v1/datasets/{id}",
        "One dataset's card",
        None,
        Some("DatasetView"),
    ),
    op(
        "get",
        "/v1/datasets/{id}/preview",
        "The first examples, exactly as read",
        None,
        Some("DatasetPreview"),
    )
    .with_query(&[query(
        "limit",
        "integer",
        "How many examples to show, 1 to 100. Default 5.",
    )]),
    op(
        "post",
        "/v1/datasets/{id}/tokenize",
        "Measure real token lengths against a model's tokenizer",
        Some("TokenizeRequest"),
        Some("DatasetTokenization"),
    ),
    op(
        "delete",
        "/v1/datasets/{id}",
        "Forget a dataset",
        None,
        None,
    ),
];

/// The paths object, built from [`OPERATIONS`].
fn paths() -> Value {
    let mut paths = serde_json::Map::new();
    for operation in OPERATIONS {
        let entry = paths
            .entry(operation.path.to_string())
            .or_insert_with(|| json!({}));
        let mut item = json!({
            "summary": operation.summary,
            "operationId": operation_id(operation),
            "responses": responses(operation),
        });
        let parameters = parameters(operation);
        if !parameters.is_empty() {
            item["parameters"] = Value::Array(parameters);
        }
        if let Some(request) = operation.request {
            item["requestBody"] = json!({
                "required": true,
                "content": {"application/json": {"schema": reference(request)}}
            });
        } else if !operation.raw_request.is_empty() {
            // The body *is* the dataset, so there is no component to point at:
            // it is bytes, and the accepted media types are the whole contract.
            let content: serde_json::Map<String, Value> = operation
                .raw_request
                .iter()
                .map(|media| {
                    (
                        (*media).to_string(),
                        json!({"schema": {"type": "string", "format": "binary"}}),
                    )
                })
                .collect();
            item["requestBody"] = json!({"required": true, "content": content});
        }
        entry[operation.method] = item;
    }
    Value::Object(paths)
}

fn operation_id(operation: &Operation) -> String {
    let tail: String = operation
        .path
        .trim_start_matches("/v1/")
        .replace(['{', '}'], "")
        .replace(['/', '.', '-'], "_");
    format!("{}_{tail}", operation.method)
}

/// Path parameters come from the template - a `{id}` in the route is a
/// parameter by construction, so deriving them cannot fall out of step with the
/// router. Query parameters are optional by definition here (every route that
/// takes one has a default) and are declared in the table.
fn parameters(operation: &Operation) -> Vec<Value> {
    operation
        .path
        .split('/')
        .filter_map(|segment| segment.strip_prefix('{')?.strip_suffix('}'))
        .map(|name| {
            json!({
                "name": name,
                "in": "path",
                "required": true,
                "schema": {"type": "string"}
            })
        })
        .chain(operation.query.iter().map(|parameter| {
            json!({
                "name": parameter.name,
                "in": "query",
                "required": false,
                "description": parameter.description,
                "schema": {"type": parameter.kind}
            })
        }))
        .collect()
}

fn responses(operation: &Operation) -> Value {
    let mut success = match (operation.method, operation.response) {
        ("delete", _) => json!({"204": {"description": "deleted"}}),
        (_, Some(schema)) if operation.created => {
            json!({
                "201": {
                    "description": "created",
                    "content": {"application/json": {"schema": reference(schema)}}
                }
            })
        }
        (_, Some(schema)) => json!({
            "200": {
                "description": "ok",
                "content": {"application/json": {"schema": reference(schema)}}
            }
        }),
        (_, None) => json!({"200": {"description": "ok"}}),
    };
    // An idempotent creation answers `200` when it made nothing: the same body,
    // a different meaning, and a client that only knows about `201` treats a
    // no-op as a failure.
    if let (Some(description), Some(schema)) = (operation.repeat, operation.response) {
        success["200"] = json!({
            "description": description,
            "content": {"application/json": {"schema": reference(schema)}}
        });
    }
    let mut responses = success;
    // Every failure of the API is one shape, so it is described once and
    // attached to every operation rather than enumerated per route.
    responses["default"] = json!({
        "description": "a problem document (RFC 9457)",
        "content": {"application/problem+json": {"schema": {"type": "object", "properties": {
            "type": {"type": "string"},
            "title": {"type": "string"},
            "status": {"type": "integer"},
            "detail": {"type": "string"},
            "errors": {"type": "array", "items": {"type": "object", "properties": {
                "pointer": {"type": "string"},
                "message": {"type": "string"}
            }}}
        }}}}
    });
    responses
}

fn reference(component: &str) -> Value {
    json!({"$ref": format!("#/components/schemas/{component}")})
}

fn info() -> Value {
    json!({
        "title": "retrograd-server",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "HTTP control plane for LoRA training runs."
    })
}

/// `GET /v1/openapi.json`
pub async fn document() -> axum::response::Response {
    Json(render()).into_response()
}

#[cfg(feature = "openapi")]
pub fn render() -> Value {
    use utoipa::OpenApi;

    #[derive(OpenApi)]
    #[openapi(components(schemas(
        crate::dto::Health,
        crate::dto::DeviceInfo,
        crate::dto::Capabilities,
        crate::dto::Features,
        crate::dto::DerivedField,
        crate::dto::ActiveDefault,
        crate::dto::Defaults,
        crate::dto::RewardEntry,
        crate::dto::Rewards,
        crate::dto::JudgeEntry,
        crate::dto::Judges,
        crate::dto::ToolEntry,
        crate::dto::McpServerEntry,
        crate::dto::McpServers,
        crate::dto::EnvironmentEntry,
        crate::dto::Environments,
        crate::dto::PreflightRequest,
        crate::dto::PreflightResponse,
        crate::dto::ModelGeometry,
        crate::dto::ForkFrom,
        crate::dto::PlanRequest,
        crate::dto::PlanResponse,
        crate::dto::RunStatus,
        crate::dto::RunProgress,
        crate::dto::RunSummary,
        crate::dto::RunListing,
        crate::dto::RunView,
        crate::dto::RunEvent,
        crate::dto::RunEventPayload,
        crate::dto::CancelAt,
        crate::dto::CancelRequest,
        crate::dto::PatchRequest,
        crate::dto::TrainingPatch,
        crate::dto::EvaluationPatch,
        crate::dto::CheckpointPatch,
        crate::dto::CommandAccepted,
        crate::dto::MetricsPage,
        crate::dto::MetricsSample,
        crate::dto::EvaluateRequest,
        crate::dto::EvaluationResult,
        crate::dto::GenerateRequest,
        crate::dto::GenerationResult,
        crate::dto::CheckpointEntry,
        crate::dto::CheckpointListing,
        crate::dto::ArtifactEntry,
        crate::dto::ArtifactListing,
        crate::dto::DatasetStatsView,
        crate::dto::DatasetView,
        crate::dto::DatasetListing,
        crate::dto::DatasetPreview,
        crate::dto::TokenizeRequest,
        crate::dto::DatasetTokenization,
    )))]
    struct ApiDoc;

    let mut document =
        serde_json::to_value(ApiDoc::openapi()).unwrap_or_else(|_| json!({"openapi": "3.1.0"}));
    document["openapi"] = json!("3.1.0");
    document["info"] = info();
    document["paths"] = paths();
    // Described unconditionally. Whether *this* deployment requires a token is a
    // fact about the deployment, not about the contract, and a document that
    // changed with it would make two servers of the same version publish
    // different APIs.
    document["components"]["securitySchemes"] = json!({
        "bearer": {"type": "http", "scheme": "bearer"}
    });
    document
}

/// Without the feature there are no derived schemas, so the document is the route
/// table alone. Still useful - a client can see the surface - and honest about
/// what is missing rather than pretending to a components section.
#[cfg(not(feature = "openapi"))]
pub fn render() -> Value {
    json!({
        "openapi": "3.1.0",
        "info": info(),
        "paths": paths(),
        "components": {
            "schemas": {},
            "securitySchemes": {"bearer": {"type": "http", "scheme": "bearer"}}
        },
        "x-retrograd-note":
            "built without the `openapi` feature, so component schemas are absent"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_operation_has_a_unique_id_and_declares_its_path_parameters() {
        let mut ids = std::collections::BTreeSet::new();
        for operation in OPERATIONS {
            let id = operation_id(operation);
            assert!(ids.insert(id.clone()), "duplicate operationId {id}");
            let parameters = parameters(operation);
            let declared = parameters
                .iter()
                .filter(|parameter| parameter["in"] == "path")
                .count();
            let braces = operation.path.matches('{').count();
            assert_eq!(
                declared, braces,
                "{} declares {declared} of {braces} path parameters",
                operation.path
            );
            // A query parameter is never required - every route that takes one
            // has a default - so a client that sends none must still be valid.
            assert!(
                parameters
                    .iter()
                    .filter(|parameter| parameter["in"] == "query")
                    .all(|parameter| parameter["required"] == false),
                "{} marks a query parameter as required",
                operation.path
            );
        }
    }

    #[test]
    fn the_document_is_a_complete_openapi_object() {
        let document = render();
        assert_eq!(document["openapi"], "3.1.0");
        assert!(document["paths"]["/v1/runs"]["post"]["requestBody"].is_object());
        assert_eq!(
            document["paths"]["/v1/runs"]["post"]["responses"]["201"]["content"]["application/json"]
                ["schema"]["$ref"],
            "#/components/schemas/RunView"
        );
        // The two methods on one path do not overwrite each other.
        assert!(document["paths"]["/v1/runs/{id}/checkpoints"]["get"].is_object());
        assert!(document["paths"]["/v1/runs/{id}/checkpoints"]["post"].is_object());
        assert!(document["paths"]["/v1/runs/{id}"]["delete"]["responses"]["204"].is_object());
        // Every operation carries the one failure shape.
        assert!(document["paths"]["/v1/health"]["get"]["responses"]["default"].is_object());
    }
}
