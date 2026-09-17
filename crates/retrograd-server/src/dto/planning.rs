//! Planning (`POST /v1/plan`, `POST /v1/runs?dry_run=true`)

use super::*;

/// The resolver's own wire types are `retrograd-plan`'s: the recipe a client
/// posts is the recipe the resolver consumes, and the plan it renders is what
/// the resolver produced. Re-exported rather than mirrored - a second copy would
/// have to be kept in step by hand, which is the failure mode the contract warns about,
/// only worse for being invisible.
pub use retrograd_plan::provenance::Provenance;
pub use retrograd_plan::recipe::Recipe;
pub use retrograd_plan::resolver::PlanSummary;

schema! {
/// `POST /v1/plan` and `POST /v1/runs`.
///
/// Exactly one of `recipe` (form (a): an intention) or `config` (form (b): a
/// complete configuration, the same schema as the CLI's TOML). Form (c) - a raw
/// TOML body - arrives as `config` too, parsed by the extractor.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlanRequest {
    #[serde(default)]
    pub recipe: Option<Recipe>,
    /// A whole configuration document. Skips the semantic phases but not the
    /// budget check.
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub config: Option<retrograd_config::ConfigDocument>,
    /// The client's own parameters: a partial document tree, spelled exactly as
    /// the CLI's TOML spells it. Every leaf it sets is **locked** - no phase may
    /// re-derive it - and everything it leaves out the server derives, from the
    /// rules `GET /v1/defaults` publishes.
    #[serde(default)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub params: Option<serde_json::Value>,
    /// Human label. No uniqueness is imposed; identity is the run id.
    #[serde(default)]
    pub name: Option<String>,
    /// Continue from a checkpoint. With `fork_from.run` this is the whole
    /// body: the parent supplies the configuration and `params` go on top.
    #[serde(default)]
    pub fork_from: Option<ForkFrom>,
}
}

schema! {
/// Where a fork starts from.
///
/// Two shapes, and the difference is who supplies the configuration. `run` names
/// a run this server knows, so the parent's own effective configuration is
/// reused - that is the fork, and the resume. `path` names a checkpoint
/// directory, which carries a manifest and no configuration at all, so a `recipe`
/// or a `config` has to come with it.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForkFrom {
    /// The parent run's id.
    #[serde(default)]
    pub run: Option<String>,
    /// A checkpoint id under the parent's checkpoint directory (`step-400`,
    /// `best`), or absent for the most recent one - the same resolution
    /// `--resume` performs.
    #[serde(default)]
    pub checkpoint: Option<String>,
    /// A `.state` directory, for a checkpoint written by something other than
    /// this server.
    #[serde(default)]
    pub path: Option<String>,
}
}

schema! {
/// What a resolution renders back.
#[derive(Clone, Debug, Serialize)]
pub struct PlanResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The effective configuration, in the same schema `config` takes on input.
    /// Server-declared values are redacted to their catalogue id.
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub effective_config: retrograd_config::ConfigDocument,
    /// Flat map, keyed by dotted field path.
    pub provenance: Provenance,
    pub plan: PlanSummary,
}
}

impl PlanRequest {
    /// An absent tree normalizes to an empty object rather than a `null`: `null`
    /// as a *value* is refused further down.
    pub fn parameters(&self) -> serde_json::Value {
        let normalize = |tree: &serde_json::Value| match tree {
            serde_json::Value::Null => serde_json::json!({}),
            value => value.clone(),
        };
        self.params
            .as_ref()
            .map(normalize)
            .unwrap_or_else(|| serde_json::json!({}))
    }

    /// Exactly one source of a base configuration, and no `null` masquerading as
    /// a value.
    pub fn form(&self) -> Result<PlanForm<'_>, &'static str> {
        match (&self.recipe, &self.config, &self.fork_from) {
            (Some(_), Some(_), _) => Err("send either a recipe or a config, not both"),
            (Some(recipe), None, _) => Ok(PlanForm::Recipe(recipe)),
            (None, Some(config), _) => Ok(PlanForm::Config(config)),
            // A fork carrying neither: its parent's configuration is the base,
            // which is what makes `{"fork_from": {"run": "…"}}` a complete body.
            (None, None, Some(fork)) => Ok(PlanForm::Fork(fork)),
            // `params` alone is not a request. It says how to train, never what:
            // there is no model, no dataset and no algorithm in it. A TOML body
            // that turned out to be partial lands here, which is why the message
            // names that case rather than only listing the three forms.
            (None, None, None) if self.params.is_some() => Err(
                "params say how to train, not what: send them alongside a recipe, a \
                 config or a fork_from",
            ),
            (None, None, None) => Err("a recipe, a config or a fork_from is required"),
        }
    }
}

pub enum PlanForm<'a> {
    Recipe(&'a Recipe),
    Config(&'a retrograd_config::ConfigDocument),
    Fork(&'a ForkFrom),
}
