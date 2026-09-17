//! The versioned HTTP schema.
//!
//! Deliberately separate from the internal types (`RunConfig`, `TrainConfig`,
//! `MemoryReport`, …): those change at the engine's pace, this is a contract.
//! The conversion is explicit, even at the price of duplication.
//!
//! All types use `snake_case`, `deny_unknown_fields` on inputs, string enums, bytes as
//! `u64` on the way out, `None` omitted rather than `null`, and any map a
//! `BTreeMap` so the serialized bytes are deterministic.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Applies the OpenAPI derive only when the `openapi` feature is on, so nothing
/// in the logic depends on it.
macro_rules! schema {
    ($(#[$meta:meta])* $vis:vis struct $name:ident { $($body:tt)* }) => {
        $(#[$meta])*
        #[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
        $vis struct $name { $($body)* }
    };
}

schema! {
/// `GET /v1/health`
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Health {
    /// Always `"ok"`: the endpoint is liveness, so a body at all is the answer.
    pub status: &'static str,
    pub version: &'static str,
    /// Backends compiled into this build, e.g. `["cpu", "metal"]`.
    pub backends: Vec<String>,
}
}

schema! {
/// One ggml device visible to this build.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct DeviceInfo {
    /// `cpu` | `gpu` | `accel` | `other`, as the runtime classifies it.
    pub kind: String,
    pub name: String,
    pub description: String,
    /// Absent when the backend does not report a memory budget, which is not
    /// the same as reporting zero.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_bytes: Option<u64>,
}
}

/// The resolved budgets come straight from `retrograd-plan`: the wire shape and
/// the shape the cost model compares against must be the same numbers, and a
/// second copy of them here would be a second truth.
pub use retrograd_plan::budget::{Budget, Budgets};

schema! {
/// `GET /v1/capabilities`
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Capabilities {
    pub execution_profile_schema_version: u32,
    pub kernel_catalog_fingerprint: String,
    /// Effective process-wide mode: `off`, `observe`, `prefer`, or `require`.
    pub rir_mode: String,
    pub rir_policy_latched: bool,
    pub backends: Vec<String>,
    pub devices: Vec<DeviceInfo>,
    /// True when device and host memory are the same physical pool (Metal,
    /// iGPU). Two independent budgets would then double-count the model.
    pub unified_memory: bool,
    pub budgets: Budgets,
    pub features: Features,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Features {
    pub openapi: bool,
    pub max_concurrent_runs: usize,
}
}

schema! {
/// One field the resolver decides when `params` leaves it out, and the rule it
/// decides it with.
///
/// This is what replaced `GET /v1/presets`. A profile was a word the server had
/// to interpret; this is the interpretation itself, published - a client that
/// wants its own "fast" or "thorough" builds one out of these rules and sends it
/// as `params`.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DerivedField {
    /// Dotted field path, the same grammar as `params`, `provenance` keys and
    /// the `PATCH` whitelist. One path grammar for the whole API.
    pub path: &'static str,
    /// The rule, in one sentence.
    pub rule: &'static str,
    /// The numbers in that sentence, named. A `BTreeMap` so the bytes are
    /// deterministic.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub thresholds: BTreeMap<&'static str, f64>,
    /// `any` | `sft` | `grpo` | `ppo` | `rollout` | `evaluation` | `checkpoint`:
    /// when the rule applies at all.
    pub applies_to: &'static str,
    /// Always true, and stated rather than implied: every one of these is a
    /// `params` field a client may set instead.
    pub client_can_set: bool,
}
}

schema! {
/// One setting this server turns on even when the run would have fitted without
/// it.
///
/// The list is the *whole* of what invariant 4 now permits without an opt-in: a
/// degradation not named here is still a 422 that asks for `allow`.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ActiveDefault {
    pub id: &'static str,
    /// When it is applied, in one sentence.
    pub condition: &'static str,
    /// What the run pays for it.
    pub cost: &'static str,
    /// What it does.
    pub note: &'static str,
    /// The dotted paths it moves. Setting any of them in `params` pins it.
    pub paths: Vec<&'static str>,
    /// Whether leaving it on changes *what* is computed rather than how fast.
    pub degrades: bool,
    /// The `plan.warnings` code this default raises, when it raises one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<&'static str>,
    /// Always true: one `params` field turns it off.
    pub client_can_disable: bool,
}
}

schema! {
/// `GET /v1/defaults`
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Defaults {
    pub derived: Vec<DerivedField>,
    pub active_defaults: Vec<ActiveDefault>,
}
}

schema! {
/// A reward declared by the operator. The command itself never leaves the
/// server.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RewardEntry {
    pub id: String,
    pub description: String,
    pub timeout_seconds: u64,
    /// How the trainer speaks to it: one worker for the run (`"persistent"`),
    /// or one process per batch (`"oneshot"`). Listed because it changes
    /// nothing a client sends and everything a client's timings look like.
    ///
    /// The spelling of [`retrograd_core::RewardMode`] rather than the type: the
    /// vocabulary is the trainer's, and publishing it as its own schema would
    /// freeze a second copy of it here.
    pub mode: &'static str,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Rewards {
    pub rewards: Vec<RewardEntry>,
}
}

schema! {
/// A judge declared by the operator. `client_settings` is the explicit list of
/// what a recipe may set on it: choices of method, with no side effect outside
/// the process. Endpoint, model and key are absent by design.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct JudgeEntry {
    pub id: String,
    pub description: String,
    /// `ruler` | `command`.
    pub kind: &'static str,
    pub client_settings: Vec<&'static str>,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Judges {
    pub judges: Vec<JudgeEntry>,
}
}

schema! {
/// One tool as the policy will actually see it.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ToolEntry {
    pub name: String,
    pub description: String,
}
}

schema! {
/// A declared MCP server, with the tools it really exposes after the
/// allow/deny filters. The catalogue connects at startup, so this list is
/// verified rather than declarative.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct McpServerEntry {
    pub id: String,
    pub description: String,
    /// A required server that fails to connect stops the server from starting;
    /// an optional one is reported `skipped` and its tools are simply absent.
    pub required: bool,
    /// `connected` | `skipped`.
    pub status: &'static str,
    pub tools: Vec<ToolEntry>,
    pub warnings: Vec<String>,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct McpServers {
    pub mcp_servers: Vec<McpServerEntry>,
}
}

schema! {
/// A declared environment. No image, no limits, no mount: the operator chose
/// those, and publishing them would be one step from letting a client choose
/// them.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct EnvironmentEntry {
    pub id: String,
    pub description: String,
    /// `http` | `container` | `local`.
    pub kind: &'static str,
    /// Empty for an `http` environment, whose tool list belongs to its own
    /// server rather than to this catalogue.
    pub tools: Vec<String>,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Environments {
    pub environments: Vec<EnvironmentEntry>,
}
}

schema! {
/// `POST /v1/preflight`
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PreflightRequest {
    pub model: String,
    /// `auto` | `cpu` | `gpu`; defaults to `auto`.
    #[serde(default)]
    pub device: Option<String>,
    /// LoRA target shorthands or patterns, as the CLI's `--targets` accepts.
    /// Absent selects the architecture's automatic profile.
    #[serde(default)]
    pub targets: Option<Vec<String>>,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct PreflightResponse {
    /// Flattened so the HTTP response is the versioned engine contract itself,
    /// not an envelope carrying a second schema.
    #[serde(flatten)]
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub report: retrograd_core::PreflightReport,
}
}

schema! {
/// Geometry of a model, as the resolver reads it. Exposed because a client that
/// wants to reason about its own budget needs the same numbers.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ModelGeometry {
    pub architecture: String,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_ff: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub n_embd_k_gqa: u32,
    pub n_embd_v_gqa: u32,
    pub n_vocab: u32,
    pub n_ctx_train: u32,
    pub n_params: u64,
    pub weight_bytes: u64,
    pub file_bytes: u64,
    pub weight_dtype: String,
    pub tied_embeddings: bool,
    pub is_recurrent: bool,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub experts: BTreeMap<String, u32>,
}
}

impl From<&retrograd_core::ModelInfo> for ModelGeometry {
    fn from(info: &retrograd_core::ModelInfo) -> Self {
        let mut experts = BTreeMap::new();
        if info.n_expert > 0 {
            experts.insert("total".to_string(), info.n_expert);
            experts.insert("used".to_string(), info.n_expert_used);
        }
        Self {
            architecture: info.architecture.clone(),
            n_layer: info.n_layer,
            n_embd: info.n_embd,
            n_ff: info.n_ff,
            n_head: info.n_head,
            n_head_kv: info.n_head_kv,
            n_embd_k_gqa: info.n_embd_k_gqa,
            n_embd_v_gqa: info.n_embd_v_gqa,
            n_vocab: info.n_vocab,
            n_ctx_train: info.n_ctx_train,
            n_params: info.n_params,
            weight_bytes: info.model_size_bytes,
            file_bytes: info.file_size_bytes,
            weight_dtype: info.dominant_weight_type.clone(),
            tied_embeddings: info.tied_embeddings,
            is_recurrent: info.is_recurrent,
            experts,
        }
    }
}

// ---------------------------------------------------------------------------
// Planning (`POST /v1/plan`, `POST /v1/runs?dry_run=true`)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Runs (`POST /v1/runs`, `GET /v1/runs[/{id}]`)
// ---------------------------------------------------------------------------

retrograd_core::wire_enum! {
    /// The stable state vocabulary that clients may branch on. Runtime
    /// transitions are documented on
    /// [`crate::runtime::registry::transition_allowed`], which is also what
    /// enforces them.
    ///
    /// The spelling beside each variant is what serde writes and what `as_str()`
    /// returns.
    ///
    /// `rename_all` stays, and is not redundant: `utoipa` builds the schema's
    /// `enum` from the variant identifiers and the *container* rule only - it
    /// does not read the per-variant `rename` the macro emits. Without this
    /// line the OpenAPI document would publish `"Queued"` while the API sends
    /// `"queued"`, silently. `run_status_is_published_as_it_is_serialized`
    /// checks the two renderers against each other.
    #[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
    #[serde(rename_all = "snake_case")]
    #[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
    pub enum RunStatus: serde {
        Queued = "queued",
        Resolving = "resolving",
        Starting = "starting",
        Running = "running",
        Paused = "paused",
        Pausing = "pausing",
        Cancelling = "cancelling",
        Completed = "completed",
        Failed = "failed",
        Cancelled = "cancelled",
        /// Found `running` in the state directory when the server started.
        /// The V1 does not re-attach; a fork resumes the work.
        Interrupted = "interrupted",
    }
}

impl RunStatus {
    /// Whether the run is over. A terminal state never changes again, which is
    /// what makes it safe to answer from the journal without a live worker.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Interrupted
        )
    }
}

/// Where a run has got to. Every figure comes from an observer event, so a field
/// is absent until the run has actually reported it once - never zero standing
/// in for unknown.
///
/// Non-finite measurements are stored as `None`, never rendered: `serde_json`
/// writes a `NaN` as `null`, which would be the one place in the API where
/// absence and nullity disagree. And the floats stay `f32` - the width the
/// engine reports them in, because widening one to `f64` writes a learning
/// rate of `2e-4` as `0.00019999999494757503`.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RunProgress {
    /// Completed epochs (SFT) or updates (PPO/GRPO).
    pub iteration: u64,
    /// What the plan said the run would do, so a client can render a fraction
    /// without fetching the plan.
    pub iterations: u64,
    pub global_step: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub train_loss: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval_loss: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub learning_rate: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_per_second: Option<f32>,
}

/// A measurement that may be `NaN` (the observer's "not evaluated this epoch"),
/// as the wire schema wants it: absent.
pub fn finite(value: f32) -> Option<f32> {
    value.is_finite().then_some(value)
}

schema! {
/// One entry of `GET /v1/runs`. Deliberately small: a listing is polled, and a
/// full configuration per run would make it expensive to poll.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct RunSummary {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub status: RunStatus,
    /// Unix seconds. Not RFC 3339: every other time in this API is a number,
    /// and one format is easier to consume than two.
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<u64>,
    pub progress: RunProgress,
    /// Present on `failed` only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}
}

schema! {
#[derive(Clone, Debug, Serialize)]
pub struct RunListing {
    pub runs: Vec<RunSummary>,
    /// Opaque: pass it back as `?cursor=` for the next page. Absent on the last
    /// page, which is how a client knows to stop.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}
}

schema! {
/// `GET /v1/runs/{id}` and the `201` of `POST /v1/runs`.
///
/// The three heavy fields are carried as pre-rendered JSON. They are produced by
/// serializing the typed resolution once, at creation, and then written to the
/// journal and served verbatim - which is both cheaper than re-rendering and the
/// only way a run restored from disk answers the same bytes it answered while it
/// was alive.
#[derive(Clone, Debug, Serialize)]
pub struct RunView {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub status: RunStatus,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finished_at: Option<u64>,
    pub progress: RunProgress,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Weights, KV caches and optimizer state stay allocated while a run is
    /// paused. True whenever this run is holding the device.
    pub holds_device: bool,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub effective_config: Box<serde_json::value::RawValue>,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub provenance: Box<serde_json::value::RawValue>,
    #[cfg_attr(feature = "openapi", schema(value_type = Object))]
    pub plan: Box<serde_json::value::RawValue>,
}
}

/// One line of `events.jsonl`, and one SSE frame.
///
/// `seq` is monotonic per run and starts at 1, so `?since=0` replays everything
/// and a client can tell a gap from a duplicate.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RunEvent {
    pub seq: u64,
    /// Unix milliseconds.
    pub at: u64,
    #[serde(flatten)]
    pub payload: RunEventPayload,
}

/// The event types the API lists.
///
/// `Metrics` is one of them because the API lists `metrics` among the SSE types
/// *and* gives it a pull route, and neither can be served from an event that
/// does not exist.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum RunEventPayload {
    Status {
        status: RunStatus,
    },
    Progress(RunProgress),
    /// One emission of the metrics bus, the same values TensorBoard receives.
    /// A `BTreeMap` so the bytes are stable, and non-finite values are absent
    /// rather than `null`.
    Metrics {
        iteration: u64,
        global_step: u64,
        values: BTreeMap<String, f32>,
    },
    Evaluation {
        iteration: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        loss: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        perplexity: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        mean_reward: Option<f32>,
        improved: bool,
        best: f64,
        stale: u32,
        keep_training: bool,
    },
    Checkpoint {
        path: String,
    },
    Memory {
        note: String,
    },
    Log {
        message: String,
    },
    Terminal {
        status: RunStatus,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

impl RunEventPayload {
    /// The `type` discriminant, as it is serialized. Written out rather than
    /// derived so the SSE `event:` field and the JSON `"type"` are the same
    /// string by construction - a client filtering on one must match the other.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Status { .. } => "status",
            Self::Progress(_) => "progress",
            Self::Metrics { .. } => "metrics",
            Self::Evaluation { .. } => "evaluation",
            Self::Checkpoint { .. } => "checkpoint",
            Self::Memory { .. } => "memory",
            Self::Log { .. } => "log",
            Self::Terminal { .. } => "terminal",
        }
    }
}

// ---------------------------------------------------------------------------
// Control
// ---------------------------------------------------------------------------

/// When a cancellation takes effect.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum CancelAt {
    /// The default, and the only one that can checkpoint: the current epoch or
    /// update finishes first, so the run stops at a point it could resume from.
    #[default]
    Boundary,
    /// At the very next progress callback, between two boundaries.
    Now,
}

schema! {
/// `POST /v1/runs/{id}/cancel`
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct CancelRequest {
    pub at: CancelAt,
    /// Write a checkpoint before stopping. Only meaningful with
    /// `at: "boundary"`; asking for it with `at: "now"` is refused rather than
    /// silently dropped, because there is no resumable point between two
    /// boundaries and a client that thought it had a checkpoint would not.
    pub checkpoint: bool,
}
}

schema! {
/// `PATCH /v1/runs/{id}` - the whitelist of the contract, and nothing else.
///
/// `deny_unknown_fields` is what makes the list closed: `n_ctx`, `lora`,
/// `algorithm` and every path are rejected by the deserializer before a handler
/// runs, and the handler turns that into the 409 that points at `fork_from`.
/// Field names follow the *document* grammar (`training.lr`), the same one
/// `params`, `provenance` and `GET /v1/defaults` speak.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct PatchRequest {
    pub training: Option<TrainingPatch>,
    pub evaluation: Option<EvaluationPatch>,
    pub checkpoint: Option<CheckpointPatch>,
}
}

schema! {
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields, default)]
pub struct TrainingPatch {
    /// The base rate the schedule multiplies. Warm-up and decay keep their
    /// shape around it.
    pub lr: Option<f32>,
}
}

schema! {
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct EvaluationPatch {
    pub every_iterations: Option<u32>,
    pub patience: Option<u32>,
}
}

schema! {
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct CheckpointPatch {
    pub every_steps: Option<u64>,
    /// `steps` | `best_eval` | `steps_and_best_eval`, the same vocabulary the
    /// configuration document uses.
    pub mode: Option<String>,
}
}

schema! {
/// What every control route answers.
///
/// One shape for pause, resume, cancel, checkpoint and patch, because a client
/// scripting them wants one thing to read: what the run is now, and when what it
/// asked for lands.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CommandAccepted {
    pub id: String,
    pub status: RunStatus,
    /// The iteration the request takes effect at - the next one, since a run is
    /// only interruptible at its progress callback. Absent when the effect
    /// was immediate, as for a run cancelled before it ever started.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applies_at_iteration: Option<u64>,
    /// The dotted paths a `PATCH` accepted, in a fixed order. Empty elsewhere.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub applied: Vec<&'static str>,
}
}

// ---------------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------------

schema! {
/// `GET /v1/runs/{id}/metrics`
///
/// The pull half of the event stream: the same events, filtered to `metrics` and
/// served from the ring or `events.jsonl`. `next_since` is what to pass back to
/// continue, so polling is a loop with no bookkeeping on the client.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricsPage {
    pub metrics: Vec<MetricsSample>,
    pub next_since: u64,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct MetricsSample {
    pub seq: u64,
    /// Unix milliseconds.
    pub at: u64,
    pub iteration: u64,
    pub global_step: u64,
    pub values: BTreeMap<String, f32>,
}
}

// ---------------------------------------------------------------------------
// Inference against a live run
// ---------------------------------------------------------------------------

schema! {
/// `POST /v1/runs/{id}/evaluate` - one evaluation out of schedule.
///
/// The whole request, and it is empty: an ad-hoc evaluation is the *configured*
/// evaluation, run now. A dataset here would be a different measurement wearing
/// the same name, and comparing it with the scheduled curve would be wrong.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields, default)]
pub struct EvaluateRequest {}
}

schema! {
/// What an ad-hoc evaluation measured, and where the run was when it did.
///
/// This result is **not** part of the run's evaluation history: it does not move
/// `best`, does not count towards patience and never writes a `best` checkpoint.
/// A client that could feed the early-stopping state by asking questions could
/// end a run by polling it.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct EvaluationResult {
    pub id: String,
    /// The iteration the evaluation ran at, so it can be lined up against the
    /// metric stream.
    pub iteration: u64,
    pub global_step: u64,
    /// SFT only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub loss: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub perplexity: Option<f64>,
    /// PPO/GRPO only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_reward: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_min: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reward_max: Option<f32>,
    pub examples: u64,
}
}

schema! {
/// `POST /v1/runs/{id}/generate` - sample against the adapter as it is now.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct GenerateRequest {
    pub prompt: String,
    /// Wrap the prompt in the model's chat template as one user turn. On by
    /// default: an instruction-tuned adapter sampled without its template
    /// answers as a text completer, which reads as a broken adapter rather than
    /// as a missing flag.
    #[serde(default = "yes")]
    pub chat: bool,
    #[serde(default)]
    pub max_new_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    /// Sampling seed. Fixed by default, so two calls at the same step return the
    /// same text and a difference is a difference in the weights.
    #[serde(default)]
    pub seed: Option<u32>,
    /// Sample the same prompt again with the adapter disabled. Doubles the cost.
    #[serde(default)]
    pub include_base: bool,
}
}

fn yes() -> bool {
    true
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct GenerationResult {
    pub id: String,
    pub iteration: u64,
    pub global_step: u64,
    pub text: String,
    pub prompt_tokens: u64,
    pub tokens: u64,
    /// Present only when `include_base` asked for it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_text: Option<String>,
}
}

// ---------------------------------------------------------------------------
// Checkpoints and artefacts
// ---------------------------------------------------------------------------

schema! {
/// One checkpoint found on disk.
///
/// A directory read, not a run's memory: a run restored from the state directory
/// lists its checkpoints exactly like a live one, and a checkpoint deleted by
/// hand disappears from the listing without anything having to be told.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CheckpointEntry {
    /// `step-000000000400` or `best`, the checkpoint's own id.
    pub id: String,
    pub path: String,
    /// `step` | `best`.
    pub kind: &'static str,
    pub global_step: u64,
    /// Total size of the state directory.
    pub bytes: u64,
    /// Unix seconds of the manifest, i.e. when the checkpoint was published.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub written_at: Option<u64>,
    /// The adapter GGUF exported beside the state directory, when it is there.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// False for a directory whose manifest is missing or of another format
    /// version: it is listed rather than hidden, because a client that asked to
    /// resume from it deserves to know why it cannot.
    pub complete: bool,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct CheckpointListing {
    /// Absent when the run writes no checkpoints, which is not the same as an
    /// empty directory.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub directory: Option<String>,
    /// Newest first.
    pub checkpoints: Vec<CheckpointEntry>,
    /// The id a resume would pick, i.e. what `fork_from` resolves to by default.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latest: Option<String>,
}
}

schema! {
/// One artefact of a run, by a fixed name rather than by path.
///
/// The name is what `GET /v1/runs/{id}/artifacts/{name}` takes. Naming the
/// members of a closed inventory is what keeps the download route from being a
/// path parameter - and therefore from being an arbitrary file reader.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ArtifactEntry {
    /// `adapter` | `run` | `events` | `observe_log` | `observe` |
    /// `checkpoints` | `tensorboard` | `wandb_export`.
    pub name: &'static str,
    pub path: String,
    /// `file` | `directory`.
    pub kind: &'static str,
    /// Whether it is on disk right now. A run that has not finished has no
    /// adapter yet, and that is worth saying rather than omitting.
    pub present: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    /// Whether `GET.../artifacts/{name}` will serve the bytes. False for a
    /// directory, and for a file over the download limit.
    pub downloadable: bool,
}
}

schema! {
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ArtifactListing {
    pub artifacts: Vec<ArtifactEntry>,
}
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

schema! {
/// Length statistics as they ride along a dataset's own card - the flattened
/// view of `retrograd_plan::DatasetStats` a client reads without fetching a
/// plan.
#[derive(Clone, Copy, Debug, Serialize, PartialEq)]
pub struct DatasetStatsView {
    /// False when these came from the character heuristic rather than a real
    /// tokenizer - true once `POST /v1/datasets/{id}/tokenize` has run.
    pub measured: bool,
    pub total_tokens: u64,
    pub p50: u32,
    pub p90: u32,
    pub p99: u32,
    pub max: u32,
}
}

schema! {
/// `POST /v1/datasets` and `GET /v1/datasets/{id}`.
///
/// `id = "ds_" + sha256(content)`: two uploads of the same bytes answer the
/// same card, which is where the idempotence of `POST /v1/datasets` comes
/// from.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetView {
    pub id: String,
    pub sha256: String,
    /// `chat-jsonl` | `text`.
    pub format: String,
    pub bytes: u64,
    pub examples: u64,
    pub created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub stats: DatasetStatsView,
}
}

impl From<&crate::datasets::DatasetMeta> for DatasetView {
    fn from(meta: &crate::datasets::DatasetMeta) -> Self {
        Self {
            id: meta.id.clone(),
            sha256: meta.sha256.clone(),
            format: meta.format.clone(),
            bytes: meta.bytes,
            examples: meta.examples,
            created_at: meta.created_at,
            name: meta.name.clone(),
            stats: DatasetStatsView {
                measured: meta.stats.measured,
                total_tokens: meta.stats.total,
                p50: meta.stats.p50,
                p90: meta.stats.p90,
                p99: meta.stats.p99,
                max: meta.stats.max,
            },
        }
    }
}

schema! {
/// `GET /v1/datasets`
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetListing {
    pub datasets: Vec<DatasetView>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_cursor: Option<String>,
}
}

schema! {
/// `GET /v1/datasets/{id}/preview`
///
/// Examples exactly as they are read off disk: a `chat-jsonl` dataset answers
/// parsed JSON objects, a `text` dataset answers raw lines - never
/// re-formatted, since the point of a preview is to show what training would
/// actually see.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetPreview {
    pub id: String,
    pub format: String,
    #[cfg_attr(feature = "openapi", schema(value_type = Vec<Object>))]
    pub examples: Vec<serde_json::Value>,
}
}

schema! {
/// `POST /v1/datasets/{id}/tokenize`
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TokenizeRequest {
    /// The model whose tokenizer and chat template decide the lengths. Subject
    /// to the same `path_roots` check as `recipe.model`.
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub model: std::path::PathBuf,
}
}

schema! {
/// `POST /v1/datasets/{id}/tokenize` - the measured lengths, and what they mean
/// for a context you have not chosen yet.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct DatasetTokenization {
    pub id: String,
    /// The cache key these lengths were stored under: the vocabulary and chat
    /// template that produced them, not the file that carried them. Every model
    /// answering the same key reuses this measurement.
    pub tokenizer: String,
    /// `measured` is `true` here by construction - a real tokenizer ran.
    pub stats: DatasetStatsView,
    /// The share of examples that would be truncated at each of a few common
    /// contexts, keyed by the context as a decimal string.
    pub truncation: std::collections::BTreeMap<String, f64>,
}
}
