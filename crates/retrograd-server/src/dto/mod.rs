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

mod artifacts;
mod control;
mod datasets;
mod inference;
mod observation;
mod planning;
mod runs;

pub use artifacts::*;
pub use control::*;
pub use datasets::*;
pub use inference::*;
pub use observation::*;
pub use planning::*;
pub use runs::*;

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
