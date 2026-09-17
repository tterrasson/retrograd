//! Runs (`POST /v1/runs`, `GET /v1/runs[/{id}]`)

use super::*;

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
