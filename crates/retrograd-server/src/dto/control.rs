//! Control

use super::*;

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
