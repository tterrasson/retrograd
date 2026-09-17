//! Inference against a live run

use super::*;

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
