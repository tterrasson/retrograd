//! Algorithm boundary. Each algorithm owns its dataset and objective while the
//! binary owns model/LoRA lifecycle, logging, and output persistence.

pub mod batch;
pub mod distill;
pub mod features;
pub mod grpo;
pub mod ppo;
mod rollout;
pub mod sft;
pub mod value;

/// The version token a persistent reward command answers at startup. Published
/// here because it is part of the `reward_command` contract this crate defines,
/// and a test or an embedder that spells it by hand would drift from it.
pub use retrograd_judge::REWARD_PROTOCOL_VERSION;

use retrograd_core::TrainMetrics;
use retrograd_metrics::MetricValue;

/// A token sequence a teacher-forced pass can score, and the mask saying which
/// of its positions carry a policy action.
///
/// A sampled rollout and a pre-generated [`batch::TrainSequence`] are the same
/// object to a scorer, and the distillation teacher scores both: the loop feeds
/// it rollouts, a caller that brought its own trajectories feeds it sequences.
/// Only the two slices are needed - a scorer has no use for a reward, a group
/// id, or the behaviour logprobs.
pub trait TokenSpan {
    fn tokens(&self) -> &[i32];
    /// Same length as [`TokenSpan::tokens`]; the first entry is never true.
    fn train_mask(&self) -> &[bool];
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RewardEvalMetrics {
    pub mean_reward: f32,
    pub reward_min: f32,
    pub reward_max: f32,
    pub examples: usize,
    /// Share of evaluated prompts whose generation budget had to be cut to fit
    /// the trained window. Training refuses such a prompt outright; evaluation
    /// keeps it, so this is what says the mean reward mixes completions scored
    /// under different budgets.
    pub budget_clamped_fraction: f32,
}

/// A point at which the run can be checkpointed and later resumed exactly: the
/// end of an SFT epoch or of a PPO/GRPO update. Mid-boundary progress events
/// carry `None`, because resuming from one would replay or skip work.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Boundary {
    /// Epochs (SFT) or updates (PPO/GRPO) fully completed, i.e. the index of
    /// the next one to run.
    pub completed_iterations: u64,
    /// Dataset cursor for the next draw. GRPO's cursor advances by more than
    /// one group per update when dynamic sampling resamples, so it cannot be
    /// derived from the update index.
    pub cursor: u64,
    /// GRPO adaptive-KL multiplier to use for the next update. This is state,
    /// not configuration: it depends on the KL measured in prior updates.
    pub kl_multiplier: Option<f32>,
}

#[derive(Clone, Debug)]
pub struct Progress {
    pub metrics: TrainMetrics,
    pub values: Vec<MetricValue>,
    pub boundary: Option<Boundary>,
    /// Free-form lines the algorithm produced while computing this event, in
    /// the order it produced them.
    ///
    /// An algorithm never prints them itself: its output shares the terminal
    /// with a live progress bar, and a line written straight to stderr lands
    /// on the bar's row and is truncated by its next redraw. The caller owns
    /// the rendering - for the CLI that is a wrapped row inside the table
    /// frame.
    pub notes: Vec<String>,
}

impl Progress {
    pub fn sft(metrics: TrainMetrics, include_eval: bool) -> Self {
        let mut values = retrograd_metrics::values_from_metrics(metrics, include_eval);
        // A step row carries the loss of the batches since the previous row,
        // an epoch row the mean over the whole epoch. Two quantities, so two
        // series: on one curve the epoch mean lands as an outlier at every
        // boundary, which is the artefact the step loss was made to remove.
        if metrics.epoch_complete {
            for value in &mut values {
                if value.name == "train/loss" {
                    value.name = "train/epoch_loss".into();
                }
            }
        }
        Self {
            metrics,
            values,
            boundary: metrics.epoch_complete.then_some(Boundary {
                completed_iterations: metrics.epoch as u64,
                cursor: 0,
                kl_multiplier: None,
            }),
            notes: Vec::new(),
        }
    }
}
