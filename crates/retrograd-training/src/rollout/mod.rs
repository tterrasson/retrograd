//! Building blocks shared by the rollout-based algorithms (PPO and GRPO):
//! prompt loading, external reward scoring, rollout sampling, and the exact
//! clipped-surrogate optimizer step over the runtime's weighted objective.
//!
//! The clipped surrogate with detached per-token coefficients reduces to a
//! weighted cross-entropy: for sampled token `a_t`, the gradient of
//! `w_t * (-log pi(a_t))` w.r.t. the logits is `w_t * (softmax - one_hot)`,
//! which equals the PPO policy gradient when
//! `w_t = A_t * r_t * [not clipped] + w_kl`. PPO anchors its k3 term to the
//! rollout policy; GRPO uses the standard fixed-reference k3 term. The weights
//! are recomputed from a fresh forward pass before every optimizer step.
//!
//! One responsibility per file: [`prompts`] reads and tokenizes datasets,
//! [`reward`] speaks the reward-process protocol and runs held-out evaluation,
//! [`judge`] blends `[grpo.judge]`'s verdicts into what that protocol returned,
//! [`sampling`] owns the row geometry and the decode paths, [`weights`] the
//! objective's per-token coefficients, [`packing`] the physical batches, and
//! [`step`] the optimizer steps built on all of them.

pub(crate) mod judge;
pub(crate) mod packing;
pub(crate) mod prompts;
pub(crate) mod reward;
pub(crate) mod sampling;
pub(crate) mod step;
pub(crate) mod weights;

#[cfg(test)]
mod tests;

pub(crate) use judge::{GroupJudge, JudgeGroup, JudgeTally};
pub(crate) use packing::WeightedStepScratch;
pub(crate) use prompts::{
    Prompt, read_eval_prompts, read_prompts, tokenize_prompt, tokenize_prompts,
};
pub(crate) use reward::{
    RewardRow, evaluate_reward_values, evaluate_rewards, evenly_spaced_subset, reward_process,
    score, score_rows,
};
pub(crate) use sampling::{
    Rollout, RowLayout, generation_room, generation_wave, sample_rollout,
    sample_rollout_groups_continuous,
};
pub(crate) use step::{
    EpochBatch, EpochState, GrpoObjective, GrpoStepParams, PpoStepParams, SchedulerHorizon,
    SurrogateStep, is_truncated, run_grpo_epoch, surrogate_step,
};
pub(crate) use weights::{
    TokenStats, check_policy_divergence, mean_std, score_train_mask, score_train_mask_group,
};
