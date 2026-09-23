//! PPO over the differentiable runtime objective.
//!
//! The sampling, weight, and optimizer-step machinery shared with GRPO lives
//! in the `rollout` module; what PPO adds is the advantage estimation - a
//! whitened per-sequence reward by default, or GAE over the linear-probe
//! value head ([`super::value`]) when the critic is enabled. The weights are
//! recomputed from a fresh forward pass before every optimizer step, so each
//! step is an exact PPO step.

use super::features::FeatureStore;
use super::observe::{PpoSlots, observed_prompts, outcome, ppo_rollouts};
use super::rollout::{
    PpoStepParams, Rollout, RowLayout, SchedulerHorizon, SurrogateStep, TokenStats,
    WeightedStepScratch, check_policy_divergence, mean_std, read_prompts, reward_process,
    sample_rollout, score, surrogate_step, tokenize_prompts,
};
use super::value::ValueHead;
use super::{Boundary, Progress};
use retrograd_config::{CriticConfig, PpoConfig};
use retrograd_core::{Error, Result, TrainConfig, TrainMetrics};
use retrograd_engine::Trainer;
use retrograd_metrics::MetricValue;
use retrograd_observe::{ObserveBatch, RolloutBatch, TrajectoryObserver};

pub fn evaluate(
    trainer: &mut Trainer,
    config: &PpoConfig,
    training: &TrainConfig,
    data: &std::path::Path,
    max_examples: Option<usize>,
) -> Result<super::RewardEvalMetrics> {
    super::rollout::evaluate_rewards(
        trainer,
        data,
        &mut reward_process(&config.reward_command, config.reward_protocol)?,
        &config.sampling,
        training,
        max_examples,
    )
}

pub fn benchmark_rewards(
    trainer: &mut Trainer,
    config: &PpoConfig,
    training: &TrainConfig,
    data: &std::path::Path,
    limit: Option<usize>,
) -> Result<Vec<f32>> {
    super::rollout::evaluate_reward_values(
        trainer,
        data,
        &mut reward_process(&config.reward_command, config.reward_protocol)?,
        &config.sampling,
        training,
        limit,
    )
}

/// Whitened advantages: `(r - mean) / (std + 1e-6)` across the rollout batch.
/// A single-rollout batch has no baseline, so it keeps its raw reward.
fn advantages(rewards: &[f32]) -> Vec<f32> {
    if rewards.len() < 2 {
        return rewards.to_vec();
    }
    let (mean, std, _) = mean_std(rewards.iter().copied());
    rewards
        .iter()
        .map(|&r| ((r as f64 - mean) / (std + 1e-6)) as f32)
        .collect()
}

/// In-place whitening of per-token advantages across the whole rollout batch.
fn whiten_tokens(advantages: &mut [Vec<f32>]) {
    let (mean, std, count) = mean_std(advantages.iter().flatten().copied());
    if count < 2 {
        return;
    }
    for row in advantages {
        for advantage in row {
            *advantage = ((*advantage as f64 - mean) / (std + 1e-6)) as f32;
        }
    }
}

/// Generalized advantage estimation over one rollout episode. `values` holds
/// `V(s_t)` for the state *before* each completion token; the only non-zero
/// per-step reward is the terminal `reward`, and the terminal value is 0.
/// Returns per-token advantages and the TD(lambda) returns
/// (`A_t + V(s_t)`) the value head regresses toward.
fn gae(reward: f32, values: &[f32], gamma: f32, lambda: f32) -> (Vec<f32>, Vec<f32>) {
    let steps = values.len();
    let mut advantages = vec![0.0_f32; steps];
    let mut returns = vec![0.0_f32; steps];
    let mut next_value = 0.0_f32;
    let mut next_advantage = 0.0_f32;
    for t in (0..steps).rev() {
        let step_reward = if t + 1 == steps { reward } else { 0.0 };
        let delta = step_reward + gamma * next_value - values[t];
        let advantage = delta + gamma * lambda * next_advantage;
        advantages[t] = advantage;
        returns[t] = advantage + values[t];
        next_value = values[t];
        next_advantage = advantage;
    }
    (advantages, returns)
}

/// What the critic reports back about one update, when it is enabled.
struct CriticStats {
    /// The value head's own post-fit regression loss.
    value_loss: f32,
    /// Bytes the feature matrix occupied in host RAM, reported so a run can tell
    /// whether this buffer exceeds 20% of the job's RAM - the point where a
    /// narrower `feature_dtype` is worth its rounding.
    feature_bytes: u64,
}

/// Per-token advantages for the batch, plus what the critic reports when the
/// value head is enabled.
fn batch_advantages(
    trainer: &mut Trainer,
    critic: &mut Option<ValueHead>,
    rollouts: &mut [Rollout],
    rewards: &[f32],
    config: &CriticConfig,
) -> Result<(Vec<Vec<f32>>, Option<CriticStats>)> {
    if !config.enabled {
        // No critic: the whitened per-sequence reward, constant over tokens.
        let advantages = advantages(rewards);
        return Ok((
            rollouts
                .iter()
                .zip(&advantages)
                .map(|(rollout, &advantage)| vec![advantage; rollout.completion_len()])
                .collect(),
            None,
        ));
    }

    let dim = trainer.hidden_size()?;
    let head = critic.get_or_insert_with(|| ValueHead::new(dim));
    let total_states = rollouts.iter().map(Rollout::completion_len).sum::<usize>();
    // The whole update's features, in the configured storage precision. This is
    // the PPO run's largest host allocation; `feature_dtype`
    // decides whether it is 4 or 2 bytes per element, and nothing else about the
    // update changes with it.
    let mut all_features = FeatureStore::with_capacity(dim, config.feature_dtype, total_states)?;
    // Staging for the fused pass when the store is narrow: one rollout's rows,
    // reused across rollouts. Empty and untouched for an F32 store, which is
    // filled in place.
    let mut staging = Vec::new();
    let mut all_returns = Vec::with_capacity(total_states);
    let mut per_rollout = Vec::with_capacity(rollouts.len());
    for (rollout, &reward) in rollouts.iter_mut().zip(rewards) {
        // The state before emitting completion token c is the prefix ending at
        // sequence index n_prompt + c - 1, so the value features are the
        // hidden states of indices n_prompt-1.. len-2 (the last token is only
        // ever a target, never a state). The fused pass writes this rollout's
        // feature rows straight into the buffer the store lends it.
        let n_prompt = rollout.first_train_index()?;
        if rollout.train_mask[n_prompt..].iter().any(|&train| !train) {
            return Err(Error::invalid(
                "the PPO critic requires a contiguous completion mask",
            ));
        }
        let tokens = &rollout.tokens;
        let old_logprobs = &mut rollout.old_logprobs;
        let filled = all_features.fill_from_tail(&mut staging, |out| {
            trainer.score_token_suffix_and_hidden_states_into(
                tokens,
                n_prompt,
                dim,
                old_logprobs,
                out,
            )
        })?;
        // Taken from the unrounded rows: the advantages of this update are the
        // ones the F32 path would have produced, whatever the store retains.
        let values = head.predict_rows(filled.as_slice());
        let (advantages, returns) = gae(reward, &values, config.gamma, config.gae_lambda);
        per_rollout.push(advantages);
        all_returns.extend_from_slice(&returns);
    }
    // Advantages come from the pre-fit head: estimate with the current value
    // function, then improve it toward the fresh returns.
    whiten_tokens(&mut per_rollout);
    let feature_bytes = all_features.allocated_bytes() as u64;
    let value_loss = head.fit_store(
        &mut all_features,
        &all_returns,
        config.value_lr,
        config.value_epochs,
    )?;
    Ok((
        per_rollout,
        Some(CriticStats {
            value_loss,
            feature_bytes,
        }),
    ))
}

/// Runs PPO end to end: sample rollouts, score them with the reward command,
/// whiten rewards into advantages, then take `ppo_epochs` exact PPO steps per
/// rollout batch through the runtime's weighted differentiable objective.
pub fn run(
    trainer: &mut Trainer,
    config: &PpoConfig,
    training: &TrainConfig,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TrainMetrics> {
    run_resumed(trainer, config, training, None, None, &mut |_, progress| {
        if progress.metrics.epoch_complete {
            on_progress(progress);
        }
        Ok(true)
    })
}

/// [`run`], restarting at a boundary restored from a checkpoint.
/// The prompt cursor is a pure function of the update index here, so only the
/// completed-update count is used; the critic is rebuilt from scratch, which
/// is why a PPO checkpoint marks its value head recreatable.
///
/// `observer` receives each wanted update's rollouts before its epochs, and
/// its outcome once they all succeeded.
pub fn run_resumed(
    trainer: &mut Trainer,
    config: &PpoConfig,
    training: &TrainConfig,
    resume: Option<Boundary>,
    observer: Option<&dyn TrajectoryObserver>,
    on_progress: &mut dyn FnMut(&mut Trainer, Progress) -> Result<bool>,
) -> Result<TrainMetrics> {
    let span = tracing::info_span!(target: "retrograd::training::ppo", "training");
    let _entered = span.enter();
    let start_update = resume.map_or(0, |boundary| boundary.completed_iterations);
    if start_update >= config.updates as u64 {
        return Err(Error::checkpoint(format!(
            "the checkpoint has {start_update} completed updates, at or past ppo.updates ({})",
            config.updates
        )));
    }
    let start_update = start_update as u32;
    let prompts = read_prompts(&config.prompts)?;
    let layout = RowLayout::resolve(trainer, training)?;
    // PPO tolerates truncated completions, so a prompt only has to leave room
    // for a single generated token; still validated once, up front.
    let prompt_rows = tokenize_prompts(trainer, &prompts, &layout, 1, &config.prompts)?;
    // Upper bound: a row is trained only up to its last active label, so a
    // short completion skips its trailing ubatches and costs less than
    // `steps_per_row`. Re-sized after every update on the steps actually
    // taken, so a decaying schedule still lands on zero.
    let total_steps = u64::from(config.updates)
        .checked_mul(u64::from(config.ppo_epochs))
        .and_then(|value| value.checked_mul(config.rollout_batch_size as u64))
        .and_then(|value| value.checked_mul(layout.steps_per_row))
        .ok_or_else(|| Error::overflow("PPO optimizer step count overflows u64"))?;
    let mut horizon = SchedulerHorizon::new(total_steps, training.warmup_steps);

    let mut final_metrics = TrainMetrics::default();
    let mut scratch = WeightedStepScratch::new(&layout);
    // One reward process for the whole run, not one per update: the command's
    // startup is identical every time it is paid, and PPO pays it `updates`
    // times otherwise.
    let mut reward = reward_process(&config.reward_command, config.reward_protocol)?;
    // The linear-probe critic persists across updates; it is created lazily so
    // disabled runs never query the hidden-state width.
    let mut critic: Option<ValueHead> = None;
    for update in start_update..config.updates {
        let step_params = PpoStepParams {
            clip_range: config.clip_range,
            kl_coefficient: config.kl_coefficient,
            scheduler_total_steps: horizon.steps(),
        };
        let update_start_step = final_metrics.global_step;
        let observer = observer.filter(|observer| observer.wants(update + 1));
        // 1. Sample one rollout per prompt, cycling deterministically.
        let mut rollouts = Vec::with_capacity(config.rollout_batch_size);
        let mut completions = Vec::with_capacity(config.rollout_batch_size);
        let first_offset = update as usize * config.rollout_batch_size;
        for index in 0..config.rollout_batch_size {
            let offset = first_offset + index;
            let prompt_tokens = &prompt_rows[offset % prompts.len()];
            // With a critic, the policy scores arrive together with the hidden
            // states in batch_advantages' fused pass, so sampling skips them.
            let (rollout, completion_text) = sample_rollout(
                trainer,
                prompt_tokens,
                &config.sampling,
                offset,
                &layout,
                /* score_policy = */ !config.critic.enabled,
            )?;
            completions.push(completion_text);
            rollouts.push(rollout);
        }

        // 2. External rewards, then per-token advantages: GAE over the value
        // head when the critic is enabled, whitened per-sequence rewards
        // otherwise. The head is fitted toward this batch's returns after the
        // advantages are taken from its pre-fit predictions.
        let rewards = score(
            &mut reward,
            completions.iter().enumerate().map(|(index, completion)| {
                (
                    prompts[(first_offset + index) % prompts.len()].reward_text(),
                    completion.as_str(),
                )
            }),
        )?;
        let mean_reward = rewards.iter().sum::<f32>() / rewards.len().max(1) as f32;
        let (advantages, critic_stats) = batch_advantages(
            trainer,
            &mut critic,
            &mut rollouts,
            &rewards,
            &config.critic,
        )?;
        match observer {
            Some(observer) => {
                let rollouts = ppo_rollouts(PpoSlots {
                    update: update + 1,
                    first_offset,
                    prompt_count: prompts.len(),
                    sampling_seed: config.sampling.seed,
                    max_new_tokens: config.sampling.max_new_tokens as usize,
                    completions,
                    lengths: rollouts.iter().map(Rollout::completion_len).collect(),
                    rewards: &rewards,
                    advantages: &advantages,
                    critic: config.critic.enabled,
                });
                observer.observe(ObserveBatch::Rollouts(RolloutBatch {
                    prompts: observed_prompts(
                        &prompts,
                        (0..config.rollout_batch_size)
                            .map(|index| (first_offset + index) % prompts.len()),
                    ),
                    rollouts,
                }));
            }
            None => drop(completions),
        }

        // 3. PPO epochs: one exact clipped-surrogate step per rollout, ratios
        // re-scored under the current policy before every step.
        for epoch in 0..config.ppo_epochs {
            let mut epoch_stats = TokenStats::default();
            for (rollout_index, (rollout, rollout_advantages)) in
                rollouts.iter().zip(&advantages).enumerate()
            {
                let (mut metrics, stats, keep_training) = surrogate_step(
                    trainer,
                    SurrogateStep {
                        rollout,
                        advantages: rollout_advantages,
                        layout: &layout,
                        params: step_params,
                        // Before the first optimizer step the policy still is
                        // the behavior policy, so re-scoring would reproduce
                        // `old_logprobs` exactly.
                        reuse_behavior_logprobs: epoch == 0 && rollout_index == 0,
                        scratch: &mut scratch,
                        on_step: &mut |trainer, mut metrics| {
                            metrics.epoch = update + 1;
                            metrics.epoch_complete = false;
                            on_progress(
                                trainer,
                                Progress {
                                    metrics,
                                    values: Vec::new(),
                                    boundary: None,
                                    notes: Vec::new(),
                                },
                            )
                        },
                    },
                )?;
                metrics.epoch = update + 1;
                metrics.epoch_complete = false;
                final_metrics = metrics;
                if !keep_training {
                    return Ok(final_metrics);
                }
                epoch_stats.surrogate_loss += stats.surrogate_loss;
                epoch_stats.kl += stats.kl;
                epoch_stats.clip_fraction += stats.clip_fraction;
                epoch_stats.ratio_max = epoch_stats.ratio_max.max(stats.ratio_max);
            }
            let n = rollouts.len().max(1) as f32;
            final_metrics.epoch = update + 1;
            final_metrics.epoch_complete = true;
            final_metrics.train_loss = epoch_stats.surrogate_loss / n;
            // Same trust-region guard as GRPO: PPO's k3 term is anchored to the
            // rollout policy rather than a frozen reference, but it saturates the
            // same way and takes the policy with it.
            check_policy_divergence(
                update + 1,
                epoch + 1,
                epoch_stats.kl / n,
                epoch_stats.clip_fraction / n,
                epoch_stats.ratio_max,
            )?;
            let closes_update = epoch + 1 == config.ppo_epochs;
            if closes_update && let Some(observer) = observer {
                observer.observe(ObserveBatch::Outcome {
                    update: update + 1,
                    entries: outcome(None, &vec![true; rollouts.len()]),
                });
            }
            let mut values = vec![
                MetricValue {
                    name: "reward/mean".into(),
                    value: mean_reward,
                },
                MetricValue {
                    name: "policy/surrogate_loss".into(),
                    value: epoch_stats.surrogate_loss / n,
                },
                MetricValue {
                    name: "policy/kl".into(),
                    value: epoch_stats.kl / n,
                },
                MetricValue {
                    name: "policy/clip_fraction".into(),
                    value: epoch_stats.clip_fraction / n,
                },
                MetricValue {
                    name: "optimizer/learning_rate".into(),
                    value: final_metrics.learning_rate,
                },
            ];
            if let Some(stats) = &critic_stats {
                values.push(MetricValue {
                    name: "policy/value_loss".into(),
                    value: stats.value_loss,
                });
                // In MiB rather than bytes: the metric backends carry f32, which
                // stops counting bytes exactly at 16 MiB and would silently round
                // the very figure a `feature_dtype` decision is read from.
                values.push(MetricValue {
                    name: "critic/feature_mib".into(),
                    value: stats.feature_bytes as f32 / (1024.0 * 1024.0),
                });
            }
            if !on_progress(
                trainer,
                Progress {
                    metrics: final_metrics,
                    values,
                    // The packing selection runs before this event and has
                    // nowhere to print; its lines ride out on it.
                    notes: std::mem::take(&mut scratch.notes),
                    // Only the last policy epoch of an update closes a
                    // resumable boundary: the next update re-samples from a
                    // deterministic cursor, an intermediate epoch does not.
                    boundary: closes_update.then(|| Boundary {
                        completed_iterations: update as u64 + 1,
                        cursor: (update as u64 + 1) * config.rollout_batch_size as u64,
                        kl_multiplier: None,
                    }),
                },
            )? {
                return Ok(final_metrics);
            }
        }
        horizon.observe(
            final_metrics.global_step.saturating_sub(update_start_step),
            (config.updates - update - 1) as u64,
            final_metrics.global_step,
        );
    }
    Ok(final_metrics)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advantages_are_whitened_across_the_batch() {
        let advs = advantages(&[1.0, 3.0]);
        // mean 2, std 1: whitened to almost exactly [-1, 1].
        assert!((advs[0] + 1.0).abs() < 1e-4, "{advs:?}");
        assert!((advs[1] - 1.0).abs() < 1e-4, "{advs:?}");
    }

    #[test]
    fn single_rollout_keeps_its_raw_reward() {
        assert_eq!(advantages(&[2.5]), vec![2.5]);
    }

    #[test]
    fn identical_rewards_produce_zero_advantages() {
        for adv in advantages(&[1.0, 1.0, 1.0]) {
            assert!(adv.abs() < 1e-6);
        }
    }

    #[test]
    fn gae_with_zero_values_and_no_discount_spreads_the_reward() {
        // V = 0, gamma = lambda = 1: A_t = R for every token (the exact
        // behavior of the critic-less path before whitening).
        let (advantages, returns) = gae(3.0, &[0.0, 0.0, 0.0], 1.0, 1.0);
        assert_eq!(advantages, vec![3.0, 3.0, 3.0]);
        assert_eq!(returns, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn gae_matches_a_hand_computed_episode() {
        // Two steps, V = [0.5, 0.25], terminal reward 1, gamma 0.9, lambda 0.8.
        // delta_1 = 1 + 0.9*0 - 0.25 = 0.75            (terminal state: V = 0)
        // delta_0 = 0 + 0.9*0.25 - 0.5 = -0.275
        // A_1 = 0.75; A_0 = -0.275 + 0.9*0.8*0.75 = 0.265
        let (advantages, returns) = gae(1.0, &[0.5, 0.25], 0.9, 0.8);
        assert!((advantages[1] - 0.75).abs() < 1e-6);
        assert!((advantages[0] - 0.265).abs() < 1e-6);
        // Returns are the value-head targets: A_t + V(s_t).
        assert!((returns[0] - (0.265 + 0.5)).abs() < 1e-6);
        assert!((returns[1] - (0.75 + 0.25)).abs() < 1e-6);
    }

    #[test]
    fn gae_credits_tokens_where_the_value_climbs() {
        // A perfect value function that already anticipates the terminal
        // reward yields zero advantage everywhere (gamma = lambda = 1).
        let (advantages, _) = gae(1.0, &[1.0, 1.0, 1.0], 1.0, 1.0);
        for advantage in advantages {
            assert!(advantage.abs() < 1e-6);
        }
    }

    #[test]
    fn whitening_normalizes_across_rollouts_of_different_lengths() {
        let mut advantages = vec![vec![1.0, 3.0], vec![2.0]];
        whiten_tokens(&mut advantages);
        let flat: Vec<f32> = advantages.iter().flatten().copied().collect();
        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        let variance =
            flat.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / flat.len() as f32;
        assert!(mean.abs() < 1e-5);
        assert!((variance - 1.0).abs() < 1e-2);
    }
}
