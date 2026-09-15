//! The PPO / Dr. GRPO token weights and the statistics read off them: the
//! clipped surrogate, its dual bound, the k3 KL terms, and the trust-region
//! guard that stops a diverged run.

use retrograd_core::{Error, Result};
use retrograd_engine::Trainer;

use super::sampling::first_train_index;
use super::step::GrpoObjective;
use crate::TokenSpan;

/// Mean, standard deviation and count of the values, accumulated in f64 so
/// extreme (but finite) f32 rewards cannot overflow into NaN/inf. Shared by
/// every path that needs a reward or advantage summary: GRPO's group
/// baseline, PPO's reward whitening, and both algorithms' published reward
/// statistics.
pub(crate) fn mean_std(values: impl Iterator<Item = f32> + Clone) -> (f64, f64, usize) {
    let mut n = 0_usize;
    let mut sum = 0.0_f64;
    for value in values.clone() {
        n += 1;
        sum += value as f64;
    }
    if n == 0 {
        return (0.0, 0.0, 0);
    }
    let mean = sum / n as f64;
    let variance = values
        .map(|value| {
            let delta = value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / n as f64;
    (mean, variance.sqrt(), n)
}

/// Aggregate diagnostics for one clipped-surrogate optimizer step.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct TokenStats {
    /// Mean k3 KL estimate (against the algorithm's chosen anchor).
    pub(crate) kl: f32,
    /// Fraction of tokens whose surrogate gradient the clip zeroed out.
    pub(crate) clip_fraction: f32,
    /// Value of the (negated) clipped surrogate objective.
    pub(crate) surrogate_loss: f32,
    /// Mean policy ratio `pi/pi_old` over the step's tokens - the off-policy
    /// drift diagnostic between optimizer epochs (1.0 before the first step).
    pub(crate) ratio_mean: f32,
    /// Largest policy ratio observed over the step's tokens.
    pub(crate) ratio_max: f32,
}

impl TokenStats {
    fn normalize(&mut self, objective_denominator: usize, token_count: usize) {
        let objective_denominator = objective_denominator.max(1) as f32;
        self.kl /= objective_denominator;
        self.surrogate_loss /= objective_denominator;
        self.clip_fraction /= token_count.max(1) as f32;
        self.ratio_mean /= token_count.max(1) as f32;
    }
}

/// Bound on the policy log-ratio before it is exponentiated. The weights that
/// survive clipping are already bounded by the clip band and the dual clip
/// below, so this is a guard against a pathological early policy producing a
/// non-finite ratio, not a part of the objective.
const POLICY_LOG_RATIO_LIMIT: f32 = 10.0;

/// Bound on the k3 log-ratio, whatever the estimator's anchor (the fixed
/// reference for Dr. GRPO, the rollout policy for PPO).
///
/// Unlike the surrogate, the k3 gradient coefficient `beta * (exp(d) - 1)` has
/// no clip to bound it: a diverged policy injects `beta * exp(d)` straight into
/// the weighted cross-entropy. At ±20, that would be `beta * 4.85e8` on a
/// single token, which makes the global gradient-norm
/// clipping rescale every other token to nothing - one saturated token then
/// owns the whole update and finishes the policy off. Five nats is already a
/// 148x probability gap, far outside any usable trust region, and keeps the
/// coefficient within the same order of magnitude as a typical advantage.
pub(super) const REFERENCE_LOG_RATIO_LIMIT: f32 = 5.0;

/// Dual-clip PPO (Ye et al., "Mastering Complex Control in MOBA Games with Deep
/// Reinforcement Learning", AAAI 2020).
///
/// The ordinary clip only bounds the ratio when it moves *with* the advantage.
/// For `A < 0` the objective is `A * r`, unbounded below as `r` grows, and its
/// gradient never vanishes - so one token whose probability the policy raised
/// while its advantage was negative can dominate an entire update. Beyond `c`
/// the objective is flattened to `c * A`, which zeroes that token's gradient
/// exactly like the ordinary clip does on the other side.
pub(super) const DUAL_CLIP_C: f32 = 3.0;

/// Clipped-surrogate outcome for one token: the gradient coefficient of
/// `-log pi(a_t)`, the value of the (non-negated) objective for reporting, and
/// whether the token's surrogate gradient was clipped away.
fn clipped_surrogate(
    advantage: f32,
    ratio: f32,
    clip_range_low: f32,
    clip_range_high: f32,
) -> (f32, f32, bool) {
    let clipped_ratio = ratio.clamp(1.0 - clip_range_low, 1.0 + clip_range_high);
    let mut objective = (advantage * ratio).min(advantage * clipped_ratio);
    let mut clipped_out = (advantage > 0.0 && ratio > 1.0 + clip_range_high)
        || (advantage < 0.0 && ratio < 1.0 - clip_range_low);
    if advantage < 0.0 && ratio > DUAL_CLIP_C {
        objective = DUAL_CLIP_C * advantage;
        clipped_out = true;
    }
    let coefficient = if clipped_out { 0.0 } else { advantage * ratio };
    (coefficient, objective, clipped_out)
}

/// A clipped surrogate this saturated means the policy left the trust region
/// the epochs assume: almost every token has already moved past its clip band,
/// so the update is carrying no usable signal and the next one will be measured
/// against a policy that no longer resembles the behavior policy.
const DIVERGENCE_CLIP_FRACTION: f32 = 0.9;

/// Per-token k3 KL against the fixed reference, above which the run is
/// diverging rather than exploring. Healthy values are well under 1.
const DIVERGENCE_KL: f32 = 50.0;

/// Refuses to keep training a policy that has left its trust region.
///
/// Without this, a diverged update is silent: the KL term saturates, the
/// gradient collapses onto a handful of tokens, and every subsequent update
/// samples completions whose rewards are all identical - so every group is
/// dropped as zero-signal and the run spends the rest of its budget sampling
/// without optimizing anything. Stopping here keeps the last checkpoint, which
/// is the only recoverable state at that point.
pub(crate) fn check_policy_divergence(
    iteration: u32,
    epoch: u32,
    kl: f32,
    clip_fraction: f32,
    ratio_max: f32,
) -> Result<()> {
    let diagnosis = if !kl.is_finite() || kl > DIVERGENCE_KL {
        format!("the k3 KL against the reference policy reached {kl:.4}")
    } else if clip_fraction > DIVERGENCE_CLIP_FRACTION {
        format!(
            "{:.1}% of the tokens are outside their clip band",
            clip_fraction * 100.0
        )
    } else {
        return Ok(());
    };
    Err(Error::runtime(format!(
        "iteration {iteration}, epoch {epoch}: the policy left its trust region - {diagnosis} \
         (largest policy ratio {ratio_max:.4}). The optimizer moved the policy far more than one \
         epoch of a rollout batch should: check that training.micro_batch * \
         training.gradient_accumulation spans training.ctx (a smaller window takes one optimizer \
         step per window-sized slice of every row instead of one per rollout), then lower \
         training.lr or the number of policy epochs. The last checkpoint is still the best policy \
         this run reached."
    )))
}

/// Per-token weights of the detached-coefficient PPO objective. The weight of
/// a token is the exact gradient coefficient of `-log pi(a_t)`:
/// `A*r` while the clipped surrogate is active, 0 once clipping binds, plus
/// `k*(1 - r)` from the k3 KL penalty toward the rollout policy.
#[cfg(test)]
pub(crate) fn ppo_token_weights(
    advantages: &[f32],
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    clip_range: f32,
    kl_coefficient: f32,
) -> (Vec<f32>, TokenStats) {
    let mut weights = Vec::with_capacity(old_logprobs.len());
    let stats = ppo_token_weights_into(
        &mut weights,
        advantages,
        old_logprobs,
        new_logprobs,
        clip_range,
        kl_coefficient,
    );
    (weights, stats)
}

pub(super) fn ppo_token_weights_into(
    weights: &mut Vec<f32>,
    advantages: &[f32],
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    clip_range: f32,
    kl_coefficient: f32,
) -> TokenStats {
    debug_assert_eq!(old_logprobs.len(), new_logprobs.len());
    debug_assert_eq!(old_logprobs.len(), advantages.len());
    weights.clear();
    weights.reserve(old_logprobs.len());
    let mut stats = TokenStats::default();
    for ((&old, &new), &advantage) in old_logprobs.iter().zip(new_logprobs).zip(advantages) {
        // Ratios explode as exp(); clamping the log-ratio keeps a pathological
        // early policy from producing non-finite weights.
        let log_ratio = (new - old).clamp(-POLICY_LOG_RATIO_LIMIT, POLICY_LOG_RATIO_LIMIT);
        let ratio = log_ratio.exp();
        let (surrogate, objective, clipped_out) =
            clipped_surrogate(advantage, ratio, clip_range, clip_range);
        // The k3 term is anchored to the rollout policy here, but it is the same
        // unbounded `exp(d)` coefficient Dr. GRPO's fixed-reference term uses,
        // and it gets the same tighter bound.
        let kl_log_ratio = (new - old).clamp(-REFERENCE_LOG_RATIO_LIMIT, REFERENCE_LOG_RATIO_LIMIT);
        let kl_ratio = kl_log_ratio.exp();
        weights.push(surrogate + kl_coefficient * (1.0 - kl_ratio));

        stats.surrogate_loss += -objective;
        stats.kl += (kl_ratio - 1.0) - kl_log_ratio;
        stats.clip_fraction += if clipped_out { 1.0 } else { 0.0 };
        stats.ratio_mean += ratio;
        stats.ratio_max = stats.ratio_max.max(ratio);
    }
    stats.normalize(old_logprobs.len(), old_logprobs.len());
    stats
}

/// Dr. GRPO coefficients. The clipped policy ratio is against the rollout
/// policy, while the KL penalty is against a fixed reference policy:
///
/// `KL_hat = exp(log p_ref - log p) - (log p_ref - log p) - 1`
///
/// For the weighted-CE reduction, the corresponding detached KL coefficient
/// is `beta * (exp(log p_ref - log p) - 1)`. The advantage is one scalar for
/// the whole completion, and the reported stats are normalized by
/// `loss_denominator` - the constant generation budget - rather than by the
/// completion's own length.
///
/// The clip is DAPO's Clip-Higher: the ratio band is `[1 - clip_range_low,
/// 1 + clip_range_high]`. A symmetric band caps how much a low-probability
/// token with positive advantage can grow at `1 + eps` of almost nothing,
/// which drives entropy collapse; a larger upper range relaxes exactly that
/// direction while keeping the usual lower bound on demotion. The direction
/// neither band bounds - a negative advantage whose ratio grew - is bounded by
/// [`DUAL_CLIP_C`].
#[cfg(test)]
pub(crate) fn grpo_token_weights(
    advantage: f32,
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    reference_logprobs: &[f32],
    objective: &GrpoObjective,
) -> (Vec<f32>, TokenStats) {
    let mut weights = Vec::with_capacity(old_logprobs.len());
    let stats = grpo_token_weights_into(
        &mut weights,
        advantage,
        None,
        old_logprobs,
        new_logprobs,
        reference_logprobs,
        objective,
    );
    (weights, stats)
}

pub(super) fn grpo_token_weights_into(
    weights: &mut Vec<f32>,
    advantage: f32,
    token_advantages: Option<&[f32]>,
    old_logprobs: &[f32],
    new_logprobs: &[f32],
    reference_logprobs: &[f32],
    objective: &GrpoObjective,
) -> TokenStats {
    let &GrpoObjective {
        clip_range_low,
        clip_range_high,
        kl_coefficient,
        loss_denominator,
    } = objective;
    debug_assert_eq!(old_logprobs.len(), new_logprobs.len());
    debug_assert!(token_advantages.is_none_or(|values| values.len() == old_logprobs.len()));
    // When `kl_coefficient == 0` the fixed-reference term is identically zero,
    // so the caller is allowed to skip the reference forward pass entirely and
    // pass an empty `reference_logprobs` slice (see the KL-zero fast path in
    // the GRPO loop). With a live anchor the slice must align token-for-token.
    let use_kl = kl_coefficient != 0.0;
    debug_assert!(!use_kl || reference_logprobs.len() == old_logprobs.len());
    weights.clear();
    weights.reserve(old_logprobs.len());
    let mut stats = TokenStats::default();
    for (index, (&old, &new)) in old_logprobs.iter().zip(new_logprobs).enumerate() {
        let advantage = token_advantages.map_or(advantage, |values| values[index]);
        let policy_log_ratio = (new - old).clamp(-POLICY_LOG_RATIO_LIMIT, POLICY_LOG_RATIO_LIMIT);
        let policy_ratio = policy_log_ratio.exp();
        let (surrogate, objective, clipped_out) =
            clipped_surrogate(advantage, policy_ratio, clip_range_low, clip_range_high);

        let (kl_weight, kl_stat) = if use_kl {
            let reference_log_ratio = (reference_logprobs[index] - new)
                .clamp(-REFERENCE_LOG_RATIO_LIMIT, REFERENCE_LOG_RATIO_LIMIT);
            let reference_ratio = reference_log_ratio.exp();
            (
                kl_coefficient * (reference_ratio - 1.0),
                reference_ratio - reference_log_ratio - 1.0,
            )
        } else {
            (0.0, 0.0)
        };
        weights.push(surrogate + kl_weight);

        stats.surrogate_loss += -objective;
        stats.kl += kl_stat;
        stats.clip_fraction += if clipped_out { 1.0 } else { 0.0 };
        stats.ratio_mean += policy_ratio;
        stats.ratio_max = stats.ratio_max.max(policy_ratio);
    }
    stats.normalize(loss_denominator, old_logprobs.len());
    stats
}

/// Scores all targets selected by `train_mask`, preserving their increasing
/// sequence order. The runtime scores one suffix efficiently; this helper then
/// selects policy-action positions and drops intervening context/tool tokens.
pub(crate) fn score_train_mask_into(
    trainer: &mut Trainer,
    tokens: &[i32],
    train_mask: &[bool],
    logprobs: &mut Vec<f32>,
    suffix_scratch: &mut Vec<f32>,
) -> Result<()> {
    let first = first_train_index(tokens, train_mask)?;
    trainer.score_token_suffix_into(tokens, first, suffix_scratch)?;
    logprobs.clear();
    logprobs.reserve(train_mask.iter().filter(|&&train| train).count());
    for (target, &score) in (first..tokens.len()).zip(suffix_scratch.iter()) {
        if train_mask[target] {
            logprobs.push(score);
        }
    }
    Ok(())
}

pub(crate) fn score_train_mask(
    trainer: &mut Trainer,
    tokens: &[i32],
    train_mask: &[bool],
) -> Result<Vec<f32>> {
    let mut logprobs = Vec::new();
    let mut suffix_scratch = Vec::new();
    score_train_mask_into(
        trainer,
        tokens,
        train_mask,
        &mut logprobs,
        &mut suffix_scratch,
    )?;
    Ok(logprobs)
}

/// Scores one prompt group through a single shared-prefix forward pass. The
/// return rows preserve the input span order and retain only positions
/// selected by each span's train mask.
///
/// Members are borrowed individually rather than as a slice: a caller may need
/// to score a *subset* of a group (truncation masking drops members, and the
/// survivors are not contiguous), and every member of the batch only has to
/// share the group's prompt.
pub(crate) fn score_train_mask_group<S: TokenSpan + ?Sized>(
    trainer: &mut Trainer,
    spans: &[&S],
) -> Result<Vec<Vec<f32>>> {
    if spans.is_empty() {
        return Ok(Vec::new());
    }
    let inputs = spans
        .iter()
        .map(|span| {
            Ok((
                span.tokens(),
                first_train_index(span.tokens(), span.train_mask())?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let suffixes = trainer.score_token_suffix_batch(&inputs)?;
    spans
        .iter()
        .zip(inputs)
        .zip(suffixes)
        .map(|((span, (_, first)), suffix)| {
            if suffix.len() != span.tokens().len() - first {
                return Err(Error::runtime(
                    "batched suffix scores do not match rollout length",
                ));
            }
            Ok((first..span.tokens().len())
                .zip(suffix)
                .filter_map(|(target, score)| span.train_mask()[target].then_some(score))
                .collect())
        })
        .collect()
}
