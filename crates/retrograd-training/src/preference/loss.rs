//! The preference losses and the per-token coefficients that train them.
//!
//! Every loss here depends on the parameters only through the summed
//! log-probability `S` of each response (or its per-token mean `S / n`). Its
//! gradient is therefore `-dL/dS` times the gradient of `log pi` on every
//! trained token of that response - one detached coefficient per sequence -
//! and the runtime's weighted cross-entropy `sum_t w_t * -log p_t` reproduces
//! it exactly with `w_t = -dL/dS` on that response's tokens. Nothing here
//! approximates the loss: the step is the autodiff gradient of the value this
//! module reports.
//!
//! Everything is computed in `f64`; a coefficient becomes an `f32` only when it
//! is written into a batch.

use retrograd_config::PreferenceLoss;
use retrograd_core::{Error, Result};

/// What the objective reads of one response: its summed log-probability under
/// the current policy, its trained-token count, and the same sum under the
/// reference when the loss has one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SideScores {
    pub(crate) sum: f64,
    pub(crate) tokens: usize,
    pub(crate) reference: Option<f64>,
}

impl SideScores {
    /// The count as the float the objective divides by. Exact: a trained-token
    /// count is bounded by the context, a `u32`, far below 2^53.
    fn count(&self) -> f64 {
        self.tokens as f64
    }

    fn mean(&self) -> f64 {
        self.sum / self.count()
    }

    fn reference(&self, loss: &PreferenceLoss) -> Result<f64> {
        self.reference.ok_or_else(|| {
            Error::invalid(format!(
                "preference loss '{}' needs a reference score for every response",
                loss.name()
            ))
        })
    }
}

/// ORPO's own quantities, for the metrics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct OrpoTerms {
    /// `-l_c`, the chosen response's per-token NLL.
    pub(crate) nll: f64,
    /// `z`, the log odds ratio of chosen over rejected.
    pub(crate) log_odds: f64,
    /// How many of the two means sat at or above the bound that keeps
    /// `log(1 - p)` finite.
    pub(crate) clamped: u32,
}

/// One pair's loss, the coefficients that train it, and what a report reads.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairTerms {
    pub(crate) loss: f64,
    /// Per-token coefficient on each side, before the division by the number
    /// of pairs in the step.
    pub(crate) chosen_weight: f64,
    pub(crate) rejected_weight: f64,
    /// Implicit rewards: `beta * (S - R)` for DPO, `beta * (l - R / n)` for
    /// IPO - the per-token form its margin is taken on - and `beta * l` for
    /// the reference-free losses.
    pub(crate) chosen_reward: f64,
    pub(crate) rejected_reward: f64,
    pub(crate) orpo: Option<OrpoTerms>,
}

/// Upper bound on a mean log-probability before `log(1 - exp(l))`: at `l = 0`
/// the response is certain and its odds are infinite.
const ORPO_MEAN_BOUND: f64 = -1e-6;

/// `log(1 + exp(x))`, without overflow for large `x` or loss of precision for
/// very negative `x`.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

/// `1 / (1 + exp(-x))`, evaluated on the side that does not overflow.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// `log(p / (1 - p))` for `p = exp(l)`, and its derivative in `l`.
fn log_odds(mean: f64) -> (f64, f64) {
    let complement = -mean.exp_m1();
    (mean - complement.ln(), 1.0 / complement)
}

/// The loss of one pair and its per-token coefficients. `-dL/dS` for a loss on
/// the sums, `-dL/dl / n` for a loss on the per-token means.
pub(crate) fn pair_terms(
    loss: &PreferenceLoss,
    chosen: SideScores,
    rejected: SideScores,
) -> Result<PairTerms> {
    if chosen.tokens == 0 || rejected.tokens == 0 {
        return Err(Error::invalid(
            "a preference response must have at least one trained token",
        ));
    }
    if !chosen.sum.is_finite() || !rejected.sum.is_finite() {
        return Err(Error::runtime(
            "a preference response scored a non-finite log-probability",
        ));
    }
    let beta = f64::from(loss.beta());
    let terms = match *loss {
        PreferenceLoss::Dpo {
            label_smoothing, ..
        } => {
            let epsilon = f64::from(label_smoothing);
            let chosen_reward = beta * (chosen.sum - chosen.reference(loss)?);
            let rejected_reward = beta * (rejected.sum - rejected.reference(loss)?);
            let margin = chosen_reward - rejected_reward;
            let weight = beta * ((1.0 - epsilon) * sigmoid(-margin) - epsilon * sigmoid(margin));
            PairTerms {
                loss: (1.0 - epsilon) * softplus(-margin) + epsilon * softplus(margin),
                chosen_weight: weight,
                rejected_weight: -weight,
                chosen_reward,
                rejected_reward,
                orpo: None,
            }
        }
        PreferenceLoss::Ipo { .. } => {
            let chosen_ratio = chosen.mean() - chosen.reference(loss)? / chosen.count();
            let rejected_ratio = rejected.mean() - rejected.reference(loss)? / rejected.count();
            let gap = chosen_ratio - rejected_ratio - 1.0 / (2.0 * beta);
            PairTerms {
                loss: gap * gap,
                chosen_weight: -2.0 * gap / chosen.count(),
                rejected_weight: 2.0 * gap / rejected.count(),
                chosen_reward: beta * chosen_ratio,
                rejected_reward: beta * rejected_ratio,
                orpo: None,
            }
        }
        PreferenceLoss::Simpo {
            gamma_beta_ratio, ..
        } => {
            let margin = beta * (chosen.mean() - rejected.mean() - f64::from(gamma_beta_ratio));
            let slope = beta * sigmoid(-margin);
            PairTerms {
                loss: softplus(-margin),
                chosen_weight: slope / chosen.count(),
                rejected_weight: -slope / rejected.count(),
                chosen_reward: beta * chosen.mean(),
                rejected_reward: beta * rejected.mean(),
                orpo: None,
            }
        }
        PreferenceLoss::Orpo { .. } => {
            let lambda = beta;
            let chosen_clamped = chosen.mean() > ORPO_MEAN_BOUND;
            let rejected_clamped = rejected.mean() > ORPO_MEAN_BOUND;
            let clamped = u32::from(chosen_clamped) + u32::from(rejected_clamped);
            let chosen_mean = chosen.mean().min(ORPO_MEAN_BOUND);
            let rejected_mean = rejected.mean().min(ORPO_MEAN_BOUND);
            let (chosen_odds, chosen_slope) = log_odds(chosen_mean);
            let (rejected_odds, rejected_slope) = log_odds(rejected_mean);
            let z = chosen_odds - rejected_odds;
            let pull = lambda * sigmoid(-z);
            // Past the bound the loss reads the bound, not the mean, so that
            // side's derivative is zero - and so is its coefficient.
            let live = |clamped: bool| f64::from(u8::from(!clamped));
            PairTerms {
                loss: -chosen_mean + lambda * softplus(-z),
                chosen_weight: live(chosen_clamped) * (1.0 + pull * chosen_slope) / chosen.count(),
                rejected_weight: live(rejected_clamped) * -pull * rejected_slope
                    / rejected.count(),
                chosen_reward: beta * chosen_mean,
                rejected_reward: beta * rejected_mean,
                orpo: Some(OrpoTerms {
                    nll: -chosen_mean,
                    log_odds: z,
                    clamped,
                }),
            }
        }
    };
    if ![
        terms.loss,
        terms.chosen_weight,
        terms.rejected_weight,
        terms.chosen_reward,
        terms.rejected_reward,
    ]
    .iter()
    .all(|value| value.is_finite())
    {
        return Err(Error::runtime(format!(
            "preference loss '{}' produced a non-finite value for a pair",
            loss.name()
        )));
    }
    Ok(terms)
}

/// A step coefficient as the `f32` the batch carries. The narrowing is refused,
/// not saturated, when the value does not survive it: the runtime would refuse
/// a non-finite weight anyway, one call later and without saying whose.
pub(crate) fn batch_weight(coefficient: f64) -> Result<f32> {
    let weight = coefficient as f32;
    if !weight.is_finite() {
        return Err(Error::invalid(format!(
            "a preference coefficient of {coefficient} does not fit a training weight"
        )));
    }
    Ok(weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    const DPO: PreferenceLoss = PreferenceLoss::Dpo {
        beta: 0.1,
        label_smoothing: 0.0,
    };
    const CDPO: PreferenceLoss = PreferenceLoss::Dpo {
        beta: 0.3,
        label_smoothing: 0.2,
    };
    const IPO: PreferenceLoss = PreferenceLoss::Ipo { beta: 0.4 };
    const SIMPO: PreferenceLoss = PreferenceLoss::Simpo {
        beta: 2.0,
        gamma_beta_ratio: 0.5,
    };
    const ORPO: PreferenceLoss = PreferenceLoss::Orpo { beta: 0.1 };

    fn side(sum: f64, tokens: usize, reference: f64) -> SideScores {
        SideScores {
            sum,
            tokens,
            reference: Some(reference),
        }
    }

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() <= 1e-7 * expected.abs().max(1.0)
    }

    /// `-dL/dS` by centred differences, the way autodiff would see the loss.
    fn numerical(loss: &PreferenceLoss, chosen: SideScores, rejected: SideScores) -> (f64, f64) {
        let h = 1e-5;
        let value = |chosen: SideScores, rejected: SideScores| {
            pair_terms(loss, chosen, rejected).unwrap().loss
        };
        let shift = |scores: SideScores, by: f64| SideScores {
            sum: scores.sum + by,
            ..scores
        };
        let chosen_slope =
            (value(shift(chosen, h), rejected) - value(shift(chosen, -h), rejected)) / (2.0 * h);
        let rejected_slope =
            (value(chosen, shift(rejected, h)) - value(chosen, shift(rejected, -h))) / (2.0 * h);
        (-chosen_slope, -rejected_slope)
    }

    #[test]
    fn every_coefficient_is_the_derivative_of_its_loss() {
        let pairs = [
            (side(-12.0, 6, -13.5), side(-9.0, 4, -8.0)),
            (side(-3.0, 3, -2.0), side(-20.0, 10, -19.0)),
            (side(-40.0, 17, -40.0), side(-41.0, 23, -39.5)),
        ];
        for loss in [DPO, CDPO, IPO, SIMPO, ORPO] {
            for (chosen, rejected) in pairs {
                let terms = pair_terms(&loss, chosen, rejected).unwrap();
                let (chosen_weight, rejected_weight) = numerical(&loss, chosen, rejected);
                assert!(
                    close(terms.chosen_weight, chosen_weight),
                    "{loss:?} chosen: {} vs {chosen_weight}",
                    terms.chosen_weight
                );
                assert!(
                    close(terms.rejected_weight, rejected_weight),
                    "{loss:?} rejected: {} vs {rejected_weight}",
                    terms.rejected_weight
                );
            }
        }
    }

    #[test]
    fn a_saturated_margin_stays_finite() {
        for margin in [-50.0, 50.0] {
            // beta = 1 so the margin is the difference of the sums.
            let loss = PreferenceLoss::Dpo {
                beta: 1.0,
                label_smoothing: 0.0,
            };
            let terms = pair_terms(&loss, side(margin, 3, 0.0), side(0.0, 3, 0.0)).unwrap();
            assert!(terms.loss.is_finite() && terms.chosen_weight.is_finite());
            if margin > 0.0 {
                assert!(terms.loss < 1e-20 && terms.chosen_weight < 1e-20);
            } else {
                assert!(close(terms.loss, 50.0));
                assert!(close(terms.chosen_weight, 1.0));
            }
        }
        let loss = PreferenceLoss::Simpo {
            beta: 50.0,
            gamma_beta_ratio: 0.0,
        };
        let terms = pair_terms(&loss, side(-2.0, 2, 0.0), side(-4.0, 2, 0.0)).unwrap();
        assert!(terms.loss.is_finite() && terms.loss > 0.0);
    }

    #[test]
    fn no_smoothing_is_plain_dpo_and_dpo_weights_are_opposite() {
        let smoothed = PreferenceLoss::Dpo {
            beta: 0.1,
            label_smoothing: 0.0,
        };
        let (chosen, rejected) = (side(-5.0, 2, -6.0), side(-7.0, 3, -6.5));
        let terms = pair_terms(&smoothed, chosen, rejected).unwrap();
        let margin = 0.1 * ((-5.0 + 6.0) - (-7.0 + 6.5));
        assert!(close(terms.loss, -(sigmoid(margin).ln())));
        assert!(close(terms.chosen_weight, 0.1 * sigmoid(-margin)));
        // A token the two responses share, in the same context, is pushed up by
        // one and down by the other: its gradient cancels exactly, as it does
        // in the loss, which does not depend on it.
        assert_eq!(terms.chosen_weight, -terms.rejected_weight);
        assert!(close(terms.chosen_reward - terms.rejected_reward, margin));
    }

    #[test]
    fn ipo_at_its_target_margin_does_not_move() {
        let beta = 0.25;
        let loss = PreferenceLoss::Ipo { beta };
        // Chosen over reference by 2 per token, rejected at its reference:
        // h = 2 = 1 / (2 beta).
        let terms = pair_terms(&loss, side(-4.0, 2, -8.0), side(-9.0, 3, -9.0)).unwrap();
        assert!(close(terms.loss, 0.0));
        assert!(terms.chosen_weight.abs() < 1e-12 && terms.rejected_weight.abs() < 1e-12);
    }

    #[test]
    fn orpo_bounds_a_certain_response_and_counts_it() {
        let terms = pair_terms(&ORPO, side(0.0, 4, 0.0), side(-8.0, 4, 0.0)).unwrap();
        let orpo = terms.orpo.expect("ORPO reports its own terms");
        assert_eq!(orpo.clamped, 1);
        assert!(terms.loss.is_finite() && terms.chosen_weight.is_finite());
        assert!(close(orpo.nll, -ORPO_MEAN_BOUND));
        // The loss is flat in a clamped side, so nothing trains it; the other
        // side keeps its coefficient.
        assert_eq!(terms.chosen_weight, 0.0);
        assert!(terms.rejected_weight < 0.0);

        let terms = pair_terms(&ORPO, side(-4.0, 4, 0.0), side(-8.0, 4, 0.0)).unwrap();
        assert_eq!(terms.orpo.unwrap().clamped, 0);
    }

    #[test]
    fn a_missing_reference_or_an_empty_response_is_refused() {
        let free = SideScores {
            sum: -1.0,
            tokens: 1,
            reference: None,
        };
        assert!(pair_terms(&DPO, free, free).is_err());
        assert!(pair_terms(&SIMPO, free, free).is_ok());
        assert!(pair_terms(&SIMPO, SideScores { tokens: 0, ..free }, free).is_err());
    }

    #[test]
    fn a_weight_that_does_not_fit_an_f32_is_refused() {
        assert_eq!(batch_weight(0.5).unwrap(), 0.5);
        assert!(batch_weight(1e300).is_err());
    }
}
