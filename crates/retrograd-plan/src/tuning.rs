//! The derivation table.
//!
//! One function per row. Each one takes an *observable* - the number of
//! examples, the token count, the VRAM budget, the width of the generation KV
//! cache - and returns a value together with the sentence that explains it. That
//! pairing is the point: the reason the client reads in `provenance` is written
//! at the place the decision is made, so the two cannot disagree.
//!
//! It derives values from explicit inputs rather than from a named profile.
//! A profile is a dict of `params` a client sends; the server knows none.
//!
//! Every threshold below is an order of magnitude, not a measurement - that is
//! said again in [`DERIVATIONS`], which is what `GET /v1/defaults` renders, so a
//! client reading the rules reads the same caveat.

use retrograd_core::{LoraConfig, ModelInfo, TargetSet};

/// A derived value and the one sentence that justifies it.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice<T> {
    pub value: T,
    pub reason: String,
}

fn choose<T>(value: T, reason: impl Into<String>) -> Choice<T> {
    Choice {
        value,
        reason: reason.into(),
    }
}

/// The observables the rules read. Everything here comes out of phase 0.
#[derive(Clone, Copy, Debug)]
pub struct Facts {
    pub examples: u64,
    /// Summed example lengths. Zero for a text corpus, whose "examples" are a
    /// windowing artefact rather than a count.
    pub total_tokens: u64,
    /// The effective VRAM budget, margin already taken out.
    pub vram_bytes: u64,
}

// ---------------------------------------------------------------------------
// LoRA
// ---------------------------------------------------------------------------

/// Examples below which the targets stay narrow and the rank low: the extra
/// parameters would mostly memorise.
pub const WIDE_TARGET_EXAMPLE_THRESHOLD: u64 = 2_000;

/// The example counts the rank steps at.
const RANK_STEPS: [(u64, u32); 4] = [(2_000, 8), (20_000, 16), (200_000, 32), (u64::MAX, 64)];

/// Share of the VRAM budget the LoRA parameters may occupy before the rank is
/// capped. Two percent: the adapter is the thing being trained, so it is meant
/// to be a rounding error next to the weights and the KV cache - a rank that is
/// not costs throughput for nothing.
const RANK_PARAMETER_BUDGET_FRACTION: u64 = 2;

/// `q`/`v` below the threshold, the architecture's automatic set above it.
///
/// An empty vector means `auto`, resolved per architecture by the runtime.
pub fn targets(facts: &Facts) -> Choice<Vec<String>> {
    if facts.examples >= WIDE_TARGET_EXAMPLE_THRESHOLD {
        choose(
            Vec::new(),
            format!(
                "{} examples: the architecture's automatic target set",
                facts.examples
            ),
        )
    } else {
        choose(
            vec!["q".to_string(), "v".to_string()],
            format!(
                "{} examples, below {WIDE_TARGET_EXAMPLE_THRESHOLD}: q and v only",
                facts.examples
            ),
        )
    }
}

/// The rank, from the amount of data, capped so the adapter stays small.
///
/// `bytes_per_rank` is what one unit of rank costs in LoRA parameters on this
/// model and this target set - computed by the caller with the cost model, so
/// the cap and the estimate cannot disagree.
pub fn rank(facts: &Facts, bytes_per_rank: u64) -> Choice<u32> {
    let (_, wanted) = RANK_STEPS
        .iter()
        .copied()
        .find(|(threshold, _)| facts.examples < *threshold)
        .unwrap_or((u64::MAX, 64));
    let allowance = facts.vram_bytes / 100 * RANK_PARAMETER_BUDGET_FRACTION;
    // No target set costs nothing per unit of rank, so a zero here is a caller
    // that has not measured rather than a free adapter: leave the rank uncapped.
    let cap = allowance
        .checked_div(bytes_per_rank)
        .map(|cap| cap.clamp(1, u32::MAX as u64) as u32)
        .unwrap_or(u32::MAX);
    // Powers of two only: the runtime is happier with them and a rank of 27
    // would suggest a precision this rule does not have.
    let capped = previous_power_of_two(wanted.min(cap)).max(1);
    if capped < wanted {
        choose(
            capped,
            format!(
                "{} examples would take rank {wanted}, lowered to {capped} to keep the \
                 adapter inside {RANK_PARAMETER_BUDGET_FRACTION}% of the VRAM budget",
                facts.examples
            ),
        )
    } else {
        choose(
            capped,
            format!("{} examples: rank {capped}", facts.examples),
        )
    }
}

/// `2 × rank`, the scaling every LoRA recipe in the literature starts from.
pub fn alpha(rank: u32) -> Choice<f32> {
    choose(rank as f32 * 2.0, "twice the rank")
}

/// What one unit of rank costs in adapter parameters, in bytes.
pub fn bytes_per_rank(
    model: &ModelInfo,
    targets: &TargetSet,
    dtype: retrograd_core::LoraDtype,
) -> u64 {
    let probe = LoraConfig {
        rank: 1,
        targets: targets.clone(),
        dtype,
        ..LoraConfig::auto(1, 2.0)
    };
    let parameters = crate::cost::trainable_parameters(model, &probe);
    parameters
        * match dtype {
            retrograd_core::LoraDtype::F32 => 4,
            retrograd_core::LoraDtype::F16 => 2,
        }
}

// ---------------------------------------------------------------------------
// Optimisation
// ---------------------------------------------------------------------------

/// Learning rate at the reference rank, and the window it is held in.
const REFERENCE_RANK: f32 = 16.0;
const REFERENCE_LR: f32 = 1.0e-4;
const LR_FLOOR: f32 = 2.0e-5;
const LR_CEILING: f32 = 3.0e-4;

/// `1e-4 × sqrt(16 / rank)`, bounded.
///
/// A wider adapter takes larger steps at the same nominal rate, because the
/// update is a sum over more parameters; scaling the rate by the square root of
/// the width is the standard correction, and the bounds keep an extreme rank
/// from producing a rate nobody would choose by hand.
pub fn learning_rate(rank: u32) -> Choice<f32> {
    let rank = rank.max(1) as f32;
    let raw = REFERENCE_LR * (REFERENCE_RANK / rank).sqrt();
    let value = raw.clamp(LR_FLOOR, LR_CEILING);
    if value != raw {
        choose(
            value,
            format!(
                "rank {rank:.0}: the scaled rate is held inside [{LR_FLOOR:e}, {LR_CEILING:e}]"
            ),
        )
    } else if rank as u32 == REFERENCE_RANK as u32 {
        choose(value, format!("rank {rank:.0}: the reference rate"))
    } else {
        choose(
            value,
            format!("rank {rank:.0}: the reference rate scaled by sqrt(16 / rank)"),
        )
    }
}

/// Steps below which a cosine decay has no room to descend.
pub const COSINE_STEP_THRESHOLD: u64 = 100;

pub fn lr_scheduler(total_steps: u64) -> Choice<&'static str> {
    if total_steps >= COSINE_STEP_THRESHOLD {
        choose(
            "cosine",
            format!("{total_steps} optimizer steps: a cosine has room to decay"),
        )
    } else {
        choose(
            "constant",
            format!(
                "{total_steps} optimizer steps, below {COSINE_STEP_THRESHOLD}: \
                 a cosine would not have time to descend"
            ),
        )
    }
}

/// Examples below which decay regularisation costs more than it buys.
pub const WEIGHT_DECAY_EXAMPLE_THRESHOLD: u64 = 500;

pub fn weight_decay(facts: &Facts) -> Choice<f32> {
    if facts.examples < WEIGHT_DECAY_EXAMPLE_THRESHOLD {
        choose(
            0.0,
            format!(
                "{} examples, below {WEIGHT_DECAY_EXAMPLE_THRESHOLD}: no decay regularisation",
                facts.examples
            ),
        )
    } else {
        choose(
            0.01,
            format!("{} examples: the usual decay", facts.examples),
        )
    }
}

// ---------------------------------------------------------------------------
// How much training
// ---------------------------------------------------------------------------

/// Supervised tokens a run aims to expose the adapter to, before the epoch count
/// is clamped. An order of magnitude: enough for a small adapter to converge on
/// a narrow task, short of the point where a LoRA on a few thousand examples
/// starts memorising them.
pub const TOKEN_EXPOSURE_TARGET: u64 = 1_500_000;
const MAX_EPOCHS: u32 = 8;

pub fn epochs(facts: &Facts) -> Choice<u32> {
    if facts.total_tokens == 0 {
        return choose(1, "no token count available: one pass");
    }
    let wanted = TOKEN_EXPOSURE_TARGET.div_ceil(facts.total_tokens);
    let value = wanted.clamp(1, MAX_EPOCHS as u64) as u32;
    choose(
        value,
        format!(
            "{} tokens: {value} pass(es) towards the {TOKEN_EXPOSURE_TARGET}-token \
             exposure target",
            facts.total_tokens
        ),
    )
}

const MIN_UPDATES: u32 = 20;
const MAX_UPDATES: u32 = 500;

/// Two passes over the prompt set, bounded.
pub fn updates(prompts: u64, prompts_per_update: usize) -> Choice<u32> {
    let per_update = prompts_per_update.max(1) as u64;
    let passes = prompts.max(1).div_ceil(per_update) * 2;
    let value = passes.clamp(MIN_UPDATES as u64, MAX_UPDATES as u64) as u32;
    choose(
        value,
        format!("two passes over {prompts} prompts, {per_update} at a time"),
    )
}

/// Inner epochs: two, one when the run is long enough that re-using a rollout
/// twice mostly amplifies its noise.
pub const INNER_EPOCH_UPDATE_THRESHOLD: u32 = 200;

pub fn inner_epochs(updates: u32) -> Choice<u32> {
    if updates > INNER_EPOCH_UPDATE_THRESHOLD {
        choose(
            1,
            format!(
                "{updates} updates, above {INNER_EPOCH_UPDATE_THRESHOLD}: \
                 one inner pass per rollout"
            ),
        )
    } else {
        choose(
            2,
            format!("{updates} updates: two inner passes per rollout"),
        )
    }
}

// ---------------------------------------------------------------------------
// Rollout width
// ---------------------------------------------------------------------------

/// Smallest group where the Dr. GRPO baseline is stable.
pub const PREFERRED_GROUP_SIZE: usize = 8;
/// A group of one has no baseline at all.
const MIN_GROUP_SIZE: usize = 2;

/// Share of the VRAM budget the *generation* KV cache may take.
///
/// A judgement call. The generation context is a second cache next to the
/// optimizer's, and it is the term that grows with the group size, so it needs a
/// ceiling that is not the whole budget - otherwise a group of 16 crowds out the
/// activations it exists to produce.
pub const GENERATION_KV_BUDGET_FRACTION: u64 = 25;

/// The largest number of sequences that can decode at once inside the
/// generation-KV allowance, as a power of two.
///
/// `element_bytes` is the KV element width the generation context will use - two
/// when `fast_sampling_context` is on, which is the default for a rollout
/// objective.
pub fn generation_capacity(
    model: &ModelInfo,
    n_ctx: u32,
    vram_bytes: u64,
    element_bytes: u64,
) -> u32 {
    let allowance = vram_bytes / 100 * GENERATION_KV_BUDGET_FRACTION;
    let mut sequences = 256u32;
    while sequences > 1 {
        let bytes = model.kv_cache_bytes(n_ctx as u64 * sequences as u64, element_bytes);
        if bytes <= allowance {
            return sequences;
        }
        sequences /= 2;
    }
    1
}

/// Narrowest micro-batch worth launching on a discrete accelerator.
///
/// On the packed rollout path `micro_batch` is not a memory lever - it is the
/// physical forward/backward width, while the optimizer window stays `ctx`,
/// because a rollout pins `gradient_accumulation` to `ctx / micro_batch`. So a
/// narrow value does not buy memory; it just cuts the same work into more
/// launches. Measured on a 4090 at 32 tokens: 137 W of 450, ~4% of memory
/// bandwidth, 100% `sm` occupancy - the profile of a device waiting on kernel
/// launches rather than computing.
pub const MIN_DISCRETE_MICRO_BATCH: u32 = 512;

/// Initial micro-batch for a run whose device memory is shared with everything else.
///
/// The minimum width for the unified case, where activations share a pool with
/// weights and caches, so narrowing the micro-batch reduces the peak.
pub const MIN_UNIFIED_MICRO_BATCH: u32 = 32;

/// The packed micro-batch a rollout starts from, before any lock.
///
/// `n_ctx` caps it - a micro-batch wider than the context has nothing to put in
/// the remainder - and both widths are powers of two, which keeps the
/// `ctx % micro_batch == 0` divisibility the loader checks.
pub fn packed_micro_batch(n_ctx: u32, discrete: bool) -> u32 {
    let width = if discrete {
        MIN_DISCRETE_MICRO_BATCH
    } else {
        MIN_UNIFIED_MICRO_BATCH
    };
    n_ctx.min(width)
}

/// Eight, lowered to the largest power of two whose group fits the generation KV.
pub fn group_size(capacity: u32) -> Choice<usize> {
    let fitting = previous_power_of_two(capacity.max(1)) as usize;
    let value = PREFERRED_GROUP_SIZE.min(fitting).max(MIN_GROUP_SIZE);
    if value > fitting {
        // The floor won over the capacity. Saying "the largest group that fits"
        // here would be false, and the run really will spill past the 25% share.
        choose(
            value,
            format!(
                "{MIN_GROUP_SIZE}: the generation KV cache holds only {fitting} sequence(s) in \
                 {GENERATION_KV_BUDGET_FRACTION}% of the VRAM budget, but a group of one has no \
                 Dr. GRPO baseline"
            ),
        )
    } else if value < PREFERRED_GROUP_SIZE {
        choose(
            value,
            format!(
                "{value}: the largest group whose generation KV cache fits \
                 {GENERATION_KV_BUDGET_FRACTION}% of the VRAM budget"
            ),
        )
    } else {
        choose(
            value,
            format!(
                "{PREFERRED_GROUP_SIZE}: the smallest group where the Dr. GRPO baseline is stable"
            ),
        )
    }
}

const MIN_PROMPTS_PER_UPDATE: usize = 2;
const MAX_PROMPTS_PER_UPDATE: usize = 16;

/// The largest `p` whose `p × group_size` rollouts still fit the generation
/// concurrency the KV allowance permits.
pub fn prompts_per_update(group_size: usize, capacity: u32) -> Choice<usize> {
    let affordable = (capacity as usize / group_size.max(1)).max(1);
    let value = affordable.clamp(MIN_PROMPTS_PER_UPDATE, MAX_PROMPTS_PER_UPDATE);
    choose(
        value,
        format!(
            "{value} prompts: beyond that the {group_size}-wide groups overflow the \
             generation cache"
        ),
    )
}

// ---------------------------------------------------------------------------
// Cadences
// ---------------------------------------------------------------------------

/// Evaluations over the whole run. Ten: often enough to see a curve bend,
/// seldom enough that evaluation is not most of the wall clock.
pub const EVALUATIONS_PER_RUN: u64 = 10;

pub fn eval_every(iterations: u64) -> Choice<u32> {
    let value = (iterations / EVALUATIONS_PER_RUN).max(1);
    choose(
        value.clamp(1, u32::MAX as u64) as u32,
        format!("{EVALUATIONS_PER_RUN} evaluations over the run's {iterations} iterations"),
    )
}

/// Evaluations without improvement before a run stops on its own.
pub const PATIENCE_EVALUATIONS: u32 = 5;

pub fn patience() -> Choice<u32> {
    choose(
        PATIENCE_EVALUATIONS,
        format!("{PATIENCE_EVALUATIONS} evaluations without a gain"),
    )
}

/// Checkpoints over the whole run.
pub const CHECKPOINTS_PER_RUN: u64 = 10;

pub fn checkpoint_every(total_steps: u64) -> Choice<u64> {
    let value = (total_steps / CHECKPOINTS_PER_RUN).max(1);
    choose(
        value,
        format!("{CHECKPOINTS_PER_RUN} checkpoints over the run's {total_steps} steps"),
    )
}

/// Share of the total steps spent warming the learning rate up.
pub const WARMUP_FRACTION: f64 = 0.03;

pub fn warmup_steps(total_steps: u64) -> Choice<u64> {
    let value = ((total_steps as f64) * WARMUP_FRACTION).round() as u64;
    choose(
        value,
        format!(
            "{:.0}% of the {total_steps} optimizer steps",
            WARMUP_FRACTION * 100.0
        ),
    )
}

/// Held-out examples one rollout evaluation scores, capped at how many the
/// eval dataset actually has: asking for more than
/// exist would silently repeat prompts rather than say so, and a plan whose
/// `max_examples` outnumbers the eval set is a number nobody can act on.
///
/// `eval_examples = 0` means the caller has not measured the eval set yet
/// (`ResolveInput.eval` absent even though a rollout evaluation is
/// configured); the four-updates rule is then the whole answer.
pub fn eval_max_examples(prompts_per_update: usize, eval_examples: u64) -> Choice<usize> {
    let wanted = prompts_per_update * 4;
    if eval_examples == 0 || (eval_examples as usize) >= wanted {
        return choose(
            wanted,
            format!("four updates' worth of prompts ({prompts_per_update} per update)"),
        );
    }
    let value = (eval_examples as usize).max(1);
    choose(
        value,
        format!(
            "four updates' worth of prompts ({prompts_per_update} per update) capped at \
             the {eval_examples} available eval examples"
        ),
    )
}

// ---------------------------------------------------------------------------
// The table `GET /v1/defaults` renders
// ---------------------------------------------------------------------------

/// Which algorithms a row applies to, so a client - and the test that checks
/// this table against a real resolution - knows where to look for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scope {
    /// Every objective.
    Any,
    /// Supervised only.
    Sft,
    /// GRPO (`reasoning-rl`, `agentic`).
    Grpo,
    /// PPO (`preference-rl`).
    Ppo,
    /// Both rollout algorithms.
    Rollout,
    /// Only when the recipe carries an `eval` dataset.
    Evaluation,
    /// Only when the recipe carries a `checkpoint_dir`.
    Checkpoint,
}

/// One row of the derivation table: a field the resolver derives, and the rule it derives it with.
#[derive(Clone, Copy, Debug)]
pub struct Derivation {
    /// Dotted document path, the grammar `params`, `provenance` and the `PATCH`
    /// whitelist all speak.
    pub path: &'static str,
    /// The rule, in one sentence.
    pub rule: &'static str,
    /// The numbers in that sentence, named. A client building its own profile
    /// wants the thresholds, not only the prose.
    pub thresholds: &'static [(&'static str, f64)],
    pub scope: Scope,
}

/// Every derivable field, with its rule. Rendered by `GET /v1/defaults`, and
/// checked against a real resolution by
/// `tests/derivation_table.rs::every_derivable_field_is_documented`.
pub static DERIVATIONS: &[Derivation] = &[
    Derivation {
        path: "training.ctx",
        rule: "the 99th percentile of the example lengths, rounded up to a power of two \
               and capped at the model's trained context",
        thresholds: &[("percentile", 0.99)],
        scope: Scope::Any,
    },
    Derivation {
        path: "lora.rank",
        rule: "8 below 2 000 examples, 16 below 20 000, 32 below 200 000, 64 above; \
               lowered so the adapter stays inside 2% of the VRAM budget",
        thresholds: &[
            ("examples_rank_8", 2_000.0),
            ("examples_rank_16", 20_000.0),
            ("examples_rank_32", 200_000.0),
            (
                "parameter_budget_percent",
                RANK_PARAMETER_BUDGET_FRACTION as f64,
            ),
        ],
        scope: Scope::Any,
    },
    Derivation {
        path: "lora.alpha",
        rule: "twice the rank",
        thresholds: &[("multiple_of_rank", 2.0)],
        scope: Scope::Any,
    },
    Derivation {
        path: "lora.targets",
        rule: "q and v below 2 000 examples, the architecture's automatic set above",
        thresholds: &[("examples", WIDE_TARGET_EXAMPLE_THRESHOLD as f64)],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.lr",
        rule: "1e-4 scaled by sqrt(16 / rank), held inside [2e-5, 3e-4]",
        thresholds: &[
            ("reference_rank", REFERENCE_RANK as f64),
            ("reference_lr", REFERENCE_LR as f64),
            ("floor", LR_FLOOR as f64),
            ("ceiling", LR_CEILING as f64),
        ],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.lr_scheduler",
        rule: "cosine from 100 optimizer steps up, constant below",
        thresholds: &[("total_steps", COSINE_STEP_THRESHOLD as f64)],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.warmup_steps",
        rule: "3% of the optimizer steps",
        thresholds: &[("fraction", WARMUP_FRACTION)],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.weight_decay",
        rule: "0.01, and 0.0 below 500 examples",
        thresholds: &[
            ("examples", WEIGHT_DECAY_EXAMPLE_THRESHOLD as f64),
            ("decay", 0.01),
        ],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.epochs",
        rule: "ceil(1.5e6 / total_tokens), clamped to [1, 8]; ignored when budget.epochs \
               or budget.minutes says otherwise",
        thresholds: &[
            ("token_exposure_target", TOKEN_EXPOSURE_TARGET as f64),
            ("max_epochs", MAX_EPOCHS as f64),
        ],
        scope: Scope::Sft,
    },
    Derivation {
        path: "training.max_grad_norm",
        rule: "left at the engine's own default of 1.0 - nothing observable would \
               justify moving it",
        thresholds: &[("value", 1.0)],
        scope: Scope::Any,
    },
    Derivation {
        path: "lora.dtype",
        rule: "left at the engine's own default (f16)",
        thresholds: &[],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.gradient_accumulation",
        rule: "the largest optimizer window (micro_batch x gradient_accumulation) whose \
               estimate fits the budget with its safety margin",
        thresholds: &[],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.micro_batch",
        rule: "divides the resolved optimizer window; on a packed rollout it is a physical \
               width jointly selected with prefix fanout and checkpointing",
        thresholds: &[
            ("discrete_tokens", MIN_DISCRETE_MICRO_BATCH as f64),
            ("unified_tokens", MIN_UNIFIED_MICRO_BATCH as f64),
        ],
        scope: Scope::Any,
    },
    Derivation {
        path: "training.shared_prefix_fanout",
        rule: "the non-dominated physical prompt-sharing width with the lowest predicted \
               execution cost at the highest allowed fidelity",
        thresholds: &[],
        scope: Scope::Grpo,
    },
    Derivation {
        path: "sampling.max_new_tokens",
        rule: "half the resolved context, so a prompt and its completion both fit",
        thresholds: &[("fraction_of_context", 0.5)],
        scope: Scope::Rollout,
    },
    Derivation {
        path: "grpo.updates",
        rule: "two passes over the prompt set, clamped to [20, 500]; ignored when \
               budget.updates says otherwise",
        thresholds: &[
            ("min", MIN_UPDATES as f64),
            ("max", MAX_UPDATES as f64),
            ("passes", 2.0),
        ],
        scope: Scope::Grpo,
    },
    Derivation {
        path: "ppo.updates",
        rule: "two passes over the prompt set, clamped to [20, 500]; ignored when \
               budget.updates says otherwise",
        thresholds: &[
            ("min", MIN_UPDATES as f64),
            ("max", MAX_UPDATES as f64),
            ("passes", 2.0),
        ],
        scope: Scope::Ppo,
    },
    Derivation {
        path: "grpo.group_size",
        rule: "8, lowered to the largest power of two whose generation KV cache fits \
               25% of the VRAM budget",
        thresholds: &[
            ("preferred", PREFERRED_GROUP_SIZE as f64),
            (
                "generation_kv_budget_percent",
                GENERATION_KV_BUDGET_FRACTION as f64,
            ),
        ],
        scope: Scope::Grpo,
    },
    Derivation {
        path: "grpo.prompts_per_update",
        rule: "the largest p whose p × group_size rollouts fit the generation \
               concurrency, clamped to [2, 16]",
        thresholds: &[
            ("min", MIN_PROMPTS_PER_UPDATE as f64),
            ("max", MAX_PROMPTS_PER_UPDATE as f64),
        ],
        scope: Scope::Grpo,
    },
    Derivation {
        path: "ppo.rollout_batch_size",
        rule: "the largest p whose p rollouts fit the generation concurrency, \
               clamped to [2, 16]",
        thresholds: &[
            ("min", MIN_PROMPTS_PER_UPDATE as f64),
            ("max", MAX_PROMPTS_PER_UPDATE as f64),
        ],
        scope: Scope::Ppo,
    },
    Derivation {
        path: "grpo.grpo_epochs",
        rule: "2, and 1 above 200 updates",
        thresholds: &[("updates", INNER_EPOCH_UPDATE_THRESHOLD as f64)],
        scope: Scope::Grpo,
    },
    Derivation {
        path: "ppo.ppo_epochs",
        rule: "2, and 1 above 200 updates",
        thresholds: &[("updates", INNER_EPOCH_UPDATE_THRESHOLD as f64)],
        scope: Scope::Ppo,
    },
    Derivation {
        path: "evaluation.every_iterations",
        rule: "ten evaluations over the run",
        thresholds: &[("evaluations_per_run", EVALUATIONS_PER_RUN as f64)],
        scope: Scope::Evaluation,
    },
    Derivation {
        path: "evaluation.patience",
        rule: "five evaluations without a gain",
        thresholds: &[("evaluations", PATIENCE_EVALUATIONS as f64)],
        scope: Scope::Evaluation,
    },
    Derivation {
        path: "evaluation.max_examples",
        rule: "four updates' worth of prompts, for a rollout objective",
        thresholds: &[("updates_worth", 4.0)],
        scope: Scope::Rollout,
    },
    Derivation {
        path: "checkpoint.mode",
        rule: "steps and best evaluation when held-out data is available, steps alone \
               otherwise",
        thresholds: &[],
        scope: Scope::Checkpoint,
    },
    Derivation {
        path: "checkpoint.every_steps",
        rule: "ten checkpoints over the run",
        thresholds: &[("checkpoints_per_run", CHECKPOINTS_PER_RUN as f64)],
        scope: Scope::Checkpoint,
    },
];

/// Largest power of two not above `value`.
fn previous_power_of_two(value: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    1u32 << (u32::BITS - 1 - value.leading_zeros())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(examples: u64, total_tokens: u64) -> Facts {
        Facts {
            examples,
            total_tokens,
            vram_bytes: 24 * 1024 * 1024 * 1024,
        }
    }

    /// One case per threshold boundary of the derivation table, on both sides of it.
    #[test]
    fn the_rank_steps_exactly_where_the_table_says() {
        for (examples, expected) in [
            (1, 8),
            (1_999, 8),
            (2_000, 16),
            (19_999, 16),
            (20_000, 32),
            (199_999, 32),
            (200_000, 64),
            (5_000_000, 64),
        ] {
            assert_eq!(
                rank(&facts(examples, 1), 0).value,
                expected,
                "{examples} examples"
            );
        }
    }

    #[test]
    fn a_rank_that_would_not_stay_small_is_capped() {
        // A tiny budget and an expensive rank unit: the cap has to bite, and it
        // has to say so.
        let small = Facts {
            examples: 500_000,
            total_tokens: 1,
            vram_bytes: 1024 * 1024,
        };
        let capped = rank(&small, 4096);
        assert!(capped.value < 64, "{capped:?}");
        assert!(capped.reason.contains("lowered"), "{capped:?}");
        assert!(capped.value.is_power_of_two());
    }

    #[test]
    fn the_targets_widen_at_the_documented_boundary() {
        assert_eq!(targets(&facts(1_999, 1)).value, vec!["q", "v"]);
        assert!(
            targets(&facts(2_000, 1)).value.is_empty(),
            "an empty target list is `auto`"
        );
    }

    #[test]
    fn the_rate_falls_as_the_rank_grows_and_stays_inside_its_window() {
        let rates: Vec<f32> = [8, 16, 32, 64]
            .into_iter()
            .map(|rank| learning_rate(rank).value)
            .collect();
        for pair in rates.windows(2) {
            assert!(pair[0] > pair[1], "{rates:?} is not decreasing");
        }
        assert_eq!(learning_rate(16).value, REFERENCE_LR);
        for rank in [1u32, 2, 4, 8, 16, 32, 64, 256, 1024] {
            let rate = learning_rate(rank).value;
            assert!(
                (LR_FLOOR..=LR_CEILING).contains(&rate),
                "rank {rank}: {rate}"
            );
        }
    }

    #[test]
    fn the_scheduler_switches_at_a_hundred_steps() {
        assert_eq!(lr_scheduler(99).value, "constant");
        assert_eq!(lr_scheduler(100).value, "cosine");
    }

    #[test]
    fn the_decay_switches_at_five_hundred_examples() {
        assert_eq!(weight_decay(&facts(499, 1)).value, 0.0);
        assert_eq!(weight_decay(&facts(500, 1)).value, 0.01);
    }

    #[test]
    fn the_epoch_count_falls_as_the_corpus_grows() {
        let counts: Vec<u32> = [50_000u64, 500_000, 1_500_000, 15_000_000]
            .into_iter()
            .map(|tokens| epochs(&facts(1_000, tokens)).value)
            .collect();
        assert_eq!(counts, [8, 3, 1, 1]);
        for pair in counts.windows(2) {
            assert!(pair[0] >= pair[1], "{counts:?} is not monotone");
        }
    }

    #[test]
    fn the_update_count_is_two_passes_inside_its_bounds() {
        assert_eq!(updates(96, 4).value, 48);
        assert_eq!(updates(4, 4).value, MIN_UPDATES, "the floor holds");
        assert_eq!(updates(100_000, 4).value, MAX_UPDATES, "the ceiling holds");
    }

    #[test]
    fn the_inner_pass_count_drops_on_a_long_run() {
        assert_eq!(inner_epochs(200).value, 2);
        assert_eq!(inner_epochs(201).value, 1);
    }

    #[test]
    fn the_group_never_falls_below_a_usable_baseline() {
        assert_eq!(group_size(64).value, PREFERRED_GROUP_SIZE);
        assert_eq!(group_size(4).value, 4);
        assert_eq!(
            group_size(1).value,
            MIN_GROUP_SIZE,
            "a group of one has no baseline"
        );
    }

    #[test]
    fn the_prompt_count_fills_the_generation_concurrency() {
        assert_eq!(prompts_per_update(8, 32).value, 4);
        assert_eq!(prompts_per_update(8, 4).value, MIN_PROMPTS_PER_UPDATE);
        assert_eq!(prompts_per_update(2, 256).value, MAX_PROMPTS_PER_UPDATE);
    }

    #[test]
    fn the_cadences_divide_the_run_into_ten() {
        assert_eq!(eval_every(100).value, 10);
        assert_eq!(eval_every(3).value, 1, "never zero");
        assert_eq!(checkpoint_every(1_000).value, 100);
        assert_eq!(checkpoint_every(3).value, 1);
        assert_eq!(warmup_steps(1_000).value, 30);
    }

    #[test]
    fn the_generation_capacity_shrinks_with_the_budget() {
        let model = crate::testing::tiny_model();
        let roomy = generation_capacity(&model, 1024, 24 * 1024 * 1024 * 1024, 2);
        let tight = generation_capacity(&model, 1024, 512 * 1024 * 1024, 2);
        assert!(roomy >= tight, "{roomy} < {tight}");
        assert!(roomy.is_power_of_two() && tight.is_power_of_two());
        assert!(tight >= 1);
    }

    #[test]
    fn every_documented_rule_names_a_dotted_path_and_a_sentence() {
        let mut seen = std::collections::BTreeSet::new();
        for derivation in DERIVATIONS {
            assert!(
                seen.insert(derivation.path),
                "{} appears twice",
                derivation.path
            );
            assert!(
                derivation.path.contains('.')
                    && !derivation.path.starts_with('.')
                    && !derivation.path.ends_with('.'),
                "'{}' is not a dotted field path",
                derivation.path
            );
            assert!(!derivation.rule.is_empty(), "{}", derivation.path);
            for (name, value) in derivation.thresholds {
                assert!(!name.is_empty(), "{}", derivation.path);
                assert!(
                    value.is_finite(),
                    "{} has a non-finite threshold",
                    derivation.path
                );
            }
        }
    }

    #[test]
    fn the_power_of_two_helper_never_overshoots() {
        assert_eq!(previous_power_of_two(0), 0);
        assert_eq!(previous_power_of_two(1), 1);
        assert_eq!(previous_power_of_two(7), 4);
        assert_eq!(previous_power_of_two(8), 8);
        assert_eq!(previous_power_of_two(u32::MAX), 1 << 31);
    }
}
