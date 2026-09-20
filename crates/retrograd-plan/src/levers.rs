//! Memory recovery levers in application order.
//!
//! The order *is* the policy - cheapest first, from free to destructive - and it
//! is expressed once, in [`LEVERS`], so a reader and the resolver cannot
//! disagree about it. Each lever knows three things: what it costs, which
//! configuration fields it moves (so an overridden field is never touched,
//! invariant 6), and whether it needs an explicit opt-in (invariant 4).
//!
//! A lever is applied *repeatedly* while it can still act and the estimate still
//! overflows, before the next one is tried. Exhausting a free lever before
//! reaching for a costly one is the whole point of having an order.
//!
//! Proactive defaults ([`crate::defaults`]) run first and enable what should be on
//! whether or not the run is cornered. A setting it already flipped is not a
//! sacrifice, so it is reported there and not here; when this phase pushes such
//! a setting *further*, the run really is paying, and the entry moves back into
//! `plan.levers` with its pre-2bis `from` (the resolver does that reconciliation,
//! so the two lists are always disjoint).
//!
//! Those repetitions are a search, not a decision. What the caller is told is
//! one [`AppliedLever`] per lever - its setting before and after the search,
//! because "`ubatch` 4096 -> 1" is the decision, and the twelve halvings that
//! got there are the loop that found it.

use retrograd_core::TrainConfig;

use crate::recipe::Allow;
use crate::rules::{AppliedRule, Rule};

/// One lever, what it costs and how it moves.
pub struct Lever {
    pub id: &'static str,
    /// What the run actually pays for this. Reported to the caller verbatim.
    pub cost: &'static str,
    /// What pulling the lever does, in one clause. Static, because it does not
    /// depend on how far the lever was pulled - that is what `state` renders.
    pub note: &'static str,
    /// Dotted configuration paths this lever moves. A lever whose every path is
    /// overridden is skipped, never applied against the caller's wishes.
    pub touches: &'static [&'static str],
    /// The opt-in this lever needs, if any. Without it the resolver stops and
    /// says what it *would* have done.
    pub requires: Option<Allow>,
    /// The lever's current setting, rendered. Read once before the search and
    /// once after, which is the whole report.
    state: fn(&TrainConfig, &Limits) -> String,
    /// Takes one step. `false` when the lever has nothing left to give.
    step: fn(&mut TrainConfig, &Limits) -> bool,
}

/// Narrowest token chunk the fused cross-entropy is worth splitting to.
///
/// Below this the per-chunk fixed cost - one pass over the projection head per
/// chunk - stops being amortised, and the lever would be trading throughput for
/// a term that is already small.
const MIN_CE_SEQ_CHUNK: u32 = 16;

/// The floors a lever may not cross, from constraints outside its own field.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// GRPO packs `n_seq_max = group_size` sequences and the runtime refuses
    /// `n_seq_max > n_batch`, so the batch - and everything derived from it -
    /// has a floor.
    pub min_batch: u32,
    /// Never go below this context: the run would stop covering its own data,
    /// or the group would stop fitting.
    pub min_ctx: u32,
    /// Physical width the selected packed micro-batch needs. A packed subgroup is
    /// one physical sequence: below this the fanout the candidate search costed
    /// stops fitting in the micro-batch, and the runtime would refuse the graph.
    /// One outside a packed run, where the micro-batch is a free memory lever.
    pub min_ubatch: u32,
    /// A rollout algorithm takes one optimizer step per `n_batch` tokens of a
    /// row, and a row is one rollout, so `retrograd-config` requires
    /// `n_batch == n_ctx` there. Nothing may propose a batch below the context
    /// on such a run; the context lever keeps the equality when it halves.
    pub whole_row_batch: bool,
    /// What the runtime would derive for `generation_batch` when the field is
    /// left at zero. The first lever needs it: it cannot halve a value the
    /// configuration does not spell out. Zero outside a rollout run.
    pub generation_batch_default: u32,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            min_batch: 1,
            min_ctx: 1,
            min_ubatch: 1,
            whole_row_batch: false,
            generation_batch_default: 0,
        }
    }
}

/// The ordered list. Index is the order phase 3 tries them in.
pub static LEVERS: &[Lever] = &[
    Lever {
        id: "generation_batch",
        cost: "rollout prefill throughput",
        note: "smaller prefill launches",
        touches: &["training.generation_batch"],
        requires: None,
        state: |training, limits| match (training.generation_batch, limits.generation_batch_default)
        {
            (0, 0) => "runtime-derived".to_string(),
            (0, derived) => format!("{derived} (runtime-derived)"),
            (explicit, _) => explicit.to_string(),
        },
        step: |training, limits| {
            // Zero means "let the runtime derive it". Pinning it to that same
            // derived value changes nothing about the run, and is what gives the
            // lever a number to halve on the next step.
            let current = if training.generation_batch == 0 {
                limits.generation_batch_default
            } else {
                training.generation_batch
            };
            let floor = training.generation_concurrency.max(1);
            if current == 0 || current <= floor {
                return false;
            }
            training.generation_batch = (current / 2).max(floor);
            true
        },
    },
    Lever {
        id: "generation_concurrency",
        cost: "rollout throughput",
        note: "fewer sequences decoded at once",
        touches: &["training.generation_concurrency"],
        requires: None,
        state: |training, _| training.generation_concurrency.to_string(),
        step: |training, _| {
            if training.generation_concurrency <= 1 {
                return false;
            }
            training.generation_concurrency = (training.generation_concurrency / 2).max(1);
            true
        },
    },
    Lever {
        id: "chunked_cross_entropy",
        cost: "vocabulary recompute",
        note: "the vocabulary logits are streamed tile by tile",
        touches: &[
            "training.chunked_cross_entropy",
            "training.chunked_ce_tiles",
            "training.chunked_ce_seq_chunk",
        ],
        requires: None,
        state: |training, _| {
            if !training.chunked_cross_entropy {
                return "off".to_string();
            }
            match training.chunked_ce_seq_chunk {
                0 => format!("{} tiles", training.chunked_ce_tiles),
                chunk => format!("{} tiles, {chunk}-token chunks", training.chunked_ce_tiles),
            }
        },
        step: |training, _| {
            if !training.chunked_cross_entropy {
                training.chunked_cross_entropy = true;
                training.chunked_ce_tiles = training.chunked_ce_tiles.max(8);
                return true;
            }
            // Doubling tiles roughly halves the peak logits term. Past 64 the
            // recompute dominates and the returns are not worth reporting as a
            // lever.
            if training.chunked_ce_tiles < 64 {
                training.chunked_ce_tiles = (training.chunked_ce_tiles * 2).min(64);
                return true;
            }
            // Finally narrow the token dimension, which caps the peak
            // independently of the sequence length. Zero is the unbounded case,
            // the whole step at once - so it enters the halving from the width
            // the step actually has, not from a number this lever invented.
            let chunk = match training.chunked_ce_seq_chunk {
                0 => training.n_ubatch,
                chunk => chunk.min(training.n_ubatch),
            };
            if chunk > MIN_CE_SEQ_CHUNK {
                training.chunked_ce_seq_chunk = (chunk / 2).max(MIN_CE_SEQ_CHUNK);
                return true;
            }
            false
        },
    },
    Lever {
        id: "gradient_checkpointing",
        cost: "one extra forward pass per segment",
        note: "layer activations are recomputed instead of retained",
        touches: &[
            "training.gradient_checkpointing",
            "training.checkpoint_every_n_layers",
        ],
        requires: None,
        state: |training, _| {
            if training.gradient_checkpointing {
                format!("every {} layers", training.checkpoint_every_n_layers)
            } else {
                "off".to_string()
            }
        },
        step: |training, _| {
            if !training.gradient_checkpointing {
                training.gradient_checkpointing = true;
                training.checkpoint_every_n_layers = training.checkpoint_every_n_layers.max(1);
                return true;
            }
            if training.checkpoint_every_n_layers < 8 {
                training.checkpoint_every_n_layers += 1;
                return true;
            }
            false
        },
    },
    Lever {
        id: "checkpoint_dtype",
        cost: "recompute bit-parity, ~1e-3 relative perturbation of the update",
        note: "the retained activations are halved",
        touches: &["training.checkpoint_dtype"],
        requires: None,
        state: |training, _| format!("{:?}", training.checkpoint_dtype).to_lowercase(),
        step: |training, _| {
            if !training.gradient_checkpointing
                || training.checkpoint_dtype != retrograd_core::CheckpointDtype::F32
            {
                return false;
            }
            training.checkpoint_dtype = retrograd_core::CheckpointDtype::F16;
            true
        },
    },
    Lever {
        id: "micro_batch",
        cost: "throughput",
        note: "a shorter physical micro-batch",
        touches: &["training.micro_batch"],
        requires: None,
        state: |training, _| training.n_ubatch.to_string(),
        step: |training, limits| {
            // Packed updates are transactional now: lowering the physical
            // ubatch adds passes while preserving one logical AdamW update. It
            // still may not go below the width the packed subgroup needs,
            // `limits.min_ubatch` carries what the candidate search selected.
            if training.n_ubatch <= limits.min_ubatch.max(1) {
                return false;
            }
            let next = training.n_ubatch / 2;
            if next < limits.min_ubatch || training.n_batch % next != 0 {
                return false;
            }
            training.n_ubatch = next;
            true
        },
    },
    // `context_length` is the only lever with a `requires` opt-in below: it is
    // the only one whose cost is unrecoverable. The KV-cache dtype case is
    // covered separately by the `kv_f16_may_fall_back` warning on the final
    // configuration.
    Lever {
        id: "context_length",
        cost: "examples above the new context are truncated",
        note: "longer examples are truncated",
        touches: &[
            "training.ctx",
            "training.gradient_accumulation",
            "training.micro_batch",
        ],
        requires: Some(Allow::TruncateContext),
        state: |training, _| training.n_ctx.to_string(),
        step: |training, limits| {
            let previous = training.n_ctx;
            let next = previous / 2;
            if previous <= limits.min_ctx || next < limits.min_ctx || next == 0 {
                return false;
            }
            training.n_ctx = next;
            // The divisibilities must survive the change: the batch geometry is
            // re-derived from the new context by the caller, but keeping the
            // configuration valid at every intermediate step means an estimate
            // is never taken on a shape the runtime would refuse.
            training.n_batch = training.n_batch.min(next).max(limits.min_batch.min(next));
            while training.n_batch > 0 && next % training.n_batch != 0 {
                training.n_batch -= 1;
            }
            training.n_batch = training.n_batch.max(1);
            training.n_ubatch = training.n_ubatch.min(training.n_batch).max(1);
            while training.n_ubatch > 1 && training.n_batch % training.n_ubatch != 0 {
                training.n_ubatch -= 1;
            }
            true
        },
    },
    Lever {
        id: "device_cpu",
        cost: "orders of magnitude slower - proposed, never chosen",
        note: "the run leaves the accelerator entirely",
        touches: &["model.device"],
        requires: None,
        state: |training, _| format!("{:?}", training.device).to_lowercase(),
        // Deliberately inert and never selected by the resolver: silently
        // moving a run to the CPU turns minutes into days.
        step: |_, _| false,
    },
];

/// A lever the resolver pulled, and how far.
///
/// One entry per lever, not per step: `from`/`to` bracket the whole search, so
/// twelve halvings of `ubatch` read as `4096 -> 1` instead of twelve near
/// identical lines.
pub type AppliedLever = AppliedRule;

impl Rule for Lever {
    fn touches(&self) -> &'static [&'static str] {
        self.touches
    }
}

impl Lever {
    /// The lever's current setting, as reported to the caller.
    pub fn state(&self, training: &TrainConfig, limits: &Limits) -> String {
        (self.state)(training, limits)
    }

    /// Takes one step. `false` when the lever has nothing left to give.
    pub fn step(&self, training: &mut TrainConfig, limits: &Limits) -> bool {
        (self.step)(training, limits)
    }

    /// Pulls the lever while `still_overflows` holds, and reports the net move.
    ///
    /// Exhausting one lever before reaching for a costlier one is the whole
    /// point of the order, so the loop lives here rather than at each call
    /// site - the resolver and the opt-in probe cannot disagree about how far a
    /// lever goes.
    pub fn pull_while(
        &self,
        training: &mut TrainConfig,
        limits: &Limits,
        still_overflows: &dyn Fn(&TrainConfig) -> bool,
    ) -> Option<AppliedLever> {
        let from = self.state(training, limits);
        let mut moved = false;
        // Every step strictly shrinks a bounded quantity, so the loop is finite;
        // the bound is a backstop against a future step function that is not.
        for _ in 0..64 {
            if !still_overflows(training) || !self.step(training, limits) {
                break;
            }
            moved = true;
        }
        moved.then(|| AppliedLever {
            id: self.id,
            to: self.state(training, limits),
            from,
            note: self.note,
            cost: self.cost,
        })
    }
}

/// The opt-ins that would unlock at least one lever this configuration has not
/// used yet - the "with `allow: [...]`, it fits" half of an
/// `insufficient_memory` answer.
pub fn unlocking_opt_ins(allowed: &[Allow]) -> Vec<Allow> {
    let mut out: Vec<Allow> = LEVERS
        .iter()
        .filter_map(|lever| lever.requires)
        .filter(|required| !allowed.contains(required))
        .collect();
    out.dedup();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn training() -> TrainConfig {
        TrainConfig {
            n_ctx: 1024,
            n_batch: 256,
            n_ubatch: 32,
            generation_concurrency: 8,
            ..Default::default()
        }
    }

    #[test]
    fn the_documented_order_is_the_implemented_order() {
        let ids: Vec<&str> = LEVERS.iter().map(|lever| lever.id).collect();
        assert_eq!(
            ids,
            [
                "generation_batch",
                "generation_concurrency",
                "chunked_cross_entropy",
                "gradient_checkpointing",
                "checkpoint_dtype",
                "micro_batch",
                "context_length",
                "device_cpu",
            ],
            "the phase 3 lever table moved"
        );
        // Every lever whose cost is unrecoverable declares its opt-in, and no
        // other claims one.
        for lever in LEVERS {
            let degrades = matches!(lever.id, "context_length");
            assert_eq!(
                lever.requires.is_some(),
                degrades,
                "{} disagrees with invariant 4",
                lever.id
            );
            assert!(!lever.touches.is_empty(), "{}", lever.id);
        }
    }

    fn lever(id: &str) -> &'static Lever {
        LEVERS.iter().find(|lever| lever.id == id).expect(id)
    }

    /// The lever starts from a configuration that already has the fused path on,
    /// since that is the default now. Turning it back on is still the first step
    /// it takes, because a client can have pinned it off and then failed to fit.
    #[test]
    fn chunked_cross_entropy_walks_from_off_to_tiles_to_token_chunks() {
        let mut config = TrainConfig {
            chunked_cross_entropy: false,
            ..training()
        };
        let limits = Limits::default();
        let cce = lever("chunked_cross_entropy");
        assert_eq!(cce.state(&config, &limits), "off");
        assert!(cce.step(&mut config, &limits));
        assert!(config.chunked_cross_entropy);
        assert_eq!(config.chunked_ce_tiles, 8);

        // Tiles double until they saturate, then the token chunk narrows.
        let mut steps = 0;
        while cce.step(&mut config, &limits) {
            steps += 1;
            assert!(steps < 16, "the lever never runs out");
        }
        assert_eq!(config.chunked_ce_tiles, 64);
        // The chunk halves from the micro-batch, not from the 512-token default:
        // a 32-token step never has more than 32 tokens to split.
        assert_eq!(config.chunked_ce_seq_chunk, MIN_CE_SEQ_CHUNK);
        assert_eq!(cce.state(&config, &limits), "64 tiles, 16-token chunks");
    }

    /// A wide step is where the token axis is worth several halvings - this is
    /// the case the default (512) is sized for, and the one this lever must
    /// handle when `seq_chunk == 0`.
    #[test]
    fn the_token_chunk_halves_down_from_a_wide_micro_batch() {
        let mut config = TrainConfig {
            n_ubatch: 1024,
            chunked_ce_tiles: 64,
            ..training()
        };
        let limits = Limits::default();
        let cce = lever("chunked_cross_entropy");
        let mut widths = vec![config.chunked_ce_seq_chunk];
        while cce.step(&mut config, &limits) {
            widths.push(config.chunked_ce_seq_chunk);
        }
        assert_eq!(widths, [512, 256, 128, 64, 32, 16]);
    }

    #[test]
    fn the_micro_batch_lever_is_a_physical_choice_for_packed_updates() {
        let mut config = training();
        assert!(lever("micro_batch").step(&mut config, &Limits::default()));
        assert_eq!(config.n_ubatch, 16);
    }

    #[test]
    fn halving_the_context_keeps_the_divisibilities() {
        let mut config = training();
        let limits = Limits {
            min_ctx: 128,
            ..Limits::default()
        };
        while lever("context_length").step(&mut config, &limits) {
            assert_eq!(config.n_ctx % config.n_batch, 0, "{config:?}");
            assert_eq!(config.n_batch % config.n_ubatch, 0, "{config:?}");
        }
        assert_eq!(config.n_ctx, 128, "it stops at the floor");
    }

    #[test]
    fn a_locked_field_makes_its_lever_unavailable() {
        let locked = |path: &str| path == "training.checkpoint_dtype";
        assert!(lever("checkpoint_dtype").is_locked_by(&locked));
        assert!(!lever("gradient_checkpointing").is_locked_by(&locked));
        // A lever with several fields is only locked when *all* of them are.
        let partial = |path: &str| path == "training.chunked_ce_tiles";
        assert!(!lever("chunked_cross_entropy").is_locked_by(&partial));
    }

    #[test]
    fn moving_to_the_cpu_is_offered_but_never_taken() {
        let mut config = training();
        assert!(!lever("device_cpu").step(&mut config, &Limits::default()));
        assert_eq!(config.device, retrograd_core::Device::Auto);
    }

    #[test]
    fn the_unlocking_opt_ins_are_the_ones_not_already_granted() {
        assert_eq!(unlocking_opt_ins(&[]), vec![Allow::TruncateContext]);
        assert!(unlocking_opt_ins(&[Allow::TruncateContext]).is_empty());
    }
}
