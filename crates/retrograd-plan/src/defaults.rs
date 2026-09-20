//! Settings that should be active even when the estimate fits.
//!
//! Memory recovery ([`crate::levers`]) moves only while the estimate
//! overflows, and everything it does costs the run something. This phase also
//! examines optional allocations that would otherwise be materialized simply
//! because the run fits.
//!
//! This phase closes it. Same shape as `levers.rs` on purpose - an ordered list,
//! `touches`, per-field lock checking - so the two phases cannot disagree about
//! what an override means. Two things differ:
//!
//! - a default is applied because it is *better*, not because the run is
//!   cornered, so it has a **condition** instead of a budget test;
//! - what it did is reported in `plan.defaults_applied`, never in `plan.levers`.
//!   "What I turned on because it is better" and "what I had to sacrifice to
//!   fit" must not read alike.
//!
//! One of the four is not numerically inert: `fast_sampling_context` changes the
//! sampling distribution. It is a default anyway, and the only one allowed to
//! change what is computed - so it carries a warning code the client can match
//! on, and a single `params` field turns it off.

use retrograd_core::TrainConfig;

use crate::Backend;
use crate::rules::{AppliedRule, Rule};

/// What a default's condition decided.
///
/// [`Verdict::Unavailable`] has no active producer. It remains available for a
/// backend that lacks an op this phase would reach for; the honest answer then
/// is the figure it would have saved, not silence or a CPU fallback.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// The condition holds: apply it.
    Apply,
    /// The condition does not hold. Nothing to report.
    Skip,
    /// The condition holds and this backend cannot honour the setting. The
    /// message says what it would have saved - applying it anyway would be a
    /// silent CPU fallback, and saying nothing would hide the cost.
    Unavailable(String),
}

/// Everything a condition reads. Rebuilt before each default is considered, so a
/// default that lowers the activation term is visible to the next one.
#[derive(Clone, Copy, Debug)]
pub struct Context {
    pub n_layer: u32,
    pub n_ctx: u32,
    /// Effective VRAM budget, safety margin already taken out.
    pub vram_bytes: u64,
    pub activation_bytes: u64,
    pub logits_bytes: u64,
    pub backend: Backend,
    pub is_rollout: bool,
    /// GRPO's group size, or PPO's rollout batch. Zero outside a rollout run.
    pub group_size: u32,
    /// Largest number of sequences whose generation KV cache fits its share of
    /// the budget ([`crate::tuning::generation_capacity`]).
    pub generation_capacity: u32,
}

/// One proactive default.
pub struct ActiveDefault {
    pub id: &'static str,
    /// The condition, in one sentence. Rendered verbatim by `GET /v1/defaults`.
    pub condition: &'static str,
    /// What the run pays for it.
    pub cost: &'static str,
    /// What it does, in one clause.
    pub note: &'static str,
    /// Dotted document paths it moves. A locked path is left alone, field by
    /// field - a client that pinned `checkpoint_every_n_layers` still gets
    /// checkpointing turned on around it.
    pub touches: &'static [&'static str],
    /// Whether leaving it on changes *what* is computed rather than how fast.
    /// True for exactly one row, and that row carries a warning.
    pub degrades: bool,
    /// Warning code the plan carries when this default is applied.
    pub warning: Option<&'static str>,
    /// Warning code when the condition holds but the backend refuses.
    pub unavailable_warning: Option<&'static str>,
    state: fn(&TrainConfig) -> String,
    verdict: fn(&Context) -> Verdict,
    apply: Apply,
}

/// What a default does, once its condition holds.
///
/// The `is_locked` predicate is a parameter rather than something checked by the
/// caller because a default respects locks *field by field*: a client that pinned
/// `checkpoint_every_n_layers` still gets checkpointing turned on around it, and
/// only the row itself knows which of its fields that is.
type Apply = fn(&mut TrainConfig, &Context, &dyn Fn(&str) -> bool);

/// A default this phase applied, and what it moved. Same shape as
/// [`crate::levers::AppliedLever`], because a client renders the two side by
/// side.
pub type AppliedDefault = AppliedRule;

/// Layers below which recomputing activations buys nothing worth a forward pass.
const CHECKPOINTING_MIN_LAYERS: u32 = 8;
/// Share of the budget the activations may take before checkpointing turns on.
const ACTIVATION_BUDGET_PERCENT: u64 = 15;
/// Peak each chunked-cross-entropy tile is sized against.
const LOGITS_TILE_BYTES: u64 = 256 * 1024 * 1024;
const MIN_CE_TILES: u32 = 2;
const MAX_CE_TILES: u32 = 16;

fn percent(bytes: u64, share: u64) -> u64 {
    bytes / 100 * share
}

/// The stride to keep one activation checkpoint at, given how far the
/// activations overshoot their share.
///
/// Two facts set the shape. The peak under checkpointing is the retained layer
/// boundaries plus the working set of the segment being recomputed, so it goes
/// as `n_layer / s + s` and bottoms out at `sqrt(n_layer)`; and the extra
/// compute is one forward pass at any stride, so nothing on the compute side
/// prefers a small `s`. Therefore `s = 1` is
/// strictly the worst choice available: the largest boundary term for the same
/// recompute.
///
/// So the search runs upward from 1 and stops at the first stride that clears
/// the share, never going past the memory optimum. Stopping early matters: a
/// wider stride recomputes a longer segment at once, and a run that already
/// fits has no reason to buy memory it will not use.
fn checkpoint_stride(n_layer: u32, activation_bytes: u64, target_bytes: u64) -> u32 {
    let optimum = retrograd_core::checkpoint_stride_for(n_layer);
    let per_layer = activation_bytes / u64::from(n_layer.max(1));
    for stride in 1..=optimum {
        let boundaries = per_layer * u64::from(n_layer / stride.max(1));
        let segment = per_layer * u64::from(stride);
        if boundaries + segment <= target_bytes {
            return stride;
        }
    }
    optimum
}

/// Tiles the vocabulary logits are streamed over: one per [`LOGITS_TILE_BYTES`]
/// of the term, bounded.
fn ce_tiles(logits_bytes: u64) -> u32 {
    let wanted = logits_bytes.div_ceil(LOGITS_TILE_BYTES);
    wanted.clamp(MIN_CE_TILES as u64, MAX_CE_TILES as u64) as u32
}

fn on_off(value: bool) -> String {
    if value { "on" } else { "off" }.to_string()
}

/// The active defaults, in application order.
///
/// `checkpoint.every_steps` and `evaluation.every_iterations` are defaults
/// too, but they are not here: both need the *step count*, which only exists
/// once the geometry is chosen, so they are derived alongside `warmup_steps`
/// after phase 3 and documented as rows of the derivation table
/// ([`crate::tuning::DERIVATIONS`]) instead. Splitting them off keeps this list
/// to what actually operates on a [`TrainConfig`].
pub static ACTIVE_DEFAULTS: &[ActiveDefault] = &[
    ActiveDefault {
        id: "gradient_checkpointing",
        condition: "at least 8 layers and activations above 15% of the VRAM budget",
        cost: "one extra forward pass per segment - numerically inert",
        note: "layer activations are recomputed instead of retained",
        touches: &[
            "training.gradient_checkpointing",
            "training.checkpoint_every_n_layers",
        ],
        degrades: false,
        warning: None,
        unavailable_warning: None,
        state: |training| {
            if training.gradient_checkpointing {
                format!("every {} layers", training.checkpoint_every_n_layers)
            } else {
                "off".to_string()
            }
        },
        verdict: |context| {
            if context.n_layer < CHECKPOINTING_MIN_LAYERS {
                return Verdict::Skip;
            }
            // Context length alone is not evidence: an 8k-token step on a small
            // model can sit at a fifth of the budget, and triggering on length
            // would pay a whole extra forward pass for memory it was not short
            // of. What matters is the share the activations actually take,
            // which is the term this recomputes.
            if context.activation_bytes > percent(context.vram_bytes, ACTIVATION_BUDGET_PERCENT) {
                Verdict::Apply
            } else {
                Verdict::Skip
            }
        },
        apply: |training, context, is_locked| {
            if !is_locked("training.gradient_checkpointing") {
                training.gradient_checkpointing = true;
            }
            if !is_locked("training.checkpoint_every_n_layers") {
                training.checkpoint_every_n_layers = checkpoint_stride(
                    context.n_layer,
                    context.activation_bytes,
                    percent(context.vram_bytes, ACTIVATION_BUDGET_PERCENT),
                );
            }
        },
    },
    ActiveDefault {
        id: "chunked_cross_entropy",
        condition: "always - every backend the project ships carries both fused nodes",
        cost: "none - the tile recompute costs less bandwidth than the tensor it avoids building",
        note: "the vocabulary logits are streamed tile by tile, over a bounded token chunk",
        touches: &[
            "training.chunked_cross_entropy",
            "training.chunked_ce_tiles",
            "training.chunked_ce_seq_chunk",
        ],
        degrades: false,
        warning: None,
        unavailable_warning: None,
        state: |training| {
            if training.chunked_cross_entropy {
                format!(
                    "{} tiles, {}-token chunks",
                    training.chunked_ce_tiles, training.chunked_ce_seq_chunk
                )
            } else {
                "off".to_string()
            }
        },
        verdict: |_| {
            // No gate at all, and both halves of that are deliberate.
            //
            // No *size* gate: never building `[n_vocab, n_tokens]` removes that
            // tensor's write
            // *and* its read, which is more bandwidth than streaming the tiles
            // spends. There is no crossover to find, so there is no threshold to
            // put one at.
            //
            // No *backend* gate either, not even for `Unknown`. It would be
            // theatre: the configuration arrives with the fused path already on
            // (it is the engine's own default), so skipping the row would leave
            // the setting exactly where it was while reading as caution. What an
            // unrecognised accelerator really deserves is a warning saying the
            // estimate assumes a probe it could not run,
            // `chunked_ce_on_an_unrecognised_backend` in `collect_warnings`.
            Verdict::Apply
        },
        apply: |training, context, is_locked| {
            if !is_locked("training.chunked_cross_entropy") {
                training.chunked_cross_entropy = true;
            }
            if !is_locked("training.chunked_ce_tiles") {
                training.chunked_ce_tiles = ce_tiles(context.logits_bytes);
            }
            if !is_locked("training.chunked_ce_seq_chunk") {
                // Vocabulary tiles bound one axis of the tile intermediate; the
                // other is the token count, so on a long context the term walks
                // back up whatever the tiles saved. Zero means "the whole step
                // at once", which is the unbounded case this is here to close;
                // any other value is already narrower than the default, and
                // widening it would be this phase spending memory.
                training.chunked_ce_seq_chunk = match training.chunked_ce_seq_chunk {
                    0 => retrograd_core::DEFAULT_CE_SEQ_CHUNK,
                    chunk => chunk.min(retrograd_core::DEFAULT_CE_SEQ_CHUNK),
                };
            }
        },
    },
    ActiveDefault {
        id: "fast_sampling_context",
        condition: "any objective that samples rollouts",
        cost: "the sampling distribution differs from the trained policy by rounding",
        note: "the sampling cache is F16 with flash attention",
        touches: &["training.fast_sampling_context"],
        degrades: true,
        warning: Some("sampling_distribution_approximated"),
        unavailable_warning: None,
        state: |training| on_off(training.fast_generation_context),
        verdict: |context| {
            if context.is_rollout {
                Verdict::Apply
            } else {
                Verdict::Skip
            }
        },
        apply: |training, _, is_locked| {
            if !is_locked("training.fast_sampling_context") {
                training.fast_generation_context = true;
            }
        },
    },
    ActiveDefault {
        id: "generation_concurrency",
        condition: "any objective that samples rollouts: one whole group decodes at once \
                    when its KV cache fits",
        cost: "none - it only sets how many completions share one decode wave",
        note: "as many sequences decode at once as the generation cache holds",
        touches: &["training.generation_concurrency"],
        degrades: false,
        warning: None,
        unavailable_warning: None,
        state: |training| match training.generation_concurrency {
            0 => "runtime-derived".to_string(),
            value => value.to_string(),
        },
        verdict: |context| {
            if context.is_rollout && context.group_size > 0 {
                Verdict::Apply
            } else {
                Verdict::Skip
            }
        },
        apply: |training, context, is_locked| {
            if is_locked("training.generation_concurrency") {
                return;
            }
            let fitting = context.generation_capacity.min(context.group_size).max(1);
            // A power of two, so a group of 8 decodes as 8 or 4 rather than as
            // some width that divides nothing.
            training.generation_concurrency = 1 << (u32::BITS - 1 - fitting.leading_zeros());
        },
    },
];

impl Rule for ActiveDefault {
    fn touches(&self) -> &'static [&'static str] {
        self.touches
    }
}

impl ActiveDefault {
    pub fn state(&self, training: &TrainConfig) -> String {
        (self.state)(training)
    }

    pub fn verdict(&self, context: &Context) -> Verdict {
        (self.verdict)(context)
    }

    /// Applies it, respecting locks field by field, and reports the net move.
    /// `None` when nothing changed - including when every field was the
    /// caller's.
    pub fn apply(
        &self,
        training: &mut TrainConfig,
        context: &Context,
        is_locked: &dyn Fn(&str) -> bool,
    ) -> Option<AppliedDefault> {
        let from = self.state(training);
        (self.apply)(training, context, is_locked);
        let to = self.state(training);
        if from == to {
            return None;
        }
        Some(AppliedDefault {
            id: self.id,
            from,
            to,
            note: self.note,
            cost: self.cost,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn context() -> Context {
        Context {
            n_layer: 24,
            n_ctx: 1024,
            vram_bytes: 8 * GIB,
            activation_bytes: 64 * 1024 * 1024,
            logits_bytes: 64 * 1024 * 1024,
            backend: Backend::Cuda,
            is_rollout: false,
            group_size: 0,
            generation_capacity: 32,
        }
    }

    fn training() -> TrainConfig {
        TrainConfig {
            n_ctx: 1024,
            n_batch: 256,
            n_ubatch: 32,
            ..Default::default()
        }
    }

    fn entry(id: &str) -> &'static ActiveDefault {
        ACTIVE_DEFAULTS.iter().find(|item| item.id == id).expect(id)
    }

    fn unlocked(_: &str) -> bool {
        false
    }

    #[test]
    fn the_documented_order_is_the_implemented_order() {
        let ids: Vec<&str> = ACTIVE_DEFAULTS.iter().map(|item| item.id).collect();
        assert_eq!(
            ids,
            [
                "gradient_checkpointing",
                "chunked_cross_entropy",
                "fast_sampling_context",
                "generation_concurrency",
            ],
            "the order of ACTIVE_DEFAULTS moved"
        );
        // Exactly one default changes what is computed, and it is the only one
        // that may carry a warning on application.
        for item in ACTIVE_DEFAULTS {
            assert_eq!(
                item.degrades,
                item.warning.is_some(),
                "{} declares a degradation without a warning, or the reverse",
                item.id
            );
            assert!(!item.touches.is_empty(), "{}", item.id);
            assert!(!item.condition.is_empty(), "{}", item.id);
            for path in item.touches {
                assert!(path.contains('.'), "'{path}' is not a dotted field path");
            }
        }
        assert_eq!(
            ACTIVE_DEFAULTS.iter().filter(|item| item.degrades).count(),
            1
        );
    }

    /// The activation share is the whole condition: a long context on a model
    /// that is nowhere near its budget must not pay an extra forward pass.
    #[test]
    fn checkpointing_follows_the_activation_share_and_not_the_context() {
        let checkpointing = entry("gradient_checkpointing");
        let long_but_light = Context {
            n_ctx: 8192,
            activation_bytes: 64 * 1024 * 1024,
            ..context()
        };
        assert_eq!(checkpointing.verdict(&long_but_light), Verdict::Skip);

        let heavy = Context {
            n_ctx: 512,
            activation_bytes: 4 * GIB,
            ..context()
        };
        assert_eq!(checkpointing.verdict(&heavy), Verdict::Apply);

        let shallow = Context {
            n_layer: 4,
            ..heavy
        };
        assert_eq!(checkpointing.verdict(&shallow), Verdict::Skip);
    }

    #[test]
    fn applying_checkpointing_reports_the_move_once() {
        let checkpointing = entry("gradient_checkpointing");
        let heavy = Context {
            activation_bytes: 4 * GIB,
            ..context()
        };
        let mut config = training();
        let applied = checkpointing
            .apply(&mut config, &heavy, &unlocked)
            .expect("it moved");
        assert_eq!(applied.from, "off");
        // 4 GiB over 24 layers against a 1.2 GiB share: no stride below the
        // memory optimum clears it, so the rule lands on sqrt(24) = 5.
        assert_eq!(applied.to, "every 5 layers");
        assert!(config.gradient_checkpointing);
        // Idempotent: a second pass has nothing to report.
        assert!(
            checkpointing
                .apply(&mut config, &heavy, &unlocked)
                .is_none()
        );
    }

    /// The stride is a memory choice: the peak is the retained boundaries plus
    /// one segment, so it falls as the stride widens up to `sqrt(n_layer)`. The
    /// rule stops at the first stride that clears the share, because a run that
    /// already fits has no reason to recompute a longer segment.
    #[test]
    fn the_checkpoint_stride_is_the_narrowest_one_that_clears_the_share() {
        let per_layer = 64 * 1024 * 1024;
        let activations = per_layer * 24;

        // Room for eight layers' worth: boundaries at stride 4 are six layers,
        // plus a four-layer segment, so stride 3 (8 + 3 = 11) does not fit and
        // stride 4 (6 + 4 = 10) is where it lands.
        assert_eq!(checkpoint_stride(24, activations, per_layer * 10), 4);
        // A generous budget is cleared by the narrowest stride there is. It has
        // to be generous: stride 1 retains *every* boundary, so clearing the
        // share at that stride means room for the whole activation term again.
        assert_eq!(checkpoint_stride(24, activations, activations * 2), 1);
        // Nothing clears it: the memory optimum, never past it.
        assert_eq!(
            checkpoint_stride(24, activations, 1),
            retrograd_core::checkpoint_stride_for(24)
        );
        // Never zero, whatever the depth.
        for n_layer in 1..64 {
            assert!(checkpoint_stride(n_layer, activations, 1) >= 1);
        }
    }

    #[test]
    fn a_locked_field_is_left_alone_and_its_neighbours_are_not() {
        let checkpointing = entry("gradient_checkpointing");
        let mut config = TrainConfig {
            checkpoint_every_n_layers: 4,
            ..training()
        };
        let locked = |path: &str| path == "training.checkpoint_every_n_layers";
        let applied = checkpointing
            .apply(&mut config, &context(), &locked)
            .expect("checkpointing itself is not locked");
        assert!(config.gradient_checkpointing);
        assert_eq!(config.checkpoint_every_n_layers, 4, "the lock held");
        assert_eq!(applied.to, "every 4 layers");

        // Every field locked: nothing to do, and nothing reported.
        let all = |path: &str| {
            matches!(
                path,
                "training.gradient_checkpointing" | "training.checkpoint_every_n_layers"
            )
        };
        let mut untouched = training();
        assert!(checkpointing.is_locked_by(&all));
        assert!(
            checkpointing
                .apply(&mut untouched, &context(), &all)
                .is_none()
        );
        assert!(!untouched.gradient_checkpointing);
    }

    /// No size gate: the fused path is not memory bought with bandwidth, it saves
    /// both, so a small logits term is not a reason to build the tensor.
    #[test]
    fn the_logits_rule_applies_at_any_size_and_sizes_its_tiles_from_the_term() {
        let cce = entry("chunked_cross_entropy");
        let small = Context {
            logits_bytes: 1024,
            ..context()
        };
        assert_eq!(cce.verdict(&small), Verdict::Apply);

        let heavy = Context {
            logits_bytes: 2 * GIB,
            ..context()
        };
        assert_eq!(cce.verdict(&heavy), Verdict::Apply);

        let mut config = TrainConfig {
            chunked_cross_entropy: false,
            chunked_ce_seq_chunk: 0,
            ..training()
        };
        let applied = cce.apply(&mut config, &heavy, &unlocked).expect("it moved");
        assert!(config.chunked_cross_entropy);
        assert_eq!(config.chunked_ce_tiles, 8, "2 GiB over 256 MiB tiles");
        assert_eq!(applied.to, "8 tiles, 512-token chunks");
        assert!(config.chunked_ce_tiles >= MIN_CE_TILES);
        assert!(config.chunked_ce_tiles <= MAX_CE_TILES);

        // The configuration now arrives with this already on, so on the common
        // path the row has nothing to report - a `defaults_applied` entry means
        // the plan really moved something.
        assert!(cce.apply(&mut training(), &heavy, &unlocked).is_none());
    }

    /// Every backend is eligible, including the unrecognised one: the configuration already
    /// carries the fused path, so a `Skip` there would read as caution while
    /// changing nothing. The unrecognised case is a warning, not an abstention.
    #[test]
    fn no_backend_is_excluded_from_the_fused_path() {
        let cce = entry("chunked_cross_entropy");
        for backend in [
            Backend::Unknown,
            Backend::Cpu,
            Backend::Metal,
            Backend::Cuda,
            Backend::Vulkan,
            Backend::Blas,
        ] {
            let facts = Context {
                backend,
                ..context()
            };
            assert_eq!(cce.verdict(&facts), Verdict::Apply, "{}", backend.id());
        }
        assert_eq!(cce.unavailable_warning, None);
    }

    /// The token axis is the one vocabulary tiles do not cover, so the rule
    /// bounds it - without ever widening a chunk the caller arrived with.
    #[test]
    fn the_token_chunk_is_bounded_but_never_widened() {
        let cce = entry("chunked_cross_entropy");

        let mut unbounded = TrainConfig {
            chunked_ce_seq_chunk: 0,
            ..training()
        };
        cce.apply(&mut unbounded, &context(), &unlocked);
        assert_eq!(
            unbounded.chunked_ce_seq_chunk,
            retrograd_core::DEFAULT_CE_SEQ_CHUNK
        );

        let mut narrow = TrainConfig {
            chunked_ce_seq_chunk: 64,
            ..training()
        };
        cce.apply(&mut narrow, &context(), &unlocked);
        assert_eq!(
            narrow.chunked_ce_seq_chunk, 64,
            "not widened to the default"
        );
    }

    #[test]
    fn the_rollout_defaults_only_exist_on_a_rollout() {
        for id in ["fast_sampling_context", "generation_concurrency"] {
            assert_eq!(entry(id).verdict(&context()), Verdict::Skip, "{id}");
        }
        let rollout = Context {
            is_rollout: true,
            group_size: 8,
            ..context()
        };
        let mut config = training();
        config.fast_generation_context = false;
        let fast = entry("fast_sampling_context")
            .apply(&mut config, &rollout, &unlocked)
            .expect("it moved");
        assert!(config.fast_generation_context);
        assert_eq!((fast.from.as_str(), fast.to.as_str()), ("off", "on"));

        let concurrency = entry("generation_concurrency")
            .apply(&mut config, &rollout, &unlocked)
            .expect("it moved");
        assert_eq!(config.generation_concurrency, 8, "one whole group at once");
        assert_eq!(concurrency.from, "runtime-derived");
    }

    #[test]
    fn a_narrow_generation_cache_lowers_the_concurrency_to_a_power_of_two() {
        let mut config = training();
        let squeezed = Context {
            is_rollout: true,
            group_size: 8,
            generation_capacity: 3,
            ..context()
        };
        entry("generation_concurrency")
            .apply(&mut config, &squeezed, &unlocked)
            .expect("it moved");
        assert_eq!(config.generation_concurrency, 2);
    }
}
