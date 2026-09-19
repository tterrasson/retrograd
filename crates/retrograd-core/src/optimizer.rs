//! Which optimizer a run uses, and what persistent state that costs.
//!
//! The half of the optimizer contract that can be decided before a graph
//! exists: the choice itself, its per-parameter eligibility policy, and the
//! state-bytes formula the planner needs. The descriptor that allocates slots,
//! builds the update step and streams them through a checkpoint lives in the
//! runtime; this module is what both ends agree on.
//!
//! Nothing here allocates. `state_bytes` is a storage formula, not a fit
//! guarantee: backend alignment, padding, the AdamW fallback of an ineligible
//! tensor and the optimizer's own scratch are added by the caller that knows
//! them.

use std::fmt;

use crate::error::{Error, Result};
use crate::trainable::{TensorRole, TrainableEntry, TrainableSet};

/// The optimizers a document may name.
///
/// `AdamW` is the default and stays the default: an experimental optimizer does
/// not become a recommended fine-tuning default because its memory formula is
/// attractive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizerKind {
    #[default]
    AdamW,
    /// llama.cpp's existing SGD semantics: no persistent state, but still a
    /// step counter and a schedule. "No slots" and "not initialized" are
    /// different states, and a resume must not confuse them.
    Sgd,
    /// Orthogonalized momentum on eligible hidden base matrices, AdamW
    /// everywhere else. Not implemented yet.
    Muon,
    /// Fixed-block, uniform-codebook Gefen. Not implemented yet.
    Gefen,
}

impl OptimizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
            Self::Muon => "muon",
            Self::Gefen => "gefen",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "adamw" | "adam_w" => Ok(Self::AdamW),
            "sgd" => Ok(Self::Sgd),
            "muon" => Ok(Self::Muon),
            "gefen" => Ok(Self::Gefen),
            other => Err(Error::config(format!(
                "training.optimizer must be adamw, sgd, muon or gefen; got '{other}'"
            ))),
        }
    }

    /// Whether this build can actually run it.
    ///
    /// A name the schema parses but the runtime cannot honour must be refused
    /// at load time: accepting a name and silently running AdamW would publish
    /// a trajectory nobody asked for, and a checkpoint that records the wrong
    /// optimizer.
    ///
    /// Muon and Gefen have no kernels here at all; AdamW and SGD are the two
    /// the update step can build, and the choice reaches it through
    /// [`Self::as_ffi`].
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::AdamW | Self::Sgd)
    }

    /// The integer the C `retro_train_config.optimizer` carries, and with it
    /// `ggml_opt_optimizer_type`: `0` AdamW, `1` SGD. Refusing to widen past
    /// what [`Self::is_implemented`] admits is deliberate - a value the runtime
    /// would read as AdamW is exactly the silent substitution that predicate
    /// exists to prevent.
    pub fn as_ffi(self) -> Result<i32> {
        match self {
            Self::AdamW => Ok(0),
            Self::Sgd => Ok(1),
            Self::Muon | Self::Gefen => Err(Error::invalid(format!(
                "optimizer {self} is not available in this build; use adamw or sgd"
            ))),
        }
    }

    /// Reads back what a runtime or a checkpoint recorded. An unknown value is
    /// an error rather than a default: "the optimizer this run used" is not a
    /// field that may be guessed.
    pub fn from_ffi(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::AdamW),
            1 => Ok(Self::Sgd),
            other => Err(Error::runtime(format!(
                "the runtime reported optimizer {other}, which this build does not know"
            ))),
        }
    }

    /// Names of the persistent per-parameter slots this optimizer keeps, in the
    /// order the state API enumerates them.
    ///
    /// SGD's empty list is the case worth naming: *no slot* and *no optimizer*
    /// are different states. An SGD run still has a step counter, a schedule
    /// and an RNG state, and a resume that concluded "no slots, so nothing was
    /// initialized" would silently restart the schedule from zero.
    pub fn slot_names(self) -> &'static [&'static str] {
        match self {
            Self::AdamW => &["m", "v"],
            Self::Sgd => &[],
            Self::Muon => &["momentum"],
            Self::Gefen => &["indices", "scales", "second_moments"],
        }
    }

    /// Persistent state bytes for one parameter of `n_elements`, under this
    /// optimizer, assuming the parameter is eligible for it.
    ///
    /// AdamW keeps two F32 tensors (`8N`), SGD keeps none, Muon one F32
    /// momentum (`4N`). Gefen is shaped per block and has no single-parameter
    /// closed form here; it is refused rather than approximated.
    fn eligible_state_bytes(self, n_elements: u64) -> u64 {
        match self {
            Self::AdamW => n_elements.saturating_mul(8),
            Self::Sgd => 0,
            Self::Muon => n_elements.saturating_mul(4),
            // Unreachable while `is_implemented` gates the choice; the fallback
            // is AdamW's because that is what an ineligible tensor gets.
            Self::Gefen => n_elements.saturating_mul(8),
        }
    }

    /// Whether this entry is eligible for the optimizer, or falls back to AdamW.
    ///
    /// Muon's v1 policy is *role*, not rank: hidden base matrices with exactly
    /// two non-trivial logical dimensions. Embeddings, the LM head, norms and
    /// biases are excluded because of what they are, and LoRA factors stay on
    /// AdamW - orthogonalizing `A` and `B` separately is not orthogonalizing
    /// `BA`, so it is a separate experiment rather than a free extension.
    pub fn is_eligible(self, entry: &TrainableEntry) -> bool {
        match self {
            Self::AdamW | Self::Sgd => true,
            Self::Muon => {
                entry.role == TensorRole::Base
                    && !entry.name.ends_with(".bias")
                    && !entry.name.contains("norm")
                    && entry.name != crate::trainable::OUTPUT_HEAD
                    && entry.ne[0] > 1
                    && entry.ne[1] > 1
                    && entry.ne[2] <= 1
                    && entry.ne[3] <= 1
            }
            Self::Gefen => entry.n_elements >= GEFEN_DEFAULT_MIN_NUMEL,
        }
    }

    /// Persistent optimizer state for a whole resolved set, with the AdamW
    /// fallback of every ineligible tensor included.
    ///
    /// The fallback is part of the total on purpose: an optimizer whose ratio
    /// only holds for the tensors it accepts reports a figure no run ever pays.
    pub fn state_bytes(self, set: &TrainableSet) -> u64 {
        set.entries
            .iter()
            .map(|entry| {
                if self.is_eligible(entry) {
                    self.eligible_state_bytes(entry.n_elements)
                } else {
                    Self::AdamW.eligible_state_bytes(entry.n_elements)
                }
            })
            .fold(0, u64::saturating_add)
    }
}

impl fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Default `min_numel` below which a Gefen-selected parameter falls back to
/// AdamW. Named rather than inlined because the plan's own warning depends on
/// it: a rank-16 LoRA factor on a 1024-wide projection has 16384 elements, so
/// "most LoRA factors are below the threshold" is false.
pub const GEFEN_DEFAULT_MIN_NUMEL: u64 = 4096;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainable::TensorDtype;

    fn entry(name: &str, role: TensorRole, ne: [i64; 4]) -> TrainableEntry {
        let n_elements = ne.iter().product::<i64>() as u64;
        TrainableEntry {
            name: name.to_string(),
            role,
            ne,
            dtype: TensorDtype::F32,
            n_elements,
            n_bytes: n_elements * 4,
            storage_id: 0,
        }
    }

    fn set(entries: Vec<TrainableEntry>) -> TrainableSet {
        TrainableSet {
            entries,
            ..Default::default()
        }
    }

    #[test]
    fn the_two_with_an_update_step_are_selectable_and_the_others_are_not() {
        assert!(OptimizerKind::AdamW.is_implemented());
        assert!(OptimizerKind::Sgd.is_implemented());
        // No kernel, no descriptor: accepting either would be a checkpoint
        // that records an optimizer the run never ran.
        assert!(!OptimizerKind::Muon.is_implemented());
        assert!(!OptimizerKind::Gefen.is_implemented());
        assert_eq!(OptimizerKind::default(), OptimizerKind::AdamW);
    }

    #[test]
    fn the_wire_value_round_trips_for_what_the_runtime_can_build() {
        for kind in [OptimizerKind::AdamW, OptimizerKind::Sgd] {
            assert_eq!(
                OptimizerKind::from_ffi(kind.as_ffi().unwrap()).unwrap(),
                kind
            );
        }
        assert!(OptimizerKind::Muon.as_ffi().is_err());
        assert!(OptimizerKind::Gefen.as_ffi().is_err());
        assert!(OptimizerKind::from_ffi(7).is_err());
    }

    #[test]
    fn sgd_keeps_no_slot_and_adamw_keeps_two() {
        assert!(OptimizerKind::Sgd.slot_names().is_empty());
        assert_eq!(OptimizerKind::AdamW.slot_names(), ["m", "v"]);
    }

    #[test]
    fn state_bytes_follow_the_published_table() {
        let one = set(vec![entry(
            "blk.0.attn_q.weight",
            TensorRole::Base,
            [64, 64, 1, 1],
        )]);
        let n = 64 * 64;
        assert_eq!(OptimizerKind::AdamW.state_bytes(&one), n * 8);
        assert_eq!(OptimizerKind::Sgd.state_bytes(&one), 0);
        assert_eq!(OptimizerKind::Muon.state_bytes(&one), n * 4);
    }

    #[test]
    fn muon_excludes_by_role_and_falls_back_to_adamw_rather_than_skipping() {
        let mixed = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.0.attn_norm.weight", TensorRole::Base, [64, 1, 1, 1]),
            entry("blk.0.attn_q.bias", TensorRole::Base, [64, 1, 1, 1]),
            entry("output.weight", TensorRole::Base, [64, 32, 1, 1]),
            entry(
                "blk.0.attn_q.weight.lora_a",
                TensorRole::LoraA,
                [64, 8, 1, 1],
            ),
        ]);
        // 4 bytes per element only on the hidden matrix; 8 everywhere else.
        let expected = 64 * 64 * 4 + 64 * 8 + 64 * 8 + 64 * 32 * 8 + 64 * 8 * 8;
        assert_eq!(OptimizerKind::Muon.state_bytes(&mixed), expected);
        // Worth saying out loud: on a LoRA-only set Muon costs exactly AdamW.
        let lora_only = set(vec![entry(
            "blk.0.attn_q.weight.lora_a",
            TensorRole::LoraA,
            [1024, 16, 1, 1],
        )]);
        assert_eq!(
            OptimizerKind::Muon.state_bytes(&lora_only),
            OptimizerKind::AdamW.state_bytes(&lora_only)
        );
    }

    #[test]
    fn a_rank_16_lora_factor_is_above_the_gefen_fallback_threshold() {
        // The plan's own correction: 16 x 1024 is 16384, not "below 4096".
        let factor = entry("a.lora_a", TensorRole::LoraA, [1024, 16, 1, 1]);
        assert!(OptimizerKind::Gefen.is_eligible(&factor));
    }

    #[test]
    fn names_round_trip_through_their_document_spelling() {
        for kind in [
            OptimizerKind::AdamW,
            OptimizerKind::Sgd,
            OptimizerKind::Muon,
            OptimizerKind::Gefen,
        ] {
            assert_eq!(OptimizerKind::parse(kind.as_str()).unwrap(), kind);
        }
        assert!(OptimizerKind::parse("lion").is_err());
    }
}
