//! Trainable tensor families per architecture.
//!
//! `full` derives the trainable set instead of reading one from the document.
//! A suffix rule fails in both directions: parameters do not all end in
//! `.weight` or `.bias` (`blk.N.ssm_a`, `blk.N.hc_attn_scale`), and "every
//! non-frozen F32 tensor" sweeps up non-parameter data.
//!
//! Families are therefore enumerated per architecture. An architecture with
//! no row refuses `full` rather than guessing. Rows arrive with the test lane
//! that resolves them against a real file.
//!
//! Scope is the architecture: dtype is the resolver's rule, the loss path
//! follows the output head, and per-device kernel differences are a runtime
//! refusal.

/// The family of a tensor name: the name minus its `blk.<N>.` prefix and its
/// `.weight`/`.bias` suffix.
///
/// Total where [`crate::TensorDesc::stem`] is partial: a name with no suffix
/// is its own family. `blk.3.attn_q.weight` and `blk.7.attn_q.bias` are both
/// `attn_q`, `blk.0.shortconv.conv.weight` is `shortconv.conv`, `blk.2.ssm_a`
/// is `ssm_a`.
pub fn tensor_family(name: &str) -> &str {
    let rest = match name.strip_prefix("blk.") {
        Some(rest) => match rest.split_once('.') {
            Some((digits, rest)) if digits.bytes().all(|byte| byte.is_ascii_digit()) => rest,
            _ => name,
        },
        None => name,
    };
    rest.strip_suffix(".weight")
        .or_else(|| rest.strip_suffix(".bias"))
        .unwrap_or(rest)
}

/// The trainable families of one architecture.
///
/// Block families repeat per layer and follow a layer range; global families
/// exist once. The input embedding and the rotary tables are in neither list:
/// they are frozen for every architecture, and listing them would make them
/// unfreezable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchitectureCapability {
    /// `general.architecture`, as the GGUF spells it.
    pub architecture: &'static str,
    /// Families under `blk.<N>.`, in the order a report lists them.
    pub block_families: &'static [&'static str],
    /// Families outside every block.
    pub global_families: &'static [&'static str],
    /// Whether a standalone model GGUF for this architecture has been
    /// exported, reloaded and compared in a test here. `false` means "not
    /// measured", never "known broken".
    pub exports_model: bool,
}

impl ArchitectureCapability {
    /// Whether `full` may take this tensor, by name alone.
    ///
    /// Position matters, not just family: block names check the block list
    /// and non-block names the global one, so an invented
    /// `blk.0.token_embd_norm.weight` does not pass on the model-wide row.
    pub fn admits(&self, name: &str) -> bool {
        let family = tensor_family(name);
        let list = if crate::trainable::block_index(name).is_some() {
            self.block_families
        } else {
            self.global_families
        };
        list.contains(&family)
    }

    /// Every family in the row, block families first; shown by a refusal of
    /// something the row does not name.
    pub fn families(&self) -> impl Iterator<Item = &'static str> {
        self.block_families
            .iter()
            .chain(self.global_families)
            .copied()
    }
}

/// The block families every dense transformer shares. Not a row: rows are
/// checked against it, so a row that drops one of them fails a test rather
/// than reading as a deliberate exclusion.
const DENSE_BLOCK: [&str; 12] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_output",
    "attn_q_norm",
    "attn_k_norm",
    "ffn_norm",
    "ffn_up",
    "ffn_down",
    "ffn_gate",
];

const LLAMA_BLOCK: [&str; 12] = DENSE_BLOCK;

/// Qwen2 is dense, with biases on the attention projections. The bias shares
/// its weight's family, so the block list is the dense one unchanged.
const QWEN2_BLOCK: [&str; 12] = DENSE_BLOCK;

/// LFM2 adds a short-convolution to the dense block. Its kernel is
/// `blk.<N>.shortconv.conv.weight`, a family with a dot in it, which is why
/// the family rule strips a suffix rather than splitting on the first
/// separator.
const LFM2_BLOCK: [&str; 15] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_output",
    "attn_q_norm",
    "attn_k_norm",
    "ffn_norm",
    "ffn_up",
    "ffn_down",
    "ffn_gate",
    "shortconv.conv",
    "shortconv.in_proj",
    "shortconv.out_proj",
];

/// Qwen3.5 is a hybrid: three gated-delta-net blocks carrying a fused QKV and
/// an output gate, then one full-attention block, repeating. The two kinds of
/// block share one row, because a family list is per architecture and a name
/// only has to be admissible *somewhere* in the model - the file decides which
/// block carries which tensor.
///
/// It normalizes after the attention rather than before the FFN, so the row
/// adds `post_attention_norm`; `ffn_norm` is in it because every row carries
/// the whole dense block, not because a Qwen3.5 file has one. `blk.N.ssm_a`
/// carries no suffix at all and `blk.N.ssm_dt.bias` carries the other one;
/// both reach their family through the same rule.
const QWEN35_BLOCK: [&str; 21] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_qkv",
    "attn_output",
    "attn_q_norm",
    "attn_k_norm",
    "attn_gate",
    "post_attention_norm",
    "ffn_norm",
    "ffn_up",
    "ffn_down",
    "ffn_gate",
    "ssm_a",
    "ssm_alpha",
    "ssm_beta",
    "ssm_conv1d",
    "ssm_dt",
    "ssm_norm",
    "ssm_out",
];

/// Global families of a dense transformer. `output` is the untied vocabulary
/// projection; a tied one is frozen by storage identity before this check.
const DENSE_GLOBAL: [&str; 3] = ["output_norm", "output", "token_embd_norm"];

/// The architectures `full` can derive a set for.
///
/// `lfm2` and `qwen2` are the CPU fixtures, each checked against a real file
/// by `tests/trainable_inventory.rs`; `qwen35` is checked there too, against
/// the opt-in model `RETRO_QWEN3NEXT_TEST_MODEL` names; `llama` follows
/// llama.cpp naming. A row lists parameters; the dtype rule decides what a
/// given file can carry.
pub const CAPABILITY_TABLE: &[ArchitectureCapability] = &[
    ArchitectureCapability {
        architecture: "llama",
        block_families: &LLAMA_BLOCK,
        global_families: &DENSE_GLOBAL,
        exports_model: false, // no fixture of this architecture runs here
    },
    ArchitectureCapability {
        architecture: "qwen2",
        block_families: &QWEN2_BLOCK,
        global_families: &DENSE_GLOBAL,
        exports_model: true,
    },
    ArchitectureCapability {
        architecture: "qwen35",
        block_families: &QWEN35_BLOCK,
        global_families: &DENSE_GLOBAL,
        // No fixture of this architecture is generated here: the row is
        // resolved against a real file by the opt-in lane in
        // `tests/trainable_inventory.rs`, and no export has been measured.
        exports_model: false,
    },
    ArchitectureCapability {
        architecture: "lfm2",
        block_families: &LFM2_BLOCK,
        global_families: &DENSE_GLOBAL,
        exports_model: true,
    },
];

/// The row for an architecture, or `None` if this build has none.
pub fn architecture_capability(architecture: &str) -> Option<&'static ArchitectureCapability> {
    CAPABILITY_TABLE
        .iter()
        .find(|row| row.architecture == architecture)
}

/// Whether this build may publish a standalone model GGUF for `architecture`.
/// An unlisted architecture answers `false`: nothing here has measured it.
pub fn architecture_exports_model(architecture: &str) -> bool {
    architecture_capability(architecture).is_some_and(|row| row.exports_model)
}

/// The architectures a standalone model export is available for, for a
/// refusal to name.
pub fn model_export_architectures() -> impl Iterator<Item = &'static str> {
    CAPABILITY_TABLE
        .iter()
        .filter(|row| row.exports_model)
        .map(|row| row.architecture)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_family_survives_the_block_prefix_and_both_suffixes() {
        assert_eq!(tensor_family("blk.3.attn_q.weight"), "attn_q");
        assert_eq!(tensor_family("blk.11.attn_q.bias"), "attn_q");
        assert_eq!(tensor_family("attn_q.weight"), "attn_q");
        assert_eq!(tensor_family("output_norm.weight"), "output_norm");
    }

    #[test]
    fn a_parameter_with_no_suffix_is_its_own_family() {
        assert_eq!(tensor_family("blk.2.ssm_a"), "ssm_a");
        assert_eq!(tensor_family("blk.2.hc_attn_scale"), "hc_attn_scale");
        assert_eq!(
            tensor_family("blk.0.shortconv.conv.weight"),
            "shortconv.conv"
        );
    }

    /// `blk.` is a prefix, not a keyword: a family starting with those
    /// letters keeps them.
    #[test]
    fn a_name_that_only_looks_like_a_block_keeps_its_prefix() {
        assert_eq!(tensor_family("blk.weight"), "blk");
        assert_eq!(tensor_family("blkx.0.attn_q.weight"), "blkx.0.attn_q");
    }

    #[test]
    fn position_is_part_of_what_a_row_admits() {
        let row = architecture_capability("lfm2").expect("the fixture's architecture has a row");
        assert!(row.admits("blk.0.attn_q.weight"));
        assert!(row.admits("blk.0.shortconv.conv.weight"));
        assert!(row.admits("token_embd_norm.weight"));
        // The same family, in the wrong half of the model.
        assert!(!row.admits("blk.0.token_embd_norm.weight"));
        assert!(!row.admits("attn_q.weight"));
    }

    #[test]
    fn no_row_can_unfreeze_the_tensors_every_policy_freezes() {
        for row in CAPABILITY_TABLE {
            for frozen in crate::trainable::ALWAYS_FROZEN {
                assert!(
                    !row.admits(frozen),
                    "{} admits '{frozen}'",
                    row.architecture
                );
            }
            assert!(
                !row.admits("blk.0.rope_freqs.weight"),
                "{}",
                row.architecture
            );
        }
    }

    /// Rows spell every family out so they can be read in one place; the
    /// price is that a line can drop in an edit. Every row starts from the
    /// dense block, so that half is checked rather than trusted.
    #[test]
    fn every_row_carries_the_whole_dense_block() {
        for row in CAPABILITY_TABLE {
            for family in DENSE_BLOCK {
                assert!(
                    row.block_families.contains(&family),
                    "{} is missing '{family}'",
                    row.architecture
                );
            }
        }
    }

    /// The hybrid row, on names taken from a real Qwen3.5 file: a
    /// gated-delta-net block, a full-attention block, and the two parameters
    /// whose suffix the family rule has to survive.
    #[test]
    fn the_hybrid_row_admits_both_kinds_of_block() {
        let row = architecture_capability("qwen35").expect("qwen35 has a row");
        for name in [
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ssm_a",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_out.weight",
            "blk.3.attn_q.weight",
            "blk.3.attn_k_norm.weight",
            "output_norm.weight",
        ] {
            assert!(row.admits(name), "the qwen35 row refuses '{name}'");
        }
        // Tied on every published checkpoint, and frozen by storage identity
        // when it is - but the row is about names, and an untied file is the
        // one that reaches this check.
        assert!(row.admits("output.weight"));
        assert!(!row.admits("token_embd.weight"));
        assert!(!row.admits("ssm_a"), "a block family outside every block");
    }

    #[test]
    fn every_row_names_a_distinct_architecture_and_distinct_families() {
        let mut architectures: Vec<&str> = CAPABILITY_TABLE
            .iter()
            .map(|row| row.architecture)
            .collect();
        let before = architectures.len();
        architectures.sort_unstable();
        architectures.dedup();
        assert_eq!(architectures.len(), before, "two rows for one architecture");

        for row in CAPABILITY_TABLE {
            let mut families: Vec<&str> = row.families().collect();
            let before = families.len();
            families.sort_unstable();
            families.dedup();
            assert_eq!(
                families.len(),
                before,
                "{} repeats a family",
                row.architecture
            );
            assert!(families.iter().all(|family| !family.is_empty()));
        }
    }

    #[test]
    fn a_model_export_is_available_only_where_a_row_grants_it() {
        assert!(architecture_exports_model("lfm2"));
        assert!(architecture_exports_model("qwen2"));
        assert!(!architecture_exports_model("llama"));
        assert!(!architecture_exports_model("an-architecture-with-no-row"));

        let granted: Vec<&str> = model_export_architectures().collect();
        assert_eq!(granted, vec!["qwen2", "lfm2"]);
        assert!(
            granted
                .iter()
                .all(|architecture| architecture_capability(architecture).is_some()),
            "a granted architecture with no row would be unreachable"
        );
    }
}
