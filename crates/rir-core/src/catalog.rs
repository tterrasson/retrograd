//! The kernel catalogue schema: what the generator publishes about every
//! (op, backend) row, read by the planner.
//!
//! `rir-gen` writes `generated/rir/catalog/catalog.json`; `retrograd-server`
//! reads it and `retrograd-plan` reasons about it. Those two ends share no
//! dependency but this crate, so the schema lives here - the same reason the
//! manifest does: a `rir-*` crate never depends on `retrograd-core`
//! (`AGENTS.md`).
//!
//! This module is the exception to "the semantic IR knows nothing about ggml",
//! and it is a deliberate one: the catalogue is by definition what RIR publishes
//! *about its ggml integration*. The IR proper still names no op. What lives
//! here is only what both ends of the file must agree on - the policy ladder and
//! the four devices it speaks about - never how a policy is decided
//! (`rir-kernels`' table), nor how a row is emitted (`rir-emit`).
//!
//! Field **order is part of the contract**: `serde_json` serializes a struct in
//! declaration order, so moving a field here moves it in the committed
//! `catalog.json`, which the regeneration-diff test reports.

use serde::{Deserialize, Serialize};

/// The catalogue publishes the claim the manifest does; `KernelShapeRule` is
/// the name its readers already import.
pub use crate::claim::{Claim, ShapeRule as KernelShapeRule};

/// Schema version of the catalogue this module describes.
///
/// One constant for the two ends, like the manifest's: the generator writes it
/// and every consumer refuses anything else, so a bump is a single edit.
///
/// 3: a variant carries its own `policy` (the pair's policy is only a ceiling),
/// and the model profiles no longer carry selections - a profile names a shape
/// to measure, and the dispatch stays per-claim.
pub const SCHEMA_VERSION: u32 = 3;

/// The ggml backends the policy table can speak about.
///
/// Distinct from [`crate::Backend`], which is the *scheduling* domain: the
/// correspondence between the two is written once, in
/// `rir_emit::manifest::ggml_backend_of`. This one is the domain the published
/// contract names, which is why it lives with the contract rather than with the
/// emitters.
///
/// The declaration order is alphabetical, and that is load-bearing: the derived
/// `Ord` orders the rows of `catalog.json` without requiring a separate name
/// comparison.
#[derive(
    Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum GgmlBackend {
    /// The default because it is the conservative row: the CPU backend is
    /// [`BackendPolicy::NativeOnly`] for every kernel, so a default-constructed
    /// policy claims nothing.
    #[default]
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl GgmlBackend {
    pub fn name(self) -> &'static str {
        match self {
            GgmlBackend::Cuda => "cuda",
            GgmlBackend::Vulkan => "vulkan",
            GgmlBackend::Metal => "metal",
            GgmlBackend::Cpu => "cpu",
        }
    }
}

/// Per-backend variant policy. The priority between a generated variant and
/// the native kernel belongs here - never implicitly to a backend's name.
/// The three values form a **ladder** in declaration order:
/// `native_only < observe_generated < prefer_generated`. The derived order lets
/// callers express a policy ceiling as a comparison.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPolicy {
    /// The backend keeps its native kernel; the RIR variant, if generated at
    /// all, is oracle/probe material only.
    #[default]
    NativeOnly,
    /// The variant is registered, its contract is evaluated and counted at the
    /// dispatch site, but the native kernel always runs. This is the state a
    /// newly integrated op enters: it produces the coverage and rejection
    /// numbers a benchmark decision needs, on a backend where nothing yet
    /// justifies encoding it.
    ///
    /// Distinct from the process-wide `observe` *mode*, which says how much any
    /// integrated op may do; this says how far **this** op on **this** backend
    /// is allowed to go, whatever the mode. The effective behaviour is the
    /// weaker of the two.
    ObserveGenerated,
    /// The generated variant is preferred when its contract matches; the
    /// native kernel remains the pre-launch fallback.
    PreferGenerated,
}

impl BackendPolicy {
    /// The published spelling, which is also what `serde` writes. All consumers
    /// use this single mapping so a renamed variant cannot silently diverge.
    pub fn name(self) -> &'static str {
        match self {
            BackendPolicy::NativeOnly => "native_only",
            BackendPolicy::ObserveGenerated => "observe_generated",
            BackendPolicy::PreferGenerated => "prefer_generated",
        }
    }

    /// Whether a variant under this policy may actually be encoded.
    pub fn dispatches(self) -> bool {
        matches!(self, BackendPolicy::PreferGenerated)
    }
}

/// Everything the generator publishes about the kernels a build carries.
///
/// `fingerprint` is a digest of `(SCHEMA_VERSION, kernels, profiles)`: an
/// execution profile quotes it, so a planner can tell whether the snapshot it
/// cached was computed against this catalogue without comparing the lists.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelCatalog {
    pub schema_version: u32,
    pub fingerprint: String,
    #[serde(default)]
    pub kernels: Vec<KernelPolicy>,
    /// Model profiles: the exact model shapes a measurement grid is derived
    /// from. A profile carries no selection - the dispatch stays per-claim,
    /// and a profile is a geometry a sweep measured, never a choice installed
    /// at runtime.
    #[serde(default)]
    pub profiles: Vec<ModelProfile>,
}

/// The two execution regimes a measurement or a promotion is made for.
///
/// One kernel can win while decoding - a handful of rows, launch-bound - and
/// lose on a prefill of hundreds of rows, or the reverse, so a measurement
/// says which regime it was taken in and a promotion which regime it covers.
#[derive(
    Clone, Copy, Debug, Default, Eq, PartialEq, Hash, Ord, PartialOrd, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Regime {
    /// At most [`Regime::DECODE_MAX_TOKENS`] tokens in the step. The default:
    /// the regime a profile exists for first.
    #[default]
    Decode,
    /// Anything larger.
    Prefill,
}

impl Regime {
    /// The largest step still measured and selected as decoding: the batch
    /// sizes the decode profiles are taken at are 1, 4, 8, 16 and 32.
    pub const DECODE_MAX_TOKENS: u32 = 32;

    /// The step sizes a decode measurement covers.
    pub const DECODE_TOKENS: [u32; 5] = [1, 4, 8, 16, 32];

    /// The step size a prefill measurement is taken at.
    pub const PREFILL_TOKENS: [u32; 1] = [512];

    /// The regime a step of `n_tokens` tokens runs in.
    pub fn of(n_tokens: u32) -> Regime {
        if n_tokens <= Self::DECODE_MAX_TOKENS {
            Regime::Decode
        } else {
            Regime::Prefill
        }
    }

    /// The published spelling, which is also what `serde` writes.
    pub fn name(self) -> &'static str {
        match self {
            Regime::Decode => "decode",
            Regime::Prefill => "prefill",
        }
    }

    /// The step sizes a measurement in this regime covers.
    pub fn tokens(self) -> &'static [u32] {
        match self {
            Regime::Decode => &Self::DECODE_TOKENS,
            Regime::Prefill => &Self::PREFILL_TOKENS,
        }
    }
}

/// The key of a model profile: the dimensions that fix every kernel shape of
/// a forward pass, and the formats its weights are stored in.
///
/// Read from the model's own metadata at load. Two models that agree on all of
/// it run the same kernels on the same shapes, whatever they are called; two
/// that differ in any of it do not.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelShape {
    /// `general.architecture`, as the model file spells it (`qwen3`).
    pub architecture: String,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_ff: u32,
    /// ggml type names of the weight matrices (`q4_K`, `q6_K`, `q8_0`, `f16`),
    /// as a set: order and repetition carry nothing.
    #[serde(default)]
    pub weight_formats: Vec<String>,
}

impl ModelShape {
    /// The same key with its format set sorted and deduplicated, which is the
    /// form two keys are compared in.
    pub fn normalized(&self) -> ModelShape {
        let mut shape = self.clone();
        shape.weight_formats.sort();
        shape.weight_formats.dedup();
        shape
    }
}

/// One profile: a model shape on one backend, published so a sweep can derive
/// the measurement grid of that model's own kernels. It selects nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelProfile {
    /// A stable name for reports and for `rir-sweep --profile` (`qwen3-0.6b-q8_0`).
    pub name: String,
    pub backend: GgmlBackend,
    pub shape: ModelShape,
}

/// One `(ggml op, op variant, backend)` row: the policy that applies to it, the
/// generated variants that serve it, and what it deliberately leaves native.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelPolicy {
    pub ggml_op: String,
    pub op_variant: Option<String>,
    pub backend: GgmlBackend,
    pub policy: BackendPolicy,
    #[serde(default)]
    pub variants: Vec<KernelVariantPolicy>,
    #[serde(default)]
    pub declared_domain: Vec<KernelDomainAssumption>,
    pub native_retired: bool,
    pub native_exception: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelVariantPolicy {
    pub variant_id: String,
    pub kernel: String,
    pub priority: u8,
    /// The variant's own policy. The pair's policy is only a **ceiling**: the
    /// effective policy at dispatch is `min(pair, variant)`, so a variant under
    /// `observe_generated` is measured but never encodes, even on a promoted
    /// pair. A variant absent from the override table carries the pair's.
    pub policy: BackendPolicy,
    #[serde(default)]
    pub constraints: Vec<String>,
    #[serde(default)]
    pub eligible_when: Claim,
    #[serde(default)]
    pub features: Vec<String>,
    pub workgroup: [u32; 3],
    pub vector_width: u32,
    pub reduction: String,
    pub scan: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelDomainAssumption {
    /// The restriction category the op leaves native, spelled as
    /// `rir_emit::DomainRestriction::name`. It stays a string here because its
    /// numbering is the fork's `ggml_rir_reject` ABI, which `rir-emit` owns and
    /// checks name by name against the C header - the catalogue publishes the
    /// word, not the number.
    pub reject: String,
    /// Kernel that declared it. An op served by one kernel per `src0` dtype
    /// publishes the same restriction several times, each excluding something
    /// different - without the name the reasons read as contradicting each other.
    pub kernel: String,
    pub why: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_published_spellings_are_the_serialized_ones() {
        // `name()` and serde must not be allowed to drift: the C registry
        // emitter prints the first, the catalogue writes the second, and the
        // planner joins rows across both.
        for policy in [
            BackendPolicy::NativeOnly,
            BackendPolicy::ObserveGenerated,
            BackendPolicy::PreferGenerated,
        ] {
            assert_eq!(
                serde_json::to_string(&policy).unwrap(),
                format!("\"{}\"", policy.name())
            );
        }
        for backend in [
            GgmlBackend::Cuda,
            GgmlBackend::Vulkan,
            GgmlBackend::Metal,
            GgmlBackend::Cpu,
        ] {
            assert_eq!(
                serde_json::to_string(&backend).unwrap(),
                format!("\"{}\"", backend.name())
            );
        }
    }

    #[test]
    fn the_policy_ladder_is_the_declaration_order() {
        assert!(BackendPolicy::NativeOnly < BackendPolicy::ObserveGenerated);
        assert!(BackendPolicy::ObserveGenerated < BackendPolicy::PreferGenerated);
        assert!(!BackendPolicy::ObserveGenerated.dispatches());
        assert!(BackendPolicy::PreferGenerated.dispatches());
    }

    #[test]
    fn a_catalogue_round_trips_through_the_schema() {
        let catalog = KernelCatalog {
            schema_version: SCHEMA_VERSION,
            fingerprint: "sha256:beef".into(),
            kernels: vec![KernelPolicy {
                ggml_op: "GGML_OP_OUT_PROD".into(),
                op_variant: None,
                backend: GgmlBackend::Vulkan,
                policy: BackendPolicy::PreferGenerated,
                variants: vec![KernelVariantPolicy {
                    variant_id: "out_prod_w4".into(),
                    kernel: "out_prod".into(),
                    priority: 2,
                    policy: BackendPolicy::ObserveGenerated,
                    eligible_when: Claim::any().on("row", ..=8),
                    workgroup: [32, 1, 1],
                    vector_width: 4,
                    reduction: "shared_tree".into(),
                    scan: "none".into(),
                    ..Default::default()
                }],
                declared_domain: vec![KernelDomainAssumption {
                    reject: "dtype".into(),
                    kernel: "out_prod".into(),
                    why: "quantized src0 is served by a sibling kernel".into(),
                }],
                native_retired: false,
                native_exception: None,
            }],
            profiles: vec![ModelProfile {
                name: "m".into(),
                backend: GgmlBackend::Vulkan,
                shape: ModelShape {
                    architecture: "qwen3".into(),
                    n_embd: 1024,
                    n_head: 16,
                    n_head_kv: 8,
                    head_dim: 128,
                    n_ff: 3072,
                    weight_formats: vec!["q4_K".into(), "q6_K".into()],
                },
            }],
        };
        let json = serde_json::to_string(&catalog).unwrap();
        assert_eq!(
            serde_json::from_str::<KernelCatalog>(&json).unwrap(),
            catalog
        );
        // The four devices and the ladder are read back as themselves, not as
        // strings a consumer re-parses - the variant's policy included.
        assert!(json.contains("\"backend\":\"vulkan\""));
        assert!(json.contains("\"policy\":\"prefer_generated\""));
        assert!(json.contains("\"policy\":\"observe_generated\""));
    }

    #[test]
    fn the_regime_follows_the_step_size() {
        assert_eq!(Regime::of(1), Regime::Decode);
        assert_eq!(Regime::of(Regime::DECODE_MAX_TOKENS), Regime::Decode);
        assert_eq!(Regime::of(Regime::DECODE_MAX_TOKENS + 1), Regime::Prefill);
        assert!(
            Regime::DECODE_TOKENS
                .iter()
                .all(|&n| Regime::of(n) == Regime::Decode)
        );
        assert!(
            Regime::PREFILL_TOKENS
                .iter()
                .all(|&n| Regime::of(n) == Regime::Prefill)
        );
        for regime in [Regime::Decode, Regime::Prefill] {
            assert_eq!(
                serde_json::to_string(&regime).unwrap(),
                format!("\"{}\"", regime.name())
            );
        }
    }

    /// The reference table publishes its profiles as shapes only.
    #[test]
    fn the_reference_profiles_carry_no_selection() {
        for p in crate::profiles::reference() {
            assert!(!p.name.is_empty());
        }
    }
}
