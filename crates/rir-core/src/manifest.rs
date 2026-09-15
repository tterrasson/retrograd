//! The manifest schema: one description of the contract, read by both ends.
//!
//! The manifest is what a generated kernel publishes about itself - bindings in
//! order, push-constant layout, grid geometry, required device features - and it
//! is written by `rir-emit` and read by `rir-runtime`. Those two crates share no
//! dependency but this one, so the schema lives here: a field added on one side
//! is a field the other side sees at compile time, which is exactly what a
//! `schema_version` alone cannot enforce.
//!
//! Field **order is part of the contract** (ADR-4 section 4) and is the declaration
//! order of `Manifest` below: `serde_json` serializes a struct in that order, so
//! moving a field here moves it in every generated artifact - which the
//! regeneration-diff test reports.
//!
//! The enums are the real ones wherever the domain already owns one - `DType`,
//! `Access`, `Backend`. Where the published vocabulary is *narrower* than the
//! domain's (`Scan` drops the tile size `ScanStrategy` carries, `Determinism`
//! adds the `elementwise` case `ReductionSemantics` has no reason to name), the
//! manifest's own enum is declared here and the domain maps onto it in one
//! exhaustive `match` - never a string built at the call site.

use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::ir::{Access, ReductionSemantics};
use crate::types::{DType, ScalarType};

/// Schema version of the manifest this crate describes.
///
/// One constant for the two ends: the emitter writes it and the runtime refuses
/// anything else, so a bump is a single edit and an immediately failing test
/// rather than two numbers to keep equal by hand.
pub const SCHEMA_VERSION: u32 = 9;

/// Reassociation semantics published for the whole kernel: the strictest of its
/// reductions, or `Elementwise` when it has none.
///
/// `ReductionSemantics` is the property of *one* reduction and has no
/// `elementwise` value - a kernel without a reduction has nothing to reassociate
/// - which is why the published vocabulary is one variant wider than the domain's.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Determinism {
    /// No reduction at all: the result of one output element depends on nothing
    /// that could be regrouped.
    Elementwise,
    Associative,
    Deterministic,
    ExactOrder,
}

impl Determinism {
    /// The strictest semantics among a kernel's reductions.
    pub fn strictest(semantics: impl IntoIterator<Item = ReductionSemantics>) -> Self {
        let mut d = Determinism::Elementwise;
        for s in semantics {
            d = match (d, s) {
                (_, ReductionSemantics::ExactOrder) | (Determinism::ExactOrder, _) => {
                    Determinism::ExactOrder
                }
                (_, ReductionSemantics::Deterministic) | (Determinism::Deterministic, _) => {
                    Determinism::Deterministic
                }
                (_, ReductionSemantics::Associative) => Determinism::Associative,
            };
        }
        d
    }
}

/// The reduction strategy a lowering used, as published. Mirrors
/// `rir_lower::ReductionStrategy`, which maps onto it in one place.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reduction {
    Serial,
    SharedTree,
    TiledStage,
    SubgroupTree,
    /// Two stages: one reduction per subgroup, then one over the subgroup
    /// totals. Published apart from `SharedTree` because
    /// it is a different topology and therefore a different grouping of the
    /// additions - the same reason `TiledStage` is not `Serial`.
    HierarchicalTree,
}

/// The scan strategy a lowering used, as published.
///
/// Narrower than `rir_lower::ScanStrategy` on purpose: the tile size carried by
/// `TiledLanes` is a lowering decision, already visible in the shader and in the
/// workgroup shape, and publishing it would put a number in the ABI that no
/// reader has anything to do with.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scan {
    Serial,
    BlockedLanes,
    TiledLanes,
}

/// How parallel axes map to hardware. Mirrors `rir_lower::ParallelMapping`.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParallelMapping {
    Workgroup,
    Invocation,
}

/// Type of one entry of the constant block. Four bytes whatever the variant,
/// the layout is a sequence of 32-bit slots.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PushType {
    U32,
    I32,
    F32,
}

/// Where a dispatcher reads a scalar parameter. One value today: ggml's
/// `op_params` block. `None` on the parameter of an oracle-only kernel, which no
/// dispatcher reads.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParamSource {
    OpParams,
}

/// A part of the ggml op this kernel does **not** claim, with the reason.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DomainRestriction {
    pub reject: String,
    /// Free prose written by whoever declared the restriction - the one field of
    /// the manifest a human composes, and the reason this file is serialized
    /// rather than concatenated.
    pub why: String,
}

/// One buffer the kernel binds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Binding {
    pub name: String,
    pub binding: u32,
    /// Dispatch-time origin of the buffer (`src0`, `dst`…), or `None` for an
    /// oracle-only kernel.
    pub source: Option<String>,
    pub dtype: DType,
    pub access: Access,
    /// The axis indexing each ggml dimension of the binding, in order (`None`
    /// for a dimension not indexed by a simple axis in any access). Together
    /// with push-constant strides, this bounds the byte range a dispatch
    /// addresses in this buffer.
    pub extents: Vec<Option<String>>,
}

impl Binding {
    /// True if the shader writes this binding.
    pub fn is_write(&self) -> bool {
        self.access == Access::Write
    }
}

/// A scalar parameter and where the dispatcher reads it: a byte offset in
/// `ggml_tensor.op_params`. Distinct from `push_constants`, which is the
/// *shader-side* layout.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: ScalarType,
    pub source: Option<ParamSource>,
    /// Absent when there is no source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_offset: Option<u32>,
}

/// One slot of the constant block.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PushConstant {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: PushType,
    /// Byte offset within the push-constant block.
    pub offset: u32,
}

/// An axis carried by the grid: `grid[d] = ceil(n_<axis> / per_workgroup)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GridAxis {
    pub axis: String,
    pub per_workgroup: u32,
}

/// An axis a **flattened** dispatch decomposes its linear index into,
/// fastest first. The divisor is
/// `ceil(n_<axis> / per_index)`, and the grid is one dimension:
/// `ceil(Π divisors / workgroup[0])`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FlatAxis {
    pub axis: String,
    pub per_index: u32,
}

/// The shape predicate a variant publishes: the product of the extents of the
/// named axes must fall within `[min, max]` inclusive.
///
/// The owned twin of `rir_lower::ShapeRule`, whose axes are `&'static str`
/// because they are written in the schedule table. A manifest is *read*, so its
/// axes are `String`; the two convert in `rir-emit`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShapeRule {
    pub axes: Vec<String>,
    pub min: u32,
    pub max: u32,
}

/// Everything a generated kernel publishes about itself.
///
/// Declaration order **is** the JSON order, and the JSON order is the contract.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub schema_version: u32,
    pub name: String,
    pub entrypoint: String,
    /// The ggml op this kernel serves. `None` marks an oracle-only kernel with
    /// no production mapping - explicit, never derived from the name.
    pub ggml_op: Option<String>,
    pub variant_id: String,
    pub production: bool,
    /// Empty means the kernel claims its op entirely - and then a contract
    /// rejection at a dispatch site contradicts this manifest (ADR-4 section 6).
    pub assumed_domain: Vec<DomainRestriction>,
    pub backend: Backend,
    pub bindings: Vec<Binding>,
    pub params: Vec<Param>,
    pub push_constants: Vec<PushConstant>,
    /// Reserved by schema 8: no lowering packs its output today, and every
    /// manifest carries `null` here. Kept because a reader may branch on the
    /// field's presence, and dropping it would be a schema change.
    pub packed_output: Option<()>,
    pub constraints: Vec<String>,
    pub determinism: Determinism,
    pub reduction: Reduction,
    pub scan: Scan,
    /// Which of the pair's variants this is. `None` with an empty
    /// `eligible_when` is the fallback: it accepts everything, which is what
    /// makes per-shape selection total.
    pub variant: Option<String>,
    pub priority: u8,
    /// Elements one invocation covers on the contiguous axis. Above one it is
    /// also a claim: the variant only accepts a tensor whose contiguous stride
    /// is one element (ADR-4 section 5).
    pub vector_width: u32,
    /// Whether the flattened index is also the **address**, in units of the
    /// element. Like `vector_width` above, it is a
    /// claim, and a strictly stronger one: `vector_width` pins `nb[0]`, this
    /// pins every stride of every binding - each one contiguous, all of them the
    /// same shape - because that is what makes the stride sum collapse into a
    /// single product. False on every lowering that decomposes.
    #[serde(default)]
    pub linear_addr: bool,
    pub eligible_when: Vec<ShapeRule>,
    /// What the device must provide, in that backend's words. Absent on the CPU,
    /// which negotiates nothing.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub features: Vec<String>,
    pub parallel_mapping: ParallelMapping,
    pub dispatch: Vec<GridAxis>,
    /// Empty on every lowering but the flattened one, and exclusive with
    /// `dispatch`: a kernel has a grid **or** a linear space, never both.
    #[serde(default)]
    pub flat: Vec<FlatAxis>,
    pub workgroup: [u32; 3],
}

/// A dtype spells itself by name - `"f32"`, `"q8_0"` - and not by serde's
/// derived shape: `DType::Quant(QuantType::Q8_0)` would serialize as a nested
/// object, and the name is what four emitters, the registry and the fork
/// already print.
impl Serialize for DType {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for DType {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let name = String::deserialize(d)?;
        DType::from_name(&name)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown dtype '{name}'")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `why` is the one field a human composes freely. Serialization handles
    /// quotes, backslashes, control characters, and newlines so the generated
    /// manifest remains valid JSON.
    #[test]
    fn a_free_text_reason_cannot_break_the_manifest() {
        let restriction = DomainRestriction {
            reject: "non_contiguous".to_string(),
            why: "nb[0] == \"elem_bytes\", C:\\path,\ttwo\nlines and \u{7}".to_string(),
        };
        let text = serde_json::to_string(&restriction).expect("plain data");
        let back: DomainRestriction = serde_json::from_str(&text).expect("valid JSON");
        assert_eq!(back, restriction);
    }

    /// The strictest reduction of a kernel decides what it publishes, and a
    /// kernel with no reduction publishes `elementwise` - the case
    /// `ReductionSemantics` has no variant for.
    #[test]
    fn the_published_determinism_is_the_strictest_reduction() {
        use ReductionSemantics::*;
        assert_eq!(Determinism::strictest([]), Determinism::Elementwise);
        assert_eq!(
            Determinism::strictest([Associative, ExactOrder, Deterministic]),
            Determinism::ExactOrder
        );
        assert_eq!(
            Determinism::strictest([Deterministic, Associative]),
            Determinism::Deterministic
        );
        assert_eq!(
            Determinism::strictest([Associative]),
            Determinism::Associative
        );
    }

    /// A dtype is a name in the manifest, quantized ones included, and the name
    /// is the one the emitters print.
    #[test]
    fn a_dtype_is_serialized_by_name() {
        for name in ["f32", "f16", "bf16", "i32", "u32", "bool", "q8_0"] {
            let d = DType::from_name(name).unwrap_or_else(|| panic!("{name}"));
            assert_eq!(serde_json::to_string(&d).unwrap(), format!("\"{name}\""));
            assert_eq!(
                serde_json::from_str::<DType>(&format!("\"{name}\"")).unwrap(),
                d
            );
        }
        assert!(serde_json::from_str::<DType>("\"q4_k\"").is_err());
    }
}
