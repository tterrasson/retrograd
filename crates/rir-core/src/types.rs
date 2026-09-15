//! Scalar and tensor element types.
//!
//! Quantized formats are **not declared here**. The canonical table lives in
//! `crate::quant_table`, which expands it into `QuantType` and its descriptors
//! and which `rir-gen` expands into `ggml-retro-quant.h` for the ggml fork
//! (ADR-3 section 2). Before that, `QuantType` restated by hand what the header
//! already said, which is the double source of truth ADR-3 closes.
//!
//! What stays here is the *semantics*: what the columns of a row mean, and - in
//! `crate::quant` - how a portable format decodes. Decoding is applied during
//! lowering from that descriptor (ADR-3 section 3), so emitters never choose an
//! approximation.

/// How a backend obtains F32 values from a block of a format
/// (ADR-3 section 5).
///
/// This is a property of the *format*, not of a kernel: it says whether the
/// decoding formula can be expressed in RIR at all. Which of the two a given
/// dispatch uses remains a schedule/backend decision.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Lowering {
    /// The formula is expanded into RIR operations and addresses. RIR then has
    /// an independent decoder - the oracle of `crate::quant` - and can emit a
    /// generated kernel that reads the format with no backend primitive.
    Portable,
    /// The block load resolves to the backend's existing primitive
    /// (`dequantize_<name>` on Metal, `dequantize()`/`get_dm()` on Vulkan, the
    /// CUDA loaders). RIR owns the contract; it does not own the optimized
    /// formula, and has no independent decoder for the format.
    NativeIntrinsic,
}

impl Lowering {
    pub fn name(self) -> &'static str {
        match self {
            Lowering::Portable => "portable",
            Lowering::NativeIntrinsic => "native_intrinsic",
        }
    }
}

/// Shape of a portable decoder. One variant per *family* of block layout, not
/// one per format: `q4_0` and `iq4_nl` differ only by their quant table, and
/// `q4_1`/`q5_1` only by carrying a min alongside the scale.
///
/// A layout with no variant here is `NativeIntrinsic` by construction - there
/// is no way to half-describe it.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum BlockShape {
    /// Not a block format: one F16 element, no scale.
    F16,
    /// `d` (F16) then `block_elements` signed bytes. `q8_0`.
    ScaleI8,
    /// `d` (F16) then packed nibbles; value = `d · (q - offset)`. `q4_0`.
    ScaleNibble { offset: i32 },
    /// `d`, `m` (F16) then packed nibbles; value = `d · q + m`. `q4_1`.
    ScaleMinNibble,
    /// `d` (F16), a 32-bit high-bit plane, then nibbles; `q5_0`.
    ScaleNibbleHigh { offset: i32 },
    /// `d`, `m` (F16), a 32-bit high-bit plane, then nibbles; `q5_1`.
    ScaleMinNibbleHigh,
    /// E8M0 byte scale then E2M1 nibbles; `mxfp4`.
    Fp4E8M0,
    /// `d` (F16) then nibbles indexing a 16-entry signed table; `iq4_nl`.
    LutNibble,
    /// A super-block with per-sub-block scales. The formulas differ enough
    /// between K quants that folding them into parameters would obscure them;
    /// the decoder matches on the format itself.
    SuperBlock,
}

/// Which training ops may read a frozen tensor of a format in place. `dequant`
/// is the narrow table shared with Metal and fused sparse CE; `out_prod` is the
/// wider one (ADR-3 section 2).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct OpFamilies {
    pub dequant: bool,
    pub out_prod: bool,
}

use crate::quant_table::QuantType;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    U32,
    Bool,
    Quant(QuantType),
}

impl DType {
    /// Element size. For a quantized dtype this is ggml `type_size`, i.e. the
    /// block size stored in `ggml_tensor.nb[0]`.
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::Bool => 1,
            DType::Quant(q) => q.desc().block_bytes as usize,
        }
    }

    /// Logical elements per `nb[0]` stride unit (1 unless quantized).
    pub fn elements_per_unit(self) -> usize {
        match self {
            DType::Quant(q) => q.desc().block_elements as usize,
            _ => 1,
        }
    }

    /// The dtype a name spells, or `None` for a name outside the domain.
    ///
    /// The inverse of `name`, and the reader half of the manifest's dtype field:
    /// an unknown spelling is a rejected manifest rather than a dtype silently
    /// treated as four bytes.
    pub fn from_name(name: &str) -> Option<DType> {
        Some(match name {
            "f32" => DType::F32,
            "f16" => DType::F16,
            "bf16" => DType::BF16,
            "i32" => DType::I32,
            "u32" => DType::U32,
            "bool" => DType::Bool,
            other => DType::Quant(QuantType::from_name(other)?),
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "i32",
            DType::U32 => "u32",
            DType::Bool => "bool",
            DType::Quant(q) => q.desc().name,
        }
    }
}

/// Type of a uniform scalar parameter (a push constant on GPU).
///
/// One variant, and that is the contract rather than a gap (ADR-1 section 6): lowering
/// puts every parameter in an F32 register and the oracle is fed `&[f32]`, so an
/// integer uniform is an integer register bank in the lowering, the oracle and
/// three emitters - not a variant to declare. An `I32` arm existed here, was
/// formatted by the manifest, and was refused by validation; a public variant no
/// caller may use is worse than none.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScalarType {
    F32,
}

impl ScalarType {
    pub fn name(self) -> &'static str {
        match self {
            ScalarType::F32 => "f32",
        }
    }
}
