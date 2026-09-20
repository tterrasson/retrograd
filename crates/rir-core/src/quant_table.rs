//! The canonical quantized-format table.
//!
//! **This is the only place a format is declared.** The table expands, in this
//! crate and at compile time, into `QuantType` and its descriptors; `rir-gen`
//! expands the same rows into `ggml-retro-quant.h` for the ggml fork. Adding a
//! format is one row here; both tables, the capability matrix and the test
//! matrix follow.
//!
//! It lives in `rir-core` - the crate at the bottom of the RIR graph - because
//! every other crate of the chain reads it and none of them is below it. The
//! table was once declared in `rir-kernels`, at the top, and reached `rir-core`
//! as a Rust file printed by `rir-gen` and committed: generation then ran
//! *upstream* of the crate that compiles the generator, so deleting the
//! committed output - or changing the shape of a descriptor - left nothing that
//! could regenerate it. Generation now only ever flows out of this table into
//! the other languages, never back into Rust.
//!
//! `block_elements`/`block_bytes` restate ggml's `blck_size`/`type_size`. They
//! are numbers here rather than symbols because the header must stay legal MSL
//! and the Rust side has no C preprocessor - so `tests/rir_quant_oracle.rs`
//! asks ggml for both at runtime and fails on any disagreement. That test, not
//! this comment, is what makes the restatement safe.
//!
//! Each row also carries its **layout**: the index plan of its payload, the
//! packing of its scales, and its formula. Lowering expands that column, so a
//! new format costs a row rather than a new function. It is *not* what the
//! oracle reads - `crate::quant` stays hand-written, one arm per format, so the
//! parity test compares two independent witnesses of the same layout instead of
//! one table with itself.

use crate::blocklayout::{
    BitPlan, BlockLayout, LutId, MinTerm, Payload, ScalePlan, SubScalePacking,
};
use crate::types::{BlockShape, Lowering, OpFamilies};

/// One quantized format, in the single place that describes it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QuantFormat {
    /// Stable RIR id. Never renumbered: it is what a manifest, a registry row
    /// and a test fixture agree on across a table reordering.
    pub id: u16,
    /// `ggml_type_name()` spelling - `q4_K`, not `q4_k`.
    pub name: &'static str,
    /// `ggml_type` enum value.
    pub ggml_type: &'static str,
    /// Metal block struct for one quantization block.
    pub metal_block: &'static str,
    /// 16-value chunks per block (`QK/16`), as the Metal templates spell it.
    pub nl: &'static str,
    /// Logical elements per block.
    pub block_elements: u32,
    /// Physical block size in bytes: ggml `type_size`, i.e. `nb[0]`.
    pub block_bytes: u32,
    /// Natural alignment of the block struct, in bytes.
    pub align: u32,
    /// Byte offset of the quantized payload within the block, after the
    /// scale(s). Meaningful for portable shapes only; 0 otherwise.
    pub data_offset: u32,
    pub shape: BlockShape,
    pub lowering: Lowering,
    /// The layout lowering expands, when it has one.
    ///
    /// A column next to `shape`, never in its place: `shape` is the family the
    /// hand-written oracle branches on, this is the detail the expansion
    /// consumes. `None` is the single fact `can_lower_dequant` derives itself
    /// from, so no format name appears in `rir-lower`.
    pub layout: Option<BlockLayout>,
    /// Op families allowed to decode this format in place.
    pub ops: OpFamilies,
}

/// Both tables.
pub const BOTH: OpFamilies = OpFamilies {
    dequant: true,
    out_prod: true,
};
/// `OUT_PROD` only. These formats have narrower Metal and fused-CE support.
pub const OUT_PROD_ONLY: OpFamilies = OpFamilies {
    dequant: false,
    out_prod: true,
};

impl QuantFormat {
    /// Lowercased `name`: suffixes Vulkan shader/pipeline names and feeds
    /// `DATA_A_<upper>`. Derived, never a column - the two spellings differing
    /// by anything but case would be a bug, not a fact to declare.
    pub fn vk_name(&self) -> String {
        self.name.to_lowercase()
    }

    pub fn is_portable(&self) -> bool {
        self.lowering == Lowering::Portable
    }
}

/// The table, expanded twice: into the format ids and into their descriptors.
///
/// One row names its `QuantType` variant and builds its `QuantFormat`, so a
/// variant cannot exist without a descriptor, nor a descriptor without a
/// variant. That is the property the generated file used to buy with a code
/// generator and a committed artifact.
macro_rules! quant_table {
    ($($variant:ident => $row:expr),+ $(,)?) => {
        /// Quantized formats RIR knows. One variant per row of the canonical
        /// table, so a format cannot exist here without existing in ggml.
        ///
        /// Variants keep the ggml spelling (`Q4_K`, `IQ2_XXS`) so a grep for a
        /// format name crosses the Rust/C boundary unchanged.
        #[expect(non_camel_case_types)]
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
        pub enum QuantType {
            $($variant,)+
        }

        /// Every format, in canonical id order.
        pub const QUANT_FORMATS: [QuantType; [$(QuantType::$variant),+].len()] =
            [$(QuantType::$variant,)+];

        impl QuantType {
            /// The row that declares this format.
            pub fn desc(self) -> QuantFormat {
                match self {
                    $(QuantType::$variant => $row,)+
                }
            }
        }
    };
}

use BlockShape::*;
use Lowering::{NativeIntrinsic, Portable};

// Shorthand: every field is spelled once, in table order.
#[expect(clippy::too_many_arguments)]
fn f(
    id: u16,
    name: &'static str,
    ggml_type: &'static str,
    metal_block: &'static str,
    nl: &'static str,
    block_elements: u32,
    block_bytes: u32,
    align: u32,
    data_offset: u32,
    shape: BlockShape,
    lowering: Lowering,
    layout: Option<BlockLayout>,
    ops: OpFamilies,
) -> QuantFormat {
    QuantFormat {
        id,
        name,
        ggml_type,
        metal_block,
        nl,
        block_elements,
        block_bytes,
        align,
        data_offset,
        shape,
        lowering,
        layout,
        ops,
    }
}

/// A layout with a whole-block F16 scale and no high plane.
fn simple(payload: BitPlan, signed: bool, d_off: u32, value: Payload) -> Option<BlockLayout> {
    Some(BlockLayout {
        payload,
        signed,
        high: None,
        scales: ScalePlan::Global { d_off },
        value,
        min: MinTerm::None,
    })
}

/// A K-quant super-block: block scale, packed sub-block scales, and the
/// formula the sub-scale enters.
#[expect(clippy::too_many_arguments)]
fn sup(
    payload: BitPlan,
    high: Option<BitPlan>,
    d_off: u32,
    dmin_off: Option<u32>,
    sub_elements: u32,
    off: u32,
    packing: SubScalePacking,
    bias: i32,
    value: Payload,
    min: MinTerm,
) -> Option<BlockLayout> {
    Some(BlockLayout {
        payload,
        signed: false,
        high,
        scales: ScalePlan::SubBlock {
            d_off,
            dmin_off,
            sub_elements,
            off,
            packing,
            bias,
        },
        value,
        min,
    })
}

const NL256: &str = "GGML_RETRO_NL_256";

quant_table! {
    // Not a block format: one F16 element. It is a row because the ops that
    // decode in place accept it, and leaving it out is what let Metal and
    // CUDA disagree about F16 before the header existed.
    //
    // No layout: there is no block to describe, and the F16 element type is
    // a `MemType`, not a quantized expansion.
    F16 => f(
        0,
        "f16",
        "GGML_TYPE_F16",
        "half4x4",
        "1",
        1,
        2,
        2,
        0,
        F16,
        Portable,
        None,
        BOTH,
    ),
    // d (F16) then 16 nibble pairs; value = d·(q - 8).
    Q4_0 => f(
        1,
        "q4_0",
        "GGML_TYPE_Q4_0",
        "block_q4_0",
        "2",
        32,
        18,
        2,
        2,
        ScaleNibble { offset: 8 },
        Portable,
        simple(BitPlan::new(2, 4, 16), false, 0, Payload::Offset(8)),
        BOTH,
    ),
    // d, m (F16) then nibbles; value = d·q + m.
    Q4_1 => f(
        2,
        "q4_1",
        "GGML_TYPE_Q4_1",
        "block_q4_1",
        "2",
        32,
        20,
        2,
        4,
        ScaleMinNibble,
        Portable,
        Some(BlockLayout {
            payload: BitPlan::new(4, 4, 16),
            signed: false,
            high: None,
            scales: ScalePlan::GlobalMin { d_off: 0, m_off: 2 },
            value: Payload::Offset(0),
            min: MinTerm::GlobalPlus,
        }),
        BOTH,
    ),
    // d (F16), a 32-bit fifth-bit plane, then nibbles; value = d·(q - 16).
    // The high plane is linear over the block - bit `e` of a 32-bit word,
    // which is `plane = 1` in the index plan, one bit per element per byte.
    Q5_0 => f(
        3,
        "q5_0",
        "GGML_TYPE_Q5_0",
        "block_q5_0",
        "2",
        32,
        22,
        2,
        6,
        ScaleNibbleHigh { offset: 16 },
        Portable,
        Some(BlockLayout {
            payload: BitPlan::new(6, 4, 16),
            signed: false,
            high: Some(BitPlan::new(2, 1, 1)),
            scales: ScalePlan::Global { d_off: 0 },
            value: Payload::Offset(16),
            min: MinTerm::None,
        }),
        BOTH,
    ),
    Q5_1 => f(
        4,
        "q5_1",
        "GGML_TYPE_Q5_1",
        "block_q5_1",
        "2",
        32,
        24,
        2,
        8,
        ScaleMinNibbleHigh,
        Portable,
        Some(BlockLayout {
            payload: BitPlan::new(8, 4, 16),
            signed: false,
            high: Some(BitPlan::new(4, 1, 1)),
            scales: ScalePlan::GlobalMin { d_off: 0, m_off: 2 },
            value: Payload::Offset(0),
            min: MinTerm::GlobalPlus,
        }),
        BOTH,
    ),
    // The pilot format: d (F16) then 32 signed bytes.
    Q8_0 => f(
        5,
        "q8_0",
        "GGML_TYPE_Q8_0",
        "block_q8_0",
        "2",
        32,
        34,
        2,
        2,
        ScaleI8,
        Portable,
        simple(BitPlan::new(2, 8, 32), true, 0, Payload::Offset(0)),
        BOTH,
    ),
    // E8M0 exponent byte then E2M1 nibbles - no F16 anywhere, hence align 1.
    //
    // **Explicitly refused rather than forgotten**.
    // Its payload is a plain nibble plan and its LUT is already carried, but
    // its *scale* is an E8M0 exponent byte: expanding it means a scalar
    // conversion op in the Loop IR and one arm in each of three emitters,
    // for a format outside the ten standard ones and absent from every
    // census. The decision is the cost of the op, not the cost of the row.
    MXFP4 => f(
        6,
        "mxfp4",
        "GGML_TYPE_MXFP4",
        "block_mxfp4",
        "2",
        32,
        17,
        1,
        1,
        Fp4E8M0,
        Portable,
        None,
        BOTH,
    ),
    // K super-blocks: 256 elements, per-sub-block 4/6-bit scales.
    //
    // scales[16] | qs[64] | d, dmin (F16). One byte per sub-block of 16
    // carries the 4-bit scale low and the 4-bit min high.
    Q2_K => f(
        7,
        "q2_K",
        "GGML_TYPE_Q2_K",
        "block_q2_K",
        NL256,
        256,
        84,
        2,
        16,
        SuperBlock,
        Portable,
        sup(
            BitPlan::new(16, 2, 32),
            None,
            80,
            Some(82),
            16,
            0,
            SubScalePacking::NibblePair,
            0,
            Payload::Offset(0),
            MinTerm::SubMinus,
        ),
        BOTH,
    ),
    // hmask[32] | qs[64] | scales[12] | d (F16). Two payload bits plus one
    // high bit, and the high bit is what turns `q − 4` into ggml's
    // `qv − (bit ? 0: 4)`: the same expression, read forwards.
    Q3_K => f(
        8,
        "q3_K",
        "GGML_TYPE_Q3_K",
        "block_q3_K",
        NL256,
        256,
        110,
        2,
        32,
        SuperBlock,
        Portable,
        sup(
            BitPlan::new(32, 2, 32),
            Some(BitPlan::new(0, 1, 32)),
            108,
            None,
            16,
            96,
            SubScalePacking::Kmask12,
            32,
            Payload::Offset(4),
            MinTerm::None,
        ),
        BOTH,
    ),
    // d, dmin (F16) | scales[12] | qs[128].
    Q4_K => f(
        9,
        "q4_K",
        "GGML_TYPE_Q4_K",
        "block_q4_K",
        NL256,
        256,
        144,
        2,
        16,
        SuperBlock,
        Portable,
        sup(
            BitPlan::new(16, 4, 32),
            None,
            0,
            Some(2),
            32,
            4,
            SubScalePacking::SixBitPairs12,
            0,
            Payload::Offset(0),
            MinTerm::SubMinus,
        ),
        BOTH,
    ),
    // d, dmin (F16) | scales[12] | qh[32] | qs[128].
    Q5_K => f(
        10,
        "q5_K",
        "GGML_TYPE_Q5_K",
        "block_q5_K",
        NL256,
        256,
        176,
        2,
        48,
        SuperBlock,
        Portable,
        sup(
            BitPlan::new(48, 4, 32),
            Some(BitPlan::new(16, 1, 32)),
            0,
            Some(2),
            32,
            4,
            SubScalePacking::SixBitPairs12,
            0,
            Payload::Offset(0),
            MinTerm::SubMinus,
        ),
        BOTH,
    ),
    // ql[128] | qh[64] | scales[16] (i8) | d (F16). The nibble plane is 64
    // bytes wide here and not 32 - the one place a plane width is not the
    // sub-block size, and the reason the index plan carries it explicitly.
    Q6_K => f(
        11,
        "q6_K",
        "GGML_TYPE_Q6_K",
        "block_q6_K",
        NL256,
        256,
        210,
        2,
        0,
        SuperBlock,
        Portable,
        sup(
            BitPlan::new(0, 4, 64),
            Some(BitPlan::new(128, 2, 32)),
            208,
            None,
            16,
            192,
            SubScalePacking::I8,
            0,
            Payload::Offset(32),
            MinTerm::None,
        ),
        BOTH,
    ),
    // IQ formats index shared grid tables. Reproducing those tables in RIR
    // would duplicate kilobytes of constants whose only consumer is a
    // scalar path slower than the primitive every backend already has.
    IQ2_XXS => f(
        12,
        "iq2_xxs",
        "GGML_TYPE_IQ2_XXS",
        "block_iq2_xxs",
        NL256,
        256,
        66,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        BOTH,
    ),
    IQ2_XS => f(
        13,
        "iq2_xs",
        "GGML_TYPE_IQ2_XS",
        "block_iq2_xs",
        NL256,
        256,
        74,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        BOTH,
    ),
    IQ2_S => f(
        14,
        "iq2_s",
        "GGML_TYPE_IQ2_S",
        "block_iq2_s",
        NL256,
        256,
        82,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        BOTH,
    ),
    IQ3_XXS => f(
        15,
        "iq3_xxs",
        "GGML_TYPE_IQ3_XXS",
        "block_iq3_xxs",
        NL256,
        256,
        98,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        BOTH,
    ),
    IQ3_S => f(
        16,
        "iq3_s",
        "GGML_TYPE_IQ3_S",
        "block_iq3_s",
        NL256,
        256,
        110,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        BOTH,
    ),
    // A 16-entry signed lookup table indexed by nibbles - small enough to
    // carry, so it stays portable. With `LutId` in the Loop IR it is the
    // same row as `q4_0` with `Payload::Lut` instead of an offset.
    IQ4_NL => f(
        17,
        "iq4_nl",
        "GGML_TYPE_IQ4_NL",
        "block_iq4_nl",
        "2",
        32,
        18,
        2,
        2,
        LutNibble,
        Portable,
        simple(BitPlan::new(2, 4, 16), false, 0, Payload::Lut(LutId::Iq4Nl)),
        BOTH,
    ),
    // d (F16) | scales_h (u16) | scales_l[4] | qs[128]. Four low bits packed
    // two per byte plus two high bits taken from the 16-bit word - the fifth
    // packing, and the last one ggml uses.
    IQ4_XS => f(
        18,
        "iq4_xs",
        "GGML_TYPE_IQ4_XS",
        "block_iq4_xs",
        NL256,
        256,
        136,
        2,
        8,
        SuperBlock,
        Portable,
        sup(
            BitPlan::new(8, 4, 16),
            None,
            0,
            None,
            32,
            4,
            SubScalePacking::FourPlusTwo { high_off: 2 },
            32,
            Payload::Lut(LutId::Iq4Nl),
            MinTerm::None,
        ),
        BOTH,
    ),
    // OUT_PROD-only extension: the Metal and fused-CE support policy
    // is narrower, so these rows expand into one table and not the other.
    Q1_0 => f(
        19,
        "q1_0",
        "GGML_TYPE_Q1_0",
        "block_q1_0",
        "8",
        128,
        18,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        OUT_PROD_ONLY,
    ),
    Q2_0 => f(
        20,
        "q2_0",
        "GGML_TYPE_Q2_0",
        "block_q2_0",
        "4",
        64,
        18,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        OUT_PROD_ONLY,
    ),
    IQ1_S => f(
        21,
        "iq1_s",
        "GGML_TYPE_IQ1_S",
        "block_iq1_s",
        NL256,
        256,
        50,
        2,
        2,
        SuperBlock,
        NativeIntrinsic,
        None,
        OUT_PROD_ONLY,
    ),
    IQ1_M => f(
        22,
        "iq1_m",
        "GGML_TYPE_IQ1_M",
        "block_iq1_m",
        NL256,
        256,
        56,
        1,
        0,
        SuperBlock,
        NativeIntrinsic,
        None,
        OUT_PROD_ONLY,
    ),
    NVFP4 => f(
        23,
        "nvfp4",
        "GGML_TYPE_NVFP4",
        "block_nvfp4",
        "4",
        64,
        36,
        1,
        4,
        SuperBlock,
        NativeIntrinsic,
        None,
        OUT_PROD_ONLY,
    ),
}

impl QuantType {
    /// Lookup by ggml type name (`"q4_K"`), the spelling the FFI and the
    /// fixtures use.
    pub fn from_name(name: &str) -> Option<QuantType> {
        QUANT_FORMATS.into_iter().find(|q| q.desc().name == name)
    }
}

/// The whole table, in canonical id order.
///
/// `rir-gen` reads it to emit `ggml-retro-quant.h`; the kernel registry reads
/// it to instantiate one kernel per lowerable format.
pub fn quant_formats() -> Vec<QuantFormat> {
    QUANT_FORMATS.into_iter().map(QuantType::desc).collect()
}
