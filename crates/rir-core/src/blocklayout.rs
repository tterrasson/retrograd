//! The layout **description** of a quantized format
//! (ADR-3 section 4).
//!
//! It answers one question, the same one that seven hand-written expansion arms
//! answered seven times: *where are element `e`'s bits, and what should multiply
//! them*. Three aspects - the payload and its index plan,
//! the scales and their packing, and the formula.
//!
//! **It drives lowering, never the oracle.** `crate::quant` remains hand-written,
//! one arm per format, and this rule is the guarantee rather than an exception:
//! two independent witnesses of the same layout make parity meaningful.
//! Generating both from this table would silently let an incorrect description
//! pass the test.
//!
//! What the description does **not** have: a way to describe a
//! `NativeIntrinsic` format. A format without a `BlockLayout` cannot be lowered;
//! `rir_lower::can_lower_dequant` derives support from this description.

/// A constant table indexed by the decoded payload.
///
/// Bounded by construction: sixteen to two hundred fifty-six `i8` entries,
/// declared here and nowhere else. Emitters print the table at the start of the
/// shader and the interpreter reads it directly - a single source.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, PartialOrd, Ord)]
pub enum LutId {
    /// `kvalues_iq4nl`: the 16-entry non-linear table shared by `iq4_nl` and
    /// `iq4_xs`.
    Iq4Nl,
}

/// `kvalues_iq4nl`, as written by ggml.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

impl LutId {
    pub fn values(self) -> &'static [i8] {
        match self {
            LutId::Iq4Nl => &KVALUES_IQ4NL,
        }
    }

    /// Table name in emitted code. A single table, shared by both formats that
    /// index it - a shader must never declare the same table
    /// twice.
    pub fn symbol(self) -> &'static str {
        match self {
            LutId::Iq4Nl => "rir_lut_iq4nl",
        }
    }

    pub const ALL: [LutId; 1] = [LutId::Iq4Nl];
}

/// The **index plan** of a bit field: the mapping
/// `element → (byte, shift)`.
///
/// This field is named explicitly instead of derived from a
/// width, because this is where ggml interleaves and where a decoder goes wrong.
/// Yet the form is unique, as shown by reading the fourteen layouts:
///
/// ```text
/// ppb   = 8 / bits                        (elements per byte)
/// group = plane · ppb                     (elements covered by one plane)
/// byte  = offset + (e / group) · plane + (e % plane)
/// shift = bits · ((e % group) / plane)
/// value = (mem[byte] >> shift) & ((1 << bits) - 1)
/// ```
///
/// `plane` is the width in bytes of a bit plane - 16 for `q4_0`, 32 for K
/// quants, 64 for the `q6_K` payload, and 1 for a linear plane such as `q5_0`'s
/// `qh`. All fourteen portable formats fit, including high-bit planes: the only
/// difference between a payload and a high plane is how it is used afterward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BitPlan {
    /// Byte offset of the plane within the block.
    pub offset: u32,
    /// Bit-field width: 1, 2, 4, or 8 - a divisor of 8, never 3.
    pub bits: u32,
    /// Width of one plane in bytes.
    pub plane: u32,
}

impl BitPlan {
    pub const fn new(offset: u32, bits: u32, plane: u32) -> Self {
        BitPlan {
            offset,
            bits,
            plane,
        }
    }

    /// Elements per byte.
    pub const fn per_byte(&self) -> u32 {
        8 / self.bits
    }

    /// Elements covered by a complete plane.
    pub const fn group(&self) -> u32 {
        self.plane * self.per_byte()
    }

    pub const fn mask(&self) -> u32 {
        (1u32 << self.bits) - 1
    }

    /// The byte and shift of element `e`. Written here so lowering and tests
    /// read the **same** definition.
    pub fn locate(&self, e: usize) -> (usize, u32) {
        let (plane, group) = (self.plane as usize, self.group() as usize);
        let byte = self.offset as usize + (e / group) * plane + (e % plane);
        let shift = self.bits * ((e % group) / plane) as u32;
        (byte, shift)
    }
}

/// Where the block scales come from.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScalePlan {
    /// One F16 scale for the entire block, at `d_off`.
    Global { d_off: u32 },
    /// An F16 scale and minimum: `d·q + m`.
    GlobalMin { d_off: u32, m_off: u32 },
    /// An F16 block scale **and** one packed scale per sub-block.
    SubBlock {
        d_off: u32,
        /// Offset of the F16 `dmin`, when the formula subtracts one.
        dmin_off: Option<u32>,
        /// Elements per sub-block: 16 or 32.
        sub_elements: u32,
        /// Offset of the packed-scale block.
        off: u32,
        packing: SubScalePacking,
        /// Bias subtracted from the sub-block scale before multiplication.
        /// 32 for `q3_K` and `iq4_xs`, 0 everywhere else.
        bias: i32,
    },
}

/// The five sub-block scale packings actually used by ggml
/// (ADR-3 section 4). One more would be a line here and an arm in
/// lowering; this is the only part of the description not derived from an index
/// plan, because ggml did not make it regular.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SubScalePacking {
    /// `q4_K`, `q5_K`: eight 6-bit (scale, min) pairs in 12 bytes,
    /// `get_scale_min_k4`.
    SixBitPairs12,
    /// `q2_K`: one byte per sub-block, low nibble = scale, high = min.
    NibblePair,
    /// `q3_K`: sixteen 6-bit scales shuffled by `kmask` in 12 bytes.
    Kmask12,
    /// `q6_K`: one signed byte per sub-block.
    I8,
    /// `iq4_xs`: four low bits packed two per byte, plus two high bits taken
    /// from a 16-bit word at `high_off`.
    FourPlusTwo { high_off: u32 },
}

/// What the decoded payload becomes before multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Payload {
    /// `q - offset`, as an integer and then converted.
    Offset(i32),
    /// `LUT[q]`.
    Lut(LutId),
}

/// What the formula subtracts or adds after multiplication.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MinTerm {
    /// Nothing: `d · payload`.
    None,
    /// `+ m`, the global F16 minimum of `q4_1`/`q5_1`.
    GlobalPlus,
    /// `− dmin · mm`, the sub-block minimum of K quants.
    SubMinus,
}

/// The complete layout of a format, as consumed by lowering.
///
/// `BlockShape` remains the **family** - what the canonical table already
/// publishes and what the oracle branches on; this is the detail, and it comes
/// alongside rather than replacing it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BlockLayout {
    /// The payload plan.
    pub payload: BitPlan,
    /// True when the 8-bit payload is **signed** (`q8_0`): it is then read as-is,
    /// without shifting or masking.
    pub signed: bool,
    /// The high-bit plan, when present. Its value is concatenated above the
    /// payload: `q = payload | (high << payload.bits)`.
    pub high: Option<BitPlan>,
    pub scales: ScalePlan,
    pub value: Payload,
    pub min: MinTerm,
}

impl BlockLayout {
    /// The sub-block scale on which element `e` depends, if any.
    pub fn sub_index(&self, e: usize) -> Option<usize> {
        match self.scales {
            ScalePlan::SubBlock { sub_elements, .. } => Some(e / sub_elements as usize),
            _ => None,
        }
    }
}
