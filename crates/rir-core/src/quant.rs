//! The dequantization oracle (ADR-3 section 3).
//!
//! One independent decoder per **portable** format, working on the exact bytes
//! ggml stores. It is what makes a quantized kernel testable: the parity rule
//! is "same bytes decoded two ways", never "two independent quantizations of
//! the same F32" - those differ by the quantizer, not by the kernel.
//!
//! `NativeIntrinsic` formats have no decoder here on purpose. Their
//! grids and 1-2 bpw layouts would be kilobytes of constants whose only
//! consumer is a scalar path slower than the loader every backend already
//! ships; RIR owns their *contract* - block geometry, families, eligibility,
//! and `dequantize_row` says so with `QuantError::NoPortableDecoder` instead of
//! guessing.
//!
//! `tests/rir_quant_oracle.rs` runs each decoder against ggml's own
//! `to_float` on identical bytes; the block geometry declared in the canonical
//! table is checked against ggml's type traits in the same test.
//!
//! **On the casts in this file** (docs/engineering/CONVERSIONS.md). This is the
//! second-densest file of the repo and **one** cast in it is a conversion to
//! treat, in `dequantize_row`. Everything else falls in two classes that the convention
//! says to leave alone, and the distinction is the whole point of having
//! reviewed rather than swept:
//!
//! - **bit patterns**  - `(q & 0x0F) as i32`, `(sc[is] as i8)`,
//!   `((h >> 15) & 1) as u32`, `(*seed >> 33) as u8`. Here the narrowing *is*
//!   the operation: a nibble reinterpreted as a signed offset, a scale byte
//!   read as `i8`, an IEEE-754 field recomposed by hand. A checked conversion
//!   would be a contradiction - it would refuse the values these decoders exist
//!   to read - and it would break the byte-exactness the oracle is judged on;
//! - **widenings**  - every `as usize` reads a field of the canonical
//!   descriptor (`block_bytes`, `block_elements`, `data_offset`), all `u32`,
//!   used as a length or an index.

use crate::quant_table::QuantType;
use crate::types::{BlockShape, Lowering};

#[derive(Clone, Copy, Debug, Eq, PartialEq, thiserror::Error)]
pub enum QuantError {
    /// The format is `NativeIntrinsic`: RIR describes it but does not decode it.
    #[error("{}: native_intrinsic format, no portable decoder",.format.desc().name)]
    NoPortableDecoder { format: QuantType },
    /// `n` is not a multiple of the block size.
    #[error("{}: {n} elements, not a multiple of the block",.format.desc().name)]
    NotBlockAligned { format: QuantType, n: usize },
    /// The byte slice is shorter than the blocks it must contain.
    #[error("{}: {need} bytes expected, {got} provided",.format.desc().name)]
    ShortInput {
        format: QuantType,
        need: usize,
        got: usize,
    },
}

/// `kvalues_iq4nl` - the 16-entry non-linear table shared by `iq4_nl`/`iq4_xs`.
const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// `kvalues_fp4` - doubled E2M1 values, shared by MXFP4 and NVFP4.
const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

fn f16(bytes: &[u8], off: usize) -> f32 {
    half_to_f32(u16::from_le_bytes([bytes[off], bytes[off + 1]]))
}

/// IEEE binary16 to binary32. Exact for every input, subnormals included: an
/// oracle that rounded here would report a kernel error that is its own.
///
/// **One of three deliberate copies, and this one is the reference decoder of
/// the canonical tables.** The other two are
/// `rir_lower::interp::f16_to_f32`, which is what the parity oracle executes,
/// and the helper `rir_emit::cpu` prints into generated Rust, which is what a
/// consumer of `generated/rir/*/cpu.rs` runs. They are not a duplication to
/// centralize: a single implementation would make every F16 test tautological,
/// since the judge and the judged would be the same code. What keeps them from
/// drifting is a witness rather than a refactor,
/// `rir-kernels/tests/f16_oracles.rs` compares all three over the **whole**
/// 65 536-value input space, which is small enough to enumerate.
///
/// It is public for that test, and for that reason only.
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if mant == 0 => sign << 31,
        0 => {
            // Subnormal: renormalize into the binary32 exponent range.
            let mut e = -1i32;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | (((127 - 13 + e) as u32) << 23) | (m << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (mant << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13),
    };
    f32::from_bits(bits)
}

/// `ggml_e8m0_to_fp32_half`: the E8M0 exponent byte scaled by 1/2, which is the
/// convention `kvalues_mxfp4` (doubled E2M1) expects.
fn e8m0_to_f32_half(x: u8) -> f32 {
    // 2^(x - 127) / 2 = 2^(x - 128), with the two smallest exponents falling
    // into binary32 subnormals.
    match x {
        0 => f32::from_bits(0x0020_0000),
        1 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits(((x as u32) - 1) << 23),
    }
}

/// `get_scale_min_k4`: the 6-bit scale/min pair of super-block `j`, packed in
/// 12 bytes for 8 sub-blocks.
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Decodes `n` elements from `src`, which must hold whole blocks of `format`.
///
/// `n` is in **logical elements**, and must be a multiple of the block size:
/// a partial block is rejected rather than padded, because the same rejection
/// is what `supports_op` owes a dispatch (the required check).
pub fn dequantize_row(format: QuantType, src: &[u8], n: usize) -> Result<Vec<f32>, QuantError> {
    let d = format.desc();
    if d.lowering != Lowering::Portable {
        return Err(QuantError::NoPortableDecoder { format });
    }
    let be = d.block_elements as usize;
    if !n.is_multiple_of(be) {
        return Err(QuantError::NotBlockAligned { format, n });
    }
    let nb = n / be;
    // Saturating, the one treated cast of this file. `n` is the caller's
    // element count, so this product is a size with nothing bounding it above
    // (docs/engineering/CONVERSIONS.md). Wrapping it would make `need` small, the check
    // below would pass, and the loop would fail further down - panicking on the
    // slice range in release, on the multiplication in debug. Saturating leaves
    // `need` at `usize::MAX`, which no slice reaches, so the refusal comes from
    // the check already written here and names the right reason.
    let need = nb.saturating_mul(d.block_bytes as usize);
    if src.len() < need {
        return Err(QuantError::ShortInput {
            format,
            need,
            got: src.len(),
        });
    }

    let mut out = vec![0.0f32; n];
    for i in 0..nb {
        let b = &src[i * d.block_bytes as usize..(i + 1) * d.block_bytes as usize];
        decode_block(format, b, &mut out[i * be..(i + 1) * be]);
    }
    Ok(out)
}

/// Random bytes for `blocks` whole blocks of `format` - **one fixture for every
/// format**, derived from the description (ADR-3 section 7).
///
/// It lives here, beside the oracle, because it is the fixture *of a format*
/// and not of a kernel. A hand-written encoder would be the wrong reference,
/// it would be a second decoder in disguise, agreeing with a wrong expansion
/// exactly where the packing is wrong. Any bit pattern is a legal block, so
/// random bytes plus `dequantize_row` is the reference, and the chain of trust stays ggml →
/// oracle → expansion with no link written twice.
///
/// The one precaution: **the F16 scales are pinned**, at the offsets the
/// description names. A random half is infinite or NaN one pattern in
/// thirty-two, and a contraction over such a block has nothing left to compare.
/// 2⁻⁶ and 2⁻⁷ keep a six-bit scale times a nibble around one, so the sums stay
/// clear of the cancellation where a relative tolerance means nothing.
///
/// Returns `None` for a format the table describes no layout for: there is then
/// no way to know which bytes are scales, and pinning the wrong ones silently
/// would be worse than refusing.
pub fn random_block_bytes(format: QuantType, blocks: usize, seed: &mut u64) -> Option<Vec<u8>> {
    let d = format.desc();
    let layout = d.layout?;
    let bb = d.block_bytes as usize;
    let mut raw = vec![0u8; bb * blocks];
    for byte in raw.iter_mut() {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (*seed >> 33) as u8;
    }
    let f16_offsets: Vec<u32> = match layout.scales {
        crate::blocklayout::ScalePlan::Global { d_off } => vec![d_off],
        crate::blocklayout::ScalePlan::GlobalMin { d_off, m_off } => vec![d_off, m_off],
        crate::blocklayout::ScalePlan::SubBlock {
            d_off, dmin_off, ..
        } => std::iter::once(d_off).chain(dmin_off).collect(),
    };
    for b in 0..blocks {
        for (i, off) in f16_offsets.iter().enumerate() {
            let bits: u16 = if i == 0 { 0x2400 } else { 0x2000 };
            let at = b * bb + *off as usize;
            raw[at..at + 2].copy_from_slice(&bits.to_le_bytes());
        }
    }
    Some(raw)
}

/// Decodes exactly one block. `y.len() == block_elements`.
///
/// Every arm mirrors the corresponding `dequantize_row_*` of `ggml-quants.c`,
/// including the interleaving: the low nibbles fill the first half of the block
/// and the high nibbles the second. Getting that backwards produces a plausible
/// distribution and a wrong tensor, which is why the parity test compares
/// element by element and not by norm.
fn decode_block(format: QuantType, b: &[u8], y: &mut [f32]) {
    let d = format.desc();
    match d.shape {
        BlockShape::F16 => y[0] = f16(b, 0),

        BlockShape::ScaleI8 => {
            let scale = f16(b, 0);
            let off = d.data_offset as usize;
            for (j, v) in y.iter_mut().enumerate() {
                *v = scale * (b[off + j] as i8) as f32;
            }
        }

        BlockShape::ScaleNibble { offset } => {
            let scale = f16(b, 0);
            let off = d.data_offset as usize;
            let half = y.len() / 2;
            for j in 0..half {
                let q = b[off + j];
                y[j] = scale * ((q & 0x0F) as i32 - offset) as f32;
                y[j + half] = scale * ((q >> 4) as i32 - offset) as f32;
            }
        }

        BlockShape::ScaleMinNibble => {
            let (scale, min) = (f16(b, 0), f16(b, 2));
            let off = d.data_offset as usize;
            let half = y.len() / 2;
            for j in 0..half {
                let q = b[off + j];
                y[j] = scale * (q & 0x0F) as f32 + min;
                y[j + half] = scale * (q >> 4) as f32 + min;
            }
        }

        BlockShape::ScaleNibbleHigh { offset } => {
            let scale = f16(b, 0);
            let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
            let off = d.data_offset as usize;
            let half = y.len() / 2;
            for j in 0..half {
                let q = b[off + j];
                let h0 = ((qh >> j) << 4) & 0x10;
                let h1 = (qh >> (j + 12)) & 0x10;
                y[j] = scale * (((q & 0x0F) as u32 | h0) as i32 - offset) as f32;
                y[j + half] = scale * (((q >> 4) as u32 | h1) as i32 - offset) as f32;
            }
        }

        BlockShape::ScaleMinNibbleHigh => {
            let (scale, min) = (f16(b, 0), f16(b, 2));
            let qh = u32::from_le_bytes([b[4], b[5], b[6], b[7]]);
            let off = d.data_offset as usize;
            let half = y.len() / 2;
            for j in 0..half {
                let q = b[off + j];
                let h0 = ((qh >> j) << 4) & 0x10;
                let h1 = (qh >> (j + 12)) & 0x10;
                y[j] = scale * ((q & 0x0F) as u32 | h0) as f32 + min;
                y[j + half] = scale * ((q >> 4) as u32 | h1) as f32 + min;
            }
        }

        BlockShape::Fp4E8M0 => {
            let scale = e8m0_to_f32_half(b[0]);
            let off = d.data_offset as usize;
            let half = y.len() / 2;
            for j in 0..half {
                let q = b[off + j];
                y[j] = scale * KVALUES_FP4[(q & 0x0F) as usize] as f32;
                y[j + half] = scale * KVALUES_FP4[(q >> 4) as usize] as f32;
            }
        }

        BlockShape::LutNibble => {
            let scale = f16(b, 0);
            let off = d.data_offset as usize;
            let half = y.len() / 2;
            for j in 0..half {
                let q = b[off + j];
                y[j] = scale * KVALUES_IQ4NL[(q & 0x0F) as usize] as f32;
                y[j + half] = scale * KVALUES_IQ4NL[(q >> 4) as usize] as f32;
            }
        }

        BlockShape::SuperBlock => decode_super_block(format, b, y),
    }
}

/// K-quant super-blocks. Each layout is spelled out because each is genuinely
/// different; parameterizing them would hide the very bit shifts a decoder can
/// get wrong.
fn decode_super_block(format: QuantType, b: &[u8], y: &mut [f32]) {
    match format {
        // scales[16] | qs[64] | d, dmin (F16)
        QuantType::Q2_K => {
            let (d, dmin) = (f16(b, 80), f16(b, 82));
            let mut o = 0usize;
            let mut is = 0usize;
            for n in (0..256).step_by(128) {
                let q = &b[16 + n / 4..];
                for j in 0..4 {
                    let shift = 2 * j;
                    for half in 0..2 {
                        let sc = b[is];
                        is += 1;
                        let dl = d * (sc & 0xF) as f32;
                        let ml = dmin * (sc >> 4) as f32;
                        for l in 0..16 {
                            let qv = (q[l + 16 * half] >> shift) & 3;
                            y[o] = dl * qv as f32 - ml;
                            o += 1;
                        }
                    }
                }
            }
        }

        // hmask[32] | qs[64] | scales[12] | d (F16)
        QuantType::Q3_K => {
            let d_all = f16(b, 108);
            let hm = &b[0..32];
            // The 12 packed bytes hold sixteen 6-bit scales, split 4+2 bits.
            let mut aux = [0u32; 4];
            // Only three words are stored; aux[3] is *computed* below, as in
            // ggml's `memcpy(aux, x[i].scales, 12)`.
            for (i, a) in aux.iter_mut().take(3).enumerate() {
                *a = u32::from_le_bytes([
                    b[96 + 4 * i],
                    b[97 + 4 * i],
                    b[98 + 4 * i],
                    b[99 + 4 * i],
                ]);
            }
            let (kmask1, kmask2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            aux[0] = (aux[0] & kmask2) | ((tmp & kmask1) << 4);
            aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            let mut scales = [0i8; 16];
            for i in 0..4 {
                for (j, s) in aux[i].to_le_bytes().iter().enumerate() {
                    scales[4 * i + j] = *s as i8;
                }
            }

            let mut o = 0usize;
            let mut is = 0usize;
            let mut m: u8 = 1;
            for n in (0..256).step_by(128) {
                let q = &b[32 + n / 4..];
                for j in 0..4 {
                    let shift = 2 * j;
                    for half in 0..2 {
                        let dl = d_all * (scales[is] as i32 - 32) as f32;
                        is += 1;
                        for l in 0..16 {
                            let idx = l + 16 * half;
                            let qv = ((q[idx] >> shift) & 3) as i32;
                            let hi = if hm[idx] & m != 0 { 0 } else { 4 };
                            y[o] = dl * (qv - hi) as f32;
                            o += 1;
                        }
                    }
                    m <<= 1;
                }
            }
        }

        // d, dmin (F16) | scales[12] | qs[128]
        QuantType::Q4_K => {
            let (d, dmin) = (f16(b, 0), f16(b, 2));
            let scales = &b[4..16];
            let qs = &b[16..];
            let mut o = 0usize;
            for (blk, is) in (0..256).step_by(64).zip((0..8).step_by(2)) {
                let q = &qs[blk / 2..];
                let (sc1, m1) = scale_min_k4(is, scales);
                let (sc2, m2) = scale_min_k4(is + 1, scales);
                let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
                for l in 0..32 {
                    y[o + l] = d1 * (q[l] & 0xF) as f32 - mm1;
                    y[o + 32 + l] = d2 * (q[l] >> 4) as f32 - mm2;
                }
                o += 64;
            }
        }

        // d, dmin (F16) | scales[12] | qh[32] | qs[128]
        QuantType::Q5_K => {
            let (d, dmin) = (f16(b, 0), f16(b, 2));
            let scales = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..];
            let mut o = 0usize;
            let (mut u1, mut u2) = (1u8, 2u8);
            for (blk, is) in (0..256).step_by(64).zip((0..8).step_by(2)) {
                let ql = &qs[blk / 2..];
                let (sc1, m1) = scale_min_k4(is, scales);
                let (sc2, m2) = scale_min_k4(is + 1, scales);
                let (d1, mm1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, mm2) = (d * sc2 as f32, dmin * m2 as f32);
                for l in 0..32 {
                    let h1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let h2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    y[o + l] = d1 * ((ql[l] & 0xF) as i32 + h1) as f32 - mm1;
                    y[o + 32 + l] = d2 * ((ql[l] >> 4) as i32 + h2) as f32 - mm2;
                }
                o += 64;
                u1 <<= 2;
                u2 <<= 2;
            }
        }

        // ql[128] | qh[64] | scales[16] (i8) | d (F16)
        QuantType::Q6_K => {
            let d = f16(b, 208);
            for n in (0..256).step_by(128) {
                let ql = &b[n / 2..];
                let qh = &b[128 + n / 4..];
                let sc = &b[192 + n / 16..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) as i32 | ((qh[l] & 3) as i32) << 4) - 32;
                    let q2 = ((ql[l + 32] & 0xF) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                    let q3 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                    let q4 = ((ql[l + 32] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                    y[n + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                    y[n + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                    y[n + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                    y[n + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
                }
            }
        }

        // d (F16) | scales_h (u16) | scales_l[4] | qs[128]
        QuantType::IQ4_XS => {
            let d = f16(b, 0);
            let scales_h = u16::from_le_bytes([b[2], b[3]]);
            let scales_l = &b[4..8];
            for ib in 0..8usize {
                let ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                    | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
                let dl = d * (ls - 32) as f32;
                let qs = &b[8 + 16 * ib..];
                for j in 0..16 {
                    y[32 * ib + j] = dl * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
                    y[32 * ib + j + 16] = dl * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
                }
            }
        }

        other => unreachable!("{}: SuperBlock without a decoder", other.desc().name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant_table::QUANT_FORMATS;

    /// Every format declared `Portable` must actually decode, and every format
    /// declared `NativeIntrinsic` must refuse rather than return zeros. Same
    /// cardinality on both sides is the required check in test form: a row added to
    /// the canonical table without a decoder fails here, not on a device.
    #[test]
    fn portability_matches_the_decoder_table() {
        for q in QUANT_FORMATS {
            let d = q.desc();
            let n = d.block_elements as usize;
            let src = vec![0x11u8; d.block_bytes as usize];
            let got = dequantize_row(q, &src, n);
            match d.lowering {
                Lowering::Portable => {
                    let v = got.unwrap_or_else(|e| panic!("{}: {e}", d.name));
                    assert_eq!(v.len(), n, "{}", d.name);
                }
                Lowering::NativeIntrinsic => assert_eq!(
                    got,
                    Err(QuantError::NoPortableDecoder { format: q }),
                    "{}: a silent decoder would be worse than none",
                    d.name
                ),
            }
        }
    }

    /// A partial block is refused before any decoding: the same rule the
    /// dispatch side owes an op whose `ne[0]` is not a multiple of the block.
    #[test]
    fn a_partial_block_is_rejected() {
        let q = QuantType::Q4_0;
        assert_eq!(
            dequantize_row(q, &[0u8; 18], 16),
            Err(QuantError::NotBlockAligned { format: q, n: 16 })
        );
        assert_eq!(
            dequantize_row(q, &[0u8; 10], 32),
            Err(QuantError::ShortInput {
                format: q,
                need: 18,
                got: 10
            })
        );
    }

    /// F16 conversion, including the subnormal path an oracle must not round.
    #[test]
    fn half_conversion_is_exact() {
        assert_eq!(half_to_f32(0x3c00), 1.0);
        assert_eq!(half_to_f32(0xbc00), -1.0);
        assert_eq!(half_to_f32(0x0000), 0.0);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24)); // smallest subnormal
        assert_eq!(half_to_f32(0x0400), 2f32.powi(-14)); // smallest normal
        assert_eq!(half_to_f32(0x7bff), 65504.0);
    }

    /// The E8M0 half-scale MXFP4 expects, at both ends of the exponent range.
    #[test]
    fn e8m0_half_scale_matches_ggml() {
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(129), 2.0);
        // 2^-127 and 2^-128 are binary32 subnormals: `powi` underflows to 0,
        // so the reference is the bit pattern ggml writes.
        assert_eq!(e8m0_to_f32_half(1), f32::from_bits(0x0040_0000));
        assert_eq!(e8m0_to_f32_half(0), f32::from_bits(0x0020_0000));
    }

    /// Q8_0 against the hand-computed reference of the original descriptor:
    /// one F16 scale then signed bytes, no interleaving.
    #[test]
    fn q8_0_decodes_scale_times_byte() {
        let mut blk = vec![0u8; 34];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        for j in 0..32 {
            blk[2 + j] = (j as i8 - 16) as u8;
        }
        let v = dequantize_row(QuantType::Q8_0, &blk, 32).unwrap();
        for (j, got) in v.iter().enumerate() {
            assert_eq!(*got, (j as i32 - 16) as f32);
        }
    }

    /// Q4_0 interleaving: low nibbles first half, high nibbles second half.
    #[test]
    fn q4_0_splits_nibbles_across_the_block() {
        let mut blk = vec![0u8; 18];
        blk[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1.0
        for j in 0..16 {
            blk[2 + j] = 0x80 | 0x01; // low = 1, high = 8
        }
        let v = dequantize_row(QuantType::Q4_0, &blk, 32).unwrap();
        assert!(v[..16].iter().all(|x| *x == -7.0), "nibble bas = 1 - 8");
        assert!(v[16..].iter().all(|x| *x == 0.0), "nibble haut = 8 - 8");
    }
}
