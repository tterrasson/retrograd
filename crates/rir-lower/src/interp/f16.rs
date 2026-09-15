//! The interpreter's F16 conversions.
//!
//! **Deliberately independent of `rir_core::quant`, and labelled as such**:
//! the oracle must not share code with what it
//! judges. `rir_core::quant::half_to_f32` decodes the canonical quantized
//! tables, `rir_emit::cpu` prints a third pair into generated Rust, and these
//! two are what the parity oracle executes. Three copies on purpose, one
//! witness against drift: `rir-kernels/tests/f16_oracles.rs` compares them over
//! the entire 65 536-value input space. A shared helper here would make that
//! test tautological - it would compare a function with itself.

/// Round-to-nearest-even narrowing, the rounding `half(x)` and `float16_t(x)`
/// perform (ADR-3 section 6). Written here rather than taken from a crate
/// for the reason every oracle in RIR is written by hand: it is the second,
/// independent witness of what a shader does.
pub fn f32_to_f16(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x007F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x0200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = (m >> shift) as u16;
        let rest = m & ((1 << shift) - 1);
        let tie = 1u32 << (shift - 1);
        let round = u16::from(rest > tie || (rest == tie && half & 1 == 1));
        return sign | (half + round);
    }
    let half = ((e as u32) << 10) as u16 | (man >> 13) as u16;
    let rest = man & 0x1FFF;
    let round = u16::from(rest > 0x1000 || (rest == 0x1000 && half & 1 == 1));
    sign | (half + round)
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31
        } else {
            // `man · 2^-24`, renormalized: the binary32 exponent field is
            // `113 - e`, which is 103 for the smallest half. `112 - e` here
            // halved every subnormal, and the generated CPU decoder said the
            // same thing, so the two agreed and no parity test could see it.
            let mut m = man;
            let mut e: i32 = 0;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            (sign << 31) | (((113 - e) as u32) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        (sign << 31) | 0x7F80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}
