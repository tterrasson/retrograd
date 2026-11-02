//! Numbers printed as nlohmann::json's `dump()` prints them.
//!
//! The decision server writes some values into the prompt as text with
//! `dump()`: the scalars of a flattened `kev` input, the canonical legend of a
//! score. A float there is printed by nlohmann's own Grisu2 and in its own
//! layout - `1e+20`, `1.5e-07`, `100000.0` - where `serde_json` prints the
//! shortest representation in another layout (`1e20`, `1.5e-7`). Grisu2 does
//! not always find the shortest digits (`4.6989500000000003e+20` where the
//! shortest is `4.69895e20`), so matching the layout is not enough: the digit
//! generation is ported as written, constants and rounding included.

use serde_json::Number;

/// `dump()` of a JSON number: integers as written, floats through Grisu2.
pub fn dump_number(number: &Number) -> String {
    if let Some(value) = number.as_u64() {
        return value.to_string();
    }
    if let Some(value) = number.as_i64() {
        return value.to_string();
    }
    match number.as_f64() {
        Some(value) => dump_f64(value),
        // A `Number` that is neither an integer nor an f64 does not exist
        // without serde_json's arbitrary precision, which this build lacks.
        None => number.to_string(),
    }
}

/// `dump()` of a float.
pub fn dump_f64(value: f64) -> String {
    // nlohmann writes a NaN or an infinity as `null`.
    if !value.is_finite() {
        return "null".into();
    }
    let mut out = String::new();
    if value.is_sign_negative() {
        out.push('-');
    }
    if value == 0.0 {
        out.push_str("0.0");
        return out;
    }
    let (digits, decimal_exponent) = grisu2(value.abs());
    format_buffer(&mut out, &digits, decimal_exponent);
    out
}

/// The smallest and largest decimal exponents written without an exponent:
/// nlohmann's `kMinExp = -4` and `kMaxExp = digits10 = 15`.
const MIN_EXP: i32 = -4;
const MAX_EXP: i32 = 15;

/// nlohmann's `format_buffer`: `digits * 10^decimal_exponent`, as `123.45`,
/// `1230.0`, `0.00123` or `1.23e+45`.
fn format_buffer(out: &mut String, digits: &[u8], decimal_exponent: i32) {
    // At most 17 digits: the widening cannot fail.
    let k = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let n = k + decimal_exponent;
    let text = |range: std::ops::Range<usize>| -> &str {
        // The buffer holds ASCII digits only.
        std::str::from_utf8(&digits[range]).unwrap_or_default()
    };
    if k <= n && n <= MAX_EXP {
        // digits[000].0
        out.push_str(text(0..digits.len()));
        out.extend(std::iter::repeat_n(
            '0',
            usize::try_from(n - k).unwrap_or(0),
        ));
        out.push_str(".0");
    } else if 0 < n && n <= MAX_EXP {
        // dig.its
        let split = usize::try_from(n).unwrap_or(0);
        out.push_str(text(0..split));
        out.push('.');
        out.push_str(text(split..digits.len()));
    } else if MIN_EXP < n && n <= 0 {
        // 0.[000]digits
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', usize::try_from(-n).unwrap_or(0)));
        out.push_str(text(0..digits.len()));
    } else {
        // d[.igits]e±XX
        out.push_str(text(0..1));
        if k > 1 {
            out.push('.');
            out.push_str(text(1..digits.len()));
        }
        let exponent = n - 1;
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        let magnitude = exponent.unsigned_abs();
        if magnitude < 10 {
            out.push('0');
        }
        out.push_str(&magnitude.to_string());
    }
}

/// `f * 2^e`.
#[derive(Clone, Copy)]
struct DiyFp {
    f: u64,
    e: i32,
}

impl DiyFp {
    fn sub(self, other: Self) -> Self {
        Self {
            f: self.f - other.f,
            e: self.e,
        }
    }

    /// The upper 64 bits of the product, rounded half up.
    fn mul(self, other: Self) -> Self {
        let (u_lo, u_hi) = (self.f & 0xFFFF_FFFF, self.f >> 32);
        let (v_lo, v_hi) = (other.f & 0xFFFF_FFFF, other.f >> 32);
        let p0 = u_lo * v_lo;
        let p1 = u_lo * v_hi;
        let p2 = u_hi * v_lo;
        let p3 = u_hi * v_hi;
        let mut q = (p0 >> 32) + (p1 & 0xFFFF_FFFF) + (p2 & 0xFFFF_FFFF);
        q += 1 << 31;
        let h = p3 + (p2 >> 32) + (p1 >> 32) + (q >> 32);
        Self {
            f: h,
            e: self.e + other.e + 64,
        }
    }

    fn normalize(mut self) -> Self {
        while self.f >> 63 == 0 {
            self.f <<= 1;
            self.e -= 1;
        }
        self
    }

    fn normalize_to(self, exponent: i32) -> Self {
        let delta = self.e - exponent;
        Self {
            f: self.f << delta,
            e: exponent,
        }
    }
}

const ALPHA: i32 = -60;

/// `(f, e, k)` with `f * 2^e ≈ 10^k`, nlohmann's `kCachedPowers`.
const CACHED_POWERS: [(u64, i32, i32); 79] = [
    (0xAB70FE17C79AC6CA, -1060, -300),
    (0xFF77B1FCBEBCDC4F, -1034, -292),
    (0xBE5691EF416BD60C, -1007, -284),
    (0x8DD01FAD907FFC3C, -980, -276),
    (0xD3515C2831559A83, -954, -268),
    (0x9D71AC8FADA6C9B5, -927, -260),
    (0xEA9C227723EE8BCB, -901, -252),
    (0xAECC49914078536D, -874, -244),
    (0x823C12795DB6CE57, -847, -236),
    (0xC21094364DFB5637, -821, -228),
    (0x9096EA6F3848984F, -794, -220),
    (0xD77485CB25823AC7, -768, -212),
    (0xA086CFCD97BF97F4, -741, -204),
    (0xEF340A98172AACE5, -715, -196),
    (0xB23867FB2A35B28E, -688, -188),
    (0x84C8D4DFD2C63F3B, -661, -180),
    (0xC5DD44271AD3CDBA, -635, -172),
    (0x936B9FCEBB25C996, -608, -164),
    (0xDBAC6C247D62A584, -582, -156),
    (0xA3AB66580D5FDAF6, -555, -148),
    (0xF3E2F893DEC3F126, -529, -140),
    (0xB5B5ADA8AAFF80B8, -502, -132),
    (0x87625F056C7C4A8B, -475, -124),
    (0xC9BCFF6034C13053, -449, -116),
    (0x964E858C91BA2655, -422, -108),
    (0xDFF9772470297EBD, -396, -100),
    (0xA6DFBD9FB8E5B88F, -369, -92),
    (0xF8A95FCF88747D94, -343, -84),
    (0xB94470938FA89BCF, -316, -76),
    (0x8A08F0F8BF0F156B, -289, -68),
    (0xCDB02555653131B6, -263, -60),
    (0x993FE2C6D07B7FAC, -236, -52),
    (0xE45C10C42A2B3B06, -210, -44),
    (0xAA242499697392D3, -183, -36),
    (0xFD87B5F28300CA0E, -157, -28),
    (0xBCE5086492111AEB, -130, -20),
    (0x8CBCCC096F5088CC, -103, -12),
    (0xD1B71758E219652C, -77, -4),
    (0x9C40000000000000, -50, 4),
    (0xE8D4A51000000000, -24, 12),
    (0xAD78EBC5AC620000, 3, 20),
    (0x813F3978F8940984, 30, 28),
    (0xC097CE7BC90715B3, 56, 36),
    (0x8F7E32CE7BEA5C70, 83, 44),
    (0xD5D238A4ABE98068, 109, 52),
    (0x9F4F2726179A2245, 136, 60),
    (0xED63A231D4C4FB27, 162, 68),
    (0xB0DE65388CC8ADA8, 189, 76),
    (0x83C7088E1AAB65DB, 216, 84),
    (0xC45D1DF942711D9A, 242, 92),
    (0x924D692CA61BE758, 269, 100),
    (0xDA01EE641A708DEA, 295, 108),
    (0xA26DA3999AEF774A, 322, 116),
    (0xF209787BB47D6B85, 348, 124),
    (0xB454E4A179DD1877, 375, 132),
    (0x865B86925B9BC5C2, 402, 140),
    (0xC83553C5C8965D3D, 428, 148),
    (0x952AB45CFA97A0B3, 455, 156),
    (0xDE469FBD99A05FE3, 481, 164),
    (0xA59BC234DB398C25, 508, 172),
    (0xF6C69A72A3989F5C, 534, 180),
    (0xB7DCBF5354E9BECE, 561, 188),
    (0x88FCF317F22241E2, 588, 196),
    (0xCC20CE9BD35C78A5, 614, 204),
    (0x98165AF37B2153DF, 641, 212),
    (0xE2A0B5DC971F303A, 667, 220),
    (0xA8D9D1535CE3B396, 694, 228),
    (0xFB9B7CD9A4A7443C, 720, 236),
    (0xBB764C4CA7A44410, 747, 244),
    (0x8BAB8EEFB6409C1A, 774, 252),
    (0xD01FEF10A657842C, 800, 260),
    (0x9B10A4E5E9913129, 827, 268),
    (0xE7109BFBA19C0C9D, 853, 276),
    (0xAC2820D9623BF429, 880, 284),
    (0x80444B5E7AA7CF85, 907, 292),
    (0xBF21E44003ACDD2D, 933, 300),
    (0x8E679C2F5E44FF8F, 960, 308),
    (0xD433179D9C8CB841, 986, 316),
    (0x9E19DB92B4E31BA9, 1013, 324),
];

/// The cached power that brings a binary exponent `e` into [α, γ].
fn cached_power(e: i32) -> (DiyFp, i32) {
    const MIN_DEC_EXP: i32 = -300;
    const DEC_STEP: i32 = 8;
    let f = ALPHA - e - 1;
    // `k = ceil(f * log10(2))`, as nlohmann computes it.
    let k = (f * 78913) / (1 << 18) + i32::from(f > 0);
    let index = (-MIN_DEC_EXP + k + (DEC_STEP - 1)) / DEC_STEP;
    // In range for every finite double: e is within [-1137, 960].
    let (f, e, k) = CACHED_POWERS[usize::try_from(index).unwrap_or(0)];
    (DiyFp { f, e }, k)
}

/// `(k, 10^(k-1))` with `10^(k-1) <= n < 10^k`.
fn largest_pow10(n: u32) -> (i32, u32) {
    let mut pow10 = 1_000_000_000_u32;
    let mut k = 10;
    while k > 1 && n < pow10 {
        pow10 /= 10;
        k -= 1;
    }
    (k, pow10)
}

fn round_last(buffer: &mut [u8], dist: u64, delta: u64, mut rest: u64, ten_k: u64) {
    let Some(last) = buffer.last_mut() else {
        return;
    };
    while rest < dist
        && delta - rest >= ten_k
        && (rest + ten_k < dist || dist - rest > rest + ten_k - dist)
    {
        *last -= 1;
        rest += ten_k;
    }
}

/// The digits and the decimal exponent of a finite positive double.
fn grisu2(value: f64) -> (Vec<u8>, i32) {
    const PRECISION: u32 = 53;
    const BIAS: i32 = 1023 + 52;
    const MIN_EXP_BIN: i32 = 1 - BIAS;
    const HIDDEN_BIT: u64 = 1 << (PRECISION - 1);

    let bits = value.to_bits();
    let biased = bits >> (PRECISION - 1);
    let fraction = bits & (HIDDEN_BIT - 1);
    // Eleven bits of exponent: the narrowing cannot truncate.
    let biased_exponent = i32::try_from(biased).unwrap_or(0);
    let v = if biased == 0 {
        DiyFp {
            f: fraction,
            e: MIN_EXP_BIN,
        }
    } else {
        DiyFp {
            f: fraction + HIDDEN_BIT,
            e: biased_exponent - BIAS,
        }
    };
    let lower_boundary_is_closer = fraction == 0 && biased > 1;
    let m_plus = DiyFp {
        f: 2 * v.f + 1,
        e: v.e - 1,
    };
    let m_minus = if lower_boundary_is_closer {
        DiyFp {
            f: 4 * v.f - 1,
            e: v.e - 2,
        }
    } else {
        DiyFp {
            f: 2 * v.f - 1,
            e: v.e - 1,
        }
    };
    let w_plus = m_plus.normalize();
    let w_minus = m_minus.normalize_to(w_plus.e);
    let v = v.normalize();

    let (c_minus_k, cached_k) = cached_power(w_plus.e);
    let w = v.mul(c_minus_k);
    let w_minus = w_minus.mul(c_minus_k);
    let w_plus = w_plus.mul(c_minus_k);
    let m_minus = DiyFp {
        f: w_minus.f + 1,
        e: w_minus.e,
    };
    let m_plus = DiyFp {
        f: w_plus.f - 1,
        e: w_plus.e,
    };
    let mut decimal_exponent = -cached_k;
    let mut buffer = Vec::with_capacity(17);

    let mut delta = m_plus.sub(m_minus).f;
    let mut dist = m_plus.sub(w).f;
    let shift = u32::try_from(-m_plus.e).unwrap_or(0);
    let one = 1_u64 << shift;
    // `-e >= 32`: the integral part fits 32 bits.
    let mut p1 = u32::try_from(m_plus.f >> shift).unwrap_or(u32::MAX);
    let mut p2 = m_plus.f & (one - 1);

    let (k, mut pow10) = largest_pow10(p1);
    let mut n = k;
    while n > 0 {
        let digit = p1 / pow10;
        p1 %= pow10;
        // One decimal digit.
        buffer.push(b'0' + u8::try_from(digit).unwrap_or(0));
        n -= 1;
        let rest = (u64::from(p1) << shift) + p2;
        if rest <= delta {
            decimal_exponent += n;
            round_last(&mut buffer, dist, delta, rest, u64::from(pow10) << shift);
            return (buffer, decimal_exponent);
        }
        pow10 /= 10;
    }
    let mut m = 0;
    loop {
        p2 *= 10;
        let digit = p2 >> shift;
        p2 &= one - 1;
        buffer.push(b'0' + u8::try_from(digit).unwrap_or(0));
        m += 1;
        delta *= 10;
        dist *= 10;
        if p2 <= delta {
            break;
        }
    }
    decimal_exponent -= m;
    round_last(&mut buffer, dist, delta, p2, one);
    (buffer, decimal_exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Printed by nlohmann::json 3.12 (the copy vendored with llama.cpp), one
    /// `json(double).dump()` per line, among them every layout and cases where
    /// Grisu2 is not the shortest representation.
    const REFERENCE: &[(u64, &str)] = &[
        (0xc151df7d6ee5e2d6, "-4685301.732781133"),
        (0x4105420800000000, "174145.0"),
        (0x4247dd645e8fadee, "204997180703.35883"),
        (0x461c9cf62eb9fcb6, "5.66741457085718e+29"),
        (0xbaa602f9af703e45, "-3.5561693938148423e-26"),
        (0x4439791a64e3bc2e, "4.6989500000000003e+20"),
        (0x43166f6d21fd4801, "1.5787410000000002e+15"),
        (0x42d183fdff000288, "77034398941194.12"),
        (0x9a431317cf6030d0, "-3.5912667368320656e-182"),
        (0x43122c235e65cfff, "1.2787699999999997e+15"),
        (0x44710ad45cdeaf6e, "5.030000000000001e+21"),
    ];

    #[test]
    fn floats_print_as_nlohmann_dump_prints_them() {
        for &(bits, expected) in REFERENCE {
            assert_eq!(dump_f64(f64::from_bits(bits)), expected, "{bits:016x}");
        }
    }

    #[test]
    fn every_layout_of_format_buffer() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100000.0, "100000.0"),
            (1e15, "1e+15"),
            (123456789012345.0, "123456789012345.0"),
            (0.5, "0.5"),
            (0.001, "0.001"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (1e20, "1e+20"),
            (1e100, "1e+100"),
            (-2.5e-300, "-2.5e-300"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (f64::NAN, "null"),
            (f64::INFINITY, "null"),
        ];
        for (value, expected) in cases {
            assert_eq!(dump_f64(value), expected, "{value:e}");
        }
    }

    #[test]
    fn integers_print_as_written() {
        let parse = |text: &str| -> Number { serde_json::from_str(text).expect("number") };
        assert_eq!(dump_number(&parse("42")), "42");
        assert_eq!(dump_number(&parse("-7")), "-7");
        assert_eq!(
            dump_number(&parse("18446744073709551615")),
            "18446744073709551615"
        );
        assert_eq!(dump_number(&parse("3.0")), "3.0");
        assert_eq!(dump_number(&parse("1e5")), "100000.0");
        assert_eq!(dump_number(&parse("2.5E+21")), "2.5e+21");
    }
}
