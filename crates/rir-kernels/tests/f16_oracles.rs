//! The three F16 conversions, compared over their **whole** input space.
//!
//! RIR carries three copies of binary16↔binary32 on purpose, each labelled where
//! it lives:
//!
//! 1. `rir_core::quant::half_to_f32` - the reference decoder of the canonical
//!    quantized tables, used by `dequantize_row`;
//! 2. `rir_lower::interp::{f16_to_f32, f32_to_f16}` - what the parity oracle
//!    executes, written by hand so that the oracle does not share code with what
//!    it judges;
//! 3. the pair `rir_emit::cpu` prints into `generated/rir/*/cpu.rs` - which
//!    *cannot* call either of the others, since emitted files have no dependency
//!    on any RIR crate.
//!
//! The question was never "four copies, keep one". Sharing an implementation
//! would make every F16 assertion in the repository tautological: the oracle and
//! the shader-side decoder would be the same function, and a wrong-but-agreeing
//! answer would pass. What the copies need is not a refactor but a witness, and
//! F16 admits the strongest one available: **there are 65 536 inputs**, so the
//! agreement is not sampled, it is enumerated.
//!
//! The generated side is read from a committed artifact rather than re-emitted,
//! for the reason the device lane reads committed files: a stale artifact is one
//! of the failures worth catching.

/// The committed generated CPU kernel of an F16 band, included for its two
/// helpers. `add_f16` is the smallest generated file that carries both.
#[allow(dead_code)]
mod generated {
    include!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../generated/rir/add_f16/cpu.rs"
    ));

    // The emitted helpers are private in the file that declares them - an
    // emitted file exports its kernel and nothing else. Two wrappers in the same
    // module reach them without the emitter printing one byte differently for a
    // test's benefit.
    pub fn widen(h: u16) -> f32 {
        f16_to_f32(h)
    }

    pub fn narrow(f: f32) -> u16 {
        f32_to_f16(f)
    }
}

/// Every half, decoded by the three copies, compared **bit for bit**.
///
/// Bits and not `==`: a NaN is not equal to itself, and the payload of a decoded
/// NaN is exactly the kind of detail two hand-written decoders disagree on.
#[test]
fn the_three_f16_decoders_agree_on_all_65536_halves() {
    for h in 0u16..=u16::MAX {
        let core = rir_core::quant::half_to_f32(h).to_bits();
        let oracle = rir_lower::interp::f16_to_f32(h).to_bits();
        let emitted = generated::widen(h).to_bits();
        assert_eq!(
            core, oracle,
            "half {h:#06x}: rir_core {core:#010x} vs interpreter {oracle:#010x}"
        );
        assert_eq!(
            core, emitted,
            "half {h:#06x}: rir_core {core:#010x} vs generated CPU {emitted:#010x}"
        );
    }
}

/// The two narrowings, on the exhaustive set of values that *have* an exact
/// half: every half decoded to F32 must narrow back to the half it came from,
/// and both copies must agree on it.
///
/// The canonical tables are read and never written, which is why `rir_core` has
/// no narrowing to compare here - two copies, not three.
#[test]
fn the_two_f16_narrowings_round_trip_every_half() {
    for h in 0u16..=u16::MAX {
        let x = rir_core::quant::half_to_f32(h);
        let oracle = rir_lower::interp::f32_to_f16(x);
        let emitted = generated::narrow(x);
        assert_eq!(
            oracle, emitted,
            "half {h:#06x} ({x}): interpreter {oracle:#06x} vs generated CPU {emitted:#06x}"
        );
        // A NaN keeps a non-zero mantissa but not necessarily *its* mantissa,
        // which is what both copies say in writing; every other half round-trips
        // exactly.
        let is_nan = h & 0x7C00 == 0x7C00 && h & 0x03FF != 0;
        if !is_nan {
            assert_eq!(oracle, h, "half {h:#06x} does not round-trip ({x})");
        }
    }
}

/// The rounding rule itself, on the values that decide it: ties, subnormal
/// boundaries, and the overflow edge.
///
/// The round trip above never exercises rounding - every input it uses is
/// already representable. These are the inputs where "round to nearest even" is
/// a choice, and where a copy that truncated instead would still pass every
/// other test in the repository.
#[test]
fn the_two_f16_narrowings_agree_where_rounding_decides() {
    let mut cases: Vec<f32> = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        // Just over the largest half: 65504 is exact, 65520 is the tie that
        // rounds to infinity.
        65504.0,
        65519.0,
        65520.0,
        // The subnormal floor: 2^-24 is the smallest half, half of it is the
        // tie that rounds to zero.
        5.960_464_5e-8,
        2.980_232_2e-8,
        2.980_232e-8,
        // Ties between two adjacent halves, on both sides of the even rule.
        1.000_488_3,
        1.001_464_8,
    ];
    // A deterministic sweep across the whole exponent range, so the tie cases
    // above are not the only mantissas tried.
    let mut seed = 0xf16_2026u64;
    for _ in 0..20_000 {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let x = f32::from_bits((seed >> 32) as u32);
        if x.is_nan() {
            continue;
        }
        cases.push(x);
    }
    for x in cases {
        let oracle = rir_lower::interp::f32_to_f16(x);
        let emitted = generated::narrow(x);
        assert_eq!(
            oracle,
            emitted,
            "{x} ({:#010x}): interpreter {oracle:#06x} vs generated CPU {emitted:#06x}",
            x.to_bits()
        );
    }
}
