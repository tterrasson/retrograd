//! The quantized-format table and the RIR dequantization oracle,
//! checked against ggml itself, then the
//! **expansion** checked against the oracle.
//!
//! Four questions, and they are deliberately separate:
//!
//! 1. does the canonical table describe the same block geometry ggml uses?
//! 2. does the portable oracle decode the same bytes to the same values as
//!    ggml's `to_float`?
//! 3. do the two sides enumerate the same set of formats?
//! 4. does the **Loop IR expansion** decode those same bytes to those same
//!    values, element by element, with no device and no kernel?
//!
//! The first is what makes the numbers in `rir_core::quant_formats()` safe
//! to restate: the generated header must stay legal MSL, so it cannot include
//! `ggml-common.h` and ask. The second is the decoding contract itself. The third
//! is the required check - a format known to one side and not the other is exactly the
//! drift the whole table exists to make impossible.
//!
//! The fourth is what makes ten formats a project rather than a swamp. Without
//! it, an expansion would only be observable through a kernel, on a device, with
//! a fixture per shape. Here the chain is complete and every link is independent:
//! ggml decodes the bytes, the hand-written oracle decodes the same bytes, and
//! the description-driven expansion decodes them a third time - the description
//! drives the expansion and *never* the oracle, which is what keeps the last
//! comparison from being a table checked against itself.

use rir_core::quant_table::{QUANT_FORMATS, QuantType};
use rir_core::types::Lowering;

/// A deterministic F32 row: mixed magnitudes and signs, so a decoder that drops
/// the sign bit or the high nibble cannot pass by luck.
fn fixture(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32) * 0.037 - 1.7;
            x.sin() * (1.0 + (i % 7) as f32 * 0.25)
        })
        .collect()
}

#[test]
fn the_table_declares_the_block_geometry_ggml_actually_uses() {
    for q in QUANT_FORMATS {
        let d = q.desc();
        let Some(id) = retrograd_engine::ggml_type_id(d.name) else {
            // The `dequant` table is the one the FFI enumerates; OUT_PROD-only
            // rows are covered by the identity test below.
            assert!(
                !d.ops.dequant,
                "{}: missing from retro_dequant_types()",
                d.name
            );
            continue;
        };
        let (be, bytes) =
            retrograd_engine::quant_traits(id).unwrap_or_else(|| panic!("{}", d.name));
        assert_eq!(
            (be as u32, bytes as u32),
            (d.block_elements, d.block_bytes),
            "{}: declared block geometry != ggml - fix rir_core::quant_formats()",
            d.name
        );
    }
}

#[test]
fn the_portable_oracle_decodes_exactly_like_ggml() {
    let mut checked = 0usize;
    for q in QUANT_FORMATS {
        let d = q.desc();
        if d.lowering != Lowering::Portable {
            continue;
        }
        let Some(id) = retrograd_engine::ggml_type_id(d.name) else {
            continue;
        };

        // Four whole blocks: enough for the per-sub-block scales of a K quant
        // to differ from one another.
        let n = 4 * d.block_elements as usize;
        let src = fixture(n);
        let Some((bytes, expected)) = retrograd_engine::quant_roundtrip(id, &src) else {
            panic!("{}: ggml refused to quantize {n} values", d.name);
        };
        let got =
            rir_core::dequantize_row(q, &bytes, n).unwrap_or_else(|e| panic!("{}: {e}", d.name));

        // Bit-for-bit: both sides run the same formula on the same bytes in
        // F32. A tolerance here would hide a wrong nibble in the low bits of a
        // small scale, which is precisely the failure worth catching.
        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: element {i} - oracle {a} != ggml {b}",
                d.name
            );
        }
        checked += 1;
    }
    assert!(checked >= 12, "only {checked} portable formats tested");
}

#[test]
fn a_native_intrinsic_format_refuses_rather_than_guesses() {
    let mut native = 0usize;
    for q in QUANT_FORMATS {
        let d = q.desc();
        if d.lowering == Lowering::Portable {
            continue;
        }
        native += 1;
        let src = vec![0u8; 4 * d.block_bytes as usize];
        let err = rir_core::dequantize_row(q, &src, 4 * d.block_elements as usize)
            .expect_err("a native_intrinsic format must not decode");
        assert_eq!(
            err,
            rir_core::QuantError::NoPortableDecoder { format: q },
            "{}",
            d.name
        );
    }
    // The distinction only means something while both sides are non-empty: a
    // table where everything became portable would silently drop the contract.
    assert!(
        native > 0,
        "no native_intrinsic format left - the refusal has nothing to check"
    );
}

#[test]
fn both_sides_enumerate_the_same_dequant_table() {
    let ggml: Vec<String> = retrograd_engine::dequant_types()
        .into_iter()
        .map(|(_, n)| n)
        .collect();
    let rir: Vec<&str> = QUANT_FORMATS
        .iter()
        .filter(|q| q.desc().ops.dequant)
        .map(|q| q.desc().name)
        .collect();
    assert_eq!(
        rir.len(),
        ggml.len(),
        "different cardinality: RIR {rir:?} vs GGML_RETRO_DEQUANT_TYPES {ggml:?}"
    );
    for name in &rir {
        assert!(
            ggml.iter().any(|g| g == name),
            "{name} missing from the ggml table"
        );
    }
}

/// The `q8_0` descriptor the pilot kernel was written against - 32 elements, 34
/// bytes, payload at offset 2 - must be what the canonical table says. The pilot
/// kernel's addressing depends on those three numbers.
#[test]
fn the_pilot_descriptor_survived_the_move_to_the_canonical_table() {
    let d = QuantType::Q8_0.desc();
    assert_eq!(
        (d.block_elements, d.block_bytes, d.data_offset),
        (32, 34, 2)
    );
}

// ---- The expansion, against the oracle, in isolation  ---------------

/// `dst[col,row] = dequant(src[col,row])`: the smallest kernel whose lowering
/// goes through the quantized decoder and nothing else.
fn dequant_copy(format: QuantType) -> rir_core::ValidatedKernel {
    use rir_core::{DType, Extent, KernelBuilder, Layout, TensorType};
    let mut k = KernelBuilder::new("dequant_copy");
    let src = k.input(
        "src",
        TensorType {
            dtype: DType::Quant(format),
            rank: 2,
            layout: Layout::Ggml,
        },
    );
    let dst = k.output("dst", TensorType::f32(2));
    let col = k.axis("col", Extent::Dim { arg: src, dim: 0 });
    let row = k.axis("row", Extent::Dim { arg: src, dim: 1 });
    let v = k.read(src, &[col, row]);
    k.write(dst, &[col, row], v);
    k.finish().expect("dequant_copy: invalid kernel")
}

/// Every format the table describes a layout for decodes, through the Loop IR,
/// **exactly** what the oracle decodes from the same bytes.
///
/// Bit-for-bit, and for the reason the ggml comparison above is: both sides run
/// the same formula in F32 on the same bytes, so a tolerance here would hide a
/// wrong nibble in the low bits of a small scale - the one failure worth
/// catching. It is also what pins the *association* of the formula: `d·sc·q −
/// dmin·m` re-associated is still the right value and no longer the same float.
#[test]
fn every_described_format_expands_to_what_the_oracle_decodes() {
    use rir_lower::interp::{BoundArg, TensorViewBytes, TensorViewMut, run};
    let mut described = 0usize;
    for q in QUANT_FORMATS {
        let d = q.desc();
        if !rir_lower::can_lower_dequant(q) {
            continue;
        }
        described += 1;
        let be = d.block_elements as usize;
        let bb = d.block_bytes as usize;
        // Four blocks per row and two rows: enough for the per-sub-block
        // scales of a K quant to differ from one another, and enough for the
        // second dimension's stride to be exercised.
        let (blocks_per_row, rows) = (4usize, 2usize);
        let mut seed = 0xF1_2026u64 ^ d.id as u64;
        let raw = rir_core::random_block_bytes(q, blocks_per_row * rows, &mut seed)
            .unwrap_or_else(|| panic!("{}: format without a description", d.name));

        let n_col = blocks_per_row * be;
        let row_bytes = blocks_per_row * bb;
        let mut expected = Vec::with_capacity(n_col * rows);
        for r in 0..rows {
            expected.extend(
                rir_core::dequantize_row(q, &raw[r * row_bytes..(r + 1) * row_bytes], n_col)
                    .unwrap_or_else(|e| panic!("{}: {e}", d.name)),
            );
        }

        let kernel = dequant_copy(q);
        let lk = rir_lower::lower(&kernel, rir_lower::Schedule::cpu_serial())
            .unwrap_or_else(|e| panic!("{}: {e}", d.name));
        let mut got = vec![0f32; n_col * rows];
        let mut args = [
            BoundArg::InBytes(TensorViewBytes {
                data: &raw,
                shape: [n_col, rows, 1, 1],
                nb: [bb, row_bytes, row_bytes * rows, row_bytes * rows],
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut got,
                shape: [n_col, rows, 1, 1],
                nb: [4, 4 * n_col, 4 * n_col * rows, 4 * n_col * rows],
            }),
        ];
        run(&lk, &mut args, &[]).unwrap_or_else(|e| panic!("{}: {e}", d.name));

        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{}: element {i} - expansion {a} != oracle {b}",
                d.name
            );
        }
    }
    // The ten standard formats plus the two the LUT made almost free. A number
    // that drops is a format that stopped being lowerable, which is a decision
    // and must be written in the table rather than discovered here.
    assert_eq!(
        described, 12,
        "12 lowerable formats expected (the ten standard ones, iq4_nl, iq4_xs)"
    );
}

/// The ten standard formats are lowerable by name exactly once.
/// `mxfp4` is **explicitly** absent: its E8M0 scale would need a new scalar
/// conversion op in three emitters, for a format outside the ten and absent
/// from every census.
#[test]
fn the_ten_standard_formats_are_lowerable_and_mxfp4_is_refused() {
    for name in [
        "q2_K", "q3_K", "q4_0", "q4_1", "q4_K", "q5_0", "q5_1", "q5_K", "q6_K", "q8_0",
    ] {
        let q =
            QuantType::from_name(name).unwrap_or_else(|| panic!("{name} missing from the table"));
        assert!(rir_lower::can_lower_dequant(q), "{name}: not lowerable");
    }
    for name in ["iq4_nl", "iq4_xs"] {
        let q = QuantType::from_name(name).unwrap();
        assert!(rir_lower::can_lower_dequant(q), "{name}: not lowerable");
    }
    let mxfp4 = QuantType::from_name("mxfp4").unwrap();
    assert!(
        !rir_lower::can_lower_dequant(mxfp4),
        "mxfp4: explicit refusal expected, not an expansion"
    );
    // Formats without a portable RIR decoder stay out regardless of shape.
    for q in QUANT_FORMATS {
        if q.desc().lowering != Lowering::Portable {
            assert!(
                !rir_lower::can_lower_dequant(q),
                "{}: native_intrinsic lowered",
                q.desc().name
            );
        }
    }
}

/// A format with no description is an explicit `UnsupportedQuantShape`, never
/// an approximation - rule 3 of the compiler, and the only reason `can_lower`
/// may be asked *before* generation rather than discovered during it.
#[test]
fn a_format_without_a_description_fails_lowering_with_its_reason() {
    let mxfp4 = QuantType::from_name("mxfp4").unwrap();
    let kernel = dequant_copy(mxfp4);
    assert!(matches!(
        rir_lower::lower(&kernel, rir_lower::Schedule::cpu_serial()),
        Err(rir_lower::LowerError::UnsupportedQuantShape { .. })
    ));
    let iq2 = QuantType::from_name("iq2_xxs").unwrap();
    let kernel = dequant_copy(iq2);
    assert!(matches!(
        rir_lower::lower(&kernel, rir_lower::Schedule::cpu_serial()),
        Err(rir_lower::LowerError::NativeIntrinsicQuant { .. })
    ));
}
