//! `sum_rows_<format>` - a **fused** quantized loader (scale plus payload, with
//! no materialized dequantized tensor) feeding a per-row reduction.
//!
//! ```text
//! y[row] = Σ_col dequant(x[col, row])
//! ```
//!
//! The write is independent of the inner axis, so lowering hoists it to row
//! scope: one write per row rather than per element.
//!
//! One kernel is built **per format**, from the canonical table
//! (ADR-3 section 2). The kernel body does not change between them: the
//! only difference is the dtype, and lowering expands the right block shape.
//! That is the phase E claim in executable form - a format is a row, not a
//! kernel rewrite. `variants()` is what decides which rows get a kernel, and it
//! asks `rir_lower::can_lower_dequant` rather than listing names.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, Layout, QUANT_FORMATS, QuantType, ReduceOp,
    ReductionSemantics, TensorType, ValidateError, ValidatedKernel,
};

/// Kernel name for a format: `sum_rows_q8_0`.
pub fn kernel_name(format: QuantType) -> String {
    format!("sum_rows_{}", format.desc().name)
}

/// The formats this kernel is generated for: every portable format whose block
/// shape lowering can expand. A format the table declares `NativeIntrinsic`, or
/// whose shape has no Loop IR expansion yet, is skipped here rather than
/// failing generation - the registry stays a statement about what exists, not
/// about what was attempted.
pub fn variants() -> Vec<QuantType> {
    QUANT_FORMATS
        .into_iter()
        .filter(|q| rir_lower::can_lower_dequant(*q) && q.desc().block_elements > 1)
        .collect()
}

pub fn build(format: QuantType) -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new(&kernel_name(format));

    let x = k.input(
        "x",
        TensorType {
            dtype: DType::Quant(format),
            rank: 2,
            layout: Layout::Ggml,
        },
    );
    let y = k.output("y", TensorType::f32(1));

    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });

    // Explicit Read plus Dequant, fused during lowering.
    let xv = k.read(x, &[col, row]);
    let s = k.reduce(ReduceOp::Sum, col, xv, ReductionSemantics::Deterministic);
    k.write(y, &[row], s);

    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::Quant(format)],
    });
    k.constrain(Constraint::Rank { arg: x, max: 2 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorViewBytes, TensorViewMut, run};

    /// Builds `n_row` rows of quantized blocks and the F32 values they encode.
    ///
    /// The bytes are random rather than obtained from a quantizer, so the
    /// reference is the *definition* of the format - the interpreter and the
    /// generated code are then compared against it, not against each other.
    fn build_rows(
        format: QuantType,
        n_col: usize,
        n_row: usize,
        row_stride_bytes: usize,
        seed: &mut u64,
    ) -> (Vec<u8>, Vec<Vec<f32>>) {
        let d = format.desc();
        let (be, bb) = (d.block_elements as usize, d.block_bytes as usize);
        assert_eq!(n_col % be, 0);
        let blocks = n_col / be;
        assert!(row_stride_bytes >= blocks * bb);
        let mut raw = vec![0u8; row_stride_bytes * n_row];
        let mut logical = Vec::new();
        for r in 0..n_row {
            // One fixture, derived from the description
            // (`rir_core::random_block_bytes`, ADR-3), so a new
            // format needs no hand-written encoder here. Random bytes are a *legal* block whatever
            // their value, and the reference is the portable oracle, which
            // `tests/rir_quant_oracle.rs` compares element by element against
            // ggml's own `to_float`. The chain of trust runs ggml → oracle →
            // lowering, with no link written twice.
            let bytes = rir_core::random_block_bytes(format, blocks, seed)
                .unwrap_or_else(|| panic!("{}: format without a description", d.name));
            let at = r * row_stride_bytes;
            raw[at..at + blocks * bb].copy_from_slice(&bytes);
            logical.push(rir_core::dequantize_row(format, &bytes, n_col).unwrap());
        }
        (raw, logical)
    }

    fn reference_sums(logical: &[Vec<f32>]) -> Vec<f32> {
        logical
            .iter()
            .map(|row| row.iter().fold(0.0f32, |a, &v| a + v))
            .collect()
    }

    /// A 2D byte view of `n_col` logical elements per row.
    fn view(
        format: QuantType,
        data: &[u8],
        n_col: usize,
        n_row: usize,
        row_stride_bytes: usize,
    ) -> TensorViewBytes<'_> {
        let d = format.desc();
        let blocks = n_col / d.block_elements as usize;
        assert!(row_stride_bytes >= blocks * d.block_bytes as usize);
        assert!(data.len() >= row_stride_bytes * n_row);
        TensorViewBytes {
            data,
            shape: [n_col, n_row, 1, 1],
            nb: [
                d.block_bytes as usize,
                row_stride_bytes,
                row_stride_bytes * n_row,
                row_stride_bytes * n_row,
            ],
        }
    }

    fn run_case(format: QuantType, n_col: usize, n_row: usize, row_stride_bytes: usize) {
        run_case_with(
            format,
            Schedule::cpu_serial(),
            n_col,
            n_row,
            row_stride_bytes,
        );
    }

    fn run_case_with(
        format: QuantType,
        schedule: Schedule,
        n_col: usize,
        n_row: usize,
        row_stride_bytes: usize,
    ) {
        let kernel = build(format).unwrap();
        let lk = rir_lower::lower(&kernel, schedule).unwrap();

        let mut seed = 0x08_2026_u64 ^ ((n_col as u64) << 24) ^ (n_row as u64);
        let (raw, logical) = build_rows(format, n_col, n_row, row_stride_bytes, &mut seed);
        let expected = reference_sums(&logical);

        let mut got = vec![0f32; n_row];
        let mut args = [
            BoundArg::InBytes(view(format, &raw, n_col, n_row, row_stride_bytes)),
            BoundArg::Out(TensorViewMut::contiguous_1d(&mut got, n_row)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for (r, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            let tol = 1e-4f32.max(1e-5 * e.abs());
            assert!(
                (g - e).abs() <= tol,
                "{} row {r} ({n_col}x{n_row}, stride {row_stride_bytes}): {g} vs {e}",
                format.desc().name
            );
        }
    }

    /// Blocks/bytes for a case sized in blocks, per format.
    fn geometry(format: QuantType, blocks: usize) -> (usize, usize) {
        let d = format.desc();
        (
            blocks * d.block_elements as usize,
            blocks * d.block_bytes as usize,
        )
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for format in variants() {
            for &(blocks, rows) in &[(1usize, 1usize), (2, 5), (3, 3)] {
                let (n_col, bytes) = geometry(format, blocks);
                run_case(format, n_col, rows, bytes);
            }
        }
    }

    /// Under the Vulkan schedule, lanes split the row's blocks, dequantize
    /// their shares, and combine them with `subgroupAdd`. This verifies the
    /// Loop IR half of quantized GPU support; GLSL compilation validates the
    /// corresponding shader.
    #[test]
    fn parity_of_the_lane_lowering_against_the_reference() {
        for format in variants() {
            for &(blocks, rows) in &[(1usize, 1usize), (2, 5), (32, 3)] {
                let (n_col, bytes) = geometry(format, blocks);
                run_case_with(format, Schedule::vulkan_subgroup(), n_col, rows, bytes);
            }
        }
    }

    #[test]
    fn parity_with_padded_rows() {
        // Row strides exceed the bytes occupied by useful blocks.
        for format in variants() {
            let (n_col, bytes) = geometry(format, 1);
            run_case(format, n_col, 4, bytes + 30);
            let (n_col, bytes) = geometry(format, 2);
            run_case(format, n_col, 2, bytes + 12);
        }
    }

    /// The table is what decides which formats exist here. Q8_0 was the pilot;
    /// Q4_0 is the one that proves a *second* block shape goes through the same
    /// path, which is the whole point of phase E.
    #[test]
    fn the_variant_list_follows_the_canonical_table() {
        let names: Vec<&str> = variants().iter().map(|q| q.desc().name).collect();
        assert!(names.contains(&"q8_0"), "{names:?}");
        assert!(names.contains(&"q4_0"), "{names:?}");
        for q in QUANT_FORMATS {
            let d = q.desc();
            if !d.is_portable() {
                assert!(
                    !names.contains(&d.name),
                    "{}: generated native_intrinsic",
                    d.name
                );
            }
        }
    }

    /// A `NativeIntrinsic` format is refused by lowering with a distinct error:
    /// RIR describes it, and says so, instead of emitting a wrong expansion.
    #[test]
    fn a_native_intrinsic_format_stops_at_lowering() {
        let format = QUANT_FORMATS
            .into_iter()
            .find(|q| !q.desc().is_portable())
            .expect("no native_intrinsic format");
        let kernel = build(format).unwrap();
        let err = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap_err();
        assert_eq!(err, rir_lower::LowerError::NativeIntrinsicQuant { format });
    }

    // The **generated** Rust CPU path, including its fused quantized loader and
    // f16_to_f32 conversion, matches the reference.
    #[expect(dead_code)]
    mod generated_q8_0 {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/sum_rows_q8_0/cpu.rs"
        ));
    }

    #[expect(dead_code)]
    mod generated_q4_0 {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/sum_rows_q4_0/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row, stride) = (64usize, 3usize, 80usize);
        let mut seed = 0x51e_2026u64;
        let (raw, logical) = build_rows(QuantType::Q8_0, n_col, n_row, stride, &mut seed);
        let expected = reference_sums(&logical);

        let mut got = vec![0f32; n_row];
        generated_q8_0::sum_rows_q8_0(
            n_row,
            n_col,
            generated_q8_0::TensorRefBytes {
                data: &raw,
                nb: [34, stride, stride * n_row, stride * n_row],
            },
            generated_q8_0::TensorRefMut {
                data: &mut got,
                nb: [4, 4 * n_row, 4 * n_row, 4 * n_row],
            },
        );

        for (r, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4f32.max(1e-5 * e.abs()),
                "row {r}: {g} vs {e}"
            );
        }
    }

    /// Same check for the nibble format: the generated CPU code must extract
    /// the same halves as the reference, not merely a plausible pair.
    #[test]
    fn parity_of_the_generated_cpu_for_the_nibble_format() {
        let (n_col, n_row, stride) = (64usize, 3usize, 48usize);
        let mut seed = 0x4_0000_2026u64;
        let (raw, logical) = build_rows(QuantType::Q4_0, n_col, n_row, stride, &mut seed);
        let expected = reference_sums(&logical);

        let mut got = vec![0f32; n_row];
        generated_q4_0::sum_rows_q4_0(
            n_row,
            n_col,
            generated_q4_0::TensorRefBytes {
                data: &raw,
                nb: [18, stride, stride * n_row, stride * n_row],
            },
            generated_q4_0::TensorRefMut {
                data: &mut got,
                nb: [4, 4 * n_row, 4 * n_row, 4 * n_row],
            },
        );

        for (r, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4f32.max(1e-5 * e.abs()),
                "row {r}: {g} vs {e}"
            );
        }
    }
}
