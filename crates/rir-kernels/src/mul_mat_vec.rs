//! `mul_mat_vec_<format>` - the decode-time product of a quantized weight
//! matrix with a handful of activation columns.
//!
//! ```text
//! dst[row, t] = Σ_col dequant(a[col, row]) · b[col, t]      (ggml: C = Aᵀ·B)
//! ```
//!
//! This is `GGML_OP_MUL_MAT` with a quantized `src0` and an F32 `src1`, in
//! the regime where `src1` has few columns (one per token of a decode step):
//! every weight is read once per step, so the cost is the weight traffic and
//! the decode that rides on it, not arithmetic. The fused quantized loader is
//! the one `sum_rows_<format>` exercises; what this kernel adds is the second
//! operand inside the contraction.
//!
//! **Axis names.** The axes take ggml's own vocabulary, `col` for the
//! contraction and `row` for the weight row, and the token index is `plane`:
//! a shape is `[col, row, plane, 1]` = `[K, N, tokens, 1]`, so a shape rule over
//! `row · plane` counts output elements - the workgroups a row-per-subgroup
//! lowering launches - which is the occupancy the reduction families already
//! arbitrate on.
//!
//! **Formats.** One kernel per weight format a published model profile stores
//! its matrices in (`rir_core::profiles`), filtered by what lowering can decode.
//! A format no profile carries gets no kernel: a GEMV is generated for a model
//! that will run it, not for every row of the format table.
//!
//! **Observed, decode only.** The fork offers `MUL_MAT` to the registry for
//! steps of at most `RIR_DECODE_MAX_TOKENS` columns; the pair encodes nothing
//! until the promotion lane has measured it against the native `mul_mv`
//! kernels on those shapes.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, Layout, QUANT_FORMATS, QuantType, ReduceOp,
    ReductionSemantics, TensorType, ValidateError, ValidatedKernel,
};

/// Kernel name for a weight format: `mul_mat_vec_q8_0`.
pub fn kernel_name(format: QuantType) -> String {
    format!("mul_mat_vec_{}", format.desc().name)
}

/// The weight formats of the reference profiles that lowering can decode, in
/// canonical table order.
pub fn variants() -> Vec<QuantType> {
    let profiled: Vec<String> = rir_core::profiles::reference()
        .into_iter()
        .flat_map(|p| p.shape.weight_formats)
        .collect();
    QUANT_FORMATS
        .into_iter()
        .filter(|q| {
            let d = q.desc();
            profiled.iter().any(|f| f == d.name)
                && d.block_elements > 1
                && rir_lower::can_lower_dequant(*q)
        })
        .collect()
}

pub fn build(format: QuantType) -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new(&kernel_name(format));

    let a = k.input(
        "a",
        TensorType {
            dtype: DType::Quant(format),
            rank: 2,
            layout: Layout::Ggml,
        },
    );
    let b = k.input("b", TensorType::f32_2d());
    let dst = k.output("dst", TensorType::f32_2d());

    // Declaration order is grid order: `row` on x, `plane` (the token) on y,
    // and `col` remains the inner contraction.
    let row = k.axis("row", Extent::Dim { arg: a, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: b, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: a, dim: 0 });

    let av = k.read(a, &[col, row]);
    let bv = k.read(b, &[col, plane]);
    let p = k.mul(av, bv);
    let s = k.reduce(ReduceOp::Sum, col, p, ReductionSemantics::Deterministic);
    k.write(dst, &[row, plane], s);

    k.constrain(Constraint::DType {
        arg: a,
        allowed: vec![DType::Quant(format)],
    });
    k.constrain(Constraint::DType {
        arg: b,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::Rank { arg: a, max: 2 });
    k.constrain(Constraint::Rank { arg: b, max: 2 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewBytes, TensorViewMut, run};

    fn fill(seed: &mut u64, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
                u * 2.0 - 1.0
            })
            .collect()
    }

    /// Runs one case under `schedule` against the definition of the format:
    /// rows decoded by the portable oracle, then a plain dot product.
    fn run_case(format: QuantType, schedule: Schedule, blocks: usize, n_row: usize, n_tok: usize) {
        let d = format.desc();
        let (be, bb) = (d.block_elements as usize, d.block_bytes as usize);
        let n_col = blocks * be;
        let row_bytes = blocks * bb;

        let mut seed = 0x6e3_2026u64 ^ ((n_col as u64) << 20) ^ ((n_row as u64) << 8);
        let mut raw = Vec::with_capacity(row_bytes * n_row);
        let mut weights = Vec::with_capacity(n_row);
        for _ in 0..n_row {
            let bytes = rir_core::random_block_bytes(format, blocks, &mut seed).unwrap();
            weights.push(rir_core::dequantize_row(format, &bytes, n_col).unwrap());
            raw.extend_from_slice(&bytes);
        }
        let x = fill(&mut seed, n_col * n_tok);

        let mut expected = vec![0f32; n_row * n_tok];
        for t in 0..n_tok {
            for (r, w) in weights.iter().enumerate() {
                expected[t * n_row + r] = w
                    .iter()
                    .zip(&x[t * n_col..(t + 1) * n_col])
                    .fold(0f32, |acc, (w, x)| acc + w * x);
            }
        }

        let kernel = build(format).unwrap();
        let lk = rir_lower::lower(&kernel, schedule).unwrap();
        let mut got = vec![0f32; n_row * n_tok];
        let mut args = [
            BoundArg::InBytes(TensorViewBytes {
                data: &raw,
                shape: [n_col, n_row, 1, 1],
                nb: [bb, row_bytes, row_bytes * n_row, row_bytes * n_row],
            }),
            BoundArg::In(TensorView::strided_2d(&x, n_col, n_tok, n_col)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_row, n_tok)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            let tol = 1e-3f32.max(1e-4 * e.abs());
            assert!(
                (g - e).abs() <= tol,
                "{} {n_col}x{n_row}x{n_tok} element {i}: {g} vs {e}",
                d.name
            );
        }
    }

    #[test]
    fn the_formats_are_the_reference_profiles_weight_formats() {
        let names: Vec<&str> = variants().iter().map(|q| q.desc().name).collect();
        assert_eq!(names, ["q8_0", "q4_K", "q6_K"]);
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for format in variants() {
            for &(blocks, rows, toks) in &[(1usize, 1usize, 1usize), (2, 5, 3), (3, 4, 16)] {
                run_case(format, Schedule::cpu_serial(), blocks, rows, toks);
            }
        }
    }

    /// Under the subgroup schedule the 32 lanes of one workgroup split the
    /// row's blocks, each multiplying its decoded share by the matching slice
    /// of `b`, and `subgroupAdd` closes the dot product.
    #[test]
    fn parity_of_the_lane_lowering_against_the_reference() {
        for format in variants() {
            for &(blocks, rows, toks) in &[(1usize, 1usize, 1usize), (2, 5, 3), (33, 3, 4)] {
                run_case(format, Schedule::vulkan_subgroup(), blocks, rows, toks);
            }
        }
    }

    /// The lane-tiled lowering: one workgroup owns several tokens and every
    /// lane decodes its share of the weight row once for all of them. Token
    /// counts that are not a whole number of tiles leave the last one partial,
    /// and its guarded indices must neither read nor write past the tensor.
    #[test]
    fn parity_of_the_lane_tiled_lowering_against_the_reference() {
        use rir_lower::GpuBackend;
        for format in variants() {
            for cols in [2u32, 4, 8] {
                for &(blocks, rows, toks) in &[(1usize, 1usize, 1usize), (2, 5, 3), (3, 4, 9)] {
                    let schedule = Schedule::gpu_subgroup_cols(GpuBackend::Vulkan, cols);
                    run_case(format, schedule, blocks, rows, toks);
                }
            }
        }
    }

    /// The tile is what the dispatch divides by, so the manifest's grid has to
    /// say so - and the decoded weight is read once per segment whatever the
    /// tile, which is the point of the lowering.
    #[test]
    fn the_lane_tile_divides_the_token_grid() {
        use rir_lower::GpuBackend;
        let kernel = build(QuantType::Q8_0).unwrap();
        let lk =
            rir_lower::lower(&kernel, Schedule::gpu_subgroup_cols(GpuBackend::Metal, 4)).unwrap();
        assert_eq!(lk.schedule.lane_tile(), 4);
        let rir_lower::Stmt::Parallel { tile, body, .. } = &lk.body[0] else {
            panic!("expected the row grid");
        };
        assert_eq!(*tile, 1);
        let rir_lower::Stmt::Parallel { tile, .. } = &body[0] else {
            panic!("expected the token grid");
        };
        assert_eq!(*tile, 4);
    }
}
