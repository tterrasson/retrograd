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
        run_case_strided(format, schedule, blocks, n_row, n_tok, 1);
    }

    /// The same, with consecutive columns of `b` `step` elements apart: a
    /// step of one is the contiguous layout, anything else the one a run may
    /// not read as consecutive elements.
    fn run_case_strided(
        format: QuantType,
        schedule: Schedule,
        blocks: usize,
        n_row: usize,
        n_tok: usize,
        step: usize,
    ) {
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
        let mut spread = vec![0f32; n_col * n_tok * step];
        for (i, v) in x.iter().enumerate() {
            spread[i * step] = *v;
        }

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
            BoundArg::In(TensorView {
                data: &spread,
                shape: [n_col, n_tok, 1, 1],
                nb: [
                    4 * step,
                    4 * step * n_col,
                    4 * step * n_col * n_tok,
                    4 * step * n_col * n_tok,
                ],
            }),
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

    /// Runs: a segment reads its four activations in one access. The sum each
    /// accumulator receives is the element loop's, term for term, with or
    /// without a lane tile, and partial tiles stay guarded. A `b` whose columns
    /// are not consecutive takes the element half of the guard.
    #[test]
    fn parity_of_the_run_lowering_against_the_reference() {
        use rir_lower::GpuBackend;
        for format in variants() {
            for cols in [1u32, 4] {
                for &(blocks, rows, toks) in &[(1usize, 1usize, 1usize), (2, 5, 3), (33, 3, 9)] {
                    let schedule = if cols == 1 {
                        Schedule::gpu_subgroup(GpuBackend::Metal)
                    } else {
                        Schedule::gpu_subgroup_cols(GpuBackend::Metal, cols)
                    };
                    for step in [1, 2] {
                        let schedule = schedule.clone().with_reduce_run(4);
                        run_case_strided(format, schedule, blocks, rows, toks, step);
                    }
                }
            }
        }
    }

    /// A run divides the segment of the quantized read; one it does not divide
    /// is refused, not read element by element under a name that says runs.
    #[test]
    fn a_run_that_does_not_divide_the_segment_is_refused() {
        use rir_lower::{GpuBackend, LowerError};
        let kernel = build(QuantType::Q8_0).unwrap();
        let err = rir_lower::lower(
            &kernel,
            Schedule::gpu_subgroup(GpuBackend::Metal).with_reduce_run(3),
        )
        .unwrap_err();
        assert!(
            matches!(err, LowerError::ReduceRunUnsupported { .. }),
            "{err}"
        );
    }

    /// The shipped GPU schedules read their activations in runs where `b` is
    /// contiguous: one `UnitStride`, whose unit half reads `b` in one
    /// `LoadRun` of four per segment and per tiled index and never element by
    /// element, and whose strided half is the element loop. The CPU keeps the
    /// serial element loop, which is the oracle's own order.
    #[test]
    fn the_shipped_schedules_read_the_activations_in_runs() {
        use rir_lower::{Stmt, child_blocks_of};
        /// `(runs of b, element reads of b)` under `stmts`.
        fn count(stmts: &[Stmt]) -> (usize, usize) {
            let mut n = (0, 0);
            for s in stmts {
                match s {
                    Stmt::LoadRun { dsts, arg, .. } => {
                        assert_eq!((dsts.len(), arg.0), (4, 1));
                        n.0 += 1;
                    }
                    Stmt::Load { arg, .. } if arg.0 == 1 => n.1 += 1,
                    _ => {
                        for b in child_blocks_of(s) {
                            let (r, e) = count(b);
                            n = (n.0 + r, n.1 + e);
                        }
                    }
                }
            }
            n
        }
        fn guards(stmts: &[Stmt], out: &mut Vec<(usize, usize, usize, usize)>) {
            for s in stmts {
                if let Stmt::UnitStride {
                    args,
                    unit,
                    strided,
                } = s
                {
                    assert_eq!(args.len(), 1);
                    let (u, s) = (count(unit), count(strided));
                    out.push((u.0, u.1, s.0, s.1));
                } else {
                    for b in child_blocks_of(s) {
                        guards(b, out);
                    }
                }
            }
        }
        let kernel = build(QuantType::Q4_K).unwrap();
        for schedule in rir_lower::schedules_for(rir_lower::Family::MulMatVec) {
            if schedule.backend() == rir_lower::Backend::Cpu {
                assert_eq!(schedule.reduce_run(), 1);
                continue;
            }
            let tile = schedule.lane_tile() as usize;
            assert_eq!(schedule.reduce_run(), 4);
            let lk = rir_lower::lower(&kernel, schedule).unwrap();
            let mut found = Vec::new();
            guards(&lk.body, &mut found);
            assert_eq!(found, [(tile, 0, 0, tile)]);
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
