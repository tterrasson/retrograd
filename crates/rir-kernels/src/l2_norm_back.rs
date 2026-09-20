//! `l2_norm_back` - the stage-A reference kernel.
//!
//! For each row:
//!
//! ```text
//! norm = sqrt(Σ x²)
//! dx   = norm > eps ? (dz − x · (Σ x·dz / Σ x²)) / norm: dz / eps
//! ```
//!
//! Two deterministic per-row reductions share the same traversal of `x`,
//! followed by an elementwise epilogue with a safety branch. This backward
//! normalization pattern represents the long tail targeted by RIR.
//!
//! The kernel carries **three outer axes** (`row`, `plane`, `batch`) plus the
//! reduction axis `col`, i.e. the full ggml rank. A rank-2 tensor is the same
//! problem with `n_plane = n_batch = 1`; what the outer axes buy is a distinct
//! `nb[2]`/`nb[3]` per argument, the only way to address a view inside a
//! packed tensor - measured as the sole blocker on the real Qwen3.5 graph.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ReduceOp, ReductionSemantics, ScalarType, TensorType,
    ValidateError, ValidatedKernel,
};

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("l2_norm_back");

    let dz = k.input("dz", TensorType::f32(4));
    let x = k.input("x", TensorType::f32(4));
    let dx = k.output("dx", TensorType::f32(4));
    let eps = k.param("eps", ScalarType::F32);

    // Logical axes only; threads do not exist at this level. Declaration order
    // is what lowering maps to the grid: `row` to x, `plane` to y, `batch` to
    // z, leaving `col` as the inner reduction axis.
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });

    let xv = k.read(x, &[col, row, plane, batch]);
    let zv = k.read(dz, &[col, row, plane, batch]);

    let xx = k.mul(xv, xv);
    let xz = k.mul(xv, zv);
    let sum_xx = k.reduce(ReduceOp::Sum, col, xx, ReductionSemantics::Deterministic);
    let sum_xz = k.reduce(ReduceOp::Sum, col, xz, ReductionSemantics::Deterministic);

    let norm = k.sqrt(sum_xx);
    let ratio = k.div(sum_xz, sum_xx);
    let scaled = k.mul(xv, ratio);
    let num = k.sub(zv, scaled);
    let normal = k.div(num, norm);
    let safe = k.div(zv, eps);
    let cond = k.gt(norm, eps);
    let value = k.select(cond, normal, safe);

    k.write(dx, &[col, row, plane, batch], value);

    k.constrain(Constraint::DType {
        arg: dz,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::Rank { arg: x, max: 4 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewMut, run};

    /// Independent analytical reference using the same formulas in direct code.
    fn reference(
        dz: &[f32],
        x: &[f32],
        dx: &mut [f32],
        n_col: usize,
        n_row: usize,
        stride_elems: usize,
        eps: f32,
    ) {
        for r in 0..n_row {
            let base = r * stride_elems;
            let mut sum_xx = 0.0f32;
            let mut sum_xz = 0.0f32;
            for c in 0..n_col {
                let xv = x[base + c];
                let zv = dz[base + c];
                sum_xx += xv * xv;
                sum_xz += xv * zv;
            }
            let norm = sum_xx.sqrt();
            for c in 0..n_col {
                let xv = x[base + c];
                let zv = dz[base + c];
                dx[base + c] = if norm > eps {
                    (zv - xv * (sum_xz / sum_xx)) / norm
                } else {
                    zv / eps
                };
            }
        }
    }

    /// Deterministic LCG generator with no rand dependency.
    fn fill(seed: &mut u64, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
            *v = (u * 2.0 - 1.0) * scale;
        }
    }

    fn assert_close(a: &[f32], b: &[f32], ctx: &str) {
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            let tol = 1e-5f32.max(1e-5 * x.abs());
            assert!((x - y).abs() <= tol, "{ctx}: element {i}: {x} vs {y}");
        }
    }

    fn run_case(n_col: usize, n_row: usize, stride_elems: usize, zero_x: bool, eps: f32) {
        let kernel = build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap();

        let len = stride_elems * n_row;
        let mut seed = 0x0dd0_2026u64 ^ ((n_col as u64) << 32) ^ (n_row as u64);
        let mut dz = vec![0f32; len];
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut dz, 2.0);
        if !zero_x {
            fill(&mut seed, &mut x, 1.5);
        }

        let mut expected = vec![0f32; len];
        reference(&dz, &x, &mut expected, n_col, n_row, stride_elems, eps);

        let mut got = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::strided_2d(&dz, n_col, n_row, stride_elems)),
            BoundArg::In(TensorView::strided_2d(&x, n_col, n_row, stride_elems)),
            BoundArg::Out(TensorViewMut::strided_2d(
                &mut got,
                n_col,
                n_row,
                stride_elems,
            )),
        ];
        run(&lk, &mut args, &[eps]).unwrap();

        // Compare logical elements only; padding remains zero.
        for r in 0..n_row {
            let base = r * stride_elems;
            assert_close(
                &got[base..base + n_col],
                &expected[base..base + n_col],
                &format!("shape {n_col}x{n_row} stride {stride_elems} zero_x={zero_x}"),
            );
            for c in n_col..stride_elems {
                assert_eq!(got[base + c], 0.0, "write outside logical elements");
            }
        }
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        // Unaligned, non-power-of-two shapes, including unit dimensions.
        for &(n_col, n_row) in &[(1, 1), (3, 5), (33, 7), (64, 4), (1, 9), (17, 1)] {
            run_case(n_col, n_row, n_col, false, 1e-6);
        }
    }

    #[test]
    fn parity_with_non_contiguous_strides() {
        for &(n_col, n_row, stride) in &[(3, 5, 8), (33, 7, 40), (1, 4, 3)] {
            run_case(n_col, n_row, stride, false, 1e-6);
        }
    }

    #[test]
    fn the_safety_branch_when_the_norm_is_below_eps() {
        // x = 0 implies norm <= eps, so dx = dz / eps across the row.
        run_case(9, 3, 9, true, 1e-3);
    }

    /// The shape class that motivated the three outer axes: `x` is a view
    /// inside a packed QKV tensor, so its planes are `gap` times further apart
    /// than the rows they contain. No fold can express that; only a distinct
    /// `nb[2]` per argument can.
    #[test]
    fn parity_on_a_rank_4_view_with_a_plane_gap() {
        let kernel = build().unwrap();
        for schedule in [Schedule::cpu_serial(), Schedule::vulkan_subgroup()] {
            let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
            // Qwen3.5's own geometry, shrunk: [n_col, n_row, n_plane, n_batch]
            // with `x` living one plane out of three.
            let (n_col, n_row, n_plane, n_batch, gap) = (33usize, 4usize, 3usize, 2usize, 3usize);

            let packed = |g: usize| {
                [
                    4,
                    4 * n_col,
                    4 * n_col * n_row * g,
                    4 * n_col * n_row * g * n_plane,
                ]
            };
            let len = |g: usize| n_col * n_row * n_plane * n_batch * g;

            let mut seed = 0x4d4d_2026u64;
            let mut dz = vec![0f32; len(1)];
            let mut x = vec![0f32; len(gap)];
            fill(&mut seed, &mut dz, 2.0);
            fill(&mut seed, &mut x, 1.5);
            let eps = 1e-6f32;

            let mut got = vec![0f32; len(1)];
            let shape = [n_col, n_row, n_plane, n_batch];
            let mut args = [
                BoundArg::In(TensorView {
                    data: &dz,
                    shape,
                    nb: packed(1),
                }),
                BoundArg::In(TensorView {
                    data: &x,
                    shape,
                    nb: packed(gap),
                }),
                BoundArg::Out(TensorViewMut {
                    data: &mut got,
                    shape,
                    nb: packed(1),
                }),
            ];
            run(&lk, &mut args, &[eps]).unwrap();

            // Independent reference walking the same two layouts by hand.
            for b in 0..n_batch {
                for p in 0..n_plane {
                    for r in 0..n_row {
                        let dst_base = ((b * n_plane + p) * n_row + r) * n_col;
                        let x_base = ((b * n_plane * gap + p * gap) * n_row + r) * n_col;
                        let (mut sum_xx, mut sum_xz) = (0.0f32, 0.0f32);
                        for c in 0..n_col {
                            sum_xx += x[x_base + c] * x[x_base + c];
                            sum_xz += x[x_base + c] * dz[dst_base + c];
                        }
                        let norm = sum_xx.sqrt();
                        for c in 0..n_col {
                            let (xv, zv) = (x[x_base + c], dz[dst_base + c]);
                            let e = if norm > eps {
                                (zv - xv * (sum_xz / sum_xx)) / norm
                            } else {
                                zv / eps
                            };
                            let g = got[dst_base + c];
                            let tol = 1e-4f32.max(2e-4 * e.abs());
                            assert!(
                                (g - e).abs() <= tol,
                                "{:?} [{b},{p},{r},{c}] : {g} vs {e}",
                                schedule.backend()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The interpreter's GPU-lowered path (Vulkan schedule, 32 lanes, and
    /// `LaneReduce`) matches the reference within relative tolerance. Its
    /// accumulation order differs from serial, so bitwise equality is not
    /// expected. This covers stage-B Loop IR parity; device parity uses the
    /// generated shader on a real GPU.
    #[test]
    fn parity_of_the_lane_lowering_against_the_reference() {
        let kernel = build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();

        for &(n_col, n_row, stride) in &[
            (1usize, 1usize, 1usize),
            (7, 3, 7),
            (33, 5, 40),
            (100, 4, 100),
            (256, 2, 300),
        ] {
            let len = stride * n_row;
            let mut seed = 0xb00c_2026u64 ^ ((n_col as u64) << 32) ^ (n_row as u64);
            let mut dz = vec![0f32; len];
            let mut x = vec![0f32; len];
            fill(&mut seed, &mut dz, 2.0);
            fill(&mut seed, &mut x, 1.5);
            let eps = 1e-6f32;

            let mut expected = vec![0f32; len];
            reference(&dz, &x, &mut expected, n_col, n_row, stride, eps);

            let mut got = vec![0f32; len];
            let mut args = [
                BoundArg::In(TensorView::strided_2d(&dz, n_col, n_row, stride)),
                BoundArg::In(TensorView::strided_2d(&x, n_col, n_row, stride)),
                BoundArg::Out(TensorViewMut::strided_2d(&mut got, n_col, n_row, stride)),
            ];
            run(&lk, &mut args, &[eps]).unwrap();

            for r in 0..n_row {
                let base = r * stride;
                for c in 0..n_col {
                    let (g, e) = (got[base + c], expected[base + c]);
                    let tol = 1e-4f32.max(2e-4 * e.abs());
                    assert!(
                        (g - e).abs() <= tol,
                        "lanes {n_col}x{n_row} stride {stride} [{r},{c}] : {g} vs {e}"
                    );
                }
            }
        }
    }

    // Include the committed, **generated** CPU code directly to verify that it
    // compiles and matches the reference. This covers emitted-code parity for
    // stage A; the interpreter tests above cover Loop IR parity.
    // `too_many_arguments` is the only lint the emitted CPU kernels trip, and it
    // is not a style choice: the arity is the kernel's shape rank plus its
    // buffers, and the emitter cannot be bent to please a lint without moving
    // `generated/`, which must regenerate byte for byte. Named rather than blanketed
    // (`clippy::all`), so the next lint an emitter change trips is reported.
    #[expect(clippy::too_many_arguments)]
    mod generated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/l2_norm_back/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row, stride) = (33usize, 7usize, 40usize);
        let len = stride * n_row;
        let mut seed = 0xfeed_2026u64;
        let mut dz = vec![0f32; len];
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut dz, 2.0);
        fill(&mut seed, &mut x, 1.5);
        let eps = 1e-6f32;

        let mut expected = vec![0f32; len];
        reference(&dz, &x, &mut expected, n_col, n_row, stride, eps);

        let mut got = vec![0f32; len];
        let nb = [4usize, 4 * stride, 4 * stride * n_row, 4 * stride * n_row];
        generated::l2_norm_back(
            n_row,
            1,
            1,
            n_col,
            generated::TensorRef { data: &dz, nb },
            generated::TensorRef { data: &x, nb },
            generated::TensorRefMut { data: &mut got, nb },
            eps,
        );

        for r in 0..n_row {
            let base = r * stride;
            assert_close(
                &got[base..base + n_col],
                &expected[base..base + n_col],
                "generated CPU",
            );
        }
    }
}
