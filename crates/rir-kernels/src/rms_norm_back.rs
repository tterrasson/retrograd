//! `rms_norm_back` - the second kernel identified by the real graph census:
//! 0.6% of traffic on Qwen3.5, 1.2% on gemma-3, and
//! above all the exact analogue of `l2_norm_back`, already promoted on both
//! backends. The same problem shape - two deterministic reductions over the
//! same row traversal, followed by an elementwise epilogue.
//!
//! ```text
//! sum_xx  = Σ x²        sum_xdz = Σ x·dz          N = extent(col)
//! rrms    = 1 / sqrt(sum_xx/N + eps)
//! scale_x = −sum_xdz / (sum_xx + eps·N)
//! dx      = (dz + x·scale_x) · rrms
//! ```
//!
//! This follows `ggml_compute_forward_rms_norm_back_f32` term for term,
//! including the form chosen there to avoid cancellation (ggml issue #1491):
//! `scale_x` divides by `sum_eps = sum_xx + eps·N` rather than composing
//! `mean_xdz / mean_eps`, and `rrms` remains built from `mean_eps`.
//!
//! **What this kernel required from the compiler.** A mean needs its divisor,
//! and nothing in the DSL could name an axis extent: the only available values
//! were a constant, a parameter, an index, and a read. `AxisExtent` fills this
//! gap with data that emitters already print as a loop bound (`pc.n_col`) - not
//! a new push constant, nor a parameter the caller must populate consistently
//! with the shape. The alternative - reducing `1.0` over the axis - would have
//! paid for another collective to recompute a number the shader knows.
//!
//! **The domain.** ggml requires `src0`, `src1`, and `dst` to have the same
//! shape; `supports_op` derives axis agreement from all three arguments being
//! indexed by the same axes. F32 only, like the native kernel.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ReduceOp, ReductionSemantics, ScalarType, TensorType,
    ValidateError, ValidatedKernel,
};

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("rms_norm_back");

    let dz = k.input("dz", TensorType::f32(4));
    let x = k.input("x", TensorType::f32(4));
    let dx = k.output("dx", TensorType::f32(4));
    let eps = k.param("eps", ScalarType::F32);

    // Same declaration order as `l2_norm_back`: `row` on x, `plane` on y,
    // `batch` on z; `col` remains the inner reduction axis. The three outer
    // axes make it possible to address a view inside a packed tensor.
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });

    let xv = k.read(x, &[col, row, plane, batch]);
    let zv = k.read(dz, &[col, row, plane, batch]);

    let xx = k.mul(xv, xv);
    let xz = k.mul(xv, zv);
    let sum_xx = k.reduce(ReduceOp::Sum, col, xx, ReductionSemantics::Deterministic);
    let sum_xdz = k.reduce(ReduceOp::Sum, col, xz, ReductionSemantics::Deterministic);

    let n = k.axis_extent(col);
    let mean_eps = {
        let mean = k.div(sum_xx, n);
        k.add(mean, eps)
    };
    let sum_eps = {
        let eps_n = k.mul(eps, n);
        k.add(sum_xx, eps_n)
    };

    let one = k.const_f32(1.0);
    let rms = k.sqrt(mean_eps);
    let rrms = k.div(one, rms);

    let zero = k.const_f32(0.0);
    let neg_xdz = k.sub(zero, sum_xdz);
    let scale_x = k.div(neg_xdz, sum_eps);

    let shifted = {
        let t = k.mul(xv, scale_x);
        k.add(zv, t)
    };
    let value = k.mul(shifted, rrms);

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

    /// Independent analytical reference, written from
    /// `ggml_compute_forward_rms_norm_back_f32`, not from the kernel.
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
            let (mut sum_xx, mut sum_xdz) = (0.0f64, 0.0f64);
            for c in 0..n_col {
                sum_xx += (x[base + c] * x[base + c]) as f64;
                sum_xdz += (x[base + c] * dz[base + c]) as f64;
            }
            let mean_eps = (sum_xx as f32) / n_col as f32 + eps;
            let sum_eps = (sum_xx as f32) + eps * n_col as f32;
            let rrms = 1.0f32 / mean_eps.sqrt();
            let scale_x = -(sum_xdz as f32) / sum_eps;
            for c in 0..n_col {
                dx[base + c] = (dz[base + c] + x[base + c] * scale_x) * rrms;
            }
        }
    }

    fn fill(seed: &mut u64, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
            *v = (u * 2.0 - 1.0) * scale;
        }
    }

    fn run_case(schedule: Schedule, n_col: usize, n_row: usize, stride: usize, zero_x: bool) {
        let kernel = build().unwrap();
        let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
        let eps = 1e-5f32;

        let len = stride * n_row;
        let mut seed = 0x7a5c_2026u64 ^ ((n_col as u64) << 32) ^ (n_row as u64);
        let mut dz = vec![0f32; len];
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut dz, 2.0);
        if !zero_x {
            fill(&mut seed, &mut x, 1.5);
        }

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
                    "{:?} {n_col}x{n_row} stride {stride} [{r},{c}] : {g} vs {e}",
                    schedule.backend()
                );
            }
            for c in n_col..stride {
                assert_eq!(got[base + c], 0.0, "write outside logical elements");
            }
        }
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for &(n_col, n_row) in &[(1, 1), (3, 5), (33, 7), (64, 4), (1, 9), (17, 1)] {
            run_case(Schedule::cpu_serial(), n_col, n_row, n_col, false);
        }
    }

    #[test]
    fn parity_with_non_contiguous_strides() {
        for &(n_col, n_row, stride) in &[(3, 5, 8), (33, 7, 40), (1, 4, 3)] {
            run_case(Schedule::cpu_serial(), n_col, n_row, stride, false);
        }
    }

    /// `x = 0` cancels `sum_xx`: `mean_eps` is exactly `eps`, `scale_x` is 0,
    /// so `dx = dz / sqrt(eps)`. This is the branch protected by ggml's epsilon,
    /// and it exists only because the kernel divides by `sum_xx + eps·N` rather
    /// than by `sum_xx`.
    #[test]
    fn the_epsilon_floor_when_the_row_is_zero() {
        run_case(Schedule::cpu_serial(), 9, 3, 9, true);
    }

    /// The same kernel under GPU lowering (32 lanes, `LaneReduce`), which is
    /// what the fork executes. The accumulation order differs from sequential,
    /// so the tolerance is relative.
    #[test]
    fn parity_of_the_lane_lowering_against_the_reference() {
        for &(n_col, n_row, stride) in &[(1, 1, 1), (7, 3, 7), (33, 5, 40), (256, 2, 300)] {
            run_case(Schedule::vulkan_subgroup(), n_col, n_row, stride, false);
        }
    }

    /// The shape class motivating the three outer axes: `x` occupies one plane
    /// out of three in a packed tensor, which no repetition can express.
    #[test]
    fn parity_on_a_rank_4_view_with_a_plane_gap() {
        let kernel = build().unwrap();
        for schedule in [Schedule::cpu_serial(), Schedule::vulkan_subgroup()] {
            let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
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

            let mut seed = 0x515a_2026u64;
            let mut dz = vec![0f32; len(1)];
            let mut x = vec![0f32; len(gap)];
            fill(&mut seed, &mut dz, 2.0);
            fill(&mut seed, &mut x, 1.5);
            let eps = 1e-5f32;

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

            for b in 0..n_batch {
                for p in 0..n_plane {
                    for r in 0..n_row {
                        let dst_base = ((b * n_plane + p) * n_row + r) * n_col;
                        let x_base = ((b * n_plane * gap + p * gap) * n_row + r) * n_col;
                        let (mut sum_xx, mut sum_xdz) = (0.0f32, 0.0f32);
                        for c in 0..n_col {
                            sum_xx += x[x_base + c] * x[x_base + c];
                            sum_xdz += x[x_base + c] * dz[dst_base + c];
                        }
                        let rrms = 1.0f32 / (sum_xx / n_col as f32 + eps).sqrt();
                        let scale_x = -sum_xdz / (sum_xx + eps * n_col as f32);
                        for c in 0..n_col {
                            let e = (dz[dst_base + c] + x[x_base + c] * scale_x) * rrms;
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

    // `too_many_arguments` is the only lint the emitted CPU kernels trip, and it
    // is not a style choice: the arity is the kernel's shape rank plus its
    // buffers, and the emitter cannot be bent to please a lint without moving
    // `generated/`, which must regenerate byte for byte. Named rather than blanketed
    // (`clippy::all`), so the next lint an emitter change trips is reported.
    #[allow(dead_code, clippy::too_many_arguments)]
    mod generated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/rms_norm_back/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row, stride) = (33usize, 7usize, 40usize);
        let len = stride * n_row;
        let mut seed = 0xfeed_2027u64;
        let mut dz = vec![0f32; len];
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut dz, 2.0);
        fill(&mut seed, &mut x, 1.5);
        let eps = 1e-5f32;

        let mut expected = vec![0f32; len];
        reference(&dz, &x, &mut expected, n_col, n_row, stride, eps);

        let mut got = vec![0f32; len];
        let nb = [4usize, 4 * stride, 4 * stride * n_row, 4 * stride * n_row];
        generated::rms_norm_back(
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
            for c in 0..n_col {
                let (g, e) = (got[base + c], expected[base + c]);
                let tol = 1e-4f32.max(2e-4 * e.abs());
                assert!((g - e).abs() <= tol, "generated CPU [{r},{c}]: {g} vs {e}");
            }
        }
    }
}
