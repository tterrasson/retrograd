//! `l2_norm_fwd` - L2 normalization forward pass and the stage-D autodiff
//! test case:
//!
//! ```text
//! y[c,r] = x[c,r] / sqrt(Σ_c x²)
//! ```
//!
//! Its backward pass, derived by transposition as `l2_norm_fwd_grad`, must
//! match both the handwritten `l2_norm_back` kernel outside the epsilon branch
//! and numerical gradients. Derived backward passes require validation before
//! use.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ReduceOp, ReductionSemantics, TensorType,
    ValidateError, ValidatedKernel,
};

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("l2_norm_fwd");

    let x = k.input("x", TensorType::f32_2d());
    let y = k.output("y", TensorType::f32_2d());

    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });

    let xv = k.read(x, &[col, row]);
    let xx = k.mul(xv, xv);
    let sum_xx = k.reduce(ReduceOp::Sum, col, xx, ReductionSemantics::Deterministic);
    let norm = k.sqrt(sum_xx);
    let yv = k.div(xv, norm);
    k.write(y, &[col, row], yv);

    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::Rank { arg: x, max: 2 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use rir_core::derive_backward;
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewMut, run};

    fn fill(seed: &mut u64, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
            // Keep values away from zero to avoid the handwritten kernel's
            // epsilon branch and condition finite differences well.
            *v = (u * 2.0 - 1.0) * scale + if *v >= 0.0 { 0.35 } else { -0.35 };
        }
    }

    /// F64 reference forward pass for numerical gradients.
    fn forward_f64(x: &[f64], n_col: usize) -> Vec<f64> {
        let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
        (0..n_col).map(|c| x[c] / norm).collect()
    }

    #[test]
    fn forward_parity_against_the_reference() {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap();

        let (n_col, n_row) = (13usize, 4usize);
        let mut seed = 0xf0d_2026u64;
        let mut x = vec![0f32; n_col * n_row];
        fill(&mut seed, &mut x, 1.0);

        let mut got = vec![0f32; n_col * n_row];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_col, n_row)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for r in 0..n_row {
            let row = &x[r * n_col..(r + 1) * n_col];
            let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
            for c in 0..n_col {
                let e = row[c] / norm;
                let g = got[r * n_col + c];
                assert!(
                    (g - e).abs() <= 1e-6f32.max(1e-6 * e.abs()),
                    "[{r},{c}] {g} vs {e}"
                );
            }
        }
    }

    /// The transposition-derived gradient matches handwritten `l2_norm_back`
    /// on well-conditioned inputs.
    #[test]
    fn the_derived_gradient_matches_the_handwritten_kernel() {
        let fwd = super::build().unwrap();
        let grad = derive_backward(&fwd).expect("autodiff l2_norm_fwd");
        assert_eq!(grad.name(), "l2_norm_fwd_grad");
        let lk_grad = rir_lower::lower(&grad, Schedule::cpu_serial()).unwrap();

        let manual = crate::l2_norm_back::build().unwrap();
        let lk_manual = rir_lower::lower(&manual, Schedule::cpu_serial()).unwrap();

        let (n_col, n_row) = (17usize, 3usize);
        let len = n_col * n_row;
        let mut seed = 0xad_2026u64;
        let mut x = vec![0f32; len];
        let mut dy = vec![0f32; len];
        fill(&mut seed, &mut x, 1.0);
        fill(&mut seed, &mut dy, 1.0);

        // Derived kernel arguments: (x, dy, dx).
        let mut dx_ad = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut dx_ad, n_col, n_row)),
        ];
        run(&lk_grad, &mut args, &[]).unwrap();

        // Handwritten kernel arguments: (dz, x, dx); small epsilon selects the
        // normal branch.
        let mut dx_manual = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, n_row)),
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut dx_manual, n_col, n_row)),
        ];
        run(&lk_manual, &mut args, &[1e-12]).unwrap();

        for i in 0..len {
            let (a, m) = (dx_ad[i], dx_manual[i]);
            assert!(
                (a - m).abs() <= 1e-5f32.max(1e-4 * m.abs()),
                "element {i}: {a} vs {m}"
            );
        }
    }

    /// The Vulkan-scheduled forward pass (32 lanes and `subgroupAdd`) matches
    /// the analytical reference within relative tolerance because its
    /// accumulation order differs from serial.
    #[test]
    fn forward_parity_of_the_lane_lowering() {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();

        for &(n_col, n_row) in &[(1usize, 1usize), (13, 4), (100, 3)] {
            let mut seed = 0x1a2_2026u64 ^ ((n_col as u64) << 16) ^ n_row as u64;
            let mut x = vec![0f32; n_col * n_row];
            fill(&mut seed, &mut x, 1.0);

            let mut got = vec![0f32; n_col * n_row];
            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
                BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_col, n_row)),
            ];
            run(&lk, &mut args, &[]).unwrap();

            for r in 0..n_row {
                let row = &x[r * n_col..(r + 1) * n_col];
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                for c in 0..n_col {
                    let e = row[c] / norm;
                    let g = got[r * n_col + c];
                    assert!(
                        (g - e).abs() <= 1e-5f32.max(1e-5 * e.abs()),
                        "[{r},{c}] {g} vs {e}"
                    );
                }
            }
        }
    }

    /// Under the Vulkan schedule, the derived gradient's **two reduction
    /// levels** use successive lane collectives; the second level consumes the
    /// first. Compare against the handwritten kernel rather than the same
    /// kernel's serial path to keep the oracle independent.
    #[test]
    fn the_derived_gradient_under_lanes_matches_the_handwritten_kernel() {
        let fwd = super::build().unwrap();
        let grad = derive_backward(&fwd).expect("autodiff l2_norm_fwd");
        let lk_grad = rir_lower::lower(&grad, Schedule::vulkan_subgroup()).unwrap();

        let manual = crate::l2_norm_back::build().unwrap();
        let lk_manual = rir_lower::lower(&manual, Schedule::cpu_serial()).unwrap();

        let (n_col, n_row) = (37usize, 3usize);
        let len = n_col * n_row;
        let mut seed = 0x2ad_2026u64;
        let mut x = vec![0f32; len];
        let mut dy = vec![0f32; len];
        fill(&mut seed, &mut x, 1.0);
        fill(&mut seed, &mut dy, 1.0);

        let mut dx_ad = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut dx_ad, n_col, n_row)),
        ];
        run(&lk_grad, &mut args, &[]).unwrap();

        let mut dx_manual = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, n_row)),
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut dx_manual, n_col, n_row)),
        ];
        run(&lk_manual, &mut args, &[1e-12]).unwrap();

        for i in 0..len {
            let (a, m) = (dx_ad[i], dx_manual[i]);
            assert!(
                (a - m).abs() <= 1e-5f32.max(1e-4 * m.abs()),
                "element {i}: {a} vs {m}"
            );
        }
    }

    /// The derived gradient matches centered finite differences computed from
    /// the F64 reference forward pass.
    #[test]
    fn the_derived_gradient_matches_the_numeric_gradient() {
        let fwd = super::build().unwrap();
        let grad = derive_backward(&fwd).unwrap();
        let lk_grad = rir_lower::lower(&grad, Schedule::cpu_serial()).unwrap();

        let n_col = 9usize;
        let mut seed = 0x96ad_2026u64;
        let mut x = vec![0f32; n_col];
        let mut dy = vec![0f32; n_col];
        fill(&mut seed, &mut x, 1.0);
        fill(&mut seed, &mut dy, 1.0);

        let mut dx = vec![0f32; n_col];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, 1)),
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, 1)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut dx, n_col, 1)),
        ];
        run(&lk_grad, &mut args, &[]).unwrap();

        let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let h = 1e-5f64;
        for j in 0..n_col {
            let mut xp = x64.clone();
            let mut xm = x64.clone();
            xp[j] += h;
            xm[j] -= h;
            let yp = forward_f64(&xp, n_col);
            let ym = forward_f64(&xm, n_col);
            let numeric: f64 = (0..n_col)
                .map(|i| dy[i] as f64 * (yp[i] - ym[i]) / (2.0 * h))
                .sum();
            let a = dx[j] as f64;
            assert!(
                (a - numeric).abs() <= 1e-3f64.max(1e-3 * numeric.abs()),
                "d x[{j}]: autodiff {a} vs numerical {numeric}"
            );
        }
    }
}
