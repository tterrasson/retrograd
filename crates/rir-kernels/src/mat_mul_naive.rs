//! `mat_mul_naive` - the deliberately limited stage-F slice: a contraction in
//! the shared IR with a **naive** three-loop schedule.
//!
//! ```text
//! c[i,j] = Σ_k a[k,i] · b[k,j]        (ggml convention: C = Aᵀ·B)
//! ```
//!
//! Stage F shares the mathematics across backends through the IR while
//! allowing schedules to differ. By design, tiling,
//! double buffering, and matrix units are **outside** this kernel's scope.
//! Production MMA kernels remain handwritten; this naive variant is an oracle
//! and an observable fallback.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ReduceOp, ReductionSemantics, TensorType,
    ValidateError, ValidatedKernel,
};

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("mat_mul_naive");

    let a = k.input("a", TensorType::f32_2d());
    let b = k.input("b", TensorType::f32_2d());
    let c = k.output("c", TensorType::f32_2d());

    let i = k.axis("i", Extent::Dim { arg: a, dim: 1 });
    let j = k.axis("j", Extent::Dim { arg: b, dim: 1 });
    let kk = k.axis("k", Extent::Dim { arg: a, dim: 0 });

    let av = k.read(a, &[kk, i]);
    let bv = k.read(b, &[kk, j]);
    let p = k.mul(av, bv);
    let s = k.reduce(ReduceOp::Sum, kk, p, ReductionSemantics::Deterministic);
    k.write(c, &[i, j], s);

    k.constrain(Constraint::DType {
        arg: a,
        allowed: vec![DType::F32],
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
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewMut, run};

    fn fill(seed: &mut u64, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
            *v = (u * 2.0 - 1.0) * scale;
        }
    }

    /// Reference C = Aᵀ·B with the same ascending-k accumulation order.
    fn reference(
        a: &[f32],
        b: &[f32],
        n_k: usize,
        n_i: usize,
        n_j: usize,
        sa: usize,
        sb: usize,
    ) -> Vec<f32> {
        let mut c = vec![0f32; n_i * n_j];
        for i in 0..n_i {
            for j in 0..n_j {
                let mut acc = 0f32;
                for k in 0..n_k {
                    acc += a[i * sa + k] * b[j * sb + k];
                }
                c[j * n_i + i] = acc;
            }
        }
        c
    }

    fn run_case(n_k: usize, n_i: usize, n_j: usize, sa: usize, sb: usize) {
        run_case_with(Schedule::cpu_serial(), n_k, n_i, n_j, sa, sb);
    }

    fn run_case_with(
        schedule: Schedule,
        n_k: usize,
        n_i: usize,
        n_j: usize,
        sa: usize,
        sb: usize,
    ) -> Vec<f32> {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, schedule).unwrap();

        let mut seed = 0x3a3_2026u64 ^ ((n_k as u64) << 20) ^ ((n_i as u64) << 10) ^ n_j as u64;
        let mut a = vec![0f32; sa * n_i];
        let mut b = vec![0f32; sb * n_j];
        fill(&mut seed, &mut a, 1.0);
        fill(&mut seed, &mut b, 1.0);
        let expected = reference(&a, &b, n_k, n_i, n_j, sa, sb);

        let mut got = vec![0f32; n_i * n_j];
        let mut args = [
            BoundArg::In(TensorView::strided_2d(&a, n_k, n_i, sa)),
            BoundArg::In(TensorView::strided_2d(&b, n_k, n_j, sb)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_i, n_j)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for idx in 0..n_i * n_j {
            let (g, e) = (got[idx], expected[idx]);
            assert!(
                (g - e).abs() <= 1e-4f32.max(1e-5 * e.abs()),
                "{n_k}x{n_i}x{n_j} element {idx}: {g} vs {e}"
            );
        }
        got
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        run_case(1, 1, 1, 1, 1);
        run_case(8, 5, 3, 8, 8);
        run_case(33, 7, 4, 33, 33);
    }

    /// The Vulkan schedule maps **both** parallel axes (i, j) onto the grid and
    /// keeps the contraction sequential within each invocation. Accumulation
    /// order is unchanged, so the result must be **bit-identical** to serial,
    /// not merely close. This distinguishes remapping from reassociation.
    #[test]
    fn the_grid_mapping_does_not_change_the_result() {
        for &(n_k, n_i, n_j, sa, sb) in &[
            (8usize, 5usize, 3usize, 12usize, 9usize),
            (33, 7, 4, 33, 33),
            (1, 1, 1, 1, 1),
        ] {
            let serial = run_case_with(Schedule::cpu_serial(), n_k, n_i, n_j, sa, sb);
            let grid = run_case_with(Schedule::vulkan_grid([16, 16, 1]), n_k, n_i, n_j, sa, sb);
            assert_eq!(
                serial, grid,
                "{n_k}x{n_i}x{n_j}: the grid changed the result"
            );
        }
    }

    #[test]
    fn parity_with_non_contiguous_strides() {
        run_case(8, 5, 3, 12, 9);
        run_case(17, 2, 6, 20, 17);
    }

    // The generated CPU i/j/k loop nest compiles and is correct.
    mod generated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/mat_mul_naive/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_k, n_i, n_j) = (16usize, 4usize, 5usize);
        let mut seed = 0x99_2026u64;
        let mut a = vec![0f32; n_k * n_i];
        let mut b = vec![0f32; n_k * n_j];
        fill(&mut seed, &mut a, 1.0);
        fill(&mut seed, &mut b, 1.0);
        let expected = reference(&a, &b, n_k, n_i, n_j, n_k, n_k);

        let mut got = vec![0f32; n_i * n_j];
        generated::mat_mul_naive(
            n_i,
            n_j,
            n_k,
            generated::TensorRef {
                data: &a,
                nb: [4, 4 * n_k, 4 * n_k * n_i, 4 * n_k * n_i],
            },
            generated::TensorRef {
                data: &b,
                nb: [4, 4 * n_k, 4 * n_k * n_j, 4 * n_k * n_j],
            },
            generated::TensorRefMut {
                data: &mut got,
                nb: [4, 4 * n_i, 4 * n_i * n_j, 4 * n_i * n_j],
            },
        );

        for idx in 0..n_i * n_j {
            assert!((got[idx] - expected[idx]).abs() <= 1e-4, "element {idx}");
        }
    }
}
