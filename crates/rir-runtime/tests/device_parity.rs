//! Device parity: the **generated** shader executed on a real GPU against the
//! oracle (the Loop IR interpreter).
//!
//! This is the link covered by neither generation nor kernel tests: they
//! validate "IR ↔ emitted code" and "GLSL compiles," not "the GPU computes the
//! same result." Nothing is copied from the generator - files read are those
//! committed under `generated/rir/`.
//!
//! Without `libvulkan`, a GPU, a GLSL compiler, or on an insufficient device,
//! every test **skips** with an explanation. It becomes effective on machines
//! with a device (vulkan/cuda lanes), where failure is a real disagreement, not
//! an environment issue.

#![allow(clippy::unwrap_used)]
// Helpers outside a `#[test]` body, which is what `allow-unwrap-in-tests`
// covers. Same reasoning, said where the configuration cannot reach.

use std::path::PathBuf;

use rir_lower::Schedule;
use rir_lower::interp::{BoundArg, TensorView, TensorViewBytes, TensorViewMut, run};
use rir_runtime::any::{AnyGpu, Artifact, Backend};
use rir_runtime::{
    Arg, Gpu, Manifest, ManifestReader, Pipeline, Values, compile_glsl, is_unavailable,
};

fn kernel_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../generated/rir")
        .join(name)
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

/// Opens the device and builds a generated-kernel pipeline, or explains why the
/// test cannot run here.
fn setup(name: &str) -> Option<(Gpu, Manifest, Vec<u32>)> {
    let dir = kernel_dir(name);
    let manifest = match Manifest::load(&dir) {
        Ok(m) => m,
        Err(e) => panic!("unreadable manifest for {name}: {e}"),
    };
    let spirv = match compile_glsl(&dir.join("kernel.comp")) {
        Ok(s) => s,
        Err(e) if is_unavailable(&e) => {
            eprintln!("device parity {name} skipped: {e}");
            return None;
        }
        Err(e) => panic!("generated shader for {name} does not compile: {e}"),
    };
    match Gpu::open() {
        Ok(gpu) => {
            eprintln!("device parity {name} on '{}'", gpu.name());
            Some((gpu, manifest, spirv))
        }
        Err(e) if is_unavailable(&e) => {
            eprintln!("device parity {name} skipped: {e}");
            None
        }
        Err(e) => panic!("opening device: {e}"),
    }
}

/// Same setup for a named generated variant (`kernel.<variant>.comp` and its
/// matching manifest). Keeping this on committed files catches a stale AOT
/// copy; emitting again inside the test would only test the compiler twice.
fn setup_variant(kernel: &str, variant: &str) -> Option<(Gpu, Manifest, Vec<u32>)> {
    let dir = kernel_dir(kernel);
    let name = format!("{kernel}/{variant}");
    let manifest_path = dir.join(format!("manifest.{variant}.vulkan.json"));
    let manifest = std::fs::read_to_string(&manifest_path)
        .unwrap_or_else(|e| panic!("unreadable manifest for {name}: {e}"));
    let manifest = Manifest::from_json(&manifest)
        .unwrap_or_else(|e| panic!("invalid manifest for {name}: {e}"));
    let spirv = match compile_glsl(&dir.join(format!("kernel.{variant}.comp"))) {
        Ok(s) => s,
        Err(e) if is_unavailable(&e) => {
            eprintln!("device parity {name} skipped: {e}");
            return None;
        }
        Err(e) => panic!("generated shader for {name} does not compile: {e}"),
    };
    match Gpu::open() {
        Ok(gpu) => {
            eprintln!("device parity {name} on '{}'", gpu.name());
            Some((gpu, manifest, spirv))
        }
        Err(e) if is_unavailable(&e) => {
            eprintln!("device parity {name} skipped: {e}");
            None
        }
        Err(e) => panic!("opening device: {e}"),
    }
}

/// `build` distinguishes "this device is insufficient" (skip) from "the
/// pipeline is malformed" (fail) - the distinction made by `is_unavailable`.
fn pipeline<'g>(gpu: &'g Gpu, m: &Manifest, spirv: &[u32], name: &str) -> Option<Pipeline<'g>> {
    match gpu.build(m, spirv) {
        Ok(p) => Some(p),
        Err(e) if is_unavailable(&e) => {
            eprintln!("device parity {name} skipped: {e}");
            None
        }
        Err(e) => panic!("building pipeline {name}: {e}"),
    }
}

/// `l2_norm_back`: two deterministic reductions in one pass, with a branching
/// epilogue. Hardware `subgroupAdd` accumulation order differs from the
/// interpreter's increasing-lane simulation, hence relative tolerance - the
/// same as lane-lowering tests.
#[test]
fn l2_norm_back_on_gpu_against_the_oracle() {
    let Some((gpu, manifest, spirv)) = setup("l2_norm_back") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "l2_norm_back") else {
        return;
    };

    let kernel = rir_kernels::l2_norm_back::build().unwrap();
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();

    // `gap` separates `x` planes by a factor > 1: the geometry of a view into a
    // packed QKV tensor actually sent to this op by the Qwen3.5 graph.
    // This is the only case enabled by the three
    // outer axes and covered by no repetition.
    for &(n_col, n_row, n_plane, n_batch, gap) in &[
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (33, 5, 1, 1, 1),
        (257, 3, 1, 1, 1),
        (128, 4, 3, 2, 1),
        (128, 4, 3, 2, 3),
    ] {
        let n_rows_total = n_row * n_plane * n_batch;
        let len = n_col * n_rows_total;
        let mut seed = 0xd0d0_2026u64 ^ ((n_col as u64) << 32) ^ n_row as u64 ^ (gap as u64) << 8;
        let mut dz = vec![0f32; len];
        let mut x = vec![0f32; len * gap];
        fill(&mut seed, &mut dz, 2.0);
        fill(&mut seed, &mut x, 1.5);
        let eps = 1e-6f32;
        let nb = |g: usize| {
            [
                4usize,
                4 * n_col,
                4 * n_col * n_row * g,
                4 * n_col * n_row * g * n_plane,
            ]
        };
        let shape = [n_col, n_row, n_plane, n_batch];

        let mut expected = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView {
                data: &dz,
                shape,
                nb: nb(1),
            }),
            BoundArg::In(TensorView {
                data: &x,
                shape,
                nb: nb(gap),
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut expected,
                shape,
                nb: nb(1),
            }),
        ];
        run(&lk, &mut args, &[eps]).unwrap();

        let mut got = vec![0f32; len];
        let mut values = Values::new();
        values
            .f32("eps", eps)
            .u32("n_row", n_row as u32)
            .u32("n_plane", n_plane as u32)
            .u32("n_batch", n_batch as u32)
            .u32("n_col", n_col as u32)
            .strides("dz", &nb(1))
            .strides("x", &nb(gap))
            .strides("dx", &nb(1));
        pipe.run(
            &mut [Arg::input(&dz), Arg::input(&x), Arg::output(&mut got)],
            &values,
        )
        .expect("dispatch l2_norm_back");

        for i in 0..len {
            let (g, e) = (got[i], expected[i]);
            let tol = 1e-4f32.max(2e-4 * e.abs());
            assert!(
                (g - e).abs() <= tol,
                "l2_norm_back {n_col}x{n_row}x{n_plane}x{n_batch} gap {gap} \
                 element {i}: gpu {g} vs oracle {e}"
            );
        }
    }
}

/// `l2_norm_fwd`: one reduction and an elementwise epilogue dividing by it. The
/// forward counterpart of `l2_norm_back`, without branching - measured error can only come
/// from `subgroupAdd` accumulation order.
#[test]
fn l2_norm_fwd_on_gpu_against_the_oracle() {
    let Some((gpu, manifest, spirv)) = setup("l2_norm_fwd") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "l2_norm_fwd") else {
        return;
    };

    let kernel = rir_kernels::l2_norm_fwd::build().unwrap();
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();

    for &(n_col, n_row) in &[(1usize, 1usize), (33, 5), (257, 3)] {
        let len = n_col * n_row;
        let mut seed = 0x1112_2026u64 ^ ((n_col as u64) << 32) ^ n_row as u64;
        let mut x = vec![0f32; len];
        fill(&mut seed, &mut x, 1.5);
        let nb = [4usize, 4 * n_col];

        let mut expected = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut expected, n_col, n_row)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        let mut got = vec![0f32; len];
        let mut values = Values::new();
        values
            .u32("n_row", n_row as u32)
            .u32("n_col", n_col as u32)
            .strides("x", &nb)
            .strides("y", &nb);
        pipe.run(&mut [Arg::input(&x), Arg::output(&mut got)], &values)
            .expect("dispatch l2_norm_fwd");

        for i in 0..len {
            let (g, e) = (got[i], expected[i]);
            let tol = 1e-5f32.max(2e-4 * e.abs());
            assert!(
                (g - e).abs() <= tol,
                "l2_norm_fwd {n_col}x{n_row} element {i}: gpu {g} vs oracle {e}"
            );
        }
    }
}

/// `l2_norm_fwd_grad`: the kernel **derived** by transposition and the table's
/// most delicate path - two successive reduction levels, hence two
/// `subgroupAdd` operations where the second consumes the first's broadcast
/// result. No other kernel covers this dependency on hardware.
#[test]
fn l2_norm_fwd_grad_on_gpu_against_the_oracle() {
    let Some((gpu, manifest, spirv)) = setup("l2_norm_fwd_grad") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "l2_norm_fwd_grad") else {
        return;
    };

    let fwd = rir_kernels::l2_norm_fwd::build().unwrap();
    let kernel = rir_core::derive_backward(&fwd).unwrap();
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();

    // Binding order follows the manifest: x, dy, dx.
    assert_eq!(
        manifest
            .bindings
            .iter()
            .map(|b| b.name.as_str())
            .collect::<Vec<_>>(),
        ["x", "dy", "dx"]
    );

    for &(n_col, n_row) in &[(1usize, 1usize), (33, 5), (257, 3)] {
        let len = n_col * n_row;
        let mut seed = 0x2223_2026u64 ^ ((n_col as u64) << 32) ^ n_row as u64;
        let mut x = vec![0f32; len];
        let mut dy = vec![0f32; len];
        fill(&mut seed, &mut x, 1.5);
        fill(&mut seed, &mut dy, 2.0);
        let nb = [4usize, 4 * n_col];

        let mut expected = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut expected, n_col, n_row)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        let mut got = vec![0f32; len];
        let mut values = Values::new();
        values
            .u32("n_row", n_row as u32)
            .u32("n_col", n_col as u32)
            .strides("x", &nb)
            .strides("dy", &nb)
            .strides("dx", &nb);
        pipe.run(
            &mut [Arg::input(&x), Arg::input(&dy), Arg::output(&mut got)],
            &values,
        )
        .expect("dispatch l2_norm_fwd_grad");

        for i in 0..len {
            let (g, e) = (got[i], expected[i]);
            let tol = 1e-4f32.max(4e-4 * e.abs());
            assert!(
                (g - e).abs() <= tol,
                "l2_norm_fwd_grad {n_col}x{n_row} element {i}: gpu {g} vs oracle {e}"
            );
        }
    }
}

/// `cumsum`: one sequential scan per invocation, hence CPU accumulation order,
/// but partial sums written one by one - the only table kernel where every inner
/// loop iteration produces a store. One floating-point sum per element with no
/// possible contraction: the result must be **bit-for-bit** equal to the oracle.
#[test]
fn cumsum_on_gpu_against_the_oracle() {
    let Some((gpu, manifest, spirv)) = setup("cumsum") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "cumsum") else {
        return;
    };

    let kernel = rir_kernels::cumsum::build().unwrap();
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_grid([64, 1, 1])).unwrap();

    // Like `l2_norm_back`, `gap` separates `x` planes: the scan must read the
    // correct plane of a packed-tensor view, which no repetition of
    // rows cannot express.
    for &(n_col, n_row, n_plane, n_batch, gap) in &[
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (7, 3, 1, 1, 1),
        (129, 70, 1, 1, 1),
        (17, 5, 3, 2, 1),
        (17, 5, 3, 2, 3),
    ] {
        let n_rows_total = n_row * n_plane * n_batch;
        let len = n_col * n_rows_total;
        let mut seed = 0x3334_2026u64 ^ ((n_col as u64) << 32) ^ n_row as u64 ^ (gap as u64) << 8;
        let mut x = vec![0f32; len * gap];
        fill(&mut seed, &mut x, 1.0);
        let nb = |g: usize| {
            [
                4usize,
                4 * n_col,
                4 * n_col * n_row * g,
                4 * n_col * n_row * g * n_plane,
            ]
        };
        let shape = [n_col, n_row, n_plane, n_batch];

        let mut expected = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView {
                data: &x,
                shape,
                nb: nb(gap),
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut expected,
                shape,
                nb: nb(1),
            }),
        ];
        run(&lk, &mut args, &[]).unwrap();

        let mut got = vec![0f32; len];
        let mut values = Values::new();
        values
            .u32("n_row", n_row as u32)
            .u32("n_plane", n_plane as u32)
            .u32("n_batch", n_batch as u32)
            .u32("n_col", n_col as u32)
            .strides("x", &nb(gap))
            .strides("y", &nb(1));
        pipe.run(&mut [Arg::input(&x), Arg::output(&mut got)], &values)
            .expect("dispatch cumsum");

        assert_eq!(
            got, expected,
            "cumsum {n_col}x{n_row}x{n_plane}x{n_batch} gap {gap} : \
             the GPU reassociated a scan"
        );
    }
}

/// `run` guards: neither a short argument nor reversed binding direction reaches
/// dispatch. The first would be out-of-bounds SSBO access; the second a silent
/// wrong result.
#[test]
fn invalid_arguments_are_refused_before_the_dispatch() {
    let Some((gpu, manifest, spirv)) = setup("l2_norm_fwd") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "l2_norm_fwd") else {
        return;
    };

    let (n_col, n_row) = (16usize, 4usize);
    let nb = [4usize, 4 * n_col];
    let mut values = Values::new();
    values
        .u32("n_row", n_row as u32)
        .u32("n_col", n_col as u32)
        .strides("x", &nb)
        .strides("y", &nb);

    let x = vec![1f32; n_col * n_row];
    let mut y = vec![0f32; n_col * n_row];
    pipe.run(&mut [Arg::input(&x), Arg::output(&mut y)], &values)
        .expect("the conforming shape passes");

    // One row fewer than announced by push constants.
    let court = vec![1f32; n_col * (n_row - 1)];
    match pipe.run(&mut [Arg::input(&court), Arg::output(&mut y)], &values) {
        Err(rir_runtime::RuntimeError::BufferTooSmall { binding, .. }) => {
            assert_eq!(binding, "x")
        }
        other => panic!("expected BufferTooSmall, {other:?}"),
    }

    // `y` is a write binding: binding it for reading would never read the result back.
    match pipe.run(&mut [Arg::input(&x), Arg::input(&x)], &values) {
        Err(rir_runtime::RuntimeError::AccessMismatch { binding, .. }) => {
            assert_eq!(binding, "y")
        }
        other => panic!("expected AccessMismatch, {other:?}"),
    }
}

/// `sum_rows_<format>`: the GPU-side quantized loader - `float16_t` scale and
/// bytes read from the same binding. This path merits a device test:
/// dequantization is the only contract part where GLSL and the interpreter may
/// disagree on byte interpretation, not merely addition order.
///
/// The loop traverses `sum_rows_quant::variants()` instead of naming `q8_0`, so
/// a format added to the canonical table is validated on device without
/// rewriting any test.
#[test]
fn sum_rows_quant_on_gpu_against_the_oracle() {
    for format in rir_kernels::sum_rows_quant::variants() {
        let name = rir_kernels::sum_rows_quant::kernel_name(format);
        let Some((gpu, manifest, spirv)) = setup(&name) else {
            return;
        };
        let Some(pipe) = pipeline(&gpu, &manifest, &spirv, &name) else {
            return;
        };

        let kernel = rir_kernels::sum_rows_quant::build(format).unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();
        let d = format.desc();
        let (be, bb) = (d.block_elements as usize, d.block_bytes as usize);

        for &blocks in &[1usize, 4, 32] {
            let (n_col, n_row) = (blocks * be, 3usize);
            let stride = blocks * bb;
            let mut seed = 0xbeef_2026u64 ^ ((n_col as u64) << 16) ^ n_row as u64;
            // The single fixture derived from the description.
            // Previous code pinned one F16 scale at
            // offset 0: correct for simple blocks, wrong for super-blocks whose
            // scales are at the end - `q2_K`'s `dmin` is at offset 82, where a
            // random half is inf or NaN once in thirty-two. Both sides then
            // returned NaN and the test failed on `NaN != NaN`, testing its
            // fixture rather than the kernel.
            let raw = rir_core::random_block_bytes(format, blocks * n_row, &mut seed)
                .unwrap_or_else(|| panic!("{name}: format without a description"));
            let nb_x = [bb, stride];

            // The oracle is the interpreter over *the same bytes*: the only
            // comparison isolating GPU decoding.
            let mut expected = vec![0f32; n_row];
            let mut args = [
                BoundArg::InBytes(TensorViewBytes {
                    data: &raw,
                    shape: [n_col, n_row, 1, 1],
                    nb: [bb, stride, stride * n_row, stride * n_row],
                }),
                BoundArg::Out(TensorViewMut::contiguous_1d(&mut expected, n_row)),
            ];
            run(&lk, &mut args, &[]).unwrap();

            let mut got = vec![0f32; n_row];
            let mut values = Values::new();
            values
                .u32("n_row", n_row as u32)
                .u32("n_col", n_col as u32)
                .strides("x", &nb_x)
                .strides("y", &[4usize]);
            pipe.run(&mut [Arg::input(&raw), Arg::output(&mut got)], &values)
                .unwrap_or_else(|e| panic!("dispatch {name} : {e:?}"));

            for r in 0..n_row {
                let (g, e) = (got[r], expected[r]);
                let tol = 1e-3f32.max(2e-4 * e.abs());
                assert!(
                    (g - e).abs() <= tol,
                    "{name} {n_col}x{n_row} row {r}: gpu {g} vs oracle {e}"
                );
            }
        }
    }
}

/// `mat_mul_naive`: `invocation` mapping over two grid dimensions with a
/// sequential contraction inside the invocation. Accumulation **order** is thus
/// the CPU's, but not the bit-for-bit result: GLSL and SPIR-V permit contracting
/// `a*b + acc` into fused multiply-add with different rounding. Measured on
/// Apple M1/MoltenVK: output exactly matches a `mul_add` reference, never
/// separate `mul` then `add`.
///
/// This does not violate the contract: `Deterministic` promises a stable result
/// across runs on one device, while `ExactOrder` promises accumulation order,
/// not a rounding mode. The bound is therefore a few ulps per accumulated term,
/// much tighter than subgroup-kernel relative tolerance, preserving the
/// distinction.
#[test]
fn mat_mul_naive_on_gpu_against_the_oracle_up_to_contraction() {
    let Some((gpu, manifest, spirv)) = setup("mat_mul_naive") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "mat_mul_naive") else {
        return;
    };

    let kernel = rir_kernels::mat_mul_naive::build().unwrap();
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_grid([16, 16, 1])).unwrap();

    for &(n_k, n_i, n_j) in &[(1usize, 1usize, 1usize), (16, 4, 5), (33, 17, 20)] {
        let mut seed = 0xa11_2026u64 ^ ((n_k as u64) << 20) ^ n_i as u64;
        let mut a = vec![0f32; n_k * n_i];
        let mut b = vec![0f32; n_k * n_j];
        fill(&mut seed, &mut a, 1.0);
        fill(&mut seed, &mut b, 1.0);
        let (nb_a, nb_b, nb_c) = ([4usize, 4 * n_k], [4usize, 4 * n_k], [4usize, 4 * n_i]);

        let mut expected = vec![0f32; n_i * n_j];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&a, n_k, n_i)),
            BoundArg::In(TensorView::contiguous_2d(&b, n_k, n_j)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut expected, n_i, n_j)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        let mut got = vec![0f32; n_i * n_j];
        let mut values = Values::new();
        values
            .u32("n_i", n_i as u32)
            .u32("n_j", n_j as u32)
            .u32("n_k", n_k as u32)
            .strides("a", &nb_a)
            .strides("b", &nb_b)
            .strides("c", &nb_c);
        pipe.run(
            &mut [Arg::input(&a), Arg::input(&b), Arg::output(&mut got)],
            &values,
        )
        .expect("dispatch mat_mul_naive");

        for idx in 0..n_i * n_j {
            let (g, e) = (got[idx], expected[idx]);
            // At worst one ulp per contracted term: contraction does not
            // propagate beyond accumulation itself.
            let tol = n_k as f32 * f32::EPSILON * e.abs().max(1.0);
            assert!(
                (g - e).abs() <= tol,
                "mat_mul_naive {n_k}x{n_i}x{n_j} element {idx}: gpu {g} vs oracle {e} \
                 (difference {}, bound {tol})",
                (g - e).abs()
            );
        }
    }
}

/// GPU `out_prod` against the oracle on real-graph geometry: full rank 4 and `a`
/// viewed inside a packed tensor.
///
/// What this catches and `mat_mul_naive` above cannot: `out_prod` is the first
/// **five-axis** kernel, hence the first whose fourth parallel axis becomes a
/// sequential shader loop for lack of a fourth grid dimension. An off-by-one in
/// this order - using `batch` for `plane` - remains invisible while either is 1,
/// as both are in most benchmark shapes.
///
/// Since this rule was introduced, it is also the only cooperative-staging check: the oracle fills
/// each whole tile alone because it has no workgroup; the shader fills it with
/// 256 invocations distributing elements and synchronizing. They agree only if
/// distribution, bounds, and barriers are correct - the half of tiling only a
/// device can judge.
#[test]
fn out_prod_on_gpu_against_the_oracle() {
    let Some((gpu, manifest, spirv)) = setup("out_prod") else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, "out_prod") else {
        return;
    };

    let kernel = rir_kernels::out_prod::build().unwrap();
    // The schedule producing the `.comp` loaded above: the oracle must execute
    // *this* staging, not a similar one.
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_grid_tiled([16, 16, 1], 16, 4)).unwrap();

    // `gap` separates `a` rows: its nb[1..3] then cannot be derived from any
    // product of extents, as with a view.
    //
    // No extent is a multiple of 16: every shape has a workgroup crossing an
    // edge, whose out-of-tensor invocations must still perform their share of
    // loads and reach barriers. One early return would block the others.
    for &(n_i, n_j, n_k, n_plane, n_batch, gap) in &[
        (1usize, 1usize, 1usize, 1usize, 1usize, 1usize),
        (17, 20, 33, 1, 1, 1),
        (5, 3, 8, 3, 2, 1),
        (4, 6, 7, 2, 3, 3),
    ] {
        let sa1 = 4 * n_i * gap;
        let nb_a = [4usize, sa1, sa1 * n_k, sa1 * n_k * n_plane];
        let nb_b = [4usize, 4 * n_j, 4 * n_j * n_k, 4 * n_j * n_k * n_plane];
        let nb_d = [4usize, 4 * n_i, 4 * n_i * n_j, 4 * n_i * n_j * n_plane];
        let len = |nb: [usize; 4]| nb[3] / 4 * n_batch;

        let mut seed = 0xd07_2026u64 ^ ((n_i as u64) << 20) ^ ((n_j as u64) << 8) ^ n_k as u64;
        let mut a = vec![0f32; len(nb_a)];
        let mut b = vec![0f32; len(nb_b)];
        fill(&mut seed, &mut a, 1.0);
        fill(&mut seed, &mut b, 1.0);

        let shape_a = [n_i, n_k, n_plane, n_batch];
        let shape_b = [n_j, n_k, n_plane, n_batch];
        let shape_d = [n_i, n_j, n_plane, n_batch];

        let mut expected = vec![0f32; len(nb_d)];
        let mut args = [
            BoundArg::In(TensorView {
                data: &a,
                shape: shape_a,
                nb: nb_a,
            }),
            BoundArg::In(TensorView {
                data: &b,
                shape: shape_b,
                nb: nb_b,
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut expected,
                shape: shape_d,
                nb: nb_d,
            }),
        ];
        run(&lk, &mut args, &[]).unwrap();

        let mut got = vec![0f32; len(nb_d)];
        let mut values = Values::new();
        values
            .u32("n_i", n_i as u32)
            .u32("n_j", n_j as u32)
            .u32("n_plane", n_plane as u32)
            .u32("n_batch", n_batch as u32)
            .u32("n_k", n_k as u32)
            .strides("a", &nb_a)
            .strides("b", &nb_b)
            .strides("dst", &nb_d);
        pipe.run(
            &mut [Arg::input(&a), Arg::input(&b), Arg::output(&mut got)],
            &values,
        )
        .expect("dispatch out_prod");

        for (idx, (&g, &e)) in got.iter().zip(&expected).enumerate() {
            // Both sides traverse the contraction in the same order: one ulp per
            // accumulated term bounds the difference, nothing more.
            let tol = n_k as f32 * f32::EPSILON * e.abs().max(1.0);
            assert!(
                (g - e).abs() <= tol,
                "out_prod {n_i}x{n_j}x{n_k} plane={n_plane} batch={n_batch} gap={gap} \
                 element {idx}: gpu {g} vs oracle {e}"
            );
        }
    }
}

/// The **blocked** scan (`ScanStrategy::BlockedLanes`) against the oracle. Two
/// things distinguish it from tests above. It cannot be bit-for-bit. Sequential
/// scan can because it has one possible order; this recombines block totals
/// through a collective tree whose hardware association differs from the
/// interpreter's simulation. This is exactly the difference between `ExactOrder`
/// and `Deterministic`.
fn cumsum_variant_on_gpu_against_the_oracle(variant: &str, schedule: Schedule) {
    let name = format!("cumsum/{variant}");
    let kernel = rir_kernels::cumsum::build().unwrap();
    let lk = rir_lower::lower(&kernel, schedule).expect("blocked-scan lowering");
    let Some((gpu, manifest, spirv)) = setup_variant("cumsum", variant) else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, &name) else {
        return;
    };

    // Lengths around 32 lanes: a block shorter than lane count, exact block,
    // block with remainder, and a long row motivating the strategy. `gap`
    // replays the packed-tensor view.
    for &(n_col, n_row, n_plane, n_batch, gap) in &[
        (1usize, 1usize, 1usize, 1usize, 1usize),
        (7, 3, 1, 1, 1),
        (32, 2, 1, 1, 1),
        (33, 2, 1, 1, 1),
        (129, 70, 1, 1, 1),
        (4096, 2, 1, 1, 1),
        (17, 5, 3, 2, 3),
    ] {
        let n_rows_total = n_row * n_plane * n_batch;
        let len = n_col * n_rows_total;
        let mut seed = 0x5CA7_2026u64 ^ ((n_col as u64) << 32) ^ n_row as u64 ^ (gap as u64) << 8;
        let mut x = vec![0f32; len * gap];
        fill(&mut seed, &mut x, 1.0);
        let nb = |g: usize| {
            [
                4usize,
                4 * n_col,
                4 * n_col * n_row * g,
                4 * n_col * n_row * g * n_plane,
            ]
        };
        let shape = [n_col, n_row, n_plane, n_batch];

        let mut expected = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView {
                data: &x,
                shape,
                nb: nb(gap),
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut expected,
                shape,
                nb: nb(1),
            }),
        ];
        run(&lk, &mut args, &[]).unwrap();

        let mut got = vec![0f32; len];
        let mut values = Values::new();
        values
            .u32("n_row", n_row as u32)
            .u32("n_plane", n_plane as u32)
            .u32("n_batch", n_batch as u32)
            .u32("n_col", n_col as u32)
            .strides("x", &nb(gap))
            .strides("y", &nb(1));
        pipe.run(&mut [Arg::input(&x), Arg::output(&mut got)], &values)
            .expect("dispatch cumsum collectif");

        for i in 0..len {
            let (g, e) = (got[i], expected[i]);
            let tol = 1e-4f32.max(2e-4 * e.abs());
            assert!(
                (g - e).abs() <= tol,
                "cumsum {variant} {n_col}x{n_row}x{n_plane}x{n_batch} gap {gap} \
                 element {i}: gpu {g} vs oracle {e}"
            );
        }
    }
}

/// The vectorized elementwise strip: the shader reads
/// and writes four elements per invocation; the oracle handles them one by one.
///
/// What only this test decides: the **tail**. A length not divisible by four
/// sends the final invocation through the scalar branch, the only pipeline
/// location where it executes. The sweep covers all three remainders and a row
/// shorter than the vector, where the vector branch is never taken. Elementwise,
/// hence bit-for-bit: no addition is regrouped.
fn band_vec4_on_gpu_against_the_oracle(kernel_name: &str, binary: bool, params: &[f32]) {
    for target in Backend::available() {
        band_vec4_on_one_backend(target, kernel_name, binary, params);
    }
}

/// The same sweep on one backend.
///
/// It is the case CUDA needed most and the one no other test reaches there: the
/// vectorized band's **tail**. `family_parity` covers a kernel's fallback, and
/// for `add`/`mul` the fallback is the scalar lowering - so without this the
/// four-wide read and its scalar remainder would go to a device untested.
fn band_vec4_on_one_backend(target: Backend, kernel_name: &str, binary: bool, params: &[f32]) {
    let name = format!("{kernel_name}/vec4/{}", target.name());
    // Since F4 the strip is indexed by (op, type); this sweep covers the F32 path,
    // so it names the member by op and takes type from the table rather than
    // reconstructing it.
    let band = *rir_kernels::elementwise::variants()
        .iter()
        .find(|b| b.kernel_name() == kernel_name)
        .unwrap_or_else(|| panic!("unknown strip member: {kernel_name}"));
    let entry = rir_kernels::registry()
        .into_iter()
        .find(|e| e.kernel.name() == kernel_name)
        .unwrap_or_else(|| panic!("{kernel_name} absent from the registry"));
    // The vectorized lowering of this backend, found by the property that
    // defines it rather than by its name: `scale` publishes its vector lowering
    // as the pair's **fallback** - ggml guarantees the layout it claims - so it
    // carries the bare kernel name, while `add` and `mul` carry `vec4`.
    let lower_backend = match target {
        Backend::Vulkan => rir_lower::Backend::Vulkan,
        Backend::Cuda => rir_lower::Backend::Cuda,
    };
    let Some(schedule) = entry
        .schedules
        .iter()
        .find(|s| s.backend() == lower_backend && s.vector_width() > 1)
        .cloned()
    else {
        eprintln!("{name} skipped: no vectorized schedule on this backend");
        return;
    };
    let variant = schedule.variant();
    let kernel = rir_kernels::elementwise::build(band).unwrap();
    let lk = rir_lower::lower(&kernel, schedule).expect("vectorized lowering");

    let dir = kernel_dir(kernel_name);
    let (source_file, manifest_file) = target.files(variant);
    let manifest = Manifest::load_file(&dir, &manifest_file)
        .unwrap_or_else(|e| panic!("{name}: {manifest_file}: {e}"));
    let source = std::fs::read_to_string(dir.join(&source_file))
        .unwrap_or_else(|e| panic!("{name}: {source_file}: {e}"));
    let params_header = std::fs::read_to_string(kernel_dir("registry").join("rir_kernel_params.h"))
        .expect("rir_kernel_params.h");
    let artifact = rir_emit::artifact_name(&lk);

    let gpu = match AnyGpu::open(target) {
        Ok(g) => {
            eprintln!("device parity {name} on '{}'", g.name());
            g
        }
        Err(e) if is_unavailable(&e) => {
            eprintln!("{name} skipped: {e}");
            return;
        }
        Err(e) => panic!("opening device: {e}"),
    };
    let pipe = match gpu.build(
        &manifest,
        &Artifact {
            artifact: &artifact,
            source: &source,
            params_header: &params_header,
        },
    ) {
        Ok(p) => p,
        Err(e) if is_unavailable(&e) => {
            eprintln!("{name} skipped: {e}");
            return;
        }
        Err(e) => panic!("building {name}: {e}"),
    };

    for &(n_col, n_row, n_plane, n_batch) in &[
        (1usize, 1usize, 1usize, 1usize),
        (2, 3, 1, 1),
        (3, 3, 2, 2),
        (4, 5, 1, 1),
        (5, 5, 1, 1),
        (7, 4, 2, 1),
        (33, 4, 3, 2),
        (4096, 16, 1, 1),
    ] {
        let len = n_col * n_row * n_plane * n_batch;
        let nb = [
            4usize,
            4 * n_col,
            4 * n_col * n_row,
            4 * n_col * n_row * n_plane,
        ];
        let shape = [n_col, n_row, n_plane, n_batch];
        let mut seed = 0x7ec4_2026u64 ^ ((n_col as u64) << 32) ^ n_row as u64;
        let mut a = vec![0f32; len];
        let mut b = vec![0f32; len];
        fill(&mut seed, &mut a, 2.0);
        fill(&mut seed, &mut b, 1.5);

        let mut expected = vec![0f32; len];
        let mut args: Vec<BoundArg> = vec![BoundArg::In(TensorView {
            data: &a,
            shape,
            nb,
        })];
        if binary {
            args.push(BoundArg::In(TensorView {
                data: &b,
                shape,
                nb,
            }));
        }
        args.push(BoundArg::Out(TensorViewMut {
            data: &mut expected,
            shape,
            nb,
        }));
        run(&lk, &mut args, params).unwrap();

        let mut got = vec![0f32; len];
        let mut values = Values::new();
        for (i, p) in kernel.params().iter().enumerate() {
            values.f32(&p.name, params[i]);
        }
        values
            .u32("n_col", n_col as u32)
            .u32("n_row", n_row as u32)
            .u32("n_plane", n_plane as u32)
            .u32("n_batch", n_batch as u32)
            .strides("a", &nb);
        if binary {
            values.strides("b", &nb);
        }
        values.strides("dst", &nb);
        flat_values(&mut values, &lk, &shape);
        let mut dispatch: Vec<Arg> = vec![Arg::input(&a)];
        if binary {
            dispatch.push(Arg::input(&b));
        }
        dispatch.push(Arg::output(&mut got));
        pipe.run(&mut dispatch, &values)
            .unwrap_or_else(|e| panic!("dispatch {name} : {e}"));

        for i in 0..len {
            let (g, e) = (got[i], expected[i]);
            // `add` and `mul` are one floating-point operation per element: only
            // one rounding is possible, so equality is **bit-for-bit**, proving
            // four-wide reads access the same bytes as scalar reads.
            //
            // `scale` performs two, `a·s + bias`, and shader compilers may
            // contract them into fused multiply-add; the oracle rounds the
            // product before addition. The difference is that rounding alone,
            // bounded by two ulps. This is the same permission documented by
            // `mat_mul_naive`, not a convenience tolerance.
            let tol = if binary {
                0.0
            } else {
                2.0 * f32::EPSILON * e.abs().max(1.0)
            };
            assert!(
                (g - e).abs() <= tol,
                "{name} {n_col}x{n_row}x{n_plane}x{n_batch} element {i}: \
                 gpu {g} vs oracle {e}"
            );
        }
    }
}

#[test]
fn add_vec4_on_gpu_against_the_oracle() {
    band_vec4_on_gpu_against_the_oracle("add", true, &[]);
}

#[test]
fn mul_vec4_on_gpu_against_the_oracle() {
    band_vec4_on_gpu_against_the_oracle("mul", true, &[]);
}

#[test]
fn scale_vec4_on_gpu_against_the_oracle() {
    band_vec4_on_gpu_against_the_oracle("scale", false, &[1.75, -0.5]);
}

#[test]
fn blocked_cumsum_on_gpu_against_the_oracle() {
    cumsum_variant_on_gpu_against_the_oracle("blocked", Schedule::vulkan_blocked_scan());
}

/// The **strided** scan (`ScanStrategy::StridedLanes`) against the oracle: its
/// last round is the partial one, where lanes past the row contribute the
/// identity to both collectives, so the lengths around 32 above are the ones
/// that decide it.
#[test]
fn strided_cumsum_on_gpu_against_the_oracle() {
    cumsum_variant_on_gpu_against_the_oracle(
        "strided",
        Schedule::gpu_strided_scan(rir_lower::GpuBackend::Vulkan),
    );
}

/// The **tiled** scan (`ScanStrategy::TiledLanes`) against
/// the oracle on the same shapes as its two predecessors. It adds an interleaved
/// access plan and a **carry** passed from one tile to the next; shapes not
/// divisible by tile size therefore matter here, and the list contains three.
#[test]
fn tiled_cumsum_on_gpu_against_the_oracle() {
    cumsum_variant_on_gpu_against_the_oracle("tiled", Schedule::vulkan_tiled_scan(256, 16));
}

/// `read_outputs` is a **safe** API: it must not cause a read beyond the Vulkan
/// mapping. Device buffers were sized by `prepare`, so reading into a longer
/// slice would copy beyond the mapping. The "same bindings, same sizes"
/// constraint was documented but unchecked; this test makes it true.
#[test]
fn read_outputs_refuses_a_binding_that_is_not_the_one_prepared() {
    let name = "cumsum";
    let Some((gpu, manifest, spirv)) = setup(name) else {
        return;
    };
    let Some(pipe) = pipeline(&gpu, &manifest, &spirv, name) else {
        return;
    };

    let (n_col, n_row) = (16usize, 2usize);
    let len = n_col * n_row;
    let x = vec![1f32; len];
    let mut y = vec![0f32; len];
    let nb = [4usize, 4 * n_col, 4 * n_col * n_row, 4 * n_col * n_row];
    let mut values = Values::new();
    values
        .u32("n_row", n_row as u32)
        .u32("n_plane", 1)
        .u32("n_batch", 1)
        .u32("n_col", n_col as u32)
        .strides("x", &nb)
        .strides("y", &nb);

    let session = pipe
        .prepare(&[Arg::input(&x), Arg::output(&mut y)], &values)
        .expect("preparing cumsum");
    session.dispatch().expect("dispatch cumsum");

    // Longer than the bound slice: the case that read beyond the mapping.
    let mut too_long = vec![0f32; len * 4];
    let err = session
        .read_outputs(&mut [Arg::input(&x), Arg::output(&mut too_long)])
        .expect_err("an oversized output must be rejected");
    assert!(
        matches!(err, rir_runtime::RuntimeError::ArgLenMismatch { .. }),
        "unexpected error: {err}"
    );

    // Shorter: no overflow, but a truncated result returned as complete - just
    // as wrong and rejected for the same reason.
    let mut too_short = vec![0f32; len / 2];
    assert!(
        session
            .read_outputs(&mut [Arg::input(&x), Arg::output(&mut too_short)])
            .is_err()
    );

    // Direction also matters: binding an input where `prepare` had an output
    // would read nothing back and return `Ok`.
    let swapped = vec![0f32; len];
    assert!(
        session
            .read_outputs(&mut [Arg::input(&x), Arg::input(&swapped)])
            .is_err()
    );

    // The original readback still works.
    session
        .read_outputs(&mut [Arg::input(&x), Arg::output(&mut y)])
        .expect("the original readback must remain accepted");
    assert_eq!(y[0], 1.0, "the scan was not read back");
}

/// The CUDA half of the contract that is checked **before** a device is touched:
/// what the manifest requires of it.
///
/// Two properties, and the second is the one worth a test. A requirement the
/// device cannot meet must be a `MissingFeature` naming it, not an `nvcc`
/// diagnostic or a launch failure about a resource. And a requirement this
/// runtime does not **recognize** must be refused just as firmly: a manifest
/// newer than its reader carries a condition nobody evaluated, and accepting it
/// is how a measurement comes back green for a kernel whose contract was never
/// checked. The Vulkan half has said so since it was written ("an unknown
/// feature is a rejection, not a shrug"); this is CUDA's.
#[cfg(feature = "cuda")]
#[test]
fn an_unmeetable_or_unknown_cuda_feature_is_refused() {
    use rir_runtime::RuntimeError;

    let gpu = match rir_runtime::cuda::Gpu::open() {
        Ok(g) => g,
        Err(e) if is_unavailable(&e) => {
            eprintln!("cuda feature check skipped: {e}");
            return;
        }
        Err(e) => panic!("opening device: {e}"),
    };
    // A real generated manifest, and the device that has to satisfy it: `scale`
    // publishes `cuda>=5.0` and `block_threads>=256`, which any CUDA device
    // meets - so the accepted case below is a fact about this device rather than
    // an empty list.
    let dir = kernel_dir("scale");
    let base = std::fs::read_to_string(dir.join("manifest.cuda.json")).expect("manifest.cuda.json");
    let manifest = Manifest::from_json(&base).expect("manifest");
    assert!(
        manifest.features.iter().any(|f| f.starts_with("cuda>=")),
        "the manifest publishes no compute capability: {:?}",
        manifest.features
    );
    gpu.check_features(&manifest)
        .expect("the device must satisfy a manifest it will be asked to run");

    // One case per published budget, plus the two ways a feature can be
    // unreadable. Each replaces the whole `features` list, so the failure is
    // attributable to the line under test.
    for (features, why) in [
        (r#"["cuda>=99.0"]"#, "a compute capability from the future"),
        (
            r#"["shared_bytes>=1099511627776"]"#,
            "a terabyte of shared memory per block",
        ),
        (r#"["block_threads>=1048576"]"#, "a million threads a block"),
        (
            r#"["quantum_entanglement"]"#,
            "a feature nobody implemented",
        ),
        (r#"["cuda>=eight.nine"]"#, "a version that is not a number"),
        (r#"["shared_bytes>=lots"]"#, "a size that is not a number"),
    ] {
        let start = base.find("\"features\":").expect("features field");
        let end = base[start..].find(']').expect("features list") + start + 1;
        let mutated = format!("{}\"features\": {features}{}", &base[..start], &base[end..]);
        let manifest = Manifest::from_json(&mutated).unwrap_or_else(|e| panic!("{why}: {e}"));
        match gpu.check_features(&manifest) {
            Err(RuntimeError::MissingFeature { feature, .. }) => {
                assert!(mutated.contains(&feature), "{why}: named {feature}")
            }
            other => panic!("{why} was not refused: {other:?}"),
        }
    }
}

/// Several CUDA devices open **in one process, at once**.
///
/// This is a property of the harness rather than of a kernel, and it needs a
/// test because the way it broke is invisible: `cargo test` runs its tests on
/// threads of one process, every `Gpu::open` compiles a host harness, and while
/// they all compiled it under one name two `nvcc` invocations wrote one shared
/// object - one of them into a file another was loading. Nothing failed
/// deterministically. It hung, or it did not, depending on which write landed
/// first.
///
/// So the instances are opened concurrently and each one is made to compile,
/// dispatch and read back its own kernel. A directory shared between them would
/// show up here as a corrupt object, a missing symbol, or a hang - and a
/// `Drop` that removed a directory still in use would show up as the second
/// half of this test failing after the first instance is gone.
#[cfg(feature = "cuda")]
#[test]
fn several_cuda_instances_coexist_in_one_process() {
    let dir = kernel_dir("scale");
    let source = std::fs::read_to_string(dir.join("kernel.cu")).expect("kernel.cu");
    let manifest = Manifest::load_file(&dir, "manifest.cuda.json").expect("manifest.cuda.json");
    let header = std::fs::read_to_string(kernel_dir("registry").join("rir_kernel_params.h"))
        .expect("rir_kernel_params.h");

    // `scale`, on a shape small enough that four of them cost a compile each and
    // nothing else: what is under test is the harness, not the kernel.
    let (n_col, n_row) = (64usize, 4usize);
    let len = n_col * n_row;
    let nb = [4usize, 4 * n_col, 4 * len, 4 * len];
    let run_one = |gpu: &rir_runtime::cuda::Gpu| {
        let pipe = gpu
            .build(
                &manifest,
                &rir_runtime::cuda::Unit {
                    artifact: "scale",
                    source: &source,
                    params_header: &header,
                },
            )
            .expect("building scale");
        let a = vec![2f32; len];
        let mut dst = vec![0f32; len];
        let mut values = Values::new();
        values
            .f32("scale", 3.0)
            .f32("bias", 1.0)
            .u32("n_col", n_col as u32)
            .u32("n_row", n_row as u32)
            .u32("n_plane", 1)
            .u32("n_batch", 1)
            .strides("a", &nb)
            .strides("dst", &nb);
        // `scale` is flattened on CUDA, so its constant
        // block carries the divisors of the decomposition - read off the
        // manifest, which is where the shader's own geometry is published.
        flat_values_from(
            &mut values,
            &manifest,
            &[("col", n_col), ("row", n_row), ("plane", 1), ("batch", 1)],
        );
        pipe.run(&mut [Arg::input(&a), Arg::output(&mut dst)], &values)
            .expect("dispatching scale");
        // 2·3 + 1: the kernel ran, and it ran on *this* instance's buffers.
        assert!(dst.iter().all(|v| (*v - 7.0).abs() < 1e-6), "{dst:?}");
    };

    // The first one decides whether this machine can run the test at all, so the
    // threads below never have to distinguish "no toolkit" from "broken".
    let first = match rir_runtime::cuda::Gpu::open() {
        Ok(g) => g,
        Err(e) if is_unavailable(&e) => {
            eprintln!("cuda instances skipped: {e}");
            return;
        }
        Err(e) => panic!("opening device: {e}"),
    };

    std::thread::scope(|s| {
        let handles: Vec<_> = (0..3)
            .map(|i| {
                s.spawn(move || {
                    let gpu = rir_runtime::cuda::Gpu::open()
                        .unwrap_or_else(|e| panic!("instance {i}: {e}"));
                    run_one(&gpu);
                })
            })
            .collect();
        run_one(&first);
        for h in handles {
            h.join().expect("an instance panicked");
        }
    });

    // The three concurrent instances are dropped by now, each having removed its
    // own directory. The one still open must be untouched by that.
    run_one(&first);
}

/// The same block, derived from the **manifest** instead of the lowering, for
/// the tests that hold a manifest and no `LoopKernel`. One computation, two
/// sources of the geometry, and both are published rather than assumed.
#[expect(unused)]
fn flat_values_from(values: &mut Values, m: &Manifest, extents: &[(&str, usize)]) {
    if m.flat.is_empty() {
        return;
    }
    let d: Vec<u32> = m
        .flat
        .iter()
        .map(|f| {
            let n = extents
                .iter()
                .find(|(name, _)| *name == f.axis)
                .unwrap_or_else(|| panic!("no extent for flat axis {}", f.axis))
                .1;
            n.div_ceil(f.per_index as usize) as u32
        })
        .collect();
    values.u32("rir_flat_total", d.iter().product());
    for (i, &di) in d.iter().enumerate().take(d.len() - 1) {
        let (mp, sh) = rir_lower::fastdiv_magic(di);
        values
            .u32(&format!("rir_flat{i}_div"), di)
            .u32(&format!("rir_flat{i}_mp"), mp)
            .u32(&format!("rir_flat{i}_sh"), sh);
    }
}

/// The constant block a flattened dispatch reads, derived
/// from the lowering under test: the number of points, then one (divisor, magic
/// multiplier, shift) triple per divisor.
///
/// Derived and not hand-written, for the reason every push constant here is:
/// what the shader declares is what the manifest declares, and a value this file
/// invented would be a second ABI. A lowering that flattens nothing declares
/// none of these names, so this writes nothing and costs nothing.
fn flat_values(values: &mut Values, lk: &rir_lower::LoopKernel, extents: &[usize]) {
    let flat = lk.flat_axes();
    if flat.is_empty() {
        return;
    }
    let d: Vec<u32> = flat
        .iter()
        .map(|(ax, per)| extents[ax.0 as usize].div_ceil(*per as usize) as u32)
        .collect();
    values.u32("rir_flat_total", d.iter().product());
    for (i, &di) in d.iter().enumerate().take(d.len() - 1) {
        let (mp, sh) = rir_lower::fastdiv_magic(di);
        values
            .u32(&format!("rir_flat{i}_div"), di)
            .u32(&format!("rir_flat{i}_mp"), mp)
            .u32(&format!("rir_flat{i}_sh"), sh);
    }
}
