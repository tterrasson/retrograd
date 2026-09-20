//! Level-1 measurement loop: time the generated
//! shader alone, without ggml or rebuilding the fork.
//!
//! What this harness provides and the promotion lane does not: the kernel is
//! **emitted in-process** from the `Kernel` and a `Schedule` chosen here.
//! Changing schedule is one line in this file, not regeneration followed by a
//! llama.cpp build - this is the loop in which scan and reduction geometries
//! are worked out (`ScanStrategy`, `vector_width`, `SharedTree`).
//!
//! What it does not and must not claim to provide: comparison with a native
//! kernel. It measures only RIR against RIR. Promotion is decided by
//! `scripts/test-rir.sh`, which goes through ggml and executes both paths on the
//! same shapes.
//!
//! It runs only on request - `RIR_TIME=1` - because a performance number from a
//! loaded machine does not belong in a correctness lane:
//!
//! ```text
//! RIR_TIME=1 cargo test -p rir-runtime --test device_timing -- --nocapture
//! ```

use rir_emit::{emit_manifest, emit_vulkan};
use rir_lower::Schedule;
use rir_runtime::any::{AnyGpu, AnyPipeline, Backend};
use rir_runtime::{
    Arg, Gpu, Manifest, ManifestReader, RuntimeError, Values, compile_glsl, is_unavailable,
};

/// Uncounted runs followed by counted runs. Enough to escape first-submission
/// noise without making this test a long wait.
const WARMUP: u32 = 3;
const ITERS: u32 = 50;

fn enabled() -> bool {
    match std::env::var_os("RIR_TIME") {
        Some(v) => v != "0",
        None => {
            eprintln!("timing skipped: set RIR_TIME=1 to run it");
            false
        }
    }
}

/// Emits, compiles, and builds a kernel pipeline for a given schedule. `None`
/// means this machine cannot measure (no GPU or compiler); a panic means the
/// kernel/schedule pair is broken, which is a real failure.
fn build<'g>(
    gpu: &'g Gpu,
    kernel: &rir_core::ValidatedKernel,
    schedule: Schedule,
    tag: &str,
) -> Option<rir_runtime::Pipeline<'g>> {
    let lk = rir_lower::lower(kernel, schedule).unwrap_or_else(|e| panic!("{tag} : lowering {e}"));
    let glsl = emit_vulkan(&lk).unwrap_or_else(|e| panic!("{tag}: emission {e}"));
    let manifest = Manifest::from_json(&emit_manifest(&lk, None))
        .unwrap_or_else(|e| panic!("{tag} : manifest {e}"));

    // `compile_glsl` takes a path (as the AOT pipeline calls it); the parent
    // directory names the temporary file and therefore carries the tag.
    let dir = std::env::temp_dir().join(format!("rir-time-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("temporary directory");
    let comp = dir.join("kernel.comp");
    std::fs::write(&comp, glsl).expect("writing shader");
    let spirv = compile_glsl(&comp);
    let _ = std::fs::remove_dir_all(&dir);

    let spirv = match spirv {
        Ok(s) => s,
        Err(e) if is_unavailable(&e) => {
            eprintln!("{tag} skipped: {e}");
            return None;
        }
        Err(e) => panic!("{tag} : compilation {e}"),
    };
    match gpu.build(&manifest, &spirv) {
        Ok(p) => Some(p),
        Err(e) if is_unavailable(&e) => {
            eprintln!("{tag} skipped: {e}");
            None
        }
        Err(e) => panic!("{tag} : pipeline {e}"),
    }
}

fn gpu() -> Option<Gpu> {
    match Gpu::open() {
        Ok(g) => {
            eprintln!("timing on '{}'", g.name());
            Some(g)
        }
        Err(e) if is_unavailable(&e) => {
            eprintln!("timing skipped: {e}");
            None
        }
        Err(e) => panic!("opening device: {e}"),
    }
}

/// One table row. Time is for one dispatch, excluding upload and readback,
/// exactly what `Session` separates.
fn report(tag: &str, shape: &str, groups: [u32; 3], t: Result<std::time::Duration, RuntimeError>) {
    match t {
        Ok(d) => println!(
            "{tag:<28} {shape:<26} {:>4}×{:<4} {:>10.1} µs",
            groups[0],
            groups[1] * groups[2],
            d.as_secs_f64() * 1e6
        ),
        Err(e) => println!("{tag:<28} {shape:<26} {:>19}", format!("failure: {e}")),
    }
}

/// The same, on any backend the build can drive.
///
/// The kernel is still **emitted in process** from a `Schedule` written in this
/// file - that is what makes this the short loop rather than the lane - and what
/// changes with a second backend is only which emitter prints it and which
/// device compiles it.
///
/// The emit-lower-compile-build sequence itself is `rir_sweep::measure::build`:
/// the offline search runs *this* loop, so the two
/// share the function rather than each keeping a copy of the parameter-header
/// derivation. What stays here is the only thing that differs - a
/// test turns an unavailable device into a skip and a broken pair into a panic
/// carrying its tag, where a tool returns a typed error.
fn build_any<'g>(
    gpu: &'g AnyGpu,
    target: Backend,
    kernel: &rir_core::ValidatedKernel,
    schedule: Schedule,
    tag: &str,
) -> Option<AnyPipeline<'g>> {
    match rir_sweep::measure::build(gpu, target, kernel, schedule) {
        Ok(built) => Some(built.pipeline),
        Err(rir_sweep::SweepError::Runtime(e)) if is_unavailable(&e) => {
            eprintln!("{tag} skipped: {e}");
            None
        }
        Err(e) => panic!("{tag}: {e}"),
    }
}

/// The constant block a flattened dispatch reads: the
/// number of points, then one (divisor, multiplier, shift) triple per divisor.
///
/// Written here rather than derived from the manifest because this file *builds*
/// the schedule it times: the widths are the question, so they are an argument.
/// A variant that is not flattened declares none of these names, and a value
/// nobody declared is never written into the block.
fn flat_values(values: &mut Values, extents: &[usize], per: &[usize]) {
    flat_values_with(values, extents, per, true)
}

/// The same block, with or without its reciprocals.
///
/// A linear-addressing variant declares the total alone:
/// it divides by nothing, so the layout carries no triple and a value written
/// under a name the shader never declared is a value nothing reads.
fn flat_values_with(values: &mut Values, extents: &[usize], per: &[usize], decompose: bool) {
    let d: Vec<u32> = extents
        .iter()
        .zip(per)
        .map(|(n, p)| n.div_ceil(*p) as u32)
        .collect();
    values.u32("rir_flat_total", d.iter().product());
    if !decompose {
        return;
    }
    for (i, &di) in d.iter().enumerate().take(d.len() - 1) {
        let (mp, sh) = rir_lower::fastdiv_magic(di);
        values
            .u32(&format!("rir_flat{i}_div"), di)
            .u32(&format!("rir_flat{i}_mp"), mp)
            .u32(&format!("rir_flat{i}_sh"), sh);
    }
}

/// The elementwise band, timed on **every backend this build can drive, in one
/// session**.
///
/// It compares RIR to RIR and decides nothing: promotion is `scripts/test-rir.sh`,
/// which goes through ggml and runs both paths on the same shapes.
///
/// **The two columns are not a ratio, and must not be read as one.** One may hope
/// this machine can compare a Vulkan and a CUDA number on the same silicon; that
/// hope is about the *lane*'s ratios against each backend's own native kernel,
/// and it does not survive down here, because the two harnesses do not put the
/// data in the same place. The Vulkan half allocates host-visible memory - the
/// deliberate choice of a runtime that validates results rather than throughput
/// (`device/mod.rs`) - while the CUDA half calls `cudaMalloc`, which is device
/// memory. The measured gap on this box is three orders of magnitude, and it is
/// the bus, not the kernel.
///
/// What each column *does* say, on its own, is the useful thing: how one
/// backend's time moves shape to shape. A kernel whose cost stops following its
/// element count is a kernel that became launch-dominated, and that crossover is
/// exactly what a promotion needs to know before it claims anything about the band.
#[test]
fn elementwise_band_cost_on_every_backend() {
    if !enabled() {
        return;
    }
    // One row per (kernel, shape), the backends side by side - read each column
    // down, not across (see above).
    println!(
        "{:<24} {:<26} {:>12} {:>12}",
        "kernel", "shape", "vulkan µs", "cuda µs"
    );
    for member in rir_kernels::elementwise::variants() {
        let name = member.kernel_name();
        // The F32 members of the band that CUDA schedules. The F16 twins share
        // their lowering and would time a memory format, not a kernel.
        if !matches!(name.as_str(), "add" | "mul" | "scale") {
            continue;
        }
        let kernel = rir_kernels::elementwise::build(member).unwrap();
        for &(n_col, n_row) in &[(4096usize, 16usize), (1024, 64), (128, 256), (33, 256)] {
            let len = n_col * n_row;
            let nb = [4usize, 4 * n_col, 4 * len, 4 * len];
            let mut t = [f64::NAN; 2];
            for (i, target) in [Backend::Vulkan, Backend::Cuda].into_iter().enumerate() {
                if !Backend::available().contains(&target) {
                    continue;
                }
                let tag = format!("{name}/vec4/{}", target.name());
                let Ok(gpu) = AnyGpu::open(target) else {
                    continue;
                };
                let Some(pipe) = build_any(
                    &gpu,
                    target,
                    &kernel,
                    Schedule::gpu_grid_vec4(
                        match target {
                            Backend::Vulkan => rir_lower::GpuBackend::Vulkan,
                            Backend::Cuda => rir_lower::GpuBackend::Cuda,
                        },
                        [256, 1, 1],
                    ),
                    &tag,
                ) else {
                    continue;
                };
                let a = vec![1f32; len];
                let b = vec![0.5f32; len];
                let mut dst = vec![0f32; len];
                let mut values = Values::new();
                values
                    .u32("n_col", n_col as u32)
                    .u32("n_row", n_row as u32)
                    .u32("n_plane", 1)
                    .u32("n_batch", 1)
                    .strides("a", &nb)
                    .strides("dst", &nb);
                for p in kernel.params() {
                    values.f32(&p.name, 0.75);
                }
                let mut args: Vec<Arg> = vec![Arg::input(&a)];
                if kernel.args().len() == 3 {
                    values.strides("b", &nb);
                    args.push(Arg::input(&b));
                }
                args.push(Arg::output(&mut dst));
                let session = pipe
                    .prepare(&args, &values)
                    .unwrap_or_else(|e| panic!("preparing {tag}: {e}"));
                t[i] = match session.time(WARMUP, ITERS) {
                    Ok(d) => d.as_secs_f64() * 1e6,
                    Err(e) => {
                        println!("{tag:<24} failure: {e}");
                        continue;
                    }
                };
            }
            println!(
                "{name:<24} {:<26} {:>12.1} {:>12.1}",
                format!("n_col={n_col} n_row={n_row}"),
                t[0],
                t[1]
            );
        }
    }
}

/// `cumsum`: the lane measured its cost as **linear in row
/// length** - this test replays it on the shader alone, without ggml noise, and
/// this is the measurement a blocked `ScanStrategy` must bend. Doubling `n_col`
/// at constant row count must stop doubling time.
#[test]
fn cumsum_scan_cost_against_row_length() {
    if !enabled() {
        return;
    }
    let Some(gpu) = gpu() else { return };
    let kernel = rir_kernels::cumsum::build().unwrap();
    // Both lowerings of the same kernel on the same shapes: this is the
    // comparison level 1 can make and the promotion lane does not.
    for (tag, schedule) in [
        ("cumsum/grid64", Schedule::vulkan_grid([64, 1, 1])),
        ("cumsum/blocked32", Schedule::vulkan_blocked_scan()),
        (
            "cumsum/shared256",
            rir_lower::schedule::bench::vulkan_shared_scan(),
        ),
        // The tiled scan keeps its predecessor opposite it here even though it
        // replaced it in the production table: they
        // differ by an access plan with *identical* serial depth, exactly the
        // difference isolated by this harness and no longer visible to a lane
        // measuring only one variant per shape after the table changes.
        ("cumsum/tiled256", Schedule::vulkan_tiled_scan(256, 16)),
    ] {
        let Some(pipe) = build(&gpu, &kernel, schedule, tag) else {
            return;
        };

        for &(n_col, n_row) in &[
            (4096usize, 1usize),
            (8192, 1),
            (16384, 1),
            (32768, 1),
            (2_000_000, 1),
            // Multi-row shapes: enough rows to occupy the GPU, unlike the
            // single-row column. This is where both strategies cross and where
            // the need for one or two variants is decided.
            (2048, 320),
            (2048, 16),
            (20000, 40),
            (512, 1000),
        ] {
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
            report(
                tag,
                &format!("n_col={n_col} n_row={n_row}"),
                session.groups(),
                session.time(WARMUP, ITERS),
            );
        }
    }
}

/// Sweep deciding the **shape rule**: at what size the scan strategies cross.
/// The lane table says "a few hundred rows"; a registry rule cannot stop there, so
/// this test sweeps the (n_col, n_row) plane around the crossover and prints the
/// blocked/sequential ratio shape by shape.
#[test]
fn cumsum_strategy_crossover_sweep() {
    if !enabled() {
        return;
    }
    let Some(gpu) = gpu() else { return };
    let kernel = rir_kernels::cumsum::build().unwrap();
    let Some(serial) = build(&gpu, &kernel, Schedule::vulkan_grid([64, 1, 1]), "grid64") else {
        return;
    };
    let Some(blocked) = build(&gpu, &kernel, Schedule::vulkan_blocked_scan(), "blocked32") else {
        return;
    };

    println!(
        "{:<12} {:>8} {:>12} {:>12} {:>8}",
        "n_col", "n_row", "grid64 µs", "blocked µs", "ratio"
    );
    for &n_col in &[16usize, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192] {
        for &n_row in &[1usize, 8, 32, 64, 128, 256, 512, 1024] {
            let len = n_col * n_row;
            if len > 16 * 1024 * 1024 {
                continue;
            }
            let x = vec![1f32; len];
            let nb = [4usize, 4 * n_col, 4 * n_col * n_row, 4 * n_col * n_row];
            let mut t = [0f64; 2];
            for (i, pipe) in [&serial, &blocked].into_iter().enumerate() {
                let mut y = vec![0f32; len];
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
                t[i] = match session.time(WARMUP, ITERS) {
                    Ok(d) => d.as_secs_f64() * 1e6,
                    Err(e) => {
                        println!("{n_col:<12} {n_row:>8}   failure: {e}");
                        continue;
                    }
                };
            }
            println!(
                "{n_col:<12} {n_row:>8} {:>12.1} {:>12.1} {:>8.2}",
                t[0],
                t[1],
                t[1] / t[0]
            );
        }
    }
}

/// `sum_rows_<format>`: isolated **format-lowering** cost.
///
/// This is the measurement the block-wise loader requested: a quantized row
/// has only one sum, so everything except
/// payload reading is decoding - block index, F16 scale, and for a K quant the
/// packed 6-bit subscale. Read the table per format: `q8_0` has only one scale
/// per block of 32, `q4_K` has the `get_scale_min_k4` pair per sub-block of 32,
/// exposing segment sharing.
///
/// Like the rest of this file, it compares nothing with the native kernel: two
/// executions compare two RIR lowerings of the same kernel on the same bytes.
#[test]
fn sum_rows_quant_decode_cost() {
    if !enabled() {
        return;
    }
    let Some(gpu) = gpu() else { return };
    for format in rir_kernels::sum_rows_quant::variants() {
        let name = rir_kernels::sum_rows_quant::kernel_name(format);
        let kernel = rir_kernels::sum_rows_quant::build(format).unwrap();
        let tag = format!("{name}/subgroup");
        let Some(pipe) = build(&gpu, &kernel, Schedule::vulkan_subgroup(), &tag) else {
            return;
        };
        let d = format.desc();
        let (be, bb) = (d.block_elements as usize, d.block_bytes as usize);

        // Many long rows: the strip where the family is actually called and the
        // only one where occupied lane count does not dominate measurement.
        for &(blocks, n_row) in &[(16usize, 256usize), (64, 256), (16, 4096)] {
            let (n_col, stride) = (blocks * be, blocks * bb);
            let raw = vec![0x11u8; stride * n_row];
            let mut y = vec![0f32; n_row];
            let mut values = Values::new();
            values
                .u32("n_row", n_row as u32)
                .u32("n_col", n_col as u32)
                .strides("x", &[bb, stride])
                .strides("y", &[4usize]);
            let session = pipe
                .prepare(&[Arg::input(&raw), Arg::output(&mut y)], &values)
                .expect("preparing quantized sum_rows");
            report(
                &tag,
                &format!("n_col={n_col} n_row={n_row}"),
                session.groups(),
                session.time(WARMUP, ITERS),
            );
        }
    }
}

/// `l2_norm_back`: the promoted pair. It is the control - if its number changes
/// by a factor, the machine changed, not the kernel.
#[test]
fn l2_norm_back_reduction_cost() {
    if !enabled() {
        return;
    }
    let Some(gpu) = gpu() else { return };
    let kernel = rir_kernels::l2_norm_back::build().unwrap();
    let tag = "l2_norm_back/subgroup";
    let Some(pipe) = build(&gpu, &kernel, Schedule::vulkan_subgroup(), tag) else {
        return;
    };

    for &(n_col, n_row, n_plane, n_batch) in &[
        (128usize, 1usize, 1usize, 1usize),
        (128, 16, 16, 1),
        (1024, 16, 16, 1),
    ] {
        let len = n_col * n_row * n_plane * n_batch;
        let (dz, x) = (vec![1f32; len], vec![0.5f32; len]);
        let mut dx = vec![0f32; len];
        let nb = [
            4usize,
            4 * n_col,
            4 * n_col * n_row,
            4 * n_col * n_row * n_plane,
        ];
        let mut values = Values::new();
        values
            .f32("eps", 1e-6)
            .u32("n_row", n_row as u32)
            .u32("n_plane", n_plane as u32)
            .u32("n_batch", n_batch as u32)
            .u32("n_col", n_col as u32)
            .strides("dz", &nb)
            .strides("x", &nb)
            .strides("dx", &nb);

        let session = pipe
            .prepare(
                &[Arg::input(&dz), Arg::input(&x), Arg::output(&mut dx)],
                &values,
            )
            .expect("preparing l2_norm_back");
        report(
            tag,
            &format!("ne=[{n_col},{n_row},{n_plane},{n_batch}]"),
            session.groups(),
            session.time(WARMUP, ITERS),
        );
    }
}

/// The three reduction geometries `rms_norm_back` can take on CUDA, on the
/// shapes the census designated.
///
/// This is the arbitration the lane cannot make: `scripts/test-rir.sh`
/// says whether the *selected* variant beats the native kernel, not which of the
/// three would have. Here the schedule is one line, so the row count is the
/// question and not the rebuild.
///
/// It compares RIR to RIR and decides no promotion. What it decides is which
/// lowering the shape rule should send `[1024,16,1,1]` to - the shape that makes
/// 60 % of the `RMS_NORM_BACK` nodes of both censused graphs.
#[test]
fn rms_norm_back_reduction_geometry_on_cuda() {
    if !enabled() {
        return;
    }
    if !Backend::available().contains(&Backend::Cuda) {
        eprintln!("rms_norm_back geometry: no CUDA device");
        return;
    }
    let Ok(gpu) = AnyGpu::open(Backend::Cuda) else {
        return;
    };
    let kernel = rir_kernels::rms_norm_back::build().unwrap();
    let cuda = rir_lower::GpuBackend::Cuda;
    for (tag, schedule) in [
        ("rms_norm_back/subgroup", Schedule::gpu_subgroup(cuda)),
        ("rms_norm_back/shared256", Schedule::gpu_shared_reduce(cuda)),
        // The 1 024-lane tree is not in the production table and never was:
        // this row is what kept it out. It is built from
        // the generic constructor with the block overridden, so removing the
        // variant did not remove the measurement that refused it.
        (
            "rms_norm_back/shared1024",
            Schedule::gpu_shared_reduce(cuda).with_block([1024, 1, 1]),
        ),
    ] {
        let Some(pipe) = build_any(&gpu, Backend::Cuda, &kernel, schedule, tag) else {
            continue;
        };
        // The census shapes, plus the two the lane runs either side of the
        // native's `ncols` switch.
        for &(n_col, n_row, n_plane, n_batch) in &[
            (1024usize, 16usize, 1usize, 1usize),
            (640, 16, 1, 1),
            (256, 8, 16, 1),
            (128, 16, 16, 1),
        ] {
            let len = n_col * n_row * n_plane * n_batch;
            let (dz, x) = (vec![1f32; len], vec![0.5f32; len]);
            let mut dx = vec![0f32; len];
            let nb = [
                4usize,
                4 * n_col,
                4 * n_col * n_row,
                4 * n_col * n_row * n_plane,
            ];
            let mut values = Values::new();
            values
                .f32("eps", 1e-6)
                .u32("n_row", n_row as u32)
                .u32("n_plane", n_plane as u32)
                .u32("n_batch", n_batch as u32)
                .u32("n_col", n_col as u32)
                .strides("dz", &nb)
                .strides("x", &nb)
                .strides("dx", &nb);
            let session = pipe
                .prepare(
                    &[Arg::input(&dz), Arg::input(&x), Arg::output(&mut dx)],
                    &values,
                )
                .unwrap_or_else(|e| panic!("preparing {tag}: {e}"));
            report(
                tag,
                &format!("ne=[{n_col},{n_row},{n_plane},{n_batch}]"),
                session.groups(),
                session.time(WARMUP, ITERS),
            );
        }
    }
}

/// The **width** of the flattened dispatch, arbitrated RIR against RIR on the
/// three shapes the lane runs for `UNARY`.
///
/// The question this answers is the one the lane could not: its ratio is against
/// a native kernel, so a flattened variant coming in at 1.20 says "slower than
/// `unary_op_kernel`" without saying *which* half of the lowering costs it - the
/// decomposition, or a vector width that divides the grid by four. Four elements
/// per thread is what was measured on Metal and Vulkan against natives that walk
/// a row per threadgroup; the CUDA native walks **one element per thread** and
/// is already flat, so the width buys no address algebra here and spends the
/// occupancy a small tensor has left.
///
/// Same three geometries the lane times, so the two tables can be read together.
#[test]
fn unary_flat_width_on_cuda() {
    if !enabled() {
        return;
    }
    if !Backend::available().contains(&Backend::Cuda) {
        eprintln!("unary flat width: no CUDA device");
        return;
    }
    let Ok(gpu) = AnyGpu::open(Backend::Cuda) else {
        return;
    };
    let kernel = rir_kernels::unary::build(rir_kernels::unary::Unary::Silu).unwrap();
    let cuda = rir_lower::GpuBackend::Cuda;
    for (tag, schedule, width) in [
        (
            "unary_silu/grid_vec4",
            Schedule::gpu_grid_vec4(cuda, [256, 1, 1]),
            4usize,
        ),
        (
            "unary_silu/flat_w4",
            Schedule::gpu_grid_flat(cuda, [256, 1, 1], 4),
            4,
        ),
        (
            "unary_silu/flat_w1",
            Schedule::gpu_grid_flat(cuda, [256, 1, 1], 1),
            1,
        ),
    ] {
        let Some(pipe) = build_any(&gpu, Backend::Cuda, &kernel, schedule, tag) else {
            continue;
        };
        for &(n_col, n_row, n_plane, n_batch) in &[
            (4096usize, 16usize, 1usize, 1usize),
            (1024, 16, 1, 1),
            (128, 16, 16, 1),
        ] {
            let len = n_col * n_row * n_plane * n_batch;
            let x = vec![0.5f32; len];
            let mut y = vec![0f32; len];
            let nb = [
                4usize,
                4 * n_col,
                4 * n_col * n_row,
                4 * n_col * n_row * n_plane,
            ];
            let mut values = Values::new();
            values
                .u32("n_col", n_col as u32)
                .u32("n_row", n_row as u32)
                .u32("n_plane", n_plane as u32)
                .u32("n_batch", n_batch as u32)
                .strides("x", &nb)
                .strides("dst", &nb);
            flat_values(
                &mut values,
                &[n_col, n_row, n_plane, n_batch],
                &[width, 1, 1, 1],
            );
            let session = pipe
                .prepare(&[Arg::input(&x), Arg::output(&mut y)], &values)
                .unwrap_or_else(|e| panic!("preparing {tag}: {e}"));
            report(
                tag,
                &format!("ne=[{n_col},{n_row},{n_plane},{n_batch}]"),
                session.groups(),
                session.time(WARMUP, ITERS),
            );
        }
    }
}

/// The **address**, arbitrated the way the width was.
///
/// `unary_flat_width_on_cuda` above settled the dispatch geometry and left one
/// thing behind: the flattened
/// grid removed the idle lanes and kept the four stride products per element,
/// against a native kernel that indexes `base + i · elem_size`. This times the
/// lowering that removes them - one multiplication, no decomposition - against
/// the decomposing variant it specializes, over the product the item asks for:
/// `block ∈ {128, 256, 512}` and `vector_width ∈ {1, 2, 4}`.
///
/// It compares RIR to RIR and decides nothing. What it can settle is which
/// geometry the table should carry, and whether the address is worth a claim at
/// all: a linear column that does not beat its decomposing twin is an item to
/// close with a number, not a variant to promote.
///
/// The shapes are the ones the width sweep used, so the two tables read
/// together - and all three have a contiguous extent divisible by four, which is
/// what the claim requires and what a dispatch site would check.
///
/// It runs on **every backend the build can drive**, unlike the width sweep
/// above. The claim is not a CUDA one - a stride sum is a stride sum in three
/// shading languages - and a machine with one backend should still be able to
/// say what removing it is worth. What stays CUDA's is the *decision*: the
/// production table flattens on `FLAT_TARGETS` alone, so a Vulkan column here is
/// evidence about the lever and not about a variant anyone ships.
#[test]
fn unary_flat_address() {
    if !enabled() {
        return;
    }
    let kernel = rir_kernels::unary::build(rir_kernels::unary::Unary::Silu).unwrap();
    for target in Backend::available() {
        let Ok(gpu) = AnyGpu::open(target) else {
            continue;
        };
        let g = match target {
            Backend::Vulkan => rir_lower::GpuBackend::Vulkan,
            Backend::Cuda => rir_lower::GpuBackend::Cuda,
        };
        println!("--- {} ---", target.name());
        let mut plan: Vec<(String, Schedule, usize, bool)> = Vec::new();
        // The baseline first, at the geometry the table ships, so every linear
        // row below is read against the variant it would specialize.
        plan.push((
            "unary_silu/flat_w1 (base)".to_string(),
            Schedule::gpu_grid_flat(g, [256, 1, 1], 1),
            1,
            true,
        ));
        for block in [128u32, 256, 512] {
            for width in [1usize, 2, 4] {
                plan.push((
                    format!("unary_silu/linear b{block} w{width}"),
                    Schedule::gpu_grid_flat_linear(g, [block, 1, 1], width as u32),
                    width,
                    false,
                ));
            }
        }
        run_address_plan(&gpu, target, &kernel, plan);
    }
}

/// One column of `unary_flat_address`: every geometry of the plan, on the three
/// shapes the width sweep used.
fn run_address_plan(
    gpu: &AnyGpu,
    target: Backend,
    kernel: &rir_core::ValidatedKernel,
    plan: Vec<(String, Schedule, usize, bool)>,
) {
    for (tag, schedule, width, decompose) in plan {
        let Some(pipe) = build_any(gpu, target, kernel, schedule, &tag) else {
            continue;
        };
        for &(n_col, n_row, n_plane, n_batch) in &[
            (4096usize, 16usize, 1usize, 1usize),
            (1024, 16, 1, 1),
            (128, 16, 16, 1),
        ] {
            let len = n_col * n_row * n_plane * n_batch;
            let x = vec![0.5f32; len];
            let mut y = vec![0f32; len];
            let nb = [
                4usize,
                4 * n_col,
                4 * n_col * n_row,
                4 * n_col * n_row * n_plane,
            ];
            let mut values = Values::new();
            values
                .u32("n_col", n_col as u32)
                .u32("n_row", n_row as u32)
                .u32("n_plane", n_plane as u32)
                .u32("n_batch", n_batch as u32)
                .strides("x", &nb)
                .strides("dst", &nb);
            flat_values_with(
                &mut values,
                &[n_col, n_row, n_plane, n_batch],
                &[width, 1, 1, 1],
                decompose,
            );
            let session = pipe
                .prepare(&[Arg::input(&x), Arg::output(&mut y)], &values)
                .unwrap_or_else(|e| panic!("preparing {tag}: {e}"));
            report(
                &tag,
                &format!("ne=[{n_col},{n_row},{n_plane},{n_batch}]"),
                session.groups(),
                session.time(WARMUP, ITERS),
            );
        }
    }
}

/// Two more reduction levers, on the shapes the width sweep left open.
///
/// `rms_norm_back_reduction_geometry_on_cuda` above settled a *width* and left a
/// sentence: the two shapes `shared_reduce` does not claim - `[256,8,16,1]` and
/// `[128,16,16,1]`, 128 and 256 rows - stay with the 32-lane fallback "against a
/// native that flattens `nrows` into a single grid dimension and addresses
/// through one `int64`, and reduces in two stages". Two chantiers, and neither
/// is a block size. This times both:
///
/// - `hier` is the two-stage tree against the flat one, at the same 256 lanes,
///   so what the two columns differ by is the topology and nothing else;
/// - `flat_rows` is the same 32-lane reduction over a linear row space, so what
///   it differs from `subgroup` by is the dispatch geometry and nothing else.
///
/// It runs on **every backend the build can drive**, unlike its neighbour, and
/// that is deliberate: the barrier count and the shared-storage size are
/// properties of the lowering, not of CUDA, and a machine with one backend
/// should still be able to say whether two barriers beat eight. Read each column
/// down and never across (`elementwise_band_cost_on_every_backend`).
#[test]
fn rms_norm_back_reduction_stages() {
    if !enabled() {
        return;
    }
    let kernel = rir_kernels::rms_norm_back::build().unwrap();
    for target in Backend::available() {
        let Ok(gpu) = AnyGpu::open(target) else {
            continue;
        };
        let g = match target {
            Backend::Vulkan => rir_lower::GpuBackend::Vulkan,
            Backend::Cuda => rir_lower::GpuBackend::Cuda,
        };
        println!("--- {} ---", target.name());
        for (tag, schedule, flat) in [
            ("rms_norm_back/subgroup", Schedule::gpu_subgroup(g), false),
            (
                "rms_norm_back/flat_rows",
                Schedule::gpu_subgroup_flat_rows(g),
                true,
            ),
            (
                "rms_norm_back/shared256",
                Schedule::gpu_shared_reduce(g),
                false,
            ),
            ("rms_norm_back/hier256", Schedule::gpu_hier_reduce(g), false),
        ] {
            let Some(pipe) = build_any(&gpu, target, &kernel, schedule, tag) else {
                continue;
            };
            for &(n_col, n_row, n_plane, n_batch) in &[
                (1024usize, 16usize, 1usize, 1usize),
                (640, 16, 1, 1),
                (256, 8, 16, 1),
                (128, 16, 16, 1),
            ] {
                let len = n_col * n_row * n_plane * n_batch;
                let (dz, x) = (vec![1f32; len], vec![0.5f32; len]);
                let mut dx = vec![0f32; len];
                let nb = [
                    4usize,
                    4 * n_col,
                    4 * n_col * n_row,
                    4 * n_col * n_row * n_plane,
                ];
                let mut values = Values::new();
                values
                    .f32("eps", 1e-6)
                    .u32("n_row", n_row as u32)
                    .u32("n_plane", n_plane as u32)
                    .u32("n_batch", n_batch as u32)
                    .u32("n_col", n_col as u32)
                    .strides("dz", &nb)
                    .strides("x", &nb)
                    .strides("dx", &nb);
                if flat {
                    // The row space, fastest axis first, one row per workgroup,
                    // so every `per` is one and the reduced axis is not in it.
                    flat_values(&mut values, &[n_row, n_plane, n_batch], &[1, 1, 1]);
                }
                let session = pipe
                    .prepare(
                        &[Arg::input(&dz), Arg::input(&x), Arg::output(&mut dx)],
                        &values,
                    )
                    .unwrap_or_else(|e| panic!("preparing {tag}: {e}"));
                report(
                    tag,
                    &format!("ne=[{n_col},{n_row},{n_plane},{n_batch}]"),
                    session.groups(),
                    session.time(WARMUP, ITERS),
                );
            }
        }
    }
}
