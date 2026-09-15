//! The multi-dispatch capability on a real device.
//!
//! Two questions, and they are not the same question.
//!
//! The first is **correctness**, and it runs in this lane unconditionally: three
//! dispatches with a scratch between them return the scan a single dispatch
//! returns. That is what makes the capability real rather than plausible - the
//! plan's strides, its barriers, its grid expressions and its scratch lifetimes
//! are all things that produce a *wrong number* when they are wrong, and this is
//! what reads that number.
//!
//! The second is **whether it is worth anything**, and it runs only under
//! `RIR_TIME=1`, like every other timing in this crate. This is not a promotion
//! result; it reports what the capability would buy if a client appeared.
//!
//! ```text
//! RIR_TIME=1 cargo test --release -p rir-runtime --test dispatch_plan -- --nocapture
//! ```

use rir_core::Backend;
use rir_kernels::scan_plan;
use rir_lower::Schedule;
use rir_runtime::{
    Arg, Gpu, Manifest, ManifestReader, Plan, PlanArg, Values, compile_glsl, is_unavailable,
};

/// Tiles of the row the parity test scans. One row of `TILE · TILES` elements,
/// the shape the plan claims: a scan of one very long row.
const TILES: usize = 256;

/// Row lengths the timing walks, all whole numbers of tiles: 64 Ki, 256 Ki and
/// 2 Mi elements. The last is `cumsum_scan_cost_against_row_length`'s own
/// longest row, so the two tables can be read against each other.
const LENGTHS: [usize; 3] = [65_536, 262_144, 2_097_152];

fn row_of(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i % 11) as f32 - 5.0) * 0.125).collect()
}

fn row() -> Vec<f32> {
    row_of(TILES * scan_plan::TILE as usize)
}

/// The scan as f64, which is the reference both paths are compared to.
fn reference(x: &[f32]) -> Vec<f64> {
    let mut acc = 0f64;
    x.iter()
        .map(|v| {
            acc += f64::from(*v);
            acc
        })
        .collect()
}

fn gpu() -> Option<Gpu> {
    match Gpu::open() {
        Ok(g) => Some(g),
        Err(e) if is_unavailable(&e) => {
            eprintln!("skipped: {e}");
            None
        }
        Err(e) => panic!("opening device: {e}"),
    }
}

/// Emits one pass and compiles it, returning what `Plan::build` binds.
fn pass(kernel: &rir_core::ValidatedKernel, schedule: Schedule) -> Option<(Manifest, Vec<u32>)> {
    let lowered = rir_lower::lower(kernel, schedule).expect("lowering a plan pass");
    let glsl = rir_emit::emit_vulkan(&lowered).expect("emitting a plan pass");
    let manifest = Manifest::from_json(&rir_emit::emit_manifest(&lowered, None)).expect("manifest");
    let dir = std::env::temp_dir().join(format!(
        "rir-plan-{}-{}",
        std::process::id(),
        rir_emit::artifact_name(&lowered)
    ));
    std::fs::create_dir_all(&dir).expect("temporary directory");
    let comp = dir.join("kernel.comp");
    std::fs::write(&comp, glsl).expect("writing the shader");
    let spirv = compile_glsl(&comp);
    let _ = std::fs::remove_dir_all(&dir);
    match spirv {
        Ok(s) => Some((manifest, s)),
        Err(e) if is_unavailable(&e) => {
            eprintln!("skipped: {e}");
            None
        }
        Err(e) => panic!("compiling a plan pass: {e}"),
    }
}

/// The artifact names of the three passes, which the plan needs and only the
/// emitter knows.
fn artifacts() -> [String; 3] {
    let kernels = scan_plan::kernels().expect("the plan's kernels");
    let schedules = scan_plan::schedules(rir_lower::GpuBackend::Vulkan);
    let mut names = Vec::new();
    for (k, s) in kernels.iter().zip(schedules) {
        let lowered = rir_lower::lower(k, s).expect("lowering");
        names.push(rir_emit::artifact_name(&lowered));
    }
    [names[0].clone(), names[1].clone(), names[2].clone()]
}

/// Three dispatches and a scratch return what one dispatch returns.
///
/// The tolerance is the plan's own: pass 0 reduces each tile with a subgroup
/// tree, so the additions inside a tile are regrouped - `Deterministic` and not
/// `ExactOrder`, which `scan_plan` states. What must **not** differ is anything
/// else: an off-by-one tile offset, a missing barrier or a stride read as bytes
/// where it is elements all produce errors orders of magnitude larger than
/// this bound, on most elements rather than the last ones.
#[test]
fn the_plan_computes_the_scan_it_replaces() {
    let Some(gpu) = gpu() else { return };
    let kernels = scan_plan::kernels().expect("the plan's kernels");
    let schedules = scan_plan::schedules(rir_lower::GpuBackend::Vulkan);
    let mut built = Vec::new();
    for (k, s) in kernels.iter().zip(schedules) {
        let Some(p) = pass(k, s) else { return };
        built.push(p);
    }
    let refs: Vec<(&Manifest, &[u32])> = built.iter().map(|(m, s)| (m, s.as_slice())).collect();
    let valid_plan = scan_plan::plan(Backend::Vulkan, artifacts());
    let mut wrong_artifact = valid_plan.clone();
    wrong_artifact.passes[0].artifact = "not_the_emitted_kernel".to_string();
    assert!(matches!(
        Plan::build(&gpu, wrong_artifact, &refs),
        Err(rir_runtime::RuntimeError::BadManifest(_))
    ));
    let plan = Plan::build(&gpu, valid_plan, &refs).expect("building the plan");

    let x = row();
    let mut y = vec![0f32; x.len()];
    let n_col = x.len() as u64;
    let axes = |name: &str| match name {
        "col" => Some(n_col),
        "row" | "plane" | "batch" => Some(1),
        _ => None,
    };
    // The budget the plan publishes, checked against what it is: one element per
    // tile in each of the two live arrays.
    assert_eq!(
        plan.plan().peak_scratch_bytes(&axes).unwrap(),
        2 * TILES as u64 * 4
    );

    let wrong_y = vec![0f32; x.len()];
    let wrong_args = vec![PlanArg::input("x", &x), PlanArg::input("y", &wrong_y)];
    assert!(matches!(
        plan.prepare(&wrong_args, &axes),
        Err(rir_runtime::RuntimeError::AccessMismatch { .. })
    ));

    let mut args = vec![PlanArg::input("x", &x), PlanArg::output("y", &mut y)];
    let session = plan.prepare(&args, &axes).expect("preparing the plan");
    session.dispatch().expect("running the plan");
    session.read_outputs(&mut args).expect("reading back");
    drop(args);

    let expect = reference(&x);
    for (i, (got, want)) in y.iter().zip(&expect).enumerate() {
        let bound = 1e-4 * (1.0 + want.abs());
        assert!(
            (f64::from(*got) - want).abs() <= bound,
            "element {i}: plan {got} against the scan {want}"
        );
    }
}

/// What the capability buys on the shape it exists for, against the three
/// single-dispatch scans of the production table.
///
/// It decides nothing (`RIR_TIME=1`, like every timing here) and it is not a
/// promotion: a multi-pass plan is promoted only for a **census client**, and
/// `CUMSUM` has none. What
/// this replaces is the sentence that would otherwise stand in for a number.
#[test]
fn a_long_row_against_the_single_dispatch_scans() {
    if std::env::var_os("RIR_TIME").is_none_or(|v| v == "0") {
        eprintln!("timing skipped: set RIR_TIME=1 to run it");
        return;
    }
    let Some(gpu) = gpu() else { return };

    // The three passes are built once: the plan's shapes are push constants, so
    // a longer row is another `prepare` and not another compilation.
    let kernels = scan_plan::kernels().expect("the plan's kernels");
    let schedules = scan_plan::schedules(rir_lower::GpuBackend::Vulkan);
    let mut built = Vec::new();
    for (k, s) in kernels.iter().zip(schedules) {
        let Some(p) = pass(k, s) else { return };
        built.push(p);
    }
    let refs: Vec<(&Manifest, &[u32])> = built.iter().map(|(m, s)| (m, s.as_slice())).collect();
    let plan = Plan::build(&gpu, scan_plan::plan(Backend::Vulkan, artifacts()), &refs)
        .expect("building the plan");

    let kernel = rir_kernels::cumsum::build().expect("cumsum");
    let singles: Vec<(&str, rir_runtime::Pipeline<'_>)> = [
        ("cumsum/grid64", Schedule::vulkan_grid([64, 1, 1])),
        ("cumsum/blocked32", Schedule::vulkan_blocked_scan()),
        ("cumsum/tiled256", Schedule::vulkan_tiled_scan(256, 16)),
    ]
    .into_iter()
    .filter_map(|(tag, schedule)| {
        let (manifest, spirv) = pass(&kernel, schedule)?;
        match gpu.build(&manifest, &spirv) {
            Ok(p) => Some((tag, p)),
            Err(e) if is_unavailable(&e) => None,
            Err(e) => panic!("{tag}: {e}"),
        }
    })
    .collect();

    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>12} {:>12}",
        "row length", "tiles", "plan(3) µs", "grid64 µs", "blocked32 µs", "tiled256 µs"
    );
    for n_col in LENGTHS {
        let x = row_of(n_col);
        let axes = |name: &str| match name {
            "col" => Some(n_col as u64),
            "row" | "plane" | "batch" => Some(1),
            _ => None,
        };
        let mut y = vec![0f32; n_col];
        let plan_us = {
            let args = vec![PlanArg::input("x", &x), PlanArg::output("y", &mut y)];
            let session = plan.prepare(&args, &axes).expect("preparing the plan");
            session.time(1, 10).expect("timing the plan").as_secs_f64() * 1e6
        };

        let mut row = format!(
            "{:<20} {:>10} {:>12.1}",
            n_col,
            n_col / scan_plan::TILE as usize,
            plan_us
        );
        for (_, pipeline) in &singles {
            let mut out = vec![0f32; n_col];
            let mut values = Values::new();
            values
                .u32("n_col", n_col as u32)
                .u32("n_row", 1)
                .u32("n_plane", 1)
                .u32("n_batch", 1)
                .strides("x", &[4, 4 * n_col, 4 * n_col, 4 * n_col])
                .strides("y", &[4, 4 * n_col, 4 * n_col, 4 * n_col]);
            let session = pipeline
                .prepare(&[Arg::input(&x), Arg::output(&mut out)], &values)
                .expect("preparing a single dispatch");
            match session.time(1, 10) {
                Ok(d) => row.push_str(&format!(" {:>12.1}", d.as_secs_f64() * 1e6)),
                Err(e) => row.push_str(&format!(" {:>12}", format!("{e}"))),
            }
        }
        println!("{row}");
    }
}
