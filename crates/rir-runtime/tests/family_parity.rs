//! Device parity, **derived** from the registry instead of written per kernel
//!
//! `device_parity.rs` holds one hand-written test per kernel it covers, and each
//! one repeats the same six steps: fill the inputs, build the views and the
//! strides, run the oracle, dispatch the shader, name the push constants,
//! compare with a tolerance. The cost of that repetition is not the lines - it
//! is that a kernel nobody wrote a test for has **no** device coverage at all,
//! and nothing says which ones those are. This file derives it for, among
//! others, RMS norm, the fourteen unary members, the five F16 band members and
//! the twelve quantized `out_prod` variants.
//!
//! What is derived, and from what:
//!
//! | datum | source |
//! |---|---|
//! | the kernels to cover | `rir_kernels::registry()`, so a new kernel is covered the day it is registered |
//! | the schedule | the registration's Vulkan **fallback** schedule (named variants keep their own tests) |
//! | each argument's `ne[]` | `LoopKernel::arg_axes` and the axis extents of the case |
//! | each argument's `nb[]` | packed, from the element type - block bytes for a quantized one |
//! | the input bytes | random floats, random halves, or `rir_core::random_block_bytes` for a format |
//! | the push constants | the manifest's own list: `n_<axis>`, `<arg>_nb<d>`, else a parameter |
//! | the shapes | one base, one per axis reduced to 1, one odd, one that exercises a fold |
//! | the tolerance | the kernel's own arithmetic: quantized, collective, F16 destination, or serial |
//!
//! The tolerance rule is deliberately not derived. Scans, quantized kernels,
//! and contractions do not have the same notion of parity. The four rules below
//! are the ones `device_parity.rs` documents, chosen by a property of the kernel
//! rather than by its name - a fifth kind of arithmetic will need a fifth rule,
//! and having to add it is the point.
//!
//! Like every device test here, this skips with an explanation when there is no
//! device, no `libvulkan`, no GLSL compiler and no CUDA toolkit.
//!
//! **Two backends**, and the derivation is what made
//! that cheap: everything above the device - the shapes, the geometry, the push
//! constants, the tolerance - is read off the Loop IR and the manifest, so the
//! second backend is a second *schedule* to look up and a second source file to
//! read, not a second harness. A kernel CUDA does not serve is not a gap here
//! either: it is a line in `FAMILY_REFUSED`, and this file prints it as such.

#![allow(clippy::unwrap_used)]
// Helpers outside a `#[test]` body, which is what `allow-unwrap-in-tests`
// covers. Same reasoning, said where the configuration cannot reach.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rir_core::{Access, DType};
use rir_kernels::registry;
use rir_lower::interp::{
    BoundArg, TensorView, TensorViewBytes, TensorViewBytesMut, TensorViewMut, f16_to_f32,
    f32_to_f16, run,
};
use rir_lower::{Backend, LoopKernel, Schedule};
use rir_runtime::any::{AnyGpu, AnyPipeline, Artifact, Backend as Target};
use rir_runtime::{Arg, Manifest, ManifestReader, Values, is_unavailable};

/// How far the two sides may differ, and why.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Tol {
    /// A quantized read: the shader decodes blocks with F16 scales, the oracle
    /// decodes the same bytes in F32. The floor is the scale's own resolution.
    Quantized,
    /// A collective: hardware `subgroupAdd` has its own accumulation topology,
    /// the oracle sums in increasing lane order.
    Collective,
    /// An F16 destination: both sides narrow at the store, so they may differ by
    /// one half-precision ulp and no more.
    HalfStore,
    /// No collective and no quantization: the difference can only be rounding of
    /// the contraction (GLSL may contract `a*b+acc` into an FMA).
    Serial { terms: usize },
}

impl Tol {
    fn bound(self, expected: f32) -> f32 {
        let e = expected.abs();
        match self {
            Tol::Quantized => 1e-3f32.max(2e-4 * e),
            Tol::Collective => 1e-4f32.max(2e-4 * e),
            Tol::HalfStore => 1e-3f32.max(1e-3 * e),
            Tol::Serial { terms } => 1e-5f32.max(terms as f32 * f32::EPSILON * e.max(1.0) * 4.0),
        }
    }
}

fn kernel_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../generated/rir")
        .join(name)
}

fn rng(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
    u * 2.0 - 1.0
}

/// A buffer of one argument, in the two shapes the interpreter and the runtime
/// need: floats for an F32 binding, bytes for an F16 or quantized one.
enum Buf {
    F32(Vec<f32>),
    Bytes(Vec<u8>),
}

impl Buf {
    fn as_arg(&self) -> Arg<'_> {
        match self {
            Buf::F32(v) => Arg::input(v),
            Buf::Bytes(v) => Arg::input(v),
        }
    }

    fn as_arg_mut(&mut self) -> Arg<'_> {
        match self {
            Buf::F32(v) => Arg::output(v),
            Buf::Bytes(v) => Arg::output(v),
        }
    }

    /// The values a comparison reads, decoded when the buffer holds halves.
    fn values(&self) -> Vec<f32> {
        match self {
            Buf::F32(v) => v.clone(),
            Buf::Bytes(v) => v
                .as_chunks::<2>()
                .0
                .iter()
                .map(|c| f16_to_f32(u16::from_le_bytes(*c)))
                .collect(),
        }
    }
}

/// The geometry of one argument under a shape assignment: logical `ne[]` and
/// packed `nb[]`, derived from `arg_axes` and the element type.
///
/// This is the whole reason a declarative harness is possible: an argument's
/// shape is not an input of the test, it is a **consequence** of the axis
/// extents and of which dimension each axis indexes - which the Loop IR
/// publishes because the manifest publishes it.
fn geometry(lk: &LoopKernel, arg: usize, extents: &[usize]) -> ([usize; 4], [usize; 4], usize) {
    let dtype = lk.args[arg].ty.dtype;
    let mut ne = [1usize; 4];
    for (d, axis) in lk.arg_axes[arg].iter().enumerate() {
        if let Some(a) = axis {
            ne[d] = extents[a.0 as usize];
        }
    }
    // `nb[0]` is one stride **unit**: an element, or a whole block for a
    // quantized format. A row is then `ne[0] / block_elements` units.
    let (unit, per_unit) = match dtype {
        DType::Quant(q) => {
            let d = q.desc();
            (d.block_bytes as usize, d.block_elements as usize)
        }
        other => (other.size_bytes(), 1),
    };
    let mut nb = [unit, 0, 0, 0];
    nb[1] = unit * ne[0].div_ceil(per_unit);
    nb[2] = nb[1] * ne[1];
    nb[3] = nb[2] * ne[2];
    let bytes = nb[3] * ne[3];
    (ne, nb, bytes)
}

/// The push-constant block, filled from the manifest's own list of names.
///
/// A value nobody asked for is not passed, and a name the manifest carries and
/// this cannot resolve is a failure rather than a default: the second case is
/// how a test starts dispatching with a zero extent and still passes.
fn values(m: &Manifest, lk: &LoopKernel, extents: &[usize], nbs: &[[usize; 4]]) -> Values {
    let mut v = Values::new();
    for pc in &m.push_constants {
        if let Some(axis) = pc.name.strip_prefix("n_") {
            let i = lk
                .axes
                .iter()
                .position(|a| a.name == axis)
                .unwrap_or_else(|| panic!("{}: push constant {} is no axis", lk.name, pc.name));
            v.u32(&pc.name, extents[i] as u32);
            continue;
        }
        if let Some((arg, dim)) = pc
            .name
            .rsplit_once("_nb")
            .and_then(|(a, d)| d.parse::<usize>().ok().map(|d| (a, d)))
            && let Some(i) = lk.args.iter().position(|x| x.name == arg)
        {
            v.u32(&pc.name, nbs[i][dim] as u32);
            continue;
        }
        // The flattened dispatch's own block. Computed
        // here rather than read from anywhere, and that is the point: the shader
        // divides by these numbers, the oracle divides by the extents, and the
        // comparison at the end is what proves `fastdiv_magic` and
        // `init_fastdiv_values` are the same function. A wrong multiplier does
        // not crash - it decomposes to the wrong indices, which is a wrong
        // result on every element but the first.
        if pc.name.starts_with("rir_flat") {
            let flat = lk.flat_axes();
            let divisor = |i: usize| {
                let (ax, per) = flat[i];
                extents[ax.0 as usize].div_ceil(per as usize) as u32
            };
            if pc.name == "rir_flat_total" {
                let total: u32 = (0..flat.len()).map(divisor).product();
                v.u32(&pc.name, total);
                continue;
            }
            let (i, field) = pc.name["rir_flat".len()..]
                .split_once('_')
                .map(|(i, f)| (i.parse::<usize>().expect("flat index"), f))
                .expect("flat push constant name");
            let d = divisor(i);
            let (mp, sh) = rir_lower::fastdiv_magic(d);
            v.u32(
                &pc.name,
                match field {
                    "div" => d,
                    "mp" => mp,
                    "sh" => sh,
                    other => panic!("{}: flat push constant field {other}", lk.name),
                },
            );
            continue;
        }
        let Some(i) = lk.params.iter().position(|p| p.name == pc.name) else {
            panic!("{}: push constant {} resolves to nothing", lk.name, pc.name);
        };
        // The parameters the kernels declare are an epsilon or a scale. A value
        // per *name* and not per kernel, so a new parameter is a compile-time
        // decision here rather than a silent zero.
        let value = match lk.params[i].name.as_str() {
            "eps" => 1e-6,
            "scale" | "bias" => 0.75,
            other => panic!("{}: no test value for parameter {other}", lk.name),
        };
        v.f32(&pc.name, value);
    }
    v
}

/// The shape assignments a kernel is tried on: axis extents, in axis order.
///
/// Edge shapes, derived: a base, then one axis at a time reduced to its minimum
/// (1, or one block for a quantized contiguous dimension), then an odd extent on
/// the contiguous axis, then - when the kernel folds an index - a shape where
/// the folded axis is genuinely smaller than the one it divides, which is the
/// only shape that exercises the fold at all.
fn shapes(lk: &LoopKernel) -> Vec<Vec<usize>> {
    let n = lk.axes.len();
    // The quantum of each axis: 1 unless a quantized argument indexes its
    // contiguous dimension with it, in which case a row must be whole blocks.
    let mut quantum = vec![1usize; n];
    for (a, arg) in lk.args.iter().enumerate() {
        if let DType::Quant(q) = arg.ty.dtype
            && let Some(Some(axis)) = lk.arg_axes[a].first()
        {
            let be = q.desc().block_elements as usize;
            quantum[axis.0 as usize] = quantum[axis.0 as usize].max(be);
        }
    }
    let base: Vec<usize> = quantum.iter().map(|q| q * 4).collect();
    let mut out = vec![base.clone()];
    for i in 0..n {
        let mut s = base.clone();
        s[i] = quantum[i];
        out.push(s);
    }
    // An odd extent, on the axis of the contiguous dimension of the output:
    // the one a partial workgroup and a vector tail are decided by.
    let mut odd = base.clone();
    for i in 0..n {
        if quantum[i] == 1 {
            odd[i] = 33;
            break;
        }
    }
    out.push(odd);
    if !lk.folds.is_empty() {
        let mut fold = base.clone();
        for (index, over) in &lk.folds {
            let i = index.0 as usize;
            let o = over.0 as usize;
            fold[i] = quantum[i] * 4;
            fold[o] = quantum[o] * 2;
        }
        out.push(fold);
    }
    // A fold requires `extent(index) % extent(over) == 0`, so an assignment that
    // shrank one side of a fold is not a shape this kernel accepts.
    out.retain(|s| {
        lk.folds
            .iter()
            .all(|(i, o)| s[i.0 as usize] % s[o.0 as usize] == 0)
    });
    out
}

/// The shapes above, less those a variant's own **layout claim** excludes.
///
/// A `vector_width > 1` lowering reads `w` consecutive indices of the contiguous
/// axis as `w` consecutive addresses. A fold over that axis breaks that unless
/// it is the identity, which is exactly the condition
/// `ggml_rir_variant_fits_layout` evaluates at a dispatch site before selecting
/// the variant - so trying it here on a shape the dispatcher would refuse tests
/// a pairing that cannot occur, and fails on a read the manifest already
/// forbids. The pair's fallback takes those shapes, and it is covered above.
fn claimed_shapes(lk: &LoopKernel) -> Vec<Vec<usize>> {
    let mut out = shapes(lk);
    if lk.vector_width() > 1
        && let Some((ax, _)) = lk.flat_axes().first().copied()
    {
        out.retain(|s| {
            lk.folds
                .iter()
                .all(|(i, o)| *i != ax || s[o.0 as usize] == s[ax.0 as usize])
        });
    }
    // The second layout claim, and the harness owes it the same treatment
    // a linear variant addresses `linear · w ·
    // elem_bytes`, which is the element's offset only while the contiguous
    // extent is a whole number of vectors. The derived `odd` shape is 33, so
    // this is not a hypothetical - it is the one shape of the set the claim
    // excludes, and the pair's decomposing variant is what covers it.
    //
    // The other half of the claim needs nothing here: this harness packs every
    // `nb[]` from the extents (`geometry`), so every binding it builds is
    // contiguous. `device_parity` is where a view with a gap is written, and it
    // is where the refusal is checked rather than avoided.
    if lk.linear_addr()
        && let Some((ax, per)) = lk.flat_axes().first().copied()
        && per > 1
    {
        out.retain(|s| s[ax.0 as usize] % per as usize == 0);
    }
    out
}

/// One kernel, one shape: fill, run both sides, compare.
fn check(name: &str, lk: &LoopKernel, pipe: &AnyPipeline<'_>, m: &Manifest, extents: &[usize]) {
    let mut seed = 0x9e37_79b9u64 ^ extents.iter().fold(1u64, |a, e| a * 31 + *e as u64);
    let n_args = lk.args.len();
    let out_index = lk
        .args
        .iter()
        .position(|a| a.access == Access::Write)
        .expect("a kernel writes something");
    assert!(
        lk.args.iter().filter(|a| a.access == Access::Write).count() == 1,
        "{name}: this harness binds exactly one written argument"
    );

    let mut geo = Vec::new();
    for a in 0..n_args {
        geo.push(geometry(lk, a, extents));
    }
    let nbs: Vec<[usize; 4]> = geo.iter().map(|(_, nb, _)| *nb).collect();

    // Inputs, in argument order; the written slot gets an empty placeholder.
    let mut inputs: Vec<Buf> = Vec::new();
    for (a, arg) in lk.args.iter().enumerate() {
        let (ne, _, bytes) = geo[a];
        if a == out_index {
            inputs.push(Buf::F32(Vec::new()));
            continue;
        }
        inputs.push(match arg.ty.dtype {
            DType::F32 => Buf::F32((0..bytes / 4).map(|_| rng(&mut seed) * 1.5).collect()),
            DType::F16 => Buf::Bytes(
                (0..bytes / 2)
                    .flat_map(|_| f32_to_f16(rng(&mut seed) * 1.5).to_le_bytes())
                    .collect(),
            ),
            DType::Quant(q) => {
                let blocks = ne[0] / q.desc().block_elements as usize * ne[1] * ne[2] * ne[3];
                Buf::Bytes(
                    rir_core::random_block_bytes(q, blocks, &mut seed)
                        .unwrap_or_else(|| panic!("{name}: format without a description")),
                )
            }
            other => panic!("{name}: no fixture for a {} binding", other.name()),
        });
    }

    let (_, _, out_bytes) = geo[out_index];
    let half_out = lk.args[out_index].ty.dtype == DType::F16;
    let mut expected = if half_out {
        Buf::Bytes(vec![0u8; out_bytes])
    } else {
        Buf::F32(vec![0f32; out_bytes / 4])
    };

    {
        // The written slot is built once, outside the loop: it is the only `&mut`
        // borrow, and the borrow checker cannot see that a loop takes it once.
        let (ne_out, nb_out, _) = geo[out_index];
        let out_bound = match &mut expected {
            Buf::Bytes(d) => BoundArg::OutBytes(TensorViewBytesMut {
                data: d,
                shape: ne_out,
                nb: nb_out,
            }),
            Buf::F32(d) => BoundArg::Out(TensorViewMut {
                data: d,
                shape: ne_out,
                nb: nb_out,
            }),
        };
        let mut args: Vec<BoundArg> = Vec::new();
        for (a, _arg) in lk.args.iter().enumerate() {
            if a == out_index {
                continue;
            }
            let (ne, nb, _) = geo[a];
            args.push(match &inputs[a] {
                Buf::F32(d) => BoundArg::In(TensorView {
                    data: d,
                    shape: ne,
                    nb,
                }),
                Buf::Bytes(d) => BoundArg::InBytes(TensorViewBytes {
                    data: d,
                    shape: ne,
                    nb,
                }),
            });
        }
        args.insert(out_index, out_bound);
        let params: Vec<f32> = lk
            .params
            .iter()
            .map(|p| match p.name.as_str() {
                "eps" => 1e-6,
                _ => 0.75,
            })
            .collect();
        run(lk, &mut args, &params).unwrap_or_else(|e| panic!("{name}: oracle: {e}"));
    }

    let mut got = if half_out {
        Buf::Bytes(vec![0u8; out_bytes])
    } else {
        Buf::F32(vec![0f32; out_bytes / 4])
    };
    let vals = values(m, lk, extents, &nbs);
    {
        let out_arg = got.as_arg_mut();
        let mut bound: Vec<Arg> = (0..n_args)
            .filter(|&a| a != out_index)
            .map(|a| inputs[a].as_arg())
            .collect();
        bound.insert(out_index, out_arg);
        pipe.run(&mut bound, &vals)
            .unwrap_or_else(|e| panic!("{name} {extents:?}: dispatch: {e:?}"));
    }

    let tol = tolerance(lk, extents, half_out);
    let (e, g) = (expected.values(), got.values());
    assert_eq!(e.len(), g.len());
    for i in 0..e.len() {
        let bound = tol.bound(e[i]);
        assert!(
            (g[i] - e[i]).abs() <= bound,
            "{name} {extents:?} element {i}: gpu {} vs oracle {} (bound {bound}, {tol:?})",
            g[i],
            e[i]
        );
    }
}

/// Which of the four rules applies, decided by a property of the kernel.
fn tolerance(lk: &LoopKernel, extents: &[usize], half_out: bool) -> Tol {
    if lk
        .args
        .iter()
        .any(|a| matches!(a.ty.dtype, DType::Quant(_)))
    {
        return Tol::Quantized;
    }
    if lk.uses_subgroup() || lk.uses_shared() {
        return Tol::Collective;
    }
    if half_out {
        return Tol::HalfStore;
    }
    Tol::Serial {
        terms: extents.iter().copied().max().unwrap_or(1),
    }
}

/// The whole registry, on one backend's fallback schedule of each kernel.
///
/// Three outcomes per kernel, and keeping them apart is the whole accounting:
/// **covered** (it ran against the oracle), **skipped** (this machine cannot run
/// it, with the reason), and **refused** (the family declared in writing that it
/// does not serve this backend). Only the third is
/// new, and it is new because CUDA is the first backend the table does not serve
/// whole.
fn every_registered_kernel_on(target: Target) {
    // Opened and dropped: the workers below each open their own, and this one
    // answers the only question that has to be answered before any of them
    // exists - whether this machine has a device at all, and which.
    match AnyGpu::open(target) {
        Ok(g) => eprintln!("family parity ({}) on '{}'", target.name(), g.name()),
        Err(e) if is_unavailable(&e) => {
            eprintln!("family parity ({}) skipped: {e}", target.name());
            return;
        }
        Err(e) => panic!("opening device: {e}"),
    }
    let backend = match target {
        Target::Vulkan => Backend::Vulkan,
        Target::Cuda => Backend::Cuda,
    };
    let gpu_backend = backend.gpu().expect("a GPU target");
    // The parameter header a generated `.cu` includes.
    // Read from the committed registry like everything else here: what is under
    // test is the artifact as it was generated, not one re-emitted in process.
    let params_header = std::fs::read_to_string(kernel_dir("registry").join("rir_kernel_params.h"))
        .expect("rir_kernel_params.h");

    // Two phases, and the split is what makes the second one parallel. This one
    // reads the tables and decides *what* must run: no device is touched, so a
    // registry defect - a kernel with neither a fallback schedule nor a written
    // refusal - still fails here with the same message, before any thread
    // exists to confuse it.
    let entries = registry();
    let mut work: Vec<(usize, String, Schedule)> = Vec::new();
    let mut refused: Vec<(String, &'static str)> = Vec::new();
    for (index, reg) in entries.iter().enumerate() {
        let name = reg.id.name();
        let Some(schedule) = reg
            .schedules
            .iter()
            .find(|s| s.backend() == backend && s.variant().is_none())
            .cloned()
        else {
            // A family that refuses this backend in writing said so; the
            // absence is then the statement, and it is reported as one.
            if let Some(why) = rir_lower::family_refusal(reg.family, gpu_backend) {
                refused.push((name, why));
                continue;
            }
            // Not a skip. A skip says "this device cannot run it"; the absence
            // of a fallback schedule with no refusal behind it says the registry
            // stopped declaring the coverage this harness exists to prove.
            // Tolerating it would let the witness stay green precisely when a
            // kernel loses its lowering - the tautological-test failure mode
            // this file exists against (ADR-5 section 8).
            panic!(
                "{name}: registered without a {} fallback schedule, and without a \
                 family refusal saying so",
                target.name()
            );
        };
        // The fallback, and every **flattened** variant beside it.
        // The flattening is the one named variant this
        // harness covers, and it is not an exception to "named variants keep
        // their own tests": it introduces *arithmetic* - a magic-number
        // decomposition - where every other variant only moves a block shape
        // around, and arithmetic is what an oracle is for. A wrong multiplier
        // computes the wrong index and nothing else notices.
        work.push((index, name.clone(), schedule));
        for s in reg.schedules.iter() {
            // Named ones only: a flattened lowering that is the pair's fallback
            // is already the schedule above.
            if s.backend() == backend && s.flatten() && s.variant().is_some() {
                work.push((index, s.artifact_name(&name), s.clone()));
            }
        }
    }

    // Phase two: every lowering above, against the oracle, across the machine's
    // cores. `user` time was three quarters of the wall clock of this one test,
    // the oracle is an interpreter and it runs on the host - so a single thread
    // was leaving the rest of the box idle for minutes.
    //
    // One `AnyGpu` **per thread**, not one shared: the wrapper's Vulkan arm
    // stages each shader through a file named after the artifact, and each
    // worker holds pipelines borrowed from its own device for as long as it uses
    // them. Two lowerings never share an artifact name, so no two workers write
    // the same path.
    let threads = std::env::var("RIR_PARITY_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()))
        .clamp(1, work.len().max(1));
    let next = AtomicUsize::new(0);
    let covered = Mutex::new(Vec::<String>::new());
    let skipped = Mutex::new(Vec::<(String, String)>::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                // The probe above already opened one, so a failure here is a
                // real one rather than "no device on this machine".
                let gpu = AnyGpu::open(target).expect("opening a second device");
                while let Some((index, artifact_name, schedule)) =
                    work.get(next.fetch_add(1, Ordering::Relaxed))
                {
                    let reg = &entries[*index];
                    let name = reg.id.name();
                    let dir = kernel_dir(&name);
                    let (source_file, manifest_file) = target.files(schedule.variant());
                    let manifest = Manifest::load_file(&dir, &manifest_file)
                        .unwrap_or_else(|e| panic!("{name}: {manifest_file}: {e}"));
                    let source = std::fs::read_to_string(dir.join(&source_file))
                        .unwrap_or_else(|e| panic!("{name}: {source_file}: {e}"));
                    let artifact = Artifact {
                        artifact: artifact_name,
                        source: &source,
                        params_header: &params_header,
                    };
                    let pipe = match gpu.build(&manifest, &artifact) {
                        Ok(p) => p,
                        Err(e) if is_unavailable(&e) => {
                            skipped
                                .lock()
                                .unwrap()
                                .push((artifact_name.clone(), format!("{e}")));
                            continue;
                        }
                        Err(e) => {
                            panic!(
                                "{artifact_name}: building the {} kernel: {e}",
                                target.name()
                            )
                        }
                    };
                    let lk = rir_lower::lower(&reg.kernel, schedule.clone())
                        .unwrap_or_else(|e| panic!("{artifact_name}: lowering: {e}"));
                    for extents in claimed_shapes(&lk) {
                        check(artifact_name, &lk, &pipe, &manifest, &extents);
                    }
                    covered.lock().unwrap().push(artifact_name.clone());
                }
            });
        }
    });
    let covered = covered.into_inner().unwrap();
    let mut skipped = skipped.into_inner().unwrap();
    skipped.sort();

    eprintln!(
        "family parity ({}): {} kernels covered, {} refused by their family",
        target.name(),
        covered.len(),
        refused.len()
    );
    for (name, why) in &skipped {
        eprintln!("family parity: {name} skipped - {why}");
    }
    for (name, why) in &refused {
        eprintln!("family parity: {name} refused - {why}");
    }

    // The four families this file exists for. A harness that quietly stopped
    // covering one of them would otherwise still be green, which is the failure
    // mode this whole item is about.
    for required in ["rms_norm", "unary_silu", "add_f16", "out_prod_q4_0"] {
        assert!(
            covered.iter().any(|c| c == required)
                || skipped.iter().any(|(n, _)| n == required)
                || refused.iter().any(|(n, _)| n == required),
            "family parity covers neither {required} nor a reason for it: {covered:?}"
        );
    }
    // What must have run, derived from the same tables the loop walked: one
    // entry per kernel whose family serves this backend, plus one per flattened
    // variant beside it.
    let expected_artifacts: Vec<String> = entries
        .iter()
        .filter(|r| rir_lower::family_refusal(r.family, gpu_backend).is_none())
        .flat_map(|r| {
            let name = r.id.name();
            let mut v = vec![name.clone()];
            v.extend(
                r.schedules
                    .iter()
                    .filter(|s| s.backend() == backend && s.flatten() && s.variant().is_some())
                    .map(|s| s.artifact_name(&name)),
            );
            v
        })
        .collect();
    // Total accounting, which is the property that matters: every registered
    // lowering is either covered, explained, or refused in writing. A harness
    // that quietly stopped enumerating would fail here rather than pass with
    // fewer cases.
    let n = expected_artifacts.len() + refused.len();
    assert_eq!(
        covered.len() + skipped.len() + refused.len(),
        n,
        "family parity accounted for {} of {n} registered lowerings",
        covered.len() + skipped.len() + refused.len()
    );
    if skipped.is_empty() {
        // The exact set, not a count above a round number: on a device that
        // skipped nothing, what must have run is every lowering the schedule
        // table publishes for this backend, rather than a number this file
        // would have to keep up to date.
        let mut expected: Vec<String> = expected_artifacts;
        expected.sort();
        let mut ran = covered.clone();
        ran.sort();
        assert_eq!(
            ran,
            expected,
            "family parity ({}) ran a different set from the one the table serves",
            target.name()
        );
    }
}

#[test]
fn every_registered_kernel_agrees_with_the_oracle_on_the_device() {
    every_registered_kernel_on(Target::Vulkan);
}

/// The same witness on CUDA, which is the first check that
/// the CUDA emitter produces kernels that **compute** rather than merely
/// compile.
///
/// It is also where the CUDA tolerance question is answered by measurement
/// rather than by intuition: the sources are compiled with the fork's
/// `-use_fast_math`, so `expf` and `sqrtf` are the fast intrinsics here as they
/// are in the shipped binary, and the bounds this harness applies are the ones
/// it already applied to Vulkan.
#[cfg(feature = "cuda")]
#[test]
fn every_registered_kernel_agrees_with_the_oracle_on_cuda() {
    every_registered_kernel_on(Target::Cuda);
}

/// A witness for the derivation itself: the geometry this harness computes is
/// the geometry the hand-written tests write out.
///
/// Without it, a wrong stride would make every case above wrong **in the same
/// way** on both sides - the oracle and the shader read the same numbers - and
/// the comparison would still pass.
#[test]
fn the_derived_geometry_is_the_one_the_handwritten_tests_use() {
    // `sum_rows_q8_0`, from `device_parity::sum_rows_quant_on_gpu_against_the_oracle`:
    // four blocks of 32 on the contiguous dimension, three rows, `nb = [34, 136]`.
    let kernel = rir_kernels::sum_rows_quant::build(rir_core::QuantType::Q8_0).unwrap();
    let lk = rir_lower::lower(&kernel, Schedule::vulkan_subgroup()).unwrap();
    let col = lk.axes.iter().position(|a| a.name == "col").unwrap();
    let row = lk.axes.iter().position(|a| a.name == "row").unwrap();
    let mut extents = vec![1usize; lk.axes.len()];
    extents[col] = 128;
    extents[row] = 3;
    let (ne, nb, bytes) = geometry(&lk, 0, &extents);
    assert_eq!(ne, [128, 3, 1, 1]);
    assert_eq!(nb, [34, 136, 408, 408]);
    assert_eq!(bytes, 408);
    let (ne_y, nb_y, bytes_y) = geometry(&lk, 1, &extents);
    assert_eq!(ne_y, [3, 1, 1, 1]);
    assert_eq!(nb_y, [4, 12, 12, 12]);
    assert_eq!(bytes_y, 12);
}
