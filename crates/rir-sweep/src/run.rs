//! The sweep itself: build the pair's fallback, then every candidate, on every
//! shape, and hand the rows to [`crate::verdict::arbitrate`].
//!
//! The order of operations is the one property of this file worth stating.
//! Every candidate is timed **against the same fixture bytes** as the fallback
//! and compared against the fallback's own output on that shape, in the same
//! process and the same device session. Two of those three are what make the
//! comparison a comparison; the third - same process - is what makes the
//! agreement check free, since the reference is already in host memory.

use rir_lower::LoopKernel;
use rir_runtime::any::{AnyGpu, Backend};

use crate::SweepError;
use crate::bind::{Fixture, push_values};
use crate::candidates::{Candidate, Origin, Subject, candidates, extents, shapes};
use crate::measure::{Built, Footprint, ITERS, Mode, REPS, Sample, WARMUP, build, time};
use crate::verdict::{Measured, Verdict, arbitrate};

/// How long the sweep runs and on what.
#[derive(Clone, Debug)]
pub struct Options {
    pub reps: u32,
    pub warmup: u32,
    pub iters: u32,
    /// `[col, row, plane, batch]` shapes, or the family's own list.
    pub shapes: Option<Vec<[usize; 4]>>,
    pub mode: Mode,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            reps: REPS,
            warmup: WARMUP,
            iters: ITERS,
            shapes: None,
            mode: Mode::Stream,
        }
    }
}

/// One candidate's line in the report.
pub struct Arbitrated {
    pub tag: String,
    pub source: String,
    pub origin: Origin,
    pub footprint: Footprint,
    pub rows: Vec<Measured>,
    pub verdict: Verdict,
}

/// Everything one (kernel, backend) sweep produced.
pub struct SweepReport {
    pub mode: Mode,
    pub kernel: String,
    pub family: &'static str,
    pub backend: Backend,
    pub device: String,
    pub base_tag: String,
    pub base_footprint: Footprint,
    pub shapes: Vec<[usize; 4]>,
    pub base: Vec<Sample>,
    pub arbitrated: Vec<Arbitrated>,
    /// Candidates the compiler refused, with its refusal. Printed rather than
    /// dropped: "this geometry does not lower for this kernel" is an answer,
    /// and one a reader would otherwise reproduce by hand.
    pub declined: Vec<(String, String)>,
}

/// Runs the sweep for one kernel on one backend.
pub fn sweep(
    subject: &Subject,
    target: Backend,
    gpu: &AnyGpu,
    options: &Options,
) -> Result<SweepReport, SweepError> {
    let gpu_backend = match target {
        Backend::Vulkan => rir_lower::GpuBackend::Vulkan,
        Backend::Cuda => rir_lower::GpuBackend::Cuda,
    };
    let all = candidates(subject, gpu_backend);
    let base = all
        .iter()
        .find(|c| c.origin == Origin::Fallback)
        .ok_or_else(|| SweepError::Unbindable {
            kernel: subject.name.clone(),
            why: format!(
                "the table carries no fallback for {} on {}; there is nothing to measure against",
                subject.family.name(),
                target.name()
            ),
        })?;
    let base_built = build(gpu, target, &subject.kernel, base.schedule.clone())?;
    let base_footprint = Footprint::of(&base_built.lowered).on(gpu);

    let shape_list = options
        .shapes
        .clone()
        .unwrap_or_else(|| shapes(subject.family));

    // The fallback first, on every shape: its output is the reference every
    // candidate is compared against, and its median is the denominator of every
    // gain.
    let mut fixtures = Vec::new();
    let mut reference = Vec::new();
    let mut base_samples = Vec::new();
    let mut kept = Vec::new();
    for shape in &shape_list {
        let ext = extents(&base_built.lowered, *shape)?;
        let fixture = Fixture::build(&base_built.lowered, &ext)?;
        let (sample, values) = run_shape(&base_built, &fixture, &ext, options)?;
        kept.push(*shape);
        fixtures.push((fixture, ext));
        reference.push(values);
        base_samples.push(sample);
    }

    let mut arbitrated = Vec::new();
    let mut declined = Vec::new();
    for candidate in all.iter().filter(|c| c.origin != Origin::Fallback) {
        match arbitrate_one(
            gpu,
            target,
            subject,
            candidate,
            &fixtures,
            &reference,
            &base_samples,
            options,
        ) {
            Ok(a) => arbitrated.push(a),
            Err(e) if e.is_unavailable() => return Err(e),
            Err(e) => declined.push((candidate.tag.clone(), e.to_string())),
        }
    }

    Ok(SweepReport {
        mode: options.mode,
        kernel: subject.name.clone(),
        family: subject.family.name(),
        backend: target,
        device: gpu.name(),
        base_tag: base.tag.clone(),
        base_footprint,
        shapes: kept,
        base: base_samples,
        arbitrated,
        declined,
    })
}

#[allow(clippy::too_many_arguments)]
fn arbitrate_one(
    gpu: &AnyGpu,
    target: Backend,
    subject: &Subject,
    candidate: &Candidate,
    fixtures: &[(Fixture, Vec<usize>)],
    reference: &[Vec<f32>],
    base: &[Sample],
    options: &Options,
) -> Result<Arbitrated, SweepError> {
    let built = build(gpu, target, &subject.kernel, candidate.schedule.clone())?;
    let footprint = Footprint::of(&built.lowered).on(gpu);
    let mut rows = Vec::new();
    for (i, (fixture, ext)) in fixtures.iter().enumerate() {
        let shape = shape_of(&built.lowered, ext);
        let (sample, values) = run_shape(&built, fixture, ext, options)?;
        rows.push(Measured {
            shape,
            base: base[i],
            candidate: sample,
            agrees: agrees(&reference[i], &values, regroup_len(&built.lowered, ext)),
        });
    }
    let verdict = arbitrate(&rows);
    Ok(Arbitrated {
        tag: candidate.tag.clone(),
        source: candidate.source.clone(),
        origin: candidate.origin,
        footprint,
        rows,
        verdict,
    })
}

/// Prepares, dispatches once to obtain the result, then times: one session, one
/// upload. The dispatch that produces the values is also the first uncounted
/// run, which is why `time` still warms up afterwards rather than counting it.
fn run_shape(
    built: &Built<'_>,
    fixture: &Fixture,
    ext: &[usize],
    options: &Options,
) -> Result<(Sample, Vec<f32>), SweepError> {
    let values = push_values(&built.manifest, &built.lowered, ext, &fixture.nbs())?;
    let mut out = fixture.output(&built.lowered);
    let mut args = fixture.bind(&mut out);
    let session = built.pipeline.prepare(&args, &values)?;
    session.dispatch()?;
    session.read_outputs(&mut args)?;
    let sample = time(
        &session,
        options.mode,
        options.reps,
        options.warmup,
        options.iters,
    )?;
    drop(args);
    Ok((sample, out.values()))
}

/// `[col, row, plane, batch]` for a kernel's axis order.
fn shape_of(lk: &LoopKernel, ext: &[usize]) -> [usize; 4] {
    let mut shape = [1usize; 4];
    for (i, name) in crate::candidates::AXES.iter().enumerate() {
        if let Some(a) = lk.axes.iter().position(|a| a.name == *name) {
            shape[i] = ext[a];
        }
    }
    shape
}

/// Number of terms a lowering may regroup on this shape: the extent of the
/// reduced or scanned axis, or one when the kernel has neither.
///
/// It is the tolerance's only free variable, and it is a property of the kernel
/// and the shape rather than a constant: two lowerings of an elementwise band
/// member compute the same expression term for term and must agree exactly,
/// while two lowerings of a 4 096-term reduction legitimately differ in the last
/// bits - which is precisely what a `SubgroupTree` against a `Serial`
/// accumulation *is*.
fn regroup_len(lk: &LoopKernel, ext: &[usize]) -> usize {
    let regroups =
        !lk.reduction_semantics.is_empty() || lk.schedule.scan() != rir_lower::ScanStrategy::Serial;
    if !regroups {
        return 1;
    }
    lk.axes
        .iter()
        .position(|a| a.name == "col")
        .map(|i| ext[i])
        .unwrap_or(1)
}

/// Whether two lowerings of one kernel returned the same bytes.
///
/// The bound is `√n · 8 · ε_f32` relative, with `n` the number of regrouped
/// terms: the error of a reassociated float sum grows like the square root of
/// its length, and the eight is slack for the epilogue. For a kernel that
/// regroups nothing, `n` is one and the bound is a handful of ULPs - a
/// vectorized band member that disagrees at all is a bug, not a rounding.
///
/// This is **not** the parity lane's check, and cannot replace it: it compares
/// RIR to RIR, so a formula wrong in both is invisible here. `family_parity`
/// compares to the interpreted oracle, and `scripts/test-rir.sh` to the native
/// kernel through ggml.
pub fn agrees(reference: &[f32], got: &[f32], regroup_len: usize) -> bool {
    if reference.len() != got.len() {
        return false;
    }
    let rel = 8.0 * f32::EPSILON as f64 * (regroup_len as f64).sqrt();
    reference.iter().zip(got).all(|(a, b)| {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() == b.is_nan();
        }
        let bound = rel * (1.0 + f64::from(a.abs()));
        (f64::from(*a) - f64::from(*b)).abs() <= bound
    })
}
