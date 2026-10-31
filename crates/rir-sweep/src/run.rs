//! The sweep itself: build the pair's fallback, then every candidate, on every
//! shape, and hand the rows to [`crate::verdict::arbitrate`].
//!
//! The order of operations is the one property of this file worth stating.
//! Every candidate is timed **against the same fixture bytes** as the fallback
//! and compared against the fallback's own output on that shape, in the same
//! process and the same device session. Two of those three are what make the
//! comparison a comparison; the third - same process - is what makes the
//! agreement check free, since the reference is already in host memory.

use rir_core::catalog::Regime;
use rir_lower::LoopKernel;
use rir_runtime::any::{AnyGpu, Backend};

use crate::SweepError;
use crate::bind::{Fixture, push_values};
use crate::candidates::{Candidate, Origin, Subject, candidates, extents_with_contraction, shapes};
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
    /// Optional substrings selecting candidates; the fallback is always kept.
    pub candidate_filters: Vec<String>,
    /// OUT_PROD reduction depth, independent of its output shape.
    pub contraction: usize,
    /// The model profile and regime the shapes were derived from, when they
    /// were: the report then says which profile its proposal is for.
    pub profile: Option<(String, Regime)>,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            reps: REPS,
            warmup: WARMUP,
            iters: ITERS,
            shapes: None,
            mode: Mode::Stream,
            candidate_filters: Vec::new(),
            contraction: 64,
            profile: None,
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
    /// See [`Options::profile`].
    pub profile: Option<(String, Regime)>,
    pub contraction: Option<usize>,
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
    let gpu_backend = target.gpu();
    let all: Vec<_> = candidates(subject, gpu_backend)
        .into_iter()
        .filter(|c| {
            c.origin == Origin::Fallback
                || options.candidate_filters.is_empty()
                || options
                    .candidate_filters
                    .iter()
                    .any(|filter| c.tag.contains(filter))
        })
        .collect();
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
        let ext = extents_with_contraction(&base_built.lowered, *shape, options.contraction)?;
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
            &base_built,
            options,
        ) {
            Ok(a) => arbitrated.push(a),
            Err(e) if e.is_unavailable() => return Err(e),
            Err(e) => declined.push((candidate.tag.clone(), e.to_string())),
        }
    }

    Ok(SweepReport {
        contraction: (subject.family == rir_lower::Family::OutProd).then_some(options.contraction),
        mode: options.mode,
        kernel: subject.name.clone(),
        family: subject.family.name(),
        profile: options.profile.clone(),
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

#[expect(clippy::too_many_arguments)]
fn arbitrate_one(
    gpu: &AnyGpu,
    target: Backend,
    subject: &Subject,
    candidate: &Candidate,
    fixtures: &[(Fixture, Vec<usize>)],
    reference: &[Vec<f32>],
    baseline: &Built<'_>,
    options: &Options,
) -> Result<Arbitrated, SweepError> {
    let built = build(gpu, target, &subject.kernel, candidate.schedule.clone())?;
    let footprint = Footprint::of(&built.lowered).on(gpu);
    let mut rows = Vec::new();
    for (i, (fixture, ext)) in fixtures.iter().enumerate() {
        let shape = shape_of(&built.lowered, ext);
        // Compilation and a long candidate list can change clock/thermal
        // state. Bracket each candidate with fresh baseline timings; include
        // baseline drift in the noise instead of comparing to a cold sample
        // from the start of the sweep.
        let (before, _) = run_shape(baseline, fixture, ext, options)?;
        let (sample, values) = run_shape(&built, fixture, ext, options)?;
        let (after, _) = run_shape(baseline, fixture, ext, options)?;
        rows.push(Measured {
            shape,
            base: bracket(before, after),
            candidate: sample,
            agrees: agrees(&reference[i], &values, regroup_len(&built.lowered, ext)),
        });
    }
    let mut verdict = arbitrate(&rows);
    // The GEMV has no `batch` axis: its output count is `row · plane`, and a
    // rule naming an axis the kernel lacks is refused by `check_schedule`.
    if subject.family == rir_lower::Family::MulMatVec
        && let Verdict::Claims { rules, .. } = &mut verdict
    {
        for rule in rules.iter_mut() {
            if rule.axes == ["row", "plane", "batch"] {
                rule.axes = &["row", "plane"];
            }
        }
    }
    if subject.family == rir_lower::Family::OutProd {
        // One measured contraction depth cannot justify a claim on every k.
        let k = u32::try_from(options.contraction).map_err(|_| SweepError::Unbindable {
            kernel: subject.name.clone(),
            why: "contraction exceeds rule counter".into(),
        })?;
        if let Verdict::Replaces { best_gain } = verdict {
            verdict = Verdict::Claims {
                rules: Vec::new(),
                best_gain,
                ties_claimed: Vec::new(),
            };
        }
        if let Verdict::Claims { rules, .. } = &mut verdict {
            for rule in rules.iter_mut() {
                rule.axes = match rule.axes {
                    ["col"] => &["i"],
                    ["row", "plane", "batch"] => &["j", "plane", "batch"],
                    _ => rule.axes,
                };
            }
            rules.push(rir_lower::ShapeRule {
                axes: &["k"],
                min: k,
                max: k,
            });
        }
    }
    Ok(Arbitrated {
        tag: candidate.tag.clone(),
        source: candidate.source.clone(),
        origin: candidate.origin,
        footprint,
        rows,
        verdict,
    })
}

/// Preserve both within-pass spread and clock drift between controls.
fn bracket(before: Sample, after: Sample) -> Sample {
    let median_us = (before.median_us + after.median_us) / 2.0;
    let drift = (before.median_us - after.median_us).abs() / median_us;
    Sample {
        median_us,
        spread: before.spread.max(after.spread).max(drift),
        groups: before.groups,
    }
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
        if let Some(a) = lk.axes.iter().position(|a| {
            a.name == *name
                || (lk.name.starts_with("out_prod")
                    && ((*name == "col" && a.name == "i") || (*name == "row" && a.name == "j")))
        }) {
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
/// When terms are regrouped, the bound scales with the larger of the element
/// and the output's root mean square. A reassociation error is proportional to
/// the partial sums, not to the result: a sum that cancels - q6_K's
/// offset-centred weights do, routinely - ends far below the partial sums it
/// went through, and a bound relative to the element alone calls every
/// regrouping of it a disagreement. The sweep has no `Σ|terms|` per element;
/// the RMS of the whole output is its scale for a random fixture, and it
/// cannot hide a wrong formula, which is off by the output's own magnitude.
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
    let scale = if regroup_len > 1 { rms(reference) } else { 0.0 };
    reference.iter().zip(got).all(|(a, b)| {
        if a.is_nan() || b.is_nan() {
            return a.is_nan() == b.is_nan();
        }
        let bound = rel * (1.0 + f64::from(a.abs()).max(scale));
        (f64::from(*a) - f64::from(*b)).abs() <= bound
    })
}

/// Root mean square of the finite elements; zero when there are none.
fn rms(values: &[f32]) -> f64 {
    let (sum, n) = values
        .iter()
        .filter(|v| v.is_finite())
        .fold((0.0f64, 0usize), |(s, n), v| {
            (s + f64::from(*v).powi(2), n + 1)
        });
    if n == 0 { 0.0 } else { (sum / n as f64).sqrt() }
}

#[cfg(test)]
mod tests {
    use super::{agrees, bracket};
    use crate::measure::Sample;

    #[test]
    fn control_drift_is_counted_as_measurement_noise() {
        let sample = |us| Sample {
            median_us: us,
            spread: 0.01,
            groups: [1, 1, 1],
        };
        let control = bracket(sample(10.0), sample(20.0));
        assert_eq!(control.median_us, 15.0);
        assert!(control.spread > 0.66);
    }

    #[test]
    fn a_regrouped_sum_is_bounded_by_the_output_scale_not_the_element() {
        // One element cancelled to near zero among peers of magnitude 600: a
        // reassociation moves it by a few ULPs of 600, not of 0.01.
        let reference = [600.0, -590.0, 0.01, 610.0];
        let regrouped = [600.0, -590.0, 0.0103, 610.0];
        assert!(agrees(&reference, &regrouped, 1024));
        // A wrong value is still wrong at that scale.
        let wrong = [600.0, -590.0, 1.0, 610.0];
        assert!(!agrees(&reference, &wrong, 1024));
        // Without regrouping the element alone sets the bound.
        assert!(!agrees(&reference, &regrouped, 1));
    }
}
