//! The measurement loop: emit in process, build on the device, time, and read
//! back what the lowering costs besides time.
//!
//! It is `device_timing`'s loop, not a second one - the test file calls
//! [`build`] and [`time`] below. What the short loop
//! provides and the promotion lane does not is exactly what a sweep needs: the
//! kernel is emitted from a `Schedule` chosen by the caller, so changing
//! geometry costs neither a regeneration nor a build of the fork. And what it
//! does not provide, a sweep does not gain by iterating: this compares RIR to
//! RIR. Promotion is `scripts/test-rir.sh`.

use rir_core::ValidatedKernel;
use rir_emit::{emit_cuda, emit_manifest, emit_vulkan};
use rir_lower::{LoopKernel, Schedule};
use rir_runtime::any::{AnyGpu, AnyPipeline, AnySession, Artifact, Backend};
use rir_runtime::{Manifest, ManifestReader};

use crate::SweepError;

/// One built candidate: the pipeline, plus the two things the *lowering* knows
/// about it and the device does not have to be asked for.
pub struct Built<'g> {
    pub pipeline: AnyPipeline<'g>,
    pub manifest: Manifest,
    pub lowered: LoopKernel,
}

/// What one candidate occupies, beside its time.
///
/// Three of the four numbers a candidate reports come from the lowering itself and are
/// therefore exact on every backend: shared storage is `LoopKernel::shared`,
/// which is what the emitters print, and the lane count is the workgroup. The
/// fourth - registers, spills, occupancy - is a **device** property, and the
/// field says which backend published it rather than reporting a zero that
/// reads like a measurement.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Footprint {
    /// Bytes of shared storage the shader declares. Every shared array in the
    /// dialect is a 32-bit word (`shared float x[n]`), so this is `4 · Σ len`.
    pub shared_bytes: u32,
    /// Invocations per workgroup.
    pub lanes: u32,
    /// Workgroups the device's shared-memory budget admits at once, when the
    /// backend publishes that budget. `None` where it does not.
    pub workgroups_by_shared: Option<u32>,
    /// Registers per thread, when the backend publishes them. Vulkan publishes
    /// none - not "zero", none - so it stays `None` there.
    pub registers: Option<u32>,
    /// Local-memory bytes per thread, i.e. spills, on a backend that reports
    /// them.
    pub spill_bytes: Option<u32>,
}

impl Footprint {
    /// What the lowering alone says, before any device is asked.
    pub fn of(lowered: &LoopKernel) -> Footprint {
        let words: u32 = lowered.shared.iter().map(|(_, len)| *len).sum();
        let block = lowered.schedule.block();
        Footprint {
            shared_bytes: words.saturating_mul(4),
            lanes: block[0] * block[1] * block[2],
            ..Footprint::default()
        }
    }

    /// The same, completed with what this device publishes.
    pub fn on(mut self, gpu: &AnyGpu) -> Footprint {
        if let Some(limit) = shared_memory_limit(gpu)
            && self.shared_bytes > 0
        {
            self.workgroups_by_shared = Some(limit / self.shared_bytes);
        }
        self
    }
}

/// The device's shared-memory budget, when the backend publishes one.
///
/// Vulkan does: `maxComputeSharedMemorySize`. What it does not publish is a
/// multiprocessor count, so the quotient above is a ceiling on **concurrent
/// workgroups per multiprocessor** and not an occupancy - which is why the field
/// is named after what it divides rather than after what a CUDA profiler would
/// call it.
// The `if let` is irrefutable in a build without the CUDA feature, and that is
// the point: the alternative arm exists only under it.
#[allow(irrefutable_let_patterns)]
fn shared_memory_limit(gpu: &AnyGpu) -> Option<u32> {
    // Written as an `if let` and not a `match`, so this file carries no
    // `#[cfg(feature = "cuda")]` of its own: `AnyGpu`'s CUDA arm exists only
    // under that feature, and a match here would stop compiling the moment a
    // caller enabled it on the runtime without enabling it on the tool.
    if let AnyGpu::Vulkan(g) = gpu {
        return Some(g.max_shared_memory());
    }
    None
}

/// Uncounted runs, then counted runs, then repetitions of that pair. See
/// [`time`] for why there are three numbers and not two.
pub const WARMUP: u32 = 3;
pub const ITERS: u32 = 50;
pub const REPS: u32 = 5;

/// What a repetition measures.
///
/// Two numbers and not one, because a device can make one of them constant.
/// `Latency` separates the repeated executions with a barrier, so `n` of them
/// cost `n` times one; `Stream` does not, so the device may overlap them and
/// what comes out is a throughput. On CUDA the two coincide - a stream
/// serializes launches at no cost - and on MoltenVK they differ by an order of
/// magnitude, because a Vulkan barrier there ends the Metal encoder and costs
/// more than any kernel this repo generates. A sweep that only knew the first
/// number would report, on that machine, that no schedule differs from any
/// other.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Mode {
    Latency,
    Stream,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Latency => "latency (barrier between runs)",
            Mode::Stream => "stream (runs the device may overlap)",
        }
    }
}

/// One timing of one candidate on one shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    /// Median of the repetitions, in microseconds.
    pub median_us: f64,
    /// Relative spread of the repetitions, `(max − min) / median`. It is the
    /// **noise floor of this measurement**, and the reason it is carried beside
    /// the median rather than averaged away: a gain below it is not a gain, and
    /// The tool refuses one.
    pub spread: f64,
    pub groups: [u32; 3],
}

/// Times a prepared session: `reps` repetitions of (`warmup` uncounted runs,
/// `iters` counted runs), reduced to a median and a spread.
///
/// Three numbers because they answer three different noises. `iters` amortizes
/// the fixed submission and fence cost, which on macOS is a several-millisecond
/// quantum that would otherwise be the whole measurement - that is
/// `Session::time`'s own reason. `warmup` pays lazy pipeline compilation and the
/// clock ramp once. `reps` is what a sweep adds: a scheduler hiccup inside one
/// submission moves a mean and cannot be seen in it, and the median of several
/// submissions is what does not move. The spread of those repetitions is then a
/// measured noise floor rather than a threshold someone chose.
pub fn time(
    session: &AnySession<'_>,
    mode: Mode,
    reps: u32,
    warmup: u32,
    iters: u32,
) -> Result<Sample, SweepError> {
    let mut runs = Vec::with_capacity(reps.max(1) as usize);
    for _ in 0..reps.max(1) {
        let d = match mode {
            Mode::Latency => session.time(warmup, iters)?,
            Mode::Stream => session.time_stream(warmup, iters)?,
        };
        runs.push(d.as_secs_f64() * 1e6);
    }
    runs.sort_by(f64::total_cmp);
    let median = runs[runs.len() / 2];
    let spread = if median > 0.0 {
        (runs[runs.len() - 1] - runs[0]) / median
    } else {
        f64::INFINITY
    };
    Ok(Sample {
        median_us: median,
        spread,
        groups: session.groups(),
    })
}

/// Lowers, emits, compiles and builds one (kernel, schedule) pair on `target`.
///
/// The parameter header is the committed one, extended in memory when the
/// artifact under arbitration has no struct in it - which is the normal case
/// here and the point of the short loop: a geometry that is the *question*
/// cannot already be in a generated header. The extension is derived from
/// `shader_params_layout`, the same function the committed header is built
/// from, so the ABI stays a derivation and never a second hand-written copy.
pub fn build<'g>(
    gpu: &'g AnyGpu,
    target: Backend,
    kernel: &ValidatedKernel,
    schedule: Schedule,
) -> Result<Built<'g>, SweepError> {
    let lowered = rir_lower::lower(kernel, schedule)?;
    let source = match target {
        Backend::Vulkan => emit_vulkan(&lowered),
        Backend::Cuda => emit_cuda(&lowered),
    }?;
    let manifest = Manifest::from_json(&emit_manifest(&lowered, None))?;
    let artifact = rir_emit::artifact_name(&lowered);
    let params_header = params_header(&lowered, &artifact)?;
    let pipeline = gpu.build(
        &manifest,
        &Artifact {
            artifact: &artifact,
            source: &source,
            params_header: &params_header,
        },
    )?;
    Ok(Built {
        pipeline,
        manifest,
        lowered,
    })
}

/// The committed `rir_kernel_params.h`, plus the struct of an artifact the
/// table does not carry.
fn params_header(lowered: &LoopKernel, artifact: &str) -> Result<String, SweepError> {
    let mut header = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../generated/rir/registry/rir_kernel_params.h"),
    )?;
    if header.contains(&format!("rir_{artifact}_params;")) {
        return Ok(header);
    }
    header.push_str(&format!("\ntypedef struct rir_{artifact}_params {{\n"));
    for (name, ty) in rir_emit::manifest::shader_params_layout(lowered) {
        let c_ty = match ty {
            "float" => "float   ",
            "int" => "int32_t ",
            _ => "uint32_t",
        };
        header.push_str(&format!("    {c_ty} {name};\n"));
    }
    header.push_str(&format!("}} rir_{artifact}_params;\n"));
    Ok(header)
}
