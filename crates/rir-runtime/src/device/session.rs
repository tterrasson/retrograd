//! The measurement session: buffers allocated and uploaded once, then one
//! submission per dispatch.

use std::cell::Cell;

use crate::manifest::ManifestDispatch;
use crate::{Arg, RuntimeError};

use super::*;

/// Maximum dispatch wait. Beyond this, the shader is considered non-terminating
/// - but its resources are *not* considered free: see `dispatch`.
pub(crate) const DISPATCH_TIMEOUT_NS: u64 = 10_000_000_000;

/// A **prepared** dispatch: buffers allocated, inputs uploaded, grid and push
/// constants resolved. The dispatcher no longer touches host memory, so
/// repeating it measures shader and submission rather than copies.
///
/// Outputs remain in device buffers until `read_outputs`: a timed session has no
/// reason to read them back after every run.
pub struct Session<'p> {
    pub(crate) pipeline: &'p Pipeline<'p>,
    pub(crate) buffers: Vec<Buffer>,
    /// What `prepare` bound, binding by binding: direction and exact size.
    /// `read_outputs` replays this to reject a slice different from the original:
    /// device buffers were sized here, so a longer slice would read beyond the
    /// Vulkan mapping.
    pub(crate) bound: Vec<BoundArg>,
    pub(crate) push: Vec<u8>,
    pub(crate) groups: [u32; 3],
    /// An in-flight dispatch prevents returning anything - see `DispatchError`
    /// and the `Drop` implementation below.
    pub(crate) in_flight: Cell<bool>,
}

impl Session<'_> {
    /// Number of workgroups derived by the manifest from push constants - needed
    /// to interpret a timing result.
    pub fn groups(&self) -> [u32; 3] {
        self.groups
    }

    /// Submits the dispatch and waits for completion. Reusable: nothing is consumed.
    pub fn dispatch(&self) -> Result<(), RuntimeError> {
        self.dispatch_n(1)
    }

    /// `n` executions chained in one submission - see `Pipeline::dispatch`.
    pub(crate) fn dispatch_n(&self, n: u32) -> Result<(), RuntimeError> {
        self.dispatch_runs(n, true)
    }

    fn dispatch_runs(&self, n: u32, separated: bool) -> Result<(), RuntimeError> {
        if self.in_flight.get() {
            return Err(RuntimeError::InFlight);
        }
        self.pipeline
            .dispatch(&self.buffers, &self.push, self.groups, n, separated)
            .map_err(|DispatchError { error, in_flight }| {
                self.in_flight.set(in_flight);
                error
            })
    }

    /// Copies output buffers back into bound slices. `args` must be the set used
    /// by `prepare`: same bindings, directions, and sizes - this is **checked**,
    /// not merely documented. `prepare` sized device buffers; a longer slice
    /// would copy beyond the mapping from a safe API.
    pub fn read_outputs(&self, args: &mut [Arg]) -> Result<(), RuntimeError> {
        self.pipeline.manifest.check_read_back(&self.bound, args)?;
        for (b, a) in self.buffers.iter().zip(args.iter_mut()) {
            if let Arg::Out(data) = a {
                b.read(data)?;
            }
        }
        Ok(())
    }

    /// Times `iters` dispatches and returns the **mean** time per dispatch after
    /// uncounted `warmup` runs - lazy driver compilation and GPU frequency ramp
    /// are paid once, not on every measurement.
    ///
    /// All `iters` executions fit in **one** submission: fixed submission and
    /// wait cost is divided by `iters` instead of paid each run - otherwise it
    /// dominates and measurement is not reproducible. What remains included is
    /// 1/`iters` of the host constant; the rest is the shader.
    pub fn time(&self, warmup: u32, iters: u32) -> Result<std::time::Duration, RuntimeError> {
        if warmup > 0 {
            self.dispatch_n(warmup)?;
        }
        let iters = iters.max(1);
        let start = std::time::Instant::now();
        self.dispatch_n(iters)?;
        Ok(start.elapsed() / iters)
    }

    /// The same as `time`, with the executions **not** separated by a barrier:
    /// `iters` independent runs of a kernel whose inputs no run modifies, which
    /// the device may overlap.
    ///
    /// It measures a throughput where `time` measures a latency, and it exists
    /// because on a device whose barrier costs more than the kernel - MoltenVK
    /// ends the Metal encoder on one - the latency is the barrier and moves with
    /// nothing else. Two schedules of one kernel are then separable here and
    /// indistinguishable there, which is a property of the measurement and not
    /// of the schedules.
    ///
    /// Legitimate **only** where the runs are independent: every kernel this
    /// runtime binds reads its inputs and writes a distinct output, so replaying
    /// one is idempotent. A kernel accumulating into one of its inputs would
    /// need the barrier, and would also be measuring a different thing.
    pub fn time_stream(
        &self,
        warmup: u32,
        iters: u32,
    ) -> Result<std::time::Duration, RuntimeError> {
        if warmup > 0 {
            self.dispatch_runs(warmup, false)?;
        }
        let iters = iters.max(1);
        let start = std::time::Instant::now();
        self.dispatch_runs(iters, false)?;
        Ok(start.elapsed() / iters)
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        if self.in_flight.get() {
            // The dispatch is incomplete and the device no longer responds:
            // destroying these buffers while the GPU may still read them would
            // violate their Vulkan lifetime. Leak them - bounded to the failed
            // dispatch - rather than risk a device-lost or
            // driver crash.
            for b in self.buffers.drain(..) {
                std::mem::forget(b);
            }
        }
    }
}

/// A dispatch failure, plus what the caller must know beyond the error: might
/// the GPU still be using dispatch resources? If so, nothing it references may
/// be destroyed.
pub(crate) struct DispatchError {
    pub(crate) error: RuntimeError,
    pub(crate) in_flight: bool,
}

impl From<RuntimeError> for DispatchError {
    /// Preparation failures (before any submission) put nothing in
    /// vol.
    fn from(error: RuntimeError) -> Self {
        DispatchError {
            error,
            in_flight: false,
        }
    }
}
