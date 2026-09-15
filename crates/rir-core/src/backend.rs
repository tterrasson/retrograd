//! Which backends exist.
//!
//! The enums live in the leaf crate because three things name a backend and
//! only one of them schedules: the schedule table (`rir-lower`), the emitters
//! (`rir-emit`) and the manifest every generated kernel publishes, which
//! `rir-runtime` reads without ever seeing a schedule. The *tables* - which
//! backends a family is scheduled on, which it refuses in writing - stay in
//! `rir-lower::schedule::backend`, where the families are.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
    Vulkan,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
            Backend::Vulkan => "vulkan",
        }
    }

    /// This backend seen as a GPU, or `None` for the CPU. The inverse of
    /// `GpuBackend::backend`, and the only way back: a `Backend` read from a
    /// lowered kernel can be matched against the GPU domain without a
    /// `unreachable!` arm.
    pub fn gpu(self) -> Option<GpuBackend> {
        match self {
            Backend::Cpu => None,
            Backend::Cuda => Some(GpuBackend::Cuda),
            Backend::Metal => Some(GpuBackend::Metal),
            Backend::Vulkan => Some(GpuBackend::Vulkan),
        }
    }
}

/// A backend that has a grid, workgroups and collectives - everything the
/// `gpu_*` schedule constructors decide about.
///
/// It exists so that a schedule carrying a workgroup shape cannot be built for
/// the CPU: `gpu_grid(Backend::Cpu, [256,1,1])` is not a value that fails
/// validation, it is a value that does not typecheck. The schedule table then
/// names its backends as **data** (`GPU_TARGETS`) instead of expressing them as
/// the conjunction of one `push` per backend, which is what let a kernel keep
/// CPU-only coverage on one backend without a word.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuBackend {
    Cuda,
    Metal,
    Vulkan,
}

impl GpuBackend {
    /// Every GPU backend the domain knows. `GPU_TARGETS` plus `GPU_REFUSED`
    /// must cover it exactly, which is what makes a refusal a statement rather
    /// than a missing line.
    pub const ALL: [GpuBackend; 3] = [GpuBackend::Cuda, GpuBackend::Metal, GpuBackend::Vulkan];

    pub fn backend(self) -> Backend {
        match self {
            GpuBackend::Cuda => Backend::Cuda,
            GpuBackend::Metal => Backend::Metal,
            GpuBackend::Vulkan => Backend::Vulkan,
        }
    }

    pub fn name(self) -> &'static str {
        self.backend().name()
    }
}

impl From<GpuBackend> for Backend {
    fn from(g: GpuBackend) -> Self {
        g.backend()
    }
}
