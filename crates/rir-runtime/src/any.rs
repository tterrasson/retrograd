//! One interface over the backends the short loop can drive.
//!
//! It exists so that `device_parity` and `device_timing` become **parameterized
//! by backend** instead of duplicated. That is worth a wrapper for a reason the
//! line count does not show: a parity test written twice is two tests that can
//! disagree about what they check, and the property this loop is for - the
//! generated code computes what the oracle computes - is the same property on
//! every device.
//!
//! Nothing is decided here. Each arm forwards to the backend's own `Gpu`,
//! `Pipeline` and `Session`, which already have the same shape: the checks above
//! the device live on `Manifest` and are called by both.
//!
//! And this machine makes the wrapper pay twice (`memory/gpu-lanes-on-this-box`):
//! the same silicon runs RIR under Vulkan and under CUDA, so the two arms can be
//! measured in one session on one shape, and a ratio that differs between them
//! cannot be blamed on the hardware.

use crate::{Arg, Manifest, RuntimeError, Values};

/// Which backend a test is running on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    Vulkan,
    Cuda,
    Metal,
}

impl Backend {
    pub fn name(self) -> &'static str {
        match self {
            Backend::Vulkan => "vulkan",
            Backend::Cuda => "cuda",
            Backend::Metal => "metal",
        }
    }

    /// The domain's name for this backend - what a manifest and a schedule
    /// carry.
    pub fn lower(self) -> rir_core::Backend {
        match self {
            Backend::Vulkan => rir_core::Backend::Vulkan,
            Backend::Cuda => rir_core::Backend::Cuda,
            Backend::Metal => rir_core::Backend::Metal,
        }
    }

    /// The same, as the GPU backend the schedule constructors take.
    pub fn gpu(self) -> rir_core::GpuBackend {
        match self {
            Backend::Vulkan => rir_core::GpuBackend::Vulkan,
            Backend::Cuda => rir_core::GpuBackend::Cuda,
            Backend::Metal => rir_core::GpuBackend::Metal,
        }
    }

    /// The generated source file of a variant on this backend - the fallback's
    /// bare name, or the variant's infix - and its manifest.
    ///
    /// Derived rather than spelled at each call site, because the naming
    /// convention is `rir_emit::artifact_files`'s and a test that restated it
    /// would be a fourth place to keep in step.
    pub fn files(self, variant: Option<&str>) -> (String, String) {
        let infix = variant.map(|v| format!(".{v}")).unwrap_or_default();
        let ext = match self {
            Backend::Vulkan => "comp",
            Backend::Cuda => "cu",
            Backend::Metal => "metal",
        };
        (
            format!("kernel{infix}.{ext}"),
            format!("manifest{infix}.{}.json", self.name()),
        )
    }

    /// The backends this build can drive. CUDA is behind a feature and Metal
    /// behind an Apple target, so a build without them must not silently report
    /// the test as passed on a backend it never touched.
    pub fn available() -> Vec<Backend> {
        let mut v = vec![Backend::Vulkan];
        if cfg!(feature = "cuda") {
            v.push(Backend::Cuda);
        }
        if cfg!(target_vendor = "apple") {
            v.push(Backend::Metal);
        }
        v
    }
}

/// What a backend needs to build a kernel: the generated source as the
/// generator wrote it, and - on CUDA - the parameter header it includes.
pub struct Artifact<'a> {
    /// `rir_<artifact>` is the launch symbol on CUDA; unused on Vulkan, whose
    /// module has one entrypoint, and on Metal, whose entrypoint the manifest
    /// names.
    pub artifact: &'a str,
    pub source: &'a str,
    /// `rir_kernel_params.h`, read by the caller from the generated registry.
    /// Empty is legitimate on Vulkan and Metal, which do not include it.
    pub params_header: &'a str,
}

pub enum AnyGpu {
    /// Boxed because an `ash::Device` carries its whole dispatch table - 1 800
    /// bytes against CUDA's hundred, which would make every `AnyGpu` that size.
    Vulkan(Box<crate::Gpu>),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::Gpu),
    #[cfg(target_vendor = "apple")]
    Metal(crate::metal::Gpu),
}

impl AnyGpu {
    pub fn open(backend: Backend) -> Result<AnyGpu, RuntimeError> {
        match backend {
            Backend::Vulkan => crate::Gpu::open().map(|g| AnyGpu::Vulkan(Box::new(g))),
            #[cfg(feature = "cuda")]
            Backend::Cuda => crate::cuda::Gpu::open().map(AnyGpu::Cuda),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => Err(RuntimeError::NoToolkit(
                "this build has no cuda feature".into(),
            )),
            #[cfg(target_vendor = "apple")]
            Backend::Metal => crate::metal::Gpu::open().map(AnyGpu::Metal),
            #[cfg(not(target_vendor = "apple"))]
            Backend::Metal => Err(RuntimeError::NoMetal("not an Apple target".into())),
        }
    }

    pub fn name(&self) -> String {
        match self {
            // Vulkan reads its name out of the device properties on each call;
            // CUDA read it once at open. One owned string either way, so the
            // caller does not have to know which.
            AnyGpu::Vulkan(g) => g.name(),
            #[cfg(feature = "cuda")]
            AnyGpu::Cuda(g) => g.name().to_string(),
            #[cfg(target_vendor = "apple")]
            AnyGpu::Metal(g) => g.name().to_string(),
        }
    }

    pub fn build(
        &self,
        manifest: &Manifest,
        artifact: &Artifact<'_>,
    ) -> Result<AnyPipeline<'_>, RuntimeError> {
        match self {
            AnyGpu::Vulkan(g) => {
                // `compile_glsl` takes a path, as the AOT pipeline calls it, so
                // the text goes back through a file. One per process and per
                // artifact: two lanes must not write each other's shader.
                let dir = std::env::temp_dir().join(format!(
                    "rir-any-{}-{}",
                    std::process::id(),
                    artifact.artifact
                ));
                std::fs::create_dir_all(&dir)?;
                let comp = dir.join("kernel.comp");
                std::fs::write(&comp, artifact.source)?;
                let spirv = crate::compile_glsl(&comp);
                let _ = std::fs::remove_dir_all(&dir);
                g.build(manifest, &spirv?).map(AnyPipeline::Vulkan)
            }
            #[cfg(feature = "cuda")]
            AnyGpu::Cuda(g) => g
                .build(
                    manifest,
                    &crate::cuda::Unit {
                        artifact: artifact.artifact,
                        source: artifact.source,
                        params_header: artifact.params_header,
                    },
                )
                .map(AnyPipeline::Cuda),
            #[cfg(target_vendor = "apple")]
            AnyGpu::Metal(g) => g.build(manifest, artifact.source).map(AnyPipeline::Metal),
        }
    }
}

pub enum AnyPipeline<'g> {
    Vulkan(crate::Pipeline<'g>),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::Pipeline<'g>),
    #[cfg(target_vendor = "apple")]
    Metal(crate::metal::Pipeline<'g>),
}

impl AnyPipeline<'_> {
    pub fn run(&self, args: &mut [Arg], values: &Values) -> Result<(), RuntimeError> {
        match self {
            AnyPipeline::Vulkan(p) => p.run(args, values),
            #[cfg(feature = "cuda")]
            AnyPipeline::Cuda(p) => p.run(args, values),
            #[cfg(target_vendor = "apple")]
            AnyPipeline::Metal(p) => p.run(args, values),
        }
    }

    pub fn prepare(&self, args: &[Arg], values: &Values) -> Result<AnySession<'_>, RuntimeError> {
        match self {
            AnyPipeline::Vulkan(p) => p.prepare(args, values).map(AnySession::Vulkan),
            #[cfg(feature = "cuda")]
            AnyPipeline::Cuda(p) => p.prepare(args, values).map(AnySession::Cuda),
            #[cfg(target_vendor = "apple")]
            AnyPipeline::Metal(p) => p.prepare(args, values).map(AnySession::Metal),
        }
    }
}

pub enum AnySession<'p> {
    Vulkan(crate::Session<'p>),
    #[cfg(feature = "cuda")]
    Cuda(crate::cuda::Session<'p>),
    #[cfg(target_vendor = "apple")]
    Metal(crate::metal::Session<'p>),
}

impl AnySession<'_> {
    pub fn groups(&self) -> [u32; 3] {
        match self {
            AnySession::Vulkan(s) => s.groups(),
            #[cfg(feature = "cuda")]
            AnySession::Cuda(s) => s.groups(),
            #[cfg(target_vendor = "apple")]
            AnySession::Metal(s) => s.groups(),
        }
    }

    pub fn dispatch(&self) -> Result<(), RuntimeError> {
        match self {
            AnySession::Vulkan(s) => s.dispatch(),
            #[cfg(feature = "cuda")]
            AnySession::Cuda(s) => s.dispatch(),
            #[cfg(target_vendor = "apple")]
            AnySession::Metal(s) => s.dispatch(),
        }
    }

    pub fn read_outputs(&self, args: &mut [Arg]) -> Result<(), RuntimeError> {
        match self {
            AnySession::Vulkan(s) => s.read_outputs(args),
            #[cfg(feature = "cuda")]
            AnySession::Cuda(s) => s.read_outputs(args),
            #[cfg(target_vendor = "apple")]
            AnySession::Metal(s) => s.read_outputs(args),
        }
    }

    pub fn time(&self, warmup: u32, iters: u32) -> Result<std::time::Duration, RuntimeError> {
        match self {
            AnySession::Vulkan(s) => s.time(warmup, iters),
            #[cfg(feature = "cuda")]
            AnySession::Cuda(s) => s.time(warmup, iters),
            #[cfg(target_vendor = "apple")]
            AnySession::Metal(s) => s.time(warmup, iters),
        }
    }

    /// Compatibility stream timing: repeated dispatches serialize shared writes.
    /// Vulkan includes a barrier; CUDA serializes launches on the same stream;
    /// Metal runs them in one serial encoder.
    pub fn time_stream(
        &self,
        warmup: u32,
        iters: u32,
    ) -> Result<std::time::Duration, RuntimeError> {
        match self {
            AnySession::Vulkan(s) => s.time_stream(warmup, iters),
            #[cfg(feature = "cuda")]
            AnySession::Cuda(s) => s.time_stream(warmup, iters),
            #[cfg(target_vendor = "apple")]
            AnySession::Metal(s) => s.time_stream(warmup, iters),
        }
    }
}
