//! Metal device and kernel, built **from the manifest** - the short loop's
//! third backend, and the one Apple machines can actually drive.
//!
//! The carrier is the one the fork already uses: the generated `kernel.metal`
//! is compiled **at run time** by the device (`newLibraryWithSource`), so no
//! offline Metal toolchain is needed, and the binding convention is the fork's
//! own (`ggml_metal_op_rir_dispatch`): tensors at buffer indices `0..n` in
//! manifest order, the params struct right after them, threadgroups of the
//! manifest's `workgroup`, as many as `Manifest::groups` derives. Nothing is
//! recomputed here - if this module had to infer a binding or a geometry, the
//! manifest would be incomplete.
//!
//! The interface is the Vulkan and CUDA one: `Gpu::open`, `Gpu::build`,
//! `Pipeline::prepare`, `Session::dispatch`/`time`/`read_outputs`, with the
//! checks above the device - binding count, direction, addressed range,
//! constant block, grid - on `Manifest` and shared by all three.
//!
//! Buffers are `StorageModeShared`: on the unified memory of Apple silicon that
//! is device memory the host can also map, so upload and read-back are copies
//! into the mapping and timing measures the kernel rather than a bus.

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily, MTLLanguageVersion, MTLLibrary,
    MTLResourceOptions, MTLSize,
};
use rir_core::Backend;
use std::{cell::Cell, marker::PhantomData};

use crate::manifest::{BoundArg, Manifest, ManifestDispatch};
use crate::{Arg, RuntimeError, Values};

/// The language version a `metal>=x.y` feature names, as the compiler takes it.
///
/// `None` for anything else, including a version newer than this runtime
/// knows: compiling at a version the bindings cannot spell would be compiling a
/// different contract from the one the manifest claims.
fn language_version(spelling: &str) -> Option<MTLLanguageVersion> {
    Some(match spelling {
        "2.0" => MTLLanguageVersion::Version2_0,
        "2.1" => MTLLanguageVersion::Version2_1,
        "2.2" => MTLLanguageVersion::Version2_2,
        "2.3" => MTLLanguageVersion::Version2_3,
        "2.4" => MTLLanguageVersion::Version2_4,
        "3.0" => MTLLanguageVersion::Version3_0,
        "3.1" => MTLLanguageVersion::Version3_1,
        "3.2" => MTLLanguageVersion::Version3_2,
        _ => return None,
    })
}

/// An Objective-C error, in the words it carries.
fn describe(error: &objc2_foundation::NSError) -> String {
    error.localizedDescription().to_string()
}

/// The device and its command queue.
pub struct Gpu {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    name: String,
    /// `simd_sum` and friends, under the rule the fork applies before admitting
    /// a RIR variant on Metal: Apple7 or Metal3 family.
    simdgroup_reduction: bool,
}

impl Gpu {
    /// Opens the system's default device.
    ///
    /// Every failure is typed and recoverable (`NoMetal`): a caller knows *why*
    /// it skips, which is the property the two other backends have.
    pub fn open() -> Result<Gpu, RuntimeError> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| RuntimeError::NoMetal("no default Metal device".into()))?;
        let queue = device
            .newCommandQueue()
            .ok_or_else(|| RuntimeError::NoMetal("the device returned no command queue".into()))?;
        let simdgroup_reduction = device.supportsFamily(MTLGPUFamily::Apple7)
            || device.supportsFamily(MTLGPUFamily::Metal3);
        Ok(Gpu {
            name: device.name().to_string(),
            device,
            queue,
            simdgroup_reduction,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Threadgroup memory one threadgroup may declare.
    pub fn max_threadgroup_memory(&self) -> usize {
        self.device.maxThreadgroupMemoryLength()
    }

    /// Checks the manifest's device-level features **before** compiling
    /// anything, and returns the language version to compile at.
    ///
    /// As on the two other backends, **an unknown feature is a rejection, not a
    /// shrug**: a manifest newer than this runtime carries a requirement this
    /// runtime does not know how to evaluate. `simdgroup_size>=N` is only
    /// parsed here - the SIMD width is a property of the compiled pipeline and
    /// is checked on it, by `check_pipeline`.
    pub fn check_features(&self, m: &Manifest) -> Result<MTLLanguageVersion, RuntimeError> {
        let mut version = None;
        for f in &m.features {
            let (ok, detail) = if let Some(v) = f.strip_prefix("metal>=") {
                match language_version(v) {
                    Some(lv) => {
                        version = Some(lv);
                        (true, String::new())
                    }
                    None => (false, "language version unknown to this runtime".into()),
                }
            } else if f == "simdgroup_reduction" {
                (
                    self.simdgroup_reduction,
                    "neither Apple7 nor Metal3 family".into(),
                )
            } else if let Some(v) = f.strip_prefix("simdgroup_size>=") {
                (v.parse::<usize>().is_ok(), "unreadable lane count".into())
            } else {
                (false, "feature unknown to this runtime".into())
            };
            if !ok {
                return Err(RuntimeError::MissingFeature {
                    feature: f.clone(),
                    detail,
                });
            }
        }
        // Every Metal manifest publishes its language version: the
        // compile-time contract is that version, and a manifest without one
        // says nothing about which language its source was written in.
        version.ok_or_else(|| {
            RuntimeError::BadManifest("metal manifest without a metal>= feature".into())
        })
    }

    /// The pipeline-level half: what only a compiled pipeline can say.
    ///
    /// The threadgroup must fit, and a variant reducing through a SIMD-group
    /// also needs the collective's width - the same `needed` the fork's device
    /// check computes before encoding a RIR variant. Threadgroup memory is read
    /// off the pipeline rather than recomputed from the source.
    fn check_pipeline(
        &self,
        m: &Manifest,
        state: &ProtocolObject<dyn MTLComputePipelineState>,
    ) -> Result<(), RuntimeError> {
        let threads = m.workgroup.iter().try_fold(1usize, |n, &w| {
            n.checked_mul(w as usize).ok_or_else(|| {
                RuntimeError::BadManifest("workgroup thread count overflows usize".into())
            })
        })?;
        let width = state.threadExecutionWidth();
        for f in &m.features {
            if let Some(Ok(lanes)) = f.strip_prefix("simdgroup_size>=").map(str::parse::<usize>)
                && width < lanes
            {
                return Err(RuntimeError::MissingFeature {
                    feature: f.clone(),
                    detail: format!("SIMD-groups of {width} lanes"),
                });
            }
        }
        let max = state.maxTotalThreadsPerThreadgroup();
        if max < threads {
            return Err(RuntimeError::MissingFeature {
                feature: format!("workgroup {:?}", m.workgroup),
                detail: format!("{max} threads per threadgroup for this pipeline"),
            });
        }
        let per_axis = self.device.maxThreadsPerThreadgroup();
        let limits = [per_axis.width, per_axis.height, per_axis.depth];
        if m.workgroup.iter().zip(limits).any(|(&w, l)| w as usize > l) {
            return Err(RuntimeError::MissingFeature {
                feature: format!("workgroup {:?}", m.workgroup),
                detail: format!("threadgroup bounded by {limits:?}"),
            });
        }
        let shared = state.staticThreadgroupMemoryLength();
        if shared > self.max_threadgroup_memory() {
            return Err(RuntimeError::MissingFeature {
                feature: format!("threadgroup memory {shared} bytes"),
                detail: format!("{} bytes per threadgroup", self.max_threadgroup_memory()),
            });
        }
        Ok(())
    }

    /// Compiles a generated `kernel.metal` at the language version its manifest
    /// claims and builds the manifest's entrypoint.
    ///
    /// Compiling at the claimed version rather than the device's default is
    /// what makes this check the published `metal>=x.y` rather than merely the
    /// syntax - the same choice the artefact test makes with the offline
    /// compiler.
    pub fn build(&self, manifest: &Manifest, source: &str) -> Result<Pipeline<'_>, RuntimeError> {
        if manifest.backend != Backend::Metal {
            return Err(RuntimeError::BadManifest(format!(
                "backend '{}': this pipeline executes metal",
                manifest.backend.name()
            )));
        }
        let version = self.check_features(manifest)?;
        let state = autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            options.setLanguageVersion(version);
            let library = self
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&options))
                .map_err(|e| RuntimeError::Compile(describe(&e)))?;
            let function = library
                .newFunctionWithName(&NSString::from_str(&manifest.entrypoint))
                .ok_or_else(|| RuntimeError::MissingSymbol {
                    name: manifest.entrypoint.clone(),
                    detail: "absent from the compiled library".into(),
                })?;
            self.device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| RuntimeError::Metal {
                    op: "newComputePipelineState",
                    detail: describe(&e),
                })
        })?;
        self.check_pipeline(manifest, &state)?;
        Ok(Pipeline {
            gpu: self,
            manifest: manifest.clone(),
            state,
        })
    }

    /// A shared buffer of at least `len` bytes. A zero-length argument stays
    /// legal for the kernel - no thread reads it - but Metal allocates nothing
    /// to bind for zero bytes.
    fn buffer(&self, len: usize) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, RuntimeError> {
        self.device
            .newBufferWithLength_options(len.max(4), MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| RuntimeError::Metal {
                op: "newBufferWithLength",
                detail: format!("{len} bytes"),
            })
    }
}

/// Copies `data` into a shared buffer at least as long.
fn upload(buffer: &ProtocolObject<dyn MTLBuffer>, data: &[u8]) {
    debug_assert!(buffer.length() >= data.len());
    // SAFETY: a `StorageModeShared` buffer maps `length()` bytes at
    // `contents()`, which `buffer()` allocated at least `data.len()` long, and
    // no command buffer touches it while the host writes: every dispatch waits
    // for completion before returning.
    unsafe {
        std::ptr::copy_nonoverlapping(
            data.as_ptr(),
            buffer.contents().as_ptr().cast::<u8>(),
            data.len(),
        );
    }
}

/// A kernel, reusable across `run` calls.
pub struct Pipeline<'g> {
    gpu: &'g Gpu,
    manifest: Manifest,
    state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
}

impl Pipeline<'_> {
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// A complete dispatch: upload inputs, execute, read outputs back into bound
    /// slices. Synchronous.
    pub fn run(&self, args: &mut [Arg], values: &Values) -> Result<(), RuntimeError> {
        let session = self.prepare(args, values)?;
        session.dispatch()?;
        session.read_outputs(args)
    }

    /// The first half of `run`: checks, allocation, upload, and resolution of
    /// grid and params. Its result can be dispatched any number of times without
    /// touching host memory - which is what makes `Session::time` measurable.
    pub fn prepare(&self, args: &[Arg], values: &Values) -> Result<Session<'_>, RuntimeError> {
        let (params, grid) = self.manifest.encode_dispatch(args, values)?;
        let mut buffers = Vec::with_capacity(args.len());
        for a in args {
            let buffer = self.gpu.buffer(a.len())?;
            if let Arg::In(data) = a {
                upload(&buffer, data);
            }
            buffers.push(buffer);
        }
        // A buffer and not `setBytes`: the latter is bounded at 4 KiB, which
        // the manifest's push-constant offsets are not. The shader sees the
        // same `constant` struct either way.
        let constants = self.gpu.buffer(params.len())?;
        upload(&constants, &params);
        let bound = args
            .iter()
            .map(|a| BoundArg {
                write: matches!(a, Arg::Out(_)),
                len: a.len(),
            })
            .collect();
        Ok(Session {
            pipeline: self,
            buffers,
            constants,
            bound,
            grid,
            not_sync: PhantomData,
        })
    }
}

/// A **prepared** dispatch: buffers allocated, inputs uploaded, grid and params
/// resolved. Repeating it measures the kernel and the submission rather than
/// copies.
///
/// There is no in-flight state, unlike the two other backends: every dispatch
/// waits for its command buffer to complete or fail, and either way the device
/// has released the buffers when the wait returns.
pub struct Session<'p> {
    pipeline: &'p Pipeline<'p>,
    buffers: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    constants: Retained<ProtocolObject<dyn MTLBuffer>>,
    bound: Vec<BoundArg>,
    grid: [u32; 3],
    // Keep mappings single-caller even if Metal's bindings become Sync: a
    // concurrent read_outputs must not race dispatch's device writes. Match
    // the Vulkan/CUDA sessions' contract without a runtime lock.
    not_sync: PhantomData<Cell<()>>,
}

impl Session<'_> {
    /// Number of threadgroups the manifest derives from the params - needed to
    /// interpret a timing result.
    pub fn groups(&self) -> [u32; 3] {
        self.grid
    }

    /// Submits the dispatch and waits for completion. Reusable.
    pub fn dispatch(&self) -> Result<(), RuntimeError> {
        self.dispatch_n(1)
    }

    /// `n` dispatches in one serial compute encoder, one command buffer, one
    /// wait.
    ///
    /// A serial encoder runs each dispatch to completion before the next, so
    /// `n` executions sharing their outputs cost `n` times one - the property
    /// the Vulkan half gets from a barrier and CUDA from a single stream, and
    /// the reason a timing here reads like a timing there.
    pub(crate) fn dispatch_n(&self, n: u32) -> Result<(), RuntimeError> {
        let pipeline = self.pipeline;
        // An empty grid dispatches nothing, which Metal does not accept as a
        // dispatch: the encoder is still committed, so the call costs what an
        // empty submission costs and returns the same way.
        let empty = self.grid.contains(&0);
        let size = |v: [u32; 3]| MTLSize {
            width: v[0] as usize,
            height: v[1] as usize,
            depth: v[2] as usize,
        };
        let (groups, threads) = (size(self.grid), size(pipeline.manifest.workgroup));
        autoreleasepool(|_| {
            let commands = pipeline
                .gpu
                .queue
                .commandBuffer()
                .ok_or(RuntimeError::Metal {
                    op: "commandBuffer",
                    detail: "the queue returned none".into(),
                })?;
            let encoder = commands
                .computeCommandEncoder()
                .ok_or(RuntimeError::Metal {
                    op: "computeCommandEncoder",
                    detail: "the command buffer returned none".into(),
                })?;
            encoder.setComputePipelineState(&pipeline.state);
            // SAFETY: each buffer is a live allocation this session owns, bound
            // at offset 0 at the index the generated signature declares for it
            // (`[[buffer(i)]]` in manifest order, the params struct after).
            unsafe {
                for (i, b) in self.buffers.iter().enumerate() {
                    encoder.setBuffer_offset_atIndex(Some(b), 0, i);
                }
                encoder.setBuffer_offset_atIndex(Some(&self.constants), 0, self.buffers.len());
            }
            if !empty {
                for _ in 0..n.max(1) {
                    encoder.dispatchThreadgroups_threadsPerThreadgroup(groups, threads);
                }
            }
            encoder.endEncoding();
            commands.commit();
            commands.waitUntilCompleted();
            if commands.status() == MTLCommandBufferStatus::Completed {
                return Ok(());
            }
            Err(RuntimeError::Metal {
                op: "command buffer",
                detail: commands
                    .error()
                    .map(|e| describe(&e))
                    .unwrap_or_else(|| format!("status {:?}", commands.status())),
            })
        })
    }

    /// Copies output buffers back into bound slices. `args` must be the set used
    /// by `prepare` - same bindings, directions and sizes, **checked**.
    pub fn read_outputs(&self, args: &mut [Arg]) -> Result<(), RuntimeError> {
        self.pipeline.manifest.check_read_back(&self.bound, args)?;
        for (b, a) in self.buffers.iter().zip(args.iter_mut()) {
            if let Arg::Out(data) = a {
                // SAFETY: the buffer was allocated by `prepare` from this very
                // slice's length (checked just above), it maps `length()` bytes
                // at `contents()`, and the last dispatch has completed.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        b.contents().as_ptr().cast::<u8>(),
                        data.as_mut_ptr(),
                        data.len(),
                    );
                }
            }
        }
        Ok(())
    }

    /// Times `iters` dispatches and returns the **mean** per dispatch after
    /// uncounted `warmup` runs - pipeline warm-up and clock ramp are paid
    /// once, not on every measurement.
    ///
    /// All `iters` dispatches share one command buffer, so the fixed commit and
    /// wait cost is divided by `iters` instead of paid each run.
    pub fn time(&self, warmup: u32, iters: u32) -> Result<std::time::Duration, RuntimeError> {
        if warmup > 0 {
            self.dispatch_n(warmup)?;
        }
        let iters = iters.max(1);
        let start = std::time::Instant::now();
        self.dispatch_n(iters)?;
        Ok(start.elapsed() / iters)
    }

    /// The same measurement, under the name the Vulkan half gives its
    /// compatibility mode: a serial encoder has no unbarriered variant, so
    /// there is nothing different to measure. The pair exists so `AnySession`
    /// has one shape.
    pub fn time_stream(
        &self,
        warmup: u32,
        iters: u32,
    ) -> Result<std::time::Duration, RuntimeError> {
        self.time(warmup, iters)
    }
}

#[cfg(test)]
mod tests {
    use super::language_version;

    /// The spellings the emitter publishes are known, and a version this
    /// runtime cannot spell is `None` - refused, never compiled at a default.
    #[test]
    fn a_language_version_is_read_as_the_manifest_spells_it() {
        assert!(language_version("2.0").is_some());
        assert!(language_version("2.1").is_some());
        assert!(language_version("3.2").is_some());
        assert!(language_version("9.0").is_none());
        assert!(language_version("2").is_none());
        assert!(language_version("").is_none());
    }
}
