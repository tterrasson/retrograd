//! rir-runtime - execute a generated kernel on a real Vulkan device.
//!
//! The runtime counterpart of the AOT pipeline: `rir-gen` writes `kernel.comp`
//! and `manifest.vulkan.json`; this crate takes them as-is and builds a compute
//! pipeline. **Nothing is recomputed here** - binding order,
//! push-constant offsets, workgroup size, workgroup count, and
//! required features all come from the manifest. If the runtime had to infer
//! anything, the manifest would be incomplete.
//!
//! Like the other `rir-*` crates, it is **outside the execution path** of
//! `retrograd-*`: nothing depends on it; it validates generated code against
//! the oracle on real hardware.
//!
//! Anything that may be missing is a typed, recoverable error, never a panic:
//! no `libvulkan` (`NoLoader`), no GPU (`NoDevice`), no GLSL compiler
//! (`NoCompiler`), or a device lacking kernel requirements
//! (`MissingFeature`). A test caller therefore knows *why* it skips.
//!
//! ```no_run
//! use rir_runtime::{Arg, Gpu, Manifest, Values};
//!
//! let dir = std::path::Path::new("generated/rir/l2_norm_back");
//! let manifest = Manifest::load(dir)?;
//! let spirv = rir_runtime::compile_glsl(&dir.join("kernel.comp"))?;
//!
//! let gpu = Gpu::open()?;                       // Err(NoDevice) without a GPU
//! let pipeline = gpu.build(&manifest, &spirv)?; // Err(MissingFeature) if insufficient
//!
//! let (n_row, n_col) = (4usize, 13usize);
//! let (dz, x) = (vec![1f32; n_row * n_col], vec![2f32; n_row * n_col]);
//!
//! let mut values = Values::new();
//! values.f32("eps", 1e-6).u32("n_row", 4).u32("n_col", 13);
//! for arg in ["dz", "x", "dx"] {
//!     values.strides(arg, &[4, 4 * n_col]);      // ggml strides, in bytes
//! }
//! // … one (name, value) pair per manifest `push_constants` entry.
//!
//! let mut dx = vec![0f32; n_row * n_col];
//! pipeline.run(&mut [Arg::input(&dz), Arg::input(&x), Arg::output(&mut dx)], &values)?;
//! # Ok::<(), rir_runtime::RuntimeError>(())
//! ```

pub mod any;
#[cfg(feature = "cuda")]
pub mod cuda;
mod device;
mod manifest;
mod spirv;

pub use device::{Gpu, Pipeline, Plan, PlanArg, PlanSession, Session};
pub use manifest::{Binding, GridAxis, Manifest, ManifestReader, PushConstant, Scalar, Values};
pub use spirv::compile_glsl;

/// An argument bound to a binding, in manifest order. Bytes are raw: this is the
/// ggml contract (byte strides), not a typed view.
pub enum Arg<'a> {
    In(&'a [u8]),
    Out(&'a mut [u8]),
}

impl<'a> Arg<'a> {
    /// Binds a typed slice for reading (f32, quantized u8…).
    pub fn input<T: Pod>(data: &'a [T]) -> Self {
        Arg::In(as_bytes(data))
    }

    /// Binds a typed slice for writing; it is rewritten after dispatch with the
    /// shader output.
    pub fn output<T: Pod>(data: &'a mut [T]) -> Self {
        Arg::Out(as_bytes_mut(data))
    }

    fn len(&self) -> usize {
        match self {
            Arg::In(d) => d.len(),
            Arg::Out(d) => d.len(),
        }
    }
}

/// Element types accepted by a binding: machine scalars without padding, bit
/// invariants, or references.
///
/// `Copy` was insufficient in both directions for two distinct reasons. On
/// input, a `T: Copy` may contain uninitialized padding (a `#[repr(Rust)]`
/// struct), and reading it as bytes is UB. Output is worse: the shader writes
/// **arbitrary** bytes into the slice, so a `bool`, `char`, enum, or `Copy`
/// reference could receive an invalid pattern through a fully safe API.
///
/// The trait is therefore sealed to types satisfying both guarantees: every
/// byte pattern is a valid value (output), and every value is fully initialized
/// (input). Adding a type here asserts both properties - not merely `Copy`.
pub trait Pod: sealed::Sealed + Copy + 'static {}

mod sealed {
    pub trait Sealed {}
}

macro_rules! impl_pod {
    ($($t:ty),* $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl Pod for $t {}
    )*};
}

impl_pod!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);

fn as_bytes<T: Pod>(data: &[T]) -> &[u8] {
    // SAFETY: `T: Pod` is a machine scalar without padding, so every byte is
    // initialized and readable as `u8`. The length is the exact slice size,
    // `u8` alignment is 1 (always lower than `T`), and the view borrows `data`,
    // so it cannot outlive it.
    unsafe { std::slice::from_raw_parts(data.as_ptr().cast(), std::mem::size_of_val(data)) }
}

fn as_bytes_mut<T: Pod>(data: &mut [T]) -> &mut [u8] {
    // SAFETY: same conditions as for reading, plus the output-specific one,
    // dispatch copies arbitrary bytes into this view, and `T: Pod` guarantees
    // every byte pattern is a valid `T`. Exclusive borrowing prevents another
    // view from observing the slice while it is rewritten.
    unsafe { std::slice::from_raw_parts_mut(data.as_mut_ptr().cast(), std::mem::size_of_val(data)) }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    /// No loadable `libvulkan` on this machine.
    #[error("libvulkan not found: {0}")]
    NoLoader(String),
    /// Loader present, but no usable compute device.
    #[error("no Vulkan compute device")]
    NoDevice,
    /// Neither `glslc` nor `glslangValidator` is in PATH.
    #[error("no GLSL compiler (glslc or glslangValidator) in PATH")]
    NoCompiler,
    /// The device cannot provide what the manifest requires.
    #[error("device does not provide '{feature}' ({detail})")]
    MissingFeature { feature: String, detail: String },
    /// The manifest is unreadable or does not describe this backend.
    #[error("invalid manifest: {0}")]
    BadManifest(String),
    /// A push-constant value declared by the manifest was not supplied or has
    /// the wrong type.
    #[error("push constant '{name}' declared by the manifest but not supplied")]
    MissingValue { name: String },
    /// As many bound arguments as bindings, in the same order.
    #[error("{got} bound arguments, {expected} manifest bindings")]
    ArgCountMismatch { expected: usize, got: usize },
    /// Bound-argument direction does not match the binding: an input bound to
    /// an output would never be read back, while an output bound to an input
    /// would give the shader an uninitialized buffer. Both would return an
    /// incorrect result **without** failing.
    #[error("binding '{binding}': expected {expected}")]
    AccessMismatch {
        binding: String,
        expected: &'static str,
    },
    /// The bound argument is smaller than the byte range that extents and
    /// supplied strides make the shader address.
    #[error("binding '{binding}': {got} bytes bound, dispatch addresses {need}")]
    BufferTooSmall {
        binding: String,
        need: usize,
        got: usize,
    },
    /// `Session::read_outputs` received a slice of a different size from the one
    /// bound by `prepare`. Device buffers were sized there: a longer slice would
    /// read beyond the mapping, while a shorter one would silently return a
    /// truncated result.
    #[error("binding '{binding}': {got} bytes read back, {expected} bound during preparation")]
    ArgLenMismatch {
        binding: String,
        expected: usize,
        got: usize,
    },
    /// GLSL → SPIR-V, or CUDA → shared object, compilation failed.
    #[error("compilation: {0}")]
    Compile(String),
    /// No usable CUDA toolkit: `nvcc` absent, or unable to compile anything on
    /// this machine. Recoverable, like `NoLoader`.
    #[error("no usable CUDA toolkit: {0}")]
    NoToolkit(String),
    /// A symbol the generated code was expected to export is absent from the
    /// object that was just compiled.
    #[error("symbol '{name}' absent from the compiled object ({detail})")]
    MissingSymbol { name: String, detail: String },
    /// A CUDA runtime call failed.
    #[error("{op} failed: {detail}")]
    Cuda { op: &'static str, detail: String },
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    /// A Vulkan call failed.
    #[error("{op} failed (VkResult {code})")]
    Vulkan { op: &'static str, code: i32 },
    /// A previous dispatch in this session never returned: the device may still
    /// read its buffers, so nothing can be resubmitted or freed.
    #[error("a previous dispatch in this session has not completed")]
    InFlight,
}

/// True if the error means "this machine cannot execute the test" rather than
/// "generated code is wrong." Device parity tests use this to skip cleanly
/// instead of failing on a machine without a GPU.
pub fn is_unavailable(e: &RuntimeError) -> bool {
    matches!(
        e,
        RuntimeError::NoLoader(_)
            | RuntimeError::NoDevice
            | RuntimeError::NoCompiler
            | RuntimeError::NoToolkit(_)
            | RuntimeError::MissingFeature { .. }
    )
}
