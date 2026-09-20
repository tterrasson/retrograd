//! Vulkan device and compute pipeline, built **from the manifest**.
//!
//! The crate's only Vulkan-facing code. `Gpu` owns the loader, instance, device,
//! queue, and command pool - shared between kernels; `Pipeline` owns what
//! belongs to one kernel - SPIR-V module, descriptor layout, and pipeline.
//! Buffers live for one `run`: the RIR contract is stateless.
//!
//! The Vulkan subset used is deliberately minimal (host-visible memory, one
//! dispatch, one wait): this runtime primarily validates the **correctness** of
//! generated code.
//!
//! It also provides the short measurement loop,
//! which justifies `Session`: `Pipeline::run` uploads, dispatches, and reads back
//! in one call, so timing it would include two host copies in addition to the
//! shader. `Pipeline::prepare` allocates and uploads once;
//! `Session::dispatch` only submits. The number remains an end-to-end host time
//! (including submission and fence wait): it compares two schedules of the same
//! kernel and does not decide promotion - level 2, `scripts/test-rir.sh`, does.

pub mod buffers;
pub mod instance;
pub mod pipeline;
pub mod plan;
pub mod session;

pub use instance::Gpu;
pub use pipeline::Pipeline;
pub use plan::{Plan, PlanArg, PlanSession};
pub use session::Session;

pub(crate) use crate::manifest::BoundArg;
pub(crate) use buffers::Buffer;
pub(crate) use instance::vk_err;
pub(crate) use pipeline::PartialPipeline;
pub(crate) use session::{DISPATCH_TIMEOUT_NS, DispatchError};
