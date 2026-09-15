//! RIR - a closed-domain, ahead-of-time source-to-source kernel compiler.
//!
//! This crate contains the **semantic IR** (ADR-1):
//! a pure tensor SSA graph with no notion of threads, blocks, shared memory,
//! or barriers. Lowering to Loop IR lives in `rir-lower`; text emitters live
//! in `rir-emit`.
//!
//! Boundary (ADR-4 section 2): RIR compiles a kernel and its contract. ggml owns the
//! graph, tensors, and buffer lifetimes. None of these crates is linked into
//! the runtime binary.

pub mod autodiff;
pub mod backend;
pub mod blocklayout;
pub mod builder;
pub mod catalog;
pub mod ids;
pub mod ir;
pub mod layout;
pub mod manifest;
pub mod plan;
pub mod quant;
pub mod quant_table;
pub mod supports;
pub mod types;
pub mod validate;

pub use autodiff::{AutodiffError, derive_backward};
pub use backend::{Backend, GpuBackend};
pub use blocklayout::{BitPlan, BlockLayout, LutId, MinTerm, Payload, ScalePlan, SubScalePacking};
pub use builder::KernelBuilder;
pub use ids::{IrId, MAX_IDS};
pub use ir::*;
pub use layout::*;
pub use plan::{DispatchPlan, PlanError};
pub use quant::{QuantError, dequantize_row, random_block_bytes};
pub use quant_table::{BOTH, OUT_PROD_ONLY, QUANT_FORMATS, QuantFormat, QuantType, quant_formats};
pub use supports::{RejectReason, TensorDesc, supports_op};
pub use types::*;
pub use validate::{ValidateError, ValidatedKernel, validate};
