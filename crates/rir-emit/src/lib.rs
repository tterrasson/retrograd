//! RIR emitters (ADR-2 section 8).
//!
//! **Golden rule: emitters make no decisions.** Each emitter prints `Stmt` as
//! text. Any heuristic here belongs in the schedule or lowering instead. An
//! `if backend ==...` in an emitter is a regression.
//!
//! Available emitters are standalone Rust for CPU, compute GLSL for Vulkan,
//! Metal Shading Language for Metal, and CUDA C++ for CUDA.
//! A `Stmt` an emitter cannot print is an explicit `EmitError`, never a silent
//! approximation - which is what the CUDA emitter's v1 subset rests on.

pub mod cpu;
pub mod cuda;
pub mod dialect;
pub mod integration;
pub mod manifest;
pub mod metal;
pub(crate) mod printer;
pub mod quant;
pub mod registry;
pub mod vulkan;

pub use cpu::emit_cpu;
pub use cuda::emit_cuda;
pub use integration::{
    ArgSource, BackendPolicy, DomainAssumption, DomainRestriction, GgmlBackend, IntegrationSpec,
    ParamSpec,
};
pub use manifest::{
    KernelNeeds, artifact_files, artifact_name, emit_manifest, entrypoint, kernel_needs,
    shader_params_layout, variant_id,
};
pub use metal::emit_metal;
pub use quant::emit_ggml_header;
pub use registry::{RegistryError, emit_params_header, emit_registry};
pub use vulkan::emit_vulkan;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum EmitError {
    /// Loop IR contains a form this backend cannot emit yet.
    #[error("{backend} emitter: unsupported {stmt}")]
    UnsupportedStmt {
        backend: &'static str,
        stmt: &'static str,
    },
}

#[cfg(test)]
pub mod testkit;
