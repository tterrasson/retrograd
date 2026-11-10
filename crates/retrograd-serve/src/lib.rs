//! `retrograd-serve`: the standalone inference server, and the code
//! `retrograd serve` delegates to.
//!
//! One process serves one base model plus any number of its LoRA checkpoints
//! under their own ids (`base` for the model alone). The weights are served by
//! the step-wise inference engine ([`retrograd_infer`]) through the session of
//! [`retrograd_openai`]: several chat requests batch in one decode, and
//! switching checkpoints activates the adapter instead of reloading the base.
//!
//! The binary is `retrograd-infer`; this crate's dependency tree carries no
//! training, agent or run-config crate.

pub mod config;
pub mod server;
pub mod source;

pub use config::{AdapterDecl, ConfigError, ServeConfig};
pub use server::{EngineStatsDto, LoadOutcome, Serve, ServeError, serve};
pub use source::AdapterSource;
// Re-exported so the delegating CLI names the admission mode without depending
// on the policy crate itself.
pub use retrograd_infer::AdmissionMode;
pub use retrograd_infer::experts::{DEFAULT_ROUTE_SAMPLE_EVERY, ExpertHistoryOptions};
pub use retrograd_openai::SessionOptions;
