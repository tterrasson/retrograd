//! Vocabulary and contracts shared by the agent stack.
//!
//! Everything here is data or a trait: no HTTP client, no subprocess, no
//! tokenizer, and - the property this crate exists to hold - **no dependency
//! on the training engine by default**. Turning a trajectory into a
//! `TrainSequence` lives in `retrograd-agent`, which is the crate that links
//! the C++/FFI bridge. That is what lets an environment, a container backend,
//! a tool or a judge be compiled and tested without a model. The one
//! exception is opt-in: [`template_parser`] runs a template-derived grammar
//! through the C ABI and sits behind the `template-parser` feature, which the
//! crates that render with a model's own template (`retrograd-agent`,
//! `retrograd-openai`) enable.
//!
//! The split every implementation in the stack must respect:
//!
//! - an **action that failed** is an observation - `Ok` with `is_error` set,
//!   which the policy reads and reacts to;
//! - a **broken world** is `Err`, and it costs the trajectory.
//!
//! Blurring the two lets a rollout continue, be scored and be trained against a
//! world that cannot answer. See [`Environment::step`] and [`Sandbox::exec`].

pub mod chat;
pub mod config;
pub mod env;
pub mod error;
pub mod interrupt;
pub mod parser;
pub mod rendering;
pub mod sandbox;
pub mod scenario;
#[cfg(feature = "template-parser")]
pub mod template_parser;
pub mod text;
pub mod tools;
pub mod trajectory;

pub use config::{AgentConfig, AgentGrpoConfig, JudgeFailurePolicy};
pub use env::{EnvState, Environment, EnvironmentFactory, StepOutcome};
pub use error::{Error, FailureKind, Result};
pub use sandbox::{
    DirEntry, ExecOutput, ExecRequest, FileKind, Lease, Sandbox, SandboxLimits, SandboxOwner,
    SandboxProvider, relative_path,
};
pub use scenario::{RolloutLimits, Scenario, TruncationPolicy};
#[cfg(feature = "template-parser")]
pub use template_parser::TemplateToolCallParser;
pub use tools::{ToolCall, ToolProvider, ToolResult, ToolSpec};
pub use trajectory::{Message, Role, Step, StepKind, Trajectory, TrajectoryGroup};
