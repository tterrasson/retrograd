//! Vocabulary and contracts shared by the agent stack.
//!
//! Everything here is data or a trait: no HTTP client, no subprocess, no
//! tokenizer, and - the property this crate exists to hold - **no dependency on
//! the training engine**. Turning a trajectory into a `TrainSequence` lives in
//! `retrograd-agent`, which is the only crate that links the C++/FFI bridge.
//! That is what lets an environment, a container backend, a tool or a judge be
//! compiled and tested without a model.
//!
//! The split every implementation in the stack must respect:
//!
//! - an **action that failed** is an observation - `Ok` with `is_error` set,
//!   which the policy reads and reacts to;
//! - a **broken world** is `Err`, and it costs the trajectory.
//!
//! Blurring the two lets a rollout continue, be scored and be trained against a
//! world that cannot answer. See [`Environment::step`] and [`Sandbox::exec`].

pub mod config;
pub mod env;
pub mod error;
pub mod interrupt;
pub mod sandbox;
pub mod scenario;
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
pub use tools::{ToolCall, ToolProvider, ToolResult, ToolSpec};
pub use trajectory::{Message, Role, Step, StepKind, Trajectory, TrajectoryGroup};
