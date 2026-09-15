//! Agentic rollout and reward backends for Retrograd.
//!
//! The crate is intentionally kept separate from the synchronous training
//! core so async transports and orchestration do not leak into `retrograd`.
//!
//! It is also the only crate of the agent stack that links the training engine:
//! the vocabulary and the contracts live in
//! [`retrograd_agent_core`], which environments, tools,
//! containers and judges build on without ever pulling in the FFI bridge.

pub mod chat;
mod grpo;
pub mod policy;
pub mod rollout;
pub mod template_parser;
pub mod train;

pub use retrograd_agent_core::trajectory;
pub use retrograd_env as env;
pub use retrograd_judge as judge;
pub use retrograd_tools as tools;

#[cfg(feature = "http-env")]
pub use env::config::Environments;
pub use env::{
    Environment, EnvironmentConfig, EnvironmentFactory, LocalConfig, SandboxEnvironment,
    SandboxEnvironmentConfig, SandboxEnvironmentFactory, SharedToolsFactory, StepOutcome,
    ToolProviderEnvironment, ToolProviderFactory,
};
pub use env::{HttpEnvironmentConfig, HttpEnvironmentFactory};
pub use grpo::{
    AgentEvalMetrics, AgentFlow, AgentRunOutcome, AgenticGrpoServices, AgenticRun, UpdateBoundary,
    UpdateHook, run_agentic_grpo,
};
pub use judge::{
    Aggregation, CommandReward, CompactionConfig, JudgeBackend, JudgeBatchMetrics, JudgeConfig,
    JudgeContext, JudgeFailurePolicy, JudgeStrategy, RewardBackend, RulerConfig, RulerJudge, Score,
};
pub use policy::{Policy, PolicyActor, PolicyGeneration, PolicyHandle, TrainerLender};
pub use retrograd_agent_core::config::{AgentConfig, AgentGrpoConfig};
pub use retrograd_agent_core::scenario::{RolloutLimits, Scenario, TruncationPolicy};
pub use retrograd_agent_core::{
    DirEntry, EnvState, Error, ExecOutput, ExecRequest, FailureKind, FileKind, Result, Sandbox,
    SandboxLimits, interrupt,
};
pub use rollout::{GroupOutcome, RolloutEngine, RolloutFailures};
pub use template_parser::TemplateToolCallParser;
pub use tools::{CompositeToolProvider, LocalTool, LocalToolProvider, ToolPlanResolve};
pub use train::{groups_to_train_sequences, to_train_sequence, to_train_sequences};
pub use trajectory::{Message, Role, Step, StepKind, Trajectory, TrajectoryGroup};
