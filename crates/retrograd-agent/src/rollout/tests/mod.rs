//! Fakes and fixtures shared by the themed test modules below.
//!
//! They live in the parent module rather than a sibling one because the whole
//! point is to have a *single* set of fakes: a policy that diverges by seed, a
//! stateful counting environment and its ledger are what most of these tests
//! are actually about, and duplicating them per theme would be duplicating the
//! thing under test.

mod basics;
mod environments;
mod failures;
mod lockstep;
mod render;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use retrograd_core::SamplingParams;

use super::RolloutEngine;
use crate::env::{EnvState, Environment, EnvironmentFactory, StepOutcome};
use crate::policy::{Policy, PolicyGeneration};
use crate::tools::{
    HermesToolCallParser, ToolCall, ToolCallParser, ToolProvider, ToolResult, ToolSpec,
};
use crate::trajectory::{Message, Role};
use crate::{Error, Result};
use retrograd_agent_core::scenario::{RolloutLimits, Scenario};

const TOOL_CALL: &str =
    "<tool_call>{\"name\":\"test__echo\",\"arguments\":{\"text\":\"hi\"}}</tool_call>";

fn logprobs(train_mask: &[bool], value: f32) -> Vec<f32> {
    vec![value; train_mask.iter().filter(|&&train| train).count()]
}

/// The framing shape every fake renders: the opening piece, then one
/// observation piece per assistant turn already in the conversation.
///
/// One piece per gap around the sampled turns is what the engine asks for, and
/// it checks that the pieces it has already committed come back unchanged - a
/// fake must return one piece per gap.
fn framing(messages: &[Message], opening: &[i32], observation: &[i32]) -> Vec<Vec<i32>> {
    let mut pieces = vec![opening.to_vec()];
    for _ in messages
        .iter()
        .filter(|message| message.role == Role::Assistant)
    {
        pieces.push(observation.to_vec());
    }
    pieces
}

#[derive(Default)]
struct FakePolicy {
    seeds: Mutex<Vec<u32>>,
}

#[async_trait]
impl Policy for FakePolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        let mut seeds = self.seeds.lock().unwrap();
        Ok(sampling
            .iter()
            .map(|params| {
                seeds.push(params.seed);
                PolicyGeneration {
                    tokens: vec![11],
                    text: "answer".into(),
                    stopped_at_eog: true,
                }
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// Calls a tool on the first turn and answers on the second. The turn is
/// read from the prompt, not a call counter: in lockstep several members share
/// one decode batch, so the prompt identifies the turn.
#[derive(Default)]
pub(crate) struct MultiTurnPolicy;

#[async_trait]
impl Policy for MultiTurnPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling
            .iter()
            .map(|_| multi_turn_generation(prompt.len()))
            .collect())
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(requests
            .iter()
            .map(|(prompt, _)| multi_turn_generation(prompt.len()))
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.25))
    }
}

fn multi_turn_generation(prompt_len: usize) -> PolicyGeneration {
    if prompt_len > 2 {
        PolicyGeneration {
            tokens: vec![11],
            text: "done".into(),
            stopped_at_eog: true,
        }
    } else {
        PolicyGeneration {
            tokens: vec![10],
            text: TOOL_CALL.into(),
            stopped_at_eog: true,
        }
    }
}

/// Branches on its sampling seed so members of one group diverge at turn 0:
/// even seeds call a tool then answer, odd seeds answer immediately. Also
/// records the shape of every decode batch it is handed.
#[derive(Default)]
struct DivergentPolicy {
    capacity: usize,
    shared_batches: Mutex<Vec<usize>>,
    continuous_batches: Mutex<Vec<usize>>,
}

impl DivergentPolicy {
    fn generation(prompt_len: usize, seed: u32) -> PolicyGeneration {
        if prompt_len > 2 {
            PolicyGeneration {
                tokens: vec![12],
                text: "done".into(),
                stopped_at_eog: true,
            }
        } else if seed.is_multiple_of(2) {
            PolicyGeneration {
                tokens: vec![10],
                text: TOOL_CALL.into(),
                stopped_at_eog: true,
            }
        } else {
            PolicyGeneration {
                tokens: vec![11],
                text: "answer".into(),
                stopped_at_eog: true,
            }
        }
    }
}

#[async_trait]
impl Policy for DivergentPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        self.shared_batches.lock().unwrap().push(sampling.len());
        Ok(sampling
            .iter()
            .map(|params| Self::generation(prompt.len(), params.seed))
            .collect())
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        self.continuous_batches.lock().unwrap().push(requests.len());
        Ok(requests
            .iter()
            .map(|(prompt, params)| Self::generation(prompt.len(), params.seed))
            .collect())
    }

    fn sequence_capacity(&self) -> usize {
        if self.capacity == 0 {
            usize::MAX
        } else {
            self.capacity
        }
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.25))
    }
}

/// Emits nothing for the members whose seed satisfies `fails`, the shape of
/// a runtime that returned an empty sequence for one row of a batch.
struct FlakyPolicy {
    fails: fn(u32) -> bool,
}

#[async_trait]
impl Policy for FlakyPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling
            .iter()
            .map(|params| PolicyGeneration {
                tokens: if (self.fails)(params.seed) {
                    Vec::new()
                } else {
                    vec![11]
                },
                text: "answer".into(),
                stopped_at_eog: true,
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// Always fills the whole per-turn budget; `eog` decides whether that was a
/// natural stop landing on the budget or a real cutoff.
struct BudgetPolicy {
    eog: bool,
}

#[async_trait]
impl Policy for BudgetPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling
            .iter()
            .map(|params| PolicyGeneration {
                tokens: vec![7; params.max_new_tokens as usize],
                text: "partial".into(),
                stopped_at_eog: self.eog,
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// Ends every sampled turn on the same token its template writes to close the
/// turn - the ordinary case for a chat model, whose end-of-generation token *is*
/// the template's turn terminator.
struct SharedTerminatorPolicy;

const TERMINATOR: i32 = 42;

#[async_trait]
impl Policy for SharedTerminatorPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[TERMINATOR, 20]))
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling
            .iter()
            .map(|_| match prompt.len() > 2 {
                true => PolicyGeneration {
                    tokens: vec![11, TERMINATOR],
                    text: "done".into(),
                    stopped_at_eog: true,
                },
                false => PolicyGeneration {
                    tokens: vec![10, TERMINATOR],
                    text: TOOL_CALL.into(),
                    stopped_at_eog: true,
                },
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// Re-renders the conversation with a different prefix after the tool turn,
/// the failure mode the engine must refuse rather than train off-policy.
struct UnstablePrefixPolicy;

#[async_trait]
impl Policy for UnstablePrefixPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        // The opening piece changes the moment a tool turn exists, so the
        // tokens the member already generated were conditioned on a prompt the
        // template cannot render.
        Ok(framing(messages, &[1, 2], &[20])
            .into_iter()
            .enumerate()
            .map(|(index, piece)| match index {
                0 if messages.iter().any(|message| message.role == Role::Tool) => vec![99, 98],
                _ => piece,
            })
            .collect())
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling
            .iter()
            .map(|_| PolicyGeneration {
                tokens: vec![10],
                text: TOOL_CALL.into(),
                stopped_at_eog: true,
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// Keeps calling tools, one slow turn at a time: the shape a rollout
/// deadline exists for.
struct SlowPolicy {
    per_turn: Duration,
}

#[async_trait]
impl Policy for SlowPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        Ok(framing(messages, &[1, 2], &[20]))
    }

    async fn generate_shared(
        &self,
        _prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        tokio::time::sleep(self.per_turn).await;
        Ok(sampling
            .iter()
            .map(|_| PolicyGeneration {
                tokens: vec![10],
                text: TOOL_CALL.into(),
                stopped_at_eog: true,
            })
            .collect())
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        tokio::time::sleep(self.per_turn).await;
        Ok(requests
            .iter()
            .map(|_| PolicyGeneration {
                tokens: vec![10],
                text: TOOL_CALL.into(),
                stopped_at_eog: true,
            })
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.5))
    }
}

/// A policy whose chat template renders tools natively. Behaves like
/// `MultiTurnPolicy` - one tool call, then an answer - and records what the
/// engine handed the template.
#[derive(Default)]
struct NativeToolPolicy {
    native_renders: Mutex<Vec<(Vec<Message>, Vec<ToolSpec>)>>,
    prompt_renders: AtomicUsize,
}

#[async_trait]
impl Policy for NativeToolPolicy {
    async fn supports_native_tools(&self) -> Result<bool> {
        Ok(true)
    }

    /// The other half of `supports_native_tools`, and answering it is not
    /// optional for a policy that says yes: a backend that renders the catalog
    /// but yields no parser falls back to the prompt-written one, both halves
    /// together. This fake's generations are `<tool_call>` literals, so the
    /// parser that reads *its* format is the Hermes one.
    async fn tool_call_parser(
        &self,
        _tools: &[ToolSpec],
    ) -> Result<Option<Arc<dyn ToolCallParser>>> {
        Ok(Some(Arc::new(HermesToolCallParser)))
    }

    async fn render_chat_framing(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        match tools.is_empty() {
            true => {
                self.prompt_renders.fetch_add(1, Ordering::Relaxed);
            }
            false => self
                .native_renders
                .lock()
                .unwrap()
                .push((messages.to_vec(), tools.to_vec())),
        }
        MultiTurnPolicy
            .render_chat_framing(messages, tools, add)
            .await
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        MultiTurnPolicy.generate_shared(prompt, sampling).await
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        MultiTurnPolicy.generate_continuous(requests).await
    }

    async fn score_masked(&self, tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        MultiTurnPolicy.score_masked(tokens, train_mask).await
    }
}

/// Counts how often the engine asks for the tool list.
#[derive(Default)]
struct CountingTools {
    listings: AtomicUsize,
}

#[async_trait]
impl ToolProvider for CountingTools {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        self.listings.fetch_add(1, Ordering::Relaxed);
        EchoTools.list_tools().await
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        EchoTools.call(call).await
    }
}

struct EchoTools;

#[async_trait]
impl ToolProvider for EchoTools {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Ok(vec![ToolSpec {
            name: "test__echo".into(),
            description: "echo".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }])
    }

    async fn call(&self, call: &ToolCall) -> Result<ToolResult> {
        Ok(ToolResult {
            call_id: call.id.clone(),
            content: call.arguments.to_string(),
            is_error: false,
        })
    }
}

/// A tool server that is down when the engine builds its instructions.
struct UnreachableTools;

#[async_trait]
impl ToolProvider for UnreachableTools {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        Err(Error::Tool("connection refused".into()))
    }

    async fn call(&self, _call: &ToolCall) -> Result<ToolResult> {
        Err(Error::Tool("connection refused".into()))
    }
}

/// Lists its tools but cannot run them: the stateless failure that stays an
/// observation instead of killing the trajectory.
struct FailingCallTools;

#[async_trait]
impl ToolProvider for FailingCallTools {
    async fn list_tools(&self) -> Result<Vec<ToolSpec>> {
        EchoTools.list_tools().await
    }

    async fn call(&self, _call: &ToolCall) -> Result<ToolResult> {
        Err(Error::Tool("connection refused".into()))
    }
}

/// Answers turn 0 instantly and then stalls forever: the deadline has to bite
/// inside the decode, not at the next turn boundary.
struct StallingPolicy;

#[async_trait]
impl Policy for StallingPolicy {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        MultiTurnPolicy
            .render_chat_framing(messages, tools, add)
            .await
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        MultiTurnPolicy.generate_shared(prompt, sampling).await
    }

    async fn generate_continuous(
        &self,
        _requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        std::future::pending().await
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(logprobs(&train_mask, -0.25))
    }
}

/// Instance bookkeeping shared by a factory and everything it hands out.
#[derive(Default)]
pub(crate) struct EnvironmentLedger {
    created: AtomicUsize,
    closed: AtomicUsize,
    state_reads: AtomicUsize,
}

/// A *stateful* environment: its step counter lives in the instance, so a
/// member reading a sibling's count shows up directly in the observations.
struct CountingEnvironment {
    ledger: Arc<EnvironmentLedger>,
    steps: usize,
    opening: Option<String>,
    reward: Option<f32>,
    done_after: Option<usize>,
    /// The step at which the instance loses its world, the shape of an HTTP
    /// session the server reaped mid-rollout.
    breaks_at: Option<usize>,
    /// A step that never answers.
    step_delay: Option<Duration>,
    /// What `state()` reports, `None` for an environment with none.
    summary: Option<String>,
}

#[async_trait]
impl Environment for CountingEnvironment {
    async fn reset(&mut self, _scenario: &Scenario, seed: u64) -> Result<Option<String>> {
        Ok(self
            .opening
            .as_ref()
            .map(|opening| format!("{opening} {seed}")))
    }

    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
        self.steps += 1;
        if let Some(delay) = self.step_delay {
            tokio::time::sleep(delay).await;
        }
        if self.breaks_at == Some(self.steps) {
            return Err(Error::Tool("environment step lost its session".into()));
        }
        Ok(StepOutcome {
            result: ToolResult {
                call_id: call.id.clone(),
                content: self.steps.to_string(),
                is_error: false,
            },
            reward: self.reward,
            done: self.done_after == Some(self.steps),
        })
    }

    async fn tools(&self) -> Result<Vec<ToolSpec>> {
        EchoTools.list_tools().await
    }

    async fn state(&mut self) -> Result<EnvState> {
        // Read *before* `close`, and only for a member that survived.
        self.ledger.state_reads.fetch_add(1, Ordering::Relaxed);
        Ok(match &self.summary {
            Some(summary) => EnvState {
                done: true,
                summary: Some(format!("{summary} after {} steps", self.steps)),
                ..Default::default()
            },
            None => EnvState::default(),
        })
    }

    async fn close(&mut self) -> Result<()> {
        self.ledger.closed.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct CountingFactory {
    pub(crate) ledger: Arc<EnvironmentLedger>,
    pub(crate) opening: Option<String>,
    pub(crate) reward: Option<f32>,
    pub(crate) done_after: Option<usize>,
    pub(crate) breaks_at: Option<usize>,
    pub(crate) step_delay: Option<Duration>,
    pub(crate) summary: Option<String>,
}

#[async_trait]
impl EnvironmentFactory for CountingFactory {
    async fn create(&self) -> Result<Box<dyn Environment>> {
        self.ledger.created.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(CountingEnvironment {
            ledger: self.ledger.clone(),
            steps: 0,
            opening: self.opening.clone(),
            reward: self.reward,
            done_after: self.done_after,
            breaks_at: self.breaks_at,
            step_delay: self.step_delay,
            summary: self.summary.clone(),
        }))
    }
}

pub(crate) fn engine_with_environments(
    policy: Arc<dyn Policy>,
    factory: Arc<CountingFactory>,
    limits: RolloutLimits,
) -> RolloutEngine {
    RolloutEngine::with_environments(policy, factory, Arc::new(HermesToolCallParser), limits)
        .unwrap()
}

pub(crate) fn scenario() -> Scenario {
    Scenario {
        id: "demo".into(),
        system: Some("be useful".into()),
        user: "hello".into(),
        metadata: Default::default(),
    }
}

fn engine_with(policy: Arc<dyn Policy>, limits: RolloutLimits) -> RolloutEngine {
    RolloutEngine::with_tools(
        policy,
        Arc::new(EchoTools),
        Arc::new(HermesToolCallParser),
        limits,
    )
    .unwrap()
}
pub(crate) fn group_limits() -> RolloutLimits {
    RolloutLimits {
        max_turns: 4,
        max_new_tokens_per_turn: 4,
        max_trajectory_tokens: 32,
        ..Default::default()
    }
}
