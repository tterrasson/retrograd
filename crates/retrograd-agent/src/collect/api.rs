//! Collection from a remote, OpenAI-compatible model.
//!
//! An API is not a [`Policy`](crate::Policy): it returns text and structured
//! calls, never the tokens or the log-probabilities a trajectory is made of.
//! So this is a text loop of its own, over the same environments, and what it
//! produces is records only - structured ones, since the calls come back
//! already parsed. The filters, the selection and the writing are the rollout
//! path's (`keep_best`); the grading is the environment's alone, because a
//! judge scores token-level trajectory groups.

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use retrograd_dataset::chat_template::{TemplateTool, ToolRenderingKind, observation_text};
use retrograd_dataset::{ChatExample, ChatMessage, ChatToolCall};
use retrograd_llm_client::{OpenAiClient, Purpose};
use serde_json::{Value, json};

use super::{
    Candidate, CollectConfig, CollectSink, CollectStats, base_seed, check_config, keep_best,
};
use crate::env::{EnvTask, Environment, EnvironmentFactory};
use crate::export::AssistantForm;
use crate::tools::ToolSpec;
use crate::{Error, Result, interrupt};
use retrograd_agent_core::scenario::{RolloutLimits, Scenario};

/// The remote model and how it is asked.
pub struct ApiGenerator {
    client: OpenAiClient,
    model: String,
    temperature: Option<f32>,
}

/// The cap on one chat-completions reply: a turn with its calls, far below it
/// for any sane reply, far above it for a runaway one.
const RESPONSE_BYTES: usize = 4 * 1024 * 1024;

impl ApiGenerator {
    /// A failed request fails the attempt the way a policy failure does, and
    /// is counted as one.
    pub fn connect(
        base_url: &str,
        api_key: String,
        timeout: Duration,
        model: impl Into<String>,
        temperature: Option<f32>,
    ) -> Result<Self> {
        Ok(Self {
            client: OpenAiClient::new(
                base_url,
                api_key,
                timeout,
                RESPONSE_BYTES,
                Purpose::Collect,
            )?,
            model: model.into(),
            temperature,
        })
    }
}

/// [`collect_trajectories`](super::collect_trajectories) with a remote model in
/// place of the policy. The `k` attempts of a scenario run concurrently; the
/// scenarios, in order.
///
/// `limits` apply in turns and seconds: a remote model has no token budget to
/// give it, and a reply the provider cut short is a failed attempt.
pub async fn collect_from_api(
    generator: &ApiGenerator,
    environments: Arc<dyn EnvironmentFactory>,
    scenarios: &[Scenario],
    limits: RolloutLimits,
    config: &CollectConfig,
    sink: &mut dyn CollectSink,
) -> Result<CollectStats> {
    if config.form == AssistantForm::Raw {
        return Err(Error::invalid(
            "an API collection writes structured turns only: the calls come back parsed, and \
             there is no generated text to keep verbatim",
        ));
    }
    // No judge on this path, so the environment is the only possible grader.
    check_config(config, false, scenarios)?;
    let mut stats = CollectStats::default();
    for (index, scenario) in scenarios.iter().enumerate() {
        if interrupt::stop_requested() {
            stats.interrupted = true;
            break;
        }
        let base = base_seed(config, index);
        let attempts = join_all((0..config.k).map(|member| {
            let seed = base.wrapping_add(member as u64);
            attempt(generator, environments.as_ref(), scenario, seed, limits)
        }))
        .await;
        let mut candidates = Vec::new();
        for (member, attempt) in attempts.into_iter().enumerate() {
            match attempt {
                Ok(Attempt::Truncated) => stats.rejected.truncated += 1,
                Ok(Attempt::Finished { reward: None, .. }) => stats.rejected.unscored += 1,
                Ok(Attempt::Finished {
                    record,
                    reward: Some(reward),
                    invalid_turns,
                    verification,
                }) => candidates.push(Candidate {
                    reward,
                    length: record.as_ref().map_or(usize::MAX, record_length),
                    member,
                    seed: base.wrapping_add(member as u64),
                    invalid_turns,
                    verification,
                    record,
                }),
                Err(error) => {
                    tracing::warn!(scenario = %scenario.id, "an API attempt failed: {error}");
                    stats.failures.record(error.failure_kind());
                }
            }
        }
        keep_best(
            scenario,
            (index, scenarios.len()),
            config.k,
            candidates,
            config,
            &mut stats,
            sink,
        )?;
    }
    Ok(stats)
}

enum Attempt {
    /// Out of turns or out of time before the episode ended.
    Truncated,
    Finished {
        record: Result<ChatExample>,
        /// `None` when the environment graded nothing.
        reward: Option<f32>,
        invalid_turns: usize,
        verification: Option<String>,
    },
}

/// One episode against a fresh environment instance, closed whatever happens.
async fn attempt(
    generator: &ApiGenerator,
    environments: &dyn EnvironmentFactory,
    scenario: &Scenario,
    seed: u64,
    limits: RolloutLimits,
) -> Result<Attempt> {
    let mut environment = environments.create().await?;
    let episode = episode(generator, environment.as_mut(), scenario, seed, limits);
    let outcome = match limits.max_rollout_secs {
        0 => episode.await,
        secs => match tokio::time::timeout(Duration::from_secs(secs), episode).await {
            Ok(outcome) => outcome,
            Err(_) => Ok(Attempt::Truncated),
        },
    };
    if let Err(error) = environment.close().await {
        tracing::warn!("closing a collection environment failed: {error}");
    }
    outcome
}

async fn episode(
    generator: &ApiGenerator,
    environment: &mut dyn Environment,
    scenario: &Scenario,
    seed: u64,
    limits: RolloutLimits,
) -> Result<Attempt> {
    let specs = environment.tools(scenario).await?;
    let tools = specs.iter().map(TemplateTool::from).collect::<Vec<_>>();
    let opening = environment.reset(scenario, seed).await?;

    let mut messages = Vec::new();
    if let Some(system) = &scenario.system {
        messages.push(ChatMessage::text("system", system.as_str()));
    }
    messages.push(ChatMessage::text("user", scenario.user.as_str()));
    // A second user turn, exactly as the rollout writes an environment's
    // opening.
    if let Some(opening) = opening {
        messages.push(ChatMessage::text("user", opening));
    }

    let mut step_rewards = None::<f32>;
    let mut done = false;
    let mut invalid_turns = 0;
    let mut finished = false;
    for _ in 0..limits.max_turns {
        let completion = generator
            .client
            .chat_completion(&request(generator, &messages, &specs))
            .await?;
        let calls = completion.tool_calls;
        messages.push(ChatMessage {
            tool_calls: calls
                .iter()
                .map(|call| ChatToolCall {
                    id: Some(call.id.clone()),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                })
                .collect(),
            ..ChatMessage::text("assistant", completion.content)
        });
        if calls.is_empty() {
            // An answer, or - when the run expects a call on every turn - a
            // turn that called nothing. The latter invalidates the whole
            // record in keep_best, even if a later turn recovers, so there is
            // no useful trace to continue collecting.
            if !limits.end_on_no_tool_call && !specs.is_empty() {
                invalid_turns += 1;
            }
            finished = true;
            break;
        }
        for call in &calls {
            let outcome = environment.step(call).await?;
            if let Some(reward) = outcome.reward {
                *step_rewards.get_or_insert(0.0) += reward;
            }
            done |= outcome.done;
            messages.push(ChatMessage {
                tool_call_id: Some(call.id.clone()),
                is_error: outcome.result.is_error,
                ..ChatMessage::text("tool", outcome.result.content)
            });
        }
        if done {
            finished = true;
            break;
        }
    }
    if !finished {
        return Ok(Attempt::Truncated);
    }

    let verification = match environment.state().await {
        Ok(state) => state
            .metadata
            .get("verification")
            .and_then(Value::as_str)
            .map(str::to_owned),
        Err(error) => {
            tracing::warn!("reading a collection environment state failed: {error}");
            None
        }
    };
    // The rollout's rule: a verifiable task that ended without being
    // submitted is a failed attempt and gets its failure reward.
    let unsubmitted = (!done)
        .then(|| EnvTask::from_scenario(scenario).ok())
        .flatten()
        .and_then(|task| task.verify.map(|verify| verify.reward_on_failure));
    let reward = match (unsubmitted, step_rewards) {
        (None, None) => None,
        (unsubmitted, steps) => Some(unsubmitted.unwrap_or(0.0) + steps.unwrap_or(0.0)),
    };
    let example = ChatExample {
        tools,
        messages,
        ..ChatExample::default()
    };
    let record = example
        .validate()
        .map(|()| example)
        .map_err(|error| Error::invalid(format!("an API trace is not a valid record: {error}")));
    Ok(Attempt::Finished {
        record,
        reward,
        invalid_turns,
        verification,
    })
}

/// The chat-completions request for the conversation so far, in the OpenAI
/// shape: arguments as a JSON string, observations under their call id.
fn request(generator: &ApiGenerator, messages: &[ChatMessage], specs: &[ToolSpec]) -> Value {
    let messages = messages
        .iter()
        .map(|message| {
            let content = if message.role == "tool" {
                observation_text(
                    ToolRenderingKind::Native,
                    message.tool_call_id.as_deref().unwrap_or_default(),
                    &message.content,
                    message.is_error,
                )
            } else {
                message.content.clone()
            };
            let mut object = json!({"role": message.role, "content": content});
            if !message.tool_calls.is_empty() {
                object["tool_calls"] = message
                    .tool_calls
                    .iter()
                    .map(|call| {
                        json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": call.arguments.to_string(),
                            },
                        })
                    })
                    .collect();
            }
            if let Some(id) = &message.tool_call_id {
                object["tool_call_id"] = json!(id);
            }
            object
        })
        .collect::<Vec<_>>();
    let mut payload = json!({"model": generator.model, "messages": messages});
    if !specs.is_empty() {
        payload["tools"] = specs
            .iter()
            .map(|spec| {
                json!({
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema,
                    },
                })
            })
            .collect();
    }
    if let Some(temperature) = generator.temperature {
        payload["temperature"] = json!(temperature);
    }
    payload
}

/// What "shorter" means for a trace with no tokens: its turns, in characters.
fn record_length(record: &ChatExample) -> usize {
    record
        .messages
        .iter()
        .filter(|message| message.role == "assistant")
        .map(|message| {
            message.content.chars().count()
                + message
                    .tool_calls
                    .iter()
                    .map(|call| call.name.len() + call.arguments.to_string().len())
                    .sum::<usize>()
        })
        .sum()
}

#[cfg(test)]
mod tests;
