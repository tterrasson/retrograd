//! The turn loop: one decode batch per turn over the live members of a group.

use futures_util::future::join_all;
use retrograd_core::SamplingParams;
use tokio::time::Instant;

use super::RolloutEngine;
use super::deadline::{before_deadline, deadline_expired, truncate};
use super::environments::step_environment;
use super::render::{ToolRendering, inject_tool_instructions, observation_message};
use super::state::RolloutState;
use crate::env::{EnvTask, Environment};
use crate::policy::PolicyGeneration;
use crate::tools::{ParsedAssistant, ToolResult};
use crate::trajectory::{Message, Provenance, Role, Step, StepKind, Trajectory};
use crate::{Error, Result};
use retrograd_agent_core::scenario::Scenario;

impl RolloutEngine {
    /// The lockstep loop. The outer `Result` is a group-wide setup failure; the
    /// inner one is per member, so one dead trajectory never destroys the rest.
    pub(super) async fn rollout_lockstep(
        &self,
        scenario: &Scenario,
        seeds: &[u64],
        environments: &mut [Box<dyn Environment>],
        // Anchored by the caller before the environments were even created, and
        // every stage below goes through `before_deadline`: the tool listing,
        // `reset`, the renders, the decodes, the environment steps and the final
        // rescore. Sampling it only at turn boundaries would bound the number of
        // turns rather than the wall clock, and one hung call would run forever.
        deadline: Option<Instant>,
    ) -> Result<Vec<Result<Trajectory>>> {
        let listing = before_deadline(
            deadline,
            self.tool_rendering(environments.first().map(|environment| &**environment)),
        )
        .await;
        let rendering = match listing {
            Some(rendering) => rendering?,
            None => {
                return Err(Error::Tool(deadline_expired(
                    "listing the environment tools",
                )));
            }
        };
        let mut messages = Vec::new();
        if let Some(system) = &scenario.system {
            messages.push(Message::text(Role::System, system));
        }
        // Only the prompt path writes a catalog into the system turn: under
        // native rendering the template already has the tools.
        if let ToolRendering::Prompt { instructions, .. } = rendering {
            inject_tool_instructions(&mut messages, instructions.clone());
        }
        messages.push(Message::text(Role::User, &scenario.user));

        // One entry per seed even with no environment at all, so the member
        // states below stay aligned with the seeds.
        let openings = if environments.is_empty() {
            seeds.iter().map(|_| Ok(None)).collect::<Vec<_>>()
        } else {
            let resets = before_deadline(
                deadline,
                join_all(
                    environments
                        .iter_mut()
                        .zip(seeds)
                        .map(|(environment, &seed)| environment.reset(scenario, seed)),
                ),
            )
            .await;
            match resets {
                Some(openings) => openings,
                // Nothing per-member exists yet to truncate, so an overrun here
                // is a group-wide setup failure like any other.
                None => return Err(Error::Tool(deadline_expired("resetting the environments"))),
            }
        };
        // No environment, or none of them opened on its own text: the members
        // share one rendered scenario and diverge only at their first sampled
        // token, which is what lets turn 0 be a single shared decode.
        let shared = openings
            .iter()
            .all(|opening| !matches!(opening, Ok(Some(_))));
        let member_messages = seeds
            .iter()
            .enumerate()
            .map(|(member, _)| {
                let mut member_messages = messages.clone();
                if let Some(Ok(Some(opening))) = openings.get(member) {
                    member_messages.push(Message::text(Role::User, opening));
                }
                member_messages
            })
            .collect::<Vec<_>>();

        // A render failure on the shared prompt is group-wide, as it always
        // was; a per-member one costs only that member.
        let shared_prompt = match shared {
            true => match before_deadline(deadline, self.render_prompt(&messages, rendering)).await
            {
                Some(prompt) => Some(prompt?),
                None => {
                    return Err(Error::PolicyGeneration(deadline_expired(
                        "rendering the scenario",
                    )));
                }
            },
            false => None,
        };
        let prompts = match &shared_prompt {
            Some(prompt) => seeds.iter().map(|_| Ok(prompt.clone())).collect::<Vec<_>>(),
            None => {
                let renders = before_deadline(
                    deadline,
                    join_all(
                        member_messages
                            .iter()
                            .map(|messages| self.render_prompt(messages, rendering)),
                    ),
                )
                .await;
                match renders {
                    Some(prompts) => prompts,
                    None => {
                        return Err(Error::PolicyGeneration(deadline_expired(
                            "rendering the scenario",
                        )));
                    }
                }
            }
        };

        let mut states = Vec::with_capacity(seeds.len());
        for (((&seed, messages), opening), prompt) in
            seeds.iter().zip(member_messages).zip(openings).zip(prompts)
        {
            let mut state = match prompt {
                Ok(prompt) => RolloutState::new(seed, messages, prompt),
                Err(error) => {
                    let mut state = RolloutState::new(seed, messages, Vec::new());
                    state.fail(error);
                    state
                }
            };
            if let Err(error) = opening {
                state.fail(error);
            }
            states.push(state);
        }
        for turn in 0..self.limits.max_turns {
            // Cheap and redundant with the per-stage bounds below, but it keeps
            // the loop from opening a turn it has no time left to run.
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                for state in states.iter_mut().filter(|state| !state.finished) {
                    state.stop(true);
                }
            }

            let mut live = Vec::with_capacity(states.len());
            for (index, state) in states.iter_mut().enumerate() {
                if state.finished {
                    continue;
                }
                if state.tokens.len() >= self.limits.max_trajectory_tokens {
                    state.stop(true);
                    continue;
                }
                live.push(index);
            }
            if live.is_empty() {
                break;
            }

            let budgets = live
                .iter()
                .map(|&index| {
                    let room = self.limits.max_trajectory_tokens - states[index].tokens.len();
                    u32::try_from((self.limits.max_new_tokens_per_turn as usize).min(room))
                        .expect("validated trajectory budget fits u32")
                })
                .collect::<Vec<_>>();
            let sampling = live
                .iter()
                .zip(&budgets)
                .map(|(&index, &max_new_tokens)| SamplingParams {
                    temperature: 1.0,
                    top_p: 1.0,
                    max_new_tokens,
                    seed: sampling_seed(states[index].seed, turn),
                })
                .collect::<Vec<_>>();

            let generated = if let (0, Some(prompt)) = (turn, &shared_prompt) {
                // Every member still sits on the shared prompt.
                before_deadline(deadline, self.generate_shared(prompt, sampling)).await
            } else {
                let requests = live
                    .iter()
                    .zip(sampling)
                    .map(|(&index, sampling)| (states[index].tokens.clone(), sampling))
                    .collect::<Vec<_>>();
                before_deadline(deadline, self.generate_continuous(requests)).await
            };
            // A decode the deadline cut short truncates the members riding it,
            // the same policy the turn boundary applies.
            let Some(generated) = generated else {
                truncate(&mut states, &live);
                break;
            };
            let generated = match generated {
                Ok(generated) if generated.len() == live.len() => generated,
                Ok(generated) => {
                    let error = Error::PolicyGeneration(format!(
                        "policy returned {} generations for {} sequences",
                        generated.len(),
                        live.len()
                    ));
                    for &index in &live {
                        states[index].fail(error.duplicate());
                    }
                    continue;
                }
                // One batched call died, so every member riding it dies with it.
                Err(error) => {
                    for &index in &live {
                        states[index].fail(error.duplicate());
                    }
                    continue;
                }
            };

            let mut pending = Vec::new();
            for ((&index, generation), max_new_tokens) in live.iter().zip(generated).zip(&budgets) {
                if let Some(parsed) = self.apply_generation(
                    &mut states[index],
                    generation,
                    *max_new_tokens,
                    turn,
                    rendering,
                ) {
                    pending.push((index, parsed));
                }
            }
            if pending.is_empty() {
                continue;
            }

            // Environments of different members are independent and I/O-bound,
            // so they step concurrently; the calls *within* one member stay
            // ordered, because a stateful environment sequences its own steps.
            let mut stepping = Vec::with_capacity(pending.len());
            let mut cursor = 0;
            for (index, environment) in environments.iter_mut().enumerate() {
                if cursor == pending.len() {
                    break;
                }
                if pending[cursor].0 != index {
                    continue;
                }
                stepping.push(step_environment(environment.as_mut(), &pending[cursor].1));
                cursor += 1;
            }
            let turns = before_deadline(deadline, join_all(stepping)).await;
            let Some(turns) = turns else {
                // Cancelled mid-step: no observation came back, so these members
                // end on the action they had already emitted.
                truncate(&mut states, &live);
                break;
            };

            let mut continuing = Vec::with_capacity(pending.len());
            for ((index, _), turn) in pending.iter().zip(turns) {
                let state = &mut states[*index];
                let first_observation = state.messages.len();
                for observation in &turn.observations {
                    state
                        .messages
                        .push(observation_message(observation, rendering));
                }
                let observed = (first_observation..state.messages.len()).collect::<Vec<_>>();
                match turn.failure {
                    // Not something the policy can react to: the environment
                    // itself broke, so the world the collected prefix is
                    // conditioned on is gone and the trajectory goes with it.
                    Some(error) => state.fail(error),
                    None => continuing.push((*index, turn.reward, turn.done, observed)),
                }
            }
            if continuing.is_empty() {
                continue;
            }

            let renders = before_deadline(
                deadline,
                join_all(continuing.iter().map(|(index, _, _, _)| {
                    self.render_framing(&states[*index].messages, rendering, true)
                })),
            )
            .await;
            let Some(renders) = renders else {
                truncate(&mut states, &live);
                break;
            };
            for ((index, reward, done, observed), render) in continuing.into_iter().zip(renders) {
                self.apply_observation(&mut states[index], render, reward, done, observed);
            }
        }

        // Only members that survived the loop are worth an exact re-score.
        let scorable = states
            .iter()
            .enumerate()
            .filter(|(_, state)| state.failure.is_none())
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let scoring_batch = scorable
            .iter()
            .map(|&index| {
                (
                    states[index].tokens.clone(),
                    states[index].train_mask.clone(),
                )
            })
            .collect::<Vec<_>>();
        let scores = if scoring_batch.is_empty() {
            Some(Ok(Vec::new()))
        } else {
            before_deadline(deadline, self.policy.score_masked_batch(scoring_batch)).await
        };
        match scores {
            Some(Ok(scores)) if scores.len() == scorable.len() => {
                for (index, old_logprobs) in scorable.into_iter().zip(scores) {
                    states[index].old_logprobs = Some(old_logprobs);
                }
            }
            Some(Ok(scores)) => {
                let error = Error::PolicyGeneration(format!(
                    "policy returned {} score rows for {} trajectories",
                    scores.len(),
                    scorable.len()
                ));
                for index in scorable {
                    states[index].fail(error.duplicate());
                }
            }
            Some(Err(error)) => {
                for index in scorable {
                    states[index].fail(error.duplicate());
                }
            }
            // The one stage that cannot degrade into a truncation: a trajectory
            // without its old logprobs is not a trajectory, so there is nothing
            // to keep. These members are counted as failures instead.
            None => {
                for index in scorable {
                    states[index].fail(Error::PolicyGeneration(deadline_expired(
                        "rescoring the trajectories",
                    )));
                }
            }
        }

        Ok(states
            .into_iter()
            .enumerate()
            .map(|(member, state)| self.finish_trajectory(scenario, member, state))
            .collect())
    }

    /// Applies one generated turn to a member, exactly as the single-trajectory
    /// loop did. Returns the parsed assistant turn when the member wants tools
    /// and may continue, `None` when it stopped for any reason.
    fn apply_generation(
        &self,
        state: &mut RolloutState,
        generation: PolicyGeneration,
        max_new_tokens: u32,
        turn: usize,
        rendering: &ToolRendering,
    ) -> Option<ParsedAssistant> {
        if generation.tokens.is_empty() {
            state.fail(Error::PolicyGeneration("policy generated no tokens".into()));
            return None;
        }
        let action_start = state.tokens.len();
        state.tokens.extend_from_slice(&generation.tokens);
        state.train_mask.resize(state.tokens.len(), true);
        let assistant = state.messages.len();
        state.push_step(
            Step {
                kind: StepKind::PolicyAction,
                token_range: (action_start, state.tokens.len()),
                reward: None,
            },
            vec![assistant],
        );

        // Read with the parser the rendering settled on, never with one chosen
        // anywhere else: a template that rendered its own catalog answers in
        // its own format, and reading that with the `<tool_call>` convention
        // finds nothing and reports nothing.
        let mut parsed = match rendering {
            ToolRendering::Native { parser, .. } => parser.parse(&generation.text),
            ToolRendering::Prompt { .. } => self.fallback_parser.parse(&generation.text),
            // Nothing to call: the whole generation is the answer.
            ToolRendering::None => ParsedAssistant {
                content: generation.text.clone(),
                ..Default::default()
            },
        };
        state.messages.push(Message {
            role: Role::Assistant,
            // The fallback chat renderer only sees content, so preserve
            // the raw tags as well as normalized tool_calls metadata.
            content: generation.text,
            tool_calls: parsed.tool_calls.clone(),
            tool_call_id: None,
            is_error: false,
        });

        // A turn that emitted its stop token is complete even when it used
        // the whole budget; only a cutoff mid-response is a truncation.
        if (generation.tokens.len() == max_new_tokens as usize && !generation.stopped_at_eog)
            || state.tokens.len() >= self.limits.max_trajectory_tokens
        {
            state.stop(true);
            return None;
        }
        if parsed.tool_calls.is_empty() && parsed.parse_errors.is_empty() {
            // Nothing to call, or the run says a turn without a call is the
            // answer: either way the trajectory is complete. `declared_tools`
            // rather than the flag alone, because a run with no catalog has no
            // other way to end - telling a policy to call a tool it was never
            // given would spend every turn of every trajectory.
            if self.limits.end_on_no_tool_call || rendering.declared_tools() == 0 {
                state.stop(false);
                return None;
            }
            // The same shape a malformed call takes: one error observation, no
            // environment step, and the turn goes on. Written here rather than
            // in a parser because no parser can know it - the text was
            // syntactically fine, it just named nothing.
            parsed.parse_errors.push(no_tool_call_observation());
        }
        // A turn that reaches here with no call has only errors to show for
        // itself - malformed calls, or the observation just written. Enough of
        // them in a row and the trajectory is cut the way a turn-budget overrun
        // cuts it: truncated, priced by `truncation`, and cheaper by every turn
        // it did not decode. One valid call resets the count.
        if parsed.tool_calls.is_empty() {
            state.failed_turns += 1;
            if self.limits.max_failed_turns > 0
                && state.failed_turns >= self.limits.max_failed_turns
            {
                state.stop(true);
                return None;
            }
        } else {
            state.failed_turns = 0;
        }
        if turn + 1 == self.limits.max_turns {
            state.stop(true);
            return None;
        }
        Some(parsed)
    }

    /// Renders the opening prompt and checks it leaves room to answer in.
    ///
    /// The scenario has no assistant turn yet, so the framing is one piece and
    /// that piece *is* the prompt.
    async fn render_prompt(
        &self,
        messages: &[Message],
        rendering: &ToolRendering,
    ) -> Result<Vec<i32>> {
        let framing = self.render_framing(messages, rendering, true).await?;
        let [prompt] = <[Vec<i32>; 1]>::try_from(framing).map_err(|framing| {
            Error::invalid(format!(
                "opening scenario rendered {} framing pieces, expected one",
                framing.len()
            ))
        })?;
        if prompt.is_empty() || prompt.len() >= self.limits.max_trajectory_tokens {
            return Err(Error::invalid(
                "rendered scenario leaves no room in max_trajectory_tokens",
            ));
        }
        Ok(prompt)
    }

    /// Folds the re-rendered conversation back into the member's token stream,
    /// carrying the environment's verdict on the turn: its step reward lands on
    /// the observation, and `done` ends the trajectory without truncating it.
    /// `observed` are the messages the turn's observations were written to.
    ///
    /// Only the *new* framing piece is appended. The sampled turns are already in
    /// `state.tokens` exactly as the policy emitted them and are never rendered
    /// or re-tokenized; what the re-render is checked for is that the pieces
    /// around them did not move.
    fn apply_observation(
        &self,
        state: &mut RolloutState,
        render: Result<Vec<Vec<i32>>>,
        reward: Option<f32>,
        done: bool,
        observed: Vec<usize>,
    ) {
        let framing = match render {
            Ok(framing) => framing,
            Err(error) => return state.fail(error),
        };
        let committed = state.framing.len();
        if framing.len() != committed + 1 {
            return state.fail(Error::invalid(format!(
                "chat template rendered {} framing pieces for a conversation that had {committed} \
                 plus one tool turn: the template is not usable for multi-turn rollouts",
                framing.len(),
            )));
        }
        if framing[..committed] != state.framing[..] {
            let at = (0..committed)
                .find(|&i| framing[i] != state.framing[i])
                .unwrap_or(0);
            return state.fail(Error::invalid(format!(
                "chat template rewrote turn {at} of the conversation when the tool result was \
                 appended, so the tokens already generated against it are stale (committed \
                 {:?}, re-render {:?})",
                state.framing[at], framing[at],
            )));
        }
        // The policy sampled its own end-of-turn token and the template opens the
        // new piece by closing the same turn, so the two are one token, not two.
        // Only an exact match counts: a model whose end-of-generation token is
        // not the one its template writes has nothing to fold together, and
        // gluing them anyway would put a pair the model never saw in the prompt.
        let observation = match framing[committed].split_first() {
            Some((&terminator, rest)) if state.tokens.last() == Some(&terminator) => rest,
            _ => &framing[committed],
        };
        if state.tokens.len() + observation.len() > self.limits.max_trajectory_tokens {
            return state.stop(true);
        }
        let observation_start = state.tokens.len();
        state.tokens.extend_from_slice(observation);
        state.train_mask.resize(state.tokens.len(), false);
        state.framing = framing;
        state.env_rewarded |= reward.is_some();
        state.environment_done |= done;
        state.push_step(
            Step {
                kind: StepKind::ToolResult,
                token_range: (observation_start, state.tokens.len()),
                reward,
            },
            observed,
        );
        if done {
            state.stop(false);
        }
    }

    fn finish_trajectory(
        &self,
        scenario: &Scenario,
        member: usize,
        state: RolloutState,
    ) -> Result<Trajectory> {
        if let Some(error) = state.failure {
            return Err(error);
        }
        let missing_verifier_reward = (!state.environment_done)
            .then(|| EnvTask::from_scenario(scenario).ok())
            .flatten()
            .and_then(|task| task.verify.map(|verify| verify.reward_on_failure));
        let final_reward = missing_verifier_reward.or_else(|| state.env_rewarded.then_some(0.0));
        let trajectory = Trajectory {
            scenario_id: scenario.id.clone(),
            messages: state.messages,
            tokens: state.tokens,
            old_logprobs: state
                .old_logprobs
                .ok_or_else(|| Error::invalid("trajectory was never scored"))?,
            train_mask: state.train_mask,
            steps: state.steps,
            // The environment graded the steps, so the trajectory's own final
            // reward is nothing on top of them - but it is *set*, which is what
            // takes the group out of the judge's hands.
            reward: final_reward,
            truncated: state.truncated,
            metadata: scenario.metadata.clone(),
            provenance: Some(Provenance {
                member,
                seed: state.seed,
                step_messages: state.step_messages,
            }),
        };
        trajectory.validate()?;
        Ok(trajectory)
    }

    /// Splits a shared-prompt batch across the backend's sequence capacity, so
    /// a `group_size` above `n_seq_max` costs extra waves instead of failing.
    async fn generate_shared(
        &self,
        prompt: &[i32],
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        let capacity = self.policy.sequence_capacity().max(1);
        let mut generations = Vec::with_capacity(sampling.len());
        for chunk in sampling.chunks(capacity) {
            generations.extend(
                self.policy
                    .generate_shared(prompt.to_vec(), chunk.to_vec())
                    .await?,
            );
        }
        Ok(generations)
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        let capacity = self.policy.sequence_capacity().max(1);
        let mut generations = Vec::with_capacity(requests.len());
        for chunk in requests.chunks(capacity) {
            generations.extend(self.policy.generate_continuous(chunk.to_vec()).await?);
        }
        Ok(generations)
    }
}

/// The observation a turn that called nothing comes back with, under
/// `end_on_no_tool_call = false`.
///
/// It says what was missing rather than that something was wrong: the text the
/// policy produced is still in the transcript above, and the only information
/// this line carries that the policy does not already have is that the turn
/// bought it nothing.
fn no_tool_call_observation() -> ToolResult {
    ToolResult {
        call_id: "no_tool_call".into(),
        content: "that turn called no tool, so nothing happened and the turn is spent. Answer \
                  with a tool call."
            .into(),
        is_error: true,
    }
}

/// Folds the trajectory seed and the turn index into the u32 sampling seed.
/// Callers separate updates and group members in different bit ranges of the
/// u64; a plain truncating cast would drop the high bits, so every update would
/// replay the same sampling seeds. The SplitMix64 finalizer diffuses all 64
/// bits into the low 32 before the cast, and keeps turns of one member distinct
/// from turn 0 of its siblings.
pub(super) fn sampling_seed(seed: u64, turn: usize) -> u32 {
    let mut state = seed ^ (turn as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    (state ^ (state >> 31)) as u32
}
