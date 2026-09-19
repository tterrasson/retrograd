use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::{mpsc, oneshot};

use crate::chat::{template_messages, template_tools};
use crate::template_parser::TemplateToolCallParser;
use crate::tools::{ToolCallParser, ToolSpec};
use crate::trajectory::Message;
use crate::{Error, Result};
use retrograd_core::{SamplingParams, TrainConfig, TrainMetrics};
use retrograd_engine::{GenerationStats, Trainer};
use retrograd_training::Progress;
use retrograd_training::batch::{
    BatchObservation, GrpoBatchParams, TrainSequence, train_grpo_batch_observed,
};

#[derive(Clone, Debug)]
pub struct PolicyGeneration {
    pub tokens: Vec<i32>,
    pub text: String,
    /// The model emitted an end-of-generation token, so the turn finished on
    /// its own rather than against `max_new_tokens`. Without this a response
    /// whose stop token lands exactly on the budget looks truncated.
    pub stopped_at_eog: bool,
}

#[async_trait]
pub trait Policy: Send + Sync {
    /// Renders the conversation into the framing around its assistant turns.
    ///
    /// Returns one piece more than there are assistant messages: the tokens
    /// before the first sampled turn, between consecutive ones, and after the
    /// last (including the generation prompt when `add_assistant`). Interleaving
    /// them with the tokens the policy sampled rebuilds the whole prompt.
    ///
    /// Rendering *around* the sampled turns rather than through them is what
    /// makes a multi-turn prompt assemblable at all. Handing a sampled turn back
    /// to the template and re-tokenizing the result loses the sampled tokens
    /// twice over: the template may trim the content or move its `<think>` block,
    /// and even byte-identical text does not re-tokenize to the ids that were
    /// sampled, because sampling is free to pick a non-canonical split.
    ///
    /// `tools` is the catalog to hand the template; empty means the caller
    /// described the tools itself and the template must not be given any.
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        add_assistant: bool,
    ) -> Result<Vec<Vec<i32>>>;

    /// Whether the model's own chat template renders a tool catalog, and can
    /// therefore be handed the tools instead of being told about them in a
    /// hand-written system prompt.
    ///
    /// Defaults to `false`: a backend that cannot answer is a backend whose
    /// template must be assumed blind to tools, which is the conservative
    /// direction - the prompt-injected catalog works for every template.
    async fn supports_native_tools(&self) -> Result<bool> {
        Ok(false)
    }

    /// The parser for the tool-call format this model's own chat template
    /// teaches it, derived from that template and from `tools`.
    ///
    /// Only meaningful next to [`Policy::supports_native_tools`]: a template
    /// that renders the catalog also decides how the call comes back, and the
    /// two answers have to be taken together or the run renders in one format
    /// and reads in another.
    ///
    /// `None` means no parser could be derived - an exotic template, or a
    /// backend that cannot answer. The caller falls back to describing the
    /// tools in the prompt, which is why the default is `Ok(None)` and no test
    /// policy has to implement this.
    async fn tool_call_parser(
        &self,
        tools: &[ToolSpec],
    ) -> Result<Option<Arc<dyn ToolCallParser>>> {
        let _ = tools;
        Ok(None)
    }

    /// Samples `sampling.len()` independent completions for one shared prompt.
    /// The backend decodes the prompt once for the whole batch, which is the
    /// shape of turn 0 of a rollout group: every member starts from the same
    /// rendered scenario and differs only by its sampling seed.
    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>>;

    /// Samples heterogeneous prompts together, one KV sequence per request.
    /// The shape of turns ≥ 1, once the members of a group have diverged.
    ///
    /// The default routes each request through [`Policy::generate_shared`]
    /// one at a time: correct, but it forfeits the batching. Backends that can
    /// decode independent sequences concurrently should override it.
    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        let mut generations = Vec::with_capacity(requests.len());
        for (prompt, sampling) in requests {
            generations.push(exactly_one(
                self.generate_shared(prompt, vec![sampling]).await?,
            )?);
        }
        Ok(generations)
    }

    async fn generate(
        &self,
        prompt: Vec<i32>,
        sampling: SamplingParams,
    ) -> Result<PolicyGeneration> {
        exactly_one(self.generate_shared(prompt, vec![sampling]).await?)
    }

    /// How many sequences the backend decodes concurrently. Callers split
    /// larger batches into successive calls instead of failing, so a group
    /// bigger than the runtime's sequence capacity stays legal. `usize::MAX`
    /// means "no limit worth chunking for".
    fn sequence_capacity(&self) -> usize {
        usize::MAX
    }

    async fn score_masked(&self, tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>>;

    /// Scores a group of completed trajectories. Backends with a shared-prefix
    /// scorer override this; the default keeps test and remote policies correct
    /// without pretending they support physical batching.
    async fn score_masked_batch(
        &self,
        sequences: Vec<(Vec<i32>, Vec<bool>)>,
    ) -> Result<Vec<Vec<f32>>> {
        let mut scores = Vec::with_capacity(sequences.len());
        for (tokens, train_mask) in sequences {
            scores.push(self.score_masked(tokens, train_mask).await?);
        }
        Ok(scores)
    }
}

fn exactly_one(mut generations: Vec<PolicyGeneration>) -> Result<PolicyGeneration> {
    if generations.len() != 1 {
        return Err(Error::PolicyGeneration(format!(
            "expected one generation, policy returned {}",
            generations.len()
        )));
    }
    Ok(generations.pop().expect("length checked above"))
}

enum PolicyRequest {
    /// Already-serialized JSON: the conversion is a pure function of the
    /// messages (`crate::chat`), so it happens on the caller's side rather than
    /// on the single thread that owns the trainer. `tools_json` is `None` when
    /// the catalog must not reach the template.
    RenderChatFraming {
        messages_json: String,
        tools_json: Option<String>,
        sentinels: Vec<String>,
        add_assistant: bool,
        reply: oneshot::Sender<Result<Vec<Vec<i32>>>>,
    },
    SupportsNativeTools {
        reply: oneshot::Sender<Result<bool>>,
    },
    /// The runtime's lifetime prefix-reuse counters. Read on the trainer's own
    /// thread like everything else here, and differenced by the caller around a
    /// rollout phase.
    GenerationStats {
        reply: oneshot::Sender<Result<GenerationStats>>,
    },
    /// Answers with the *serialized* parser rather than a built one: the blob
    /// is what crosses the channel, and the handle wraps it on the caller's
    /// side. Sent once per run.
    ToolCallParser {
        tools_json: Option<String>,
        reply: oneshot::Sender<Result<Option<String>>>,
    },
    GenerateShared {
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
        reply: oneshot::Sender<Result<Vec<PolicyGeneration>>>,
    },
    GenerateContinuous {
        requests: Vec<(Vec<i32>, SamplingParams)>,
        reply: oneshot::Sender<Result<Vec<PolicyGeneration>>>,
    },
    ScoreMasked {
        tokens: Vec<i32>,
        train_mask: Vec<bool>,
        reply: oneshot::Sender<Result<Vec<f32>>>,
    },
    ScoreMaskedBatch {
        sequences: Vec<(Vec<i32>, Vec<bool>)>,
        reply: oneshot::Sender<Result<Vec<Vec<f32>>>>,
    },
    /// Boxed: these fields are bigger than any other variant's, and the enum
    /// travels by value through the channel on every call.
    TrainBatch {
        job: Box<TrainBatchJob>,
        reply: oneshot::Sender<Result<(TrainMetrics, Vec<Progress>)>>,
    },
    /// Hands the trainer to the borrower for the duration of one job, then
    /// takes it back.
    ///
    /// Only the handshake travels on this channel: `Trainer` is `!Send`, and
    /// `PolicyHandle` has to stay `Send` because `Policy`'s futures are. The
    /// trainer itself moves through a slot both sides share on the one thread
    /// that owns it (`PolicyActor::spawn_local`), which is also why a boxed
    /// closure is not an option: it would be `'static` and could borrow neither
    /// the run controller nor the observer - precisely what a checkpoint at an
    /// update boundary has to touch.
    ///
    /// Safe because an update boundary is quiet: the rollouts, the judging and
    /// the optimizer step of that update have completed, and the next update
    /// has not started.
    Lend {
        ready: oneshot::Sender<()>,
        done: oneshot::Receiver<()>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

/// One GRPO update's inputs, boxed so the request that carries it stays small.
struct TrainBatchJob {
    sequences: Vec<TrainSequence>,
    params: GrpoBatchParams,
    training: TrainConfig,
    observation: Option<BatchObservation>,
}

#[derive(Clone)]
pub struct PolicyHandle {
    tx: mpsc::Sender<PolicyRequest>,
    sequence_capacity: usize,
}

impl PolicyHandle {
    async fn request<T>(
        &self,
        build: impl FnOnce(oneshot::Sender<Result<T>>) -> PolicyRequest,
    ) -> Result<T> {
        let (reply, receive) = oneshot::channel();
        self.tx
            .send(build(reply))
            .await
            .map_err(|_| Error::PolicyStopped)?;
        receive.await.map_err(|_| Error::PolicyStopped)?
    }

    /// Lifetime prefix-reuse counters of the generation context. Difference two
    /// snapshots around a rollout phase to attribute one update.
    pub async fn generation_stats(&self) -> Result<GenerationStats> {
        self.request(|reply| PolicyRequest::GenerationStats { reply })
            .await
    }

    /// `observation` receives the selection before the epochs and the
    /// outcome after them.
    pub async fn train_grpo_batch(
        &self,
        sequences: Vec<TrainSequence>,
        params: GrpoBatchParams,
        training: TrainConfig,
        observation: Option<BatchObservation>,
    ) -> Result<(TrainMetrics, Vec<Progress>)> {
        let job = Box::new(TrainBatchJob {
            sequences,
            params,
            training,
            observation,
        });
        self.request(|reply| PolicyRequest::TrainBatch { job, reply })
            .await
    }
}

#[async_trait]
impl Policy for PolicyHandle {
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        tools: &[ToolSpec],
        add_assistant: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let (messages_json, sentinels) = template_messages(messages)?;
        let tools_json = match tools.is_empty() {
            true => None,
            false => Some(template_tools(tools)?),
        };
        self.request(|reply| PolicyRequest::RenderChatFraming {
            messages_json,
            tools_json,
            sentinels,
            add_assistant,
            reply,
        })
        .await
    }

    async fn supports_native_tools(&self) -> Result<bool> {
        self.request(|reply| PolicyRequest::SupportsNativeTools { reply })
            .await
    }

    async fn tool_call_parser(
        &self,
        tools: &[ToolSpec],
    ) -> Result<Option<Arc<dyn ToolCallParser>>> {
        let tools_json = match tools.is_empty() {
            true => None,
            false => Some(template_tools(tools)?),
        };
        let parser = self
            .request(|reply| PolicyRequest::ToolCallParser { tools_json, reply })
            .await?;
        Ok(parser
            .map(|parser| Arc::new(TemplateToolCallParser::new(parser)) as Arc<dyn ToolCallParser>))
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        self.request(|reply| PolicyRequest::GenerateShared {
            prompt,
            sampling,
            reply,
        })
        .await
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        self.request(|reply| PolicyRequest::GenerateContinuous { requests, reply })
            .await
    }

    fn sequence_capacity(&self) -> usize {
        self.sequence_capacity
    }

    async fn score_masked(&self, tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        self.request(|reply| PolicyRequest::ScoreMasked {
            tokens,
            train_mask,
            reply,
        })
        .await
    }

    async fn score_masked_batch(
        &self,
        sequences: Vec<(Vec<i32>, Vec<bool>)>,
    ) -> Result<Vec<Vec<f32>>> {
        self.request(|reply| PolicyRequest::ScoreMaskedBatch { sequences, reply })
            .await
    }
}

/// Local-task actor around `Trainer`. `Trainer` never crosses an OS-thread
/// boundary: the actor is spawned with `spawn_local` and must be driven by a
/// Tokio `LocalSet` on the thread that created the trainer.
pub struct PolicyActor {
    handle: PolicyHandle,
    returned: Rc<RefCell<Option<Trainer>>>,
    lent: Rc<RefCell<Option<Trainer>>>,
    task: tokio::task::JoinHandle<()>,
}

/// Exclusive, scoped access to the trainer the policy actor owns.
///
/// `!Send` and `!Sync` by construction - it is only ever used on the same
/// thread as the actor, which is what makes lending a `!Send` trainer sound
/// without any unsafe code.
#[derive(Clone)]
pub struct TrainerLender {
    tx: mpsc::Sender<PolicyRequest>,
    lent: Rc<RefCell<Option<Trainer>>>,
}

impl TrainerLender {
    /// Runs `job` against the trainer, exclusively, and gives it back.
    ///
    /// A panic inside `job` drops the trainer with it and stops the actor,
    /// the same outcome a panic anywhere else in the run has, so it is not
    /// special-cased.
    pub async fn with_trainer<T>(&self, job: impl FnOnce(&mut Trainer) -> Result<T>) -> Result<T> {
        let (ready, wait_ready) = oneshot::channel();
        let (finished, done) = oneshot::channel();
        self.tx
            .send(PolicyRequest::Lend { ready, done })
            .await
            .map_err(|_| Error::PolicyStopped)?;
        wait_ready.await.map_err(|_| Error::PolicyStopped)?;
        let mut trainer = self.lent.borrow_mut().take().ok_or(Error::PolicyStopped)?;
        let result = job(&mut trainer);
        self.lent.borrow_mut().replace(trainer);
        // The actor is waiting on this to take the trainer back; the run is
        // stuck without it, so a closed channel is the actor already gone.
        finished.send(()).map_err(|_| Error::PolicyStopped)?;
        result
    }
}

/// Renders the conversation and tokenizes each framing piece on its own.
///
/// Piece-by-piece rather than in one pass over the joined text, because the
/// sampled turns that go between them are *not* re-tokenized: only the first
/// piece opens the stream, so the rest are tokenized as fragments and no BOS
/// lands in the middle of a prompt.
fn render_framing(
    trainer: &Trainer,
    messages_json: &str,
    tools_json: Option<&str>,
    sentinels: &[String],
    add_assistant: bool,
) -> Result<Vec<Vec<i32>>> {
    let rendered = trainer.format_chat_messages(messages_json, tools_json, add_assistant)?;
    crate::chat::split_assistant_spans(&rendered, sentinels)?
        .into_iter()
        .enumerate()
        .map(|(index, segment)| {
            let tokens = match index {
                0 => trainer.tokenize_text(&segment),
                _ => trainer.tokenize_fragment(&segment),
            };
            tokens.map_err(Error::from)
        })
        .collect()
}

/// How much of a generation the tool-call parser gets to see.
///
/// The turn's own closer (LFM2's `<|im_end|>`, the same shape for any
/// template with a turn-end control token) is a control token like the call's
/// own delimiters, but it is not part of the call: the parser is derived from
/// a span that stops before it - a generation stops there, it does not
/// include it, see `chat_parser_roundtrip.rs`'s `sampled()` - so it is
/// excluded from the text the parser reads. It stays in the token sequence
/// training scores, which includes the position where the policy chose to
/// stop.
///
/// `saturating_sub` rather than an assumed non-empty slice: `tokens_len == 0`
/// cannot pair with `stopped_at_eog == true` in `finish_generation` (there is
/// no last token to have been the closer), but the arithmetic stays safe on
/// its own rather than trusting that pairing to hold forever.
fn tool_call_parse_end(tokens_len: usize, stopped_at_eog: bool) -> usize {
    if stopped_at_eog {
        tokens_len.saturating_sub(1)
    } else {
        tokens_len
    }
}

/// Detokenizes a sampled row and decides whether it stopped on its own.
fn finish_generation(trainer: &mut Trainer, tokens: Vec<i32>) -> Result<PolicyGeneration> {
    let stopped_at_eog = match tokens.last() {
        Some(&last) => trainer.is_eog_token(last)?,
        None => false,
    };
    // `true`: this text is read back by a tool-call parser, and a model's own
    // call-format delimiters (LFM2's `<|tool_call_start|>`, and the like) are
    // control tokens - dropping them the way a human-facing render would is
    // what makes a syntactically perfect call unparseable (see
    // `Trainer::detokenize`).
    let parse_end = tool_call_parse_end(tokens.len(), stopped_at_eog);
    let text = trainer.detokenize(&tokens[..parse_end], true)?;
    Ok(PolicyGeneration {
        tokens,
        text,
        stopped_at_eog,
    })
}

/// The policy task panicked or was cancelled. The cause is kept as text because
/// `Error::duplicate` copies a policy failure onto every member of its batch,
/// and a `JoinError` cannot be copied.
fn policy_task_failed(error: impl std::fmt::Display) -> Error {
    Error::PolicyTask(error.to_string())
}

#[cfg(test)]
mod tool_call_parse_end_tests {
    use super::tool_call_parse_end;

    #[test]
    fn a_turn_that_stopped_on_the_closer_excludes_only_that_last_token() {
        assert_eq!(tool_call_parse_end(5, true), 4);
    }

    #[test]
    fn a_turn_cut_by_the_token_budget_keeps_every_token() {
        assert_eq!(tool_call_parse_end(5, false), 5);
    }

    #[test]
    fn an_empty_generation_never_underflows() {
        assert_eq!(tool_call_parse_end(0, false), 0);
        assert_eq!(tool_call_parse_end(0, true), 0);
    }
}

/// Detaches one request's slice of a merged wave. The whole slice leaves the
/// queue before anything downstream can fail: a short-circuiting `collect` over
/// `pop_front` would leave this request's tail behind, and the next request of
/// the wave would then answer with another group's tokens instead of an error.
fn take_rows(count: usize, rows: &mut VecDeque<Vec<i32>>) -> Result<Vec<Vec<i32>>> {
    if rows.len() < count {
        rows.clear();
        return Err(Error::PolicyGeneration(
            "generation batch lost a row".into(),
        ));
    }
    Ok(rows.drain(..count).collect())
}

enum QueuedGeneration {
    Shared {
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
        reply: oneshot::Sender<Result<Vec<PolicyGeneration>>>,
    },
    Continuous {
        requests: Vec<(Vec<i32>, SamplingParams)>,
        reply: oneshot::Sender<Result<Vec<PolicyGeneration>>>,
    },
}

/// What [`QueuedGeneration::from_request`] answers: a request that joins the
/// wave, or the one it was handed back untouched.
///
/// Deliberately not a `Result`: nothing here failed, and the second case is
/// the ordinary path for every non-generation request. `Result` would have
/// made `clippy::result_large_err` suggest boxing, which would allocate once
/// per render or capability request; large payloads stay boxed at the source
/// instead.
enum Queued {
    Generation(QueuedGeneration),
    Other(PolicyRequest),
}

impl QueuedGeneration {
    fn from_request(request: PolicyRequest) -> Queued {
        match request {
            PolicyRequest::GenerateShared {
                prompt,
                sampling,
                reply,
            } => Queued::Generation(Self::Shared {
                prompt,
                sampling,
                reply,
            }),
            PolicyRequest::GenerateContinuous { requests, reply } => {
                Queued::Generation(Self::Continuous { requests, reply })
            }
            other => Queued::Other(other),
        }
    }

    fn into_request(self) -> PolicyRequest {
        match self {
            Self::Shared {
                prompt,
                sampling,
                reply,
            } => PolicyRequest::GenerateShared {
                prompt,
                sampling,
                reply,
            },
            Self::Continuous { requests, reply } => {
                PolicyRequest::GenerateContinuous { requests, reply }
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Shared { sampling, .. } => sampling.len(),
            Self::Continuous { requests, .. } => requests.len(),
        }
    }

    fn send_error(self, error: &Error) {
        match self {
            Self::Shared { reply, .. } | Self::Continuous { reply, .. } => {
                let _ = reply.send(Err(error.duplicate()));
            }
        }
    }

    fn send_rows(self, trainer: &mut Trainer, rows: &mut VecDeque<Vec<i32>>) {
        let result = take_rows(self.len(), rows).and_then(|taken| {
            taken
                .into_iter()
                .map(|tokens| finish_generation(trainer, tokens))
                .collect()
        });
        match self {
            Self::Shared { reply, .. } | Self::Continuous { reply, .. } => {
                let _ = reply.send(result);
            }
        }
    }
}

async fn collect_generation_wave(
    trainer: &mut Trainer,
    rx: &mut mpsc::Receiver<PolicyRequest>,
    deferred: &mut VecDeque<PolicyRequest>,
    first: QueuedGeneration,
    capacity: usize,
) {
    let mut queued = vec![first];
    let mut count = queued[0].len();
    // A tiny bounded window absorbs groups which became ready in adjacent
    // scheduler polls. Render requests are serviced inside the window because
    // another group's first generation cannot be queued until its prompt has
    // been rendered. One millisecond is negligible beside a decode while still
    // bounding latency for a genuinely solitary request.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_millis(1);
    while count < capacity {
        let request = match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(request)) => request,
            Ok(None) | Err(_) => break,
        };
        match QueuedGeneration::from_request(request) {
            Queued::Generation(generation) if count + generation.len() <= capacity => {
                count += generation.len();
                queued.push(generation);
            }
            Queued::Generation(generation) => {
                deferred.push_back(generation.into_request());
                break;
            }
            Queued::Other(PolicyRequest::RenderChatFraming {
                messages_json,
                tools_json,
                sentinels,
                add_assistant,
                reply,
            }) => {
                let _ = reply.send(render_framing(
                    trainer,
                    &messages_json,
                    tools_json.as_deref(),
                    &sentinels,
                    add_assistant,
                ));
            }
            Queued::Other(PolicyRequest::SupportsNativeTools { reply }) => {
                let _ = reply.send(trainer.chat_template_supports_tools().map_err(Error::from));
            }
            Queued::Other(PolicyRequest::GenerationStats { reply }) => {
                let _ = reply.send(trainer.generation_stats().map_err(Error::from));
            }
            Queued::Other(PolicyRequest::ToolCallParser { tools_json, reply }) => {
                let _ = reply.send(
                    trainer
                        .tool_call_parser(tools_json.as_deref())
                        .map_err(Error::from),
                );
            }
            Queued::Other(other) => deferred.push_back(other),
        }
    }
    run_generation_wave(trainer, queued);
}

/// Coalesces generation requests which different rollout groups made ready in
/// the same scheduling wave. Heterogeneous prompts go through the continuous
/// runtime, which still detects and copies shared prefixes.
fn run_generation_wave(trainer: &mut Trainer, queued: Vec<QueuedGeneration>) {
    let expected = queued.iter().map(QueuedGeneration::len).sum::<usize>();
    // Every wave goes through the continuous path, a shared prompt included. The
    // runtime recognizes rows with an identical prompt and decodes that prompt
    // once, so the shared path would save nothing there; what it would cost is
    // the prefix cache. It clears the generation context on entry and on exit,
    // and a turn-0 wave arriving while other groups are mid-trajectory would
    // evict every prefix they hold - and leave its own group's turn 0 for turn
    // 1 to decode again.
    let sequences = queued
        .iter()
        .flat_map(|request| match request {
            QueuedGeneration::Shared {
                prompt, sampling, ..
            } => sampling
                .iter()
                .map(|params| (prompt.as_slice(), *params))
                .collect::<Vec<_>>(),
            QueuedGeneration::Continuous { requests, .. } => requests
                .iter()
                .map(|(prompt, params)| (prompt.as_slice(), *params))
                .collect::<Vec<_>>(),
        })
        .collect::<Vec<_>>();
    let generated = trainer.generate_tokens_continuous(&sequences);
    let mut rows = match generated {
        Ok(rows) if rows.len() == expected => VecDeque::from(rows),
        Ok(rows) => {
            let error = Error::PolicyGeneration(format!(
                "generation runtime returned {} rows for a {expected}-row wave",
                rows.len()
            ));
            for request in queued {
                request.send_error(&error);
            }
            return;
        }
        Err(error) => {
            let error = Error::from(error);
            for request in queued {
                request.send_error(&error);
            }
            return;
        }
    };
    for request in queued {
        request.send_rows(trainer, &mut rows);
    }
}

fn first_masked_target(tokens: &[i32], train_mask: &[bool]) -> Result<usize> {
    if tokens.len() != train_mask.len() {
        return Err(Error::invalid("tokens and train_mask lengths differ"));
    }
    let first = train_mask
        .iter()
        .position(|selected| *selected)
        .ok_or_else(|| Error::invalid("train_mask selects no target token"))?;
    if first == 0 || first >= tokens.len() {
        return Err(Error::invalid(
            "train_mask must select a token after the first input token",
        ));
    }
    Ok(first)
}

/// Scores equal-prefix rows together, retaining only the selected policy
/// positions afterwards. Distinct environment openings form separate physical
/// batches instead of making the shared-prefix runtime reject the whole group.
fn score_masked_batch(
    trainer: &mut Trainer,
    sequences: &[(Vec<i32>, Vec<bool>)],
    capacity: usize,
) -> Result<Vec<Vec<f32>>> {
    if sequences.is_empty() {
        return Err(Error::invalid("masked scoring batch requires sequences"));
    }
    let first = sequences
        .iter()
        .map(|(tokens, mask)| first_masked_target(tokens, mask))
        .collect::<Result<Vec<_>>>()?;
    let mut prefix_groups: Vec<Vec<usize>> = Vec::new();
    for index in 0..sequences.len() {
        let prefix = &sequences[index].0[..first[index]];
        match prefix_groups.iter_mut().find(|group| {
            let leader = group[0];
            sequences[leader].0[..first[leader]] == *prefix
        }) {
            Some(group) => group.push(index),
            None => prefix_groups.push(vec![index]),
        }
    }

    let mut output = vec![None; sequences.len()];
    for group in prefix_groups {
        for chunk in group.chunks(capacity.max(1)) {
            let inputs = chunk
                .iter()
                .map(|&index| (sequences[index].0.as_slice(), first[index]))
                .collect::<Vec<_>>();
            let suffixes = trainer
                .score_token_suffix_batch(&inputs)
                .map_err(Error::from)?;
            for (&index, suffix) in chunk.iter().zip(suffixes) {
                let mask = &sequences[index].1;
                output[index] = Some(
                    (first[index]..sequences[index].0.len())
                        .zip(suffix)
                        .filter_map(|(target, score)| mask[target].then_some(score))
                        .collect(),
                );
            }
        }
    }
    Ok(output
        .into_iter()
        .map(|row| row.expect("every validated scoring row belongs to a prefix group"))
        .collect())
}

impl PolicyActor {
    /// `sequence_capacity` is the number of rollout sequences the trainer can
    /// decode concurrently - `TrainConfig::generation_concurrency`, resolved to
    /// `n_seq_max` when left at zero. It is not derivable from `Trainer`, which
    /// does not retain its config, so the caller supplies it and the engine
    /// chunks its batches accordingly.
    pub fn spawn_local(
        trainer: Trainer,
        capacity: usize,
        sequence_capacity: usize,
        scoring_capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(Error::invalid(
                "policy actor channel capacity must be greater than zero",
            ));
        }
        if sequence_capacity == 0 {
            return Err(Error::invalid(
                "policy actor sequence capacity must be greater than zero",
            ));
        }
        if scoring_capacity == 0 {
            return Err(Error::invalid(
                "policy actor scoring capacity must be greater than zero",
            ));
        }
        let (tx, mut rx) = mpsc::channel(capacity);
        let returned = Rc::new(RefCell::new(None));
        let task_returned = Rc::clone(&returned);
        let lent: Rc<RefCell<Option<Trainer>>> = Rc::new(RefCell::new(None));
        let task_lent = Rc::clone(&lent);
        let task = tokio::task::spawn_local(async move {
            let mut trainer = trainer;
            let mut deferred = VecDeque::new();
            loop {
                let request = match deferred.pop_front() {
                    Some(request) => Some(request),
                    None => rx.recv().await,
                };
                let Some(request) = request else { break };
                match request {
                    PolicyRequest::RenderChatFraming {
                        messages_json,
                        tools_json,
                        sentinels,
                        add_assistant,
                        reply,
                    } => {
                        let _ = reply.send(render_framing(
                            &trainer,
                            &messages_json,
                            tools_json.as_deref(),
                            &sentinels,
                            add_assistant,
                        ));
                    }
                    PolicyRequest::SupportsNativeTools { reply } => {
                        let _ =
                            reply.send(trainer.chat_template_supports_tools().map_err(Error::from));
                    }
                    PolicyRequest::GenerationStats { reply } => {
                        let _ = reply.send(trainer.generation_stats().map_err(Error::from));
                    }
                    PolicyRequest::ToolCallParser { tools_json, reply } => {
                        let _ = reply.send(
                            trainer
                                .tool_call_parser(tools_json.as_deref())
                                .map_err(Error::from),
                        );
                    }
                    PolicyRequest::GenerateShared {
                        prompt,
                        sampling,
                        reply,
                    } => {
                        // Logprob-free sampling: `score_masked` re-scores the
                        // assembled trajectory exactly, so rollout-time
                        // logprobs would be thrown away.
                        collect_generation_wave(
                            &mut trainer,
                            &mut rx,
                            &mut deferred,
                            QueuedGeneration::Shared {
                                prompt,
                                sampling,
                                reply,
                            },
                            sequence_capacity,
                        )
                        .await;
                    }
                    PolicyRequest::GenerateContinuous { requests, reply } => {
                        collect_generation_wave(
                            &mut trainer,
                            &mut rx,
                            &mut deferred,
                            QueuedGeneration::Continuous { requests, reply },
                            sequence_capacity,
                        )
                        .await;
                    }
                    PolicyRequest::ScoreMasked {
                        tokens,
                        train_mask,
                        reply,
                    } => {
                        let result = trainer
                            .score_masked_tokens(&tokens, &train_mask)
                            .map_err(Error::from);
                        let _ = reply.send(result);
                    }
                    PolicyRequest::ScoreMaskedBatch { sequences, reply } => {
                        let _ = reply.send(score_masked_batch(
                            &mut trainer,
                            &sequences,
                            scoring_capacity,
                        ));
                    }
                    PolicyRequest::TrainBatch { job, reply } => {
                        let mut progress = Vec::new();
                        let result = train_grpo_batch_observed(
                            &mut trainer,
                            &job.sequences,
                            &job.params,
                            &job.training,
                            job.observation.as_ref(),
                            &mut |value| progress.push(value),
                        )
                        .map(|metrics| (metrics, progress))
                        .map_err(Error::from);
                        let _ = reply.send(result);
                    }
                    PolicyRequest::Lend { ready, done } => {
                        task_lent.borrow_mut().replace(trainer);
                        if ready.send(()).is_ok() {
                            let _ = done.await;
                        }
                        match task_lent.borrow_mut().take() {
                            Some(returned) => trainer = returned,
                            // The borrower was dropped mid-job and the trainer
                            // went with it. Leave without filling the return
                            // slot, so `into_trainer` reports the loss instead
                            // of the actor serving requests with no trainer.
                            None => return,
                        }
                    }
                    PolicyRequest::Shutdown { reply } => {
                        let _ = reply.send(());
                        break;
                    }
                }
            }
            task_returned.borrow_mut().replace(trainer);
        });
        Ok(Self {
            handle: PolicyHandle {
                tx,
                sequence_capacity,
            },
            returned,
            lent,
            task,
        })
    }

    pub fn handle(&self) -> PolicyHandle {
        self.handle.clone()
    }

    /// A borrower of the trainer, for the code that runs between updates.
    ///
    /// Deliberately not on [`PolicyHandle`]: that one is `Send` because the
    /// rollout engine passes it around as `Arc<dyn Policy>`, and a `!Send`
    /// trainer cannot travel with it.
    pub fn lender(&self) -> TrainerLender {
        TrainerLender {
            tx: self.handle.tx.clone(),
            lent: Rc::clone(&self.lent),
        }
    }

    pub async fn into_trainer(self) -> Result<Trainer> {
        let (reply, receive) = oneshot::channel();
        self.handle
            .tx
            .send(PolicyRequest::Shutdown { reply })
            .await
            .map_err(|_| Error::PolicyStopped)?;
        receive.await.map_err(|_| Error::PolicyStopped)?;
        self.task.await.map_err(policy_task_failed)?;
        self.returned
            .borrow_mut()
            .take()
            .ok_or(Error::PolicyStopped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wave answers several groups out of one queue, so a request that fails
    /// mid-slice must still consume the slice it owns. The next group must not
    /// receive rows left by the failed request.
    #[test]
    fn a_request_consumes_its_whole_slice_of_the_wave() {
        let mut rows = VecDeque::from(vec![vec![1], vec![2], vec![3], vec![4]]);
        assert_eq!(take_rows(2, &mut rows).unwrap(), [vec![1], vec![2]]);
        assert_eq!(take_rows(2, &mut rows).unwrap(), [vec![3], vec![4]]);
        assert!(take_rows(1, &mut rows).is_err());
    }

    /// A short wave cannot leave rows behind either: whoever comes next would
    /// read them as its own.
    #[test]
    fn a_wave_too_short_for_a_request_answers_nobody_with_stale_rows() {
        let mut rows = VecDeque::from(vec![vec![1], vec![2]]);
        assert!(take_rows(3, &mut rows).is_err());
        assert!(rows.is_empty());
    }
}
