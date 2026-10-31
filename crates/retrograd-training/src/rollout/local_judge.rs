//! The in-process judge transport: the RULER protocol executed on the
//! training thread's own model, with no HTTP hop and no async runtime in the
//! path.
//!
//! This is what makes self-judging possible: pointing an HTTP judge at the
//! served run deadlocks, because the request waits on the training thread
//! while that thread waits on the request. Here the judge's requests are
//! rendered with the model's own chat template, tokenized, sampled under the
//! selected weights and detokenized - all on the thread that owns the
//! [`Trainer`].
//!
//! Generation goes through the step-wise scheduler
//! ([`retrograd_infer::Scheduler`]) on the same decode backend the rollouts
//! use ([`ScheduledJudgeModel`]): the judge's requests are admitted, packed
//! and prefix-cached exactly like rollout requests, bounded by the same slot
//! and token budgets, on the trainer's own thread and with no weight update
//! able to interleave. A `snapshot` judge generates on the run's own
//! generation context, under a frozen per-row copy of the adapter; only when
//! that context cannot serve it does it keep a separate engine, and the
//! scheduler is then built over that engine, not over the run's.
//!
//! The pieces are kept apart on purpose. [`JudgeModel`] is the model surface
//! the transport drives (implemented by [`Trainer`] and by
//! [`TrainerBackend`], faked in tests); [`plan_wave`] is the pure batching
//! decision; [`SnapshotJudge`] owns the frozen copy of the weights a
//! `snapshot` judge reads; [`LocalJudge`] wires them into the protocol and
//! applies the same failure policy as the HTTP path.

use std::path::{Path, PathBuf};

use retrograd_agent_core::config::JudgeFailurePolicy;
use retrograd_agent_core::trajectory::TrajectoryGroup;
use retrograd_core::{Error, Result, SamplingParams, SamplingParamsV2, TrainConfig};
use retrograd_engine::{AdapterId, JudgeWeights, SnapshotSource, Trainer};
use retrograd_infer::{
    DecodeBackend, Request, Scheduler, SchedulerOptions, TrainerBackend, run_requests_with,
};
use retrograd_judge::{
    JudgeBatchMetrics, JudgeCacheMode, JudgeReply, JudgeRequest, JudgeTransport,
    JudgeTransportError, RULER_PROTOCOL_VERSION, RulerConfig, RulerJudge, ScoreRequest,
    apply_group_scores,
};
use retrograd_metrics::MetricValue;

use super::driver::scheduler_options;
use super::sampling::RowLayout;

/// The model surface the local transport drives. `Trainer` implements it; a
/// fake implements it in tests, which is how the transport is exercised
/// without a GGUF.
pub trait JudgeModel {
    /// Renders one judge prompt through the model's chat template (a single
    /// user turn, assistant turn opened), as text, exactly as a serving
    /// request would be. Scoring needs the text too: a label is tokenized in
    /// the context of the rendered prompt, not in isolation.
    fn render_chat_text(&mut self, prompt: &str) -> Result<String>;
    /// Tokenizes raw text with the model's tokenizer.
    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>>;
    /// Renders one judge prompt through the model's chat template and
    /// tokenizes the result.
    fn render_chat(&mut self, prompt: &str) -> Result<Vec<i32>> {
        let rendered = self.render_chat_text(prompt)?;
        self.tokenize_text(&rendered)
    }
    /// Samples one completion per `(prompt, params)`, batched into a single
    /// call. Returns only the tokens; the judge trains on nothing. The
    /// extended parameters carry the output grammar of a structured answer.
    fn generate(&mut self, sequences: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>>;
    /// Teacher-forced log-probabilities of each of `targets` as the next
    /// token after `prompt`: `log p(targets[i] | prompt)`. The two label
    /// suffixes of one scoring request share the prompt, so the runtime
    /// branches them off one decoded prefix.
    fn score_next_logprobs(&mut self, prompt: &[i32], targets: &[i32]) -> Result<Vec<f32>>;
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    fn context_size(&self) -> Result<usize>;
}

impl JudgeModel for Trainer {
    fn render_chat_text(&mut self, prompt: &str) -> Result<String> {
        let messages = serde_json::json!([{"role": "user", "content": prompt}]).to_string();
        self.format_chat_messages(&messages, None, true)
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
        Trainer::tokenize_text(self, text)
    }

    fn generate(&mut self, sequences: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
        // Forced reasoning tokens are kept in the reply rather than refused:
        // the judge does not train on its own output, so a forced token is
        // just part of the answer it reads.
        Ok(self
            .generate_sampled_continuous_v2(sequences)?
            .into_iter()
            .map(|row| row.tokens)
            .collect())
    }

    fn score_next_logprobs(&mut self, prompt: &[i32], targets: &[i32]) -> Result<Vec<f32>> {
        if targets.is_empty() {
            return Err(Error::invalid("scoring needs at least one target token"));
        }
        // Each target is scored as the single completion token of
        // `prompt + [target]`: all rows share the prompt, so one batched call
        // decodes the prefix once and branches per candidate.
        let sequences: Vec<(Vec<i32>, usize)> = targets
            .iter()
            .map(|target| {
                let mut tokens = prompt.to_vec();
                tokens.push(*target);
                (tokens, prompt.len())
            })
            .collect();
        let refs: Vec<(&[i32], usize)> = sequences
            .iter()
            .map(|(tokens, n_prompt)| (tokens.as_slice(), *n_prompt))
            .collect();
        let rows = self.score_token_suffix_batch(&refs)?;
        Ok(rows.into_iter().map(|mut row| row.remove(0)).collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.detokenize(tokens, false)
    }

    fn context_size(&self) -> Result<usize> {
        Trainer::context_size(self)
    }
}

/// A `&mut Trainer` as a [`JudgeModel`]: the text and scoring surface of the
/// model the scheduler drives, reached through the scheduler's backend.
impl JudgeModel for TrainerBackend<'_> {
    fn render_chat_text(&mut self, prompt: &str) -> Result<String> {
        JudgeModel::render_chat_text(self.0, prompt)
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
        JudgeModel::tokenize_text(self.0, text)
    }

    fn generate(&mut self, sequences: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
        JudgeModel::generate(self.0, sequences)
    }

    fn score_next_logprobs(&mut self, prompt: &[i32], targets: &[i32]) -> Result<Vec<f32>> {
        JudgeModel::score_next_logprobs(self.0, prompt, targets)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        JudgeModel::detokenize(self.0, tokens)
    }

    fn context_size(&self) -> Result<usize> {
        JudgeModel::context_size(self.0)
    }
}

/// A [`JudgeModel`] whose generations go through the step-wise scheduler
/// ([`retrograd_infer::Scheduler`]) on the same decode backend the rollouts
/// use, instead of one batched FFI generator call per wave: the judge's
/// requests are admitted, packed and prefix-cached exactly like rollout
/// requests, under the same slot and token budgets, on the backend's own
/// thread.
///
/// Everything but `generate` is delegated to the backend's model surface
/// (render, tokenize, detokenize, scoring). `generate` submits one
/// single-completion request per sequence and returns the sampled tokens in
/// request order; a request the pool cannot admit comes back as that
/// request's error, never as a hang. Forced reasoning tokens are kept - the
/// judge reads its own forced markers - unlike the rollout driver, which
/// refuses them.
pub struct ScheduledJudgeModel<B: DecodeBackend> {
    scheduler: Scheduler<B>,
}

impl<B: DecodeBackend> ScheduledJudgeModel<B> {
    /// Builds the scheduler over `backend` and initializes it, which clears
    /// every pool slot once - matching the batched generators, which clear
    /// the generation context on entry.
    pub fn new(backend: B, options: SchedulerOptions) -> Result<Self> {
        let mut scheduler = Scheduler::new(backend, options);
        scheduler.initialize()?;
        Ok(Self { scheduler })
    }

    pub fn scheduler(&self) -> &Scheduler<B> {
        &self.scheduler
    }
}

impl<B> JudgeModel for ScheduledJudgeModel<B>
where
    B: DecodeBackend + JudgeModel,
{
    fn render_chat_text(&mut self, prompt: &str) -> Result<String> {
        self.scheduler.backend_mut().render_chat_text(prompt)
    }

    fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
        self.scheduler.backend().tokenize_text(text)
    }

    fn generate(&mut self, sequences: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let requests: Vec<Request> = sequences
            .iter()
            .map(|(prompt, params)| Request::single((*prompt).to_vec(), params.clone()))
            .collect();
        let outcomes = run_requests_with(&mut self.scheduler, requests, |_| {})?;
        outcomes
            .into_iter()
            .map(|outcome| {
                if let Some(error) = &outcome.error {
                    return Err(Error::runtime(format!("the judge request failed: {error}")));
                }
                match outcome.completions.as_slice() {
                    // Forced tokens stay in: the judge reads them as part of
                    // its answer, it trains on none of them.
                    [single] => Ok(single.tokens.clone()),
                    _ => Err(Error::runtime(
                        "a single-completion judge request returned no completion",
                    )),
                }
            })
            .collect()
    }

    fn score_next_logprobs(&mut self, prompt: &[i32], targets: &[i32]) -> Result<Vec<f32>> {
        self.scheduler
            .backend_mut()
            .score_next_logprobs(prompt, targets)
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.scheduler.backend().detokenize(tokens)
    }

    fn context_size(&self) -> Result<usize> {
        // The generation pool's per-sequence limit - what the scheduler can
        // actually reserve - not the training context's row width.
        Ok(DecodeBackend::context_size(self.scheduler.backend())? as usize)
    }
}

/// The batching decision for one wave of judge requests, as a pure function.
///
/// A request is feasible when its rendered prompt plus the generation budget
/// fits the model's context window; infeasible ones come back as per-request
/// errors and are never sent. The feasible ones are packed `batch_len` at a
/// time - `batch_len` is the run's `generation_concurrency`, the number of
/// sequences the runtime decodes together - so no wave ever asks for an
/// unbounded number of sequences in one call.
#[derive(Clone, Debug, PartialEq)]
pub struct WavePlan {
    /// Batches of request indices, in request order within each batch.
    pub batches: Vec<Vec<usize>>,
    /// Requests that cannot be served, with the reason.
    pub errors: Vec<(usize, String)>,
}

pub fn plan_wave(
    prompts: &[std::result::Result<Vec<i32>, String>],
    batch_len: usize,
    context: usize,
    max_new_tokens: u32,
) -> WavePlan {
    let batch_len = batch_len.max(1);
    let budget = max_new_tokens as usize;
    let mut plan = WavePlan {
        batches: Vec::new(),
        errors: Vec::new(),
    };
    let mut current: Vec<usize> = Vec::new();
    for (index, prompt) in prompts.iter().enumerate() {
        let tokens = match prompt {
            Ok(tokens) => tokens,
            Err(message) => {
                plan.errors
                    .push((index, format!("render the judge prompt: {message}")));
                continue;
            }
        };
        if tokens.is_empty() {
            plan.errors.push((
                index,
                "the chat template rendered an empty prompt".to_owned(),
            ));
            continue;
        }
        if tokens
            .len()
            .checked_add(budget)
            .is_none_or(|total| total > context)
        {
            plan.errors.push((
                index,
                format!(
                    "judge prompt of {} tokens plus the {} token budget exceeds the {} token \
                     context window",
                    tokens.len(),
                    budget,
                    context
                ),
            ));
            continue;
        }
        if current.len() >= batch_len {
            plan.batches.push(std::mem::take(&mut current));
        }
        current.push(index);
    }
    if !current.is_empty() {
        plan.batches.push(current);
    }
    plan
}

/// The output grammar a wire request asks for, as GBNF text: a strict
/// `json_schema` attempt constrains to the request's schema, the degraded
/// `json_object` retry to any JSON object. A request with no structured
/// format is generated unconstrained (the schema is still stated in the
/// prompt text, as the protocol always does).
fn grammar_of(request: &JudgeRequest) -> Option<std::result::Result<std::sync::Arc<str>, String>> {
    let format = request.response_format.as_ref()?;
    match format.get("type").and_then(|value| value.as_str()) {
        Some("json_schema") => Some(
            retrograd_engine::grammar_from_json_schema(&request.schema.to_string())
                .map(|gbnf| std::sync::Arc::from(gbnf.as_str()))
                .map_err(|error| error.to_string()),
        ),
        Some("json_object") => Some(
            retrograd_engine::json_object_grammar()
                .map(|gbnf| std::sync::Arc::from(gbnf.as_str()))
                .map_err(|error| error.to_string()),
        ),
        _ => None,
    }
}

/// The in-process [`JudgeTransport`]: renders each request through the model's
/// template, plans the wave, samples every batch with the extended sampler
/// (structured answers are grammar-constrained), and detokenizes each reply.
///
/// Borrows the model mutably for the duration of the call - which is exactly
/// why the protocol takes the transport as a parameter instead of storing it.
pub struct LocalJudgeTransport<'a, M: JudgeModel + ?Sized> {
    model: &'a mut M,
    sampling: SamplingParamsV2,
    max_new_tokens: u32,
    batch_len: usize,
}

impl<'a, M: JudgeModel + ?Sized> LocalJudgeTransport<'a, M> {
    pub fn new(
        model: &'a mut M,
        sampling: SamplingParams,
        max_new_tokens: u32,
        batch_len: usize,
    ) -> Self {
        Self {
            model,
            sampling: SamplingParamsV2::from(sampling),
            max_new_tokens,
            batch_len,
        }
    }
}

impl<M: JudgeModel + ?Sized> JudgeTransport for LocalJudgeTransport<'_, M> {
    fn complete(
        &mut self,
        requests: &[JudgeRequest],
    ) -> Vec<std::result::Result<JudgeReply, JudgeTransportError>> {
        if requests.is_empty() {
            return Vec::new();
        }
        let prompts: Vec<std::result::Result<Vec<i32>, String>> = requests
            .iter()
            .map(|request| {
                self.model
                    .render_chat(&request.prompt)
                    .map_err(|error| error.to_string())
            })
            .collect();
        let context = match self.model.context_size() {
            Ok(context) => context,
            Err(error) => {
                return requests
                    .iter()
                    .map(|_| {
                        Err(JudgeTransportError::Failed(format!(
                            "read the judge context window: {error}"
                        )))
                    })
                    .collect();
            }
        };
        let plan = plan_wave(&prompts, self.batch_len, context, self.max_new_tokens);
        let mut replies: Vec<Option<std::result::Result<JudgeReply, JudgeTransportError>>> =
            (0..requests.len()).map(|_| None).collect();
        for (index, message) in plan.errors {
            replies[index] = Some(Err(JudgeTransportError::Failed(message)));
        }
        for batch in &plan.batches {
            let mut sequences = Vec::with_capacity(batch.len());
            let mut slots = Vec::with_capacity(batch.len());
            for &index in batch {
                // Feasible prompts are `Ok` by construction of the plan.
                let tokens = prompts[index]
                    .as_ref()
                    .expect("plan_wave only batches feasible prompts");
                // Each request draws its own seed, derived from the wave's, so
                // a replay of the same wave reproduces the same draws. The
                // seed is a wrapping bit pattern, so retaining its low 32 bits
                // is intentional even for a larger request index.
                let mut params = self.sampling.clone();
                params.temperature = requests[index]
                    .temperature
                    .unwrap_or(self.sampling.temperature);
                // The wire protocol uses zero for greedy decoding. The native
                // sampler requires positive temperature; top_k=1 implements
                // greedy selection after grammar masking without division by zero.
                if params.temperature == 0.0 {
                    params.temperature = 1.0;
                    params.top_k = 1;
                }
                params.max_new_tokens = self.max_new_tokens;
                params.seed = self.sampling.seed.wrapping_add(index as u32);
                // A structured answer is constrained by its grammar, not only
                // by the schema stated in the prompt: that is what keeps the
                // syntax failures the failure policy would otherwise count.
                match grammar_of(&requests[index]) {
                    Some(Ok(gbnf)) => params.grammar_gbnf = Some(gbnf),
                    Some(Err(message)) => {
                        replies[index] = Some(Err(JudgeTransportError::Failed(format!(
                            "build the output grammar: {message}"
                        ))));
                        continue;
                    }
                    None => {}
                }
                sequences.push((tokens.as_slice(), params));
                slots.push(index);
            }
            if sequences.is_empty() {
                continue;
            }
            match self.model.generate(&sequences) {
                Ok(rows) if rows.len() != slots.len() => {
                    for &index in &slots {
                        replies[index] = Some(Err(JudgeTransportError::Arity {
                            requested: slots.len(),
                            returned: rows.len(),
                        }));
                    }
                }
                Ok(rows) => {
                    for (row, &index) in rows.iter().zip(&slots) {
                        replies[index] = Some(match self.model.detokenize(row) {
                            Ok(content) => Ok(JudgeReply { content }),
                            Err(error) => Err(JudgeTransportError::Failed(format!(
                                "detokenize the judge reply: {error}"
                            ))),
                        });
                    }
                }
                Err(error) => {
                    for &index in &slots {
                        replies[index] = Some(Err(JudgeTransportError::Failed(error.to_string())));
                    }
                }
            }
        }
        replies
            .into_iter()
            .map(|reply| reply.expect("every request got a slot in the wave"))
            .collect()
    }

    /// Scores each request's target labels: the prompt is rendered through the
    /// template, each label is tokenized *in the context of that render* -
    /// spacing and case included - and refused when it is not a single token
    /// there, then the logprobs come from one branched teacher-forced call.
    fn score(
        &mut self,
        requests: &[ScoreRequest],
    ) -> Vec<std::result::Result<Vec<f32>, JudgeTransportError>> {
        let context = match self.model.context_size() {
            Ok(context) => context,
            Err(error) => {
                return requests
                    .iter()
                    .map(|_| {
                        Err(JudgeTransportError::Failed(format!(
                            "read the judge context window: {error}"
                        )))
                    })
                    .collect();
            }
        };
        requests
            .iter()
            .map(|request| {
                let rendered = self
                    .model
                    .render_chat_text(&request.prompt)
                    .map_err(|error| {
                        JudgeTransportError::Failed(format!("render the judge prompt: {error}"))
                    })?;
                let base = self.model.tokenize_text(&rendered).map_err(|error| {
                    JudgeTransportError::Failed(format!("tokenize the judge prompt: {error}"))
                })?;
                if base.is_empty() {
                    return Err(JudgeTransportError::Failed(
                        "the chat template rendered an empty prompt".to_owned(),
                    ));
                }
                if base.len() >= context {
                    return Err(JudgeTransportError::Failed(format!(
                        "judge prompt of {} tokens fills the {} token context window, no room \
                         for the label",
                        base.len(),
                        context
                    )));
                }
                let mut targets = Vec::with_capacity(request.targets.len());
                for label in &request.targets {
                    let full = self
                        .model
                        .tokenize_text(&format!("{rendered}{label}"))
                        .map_err(|error| {
                            JudgeTransportError::Failed(format!(
                                "tokenize the label '{label}': {error}"
                            ))
                        })?;
                    match full.strip_prefix(base.as_slice()) {
                        Some([token]) => targets.push(*token),
                        other => {
                            return Err(JudgeTransportError::Failed(format!(
                                "the label '{label}' is not a single token in this template's \
                                 context ({} tokens added): pointwise_logprob refuses \
                                 multi-token labels rather than truncate to the first",
                                other.map_or(0, <[i32]>::len)
                            )));
                        }
                    }
                }
                self.model
                    .score_next_logprobs(&base, &targets)
                    .map_err(|error| JudgeTransportError::Failed(error.to_string()))
            })
            .collect()
    }
}

/// Whether a `snapshot` judge's frozen policy can be a per-row adapter of the
/// run's own generation context: the context must exist (multi-sequence or
/// fast sampling), and the strategy must only generate - `pointwise_logprob`
/// scores labels teacher-forced on the training context, which per-row
/// adapters never reach. The memory planner reads the same rule.
pub fn snapshot_on_generation_context(training: &TrainConfig, pointwise: bool) -> bool {
    !pointwise
        && (training.effective_generation_concurrency() > 1
            || training.fast_generation_context
            || training.generation_batch != 0)
}

/// Where the frozen copy lives.
enum Frozen {
    /// A per-row adapter of the run's generation context, once taken.
    Adapter(Option<AdapterId>),
    /// A separate forward-only engine on the same GGUF, with the snapshot
    /// exported to `dir` and loaded on top.
    Engine { engine: Box<Trainer>, dir: PathBuf },
}

/// The frozen copy of the policy a `snapshot` judge reads.
///
/// Wherever it can, the snapshot is a second adapter on the run's own base:
/// the trained adapter's tensors are copied, device to device and without a
/// file, into a per-row adapter of the generation context, and the judge
/// generates under it - no second copy of the base weights, and the trained
/// adapter is never toggled. Otherwise (a `pointwise_logprob` strategy, or a
/// run with no dedicated generation context) it is one more copy of the base
/// weights in a forward-only engine owned here, loaded with the adapter
/// exported every `refresh_every` updates. Either way it is never the run's
/// anchor: replacing that would change the KL term, so `with_weights` is
/// handed the snapshot explicitly.
pub struct SnapshotJudge {
    frozen: Frozen,
    refresh_every: u32,
    next_refresh: u32,
    version: u64,
    fingerprint: String,
}

impl SnapshotJudge {
    /// Refused when the run trains base weights: a snapshot copies *the
    /// adapter*, and there is no frozen base to copy it onto. `dir` is where
    /// a separate engine's exports go; the per-row form writes no file.
    pub fn new(
        model_path: impl AsRef<Path>,
        training: &TrainConfig,
        refresh_every: u32,
        dir: Option<PathBuf>,
        pointwise: bool,
    ) -> Result<Self> {
        if refresh_every == 0 {
            return Err(Error::config(
                "the local judge's snapshot refresh_every must be at least one update",
            ));
        }
        if training.trainable.policy.trains_base_weights() {
            return Err(Error::config(
                "a snapshot judge copies the trained adapter onto a frozen base, and this run \
                 trains its base weights, so there is no frozen base to copy onto: use \
                 weights = \"reference\", or wait for a full frozen copy of the base",
            ));
        }
        let frozen = if snapshot_on_generation_context(training, pointwise) {
            Frozen::Adapter(None)
        } else {
            let dir = dir.ok_or_else(|| {
                Error::config("snapshot judge weights need a directory for the frozen adapters")
            })?;
            std::fs::create_dir_all(&dir)?;
            // Forward-only, like the reference anchor: no adapter yet, no
            // backward graph, no optimizer state - weights plus KV, and the
            // rest of the geometry travels from the run so the judge scores
            // the same shapes.
            let engine = Trainer::new(
                model_path,
                TrainConfig {
                    device: training.device,
                    n_ctx: training.n_ctx,
                    n_batch: training.n_batch,
                    n_ubatch: training.n_ubatch,
                    n_seq_max: training.n_seq_max,
                    generation_concurrency: training.generation_concurrency,
                    generation_batch: training.generation_batch,
                    fast_generation_context: training.fast_generation_context,
                    generation_kv_dtype: training.generation_kv_dtype,
                    threads: training.threads,
                    kv_dtype: training.kv_dtype,
                    ..TrainConfig::default()
                },
            )?;
            Frozen::Engine {
                engine: Box::new(engine),
                dir,
            }
        };
        Ok(Self {
            frozen,
            refresh_every,
            next_refresh: 0,
            version: 0,
            fingerprint: String::new(),
        })
    }

    /// Whether the snapshot is a per-row adapter of the run's own generation
    /// context rather than a separate engine.
    pub fn on_generation_context(&self) -> bool {
        matches!(self.frozen, Frozen::Adapter(_))
    }

    /// The snapshot as [`Trainer::with_weights`] takes it; `None` before the
    /// first refresh.
    pub fn source(&mut self) -> Option<SnapshotSource<'_>> {
        match &mut self.frozen {
            Frozen::Adapter(id) => id.map(SnapshotSource::Adapter),
            Frozen::Engine { engine, .. } => Some(SnapshotSource::Engine(engine)),
        }
    }

    /// The separate engine holding the snapshot, when it is one.
    pub fn engine(&mut self) -> Option<&mut Trainer> {
        match &mut self.frozen {
            Frozen::Engine { engine, .. } => Some(engine),
            Frozen::Adapter(_) => None,
        }
    }

    /// The identity of the frozen weights currently loaded: unique per
    /// refresh, so a verdict cache keyed on it never reads a previous one.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// Freezes the policy's current adapter. A per-row snapshot replaces the
    /// previous one in place (it is unloaded before the copy is taken, so the
    /// peak holds one); an engine snapshot goes through one adapter on disk,
    /// the previous export superseded, not kept.
    pub fn refresh(&mut self, policy: &mut Trainer) -> Result<()> {
        match &mut self.frozen {
            Frozen::Adapter(id) => {
                let taken = policy.generation_adapter_snapshot(id.take(), 1.0)?;
                *id = Some(taken);
                self.fingerprint = format!("generation-adapter-{}", self.version + 1);
            }
            Frozen::Engine { engine, dir } => {
                let path = dir.join("judge-snapshot.gguf");
                policy.save_lora(&path)?;
                engine.load_lora(&path)?;
                self.fingerprint = retrograd_checkpoint::fingerprint_file(&path)?;
            }
        }
        self.version += 1;
        Ok(())
    }

    /// Refreshes when `update` has reached the next boundary. The first call -
    /// before any collection - freezes the starting policy, so every update is
    /// judged by one immutable version of the weights.
    pub fn maybe_refresh(&mut self, update: u32, policy: &mut Trainer) -> Result<()> {
        if update >= self.next_refresh {
            self.refresh(policy)?;
            self.next_refresh = update.saturating_add(self.refresh_every);
        }
        Ok(())
    }
}

/// The cache-key identity of a local judge: everything a cached verdict
/// depends on besides the prompt - the model file (which carries the template
/// and the tokenizer), the weights, the sampling parameters and budgets, and
/// the protocol version.
fn local_cache_identity(
    model_fingerprint: &str,
    weights: JudgeWeights,
    snapshot_fingerprint: &str,
    sampling: SamplingParams,
    max_new_tokens: u32,
    judge_reasoning_budget: Option<u32>,
) -> String {
    format!(
        "local|model={model_fingerprint}|weights={weights:?}|snapshot={snapshot_fingerprint}|\
         sampling={sampling:?}|max_new_tokens={max_new_tokens}|judge_budget={judge_reasoning_budget:?}|\
         protocol={RULER_PROTOCOL_VERSION}"
    )
}

/// Verifies that every `pointwise_logprob` label is a single token in the
/// context of the model's chat template, and refuses the run when one is not:
/// a multi-token label would give a truncated first-token probability, which
/// is not a verdict anyone configured.
fn verify_pointwise_labels<M: JudgeModel + ?Sized>(model: &mut M) -> Result<()> {
    let rendered = model.render_chat_text("ping")?;
    let base = model.tokenize_text(&rendered)?;
    for label in retrograd_judge::POINTWISE_LABELS {
        let full = model.tokenize_text(&format!("{rendered}{label}"))?;
        if !matches!(full.strip_prefix(base.as_slice()), Some([_])) {
            return Err(Error::invalid(format!(
                "the pointwise label '{label}' is not a single token in this template's context \
                 ({} tokens for '{label}'): choose labels the tokenizer keeps whole, or drop the \
                 pointwise_logprob strategy",
                full.len().saturating_sub(base.len())
            )));
        }
    }
    Ok(())
}

/// A judge that executes the RULER protocol in process, on the training
/// thread's model, under a chosen set of weights.
pub struct LocalJudge {
    judge: RulerJudge,
    weights: JudgeWeights,
    snapshot: Option<SnapshotJudge>,
    sampling: SamplingParams,
    max_new_tokens: u32,
    batch_len: usize,
    judge_reasoning_budget: Option<u32>,
    model_fingerprint: String,
    /// The run's training geometry, kept to size the judge's scheduler (slot
    /// count, generation batch, admission mode) exactly like the rollouts'.
    training: TrainConfig,
    /// Whether the strategy is `pointwise_logprob`, whose labels are verified
    /// once against the model's template before the first group is scored.
    pointwise: bool,
    labels_verified: bool,
}

impl LocalJudge {
    /// Builds the local judge around a RULER configuration (rubrics, strategy,
    /// context budgets, compaction - the transport fields are unused, the
    /// selection itself was validated at load).
    ///
    /// `snapshot` weights need `snapshot_dir` and `refresh_every`; the first
    /// is refused at preflight when the run trains base weights.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ruler: RulerConfig,
        weights: JudgeWeights,
        sampling: SamplingParams,
        max_new_tokens: u32,
        batch_len: usize,
        model_path: &Path,
        training: &TrainConfig,
        snapshot_dir: Option<PathBuf>,
        refresh_every: u32,
        judge_reasoning_budget: Option<u32>,
    ) -> Result<Self> {
        ruler.validate_declaration()?;
        if max_new_tokens == 0 {
            return Err(Error::invalid(
                "the local judge needs a positive generation budget",
            ));
        }
        if batch_len == 0 {
            return Err(Error::invalid(
                "the local judge needs a positive batch length (training.generation_concurrency)",
            ));
        }
        let pointwise = ruler.strategy == retrograd_judge::JudgeStrategy::PointwiseLogprob;
        let snapshot = match weights {
            JudgeWeights::Snapshot => Some(SnapshotJudge::new(
                model_path,
                training,
                refresh_every,
                snapshot_dir,
                pointwise,
            )?),
            _ => None,
        };
        let model_fingerprint = retrograd_checkpoint::fingerprint_file_cached(model_path)?;
        // The cache follows the weights: `policy` moves under every update, so
        // no verdict may be reused at all; `snapshot` is refreshed, so verdicts
        // live in memory only and are re-keyed on the snapshot fingerprint;
        // `base` and `reference` are stable and keep the configured disk cache.
        let (mode, identity) = match weights {
            JudgeWeights::Policy => (JudgeCacheMode::Disabled, String::new()),
            JudgeWeights::Snapshot => (
                JudgeCacheMode::Memory,
                local_cache_identity(
                    &model_fingerprint,
                    weights,
                    "none-yet",
                    sampling,
                    max_new_tokens,
                    judge_reasoning_budget,
                ),
            ),
            _ => (
                JudgeCacheMode::Persistent,
                local_cache_identity(
                    &model_fingerprint,
                    weights,
                    "",
                    sampling,
                    max_new_tokens,
                    judge_reasoning_budget,
                ),
            ),
        };
        let judge =
            RulerJudge::with_cache(ruler, mode, identity).map_err(retrograd_core::Error::from)?;
        Ok(Self {
            judge,
            weights,
            snapshot,
            sampling,
            max_new_tokens,
            batch_len,
            judge_reasoning_budget,
            model_fingerprint,
            training: training.clone(),
            pointwise,
            labels_verified: false,
        })
    }

    pub fn weights(&self) -> JudgeWeights {
        self.weights
    }

    /// Scores every group in order, on `trainer`, through the local transport.
    ///
    /// Groups are judged one after another - a single model serves them - and
    /// the failure policy, the dropped-fraction threshold and the batch
    /// metrics are applied exactly as the HTTP path applies them: an
    /// unparsable answer is a failed group, nothing else.
    pub fn score_groups(
        &mut self,
        trainer: &mut Trainer,
        groups: &mut [TrajectoryGroup],
        failure_policy: JudgeFailurePolicy,
        max_dropped_fraction: f32,
        drop_degenerate_groups: bool,
    ) -> Result<JudgeBatchMetrics> {
        if self.weights == JudgeWeights::Snapshot
            && self
                .snapshot
                .as_ref()
                .is_some_and(|snapshot| snapshot.version() == 0)
        {
            self.maybe_refresh_snapshot(0, trainer)?;
        }
        // An attached reference can be a different model file, with its own
        // template and tokenizer. Its identity, rather than the policy's,
        // determines whether a persistent verdict can be reused.
        if self.weights == JudgeWeights::Reference {
            let fingerprint = match trainer.reference_path() {
                Some(path) => retrograd_checkpoint::fingerprint_file_cached(path)?,
                None => self.model_fingerprint.clone(),
            };
            self.judge.set_cache_identity(local_cache_identity(
                &fingerprint,
                self.weights,
                "",
                self.sampling,
                self.max_new_tokens,
                self.judge_reasoning_budget,
            ));
        }
        // A pointwise judge reads its labels as single tokens of the model's
        // own template: verified once, here, under the selected weights, and
        // refused for the whole run rather than per group when they are not.
        if self.pointwise && !self.labels_verified {
            let snapshot = self.snapshot.as_mut().and_then(SnapshotJudge::source);
            trainer.with_weights(self.weights, snapshot, |model| {
                verify_pointwise_labels(model)
            })?;
            self.labels_verified = true;
        }
        self.score_groups_inner(
            trainer,
            groups,
            failure_policy,
            max_dropped_fraction,
            drop_degenerate_groups,
        )
    }

    fn score_groups_inner(
        &mut self,
        trainer: &mut Trainer,
        groups: &mut [TrajectoryGroup],
        failure_policy: JudgeFailurePolicy,
        max_dropped_fraction: f32,
        drop_degenerate_groups: bool,
    ) -> Result<JudgeBatchMetrics> {
        let judge = &self.judge;
        let sampling = self.sampling;
        let max_new_tokens = self.max_new_tokens;
        let batch_len = self.batch_len;
        let weights = self.weights;
        let budget = self.judge_reasoning_budget;
        let training = &self.training;
        let pointwise = self.pointwise;
        // One weights selection for the whole judging session, not one per
        // group: every request of the update runs under the same weights -
        // `base` included - and the snapshot's version is immutable for the
        // whole update. The scheduler lives and dies inside this call, on the
        // trainer's own thread, so no weight update can interleave with a
        // judge request.
        let snapshot = self.snapshot.as_mut().and_then(SnapshotJudge::source);
        // The protocol speaks the agent façade; the training loop speaks the
        // core one, and the two convert at this boundary with the class kept.
        let results = trainer.with_weights(weights, snapshot, |model| {
            model.with_reasoning_budget(budget, |model| {
                // Label scoring uses the training context, not the generation
                // pool. Keep its own context limit and avoid initializing a
                // scheduler for a strategy that never generates.
                if pointwise {
                    let mut transport =
                        LocalJudgeTransport::new(model, sampling, max_new_tokens, batch_len);
                    return Ok(groups
                        .iter()
                        .map(|group| judge.score_group_via(&mut transport, group))
                        .collect::<Vec<_>>());
                }
                // The judge's requests go through the same step-wise
                // scheduler as the rollouts, over the same backend (the
                // run's trainer - for `snapshot`, with the snapshot adapter
                // selected on its generation context, or the snapshot's own
                // engine when it has one), sized like the rollouts': same
                // slots, same generation batch, same admission mode.
                let layout = RowLayout::resolve(model, training)?;
                let options = scheduler_options(
                    layout.rollout_admission,
                    layout.generation_batch,
                    layout.generation_slots,
                    layout.generation_slots,
                );
                let mut model = ScheduledJudgeModel::new(TrainerBackend(model), options)?;
                // A listwise group is judged only once complete: each
                // group's waves are submitted in turn, and within a wave
                // freed slots are refilled from the queue.
                let mut results: Vec<
                    std::result::Result<Vec<retrograd_judge::Score>, retrograd_agent_core::Error>,
                > = Vec::with_capacity(groups.len());
                for group in groups.iter() {
                    let mut transport =
                        LocalJudgeTransport::new(&mut model, sampling, max_new_tokens, batch_len);
                    results.push(judge.score_group_via(&mut transport, group));
                }
                Ok::<_, Error>(results)
            })
        })?;
        apply_group_scores(
            groups,
            results,
            failure_policy,
            max_dropped_fraction,
            drop_degenerate_groups,
        )
        .map_err(retrograd_core::Error::from)
    }

    /// Refreshes the snapshot at an update boundary, and re-keys the verdict
    /// cache on the new fingerprint so no verdict from the previous snapshot
    /// is read back.
    pub fn maybe_refresh_snapshot(&mut self, update: u32, policy: &mut Trainer) -> Result<()> {
        let Some(snapshot) = self.snapshot.as_mut() else {
            return Ok(());
        };
        let before = snapshot.version();
        snapshot.maybe_refresh(update, policy)?;
        if snapshot.version() != before {
            self.judge.set_cache_identity(local_cache_identity(
                &self.model_fingerprint,
                self.weights,
                snapshot.fingerprint(),
                self.sampling,
                self.max_new_tokens,
                self.judge_reasoning_budget,
            ));
        }
        Ok(())
    }

    pub fn metric_values(&self) -> Vec<MetricValue> {
        self.judge.metric_values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::trajectory::{Message, Role, Trajectory};
    use retrograd_judge::{JudgeStrategy, Score};

    /// A model that answers every prompt with a canned verdict, recording what
    /// it was asked. In `pairwise` mode it answers the two-way verdict format.
    pub(super) struct FakeModel {
        pub(super) context: usize,
        pub(super) batch_seen: Vec<usize>,
        pub(super) prompts: Vec<String>,
        pub(super) fail_generate: bool,
        pub(super) pairwise: bool,
        /// Logprobs returned per target by `score_next_logprobs`.
        pub(super) label_logprobs: Vec<f32>,
        pub(super) scored: Vec<(usize, Vec<i32>)>,
        pub(super) retokenize_suffix: bool,
    }

    impl FakeModel {
        pub(super) fn new() -> Self {
            Self {
                context: 4096,
                batch_seen: Vec::new(),
                prompts: Vec::new(),
                fail_generate: false,
                pairwise: false,
                label_logprobs: vec![0.0, -1.0],
                scored: Vec::new(),
                retokenize_suffix: false,
            }
        }
    }

    impl JudgeModel for FakeModel {
        fn render_chat_text(&mut self, prompt: &str) -> Result<String> {
            self.prompts.push(prompt.to_owned());
            // Trailing space, so a label appended to the render is its own
            // whitespace-separated token - as the single-token check expects.
            Ok(format!("<chat>{prompt} "))
        }

        fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
            // One "token" per whitespace-separated word.
            let mut tokens = vec![0_i32; text.split_whitespace().count().max(1)];
            if self.retokenize_suffix && (text.ends_with("yes") || text.ends_with("no")) {
                tokens[0] = 1;
            }
            Ok(tokens)
        }

        fn generate(&mut self, sequences: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
            for (_, params) in sequences {
                assert!(params.temperature > 0.0);
                assert_eq!(
                    params.top_k, 1,
                    "the zero-temperature judge decodes greedily"
                );
            }
            self.batch_seen.push(sequences.len());
            if self.fail_generate {
                return Err(Error::runtime("the fake model refuses"));
            }
            Ok(sequences.iter().map(|_| vec![1_i32, 2, 3]).collect())
        }

        fn score_next_logprobs(&mut self, prompt: &[i32], targets: &[i32]) -> Result<Vec<f32>> {
            self.scored.push((prompt.len(), targets.to_vec()));
            Ok(self.label_logprobs.clone())
        }

        fn detokenize(&self, _tokens: &[i32]) -> Result<String> {
            if self.pairwise {
                Ok(r#"{"winner":"a","explanation":"first is better"}"#.to_owned())
            } else {
                Ok(r#"{"scores":[{"trajectory_id":0,"explanation":"e","score":0.8},{"trajectory_id":1,"explanation":"e","score":0.2}]}"#.to_owned())
            }
        }

        fn context_size(&self) -> Result<usize> {
            Ok(self.context)
        }
    }

    pub(super) fn group() -> TrajectoryGroup {
        let trajectory = |answer: &str| Trajectory {
            scenario_id: "s".into(),
            messages: vec![
                Message::text(Role::User, "question"),
                Message::text(Role::Assistant, answer),
            ],
            tokens: vec![1, 2],
            old_logprobs: vec![-1.0],
            train_mask: vec![false, true],
            steps: vec![],
            reward: None,
            truncated: false,
            metadata: Default::default(),
            provenance: None,
        };
        TrajectoryGroup {
            group_id: 1,
            scenario_id: "s".into(),
            trajectories: vec![trajectory("a"), trajectory("b")],
        }
    }

    pub(super) fn sampling() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_new_tokens: 64,
            seed: 7,
        }
    }

    fn ruler_config() -> RulerConfig {
        RulerConfig {
            base_url: String::new(),
            model: "local".into(),
            strategy: JudgeStrategy::Listwise,
            max_retries: 0,
            ..Default::default()
        }
    }

    #[test]
    fn the_wave_is_batched_by_concurrency_and_refuses_what_does_not_fit() {
        // Five prompts of 10 tokens, budget 64, context 128: 10+64 fits, so
        // all five are feasible and packed 2 at a time.
        let prompts: Vec<std::result::Result<Vec<i32>, String>> = vec![Ok(vec![0_i32; 10]); 5];
        let plan = plan_wave(&prompts, 2, 128, 64);
        assert_eq!(plan.batches, vec![vec![0, 1], vec![2, 3], vec![4]]);
        assert!(plan.errors.is_empty());
        // A prompt that does not leave room for the budget is refused, not sent.
        let prompts: Vec<std::result::Result<Vec<i32>, String>> =
            vec![Ok(vec![0_i32; 100]), Ok(vec![0_i32; 10])];
        let plan = plan_wave(&prompts, 4, 128, 64);
        assert_eq!(plan.batches, vec![vec![1]]);
        assert_eq!(plan.errors.len(), 1);
        assert!(plan.errors[0].1.contains("exceeds"), "{:?}", plan.errors[0]);
        // A failed render is reported per request.
        let prompts: Vec<std::result::Result<Vec<i32>, String>> =
            vec![Err("template failed".to_owned()), Ok(vec![1])];
        let plan = plan_wave(&prompts, 4, 128, 64);
        assert_eq!(plan.batches, vec![vec![1]]);
        assert!(plan.errors[0].1.contains("template failed"));
        // An empty render is refused too.
        let prompts: Vec<std::result::Result<Vec<i32>, String>> = vec![Ok(Vec::new())];
        let plan = plan_wave(&prompts, 4, 128, 64);
        assert!(plan.batches.is_empty());
        assert_eq!(plan.errors.len(), 1);
    }

    #[test]
    fn the_transport_scores_a_group_end_to_end_through_the_model() {
        let judge = RulerJudge::new(ruler_config()).unwrap();
        let mut model = FakeModel::new();
        let requests = vec![JudgeRequest::structured(
            "judge this".to_owned(),
            serde_json::json!({"type": "object"}),
            "ruler_scores",
            Some(0.0),
        )];
        let replies = {
            let mut transport = LocalJudgeTransport::new(&mut model, sampling(), 64, 4);
            transport.complete(&requests)
        };
        assert_eq!(replies.len(), 1);
        let content = replies[0].as_ref().expect("a reply").content.clone();
        assert!(content.contains("\"score\":0.8"), "{content}");
        // The model saw the prompt once, in one batch of one.
        assert_eq!(model.prompts.len(), 1);
        assert!(model.prompts[0].contains("judge this"));
        assert_eq!(model.batch_seen, vec![1]);
        // And the protocol parses the reply into scores.
        let scores = judge
            .score_group_via(
                &mut LocalJudgeTransport::new(&mut model, sampling(), 64, 4),
                &group(),
            )
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!(matches!(scores[0], Score { value, .. } if (value - 0.8).abs() < 1e-6));
    }

    #[test]
    fn a_generation_failure_is_a_per_request_error_not_a_panic() {
        let mut model = FakeModel::new();
        model.fail_generate = true;
        let requests = vec![JudgeRequest::structured(
            "judge this".to_owned(),
            serde_json::json!({}),
            "ruler_scores",
            None,
        )];
        let replies = LocalJudgeTransport::new(&mut model, sampling(), 64, 4).complete(&requests);
        assert!(replies[0].is_err());
        assert!(
            replies[0]
                .as_ref()
                .unwrap_err()
                .to_string()
                .contains("refuses"),
            "{:?}",
            replies[0]
        );
    }

    #[test]
    fn a_model_must_return_exactly_one_row_per_request() {
        struct WrongArityModel {
            rows: usize,
        }
        impl JudgeModel for WrongArityModel {
            fn render_chat_text(&mut self, _: &str) -> Result<String> {
                Ok("x ".to_owned())
            }
            fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
                Ok(vec![0_i32; text.split_whitespace().count().max(1)])
            }
            fn generate(&mut self, _: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
                Ok(vec![vec![1]; self.rows])
            }
            fn score_next_logprobs(&mut self, _: &[i32], _: &[i32]) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            fn detokenize(&self, _: &[i32]) -> Result<String> {
                Ok("{}".into())
            }
            fn context_size(&self) -> Result<usize> {
                Ok(128)
            }
        }
        let request =
            JudgeRequest::structured("judge".into(), serde_json::json!({}), "score", None);
        for rows in [0, 2] {
            let mut model = WrongArityModel { rows };
            let replies = LocalJudgeTransport::new(&mut model, sampling(), 64, 1)
                .complete(std::slice::from_ref(&request));
            assert_eq!(replies.len(), 1);
            assert!(
                matches!(replies[0], Err(JudgeTransportError::Arity { requested: 1, returned }) if returned == rows)
            );
        }
    }

    #[test]
    fn a_wave_never_sends_more_sequences_than_the_concurrency() {
        let judge = RulerJudge::new(RulerConfig {
            strategy: JudgeStrategy::Pairwise {
                max_pairs: None,
                both_orders: false,
                aggregation: retrograd_judge::Aggregation::WinRate,
            },
            ..ruler_config()
        })
        .unwrap();
        let mut model = FakeModel::new();
        model.pairwise = true;
        // Three trajectories -> ring of three pairs -> one wave of three
        // requests, packed 2 at a time.
        let group = TrajectoryGroup {
            group_id: 1,
            scenario_id: "s".into(),
            trajectories: (0..3)
                .map(|index| {
                    let mut trajectory = group().trajectories[0].clone();
                    trajectory.messages[1] =
                        Message::text(Role::Assistant, format!("answer-{index}"));
                    trajectory
                })
                .collect(),
        };
        let scores = judge
            .score_group_via(
                &mut LocalJudgeTransport::new(&mut model, sampling(), 64, 2),
                &group,
            )
            .unwrap();
        assert_eq!(scores.len(), 3);
        assert!(
            model.batch_seen.iter().all(|&batch| batch <= 2),
            "a batch exceeded the concurrency: {:?}",
            model.batch_seen
        );
        assert_eq!(model.batch_seen.iter().sum::<usize>(), 3);
    }
}

#[cfg(test)]
mod pointwise_tests {
    use super::tests::{FakeModel, group, sampling};
    use super::*;
    use retrograd_judge::{JudgeStrategy, ScoreRequest};

    fn score_requests() -> Vec<ScoreRequest> {
        vec![ScoreRequest {
            prompt: "judge this".to_owned(),
            targets: retrograd_judge::POINTWISE_LABELS
                .iter()
                .map(|label| label.to_string())
                .collect(),
        }]
    }

    #[test]
    fn scoring_tokenizes_each_label_in_the_template_context() {
        let mut model = FakeModel::new();
        let replies =
            LocalJudgeTransport::new(&mut model, sampling(), 64, 4).score(&score_requests());
        assert_eq!(replies.len(), 1);
        let logprobs = replies[0].as_ref().expect("a scoring reply").clone();
        assert_eq!(logprobs, vec![0.0, -1.0]);
        // One branched call per request, with the two label ids as targets.
        assert_eq!(model.scored.len(), 1);
        assert_eq!(model.scored[0].1.len(), 2);
    }

    #[test]
    fn a_label_must_preserve_the_tokenized_prompt_prefix() {
        let mut model = FakeModel::new();
        model.retokenize_suffix = true;
        assert!(verify_pointwise_labels(&mut model).is_err());
        let replies =
            LocalJudgeTransport::new(&mut model, sampling(), 64, 4).score(&score_requests());
        assert!(replies[0].is_err());
        assert!(model.scored.is_empty());
    }

    #[test]
    fn a_multi_token_label_is_refused_not_truncated() {
        let mut model = FakeModel::new();
        let requests = vec![ScoreRequest {
            prompt: "judge this".to_owned(),
            targets: vec!["yes".to_owned(), "no way".to_owned()],
        }];
        let replies = LocalJudgeTransport::new(&mut model, sampling(), 64, 4).score(&requests);
        let error = replies[0].as_ref().unwrap_err().to_string();
        assert!(error.contains("not a single token"), "{error}");
        assert!(error.contains("no way"), "{error}");
        assert!(model.scored.is_empty());
    }

    #[test]
    fn the_label_preflight_refuses_a_multi_token_label() {
        let mut model = FakeModel::new();
        assert!(verify_pointwise_labels(&mut model).is_ok());
        // A model whose template glues the label onto the previous token.
        struct GluedModel;
        impl JudgeModel for GluedModel {
            fn render_chat_text(&mut self, _: &str) -> Result<String> {
                Ok("ping".to_owned())
            }
            fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
                // One token for the whole text: any appended label merges.
                Ok(vec![0_i32; text.split_whitespace().count().max(1)])
            }
            fn generate(&mut self, _: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
                Ok(Vec::new())
            }
            fn score_next_logprobs(&mut self, _: &[i32], _: &[i32]) -> Result<Vec<f32>> {
                Ok(Vec::new())
            }
            fn detokenize(&self, _: &[i32]) -> Result<String> {
                Ok(String::new())
            }
            fn context_size(&self) -> Result<usize> {
                Ok(128)
            }
        }
        // "ping" is one token, "pingyes" is still one token: not +1.
        let error = verify_pointwise_labels(&mut GluedModel).unwrap_err();
        assert!(error.to_string().contains("not a single token"), "{error}");
    }

    #[test]
    fn pointwise_scores_a_group_end_to_end_through_the_model() {
        let judge = RulerJudge::with_cache(
            RulerConfig {
                strategy: JudgeStrategy::PointwiseLogprob,
                max_retries: 0,
                ..Default::default()
            },
            JudgeCacheMode::Memory,
            "local-test".into(),
        )
        .unwrap();
        let mut model = FakeModel::new();
        // log(1) vs log(e^-1): P(yes) = 1/(1+e^-1) ~ 0.731
        model.label_logprobs = vec![0.0, -1.0];
        let scores = judge
            .score_group_via(
                &mut LocalJudgeTransport::new(&mut model, sampling(), 64, 4),
                &group(),
            )
            .unwrap();
        assert_eq!(scores.len(), 2);
        let expected = 1.0 / (1.0 + (-1.0_f32).exp());
        for score in &scores {
            assert!((score.value - expected).abs() < 1e-6, "{}", score.value);
            assert!(score.valid);
            assert!(
                score
                    .explanation
                    .as_ref()
                    .is_some_and(|e| e.contains("P(yes)"))
            );
        }
        // One scoring request per trajectory.
        assert_eq!(model.scored.len(), 2);
    }
}

#[cfg(test)]
mod scheduled_tests {
    //! The judge's generation path through the step-wise scheduler: same
    //! backend, same admission as the rollouts, one request per sequence, on
    //! the backend's own thread - exercised on a fake pool, no model.

    use std::path::Path;
    use std::time::Duration;

    use retrograd_engine::{EngineMemory, StepOutput, StepRow};
    use retrograd_infer::{
        AdmissionMode, DecodeBackend, FakeBackend, InferError, SchedulerOptions, fake_sampling_v2,
    };

    use super::*;

    fn pieces(vocab: usize) -> Vec<String> {
        (0..vocab)
            .map(|index| {
                char::from_u32('a' as u32 + (index as u32 % 26))
                    .expect("a letter")
                    .to_string()
            })
            .collect()
    }

    /// A `FakeBackend` that is also a [`JudgeModel`]: the text surface is
    /// canned, and the batched generator refuses loudly - the scheduled model
    /// must drive the scheduler, never that fallback.
    struct FakeJudgeBackend {
        fake: FakeBackend,
        prompts: Vec<String>,
        batched_called: bool,
    }

    impl FakeJudgeBackend {
        fn new() -> Self {
            Self::with_ctx(32, 128)
        }

        fn with_ctx(n_ctx_seq: u32, cells_total: u64) -> Self {
            Self {
                fake: FakeBackend::new(4, n_ctx_seq, cells_total, 100, 99, pieces(100)),
                prompts: Vec::new(),
                batched_called: false,
            }
        }
    }

    impl DecodeBackend for FakeJudgeBackend {
        fn step(
            &mut self,
            rows: &[StepRow<'_>],
            out: &mut StepOutput,
        ) -> std::result::Result<(), InferError> {
            self.fake.step(rows, out)
        }
        fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> std::result::Result<(), InferError> {
            self.fake.seq_rm(slot, p0, p1)
        }
        fn seq_cp(
            &mut self,
            src: i32,
            dst: i32,
            p0: i32,
            p1: i32,
        ) -> std::result::Result<(), InferError> {
            self.fake.seq_cp(src, dst, p0, p1)
        }
        fn seq_pos(&self, slot: i32) -> std::result::Result<(i32, i32), InferError> {
            self.fake.seq_pos(slot)
        }
        fn seq_save_len(&self, slot: i32, flags: u32) -> std::result::Result<usize, InferError> {
            self.fake.seq_save_len(slot, flags)
        }
        fn seq_save_into(
            &mut self,
            slot: i32,
            flags: u32,
            buffer: &mut [u8],
        ) -> std::result::Result<usize, InferError> {
            self.fake.seq_save_into(slot, flags, buffer)
        }
        fn seq_load(
            &mut self,
            slot: i32,
            flags: u32,
            data: &[u8],
        ) -> std::result::Result<usize, InferError> {
            self.fake.seq_load(slot, flags, data)
        }
        fn kv_unified(&self) -> std::result::Result<bool, InferError> {
            self.fake.kv_unified()
        }
        fn weights_version(&self) -> std::result::Result<u64, InferError> {
            self.fake.weights_version()
        }
        fn model_fingerprint(&self) -> std::result::Result<u64, InferError> {
            self.fake.model_fingerprint()
        }
        fn is_hybrid(&self) -> std::result::Result<bool, InferError> {
            self.fake.is_hybrid()
        }
        fn memory(&self) -> std::result::Result<EngineMemory, InferError> {
            self.fake.memory()
        }
        fn request_free(&mut self, request: i32) -> std::result::Result<(), InferError> {
            self.fake.request_free(request)
        }
        fn is_eog_token(&self, token: i32) -> std::result::Result<bool, InferError> {
            self.fake.is_eog_token(token)
        }
        fn detokenize_bytes(
            &self,
            tokens: &[i32],
            unparse_special: bool,
        ) -> std::result::Result<Vec<u8>, InferError> {
            self.fake.detokenize_bytes(tokens, unparse_special)
        }
        fn context_size(&self) -> std::result::Result<u32, InferError> {
            self.fake.context_size()
        }
        fn vocab_size(&self) -> std::result::Result<u32, InferError> {
            self.fake.vocab_size()
        }
        fn load_adapter(&mut self, name: &str, path: &Path) -> std::result::Result<(), InferError> {
            self.fake.load_adapter(name, path)
        }
        fn activate_adapter(&mut self, name: Option<&str>) -> std::result::Result<(), InferError> {
            self.fake.activate_adapter(name)
        }
        fn active_adapter(&self) -> std::result::Result<Option<String>, InferError> {
            self.fake.active_adapter()
        }
    }

    impl JudgeModel for FakeJudgeBackend {
        fn render_chat_text(&mut self, prompt: &str) -> Result<String> {
            self.prompts.push(prompt.to_owned());
            Ok(format!("<chat>{prompt} "))
        }
        fn tokenize_text(&self, text: &str) -> Result<Vec<i32>> {
            Ok(vec![0_i32; text.split_whitespace().count().max(1)])
        }
        fn generate(&mut self, _: &[(&[i32], SamplingParamsV2)]) -> Result<Vec<Vec<i32>>> {
            self.batched_called = true;
            Err(Error::runtime(
                "the batched generator must not be called behind the scheduler",
            ))
        }
        fn score_next_logprobs(&mut self, _: &[i32], targets: &[i32]) -> Result<Vec<f32>> {
            Ok(vec![0.0; targets.len()])
        }
        fn detokenize(&self, _tokens: &[i32]) -> Result<String> {
            // The listwise verdict format the protocol parses.
            Ok(r#"{"scores":[{"trajectory_id":0,"explanation":"e","score":0.8},{"trajectory_id":1,"explanation":"e","score":0.2}]}"#.to_owned())
        }
        fn context_size(&self) -> Result<usize> {
            Ok(DecodeBackend::context_size(&self.fake)? as usize)
        }
    }

    fn options() -> SchedulerOptions {
        SchedulerOptions {
            mode: AdmissionMode::Lockstep,
            n_batch: 16,
            queue_capacity: 8,
            event_capacity: 8,
            slow_client_timeout: Duration::ZERO,
            ..SchedulerOptions::default()
        }
    }

    #[test]
    fn judge_generation_runs_through_the_scheduler_in_request_order() {
        let mut model = ScheduledJudgeModel::new(FakeJudgeBackend::new(), options()).unwrap();
        // The text surface delegates to the backend's model.
        let rendered = model.render_chat_text("judge this").unwrap();
        assert_eq!(rendered, "<chat>judge this ");
        let prompt = model.tokenize_text(&rendered).unwrap();
        assert!(!prompt.is_empty());
        assert_eq!(
            model.context_size().unwrap(),
            DecodeBackend::context_size(&model.scheduler().backend().fake).unwrap() as usize
        );
        let params = [fake_sampling_v2(1, 4), fake_sampling_v2(2, 6)];
        let sequences: Vec<(&[i32], SamplingParamsV2)> = params
            .iter()
            .map(|params| (prompt.as_slice(), params.clone()))
            .collect();
        let rows = model.generate(&sequences).unwrap();
        // One completion per sequence, in request order, sized by each
        // request's own budget.
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].len(), 4);
        assert_eq!(rows[1].len(), 6);
        // The requests went through the scheduler (steps recorded, both
        // request ids seen), and the batched generator was never called.
        let backend = model.scheduler().backend();
        assert!(!backend.batched_called);
        assert!(!backend.fake.steps.is_empty());
        assert_eq!(backend.fake.request_order.len(), 2);
    }

    #[test]
    fn an_unreservable_judge_prompt_fails_the_call_instead_of_hanging() {
        let mut model = ScheduledJudgeModel::new(FakeJudgeBackend::new(), options()).unwrap();
        // n_ctx_seq is 32: a 40-token prompt can never be reserved.
        let good = vec![1_i32; 4];
        let bad = vec![1_i32; 40];
        let params = fake_sampling_v2(9, 3);
        let sequences: Vec<(&[i32], SamplingParamsV2)> =
            vec![(good.as_slice(), params.clone()), (bad.as_slice(), params)];
        let error = model.generate(&sequences).unwrap_err();
        assert!(
            error.to_string().contains("judge request failed"),
            "{error}"
        );
    }

    #[test]
    fn the_transport_drives_the_scheduled_model_end_to_end() {
        // The full path a local judge takes: render through the template,
        // plan the wave, generate through the scheduler, detokenize.
        let judge = RulerJudge::new(RulerConfig {
            strategy: JudgeStrategy::Listwise,
            max_retries: 0,
            ..RulerConfig {
                base_url: String::new(),
                model: "local".into(),
                ..Default::default()
            }
        })
        .unwrap();
        // A wide pool: the rendered RULER prompt is ~50 tokens.
        let mut model =
            ScheduledJudgeModel::new(FakeJudgeBackend::with_ctx(256, 1024), options()).unwrap();
        let scores = judge
            .score_group_via(
                &mut LocalJudgeTransport::new(&mut model, sampling(), 8, 4),
                &group(),
            )
            .unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores.iter().all(|score| score.valid));
        let backend = model.scheduler().backend();
        assert!(!backend.batched_called);
        assert!(!backend.fake.steps.is_empty());
        // The judge prompt was rendered through the template exactly once.
        assert_eq!(backend.prompts.len(), 1);
        assert!(backend.prompts[0].contains("question"));
    }

    use super::tests::{group, sampling};
    use retrograd_judge::JudgeStrategy;
}
