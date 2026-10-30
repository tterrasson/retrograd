//! The in-process judge transport: the RULER protocol executed on the
//! training thread's own model, with no HTTP hop and no async runtime in the
//! path.
//!
//! This is what makes self-judging possible: pointing an HTTP judge at the
//! served run deadlocks, because the request waits on the training thread
//! while that thread waits on the request. Here the judge's requests are
//! rendered with the model's own chat template, tokenized, sampled through
//! [`Trainer::generate_sampled_continuous`] under the selected weights, and
//! detokenized - all on the thread that owns the [`Trainer`].
//!
//! The pieces are kept apart on purpose. [`JudgeModel`] is the model surface
//! the transport drives (implemented by [`Trainer`], faked in tests);
//! [`plan_wave`] is the pure batching decision; [`SnapshotJudge`] owns the
//! frozen copy of the weights a `snapshot` judge reads; [`LocalJudge`] wires
//! them into the protocol and applies the same failure policy as the HTTP
//! path.

use std::path::{Path, PathBuf};

use retrograd_agent_core::config::JudgeFailurePolicy;
use retrograd_agent_core::trajectory::TrajectoryGroup;
use retrograd_core::{Error, Result, SamplingParams, TrainConfig};
use retrograd_engine::{JudgeWeights, Trainer};
use retrograd_judge::{
    JudgeBatchMetrics, JudgeCacheMode, JudgeReply, JudgeRequest, JudgeTransport,
    JudgeTransportError, RULER_PROTOCOL_VERSION, RulerConfig, RulerJudge, apply_group_scores,
};
use retrograd_metrics::MetricValue;

/// The model surface the local transport drives. `Trainer` implements it; a
/// fake implements it in tests, which is how the transport is exercised
/// without a GGUF.
pub trait JudgeModel {
    /// Renders one judge prompt through the model's chat template (a single
    /// user turn, assistant turn opened) and tokenizes the result, exactly as
    /// a serving request would be.
    fn render_chat(&mut self, prompt: &str) -> Result<Vec<i32>>;
    /// Samples one completion per `(prompt, params)`, batched into a single
    /// call. Returns only the tokens; the judge trains on nothing.
    fn generate(&mut self, sequences: &[(&[i32], SamplingParams)]) -> Result<Vec<Vec<i32>>>;
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    fn context_size(&self) -> Result<usize>;
}

impl JudgeModel for Trainer {
    fn render_chat(&mut self, prompt: &str) -> Result<Vec<i32>> {
        let messages = serde_json::json!([{"role": "user", "content": prompt}]).to_string();
        let rendered = self.format_chat_messages(&messages, None, true)?;
        self.tokenize_text(&rendered)
    }

    fn generate(&mut self, sequences: &[(&[i32], SamplingParams)]) -> Result<Vec<Vec<i32>>> {
        // Forced reasoning tokens are kept in the reply rather than refused:
        // the judge does not train on its own output, so a forced token is
        // just part of the answer it reads.
        Ok(self
            .generate_sampled_continuous(sequences)?
            .into_iter()
            .map(|row| row.tokens)
            .collect())
    }

    fn detokenize(&self, tokens: &[i32]) -> Result<String> {
        self.detokenize(tokens, false)
    }

    fn context_size(&self) -> Result<usize> {
        Trainer::context_size(self)
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

/// The in-process [`JudgeTransport`]: renders each request through the model's
/// template, plans the wave, samples every batch with
/// `generate_sampled_continuous`, and detokenizes each reply.
///
/// Borrows the model mutably for the duration of the call - which is exactly
/// why the protocol takes the transport as a parameter instead of storing it.
pub struct LocalJudgeTransport<'a, M: JudgeModel + ?Sized> {
    model: &'a mut M,
    sampling: SamplingParams,
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
            sampling,
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
                // a replay of the same wave reproduces the same draws.
                let params = SamplingParams {
                    temperature: requests[index]
                        .temperature
                        .unwrap_or(self.sampling.temperature),
                    top_p: self.sampling.top_p,
                    max_new_tokens: self.max_new_tokens,
                    // The seed is a wrapping bit pattern, so retaining its low
                    // 32 bits is intentional even for a larger request index.
                    seed: self.sampling.seed.wrapping_add(index as u32),
                };
                sequences.push((tokens.as_slice(), params));
                slots.push(index);
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
}

/// The frozen copy of the policy a `snapshot` judge reads.
///
/// Version 1 is one more copy of the base weights plus the trained adapter,
/// exported to a file every `refresh_every` updates and loaded into a
/// forward-only engine owned here. The engine is never the run's anchor:
/// replacing that would change the KL term, so the snapshot is a separate
/// `Trainer` and `with_weights` hands it over explicitly.
pub struct SnapshotJudge {
    engine: Trainer,
    dir: PathBuf,
    refresh_every: u32,
    next_refresh: u32,
    version: u64,
    fingerprint: String,
}

impl SnapshotJudge {
    /// Builds the judge engine from the same GGUF the run trains on. Refused
    /// when the run trains base weights: a snapshot copies *the adapter*, and
    /// there is no frozen base to copy it onto.
    pub fn new(
        model_path: impl AsRef<Path>,
        training: &TrainConfig,
        refresh_every: u32,
        dir: impl Into<PathBuf>,
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
        let dir = dir.into();
        std::fs::create_dir_all(&dir)?;
        // Forward-only, like the reference anchor: no adapter yet, no backward
        // graph, no optimizer state - weights plus KV, and the rest of the
        // geometry travels from the run so the judge scores the same shapes.
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
        Ok(Self {
            engine,
            dir,
            refresh_every,
            next_refresh: 0,
            version: 0,
            fingerprint: String::new(),
        })
    }

    pub fn engine(&mut self) -> &mut Trainer {
        &mut self.engine
    }

    /// The content fingerprint of the frozen adapter currently loaded.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    /// Freezes the policy's current adapter into the judge engine. The peak is
    /// one adapter on disk plus the two engines already resident (the run's
    /// and this one's base copy); the previous export is superseded, not kept.
    pub fn refresh(&mut self, policy: &mut Trainer) -> Result<()> {
        let path = self.dir.join("judge-snapshot.gguf");
        policy.save_lora(&path)?;
        self.engine.load_lora(&path)?;
        self.fingerprint = retrograd_checkpoint::fingerprint_file(&path)?;
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
}

impl LocalJudge {
    /// Builds the local judge around a RULER configuration (rubrics, strategy,
    /// context budgets, compaction - the transport fields are unused).
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
        let snapshot = match weights {
            JudgeWeights::Snapshot => Some(SnapshotJudge::new(
                model_path,
                training,
                refresh_every,
                snapshot_dir.ok_or_else(|| {
                    Error::config("snapshot judge weights need a directory for the frozen adapters")
                })?,
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
        // The protocol speaks the agent façade; the training loop speaks the
        // core one, and the two convert at this boundary with the class kept.
        let mut results: Vec<
            std::result::Result<Vec<retrograd_judge::Score>, retrograd_agent_core::Error>,
        > = Vec::with_capacity(groups.len());
        for group in groups.iter() {
            let snapshot_engine = self.snapshot.as_mut().map(SnapshotJudge::engine);
            let result = trainer.with_weights(weights, snapshot_engine, |model| {
                model.with_reasoning_budget(self.judge_reasoning_budget, |model| {
                    let mut transport =
                        LocalJudgeTransport::new(model, sampling, max_new_tokens, batch_len);
                    judge
                        .score_group_via(&mut transport, group)
                        .map_err(retrograd_core::Error::from)
                })
            });
            results.push(result.map_err(retrograd_agent_core::Error::from));
        }
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
    struct FakeModel {
        context: usize,
        batch_seen: Vec<usize>,
        prompts: Vec<String>,
        fail_generate: bool,
        pairwise: bool,
    }

    impl FakeModel {
        fn new() -> Self {
            Self {
                context: 4096,
                batch_seen: Vec::new(),
                prompts: Vec::new(),
                fail_generate: false,
                pairwise: false,
            }
        }
    }

    impl JudgeModel for FakeModel {
        fn render_chat(&mut self, prompt: &str) -> Result<Vec<i32>> {
            self.prompts.push(prompt.to_owned());
            // One "token" per 8 characters: enough to exercise the budgets.
            Ok(vec![0_i32; (prompt.len() / 8).max(1)])
        }

        fn generate(&mut self, sequences: &[(&[i32], SamplingParams)]) -> Result<Vec<Vec<i32>>> {
            self.batch_seen.push(sequences.len());
            if self.fail_generate {
                return Err(Error::runtime("the fake model refuses"));
            }
            Ok(sequences.iter().map(|_| vec![1_i32, 2, 3]).collect())
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

    fn group() -> TrajectoryGroup {
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

    fn sampling() -> SamplingParams {
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
            fn render_chat(&mut self, _: &str) -> Result<Vec<i32>> {
                Ok(vec![1])
            }
            fn generate(&mut self, _: &[(&[i32], SamplingParams)]) -> Result<Vec<Vec<i32>>> {
                Ok(vec![vec![1]; self.rows])
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
