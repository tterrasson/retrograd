//! The RULER protocol, separated from its transport.
//!
//! Everything in this module is pure protocol: render a group into
//! [`JudgeRequest`]s, execute them through whatever [`JudgeTransport`] the
//! caller binds, parse the replies into [`Score`]s. The crate's `http` feature
//! only adds one transport (an OpenAI-compatible endpoint) plus the blocking adapter that
//! drives it for the async API; the protocol itself - rendering, budgeting,
//! compaction, chunking, pairwise aggregation, the verdict cache and the
//! `judge/*` metrics - compiles and runs without it, so an in-process judge
//! links neither `reqwest` nor `rustls`.
//!
//! Execution is organized in *waves*: compaction is one wave of summary
//! requests, the verdict is another (one request per chunk, one per pair), and
//! a transport runs a wave's requests concurrently, bounded by its own
//! configuration. A request that fails - transport error or unparsable answer -
//! is retried on its own, with a degraded response format, exactly as the HTTP
//! path always did.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use retrograd_agent_core::trajectory::TrajectoryGroup;
use retrograd_agent_core::{Error, Result};
use retrograd_metrics::MetricValue;

#[cfg(feature = "http")]
use crate::RewardBackend;
use crate::Score;
use crate::render::compact;
use crate::render::prompt::{self, DEFAULT_PAIRWISE_RUBRIC, DEFAULT_RUBRIC, RenderedGroup};
use crate::transport::{JudgeRequest, JudgeTransport, JudgeTransportError};

#[cfg(feature = "http")]
mod http;
#[cfg(feature = "http")]
pub use http::HttpJudgeTransport;

mod chunked;
mod listwise;
mod pairwise;

#[cfg(test)]
use crate::render::compact::CompactionConfig;
#[cfg(test)]
use crate::transport::JudgeReply;
#[cfg(test)]
use retrograd_spec::judge::{Aggregation, JudgeContext};
pub use retrograd_spec::judge::{JudgeStrategy, RulerConfig};

/// Version of the render/parse protocol, mixed into every cache key: a cached
/// verdict written by an older prompt or parser must not be read as today's.
pub const RULER_PROTOCOL_VERSION: u32 = 1;

/// How much of the verdict cache a judge uses.
///
/// The default is the historical behaviour (memory plus the on-disk JSONL
/// file) for a judge whose weights do not move - an HTTP endpoint. A judge
/// whose weights *do* move (the in-process policy, a snapshot that is refreshed)
/// must not read a verdict produced under other weights, so it drops the disk
/// cache by default and keys what it keeps on a weights fingerprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JudgeCacheMode {
    /// In-memory plus the on-disk JSONL cache file.
    #[default]
    Persistent,
    /// In-memory only: verdicts are reusable while the weights that produced
    /// them are loaded, and nothing is written to disk.
    Memory,
    /// No cache at all.
    Disabled,
}

/// The rubric a rendered group is judged against.
///
/// It reads a `RenderedGroup`, which is this crate's rendering and not part of
/// the schema - so it is added to the declaration here rather than moved with
/// it.
trait Rubrics {
    fn listwise_rubric<'a>(&'a self, rendered: &'a RenderedGroup) -> &'a str;
    fn pairwise_rubric(&self, rendered: &RenderedGroup) -> String;
}

impl Rubrics for RulerConfig {
    /// The rubric a request should carry. A scenario that declares its own wins
    /// over the run-wide one: task-specific criteria are worth far more than a
    /// generic wording, and a scenario is the only place that knows them.
    fn listwise_rubric<'a>(&'a self, rendered: &'a RenderedGroup) -> &'a str {
        rendered
            .rubric
            .as_deref()
            .or(self.rubric.as_deref())
            .unwrap_or(DEFAULT_RUBRIC)
    }

    /// Same rule, one composition apart: a scenario rubric is *criteria*, while
    /// a pairwise rubric also carries the answer format ("a", "b" or "tie").
    /// Substituting one for the other would ask for a per-trajectory score in a
    /// two-way comparison, so the scenario's criteria are appended instead.
    fn pairwise_rubric(&self, rendered: &RenderedGroup) -> String {
        let base = self
            .pairwise_rubric
            .as_deref()
            .unwrap_or(DEFAULT_PAIRWISE_RUBRIC);
        match &rendered.rubric {
            Some(scenario) => format!("{base}\n\nTask-specific criteria:\n{scenario}"),
            None => base.to_owned(),
        }
    }
}

#[derive(Default)]
struct Stats {
    requests: AtomicU64,
    retries: AtomicU64,
    latency_micros: AtomicU64,
    cache_hits: AtomicU64,
    groups: AtomicU64,
    prompt_chars_max: AtomicU64,
    elided_chars: AtomicU64,
    rendered_chars: AtomicU64,
    compacted_chars: AtomicU64,
    pairs_judged: AtomicU64,
    position_disagreements: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct RulerStats {
    pub requests: u64,
    pub retries: u64,
    pub latency_ms_mean: f32,
    pub cache_hits: u64,
    pub groups: u64,
    /// Largest prompt actually sent, in characters. The number to watch when
    /// tuning `JudgeContext`: close to the model's window means chunking is
    /// about to kick in (or already has).
    pub prompt_chars_max: u64,
    /// Share of rendered characters that budgeting had to drop. A judge reading
    /// mostly elided trajectories is judging summaries, not behaviour.
    pub elided_fraction: f32,
    /// Share of rendered characters replaced by an LLM summary. Watch it exactly
    /// like `elided_fraction`: a judge reading mostly summaries is judging
    /// summaries.
    pub compacted_fraction: f32,
    /// Share of both-order comparisons where the judge contradicted itself once
    /// the trajectories were exchanged. The single number that says whether the
    /// chosen judge is reading the trajectories or their positions; near 0.5 it
    /// is answering at random.
    pub position_disagreement: f32,
    pub requests_per_group: f32,
}

/// One cached judge response. Verdicts and compaction summaries share the file:
/// they share a model, a determinism requirement and a lifetime, and two files
/// would be two things to keep in step.
#[derive(Serialize, Deserialize)]
struct CacheRow {
    key: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    scores: Option<Vec<Score>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    text: Option<String>,
}

/// The protocol state a judge carries from one group to the next: the
/// configuration, the verdict caches and the counters. Shared with the HTTP
/// blocking jobs through an `Arc`, which is why the caches are behind mutexes
/// and the counters atomics.
struct RulerCore {
    config: RulerConfig,
    /// Mixed into every cache key in place of the model name when the judge is
    /// not an endpoint: the transport identity, the weights fingerprint, the
    /// sampling parameters, the protocol version. Behind a mutex because an
    /// in-process judge re-keys its cache when its snapshot is refreshed.
    cache_identity: Mutex<String>,
    cache_mode: JudgeCacheMode,
    cache: Mutex<HashMap<String, Vec<Score>>>,
    text_cache: Mutex<HashMap<String, String>>,
    stats: Stats,
}

/// A RULER judge: the protocol, plus an optional bound transport.
///
/// Built with [`RulerJudge::new`], the judge is transport-agnostic and scores
/// groups through [`RulerJudge::score_group_via`], the transport being passed
/// at execution time - an in-process transport borrows the model mutably, so
/// it cannot live inside the judge. Built with `RulerJudge::with_http` (the
/// `http` feature), the OpenAI-compatible transport is bound and driven by a
/// blocking job, which is what keeps the async
/// [`crate::RewardBackend`] API free of any nested runtime.
pub struct RulerJudge {
    core: Arc<RulerCore>,
    #[cfg(feature = "http")]
    http: Option<http::HttpHandle>,
}

impl RulerJudge {
    /// Builds the protocol only: no transport is bound, and the transport
    /// fields of the configuration (`base_url`, `api_key_env`) are not checked,
    /// since there is no endpoint to check them for. Groups are scored with
    /// [`RulerJudge::score_group_via`].
    ///
    /// The transport is passed at execution time rather than captured here.
    pub fn new(config: RulerConfig) -> Result<Self> {
        Self::build(config, JudgeCacheMode::Persistent, None)
    }

    /// Same, with an explicit cache policy and cache-key identity. An
    /// in-process judge whose weights move uses this to keep its cache keyed on
    /// the weights fingerprint (or to disable it).
    pub fn with_cache(config: RulerConfig, mode: JudgeCacheMode, identity: String) -> Result<Self> {
        Self::build(config, mode, Some(identity))
    }

    fn build(
        config: RulerConfig,
        cache_mode: JudgeCacheMode,
        identity: Option<String>,
    ) -> Result<Self> {
        config.validate_protocol()?;
        let (cache, text_cache) = if cache_mode == JudgeCacheMode::Persistent {
            load_cache(config.cache_path.as_ref())?
        } else {
            Caches::default()
        };
        let identity = serde_json::json!([
            identity.unwrap_or_else(|| format!("{}|{}", config.base_url, config.model)),
            config.temperature,
            RULER_PROTOCOL_VERSION,
        ])
        .to_string();
        Ok(Self {
            core: Arc::new(RulerCore {
                config,
                cache_identity: Mutex::new(identity),
                cache_mode,
                cache: Mutex::new(cache),
                text_cache: Mutex::new(text_cache),
                stats: Stats::default(),
            }),
            #[cfg(feature = "http")]
            http: None,
        })
    }

    /// Binds the OpenAI-compatible HTTP transport, reading the API key from
    /// the configured environment variable. Blocking jobs drive its requests;
    /// the judge answers the async [`crate::RewardBackend`] API directly.
    #[cfg(feature = "http")]
    pub fn with_http(config: RulerConfig) -> Result<Self> {
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            Error::invalid(format!(
                "RULER API key environment variable '{}' is not set",
                config.api_key_env
            ))
        })?;
        Self::with_http_api_key(config, api_key)
    }

    /// Builds a judge with an explicit key, bound to the HTTP transport.
    /// Intended for embedders and tests that already manage secrets outside
    /// serialized configuration.
    #[cfg(feature = "http")]
    pub fn with_api_key(config: RulerConfig, api_key: String) -> Result<Self> {
        Self::with_http_api_key(config, api_key)
    }

    #[cfg(feature = "http")]
    fn with_http_api_key(config: RulerConfig, api_key: String) -> Result<Self> {
        config.validate()?;
        let core = Self::build(config.clone(), JudgeCacheMode::Persistent, None)?.core;
        let http = Some(http::HttpHandle::new(Arc::clone(&core), &config, api_key)?);
        Ok(Self { core, http })
    }

    /// Scores one group through `transport`: render, execute the waves, parse.
    ///
    /// This is the whole execution phase, and the transport is its only
    /// parameter: nothing here knows or cares whether the requests went over
    /// HTTP or through an in-process model.
    pub fn score_group_via(
        &self,
        transport: &mut dyn JudgeTransport,
        group: &TrajectoryGroup,
    ) -> Result<Vec<Score>> {
        self.core.score_group_via(transport, group)
    }

    /// Scores one group over the bound HTTP transport, from an async context.
    ///
    /// The work runs on Tokio's blocking pool, so the calling executor is
    /// never blocked and no runtime is nested inside it.
    #[cfg(feature = "http")]
    pub async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
        match &self.http {
            Some(handle) => handle.score_group(group).await,
            None => Err(Error::invalid(
                "this judge has no transport bound: score it with score_group_via and an \
                 explicit transport",
            )),
        }
    }

    pub fn stats(&self) -> RulerStats {
        self.core.stats()
    }

    /// Re-keys the verdict cache. An in-process judge whose weights moved
    /// (a refreshed snapshot) calls this so no verdict written under the old
    /// weights is ever read back.
    pub fn set_cache_identity(&self, identity: String) {
        let identity = serde_json::json!([
            identity,
            self.core.config.temperature,
            RULER_PROTOCOL_VERSION,
        ])
        .to_string();
        let mut current = self
            .core
            .cache_identity
            .lock()
            .expect("judge cache identity lock");
        if *current != identity {
            *current = identity;
            self.core
                .cache
                .lock()
                .expect("judge score cache lock")
                .clear();
            self.core
                .text_cache
                .lock()
                .expect("judge text cache lock")
                .clear();
        }
    }

    pub fn metric_values(&self) -> Vec<MetricValue> {
        self.core.metric_values()
    }
}

#[cfg(feature = "http")]
#[async_trait::async_trait]
impl RewardBackend for RulerJudge {
    async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
        RulerJudge::score_group(self, group).await
    }

    fn metric_values(&self) -> Vec<MetricValue> {
        self.core.metric_values()
    }
}

impl RulerCore {
    fn cache_identity(&self) -> String {
        self.cache_identity
            .lock()
            .expect("judge cache identity lock")
            .clone()
    }

    fn score_group_via(
        &self,
        transport: &mut dyn JudgeTransport,
        group: &TrajectoryGroup,
    ) -> Result<Vec<Score>> {
        let mut rendered = prompt::render_group(group, &self.config.context)?;
        self.stats.groups.fetch_add(1, Ordering::Relaxed);
        self.stats
            .rendered_chars
            .fetch_add(rendered.stats.rendered_chars as u64, Ordering::Relaxed);
        self.stats
            .elided_chars
            .fetch_add(rendered.stats.elided_chars as u64, Ordering::Relaxed);
        self.compact_group_via(transport, &mut rendered)?;
        match self.config.strategy {
            JudgeStrategy::Listwise => self.score_listwise_via(transport, &rendered),
            JudgeStrategy::Auto => {
                let total =
                    prompt::request_overhead(self.config.listwise_rubric(&rendered), &rendered)
                        + rendered
                            .trajectories
                            .iter()
                            .map(|trajectory| trajectory.chars)
                            .sum::<usize>();
                if total <= self.config.context.max_request_chars {
                    self.score_listwise_via(transport, &rendered)
                } else {
                    self.score_chunked_via(transport, &rendered, true)
                }
            }
            JudgeStrategy::Chunked { anchor } => {
                self.score_chunked_via(transport, &rendered, anchor)
            }
            JudgeStrategy::Pairwise {
                max_pairs,
                both_orders,
                aggregation,
            } => self.score_pairwise_via(transport, &rendered, max_pairs, both_orders, aggregation),
        }
    }

    fn stats(&self) -> RulerStats {
        let requests = self.stats.requests.load(Ordering::Relaxed);
        let latency = self.stats.latency_micros.load(Ordering::Relaxed);
        let groups = self.stats.groups.load(Ordering::Relaxed);
        let rendered = self.stats.rendered_chars.load(Ordering::Relaxed);
        let elided = self.stats.elided_chars.load(Ordering::Relaxed);
        let compacted = self.stats.compacted_chars.load(Ordering::Relaxed);
        let pairs = self.stats.pairs_judged.load(Ordering::Relaxed);
        RulerStats {
            requests,
            retries: self.stats.retries.load(Ordering::Relaxed),
            latency_ms_mean: if requests == 0 {
                0.0
            } else {
                latency as f32 / requests as f32 / 1000.0
            },
            cache_hits: self.stats.cache_hits.load(Ordering::Relaxed),
            groups,
            prompt_chars_max: self.stats.prompt_chars_max.load(Ordering::Relaxed),
            elided_fraction: if rendered + elided == 0 {
                0.0
            } else {
                elided as f32 / (rendered + elided) as f32
            },
            compacted_fraction: if rendered == 0 {
                0.0
            } else {
                compacted as f32 / rendered as f32
            },
            position_disagreement: if pairs == 0 {
                0.0
            } else {
                self.stats.position_disagreements.load(Ordering::Relaxed) as f32 / pairs as f32
            },
            requests_per_group: if groups == 0 {
                0.0
            } else {
                requests as f32 / groups as f32
            },
        }
    }

    fn metric_values(&self) -> Vec<MetricValue> {
        let stats = self.stats();
        [
            ("judge/latency_ms_mean", stats.latency_ms_mean),
            ("judge/retry_count", stats.retries as f32),
            ("judge/requests_per_group", stats.requests_per_group),
            ("judge/prompt_chars_max", stats.prompt_chars_max as f32),
            ("judge/elided_fraction", stats.elided_fraction),
            ("judge/compacted_fraction", stats.compacted_fraction),
            ("judge/position_disagreement", stats.position_disagreement),
            (
                "judge/cache_hit_fraction",
                if stats.requests + stats.cache_hits == 0 {
                    0.0
                } else {
                    stats.cache_hits as f32 / (stats.requests + stats.cache_hits) as f32
                },
            ),
        ]
        .into_iter()
        .map(|(name, value)| MetricValue {
            name: name.into(),
            value,
        })
        .collect()
    }

    /// Sends `requests` as one wave and parses each reply with
    /// `parse(request_index, request, content)`, retrying the requests that
    /// fail - transport error or unparsable answer - on their own, with a
    /// degraded response format, exactly as the HTTP path always did.
    ///
    /// Returns one result per request, in request order.
    fn complete_wave<T>(
        &self,
        transport: &mut dyn JudgeTransport,
        requests: Vec<JudgeRequest>,
        parse: impl Fn(usize, &JudgeRequest, &str) -> Result<T>,
    ) -> Vec<Result<T>> {
        let started = Instant::now();
        let count = requests.len();
        let mut attempts = vec![0_u32; count];
        let mut results: Vec<Option<Result<T>>> = (0..count).map(|_| None).collect();
        let mut pending: Vec<usize> = (0..count).collect();
        while !pending.is_empty() {
            let wave: Vec<JudgeRequest> = pending
                .iter()
                .map(|&index| self.wire_request(&requests[index], attempts[index]))
                .collect();
            for request in &wave {
                let chars = request.prompt.chars().count() as u64;
                self.stats
                    .prompt_chars_max
                    .fetch_max(chars, Ordering::Relaxed);
            }
            let replies = transport.complete(&wave);
            if replies.len() != wave.len() {
                for &index in &pending {
                    results[index] = Some(Err(JudgeTransportError::Arity {
                        requested: wave.len(),
                        returned: replies.len(),
                    }
                    .into()));
                }
                break;
            }
            let mut retry = Vec::new();
            for (slot, reply) in replies.into_iter().enumerate() {
                let index = pending[slot];
                let outcome = match reply {
                    Ok(reply) => parse(index, &requests[index], &reply.content),
                    Err(error) => Err(Error::from(error)),
                };
                match outcome {
                    Ok(value) => results[index] = Some(Ok(value)),
                    Err(error) => {
                        if attempts[index] >= self.config.max_retries {
                            results[index] = Some(Err(error));
                        } else {
                            attempts[index] += 1;
                            self.stats.retries.fetch_add(1, Ordering::Relaxed);
                            retry.push(index);
                        }
                    }
                }
            }
            // All failures in this wave back off together; sleeping once
            // per failed request would multiply the delay by the wave size.
            if let Some(attempt) = retry.iter().map(|&index| attempts[index]).max() {
                let backoff = 100_u64
                    .saturating_mul(1_u64 << attempt.min(8))
                    .saturating_add((u64::from(attempt) * 37) % 53);
                std::thread::sleep(Duration::from_millis(backoff));
            }
            pending = retry;
        }
        self.stats
            .requests
            .fetch_add(count as u64, Ordering::Relaxed);
        self.stats.latency_micros.fetch_add(
            started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
        results
            .into_iter()
            .map(|result| result.expect("every request ends with a result"))
            .collect()
    }

    /// The request as the transport sees it: the protocol states the schema in
    /// the prompt text - not only in `response_format`, because an endpoint
    /// that renders that field into prompt text has nothing to say about a bare
    /// `json_object` and prints "any object" - and asks for structured output
    /// first, degrading to the widest format on retries.
    ///
    /// The cache keys on `request.prompt` *before* this suffix, so rewording it
    /// does not invalidate a single cached verdict.
    fn wire_request(&self, request: &JudgeRequest, attempt: u32) -> JudgeRequest {
        let mut wire = request.clone();
        let shape = format!(
            "\n\nAnswer with a single JSON object matching this schema exactly, and nothing \
             else - no prose, no markdown fence:\n{}",
            request.schema
        );
        wire.prompt = format!("{}{shape}", request.prompt);
        // Structured decoding first, then the widest format the OpenAI
        // protocol has: an endpoint that rejects `json_schema` outright gets
        // a second chance at answering. That degradation is about the
        // transport only - `shape` above carries the schema either way.
        wire.response_format = Some(if attempt == 0 {
            serde_json::json!({
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name,
                    "strict": true,
                    "schema": request.schema
                }
            })
        } else {
            serde_json::json!({"type": "json_object"})
        });
        wire
    }

    /// Summarises the middle of every member, or of none of them, in one wave
    /// through the same transport as the verdicts - a local judge must not hide
    /// an HTTP call in its compaction.
    ///
    /// The decision is taken once from the group's longest member, and the same
    /// budget is applied to all: a member compacted harder than its siblings
    /// would be handicapped for a reason that has nothing to do with what it
    /// did, and that handicap would land directly in the GRPO advantage. A
    /// member whose middle is empty simply costs no request.
    fn compact_group_via(
        &self,
        transport: &mut dyn JudgeTransport,
        rendered: &mut prompt::RenderedGroup,
    ) -> Result<()> {
        let Some(config) = &self.config.compaction else {
            return Ok(());
        };
        let sizes = rendered
            .trajectories
            .iter()
            .map(|trajectory| trajectory.chars)
            .collect::<Vec<_>>();
        if !compact::plan_group(config, &sizes) {
            return Ok(());
        }
        // One entry per member: (trajectory index, range, cache key, cached summary).
        let mut members: Vec<(usize, std::ops::Range<usize>, String, Option<String>)> = Vec::new();
        for (index, trajectory) in rendered.trajectories.iter().enumerate() {
            let range = config.middle(trajectory.messages.len());
            if range.is_empty() {
                continue;
            }
            let prompt = compact::compaction_prompt(config, &trajectory.messages[range.clone()])?;
            let key = cache_key(&self.cache_identity(), &prompt);
            let cached = self.cached_text(&key);
            members.push((index, range, key, cached));
        }
        // The wave carries only the members without a cached summary; a failed
        // summary is not a failed group: that member keeps its full transcript,
        // which is more context than its siblings rather than less, and the
        // request budget is the only thing that suffers.
        let unsent: Vec<usize> = members
            .iter()
            .enumerate()
            .filter(|(_, (_, _, _, cached))| cached.is_none())
            .map(|(position, _)| position)
            .collect();
        let requests: Vec<JudgeRequest> = unsent
            .iter()
            .map(|&position| {
                let (index, range, _, _) = &members[position];
                let prompt = compact::compaction_prompt(
                    config,
                    &rendered.trajectories[*index].messages[range.clone()],
                )
                .expect("the prompt was built once above for the same range");
                // A summary is never a verdict: temperature 0, whatever the
                // judge samples its verdicts at.
                JudgeRequest::structured(
                    prompt,
                    compact::compaction_schema(),
                    "ruler_summary",
                    Some(0.0),
                )
            })
            .collect();
        if !requests.is_empty() {
            let results = self.complete_wave(transport, requests, |_, _, content| {
                compact::parse_compaction(content)
            });
            for (&position, result) in unsent.iter().zip(results) {
                let (_, _, key, cached) = &mut members[position];
                match result {
                    Ok(summary) => {
                        // The budget is advisory for the model and enforced here,
                        // so one verbose summary cannot undo the budgeting the
                        // whole pipeline exists for.
                        let (summary, _) = crate::render::elide(
                            &summary,
                            config.target_chars,
                            self.config.context.head_ratio,
                        );
                        self.store_text(key.clone(), &summary)?;
                        *cached = Some(summary);
                    }
                    Err(error) => {
                        tracing::warn!("compacting a judge transcript failed: {error}");
                    }
                }
            }
        }
        for (index, range, _, summary) in members {
            let Some(summary) = summary else { continue };
            let trajectory = &mut rendered.trajectories[index];
            let before = trajectory
                .messages
                .iter()
                .map(|message| message.content.chars().count())
                .sum::<usize>();
            compact::splice(&mut trajectory.messages, range, &summary);
            let after = trajectory
                .messages
                .iter()
                .map(|message| message.content.chars().count())
                .sum::<usize>();
            self.stats
                .compacted_chars
                .fetch_add(before.saturating_sub(after) as u64, Ordering::Relaxed);
            trajectory.chars = after
                + trajectory
                    .env_summary
                    .as_ref()
                    .map_or(0, |summary| summary.chars().count());
        }
        Ok(())
    }

    fn cached(&self, key: &str) -> Option<Vec<Score>> {
        if self.cache_mode == JudgeCacheMode::Disabled {
            return None;
        }
        let hit = self
            .cache
            .lock()
            .expect("judge score cache lock")
            .get(key)
            .cloned();
        if hit.is_some() {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    fn store(&self, key: String, scores: &[Score]) -> Result<()> {
        if self.cache_mode == JudgeCacheMode::Disabled {
            return Ok(());
        }
        if self.cache_mode == JudgeCacheMode::Persistent {
            append_cache(
                self.config.cache_path.as_ref(),
                CacheRow {
                    key: key.clone(),
                    scores: Some(scores.to_vec()),
                    text: None,
                },
            )?;
        }
        self.cache
            .lock()
            .expect("judge score cache lock")
            .insert(key, scores.to_vec());
        Ok(())
    }

    fn cached_text(&self, key: &str) -> Option<String> {
        if self.cache_mode == JudgeCacheMode::Disabled {
            return None;
        }
        let hit = self
            .text_cache
            .lock()
            .expect("judge text cache lock")
            .get(key)
            .cloned();
        if hit.is_some() {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
        }
        hit
    }

    fn store_text(&self, key: String, text: &str) -> Result<()> {
        if self.cache_mode == JudgeCacheMode::Disabled {
            return Ok(());
        }
        if self.cache_mode == JudgeCacheMode::Persistent {
            append_cache(
                self.config.cache_path.as_ref(),
                CacheRow {
                    key: key.clone(),
                    scores: None,
                    text: Some(text.to_owned()),
                },
            )?;
        }
        self.text_cache
            .lock()
            .expect("judge text cache lock")
            .insert(key, text.to_owned());
        Ok(())
    }
}

/// The cache key of one request: the judge identity (model, and for an
/// in-process judge the weights fingerprint and sampling parameters) plus the
/// prompt, hashed together.
pub(crate) fn cache_key(identity: &str, prompt: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in identity.bytes().chain([0]).chain(prompt.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

type Caches = (HashMap<String, Vec<Score>>, HashMap<String, String>);

/// A failure of the on-disk verdict cache, naming the operation that hit it.
#[derive(Debug, thiserror::Error)]
pub(crate) enum JudgeCacheError {
    #[error("{op}: {source}")]
    Io {
        op: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("{op}: {source}")]
    Json {
        op: &'static str,
        #[source]
        source: serde_json::Error,
    },
    #[error("judge cache row '{key}' carries neither scores nor a summary")]
    EmptyRow { key: String },
}

/// The cache belongs to the reward backend, so a cache that cannot be read or
/// written is a reward failure.
impl From<JudgeCacheError> for Error {
    fn from(error: JudgeCacheError) -> Self {
        Error::Reward(error.to_string())
    }
}

fn load_cache(path: Option<&PathBuf>) -> Result<Caches> {
    let mut caches: Caches = (HashMap::new(), HashMap::new());
    let Some(path) = path else {
        return Ok(caches);
    };
    if !path.exists() {
        return Ok(caches);
    }
    let file = File::open(path).map_err(|source| JudgeCacheError::Io {
        op: "open judge cache",
        source,
    })?;
    for line in BufReader::new(file).lines() {
        let line = line.map_err(|source| JudgeCacheError::Io {
            op: "read judge cache",
            source,
        })?;
        let row: CacheRow =
            serde_json::from_str(&line).map_err(|source| JudgeCacheError::Json {
                op: "parse judge cache",
                source,
            })?;
        match (row.scores, row.text) {
            (Some(scores), _) => {
                caches.0.insert(row.key, scores);
            }
            (None, Some(text)) => {
                caches.1.insert(row.key, text);
            }
            (None, None) => {
                return Err(JudgeCacheError::EmptyRow { key: row.key }.into());
            }
        }
    }
    Ok(caches)
}

fn append_cache(path: Option<&PathBuf>, row: CacheRow) -> Result<()> {
    static CACHE_FILE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    let Some(path) = path else {
        return Ok(());
    };
    let _guard = CACHE_FILE_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("judge cache append lock");
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|error| Error::Reward(format!("open judge cache for append: {error}")))?;
    let mut encoded = serde_json::to_vec(&row)
        .map_err(|error| Error::Reward(format!("encode judge cache: {error}")))?;
    encoded.push(b'\n');
    file.write_all(&encoded)
        .map_err(|error| Error::Reward(format!("write judge cache row: {error}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::trajectory::{Message, Role, Trajectory};

    type TransportResult = std::result::Result<JudgeReply, JudgeTransportError>;

    /// A transport that answers from a queue of canned replies, recording the
    /// prompts it was sent.
    struct FakeTransport {
        replies: std::collections::VecDeque<TransportResult>,
        prompts: Vec<String>,
    }

    impl FakeTransport {
        fn new(replies: Vec<TransportResult>) -> Self {
            Self {
                replies: replies.into(),
                prompts: Vec::new(),
            }
        }
    }

    impl JudgeTransport for FakeTransport {
        fn complete(&mut self, requests: &[JudgeRequest]) -> Vec<TransportResult> {
            for request in requests {
                self.prompts.push(request.prompt.clone());
            }
            requests
                .iter()
                .map(|_| {
                    self.replies
                        .pop_front()
                        .expect("the fake transport was given a reply per request")
                })
                .collect()
        }
    }

    fn trajectory(answer: &str) -> Trajectory {
        Trajectory {
            scenario_id: "s".into(),
            messages: vec![
                Message::text(Role::System, "common"),
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
        }
    }

    fn group(answers: &[&str]) -> TrajectoryGroup {
        TrajectoryGroup {
            group_id: 1,
            scenario_id: "s".into(),
            trajectories: answers.iter().map(|answer| trajectory(answer)).collect(),
        }
    }

    /// A `{"scores":[...]}` listwise judge response, built from
    /// `(trajectory_id, explanation, score)` triples.
    fn scores_json(entries: &[(usize, &str, f64)]) -> String {
        let scores: Vec<_> = entries
            .iter()
            .map(|&(trajectory_id, explanation, score)| {
                serde_json::json!({
                    "trajectory_id": trajectory_id,
                    "explanation": explanation,
                    "score": score,
                })
            })
            .collect();
        serde_json::json!({ "scores": scores }).to_string()
    }

    fn reply(content: &str) -> TransportResult {
        Ok(JudgeReply {
            content: content.to_owned(),
        })
    }

    /// The configuration a transport-agnostic judge is built with: the
    /// transport fields are placeholders, they are validated only when a
    /// transport is bound.
    fn local_config() -> RulerConfig {
        RulerConfig {
            base_url: String::new(),
            model: "local".into(),
            api_key_env: String::new(),
            strategy: JudgeStrategy::Listwise,
            max_retries: 0,
            ..Default::default()
        }
    }

    #[test]
    fn a_group_is_scored_through_any_transport_render_transport_parse() {
        let judge = RulerJudge::new(RulerConfig {
            timeout_secs: 0,
            max_concurrency: 0,
            ..local_config()
        })
        .unwrap();
        let mut transport = FakeTransport::new(vec![reply(&scores_json(&[
            (0, "ok", 0.75),
            (1, "ok", 0.25),
        ]))]);
        let scores = judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        assert_eq!(scores[0].value, 0.75);
        assert_eq!(scores[1].value, 0.25);
        // The transport saw the prompt with the schema stated in the text.
        assert_eq!(transport.prompts.len(), 1);
        assert!(transport.prompts[0].contains("Answer with a single JSON object"));
        assert!(transport.prompts[0].contains("\"scores\""));
        assert_eq!(judge.stats().requests, 1);
    }

    #[test]
    fn a_malformed_reply_is_retried_with_a_degraded_format_then_accepted() {
        let mut config = local_config();
        config.max_retries = 1;
        let judge = RulerJudge::new(config).unwrap();
        let mut transport = FakeTransport::new(vec![
            reply("not JSON"),
            reply(&scores_json(&[(0, "ok", 0.4), (1, "ok", 0.6)])),
        ]);
        let scores = judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        assert_eq!(scores[1].value, 0.6);
        assert_eq!(judge.stats().retries, 1);
        assert_eq!(judge.stats().requests, 1);
        // The retry asked for the widest format instead of a strict schema.
        assert!(transport.prompts[1].contains("Answer with a single JSON object"));
    }

    #[test]
    fn a_transport_error_is_a_retryable_request_failure() {
        let mut config = local_config();
        config.max_retries = 0;
        let judge = RulerJudge::new(config).unwrap();
        let mut transport =
            FakeTransport::new(vec![Err(JudgeTransportError::Failed("boom".into()))]);
        let error = judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap_err();
        assert!(error.to_string().contains("boom"), "{error}");
    }

    #[test]
    fn a_cached_verdict_is_reused_without_touching_the_transport() {
        let judge = RulerJudge::new(local_config()).unwrap();
        let mut transport = FakeTransport::new(vec![reply(&scores_json(&[
            (0, "ok", 0.75),
            (1, "ok", 0.25),
        ]))]);
        let first = judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        // The transport is empty now: a second scoring can only come from the cache.
        let second = judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        assert_eq!(first[0].value, second[0].value);
        assert_eq!(judge.stats().requests, 1);
        assert_eq!(judge.stats().cache_hits, 1);
    }

    #[test]
    fn a_disabled_cache_never_reads_nor_writes() {
        let judge = RulerJudge::with_cache(
            local_config(),
            JudgeCacheMode::Disabled,
            "local:policy".into(),
        )
        .unwrap();
        let mut transport = FakeTransport::new(vec![
            reply(&scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)])),
            reply(&scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)])),
        ]);
        judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        // Both scorings hit the transport: nothing was cached.
        assert_eq!(transport.prompts.len(), 2);
        assert_eq!(judge.stats().cache_hits, 0);
    }

    #[test]
    fn the_cache_identity_separates_judges_that_share_a_prompt() {
        // Two judges, two identities, same prompt: the second must not read the
        // first's verdict. This is what a weights fingerprint buys an
        // in-process judge.
        let first =
            RulerJudge::with_cache(local_config(), JudgeCacheMode::Memory, "v1".into()).unwrap();
        let second =
            RulerJudge::with_cache(local_config(), JudgeCacheMode::Memory, "v2".into()).unwrap();
        let mut transport = FakeTransport::new(vec![
            reply(&scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)])),
            reply(&scores_json(&[(0, "ok", 0.1), (1, "ok", 0.9)])),
        ]);
        let a = first
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        let b = second
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        assert_eq!(a[0].value, 0.75);
        assert_eq!(b[0].value, 0.1);
        // And the same judge under a new identity does not read the old one.
        let third =
            RulerJudge::with_cache(local_config(), JudgeCacheMode::Memory, "v3".into()).unwrap();
        let mut transport =
            FakeTransport::new(vec![reply(&scores_json(&[(0, "ok", 0.5), (1, "ok", 0.5)]))]);
        let c = third
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        assert_eq!(c[0].value, 0.5);
    }

    #[test]
    fn compaction_goes_through_the_same_transport_not_a_hidden_call() {
        let mut config = local_config();
        config.compaction = Some(CompactionConfig {
            trigger_chars: 500,
            target_chars: 200,
            keep_last: 2,
        });
        let judge = RulerJudge::new(config).unwrap();
        let long = |answer: &str| {
            let mut trajectory = trajectory(answer);
            for turn in 0..8 {
                trajectory.messages.push(Message::text(
                    Role::Assistant,
                    format!("middle-{turn} {}", "y".repeat(200)),
                ));
            }
            trajectory
                .messages
                .push(Message::text(Role::Assistant, "done"));
            trajectory
        };
        let group = TrajectoryGroup {
            group_id: 1,
            scenario_id: "s".into(),
            trajectories: vec![long("a"), long("b")],
        };
        let mut transport = FakeTransport::new(vec![
            reply(r#"{"summary":"they poked at it"}"#),
            reply(r#"{"summary":"they poked at it"}"#),
            reply(&scores_json(&[(0, "ok", 0.4), (1, "ok", 0.6)])),
        ]);
        let scores = judge.score_group_via(&mut transport, &group).unwrap();
        assert_eq!(scores.len(), 2);
        // Two summary requests - one wave - then the verdict: all through the
        // one transport, and the summaries at temperature 0.
        assert_eq!(transport.prompts.len(), 3);
        assert!(transport.prompts[0].contains("Summarise the middle"));
        assert!(judge.stats().compacted_fraction > 0.0);
    }

    #[test]
    fn pairwise_runs_its_whole_schedule_through_one_transport() {
        let mut config = local_config();
        config.strategy = JudgeStrategy::Pairwise {
            max_pairs: None,
            both_orders: false,
            aggregation: Aggregation::WinRate,
        };
        let judge = RulerJudge::new(config).unwrap();
        let mut transport = FakeTransport::new(vec![
            reply(r#"{"winner":"a","explanation":"0>1"}"#),
            reply(r#"{"winner":"a","explanation":"1>2"}"#),
            reply(r#"{"winner":"b","explanation":"2>0"}"#),
        ]);
        let scores = judge
            .score_group_via(&mut transport, &group(&["alpha", "beta", "gamma"]))
            .unwrap();
        // Ring (0,1), (1,2), (2,0): 0 won both its matches, 1 won one of two,
        // 2 lost both.
        assert_eq!(scores[0].value, 1.0);
        assert_eq!(scores[1].value, 0.5);
        assert_eq!(scores[2].value, 0.0);
        assert_eq!(transport.prompts.len(), 3);
        assert_eq!(judge.stats().requests, 3);
    }

    #[test]
    fn a_transport_that_returns_the_wrong_arity_fails_the_wave() {
        struct ShortTransport;
        impl JudgeTransport for ShortTransport {
            fn complete(&mut self, requests: &[JudgeRequest]) -> Vec<TransportResult> {
                (0..requests.len().saturating_sub(1))
                    .map(|_| reply("{}"))
                    .collect()
            }
        }
        let judge = RulerJudge::new(local_config()).unwrap();
        let mut transport = ShortTransport;
        let error = judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap_err();
        assert!(error.to_string().contains("replies for"), "{error}");
    }

    /// A cache failure reaches the agent stack as a reward failure, with the
    /// operation that hit it still in the message.
    #[test]
    fn a_cache_failure_is_a_reward_error_that_names_the_operation() {
        let error: Error = JudgeCacheError::Io {
            op: "open judge cache",
            source: std::io::Error::from(std::io::ErrorKind::PermissionDenied),
        }
        .into();
        assert!(matches!(error, Error::Reward(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            "reward backend error: open judge cache: permission denied"
        );
        let empty: Error = JudgeCacheError::EmptyRow { key: "k".into() }.into();
        assert_eq!(
            empty.to_string(),
            "reward backend error: judge cache row 'k' carries neither scores nor a summary"
        );
    }

    #[test]
    fn strategies_and_context_round_trip_through_serde() {
        let config: RulerConfig = serde_json::from_str(
            r#"{"base_url":"http://x/v1","model":"m","strategy":{"mode":"pairwise","max_pairs":12},
                "context":{"max_request_chars":1000,"max_trajectory_chars":500,"max_message_chars":100,"head_ratio":0.5}}"#,
        )
        .unwrap();
        assert_eq!(
            config.strategy,
            JudgeStrategy::Pairwise {
                max_pairs: Some(12),
                both_orders: true,
                aggregation: Aggregation::WinRate,
            }
        );
        assert_eq!(config.context.max_request_chars, 1000);
        config.validate().unwrap();

        // Defaults stay listwise-with-fallback and a sane budget.
        let minimal: RulerConfig =
            serde_json::from_str(r#"{"base_url":"http://x/v1","model":"m"}"#).unwrap();
        assert_eq!(minimal.strategy, JudgeStrategy::Auto);
        assert_eq!(minimal.context, JudgeContext::default());
        minimal.validate().unwrap();

        let invalid: RulerConfig = serde_json::from_str(
            r#"{"base_url":"http://x/v1","model":"m","strategy":{"mode":"pairwise","max_pairs":0}}"#,
        )
        .unwrap();
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn the_cache_round_trips_through_its_jsonl_file() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let path = tmp.path().join("cache.jsonl");
        let scores = vec![Score {
            value: 0.5,
            valid: true,
            explanation: Some("why".into()),
            error: None,
        }];
        let row = |key: &str| CacheRow {
            key: key.to_owned(),
            scores: Some(scores.clone()),
            text: None,
        };
        append_cache(Some(&path), row("k1")).unwrap();
        append_cache(Some(&path), row("k2")).unwrap();
        // Verdicts and summaries share the file and are read back into their own
        // maps: a summary must never be handed out as a score.
        append_cache(
            Some(&path),
            CacheRow {
                key: "k3".into(),
                scores: None,
                text: Some("a summary".into()),
            },
        )
        .unwrap();
        let (cache, texts) = load_cache(Some(&path)).unwrap();
        assert_eq!(cache.len(), 2);
        assert_eq!(texts.get("k3").map(String::as_str), Some("a summary"));
        assert_eq!(cache.get("k1"), Some(&scores));
    }

    #[test]
    fn nonpersistent_modes_ignore_an_existing_invalid_disk_cache() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache.jsonl");
        std::fs::write(&path, "not JSON").unwrap();
        for mode in [JudgeCacheMode::Memory, JudgeCacheMode::Disabled] {
            let config = RulerConfig {
                cache_path: Some(path.clone()),
                ..local_config()
            };
            let judge = RulerJudge::with_cache(config, mode, "local".into()).unwrap();
            assert!(judge.core.cache.lock().unwrap().is_empty());
            assert!(judge.core.text_cache.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn a_failed_disk_write_is_not_reused_as_a_successful_verdict() {
        let tmp = tempfile::tempdir().unwrap();
        let config = RulerConfig {
            cache_path: Some(tmp.path().join("missing/cache.jsonl")),
            ..local_config()
        };
        let judge = RulerJudge::new(config).unwrap();
        let content = scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)]);
        let mut transport = FakeTransport::new(vec![reply(&content), reply(&content)]);
        for _ in 0..2 {
            assert!(
                judge
                    .score_group_via(&mut transport, &group(&["a", "b"]))
                    .is_err()
            );
        }
        assert_eq!(transport.prompts.len(), 2);
        assert!(judge.core.cache.lock().unwrap().is_empty());
        assert!(judge.core.store_text("summary".into(), "text").is_err());
        assert!(judge.core.cached_text("summary").is_none());
    }

    #[test]
    fn changing_the_identity_evicts_previous_in_memory_verdicts() {
        let judge =
            RulerJudge::with_cache(local_config(), JudgeCacheMode::Memory, "v1".into()).unwrap();
        judge.core.store("key".into(), &[]).unwrap();
        judge.core.store_text("summary".into(), "text").unwrap();
        judge.set_cache_identity("v1".into());
        assert!(judge.core.cached("key").is_some());
        judge.set_cache_identity("v2".into());
        assert!(judge.core.cached("key").is_none());
        assert!(judge.core.cached_text("summary").is_none());
    }

    #[test]
    fn a_memory_only_cache_still_shares_nothing_with_the_disk() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let path = tmp.path().join("cache.jsonl");
        let mut config = local_config();
        config.cache_path = Some(path.clone());
        let judge = RulerJudge::with_cache(config, JudgeCacheMode::Memory, "local".into()).unwrap();
        let mut transport = FakeTransport::new(vec![
            reply(&scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)])),
            reply(&scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)])),
        ]);
        judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        // Served from memory on the second call...
        judge
            .score_group_via(&mut transport, &group(&["a", "b"]))
            .unwrap();
        assert_eq!(judge.stats().cache_hits, 1);
        // ...but nothing was written to disk.
        assert!(!path.exists());
    }
}
