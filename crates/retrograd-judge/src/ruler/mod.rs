// Everything below is what only the judge itself uses. Its transport is behind
// `http`, so a build whose rewards are all commands links none of it - and the
// re-exported declarations (`RulerConfig`, `JudgeStrategy`) stay available to a
// configuration parser. Every import here is therefore gated, including the
// ones a `use` at the top of the file would make look unconditional.
#[cfg(feature = "http")]
use futures_util::future::join_all;
#[cfg(feature = "http")]
use std::collections::HashMap;
#[cfg(feature = "http")]
use std::fs::{File, OpenOptions};
#[cfg(feature = "http")]
use std::io::{BufRead, BufReader, Write};
#[cfg(feature = "http")]
use std::path::PathBuf;
#[cfg(feature = "http")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "http")]
use std::sync::{Arc, Mutex, OnceLock};
#[cfg(feature = "http")]
use std::time::{Duration, Instant};

#[cfg(feature = "http")]
use async_trait::async_trait;
#[cfg(feature = "http")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "http")]
use crate::render::compact::CompactionConfig;
#[cfg(feature = "http")]
use retrograd_agent_core::{Error, Result};

#[cfg(feature = "http")]
use retrograd_agent_core::trajectory::TrajectoryGroup;

#[cfg(feature = "http")]
use crate::client::OpenAiClient;
#[cfg(feature = "http")]
use crate::render::compact;
#[cfg(feature = "http")]
use crate::render::prompt::{self, DEFAULT_PAIRWISE_RUBRIC, DEFAULT_RUBRIC, RenderedGroup};
#[cfg(feature = "http")]
use crate::{RewardBackend, Score};
#[cfg(feature = "http")]
use retrograd_llm_client::Purpose;

#[cfg(test)]
use retrograd_spec::judge::{Aggregation, JudgeContext};
pub use retrograd_spec::judge::{JudgeStrategy, RulerConfig};

/// The rubric a rendered group is judged against.
///
/// It reads a `RenderedGroup`, which is this crate's rendering and not part of
/// the schema - so it is added to the declaration here rather than moved with
/// it.
#[cfg(feature = "http")]
trait Rubrics {
    fn listwise_rubric<'a>(&'a self, rendered: &'a RenderedGroup) -> &'a str;
    fn pairwise_rubric(&self, rendered: &RenderedGroup) -> String;
}

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
mod chunked;
#[cfg(feature = "http")]
mod listwise;
#[cfg(feature = "http")]
mod pairwise;

#[cfg(feature = "http")]
pub struct RulerJudge {
    config: RulerConfig,
    client: OpenAiClient,
    semaphore: Arc<tokio::sync::Semaphore>,
    cache: Mutex<HashMap<String, Vec<Score>>>,
    text_cache: Mutex<HashMap<String, String>>,
    stats: Stats,
}

#[cfg(feature = "http")]
impl RulerJudge {
    pub fn new(config: RulerConfig) -> Result<Self> {
        config.validate()?;
        let api_key = std::env::var(&config.api_key_env).map_err(|_| {
            Error::invalid(format!(
                "RULER API key environment variable '{}' is not set",
                config.api_key_env
            ))
        })?;
        Self::with_api_key(config, api_key)
    }

    /// Builds a judge with an explicit key. Intended for embedders and tests
    /// that already manage secrets outside serialized configuration.
    pub fn with_api_key(config: RulerConfig, api_key: String) -> Result<Self> {
        config.validate()?;
        let client = OpenAiClient::new(
            &config.base_url,
            api_key,
            Duration::from_secs(config.timeout_secs),
            8 * 1024 * 1024,
            Purpose::Judge,
        )?;
        let (cache, text_cache) = load_cache(config.cache_path.as_ref())?;
        Ok(Self {
            semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrency)),
            config,
            client,
            cache: Mutex::new(cache),
            text_cache: Mutex::new(text_cache),
            stats: Stats::default(),
        })
    }

    pub fn stats(&self) -> RulerStats {
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

    /// Sends one prompt and parses it with `parse`, retrying on a malformed
    /// response. Every attempt states the schema in the prompt; the second one
    /// also drops the JSON-schema *request format*, which some
    /// OpenAI-compatible endpoints reject outright.
    pub(super) async fn request<T>(
        &self,
        prompt: &str,
        schema: serde_json::Value,
        schema_name: &str,
        parse: impl Fn(&str) -> Result<T>,
    ) -> Result<T> {
        self.request_at(self.config.temperature, prompt, schema, schema_name, parse)
            .await
    }

    /// Same request, with the sampling temperature stated explicitly.
    ///
    /// Verdicts use whatever the configuration asks for; compaction does not get
    /// that choice (see [`Self::summarise`]).
    async fn request_at<T>(
        &self,
        temperature: Option<f32>,
        prompt: &str,
        schema: serde_json::Value,
        schema_name: &str,
        parse: impl Fn(&str) -> Result<T>,
    ) -> Result<T> {
        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|_| Error::Reward("judge semaphore closed".into()))?;
        let chars = prompt.chars().count() as u64;
        self.stats
            .prompt_chars_max
            .fetch_max(chars, Ordering::Relaxed);
        let started = Instant::now();
        let mut last_error = None;
        // The schema is stated in the prompt, not only in `response_format`: an
        // endpoint that renders that field into prompt text has nothing to say
        // about a bare `json_object` and prints "any object", so a judge never
        // told about `winner` and `explanation` answers with field names of its
        // own invention, `parse` rejects it, and every retry burns on the same
        // failure. In the text the shape survives whatever the transport drops.
        //
        // The callers key their cache on `prompt` alone, before this suffix, so
        // rewording it does not invalidate a single cached verdict.
        let shape = format!(
            "\n\nAnswer with a single JSON object matching this schema exactly, and nothing \
             else - no prose, no markdown fence:\n{schema}"
        );
        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                self.stats.retries.fetch_add(1, Ordering::Relaxed);
                let backoff = 100_u64
                    .saturating_mul(1_u64 << attempt.min(8))
                    .saturating_add((attempt as u64 * 37) % 53);
                tokio::time::sleep(Duration::from_millis(backoff)).await;
            }
            // Structured decoding first, then the widest format the OpenAI
            // protocol has: an endpoint that rejects `json_schema` outright gets
            // a second chance at answering. That degradation is about the
            // transport only - `shape` above carries the schema either way.
            let response_format = if attempt == 0 {
                serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": true,
                        "schema": schema
                    }
                })
            } else {
                serde_json::json!({"type": "json_object"})
            };
            let mut payload = serde_json::json!({
                "model": self.config.model,
                "messages": [{"role": "user", "content": format!("{prompt}{shape}")}],
                "response_format": response_format
            });
            if let Some(temperature) = temperature {
                payload["temperature"] = serde_json::json!(temperature);
            }
            match self
                .client
                .chat_completion(&payload)
                .await
                .and_then(|completion| parse(&completion.content))
            {
                Ok(parsed) => {
                    self.record_request(started);
                    return Ok(parsed);
                }
                Err(error) => last_error = Some(error),
            }
        }
        self.record_request(started);
        Err(last_error.unwrap_or_else(|| Error::Reward("judge failed without an error".into())))
    }

    pub(super) fn record_request(&self, started: Instant) {
        self.stats.requests.fetch_add(1, Ordering::Relaxed);
        self.stats.latency_micros.fetch_add(
            started.elapsed().as_micros().min(u64::MAX as u128) as u64,
            Ordering::Relaxed,
        );
    }

    /// Summarises the middle of every member, or of none of them.
    ///
    /// The decision is taken once from the group's longest member, and the same
    /// budget is applied to all: a member compacted harder than its siblings
    /// would be handicapped for a reason that has nothing to do with what it
    /// did, and that handicap would land directly in the GRPO advantage. A
    /// member whose middle is empty simply costs no request.
    async fn compact_group(&self, rendered: &mut prompt::RenderedGroup) -> Result<()> {
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
        let summaries = join_all(rendered.trajectories.iter().map(|trajectory| {
            let range = config.middle(trajectory.messages.len());
            async move {
                if range.is_empty() {
                    return Ok(None);
                }
                self.summarise(config, &trajectory.messages[range])
                    .await
                    .map(Some)
            }
        }))
        .await;
        for (trajectory, summary) in rendered.trajectories.iter_mut().zip(summaries) {
            // A failed summary is not a failed group: that member keeps its full
            // transcript, which is more context than its siblings rather than
            // less, and the request budget is the only thing that suffers.
            let Some(summary) = summary.unwrap_or_else(|error| {
                tracing::warn!("compacting a judge transcript failed: {error}");
                None
            }) else {
                continue;
            };
            let range = config.middle(trajectory.messages.len());
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

    /// One summary request, deterministic and cached by content hash: the same
    /// transcript must compact to the same text on a replay, otherwise the
    /// verdict cache below never hits again.
    ///
    /// The temperature is forced to zero here rather than inherited from
    /// [`RulerConfig::temperature`], and the cache is not what makes this true:
    /// a cache makes the *second* summary stable, while the first one is what
    /// ends up in the prompt every member is judged from. Two runs without a
    /// shared cache would otherwise compact the same transcripts differently,
    /// and produce different rewards from identical trajectories. A judge
    /// configured to sample its verdicts is free to do so; summarising is not a
    /// verdict.
    async fn summarise(
        &self,
        config: &CompactionConfig,
        messages: &[crate::render::JudgeMessage],
    ) -> Result<String> {
        let prompt = compact::compaction_prompt(config, messages)?;
        let key = cache_key(&self.config.model, &prompt);
        if let Some(summary) = self.cached_text(&key) {
            return Ok(summary);
        }
        let summary = self
            .request_at(
                Some(0.0),
                &prompt,
                compact::compaction_schema(),
                "ruler_summary",
                compact::parse_compaction,
            )
            .await?;
        // The budget is advisory for the model and enforced here, so one verbose
        // summary cannot undo the budgeting the whole pipeline exists for.
        let (summary, _) = crate::render::elide(
            &summary,
            config.target_chars,
            self.config.context.head_ratio,
        );
        self.store_text(key, &summary)?;
        Ok(summary)
    }

    pub(super) fn cached(&self, key: &str) -> Option<Vec<Score>> {
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

    pub(super) fn store(&self, key: String, scores: &[Score]) -> Result<()> {
        self.cache
            .lock()
            .expect("judge score cache lock")
            .insert(key.clone(), scores.to_vec());
        append_cache(
            self.config.cache_path.as_ref(),
            CacheRow {
                key,
                scores: Some(scores.to_vec()),
                text: None,
            },
        )
    }

    fn cached_text(&self, key: &str) -> Option<String> {
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
        self.text_cache
            .lock()
            .expect("judge text cache lock")
            .insert(key.clone(), text.to_owned());
        append_cache(
            self.config.cache_path.as_ref(),
            CacheRow {
                key,
                scores: None,
                text: Some(text.to_owned()),
            },
        )
    }
}

#[cfg(feature = "http")]
#[async_trait]
impl RewardBackend for RulerJudge {
    async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
        let mut rendered = prompt::render_group(group, &self.config.context)?;
        self.stats.groups.fetch_add(1, Ordering::Relaxed);
        self.stats
            .rendered_chars
            .fetch_add(rendered.stats.rendered_chars as u64, Ordering::Relaxed);
        self.stats
            .elided_chars
            .fetch_add(rendered.stats.elided_chars as u64, Ordering::Relaxed);
        self.compact_group(&mut rendered).await?;
        match self.config.strategy {
            JudgeStrategy::Listwise => self.score_listwise(&rendered).await,
            JudgeStrategy::Auto => {
                let total =
                    prompt::request_overhead(self.config.listwise_rubric(&rendered), &rendered)
                        + rendered
                            .trajectories
                            .iter()
                            .map(|trajectory| trajectory.chars)
                            .sum::<usize>();
                if total <= self.config.context.max_request_chars {
                    self.score_listwise(&rendered).await
                } else {
                    self.score_chunked(&rendered, true).await
                }
            }
            JudgeStrategy::Chunked { anchor } => self.score_chunked(&rendered, anchor).await,
            JudgeStrategy::Pairwise {
                max_pairs,
                both_orders,
                aggregation,
            } => {
                self.score_pairwise(&rendered, max_pairs, both_orders, aggregation)
                    .await
            }
        }
    }

    fn metric_values(&self) -> Vec<retrograd_metrics::MetricValue> {
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
        .map(|(name, value)| retrograd_metrics::MetricValue {
            name: name.into(),
            value,
        })
        .collect()
    }
}

#[cfg(feature = "http")]
pub(super) fn cache_key(model: &str, prompt: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in model.bytes().chain([0]).chain(prompt.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

#[cfg(feature = "http")]
type Caches = (HashMap<String, Vec<Score>>, HashMap<String, String>);

/// A failure of the on-disk verdict cache, naming the operation that hit it.
#[cfg(feature = "http")]
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
#[cfg(feature = "http")]
impl From<JudgeCacheError> for Error {
    fn from(error: JudgeCacheError) -> Self {
        Error::Reward(error.to_string())
    }
}

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
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

#[cfg(feature = "http")]
#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    use super::*;
    use retrograd_agent_core::trajectory::{Message, Role, Trajectory};

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

    fn config(address: std::net::SocketAddr) -> RulerConfig {
        RulerConfig {
            base_url: format!("http://{address}/v1"),
            model: "mock".into(),
            timeout_secs: 5,
            max_retries: 1,
            max_concurrency: 4,
            ..Default::default()
        }
    }

    /// Serves one canned `content` per connection, in order, and reports the
    /// request bodies it saw.
    fn serve(contents: Vec<String>) -> (std::net::SocketAddr, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for content in contents {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let mut request = Vec::new();
                let mut buffer = [0_u8; 8192];
                loop {
                    let read = match stream.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => read,
                    };
                    request.extend_from_slice(&buffer[..read]);
                    let text = String::from_utf8_lossy(&request);
                    if let Some(index) = text.find("\r\n\r\n") {
                        let length = text
                            .to_ascii_lowercase()
                            .split("content-length:")
                            .nth(1)
                            .and_then(|rest| rest.split("\r\n").next())
                            .and_then(|value| value.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        if request.len() >= index + 4 + length {
                            break;
                        }
                    }
                }
                let text = String::from_utf8_lossy(&request).to_string();
                let body = text
                    .split_once("\r\n\r\n")
                    .map(|(_, body)| body.to_owned())
                    .unwrap_or_default();
                tx.send(body).ok();
                let payload = serde_json::json!({
                    "choices": [{"message": {"content": content}}]
                })
                .to_string();
                write!(
                    stream,
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    payload.len(),
                    payload
                )
                .ok();
            }
        });
        (address, rx)
    }

    #[tokio::test]
    async fn malformed_response_is_retried_then_accepted() {
        let (address, _requests) = serve(vec![
            "not JSON".to_owned(),
            scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)]),
        ]);
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                strategy: JudgeStrategy::Listwise,
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let scores = judge.score_group(&group(&["a", "b"])).await.unwrap();
        assert_eq!(scores[0].value, 0.75);
        assert_eq!(judge.stats().retries, 1);
    }

    #[tokio::test]
    async fn an_unresponsive_endpoint_times_out_instead_of_hanging() {
        // Accept the connection and never answer: without a timeout the whole
        // update would block on one group forever.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let server = std::thread::spawn(move || {
            let mut held = Vec::new();
            listener.set_nonblocking(true).expect("nonblocking");
            while stop_rx.try_recv().is_err() {
                if let Ok((stream, _)) = listener.accept() {
                    held.push(stream);
                }
                std::thread::sleep(Duration::from_millis(10));
            }
        });
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                timeout_secs: 1,
                max_retries: 0,
                strategy: JudgeStrategy::Listwise,
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let error = judge.score_group(&group(&["a", "b"])).await.unwrap_err();
        stop_tx.send(()).ok();
        server.join().unwrap();
        assert!(
            error.to_string().to_lowercase().contains("time"),
            "expected a timeout error, got: {error}"
        );
    }

    #[tokio::test]
    async fn auto_falls_back_to_chunking_when_the_group_overruns_the_budget() {
        // Two chunks of two, the second also carrying the anchor. Prompt ids are
        // *presentation* positions, not trajectory indices: for group 1 the
        // permutation leaves chunk 0 as `[0, 1]` and presents chunk 0 + anchor
        // as `[2, 0, 3]`. So the second response scores trajectory 2 at 0.1, the
        // anchor at 0.2 and trajectory 3 at 0.3 - and the anchor's own chunk
        // scored it 0.5, shifting the second chunk by +0.3.
        let (address, requests) = serve(vec![
            scores_json(&[(0, "anchor", 0.5), (1, "second", 0.6)]),
            scores_json(&[(0, "third", 0.1), (1, "anchor", 0.2), (2, "fourth", 0.3)]),
        ]);
        // Distinct answers: identical ones would be absorbed by the shared
        // prefix and cost nothing, so nothing would need chunking.
        let answers = (0..4)
            .map(|index| format!("{}{index}", "x".repeat(400)))
            .collect::<Vec<_>>();
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                context: JudgeContext {
                    max_request_chars: 1_200,
                    max_trajectory_chars: 1_000,
                    max_message_chars: 1_000,
                    head_ratio: 0.5,
                    include_env_state: true,
                },
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let answers = answers.iter().map(String::as_str).collect::<Vec<_>>();
        let scores = judge.score_group(&group(&answers)).await.unwrap();
        assert_eq!(scores.len(), 4);
        assert!(scores.iter().all(|score| score.valid), "{scores:?}");
        assert_eq!(scores[0].value, 0.5);
        assert_eq!(scores[1].value, 0.6);
        // Second chunk shifted by 0.5 - 0.2, and each score landed back on the
        // trajectory it was about rather than on the position it was shown in.
        assert!((scores[2].value - 0.4).abs() < 1e-6, "{:?}", scores[2]);
        assert_eq!(scores[2].explanation.as_deref(), Some("third"));
        assert!((scores[3].value - 0.6).abs() < 1e-6, "{:?}", scores[3]);
        assert_eq!(scores[3].explanation.as_deref(), Some("fourth"));
        let bodies: Vec<String> = requests.try_iter().collect();
        assert_eq!(bodies.len(), 2, "expected exactly two judge requests");
        assert_eq!(judge.stats().requests_per_group, 2.0);
    }

    #[tokio::test]
    async fn pairwise_aggregates_win_rates_and_survives_one_failed_comparison() {
        // Ring schedule over three trajectories: (0,1), (1,2), (2,0).
        let (address, _requests) = serve(vec![
            r#"{"winner":"a","explanation":"0>1"}"#.to_owned(),
            r#"{"winner":"a","explanation":"1>2"}"#.to_owned(),
            "unparseable".to_owned(),
        ]);
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                strategy: JudgeStrategy::Pairwise {
                    max_pairs: Some(3),
                    // One order only: this test is about aggregation surviving a
                    // failed comparison, not about position bias.
                    both_orders: false,
                    aggregation: Aggregation::WinRate,
                },
                max_retries: 0,
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let scores = judge.score_group(&group(&["a", "b", "c"])).await.unwrap();
        // 0 won its only completed match, 1 won one of two, 2 lost its only one.
        assert_eq!(scores[0].value, 1.0);
        assert_eq!(scores[1].value, 0.5);
        assert_eq!(scores[2].value, 0.0);
        assert!(scores.iter().all(|score| score.valid));
    }

    #[tokio::test]
    async fn a_pair_the_judge_answers_by_position_becomes_a_tie() {
        // Two trajectories, one pair, judged in both orders. The judge answers
        // "a" whichever way round it is asked: it is reading the position.
        let (address, requests) = serve(vec![
            r#"{"winner":"a","explanation":"first looks better"}"#.to_owned(),
            r#"{"winner":"a","explanation":"first looks better"}"#.to_owned(),
        ]);
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                strategy: JudgeStrategy::Pairwise {
                    max_pairs: None,
                    both_orders: true,
                    aggregation: Aggregation::WinRate,
                },
                max_retries: 0,
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let scores = judge.score_group(&group(&["a", "b"])).await.unwrap();
        assert_eq!(scores[0].value, 0.5);
        assert_eq!(scores[1].value, 0.5);
        assert!(
            scores[0]
                .explanation
                .as_deref()
                .unwrap()
                .contains("order-dependent")
        );
        let stats = judge.stats();
        assert_eq!(stats.position_disagreement, 1.0);
        // Two requests for one pair: the cost `both_orders` buys the number with.
        assert_eq!(requests.try_iter().count(), 2);
        assert_eq!(stats.requests, 2);
    }

    #[tokio::test]
    async fn compaction_applies_to_every_member_of_the_group_or_to_none() {
        // One long member and one short one. Both get compacted - a summary
        // judged against a full transcript would hand the difference to GRPO as
        // if it were performance.
        let (address, requests) = serve(vec![
            r#"{"summary":"they poked at it"}"#.to_owned(),
            r#"{"summary":"they poked at it"}"#.to_owned(),
            scores_json(&[(0, "ok", 0.4), (1, "ok", 0.6)]),
        ]);
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
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                strategy: JudgeStrategy::Listwise,
                max_retries: 0,
                compaction: Some(crate::render::compact::CompactionConfig {
                    trigger_chars: 500,
                    target_chars: 200,
                    keep_last: 2,
                }),
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let scores = judge.score_group(&group).await.unwrap();
        assert_eq!(scores.len(), 2);
        let bodies: Vec<String> = requests.try_iter().collect();
        // Two summary requests - one per member - then one scoring request.
        assert_eq!(bodies.len(), 3, "{bodies:?}");
        let scoring = bodies.last().unwrap();
        assert_eq!(scoring.matches("messages summarised").count(), 2);
        assert!(!scoring.contains("middle-3"), "middle turns survived");
        // The opening and the closing turns are never summarised away.
        assert!(scoring.contains("done"));
        assert!(judge.stats().compacted_fraction > 0.0);
    }

    /// Compaction is part of the prompt every member is judged from, so it is
    /// held to a stricter rule than the verdict: temperature 0, whatever the
    /// judge is configured to sample its verdicts at, and whether or not the
    /// configuration states a temperature at all. Leaving the field out is not
    /// equivalent - the provider then picks its own default, which is not zero.
    #[tokio::test]
    async fn a_summary_is_requested_at_temperature_zero_whatever_the_judge_samples_at() {
        for temperature in [Some(0.7), None] {
            let (address, requests) = serve(vec![
                r#"{"summary":"they poked at it"}"#.to_owned(),
                r#"{"summary":"they poked at it"}"#.to_owned(),
                scores_json(&[(0, "ok", 0.4), (1, "ok", 0.6)]),
            ]);
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
            let judge = RulerJudge::with_api_key(
                RulerConfig {
                    strategy: JudgeStrategy::Listwise,
                    max_retries: 0,
                    temperature,
                    compaction: Some(crate::render::compact::CompactionConfig {
                        trigger_chars: 500,
                        target_chars: 200,
                        keep_last: 2,
                    }),
                    ..config(address)
                },
                "test-key".into(),
            )
            .unwrap();
            judge
                .score_group(&TrajectoryGroup {
                    group_id: 1,
                    scenario_id: "s".into(),
                    trajectories: vec![long("a"), long("b")],
                })
                .await
                .unwrap();

            let bodies: Vec<serde_json::Value> = requests
                .try_iter()
                .map(|body| serde_json::from_str(&body).expect("a JSON payload"))
                .collect();
            let (summaries, verdict) = bodies.split_at(2);
            for summary in summaries {
                assert_eq!(
                    summary["temperature"],
                    serde_json::json!(0.0),
                    "{temperature:?}: a summary must be requested at temperature 0"
                );
            }
            // The verdict keeps what the operator asked for, including nothing.
            assert_eq!(
                verdict[0].get("temperature").cloned(),
                temperature.map(|value| serde_json::json!(value))
            );
        }
    }

    #[tokio::test]
    async fn a_group_under_the_trigger_is_never_summarised() {
        let (address, requests) = serve(vec![scores_json(&[(0, "ok", 0.4), (1, "ok", 0.6)])]);
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                strategy: JudgeStrategy::Listwise,
                compaction: Some(crate::render::compact::CompactionConfig::default()),
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        judge.score_group(&group(&["a", "b"])).await.unwrap();
        assert_eq!(
            requests.try_iter().count(),
            1,
            "a short group paid for a summary"
        );
        assert_eq!(judge.stats().compacted_fraction, 0.0);
    }

    #[tokio::test]
    async fn a_cached_verdict_is_reused_without_a_request() {
        let (address, _requests) =
            serve(vec![r#"{"winner":"b","explanation":"b wins"}"#.to_owned()]);
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                strategy: JudgeStrategy::Pairwise {
                    max_pairs: None,
                    both_orders: false,
                    aggregation: Aggregation::WinRate,
                },
                ..config(address)
            },
            "test-key".into(),
        )
        .unwrap();
        let group = group(&["a", "b"]);
        let first = judge.score_group(&group).await.unwrap();
        // The single server response is consumed; a second call can only pass
        // from the cache.
        let second = judge.score_group(&group).await.unwrap();
        assert_eq!(first[1].value, 1.0);
        assert_eq!(first[1].value, second[1].value);
        assert_eq!(judge.stats().requests, 1);
        assert_eq!(judge.stats().cache_hits, 1);
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
        let path = std::env::temp_dir().join(format!(
            "retrograd-judge-cache-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_file(&path).ok();
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
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn concurrent_cache_stores_leave_complete_jsonl_rows() {
        let path = std::env::temp_dir().join(format!(
            "retrograd-judge-concurrent-cache-{}-{:?}.jsonl",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_file(&path).ok();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut judge_config = config(listener.local_addr().unwrap());
        judge_config.cache_path = Some(path.clone());
        let judge = Arc::new(RulerJudge::with_api_key(judge_config, "test".into()).unwrap());
        let threads = (0..16)
            .map(|index| {
                let judge = judge.clone();
                std::thread::spawn(move || {
                    judge
                        .store(
                            format!("key-{index}"),
                            &[Score {
                                value: index as f32,
                                valid: true,
                                explanation: Some("x".repeat(1024)),
                                error: None,
                            }],
                        )
                        .unwrap();
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            thread.join().unwrap();
        }
        let (scores, _) = load_cache(Some(&path)).unwrap();
        assert_eq!(scores.len(), 16);
        std::fs::remove_file(&path).ok();
    }
}
