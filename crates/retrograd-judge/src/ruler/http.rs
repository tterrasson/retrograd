//! The OpenAI-compatible HTTP transport, behind the `http` feature - the only
//! place in this crate that links `reqwest` and `rustls`.
//!
//! Two shapes are provided, for two kinds of caller:
//!
//! - [`HttpJudgeTransport`] implements the synchronous [`JudgeTransport`] and
//!   *owns* a tokio runtime, created on the thread that built it. Its `complete`
//!   drives the runtime with `block_on`, which is only sound off an executor:
//!   this adapter is for plain threads - the training thread - and must never be
//!   called from inside an async context.
//! - The async API (`RewardBackend for RulerJudge`) runs each group on
//!   Tokio's blocking pool. Each job owns its runtime and shares the HTTP
//!   client and semaphore with the other groups, so groups overlap while the
//!   configured request concurrency remains global.

use std::sync::Arc;
use std::time::Duration;

use futures_util::future::join_all;
use retrograd_agent_core::trajectory::TrajectoryGroup;
use retrograd_agent_core::{Error, Result};
use retrograd_llm_client::Purpose;

use crate::Score;
use crate::client::OpenAiClient;
use crate::transport::{JudgeReply, JudgeRequest, JudgeTransport, JudgeTransportError};

use super::{RulerConfig, RulerCore};

/// The synchronous HTTP transport: an [`OpenAiClient`], a concurrency
/// semaphore, and the runtime that drives both.
pub struct HttpJudgeTransport {
    state: Arc<HttpState>,
    runtime: tokio::runtime::Runtime,
}

struct HttpState {
    model: String,
    client: OpenAiClient,
    semaphore: Arc<tokio::sync::Semaphore>,
}

impl HttpState {
    fn new(config: &RulerConfig, api_key: String) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            model: config.model.clone(),
            client: OpenAiClient::new(
                &config.base_url,
                api_key,
                Duration::from_secs(config.timeout_secs),
                8 * 1024 * 1024,
                Purpose::Judge,
            )?,
            semaphore: Arc::new(tokio::sync::Semaphore::new(config.max_concurrency)),
        })
    }
}

impl HttpJudgeTransport {
    pub fn new(config: &RulerConfig, api_key: String) -> Result<Self> {
        Self::from_state(Arc::new(HttpState::new(config, api_key)?))
    }

    fn from_state(state: Arc<HttpState>) -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| Error::Reward(format!("start the judge HTTP runtime: {error}")))?;
        Ok(Self { state, runtime })
    }
}

impl JudgeTransport for HttpJudgeTransport {
    fn complete(
        &mut self,
        requests: &[JudgeRequest],
    ) -> Vec<std::result::Result<JudgeReply, JudgeTransportError>> {
        let model = self.state.model.clone();
        let client = &self.state.client;
        let semaphore = Arc::clone(&self.state.semaphore);
        self.runtime.block_on(async move {
            join_all(requests.iter().map(|request| {
                let semaphore = Arc::clone(&semaphore);
                let model = model.clone();
                async move {
                    let _permit = semaphore.acquire().await.map_err(|_| {
                        JudgeTransportError::Failed("judge semaphore closed".into())
                    })?;
                    let mut payload = serde_json::json!({
                        "model": model,
                        "messages": [{"role": "user", "content": request.prompt}],
                    });
                    if let Some(format) = &request.response_format {
                        payload["response_format"] = format.clone();
                    }
                    if let Some(temperature) = request.temperature {
                        payload["temperature"] = serde_json::json!(temperature);
                    }
                    let completion = client
                        .chat_completion(&payload)
                        .await
                        .map_err(|error| JudgeTransportError::Failed(error.to_string()))?;
                    Ok(JudgeReply {
                        content: completion.content,
                    })
                }
            }))
            .await
        })
    }
}

/// The shared HTTP state of the async judge. No runtime is created or dropped
/// on the caller's executor thread.
pub(super) struct HttpHandle {
    core: Arc<RulerCore>,
    state: Arc<HttpState>,
}

impl HttpHandle {
    pub(super) fn new(core: Arc<RulerCore>, config: &RulerConfig, api_key: String) -> Result<Self> {
        Ok(Self {
            core,
            state: Arc::new(HttpState::new(config, api_key)?),
        })
    }

    pub(super) async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
        let core = Arc::clone(&self.core);
        let state = Arc::clone(&self.state);
        let group = group.clone();
        tokio::task::spawn_blocking(move || {
            let mut transport = HttpJudgeTransport::from_state(state)?;
            core.score_group_via(&mut transport, &group)
        })
        .await
        .map_err(|error| Error::Reward(format!("the judge HTTP job failed: {error}")))?
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    use super::super::{JudgeStrategy, RulerConfig, RulerJudge};
    use crate::render::compact::CompactionConfig;
    use crate::ruler::RulerStats;
    use retrograd_agent_core::trajectory::{Message, Role, Trajectory, TrajectoryGroup};
    use retrograd_spec::judge::{Aggregation, JudgeContext};

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

    fn read_request(stream: &mut std::net::TcpStream) -> String {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 8192];
        loop {
            let read = stream.read(&mut buffer).expect("read judge request");
            assert_ne!(read, 0, "incomplete judge request");
            request.extend_from_slice(&buffer[..read]);
            let text = String::from_utf8_lossy(&request);
            if let Some(index) = text.find("\r\n\r\n") {
                let length = text[..index]
                    .to_ascii_lowercase()
                    .split("content-length:")
                    .nth(1)
                    .and_then(|rest| rest.split("\r\n").next())
                    .and_then(|value| value.trim().parse::<usize>().ok())
                    .unwrap_or(0);
                if request.len() >= index + 4 + length {
                    return String::from_utf8(request).unwrap();
                }
            }
        }
    }

    /// Serves one canned `content` per connection, in order, and reports the
    /// request bodies it saw.
    ///
    /// Order is only meaningful for requests the judge sends one after the
    /// other: concurrent ones reach `accept` in whatever order their
    /// connections land, and are answered through [`serve_by`] instead.
    fn serve(contents: Vec<String>) -> (std::net::SocketAddr, mpsc::Receiver<String>) {
        let count = contents.len();
        let contents = std::sync::Mutex::new(contents.into_iter());
        serve_by(count, move |_| contents.lock().unwrap().next().unwrap())
    }

    /// Serves `count` connections, each answered with the `content` that
    /// `respond` picks from the request body.
    fn serve_by(
        count: usize,
        respond: impl Fn(&str) -> String + Send + 'static,
    ) -> (std::net::SocketAddr, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for _ in 0..count {
                let Ok((mut stream, _)) = listener.accept() else {
                    return;
                };
                let text = read_request(&mut stream);
                let body = text
                    .split_once("\r\n\r\n")
                    .map(|(_, body)| body.to_owned())
                    .unwrap_or_default();
                let content = respond(&body);
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
    async fn separate_groups_can_have_requests_in_flight_together() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        // The endpoint answers neither connection until both groups arrive.
        // A worker that serializes whole groups times out on the first one.
        let server = std::thread::spawn(move || {
            listener.set_nonblocking(true).unwrap();
            let deadline = std::time::Instant::now() + Duration::from_secs(3);
            let mut streams = Vec::new();
            while streams.len() < 2 && std::time::Instant::now() < deadline {
                match listener.accept() {
                    Ok((stream, _)) => streams.push(stream),
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("accept: {error}"),
                }
            }
            assert_eq!(streams.len(), 2, "the groups did not overlap");
            let content = scores_json(&[(0, "ok", 0.75), (1, "ok", 0.25)]);
            let payload =
                serde_json::json!({"choices": [{"message": {"content": content}}]}).to_string();
            for mut stream in streams {
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                read_request(&mut stream);
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", payload.len(), payload).unwrap();
            }
        });
        let judge = RulerJudge::with_api_key(
            RulerConfig {
                timeout_secs: 2,
                max_retries: 0,
                max_concurrency: 2,
                strategy: JudgeStrategy::Listwise,
                ..config(address)
            },
            "test".into(),
        )
        .unwrap();
        let first = group(&["a", "b"]);
        let second = group(&["c", "d"]);
        let (first, second) = tokio::join!(judge.score_group(&first), judge.score_group(&second));
        server.join().unwrap();
        assert!(first.is_ok(), "{first:?}");
        assert!(second.is_ok(), "{second:?}");
    }

    #[tokio::test]
    async fn an_invalid_cache_can_fail_construction_inside_an_executor() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache.jsonl");
        std::fs::write(&path, "invalid JSON").unwrap();
        let result = RulerJudge::with_api_key(
            RulerConfig {
                cache_path: Some(path),
                ..config("127.0.0.1:1".parse().unwrap())
            },
            "test".into(),
        );
        assert!(result.is_err());
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
        // scored it 0.5, shifting the second chunk by +0.3. The two chunks are
        // requested concurrently, so each answer is picked by the chunk it is
        // about: only the second one shows trajectory 2.
        let (address, requests) = serve_by(2, |body| {
            if body.contains("x2") {
                scores_json(&[(0, "third", 0.1), (1, "anchor", 0.2), (2, "fourth", 0.3)])
            } else {
                scores_json(&[(0, "anchor", 0.5), (1, "second", 0.6)])
            }
        });
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
        // Ring schedule over three trajectories: (0,1), (1,2), (2,0), compared
        // concurrently - so the verdict is picked by the pair a request shows,
        // and it is (2,0) that fails.
        let (address, _requests) = serve_by(3, |body| {
            if body.contains("alpha") && body.contains("beta") {
                r#"{"winner":"a","explanation":"0>1"}"#.to_owned()
            } else if body.contains("beta") && body.contains("gamma") {
                r#"{"winner":"a","explanation":"1>2"}"#.to_owned()
            } else {
                "unparseable".to_owned()
            }
        });
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
        let scores = judge
            .score_group(&group(&["alpha", "beta", "gamma"]))
            .await
            .unwrap();
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
        let stats: RulerStats = judge.stats();
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
                compaction: Some(CompactionConfig {
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
                    compaction: Some(CompactionConfig {
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
                compaction: Some(CompactionConfig::default()),
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

    #[tokio::test]
    async fn concurrent_cache_stores_leave_complete_jsonl_rows() {
        let tmp = tempfile::tempdir().expect("create temp dir");
        let path = tmp.path().join("cache.jsonl");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut judge_config = config(listener.local_addr().unwrap());
        judge_config.cache_path = Some(path.clone());
        let judge = Arc::new(RulerJudge::with_api_key(judge_config, "test".into()).unwrap());
        let threads = (0..16)
            .map(|index| {
                let judge = judge.clone();
                std::thread::spawn(move || {
                    judge
                        .core
                        .store(
                            format!("key-{index}"),
                            &[crate::Score {
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
        let (scores, _) = super::super::load_cache(Some(&path)).unwrap();
        assert_eq!(scores.len(), 16);
    }
}
