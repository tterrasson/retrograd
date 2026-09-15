//! HTTP environment client.
//!
//! The agent infrastructure worth reusing - containers, skills, sandboxes - is
//! rarely written in Rust. This is where it plugs in: as an *environment*, never
//! as an inference provider. Tokenization, batching and the prefix invariant all
//! stay on this side of the wire; what crosses it is `reset`/`step`/`tools`.
//!
//! Wire contract, JSON bodies throughout:
//!
//! | Route         | Request                     | Response                                      |
//! | ------------- | --------------------------- | --------------------------------------------- |
//! | `GET /tools`  | - | `{"tools": [ToolSpec]}`                        |
//! | `POST /reset` | `{"scenario": …, "seed": n}`| `{"env_id": "…", "observation": null \| "…"}`  |
//! | `POST /step`  | `{"env_id": …, "call": …}`  | `{"result": ToolResult, "reward": null, "done": false}` |
//! | `POST /close` | `{"env_id": …}`             | any 2xx, body never read - `204` is fine       |
//! | `GET /state/{env_id}` | - | `EnvState`, or `404` for a server with none    |
//!
//! `/tools` is fetched once per run by the engine, so it must describe the
//! environment kind rather than a live instance.
//!
//! This is also the bridge to an OpenEnv-style server: `reset`/`step`/`state`/
//! `close` is the same shape under another name, and everything that must stay
//! in Rust - tokenization, batching, the prefix invariant - stays on this side.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use retrograd_agent_core::env::{EnvState, Environment, EnvironmentFactory, StepOutcome};
use retrograd_agent_core::scenario::Scenario;
use retrograd_agent_core::tools::{ToolCall, ToolResult, ToolSpec};
use retrograd_agent_core::{Error, Result};

pub use retrograd_spec::env::HttpEnvironmentConfig;

/// Shares one connection pool across every instance it hands out.
pub struct HttpEnvironmentFactory {
    client: reqwest::Client,
    config: Arc<HttpEnvironmentConfig>,
}

impl HttpEnvironmentFactory {
    pub fn new(config: HttpEnvironmentConfig) -> Result<Self> {
        config.validate()?;
        let mut headers = reqwest::header::HeaderMap::new();
        for (name, value) in &config.headers {
            let name = http::HeaderName::from_bytes(name.as_bytes())
                .map_err(|error| Error::invalid(format!("invalid environment header: {error}")))?;
            let value = http::HeaderValue::from_str(value).map_err(|error| {
                Error::invalid(format!("invalid environment header value: {error}"))
            })?;
            headers.insert(name, value);
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .pool_max_idle_per_host(config.pool_size)
            .connect_timeout(Duration::from_secs(config.connect_timeout_secs))
            .timeout(Duration::from_secs(config.request_timeout_secs))
            .build()
            .map_err(|error| Error::Tool(format!("build environment HTTP client: {error}")))?;
        Ok(Self {
            client,
            config: Arc::new(config),
        })
    }
}

#[async_trait]
impl EnvironmentFactory for HttpEnvironmentFactory {
    async fn create(&self) -> Result<Box<dyn Environment>> {
        Ok(Box::new(HttpEnvironment {
            client: self.client.clone(),
            config: self.config.clone(),
            env_id: None,
        }))
    }
}

pub struct HttpEnvironment {
    client: reqwest::Client,
    config: Arc<HttpEnvironmentConfig>,
    /// Allocated by `reset`, required by every later call. `None` before the
    /// first reset and after `close`.
    env_id: Option<String>,
}

#[derive(Serialize)]
struct ResetRequest<'a> {
    scenario: &'a Scenario,
    seed: u64,
}

#[derive(Deserialize)]
struct ResetResponse {
    env_id: String,
    #[serde(default)]
    observation: Option<String>,
}

#[derive(Serialize)]
struct StepRequest<'a> {
    env_id: &'a str,
    call: &'a ToolCall,
}

#[derive(Deserialize)]
struct StepResponse {
    result: ToolResult,
    #[serde(default)]
    reward: Option<f32>,
    #[serde(default)]
    done: bool,
}

#[derive(Serialize)]
struct CloseRequest<'a> {
    env_id: &'a str,
}

#[derive(Deserialize)]
struct ToolsResponse {
    tools: Vec<ToolSpec>,
}

impl HttpEnvironment {
    fn url(&self, route: &str) -> String {
        format!("{}/{route}", self.config.base_url.trim_end_matches('/'))
    }

    fn env_id(&self) -> Result<&str> {
        self.env_id
            .as_deref()
            .ok_or_else(|| Error::Tool("environment was stepped before reset".into()))
    }

    fn max_response_bytes(&self) -> usize {
        // JSON escaping and envelope fields need some headroom beyond the text
        // that ultimately enters the prompt, but the wire body still needs a
        // hard allocation bound.
        self.config
            .max_result_bytes
            .saturating_mul(4)
            .max(64 * 1024)
    }

    async fn read_body_limited(
        mut response: reqwest::Response,
        max_bytes: usize,
        route: &str,
    ) -> Result<Vec<u8>> {
        let mut body = Vec::with_capacity(max_bytes.min(64 * 1024));
        while let Some(chunk) = response.chunk().await.map_err(|error| {
            Error::Tool(format!("environment {route} response read failed: {error}"))
        })? {
            if body.len().saturating_add(chunk.len()) > max_bytes {
                return Err(Error::Tool(format!(
                    "environment {route} response exceeded {max_bytes} bytes"
                )));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(body)
    }

    /// Sends one request and checks its status, without touching the body.
    ///
    /// A 404 or 410 on a route carrying an `env_id` means the server no longer
    /// knows this instance - it restarted, or reaped the session. That fails the
    /// trajectory rather than silently starting a fresh episode mid-rollout,
    /// which would leave the collected prefix conditioned on a world that no
    /// longer exists. The two callers that may *absorb* that case get it as a
    /// variant, never as a substring of a rendered message.
    async fn dispatch(
        &self,
        request: reqwest::RequestBuilder,
        route: &str,
    ) -> std::result::Result<reqwest::Response, CallError> {
        let response = request.send().await.map_err(|error| {
            let reason = if error.is_timeout() {
                format!("timed out after {}s", self.config.request_timeout_secs)
            } else {
                error.to_string()
            };
            CallError::Failed(Error::Tool(format!("environment {route} failed: {reason}")))
        })?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND || status == reqwest::StatusCode::GONE {
            return Err(CallError::SessionLost {
                route: route.to_string(),
                status,
            });
        }
        if !status.is_success() {
            let body = Self::read_body_limited(response, 512, route)
                .await
                .unwrap_or_default();
            return Err(CallError::Failed(Error::Tool(format!(
                "environment {route} returned {status}: {}",
                String::from_utf8_lossy(&body)
            ))));
        }
        Ok(response)
    }

    /// The routes whose answer the client actually reads.
    async fn send<T: for<'de> Deserialize<'de>>(
        &self,
        request: reqwest::RequestBuilder,
        route: &str,
    ) -> std::result::Result<T, CallError> {
        let response = self.dispatch(request, route).await?;
        let body = Self::read_body_limited(response, self.max_response_bytes(), route)
            .await
            .map_err(CallError::Failed)?;
        serde_json::from_slice(&body).map_err(|error| {
            CallError::Failed(Error::Tool(format!(
                "environment {route} sent invalid JSON: {error}"
            )))
        })
    }
}

/// One call's outcome, with the case the caller may decide to absorb kept as a
/// variant. `state` and `close` both treat a reaped session as success, and
/// reading that from `to_string()` would make a message part of the contract.
#[derive(Debug, thiserror::Error)]
enum CallError {
    #[error("environment {route} lost its session ({status}); the server restarted or reaped it")]
    SessionLost {
        route: String,
        status: reqwest::StatusCode,
    },
    #[error("{0}")]
    Failed(#[from] Error),
}

impl From<CallError> for Error {
    fn from(error: CallError) -> Self {
        match error {
            CallError::SessionLost { .. } => Error::Tool(error.to_string()),
            CallError::Failed(inner) => inner,
        }
    }
}

#[async_trait]
impl Environment for HttpEnvironment {
    async fn reset(&mut self, scenario: &Scenario, seed: u64) -> Result<Option<String>> {
        if self.env_id.is_some() {
            self.close().await?;
        }
        let request = self
            .client
            .post(self.url("reset"))
            .json(&ResetRequest { scenario, seed });
        let mut response: ResetResponse = self.send(request, "reset").await?;
        if response.env_id.is_empty() {
            return Err(Error::Tool("environment reset returned no env_id".into()));
        }
        self.env_id = Some(response.env_id);
        // The opening observation enters the prompt like any step result does,
        // and it is the one an environment is most likely to make huge - a full
        // repository listing, a page dump. Capped on the same budget.
        if let Some(observation) = &mut response.observation {
            truncate_utf8(observation, self.config.max_result_bytes);
        }
        Ok(response.observation)
    }

    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
        let env_id = self.env_id()?;
        let request = self
            .client
            .post(self.url("step"))
            .json(&StepRequest { env_id, call });
        let mut response: StepResponse = self.send(request, "step").await?;
        if response.reward.is_some_and(|reward| !reward.is_finite()) {
            return Err(Error::Tool(
                "environment step returned a non-finite reward".into(),
            ));
        }
        truncate_utf8(&mut response.result.content, self.config.max_result_bytes);
        Ok(StepOutcome {
            result: response.result,
            reward: response.reward,
            done: response.done,
        })
    }

    async fn tools(&self) -> Result<Vec<ToolSpec>> {
        let request = self.client.get(self.url("tools"));
        let response: ToolsResponse = self.send(request, "tools").await?;
        Ok(response.tools)
    }

    /// `GET /state/{env_id}`, and the one route a server may simply not have:
    /// a `404` here means "this environment reports no state", not "the session
    /// is gone", so it degrades to the empty state instead of costing the
    /// trajectory. The distinction is safe because `state` is only ever read
    /// after the rollout, by the judge - nothing downstream is conditioned on it.
    async fn state(&mut self) -> Result<EnvState> {
        let Some(env_id) = &self.env_id else {
            return Ok(EnvState::default());
        };
        let request = self.client.get(self.url(&format!("state/{env_id}")));
        match self.send::<EnvState>(request, "state").await {
            Ok(state) => Ok(state),
            Err(CallError::SessionLost { .. }) => Ok(EnvState::default()),
            Err(error) => Err(error.into()),
        }
    }

    async fn close(&mut self) -> Result<()> {
        // Idempotent: a rollout that already closed, or never reset, has
        // nothing to release.
        let Some(env_id) = self.env_id.take() else {
            return Ok(());
        };
        let request = self
            .client
            .post(self.url("close"))
            .json(&CloseRequest { env_id: &env_id });
        // The contract says the answer to `/close` is ignored, so only its status
        // is read: a bare `204 No Content` - the obvious thing for a server to
        // reply here - must not come back as a decoding error and a teardown
        // warning. A session the server already dropped is closed by definition.
        match self.dispatch(request, "close").await {
            Ok(_) => Ok(()),
            Err(CallError::SessionLost { .. }) => Ok(()),
            Err(error) => {
                self.env_id = Some(env_id);
                Err(error.into())
            }
        }
    }
}

impl Drop for HttpEnvironment {
    fn drop(&mut self) {
        // The engine closes every environment it creates, including on the
        // failure and deadline paths, so reaching here with a live `env_id`
        // means the run was interrupted outside that loop. Best effort: fire a
        // detached close if a runtime is still turning, and say so if not,
        // a leaked container is worth a line in the log.
        let Some(env_id) = self.env_id.take() else {
            return;
        };
        let request = self
            .client
            .post(self.url("close"))
            .json(&CloseRequest { env_id: &env_id });
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                handle.spawn(async move {
                    let _ = request.send().await;
                });
            }
            Err(_) => tracing::warn!(
                "environment '{env_id}' was dropped without being closed and no runtime is \
                 available to release it"
            ),
        }
    }
}

/// Byte cap that never cuts a character in half, and marks the cut so the
/// policy can tell an elided observation from a short one.
fn truncate_utf8(content: &mut String, max_bytes: usize) {
    retrograd_agent_core::text::truncate_utf8(content, max_bytes, "\n[observation truncated]");
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use axum::extract::State;
    use axum::routing::{get, post};
    use axum::{Json, Router};

    use super::*;

    /// Server-side state of the fake environment: one counter per session, so
    /// the test can tell whether two rollouts really got separate instances.
    #[derive(Default)]
    struct FakeState {
        next_id: AtomicUsize,
        sessions: Mutex<std::collections::HashMap<String, usize>>,
        /// `reset` refuses these many calls before working.
        failing_resets: AtomicUsize,
        /// `close` refuses these many calls before releasing the session.
        failing_closes: AtomicUsize,
        /// `step` hangs on a call naming this tool.
        hanging_tool: Mutex<Option<String>>,
        /// What `reset` answers with, `None` for no opening observation.
        reset_observation: Mutex<Option<String>>,
        /// Whether `GET /state` is implemented at all.
        reports_state: AtomicBool,
    }

    async fn serve(state: Arc<FakeState>) -> String {
        let app =
            Router::new()
                .route(
                    "/tools",
                    get(|| async {
                        Json(serde_json::json!({"tools": [{
                            "name": "count",
                            "description": "increment and read the session counter",
                            "input_schema": {"type": "object"}
                        }]}))
                    }),
                )
                .route(
                    "/reset",
                    post(
                        |State(state): State<Arc<FakeState>>,
                         Json(_body): Json<serde_json::Value>| async move {
                            if state
                                .failing_resets
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                                    left.checked_sub(1)
                                })
                                .is_ok()
                            {
                                return (
                                    axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                                    Json(serde_json::json!({"error": "environment is warming up"})),
                                );
                            }
                            let id =
                                format!("env-{}", state.next_id.fetch_add(1, Ordering::SeqCst));
                            state.sessions.lock().unwrap().insert(id.clone(), 0);
                            let observation = state.reset_observation.lock().unwrap().clone();
                            (
                                axum::http::StatusCode::OK,
                                Json(serde_json::json!({"env_id": id, "observation": observation})),
                            )
                        },
                    ),
                )
                .route(
                    "/step",
                    post(
                        |State(state): State<Arc<FakeState>>,
                         Json(body): Json<serde_json::Value>| async move {
                            let env_id = body["env_id"].as_str().unwrap_or_default().to_owned();
                            let call = &body["call"];
                            if state.hanging_tool.lock().unwrap().as_deref()
                                == call["name"].as_str()
                            {
                                // Longer than any timeout the tests configure.
                                tokio::time::sleep(Duration::from_secs(3600)).await;
                            }
                            let mut sessions = state.sessions.lock().unwrap();
                            let Some(counter) = sessions.get_mut(&env_id) else {
                                return (
                                    axum::http::StatusCode::NOT_FOUND,
                                    Json(serde_json::json!({"error": "unknown env_id"})),
                                );
                            };
                            *counter += 1;
                            let value = *counter;
                            (
                                axum::http::StatusCode::OK,
                                Json(serde_json::json!({
                                    "result": {
                                        "call_id": call["id"],
                                        "content": value.to_string(),
                                        "is_error": false
                                    },
                                    "reward": 0.25,
                                    "done": value >= 2
                                })),
                            )
                        },
                    ),
                )
                .route(
                    "/state/{env_id}",
                    get(
                        |State(state): State<Arc<FakeState>>,
                         axum::extract::Path(env_id): axum::extract::Path<String>| async move {
                            // A server that has no state to report answers 404,
                            // and the client must read that as "nothing to say"
                            // rather than as a lost session.
                            if !state.reports_state.load(Ordering::SeqCst) {
                                return (
                                    axum::http::StatusCode::NOT_FOUND,
                                    Json(serde_json::json!({"error": "no state"})),
                                );
                            }
                            let sessions = state.sessions.lock().unwrap();
                            let Some(counter) = sessions.get(&env_id) else {
                                return (
                                    axum::http::StatusCode::NOT_FOUND,
                                    Json(serde_json::json!({"error": "unknown env_id"})),
                                );
                            };
                            (
                                axum::http::StatusCode::OK,
                                Json(serde_json::json!({
                                    "done": *counter >= 2,
                                    "summary": format!("counter={counter}")
                                })),
                            )
                        },
                    ),
                )
                .route(
                    "/close",
                    post(
                        |State(state): State<Arc<FakeState>>,
                         Json(body): Json<serde_json::Value>| async move {
                            if state
                                .failing_closes
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |left| {
                                    left.checked_sub(1)
                                })
                                .is_ok()
                            {
                                return axum::http::StatusCode::INTERNAL_SERVER_ERROR;
                            }
                            let env_id = body["env_id"].as_str().unwrap_or_default();
                            state.sessions.lock().unwrap().remove(env_id);
                            // No body at all: the contract ignores the answer to
                            // `/close`, so `204` is the natural reply and the
                            // client must not try to decode JSON out of it.
                            axum::http::StatusCode::NO_CONTENT
                        },
                    ),
                )
                .with_state(state);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let _ = axum::serve(listener, app).await;
        });
        format!("http://{address}")
    }

    fn scenario() -> Scenario {
        Scenario {
            id: "demo".into(),
            system: None,
            user: "hello".into(),
            metadata: Default::default(),
        }
    }

    fn call(id: &str, name: &str) -> ToolCall {
        ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn two_instances_hold_independent_sessions() {
        let state = Arc::new(FakeState::default());
        let base = serve(state.clone()).await;
        let factory = HttpEnvironmentFactory::new(HttpEnvironmentConfig::new(base)).unwrap();

        let tools = factory.create().await.unwrap().tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "count");

        let mut first = factory.create().await.unwrap();
        let mut second = factory.create().await.unwrap();
        assert_eq!(first.reset(&scenario(), 1).await.unwrap(), None);
        second.reset(&scenario(), 2).await.unwrap();

        let outcome = first.step(&call("a", "count")).await.unwrap();
        assert_eq!(outcome.result.content, "1");
        assert_eq!(outcome.reward, Some(0.25));
        assert!(!outcome.done);
        // The second instance must start from its own counter, not the first's.
        assert_eq!(
            second
                .step(&call("b", "count"))
                .await
                .unwrap()
                .result
                .content,
            "1"
        );
        // …and the environment can end the episode itself.
        assert!(first.step(&call("c", "count")).await.unwrap().done);

        // `state` is what the judge reads instead of the transcript. This
        // server implements it, so it reports the counter the steps moved.
        state.reports_state.store(true, Ordering::SeqCst);
        let reported = first.state().await.unwrap();
        assert_eq!(reported.summary.as_deref(), Some("counter=2"));
        assert!(reported.done);

        // The fake answers `/close` with a bodyless 204: a client that insisted
        // on decoding JSON here would warn on every single teardown.
        first.close().await.unwrap();
        second.close().await.unwrap();
        assert!(
            state.sessions.lock().unwrap().is_empty(),
            "close must release the server-side session"
        );
        // Closing twice is not an error: the engine closes on every path.
        first.close().await.unwrap();
    }

    #[tokio::test]
    async fn a_failing_reset_is_an_error_not_a_silent_episode() {
        let state = Arc::new(FakeState::default());
        state.failing_resets.store(1, Ordering::SeqCst);
        let base = serve(state).await;
        let factory = HttpEnvironmentFactory::new(HttpEnvironmentConfig::new(base)).unwrap();
        let mut environment = factory.create().await.unwrap();
        let error = environment.reset(&scenario(), 1).await.unwrap_err();
        assert!(error.to_string().contains("500"), "{error}");
        // Nothing was allocated, so stepping must say so rather than guess.
        let error = environment.step(&call("a", "count")).await.unwrap_err();
        assert!(error.to_string().contains("before reset"), "{error}");
        // The next reset succeeds; the failure was per-attempt, not terminal.
        environment.reset(&scenario(), 1).await.unwrap();
    }

    #[tokio::test]
    async fn reset_closes_the_previous_session_and_close_can_be_retried() {
        let state = Arc::new(FakeState::default());
        let base = serve(state.clone()).await;
        let factory = HttpEnvironmentFactory::new(HttpEnvironmentConfig::new(base)).unwrap();
        let mut environment = factory.create().await.unwrap();
        environment.reset(&scenario(), 1).await.unwrap();
        environment.reset(&scenario(), 2).await.unwrap();
        assert_eq!(state.sessions.lock().unwrap().len(), 1);

        state.failing_closes.store(1, Ordering::SeqCst);
        assert!(environment.close().await.is_err());
        assert_eq!(state.sessions.lock().unwrap().len(), 1);
        environment.close().await.unwrap();
        assert!(state.sessions.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn a_hanging_step_times_out_instead_of_stalling_the_rollout() {
        let state = Arc::new(FakeState::default());
        *state.hanging_tool.lock().unwrap() = Some("count".into());
        let base = serve(state).await;
        let factory =
            HttpEnvironmentFactory::new(HttpEnvironmentConfig::new(base).with_request_timeout(1))
                .unwrap();
        let mut environment = factory.create().await.unwrap();
        environment.reset(&scenario(), 1).await.unwrap();
        let error = environment.step(&call("a", "count")).await.unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
    }

    #[tokio::test]
    async fn a_lost_session_fails_the_trajectory_rather_than_restarting_it() {
        let state = Arc::new(FakeState::default());
        let base = serve(state.clone()).await;
        let factory = HttpEnvironmentFactory::new(HttpEnvironmentConfig::new(base)).unwrap();
        let mut environment = factory.create().await.unwrap();
        environment.reset(&scenario(), 1).await.unwrap();
        // The server restarted: the id the trajectory is conditioned on is gone.
        state.sessions.lock().unwrap().clear();
        let error = environment.step(&call("a", "count")).await.unwrap_err();
        assert!(error.to_string().contains("lost its session"), "{error}");
        // Closing a session the server already forgot is still a clean close.
        environment.close().await.unwrap();
    }

    #[test]
    fn config_rejects_urls_and_budgets_that_cannot_work() {
        for config in [
            HttpEnvironmentConfig::new(""),
            HttpEnvironmentConfig::new("127.0.0.1:8080"),
            HttpEnvironmentConfig::new("http://x").with_request_timeout(0),
        ] {
            assert!(config.validate().is_err(), "accepted {}", config.base_url);
        }
        HttpEnvironmentConfig::new("http://127.0.0.1:8080")
            .validate()
            .unwrap();
    }

    #[tokio::test]
    async fn a_huge_opening_observation_is_capped_like_a_step_result() {
        let state = Arc::new(FakeState::default());
        *state.reset_observation.lock().unwrap() = Some("é".repeat(4096));
        let base = serve(state).await;
        let mut config = HttpEnvironmentConfig::new(base);
        config.max_result_bytes = 256;
        let factory = HttpEnvironmentFactory::new(config).unwrap();
        let mut environment = factory.create().await.unwrap();
        // Uncapped, the opening observation alone would eat the prompt budget
        // before the policy has sampled a single token.
        let observation = environment.reset(&scenario(), 1).await.unwrap().unwrap();
        assert!(observation.len() <= 256, "{} bytes", observation.len());
        assert!(observation.ends_with("[observation truncated]"));
    }

    #[tokio::test]
    async fn an_http_body_is_rejected_before_it_can_grow_without_bound() {
        let state = Arc::new(FakeState::default());
        *state.reset_observation.lock().unwrap() = Some("x".repeat(128 * 1024));
        let base = serve(state).await;
        let mut config = HttpEnvironmentConfig::new(base);
        config.max_result_bytes = 256;
        let factory = HttpEnvironmentFactory::new(config).unwrap();
        let mut environment = factory.create().await.unwrap();
        let error = environment.reset(&scenario(), 1).await.unwrap_err();
        assert!(error.to_string().contains("exceeded"), "{error}");
    }

    #[test]
    fn observation_truncation_preserves_utf8_and_marks_the_cut() {
        let mut text = "é".repeat(20);
        truncate_utf8(&mut text, 32);
        assert!(text.is_char_boundary(text.len()));
        assert!(text.ends_with("[observation truncated]"));
        let mut short = "ok".to_owned();
        truncate_utf8(&mut short, 32);
        assert_eq!(short, "ok");
    }
}
