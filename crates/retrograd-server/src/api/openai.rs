//! `/v1/models` and `/v1/chat/completions` over the weights of a run.
//!
//! The protocol is `retrograd-openai`'s; what this module adds is what a model
//! id means on a server with runs:
//!
//! ```text
//! <run-id>           the run's current weights: live while it trains, its
//!                    final adapter once it is over
//! <run-id>@final     the adapter (or model) the run wrote when it finished
//! <run-id>@best      a checkpoint, by id
//! <run-id>@step-<N>  a checkpoint, by step
//! <run-id>@latest    the checkpoint `--resume` would pick
//! <run-id>@base      the run's base model, without its adapter
//! ```
//!
//! A live run answers on its own thread, at its next progress callback, like
//! `generate` - its weights are the ones being trained and never leave it. Any
//! other id is served by the serving session, which loads the weights beside
//! nothing: it borrows the device permit the runs queue on, and a run that
//! wants it back gets it.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use retrograd_openai::{
    DeviceLease, Lease, LiveRunner, ModelCard, ModelSource, OpenAiError, Target, TrainerJob,
    WeightsSpec,
};
use tokio::sync::{Semaphore, oneshot};
use uuid::Uuid;

use super::artifacts::scan_checkpoints;
use super::inference::{dispatch, wait};
use crate::dto;
use crate::error::{ApiError, ProblemKind};
use crate::runtime::control::RunCommand;
use crate::runtime::registry::ServingSpec;
use crate::runtime::{RunHandle, RunRegistry};
use crate::state::AppState;

/// Whether a request path belongs to the OpenAI contract, whose errors are
/// OpenAI envelopes rather than problem documents - the SDKs read nothing else.
pub fn is_openai_path(path: &str) -> bool {
    path == "/v1/chat/completions" || path == "/v1/models" || path.starts_with("/v1/models/")
}

/// The runs of this server, as model ids.
pub struct RunModels {
    state: AppState,
}

impl RunModels {
    pub fn new(state: AppState) -> Self {
        Self { state }
    }
}

/// What the suffix of an id asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Weights {
    /// No suffix: live, or else final.
    Current,
    Final,
    Base,
    Latest,
    /// A checkpoint, by its id (`best`, `step-000000000400`) or by its step.
    Checkpoint(String),
}

fn parse_id(model: &str) -> Option<(Uuid, Weights)> {
    let (run, suffix) = match model.split_once('@') {
        Some((run, suffix)) => (run, Some(suffix)),
        None => (model, None),
    };
    let weights = match suffix {
        None => Weights::Current,
        Some("final") => Weights::Final,
        Some("base") => Weights::Base,
        Some("latest") => Weights::Latest,
        Some(checkpoint) if !checkpoint.is_empty() => Weights::Checkpoint(checkpoint.to_owned()),
        Some(_) => return None,
    };
    Some((run.parse().ok()?, weights))
}

fn is_live(handle: &RunHandle) -> bool {
    matches!(
        handle.status(),
        dto::RunStatus::Running | dto::RunStatus::Pausing | dto::RunStatus::Paused
    )
}

/// The run's serving half, from `run.json` - or, for a run older than the
/// field, rebuilt from the configuration it rendered, which says the same.
fn serving(handle: &RunHandle) -> Option<ServingSpec> {
    if let Some(spec) = &handle.record.artifacts.serving {
        return Some(spec.clone());
    }
    let document: retrograd_config::ConfigDocument =
        serde_json::from_str(handle.record.effective_config.get()).ok()?;
    // The rendered paths are absolute, so the root they would be joined to is
    // never read.
    let config = retrograd_config::build(document, Path::new(".")).ok()?;
    Some(super::runs::serving_of(&config))
}

fn weights(serving: &ServingSpec, model: &Path, adapter: Option<PathBuf>) -> WeightsSpec {
    WeightsSpec {
        model: model.to_path_buf(),
        adapter,
        n_ctx: serving.n_ctx,
        // Written by this server from a `Device`; anything else is a hand-edited
        // `run.json`, and letting the runtime pick is the safe reading of it.
        device: serving.device.parse().unwrap_or_default(),
        chat_template_variables: serving.chat_template_variables.clone(),
    }
}

/// The weights the run wrote when it finished, if they are on disk.
fn final_weights(serving: &ServingSpec) -> Option<WeightsSpec> {
    if !serving.output.is_file() {
        return None;
    }
    match serving.output_kind.as_str() {
        "adapter" => Some(weights(
            serving,
            &serving.model,
            Some(serving.output.clone()),
        )),
        "model" => Some(weights(serving, &serving.output, None)),
        // A trainable bundle needs this loader's own reader, which the
        // serving session does not have.
        _ => None,
    }
}

/// A checkpoint that can be served: complete, with its adapter exported
/// beside it.
struct Servable {
    id: String,
    adapter: PathBuf,
    written_at: Option<u64>,
}

fn servable_checkpoints(handle: &RunHandle) -> Vec<Servable> {
    let Some(directory) = &handle.record.artifacts.checkpoint_directory else {
        return Vec::new();
    };
    scan_checkpoints(directory)
        .into_iter()
        .filter(|entry| entry.complete)
        .filter_map(|entry| {
            Some(Servable {
                adapter: entry.adapter.map(PathBuf::from)?,
                id: entry.id,
                written_at: entry.written_at,
            })
        })
        .collect()
}

fn not_found(model: &str, handle: &RunHandle) -> OpenAiError {
    let mut available = Vec::new();
    if serving(handle).as_ref().and_then(final_weights).is_some() {
        available.push("@final".to_owned());
    }
    available.extend(
        servable_checkpoints(handle)
            .into_iter()
            .map(|checkpoint| format!("@{}", checkpoint.id)),
    );
    available.push("@base".to_owned());
    OpenAiError::model_not_found(
        model,
        format!(
            "this run has no such weights on disk; it can serve {}",
            available.join(", ")
        ),
    )
}

#[async_trait]
impl ModelSource for RunModels {
    async fn list(&self) -> Result<Vec<ModelCard>, OpenAiError> {
        let mut cards = Vec::new();
        for handle in self.state.registry.handles() {
            let id = handle.id.to_string();
            let created = handle.created_at;
            let serving = serving(&handle);
            let final_weights = serving.as_ref().and_then(final_weights);
            if is_live(&handle) {
                cards.push(ModelCard::new(&id, created).live());
            } else if final_weights.is_some() {
                cards.push(ModelCard::new(&id, created));
            }
            if final_weights.is_some() {
                cards.push(ModelCard::new(format!("{id}@final"), created));
            }
            for checkpoint in servable_checkpoints(&handle) {
                cards.push(ModelCard::new(
                    format!("{id}@{}", checkpoint.id),
                    checkpoint.written_at.unwrap_or(created),
                ));
            }
            if serving.is_some() {
                let base = ModelCard::new(format!("{id}@base"), created);
                cards.push(if is_live(&handle) { base.live() } else { base });
            }
        }
        Ok(cards)
    }

    async fn resolve(&self, model: &str) -> Result<Target, OpenAiError> {
        let (run, suffix) = parse_id(model).ok_or_else(|| {
            OpenAiError::model_not_found(
                model,
                "a model id here is a run id, optionally followed by @final, @best, @latest, \
                 @step-<N> or @base",
            )
        })?;
        let handle = self
            .state
            .registry
            .get(&run)
            .ok_or_else(|| OpenAiError::model_not_found(model, "no run has this id"))?;
        let live = is_live(&handle);
        let runner = || -> Arc<dyn LiveRunner> {
            Arc::new(LiveRun {
                handle: handle.clone(),
                state: self.state.clone(),
            })
        };
        match (&suffix, live) {
            (Weights::Current, true) => {
                return Ok(Target::Live {
                    runner: runner(),
                    base: false,
                });
            }
            // The run holds the device; the model it started from is the same
            // trainer with its adapter disabled. Not its anchor, which may be
            // another file entirely.
            (Weights::Base, true) => {
                return Ok(Target::Live {
                    runner: runner(),
                    base: true,
                });
            }
            (Weights::Final, true) => {
                return Err(OpenAiError::conflict(format!(
                    "run {run} is still training, so it has no final weights yet; serve its \
                     current weights with model `{run}`"
                )));
            }
            (Weights::Latest | Weights::Checkpoint(_), true) => {
                return Err(OpenAiError::busy(
                    "device_busy",
                    format!(
                        "run {run} holds the device, so its checkpoints cannot be loaded beside \
                         it; serve its live weights with model `{run}`, or retry when it \
                         finishes"
                    ),
                    30,
                ));
            }
            _ => {}
        }
        let serving = serving(&handle).ok_or_else(|| {
            OpenAiError::conflict(format!(
                "run {run} predates the serving metadata this server records, so its weights \
                 cannot be reloaded"
            ))
        })?;
        let cold = |spec: WeightsSpec, name: &str| Target::Cold {
            spec,
            fingerprint: format!("{run}@{name}"),
        };
        match suffix {
            Weights::Current | Weights::Final => final_weights(&serving)
                .map(|spec| cold(spec, "final"))
                .ok_or_else(|| not_found(model, &handle)),
            Weights::Base => Ok(cold(weights(&serving, &serving.model, None), "base")),
            Weights::Latest => {
                let directory = handle
                    .record
                    .artifacts
                    .checkpoint_directory
                    .as_ref()
                    .ok_or_else(|| not_found(model, &handle))?;
                let state = retrograd_run::latest_checkpoint(directory)
                    .map_err(|_| not_found(model, &handle))?;
                let adapter = state.with_extension("gguf");
                if !adapter.is_file() {
                    return Err(not_found(model, &handle));
                }
                let id = state
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("latest")
                    .to_owned();
                Ok(cold(weights(&serving, &serving.model, Some(adapter)), &id))
            }
            Weights::Checkpoint(wanted) => {
                let step = wanted
                    .strip_prefix("step-")
                    .and_then(|digits| digits.parse::<u64>().ok());
                let checkpoint = servable_checkpoints(&handle)
                    .into_iter()
                    .find(|checkpoint| {
                        checkpoint.id == wanted
                            || step.is_some()
                                && checkpoint
                                    .id
                                    .strip_prefix("step-")
                                    .and_then(|digits| digits.parse::<u64>().ok())
                                    == step
                    })
                    .ok_or_else(|| not_found(model, &handle))?;
                Ok(cold(
                    weights(&serving, &serving.model, Some(checkpoint.adapter)),
                    &checkpoint.id,
                ))
            }
        }
    }
}

/// A training run lending its trainer to one chat request.
struct LiveRun {
    handle: Arc<RunHandle>,
    state: AppState,
}

#[async_trait]
impl LiveRunner for LiveRun {
    async fn run(&self, job: TrainerJob) -> Result<String, OpenAiError> {
        let (reply, answer) = oneshot::channel();
        dispatch(&self.handle, RunCommand::WithTrainer(job, reply))?;
        let (at, ()) = wait(&self.state, answer, "chat completion").await?;
        // Not an event of the run: nothing about its training changed.
        tracing::info!(
            run_id = %self.handle.id,
            global_step = at.global_step,
            "served a chat completion from a live run"
        );
        Ok(format!("{}@step-{}", self.handle.id, at.global_step))
    }
}

/// The device permit the runs queue on, lent to the serving session.
pub struct DevicePermit {
    pub device: Arc<Semaphore>,
    pub registry: Arc<RunRegistry>,
    /// How long a load waits for the device before it is refused. Zero refuses
    /// at once: a request should not sit behind a training run for hours.
    pub wait: Duration,
}

/// How often a waiting load looks at the permit again.
const PERMIT_POLL: Duration = Duration::from_millis(50);

impl DeviceLease for DevicePermit {
    fn acquire(&self) -> Result<Lease, OpenAiError> {
        let deadline = Instant::now() + self.wait;
        loop {
            match self.device.clone().try_acquire_owned() {
                Ok(permit) => return Ok(Box::new(permit)),
                Err(tokio::sync::TryAcquireError::Closed) => {
                    return Err(OpenAiError::internal("the device queue is closed"));
                }
                Err(tokio::sync::TryAcquireError::NoPermits) if Instant::now() < deadline => {
                    std::thread::sleep(PERMIT_POLL);
                }
                Err(tokio::sync::TryAcquireError::NoPermits) => {
                    let holder = self
                        .registry
                        .live_handles()
                        .into_iter()
                        .find(|handle| handle.holds_device());
                    let message = match holder {
                        Some(run) => format!(
                            "the device is held by run {id}; serve this run's live weights with \
                             model `{id}`, or retry when it finishes",
                            id = run.id
                        ),
                        None => "the device is held by a training run; retry when it finishes"
                            .to_owned(),
                    };
                    return Err(OpenAiError::busy("device_busy", message, 30));
                }
            }
        }
    }
}

/// A problem document's sentence, as an OpenAI client reads one. Kept here,
/// beside the type it translates from: `retrograd-openai` knows nothing of this
/// server's errors.
impl From<ApiError> for OpenAiError {
    fn from(error: ApiError) -> Self {
        let translated = match error.kind {
            ProblemKind::InvalidRequest
            | ProblemKind::ServerDeclared
            | ProblemKind::UnknownCatalogId
            | ProblemKind::InsufficientMemory
            | ProblemKind::UnsupportedMediaType
            | ProblemKind::PayloadTooLarge => {
                OpenAiError::invalid(error.detail).with_status(error.kind.status())
            }
            ProblemKind::ForbiddenPath => {
                OpenAiError::invalid(error.detail).with_status(error.kind.status())
            }
            ProblemKind::NotFound => OpenAiError::not_found(error.detail),
            ProblemKind::Conflict => OpenAiError::conflict(error.detail),
            ProblemKind::DeviceBusy => OpenAiError::busy("device_busy", error.detail, 30),
            ProblemKind::Timeout => OpenAiError::timeout(error.detail),
            ProblemKind::Unauthorized => OpenAiError::unauthorized(error.detail),
            ProblemKind::MethodNotAllowed | ProblemKind::NotImplemented | ProblemKind::Internal => {
                OpenAiError::internal(error.detail).with_status(error.kind.status())
            }
        };
        // Validation errors are 400 in this contract, not 422: the SDKs retry
        // neither, and 400 is what they expect.
        let translated = match translated.status {
            http::StatusCode::UNPROCESSABLE_ENTITY => {
                translated.with_status(http::StatusCode::BAD_REQUEST)
            }
            _ => translated,
        };
        match error.errors.first() {
            Some(field) => {
                translated.with_param(field.pointer.trim_start_matches('/').replace('/', "."))
            }
            None => translated,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_is_a_run_and_a_suffix() {
        let run = Uuid::new_v4();
        assert_eq!(parse_id(&run.to_string()), Some((run, Weights::Current)));
        assert_eq!(
            parse_id(&format!("{run}@final")),
            Some((run, Weights::Final))
        );
        assert_eq!(parse_id(&format!("{run}@base")), Some((run, Weights::Base)));
        assert_eq!(
            parse_id(&format!("{run}@latest")),
            Some((run, Weights::Latest))
        );
        assert_eq!(
            parse_id(&format!("{run}@step-40")),
            Some((run, Weights::Checkpoint("step-40".into())))
        );
        assert_eq!(parse_id(&format!("{run}@")), None);
        assert_eq!(parse_id("base"), None, "a bare base is ambiguous here");
    }

    #[test]
    fn a_problem_reads_as_the_envelope_of_the_same_failure() {
        let error = OpenAiError::from(ApiError::invalid("bad").with_field(
            "/messages/0",
            crate::error::ErrorCode::InvalidValue,
            "x",
        ));
        assert_eq!(error.status, http::StatusCode::BAD_REQUEST);
        assert_eq!(error.param.as_deref(), Some("messages.0"));
        let busy = OpenAiError::from(ApiError::new(ProblemKind::DeviceBusy, "held"));
        assert_eq!(busy.status, http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(busy.code, Some("device_busy"));
        let gone = OpenAiError::from(ApiError::new(ProblemKind::Conflict, "gone"));
        assert_eq!(gone.status, http::StatusCode::CONFLICT);
        assert!(OpenAiError::from(ApiError::new(ProblemKind::Timeout, "slow")).status == 504);
    }

    #[test]
    fn only_the_openai_routes_answer_in_its_envelope() {
        assert!(is_openai_path("/v1/chat/completions"));
        assert!(is_openai_path("/v1/models"));
        assert!(is_openai_path("/v1/models/abc@best"));
        assert!(!is_openai_path("/v1/modelsx"));
        assert!(!is_openai_path("/v1/runs"));
    }
}
