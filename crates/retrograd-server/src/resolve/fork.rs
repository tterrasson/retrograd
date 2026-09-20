//! `fork_from`: continuing from a checkpoint, and branching from one.
//!
//! The plan calls this out as one operation with two uses, and it is: `--resume`
//! is a fork that changes nothing, and an experiment branch is a fork that
//! changes something. Both come down to the same three questions - which
//! checkpoint, which configuration, and is the pair compatible.
//!
//! The answer to the second is what distinguishes the two request shapes.
//! `{"run": "<id>"}` names a run this server knows, so the parent's own effective
//! configuration is the base. `{"path": "…/step-400.state"}` names a directory
//! that carries a manifest and no configuration at all, so the body has to bring
//! one.
//!
//! Two design points worth stating.
//!
//! **The checkpoint enters as an override.** `fork_from` resolves to
//! `checkpoint.resume_from`, deep-merged with the client's own `params` before
//! anything else runs. It therefore inherits the whole of invariant 6 for free: a
//! resume path is never re-derived, it shows up in the provenance as overridden,
//! and the resolver needs no special case for a fork.
//!
//! **Compatibility is checked twice, on purpose.** `RunController::begin` is the
//! authority - it compares the model signature, which needs a loaded model, and
//! it is what protects a resume against corruption. But it runs on the worker,
//! minutes later, and a failure there is a `failed` run rather than a refused
//! request. So the parts that can be checked from files alone are checked here,
//! and answered as the 409 the contract requires.

use std::path::{Path, PathBuf};

use retrograd_checkpoint::{self as checkpoint, Manifest};
use retrograd_config::{ConfigDocument, RunConfig};
use serde_json::Value;

use crate::dto;
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::state::AppState;

/// The document sections a change to which changes what is trained, and therefore
/// what a resume may not cross. The same ground `trajectory_signature` covers,
/// named in the *document* grammar so the pointers a 409 reports are the ones the
/// client wrote.
const TRAJECTORY_SECTIONS: [&str; 6] =
    ["training", "sft", "ppo", "grpo", "evaluation", "reference"];

/// What a fork continues from.
pub struct ForkTarget {
    /// The parent's effective configuration, with the catalogue ids resolved back
    /// to what they stand for. `None` when the fork named a bare path, which
    /// carries no configuration.
    pub document: Option<ConfigDocument>,
    /// The `.state` directory the run resumes from.
    pub state_dir: PathBuf,
    pub manifest: Manifest,
    /// The parent run's id, when there was one.
    pub parent: Option<String>,
    /// The catalogue id the parent's reward came from, carried through so the
    /// fork's own effective configuration redacts to the same marker. Without it,
    /// a fork of a fork would find `<reward>` and have nothing to look up.
    pub reward_id: Option<String>,
}

impl ForkTarget {
    /// The synthetic override a fork contributes.
    pub fn as_override(&self) -> Value {
        serde_json::json!({
            "checkpoint": {"resume_from": self.state_dir.to_string_lossy()}
        })
    }
}

/// Resolves `fork_from` into a checkpoint and, when it named a run, a base
/// configuration.
pub async fn target(state: &AppState, fork: &dto::ForkFrom) -> ApiResult<ForkTarget> {
    match (&fork.run, &fork.path) {
        (Some(_), Some(_)) => Err(ApiError::invalid(
            "fork_from takes either a run or a path, not both",
        )
        .with_field("/fork_from", ErrorCode::InvalidValue, "ambiguous source")),
        (None, None) => Err(
            ApiError::invalid("fork_from needs a run or a path to continue from").with_field(
                "/fork_from",
                ErrorCode::MissingField,
                "empty",
            ),
        ),
        (Some(run), None) => from_run(state, run, fork.checkpoint.as_deref()),
        (None, Some(path)) => from_path(state, path, fork.checkpoint.as_deref()),
    }
}

/// A fork of a run this server knows.
fn from_run(state: &AppState, run: &str, id: Option<&str>) -> ApiResult<ForkTarget> {
    let parent = super::super::api::runs::lookup(state, run).map_err(|_| {
        ApiError::not_found(format!("no run with id {run} to fork from")).with_field(
            "/fork_from/run",
            ErrorCode::NotFound,
            "unknown run",
        )
    })?;
    let directory = parent
        .record
        .artifacts
        .checkpoint_directory
        .clone()
        .ok_or_else(|| {
            ApiError::new(
                ProblemKind::Conflict,
                "the parent run writes no checkpoints, so there is nothing to fork from",
            )
            .with_field(
                "/fork_from/run",
                ErrorCode::Conflict,
                "no checkpoint directory",
            )
        })?;
    let state_dir = pick(&directory, id)?;
    let manifest = manifest(&state_dir)?;
    let (document, reward_id) = parent_document(state, &parent.record.effective_config)?;
    Ok(ForkTarget {
        document: Some(document),
        state_dir,
        manifest,
        parent: Some(parent.id.to_string()),
        reward_id,
    })
}

/// A fork of a checkpoint written by something else - the CLI, or a previous
/// deployment. The configuration has to come from the request.
fn from_path(state: &AppState, path: &str, id: Option<&str>) -> ApiResult<ForkTarget> {
    // Validated but not substituted: `resolve_path` canonicalizes, and what goes
    // into `checkpoint.resume_from` ends up in the effective configuration. Returning
    // `/private/var/…` for a `/var/…` path the caller sent describes the server's
    // filesystem for no gain.
    state.resolve_path(path, "/fork_from/path")?;
    let candidate = PathBuf::from(path);
    // A directory of checkpoints is as good an answer as one checkpoint: it is
    // what `--resume` without a path means, and refusing it would make the caller
    // do the `latest` resolution the server already has.
    let state_dir = if candidate.join(checkpoint::MANIFEST_FILE).is_file() {
        candidate
    } else {
        pick(&candidate, id)?
    };
    let manifest = manifest(&state_dir)?;
    Ok(ForkTarget {
        document: None,
        state_dir,
        manifest,
        parent: None,
        reward_id: None,
    })
}

/// A parent's configuration and the catalogue id its reward came from.
type ParentConfig = (ConfigDocument, Option<String>);

/// The parent's configuration, with `<reward:id>` turned back into the command
/// the operator declared.
///
/// The stored document is the *redacted* one - that is what the contract requires of
/// everything that leaves the process, and `run.json` holds the bytes the API
/// answered. So a fork has to substitute the catalogue a second time. The
/// alternative, keeping an unredacted copy on disk, would put the operator's
/// command in a second place for no gain.
fn parent_document(
    state: &AppState,
    stored: &serde_json::value::RawValue,
) -> ApiResult<ParentConfig> {
    let mut document: ConfigDocument = serde_json::from_str(stored.get()).map_err(|error| {
        ApiError::internal(format!(
            "the parent run's stored configuration could not be re-read: {error}"
        ))
    })?;
    let mut reward_id = None;
    for command in [
        document.grpo.as_mut().map(|grpo| &mut grpo.reward_command),
        document.ppo.as_mut().map(|ppo| &mut ppo.reward_command),
    ]
    .into_iter()
    .flatten()
    {
        let (resolved, id) = unredact(state, command)?;
        *command = resolved;
        reward_id = reward_id.or(id);
    }
    Ok((document, reward_id))
}

type Unredacted = (Vec<String>, Option<String>);

fn unredact(state: &AppState, command: &[String]) -> ApiResult<Unredacted> {
    let [single] = command else {
        return Ok((command.to_vec(), None));
    };
    let Some(marker) = single
        .strip_prefix("<reward:")
        .and_then(|rest| rest.strip_suffix('>'))
    else {
        // `<reward>` with no id: a run created before the id was recorded, or one
        // whose reward the catalogue no longer names. Either way the command
        // cannot be reconstructed, and guessing one would run something else.
        if single.starts_with('<') {
            return Err(ApiError::new(
                ProblemKind::UnknownCatalogId,
                "the parent run's reward cannot be identified, so it cannot be forked; \
                 send a recipe naming a declared reward instead",
            )
            .with_field(
                "/fork_from/run",
                ErrorCode::UnknownCatalogId,
                "the parent's reward has no catalogue id",
            ));
        }
        return Ok((command.to_vec(), None));
    };
    let entry = state.catalog.reward(marker).map_err(|error| {
        ApiError::new(ProblemKind::UnknownCatalogId, error.to_string()).with_field(
            "/fork_from/run",
            ErrorCode::UnknownCatalogId,
            format!("the parent used reward '{marker}', which this server no longer declares"),
        )
    })?;
    Ok((entry.command.clone(), Some(marker.to_string())))
}

/// Resolves a checkpoint id inside a directory, or the latest when none is given.
fn pick(directory: &Path, id: Option<&str>) -> ApiResult<PathBuf> {
    let Some(id) = id else {
        // The same resolution `--resume` performs, so a fork with no id picks what
        // the CLI would.
        return retrograd_run::latest_checkpoint(directory).map_err(|error| {
            ApiError::invalid(error.to_string()).with_field(
                "/fork_from/checkpoint",
                ErrorCode::NotFound,
                "no checkpoint to resume from",
            )
        });
    };
    // An id is a directory *name*, not a path. Without this, `../../etc` would
    // walk out of a directory the caller was already allowed to name - the one
    // place a path root does not help, because the escape happens after the root
    // check.
    if id.is_empty()
        || id.contains('/')
        || id.contains('\\')
        || id.contains("..")
        || id.starts_with('.')
    {
        return Err(
            ApiError::invalid(format!("'{id}' is not a checkpoint id")).with_field(
                "/fork_from/checkpoint",
                ErrorCode::InvalidValue,
                "a name like 'step-000000000400' or 'best', with no path separators",
            ),
        );
    }
    // Both spellings, because both are what a client has in hand: the id as the
    // listing reports it, and the directory name as it is on disk.
    let candidate = if id.ends_with(&format!(".{}", checkpoint::STATE_SUFFIX)) {
        directory.join(id)
    } else {
        directory.join(format!("{id}.{}", checkpoint::STATE_SUFFIX))
    };
    if !candidate.is_dir() {
        return Err(ApiError::not_found(format!(
            "no checkpoint '{id}' in {}",
            directory.display()
        ))
        .with_field(
            "/fork_from/checkpoint",
            ErrorCode::NotFound,
            "unknown checkpoint",
        ));
    }
    Ok(candidate)
}

fn manifest(state_dir: &Path) -> ApiResult<Manifest> {
    checkpoint::read_manifest(state_dir).map_err(|error| {
        // 422 rather than 404: the directory is there, it is just not a complete
        // checkpoint - an interrupted write, or one from another format version.
        ApiError::invalid(error.to_string()).with_field(
            "/fork_from",
            ErrorCode::InvalidValue,
            "not a complete checkpoint",
        )
    })
}

/// Refuses a fork whose configuration would continue on a different trajectory.
///
/// What is compared here is what a file can answer: the algorithm, the base model
/// by size, and the trajectory fingerprint. The model *signature* - the runtime's
/// architecture and shape hyperparameters - is not, because reading it needs a
/// loaded model; `RunController::begin` compares it on the worker and remains the
/// authority.
pub async fn check_compatible(
    state: &AppState,
    target: &ForkTarget,
    config: &RunConfig,
    params: &Value,
) -> ApiResult<()> {
    let conflict = |message: String| {
        ApiError::new(ProblemKind::Conflict, message).with_field(
            "/fork_from",
            ErrorCode::Conflict,
            "incompatible with this configuration",
        )
    };
    let algorithm = match &config.algorithm {
        retrograd_config::Algorithm::Sft(_) => "sft",
        retrograd_config::Algorithm::Ppo(_) => "ppo",
        retrograd_config::Algorithm::Grpo(_) => "grpo",
        retrograd_config::Algorithm::Distill(_) => "distill",
        retrograd_config::Algorithm::AgentGrpo(_) => "agent_grpo",
    };
    if target.manifest.algorithm != algorithm {
        return Err(conflict(format!(
            "the checkpoint was written by {} and this run is {algorithm}",
            target.manifest.algorithm
        )));
    }
    let model_bytes = retrograd_run::model_bytes(&config.model);
    if target.manifest.model_bytes != model_bytes {
        return Err(conflict(format!(
            "the checkpoint was taken on a {}-byte model and this run's model is {model_bytes} \
             bytes",
            target.manifest.model_bytes
        ))
        .with_field("/model/path", ErrorCode::Conflict, "a different base model"));
    }
    let reference_fingerprint = match &config.reference {
        Some(reference) => state.model_fingerprint(&reference.model).await?,
        None => String::new(),
    };
    if target.manifest.reference_fingerprint != reference_fingerprint {
        return Err(
            conflict("the checkpoint uses a different fixed reference".to_string()).with_field(
                "/config/reference/model",
                ErrorCode::Conflict,
                "a different fixed reference",
            ),
        );
    }
    let signature = retrograd_run::trajectory_signature(config).map_err(ApiError::from)?;
    if signature == target.manifest.trajectory_signature {
        return Ok(());
    }
    // The signature is a hash, so it cannot name what changed. What *can* name it
    // is the request: the parameters the client sent that fall in a section the
    // trajectory depends on are exactly the candidates.
    let mut problem = conflict(
        "this configuration would continue the checkpoint on a different training \
         trajectory; a fork that changes what is trained has to start from scratch"
            .to_string(),
    );
    let suspects: Vec<String> = retrograd_plan::merge::overridden_paths(params)
        .into_iter()
        .filter(|path| {
            TRAJECTORY_SECTIONS
                .iter()
                .any(|section| path.split('.').next() == Some(section))
        })
        .collect();
    for path in &suspects {
        problem = problem.with_field(
            format!("/params/{}", path.replace('.', "/")),
            ErrorCode::Conflict,
            "changes the training trajectory",
        );
    }
    if suspects.is_empty() {
        problem = problem.with_field(
            "/config",
            ErrorCode::Conflict,
            "the configuration differs from the one the checkpoint was written by",
        );
    }
    Err(problem)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_checkpoint_id_cannot_be_a_path() {
        let dir = std::env::temp_dir();
        for bad in ["../escape", "a/b", "..", ".hidden", ""] {
            let error = pick(&dir, Some(bad)).expect_err(bad);
            assert_eq!(error.kind, ProblemKind::InvalidRequest, "{bad}");
        }
    }

    #[test]
    fn the_fork_override_is_a_resume_path() {
        let target = ForkTarget {
            document: None,
            state_dir: PathBuf::from("/runs/ckpt/step-000000000400.state"),
            manifest: Manifest {
                format_version: retrograd_checkpoint::FORMAT_VERSION,
                checkpoint_id: "step-000000000400".into(),
                global_step: 400,
                adapter: Some("adapter.gguf".into()),
                trainable: None,
                trainable_policy: "lora".into(),
                files: Vec::new(),
                app_version: String::new(),
                llama_cpp_commit: String::new(),
                model_signature: String::new(),
                model_bytes: 0,
                model_fingerprint: String::new(),
                reference_fingerprint: String::new(),
                algorithm: "sft".into(),
                trajectory_signature: String::new(),
                resume_boundary: "epoch".into(),
                artifacts: Default::default(),
            },
            parent: None,
            reward_id: None,
        };
        assert_eq!(
            target.as_override(),
            serde_json::json!({
                "checkpoint": {"resume_from": "/runs/ckpt/step-000000000400.state"}
            })
        );
    }
}
