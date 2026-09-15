//! Where a tool actually runs.
//!
//! [`Sandbox`] is the seam that lets a tool be written once and executed either
//! inside a container (`retrograd-container`) or, deliberately and explicitly,
//! straight on the host. A tool crate therefore never depends on Docker, and a
//! container backend never depends on the tool catalogue.
//!
//! # Observations are training tokens
//!
//! Whatever a sandbox lets out of [`Sandbox::exec`] ends up tokenized into the
//! trajectory. It must be a deterministic function of the action and of nothing
//! else. Concretely, an implementation must never leak:
//!
//! - a container id, a hostname or a pid;
//! - a host absolute path (everything is reported relative to [`Sandbox::workdir`]);
//! - a duration or a timestamp;
//! - an unsorted directory listing.
//!
//! Two members of one group would otherwise receive observations that differ by
//! noise, and the group-relative baseline would be measuring that noise.
//!
//! What the rule can and cannot cover: everything the *sandbox* contributes is
//! fixed - the working directory, the hostname, the listing order, the wording
//! of a timeout. What a command deliberately prints is not, and cannot be: a
//! `bash` tool can always run `date`, `echo $$` or `head -c8 /dev/urandom`. That
//! part of the guarantee belongs to the task author, who chooses which tools the
//! profile advertises; the sandbox's job is to add no difference of its own.
//!
//! # Failed action against broken world
//!
//! Every method returns [`Error::Sandbox`] - and only that variant - when the
//! sandbox itself is unusable. Any other error is a failed *action*, which a
//! tool is free to render as an observation. Blurring the two is how a
//! trajectory ends up scored and trained against a container that died halfway
//! through it; see [`Error::is_sandbox_failure`].

use std::collections::BTreeMap;
use std::ops::Deref;
use std::path::{Component, Path};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExecRequest {
    /// Argv, never a shell string: quoting a command built from model output is
    /// how an argument becomes an injection.
    pub argv: Vec<String>,
    pub stdin: Option<Vec<u8>>,
    /// Relative to [`Sandbox::workdir`] when relative.
    pub cwd: Option<String>,
    pub env: BTreeMap<String, String>,
    pub timeout: Duration,
    /// Cap on captured output. An unbounded tool result silently eats the
    /// trajectory's whole token budget.
    pub max_output_bytes: usize,
}

impl ExecRequest {
    pub fn new<I, S>(argv: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            argv: argv.into_iter().map(Into::into).collect(),
            stdin: None,
            cwd: None,
            env: BTreeMap::new(),
            timeout: Duration::from_secs(30),
            max_output_bytes: 64 * 1024,
        }
    }

    pub fn with_stdin(mut self, stdin: impl Into<Vec<u8>>) -> Self {
        self.stdin = Some(stdin.into());
        self
    }

    pub fn with_cwd(mut self, cwd: impl Into<String>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_output_bytes(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes;
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecOutput {
    /// `None` when the process was killed by a signal - including by the
    /// sandbox's own memory limit.
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    /// Output hit `max_output_bytes` and was cut on a UTF-8 boundary.
    pub truncated: bool,
}

impl ExecOutput {
    pub fn succeeded(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileKind {
    File,
    Dir,
    Symlink,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirEntry {
    /// Relative to [`Sandbox::workdir`], never absolute on the host.
    pub path: String,
    pub kind: FileKind,
    pub size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SandboxLimits {
    pub cpus: f32,
    pub memory_bytes: u64,
    pub pids: u32,
    pub exec_timeout: Duration,
    pub max_output_bytes: usize,
    /// Whether the sandbox can reach the network at all. Tools word their own
    /// error messages from this rather than letting the model discover it by
    /// timing out.
    pub network: bool,
}

impl Default for SandboxLimits {
    fn default() -> Self {
        Self {
            cpus: 1.0,
            memory_bytes: 1024 * 1024 * 1024,
            pids: 256,
            exec_timeout: Duration::from_secs(30),
            max_output_bytes: 64 * 1024,
            network: false,
        }
    }
}

/// One episode's execution environment. Owned by a single trajectory.
#[async_trait]
pub trait Sandbox: Send + Sync {
    /// Runs a command.
    ///
    /// **A non-zero exit is not an error.** It comes back as an [`ExecOutput`]
    /// the policy reads and reacts to - a failing test suite is the whole point
    /// of the task. [`Error::Sandbox`] is reserved for a sandbox that is no
    /// longer usable: a dead container, an unreachable daemon. That one costs
    /// the trajectory, because everything already collected is conditioned on a
    /// world that has gone away. Same split as [`crate::Environment::step`].
    async fn exec(&self, request: ExecRequest) -> Result<ExecOutput>;

    /// [`Error::Sandbox`] if the sandbox is gone; any other error means the file
    /// could not be read, which is the policy's problem and a tool's to render.
    async fn read_file(&self, path: &str) -> Result<Vec<u8>>;
    async fn write_file(&self, path: &str, bytes: &[u8]) -> Result<()>;
    /// Sorted, so the observation does not depend on filesystem iteration order.
    async fn list_dir(&self, path: &str) -> Result<Vec<DirEntry>>;
    async fn remove(&self, path: &str) -> Result<()>;

    /// Root of the episode. Every relative path resolves against it, and every
    /// path reported back is relative to it.
    fn workdir(&self) -> &str;

    fn limits(&self) -> SandboxLimits;
}

/// Normalizes a path coming from a tool call into one relative to `workdir`.
///
/// Paths are model output, so `..` and host-absolute paths are the whole
/// attack. They are refused *by construction* rather than by canonicalizing
/// afterwards, which cannot see a file that does not exist yet. Returns the
/// empty string for the root itself.
pub fn relative_path(workdir: &str, path: &str) -> Result<String> {
    let trimmed = path
        .strip_prefix(workdir)
        .map(|rest| rest.trim_start_matches('/'))
        .unwrap_or(path);
    let mut parts = Vec::new();
    for component in Path::new(trimmed).components() {
        match component {
            Component::Normal(part) => parts.push(part.to_string_lossy().into_owned()),
            Component::CurDir => {}
            _ => {
                return Err(Error::invalid(format!(
                    "path '{path}' escapes the sandbox root"
                )));
            }
        }
    }
    Ok(parts.join("/"))
}

/// Who a [`Lease`] goes back to.
///
/// Deliberately synchronous: a lease is dropped on the rollout's hot path and,
/// more importantly, on its failure paths - a deadline, a poisoned trajectory,
/// an unwind - where there is no async context left to await cleanup in. An
/// implementation records the return and does the slow part (wiping a
/// workspace, killing leftovers, destroying a container) elsewhere.
pub trait SandboxOwner: Send + Sync {
    /// `healthy` is false when the sandbox must not be handed to another
    /// episode. Recycling on doubt is the one failure that shows up in no
    /// metric: the next episode inherits state, the group's members stop being
    /// independent, and the relative baseline measures the contamination.
    fn release(&self, token: u64, healthy: bool);
}

/// One episode's exclusive hold on a sandbox.
///
/// Returned on drop, on every path, so a trajectory that dies mid-turn does not
/// strand the world it was acting on.
pub struct Lease {
    sandbox: Arc<dyn Sandbox>,
    owner: Arc<dyn SandboxOwner>,
    token: u64,
    healthy: bool,
}

impl Lease {
    /// `token` is the owner's own name for this sandbox; nothing outside the
    /// owner interprets it, and it must never reach an observation.
    pub fn new(sandbox: Arc<dyn Sandbox>, owner: Arc<dyn SandboxOwner>, token: u64) -> Self {
        Self {
            sandbox,
            owner,
            token,
            healthy: true,
        }
    }

    pub fn sandbox(&self) -> Arc<dyn Sandbox> {
        self.sandbox.clone()
    }

    /// Declares the sandbox unfit for another episode. Cheap insurance: the
    /// cost is one container creation, the alternative is a silently
    /// contaminated group.
    pub fn poison(&mut self) {
        self.healthy = false;
    }
}

impl Deref for Lease {
    type Target = dyn Sandbox;

    fn deref(&self) -> &Self::Target {
        self.sandbox.as_ref()
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.owner.release(self.token, self.healthy);
    }
}

/// Where an episode gets its sandbox from: a container pool, a single local
/// directory, anything.
///
/// This is what keeps `retrograd-env` free of Docker - the environment holds a
/// `dyn SandboxProvider` and never learns which one it got.
#[async_trait]
pub trait SandboxProvider: Send + Sync {
    async fn acquire(&self) -> Result<Lease>;

    /// Best-effort warm-up of `count` instances. Called while the optimizer
    /// step runs, which is dead time on the environment side.
    async fn prewarm(&self, count: usize) -> Result<()> {
        let _ = count;
        Ok(())
    }

    async fn shutdown(&self) {}

    /// Destroys every sandbox of this provider immediately, returning how many
    /// went. Unlike [`shutdown`](Self::shutdown) it does not drain: it is what a
    /// signal handler calls, and a lease still out is never coming back. See
    /// [`crate::interrupt`].
    async fn force_cleanup(&self) -> usize {
        0
    }

    /// What acquiring sandboxes cost. Surfaced through
    /// [`EnvironmentFactory::metric_values`](crate::EnvironmentFactory::metric_values)
    /// so the pool's behaviour is visible next to the training curves rather
    /// than only in a debugger.
    fn metric_values(&self) -> Vec<retrograd_metrics::MetricValue> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_path_is_normalized_against_the_workdir_or_refused() {
        assert_eq!(
            relative_path("/work", "/work/src/a.py").unwrap(),
            "src/a.py"
        );
        assert_eq!(relative_path("/work", "./src/a.py").unwrap(), "src/a.py");
        assert_eq!(relative_path("/work", "/work").unwrap(), "");
        assert_eq!(relative_path("/work", ".").unwrap(), "");
        for escape in ["../etc/passwd", "a/../../b", "/etc/passwd"] {
            let error = relative_path("/work", escape).unwrap_err();
            assert!(error.to_string().contains("escapes"), "{escape}: {error}");
        }
    }
}
