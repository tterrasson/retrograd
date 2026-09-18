//! One container, seen as a [`Sandbox`].

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use bollard::container::LogOutput;
use bollard::exec::{CreateExecOptions, StartExecOptions, StartExecResults};
use bollard::query_parameters::{
    CreateContainerOptions, RemoveContainerOptionsBuilder, StartContainerOptions,
};
use futures_util::StreamExt;
use retrograd_agent_core::text::truncate_utf8;
use retrograd_agent_core::{
    DirEntry, Error, ExecOutput, ExecRequest, FileKind, Result, Sandbox, SandboxLimits,
    relative_path,
};
use tokio::io::AsyncWriteExt;

use crate::client::{DockerClient, docker_error};
use crate::metrics::SandboxMetrics;
use crate::pool::{ManagedSandbox, SandboxSource};
use crate::reaper::RunLabels;
use crate::spec::{ContainerSpec, WORKDIR};

/// Grace on top of the caller's timeout before we stop waiting on the daemon.
///
/// The in-container `timeout` should have killed the process long before; if it
/// has not, the container is no longer doing what we asked and is retired rather
/// than handed to the next episode.
const WATCHDOG_GRACE: Duration = Duration::from_secs(10);

/// Truncation marker. Fixed text, on purpose: it becomes training tokens, so it
/// must be the same bytes for every episode that hits the cap.
const TRUNCATION_MARKER: &str = "\n[output truncated]";

/// Cap on a single [`Sandbox::read_file`]. The file's bytes cross the socket into
/// our heap, and an episode is free to generate a gigabyte.
const MAX_READ_BYTES: usize = 8 * 1024 * 1024;

/// Docker reports an exec's status as `i64`; a POSIX exit code is a byte, and
/// the `Option` already means "no status". A value that does not fit `i32` is
/// therefore not an exit code, and reads as absent rather than as a truncation.
fn exit_code(reported: Option<i64>) -> Option<i32> {
    reported.and_then(|code| i32::try_from(code).ok())
}

/// What [`ContainerSandbox::exec_bytes`] observed.
struct RawExec {
    exit_code: Option<i32>,
    stdout: Vec<u8>,
    stderr: String,
    /// The command produced more than the cap; `stdout` is short of the truth
    /// and a file read must fail rather than return a prefix.
    overflowed: bool,
}

/// A live container, reached through Docker execs and exposed to the rest of
/// the stack as a [`Sandbox`].
pub struct ContainerSandbox {
    client: DockerClient,
    id: String,
    limits: SandboxLimits,
    timeout_wrapper: bool,
    /// Set when the container stopped being trustworthy - a watchdog fire, a
    /// failed cleanup. The pool destroys rather than recycles it.
    healthy: AtomicBool,
    metrics: Arc<SandboxMetrics>,
}

impl ContainerSandbox {
    pub(crate) fn new(
        client: DockerClient,
        id: String,
        limits: SandboxLimits,
        timeout_wrapper: bool,
        metrics: Arc<SandboxMetrics>,
    ) -> Self {
        Self {
            client,
            id,
            limits,
            timeout_wrapper,
            healthy: AtomicBool::new(true),
            metrics,
        }
    }

    pub(crate) fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Relaxed)
    }

    pub(crate) fn poison(&self) {
        self.healthy.store(false, Ordering::Relaxed);
        self.metrics.record_broken();
    }

    pub(crate) async fn destroy(&self) {
        let options = RemoveContainerOptionsBuilder::default()
            .force(true)
            .v(true)
            .build();
        if let Err(error) = self
            .client
            .docker()
            .remove_container(&self.id, Some(options))
            .await
        {
            // Nothing useful to do about it here; the reaper is the backstop.
            tracing::warn!(error = %error, "removing a sandbox container failed");
        }
    }

    /// Empties the workspace and kills whatever the episode left running.
    ///
    /// `Err` means the caller must destroy the container. Both halves matter:
    /// leftover files contaminate the next episode's observations, and a
    /// leftover background process contaminates its timings and its cpu budget.
    pub(crate) async fn reset_workspace(&self) -> Result<()> {
        // `find -delete` on the contents rather than `rm -rf /work`: the mount
        // point itself is a tmpfs and must survive.
        let wipe = self
            .exec_inner(
                ExecRequest::new(["find", WORKDIR, "-mindepth", "1", "-delete"])
                    .with_timeout(Duration::from_secs(30)),
            )
            .await?;
        if !wipe.succeeded() {
            return Err(Error::sandbox(format!(
                "wiping the workspace failed: {}",
                wipe.stderr.trim()
            )));
        }
        // Everything but pid 1 and the cleanup itself. Excluding `$$` and
        // `$PPID` is not a detail: this shell is in `/proc` too, and a list that
        // includes it makes the cleanup kill itself, exit non-zero, and cost the
        // container the pool was recycling - `ReusePolicy::Workspace` then
        // silently degrades to creating a fresh container every episode.
        let kill = self
            .exec_inner(
                ExecRequest::new([
                    "sh",
                    "-c",
                    "for p in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$'); do \
                     [ \"$p\" = 1 ] && continue; \
                     [ \"$p\" = \"$$\" ] && continue; \
                     [ \"$p\" = \"$PPID\" ] && continue; \
                     kill -9 \"$p\" 2>/dev/null; \
                     done; exit 0",
                ])
                .with_timeout(Duration::from_secs(10)),
            )
            .await?;
        if !kill.succeeded() {
            return Err(Error::sandbox("killing leftover processes failed"));
        }
        Ok(())
    }

    /// Refuses a custom image before it enters the pool when the utilities the
    /// sandbox itself relies on are absent. This checks capabilities, not a
    /// distro label: a non-Debian image is fine when it provides those capabilities.
    async fn check_capabilities(&self) -> Result<()> {
        let mut required = "sh find grep ls cat kill mkdir".to_string();
        if self.timeout_wrapper {
            required.push_str(" timeout");
        }
        let script = format!(
            "for command in {required}; do command -v \"$command\" >/dev/null || {{ echo missing:$command >&2; exit 1; }}; done; \
             find /work -mindepth 1 -maxdepth 1 -printf '' >/dev/null"
        );
        let checked = self
            .exec_inner(
                ExecRequest::new(["sh", "-c", script.as_str()])
                    .with_timeout(Duration::from_secs(10)),
            )
            .await?;
        if checked.succeeded() {
            Ok(())
        } else {
            Err(Error::sandbox(format!(
                "container image lacks required sandbox utilities: {}",
                checked.stderr.trim()
            )))
        }
    }

    /// `exec` without the metrics and without the timeout wrapper, for our own
    /// housekeeping commands.
    async fn exec_inner(&self, request: ExecRequest) -> Result<ExecOutput> {
        self.run(request, false).await
    }

    /// An exec whose stdout is collected byte for byte, for file transfers.
    ///
    /// Files move through an exec and not through the daemon's copy API, in
    /// either direction, because that API resolves the path against the
    /// container's rootfs *on the host* and therefore cannot see a tmpfs mount,
    /// which is exactly what the workspace is. An upload into a read-only rootfs
    /// is refused outright ("container rootfs is marked read-only"), and a
    /// download silently reads past the mount instead of through it. An exec
    /// runs inside the container's mount namespace, where `/work` is the tmpfs,
    /// and as the container's own user, so what it creates belongs to the
    /// episode rather than to root.
    ///
    /// [`Self::run`] cannot serve for the download half: it decodes output as
    /// lossy UTF-8, which rewrites any file that is not text.
    async fn exec_bytes(&self, argv: Vec<String>, cap: usize) -> Result<RawExec> {
        let config = CreateExecOptions {
            cmd: Some(argv),
            working_dir: Some(WORKDIR.to_string()),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            tty: Some(false),
            ..Default::default()
        };
        let exec = self
            .client
            .docker()
            .create_exec(&self.id, config)
            .await
            .map_err(|error| {
                self.poison();
                docker_error("create an exec in the sandbox", error)
            })?;
        let started = self
            .client
            .docker()
            .start_exec(&exec.id, Some(StartExecOptions::default()))
            .await
            .map_err(|error| {
                self.poison();
                docker_error("start an exec in the sandbox", error)
            })?;
        let StartExecResults::Attached { mut output, .. } = started else {
            self.poison();
            return Err(Error::sandbox("the daemon detached an attached exec"));
        };

        let collect = async {
            let mut stdout = Vec::new();
            let mut stderr = String::new();
            let mut overflowed = false;
            while let Some(chunk) = output.next().await {
                let chunk = chunk.map_err(|error| docker_error("read exec output", error))?;
                match chunk {
                    LogOutput::StdOut { message } | LogOutput::Console { message } => {
                        // Bounded as we read, and the rest of the stream is
                        // drained rather than dropped: breaking here would kill
                        // `cat` with a broken pipe and lose the exit code.
                        if stdout.len() + message.len() <= cap {
                            stdout.extend_from_slice(&message);
                        } else {
                            overflowed = true;
                        }
                    }
                    LogOutput::StdErr { message } => {
                        if stderr.len() < 4096 {
                            stderr.push_str(&String::from_utf8_lossy(&message));
                        }
                    }
                    LogOutput::StdIn { .. } => continue,
                }
            }
            Ok::<_, Error>((stdout, stderr, overflowed))
        };

        let watchdog = self.limits.exec_timeout + WATCHDOG_GRACE;
        let (stdout, stderr, overflowed) = match tokio::time::timeout(watchdog, collect).await {
            Ok(collected) => collected.inspect_err(|_| self.poison())?,
            Err(_) => {
                self.poison();
                return Err(Error::sandbox("reading a file from the sandbox timed out"));
            }
        };
        let inspected = self
            .client
            .docker()
            .inspect_exec(&exec.id)
            .await
            .map_err(|error| {
                self.poison();
                docker_error("inspect an exec in the sandbox", error)
            })?;
        Ok(RawExec {
            exit_code: exit_code(inspected.exit_code),
            stdout,
            stderr,
            overflowed,
        })
    }

    async fn run(&self, request: ExecRequest, wrap: bool) -> Result<ExecOutput> {
        if request.argv.is_empty() {
            return Err(Error::invalid("exec requires a non-empty argv"));
        }
        let cwd = match &request.cwd {
            Some(cwd) => absolute(cwd)?,
            None => WORKDIR.to_string(),
        };
        let argv = wrapped_argv(&request.argv, request.timeout, wrap && self.timeout_wrapper);
        if wrap {
            // Only the policy's own commands are counted: our housekeeping
            // would otherwise dilute `env/exec_timeout_fraction`.
            self.metrics.record_exec();
        }
        let config = CreateExecOptions {
            cmd: Some(argv),
            env: Some(
                request
                    .env
                    .iter()
                    .map(|(key, value)| format!("{key}={value}"))
                    .collect(),
            ),
            working_dir: Some(cwd),
            attach_stdout: Some(true),
            attach_stderr: Some(true),
            attach_stdin: Some(request.stdin.is_some()),
            tty: Some(false),
            ..Default::default()
        };
        let exec = self
            .client
            .docker()
            .create_exec(&self.id, config)
            .await
            .map_err(|error| {
                self.poison();
                docker_error("create an exec in the sandbox", error)
            })?;

        let started = self
            .client
            .docker()
            .start_exec(&exec.id, Some(StartExecOptions::default()))
            .await
            .map_err(|error| {
                self.poison();
                docker_error("start an exec in the sandbox", error)
            })?;
        let StartExecResults::Attached {
            mut output,
            mut input,
        } = started
        else {
            self.poison();
            return Err(Error::sandbox("the daemon detached an attached exec"));
        };

        if let Some(stdin) = request.stdin.as_ref() {
            input.write_all(stdin).await.ok();
            input.shutdown().await.ok();
        }
        drop(input);

        let max_output_bytes = request.max_output_bytes.min(self.limits.max_output_bytes);
        let collect = async {
            let mut stdout = String::new();
            let mut stderr = String::new();
            let mut overflowed = false;
            while let Some(chunk) = output.next().await {
                let chunk = chunk.map_err(|error| docker_error("read exec output", error))?;
                let (sink, bytes) = match chunk {
                    LogOutput::StdOut { message } => (&mut stdout, message),
                    LogOutput::StdErr { message } => (&mut stderr, message),
                    LogOutput::Console { message } => (&mut stdout, message),
                    LogOutput::StdIn { .. } => continue,
                };
                // Bounded as we read: a command that prints forever must not be
                // able to grow our heap before we cut it.
                if sink.len() < max_output_bytes.saturating_mul(2) {
                    sink.push_str(&String::from_utf8_lossy(&bytes));
                } else {
                    overflowed = true;
                }
            }
            Ok::<_, Error>((stdout, stderr, overflowed))
        };

        let watchdog = request.timeout + WATCHDOG_GRACE;
        let (mut stdout, mut stderr, overflowed) =
            match tokio::time::timeout(watchdog, collect).await {
                Ok(collected) => collected.inspect_err(|_| self.poison())?,
                Err(_) => {
                    // The in-container kill did not happen. We no longer know
                    // what is running in there, so the container does not get
                    // reused - see the pool's "when in doubt, destroy".
                    self.poison();
                    self.metrics.record_timeout();
                    return Ok(ExecOutput {
                        exit_code: None,
                        stdout: String::new(),
                        stderr: timeout_message(request.timeout),
                        timed_out: true,
                        truncated: false,
                    });
                }
            };

        let inspected = self
            .client
            .docker()
            .inspect_exec(&exec.id)
            .await
            .map_err(|error| {
                self.poison();
                docker_error("inspect an exec in the sandbox", error)
            })?;
        let exit_code = exit_code(inspected.exit_code);
        // 124 is what coreutils `timeout` exits with when it fires. Turning it
        // back into a timeout is what makes the in-container kill invisible to
        // the caller, who sees the same observation either way.
        let timed_out = wrap && self.timeout_wrapper && exit_code == Some(124);
        if timed_out {
            self.metrics.record_timeout();
            stdout.clear();
            stderr = timeout_message(request.timeout);
        }

        let before = stdout.len() + stderr.len();
        truncate_utf8(&mut stdout, max_output_bytes, TRUNCATION_MARKER);
        truncate_utf8(&mut stderr, max_output_bytes, TRUNCATION_MARKER);
        Ok(ExecOutput {
            exit_code: if timed_out { None } else { exit_code },
            truncated: overflowed || stdout.len() + stderr.len() < before,
            stdout,
            stderr,
            timed_out,
        })
    }
}

/// The one wording a timed-out command ever produces. Fixed, and without the
/// elapsed time: a duration differs between two identical actions, and the
/// difference would be trained on.
fn timeout_message(timeout: Duration) -> String {
    format!("command timed out after {} seconds", timeout.as_secs())
}

/// The first non-empty line of a command's stderr, for a one-line tool error.
fn first_line(stderr: &str) -> Option<&str> {
    stderr.lines().map(str::trim).find(|line| !line.is_empty())
}

/// Absolute in-container path for a tool-supplied path.
fn absolute(path: &str) -> Result<String> {
    let relative = relative_path(WORKDIR, path)?;
    Ok(if relative.is_empty() {
        WORKDIR.to_string()
    } else {
        format!("{WORKDIR}/{relative}")
    })
}

/// Wraps argv so the kill happens inside the container.
///
/// Without this we can only stop *waiting*; the process keeps running, keeps its
/// cpu share and is still there when the container is recycled. `timeout` exits
/// 124 when it fires, which is how the observation gets its `timed_out` flag
/// without us having to guess.
fn wrapped_argv(argv: &[String], timeout: Duration, wrap: bool) -> Vec<String> {
    if !wrap {
        return argv.to_vec();
    }
    let mut wrapped = vec![
        "timeout".to_string(),
        "-k".to_string(),
        "5".to_string(),
        timeout.as_secs().max(1).to_string(),
    ];
    wrapped.extend(argv.iter().cloned());
    wrapped
}

#[async_trait]
impl Sandbox for ContainerSandbox {
    async fn exec(&self, request: ExecRequest) -> Result<ExecOutput> {
        self.run(request, true).await
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let absolute = absolute(path)?;
        // `cat`, not the daemon's copy API: see [`ContainerSandbox::exec_bytes`].
        let read = self
            .exec_bytes(vec!["cat".to_string(), absolute], MAX_READ_BYTES)
            .await?;
        if read.exit_code != Some(0) {
            // A failed read is the policy's mistake and becomes an observation;
            // the container is fine, the path is not.
            let reason = if read.stderr.contains("No such file") {
                "no such file"
            } else if read.stderr.contains("Is a directory") {
                "is a directory"
            } else if read.stderr.contains("Permission denied") {
                "permission denied"
            } else {
                "cannot be read"
            };
            return Err(Error::Tool(format!("read '{path}': {reason}")));
        }
        if read.overflowed {
            return Err(Error::Tool(format!(
                "read '{path}': larger than {MAX_READ_BYTES} bytes"
            )));
        }
        Ok(read.stdout)
    }

    async fn write_file(&self, path: &str, bytes: &[u8]) -> Result<()> {
        let relative = relative_path(WORKDIR, path)?;
        if relative.is_empty() {
            return Err(Error::invalid("cannot write to the workspace root itself"));
        }
        // A tool writing `src/a.py` into an empty workspace is the normal case,
        // and a redirection does not create the directory it writes into.
        if let Some((parent, _)) = relative.rsplit_once('/') {
            let created = self
                .exec_inner(ExecRequest::new([
                    "mkdir",
                    "-p",
                    &format!("{WORKDIR}/{parent}"),
                ]))
                .await?;
            if !created.succeeded() {
                return Err(Error::Tool(format!(
                    "write '{path}': cannot create parent directory"
                )));
            }
        }

        // Through an exec rather than an upload: see
        // [`ContainerSandbox::exec_bytes`]. The path is a separate argument, so
        // a model-supplied name is data to the shell and never script.
        let written = self
            .exec_inner(
                ExecRequest::new([
                    "sh",
                    "-c",
                    "cat > \"$1\"",
                    "sh",
                    &format!("{WORKDIR}/{relative}"),
                ])
                .with_stdin(bytes.to_vec()),
            )
            .await?;
        if !written.succeeded() {
            return Err(Error::Tool(format!(
                "write '{path}': {}",
                first_line(&written.stderr).unwrap_or("the write failed")
            )));
        }
        Ok(())
    }

    async fn list_dir(&self, path: &str) -> Result<Vec<DirEntry>> {
        let absolute = absolute(path)?;
        // `-printf` is GNU findutils, which every Debian-based image has. It is
        // used rather than parsing `ls` because the fields are unambiguous, and
        // rather than downloading the directory as a tar because that would pull
        // the whole content over the socket just to name it.
        let listing = self
            .exec_inner(ExecRequest::new([
                "find",
                &absolute,
                "-mindepth",
                "1",
                "-maxdepth",
                "1",
                "-printf",
                "%y\\t%s\\t%P\\n",
            ]))
            .await?;
        if !listing.succeeded() {
            return Err(Error::Tool(format!(
                "list '{path}': {}",
                listing.stderr.trim()
            )));
        }
        let mut entries = Vec::new();
        for line in listing.stdout.lines() {
            let mut fields = line.splitn(3, '\t');
            let (Some(kind), Some(size), Some(name)) =
                (fields.next(), fields.next(), fields.next())
            else {
                continue;
            };
            entries.push(DirEntry {
                path: name.to_string(),
                kind: match kind {
                    "d" => FileKind::Dir,
                    "l" => FileKind::Symlink,
                    _ => FileKind::File,
                },
                size: size.parse().unwrap_or(0),
            });
        }
        // `find` walks in filesystem order, which is not a property of the
        // action the policy took.
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(entries)
    }

    async fn remove(&self, path: &str) -> Result<()> {
        let absolute = absolute(path)?;
        if absolute == WORKDIR {
            return Err(Error::invalid("cannot remove the workspace root itself"));
        }
        let removed = self
            .exec_inner(ExecRequest::new(["rm", "-rf", "--", &absolute]))
            .await?;
        if removed.succeeded() {
            Ok(())
        } else {
            Err(Error::Tool(format!(
                "remove '{path}': {}",
                removed.stderr.trim()
            )))
        }
    }

    fn workdir(&self) -> &str {
        WORKDIR
    }

    fn limits(&self) -> SandboxLimits {
        self.limits
    }
}

/// Creates containers from one [`ContainerSpec`], for the pool to manage.
///
/// The image is resolved once, at construction: pulling on the first rollout
/// would put a multi-gigabyte download inside a trajectory's deadline.
pub struct ContainerSource {
    client: DockerClient,
    spec: ContainerSpec,
    /// Digest-pinned, so every container of the run is the same environment.
    image: String,
    labels: RunLabels,
    metrics: Arc<SandboxMetrics>,
}

impl ContainerSource {
    pub async fn new(
        client: DockerClient,
        spec: ContainerSpec,
        run: impl Into<String>,
    ) -> Result<Self> {
        Self::with_metrics(client, spec, run, Arc::new(SandboxMetrics::default())).await
    }

    pub async fn with_metrics(
        client: DockerClient,
        spec: ContainerSpec,
        run: impl Into<String>,
        metrics: Arc<SandboxMetrics>,
    ) -> Result<Self> {
        spec.validate()?;
        let image = crate::ensure_image(&client, &spec.image).await?;
        let labels = RunLabels::new(run, spec.fingerprint());
        Ok(Self {
            client,
            spec,
            image,
            labels,
            metrics,
        })
    }

    /// The digest every container of this source runs. Belongs in the run
    /// metadata: it is the only thing that makes the environment reproducible.
    pub fn image(&self) -> &str {
        &self.image
    }

    pub fn labels(&self) -> &RunLabels {
        &self.labels
    }
}

#[async_trait]
impl SandboxSource for ContainerSource {
    async fn create(&self) -> Result<Arc<dyn ManagedSandbox>> {
        let body = self
            .spec
            .to_create_body(&self.image, self.labels.to_map().into_iter().collect());
        let created = self
            .client
            .docker()
            .create_container(None::<CreateContainerOptions>, body)
            .await
            .map_err(|error| docker_error("create a sandbox container", error))?;
        self.client
            .docker()
            .start_container(&created.id, None::<StartContainerOptions>)
            .await
            .map_err(|error| docker_error("start a sandbox container", error))?;
        let sandbox = Arc::new(ContainerSandbox::new(
            self.client.clone(),
            created.id,
            self.spec.limits(),
            self.spec.timeout_wrapper,
            self.metrics.clone(),
        ));
        if let Err(error) = sandbox.check_capabilities().await {
            sandbox.destroy().await;
            return Err(error);
        }
        Ok(Arc::new(ContainerInstance { sandbox }))
    }

    /// Every container carrying this run's label, leased or idle, gone.
    ///
    /// Going through the daemon rather than through the pool's bookkeeping is
    /// the point: the containers that matter here are the ones leased to
    /// rollouts that are being abandoned, and the label is the only view that
    /// includes them.
    async fn force_cleanup(&self) -> usize {
        match crate::reaper::reap_run(&self.client, &self.labels.run).await {
            Ok(removed) => removed,
            Err(error) => {
                tracing::warn!(%error, "could not list this run's containers to remove them");
                0
            }
        }
    }
}

struct ContainerInstance {
    sandbox: Arc<ContainerSandbox>,
}

#[async_trait]
impl ManagedSandbox for ContainerInstance {
    fn sandbox(&self) -> Arc<dyn Sandbox> {
        self.sandbox.clone()
    }

    async fn recycle(&self) -> Result<()> {
        self.sandbox.reset_workspace().await
    }

    async fn destroy(&self) {
        self.sandbox.destroy().await;
    }

    fn healthy(&self) -> bool {
        self.sandbox.is_healthy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// These cover the pure halves - path handling and fixed wordings - which
    /// are exactly the parts that must hold without a daemon.
    #[test]
    fn paths_resolve_under_the_shared_workdir_and_cannot_leave_it() {
        assert_eq!(absolute("src/a.py").unwrap(), "/work/src/a.py");
        assert_eq!(absolute("/work/src/a.py").unwrap(), "/work/src/a.py");
        assert_eq!(absolute(".").unwrap(), "/work");
        assert!(absolute("../etc/passwd").is_err());
    }

    /// The kill has to happen inside the container: stopping our own wait leaves
    /// the process running, holding cpu and waiting to pollute the next episode.
    #[test]
    fn a_command_is_wrapped_so_the_kill_happens_inside() {
        let argv = vec!["pytest".to_string(), "-q".to_string()];
        assert_eq!(
            wrapped_argv(&argv, Duration::from_secs(30), true),
            ["timeout", "-k", "5", "30", "pytest", "-q"]
        );
        assert_eq!(wrapped_argv(&argv, Duration::from_secs(30), false), argv);
    }

    /// A duration in an observation differs between two identical actions, so
    /// the message carries the *budget*, which is configuration, and never the
    /// elapsed time.
    #[test]
    fn the_timeout_wording_is_fixed() {
        assert_eq!(
            timeout_message(Duration::from_secs(30)),
            "command timed out after 30 seconds"
        );
        assert_eq!(
            timeout_message(Duration::from_secs(30)),
            timeout_message(Duration::from_secs(30))
        );
    }
}
