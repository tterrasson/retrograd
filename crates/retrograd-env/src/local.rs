//! A sandbox that is not one.
//!
//! [`LocalSandbox`] runs commands straight on the host, in a temporary
//! directory. It exists for two reasons and no others:
//!
//! 1. it makes session tools testable in the fast lane, with no daemon and no
//!    image - without it, every tool would only be exercisable in the container
//!    lane;
//! 2. some tasks are pure computation and their operator knows it.
//!
//! It is **not** an isolation boundary. Model-generated code runs with the
//! privileges of the training process, on the training machine, with the
//! network the training process has. Hence [`LocalSandboxConfig::allow_unsandboxed`]:
//! this must be something someone typed, never something obtained by omission.
//!
//! One honest limitation, next to the determinism rule of
//! [`retrograd_agent_core::sandbox`]: paths inside the sandbox are reported
//! relative to a logical root, but a command that prints its own absolute
//! working directory (`pwd`, a compiler diagnostic, a stack trace) will leak the
//! temporary directory's real name, which differs per episode. Those bytes end
//! up in the trajectory. A container gives every episode the *same* `/work`,
//! which is what actually fixes this.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, Weak};

use async_trait::async_trait;
use retrograd_agent_core::text::truncate_utf8;
use retrograd_agent_core::{
    DirEntry, Error, ExecOutput, ExecRequest, FileKind, Lease, Result, Sandbox, SandboxLimits,
    SandboxOwner, SandboxProvider, relative_path,
};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWriteExt};

/// The path every sandbox reports, whatever the host directory really is.
pub const LOGICAL_ROOT: &str = "/work";

#[derive(Clone, Debug)]
pub struct LocalSandboxConfig {
    /// Must be `true`. There is no default that turns this on: running
    /// model-generated code unconfined is a decision, not a fallback.
    pub allow_unsandboxed: bool,
    pub limits: SandboxLimits,
}

impl Default for LocalSandboxConfig {
    fn default() -> Self {
        Self {
            allow_unsandboxed: false,
            limits: SandboxLimits {
                // The host has a network whether we like it or not; saying
                // otherwise would make tools word their errors from a lie.
                network: true,
                ..SandboxLimits::default()
            },
        }
    }
}

#[derive(Debug)]
pub struct LocalSandbox {
    root: tempfile::TempDir,
    limits: SandboxLimits,
}

impl LocalSandbox {
    pub fn new(config: LocalSandboxConfig) -> Result<Self> {
        if !config.allow_unsandboxed {
            return Err(Error::invalid(
                "a local sandbox executes model-generated code on the host with no isolation; \
                 set allow_unsandboxed to run without a container",
            ));
        }
        tracing::warn!(
            "using an unsandboxed local execution environment: tool calls run on this machine"
        );
        let root = tempfile::Builder::new()
            .prefix("retrograd-sandbox-")
            .tempdir()
            .map_err(|error| Error::Tool(format!("create local sandbox directory: {error}")))?;
        Ok(Self {
            root,
            limits: config.limits,
        })
    }

    /// Resolves a sandbox path against the real root. The rule itself lives in
    /// `retrograd-agent-core` because every sandbox owes the model the same
    /// answer for the same path.
    fn resolve(&self, path: &str) -> Result<PathBuf> {
        Ok(self.root.path().join(relative_path(LOGICAL_ROOT, path)?))
    }

    /// Renders a host path back as a sandbox path. Never lets the temporary
    /// directory's real name reach an observation.
    fn logical(&self, path: &Path) -> String {
        path.strip_prefix(self.root.path())
            .map(|rest| rest.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| String::new())
    }
}

async fn read_limited<R: AsyncRead + Unpin>(
    mut reader: R,
    max_bytes: usize,
) -> std::io::Result<(Vec<u8>, bool)> {
    let mut kept = Vec::with_capacity(max_bytes.min(64 * 1024));
    let mut chunk = [0_u8; 8192];
    let mut truncated = false;
    loop {
        let read = reader.read(&mut chunk).await?;
        if read == 0 {
            break;
        }
        let remaining = max_bytes.saturating_sub(kept.len());
        let copy = read.min(remaining);
        kept.extend_from_slice(&chunk[..copy]);
        truncated |= copy < read;
    }
    Ok((kept, truncated))
}

#[cfg(unix)]
fn kill_process_group(pid: Option<u32>) {
    if let Some(pid) = pid.and_then(|pid| i32::try_from(pid).ok()) {
        // SAFETY: the child was placed in a process group whose id is its pid;
        // a negative pid addresses that group. Failure only means it exited.
        unsafe {
            libc::kill(-pid, libc::SIGKILL);
        }
    }
}

#[cfg(not(unix))]
fn kill_process_group(_pid: Option<u32>) {}

#[async_trait]
impl Sandbox for LocalSandbox {
    async fn exec(&self, request: ExecRequest) -> Result<ExecOutput> {
        let Some((program, arguments)) = request.argv.split_first() else {
            return Err(Error::invalid("exec requires a non-empty argv"));
        };
        let cwd = match &request.cwd {
            Some(cwd) => self.resolve(cwd)?,
            None => self.root.path().to_path_buf(),
        };
        let mut command = tokio::process::Command::new(program);
        command
            .args(arguments)
            .current_dir(&cwd)
            .envs(&request.env)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        command.process_group(0);
        // A program that does not exist is the policy's mistake, not a broken
        // sandbox: it comes back as an observation to react to, worded and coded
        // the way a shell in a container words it - 127 and a fixed sentence,
        // never the host's own error text, which differs between machines and
        // would be trained on.
        let mut child = match command.spawn() {
            Ok(child) => child,
            Err(error) => {
                let (code, reason) = if error.kind() == std::io::ErrorKind::NotFound {
                    (127, "command not found")
                } else {
                    (126, "cannot execute")
                };
                return Ok(ExecOutput {
                    exit_code: Some(code),
                    stdout: String::new(),
                    stderr: format!("{program}: {reason}"),
                    timed_out: false,
                    truncated: false,
                });
            }
        };
        let stdin_task = child.stdin.take().map(|mut pipe| {
            let input = request.stdin.unwrap_or_default();
            tokio::spawn(async move {
                let result = pipe.write_all(&input).await;
                let _ = pipe.shutdown().await;
                result
            })
        });
        let max_output_bytes = request.max_output_bytes.min(self.limits.max_output_bytes);
        let stdout_task = tokio::spawn(read_limited(
            child.stdout.take().expect("piped child stdout"),
            max_output_bytes,
        ));
        let stderr_task = tokio::spawn(read_limited(
            child.stderr.take().expect("piped child stderr"),
            max_output_bytes,
        ));
        let pid = child.id();
        let deadline = tokio::time::Instant::now() + request.timeout;
        let status = match tokio::time::timeout_at(deadline, child.wait()).await {
            Ok(status) => Some(
                status.map_err(|error| Error::sandbox(format!("wait for '{program}': {error}")))?,
            ),
            Err(_) => {
                kill_process_group(pid);
                let _ = child.kill().await;
                let _ = child.wait().await;
                None
            }
        };
        let collected = tokio::time::timeout_at(deadline, async {
            let stdout = stdout_task
                .await
                .map_err(|error| Error::sandbox(format!("join stdout reader: {error}")))?
                .map_err(|error| Error::sandbox(format!("read stdout: {error}")))?;
            let stderr = stderr_task
                .await
                .map_err(|error| Error::sandbox(format!("join stderr reader: {error}")))?
                .map_err(|error| Error::sandbox(format!("read stderr: {error}")))?;
            if let Some(task) = stdin_task {
                let _ = task.await;
            }
            Ok::<_, Error>((stdout, stderr))
        })
        .await;
        let ((stdout_bytes, stdout_cut), (stderr_bytes, stderr_cut)) = match collected {
            Ok(result) => result?,
            Err(_) => {
                kill_process_group(pid);
                return Ok(ExecOutput {
                    exit_code: None,
                    stdout: String::new(),
                    stderr: format!(
                        "command timed out after {} seconds",
                        request.timeout.as_secs()
                    ),
                    timed_out: true,
                    truncated: false,
                });
            }
        };
        if status.is_none() {
            return Ok(ExecOutput {
                exit_code: None,
                stdout: String::new(),
                stderr: format!(
                    "command timed out after {} seconds",
                    request.timeout.as_secs()
                ),
                timed_out: true,
                truncated: stdout_cut || stderr_cut,
            });
        }

        let mut stdout = String::from_utf8_lossy(&stdout_bytes).into_owned();
        let mut stderr = String::from_utf8_lossy(&stderr_bytes).into_owned();
        if stdout_cut {
            stdout.push(' ');
        }
        if stderr_cut {
            stderr.push(' ');
        }
        let before = stdout.len() + stderr.len();
        truncate_utf8(&mut stdout, max_output_bytes, "\n[output truncated]");
        truncate_utf8(&mut stderr, max_output_bytes, "\n[output truncated]");
        let truncated = stdout_cut || stderr_cut || stdout.len() + stderr.len() < before;
        Ok(ExecOutput {
            exit_code: status.and_then(|status| status.code()),
            truncated,
            stdout,
            stderr,
            timed_out: false,
        })
    }

    async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let resolved = self.resolve(path)?;
        tokio::fs::read(&resolved)
            .await
            .map_err(|error| fs_failure("read", path, &error))
    }

    async fn write_file(&self, path: &str, bytes: &[u8]) -> Result<()> {
        let resolved = self.resolve(path)?;
        if let Some(parent) = resolved.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|error| fs_failure("create the parent of", path, &error))?;
        }
        tokio::fs::write(&resolved, bytes)
            .await
            .map_err(|error| fs_failure("write", path, &error))
    }

    async fn list_dir(&self, path: &str) -> Result<Vec<DirEntry>> {
        let resolved = self.resolve(path)?;
        let mut reader = tokio::fs::read_dir(&resolved)
            .await
            .map_err(|error| fs_failure("list", path, &error))?;
        let mut entries = Vec::new();
        while let Some(entry) = reader
            .next_entry()
            .await
            .map_err(|error| fs_failure("list", path, &error))?
        {
            let metadata = entry
                .metadata()
                .await
                .map_err(|error| fs_failure("stat in", path, &error))?;
            entries.push(DirEntry {
                path: self.logical(&entry.path()),
                kind: if metadata.is_dir() {
                    FileKind::Dir
                } else if metadata.is_symlink() {
                    FileKind::Symlink
                } else {
                    FileKind::File
                },
                size: metadata.len(),
            });
        }
        // Filesystem iteration order is not a property of the action, and the
        // listing becomes training tokens.
        entries.sort_by(|left, right| left.path.cmp(&right.path));
        Ok(entries)
    }

    async fn remove(&self, path: &str) -> Result<()> {
        let resolved = self.resolve(path)?;
        let metadata = tokio::fs::symlink_metadata(&resolved)
            .await
            .map_err(|error| Error::Tool(format!("remove '{path}': {}", error.kind())))?;
        let removal = if metadata.is_dir() {
            tokio::fs::remove_dir_all(&resolved).await
        } else {
            tokio::fs::remove_file(&resolved).await
        };
        removal.map_err(|error| Error::Tool(format!("remove '{path}': {}", error.kind())))
    }

    fn workdir(&self) -> &str {
        LOGICAL_ROOT
    }

    fn limits(&self) -> SandboxLimits {
        self.limits
    }
}

/// A filesystem failure as the policy reads it: the action, the path as the
/// policy named it, and the error kind.
fn fs_failure(action: &str, path: &str, error: &std::io::Error) -> Error {
    Error::Tool(format!("{action} '{path}': {}", error.kind()))
}

/// One fresh [`LocalSandbox`] per episode, and no pool.
///
/// There is nothing to reuse: a temporary directory costs a `mkdir`, not a
/// container start, and creating one per episode buys the isolation a pool has
/// to work for. Releasing a lease drops the directory, which deletes it.
pub struct LocalSandboxProvider {
    me: Weak<Self>,
    config: LocalSandboxConfig,
    live: Mutex<HashMap<u64, Arc<dyn Sandbox>>>,
    next_token: Mutex<u64>,
}

impl LocalSandboxProvider {
    /// Refuses an unconfined provider here rather than at the first
    /// acquisition: the decision belongs to whoever loaded the configuration,
    /// and they are still on the stack.
    pub fn new(config: LocalSandboxConfig) -> Result<Arc<Self>> {
        if !config.allow_unsandboxed {
            return Err(Error::invalid(
                "a local sandbox executes model-generated code on the host with no isolation; \
                 set allow_unsandboxed to run without a container",
            ));
        }
        Ok(Arc::new_cyclic(|me| Self {
            me: me.clone(),
            config,
            live: Mutex::new(HashMap::new()),
            next_token: Mutex::new(0),
        }))
    }
}

#[async_trait]
impl SandboxProvider for LocalSandboxProvider {
    async fn acquire(&self) -> Result<Lease> {
        let sandbox: Arc<dyn Sandbox> = Arc::new(LocalSandbox::new(self.config.clone())?);
        let token = {
            let mut next = self
                .next_token
                .lock()
                .expect("local sandbox token counter lock");
            *next += 1;
            *next
        };
        self.live
            .lock()
            .expect("local sandbox registry lock")
            .insert(token, sandbox.clone());
        let owner = self
            .me
            .upgrade()
            .ok_or_else(|| Error::sandbox("local sandbox provider was dropped"))?;
        Ok(Lease::new(sandbox, owner, token))
    }

    async fn shutdown(&self) {
        self.live
            .lock()
            .expect("local sandbox registry lock")
            .clear();
    }
}

impl SandboxOwner for LocalSandboxProvider {
    /// `healthy` is irrelevant here: nothing is ever reused, so the answer to
    /// both cases is the same - drop the sandbox, which removes its directory.
    fn release(&self, token: u64, _healthy: bool) {
        self.live
            .lock()
            .expect("local sandbox registry lock")
            .remove(&token);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    #[test]
    fn a_filesystem_failure_names_the_action_the_path_and_the_kind() {
        let io = std::io::Error::from(std::io::ErrorKind::NotFound);
        let error = fs_failure("read", "notes.txt", &io);
        assert!(matches!(error, Error::Tool(_)), "{error:?}");
        assert_eq!(
            error.to_string(),
            "tool error: read 'notes.txt': entity not found"
        );
    }

    use super::*;

    fn sandbox() -> LocalSandbox {
        LocalSandbox::new(LocalSandboxConfig {
            allow_unsandboxed: true,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn running_unconfined_is_never_something_you_get_by_omission() {
        let error = LocalSandbox::new(LocalSandboxConfig::default()).unwrap_err();
        assert!(error.to_string().contains("allow_unsandboxed"), "{error}");
    }

    #[tokio::test]
    async fn files_round_trip_and_listings_are_sorted() {
        let sandbox = sandbox();
        for name in ["c.txt", "a.txt", "b.txt"] {
            sandbox.write_file(name, name.as_bytes()).await.unwrap();
        }
        sandbox
            .write_file("nested/deep.txt", b"deep")
            .await
            .unwrap();
        assert_eq!(sandbox.read_file("a.txt").await.unwrap(), b"a.txt");

        let entries = sandbox.list_dir(".").await.unwrap();
        let paths = entries
            .iter()
            .map(|entry| entry.path.as_str())
            .collect::<Vec<_>>();
        assert_eq!(paths, ["a.txt", "b.txt", "c.txt", "nested"]);
        // No host path ever reaches an observation.
        assert!(entries.iter().all(|entry| !entry.path.starts_with('/')));

        sandbox.remove("nested").await.unwrap();
        assert!(sandbox.read_file("nested/deep.txt").await.is_err());
    }

    /// Paths come from model output. `..` and absolute paths are the whole
    /// attack, and they are refused before they touch the filesystem - including
    /// for a file that does not exist yet, which canonicalizing could not catch.
    #[tokio::test]
    async fn a_path_cannot_escape_the_sandbox_root() {
        let sandbox = sandbox();
        for path in ["../escape.txt", "a/../../escape.txt", "/etc/passwd"] {
            let error = sandbox.write_file(path, b"x").await.unwrap_err();
            assert!(error.to_string().contains("escapes"), "{path}: {error}");
        }
        // The logical root is a prefix the tools may legitimately spell.
        sandbox.write_file("/work/inside.txt", b"x").await.unwrap();
        assert_eq!(sandbox.read_file("inside.txt").await.unwrap(), b"x");
    }

    #[tokio::test]
    async fn a_failing_command_is_an_observation_and_a_hanging_one_is_bounded() {
        let sandbox = sandbox();
        let failure = sandbox
            .exec(ExecRequest::new([
                "sh",
                "-c",
                "echo out; echo err >&2; exit 3",
            ]))
            .await
            .expect("a non-zero exit is not a sandbox failure");
        assert_eq!(failure.exit_code, Some(3));
        assert_eq!(failure.stdout.trim(), "out");
        assert_eq!(failure.stderr.trim(), "err");
        assert!(!failure.succeeded());

        let timed_out = sandbox
            .exec(ExecRequest::new(["sleep", "30"]).with_timeout(Duration::from_millis(50)))
            .await
            .unwrap();
        assert!(timed_out.timed_out && timed_out.exit_code.is_none());

        // A command that does not exist reads like it does in a container, and
        // never like this machine's `std::io::Error` text.
        let missing = sandbox
            .exec(ExecRequest::new(["definitely-not-a-program"]))
            .await
            .expect("an unknown command is an observation");
        assert_eq!(missing.exit_code, Some(127));
        assert_eq!(
            missing.stderr,
            "definitely-not-a-program: command not found"
        );
    }

    /// Two episodes must not see each other's files. Here it is free - a new
    /// directory per lease - and the test exists because the container pool's
    /// version of this property is the one protecting the GRPO baseline.
    #[tokio::test]
    async fn two_leases_never_share_a_workspace() {
        let provider = LocalSandboxProvider::new(LocalSandboxConfig {
            allow_unsandboxed: true,
            ..Default::default()
        })
        .unwrap();
        let first = provider.acquire().await.unwrap();
        first.write_file("marker.txt", b"x").await.unwrap();
        drop(first);

        let second = provider.acquire().await.unwrap();
        assert!(second.read_file("marker.txt").await.is_err());
        drop(second);
        assert!(provider.live.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn output_is_capped_on_a_character_boundary() {
        let sandbox = sandbox();
        let output = sandbox
            .exec(
                ExecRequest::new(["sh", "-c", "printf 'é%.0s' $(seq 1 500)"])
                    .with_max_output_bytes(64),
            )
            .await
            .unwrap();
        assert!(output.truncated);
        assert!(output.stdout.len() <= 64, "{}", output.stdout.len());
        assert!(output.stdout.is_char_boundary(output.stdout.len()));
    }

    #[tokio::test]
    async fn output_is_drained_while_the_process_runs_and_descendants_are_killed() {
        let sandbox = sandbox();
        let output = sandbox
            .exec(
                ExecRequest::new(["sh", "-c", "yes x | head -c 1000000"])
                    .with_max_output_bytes(128),
            )
            .await
            .unwrap();
        assert_eq!(output.exit_code, Some(0));
        assert!(output.truncated);
        assert!(output.stdout.len() <= 128);

        let timed_out = sandbox
            .exec(
                ExecRequest::new(["sh", "-c", "(sleep 0.2; touch survived-timeout) & wait"])
                    .with_timeout(Duration::from_millis(20)),
            )
            .await
            .unwrap();
        assert!(timed_out.timed_out);
        tokio::time::sleep(Duration::from_millis(300)).await;
        assert!(sandbox.read_file("survived-timeout").await.is_err());
    }
}
