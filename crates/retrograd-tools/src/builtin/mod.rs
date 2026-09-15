//! The tools a coding task is given by default.
//!
//! Every one of them is a [`SessionTool`]: it acts on the
//! sandbox of *its* trajectory and on nothing else. Nothing here knows what a
//! container is.
//!
//! # Why the wording is frozen
//!
//! A tool's output - including its error text - is tokenized into the
//! trajectory and trained on. It must therefore be a deterministic function of
//! the arguments and the sandbox state: the same bad call produces the same
//! sentence, byte for byte, in every member of every group. A message that
//! varies (a duration, a path, a retry count) teaches the model to react to
//! noise, and the group-relative baseline measures that noise instead of the
//! policy. The helpers below exist so that no tool words the same failure twice.

use std::sync::Arc;

use retrograd_agent_core::text::truncate_utf8;
use retrograd_agent_core::{ExecOutput, ExecRequest, Sandbox};

use crate::session::{SessionTool, ToolOutcome};

mod exec;
mod files;

pub use exec::{Interpreter, RunTests, Shell};
pub use files::{EditFile, Grep, ListDir, ReadFile, Submit, WriteFile};
pub use retrograd_spec::tools::Profile;

/// The tools a profile names, built.
///
/// The preset itself is a declaration and lives in `retrograd-spec` -
/// resolving it against the built-in registry is this crate's job, so the
/// method that does it is added here rather than moved with the enum.
pub trait ProfileTools {
    fn tools(self) -> Vec<Arc<dyn SessionTool>>;
}

impl ProfileTools for Profile {
    /// The tools of the profile, in a stable order.
    ///
    /// The interpreter and the test command are the only parts that differ.
    /// `python` on a `node` image would be a tool the model is told about and
    /// cannot use, so each profile advertises the one its image actually has.
    fn tools(self) -> Vec<Arc<dyn SessionTool>> {
        builtin_registry()
            .resolve(&self.tool_names())
            .expect("profile registry names are built-ins")
            .into_iter()
            .map(|(_, tool)| match tool {
                crate::RegisteredTool::Session(tool) => tool,
                crate::RegisteredTool::Shared(_) => unreachable!("profiles contain session tools"),
            })
            .collect()
    }
}

/// Constructs one built-in. A plain `fn` pointer rather than a boxed closure:
/// every built-in is a unit struct or a `Default`, so there is nothing to
/// capture.
type BuiltinCtor = fn() -> Arc<dyn SessionTool>;

/// One row of the built-in table: the registry name, and how to build it.
type BuiltinDef = (&'static str, BuiltinCtor);

struct BuiltinFactory {
    name: &'static str,
    build: BuiltinCtor,
}

impl crate::ToolFactory for BuiltinFactory {
    fn name(&self) -> &str {
        self.name
    }

    fn describe(&self) -> retrograd_agent_core::ToolSpec {
        (self.build)().spec()
    }

    fn build(
        &self,
        _params: &serde_json::Value,
    ) -> retrograd_agent_core::Result<crate::RegisteredTool> {
        Ok(crate::RegisteredTool::Session((self.build)()))
    }
}

/// Built-ins are ordinary registry entries; profiles only select their names.
pub(crate) fn builtin_registry() -> crate::ToolRegistry {
    let definitions: [BuiltinDef; 11] = [
        ("shell", || Arc::new(Shell::default())),
        ("python", || Arc::new(Interpreter::python())),
        ("node", || Arc::new(Interpreter::node())),
        ("pytest", || Arc::new(RunTests::pytest())),
        ("npm_test", || Arc::new(RunTests::npm())),
        ("read_file", || Arc::new(ReadFile)),
        ("write_file", || Arc::new(WriteFile)),
        ("edit_file", || Arc::new(EditFile)),
        ("list_dir", || Arc::new(ListDir)),
        ("grep", || Arc::new(Grep)),
        ("submit", || Arc::new(Submit)),
    ];
    let mut registry = crate::ToolRegistry::default();
    for (name, build) in definitions {
        registry
            .register(Arc::new(BuiltinFactory { name, build }))
            .expect("builtin tool definitions are valid and unique");
    }
    registry
}

/// Renders a command's result as the model will read it.
///
/// One shape for every tool that runs something, and deliberately without a
/// duration, a pid or a host path - see the module docs.
pub fn render_exec(output: &ExecOutput) -> ToolOutcome {
    let mut text = String::new();
    if output.timed_out {
        text.push_str("command timed out and was killed\n");
    }
    match output.exit_code {
        Some(code) => text.push_str(&format!("exit code: {code}\n")),
        None => text.push_str("exit code: killed by a signal\n"),
    }
    for (label, stream) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
        if !stream.trim().is_empty() {
            text.push_str(&format!("{label}:\n{}\n", stream.trim_end()));
        }
    }
    if output.truncated {
        text.push_str("[output truncated]\n");
    }
    ToolOutcome {
        content: text,
        is_error: !output.succeeded(),
        reward: None,
        done: false,
    }
}

/// An [`ExecRequest`] already bounded by the sandbox's own limits, so no tool
/// has to remember to do it.
pub fn bounded_request(sandbox: &dyn Sandbox, argv: Vec<String>) -> ExecRequest {
    let limits = sandbox.limits();
    ExecRequest::new(argv)
        .with_timeout(limits.exec_timeout)
        .with_max_output_bytes(limits.max_output_bytes)
}

/// Reads a required string argument, or produces the one sentence that says so.
pub fn string_arg(
    arguments: &serde_json::Value,
    name: &str,
) -> std::result::Result<String, ToolOutcome> {
    match arguments.get(name) {
        Some(serde_json::Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(ToolOutcome::error(format!(
            "argument '{name}' must be a string"
        ))),
        None => Err(ToolOutcome::error(format!(
            "missing required argument '{name}'"
        ))),
    }
}

/// Reads an optional list-of-strings argument. A wrong shape is refused rather
/// than ignored: silently dropping an argument the model spelled is how it
/// learns a call worked when it did not.
pub fn string_list_arg(
    arguments: &serde_json::Value,
    name: &str,
) -> std::result::Result<Vec<String>, ToolOutcome> {
    let Some(value) = arguments.get(name) else {
        return Ok(Vec::new());
    };
    let Some(items) = value.as_array() else {
        return Err(ToolOutcome::error(format!(
            "argument '{name}' must be an array of strings"
        )));
    };
    items
        .iter()
        .map(|item| {
            item.as_str().map(str::to_owned).ok_or_else(|| {
                ToolOutcome::error(format!("argument '{name}' must be an array of strings"))
            })
        })
        .collect()
}

/// Turns a *failed action* into an observation, and lets a broken world through.
///
/// The distinction is the invariant the whole stack rests on, and it cannot be
/// made by looking at the operation that failed: `read_file` answers `Err` both
/// for a file the model invented and for a container that died. Only the error
/// says which ([`Error::is_sandbox_failure`](retrograd_agent_core::Error::is_sandbox_failure)), so every tool asks here rather
/// than assuming the first case - assuming it is how a rollout keeps going, gets
/// scored and gets trained on a world that no longer exists.
///
/// ```ignore
/// let bytes = match observed(sandbox.read_file(&path).await, format!("cannot read '{path}'"))? {
///     Ok(bytes) => bytes,
///     Err(outcome) => return Ok(outcome),
/// };
/// ```
pub fn observed<T>(
    result: retrograd_agent_core::Result<T>,
    message: String,
) -> retrograd_agent_core::Result<std::result::Result<T, ToolOutcome>> {
    match result {
        Ok(value) => Ok(Ok(value)),
        Err(error) if error.is_sandbox_failure() => Err(error),
        Err(_) => Ok(Err(ToolOutcome::error(message))),
    }
}

/// Caps a payload the same way every tool does.
pub fn cap(mut text: String, max_bytes: usize) -> String {
    truncate_utf8(&mut text, max_bytes, "\n[output truncated]");
    text
}

fn object_schema(properties: serde_json::Value, required: &[&str]) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required,
    })
}

#[cfg(test)]
pub(crate) mod testing {
    //! A sandbox that keeps its files in memory and answers `exec` from a
    //! script. Enough to pin every tool's wording without a daemon, and without
    //! `retrograd-tools` depending on the crate that owns `LocalSandbox`.

    use std::collections::BTreeMap;
    use std::sync::Mutex;

    use async_trait::async_trait;
    use retrograd_agent_core::{
        DirEntry, Error, ExecOutput, ExecRequest, FileKind, Result, Sandbox, SandboxLimits,
        relative_path,
    };

    #[derive(Default)]
    pub struct FakeSandbox {
        pub files: Mutex<BTreeMap<String, Vec<u8>>>,
        pub calls: Mutex<Vec<Vec<String>>>,
        pub reply: Mutex<Option<ExecOutput>>,
        /// Answers every operation the way a dead container does.
        pub broken: bool,
    }

    impl FakeSandbox {
        pub fn new() -> Self {
            Self::default()
        }

        /// The sandbox is gone: every call fails with [`Error::Sandbox`], which
        /// is the one failure a tool must not turn into an observation.
        pub fn broken() -> Self {
            Self {
                broken: true,
                ..Self::default()
            }
        }

        fn dead(&self) -> Result<()> {
            if self.broken {
                return Err(Error::sandbox("the container is gone"));
            }
            Ok(())
        }

        pub fn with_file(self, path: &str, contents: &str) -> Self {
            self.files
                .lock()
                .unwrap()
                .insert(path.into(), contents.as_bytes().to_vec());
            self
        }

        pub fn replying(self, output: ExecOutput) -> Self {
            *self.reply.lock().unwrap() = Some(output);
            self
        }

        pub fn last_argv(&self) -> Vec<String> {
            self.calls
                .lock()
                .unwrap()
                .last()
                .cloned()
                .unwrap_or_default()
        }
    }

    #[async_trait]
    impl Sandbox for FakeSandbox {
        async fn exec(&self, request: ExecRequest) -> Result<ExecOutput> {
            self.dead()?;
            self.calls.lock().unwrap().push(request.argv);
            Ok(self.reply.lock().unwrap().clone().unwrap_or(ExecOutput {
                exit_code: Some(0),
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                truncated: false,
            }))
        }

        async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
            self.dead()?;
            let path = relative_path(self.workdir(), path)?;
            self.files
                .lock()
                .unwrap()
                .get(&path)
                .cloned()
                .ok_or_else(|| Error::Tool(format!("read '{path}': NotFound")))
        }

        async fn write_file(&self, path: &str, bytes: &[u8]) -> Result<()> {
            self.dead()?;
            let path = relative_path(self.workdir(), path)?;
            self.files.lock().unwrap().insert(path, bytes.to_vec());
            Ok(())
        }

        async fn list_dir(&self, path: &str) -> Result<Vec<DirEntry>> {
            self.dead()?;
            let prefix = relative_path(self.workdir(), path)?;
            Ok(self
                .files
                .lock()
                .unwrap()
                .iter()
                .filter(|(name, _)| name.starts_with(&prefix))
                .map(|(name, bytes)| DirEntry {
                    path: name.clone(),
                    kind: FileKind::File,
                    size: bytes.len() as u64,
                })
                .collect())
        }

        async fn remove(&self, path: &str) -> Result<()> {
            let path = relative_path(self.workdir(), path)?;
            self.files.lock().unwrap().remove(&path);
            Ok(())
        }

        fn workdir(&self) -> &str {
            "/work"
        }

        fn limits(&self) -> SandboxLimits {
            SandboxLimits::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_advertises_only_tools_its_image_has() {
        let names = |profile: Profile| {
            profile
                .tools()
                .iter()
                .map(|tool| tool.spec().name)
                .collect::<Vec<_>>()
        };
        let python = names(Profile::Python);
        assert!(python.contains(&"python".to_owned()) && !python.contains(&"node".to_owned()));
        let typescript = names(Profile::Typescript);
        assert!(
            typescript.contains(&"node".to_owned()) && !typescript.contains(&"python".to_owned())
        );
        assert_eq!(names(Profile::Custom), Vec::<String>::new());

        // Both profiles are buildable as a set: no duplicate, no broken schema.
        for profile in [Profile::Python, Profile::Typescript] {
            crate::ToolSet::builder()
                .extend(profile.tools())
                .build()
                .expect("a profile must be a valid tool set");
        }
    }

    /// The rendering is what the model reads and is trained on, so its shape is
    /// pinned here rather than left to whoever edits `render_exec` next.
    #[test]
    fn a_command_result_reads_the_same_way_whatever_produced_it() {
        let outcome = render_exec(&ExecOutput {
            exit_code: Some(1),
            stdout: "one\n".into(),
            stderr: "two\n".into(),
            timed_out: false,
            truncated: true,
        });
        assert_eq!(
            outcome.content,
            "exit code: 1\nstdout:\none\nstderr:\ntwo\n[output truncated]\n"
        );
        assert!(outcome.is_error);

        let killed = render_exec(&ExecOutput {
            exit_code: None,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: true,
            truncated: false,
        });
        assert_eq!(
            killed.content,
            "command timed out and was killed\nexit code: killed by a signal\n"
        );
    }
}
