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
use retrograd_agent_core::{Error, ExecOutput, ExecRequest, Result, Sandbox};
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::session::{SessionTool, ToolOutcome};
use crate::{RegisteredTool, ToolDefinitions, ToolRegistry};

mod exec;
mod files;

pub use exec::{Interpreter, RunTests, Shell};
pub use files::{EditFile, Grep, ListDir, ReadFile, Submit, WriteFile};

/// The built-in tools and toolsets, declared like any user file.
const BUILTIN_DEFINITIONS: &str = include_str!("builtin.toml");

/// The built-in factories, then the definitions that use them.
pub(crate) fn builtin_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::default();
    let factories: [(&str, Arc<dyn crate::ToolFactory>); 9] = [
        (
            "shell",
            Arc::new(|params: &serde_json::Value| {
                let ShellParams { program } = params_of("shell", params)?;
                Ok(session(match program {
                    Some(program) => Shell::new(program),
                    None => Shell::default(),
                }))
            }),
        ),
        (
            "interpreter",
            Arc::new(|params: &serde_json::Value| {
                let InterpreterParams { language, argv } =
                    required_params_of("interpreter", params)?;
                Ok(session(Interpreter::new(language, argv)))
            }),
        ),
        (
            "run_tests",
            Arc::new(|params: &serde_json::Value| {
                let CommandParams { argv } = required_params_of("run_tests", params)?;
                Ok(session(RunTests::new(argv)))
            }),
        ),
        ("read_file", unit("read_file", || session(ReadFile))),
        ("write_file", unit("write_file", || session(WriteFile))),
        ("edit_file", unit("edit_file", || session(EditFile))),
        ("list_dir", unit("list_dir", || session(ListDir))),
        ("grep", unit("grep", || session(Grep))),
        ("submit", unit("submit", || session(Submit))),
    ];
    for (name, factory) in factories {
        registry
            .register_factory(name, factory)
            .expect("builtin factory names are unique");
    }
    let definitions: ToolDefinitions =
        toml::from_str(BUILTIN_DEFINITIONS).expect("builtin.toml is valid");
    registry
        .define(definitions)
        .expect("builtin.toml names only builtin factories, once each");
    registry
}

fn session(tool: impl SessionTool + 'static) -> RegisteredTool {
    RegisteredTool::Session(Arc::new(tool))
}

/// A factory for a tool that takes no parameters, and says so when given some.
fn unit(name: &'static str, build: fn() -> RegisteredTool) -> Arc<dyn crate::ToolFactory> {
    Arc::new(move |params: &serde_json::Value| {
        if !params.is_null() && params.as_object().is_none_or(|map| !map.is_empty()) {
            return Err(Error::invalid(format!("factory '{name}' takes no params")));
        }
        Ok(build())
    })
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ShellParams {
    program: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct InterpreterParams {
    language: String,
    argv: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CommandParams {
    argv: Vec<String>,
}

/// Optional parameters: absent means the factory's defaults.
fn params_of<T: DeserializeOwned + Default>(
    factory: &str,
    params: &serde_json::Value,
) -> Result<T> {
    if params.is_null() {
        return Ok(T::default());
    }
    required_params_of(factory, params)
}

fn required_params_of<T: DeserializeOwned>(factory: &str, params: &serde_json::Value) -> Result<T> {
    T::deserialize(params)
        .map_err(|error| Error::invalid(format!("factory '{factory}': invalid params: {error}")))
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

    /// A builtin's spec is what the model reads and is trained on, so a version
    /// is frozen: changing its name, description or schema means adding
    /// `id@2` to `builtin.toml`, then a row here - never editing a row.
    #[test]
    fn builtin_versions_are_frozen() {
        let registry = crate::ToolRegistry::builtin();
        let actual = registry
            .definitions()
            .map(|definition| {
                let built = registry.build(&definition.key()).unwrap();
                let spec = serde_json::to_string(&built.entry.spec).unwrap();
                (
                    definition.key().to_string(),
                    crate::registry::sha256_hex(spec.as_bytes()),
                )
            })
            .collect::<Vec<_>>();
        let frozen: &[(&str, &str)] = &[
            (
                "bash@1",
                "845203a7a48b3cba77be9e97659961d3d81a36fac7d943096d7e744ea94fe2d9",
            ),
            (
                "edit_file@1",
                "f33b17304d61a40651497df52502aafdf1933489ff96c587b97f7907f1d905d2",
            ),
            (
                "grep@1",
                "c46bcd8744a31941f7bf8576fcc18eb5a6e2ed8b97b60f9ec9679c02c7a4f0fd",
            ),
            (
                "list_dir@1",
                "597226b4b6d7b05b460716147250726d00cbc93b3d3468498ef638b9cc042de6",
            ),
            (
                "node@1",
                "73150511b4ef4914743edb3e6a07e100e95e520d22841af033e9eb50ad19676f",
            ),
            (
                "npm_test@1",
                "e68488d741f97cf8194e34ec1ac8515f11c7dc23d582f927b7a1a975bc6c3bd0",
            ),
            (
                "pytest@1",
                "d8bebc8e0c815702c5b9260621a67d1a53c110464c3f0b1035e06b36613ca827",
            ),
            (
                "python@1",
                "a7ae0b79fc612e914697a4c64657ed6abef286e3b2b017f2537f191e4ab13f01",
            ),
            (
                "read_file@1",
                "696419383b5e89ba82aff5f65a4702dd83bc9450417a628463a51b3ca03c76a1",
            ),
            (
                "submit@1",
                "892e611c0f75bab94969139c3f1bf8c967c941c065a04d8fcff52cd65a3accb5",
            ),
            (
                "write_file@1",
                "34652e2b731f19b5b35d4fa3cf0836046a04c2e7676a7cc7ab29f5763802b7cf",
            ),
        ];
        assert_eq!(
            actual,
            frozen
                .iter()
                .map(|(reference, sha)| (reference.to_string(), sha.to_string()))
                .collect::<Vec<_>>()
        );
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
