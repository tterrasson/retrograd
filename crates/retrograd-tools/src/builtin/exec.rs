//! Tools that run something.

use async_trait::async_trait;
use retrograd_agent_core::{Result, Sandbox, ToolSpec};

use super::{bounded_request, object_schema, render_exec, string_arg, string_list_arg};
use crate::session::{SessionTool, ToolOutcome};

/// A shell command.
///
/// The command is handed to `sh -c` as a *single* argument, never assembled by
/// us from several: quoting model output into a command line is how an argument
/// becomes an injection into the harness rather than into the sandbox.
pub struct Shell {
    program: String,
}

impl Shell {
    pub fn new(program: impl Into<String>) -> Self {
        Self {
            program: program.into(),
        }
    }
}

impl Default for Shell {
    fn default() -> Self {
        Self {
            // `bash`, not `sh`: the model writes bash, and every image the
            // profiles ship has it. `-c` and not `-lc` - a login shell sources
            // profile scripts, whose output would differ between images and
            // land in the training tokens.
            program: "bash".into(),
        }
    }
}

#[async_trait]
impl SessionTool for Shell {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "bash".into(),
            description: "Run a shell command in the workspace and return its exit code, \
                          stdout and stderr."
                .into(),
            input_schema: object_schema(
                serde_json::json!({
                    "command": {"type": "string", "description": "The shell command to run."}
                }),
                &["command"],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let command = match string_arg(&arguments, "command") {
            Ok(command) => command,
            Err(outcome) => return Ok(outcome),
        };
        let request = bounded_request(sandbox, vec![self.program.clone(), "-c".into(), command]);
        Ok(render_exec(&sandbox.exec(request).await?))
    }
}

/// An interpreter fed a snippet on the command line.
pub struct Interpreter {
    name: String,
    argv: Vec<String>,
}

impl Interpreter {
    pub fn new(name: impl Into<String>, argv: Vec<String>) -> Self {
        Self {
            name: name.into(),
            argv,
        }
    }

    pub fn python() -> Self {
        Self::new("python", vec!["python3".into(), "-c".into()])
    }

    pub fn node() -> Self {
        Self::new("node", vec!["node".into(), "-e".into()])
    }
}

#[async_trait]
impl SessionTool for Interpreter {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: self.name.clone(),
            description: format!("Run a {} snippet in the workspace.", self.name),
            input_schema: object_schema(
                serde_json::json!({
                    "code": {"type": "string", "description": "Source to execute."}
                }),
                &["code"],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let code = match string_arg(&arguments, "code") {
            Ok(code) => code,
            Err(outcome) => return Ok(outcome),
        };
        let mut argv = self.argv.clone();
        argv.push(code);
        Ok(render_exec(
            &sandbox.exec(bounded_request(sandbox, argv)).await?,
        ))
    }
}

/// The task's test command.
///
/// It carries no reward of its own. What the tests say is the environment's
/// verdict to give - through the task's `verify` command, once, at submission,
/// not something the model can farm by calling a tool repeatedly.
pub struct RunTests {
    argv: Vec<String>,
}

impl RunTests {
    pub fn new(argv: Vec<String>) -> Self {
        Self { argv }
    }

    pub fn pytest() -> Self {
        Self::new(vec![
            "python3".into(),
            "-m".into(),
            "pytest".into(),
            "-q".into(),
        ])
    }

    pub fn npm() -> Self {
        Self::new(vec!["npm".into(), "test".into(), "--silent".into()])
    }
}

#[async_trait]
impl SessionTool for RunTests {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "run_tests".into(),
            description: format!("Run the test suite (`{}`).", self.argv.join(" ")),
            input_schema: object_schema(
                serde_json::json!({
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Extra arguments, appended to the test command."
                    }
                }),
                &[],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let extra = match string_list_arg(&arguments, "args") {
            Ok(extra) => extra,
            Err(outcome) => return Ok(outcome),
        };
        let mut argv = self.argv.clone();
        argv.extend(extra);
        Ok(render_exec(
            &sandbox.exec(bounded_request(sandbox, argv)).await?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use retrograd_agent_core::ExecOutput;

    use super::super::testing::FakeSandbox;
    use super::*;

    #[tokio::test]
    async fn a_command_reaches_the_sandbox_as_one_argument() {
        let sandbox = FakeSandbox::new();
        Shell::default()
            .call(
                &sandbox,
                serde_json::json!({"command": "echo 'a b'; rm -rf /"}),
            )
            .await
            .unwrap();
        assert_eq!(sandbox.last_argv(), ["bash", "-c", "echo 'a b'; rm -rf /"]);
    }

    /// A missing or mistyped argument is an observation with a frozen wording,
    /// never an `Err`: the policy has to read it and correct itself.
    #[tokio::test]
    async fn a_bad_argument_is_an_observation_with_a_fixed_wording() {
        let sandbox = FakeSandbox::new();
        let outcome = Shell::default()
            .call(&sandbox, serde_json::json!({}))
            .await
            .unwrap();
        assert_eq!(outcome.content, "missing required argument 'command'");
        assert!(outcome.is_error);

        let outcome = Shell::default()
            .call(&sandbox, serde_json::json!({"command": 3}))
            .await
            .unwrap();
        assert_eq!(outcome.content, "argument 'command' must be a string");

        let outcome = RunTests::pytest()
            .call(&sandbox, serde_json::json!({"args": "-x"}))
            .await
            .unwrap();
        assert_eq!(
            outcome.content,
            "argument 'args' must be an array of strings"
        );
        // The command never ran, so nothing was observed about the world.
        assert!(sandbox.last_argv().is_empty());
    }

    #[tokio::test]
    async fn a_failing_test_run_is_an_observation_not_a_reward() {
        let sandbox = FakeSandbox::new().replying(ExecOutput {
            exit_code: Some(1),
            stdout: "1 failed".into(),
            stderr: String::new(),
            timed_out: false,
            truncated: false,
        });
        let outcome = RunTests::pytest()
            .call(&sandbox, serde_json::json!({"args": ["-x"]}))
            .await
            .unwrap();
        assert_eq!(sandbox.last_argv(), ["python3", "-m", "pytest", "-q", "-x"]);
        assert!(outcome.is_error && outcome.reward.is_none() && !outcome.done);
    }
}
