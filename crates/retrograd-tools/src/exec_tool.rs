//! A tool written in any language, run inside the trajectory's sandbox.
//!
//! The program gets the model's arguments as one JSON object on stdin. That is
//! the whole contract on the way in; on the way out, `protocol = "text"` renders
//! the command like every other one, and `protocol = "json"` reads stdout as an
//! [`ExecReply`] - which is how a tool in Python grades an action (`reward`) or
//! ends the episode (`done`) exactly like a Rust `SessionTool` can.
//!
//! Running in the sandbox is the point: the tool sees the trajectory's own
//! workspace and nothing else, so it is isolated per group member for free -
//! which an MCP server on the host cannot be.

use std::time::Duration;

use async_trait::async_trait;
use retrograd_agent_core::{Error, ExecRequest, Result, Sandbox, ToolSpec};

use crate::builtin::render_exec;
use crate::registry::sha256_hex;
use crate::session::{SessionTool, ToolOutcome};
use crate::{ExecImpl, ExecProtocol, ExecReply, ToolDefinition};

pub struct ExecTool {
    spec: ToolSpec,
    argv: Vec<String>,
    protocol: ExecProtocol,
    timeout: Option<Duration>,
}

impl ExecTool {
    /// Reads the script, if there is one, from the host: the tool is fixed when
    /// the run starts, not when it is called.
    pub fn from_definition(definition: &ToolDefinition, exec: &ExecImpl) -> Result<Self> {
        let mut argv = exec.argv.clone();
        if let Some(script) = &exec.script {
            let source = std::fs::read_to_string(script).map_err(|error| {
                Error::invalid(format!(
                    "tool '{}': read script {}: {error}",
                    definition.key(),
                    script.display()
                ))
            })?;
            argv.push(source);
        }
        Ok(Self {
            spec: ToolSpec {
                name: definition.exposed_name().to_owned(),
                description: definition.description.clone().unwrap_or_default(),
                input_schema: definition
                    .input_schema
                    .clone()
                    .unwrap_or_else(|| serde_json::json!({"type": "object", "properties": {}})),
            },
            argv,
            protocol: exec.protocol,
            timeout: exec.timeout_secs.map(Duration::from_secs),
        })
    }

    /// What decides this tool's behaviour: the argv with the script inlined,
    /// the protocol and the timeout.
    pub fn fingerprint(&self) -> String {
        let view = serde_json::json!({
            "argv": self.argv,
            "protocol": self.protocol,
            "timeout_secs": self.timeout.map(|timeout| timeout.as_secs()),
        });
        sha256_hex(view.to_string().as_bytes())
    }

    fn reply(&self, stdout: &str) -> Result<ToolOutcome> {
        let reply: ExecReply = serde_json::from_str(stdout.trim()).map_err(|error| {
            Error::Tool(format!(
                "exec tool '{}' broke the json protocol: {error}",
                self.spec.name
            ))
        })?;
        if reply.reward.is_some_and(|reward| !reward.is_finite()) {
            return Err(Error::Tool(format!(
                "exec tool '{}' replied with a non-finite reward",
                self.spec.name
            )));
        }
        Ok(ToolOutcome {
            content: reply.content,
            is_error: reply.is_error,
            reward: reply.reward,
            done: reply.done,
        })
    }
}

#[async_trait]
impl SessionTool for ExecTool {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    /// A command that failed or timed out is an observation in both protocols:
    /// the model may have passed arguments the tool refused. A `json` tool that
    /// exits 0 without a well-formed reply is a broken tool, and costs the
    /// trajectory rather than training on whatever it printed.
    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let limits = sandbox.limits();
        let request = ExecRequest::new(self.argv.clone())
            .with_stdin(arguments.to_string())
            .with_timeout(self.timeout.unwrap_or(limits.exec_timeout))
            .with_max_output_bytes(limits.max_output_bytes);
        let output = sandbox.exec(request).await?;
        match self.protocol {
            ExecProtocol::Text => Ok(render_exec(&output)),
            ExecProtocol::Json if !output.succeeded() => Ok(render_exec(&output)),
            ExecProtocol::Json if output.truncated => Err(Error::Tool(format!(
                "exec tool '{}' printed more than the output cap; its reply was cut",
                self.spec.name
            ))),
            ExecProtocol::Json => self.reply(&output.stdout),
        }
    }
}

#[cfg(test)]
mod tests {
    use retrograd_agent_core::ExecOutput;

    use super::*;
    use crate::builtin::testing::FakeSandbox;

    fn tool(source: &str) -> ExecTool {
        let definitions: crate::ToolDefinitions = toml::from_str(source).unwrap();
        let definition = &definitions.tools[0];
        ExecTool::from_definition(definition, definition.exec.as_ref().unwrap()).unwrap()
    }

    fn replying(stdout: &str, exit_code: i32) -> FakeSandbox {
        FakeSandbox::new().replying(ExecOutput {
            exit_code: Some(exit_code),
            stdout: stdout.into(),
            stderr: String::new(),
            timed_out: false,
            truncated: false,
        })
    }

    const JSON_TOOL: &str = r#"
        [[tool]]
        id = "check"
        version = 1
        description = "Check the answer."
        exec = { argv = ["python3", "check.py"], protocol = "json" }
    "#;

    #[tokio::test]
    async fn a_json_tool_can_grade_and_finish_the_episode() {
        let sandbox = replying(r#"{"content": "correct", "reward": 1.0, "done": true}"#, 0);
        let outcome = tool(JSON_TOOL)
            .call(&sandbox, serde_json::json!({"answer": 42}))
            .await
            .unwrap();
        assert_eq!(
            outcome,
            ToolOutcome::ok("correct").with_reward(1.0).finished()
        );
        assert_eq!(sandbox.last_argv(), ["python3", "check.py"]);
    }

    #[tokio::test]
    async fn a_failed_command_is_an_observation_and_a_bad_reply_a_broken_tool() {
        let outcome = tool(JSON_TOOL)
            .call(&replying("", 2), serde_json::json!({}))
            .await
            .unwrap();
        assert!(outcome.is_error && outcome.content.starts_with("exit code: 2"));

        let error = tool(JSON_TOOL)
            .call(&replying("not json", 0), serde_json::json!({}))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("broke the json protocol"),
            "{error}"
        );
        assert!(!error.is_sandbox_failure());
    }

    #[tokio::test]
    async fn a_text_tool_reads_like_any_command() {
        let outcome = tool(
            r#"
            [[tool]]
            id = "lint"
            version = 1
            description = "Lint."
            exec = { argv = ["ruff", "check"] }
            "#,
        )
        .call(&replying("all good\n", 0), serde_json::json!({}))
        .await
        .unwrap();
        assert_eq!(outcome.content, "exit code: 0\nstdout:\nall good\n");
    }

    #[test]
    fn the_script_is_part_of_the_argv_and_of_the_fingerprint() {
        let directory = std::env::temp_dir().join(format!("retrograd-exec-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let script = directory.join("t.py");
        let source = |body: &str| {
            std::fs::write(&script, body).unwrap();
            tool(&format!(
                "[[tool]]\nid = \"t\"\nversion = 1\ndescription = \"d\"\n\
                 exec = {{ argv = [\"python3\", \"-c\"], script = {:?} }}\n",
                script.display().to_string()
            ))
        };
        let first = source("print(1)");
        assert_eq!(first.argv, ["python3", "-c", "print(1)"]);
        let second = source("print(2)");
        assert_ne!(first.fingerprint(), second.fingerprint());
        std::fs::remove_dir_all(&directory).unwrap();
    }
}
