//! Tools that read and change the workspace.

use async_trait::async_trait;
use retrograd_agent_core::{FileKind, Result, Sandbox, ToolSpec};

use super::{bounded_request, cap, object_schema, observed, render_exec, string_arg};
use crate::session::{SessionTool, ToolOutcome};

pub struct ReadFile;

#[async_trait]
impl SessionTool for ReadFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "read_file".into(),
            description: "Read a UTF-8 file from the workspace.".into(),
            input_schema: object_schema(
                serde_json::json!({
                    "path": {"type": "string", "description": "Path relative to the workspace."}
                }),
                &["path"],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let path = match string_arg(&arguments, "path") {
            Ok(path) => path,
            Err(outcome) => return Ok(outcome),
        };
        // A missing file is the policy's mistake and becomes an observation; a
        // dead container is not, and `observed` is what keeps the two apart.
        let bytes = match observed(
            sandbox.read_file(&path).await,
            format!("cannot read '{path}'"),
        )? {
            Ok(bytes) => bytes,
            Err(outcome) => return Ok(outcome),
        };
        Ok(ToolOutcome::ok(cap(
            String::from_utf8_lossy(&bytes).into_owned(),
            sandbox.limits().max_output_bytes,
        )))
    }
}

pub struct WriteFile;

#[async_trait]
impl SessionTool for WriteFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "write_file".into(),
            description: "Create or overwrite a file in the workspace.".into(),
            input_schema: object_schema(
                serde_json::json!({
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                }),
                &["path", "content"],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let (path, content) = match (
            string_arg(&arguments, "path"),
            string_arg(&arguments, "content"),
        ) {
            (Ok(path), Ok(content)) => (path, content),
            (Err(outcome), _) | (_, Err(outcome)) => return Ok(outcome),
        };
        match observed(
            sandbox.write_file(&path, content.as_bytes()).await,
            format!("cannot write '{path}'"),
        )? {
            Ok(()) => Ok(ToolOutcome::ok(format!(
                "wrote {} bytes to '{path}'",
                content.len()
            ))),
            Err(outcome) => Ok(outcome),
        }
    }
}

/// Exact string replacement.
///
/// Both failure modes are refused rather than guessed at: a string that is
/// absent, and one that occurs more than once. Editing the first of several
/// matches would make the result depend on where in the file the model happened
/// to be thinking, which is exactly the kind of silent difference between two
/// members of a group that the reward should not contain.
pub struct EditFile;

#[async_trait]
impl SessionTool for EditFile {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "edit_file".into(),
            description: "Replace an exact, unique string in a file. Fails if the string is \
                          absent or appears more than once."
                .into(),
            input_schema: object_schema(
                serde_json::json!({
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"}
                }),
                &["path", "old_string", "new_string"],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let (path, old, new) = match (
            string_arg(&arguments, "path"),
            string_arg(&arguments, "old_string"),
            string_arg(&arguments, "new_string"),
        ) {
            (Ok(path), Ok(old), Ok(new)) => (path, old, new),
            (Err(outcome), _, _) | (_, Err(outcome), _) | (_, _, Err(outcome)) => {
                return Ok(outcome);
            }
        };
        if old.is_empty() {
            return Ok(ToolOutcome::error(
                "argument 'old_string' must not be empty",
            ));
        }
        let bytes = match observed(
            sandbox.read_file(&path).await,
            format!("cannot read '{path}'"),
        )? {
            Ok(bytes) => bytes,
            Err(outcome) => return Ok(outcome),
        };
        let contents = String::from_utf8_lossy(&bytes).into_owned();
        match contents.matches(old.as_str()).count() {
            0 => Ok(ToolOutcome::error(format!(
                "old_string not found in '{path}'"
            ))),
            1 => {
                let edited = contents.replace(old.as_str(), &new);
                match observed(
                    sandbox.write_file(&path, edited.as_bytes()).await,
                    format!("cannot write '{path}'"),
                )? {
                    Ok(()) => Ok(ToolOutcome::ok(format!("edited '{path}'"))),
                    Err(outcome) => Ok(outcome),
                }
            }
            count => Ok(ToolOutcome::error(format!(
                "old_string appears {count} times in '{path}'; include more context to make it unique"
            ))),
        }
    }
}

pub struct ListDir;

#[async_trait]
impl SessionTool for ListDir {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "list_dir".into(),
            description: "List a workspace directory, sorted by path.".into(),
            input_schema: object_schema(
                serde_json::json!({
                    "path": {"type": "string", "description": "Defaults to the workspace root."}
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
        let path = match arguments.get("path") {
            Some(serde_json::Value::String(path)) => path.clone(),
            Some(_) => return Ok(ToolOutcome::error("argument 'path' must be a string")),
            None => ".".into(),
        };
        let entries = match observed(
            sandbox.list_dir(&path).await,
            format!("cannot list '{path}'"),
        )? {
            Ok(entries) => entries,
            Err(outcome) => return Ok(outcome),
        };
        if entries.is_empty() {
            return Ok(ToolOutcome::ok("(empty)"));
        }
        let listing = entries
            .iter()
            .map(|entry| match entry.kind {
                FileKind::Dir => format!("dir  {}", entry.path),
                FileKind::Symlink => format!("link {}", entry.path),
                FileKind::File => format!("file {} ({} bytes)", entry.path, entry.size),
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ToolOutcome::ok(cap(
            listing,
            sandbox.limits().max_output_bytes,
        )))
    }
}

/// Recursive search.
///
/// The matches are sorted in the sandbox rather than left in `grep -r`'s
/// traversal order, which is the filesystem's and not a property of the action.
pub struct Grep;

#[async_trait]
impl SessionTool for Grep {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "grep".into(),
            description: "Search the workspace for a basic regular expression. Results are \
                          sorted."
                .into(),
            input_schema: object_schema(
                serde_json::json!({
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Defaults to the workspace root."}
                }),
                &["pattern"],
            ),
        }
    }

    async fn call(
        &self,
        sandbox: &dyn Sandbox,
        arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        let pattern = match string_arg(&arguments, "pattern") {
            Ok(pattern) => pattern,
            Err(outcome) => return Ok(outcome),
        };
        let path = match arguments.get("path") {
            Some(serde_json::Value::String(path)) => path.clone(),
            Some(_) => return Ok(ToolOutcome::error("argument 'path' must be a string")),
            None => ".".into(),
        };
        // Positional arguments, so a pattern is never spliced into the script.
        let request = bounded_request(
            sandbox,
            vec![
                "sh".into(),
                "-c".into(),
                "grep -rnI -- \"$1\" \"$2\" | LC_ALL=C sort".into(),
                "grep".into(),
                pattern,
                path,
            ],
        );
        let output = sandbox.exec(request).await?;
        // grep exits 1 on "no match", which is an answer and not a failure.
        if output.exit_code == Some(1) && output.stderr.trim().is_empty() {
            return Ok(ToolOutcome::ok("no match"));
        }
        Ok(render_exec(&output))
    }
}

/// Ends the episode.
///
/// It carries no reward: what the attempt was worth is the environment's
/// verdict, computed from the task's own `verify` command once the model has
/// stopped acting.
pub struct Submit;

#[async_trait]
impl SessionTool for Submit {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "submit".into(),
            description: "Declare the task finished. No further tool call is possible.".into(),
            input_schema: object_schema(
                serde_json::json!({
                    "message": {"type": "string", "description": "Optional summary of the change."}
                }),
                &[],
            ),
        }
    }

    async fn call(
        &self,
        _sandbox: &dyn Sandbox,
        _arguments: serde_json::Value,
    ) -> Result<ToolOutcome> {
        Ok(ToolOutcome::ok("submitted").finished())
    }
}

#[cfg(test)]
mod tests {
    use retrograd_agent_core::ExecOutput;

    use super::super::testing::FakeSandbox;
    use super::*;

    #[tokio::test]
    async fn an_edit_refuses_an_absent_or_ambiguous_target() {
        let sandbox = FakeSandbox::new().with_file("a.py", "x = 1\ny = 1\n");
        let edit = |old: &'static str, new: &'static str| {
            EditFile.call(
                &sandbox,
                serde_json::json!({"path": "a.py", "old_string": old, "new_string": new}),
            )
        };

        let outcome = edit("z = 1", "z = 2").await.unwrap();
        assert_eq!(outcome.content, "old_string not found in 'a.py'");
        let outcome = edit(" = 1", " = 2").await.unwrap();
        assert_eq!(
            outcome.content,
            "old_string appears 2 times in 'a.py'; include more context to make it unique"
        );

        let outcome = edit("x = 1", "x = 2").await.unwrap();
        assert_eq!(
            (outcome.content.as_str(), outcome.is_error),
            ("edited 'a.py'", false)
        );
        assert_eq!(
            sandbox.read_file("a.py").await.unwrap(),
            b"x = 2\ny = 1\n".to_vec()
        );
    }

    /// A missing file is a failed action, so it is an observation. The sandbox
    /// reports it as `Err` - the same shape it uses for a dead container - and
    /// the tool is where the two are told apart.
    #[tokio::test]
    async fn a_missing_file_is_an_observation_and_not_a_lost_trajectory() {
        let sandbox = FakeSandbox::new();
        let outcome = ReadFile
            .call(&sandbox, serde_json::json!({"path": "nope.py"}))
            .await
            .expect("a missing file must not cost the trajectory");
        assert_eq!(
            (outcome.content.as_str(), outcome.is_error),
            ("cannot read 'nope.py'", true)
        );
    }

    /// The other half of that split, and the one that is silent when it is
    /// wrong: a sandbox that has gone away must reach the engine as an `Err`.
    /// Rendering it as "cannot read" would let the rollout continue, be scored
    /// and be trained on a world that stopped existing halfway through.
    #[tokio::test]
    async fn a_dead_sandbox_kills_the_trajectory_instead_of_becoming_text() {
        let sandbox = FakeSandbox::broken();
        let calls: Vec<(&str, serde_json::Value)> = vec![
            ("read", serde_json::json!({"path": "a.py"})),
            ("write", serde_json::json!({"path": "a.py", "content": "x"})),
            (
                "edit",
                serde_json::json!({"path": "a.py", "old_string": "x", "new_string": "y"}),
            ),
            ("list", serde_json::json!({})),
            ("grep", serde_json::json!({"pattern": "x"})),
        ];
        for (name, arguments) in calls {
            let outcome = match name {
                "read" => ReadFile.call(&sandbox, arguments).await,
                "write" => WriteFile.call(&sandbox, arguments).await,
                "edit" => EditFile.call(&sandbox, arguments).await,
                "list" => ListDir.call(&sandbox, arguments).await,
                _ => Grep.call(&sandbox, arguments).await,
            };
            let error = match outcome {
                Ok(outcome) => panic!("{name} hid a broken sandbox as '{}'", outcome.content),
                Err(error) => error,
            };
            assert!(error.is_sandbox_failure(), "{name}: {error}");
        }
    }

    #[tokio::test]
    async fn a_pattern_is_passed_positionally_and_no_match_is_an_answer() {
        let sandbox = FakeSandbox::new().replying(ExecOutput {
            exit_code: Some(1),
            stdout: String::new(),
            stderr: String::new(),
            timed_out: false,
            truncated: false,
        });
        let outcome = Grep
            .call(&sandbox, serde_json::json!({"pattern": "\"; rm -rf /"}))
            .await
            .unwrap();
        assert_eq!(
            (outcome.content.as_str(), outcome.is_error),
            ("no match", false)
        );
        let argv = sandbox.last_argv();
        assert_eq!(argv[..2], ["sh", "-c"]);
        assert_eq!(argv[4], "\"; rm -rf /");
    }

    #[tokio::test]
    async fn submit_ends_the_episode_without_grading_it() {
        let outcome = Submit
            .call(&FakeSandbox::new(), serde_json::json!({}))
            .await
            .unwrap();
        assert!(outcome.done && outcome.reward.is_none() && !outcome.is_error);
    }
}
