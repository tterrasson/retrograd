//! What a scenario asks of a sandbox-backed environment.
//!
//! The declaration lives in `scenario.metadata.env`, so a task is data in the
//! dataset rather than code in the trainer:
//!
//! ```json
//! {
//!   "id": "fix-parser-3",
//!   "user": "The test test_parse_empty fails. Fix it.",
//!   "metadata": {
//!     "env": {
//!       "files": {"src/parser.py": "...", "tests/test_parser.py": "..."},
//!       "setup": ["pip install -e."],
//!       "verify": {"command": ["pytest", "-q"], "reward_on_success": 1.0},
//!       "summary": ["git", "diff"]
//!     }
//!   }
//! }
//! ```
//!
//! `verify` is the point of the whole thing: it produces a **verifiable
//! reward**, which takes the group out of the judge's hands and replaces an
//! LLM's opinion with a command's exit code.
//!
//! Every task is parsed and validated in
//! [`prepare`](retrograd_agent_core::EnvironmentFactory::prepare), before the
//! first rollout. A typo in the 300th scenario is a configuration error the
//! operator fixes in a second; discovering it mid-run is a lost run.

use std::collections::BTreeMap;
use std::time::Duration;

use retrograd_agent_core::{Error, Result, Scenario, relative_path};
use retrograd_spec::env::Profile;
use serde::{Deserialize, Serialize};

/// The path every episode works in, whatever backs the sandbox.
pub const WORKDIR: &str = "/work";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EnvTask {
    /// Overrides the configured image. A pool runs one image, so this is only
    /// accepted when it agrees with the configuration - see
    /// [`EnvTask::check_image`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<Profile>,
    /// The toolset this scenario's trajectories get, among those the
    /// environment declares selectable. Absent means the environment's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub toolset: Option<String>,
    /// Files written into the workspace before the first turn.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub files: BTreeMap<String, String>,
    /// Shell commands run after the files are in place. A failing one is a
    /// broken world, not a failed action: the episode would start from a state
    /// the scenario does not describe.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub setup: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<Verify>,
    /// Argv whose stdout becomes [`EnvState::summary`](retrograd_agent_core::EnvState),
    /// typically `["git", "diff"]`. It is what the judge reads instead of the
    /// transcript.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<Vec<String>>,
    /// Appended to the scenario as an opening observation. Identical for every
    /// member of a group, so turn 0 stays a single shared prompt.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
}

/// The command that decides what the attempt was worth.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verify {
    pub command: Vec<String>,
    #[serde(default = "one")]
    pub reward_on_success: f32,
    #[serde(default)]
    pub reward_on_failure: f32,
    /// Budget for this one command. Falls back to the environment's
    /// `verify_timeout_secs`, then to the sandbox's own exec timeout - see
    /// [`Verify::timeout`]. A test suite is usually the longest command of the
    /// episode, and it runs once.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

impl Verify {
    /// The budget this command actually runs under: what the scenario asked
    /// for, else what the environment configured, else the sandbox's own exec
    /// timeout.
    ///
    /// The task wins because it is the one that knows what it is running: an
    /// environment-wide budget is sized for the median suite, and a scenario
    /// whose suite is slower has no other way to say so.
    pub fn timeout(&self, configured: Option<Duration>, exec_timeout: Duration) -> Duration {
        self.timeout_secs
            .map(Duration::from_secs)
            .or(configured)
            .unwrap_or(exec_timeout)
    }
}

fn one() -> f32 {
    1.0
}

impl EnvTask {
    /// Reads `scenario.metadata.env`. A scenario without one is a valid task
    /// with an empty workspace - that is how a pure-shell scenario is written.
    pub fn from_scenario(scenario: &Scenario) -> Result<Self> {
        let Some(declaration) = scenario.metadata.get("env") else {
            return Ok(Self::default());
        };
        let task: Self = serde_json::from_value(declaration.clone()).map_err(|error| {
            Error::invalid(format!(
                "scenario '{}': invalid metadata.env: {error}",
                scenario.id
            ))
        })?;
        task.validate(&scenario.id)?;
        Ok(task)
    }

    pub fn validate(&self, scenario_id: &str) -> Result<()> {
        let context =
            |message: String| Error::invalid(format!("scenario '{scenario_id}': {message}"));
        for path in self.files.keys() {
            // Model output is not the only untrusted path: a dataset is written
            // by hand too, and `../../etc` in a fixture would be written to the
            // host by whatever backs the sandbox.
            let relative = relative_path(WORKDIR, path)
                .map_err(|error| context(format!("file '{path}': {error}")))?;
            if relative.is_empty() {
                return Err(context(format!("file '{path}' is not a file path")));
            }
        }
        for command in &self.setup {
            if command.trim().is_empty() {
                return Err(context("a setup command must not be empty".into()));
            }
        }
        if let Some(verify) = &self.verify {
            if verify.command.iter().all(|part| part.trim().is_empty()) {
                return Err(context("verify.command must not be empty".into()));
            }
            for reward in [verify.reward_on_success, verify.reward_on_failure] {
                if !reward.is_finite() {
                    return Err(context("verify rewards must be finite".into()));
                }
            }
            // A zero budget kills every verification before it starts, and the
            // whole group would be graded as failing for a reason no rollout
            // shows.
            if verify.timeout_secs == Some(0) {
                return Err(context("verify.timeout_secs must be positive".into()));
            }
        }
        if self.summary.as_ref().is_some_and(|argv| argv.is_empty()) {
            return Err(context("summary must not be an empty argv".into()));
        }
        Ok(())
    }

    /// A pool serves one image to every episode, so a scenario that names a
    /// different one is refused rather than silently run somewhere else. Two
    /// images in one run means two pools, which is a configuration to write, not
    /// a default to guess.
    ///
    /// Two spellings of the same image are accepted: the reference the operator
    /// configured (usually a tag) and the digest it resolved to. A dataset
    /// written against `python:3.12-slim` and one written against the digest
    /// that was actually verified are both telling the truth, and refusing the
    /// second - which is the more precise of the two - would be perverse.
    pub fn check_image(
        &self,
        scenario_id: &str,
        configured: Option<&str>,
        pinned: Option<&str>,
    ) -> Result<()> {
        let Some(wanted) = &self.image else {
            return Ok(());
        };
        let accepted = [configured, pinned];
        // Nothing configured at all (a local sandbox) means nothing to disagree
        // with.
        if accepted.iter().all(Option::is_none) || accepted.contains(&Some(wanted.as_str())) {
            return Ok(());
        }
        Err(Error::invalid(format!(
            "scenario '{scenario_id}' asks for image '{wanted}' but the environment runs '{}'; \
             one pool runs one image",
            pinned.or(configured).unwrap_or_default()
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scenario(metadata: serde_json::Value) -> Scenario {
        Scenario {
            id: "s".into(),
            system: None,
            user: "do it".into(),
            metadata: metadata.as_object().cloned().unwrap(),
        }
    }

    #[test]
    fn a_scenario_without_a_declaration_is_a_valid_empty_task() {
        let task = EnvTask::from_scenario(&scenario(serde_json::json!({}))).unwrap();
        assert_eq!(task, EnvTask::default());
        // Another environment's metadata is not ours to reject.
        assert!(
            EnvTask::from_scenario(&scenario(serde_json::json!({"rubric": "be nice"}))).is_ok()
        );
    }

    #[test]
    fn the_documented_declaration_parses_with_its_defaults() {
        let task = EnvTask::from_scenario(&scenario(serde_json::json!({
            "env": {
                "files": {"src/parser.py": "x = 1"},
                "setup": ["pip install -e ."],
                "verify": {"command": ["pytest", "-q"]},
                "summary": ["git", "diff"]
            }
        })))
        .unwrap();
        let verify = task.verify.unwrap();
        assert_eq!(
            (verify.reward_on_success, verify.reward_on_failure),
            (1.0, 0.0)
        );
        assert_eq!(task.summary.unwrap(), ["git", "diff"]);
    }

    /// Every one of these fails at `prepare`, which is the whole reason the type
    /// exists: none of them is discoverable from a rollout without losing it.
    #[test]
    fn a_malformed_task_is_refused_before_the_first_rollout() {
        for broken in [
            serde_json::json!({"env": {"files": {"../escape.py": "x"}}}),
            serde_json::json!({"env": {"files": {"/etc/passwd": "x"}}}),
            serde_json::json!({"env": {"setup": [" "]}}),
            serde_json::json!({"env": {"verify": {"command": []}}}),
            serde_json::json!({"env": {"summary": []}}),
            serde_json::json!({"env": {"verify": {"command": ["pytest"], "timeout_secs": 0}}}),
            // A typo is a typo, not an option that gets ignored.
            serde_json::json!({"env": {"verify_command": ["pytest"]}}),
        ] {
            let error = EnvTask::from_scenario(&scenario(broken.clone()))
                .expect_err(&format!("accepted {broken}"));
            assert!(error.to_string().contains("scenario 's'"), "{error}");
        }
    }

    /// A per-task budget beats the environment's, which beats the sandbox's own
    /// exec timeout. The scenario is the only one that knows how long its suite
    /// takes, and a declared `timeout_secs` that quietly did nothing would look
    /// configured and grade the group on a command that never finished.
    #[test]
    fn the_task_budget_wins_over_the_environment_and_the_sandbox() {
        let exec = Duration::from_secs(30);
        let configured = Some(Duration::from_secs(120));
        let declared = Verify {
            command: vec!["pytest".into()],
            reward_on_success: 1.0,
            reward_on_failure: 0.0,
            timeout_secs: Some(600),
        };
        assert_eq!(declared.timeout(configured, exec), Duration::from_secs(600));
        assert_eq!(declared.timeout(None, exec), Duration::from_secs(600));

        let silent = Verify {
            timeout_secs: None,
            ..declared
        };
        assert_eq!(silent.timeout(configured, exec), Duration::from_secs(120));
        assert_eq!(silent.timeout(None, exec), exec);
    }

    #[test]
    fn a_scenario_cannot_pick_an_image_the_pool_does_not_run() {
        let task = EnvTask {
            image: Some("other:1".into()),
            ..Default::default()
        };
        assert!(
            task.check_image("s", Some("python:3.12-slim"), None)
                .is_err()
        );
        assert!(task.check_image("s", Some("other:1"), None).is_ok());
        // No configured image (a local sandbox) means nothing to disagree with.
        assert!(task.check_image("s", None, None).is_ok());
    }

    /// The configured tag and the digest it resolved to are the same image, and
    /// a scenario is entitled to name either. Naming the digest is the more
    /// precise of the two, and is accepted by the same image check.
    #[test]
    fn a_scenario_may_name_the_tag_or_the_digest_that_tag_resolved_to() {
        let digest = "python@sha256:beef";
        let by_digest = EnvTask {
            image: Some(digest.into()),
            ..Default::default()
        };
        by_digest
            .check_image("s", Some("python:3.12-slim"), Some(digest))
            .expect("the digest actually running must be accepted");

        let by_tag = EnvTask {
            image: Some("python:3.12-slim".into()),
            ..Default::default()
        };
        by_tag
            .check_image("s", Some("python:3.12-slim"), Some(digest))
            .expect("the configured tag must still be accepted");

        let elsewhere = EnvTask {
            image: Some("node:22-slim".into()),
            ..Default::default()
        };
        let error = elsewhere
            .check_image("s", Some("python:3.12-slim"), Some(digest))
            .unwrap_err()
            .to_string();
        // The message names what actually runs, not the tag it was written as.
        assert!(error.contains(digest), "{error}");
    }
}
