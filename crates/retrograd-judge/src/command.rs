//! `type = "command"`: a local process scores a group, over the JSONL reward
//! protocol of [`crate::process`].
//!
//! One process per group, and deliberately so - [`RewardMode::OneShot`]. This
//! backend is reached through `&self` from an async runtime, whereas a
//! persistent worker is a mutable session with a lifetime; the trainer's own
//! `reward_command`, which owns one for the length of its loop, is where the
//! spawn is worth amortizing (`docs/engineering/GRPO.md`).

use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use retrograd_agent_core::trajectory::{Role, TrajectoryGroup};
use retrograd_agent_core::{Error, Result};
use retrograd_core::{RewardMode, RewardProtocol};

use crate::process::RewardProcess;
use crate::{RewardBackend, Score};

#[derive(Clone, Debug)]
pub struct CommandReward {
    command: Vec<String>,
    timeout: Duration,
}

impl CommandReward {
    pub fn new(command: Vec<String>) -> Result<Self> {
        Self::with_timeout(command, Duration::from_secs(30))
    }

    /// Creates a local reward backend with a bounded command execution time.
    /// The explicit constructor is useful for embedders and tests; serialized
    /// configurations keep the conservative thirty-second default above.
    pub fn with_timeout(command: Vec<String>, timeout: Duration) -> Result<Self> {
        // Constructed here rather than at call time so a command that cannot
        // work is refused when it is declared, not on the first scored group.
        RewardProcess::new(&command, protocol(timeout))?;
        Ok(Self { command, timeout })
    }
}

fn protocol(timeout: Duration) -> RewardProtocol {
    RewardProtocol {
        mode: RewardMode::OneShot,
        timeout,
    }
}

#[derive(Serialize)]
struct Request<'a> {
    prompt: &'a str,
    completion: &'a str,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    reward: f32,
}

#[async_trait]
impl RewardBackend for CommandReward {
    async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
        let command = self.command.clone();
        let pairs = group
            .trajectories
            .iter()
            .map(|trajectory| {
                let prompt = trajectory
                    .messages
                    .iter()
                    .filter(|message| matches!(message.role, Role::System | Role::User))
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                let completion = trajectory
                    .messages
                    .iter()
                    .filter(|message| message.role == Role::Assistant)
                    .map(|message| message.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n");
                (prompt, completion)
            })
            .collect::<Vec<_>>();
        let timeout = self.timeout;
        tokio::task::spawn_blocking(move || run_command(&command, &pairs, timeout))
            .await
            .map_err(|error| Error::Reward(format!("reward command task failed: {error}")))?
    }
}

fn run_command(
    command: &[String],
    pairs: &[(String, String)],
    timeout: Duration,
) -> Result<Vec<Score>> {
    let responses: Vec<Response> = RewardProcess::new(command, protocol(timeout))?.call(
        pairs
            .iter()
            .map(|(prompt, completion)| Request { prompt, completion }),
    )?;
    responses
        .into_iter()
        .map(|response| {
            if !response.reward.is_finite() {
                return Err(Error::Reward("reward is not finite".into()));
            }
            Ok(Score {
                value: response.reward,
                valid: true,
                explanation: None,
                error: None,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::trajectory::{Message, Role, Trajectory};

    fn group() -> TrajectoryGroup {
        let trajectory = Trajectory {
            scenario_id: "scenario".into(),
            messages: vec![
                Message::text(Role::User, "prompt"),
                Message::text(Role::Assistant, "completion"),
            ],
            tokens: vec![1, 2],
            old_logprobs: vec![-0.5],
            train_mask: vec![false, true],
            steps: vec![],
            reward: None,
            truncated: false,
            metadata: Default::default(),
            provenance: None,
        };
        TrajectoryGroup {
            group_id: 1,
            scenario_id: "scenario".into(),
            trajectories: vec![trajectory],
        }
    }

    fn shell(script: &str) -> Vec<String> {
        vec!["sh".into(), "-c".into(), script.into()]
    }

    #[tokio::test]
    async fn command_reward_accepts_one_json_response_per_trajectory() {
        let judge = CommandReward::new(shell(
            "while read line; do printf '{\"reward\":0.75}\\n'; done",
        ))
        .unwrap();
        let scores = judge.score_group(&group()).await.unwrap();
        assert_eq!(scores.len(), 1);
        assert_eq!(scores[0].value, 0.75);
    }

    #[tokio::test]
    async fn command_reward_drains_stderr_before_the_process_exits() {
        let judge = CommandReward::new(shell(
            "head -c 200000 /dev/zero >&2; printf '{\"reward\":0.5}\\n'",
        ))
        .unwrap();
        let scores = judge.score_group(&group()).await.unwrap();
        assert_eq!(scores[0].value, 0.5);
    }

    #[tokio::test]
    async fn command_reward_rejects_invalid_and_non_finite_output() {
        for script in ["printf 'not-json\\n'", "printf '{\"reward\":null}\\n'"] {
            let judge = CommandReward::new(shell(script)).unwrap();
            let error = judge.score_group(&group()).await.unwrap_err();
            assert!(
                error.to_string().contains("invalid reward response"),
                "{script} -> {error}"
            );
        }

        // A JSON number valid for f64 but beyond f32::MAX: serde_json parses it
        // into f32 as infinity instead of erroring, so this exercises the
        // is_finite() check rather than the JSON-parsing branch above.
        let judge = CommandReward::new(shell("printf '{\"reward\":1e39}\\n'")).unwrap();
        let error = judge.score_group(&group()).await.unwrap_err();
        assert!(error.to_string().contains("not finite"), "{error}");
    }

    #[tokio::test]
    async fn command_reward_reports_a_missing_process_and_timeout() {
        let missing = CommandReward::new(vec!["/definitely/not/a/reward-command".into()]).unwrap();
        assert!(missing.score_group(&group()).await.is_err());

        let timeout =
            CommandReward::with_timeout(shell("sleep 1"), Duration::from_millis(20)).unwrap();
        let error = timeout.score_group(&group()).await.unwrap_err();
        assert!(error.to_string().contains("timed out"), "{error}");
    }

    /// The transport rejects an empty command and a zero timeout at declaration.
    #[tokio::test]
    async fn an_empty_command_or_a_zero_timeout_is_refused_at_declaration() {
        assert!(CommandReward::new(vec![]).is_err());
        assert!(CommandReward::with_timeout(shell("true"), Duration::ZERO).is_err());
    }
}
