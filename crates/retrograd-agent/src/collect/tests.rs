//! Collection over a scripted world: the environment decides, member by member,
//! what the policy does and what it earns, so every filter and every tie can
//! be set up on purpose.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use retrograd_core::SamplingParams;
use retrograd_dataset::{ChatExample, ChatMessage, PreferenceExample};
use serde_json::json;

use super::*;
use crate::env::{EnvState, Environment, EnvironmentFactory, StepOutcome};
use crate::judge::Score;
use crate::policy::{Policy, PolicyGeneration};
use crate::tools::{HermesToolCallParser, ToolCall, ToolResult, ToolSpec};
use crate::trajectory::{Message, Role};
use retrograd_agent_core::scenario::RolloutLimits;

const CALL: i32 = 500;
const BAD: i32 = 501;
const ANSWER: i32 = 502;

/// Reads its script off the opening the environment wrote: `mode:call` calls
/// once then answers, `mode:bad` writes a malformed call first, `mode:loop`
/// never stops calling.
struct ScriptPolicy;

fn bytes(text: &str) -> impl Iterator<Item = i32> + '_ {
    text.bytes().map(i32::from)
}

#[async_trait]
impl Policy for ScriptPolicy {
    /// One piece per gap: the messages in it, byte for byte, so an observation
    /// costs its length and a re-render never moves a committed piece.
    async fn render_chat_framing(
        &self,
        messages: &[Message],
        _tools: &[ToolSpec],
        _add: bool,
    ) -> Result<Vec<Vec<i32>>> {
        let mut pieces = vec![vec![1]];
        for message in messages {
            match message.role {
                Role::Assistant => pieces.push(vec![20]),
                _ => pieces
                    .last_mut()
                    .expect("opened above")
                    .extend(bytes(&message.content)),
            }
        }
        Ok(pieces)
    }

    async fn generate_shared(
        &self,
        prompt: Vec<i32>,
        sampling: Vec<SamplingParams>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(sampling.iter().map(|_| generation(&prompt)).collect())
    }

    async fn generate_continuous(
        &self,
        requests: Vec<(Vec<i32>, SamplingParams)>,
    ) -> Result<Vec<PolicyGeneration>> {
        Ok(requests
            .iter()
            .map(|(prompt, _)| generation(prompt))
            .collect())
    }

    async fn score_masked(&self, _tokens: Vec<i32>, train_mask: Vec<bool>) -> Result<Vec<f32>> {
        Ok(vec![
            -0.5;
            train_mask.iter().filter(|&&trained| trained).count()
        ])
    }
}

fn generation(prompt: &[i32]) -> PolicyGeneration {
    let text = prompt
        .iter()
        .filter_map(|&token| u8::try_from(token).ok())
        .map(char::from)
        .collect::<String>();
    let done = prompt
        .iter()
        .filter(|&&token| token == CALL || token == BAD)
        .count();
    let (token, text) = match (text.contains("mode:loop"), text.contains("mode:bad"), done) {
        (true, _, _) | (false, false, 0) | (false, true, 1) => (
            CALL,
            r#"<tool_call>{"name":"echo","arguments":{"text":"hi"}}</tool_call>"#,
        ),
        (false, true, 0) => (BAD, "<tool_call>{bad}</tool_call>"),
        _ => (ANSWER, "done"),
    };
    PolicyGeneration {
        tokens: vec![token],
        text: text.into(),
        stopped_at_eog: true,
    }
}

/// What one member of a group does and earns, chosen by its seed.
#[derive(Clone)]
pub(super) struct Plan {
    mode: &'static str,
    /// The environment's reward for a call; `None` leaves grading to a judge.
    reward: Option<f32>,
    observation: &'static str,
    verified: bool,
}

pub(super) fn plan(mode: &'static str, reward: Option<f32>, observation: &'static str) -> Plan {
    Plan {
        mode,
        reward,
        observation,
        verified: true,
    }
}

#[derive(Default)]
pub(super) struct ScriptedFactory {
    pub(super) plans: Vec<Plan>,
    pub(super) created: AtomicUsize,
}

struct ScriptedEnvironment {
    plans: Vec<Plan>,
    plan: Option<Plan>,
}

#[async_trait]
impl EnvironmentFactory for ScriptedFactory {
    async fn create(&self) -> Result<Box<dyn Environment>> {
        self.created.fetch_add(1, Ordering::Relaxed);
        Ok(Box::new(ScriptedEnvironment {
            plans: self.plans.clone(),
            plan: None,
        }))
    }
}

#[async_trait]
impl Environment for ScriptedEnvironment {
    async fn reset(&mut self, _scenario: &Scenario, seed: u64) -> Result<Option<String>> {
        let plan = self.plans[seed as usize % self.plans.len()].clone();
        let opening = format!("mode:{}", plan.mode);
        self.plan = Some(plan);
        Ok(Some(opening))
    }

    async fn step(&mut self, call: &ToolCall) -> Result<StepOutcome> {
        let plan = self.plan.as_ref().expect("reset first");
        Ok(StepOutcome {
            result: ToolResult::ok(call.id.clone(), plan.observation),
            reward: plan.reward,
            done: false,
        })
    }

    async fn tools(&self, _scenario: &Scenario) -> Result<Vec<ToolSpec>> {
        Ok(vec![ToolSpec {
            name: "echo".into(),
            description: "echoes".into(),
            input_schema: json!({"type": "object"}),
        }])
    }

    async fn state(&mut self) -> Result<EnvState> {
        let plan = self.plan.as_ref().expect("reset first");
        Ok(EnvState {
            metadata: match (plan.verified, plan.reward) {
                (true, Some(_)) => json!({"verification": "passed"}),
                (false, Some(_)) => json!({"verification": "failed"}),
                (_, None) => serde_json::Value::Null,
            },
            ..Default::default()
        })
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Scores the members in order from `scores`, and counts its calls.
#[derive(Default)]
struct ListJudge {
    scores: Vec<f32>,
    calls: AtomicUsize,
}

#[async_trait]
impl RewardBackend for ListJudge {
    async fn score_group(&self, group: &TrajectoryGroup) -> Result<Vec<Score>> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        Ok(group
            .trajectories
            .iter()
            .map(|trajectory| Score {
                value: self.scores[trajectory.provenance.as_ref().unwrap().member],
                valid: true,
                explanation: None,
                error: None,
            })
            .collect())
    }
}

pub(super) fn scenario() -> Scenario {
    Scenario {
        id: "task".into(),
        system: Some("be useful".into()),
        user: "go".into(),
        metadata: Default::default(),
    }
}

pub(super) fn config(k: usize, keep: usize) -> CollectConfig {
    CollectConfig {
        k,
        keep,
        min_reward: None,
        require_verified: false,
        form: AssistantForm::Structured,
        seed: 0,
        judge_failure: JudgeFailurePolicy::DropGroup,
        environment_grades: true,
        generator: "model.gguf".into(),
        pairs: None,
    }
}

fn engine(factory: Arc<ScriptedFactory>) -> RolloutEngine {
    RolloutEngine::with_environments(
        Arc::new(ScriptPolicy),
        factory,
        Arc::new(HermesToolCallParser),
        RolloutLimits {
            max_turns: 4,
            max_new_tokens_per_turn: 4,
            max_trajectory_tokens: 1024,
            ..Default::default()
        },
    )
    .unwrap()
}

async fn run(
    plans: Vec<Plan>,
    reward: Option<Arc<dyn RewardBackend>>,
    config: &CollectConfig,
) -> (Result<CollectStats>, Vec<ChatExample>, Arc<ScriptedFactory>) {
    let factory = Arc::new(ScriptedFactory {
        plans,
        ..Default::default()
    });
    let mut records = Vec::new();
    let stats = collect_trajectories(
        &engine(factory.clone()),
        reward,
        &[scenario()],
        config,
        &mut records,
    )
    .await;
    (stats, records, factory)
}

fn members(records: &[ChatExample]) -> Vec<u64> {
    records
        .iter()
        .map(|record| record.metadata["member"].as_u64().unwrap())
        .collect()
}

#[tokio::test]
async fn the_best_distinct_traces_are_kept_the_shortest_first() {
    let (stats, records, _) = run(
        vec![
            plan("call", Some(1.0), "ok"),
            plan("call", Some(0.0), "no"),
            // The same trace as member 0.
            plan("call", Some(1.0), "ok"),
            plan("call", Some(1.0), "ok, and a longer observation"),
        ],
        None,
        &config(4, 2),
    )
    .await;
    let stats = stats.unwrap();
    assert_eq!(members(&records), [0, 3]);
    assert_eq!(stats.kept, 2);
    assert_eq!(stats.rejected.duplicate, 1);
    assert_eq!(stats.rejected.over_keep, 1);
    assert_eq!(stats.attempted, 4);
    assert_eq!(
        stats.scenarios,
        [ScenarioReport {
            id: "task".into(),
            attempted: 4,
            passed: 4,
            kept: 2,
            pass_rate: 1.0,
        }]
    );
    let metadata = &records[0].metadata;
    assert_eq!(metadata["scenario_id"], "task");
    assert_eq!(metadata["reward"], 1.0);
    assert_eq!(metadata["verification"], "passed");
    assert_eq!(metadata["turns"], 2);
    assert_eq!(metadata["generator"], "model.gguf");
    assert_eq!(metadata["form"], "structured");
    assert_eq!(metadata["seed"], 0);
}

#[tokio::test]
async fn every_filter_counts_what_it_rejects() {
    let unverified = Plan {
        verified: false,
        ..plan("call", Some(1.0), "ok")
    };
    let mut config = config(5, 5);
    config.min_reward = Some(1.0);
    config.require_verified = true;
    let (stats, records, _) = run(
        vec![
            plan("loop", Some(1.0), "ok"),
            plan("bad", Some(1.0), "ok"),
            plan("call", Some(0.5), "ok"),
            unverified,
            plan("call", Some(1.0), "verified"),
        ],
        None,
        &config,
    )
    .await;
    let stats = stats.unwrap();
    assert_eq!(members(&records), [4]);
    assert_eq!(
        stats.rejected,
        Rejections {
            truncated: 1,
            invalid_turns: 1,
            below_min_reward: 1,
            unverified: 1,
            ..Default::default()
        }
    );
    assert_eq!(stats.scenarios[0].passed, 1);
    assert_eq!(stats.scenarios[0].pass_rate, 0.2);
    assert_eq!(stats.unsolved().count(), 0);
}

#[tokio::test]
async fn a_group_its_environment_graded_never_reaches_the_judge() {
    let judge = Arc::new(ListJudge {
        scores: vec![0.0, 1.0],
        ..Default::default()
    });
    let (stats, records, _) = run(
        vec![plan("call", Some(1.0), "a"), plan("call", Some(0.0), "b")],
        Some(judge.clone()),
        &config(2, 1),
    )
    .await;
    stats.unwrap();
    assert_eq!(judge.calls.load(Ordering::Relaxed), 0);
    assert_eq!(members(&records), [0], "the environment's reward decides");

    // Nothing graded by the environment: the judge decides.
    let (stats, records, _) = run(
        vec![plan("call", None, "a"), plan("call", None, "b")],
        Some(judge.clone()),
        &config(2, 1),
    )
    .await;
    stats.unwrap();
    assert_eq!(judge.calls.load(Ordering::Relaxed), 1);
    assert_eq!(members(&records), [1]);
}

#[tokio::test]
async fn a_collection_nothing_can_grade_is_refused_before_any_rollout() {
    let mut config = config(2, 1);
    config.environment_grades = false;
    let (stats, records, factory) = run(vec![plan("call", Some(1.0), "a")], None, &config).await;
    let error = stats.unwrap_err();
    assert!(error.to_string().contains("nothing grades"), "{error}");
    assert!(records.is_empty());
    assert_eq!(factory.created.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn what_is_written_reads_back_as_a_valid_dataset() {
    let (stats, records, _) = run(
        vec![plan("call", Some(1.0), "a"), plan("call", Some(1.0), "b")],
        None,
        &config(2, 2),
    )
    .await;
    stats.unwrap();
    let path = std::env::temp_dir().join(format!("retrograd-collect-{}.jsonl", std::process::id()));
    let lines = records
        .iter()
        .map(|record| serde_json::to_string(record).unwrap() + "\n")
        .collect::<String>();
    std::fs::write(&path, lines).unwrap();
    let read = retrograd_dataset::read_chat_jsonl(&path).unwrap();
    std::fs::remove_file(&path).unwrap();
    assert_eq!(read.len(), 2);
    for record in read {
        record.example.validate().unwrap();
        assert!(record.example.is_tool_record());
        assert_eq!(
            record.example.messages[0].content, "be useful",
            "the catalog is in tools, not in the system turn"
        );
    }
}

#[tokio::test]
async fn a_collection_is_reproducible() {
    let plans = || {
        vec![
            plan("call", Some(1.0), "a"),
            plan("call", Some(0.5), "bb"),
            plan("bad", Some(1.0), "c"),
        ]
    };
    let (_, first, _) = run(plans(), None, &config(3, 3)).await;
    let (_, second, _) = run(plans(), None, &config(3, 3)).await;
    assert!(!first.is_empty());
    assert_eq!(
        serde_json::to_string(&first).unwrap(),
        serde_json::to_string(&second).unwrap()
    );
}

/// A sink that keeps the pairs, and refuses a plain record.
#[derive(Default)]
struct Pairs(Vec<PreferenceExample>);

impl CollectSink for Pairs {
    fn record(&mut self, _example: ChatExample) -> Result<()> {
        panic!("pairs mode writes no plain record")
    }

    fn record_pair(&mut self, pair: PreferenceExample) -> Result<()> {
        self.0.push(pair);
        Ok(())
    }
}

fn pairs_config(k: usize, min_gap: f32) -> CollectConfig {
    CollectConfig {
        pairs: Some(PairsConfig { min_gap }),
        ..config(k, 1)
    }
}

async fn run_pairs(
    plans: Vec<Plan>,
    config: &CollectConfig,
) -> (CollectStats, Vec<PreferenceExample>) {
    let factory = Arc::new(ScriptedFactory {
        plans,
        ..Default::default()
    });
    let mut pairs = Pairs::default();
    let stats = collect_trajectories(&engine(factory), None, &[scenario()], config, &mut pairs)
        .await
        .unwrap();
    (stats, pairs.0)
}

#[tokio::test]
async fn the_best_trace_is_paired_with_the_worst_complete_one() {
    let (stats, pairs) = run_pairs(
        vec![
            plan("call", Some(0.5), "half"),
            plan("call", Some(1.0), "right"),
            // Never finishes: truncated, so never the rejected side.
            plan("loop", Some(0.0), "x"),
            // A malformed call: the worst, but its error observation answers
            // no call, so it cannot be written as a record.
            plan("bad", Some(0.0), "wrong"),
        ],
        &pairs_config(4, 0.5),
    )
    .await;
    assert_eq!(pairs.len(), 1);
    assert_eq!(stats.kept, 1);
    let pair = &pairs[0];
    pair.validate().unwrap();
    assert_eq!(pair.metadata["chosen_reward"], 1.0);
    assert_eq!(pair.metadata["chosen_seed"], 1);
    // The next worst stands in for the one that cannot be written.
    assert_eq!(pair.metadata["rejected_reward"], 0.5);
    assert_eq!(pair.metadata["rejected_seed"], 0);
    assert_eq!(pair.metadata["scenario_id"], "task");
    // The prompt is what both traces open with, up to the first answer.
    assert_eq!(pair.prompt.last().unwrap().role, "user");
    assert_eq!(pair.chosen[0].role, "assistant");
    assert!(!pair.tools.is_empty(), "the catalog is carried");
    assert_ne!(pair.chosen, pair.rejected);
}

#[tokio::test]
async fn a_group_that_prefers_too_little_gives_no_pair() {
    let (stats, pairs) = run_pairs(
        vec![plan("call", Some(1.0), "a"), plan("call", Some(0.8), "b")],
        &pairs_config(2, 0.5),
    )
    .await;
    assert!(pairs.is_empty());
    assert_eq!(stats.pairs.no_gap, 1);
    assert_eq!(stats.kept, 0);

    // Everything failed the filters: nothing to prefer.
    let mut config = pairs_config(2, 0.5);
    config.min_reward = Some(2.0);
    let (stats, pairs) = run_pairs(
        vec![plan("call", Some(1.0), "a"), plan("call", Some(0.0), "b")],
        &config,
    )
    .await;
    assert!(pairs.is_empty());
    assert_eq!(stats.pairs.no_gap, 1);

    // One graded trace is nothing to compare it with.
    let (stats, _) = run_pairs(
        vec![plan("call", Some(1.0), "a"), plan("loop", Some(0.0), "b")],
        &pairs_config(2, 0.5),
    )
    .await;
    assert_eq!(stats.pairs.single_member, 1);
}

#[test]
fn a_pair_is_cut_where_both_traces_are_answered_next() {
    let message = |role: &str, content: &str| ChatMessage::text(role, content);
    let record = |messages: Vec<ChatMessage>| ChatExample {
        messages,
        ..ChatExample::default()
    };
    let chosen = record(vec![
        message("system", "s"),
        message("user", "u"),
        message("assistant", "a"),
        message("user", "again"),
        message("assistant", "good"),
    ]);
    let rejected = record(vec![
        message("system", "s"),
        message("user", "u"),
        message("assistant", "a"),
        message("user", "again"),
        message("assistant", "bad"),
    ]);
    let pair = preference_pair(&chosen, &rejected).unwrap();
    assert_eq!(pair.prompt.len(), 4, "the shared turns are context");
    assert_eq!(pair.chosen, [message("assistant", "good")]);
    pair.validate().unwrap();

    let other = record(vec![
        message("user", "something else"),
        message("assistant", "x"),
    ]);
    assert!(preference_pair(&chosen, &other).is_none());
}

#[test]
fn pairs_mode_needs_two_attempts_and_one_pair_per_scenario() {
    let scenarios = [scenario()];
    let error = check_config(&pairs_config(1, 0.5), false, &scenarios).unwrap_err();
    assert!(error.to_string().contains("two attempts"), "{error}");
    let mut config = pairs_config(2, 0.5);
    config.keep = 2;
    let error = check_config(&config, false, &scenarios).unwrap_err();
    assert!(error.to_string().contains("--keep"), "{error}");
    let error = check_config(&pairs_config(2, f32::NAN), false, &scenarios).unwrap_err();
    assert!(error.to_string().contains("--min-gap"), "{error}");
}
