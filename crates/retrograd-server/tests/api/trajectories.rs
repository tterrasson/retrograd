//! `GET /v1/runs/{id}/trajectories…` over an export written by the real sink.
//!
//! The run is one the registry reads back from the state directory - a
//! `run.json` naming an `[observe]` directory - and the directory is filled by
//! `ObserveSink` itself, so what the routes read is the format as the trainer
//! writes it, not a fixture that agrees with the reader by construction.

use std::path::Path;

use http::StatusCode;
use retrograd_observe::{
    Algorithm, ObserveBatch, ObserveSink, ObservedMessage, ObservedPrompt, ObservedRollout,
    ObservedToolCall, OutcomeEntry, RolloutBatch, RolloutContent, RunInfo, SinkConfig, StepReward,
};
use serde_json::json;

use crate::support::*;

const RUN: &str = "6f1c1c1e-8a55-4a6e-9a1b-2a7f3b9c0d11";
const PLAIN_RUN: &str = "0b7d8e22-4a3c-4d5e-8f10-9a8b7c6d5e4f";

/// A finished run in the state directory, with `observe` as its export.
fn restored_run(fixture: &Fixture, id: &str, observe: Option<&Path>) {
    let directory = fixture.state_dir().join(id);
    std::fs::create_dir_all(&directory).expect("the run directory");
    let document = json!({
        "id": id,
        "status": "completed",
        "created_at": 1,
        "effective_config": {"run": {"algorithm": "agent_grpo"}, "model": {"path": "/m/qwen.gguf"}},
        "provenance": {},
        "plan": {},
        "artifacts": {"observe": observe},
    });
    std::fs::write(directory.join("run.json"), document.to_string()).expect("run.json");
}

fn sink(directory: &Path, every: u32, resumed_from_update: Option<u32>) -> ObserveSink {
    ObserveSink::open(
        &SinkConfig {
            directory: directory.to_path_buf(),
            every,
            max_text_chars: 0,
        },
        RunInfo {
            algorithm: Algorithm::AgentGrpo,
            model: "/models/qwen.gguf".into(),
            resumed_from_update,
            params: Default::default(),
        },
    )
    .expect("open the sink")
}

fn conversation(member: usize) -> RolloutContent {
    RolloutContent::Conversation {
        messages: vec![
            ObservedMessage {
                role: "assistant".into(),
                content: String::new(),
                tool_calls: vec![ObservedToolCall {
                    id: format!("call-{member}"),
                    name: "search".into(),
                    arguments: json!({"query": "rust"}),
                }],
                tool_call_id: None,
                is_error: false,
            },
            ObservedMessage {
                role: "tool".into(),
                content: "no result".into(),
                tool_calls: Vec::new(),
                tool_call_id: Some(format!("call-{member}")),
                is_error: member == 1,
            },
            ObservedMessage::text("assistant", format!("final answer {member}")),
        ],
        prefix: false,
        step_rewards: vec![StepReward {
            step_index: 0,
            kind: "tool".into(),
            reward: 0.25,
            message_indices: vec![0],
        }],
        terminal_reward_raw: Some(1.0),
        judge_explanation: Some("good".into()),
        metadata: Default::default(),
    }
}

fn rollouts(update: u32, groups: usize) -> ObserveBatch {
    let prompts = (0..groups)
        .map(|group| ObservedPrompt {
            key: format!("s:{group}"),
            messages: vec![ObservedMessage::text(
                "user",
                format!("task {group} ").repeat(200),
            )],
            reward_text: None,
            metadata: None,
        })
        .collect();
    let rollouts = (0..groups)
        .flat_map(|group| {
            (0..2).map(move |member| ObservedRollout {
                update,
                group: Some(group),
                member,
                prompt: format!("s:{group}"),
                seed: 7,
                tokens: 40,
                truncated: false,
                reward: Some(member as f32),
                reward_raw: Some(member as f32),
                judge_term: None,
                advantage: Some(if member == 0 { -1.0 } else { 1.0 }),
                advantage_min: None,
                advantage_max: None,
                eligible: Some(true),
                trained: None,
                skip_reason: None,
                content: conversation(member),
            })
        })
        .collect();
    ObserveBatch::Rollouts(RolloutBatch { prompts, rollouts })
}

/// Two updates, texts for the second only (`every = 2`), three groups.
fn write_export(directory: &Path) {
    let mut sink = sink(directory, 2, None);
    sink.update_summary(1, [("reward/mean", 0.25)]);
    sink.observer().observe(rollouts(2, 3));
    sink.observer().observe(ObserveBatch::Outcome {
        update: 2,
        entries: vec![OutcomeEntry {
            group: Some(0),
            member: 1,
            trained: true,
        }],
    });
    sink.update_summary(2, [("reward/mean", 0.5)]);
    sink.finish();
    assert!(sink.take_warnings().iter().all(|w| !w.contains("failed")));
}

fn router(fixture: &Fixture) -> axum::Router {
    build_router(state_with(
        fixture,
        FakeEngine::succeeding(),
        "calibrate_runs = false\n",
    ))
}

#[tokio::test]
async fn the_overview_lists_every_update_and_which_ones_carry_texts() {
    let fixture = Fixture::new("trajectories-overview");
    let observe = fixture.dir.join("observe");
    write_export(&observe);
    restored_run(&fixture, RUN, Some(&observe));
    let router = router(&fixture);

    let (status, body) = get(&router, &format!("/v1/runs/{RUN}/trajectories")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["observed"], true);
    assert_eq!(body["algorithm"], "agent_grpo");
    assert_eq!(body["model"], "qwen.gguf", "a file name, not a path");
    assert_eq!(body["every"], 2);
    assert_eq!(body["segments"], 1);
    let updates = body["updates"].as_array().expect("updates");
    assert_eq!(updates.len(), 2);
    assert_eq!(updates[0]["update"], 1);
    assert_eq!(updates[0]["texts"], false);
    assert_eq!(updates[0]["metrics"]["reward/mean"], 0.25);
    assert_eq!(updates[1]["texts"], true);
    assert_eq!(updates[1]["groups"], 3);
    assert_eq!(updates[1]["rollouts"], 6);
    assert_eq!(updates[1]["status"], "completed");
}

#[tokio::test]
async fn an_update_lists_its_groups_a_page_at_a_time_without_texts() {
    let fixture = Fixture::new("trajectories-update");
    let observe = fixture.dir.join("observe");
    write_export(&observe);
    restored_run(&fixture, RUN, Some(&observe));
    let router = router(&fixture);

    let uri = format!("/v1/runs/{RUN}/trajectories/updates/2?limit=2&preview_chars=10");
    let (status, body) = get(&router, &uri).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["summary"]["update"], 2);
    let groups = body["groups"].as_array().expect("groups");
    assert_eq!(groups.len(), 2);
    assert_eq!(body["next_cursor"], "2");
    let first = &groups[0];
    assert_eq!(first["group"], 0);
    assert_eq!(first["prompt"]["key"], "s:0");
    assert_eq!(first["prompt"]["cut"], true);
    assert_eq!(
        first["prompt"]["messages"][0]["content"]
            .as_str()
            .expect("content")
            .chars()
            .count(),
        10
    );
    assert_eq!(first["reward_mean"], 0.5);
    assert_eq!(first["reward_std"], 0.5);
    assert_eq!(first["trained"], 1);
    let member = &first["members"][1];
    assert_eq!(member["trained"], true);
    assert_eq!(member["turns"], 2);
    assert_eq!(member["tool_calls"], 1);
    assert_eq!(member["tool_errors"], 1);
    assert!(member.get("conversation").is_none(), "no text in a listing");

    let (status, body) = get(
        &router,
        &format!("/v1/runs/{RUN}/trajectories/updates/2?cursor=2&limit=2"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["groups"].as_array().expect("groups").len(), 1);
    assert!(body.get("next_cursor").is_none());

    // An update whose texts were not exported: its numbers, and no group.
    let (status, body) = get(&router, &format!("/v1/runs/{RUN}/trajectories/updates/1")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["groups"], json!([]));
    assert_eq!(body["summary"]["texts"], false);

    let (status, _) = get(&router, &format!("/v1/runs/{RUN}/trajectories/updates/9")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_group_carries_every_text_and_can_be_read_one_member_at_a_time() {
    let fixture = Fixture::new("trajectories-group");
    let observe = fixture.dir.join("observe");
    write_export(&observe);
    restored_run(&fixture, RUN, Some(&observe));
    let router = router(&fixture);

    let uri = format!("/v1/runs/{RUN}/trajectories/updates/2/groups/0");
    let (status, body) = get(&router, &uri).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["prompt"]["cut"], false);
    let members = body["members"].as_array().expect("members");
    assert_eq!(members.len(), 2);
    let conversation = &members[1]["conversation"];
    assert_eq!(
        conversation["messages"][0]["tool_calls"][0]["name"],
        "search"
    );
    assert_eq!(conversation["messages"][1]["tool_call_id"], "call-1");
    assert_eq!(conversation["messages"][1]["is_error"], true);
    assert_eq!(
        conversation["step_rewards"][0]["message_indices"],
        json!([0])
    );
    assert_eq!(conversation["judge_explanation"], "good");

    let (status, body) = get(&router, &format!("{uri}?member=1")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["members"].as_array().expect("members").len(), 1);

    // Groups are numbered: `-` is PPO's, and this export is not PPO.
    let (status, _) = get(
        &router,
        &format!("/v1/runs/{RUN}/trajectories/updates/2/groups/-"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_resume_hides_the_updates_it_replays() {
    let fixture = Fixture::new("trajectories-resume");
    let observe = fixture.dir.join("observe");
    let mut first = sink(&observe, 1, None);
    first.update_summary(1, []);
    first.update_summary(2, []);
    first.update_summary(3, []);
    first.finish();
    let mut second = sink(&observe, 1, Some(1));
    second.update_summary(2, [("reward/mean", 1.0)]);
    second.finish();
    restored_run(&fixture, RUN, Some(&observe));

    let (status, body) = get(&router(&fixture), &format!("/v1/runs/{RUN}/trajectories")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["segments"], 2);
    let seen: Vec<(u64, u64)> = body["updates"]
        .as_array()
        .expect("updates")
        .iter()
        .map(|update| {
            (
                update["update"].as_u64().expect("update"),
                update["segment"].as_u64().expect("segment"),
            )
        })
        .collect();
    assert_eq!(seen, [(1, 0), (2, 1)]);
}

#[tokio::test]
async fn a_run_without_an_export_says_so_rather_than_failing() {
    let fixture = Fixture::new("trajectories-none");
    restored_run(&fixture, PLAIN_RUN, None);
    let router = router(&fixture);

    let (status, body) = get(&router, &format!("/v1/runs/{PLAIN_RUN}/trajectories")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["observed"], false);
    assert_eq!(body["updates"], json!([]));

    let (status, _) = get(
        &router,
        &format!("/v1/runs/{PLAIN_RUN}/trajectories/updates/1"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, _) = get(&router, &format!("/v1/runs/{RUN}/trajectories")).await;
    assert_eq!(
        status,
        StatusCode::NOT_FOUND,
        "an unknown run is not an empty export"
    );
}

#[tokio::test]
async fn a_run_listing_says_which_runs_are_observed() {
    let fixture = Fixture::new("trajectories-listing");
    let observe = fixture.dir.join("observe");
    write_export(&observe);
    restored_run(&fixture, RUN, Some(&observe));
    restored_run(&fixture, PLAIN_RUN, None);
    let (status, body) = get(&router(&fixture), "/v1/runs").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let observed = |id: &str| {
        body["runs"]
            .as_array()
            .expect("runs")
            .iter()
            .find(|run| run["id"] == id)
            .map(|run| run["observed"].clone())
    };
    assert_eq!(observed(RUN), Some(json!(true)));
    assert_eq!(observed(PLAIN_RUN), Some(json!(false)));
}
