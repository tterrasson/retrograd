//! Step 9: `fork_from`  - resume and branch, which are one operation.
//!
//! The two shapes differ in who supplies the configuration: `{"run": …}` reuses
//! the parent's, `{"path": …}` needs one in the body. Both resolve to a
//! `checkpoint.resume_from` override, and both are refused when the pair would
//! continue on a different trajectory.

mod support;

use http::StatusCode;
use serde_json::json;
use support::*;

/// A finished parent with a checkpoint whose manifest matches its own
/// configuration - which is what a fork needs to be *accepted*, and which means
/// the fingerprint has to be computed the way the server computes it.
async fn parent(fixture: &Fixture, name: &str) -> (axum::Router, String, std::path::PathBuf) {
    let router = router_for(fixture, FakeEngine::succeeding());
    let (status, created) = post(&router, "/v1/runs", recipe_with_schedules(fixture, name)).await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let id = created["id"].as_str().expect("an id").to_string();
    let view = wait_for_terminal(&router, &id).await;

    let directory = fixture.dir.join("ckpt");
    std::fs::create_dir_all(&directory).expect("the checkpoint directory");
    let signature = trajectory_of(&view["effective_config"]);
    write_checkpoint(
        &directory,
        "step-000000000010",
        10,
        "sft",
        &signature,
        &fixture.dir.join("model.gguf"),
    );
    write_checkpoint(
        &directory,
        "step-000000000020",
        20,
        "sft",
        &signature,
        &fixture.dir.join("model.gguf"),
    );
    (router, id, directory)
}

#[tokio::test]
async fn a_fork_that_drops_the_checkpoint_reference_is_refused_before_startup() {
    let fixture = Fixture::new("fork-reference");
    let (router, id, directory) = parent(&fixture, "parent").await;
    let path = directory.join("step-000000000020.state");
    let mut checkpoint = retrograd_checkpoint::Checkpoint::read(&path).unwrap();
    checkpoint.manifest.reference_fingerprint = "saved-anchor".into();
    checkpoint
        .write(&path, |paths| {
            std::fs::write(paths.adapter.as_ref().unwrap(), b"not a real adapter")
                .map_err(Into::into)
        })
        .unwrap();
    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"run": id}}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error["pointer"] == "/config/reference/model"),
        "{body}"
    );
}

#[tokio::test]
async fn a_fork_of_a_run_reuses_its_configuration_and_resumes_the_latest_checkpoint() {
    let fixture = Fixture::new("fork-run");
    let (router, id, directory) = parent(&fixture, "parent").await;

    // The whole body: no recipe, no config. The parent supplies both.
    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"run": id}, "name": "child"}),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["name"], "child");
    // Latest by step, the same resolution `--resume` performs.
    assert_eq!(
        body["effective_config"]["checkpoint"]["resume_from"],
        directory
            .join("step-000000000020.state")
            .to_string_lossy()
            .as_ref()
    );
    // A fork's resume path is an override, so it reads as one in the provenance,
    // invariant 6 without a special case.
    assert_eq!(
        body["provenance"]["checkpoint.resume_from"]["source"], "override",
        "{}",
        body["provenance"]
    );
    // And the parent's own choices came through unchanged.
    assert_eq!(
        body["effective_config"]["sft"]["data"],
        fixture.path("data.jsonl")
    );
}

#[tokio::test]
async fn a_fork_can_name_the_checkpoint_and_change_what_is_not_the_trajectory() {
    let fixture = Fixture::new("fork-named");
    let (router, id, directory) = parent(&fixture, "parent").await;

    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({
            "fork_from": {"run": id, "checkpoint": "step-000000000010"},
            // A checkpoint cadence is not part of the trajectory, so changing it
            // is a legitimate fork of the same run.
            "params": {"checkpoint": {"every_steps": 5}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["effective_config"]["checkpoint"]["resume_from"],
        directory
            .join("step-000000000010.state")
            .to_string_lossy()
            .as_ref()
    );
    assert_eq!(body["effective_config"]["checkpoint"]["every_steps"], 5);

    // The id as the listing reports it, and the directory name as it is on disk:
    // both are what a client has in hand.
    let (status, _) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"run": id, "checkpoint": "step-000000000010.state"}}),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn a_fork_that_changes_the_trajectory_is_refused_and_says_which_field() {
    let fixture = Fixture::new("fork-trajectory");
    let (router, id, _) = parent(&fixture, "parent").await;

    // The learning rate is part of what a resume must find unchanged: continuing
    // an optimizer's state onto a different schedule is a different run.
    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({
            "fork_from": {"run": id},
            "params": {"training": {"lr": 0.000123}}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    let pointers: Vec<&str> = body["errors"]
        .as_array()
        .expect("field errors")
        .iter()
        .map(|error| error["pointer"].as_str().unwrap_or_default())
        .collect();
    assert!(
        pointers.contains(&"/params/training/lr"),
        "the refusal names the parameter that caused it: {body}"
    );
}

#[tokio::test]
async fn a_fork_of_a_bare_path_needs_a_configuration() {
    let fixture = Fixture::new("fork-path");
    let (router, id, directory) = parent(&fixture, "parent").await;
    let state_dir = directory.join("step-000000000020.state");

    // A checkpoint directory carries a manifest and no configuration, so this
    // cannot be a complete request - and saying that is more use than resolving
    // something the caller did not describe.
    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"path": state_dir.to_string_lossy()}}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["errors"][0]["pointer"], "/fork_from/path");

    // With the configuration alongside it, the same path is accepted: this is how
    // a checkpoint written by the CLI is picked up.
    let (_, parent_view) = get(&router, &format!("/v1/runs/{id}")).await;
    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({
            "config": parent_view["effective_config"],
            "fork_from": {"path": state_dir.to_string_lossy()}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["effective_config"]["checkpoint"]["resume_from"],
        state_dir.to_string_lossy().as_ref()
    );

    // A directory of checkpoints is as good an answer as one checkpoint: it is
    // what `--resume` with no path means.
    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({
            "config": parent_view["effective_config"],
            "fork_from": {"path": directory.to_string_lossy()}
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(
        body["effective_config"]["checkpoint"]["resume_from"],
        directory
            .join("step-000000000020.state")
            .to_string_lossy()
            .as_ref()
    );
}

#[tokio::test]
async fn a_fork_refuses_what_it_cannot_resolve() {
    let fixture = Fixture::new("fork-errors");
    let (router, id, directory) = parent(&fixture, "parent").await;

    for (body, status) in [
        // Neither half.
        (json!({"fork_from": {}}), StatusCode::UNPROCESSABLE_ENTITY),
        // Both halves.
        (
            json!({"fork_from": {"run": id.clone(), "path": directory.to_string_lossy()}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        // A run that does not exist.
        (
            json!({"fork_from": {"run": "00000000-0000-4000-8000-000000000000"}}),
            StatusCode::NOT_FOUND,
        ),
        // A checkpoint that does not exist.
        (
            json!({"fork_from": {"run": id.clone(), "checkpoint": "step-000000009999"}}),
            StatusCode::NOT_FOUND,
        ),
        // An id that is a path. The root check happens before this one, so without
        // the name check a caller could walk out of a directory it was allowed to
        // name.
        (
            json!({"fork_from": {"run": id.clone(), "checkpoint": "../../etc"}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        // Saying the same thing twice.
        (
            json!({
                "fork_from": {"run": id.clone()},
                "params": {"checkpoint": {"resume_from": "/somewhere/else.state"}}
            }),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
        // A fork *and* a base of its own, from a run that already has one.
        (
            json!({"fork_from": {"run": id.clone()}, "recipe": {}}),
            StatusCode::UNPROCESSABLE_ENTITY,
        ),
    ] {
        let (answer, rendered) = post(&router, "/v1/runs?dry_run=true", body.clone()).await;
        assert_eq!(answer, status, "{body} -> {rendered}");
    }
}

#[tokio::test]
async fn a_fork_of_a_run_with_no_checkpoints_has_nothing_to_continue_from() {
    let fixture = Fixture::new("fork-no-checkpoints");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "plain")).await;
    let id = created["id"].as_str().expect("an id").to_string();
    wait_for_terminal(&router, &id).await;

    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"run": id}}),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert_eq!(body["errors"][0]["pointer"], "/fork_from/run");
}

#[tokio::test]
async fn an_incomplete_checkpoint_is_not_forkable() {
    let fixture = Fixture::new("fork-incomplete");
    let (router, id, directory) = parent(&fixture, "parent").await;
    // A write that was interrupted before the manifest landed. `latest_checkpoint`
    // picks it because it has the highest step; the manifest read is what catches
    // it, and 422 rather than 404 because the directory *is* there.
    std::fs::create_dir_all(directory.join("step-000000000030.state")).expect("a partial write");

    let (status, body) = post(
        &router,
        "/v1/runs?dry_run=true",
        json!({"fork_from": {"run": id}}),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body}");
    assert_eq!(body["errors"][0]["pointer"], "/fork_from");
}

#[tokio::test]
async fn a_forked_run_actually_starts_from_the_checkpoint() {
    let fixture = Fixture::new("fork-create");
    let (router, id, directory) = parent(&fixture, "parent").await;

    let (status, created) = post(
        &router,
        "/v1/runs",
        json!({"fork_from": {"run": id}, "name": "child"}),
    )
    .await;
    assert_eq!(status, StatusCode::CREATED, "{created}");
    let child = created["id"].as_str().expect("an id").to_string();
    assert_ne!(child, id, "a fork is a new run, not a restart");
    assert_eq!(
        created["effective_config"]["checkpoint"]["resume_from"],
        directory
            .join("step-000000000020.state")
            .to_string_lossy()
            .as_ref()
    );
    let view = wait_for_terminal(&router, &child).await;
    assert_eq!(view["status"], "completed", "{view}");
    // Both runs are in the history, which is what makes a fork a branch rather
    // than a continuation of one record.
    let (_, listing) = get(&router, "/v1/runs").await;
    assert_eq!(listing["runs"].as_array().expect("a list").len(), 2);
}
