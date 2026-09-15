//! Step 9: what a run left behind - checkpoints, artefacts, download, purge.
//!
//! Every assertion here is about the *filesystem*, which is the point of the
//! design: these routes read the directory as it is now, so they answer for a run
//! this process only restored from disk, and a checkpoint removed by hand
//! disappears from the listing without anything being told.

mod support;

use http::StatusCode;
use serde_json::json;
use support::*;

/// A finished run with a checkpoint directory, and its id.
async fn finished(fixture: &Fixture, name: &str) -> (axum::Router, String, serde_json::Value) {
    let router = router_for(fixture, FakeEngine::succeeding());
    let (status, body) = post(&router, "/v1/runs", recipe_with_schedules(fixture, name)).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().expect("an id").to_string();
    let view = wait_for_terminal(&router, &id).await;
    (router, id, view)
}

#[tokio::test]
async fn the_checkpoint_listing_is_a_directory_read_newest_first() {
    let fixture = Fixture::new("artifacts-checkpoints");
    let (router, id, _) = finished(&fixture, "ckpt").await;
    let directory = fixture.dir.join("ckpt");
    std::fs::create_dir_all(&directory).expect("the checkpoint directory");
    write_checkpoint(
        &directory,
        "step-000000000010",
        10,
        "sft",
        "signature",
        &fixture.dir.join("model.gguf"),
    );
    write_checkpoint(
        &directory,
        "step-000000000020",
        20,
        "sft",
        "signature",
        &fixture.dir.join("model.gguf"),
    );
    write_checkpoint(
        &directory,
        "best",
        15,
        "sft",
        "signature",
        &fixture.dir.join("model.gguf"),
    );
    // Not a checkpoint: no manifest. Listed, not hidden - a client that asked to
    // resume from it deserves to know why it cannot.
    std::fs::create_dir_all(directory.join("step-000000000005.state")).expect("a partial write");

    let (status, body) = get(&router, &format!("/v1/runs/{id}/checkpoints")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entries = body["checkpoints"].as_array().expect("a list");
    let ids: Vec<&str> = entries
        .iter()
        .map(|entry| entry["id"].as_str().unwrap_or_default())
        .collect();
    assert_eq!(
        ids,
        [
            "step-000000000020",
            "best",
            "step-000000000010",
            "step-000000000005"
        ],
        "newest step first, ties broken by id: {body}"
    );
    let newest = &entries[0];
    assert_eq!(newest["kind"], "step");
    assert_eq!(newest["global_step"], 20);
    assert_eq!(newest["complete"], true);
    assert!(newest["bytes"].as_u64().expect("a size") > 0, "{newest}");
    assert!(newest["written_at"].as_u64().is_some(), "{newest}");
    // The sibling GGUF export `Checkpoint::write` publishes.
    assert!(
        newest["adapter"]
            .as_str()
            .unwrap_or_default()
            .ends_with(".gguf"),
        "{newest}"
    );
    assert_eq!(entries[1]["kind"], "best");
    let partial = entries
        .iter()
        .find(|e| e["id"] == "step-000000000005")
        .unwrap();
    assert_eq!(partial["complete"], false, "{partial}");
    // The default a fork would pick, resolved the way `--resume` resolves it.
    assert_eq!(body["latest"], "step-000000000020");
    assert!(body["directory"].as_str().is_some(), "{body}");
}

#[tokio::test]
async fn a_run_without_a_checkpoint_directory_says_so_rather_than_404() {
    let fixture = Fixture::new("artifacts-no-checkpoints");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "plain")).await;
    let id = created["id"].as_str().expect("an id").to_string();
    wait_for_terminal(&router, &id).await;

    let (status, body) = get(&router, &format!("/v1/runs/{id}/checkpoints")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Not the same as an empty directory: there is no directory.
    assert!(body.get("directory").is_none(), "{body}");
    assert!(body["checkpoints"].as_array().expect("a list").is_empty());
    assert!(body.get("latest").is_none(), "{body}");
}

#[tokio::test]
async fn the_artifact_inventory_names_what_the_run_writes() {
    let fixture = Fixture::new("artifacts-inventory");
    let (router, id, _) = finished(&fixture, "inventory").await;

    let (status, body) = get(&router, &format!("/v1/runs/{id}/artifacts")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let names: Vec<&str> = body["artifacts"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|entry| entry["name"].as_str().unwrap_or_default())
        .collect();
    // A fixed order, so a client diffing two listings sees changes and not
    // reordering. The adapter comes from the configuration; `run` and `events` are
    // the server's own and are always there.
    assert_eq!(names, ["adapter", "run", "events", "checkpoints"], "{body}");

    let events = body["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "events")
        .expect("the event log");
    assert_eq!(events["kind"], "file");
    assert_eq!(events["present"], true);
    assert_eq!(events["downloadable"], true);
    assert!(events["bytes"].as_u64().expect("a size") > 0);

    // The fake engine never saves an adapter, so the file is not there - and
    // saying so is more use than omitting the entry.
    let adapter = body["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["name"] == "adapter")
        .expect("the adapter");
    assert_eq!(adapter["present"], false, "{adapter}");
    assert_eq!(adapter["downloadable"], false, "{adapter}");
}

#[tokio::test]
async fn an_artifact_is_downloaded_by_name_and_nothing_else_is() {
    let fixture = Fixture::new("artifacts-download");
    let (router, id, _) = finished(&fixture, "download").await;

    let (status, content_type, bytes) = send_raw(
        &router,
        axum::http::Request::builder()
            .uri(format!("/v1/runs/{id}/artifacts/events"))
            .body(axum::body::Body::empty())
            .expect("the request"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/x-ndjson");
    let text = String::from_utf8(bytes).expect("utf-8");
    assert!(text.contains("\"type\":\"terminal\""), "{text}");

    // A name outside the inventory is a 404 that says what the inventory is.
    let (status, body) = get(&router, &format!("/v1/runs/{id}/artifacts/secrets")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("events"),
        "{body}"
    );
    // A traversal is not a name either: there is no path parameter to traverse.
    let (status, _) = get(
        &router,
        &format!("/v1/runs/{id}/artifacts/..%2F..%2Fetc%2Fpasswd"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    // A directory is not served inline; the listing route is.
    let (status, body) = get(&router, &format!("/v1/runs/{id}/artifacts/checkpoints")).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    // Neither is a file that is not there yet.
    let (status, body) = get(&router, &format!("/v1/runs/{id}/artifacts/adapter")).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
}

#[tokio::test]
async fn deleting_a_run_forgets_its_bookkeeping_and_keeps_the_client_s_files() {
    let fixture = Fixture::new("artifacts-delete");
    let (router, id, _) = finished(&fixture, "delete").await;
    let directory = fixture.dir.join("ckpt");
    std::fs::create_dir_all(&directory).expect("the checkpoint directory");
    write_checkpoint(
        &directory,
        "step-000000000010",
        10,
        "sft",
        "signature",
        &fixture.dir.join("model.gguf"),
    );
    let run_dir = fixture.state_dir().join(&id);
    assert!(run_dir.is_dir());

    let (status, _, body) = send_raw(
        &router,
        axum::http::Request::builder()
            .method("DELETE")
            .uri(format!("/v1/runs/{id}"))
            .body(axum::body::Body::empty())
            .expect("the request"),
    )
    .await;
    assert_eq!(status, StatusCode::NO_CONTENT);
    assert!(body.is_empty(), "a 204 has no body");

    // Gone from the registry and from the state directory.
    let (status, _) = get(&router, &format!("/v1/runs/{id}")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(!run_dir.exists(), "the run's own directory was removed");
    let (_, listing) = get(&router, "/v1/runs").await;
    assert!(
        listing["runs"].as_array().expect("a list").is_empty(),
        "{listing}"
    );

    // But the checkpoints are the client's, at a path the client chose, possibly
    // shared with another run. Purging a record must not delete someone's work.
    assert!(
        directory.join("step-000000000010.state").is_dir(),
        "the checkpoint survived the purge"
    );

    // And a second delete is a 404, not a 204: the run is gone, so there is
    // nothing to answer for.
    let (status, _) = send(
        &router,
        axum::http::Request::builder()
            .method("DELETE")
            .uri(format!("/v1/runs/{id}"))
            .body(axum::body::Body::empty())
            .expect("the request"),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_live_run_cannot_be_deleted() {
    let fixture = Fixture::new("artifacts-delete-live");
    let gate = std::sync::Arc::new(Gate::default());
    let router = router_for(&fixture, FakeEngine::gated(gate.clone()));
    let (_, created) = post(&router, "/v1/runs", recipe(&fixture, "live")).await;
    let id = created["id"].as_str().expect("an id").to_string();
    wait_for_status(&router, &id, "running").await;

    let (status, body) = send(
        &router,
        axum::http::Request::builder()
            .method("DELETE")
            .uri(format!("/v1/runs/{id}"))
            .body(axum::body::Body::empty())
            .expect("the request"),
    )
    .await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(
        body["detail"]
            .as_str()
            .unwrap_or_default()
            .contains("cancel it first"),
        "{body}"
    );

    gate.release();
    wait_for_terminal(&router, &id).await;
}

#[tokio::test]
async fn a_restored_run_still_knows_where_its_outputs_are() {
    let fixture = Fixture::new("artifacts-restored");
    let directory = fixture.dir.join("ckpt");
    {
        let (router, id, _) = finished(&fixture, "restored").await;
        std::fs::create_dir_all(&directory).expect("the checkpoint directory");
        write_checkpoint(
            &directory,
            "step-000000000030",
            30,
            "sft",
            "signature",
            &fixture.dir.join("model.gguf"),
        );
        drop(router);

        // A second server over the same state directory: nothing of this run is in
        // memory, only what `run.json` recorded. That is the case these routes
        // exist for, and the reason the artefact paths are stored typed rather than
        // re-derived from the rendered configuration.
        let reopened = router_for(&fixture, FakeEngine::succeeding());
        let (status, body) = get(&reopened, &format!("/v1/runs/{id}/checkpoints")).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["latest"], "step-000000000030", "{body}");
        let (_, artifacts) = get(&reopened, &format!("/v1/runs/{id}/artifacts")).await;
        let names: Vec<&str> = artifacts["artifacts"]
            .as_array()
            .expect("a list")
            .iter()
            .map(|entry| entry["name"].as_str().unwrap_or_default())
            .collect();
        assert_eq!(
            names,
            ["adapter", "run", "events", "checkpoints"],
            "{artifacts}"
        );
        // Its own history is still downloadable, which is what makes the event
        // log worth keeping beside `run.json`. Read raw, because an event log is
        // NDJSON and not one JSON document.
        let (status, content_type, bytes) = send_raw(
            &reopened,
            axum::http::Request::builder()
                .uri(format!("/v1/runs/{id}/artifacts/events"))
                .body(axum::body::Body::empty())
                .expect("the request"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(content_type, "application/x-ndjson");
        assert!(
            String::from_utf8_lossy(&bytes).contains("\"type\":\"terminal\""),
            "the restored run's history is still there"
        );
    }
    let _ = json!({});
}
