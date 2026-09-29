//! `retrograd-server view`: one directory's trajectories, read-only.

use http::StatusCode;
use retrograd_observe::{Algorithm, ObserveSink, RunInfo, SinkConfig};
use retrograd_server::build_viewer_router;
use serde_json::json;

use crate::support::*;

fn export(directory: &std::path::Path) {
    let mut sink = ObserveSink::open(
        &SinkConfig {
            directory: directory.to_path_buf(),
            every: 1,
            max_text_chars: 0,
        },
        RunInfo {
            algorithm: Algorithm::Grpo,
            model: "model.gguf".into(),
            resumed_from_update: None,
            params: Default::default(),
        },
    )
    .expect("open the sink");
    sink.update_summary(1, [("reward/mean", 0.5)]);
    sink.finish();
}

#[tokio::test]
async fn the_viewer_serves_one_run_and_says_what_it_is() {
    let fixture = Fixture::new("viewer-serves");
    let directory = fixture.dir.join("observe");
    export(&directory);
    let router = build_viewer_router(directory);

    let (status, body) = get(&router, "/v1/capabilities").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["features"]["mode"], "viewer");
    assert_eq!(body["features"]["auth"], false);

    let (status, body) = get(&router, "/v1/health").await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, body) = get(&router, "/v1/runs/local/trajectories").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["observed"], true);
    assert_eq!(body["updates"][0]["metrics"]["reward/mean"], 0.5);

    let (status, body) = get(&router, "/v1/runs/local/trajectories/updates/1").await;
    assert_eq!(status, StatusCode::OK, "{body}");

    let (status, _) = get(&router, "/v1/runs/other/trajectories").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn nothing_can_be_written_through_the_viewer() {
    let fixture = Fixture::new("viewer-read-only");
    let directory = fixture.dir.join("observe");
    export(&directory);
    let router = build_viewer_router(directory);

    for uri in [
        "/v1/runs",
        "/v1/plan",
        "/v1/datasets",
        "/v1/runs/local/pause",
    ] {
        let (status, body) = post(&router, uri, json!({})).await;
        assert!(
            status == StatusCode::NOT_FOUND || status == StatusCode::METHOD_NOT_ALLOWED,
            "{uri}: {status} {body}"
        );
    }
    let (status, _) = post(&router, "/v1/runs/local/trajectories", json!({})).await;
    assert_eq!(status, StatusCode::METHOD_NOT_ALLOWED);
    let (status, _) = get(&router, "/v1/runs").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn a_directory_without_a_log_is_not_an_export() {
    let fixture = Fixture::new("viewer-empty");
    let router = build_viewer_router(fixture.dir.clone());
    let (status, body) = get(&router, "/v1/runs/local/trajectories").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["observed"], false);
}
