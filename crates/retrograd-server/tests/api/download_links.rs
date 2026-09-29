//! Signed download links: one artefact, one minute, in place of the token.

use http::StatusCode;
use serde_json::Value;

use crate::support::*;

const TOKEN: &str = "a-long-enough-test-token";

async fn finished_run(
    fixture: &Fixture,
    extra: &str,
) -> (axum::Router, retrograd_server::AppState, String) {
    let state = state_with(
        fixture,
        FakeEngine::succeeding(),
        &format!("calibrate_runs = false\n{extra}"),
    );
    let router = build_router(state.clone());
    let bearer = format!("Bearer {TOKEN}");
    let headers: &[(&str, &str)] = if extra.contains("auth_token") {
        &[("authorization", bearer.as_str())]
    } else {
        &[]
    };
    let (status, body) = post_with(&router, "/v1/runs", recipe(fixture, "links"), headers).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    let id = body["id"].as_str().expect("an id").to_string();
    loop {
        let (_, view) = get_with(&router, &format!("/v1/runs/{id}"), headers).await;
        if view["status"] == "completed" {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    (router, state, id)
}

fn href(body: &Value) -> String {
    body["href"].as_str().expect("an href").to_string()
}

#[tokio::test]
async fn a_link_opens_its_artifact_without_the_token() {
    let fixture = Fixture::new("links-valid");
    let (router, _, id) = finished_run(&fixture, &format!("auth_token = \"{TOKEN}\"\n")).await;
    let bearer = format!("Bearer {TOKEN}");

    let (status, body) = send(
        &router,
        http::Request::builder()
            .method("POST")
            .uri(format!("/v1/runs/{id}/artifacts/events/link"))
            .header("authorization", &bearer)
            .body(axum::body::Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let link = href(&body);
    assert!(link.starts_with(&format!("/v1/runs/{id}/artifacts/events?sig=")));
    assert!(body["expires_at"].as_u64().is_some());

    let (status, content_type, bytes) = send_raw(
        &router,
        http::Request::builder()
            .uri(&link)
            .body(axum::body::Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{content_type}");
    assert!(!bytes.is_empty());

    // Without the link, the same download still needs the token.
    let (status, _) = get(&router, &format!("/v1/runs/{id}/artifacts/events")).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn a_link_opens_nothing_but_its_own_artifact_and_not_after_it_expires() {
    let fixture = Fixture::new("links-refused");
    let (router, state, id) = finished_run(&fixture, &format!("auth_token = \"{TOKEN}\"\n")).await;
    let now = retrograd_server::runtime::unix_seconds();

    let expired = state.link_key.sign(&id, "events", now - 1);
    let (status, _) = get(
        &router,
        &format!(
            "/v1/runs/{id}/artifacts/events?sig={expired}&exp={}",
            now - 1
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "expired");

    let valid = state.link_key.sign(&id, "events", now + 60);
    let (status, _) = get(
        &router,
        &format!("/v1/runs/{id}/artifacts/run?sig={valid}&exp={}", now + 60),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "signed for another artifact"
    );

    let (status, _) = get(
        &router,
        &format!("/v1/runs/{id}?sig={valid}&exp={}", now + 60),
    )
    .await;
    assert_eq!(
        status,
        StatusCode::UNAUTHORIZED,
        "replayed on another route"
    );

    let (status, _) = get(
        &router,
        &format!(
            "/v1/runs/{id}/artifacts/events?sig=00{valid}&exp={}",
            now + 60
        ),
    )
    .await;
    assert_eq!(status, StatusCode::UNAUTHORIZED, "tampered");

    let (status, _, _) = send_raw(
        &router,
        http::Request::builder()
            .uri(format!(
                "/v1/runs/{id}/artifacts/events?sig={valid}&exp={}",
                now + 60
            ))
            .body(axum::body::Body::empty())
            .expect("request"),
    )
    .await;
    assert_ne!(status, StatusCode::UNAUTHORIZED, "the genuine link");
}

#[tokio::test]
async fn a_link_is_refused_exactly_as_its_download_would_be() {
    let fixture = Fixture::new("links-inventory");
    let (router, _, id) = finished_run(&fixture, "").await;
    let (status, _) = post_empty(&router, &format!("/v1/runs/{id}/artifacts/nope/link")).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = post_empty(&router, &format!("/v1/runs/{id}/artifacts/events/link")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // Without authentication a signature is simply ignored.
    let (status, _, _) = send_raw(
        &router,
        http::Request::builder()
            .uri(format!("/v1/runs/{id}/artifacts/events?sig=garbage&exp=1"))
            .body(axum::body::Body::empty())
            .expect("request"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
}
