//! The web interface under `/`, and what it must not change about `/v1`.
//!
//! With the `ui` feature these run against the real `web/dist` the build
//! embedded - `scripts/test-web.sh` builds it first. Without it, `/` stays what
//! it always was: an unknown path of the API.

use http::StatusCode;

use crate::support::*;

#[cfg(feature = "ui")]
mod embedded {
    use http::header;

    use super::*;

    const TOKEN: &str = "a-long-enough-test-token";

    fn router(fixture: &Fixture, extra: &str) -> axum::Router {
        build_router(state_with(
            fixture,
            FakeEngine::succeeding(),
            &format!("calibrate_runs = false\nauth_token = \"{TOKEN}\"\n{extra}"),
        ))
    }

    async fn fetch(router: &axum::Router, uri: &str) -> (StatusCode, http::HeaderMap, Vec<u8>) {
        use http_body_util::BodyExt;
        use tower::ServiceExt;

        let response = router
            .clone()
            .oneshot(
                http::Request::builder()
                    .uri(uri)
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("route");
        let status = response.status();
        let headers = response.headers().clone();
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes()
            .to_vec();
        (status, headers, body)
    }

    #[tokio::test]
    async fn the_interface_is_served_without_a_token_on_every_route_of_its_own() {
        let fixture = Fixture::new("ui-served");
        let router = router(&fixture, "");
        let (status, headers, index) = fetch(&router, "/").await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            headers[header::CONTENT_TYPE]
                .to_str()
                .expect("ascii")
                .starts_with("text/html")
        );
        assert_eq!(
            headers[header::CONTENT_SECURITY_POLICY],
            retrograd_ui::CONTENT_SECURITY_POLICY
        );
        let (status, _, route) = fetch(&router, "/runs/abc/trajectories").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(route, index, "an application route is the application");
    }

    #[tokio::test]
    async fn a_missing_asset_is_a_plain_404_and_the_api_is_unchanged() {
        let fixture = Fixture::new("ui-api");
        let router = router(&fixture, "");
        let (status, headers, _) = fetch(&router, "/assets/missing-0000.js").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(
            !headers[header::CONTENT_TYPE]
                .to_str()
                .expect("ascii")
                .contains("html"),
            "a missing script must not receive the index"
        );
        let (status, headers, _) = fetch(&router, "/v1/nope").await;
        assert_eq!(
            status,
            StatusCode::UNAUTHORIZED,
            "the API still wants its token"
        );
        assert_eq!(headers[header::CONTENT_TYPE], "application/problem+json");
        let (status, _, _) = fetch(&router, "/v1/runs").await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _, _) = fetch(&router, "/v1/health").await;
        assert_eq!(status, StatusCode::OK);
    }

    #[tokio::test]
    async fn an_unknown_api_path_is_still_a_problem_document() {
        let fixture = Fixture::new("ui-unknown");
        let router = build_router(state_with(
            &fixture,
            FakeEngine::succeeding(),
            "calibrate_runs = false\n",
        ));
        let (status, body) = get(&router, "/v1/nope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["type"].as_str().expect("type").ends_with("not-found"));
    }

    #[tokio::test]
    async fn the_operator_can_turn_the_interface_off() {
        let fixture = Fixture::new("ui-off");
        let router = build_router(state_with(
            &fixture,
            FakeEngine::succeeding(),
            "calibrate_runs = false\nui = false\n",
        ));
        let (status, body) = get(&router, "/").await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{body}");
        let (_, capabilities) = get(&router, "/v1/capabilities").await;
        assert_eq!(capabilities["features"]["ui"], false);
    }
}

#[cfg(not(feature = "ui"))]
#[tokio::test]
async fn without_the_interface_the_root_is_an_unknown_path() {
    let fixture = Fixture::new("ui-absent");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, body) = get(&router, "/").await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert!(body["type"].as_str().expect("type").ends_with("not-found"));
    let (_, capabilities) = get(&router, "/v1/capabilities").await;
    assert_eq!(capabilities["features"]["ui"], false);
}
