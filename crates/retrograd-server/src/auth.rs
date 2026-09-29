//! The shared bearer token.
//!
//! One token, checked on every route, or no authentication at all. That is the
//! complete authentication model. Anything richer is out of scope, and
//! the choice between the two is not the client's: [`ServerConfig::validate`](crate::ServerConfig::validate)
//! refuses to serve a non-loopback address without a token, so "no token" only
//! ever means "reachable from this machine only".
//!
//! Two details that are the difference between a token check and a working one.
//! The comparison is **constant-time**: a byte-by-byte early exit leaks the token
//! one character at a time to anyone who can measure a few thousand requests. And
//! `/v1/health` is exempt, because a liveness probe is the one caller that has no
//! credentials by definition, and it learns nothing a port scan does not.
//!
//! Two more ways through, each narrower than a token:
//!
//! - outside `/v1`, in a build that serves the web interface, everything is
//!   public: those are static files, the same for every installation, with no
//!   data in them - the data is behind `/v1`, where the interface asks for the
//!   token like any other client;
//! - a download of one artefact accepts a signed link in place of the token
//!   ([`crate::links`]), because a browser cannot attach a header to a
//!   navigation. Every other route ignores a `sig`.

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http::header;

use crate::error::{ApiError, ErrorCode, ProblemKind};
use crate::state::AppState;

/// Paths under `/v1` reachable without a token: `/v1/health`, and nothing else
/// under `/v1`.
const PUBLIC: [&str; 1] = ["/v1/health"];

/// Whether `path` belongs to the API rather than to the web interface.
pub fn is_api_path(path: &str) -> bool {
    path == "/v1" || path.starts_with("/v1/")
}

pub async fn require_token(
    State(state): State<AppState>,
    request: Request,
    next: Next,
) -> Response {
    let Some(expected) = state.config.auth_token.as_deref() else {
        return next.run(request).await;
    };
    let path = request.uri().path();
    if PUBLIC.contains(&path) || (!is_api_path(path) && state.config.serves_ui()) {
        return next.run(request).await;
    }
    if request.method() == http::Method::GET && signed(&state, path, request.uri().query()) {
        return next.run(request).await;
    }
    let presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(bearer);
    match presented {
        Some(token) if constant_time_eq(token.as_bytes(), expected.as_bytes()) => {
            next.run(request).await
        }
        // The OpenAI clients read their own envelope and show an empty message
        // for anything else, so the refusal is spelled the way they read it.
        _ if crate::api::openai::is_openai_path(request.uri().path()) => {
            retrograd_openai::OpenAiError::unauthorized("this server requires a bearer token")
                .into_response()
        }
        // One answer for a missing token and a wrong one: which of the two it was
        // is not information a caller without a token is entitled to.
        _ => ApiError::new(
            ProblemKind::Unauthorized,
            "this server requires a bearer token",
        )
        .with_field(
            "/headers/authorization",
            ErrorCode::Unauthorized,
            "missing or invalid",
        )
        .into_response(),
    }
}

/// Whether the request is a download of one artefact carrying a valid,
/// unexpired signature for exactly that artefact.
fn signed(state: &AppState, path: &str, query: Option<&str>) -> bool {
    let Some((run, name)) = crate::links::artifact_route(path) else {
        return false;
    };
    let Some((signature, expires)) = query.and_then(crate::links::signature_of) else {
        return false;
    };
    state.link_key.verify(
        run,
        name,
        expires,
        signature,
        crate::runtime::unix_seconds(),
    )
}

/// The token out of an `Authorization` header, scheme matched case-insensitively
/// as RFC 9110 requires.
fn bearer(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim())
        .filter(|token| !token.is_empty())
}

/// Compares without an early exit.
///
/// The length is deliberately folded into the accumulator rather than checked
/// first: an early `return false` on a length mismatch would still leak the
/// token's length, which is the first thing an attacker needs.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = (left.len() ^ right.len()) as u8;
    // Walk the longer of the two so the loop count does not depend on where the
    // shorter one ends.
    let length = left.len().max(right.len());
    for index in 0..length {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= a ^ b;
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bearer_header_is_parsed_case_insensitively() {
        assert_eq!(bearer("Bearer abc"), Some("abc"));
        assert_eq!(bearer("bearer abc"), Some("abc"));
        assert_eq!(bearer("BEARER  abc "), Some("abc"));
        assert_eq!(bearer("Basic abc"), None);
        assert_eq!(bearer("Bearer "), None);
        assert_eq!(bearer("abc"), None);
    }

    #[test]
    fn the_comparison_does_not_exit_early() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(!constant_time_eq(b"secret", b"secrez"));
        assert!(!constant_time_eq(b"secret", b"secret-longer"));
        assert!(!constant_time_eq(b"", b"x"));
        assert!(constant_time_eq(b"", b""));
    }
}
