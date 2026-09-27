//! The API key of a process that serves one model.
//!
//! A server with runs has its own authentication over every route; this is for
//! the process that has nothing but these three, and whose clients send the
//! `Authorization: Bearer <api_key>` every OpenAI SDK sends.

use std::sync::Arc;

use axum::extract::{Request, State};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use http::header;

use crate::error::OpenAiError;

/// Refuses a request whose bearer token is not `key`, in the OpenAI envelope -
/// an SDK shows an empty message for anything else.
pub async fn require_api_key(
    State(key): State<Arc<str>>,
    request: Request,
    next: Next,
) -> Response {
    let presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(bearer);
    match presented {
        Some(token) if constant_time_eq(token.as_bytes(), key.as_bytes()) => {
            next.run(request).await
        }
        _ => OpenAiError::unauthorized("a valid API key is required as a bearer token")
            .into_response(),
    }
}

/// The token of an `Authorization` header, the scheme matched
/// case-insensitively as RFC 9110 requires.
pub fn bearer(value: &str) -> Option<&str> {
    let (scheme, token) = value.split_once(' ')?;
    scheme
        .eq_ignore_ascii_case("bearer")
        .then(|| token.trim())
        .filter(|token| !token.is_empty())
}

/// Compares without an early exit, the length folded into the accumulator so
/// that it does not leak either.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = u8::from(left.len() != right.len());
    for index in 0..left.len().max(right.len()) {
        difference |=
            left.get(index).copied().unwrap_or(0) ^ right.get(index).copied().unwrap_or(0);
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bearer_token_is_read_and_compared_whole() {
        assert_eq!(bearer("Bearer sk-1"), Some("sk-1"));
        assert_eq!(bearer("bearer  sk-1 "), Some("sk-1"));
        assert_eq!(bearer("Basic sk-1"), None);
        assert!(constant_time_eq(b"sk-1", b"sk-1"));
        assert!(!constant_time_eq(b"sk-1", b"sk-2"));
        assert!(!constant_time_eq(b"sk-1", b"sk-1-and-more"));
        // Lengths 256 apart, whose XOR truncated to a byte would be zero.
        assert!(!constant_time_eq(&[0; 256], b""));
    }
}
