//! One hardened client for the OpenAI-compatible chat-completions surface.

use std::time::Duration;

use retrograd_agent_core::{Error, Result};
use serde::Deserialize;
use serde_json::Value;

/// Which agent-stack facade variant a failed request belongs to.
#[derive(Clone, Copy, Debug)]
pub enum Purpose {
    Judge,
    ScenarioGeneration,
}

impl Purpose {
    fn label(self) -> &'static str {
        match self {
            Self::Judge => "judge",
            Self::ScenarioGeneration => "scenario generation",
        }
    }

    fn runtime_error(self, message: impl Into<String>) -> Error {
        match self {
            Self::Judge => Error::Reward(message.into()),
            Self::ScenarioGeneration => Error::Tool(message.into()),
        }
    }
}

/// A client for one OpenAI-compatible `/chat/completions` endpoint.
///
/// Requests are bearer-authenticated and bounded by both a timeout and a cap
/// on response bytes, and a completion the provider cut short (`finish_reason
/// == "length"`) is reported as an error rather than returned as partial
/// content.
#[derive(Clone)]
pub struct OpenAiClient {
    http: reqwest::Client,
    endpoint: reqwest::Url,
    api_key: String,
    timeout: Duration,
    max_response_bytes: usize,
    purpose: Purpose,
}

/// The subset of a chat-completion response every current caller needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Completion {
    pub content: String,
}

#[derive(Deserialize)]
struct CompletionResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: Message,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct Message {
    content: Option<String>,
}

impl OpenAiClient {
    pub fn new(
        base_url: &str,
        api_key: String,
        timeout: Duration,
        max_response_bytes: usize,
        purpose: Purpose,
    ) -> Result<Self> {
        let endpoint = endpoint(base_url)
            .map_err(|message| Error::invalid(format!("{} base_url {message}", purpose.label())))?;
        let http = reqwest::Client::builder().build().map_err(|error| {
            purpose.runtime_error(format!("build {} HTTP client: {error}", purpose.label()))
        })?;
        Ok(Self {
            http,
            endpoint,
            api_key,
            timeout,
            max_response_bytes,
            purpose,
        })
    }

    /// Sends `payload` as the request body and returns the first choice's
    /// content.
    ///
    /// The response body is read incrementally and the call fails as soon as
    /// it exceeds `max_response_bytes`, rather than buffering an unbounded
    /// reply first. A response truncated by the provider is an error, not a
    /// partial success.
    pub async fn chat_completion(&self, payload: &Value) -> Result<Completion> {
        let response = self
            .http
            .post(self.endpoint.clone())
            .bearer_auth(&self.api_key)
            .timeout(self.timeout)
            .json(payload)
            .send()
            .await
            .map_err(|error| {
                if error.is_timeout() {
                    self.purpose.runtime_error(format!(
                        "{} request timed out after {}s",
                        self.purpose.label(),
                        self.timeout.as_secs()
                    ))
                } else {
                    self.purpose
                        .runtime_error(format!("{} request failed: {error}", self.purpose.label()))
                }
            })?;
        let status = response.status();
        let mut response = response;
        let mut body = Vec::with_capacity(self.max_response_bytes.min(64 * 1024));
        while let Some(chunk) = response.chunk().await.map_err(|error| {
            self.purpose
                .runtime_error(format!("read {} response: {error}", self.purpose.label()))
        })? {
            if body.len().saturating_add(chunk.len()) > self.max_response_bytes {
                return Err(self.purpose.runtime_error(format!(
                    "{} response exceeded {} bytes",
                    self.purpose.label(),
                    self.max_response_bytes
                )));
            }
            body.extend_from_slice(&chunk);
        }
        if !status.is_success() {
            return Err(self.purpose.runtime_error(format!(
                "{} endpoint returned HTTP {status}: {}",
                self.purpose.label(),
                String::from_utf8_lossy(&body[..body.len().min(512)]).trim()
            )));
        }
        decode_completion(self.purpose, &body)
    }
}

fn decode_completion(purpose: Purpose, body: &[u8]) -> Result<Completion> {
    let completion: CompletionResponse = serde_json::from_slice(body).map_err(|error| {
        purpose.runtime_error(format!(
            "{} response is not a valid completion object: {error}",
            purpose.label()
        ))
    })?;
    let choice = completion.choices.into_iter().next().ok_or_else(|| {
        purpose.runtime_error(format!("{} response contained no choice", purpose.label()))
    })?;
    if choice.finish_reason.as_deref() == Some("length") {
        return Err(purpose.runtime_error(format!(
            "{} response was truncated by the provider",
            purpose.label()
        )));
    }
    let content = choice.message.content.ok_or_else(|| {
        purpose.runtime_error(format!(
            "{} response contained no message content",
            purpose.label()
        ))
    })?;
    Ok(Completion { content })
}

fn endpoint(base_url: &str) -> std::result::Result<reqwest::Url, &'static str> {
    let mut url = reqwest::Url::parse(base_url).map_err(|_| "is not a valid URL")?;
    if !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("must not contain credentials, a query string, or a fragment");
    }
    let already_complete = url
        .path()
        .trim_end_matches('/')
        .ends_with("/chat/completions");
    if !already_complete {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| "cannot be used as an HTTP base")?;
        segments.pop_if_empty().push("chat").push("completions");
    }
    Ok(url)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_is_normalized_once_and_rejects_credentials() {
        assert_eq!(
            endpoint("https://example.test/v1/").unwrap().as_str(),
            "https://example.test/v1/chat/completions"
        );
        assert_eq!(
            endpoint("https://example.test/v1/chat/completions/")
                .unwrap()
                .as_str(),
            "https://example.test/v1/chat/completions/"
        );
        assert!(endpoint("https://user@example.test/v1").is_err());
        assert!(endpoint("https://example.test/v1?token=secret").is_err());
    }

    #[test]
    fn a_provider_length_stop_is_never_accepted_as_content() {
        let error = decode_completion(
            Purpose::Judge,
            br#"{"choices":[{"message":{"content":"partial"},"finish_reason":"length"}]}"#,
        )
        .unwrap_err();
        assert!(matches!(error, Error::Reward(_)));
        assert!(error.to_string().contains("truncated"));
    }
}
