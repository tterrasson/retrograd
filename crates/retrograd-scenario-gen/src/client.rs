use std::time::Duration;

use retrograd_agent_core::{Error, Result};
use retrograd_config::ScenarioGenerationConfig;
use retrograd_llm_client::{OpenAiClient, Purpose};
use serde::Deserialize;
use serde_json::json;

use crate::ScenarioDraft;
use crate::prompt::response_schema;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DraftResponse {
    scenarios: Vec<ScenarioDraft>,
}

pub async fn request(
    config: &ScenarioGenerationConfig,
    prompt: &str,
    count: usize,
) -> Result<Vec<ScenarioDraft>> {
    let key = std::env::var(&config.api_key_env).map_err(|_| {
        Error::invalid(format!(
            "scenario generation API key environment variable '{}' is not set",
            config.api_key_env
        ))
    })?;
    let limit = count.saturating_mul(64 * 1024).max(64 * 1024);
    let client = OpenAiClient::new(
        &config.base_url,
        key,
        Duration::from_secs(config.timeout_secs),
        limit,
        Purpose::ScenarioGeneration,
    )?;
    let completion = client.chat_completion(&json!({
            "model": config.model,
            "messages": [{"role": "system", "content": prompt}],
            "response_format": {"type": "json_schema", "json_schema": response_schema(count, config.min_difficulty, config.max_difficulty)},
            "max_completion_tokens": 1024usize.saturating_add(count.saturating_mul(768)),
            "seed": config.seed,
        })).await?;
    let drafts: DraftResponse = serde_json::from_str(&completion.content).map_err(|_| {
        Error::invalid("scenario generation content did not match the strict schema")
    })?;
    if drafts.scenarios.len() != count {
        return Err(Error::invalid(format!(
            "scenario generation returned {} drafts, expected {count}",
            drafts.scenarios.len()
        )));
    }
    Ok(drafts.scenarios)
}
