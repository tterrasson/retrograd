//! From the legacy Completions wire request to the common
//! [`GenerationRequest`]: each prompt is
//! served raw - no chat template, no tools, no parser - with the same
//! sampling contract as chat (A2): a field that would change the sampled
//! distribution is honoured or refused with the field named.

use retrograd_core::SamplingParamsV2;
use serde_json::Value;

use crate::chat::convert as chat;
use crate::completions::wire::CompletionRequest;
use crate::error::OpenAiError;
use crate::request::GenerationRequest;

/// A validated legacy completions request.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub model: String,
    /// One entry per prompt element: a string `prompt` is one, a list of
    /// strings is many.
    pub prompts: Vec<String>,
    pub stream: bool,
    /// Prepend each prompt to its completion's text.
    pub echo: bool,
    pub stop: Vec<String>,
    pub n: u32,
    /// How many top-logprob pairs each emitted token carries, when the client
    /// asked for logprobs.
    pub logprobs: Option<u32>,
    pub sampling: SamplingParamsV2,
}

impl Prepared {
    /// The common request for one prompt: raw text, `n` choices.
    pub fn request(&self, prompt: &str) -> GenerationRequest {
        GenerationRequest {
            raw_prompt: Some(prompt.to_owned()),
            messages: Vec::new(),
            tools: Vec::new(),
            sampling: self.sampling.clone(),
            stop: self.stop.clone(),
            n: self.n,
            logprobs: self.logprobs,
            tool_constraint: None,
            reasoning_budget: None,
        }
    }
}

pub fn prepare(request: CompletionRequest) -> Result<Prepared, OpenAiError> {
    if let Some(suffix) = &request.suffix
        && !suffix.is_null()
    {
        return Err(OpenAiError::unsupported(
            "suffix",
            "`suffix` is not served: the completion is the whole answer, nothing is generated \
             after it to fill a suffix",
        ));
    }
    if let Some(best_of) = request.best_of {
        return Err(OpenAiError::unsupported(
            "best_of",
            format!(
                "`best_of: {best_of}` is not served: `n` completions are generated and all are \
                 returned, none is picked as a `best_of`"
            ),
        ));
    }
    if request.model.trim().is_empty() {
        return Err(OpenAiError::invalid_param("model", "`model` is required"));
    }
    let prompts = match &request.prompt {
        None | Some(Value::Null) => {
            return Err(OpenAiError::invalid_param("prompt", "`prompt` is required"));
        }
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => {
            if items.is_empty() {
                return Err(OpenAiError::invalid_param(
                    "prompt",
                    "`prompt` must contain at least one string",
                ));
            }
            if items.len() > 256 {
                return Err(OpenAiError::invalid_param(
                    "prompt",
                    "at most 256 prompts are served",
                ));
            }
            items
                .iter()
                .enumerate()
                .map(|(index, item)| match item.as_str() {
                    Some(text) => Ok(text.to_owned()),
                    None => Err(OpenAiError::unsupported(
                        format!("prompt[{index}]"),
                        "only string prompts are served; token arrays are not (send the text, \
                         the server tokenizes it with the model's own tokenizer)",
                    )),
                })
                .collect::<Result<Vec<_>, _>>()?
        }
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                "prompt",
                "`prompt` must be a string or a list of strings",
            ));
        }
    };
    let n = request.n.unwrap_or(1);
    if n == 0 || n > chat::MAX_N {
        return Err(OpenAiError::invalid_param(
            "n",
            format!("`n` must be between 1 and {}", chat::MAX_N),
        ));
    }
    // Narrowing checked against MAX_N above: cannot lose anything.
    let n = u32::try_from(n).map_err(|_| OpenAiError::invalid_param("n", "`n` does not fit"))?;
    // Zero still requests the sampled token's logprob, without top pairs.
    let logprobs = match &request.logprobs {
        None | Some(Value::Null) => None,
        Some(Value::Number(value)) => {
            let top = value.as_u64().ok_or_else(|| {
                OpenAiError::invalid_param(
                    "logprobs",
                    "`logprobs` must be an integer between 0 and 20",
                )
            })?;
            if top > chat::MAX_TOP_LOGPROBS {
                return Err(OpenAiError::invalid_param(
                    "logprobs",
                    format!("`logprobs` must be at most {}", chat::MAX_TOP_LOGPROBS),
                ));
            }
            // Bounded by MAX_TOP_LOGPROBS above.
            Some(top as u32)
        }
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                "logprobs",
                "`logprobs` must be an integer between 0 and 20 (the legacy completions form)",
            ));
        }
    };
    if request.echo == Some(true) && logprobs.is_some() {
        return Err(OpenAiError::unsupported(
            "echo",
            "`echo` with `logprobs` requires prompt token scores, which are not served",
        ));
    }
    let sampling = SamplingParamsV2 {
        temperature: chat::temperature(request.temperature)?,
        top_p: chat::top_p(request.top_p)?,
        max_new_tokens: chat::max_tokens(request.max_tokens, None)?,
        seed: chat::seed(request.seed)?,
        top_k: chat::top_k(request.top_k)?,
        min_p: chat::min_p(request.min_p)?,
        repeat_penalty: chat::repeat_penalty(request.repeat_penalty)?,
        frequency_penalty: chat::penalty("frequency_penalty", request.frequency_penalty)?,
        presence_penalty: chat::penalty("presence_penalty", request.presence_penalty)?,
        penalty_window: chat::penalty_window(request.penalty_window)?,
        logit_bias: chat::logit_bias(request.logit_bias.as_ref())?,
        grammar_gbnf: None,
    };
    Ok(Prepared {
        model: request.model,
        prompts,
        stream: request.stream.unwrap_or(false),
        echo: request.echo.unwrap_or(false),
        stop: chat::stop(request.stop)?,
        n,
        logprobs,
        sampling,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prep(body: Value) -> Result<Prepared, OpenAiError> {
        prepare(serde_json::from_value(body).expect("well-formed"))
    }

    #[test]
    fn a_string_prompt_is_one_raw_prompt_and_a_list_is_many() {
        let prepared = prep(json!({"model": "m", "prompt": "once upon"})).unwrap();
        assert_eq!(prepared.prompts, vec!["once upon".to_owned()]);
        assert!(!prepared.stream && !prepared.echo && prepared.n == 1);
        let request = prepared.request("once upon");
        assert_eq!(request.raw_prompt.as_deref(), Some("once upon"));
        assert!(request.messages.is_empty());

        let prepared = prep(json!({"model": "m", "prompt": ["a", "b"], "n": 2})).unwrap();
        assert_eq!(prepared.prompts.len(), 2);
        assert_eq!(prepared.n, 2);
    }

    #[test]
    fn token_arrays_suffix_and_best_of_are_refused_with_the_field_named() {
        for (field, value, param) in [
            ("prompt", json!([[1, 2, 3]]), "prompt[0]"),
            ("suffix", json!("…"), "suffix"),
            ("best_of", json!(4), "best_of"),
        ] {
            let mut body = json!({"model": "m", "prompt": "x"});
            body[field] = value;
            let error = prep(body).unwrap_err();
            assert_eq!(error.param.as_deref(), Some(param), "{field}");
        }
    }

    #[test]
    fn legacy_logprobs_is_an_integer_count() {
        let prepared = prep(json!({"model": "m", "prompt": "x", "logprobs": 5})).unwrap();
        assert_eq!(prepared.logprobs, Some(5));
        let prepared = prep(json!({"model": "m", "prompt": "x", "logprobs": 0})).unwrap();
        assert_eq!(prepared.logprobs, Some(0));
        let error = prep(json!({"model": "m", "prompt": "x", "logprobs": 21})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("logprobs"));
        let error = prep(json!({"model": "m", "prompt": "x", "logprobs": true})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("logprobs"));
    }
}
