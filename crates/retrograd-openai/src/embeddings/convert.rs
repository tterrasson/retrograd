//! From the embeddings wire request to the session's [`EmbedItem`] list:
//! `input` in its four shapes, `encoding_format`, and the refusals - with the
//! field named. The lot is bounded here so a request cannot queue an unbounded
//! number of prefill passes behind the decodes.

use serde_json::Value;

use crate::embeddings::wire::{EmbeddingsRequest, EncodingFormat};
use crate::error::OpenAiError;
use crate::session::EmbedItem;

/// How many inputs one request may carry. Each is a prefill pass on the
/// serving context, so the bound keeps the delay added to the decodes finite.
pub const MAX_EMBED_INPUTS: usize = 256;

/// A validated embeddings request.
#[derive(Clone, Debug)]
pub struct Prepared {
    pub model: String,
    pub inputs: Vec<EmbedItem>,
    pub encoding: EncodingFormat,
}

pub fn prepare(request: EmbeddingsRequest) -> Result<Prepared, OpenAiError> {
    if let Some(dimensions) = request.dimensions {
        return Err(OpenAiError::unsupported(
            "dimensions",
            format!(
                "`dimensions: {dimensions}` is not served: the model's own embedding width is \
                 returned"
            ),
        ));
    }
    if request.model.trim().is_empty() {
        return Err(OpenAiError::invalid_param("model", "`model` is required"));
    }
    let encoding = match request.encoding_format.as_deref() {
        None | Some("float") => EncodingFormat::Float,
        Some("base64") => EncodingFormat::Base64,
        Some(other) => {
            return Err(OpenAiError::invalid_param(
                "encoding_format",
                format!("`encoding_format` must be `float` or `base64`; got `{other}`"),
            ));
        }
    };
    let inputs = match &request.input {
        None | Some(Value::Null) => {
            return Err(OpenAiError::invalid_param("input", "`input` is required"));
        }
        Some(Value::String(text)) => vec![EmbedItem::Text(text.clone())],
        Some(Value::Array(items)) => {
            if items.is_empty() {
                return Err(OpenAiError::invalid_param(
                    "input",
                    "`input` must contain at least one element",
                ));
            }
            // A flat token array is one input, regardless of its token count.
            let all_ids = items.iter().all(Value::is_i64);
            if !all_ids && items.len() > MAX_EMBED_INPUTS {
                return Err(OpenAiError::invalid_param(
                    "input",
                    format!("`input` takes at most {MAX_EMBED_INPUTS} elements"),
                ));
            }
            let all_strings = items.iter().all(Value::is_string);
            let all_lists = items.iter().all(Value::is_array);
            let parsed: Result<Vec<EmbedItem>, OpenAiError> =
                match (all_strings, all_ids, all_lists) {
                    (true, _, _) => Ok(items
                        .iter()
                        .map(|item| {
                            EmbedItem::Text(item.as_str().expect("checked above").to_owned())
                        })
                        .collect()),
                    (false, true, _) => Ok(vec![EmbedItem::Tokens(token_ids(items, "input")?)]),
                    (false, false, true) => items
                        .iter()
                        .enumerate()
                        .map(|(index, list)| {
                            Ok(EmbedItem::Tokens(token_ids(
                                list.as_array().expect("checked above"),
                                &format!("input[{index}]"),
                            )?))
                        })
                        .collect(),
                    _ => Err(OpenAiError::invalid_param(
                        "input",
                        "`input` must be a string, a list of strings, a list of token ids, or a \
                         list of token id lists - not a mix",
                    )),
                };
            parsed?
        }
        Some(_) => {
            return Err(OpenAiError::invalid_param(
                "input",
                "`input` must be a string, a list of strings, a list of token ids, or a list of \
                 token id lists",
            ));
        }
    };
    Ok(Prepared {
        model: request.model,
        inputs,
        encoding,
    })
}

/// A list of JSON numbers as token ids; an out-of-range or non-integer element
/// is refused with its position named.
fn token_ids(items: &[Value], at: &str) -> Result<Vec<i32>, OpenAiError> {
    if items.is_empty() {
        return Err(OpenAiError::invalid_param(
            at,
            "an input must contain at least one token",
        ));
    }
    items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let id = item.as_i64().ok_or_else(|| {
                OpenAiError::invalid_param(
                    format!("{at}[{index}]"),
                    "a token id must be an integer",
                )
            })?;
            if id < 0 {
                return Err(OpenAiError::invalid_param(
                    format!("{at}[{index}]"),
                    "a token id must be non-negative",
                ));
            }
            i32::try_from(id).map_err(|_| {
                OpenAiError::invalid_param(
                    format!("{at}[{index}]"),
                    format!("a token id must fit in an i32; got {id}"),
                )
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn prep(body: Value) -> Result<Prepared, OpenAiError> {
        prepare(serde_json::from_value(body).expect("well-formed"))
    }

    #[test]
    fn the_four_input_shapes() {
        let prepared = prep(json!({"model": "m", "input": "hello"})).unwrap();
        assert_eq!(prepared.inputs.len(), 1);
        assert!(matches!(&prepared.inputs[0], EmbedItem::Text(t) if t == "hello"));

        let prepared = prep(json!({"model": "m", "input": ["a", "b"]})).unwrap();
        assert_eq!(prepared.inputs.len(), 2);

        let prepared = prep(json!({"model": "m", "input": [1, 2, 3]})).unwrap();
        assert_eq!(prepared.inputs, vec![EmbedItem::Tokens(vec![1, 2, 3])]);

        let prepared = prep(json!({"model": "m", "input": [[1, 2], [3]]})).unwrap();
        assert_eq!(
            prepared.inputs,
            vec![EmbedItem::Tokens(vec![1, 2]), EmbedItem::Tokens(vec![3])]
        );
    }

    #[test]
    fn refusals_name_the_field() {
        let error = prep(json!({"model": "m", "input": "x", "dimensions": 512})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("dimensions"));

        let error =
            prep(json!({"model": "m", "input": "x", "encoding_format": "int8"})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("encoding_format"));

        let error = prep(json!({"model": "m", "input": ["a", 3]})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input"));

        let error = prep(json!({"model": "m", "input": [[1, 2000000000000_i64]]})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input[0][1]"));

        let many = (0..=256).map(|i| i.to_string()).collect::<Vec<_>>();
        let error = prep(json!({"model": "m", "input": many})).unwrap_err();
        assert_eq!(error.param.as_deref(), Some("input"));
    }
}
