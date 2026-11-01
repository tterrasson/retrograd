//! The context of the `systemone` template.
//!
//! Every decision model is rendered through a Jinja template named
//! `systemone`, with the same frozen context, so templates are
//! interchangeable between models and between this crate and llama-server:
//!
//! ```text
//! { id, type, instructions, state, options: [{ key, description, label? }], images }
//! ```
//!
//! Values are raw JSON, the template serializes the ones that are not strings.
//! Rendering itself is the caller's: a [`TemplateRenderer`].

use std::collections::BTreeMap;
use std::fmt::Display;

use crate::json::{Json, Object};

use crate::DecisionError;
use crate::readout::Readout;
use crate::request::Question;

/// Objects nested deeper than this are refused, as llama-server does, so a
/// hostile body cannot overflow the stack of the recursive walks below.
const RECURSION_LIMIT: usize = 128;

/// Renders the `systemone` template for a context.
pub trait TemplateRenderer {
    type Error: Display;

    fn render(&self, context: &Json) -> Result<String, Self::Error>;
}

impl<F, E: Display> TemplateRenderer for F
where
    F: Fn(&Json) -> Result<String, E>,
{
    type Error = E;

    fn render(&self, context: &Json) -> Result<String, E> {
        self(context)
    }
}

/// What one prompt is made of.
#[derive(Clone, Copy, Debug)]
pub struct PromptInput<'a> {
    pub state: &'a Json,
    pub question_id: &'a str,
    pub question: &'a Question,
    /// Which of the [`Readout::n_variants`] prompts of the question: the
    /// options are in reverse order for any variant after the first.
    pub variant: usize,
    pub n_images: usize,
    /// The text of the media marker of the multimodal runtime, one per image in
    /// `images`.
    pub media_marker: &'a str,
}

/// Builds the context of the template for one prompt.
pub fn context(readout: &Readout, input: &PromptInput<'_>) -> Result<Json, DecisionError> {
    let question = input.question;
    if input.variant >= readout.n_variants(question) {
        return Err(DecisionError::Evaluation(format!(
            "variant {} of a question that has {}",
            input.variant,
            readout.n_variants(question)
        )));
    }

    let mut context = Object::new();
    context.insert("id".into(), input.question_id.into());
    context.insert("type".into(), question.kind.name().into());
    context.insert("instructions".into(), question.instructions.clone());
    context.insert("state".into(), input.state.clone());
    context.insert("options".into(), options(readout, question, input.variant)?);
    let mut context = Json::Object(context);

    // lev was trained with sorted keys.
    if readout.sorts_keys() {
        context = sort_keys(&context, 0)?;
    }

    // The kev template only takes text.
    if matches!(readout, Readout::Pointer { .. }) {
        context.set("state", kev_text(input.state)?.into());
        context.set("instructions", kev_text(&question.instructions)?.into());
    }

    // The input must not contain the marker of the options.
    if let Some(marker) = readout.marker_text() {
        replace_text(&mut context, marker, " ", 0)?;
    }

    // The template puts one media marker per image.
    let mut images = Vec::new();
    if input.n_images > 0 {
        replace_text(&mut context, input.media_marker, " ", 0)?;
        images = vec![Json::from(input.media_marker); input.n_images];
    }
    context.set("images", Json::Array(images));
    Ok(context)
}

/// Builds the context and renders it.
pub fn render<R: TemplateRenderer>(
    readout: &Readout,
    input: &PromptInput<'_>,
    renderer: &R,
) -> Result<String, DecisionError> {
    let context = context(readout, input)?;
    renderer
        .render(&context)
        .map_err(|error| DecisionError::Render(error.to_string()))
}

fn options(readout: &Readout, question: &Question, variant: usize) -> Result<Json, DecisionError> {
    let labels = match readout {
        Readout::LabelLogits { label_texts, .. } => label_texts.as_slice(),
        _ => &[],
    };
    let is_kev = matches!(readout, Readout::Pointer { .. });
    let n = question.options.len();
    let mut out = Vec::with_capacity(n);
    for position in 0..n {
        // The second variant shows the options in the reverse order.
        let option = &question.options[if variant == 0 {
            position
        } else {
            n - 1 - position
        }];
        let mut entry = Object::new();
        if is_kev {
            entry.insert(
                "key".into(),
                kev_text(&Json::String(option.key.clone()))?.into(),
            );
            entry.insert(
                "description".into(),
                if option.description.is_null() {
                    Json::Null
                } else {
                    kev_text(&option.description)?.into()
                },
            );
        } else {
            entry.insert("key".into(), option.key.clone().into());
            entry.insert("description".into(), option.description.clone());
        }
        if let Some(label) = labels.get(position) {
            entry.insert("label".into(), label.clone().into());
        }
        out.push(Json::Object(entry));
    }
    Ok(Json::Array(out))
}

fn too_deep() -> DecisionError {
    DecisionError::invalid(format!(
        "JSON is nested more than {RECURSION_LIMIT} levels deep"
    ))
}

/// Replaces `search` in every string of `value`, in place. Keys are left alone.
fn replace_text(
    value: &mut Json,
    search: &str,
    replace: &str,
    depth: usize,
) -> Result<(), DecisionError> {
    if depth >= RECURSION_LIMIT {
        return Err(too_deep());
    }
    match value {
        Json::String(text) => {
            if text.contains(search) {
                *text = text.replace(search, replace);
            }
        }
        Json::Array(items) => {
            for item in items {
                replace_text(item, search, replace, depth + 1)?;
            }
        }
        Json::Object(members) => {
            for item in members.values_mut() {
                replace_text(item, search, replace, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Sorts the keys of every object of `value`, byte-wise.
fn sort_keys(value: &Json, depth: usize) -> Result<Json, DecisionError> {
    if depth >= RECURSION_LIMIT {
        return Err(too_deep());
    }
    Ok(match value {
        Json::Array(items) => Json::Array(
            items
                .iter()
                .map(|item| sort_keys(item, depth + 1))
                .collect::<Result<_, _>>()?,
        ),
        Json::Object(members) => {
            let mut sorted = BTreeMap::new();
            for (key, item) in members {
                sorted.insert(key.as_str(), sort_keys(item, depth + 1)?);
            }
            Json::Object(
                sorted
                    .into_iter()
                    .map(|(key, item)| (key.to_owned(), item))
                    .collect(),
            )
        }
        other => other.clone(),
    })
}

/// kev flattens a JSON value into text: the keys of an object stay as labels,
/// the items of an array become `- ` lines.
fn kev_render(value: &Json, indent: usize, depth: usize) -> Result<String, DecisionError> {
    if depth >= RECURSION_LIMIT {
        return Err(too_deep());
    }
    let pad = " ".repeat(2 * indent);
    Ok(match value {
        Json::Null => String::new(),
        Json::String(text) => text.clone(),
        Json::Bool(true) => "True".into(),
        Json::Bool(false) => "False".into(),
        Json::Array(items) => {
            let mut out = String::new();
            for item in items {
                let text = kev_render(item, indent + 1, depth + 1)?;
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push_str("- ");
                out.push_str(text.trim_start_matches([' ', '\t', '\n', '\r']));
            }
            out
        }
        Json::Object(members) => {
            let mut out = String::new();
            for (key, item) in members {
                let nested = item.is_object() || item.is_array();
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&pad);
                out.push_str(key);
                out.push_str(if nested { ":\n" } else { ": " });
                out.push_str(&kev_render(
                    item,
                    if nested { indent + 1 } else { 0 },
                    depth + 1,
                )?);
            }
            out
        }
        Json::Number(number) => number.to_string(),
    })
}

/// Text written as `<|name|>` would be parsed as a special token: its bars
/// become U+00A6, so it stays text.
pub fn escape_special_tokens(text: &str) -> String {
    let is_name = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let bytes = text.as_bytes();
    let mut out = String::new();
    // The bytes of `text` that are already in `out`.
    let mut done = 0;
    let mut from = 0;
    while let Some(offset) = text[from..].find("<|") {
        let start = from + offset;
        let mut end = start + 2;
        while end < bytes.len() && is_name(bytes[end]) {
            end += 1;
        }
        if end > start + 2 && text[end..].starts_with("|>") {
            out.push_str(&text[done..start]);
            out.push_str("<\u{a6}");
            out.push_str(&text[start + 2..end]);
            out.push_str("\u{a6}>");
            done = end + 2;
        }
        from = done.max(start + 1);
    }
    if done == 0 {
        return text.to_owned();
    }
    out.push_str(&text[done..]);
    out
}

/// kev text input: the flattened value, special tokens escaped.
fn kev_text(value: &Json) -> Result<String, DecisionError> {
    Ok(escape_special_tokens(&kev_render(value, 0, 0)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readout::LabelFamily;
    use crate::request::{QuestionKind, QuestionOption};

    fn question(kind: QuestionKind, options: &[(&str, Json)]) -> Question {
        Question {
            kind,
            instructions: json!("pick"),
            options: options
                .iter()
                .map(|(key, description)| QuestionOption {
                    key: (*key).into(),
                    description: description.clone(),
                })
                .collect(),
        }
    }

    fn input<'a>(state: &'a Json, question: &'a Question, variant: usize) -> PromptInput<'a> {
        PromptInput {
            state,
            question_id: "q",
            question,
            variant,
            n_images: 0,
            media_marker: "<__media__>",
        }
    }

    fn laya() -> Readout {
        Readout::MarkerColumn {
            marker: 4,
            sep: 1,
            marker_text: "<mask>".into(),
        }
    }

    fn lev() -> Readout {
        Readout::LabelLogits {
            labels: (0..26).collect(),
            label_texts: ('A'..='Z').map(String::from).collect(),
            n_variants: 2,
            noul_scale: Some(9),
            family: LabelFamily::Codes,
        }
    }

    #[test]
    fn context_has_the_frozen_keys_in_order() {
        let state = json!("hello");
        let q = question(
            QuestionKind::Choice,
            &[("a", json!("first")), ("b", Json::Null)],
        );
        let context = context(&laya(), &input(&state, &q, 0)).expect("context");
        let keys: Vec<_> = context
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            ["id", "type", "instructions", "state", "options", "images"]
        );
        assert_eq!(context["type"], "choice");
        assert_eq!(
            context["options"],
            json!([{"key": "a", "description": "first"}, {"key": "b", "description": null}])
        );
        assert_eq!(context["images"], json!([]));
    }

    #[test]
    fn marker_text_is_replaced_in_values_not_in_keys() {
        let state = json!({"<mask>": "a <mask> b", "list": ["<mask><mask>"]});
        let mut q = question(
            QuestionKind::Choice,
            &[("<mask>", json!("d <mask>")), ("x", Json::Null)],
        );
        q.instructions = json!("what <mask>?");
        let context = context(&laya(), &input(&state, &q, 0)).expect("context");
        assert_eq!(context["state"], json!({"<mask>": "a   b", "list": ["  "]}));
        assert_eq!(context["instructions"], "what  ?");
        // Option keys are values of the options array: replaced as well.
        assert_eq!(
            context["options"][0],
            json!({"key": " ", "description": "d  "})
        );
    }

    #[test]
    fn readouts_without_a_marker_text_leave_inputs_alone() {
        let state = json!("a <mask> b");
        let q = question(
            QuestionKind::Noul,
            &[("false", Json::Null), ("true", Json::Null)],
        );
        let context = context(&lev(), &input(&state, &q, 0)).expect("context");
        assert_eq!(context["state"], "a <mask> b");
    }

    #[test]
    fn second_variant_reverses_options_but_labels_follow_the_position() {
        let state = json!("s");
        let q = question(
            QuestionKind::Choice,
            &[("a", json!("A!")), ("b", json!("B!")), ("c", json!("C!"))],
        );
        let first = context(&lev(), &input(&state, &q, 0)).expect("context");
        let second = context(&lev(), &input(&state, &q, 1)).expect("context");
        let shown = |context: &Json| -> Vec<(String, String)> {
            context["options"]
                .as_array()
                .expect("options")
                .iter()
                .map(|o| {
                    (
                        o["key"].as_str().expect("key").into(),
                        o["label"].as_str().expect("label").into(),
                    )
                })
                .collect()
        };
        assert_eq!(
            shown(&first),
            [
                ("a".into(), "A".into()),
                ("b".into(), "B".into()),
                ("c".into(), "C".into())
            ]
        );
        assert_eq!(
            shown(&second),
            [
                ("c".into(), "A".into()),
                ("b".into(), "B".into()),
                ("a".into(), "C".into())
            ]
        );
    }

    #[test]
    fn variant_out_of_range_is_refused() {
        let state = json!("s");
        let q = question(
            QuestionKind::Choice,
            &[("a", Json::Null), ("b", Json::Null)],
        );
        assert!(context(&laya(), &input(&state, &q, 1)).is_err());
        let single = question(QuestionKind::Choice, &[("a", Json::Null)]);
        assert!(context(&lev(), &input(&state, &single, 1)).is_err());
    }

    #[test]
    fn lev_sorts_every_object_key() {
        let state = crate::ordered(r#"{"z": 1, "a": {"y": 1, "b": 2}}"#);
        let q = question(
            QuestionKind::Choice,
            &[
                ("a", crate::ordered(r#"{"k2": 1, "k1": 2}"#)),
                ("b", Json::Null),
            ],
        );
        let context = context(&lev(), &input(&state, &q, 0)).expect("context");
        assert_eq!(
            serde_json::to_string(&context["state"]).expect("json"),
            r#"{"a":{"b":2,"y":1},"z":1}"#
        );
        assert_eq!(
            serde_json::to_string(&context["options"][0]).expect("json"),
            r#"{"description":{"k1":2,"k2":1},"key":"a","label":"A"}"#
        );
    }

    #[test]
    fn images_get_one_marker_each_and_the_marker_text_is_scrubbed() {
        let state = json!("see <__media__>");
        let q = question(
            QuestionKind::Noul,
            &[("true", Json::Null), ("false", Json::Null)],
        );
        let mut prompt = input(&state, &q, 0);
        prompt.n_images = 2;
        let context = context(&lev(), &prompt).expect("context");
        assert_eq!(context["state"], "see  ");
        assert_eq!(context["images"], json!(["<__media__>", "<__media__>"]));
    }

    #[test]
    fn kev_gets_flattened_text() {
        let kev = Readout::Pointer { marker: 9 };
        let state = crate::ordered(
            r#"{"name": "Ann", "tags": ["a", "b"], "meta": {"ok": true, "n": null}}"#,
        );
        let mut q = question(
            QuestionKind::Choice,
            &[("<|im_end|>", json!("say <|x|>")), ("b", Json::Null)],
        );
        q.instructions = json!(["one", {"k": "v"}]);
        let context = context(&kev, &input(&state, &q, 0)).expect("context");
        assert_eq!(
            context["state"],
            "name: Ann\ntags:\n  - a\n  - b\nmeta:\n  ok: True\n  n: "
        );
        assert_eq!(context["instructions"], "- one\n- k: v");
        assert_eq!(
            context["options"][0],
            json!({"key": "<\u{a6}im_end\u{a6}>", "description": "say <\u{a6}x\u{a6}>"})
        );
        assert_eq!(context["options"][1]["description"], Json::Null);
    }

    #[test]
    fn escape_only_rewrites_whole_special_token_spellings() {
        assert_eq!(
            escape_special_tokens("a <|b_1|> c"),
            "a <\u{a6}b_1\u{a6}> c"
        );
        assert_eq!(
            escape_special_tokens("<|a|><|b|>"),
            "<\u{a6}a\u{a6}><\u{a6}b\u{a6}>"
        );
        for untouched in ["<||>", "<|a b|>", "<|a", "|a|>", "plain", "<|<|a|>"] {
            let expected = if untouched == "<|<|a|>" {
                "<|<\u{a6}a\u{a6}>"
            } else {
                untouched
            };
            assert_eq!(escape_special_tokens(untouched), expected, "{untouched}");
        }
    }

    #[test]
    fn deeply_nested_input_is_an_invalid_request() {
        let mut state = json!("leaf");
        for _ in 0..RECURSION_LIMIT {
            state = json!([state]);
        }
        let q = question(
            QuestionKind::Noul,
            &[("false", Json::Null), ("true", Json::Null)],
        );
        assert!(matches!(
            context(&laya(), &input(&state, &q, 0)),
            Err(DecisionError::InvalidRequest(_))
        ));
    }

    #[test]
    fn closures_render_and_their_errors_surface() {
        let state = json!("s");
        let q = question(
            QuestionKind::Noul,
            &[("false", Json::Null), ("true", Json::Null)],
        );
        let ok = render(&laya(), &input(&state, &q, 0), &|context: &Json| {
            Ok::<_, String>(format!("{}!", context["type"].as_str().expect("type")))
        })
        .expect("rendered");
        assert_eq!(ok, "noul!");
        let failed = render(&laya(), &input(&state, &q, 0), &|_: &Json| {
            Err::<String, _>("boom")
        });
        assert!(matches!(failed, Err(DecisionError::Render(message)) if message == "boom"));
    }
}
