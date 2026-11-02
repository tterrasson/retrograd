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
//! A type may add to it ([`crate::readout::DecisionType`]): the list of all
//! the questions (`nimble`), the markers of the pieces of the d1-omni layout;
//! `h2o-lightning` gets texts rendered as its Python reference renders them,
//! and a joint head gets every question at once ([`joint_context`]).
//! Rendering itself is the caller's: a [`TemplateRenderer`].

use std::collections::BTreeMap;
use std::fmt::Display;

use indexmap::IndexMap;

use crate::json::{Json, Object};

use crate::DecisionError;
use crate::layout::d1omni::{D1OMNI_MARKER, D1OMNI_SEP, MARK_OPTION, MARK_QUESTION, MARK_STATE};
use crate::layout::joint::{JOINT_MARK_OPTION, JOINT_MARK_QUESTION, JOINT_MARKER, JOINT_SEP};
use crate::python::{self, Dumps};
use crate::readout::{Head, InputText, PromptScope, Readout, Vocabulary};
use crate::request::{Question, QuestionKind};

/// Objects nested deeper than this are refused, as llama-server does, so a
/// hostile body cannot overflow the stack of the recursive walks below.
pub(crate) const RECURSION_LIMIT: usize = 128;

/// A longer `h2o-lightning` state keeps its head and its tail, half of this
/// each.
const H2O_MAX_STATE_TOKENS: usize = 32000;

/// What separates the head and the tail of a cut `h2o-lightning` state.
const H2O_CUT: &str = "\n...\n";

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
    /// The state as the model reads it: [`prepare_state`] of the request's.
    pub state: &'a Json,
    /// Every question of the request, which a prompt that lists them all
    /// needs; `None` lists `question` alone.
    pub questions: Option<&'a IndexMap<String, Question>>,
    pub question_id: &'a str,
    pub question: &'a Question,
    /// The label of each option of `question`, by position:
    /// [`Readout::question_labels`]. Empty for a readout that shows none.
    pub labels: &'a [String],
    /// Which of the [`Readout::n_variants`] prompts of the question: the
    /// options are in reverse order for any variant after the first.
    pub variant: usize,
    pub n_images: usize,
    /// The text of the media marker of the multimodal runtime, one per image in
    /// `images`.
    pub media_marker: &'a str,
}

/// The state as the model reads it, done once per request: the state of the
/// request for most types; `h2o-lightning` reads compact JSON text, cut to
/// its token budget, special token texts quoted.
pub fn prepare_state(
    readout: &Readout,
    state: &Json,
    vocab: &impl Vocabulary,
) -> Result<Json, DecisionError> {
    if readout.kind.input != InputText::H2o {
        return Ok(state.clone());
    }
    let text = match state.as_str() {
        Some(text) => text.to_owned(),
        None => python::dumps(
            state,
            Dumps {
                indent: None,
                item_separator: ",",
                key_separator: ":",
                ensure_ascii: false,
            },
        )?,
    };
    Ok(h2o_quote(&h2o_cut_state(text, vocab), &readout.quoted).into())
}

/// Builds the context of the template for one prompt of a type that renders
/// one question per prompt.
pub fn context(readout: &Readout, input: &PromptInput<'_>) -> Result<Json, DecisionError> {
    let question = input.question;
    let kind = readout.kind;
    if input.variant >= readout.n_variants(question) {
        return Err(DecisionError::Evaluation(format!(
            "variant {} of a question that has {}",
            input.variant,
            readout.n_variants(question)
        )));
    }
    if matches!(kind.prompt, PromptScope::Joint | PromptScope::JointQuestion) {
        return Err(DecisionError::Evaluation(
            "a joint head reads the joint context of its questions".into(),
        ));
    }
    if shows_labels(readout) && input.labels.len() != question.options.len() {
        return Err(DecisionError::Evaluation(format!(
            "{} labels for a question of {} options",
            input.labels.len(),
            question.options.len()
        )));
    }

    let mut context = if kind.input == InputText::H2o {
        h2o_context(readout, input)?
    } else {
        let mut context = Object::new();
        context.insert("id".into(), input.question_id.into());
        context.insert("type".into(), question.kind.name().into());
        context.insert("instructions".into(), question.instructions.clone());
        context.insert("state".into(), input.state.clone());
        context.insert(
            "options".into(),
            options(readout, question, input.variant, input.labels)?,
        );
        Json::Object(context)
    };

    // The nimble prompt lists all the questions of the request.
    if kind.prompt == PromptScope::AllQuestions {
        let single = IndexMap::from([(input.question_id.to_owned(), question.clone())]);
        let questions = input.questions.unwrap_or(&single);
        let mut listed = Vec::with_capacity(questions.len());
        for (id, other) in questions {
            let labels = match &readout.head {
                Head::LabelLogits { label_texts, .. } => {
                    label_texts.get(..other.options.len()).unwrap_or(&[])
                }
                _ => &[],
            };
            let mut entry = Object::new();
            entry.insert("id".into(), id.clone().into());
            entry.insert("type".into(), other.kind.name().into());
            entry.insert("instructions".into(), other.instructions.clone());
            entry.insert("options".into(), options(readout, other, 0, labels)?);
            listed.push(Json::Object(entry));
        }
        context.set("questions", Json::Array(listed));
    }

    if kind.sorts_keys {
        context = sort_keys(&context, 0)?;
    }

    // The kev template only takes text.
    if kind.input == InputText::Kev {
        context.set("state", kev_text(input.state)?.into());
        context.set("instructions", kev_text(&question.instructions)?.into());
    }

    // The OCR text must not be parsed as special tokens.
    if kind.input == InputText::OcrEscape {
        let Some(text) = input.state.as_str() else {
            return Err(DecisionError::Evaluation(
                "the state of an OCR consistency question is not text".into(),
            ));
        };
        context.set("state", escape_special_tokens(text).into());
    }

    // The input must not contain the marker of the options.
    if let Some(marker) = readout.marker_text() {
        replace_text(&mut context, marker, " ", 0)?;
    }

    // A spoken question is heard where its text would be, after the media of
    // the state.
    let spoken = question.is_spoken();
    if spoken {
        context.set("instructions", "".into());
    }
    if kind.input == InputText::D1Break {
        context = escape(&context, &|text| d1_break(text, &readout.quoted), 0)?;
    }
    if kind.input == InputText::D1OmniEscape {
        context = escape(&context, &escape_special_tokens, 0)?;
        replace_text(&mut context, D1OMNI_MARKER, "<<d1omni ", 0)?;
        context.set("audio", Json::Bool(false));
        context.set("sep", D1OMNI_SEP.into());
        context.set("mark_state", MARK_STATE.into());
        context.set("mark_question", MARK_QUESTION.into());
        context.set("mark_option", MARK_OPTION.into());
    }

    // The template puts one media marker per image.
    let mut images = Vec::new();
    if input.n_images > 0 || spoken {
        replace_text(&mut context, input.media_marker, " ", 0)?;
        images = vec![Json::from(input.media_marker); input.n_images];
    }
    context.set("images", Json::Array(images));
    if spoken {
        context.set("instructions", input.media_marker.into());
    }
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

/// The context of a joint head: the state and every question, keys sorted,
/// and the texts that mark the pieces of the prompt
/// ([`crate::layout::joint`]).
///
/// ```text
/// { questions: [{ id, instructions, options: [{ description, key }], type }], state,
///   images, sep, mark_question, mark_option }
/// ```
pub fn joint_context(
    state: &Json,
    questions: &[(&str, &Question)],
    n_images: usize,
    media_marker: &str,
) -> Result<Json, DecisionError> {
    let mut listed = Vec::with_capacity(questions.len());
    for (id, question) in questions {
        let options = question
            .options
            .iter()
            .map(|option| {
                let mut entry = Object::new();
                entry.insert("key".into(), option.key.clone().into());
                entry.insert("description".into(), option.description.clone());
                Json::Object(entry)
            })
            .collect();
        let mut entry = Object::new();
        entry.insert("id".into(), (*id).into());
        entry.insert("type".into(), question.kind.name().into());
        entry.insert("instructions".into(), question.instructions.clone());
        entry.insert("options".into(), Json::Array(options));
        listed.push(Json::Object(entry));
    }
    let mut context = Object::new();
    context.insert("state".into(), state.clone());
    context.insert("questions".into(), Json::Array(listed));

    // Raw values with sorted keys, and no marker in the input.
    let mut context = sort_keys(&Json::Object(context), 0)?;
    replace_text(&mut context, JOINT_MARKER, "<<clef ", 0)?;

    let mut images = Vec::new();
    if n_images > 0 {
        replace_text(&mut context, media_marker, " ", 0)?;
        images = vec![Json::from(media_marker); n_images];
    }
    context.set("images", Json::Array(images));
    context.set("sep", JOINT_SEP.into());
    context.set("mark_question", JOINT_MARK_QUESTION.into());
    context.set("mark_option", JOINT_MARK_OPTION.into());
    Ok(context)
}

/// Whether the template of `readout` shows a label per option.
fn shows_labels(readout: &Readout) -> bool {
    match &readout.head {
        Head::LabelLogits { label_texts, .. } => !label_texts.is_empty(),
        Head::LabelGroups => true,
        _ => false,
    }
}

fn options(
    readout: &Readout,
    question: &Question,
    variant: usize,
    labels: &[String],
) -> Result<Json, DecisionError> {
    let is_kev = readout.kind.input == InputText::Kev;
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

/// The `h2o-lightning` context: texts, as its reference shim writes them
/// with Python. The instructions are JSON with an indent of one, the
/// descriptions default JSON, the state comes from [`prepare_state`].
///
/// ```text
/// { type, state, instructions, options: [{ label, key, description }] }
/// ```
fn h2o_context(readout: &Readout, input: &PromptInput<'_>) -> Result<Json, DecisionError> {
    let question = input.question;
    let quoted = &readout.quoted;
    let instructions = match question.instructions.as_str() {
        Some(text) => text.to_owned(),
        None => python::dumps(
            &question.instructions,
            Dumps {
                indent: Some(1),
                item_separator: ",",
                key_separator: ": ",
                ensure_ascii: true,
            },
        )?,
    };
    let instructions = python::lstrip(&instructions);

    let n = question.options.len();
    let mut options = Vec::with_capacity(n);
    for (i, option) in question.options.iter().enumerate() {
        // Null when not given, the template has a default; a score level is
        // shown as JSON null.
        let mut description =
            if question.kind == QuestionKind::Score && option.description.is_null() {
                Json::from("null")
            } else if option.description.is_null() {
                Json::Null
            } else {
                let text = match option.description.as_str() {
                    Some(text) => text.to_owned(),
                    None => python::dumps(
                        &option.description,
                        Dumps {
                            indent: None,
                            item_separator: ", ",
                            key_separator: ": ",
                            ensure_ascii: false,
                        },
                    )?,
                };
                Json::from(h2o_quote(&text, quoted))
            };
        let key = h2o_quote(&option.key, quoted);

        // The user turn is stripped like Python's `str.strip()`, the template
        // trims ASCII whitespace only: the text that ends the turn is stripped
        // here, a space keeps a description that is only whitespace.
        if i + 1 == n {
            let is_key = question.kind == QuestionKind::Choice && description.is_null();
            let text = if is_key {
                Some(key.clone())
            } else {
                description.as_str().map(str::to_owned)
            };
            if let Some(text) = text {
                let stripped = python::rstrip(&text);
                if stripped != text {
                    description = Json::from(if stripped.is_empty() { " " } else { stripped });
                }
            }
        }

        let Some(label) = input.labels.get(i) else {
            return Err(DecisionError::Evaluation(
                "an option without a label".into(),
            ));
        };
        let mut entry = Object::new();
        entry.insert("label".into(), label.clone().into());
        entry.insert("key".into(), key.into());
        entry.insert("description".into(), description);
        options.push(Json::Object(entry));
    }

    let mut context = Object::new();
    context.insert("type".into(), question.kind.name().into());
    context.insert("state".into(), input.state.clone());
    context.insert(
        "instructions".into(),
        h2o_quote(instructions, quoted).into(),
    );
    context.insert("options".into(), Json::Array(options));
    Ok(Json::Object(context))
}

/// A state longer than the budget keeps its head and its tail around a
/// `...` line. Each half is the text of its token pieces; a cut inside a
/// character gives U+FFFD, as in a JSON response of the server.
fn h2o_cut_state(text: String, vocab: &impl Vocabulary) -> String {
    // A byte-level BPE never has more tokens than bytes.
    if text.len() <= H2O_MAX_STATE_TOKENS {
        return text;
    }
    let tokens = vocab.tokenize(&text, true);
    if tokens.len() <= H2O_MAX_STATE_TOKENS {
        return text;
    }
    let half = H2O_MAX_STATE_TOKENS / 2;
    let text_of = |tokens: &[crate::Token]| {
        let bytes: Vec<u8> = tokens
            .iter()
            .flat_map(|&token| vocab.token_bytes(token))
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    };
    format!(
        "{}{H2O_CUT}{}",
        text_of(&tokens[..half]),
        text_of(&tokens[tokens.len() - half..])
    )
}

/// The first bytes of `spellings`, each once.
fn first_bytes(spellings: &[String]) -> Vec<u8> {
    let mut first = Vec::new();
    for spelling in spellings {
        if let Some(&byte) = spelling.as_bytes().first()
            && !first.contains(&byte)
        {
            first.push(byte);
        }
    }
    first
}

/// The spelling of `spellings` (longest first) that `text` starts with.
fn spelling_at<'a>(text: &[u8], spellings: &'a [String]) -> Option<&'a String> {
    spellings
        .iter()
        .find(|spelling| text.starts_with(spelling.as_bytes()))
}

/// `h2o-lightning`: the special token texts written in the input are quoted
/// with JSON escapes, so they stay text.
fn h2o_quote(text: &str, spellings: &[String]) -> String {
    let first = first_bytes(spellings);
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hit = if first.contains(&bytes[i]) {
            spelling_at(&bytes[i..], spellings)
        } else {
            None
        };
        match hit {
            Some(spelling) => {
                out.extend_from_slice(
                    spelling
                        .replace('<', "\\u003c")
                        .replace('>', "\\u003e")
                        .as_bytes(),
                );
                i += spelling.len();
            }
            None => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    // The spellings and the text are UTF-8, and a spelling starts and ends
    // on a character boundary.
    String::from_utf8_lossy(&out).into_owned()
}

/// The lfm3 d1 models: a zero-width space after the first character of each
/// special token text in the input (d1 `prompt.py`: `plain`), again until a
/// match that overlapped a broken one is broken too.
fn d1_break(text: &str, spellings: &[String]) -> String {
    const ZWSP: &str = "\u{200b}";
    let first = first_bytes(spellings);
    let mut current = text.as_bytes().to_vec();
    loop {
        let mut changed = false;
        let mut out = Vec::with_capacity(current.len());
        let mut i = 0;
        while i < current.len() {
            let hit = if first.contains(&current[i]) {
                spelling_at(&current[i..], spellings)
            } else {
                None
            };
            let Some(spelling) = hit else {
                out.push(current[i]);
                i += 1;
                continue;
            };
            // After the first character, its UTF-8 bytes.
            let lead = spelling.as_bytes()[0];
            let width = match lead {
                0..0x80 => 1,
                0x80..0xe0 => 2,
                0xe0..0xf0 => 3,
                _ => 4,
            }
            .min(spelling.len());
            out.extend_from_slice(&spelling.as_bytes()[..width]);
            out.extend_from_slice(ZWSP.as_bytes());
            out.extend_from_slice(&spelling.as_bytes()[width..]);
            i += spelling.len();
            changed = true;
        }
        current = out;
        if !changed {
            break;
        }
    }
    String::from_utf8_lossy(&current).into_owned()
}

pub(crate) fn too_deep() -> DecisionError {
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
            if !search.is_empty() && text.contains(search) {
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

/// Applies `escape` to every string of `value`, keys included. Two keys that
/// escape to the same text are one member, at the place of the first, with
/// the value of the last.
fn escape(
    value: &Json,
    escape_text: &dyn Fn(&str) -> String,
    depth: usize,
) -> Result<Json, DecisionError> {
    if depth >= RECURSION_LIMIT {
        return Err(too_deep());
    }
    Ok(match value {
        Json::String(text) => Json::String(escape_text(text)),
        Json::Array(items) => Json::Array(
            items
                .iter()
                .map(|item| escape(item, escape_text, depth + 1))
                .collect::<Result<_, _>>()?,
        ),
        Json::Object(members) => {
            let mut out = Object::new();
            for (key, item) in members {
                out.insert(escape_text(key), escape(item, escape_text, depth + 1)?);
            }
            Json::Object(out)
        }
        other => other.clone(),
    })
}

/// Sorts the keys of every object of `value`, byte-wise.
pub(crate) fn sort_keys(value: &Json, depth: usize) -> Result<Json, DecisionError> {
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
        // `val.dump()` in the server: nlohmann's float format, not serde_json's.
        Json::Number(number) => crate::json::dump_number(number),
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
    use crate::readout::SpecialToken;
    use crate::readout::tests::{Fake, kind};
    use crate::request::QuestionOption;

    /// The code labels A..Z, the first ones of a `lev`.
    const CODES: [&str; 26] = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z",
    ];

    fn codes(n: usize) -> Vec<String> {
        CODES[..n].iter().map(|code| (*code).to_owned()).collect()
    }

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
            questions: None,
            question_id: "q",
            question,
            labels: &[],
            variant,
            n_images: 0,
            media_marker: "<__media__>",
        }
    }

    fn labelled<'a>(
        state: &'a Json,
        question: &'a Question,
        variant: usize,
        labels: &'a [String],
    ) -> PromptInput<'a> {
        PromptInput {
            labels,
            ..input(state, question, variant)
        }
    }

    fn laya() -> Readout {
        Readout::new(
            kind("laya"),
            Head::MarkerColumn {
                marker: 4,
                sep: 1,
                marker_text: "<mask>".into(),
            },
        )
    }

    fn lev() -> Readout {
        Readout::new(
            kind("lev"),
            Head::LabelLogits {
                labels: (0..26).collect(),
                label_texts: codes(26),
            },
        )
    }

    fn kev() -> Readout {
        Readout::new(kind("kev"), Head::Pointer { marker: 9 })
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
        let context =
            context(&lev(), &labelled(&state, &q, 0, &codes(q.options.len()))).expect("context");
        assert_eq!(context["state"], "a <mask> b");
    }

    #[test]
    fn second_variant_reverses_options_but_labels_follow_the_position() {
        let state = json!("s");
        let q = question(
            QuestionKind::Choice,
            &[("a", json!("A!")), ("b", json!("B!")), ("c", json!("C!"))],
        );
        let first =
            context(&lev(), &labelled(&state, &q, 0, &codes(q.options.len()))).expect("context");
        let second =
            context(&lev(), &labelled(&state, &q, 1, &codes(q.options.len()))).expect("context");
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
        assert!(context(&lev(), &labelled(&state, &single, 1, &codes(1))).is_err());
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
        let context =
            context(&lev(), &labelled(&state, &q, 0, &codes(q.options.len()))).expect("context");
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
        let labels = codes(2);
        let mut prompt = labelled(&state, &q, 0, &labels);
        prompt.n_images = 2;
        let context = context(&lev(), &prompt).expect("context");
        assert_eq!(context["state"], "see  ");
        assert_eq!(context["images"], json!(["<__media__>", "<__media__>"]));
    }

    #[test]
    fn kev_gets_flattened_text() {
        let kev = kev();
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
    fn kev_writes_numbers_as_the_server_dumps_them() {
        let kev = kev();
        let state = crate::ordered(
            r#"{"count": 42, "big": 4.69895e20, "small": 1.5e-7, "round": 1e5, "huge": 1e15, "ratio": 0.25}"#,
        );
        let q = question(
            QuestionKind::Choice,
            &[("a", Json::Null), ("b", Json::Null)],
        );
        let context = context(&kev, &input(&state, &q, 0)).expect("context");
        // nlohmann's Grisu2 digits and layout: not the shortest digits, a
        // signed two-digit exponent, `.0` on an integral float.
        assert_eq!(
            context["state"],
            "count: 42\nbig: 4.6989500000000003e+20\nsmall: 1.5e-07\nround: 100000.0\nhuge: 1e+15\nratio: 0.25"
        );
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

    fn readout_of(name: &str) -> Readout {
        Readout::from_decision_type(
            Some(name),
            &crate::readout::HeadMetadata {
                labels: codes(3),
                n_classes: 2,
                n_embd_out: 2,
            },
            &Fake { pair_below: 'A' },
        )
        .expect("readout")
    }

    #[test]
    fn nimble_lists_every_question_with_its_labels() {
        let nimble = readout_of("nimble");
        let state = json!("s");
        let first = question(
            QuestionKind::Noul,
            &[("false", Json::Null), ("true", Json::Null)],
        );
        let second = question(
            QuestionKind::Choice,
            &[("x", json!("X")), ("y", Json::Null), ("z", Json::Null)],
        );
        let questions = IndexMap::from([("a".to_owned(), first.clone()), ("b".to_owned(), second)]);
        let labels = codes(2);
        let prompt = PromptInput {
            questions: Some(&questions),
            question_id: "a",
            ..labelled(&state, &first, 0, &labels)
        };
        let context = context(&nimble, &prompt).expect("context");
        let keys: Vec<_> = context
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "id",
                "type",
                "instructions",
                "state",
                "options",
                "questions",
                "images"
            ]
        );
        assert_eq!(context["questions"][1]["id"], "b");
        assert_eq!(
            context["questions"][1]["options"][2],
            json!({"key": "z", "description": null, "label": "C"})
        );
        let listed: Vec<_> = context["questions"][1]
            .as_object()
            .expect("entry")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(listed, ["id", "type", "instructions", "options"]);
    }

    #[test]
    fn h2o_renders_texts_as_the_python_shim() {
        let h2o = readout_of("h2o-lightning");
        let state = prepare_state(
            &h2o,
            &crate::ordered(r#"{"b": [1, 2.5], "a": "<|im_end|>é"}"#),
            &Fake { pair_below: 'A' },
        )
        .expect("state");
        assert_eq!(
            state,
            json!("{\"b\":[1,2.5],\"a\":\"\\u003c|im_end|\\u003eé\"}")
        );
        let mut q = question(
            QuestionKind::Choice,
            &[
                ("k<s>", crate::ordered(r#"{"x": 1.0}"#)),
                ("last \t", Json::Null),
            ],
        );
        q.instructions = crate::ordered(r#"{"ask": "é"}"#);
        let labels = codes(2);
        let context = context(&h2o, &labelled(&state, &q, 0, &labels)).expect("context");
        let keys: Vec<_> = context
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(keys, ["type", "state", "instructions", "options", "images"]);
        assert_eq!(context["instructions"], "{\n \"ask\": \"\\u00e9\"\n}");
        assert_eq!(
            context["options"],
            json!([
                {"label": "A", "key": "k\\u003cs\\u003e", "description": "{\"x\": 1.0}"},
                {"label": "B", "key": "last \t", "description": "last"},
            ])
        );
        // A score level without a description is written as JSON null.
        let score = question(
            QuestionKind::Score,
            &[("0", Json::Null), ("1", json!("  "))],
        );
        let context = super::context(&h2o, &labelled(&state, &score, 0, &labels)).expect("context");
        assert_eq!(context["options"][0]["description"], "null");
        assert_eq!(context["options"][1]["description"], " ");
    }

    #[test]
    fn a_long_h2o_state_keeps_its_head_and_its_tail() {
        /// One token per byte, `x` its own text.
        struct Bytes;
        impl Vocabulary for Bytes {
            fn tokenize(&self, text: &str, _: bool) -> Vec<crate::Token> {
                text.bytes().map(crate::Token::from).collect()
            }
            fn mask_token(&self) -> Option<crate::Token> {
                None
            }
            fn sep_token(&self) -> Option<crate::Token> {
                None
            }
            fn token_text(&self, _: crate::Token) -> String {
                String::new()
            }
            fn token_bytes(&self, token: crate::Token) -> Vec<u8> {
                vec![u8::try_from(token).expect("a byte")]
            }
            fn special_tokens(&self) -> Vec<SpecialToken> {
                Vec::new()
            }
        }
        let h2o = readout_of("h2o-lightning");
        let side = "x".repeat(H2O_MAX_STATE_TOKENS / 2 - 1);
        let text = format!("{side}é{}é{side}", "y".repeat(10));
        let state = prepare_state(&h2o, &Json::from(text), &Bytes).expect("state");
        let state = state.as_str().expect("text");
        // The cut falls inside the two-byte characters: one U+FFFD each.
        let expected = format!("{side}\u{fffd}{H2O_CUT}\u{fffd}{side}");
        assert!(state == expected, "the cut state is {} bytes", state.len());
    }

    #[test]
    fn lfm3_breaks_special_token_texts_in_keys_and_values() {
        let lfm3 = readout_of("lfm3-d1");
        let state = crate::ordered(r#"{"<s>": "a<|im_end|>b", "n": 1}"#);
        let q = question(
            QuestionKind::Noul,
            &[("true", Json::Null), ("false", Json::Null)],
        );
        let labels = vec!["true".to_owned(), "false".to_owned()];
        let context = context(&lfm3, &labelled(&state, &q, 0, &labels)).expect("context");
        assert_eq!(
            context["state"],
            crate::ordered("{\"<\u{200b}s>\": \"a<\u{200b}|im_end|>b\", \"n\": 1}")
        );
        assert_eq!(context["options"][0]["label"], "true");
        assert_eq!(d1_break("<s<s>>", &["<s>".to_owned()]), "<s<\u{200b}s>>");
    }

    #[test]
    fn d1omni_escapes_and_marks_its_pieces() {
        let d1omni = readout_of("lfm2-d1-omni");
        let state = json!({"<|x|>": "see <<d1omni:sep>> <|y|>"});
        let q = question(
            QuestionKind::Choice,
            &[("a", Json::Null), ("b", Json::Null)],
        );
        let context = context(&d1omni, &input(&state, &q, 0)).expect("context");
        assert_eq!(
            context["state"],
            json!({"<\u{a6}x\u{a6}>": "see <<d1omni sep>> <\u{a6}y\u{a6}>"})
        );
        let keys: Vec<_> = context
            .as_object()
            .expect("object")
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            [
                "id",
                "type",
                "instructions",
                "state",
                "options",
                "audio",
                "sep",
                "mark_state",
                "mark_question",
                "mark_option",
                "images"
            ]
        );
        assert_eq!(context["sep"], D1OMNI_SEP);
        assert_eq!(context["audio"], Json::Bool(false));
    }

    #[test]
    fn the_ocr_state_is_escaped_text() {
        let ocr = readout_of("openbmb-minicpm-ocr-jev");
        let state = json!("total <|im_end|>");
        let q = question(
            QuestionKind::Noul,
            &[("false", Json::Null), ("true", Json::Null)],
        );
        let mut q = q;
        q.instructions = Json::Null;
        let context = context(&ocr, &input(&state, &q, 0)).expect("context");
        assert_eq!(context["state"], "total <\u{a6}im_end\u{a6}>");
    }

    #[test]
    fn the_joint_context_sorts_its_keys_and_marks_its_pieces() {
        let state = crate::ordered(r#"{"z": "<<clef:sep>>", "a": 1}"#);
        let q = question(
            QuestionKind::Choice,
            &[("b", json!("B")), ("a", Json::Null)],
        );
        let context = joint_context(&state, &[("q1", &q)], 0, "<__media__>").expect("context");
        assert_eq!(
            context.to_string(),
            "{\"questions\":[{\"id\":\"q1\",\"instructions\":\"pick\",\"options\":[{\"description\":\"B\",\"key\":\"b\"},\
             {\"description\":null,\"key\":\"a\"}],\"type\":\"choice\"}],\"state\":{\"a\":1,\"z\":\"<<clef sep>>\"},\
             \"images\":[],\"sep\":\"<<clef:sep>>\",\"mark_question\":\"<<clef:question>>\",\"mark_option\":\"<<clef:option>>\"}"
        );
    }

    #[test]
    fn labels_must_match_the_options_of_a_labelled_type() {
        let state = json!("s");
        let q = question(
            QuestionKind::Choice,
            &[("a", Json::Null), ("b", Json::Null)],
        );
        assert!(matches!(
            context(&lev(), &input(&state, &q, 0)),
            Err(DecisionError::Evaluation(_))
        ));
        assert!(matches!(
            context(&readout_of("clef"), &input(&state, &q, 0)),
            Err(DecisionError::Evaluation(_))
        ));
    }
}
