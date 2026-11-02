//! The `/v1/systemone` request body and its validation.
//!
//! The messages of a rejected request are the ones llama-server returns, so a
//! client sees the same sentence from both.

use crate::json::Json;
use indexmap::IndexMap;

use crate::DecisionError;
use crate::readout::{DecisionType, Instructions, NoulOrder, QuestionShape, Readout, Spoken};

/// Media a request may carry, images and audio together.
const MAX_MEDIA: usize = 8;

/// Levels a `score` question may have.
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

/// The media a model that checks the consistency of an OCR text reads: the
/// original image, then the image rendered from the text.
const OCR_N_IMAGES: usize = 2;

/// The type of a question, which is also the column its score is read from in
/// a `laya` output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QuestionKind {
    Choice,
    Score,
    Noul,
}

impl QuestionKind {
    /// The name the request, the template and the temperature keys spell.
    pub fn name(self) -> &'static str {
        match self {
            Self::Choice => "choice",
            Self::Score => "score",
            Self::Noul => "noul",
        }
    }

    /// The column of the model output that holds the score of this type.
    pub fn column(self) -> usize {
        match self {
            Self::Choice => 0,
            Self::Score => 1,
            Self::Noul => 2,
        }
    }
}

/// One option of a question, in the order of the model outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct QuestionOption {
    pub key: String,
    /// `Json::Null` when the request gives none.
    pub description: Json,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Question {
    pub kind: QuestionKind,
    pub instructions: Json,
    /// `choice`: the criteria in request order, or sorted by key for a type
    /// that sorts them. `score`: one option per level, keyed by its index.
    /// `noul`: `false` then `true`, or the other way round for a readout that
    /// lists `true` first.
    pub options: Vec<QuestionOption>,
}

impl Question {
    /// Whether the instructions are an audio clip, an `input_audio` part in
    /// place of the text.
    pub fn is_spoken(&self) -> bool {
        is_spoken(&self.instructions)
    }
}

fn is_spoken(instructions: &Json) -> bool {
    instructions.is_object()
        && instructions.get("type").and_then(Json::as_str) == Some("input_audio")
}

/// Text, an object or an array: what a strict type accepts as a payload.
fn is_payload(value: &Json) -> bool {
    value.is_string() || value.is_object() || value.is_array()
}

/// What a question is checked against. A readout fixes it
/// ([`QuestionRules::of`]); a dataset uses [`QuestionRules::generic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuestionRules {
    pub kind: &'static DecisionType,
    pub n_options_max: usize,
}

impl QuestionRules {
    pub fn of(readout: &Readout) -> Self {
        Self {
            kind: readout.kind,
            n_options_max: readout.n_options_max(),
        }
    }

    /// The limits no readout exceeds, the checks of the standard types,
    /// `noul` as `[false, true]`.
    pub fn generic() -> Self {
        Self {
            kind: DecisionType::named("laya").expect("laya is a decision type"),
            n_options_max: 255,
        }
    }
}

/// A validated request. `questions` keeps the order of the body, which is the
/// order of the answers.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneRequest {
    /// The state without its media: text, an object or an array, or null for
    /// a type that asks about media only.
    pub state: Json,
    /// Media as data URLs (or base64 audio), in the order they appear: the
    /// `files` and `images` arrays, then the messages of the state.
    pub images: Vec<String>,
    pub questions: IndexMap<String, Question>,
}

impl SystemOneRequest {
    /// A request that asks one question about `state`, with no media: how a
    /// labelled example is put to the model.
    pub fn single(state: Json, question_id: String, question: Question) -> Self {
        Self {
            state,
            images: Vec::new(),
            questions: IndexMap::from([(question_id, question)]),
        }
    }

    /// Parses and validates a body.
    pub fn from_slice(body: &[u8], readout: &Readout) -> Result<Self, DecisionError> {
        let body: Json = serde_json::from_slice(body)
            .map_err(|error| DecisionError::invalid(format!("invalid JSON: {error}")))?;
        Self::parse(&body, readout)
    }

    /// Validates a body as llama-server does, in its order: the questions,
    /// then the state and its media. Media the model has no place for, and
    /// spoken instructions, are [`DecisionError::ImagesUnsupported`].
    pub fn parse(body: &Json, readout: &Readout) -> Result<Self, DecisionError> {
        let rules = QuestionRules::of(readout);
        let questions = parse_questions(body, rules)?;
        let mut images = Vec::new();
        let state = parse_state(body, rules.kind, &mut images)?;
        if !images.is_empty() && !readout.accepts_images() {
            return Err(DecisionError::ImagesUnsupported);
        }
        // A spoken question is heard where its text would be: an audio clip
        // of the prompt, after the media of the state.
        if rules.kind.spoken == Spoken::Media
            && let Some(question) = questions.values().find(|question| question.is_spoken())
        {
            let audio = question.instructions.get("input_audio");
            let data = audio
                .filter(|audio| audio.is_object())
                .and_then(|audio| audio.get("data"));
            load_audio(data, &mut images)?;
            return Err(DecisionError::ImagesUnsupported);
        }
        Ok(Self {
            state,
            images,
            questions,
        })
    }
}

fn parse_questions(
    body: &Json,
    rules: QuestionRules,
) -> Result<IndexMap<String, Question>, DecisionError> {
    let kind = rules.kind;
    match body.get("state") {
        None => return Err(DecisionError::invalid("\"state\" must be provided")),
        Some(Json::Null) if !kind.null_state => {
            return Err(DecisionError::invalid("\"state\" must be provided"));
        }
        Some(state) if kind.strict_payload && !is_payload(state) => {
            return Err(DecisionError::invalid(
                "\"state\" must be text, an object or an array",
            ));
        }
        Some(_) => {}
    }
    let Some(questions) = body
        .get("questions")
        .and_then(Json::as_object)
        .filter(|questions| !questions.is_empty())
    else {
        return Err(DecisionError::invalid(
            "\"questions\" must be a non-empty object",
        ));
    };
    if kind.questions == QuestionShape::OcrConsistency && questions.len() != 1 {
        return Err(DecisionError::invalid(
            "this model answers exactly one question",
        ));
    }
    questions
        .iter()
        .map(|(id, question)| Ok((id.clone(), parse_question(id, question, rules)?)))
        .collect()
}

/// Validates one question as `parse_questions` of llama-server does.
pub fn parse_question(
    id: &str,
    question: &Json,
    rules: QuestionRules,
) -> Result<Question, DecisionError> {
    let kind = rules.kind;
    let err = |message: &str| DecisionError::invalid(format!("questions.{id}: {message}"));
    let Some(question) = question.as_object() else {
        return Err(err("must be an object"));
    };
    let instructions = match question.get("instructions") {
        Some(instructions) => instructions.clone(),
        None if kind.instructions == Instructions::DefaultEmpty => Json::from(""),
        None => Json::Null,
    };
    if instructions.is_null() && kind.instructions == Instructions::Required {
        return Err(err("\"instructions\" must be provided"));
    }
    if kind.strict_payload && (!is_payload(&instructions) || instructions == "") {
        return Err(err(
            "\"instructions\" must be non-empty text, an object or an array",
        ));
    }
    if kind.spoken == Spoken::Refused && is_spoken(&instructions) {
        return Err(err("spoken instructions need a model with audio input"));
    }
    let criteria = question.get("criteria").unwrap_or(&Json::Null);
    let lenient = kind.questions == QuestionShape::Lenient;
    // A type of another JSON type reads as the default, as llama-server's
    // `json_value` does.
    let type_name = question
        .get("type")
        .and_then(Json::as_str)
        .unwrap_or(if lenient { "choice" } else { "" });

    let (question_kind, mut options) = match type_name {
        "choice" => {
            let Some(criteria) = criteria.as_object().filter(|criteria| !criteria.is_empty())
            else {
                return Err(err("\"criteria\" must be a non-empty object"));
            };
            let mut options = Vec::with_capacity(criteria.len());
            for (key, description) in criteria {
                if kind.strict_payload
                    && (key.is_empty() || !(description.is_null() || is_payload(description)))
                {
                    return Err(err(
                        "option keys must be non-empty, descriptions must be null, text, an object or an array",
                    ));
                }
                options.push(QuestionOption {
                    key: key.clone(),
                    description: description.clone(),
                });
            }
            if kind.strict_payload && options.len() < 2 {
                return Err(err("\"criteria\" must have at least 2 options"));
            }
            if kind.choice_sorted {
                options.sort_by(|a, b| a.key.cmp(&b.key));
            }
            (QuestionKind::Choice, options)
        }
        "score" => {
            let levels = criteria.as_array();
            let levels = if lenient {
                levels
                    .filter(|levels| !levels.is_empty())
                    .ok_or_else(|| err("\"criteria\" must be a non-empty array"))?
            } else {
                levels
                    .filter(|levels| SCORE_LEVELS.contains(&levels.len()))
                    .ok_or_else(|| err("\"criteria\" must be an array of 2 to 10 levels"))?
            };
            let mut options = Vec::with_capacity(levels.len());
            for (level, description) in levels.iter().enumerate() {
                if kind.strict_payload && !is_payload(description) {
                    return Err(err(
                        "level descriptions must be text, an object or an array",
                    ));
                }
                options.push(QuestionOption {
                    key: level.to_string(),
                    description: description.clone(),
                });
            }
            (QuestionKind::Score, options)
        }
        "noul" => {
            if !criteria.is_null() && !criteria.is_object() {
                return Err(err("\"criteria\" must be an object"));
            }
            if kind.strict_payload
                && let Some(criteria) = criteria.as_object()
                && criteria.iter().any(|(key, description)| {
                    (key != "false" && key != "true") || !is_payload(description)
                })
            {
                return Err(err(
                    "\"criteria\" may only describe \"false\" and \"true\" with text, an object or an array",
                ));
            }
            let options: Vec<QuestionOption> = ["false", "true"]
                .into_iter()
                .map(|key| {
                    let alias = if key == "true" { "yes" } else { "no" };
                    let description = match criteria.get(key) {
                        Some(description) => description.clone(),
                        None if kind.noul_aliases => {
                            criteria.get(alias).cloned().unwrap_or(Json::Null)
                        }
                        None => Json::Null,
                    };
                    QuestionOption {
                        key: key.to_owned(),
                        description,
                    }
                })
                .collect();
            (QuestionKind::Noul, options)
        }
        _ => return Err(err("\"type\" must be one of: choice, score, noul")),
    };

    if question_kind == QuestionKind::Noul {
        let swap = match kind.noul_order {
            NoulOrder::FalseFirst => false,
            NoulOrder::TrueFirst => true,
            NoulOrder::CriteriaOrder => criteria.as_object().is_some_and(|criteria| {
                criteria.len() == 2 && criteria.keys().next().map(String::as_str) == Some("true")
            }),
        };
        if swap {
            options.swap(0, 1);
        }
    }

    // The classes are fixed: "0" is inconsistent, "1" is consistent.
    if kind.questions == QuestionShape::OcrConsistency {
        if !instructions.is_null() {
            return Err(err(
                "\"instructions\" is not used by this model, the task is part of its prompt",
            ));
        }
        match question_kind {
            QuestionKind::Choice => {
                let fixed = options.len() == 2
                    && options[0].key == "0"
                    && options[1].key == "1"
                    && options.iter().all(|option| option.description.is_null());
                if !fixed {
                    return Err(err("\"criteria\" must be {\"0\": null, \"1\": null}"));
                }
            }
            QuestionKind::Noul => {
                if !criteria.is_null() {
                    return Err(err("\"criteria\" is not supported for a noul question"));
                }
            }
            QuestionKind::Score => {
                return Err(err("only choice and noul questions are supported"));
            }
        }
    }

    if options.len() > rules.n_options_max {
        return Err(err(&format!(
            "too many options ({}), this model supports at most {}",
            options.len(),
            rules.n_options_max
        )));
    }
    Ok(Question {
        kind: question_kind,
        instructions,
        options,
    })
}

/// Collects the media of the body into `media` and returns the state without
/// the media parts of its messages.
fn parse_state(
    body: &Json,
    kind: &DecisionType,
    media: &mut Vec<String>,
) -> Result<Json, DecisionError> {
    let ocr = kind.questions == QuestionShape::OcrConsistency;
    // `parse_questions` has checked that the state is there.
    let state = body.get("state").unwrap_or(&Json::Null);
    if ocr && !state.is_string() {
        return Err(DecisionError::invalid("\"state\" must be the OCR text"));
    }
    if let Some(videos) = body.get("videos") {
        let empty = match videos {
            Json::Null => true,
            Json::Array(items) => items.is_empty(),
            Json::Object(items) => items.is_empty(),
            _ => false,
        };
        if !empty {
            return Err(DecisionError::invalid("\"videos\" is not supported"));
        }
    }
    // "images" is an alias of "files".
    for key in ["files", "images"] {
        match body.get(key) {
            None | Some(Json::Null) => {}
            Some(Json::Array(urls)) => {
                for url in urls {
                    load_media(url, media)?;
                }
            }
            Some(_) => {
                return Err(DecisionError::invalid(format!(
                    "\"{key}\" must be an array"
                )));
            }
        }
    }
    if ocr && media.len() != OCR_N_IMAGES {
        return Err(DecisionError::invalid(format!(
            "this model needs exactly 2 images (the original, then the rendered one), got {}",
            media.len()
        )));
    }

    let wrapped = state.get("messages");
    let messages = wrapped.unwrap_or(state);
    let Json::Array(messages) = messages else {
        return Ok(state.clone());
    };

    // Chat messages: the image and audio parts leave the content, and the
    // video parts for a model that reads them.
    let mut messages_out = Vec::with_capacity(messages.len());
    for message in messages {
        let Some(parts) = message.get("content").and_then(Json::as_array) else {
            messages_out.push(message.clone());
            continue;
        };
        let mut content = Vec::with_capacity(parts.len());
        for part in parts {
            match (
                part.get("type").and_then(Json::as_str),
                part.get("image_url"),
                part.get("input_audio"),
                part.get("video_url"),
            ) {
                (Some("image_url"), Some(image_url), _, _) => {
                    load_media(image_url.get("url").unwrap_or(image_url), media)?;
                }
                (Some("input_audio"), _, Some(audio), _) => {
                    load_audio(audio.get("data").or_else(|| audio.get("url")), media)?;
                }
                (Some("video_url"), _, _, Some(video_url)) if kind.video_parts => {
                    load_media(video_url.get("url").unwrap_or(video_url), media)?;
                }
                _ => content.push(part.clone()),
            }
        }
        let mut message = message.clone();
        message.set("content", Json::Array(content));
        messages_out.push(message);
    }

    if wrapped.is_none() {
        return Ok(Json::Array(messages_out));
    }
    let mut state = state.clone();
    state.set("messages", Json::Array(messages_out));
    Ok(state)
}

fn load_media(url: &Json, media: &mut Vec<String>) -> Result<(), DecisionError> {
    let Some(url) = url.as_str().filter(|url| url.starts_with("data:")) else {
        return Err(DecisionError::invalid(
            "media must be data URLs (data:image/..., data:audio/... or data:video/...;base64,...)",
        ));
    };
    if media.len() >= MAX_MEDIA {
        return Err(DecisionError::invalid(format!(
            "too many images, the maximum is {MAX_MEDIA}"
        )));
    }
    media.push(url.to_owned());
    Ok(())
}

/// Audio is base64 data, as a data URL or not (OpenAI `input_audio`).
fn load_audio(data: Option<&Json>, media: &mut Vec<String>) -> Result<(), DecisionError> {
    let Some(data) = data
        .and_then(Json::as_str)
        .filter(|data| !data.starts_with("http") && !data.starts_with("file://"))
    else {
        return Err(DecisionError::invalid("audio must be base64 data"));
    };
    if media.len() >= MAX_MEDIA {
        return Err(DecisionError::invalid(format!(
            "too many media files, the maximum is {MAX_MEDIA}"
        )));
    }
    media.push(data.to_owned());
    Ok(())
}

#[cfg(test)]
mod tests {

    use super::*;
    use crate::json::Object;
    use crate::readout::Head;
    use crate::readout::tests::kind;

    fn rules() -> QuestionRules {
        QuestionRules::generic()
    }

    fn parse(body: &Json) -> Result<IndexMap<String, Question>, DecisionError> {
        parse_questions(body, rules())
    }

    fn message(result: Result<impl std::fmt::Debug, DecisionError>) -> String {
        match result {
            Err(DecisionError::InvalidRequest(message)) => message,
            other => panic!("expected an invalid request, got {other:?}"),
        }
    }

    #[test]
    fn question_order_is_the_order_of_the_body() {
        // Not alphabetical, and not the order a sorted map would give.
        let body = crate::ordered(
            r#"{"state": "s", "questions": {
                "urgent": {"type": "noul", "instructions": "i"},
                "intent": {"type": "choice", "instructions": "i", "criteria": {"b": null, "a": "x"}},
                "frustration": {"type": "score", "instructions": "i", "criteria": ["lo", "hi"]}
            }}"#,
        );
        let questions = parse(&body).expect("valid");
        let ids: Vec<_> = questions.keys().map(String::as_str).collect();
        assert_eq!(ids, ["urgent", "intent", "frustration"]);
        // Criteria keep their order too.
        let keys: Vec<_> = questions["intent"]
            .options
            .iter()
            .map(|o| o.key.as_str())
            .collect();
        assert_eq!(keys, ["b", "a"]);
    }

    #[test]
    fn state_object_keeps_the_order_of_the_body() {
        let body: Json = serde_json::from_str(
            r#"{"state": {"zeta": 1, "alpha": {"y": 1, "x": 2}}, "questions": {"q": {"type": "noul", "instructions": "i"}}}"#,
        )
        .expect("json");
        let request = SystemOneRequest::parse(&body, &laya()).expect("valid");
        assert_eq!(
            serde_json::to_string(&request.state).expect("serialise"),
            r#"{"zeta":1,"alpha":{"y":1,"x":2}}"#
        );
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

    /// A readout of `name` whose labels are `n` placeholder tokens.
    fn labelled(name: &str, n: usize) -> Readout {
        Readout::new(
            kind(name),
            Head::LabelLogits {
                labels: vec![0; n],
                label_texts: vec!["A".into(); n],
            },
        )
    }

    fn rules_of(name: &str) -> QuestionRules {
        QuestionRules {
            kind: kind(name),
            n_options_max: 255,
        }
    }

    #[test]
    fn noul_options_are_false_then_true_with_their_descriptions() {
        let body = json!({"state": "s", "questions": {"q": {
            "type": "noul", "instructions": "i", "criteria": {"true": "yes", "other": "ignored"}}}});
        let question = &parse(&body).expect("valid")["q"];
        assert_eq!(
            question.options,
            vec![
                QuestionOption {
                    key: "false".into(),
                    description: Json::Null
                },
                QuestionOption {
                    key: "true".into(),
                    description: json!("yes")
                },
            ]
        );

        let swapped = parse_questions(&body, rules_of("openjev")).expect("valid");
        assert_eq!(swapped["q"].options[0].key, "true");
        assert_eq!(swapped["q"].options[0].description, json!("yes"));
    }

    #[test]
    fn score_levels_are_keyed_by_index() {
        let body = json!({"state": "s", "questions": {"q": {
            "type": "score", "instructions": "i", "criteria": ["a", {"k": 1}, null]}}});
        let question = &parse(&body).expect("valid")["q"];
        let keys: Vec<_> = question.options.iter().map(|o| o.key.as_str()).collect();
        assert_eq!(keys, ["0", "1", "2"]);
        assert_eq!(question.options[1].description, json!({"k": 1}));
    }

    #[test]
    fn rejections_carry_the_llama_server_messages() {
        let q = |question: Json| json!({"state": "s", "questions": {"q": question}});
        assert_eq!(
            message(parse(&json!({"questions": {}}))),
            "\"state\" must be provided"
        );
        assert_eq!(
            message(parse(&json!({"state": null, "questions": {}}))),
            "\"state\" must be provided"
        );
        assert_eq!(
            message(parse(&json!({"state": "s", "questions": {}}))),
            "\"questions\" must be a non-empty object"
        );
        assert_eq!(
            message(parse(&json!({"state": "s", "questions": []}))),
            "\"questions\" must be a non-empty object"
        );
        assert_eq!(
            message(parse(&q(json!("x")))),
            "questions.q: must be an object"
        );
        assert_eq!(
            message(parse(&q(json!({"type": "noul"})))),
            "questions.q: \"instructions\" must be provided"
        );
        assert_eq!(
            message(parse(&q(json!({"type": "rank", "instructions": "i"})))),
            "questions.q: \"type\" must be one of: choice, score, noul"
        );
        assert_eq!(
            message(parse(&q(json!({"type": 3, "instructions": "i"})))),
            "questions.q: \"type\" must be one of: choice, score, noul"
        );
        assert_eq!(
            message(parse(&q(
                json!({"type": "choice", "instructions": "i", "criteria": {}})
            ))),
            "questions.q: \"criteria\" must be a non-empty object"
        );
        assert_eq!(
            message(parse(&q(
                json!({"type": "choice", "instructions": "i", "criteria": ["a"]})
            ))),
            "questions.q: \"criteria\" must be a non-empty object"
        );
        assert_eq!(
            message(parse(&q(
                json!({"type": "noul", "instructions": "i", "criteria": ["a"]})
            ))),
            "questions.q: \"criteria\" must be an object"
        );
    }

    #[test]
    fn score_needs_two_to_ten_levels() {
        let q = |n: usize| {
            json!({"state": "s", "questions": {"q": {
                "type": "score", "instructions": "i", "criteria": vec!["l"; n]}}})
        };
        for bad in [0, 1, 11] {
            assert_eq!(
                message(parse(&q(bad))),
                "questions.q: \"criteria\" must be an array of 2 to 10 levels"
            );
        }
        for good in [2, 10] {
            assert_eq!(parse(&q(good)).expect("valid")["q"].options.len(), good);
        }
    }

    #[test]
    fn too_many_options_names_the_limit() {
        let criteria: Object = (0..53).map(|i| (format!("k{i}"), Json::Null)).collect();
        let body = json!({"state": "s", "questions": {"q": {
            "type": "choice", "instructions": "i", "criteria": criteria}}});
        let letters = QuestionRules {
            n_options_max: 52,
            ..rules_of("openjev")
        };
        assert_eq!(
            message(parse_questions(&body, letters)),
            "questions.q: too many options (53), this model supports at most 52"
        );
        assert!(parse(&body).is_ok());
    }

    fn openjev_like() -> Readout {
        Readout::new(
            kind("openjev"),
            Head::LabelLogits {
                labels: vec![0; 52],
                label_texts: Vec::new(),
            },
        )
    }

    fn with_state(state: Json) -> Json {
        json!({"state": state, "images": ["data:image/png;base64,AAAA"],
               "questions": {"q": {"type": "noul", "instructions": "i"}}})
    }

    #[test]
    fn media_come_from_files_images_and_chat_messages() {
        let state = json!({"messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBBB"}},
                {"type": "input_audio", "input_audio": {"data": "QUFB"}},
            ]},
            {"role": "assistant", "content": "plain"},
        ]});
        let mut body = with_state(state);
        body.set("files", json!(["data:image/png;base64,FFFF"]));
        let request = SystemOneRequest::parse(&body, &openjev_like()).expect("valid");
        assert_eq!(
            request.images,
            [
                "data:image/png;base64,FFFF",
                "data:image/png;base64,AAAA",
                "data:image/png;base64,BBBB",
                "QUFB"
            ]
        );
        assert_eq!(
            request.state,
            json!({"messages": [
                {"role": "user", "content": [{"type": "text", "text": "look"}]},
                {"role": "assistant", "content": "plain"},
            ]})
        );
    }

    #[test]
    fn bare_message_array_state_stays_an_array() {
        let state = json!([{"role": "user", "content": [
            {"type": "image_url", "image_url": "data:image/png;base64,BBBB"}]}]);
        let request = SystemOneRequest::parse(&with_state(state), &openjev_like()).expect("valid");
        assert_eq!(request.state, json!([{"role": "user", "content": []}]));
        assert_eq!(request.images.len(), 2);
    }

    #[test]
    fn media_are_validated_and_bounded() {
        let mut body = with_state(json!("s"));
        body.set("images", json!(["https://example.com/a.png"]));
        assert!(
            message(SystemOneRequest::parse(&body, &openjev_like()))
                .starts_with("media must be data URLs")
        );

        body.set("images", json!(vec!["data:image/png;base64,AA"; 9]));
        assert_eq!(
            message(SystemOneRequest::parse(&body, &openjev_like())),
            "too many images, the maximum is 8"
        );

        body.set("images", json!("x"));
        assert_eq!(
            message(SystemOneRequest::parse(&body, &openjev_like())),
            "\"images\" must be an array"
        );

        let mut body = with_state(json!("s"));
        body.set("videos", json!(["data:video/mp4;base64,AA"]));
        assert_eq!(
            message(SystemOneRequest::parse(&body, &openjev_like())),
            "\"videos\" is not supported"
        );
        body.set("videos", json!([]));
        assert!(SystemOneRequest::parse(&body, &openjev_like()).is_ok());

        let state = json!([{"content": [{"type": "input_audio", "input_audio": {"url": "http://x/a.wav"}}]}]);
        assert_eq!(
            message(SystemOneRequest::parse(&with_state(state), &openjev_like())),
            "audio must be base64 data"
        );
    }

    #[test]
    fn images_need_a_readout_with_a_place_for_them() {
        assert!(matches!(
            SystemOneRequest::parse(&with_state(json!("s")), &laya()),
            Err(DecisionError::ImagesUnsupported)
        ));
    }

    #[test]
    fn from_slice_reports_malformed_json() {
        assert!(message(SystemOneRequest::from_slice(b"{", &laya())).starts_with("invalid JSON"));
    }

    fn question_of(name: &str, question: Json) -> Result<Question, DecisionError> {
        parse_question("q", &question, rules_of(name))
    }

    #[test]
    fn strict_payload_types_refuse_what_is_not_text_an_object_or_an_array() {
        let strict = labelled("vllm-decision2", 255);
        let body =
            |state: Json, question: Json| json!({"state": state, "questions": {"q": question}});
        let choice =
            json!({"type": "choice", "instructions": "i", "criteria": {"a": null, "b": "x"}});
        assert!(SystemOneRequest::parse(&body(json!("s"), choice.clone()), &strict).is_ok());
        assert_eq!(
            message(SystemOneRequest::parse(
                &body(json!(3), choice.clone()),
                &strict
            )),
            "\"state\" must be text, an object or an array"
        );
        for (question, expected) in [
            (
                json!({"type": "choice", "instructions": "", "criteria": {"a": null, "b": null}}),
                "questions.q: \"instructions\" must be non-empty text, an object or an array",
            ),
            (
                json!({"type": "choice", "instructions": 4, "criteria": {"a": null, "b": null}}),
                "questions.q: \"instructions\" must be non-empty text, an object or an array",
            ),
            (
                json!({"type": "choice", "instructions": "i", "criteria": {"": null, "b": null}}),
                "questions.q: option keys must be non-empty, descriptions must be null, text, an object or an array",
            ),
            (
                json!({"type": "choice", "instructions": "i", "criteria": {"a": 1, "b": null}}),
                "questions.q: option keys must be non-empty, descriptions must be null, text, an object or an array",
            ),
            (
                json!({"type": "choice", "instructions": "i", "criteria": {"a": null}}),
                "questions.q: \"criteria\" must have at least 2 options",
            ),
            (
                json!({"type": "score", "instructions": "i", "criteria": ["lo", null]}),
                "questions.q: level descriptions must be text, an object or an array",
            ),
            (
                json!({"type": "noul", "instructions": "i", "criteria": {"yes": "y"}}),
                "questions.q: \"criteria\" may only describe \"false\" and \"true\" with text, an object or an array",
            ),
        ] {
            assert_eq!(
                message(SystemOneRequest::parse(
                    &body(json!("s"), question),
                    &strict
                )),
                expected
            );
        }
    }

    #[test]
    fn criteria_order_puts_true_first_only_when_the_criteria_do() {
        let options = |text: &str| -> Vec<String> {
            question_of("vllm-decision2", crate::ordered(text))
                .expect("valid")
                .options
                .into_iter()
                .map(|option| option.key)
                .collect()
        };
        assert_eq!(
            options(
                r#"{"type": "noul", "instructions": "i", "criteria": {"true": "t", "false": "f"}}"#
            ),
            ["true", "false"]
        );
        assert_eq!(
            options(
                r#"{"type": "noul", "instructions": "i", "criteria": {"false": "f", "true": "t"}}"#
            ),
            ["false", "true"]
        );
        assert_eq!(
            options(r#"{"type": "noul", "instructions": "i", "criteria": {"true": "t"}}"#),
            ["false", "true"]
        );
    }

    #[test]
    fn sorted_types_list_the_criteria_by_key() {
        let question = question_of(
            "clef",
            crate::ordered(r#"{"type": "choice", "instructions": "i", "criteria": {"b": null, "B": null, "a": null}}"#),
        )
        .expect("valid");
        let keys: Vec<_> = question.options.iter().map(|o| o.key.as_str()).collect();
        assert_eq!(keys, ["B", "a", "b"]);
    }

    #[test]
    fn lenient_types_default_the_type_and_the_instructions() {
        let question =
            question_of("h2o-lightning", json!({"criteria": {"a": null}})).expect("valid");
        assert_eq!(question.kind, QuestionKind::Choice);
        assert_eq!(question.instructions, json!(""));
        let score = question_of(
            "h2o-lightning",
            json!({"type": "score", "criteria": ["only"]}),
        )
        .expect("one level");
        assert_eq!(score.options.len(), 1);
        assert_eq!(
            message(question_of(
                "h2o-lightning",
                json!({"type": "score", "criteria": []})
            )),
            "questions.q: \"criteria\" must be a non-empty array"
        );
        // The descriptions of a noul are also read under no and yes.
        let noul = question_of(
            "h2o-lightning",
            json!({"type": "noul", "criteria": {"yes": "Y", "false": "F", "no": "N"}}),
        )
        .expect("valid");
        assert_eq!(noul.options[0].key, "true");
        assert_eq!(noul.options[0].description, json!("Y"));
        assert_eq!(noul.options[1].description, json!("F"));
        // A standard type ignores the aliases.
        let plain = question_of(
            "laya",
            json!({"type": "noul", "instructions": "i", "criteria": {"yes": "Y"}}),
        )
        .expect("valid");
        assert!(plain.options[1].description.is_null());
    }

    #[test]
    fn optional_instructions_and_null_states() {
        assert!(question_of("vllm-d3", json!({"type": "noul"})).is_ok());
        assert!(question_of("vllm-d3", json!({"type": "noul", "instructions": null})).is_ok());
        let body =
            json!({"state": null, "questions": {"q": {"type": "noul", "instructions": "i"}}});
        assert!(SystemOneRequest::parse(&body, &labelled("lfm2-d1", 255)).is_ok());
        assert_eq!(
            message(SystemOneRequest::parse(
                &body,
                &labelled("pplx-decider", 255)
            )),
            "\"state\" must be provided"
        );
        let missing = json!({"questions": {"q": {"type": "noul", "instructions": "i"}}});
        assert_eq!(
            message(SystemOneRequest::parse(&missing, &labelled("lfm2-d1", 255))),
            "\"state\" must be provided"
        );
    }

    #[test]
    fn spoken_instructions_are_media_refused_or_plain_by_type() {
        let spoken = json!({"type": "input_audio", "input_audio": {"data": "QUFB"}});
        let body =
            json!({"state": "s", "questions": {"q": {"type": "noul", "instructions": spoken}}});
        assert!(matches!(
            SystemOneRequest::parse(&body, &labelled("lfm3-d1-omni", 255)),
            Err(DecisionError::ImagesUnsupported)
        ));
        assert!(matches!(
            SystemOneRequest::parse(&body, &laya()),
            Err(DecisionError::ImagesUnsupported)
        ));
        assert_eq!(
            message(SystemOneRequest::parse(&body, &labelled("lfm3-d1", 255))),
            "questions.q: spoken instructions need a model with audio input"
        );
        // A joint prompt is built from the raw values: an object like any other.
        assert!(SystemOneRequest::parse(&body, &labelled("clef", 255)).is_ok());
        let bad = json!({"state": "s", "questions": {"q": {"type": "noul",
            "instructions": {"type": "input_audio", "input_audio": {"url": "http://x"}}}}});
        assert_eq!(
            message(SystemOneRequest::parse(&bad, &laya())),
            "audio must be base64 data"
        );
    }

    #[test]
    fn video_parts_are_media_for_the_models_that_read_them() {
        let body = json!({"state": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AA"}}]}],
            "questions": {"q": {"type": "noul", "instructions": "i"}}});
        let lfm3 = SystemOneRequest::parse(&body, &labelled("lfm3-d1", 255)).expect("valid");
        assert_eq!(lfm3.images, ["data:video/mp4;base64,AA"]);
        assert_eq!(lfm3.state, json!([{"role": "user", "content": []}]));
        let openjev = SystemOneRequest::parse(&body, &openjev_like()).expect("valid");
        assert!(openjev.images.is_empty());
    }

    #[test]
    fn the_ocr_consistency_question_is_fixed_and_needs_two_images() {
        let ocr = Readout::new(
            kind("openbmb-minicpm-ocr-jev"),
            Head::Pooled { n_classes: 2 },
        );
        let body = |state: Json, question: Json, images: usize| {
            json!({"state": state, "questions": {"q": question},
                   "images": vec!["data:image/png;base64,AA"; images]})
        };
        let choice = json!({"type": "choice", "criteria": {"1": null, "0": null}});
        assert_eq!(
            message(SystemOneRequest::parse(
                &body(json!("text"), choice.clone(), 0),
                &ocr
            )),
            "this model needs exactly 2 images (the original, then the rendered one), got 0"
        );
        // The two images make a valid request, which a runtime without a
        // multimodal projector refuses as media.
        let valid =
            SystemOneRequest::parse(&body(json!("text"), choice.clone(), 2), &ocr).expect("valid");
        assert_eq!(valid.images.len(), 2);
        assert_eq!(valid.questions["q"].options[0].key, "0");
        assert_eq!(
            message(SystemOneRequest::parse(
                &body(json!({"a": 1}), choice.clone(), 2),
                &ocr
            )),
            "\"state\" must be the OCR text"
        );
        let two = json!({"state": "t", "questions": {"a": choice.clone(), "b": choice.clone()}});
        assert_eq!(
            message(SystemOneRequest::parse(&two, &ocr)),
            "this model answers exactly one question"
        );
        for (question, expected) in [
            (
                json!({"type": "choice", "instructions": "i", "criteria": {"0": null, "1": null}}),
                "questions.q: \"instructions\" is not used by this model, the task is part of its prompt",
            ),
            (
                json!({"type": "choice", "criteria": {"0": "no", "1": null}}),
                "questions.q: \"criteria\" must be {\"0\": null, \"1\": null}",
            ),
            (
                json!({"type": "noul", "criteria": {"true": "t"}}),
                "questions.q: \"criteria\" is not supported for a noul question",
            ),
            (
                json!({"type": "score", "criteria": ["a", "b"]}),
                "questions.q: only choice and noul questions are supported",
            ),
        ] {
            assert_eq!(
                message(SystemOneRequest::parse(
                    &body(json!("t"), question, 2),
                    &ocr
                )),
                expected
            );
        }
    }
}
