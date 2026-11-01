//! The `/v1/systemone` request body and its validation.
//!
//! The messages of a rejected request are the ones llama-server returns, so a
//! client sees the same sentence from both.

use crate::json::Json;
use indexmap::IndexMap;

use crate::DecisionError;
use crate::readout::Readout;

/// Media a request may carry, images and audio together.
const MAX_MEDIA: usize = 8;

/// Levels a `score` question may have.
const SCORE_LEVELS: std::ops::RangeInclusive<usize> = 2..=10;

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
    /// `choice`: the criteria in request order. `score`: one option per level,
    /// keyed by its index. `noul`: `false` then `true`, or the other way round
    /// for a readout that lists `true` first.
    pub options: Vec<QuestionOption>,
}

/// What a question is checked against. A readout fixes it
/// ([`QuestionRules::of`]); a dataset uses [`QuestionRules::generic`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuestionRules {
    pub n_options_max: usize,
    pub noul_true_first: bool,
}

impl QuestionRules {
    pub fn of(readout: &Readout) -> Self {
        Self {
            n_options_max: readout.n_options_max(),
            noul_true_first: readout.noul_true_first(),
        }
    }

    /// The limits no readout exceeds, `noul` as `[false, true]`.
    pub fn generic() -> Self {
        Self {
            n_options_max: 255,
            noul_true_first: false,
        }
    }
}

/// A validated request. `questions` keeps the order of the body, which is the
/// order of the answers.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneRequest {
    /// The state without its media: text, an object or an array.
    pub state: Json,
    /// Media as data URLs (or base64 audio), in the order they appear: the
    /// `files` and `images` arrays, then the messages of the state.
    pub images: Vec<String>,
    pub questions: IndexMap<String, Question>,
}

impl SystemOneRequest {
    /// Parses and validates a body.
    pub fn from_slice(body: &[u8], readout: &Readout) -> Result<Self, DecisionError> {
        let body: Json = serde_json::from_slice(body)
            .map_err(|error| DecisionError::invalid(format!("invalid JSON: {error}")))?;
        Self::parse(&body, readout)
    }

    pub fn parse(body: &Json, readout: &Readout) -> Result<Self, DecisionError> {
        let questions = parse_questions(body, QuestionRules::of(readout))?;
        let mut images = Vec::new();
        let state = parse_state(body, &mut images)?;
        if !images.is_empty() && !readout.accepts_images() {
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
    if body.get("state").is_none_or(Json::is_null) {
        return Err(DecisionError::invalid("\"state\" must be provided"));
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
    let err = |message: &str| DecisionError::invalid(format!("questions.{id}: {message}"));
    let Some(question) = question.as_object() else {
        return Err(err("must be an object"));
    };
    let instructions = question.get("instructions").cloned().unwrap_or(Json::Null);
    if instructions.is_null() {
        return Err(err("\"instructions\" must be provided"));
    }
    let criteria = question.get("criteria").unwrap_or(&Json::Null);

    let (kind, mut options) = match question.get("type").and_then(Json::as_str) {
        Some("choice") => {
            let Some(criteria) = criteria.as_object().filter(|criteria| !criteria.is_empty())
            else {
                return Err(err("\"criteria\" must be a non-empty object"));
            };
            let options: Vec<QuestionOption> = criteria
                .iter()
                .map(|(key, description)| QuestionOption {
                    key: key.clone(),
                    description: description.clone(),
                })
                .collect();
            (QuestionKind::Choice, options)
        }
        Some("score") => {
            let Some(levels) = criteria
                .as_array()
                .filter(|levels| SCORE_LEVELS.contains(&levels.len()))
            else {
                return Err(err("\"criteria\" must be an array of 2 to 10 levels"));
            };
            let options: Vec<QuestionOption> = levels
                .iter()
                .enumerate()
                .map(|(level, description)| QuestionOption {
                    key: level.to_string(),
                    description: description.clone(),
                })
                .collect();
            (QuestionKind::Score, options)
        }
        Some("noul") => {
            if !criteria.is_null() && !criteria.is_object() {
                return Err(err("\"criteria\" must be an object"));
            }
            let options: Vec<QuestionOption> = ["false", "true"]
                .into_iter()
                .map(|key| QuestionOption {
                    key: key.to_owned(),
                    description: criteria.get(key).cloned().unwrap_or(Json::Null),
                })
                .collect();
            (QuestionKind::Noul, options)
        }
        _ => return Err(err("\"type\" must be one of: choice, score, noul")),
    };

    if kind == QuestionKind::Noul && rules.noul_true_first {
        options.swap(0, 1);
    }
    if options.len() > rules.n_options_max {
        return Err(err(&format!(
            "too many options ({}), this model supports at most {}",
            options.len(),
            rules.n_options_max
        )));
    }
    Ok(Question {
        kind,
        instructions,
        options,
    })
}

/// Collects the media of the body into `media` and returns the state without
/// the media parts of its messages.
fn parse_state(body: &Json, media: &mut Vec<String>) -> Result<Json, DecisionError> {
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

    // `parse_questions` has checked that the state is there.
    let state = body.get("state").unwrap_or(&Json::Null);
    let wrapped = state.get("messages");
    let messages = wrapped.unwrap_or(state);
    let Json::Array(messages) = messages else {
        return Ok(state.clone());
    };

    // Chat messages: the image and audio parts leave the content.
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
            ) {
                (Some("image_url"), Some(image_url), _) => {
                    load_media(image_url.get("url").unwrap_or(image_url), media)?;
                }
                (Some("input_audio"), _, Some(audio)) => {
                    load_audio(audio.get("data").or_else(|| audio.get("url")), media)?;
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
        Readout::MarkerColumn {
            marker: 4,
            sep: 1,
            marker_text: "<mask>".into(),
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

        let swapped = parse_questions(
            &body,
            QuestionRules {
                n_options_max: 255,
                noul_true_first: true,
            },
        )
        .expect("valid");
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
            noul_true_first: true,
        };
        assert_eq!(
            message(parse_questions(&body, letters)),
            "questions.q: too many options (53), this model supports at most 52"
        );
        assert!(parse(&body).is_ok());
    }

    fn openjev_like() -> Readout {
        Readout::LabelLogits {
            labels: vec![0; 52],
            label_texts: Vec::new(),
            n_variants: 1,
            noul_scale: None,
            family: crate::readout::LabelFamily::Letters,
        }
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
}
