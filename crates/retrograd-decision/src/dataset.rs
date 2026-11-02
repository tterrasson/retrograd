//! Training examples for decision models.
//!
//! One JSON object per line. The native format is aligned on the request of
//! the service, one question per line:
//!
//! ```json
//! {"state": "...", "question": {"type": "choice", "instructions": "...",
//!   "criteria": {"billing": "...", "shipping": null}},
//!  "answer": "billing"}
//! ```
//!
//! `answer` is the key of the option (`choice`), the index of the level
//! (`score`) or a boolean (`noul`). It can be replaced by `probabilities`, soft
//! targets keyed by option key (`"0"`, `"1"`, ... for the levels of a score,
//! `"false"` and `"true"` for a `noul`): distillation, or an ordinal label
//! smoothing. Exactly one of the two is given.
//!
//! A line may also ask several questions of the same state, as the service
//! does: `"questions": {id: {..., "answer": ...}}`, each question carrying its
//! own `answer` or `probabilities`. It expands to one example per question.
//!
//! The `julia` format is the one of the original training code
//! (`julia/data.py`): `state`, `question` (text), `options` (2 to 20 rendered
//! descriptions), `type` (default `choice`), `target` (index of the right
//! option) and optionally `teacher_logits`. An option has no key there, so a
//! choice gets the keys `"0"`, `"1"`, ...; when `teacher_logits` is present
//! the example is distilled from `softmax(teacher_logits)` and `target` is not
//! read.

use std::path::Path;

use crate::json::{Json, Object};

use crate::DecisionError;
use crate::answer::softmax;
use crate::request::{Question, QuestionKind, QuestionOption, QuestionRules, parse_question};

/// Options of a question in the julia format.
const JULIA_OPTIONS: std::ops::RangeInclusive<usize> = 2..=20;

/// How far the sum of a `probabilities` object may be from 1.
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-3;

/// The layout of a training file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DataFormat {
    Native,
    Julia,
}

/// What the model should output for an example.
#[derive(Clone, Debug, PartialEq)]
pub enum Target {
    /// One right option, as an index into `Question::options`.
    Hard(usize),
    /// A probability for each option of `Question::options`, summing to 1.
    Soft(Vec<f64>),
}

impl Target {
    /// The target as a distribution over `n` options.
    pub fn distribution(&self, n: usize) -> Vec<f64> {
        match self {
            Self::Hard(index) => {
                let mut target = vec![0.0; n];
                target[*index] = 1.0;
                target
            }
            Self::Soft(probabilities) => probabilities.clone(),
        }
    }
}

/// One question about a state, with its right answer.
#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    pub state: Json,
    pub question_id: String,
    pub question: Question,
    pub target: Target,
}

impl Example {
    /// Aligns a dataset's canonical false/true order with the model's readout.
    /// Target mass follows the option key, including a hard boolean target.
    pub fn for_readout(mut self, readout: &crate::readout::Readout) -> Self {
        if self.question.kind == QuestionKind::Noul
            && readout.noul_true_first()
            && self
                .question
                .options
                .first()
                .is_some_and(|o| o.key == "false")
        {
            self.question.options.swap(0, 1);
            self.target = match self.target {
                Target::Hard(index) => Target::Hard(1 - index),
                Target::Soft(mut probabilities) => {
                    probabilities.swap(0, 1);
                    Target::Soft(probabilities)
                }
            };
        }
        self
    }
}

/// Reads a training file.
pub fn read_jsonl(path: &Path, format: DataFormat) -> Result<Vec<Example>, DecisionError> {
    let text = std::fs::read_to_string(path)?;
    parse_jsonl(&text, &path.display().to_string(), format)
}

/// Parses the text of a training file. `path` names it in the errors. Blank
/// lines are skipped; a file with no example is an error.
pub fn parse_jsonl(
    text: &str,
    path: &str,
    format: DataFormat,
) -> Result<Vec<Example>, DecisionError> {
    let at = |line: usize, message: String| DecisionError::Dataset {
        path: path.to_owned(),
        line,
        message,
    };
    let mut examples = Vec::new();
    // Whether the first julia row has teacher logits, and the line of it.
    let mut teacher_first: Option<bool> = None;
    for (index, text) in text.lines().enumerate() {
        let line = index + 1;
        if text.trim().is_empty() {
            continue;
        }
        let row: Json = serde_json::from_str(text)
            .map_err(|error| at(line, format!("invalid JSON: {error}")))?;
        let parsed = match format {
            DataFormat::Native => parse_native(&row),
            DataFormat::Julia => parse_julia(&row),
        }
        .map_err(|message| at(line, message))?;
        if format == DataFormat::Julia {
            let has_teacher = row
                .get("teacher_logits")
                .is_some_and(|teacher| !teacher.is_null());
            match teacher_first {
                None => teacher_first = Some(has_teacher),
                Some(first) if first != has_teacher => {
                    return Err(at(
                        line,
                        "the dataset mixes rows with and without teacher logits".into(),
                    ));
                }
                Some(_) => {}
            }
        }
        examples.extend(parsed);
    }
    if examples.is_empty() {
        return Err(at(0, "dataset contains no examples".into()));
    }
    Ok(examples)
}

fn parse_native(row: &Json) -> Result<Vec<Example>, String> {
    let Some(row) = row.as_object() else {
        return Err("a line must be a JSON object".into());
    };
    let state = match row.get("state") {
        None | Some(Json::Null) => return Err("\"state\" must be provided".into()),
        Some(state) => state,
    };
    let rules = QuestionRules::generic();
    let one = |id: &str, question: &Json, target_at: &Object| {
        let question = parse_question(id, question, rules).map_err(|error| error.to_string())?;
        let target = parse_target(target_at, &question)?;
        Ok(Example {
            state: state.clone(),
            question_id: id.to_owned(),
            question,
            target,
        })
    };

    match (row.get("question"), row.get("questions")) {
        (Some(question), None) => Ok(vec![one("question", question, row)?]),
        (None, Some(Json::Object(questions))) if !questions.is_empty() => questions
            .iter()
            .map(|(id, question)| {
                let Some(question_object) = question.as_object() else {
                    return Err(format!("questions.{id}: must be an object"));
                };
                one(id, question, question_object)
            })
            .collect(),
        (None, Some(_)) => Err("\"questions\" must be a non-empty object".into()),
        (None, None) => Err("one of \"question\" and \"questions\" must be provided".into()),
        (Some(_), Some(_)) => Err("\"question\" and \"questions\" are exclusive".into()),
    }
}

/// The target of `question`, read from the `answer` or `probabilities` member
/// of `holder`.
fn parse_target(holder: &Object, question: &Question) -> Result<Target, String> {
    let n = question.options.len();
    match (holder.get("answer"), holder.get("probabilities")) {
        (Some(answer), None) => {
            let index = match (question.kind, answer) {
                (QuestionKind::Choice, Json::String(key)) => question
                    .options
                    .iter()
                    .position(|option| option.key == *key)
                    .ok_or_else(|| format!("\"answer\" {key:?} is not one of the criteria keys"))?,
                (QuestionKind::Choice, _) => {
                    return Err("\"answer\" of a choice question must be a criteria key".into());
                }
                (QuestionKind::Score, answer) => {
                    let level = answer
                        .as_u64()
                        .and_then(|level| usize::try_from(level).ok())
                        .filter(|level| *level < n);
                    level.ok_or_else(|| {
                        format!(
                            "\"answer\" of a score question must be a level index from 0 to {}",
                            n - 1
                        )
                    })?
                }
                (QuestionKind::Noul, Json::Bool(truth)) => position_of_truth(question, *truth)?,
                (QuestionKind::Noul, _) => {
                    return Err("\"answer\" of a noul question must be true or false".into());
                }
            };
            Ok(Target::Hard(index))
        }
        (None, Some(probabilities)) => {
            parse_probabilities(probabilities, question).map(Target::Soft)
        }
        (None, None) => Err("\"answer\" or \"probabilities\" must be provided".into()),
        (Some(_), Some(_)) => Err("\"answer\" and \"probabilities\" are exclusive".into()),
    }
}

fn position_of_truth(question: &Question, truth: bool) -> Result<usize, String> {
    let key = if truth { "true" } else { "false" };
    question
        .options
        .iter()
        .position(|option| option.key == key)
        .ok_or_else(|| "a noul question has no true and false options".to_owned())
}

fn parse_probabilities(value: &Json, question: &Question) -> Result<Vec<f64>, String> {
    let Some(object) = value.as_object() else {
        return Err("\"probabilities\" must be an object keyed by option".into());
    };
    let mut probabilities = Vec::with_capacity(question.options.len());
    for option in &question.options {
        let Some(p) = object.get(&option.key).and_then(Json::as_f64) else {
            return Err(format!(
                "\"probabilities\" has no number for option {:?}",
                option.key
            ));
        };
        if !p.is_finite() || p < 0.0 {
            return Err(format!(
                "probability of option {:?} must be finite and not negative",
                option.key
            ));
        }
        probabilities.push(p);
    }
    if object.len() != probabilities.len() {
        return Err("\"probabilities\" has keys that are not options".into());
    }
    let sum: f64 = probabilities.iter().sum();
    if (sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
        return Err(format!("\"probabilities\" must sum to 1, got {sum}"));
    }
    probabilities.iter_mut().for_each(|p| *p /= sum);
    Ok(probabilities)
}

/// A row of the original training code, validated as `validate_row` does.
fn parse_julia(row: &Json) -> Result<Vec<Example>, String> {
    let Some(row) = row.as_object() else {
        return Err("request must be a JSON object".into());
    };
    let state = row
        .get("state")
        .filter(|state| state.is_string() || state.is_object() || state.is_array());
    let instructions = row.get("question").filter(|question| question.is_string());
    let (Some(state), Some(instructions)) = (state, instructions) else {
        return Err("state must be text/JSON and question must be text".into());
    };
    let descriptions: Vec<&str> = row
        .get("options")
        .and_then(Json::as_array)
        .filter(|options| JULIA_OPTIONS.contains(&options.len()))
        .and_then(|options| {
            options
                .iter()
                .map(|option| option.as_str().filter(|text| !text.is_empty()))
                .collect()
        })
        .ok_or("options must contain 2-20 nonempty rendered descriptions")?;
    let kind = match row.get("type") {
        None => QuestionKind::Choice,
        Some(Json::String(name)) => match name.as_str() {
            "choice" => QuestionKind::Choice,
            "score" => QuestionKind::Score,
            "noul" => QuestionKind::Noul,
            _ => return Err("type must be choice, score, or noul".into()),
        },
        Some(_) => return Err("type must be choice, score, or noul".into()),
    };
    if kind == QuestionKind::Noul && descriptions.len() != 2 {
        return Err("noul options must be ordered [false, true]".into());
    }
    let n = descriptions.len();

    let teacher = match row.get("teacher_logits") {
        None | Some(Json::Null) => None,
        Some(Json::Array(logits)) => {
            let logits: Option<Vec<f64>> = logits
                .iter()
                .map(|logit| logit.as_f64().filter(|logit| logit.is_finite()))
                .collect();
            match logits {
                Some(logits) if logits.len() == n => Some(logits),
                _ => {
                    return Err("teacher logits must be finite and match option count/order".into());
                }
            }
        }
        Some(_) => return Err("teacher logits must be finite and match option count/order".into()),
    };
    let target = match (&teacher, row.get("target")) {
        (Some(logits), _) => Target::Soft(softmax(logits, 1.0)),
        (None, None) => return Err("training requires target".into()),
        (None, Some(target)) => {
            let index = target
                .as_u64()
                .and_then(|index| usize::try_from(index).ok())
                .filter(|index| *index < n);
            Target::Hard(index.ok_or("target must index the supplied option list")?)
        }
    };

    let options = descriptions
        .iter()
        .enumerate()
        .map(|(i, text)| QuestionOption {
            key: match kind {
                QuestionKind::Noul => ["false", "true"][i].to_owned(),
                _ => i.to_string(),
            },
            description: Json::String((*text).to_owned()),
        })
        .collect();
    Ok(vec![Example {
        state: state.clone(),
        question_id: "question".to_owned(),
        question: Question {
            kind,
            instructions: instructions.clone(),
            options,
        },
        target,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str, format: DataFormat) -> Result<Vec<Example>, DecisionError> {
        parse_jsonl(text, "train.jsonl", format)
    }

    fn native(text: &str) -> Vec<Example> {
        parse(text, DataFormat::Native).expect("valid dataset")
    }

    fn error_at(text: &str, format: DataFormat) -> (usize, String) {
        match parse(text, format) {
            Err(DecisionError::Dataset {
                path,
                line,
                message,
            }) => {
                assert_eq!(path, "train.jsonl");
                (line, message)
            }
            other => panic!("expected a dataset error, got {other:?}"),
        }
    }

    const CHOICE: &str = r#"{"state": "s", "question": {"type": "choice", "instructions": "i", "criteria": {"billing": "pay", "shipping": null, "other": null}}, "answer": "shipping"}"#;

    #[test]
    fn a_choice_line_gives_the_index_of_its_answer() {
        let examples = native(CHOICE);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].target, Target::Hard(1));
        assert_eq!(examples[0].question.options[1].key, "shipping");
        assert_eq!(examples[0].question_id, "question");
        assert_eq!(examples[0].target.distribution(3), [0.0, 1.0, 0.0]);
    }

    #[test]
    fn score_answers_are_level_indices_and_noul_answers_are_booleans() {
        let text = r#"
{"state": ["a"], "question": {"type": "score", "instructions": "i", "criteria": ["lo", "mid", "hi"]}, "answer": 2}

{"state": {"k": 1}, "question": {"type": "noul", "instructions": "i"}, "answer": true}
{"state": "s", "question": {"type": "noul", "instructions": "i"}, "answer": false}
"#;
        let examples = native(text);
        let targets: Vec<_> = examples.iter().map(|e| e.target.clone()).collect();
        assert_eq!(targets, [Target::Hard(2), Target::Hard(1), Target::Hard(0)]);
        // `noul` is [false, true] whatever the serving readout.
        assert_eq!(examples[1].question.options[1].key, "true");
        assert_eq!(examples[1].state, json!({"k": 1}));
    }

    #[test]
    fn probabilities_are_soft_targets_keyed_by_option() {
        let text = r#"{"state": "s", "question": {"type": "score", "instructions": "i", "criteria": ["a", "b", "c"]}, "probabilities": {"0": 0.2, "1": 0.5, "2": 0.3}}"#;
        assert_eq!(native(text)[0].target, Target::Soft(vec![0.2, 0.5, 0.3]));

        let noul = r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "probabilities": {"true": 0.9, "false": 0.1}}"#;
        assert_eq!(native(noul)[0].target, Target::Soft(vec![0.1, 0.9]));

        // Slightly off 1 is normalized.
        let off = r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "probabilities": {"true": 0.5, "false": 0.5004}}"#;
        let Target::Soft(p) = &native(off)[0].target else {
            panic!("soft target");
        };
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_state_with_several_questions_expands_to_one_example_each() {
        let text = concat!(
            r#"{"state": {"msg": "hi"}, "questions": {"#,
            r#""intent": {"type": "choice", "instructions": "i", "criteria": {"a": null, "b": null}, "answer": "b"}, "#,
            r#""urgent": {"type": "noul", "instructions": "u", "answer": true}, "#,
            r#""level": {"type": "score", "instructions": "l", "criteria": ["x", "y"], "probabilities": {"0": 0.25, "1": 0.75}}"#,
            "}}"
        );
        let examples = native(text);
        let ids: Vec<_> = examples.iter().map(|e| e.question_id.as_str()).collect();
        assert_eq!(ids, ["intent", "urgent", "level"]);
        assert!(examples.iter().all(|e| e.state == json!({"msg": "hi"})));
        assert_eq!(examples[0].target, Target::Hard(1));
        assert_eq!(examples[1].target, Target::Hard(1));
        assert_eq!(examples[2].target, Target::Soft(vec![0.25, 0.75]));
    }

    #[test]
    fn errors_name_the_line_and_what_is_wrong() {
        let bad_answer = CHOICE.replace("shipping\"}", "teleport\"}");
        let text = format!("{CHOICE}\n\n{bad_answer}\n");
        let (line, message) = error_at(&text, DataFormat::Native);
        assert_eq!(line, 3);
        assert!(
            message.contains("\"teleport\" is not one of the criteria keys"),
            "{message}"
        );

        let cases = [
            ("{", "invalid JSON"),
            ("[]", "a line must be a JSON object"),
            (r#"{"question": {}}"#, "\"state\" must be provided"),
            (
                r#"{"state": "s"}"#,
                "one of \"question\" and \"questions\" must be provided",
            ),
            (
                r#"{"state": "s", "question": {}, "questions": {}}"#,
                "exclusive",
            ),
            (
                r#"{"state": "s", "question": {"type": "choice", "instructions": "i", "criteria": {}}, "answer": "a"}"#,
                "\"criteria\" must be a non-empty object",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}}"#,
                "\"answer\" or \"probabilities\" must be provided",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "answer": true, "probabilities": {}}"#,
                "\"answer\" and \"probabilities\" are exclusive",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "answer": "yes"}"#,
                "must be true or false",
            ),
            (
                r#"{"state": "s", "question": {"type": "score", "instructions": "i", "criteria": ["a", "b"]}, "answer": 2}"#,
                "level index from 0 to 1",
            ),
            (
                r#"{"state": "s", "question": {"type": "score", "instructions": "i", "criteria": ["a", "b"]}, "answer": 1.5}"#,
                "level index",
            ),
            (
                r#"{"state": "s", "question": {"type": "choice", "instructions": "i", "criteria": {"a": null}}, "answer": 0}"#,
                "must be a criteria key",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "probabilities": {"true": 0.6, "false": 0.6}}"#,
                "must sum to 1",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "probabilities": {"true": 1.0}}"#,
                "no number for option \"false\"",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "probabilities": {"true": 1.0, "false": 0.0, "maybe": 0.0}}"#,
                "keys that are not options",
            ),
            (
                r#"{"state": "s", "question": {"type": "noul", "instructions": "i"}, "probabilities": {"true": 1.5, "false": -0.5}}"#,
                "not negative",
            ),
        ];
        for (text, expected) in cases {
            let (line, message) = error_at(text, DataFormat::Native);
            assert_eq!(line, 1, "{text}");
            assert!(message.contains(expected), "{text}: {message}");
        }
    }

    #[test]
    fn an_empty_file_is_an_error() {
        let (line, message) = error_at("\n  \n", DataFormat::Native);
        assert_eq!(
            (line, message.as_str()),
            (0, "dataset contains no examples")
        );
    }

    #[test]
    fn question_errors_come_from_the_request_validation() {
        let text = r#"{"state": "s", "question": {"type": "score", "instructions": "i", "criteria": ["only"]}, "answer": 0}"#;
        let (_, message) = error_at(text, DataFormat::Native);
        assert_eq!(
            message,
            "questions.question: \"criteria\" must be an array of 2 to 10 levels"
        );
    }

    #[test]
    fn dataset_errors_convert_with_their_path_and_line() {
        let error = parse("{", DataFormat::Native).expect_err("invalid");
        let core: retrograd_core::Error = error.into();
        assert!(
            matches!(core, retrograd_core::Error::Dataset { line: 1, ref path, .. } if path == "train.jsonl")
        );
    }

    #[test]
    fn read_jsonl_reads_a_file() {
        let path =
            std::env::temp_dir().join(format!("retrograd-decision-{}.jsonl", std::process::id()));
        std::fs::write(&path, CHOICE).expect("write");
        let examples = read_jsonl(&path, DataFormat::Native).expect("read");
        std::fs::remove_file(&path).expect("cleanup");
        assert_eq!(examples.len(), 1);
        let missing = read_jsonl(&path, DataFormat::Native);
        assert!(matches!(missing, Err(DecisionError::Io(_))));
    }

    const JULIA: &str = r#"{"state": {"k": "v"}, "question": "pick one", "options": ["red", "green", "blue"], "target": 2}"#;

    #[test]
    fn julia_rows_become_keyed_choices() {
        let examples = parse(JULIA, DataFormat::Julia).expect("valid");
        let example = &examples[0];
        assert_eq!(example.question.kind, QuestionKind::Choice);
        assert_eq!(example.question.instructions, json!("pick one"));
        let keys: Vec<_> = example
            .question
            .options
            .iter()
            .map(|o| o.key.as_str())
            .collect();
        assert_eq!(keys, ["0", "1", "2"]);
        assert_eq!(example.question.options[2].description, json!("blue"));
        assert_eq!(example.target, Target::Hard(2));
    }

    #[test]
    fn julia_noul_is_false_then_true_and_teacher_logits_distill() {
        let text = r#"{"state": "s", "question": "q", "type": "noul", "options": ["no", "yes"], "teacher_logits": [0.0, 2.0]}"#;
        let example = &parse(text, DataFormat::Julia).expect("valid")[0];
        let keys: Vec<_> = example
            .question
            .options
            .iter()
            .map(|o| o.key.as_str())
            .collect();
        assert_eq!(keys, ["false", "true"]);
        let Target::Soft(p) = &example.target else {
            panic!("soft target");
        };
        assert!((p[1] - 1.0 / (1.0 + (-2.0_f64).exp())).abs() < 1e-12);
    }

    #[test]
    fn julia_rows_are_validated_like_the_original() {
        let cases = [
            (
                r#"{"state": 3, "question": "q", "options": ["a", "b"], "target": 0}"#,
                "state must be text/JSON",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a"], "target": 0}"#,
                "options must contain 2-20",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", ""], "target": 0}"#,
                "options must contain 2-20",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b"], "type": "rank", "target": 0}"#,
                "type must be",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b", "c"], "type": "noul", "target": 0}"#,
                "noul options must be ordered",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b"], "target": 2}"#,
                "target must index",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b"], "target": true}"#,
                "target must index",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b"]}"#,
                "training requires target",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b"], "target": 0, "teacher_logits": [1.0]}"#,
                "teacher logits must be finite",
            ),
            (
                r#"{"state": "s", "question": "q", "options": ["a", "b"], "target": 0, "teacher_logits": [1.0, "x"]}"#,
                "teacher logits must be finite",
            ),
        ];
        for (text, expected) in cases {
            let (line, message) = error_at(text, DataFormat::Julia);
            assert_eq!(line, 1);
            assert!(message.contains(expected), "{text}: {message}");
        }
    }

    #[test]
    fn julia_files_do_not_mix_teacher_and_hard_rows() {
        let with = r#"{"state": "s", "question": "q", "options": ["a", "b"], "teacher_logits": [0.0, 1.0]}"#;
        let (line, message) = error_at(&format!("{with}\n{with}\n{JULIA}"), DataFormat::Julia);
        // The third line is 3 options and has no teacher.
        assert_eq!(line, 3);
        assert!(message.contains("mixes rows"));
        assert!(parse(&format!("{with}\n{with}"), DataFormat::Julia).is_ok());
    }
}
