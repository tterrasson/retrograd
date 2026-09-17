//! Request extractors that fail the way the rest of the API fails.
//!
//! `axum::Json`'s own rejection answers with a plain-text body, so a malformed
//! request would be the one error in the API that is not a problem document,
//! and it is the error a client hits most often while integrating.
//!
//! Every JSON body in the API is deserialized through [`from_json_slice`] or
//! [`from_json_value`], so a client always gets a `pointer`, a `code`, and - for
//! the single most common typo, an unknown field - a `hint` naming the field it
//! probably meant.

use axum::extract::FromRequest;
use http::header;
use retrograd_core::escape_json_pointer;
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::error::{ApiError, ErrorCode, ProblemKind};
use crate::problem_from_rejection;

/// Drop-in replacement for `axum::Json` on the request side.
pub struct Json<T>(pub T);

impl<T, S> FromRequest<S> for Json<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = ApiError;

    async fn from_request(
        request: axum::extract::Request,
        state: &S,
    ) -> Result<Self, Self::Rejection> {
        // The content type is checked here rather than left to `axum::Json`,
        // because reading the body as `Bytes` is what lets the deserialization
        // failure carry a pointer and a code. Dropping the check with it would
        // mean a client that mislabels its body is told a field is missing
        // instead of being told the label is wrong.
        if !is_json(request.headers()) {
            return Err(ApiError::new(
                ProblemKind::UnsupportedMediaType,
                "this route reads a JSON body",
            )
            .with_field("", ErrorCode::InvalidValue, "expected a JSON body")
            .with_hint("send `content-type: application/json`"));
        }
        let bytes = axum::body::Bytes::from_request(request, state)
            .await
            .map_err(|rejection| {
                problem_from_rejection(rejection.status(), rejection.body_text())
            })?;
        from_json_slice(&bytes).map(Self)
    }
}

/// `application/json`, and the `+json` structured suffix RFC 6839 defines.
/// Parameters (`; charset=utf-8`) are ignored, as everywhere else.
fn is_json(headers: &header::HeaderMap) -> bool {
    let Some(value) = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let essence = value
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase();
    essence == "application/json" || essence.ends_with("+json")
}

/// Deserializes a JSON body, turning a failure into a problem document with a
/// JSON Pointer, a closed `code`, and - for an unknown field - a spelling
/// suggestion.
pub fn from_json_slice<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, ApiError> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    serde_path_to_error::deserialize(&mut deserializer).map_err(to_api_error)
}

/// The same, starting from a tree already parsed rather than raw bytes - for
/// the routes that inspect the body (the contract guard, the TOML-vs-JSON form
/// detection) before typing it.
pub fn from_json_value<T: DeserializeOwned>(value: Value) -> Result<T, ApiError> {
    serde_path_to_error::deserialize(value).map_err(to_api_error)
}

fn to_api_error(error: serde_path_to_error::Error<serde_json::Error>) -> ApiError {
    let pointer = dotted_to_pointer(&error.path().to_string());
    let message = error.into_inner().to_string();
    match parse_unknown_field(&message) {
        Some((field, known)) => {
            let mut api = ApiError::invalid(format!("invalid request: {message}")).with_field(
                pointer,
                ErrorCode::UnknownField,
                format!("unknown field '{field}'"),
            );
            api = api.with_hint(match suggest(&field, &known) {
                Some(suggestion) => format!(
                    "did you mean '{suggestion}'? known fields: {}",
                    known.join(", ")
                ),
                None => format!("known fields: {}", known.join(", ")),
            });
            api
        }
        None => match parse_missing_field(&message) {
            // serde reports the *containing* struct as the path, so the field
            // has to be appended by hand - without it the pointer would be the
            // empty root on a top-level miss, which names nothing.
            Some(field) => ApiError::invalid(format!("invalid request: {message}")).with_field(
                format!("{pointer}/{}", escape_json_pointer(&field)),
                ErrorCode::MissingField,
                format!("missing field '{field}'"),
            ),
            None => ApiError::invalid(format!("invalid request: {message}")).with_field(
                pointer,
                ErrorCode::InvalidValue,
                message,
            ),
        },
    }
}

/// `serde_path_to_error`'s own rendering: dot-separated field names, `[n]` for
/// a sequence index, and `.` alone for the root. Converted to the RFC 6901
/// pointer the rest of the API already uses.
fn dotted_to_pointer(path: &str) -> String {
    if path == "." {
        return String::new();
    }
    let normalized = path.replace('[', ".").replace(']', "");
    let mut pointer = String::new();
    for segment in normalized.split('.') {
        if segment.is_empty() {
            continue;
        }
        pointer.push('/');
        pointer.push_str(&escape_json_pointer(segment));
    }
    pointer
}

/// Pulls the offending field and the known ones out of serde's own message,
/// `unknown field \`profil\`, expected one of \`objective\`, \`model\`, …` or,
/// with a single expected field, `unknown field \`profil\`, expected \`path\``.
///
/// Parsed from the message rather than sourced from a field list kept beside
/// the type: serde already knows the schema and renders it every time
/// `deny_unknown_fields` fires, so a second, hand-maintained list would only be
/// a second place for the two to disagree.
fn parse_unknown_field(message: &str) -> Option<(String, Vec<String>)> {
    if !message.starts_with("unknown field") {
        return None;
    }
    let mut quoted = message.split('`').skip(1).step_by(2);
    let field = quoted.next()?.to_string();
    let known: Vec<String> = quoted.map(str::to_string).collect();
    Some((field, known))
}

/// The field named by serde's `missing field \`model\`` - the other half of the
/// `deny_unknown_fields` pair, and the one a client hits first while
/// integrating. Same reasoning as [`parse_unknown_field`]: serde renders it, so
/// nothing here duplicates the schema.
fn parse_missing_field(message: &str) -> Option<String> {
    if !message.starts_with("missing field") {
        return None;
    }
    message.split('`').nth(1).map(str::to_string)
}

/// The closest known field within edit distance 2, or none. A suggestion
/// beyond that distance is as likely to mislead as to help
/// (`xyzzy` must suggest nothing).
fn suggest(field: &str, known: &[String]) -> Option<String> {
    known
        .iter()
        .map(|candidate| (candidate, levenshtein(field, candidate)))
        .filter(|(_, distance)| *distance <= 2)
        .min_by_key(|(_, distance)| *distance)
        .map(|(candidate, _)| candidate.clone())
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut row: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut previous_diagonal = row[0];
        row[0] = i;
        for j in 1..=b.len() {
            let previous_above = row[j];
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            row[j] = (row[j] + 1)
                .min(row[j - 1] + 1)
                .min(previous_diagonal + cost);
            previous_diagonal = previous_above;
        }
    }
    row[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    // The fields are the schema this test exercises: `Deserialize` reads them,
    // no line here does. Deleting them would delete the test's subject.
    #[derive(Deserialize, Debug)]
    #[serde(deny_unknown_fields)]
    #[expect(dead_code)]
    struct Recipe {
        objective: Option<String>,
        model: Option<String>,
    }

    #[test]
    fn an_unknown_field_suggests_the_closest_known_one() {
        let error = from_json_slice::<Recipe>(br#"{"objectif":"x"}"#).unwrap_err();
        assert_eq!(error.errors[0].pointer, "/objectif");
        assert_eq!(error.errors[0].code, ErrorCode::UnknownField);
        let hint = error.errors[0].hint.as_deref().unwrap_or_default();
        assert!(hint.contains("did you mean 'objective'"), "{hint}");
    }

    #[test]
    fn a_field_with_no_close_match_gets_only_the_known_list() {
        let error = from_json_slice::<Recipe>(br#"{"xyzzy":"x"}"#).unwrap_err();
        let hint = error.errors[0].hint.as_deref().unwrap_or_default();
        assert!(!hint.contains("did you mean"), "{hint}");
        assert!(hint.contains("known fields"), "{hint}");
    }

    /// The other half of `deny_unknown_fields`, and the code a client's `match`
    /// most wants to tell apart from a value it got wrong.
    #[test]
    fn a_missing_field_is_named_by_code_and_by_pointer() {
        // Same as `Recipe` above: the field is the schema, not a value read.
        #[derive(Deserialize, Debug)]
        #[expect(dead_code)]
        struct Required {
            model: String,
        }
        let error = from_json_slice::<Required>(br#"{}"#).unwrap_err();
        assert_eq!(error.errors[0].pointer, "/model");
        assert_eq!(error.errors[0].code, ErrorCode::MissingField);
    }

    #[test]
    fn a_body_that_is_not_labelled_json_is_a_415_and_not_a_field_error() {
        let headers = header::HeaderMap::new();
        assert!(!is_json(&headers), "an unlabelled body is not JSON");
        for (value, expected) in [
            ("application/json", true),
            ("application/json; charset=utf-8", true),
            ("application/merge-patch+json", true),
            ("APPLICATION/JSON", true),
            ("text/plain", false),
            ("application/toml", false),
        ] {
            let mut headers = header::HeaderMap::new();
            headers.insert(header::CONTENT_TYPE, value.parse().expect("a header"));
            assert_eq!(is_json(&headers), expected, "{value}");
        }
    }

    #[test]
    fn a_nested_unknown_field_gets_a_pointer_into_the_object() {
        #[derive(Deserialize, Debug)]
        #[serde(deny_unknown_fields)]
        struct Outer {
            // Nesting is the point of the test; the value is never read.
            #[expect(dead_code)]
            recipe: Recipe,
        }
        let error = from_json_slice::<Outer>(br#"{"recipe":{"objectif":"x"}}"#).unwrap_err();
        assert_eq!(error.errors[0].pointer, "/recipe/objectif");
    }
}
