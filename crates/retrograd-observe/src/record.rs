//! One line of `observe.jsonl`: the envelope every record shares, and the cut
//! applied to long texts.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::SCHEMA_VERSION;

/// Keys whose string value is free text, cut by `max_text_chars`.
const TEXT_KEYS: [&str; 4] = ["completion", "content", "reward_text", "judge_explanation"];
/// Keys under which every string leaf is cut. Object keys, and therefore the
/// structure of the JSON, are left alone.
const NESTED_TEXT_KEYS: [&str; 2] = ["arguments", "metadata"];

/// Where a record sits: its segment, and its place inside the batch that
/// wrote it. Recovery reads nothing else.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
pub(crate) struct Header {
    pub v: u32,
    pub segment: u32,
    pub batch_id: u64,
    pub batch_index: usize,
    pub batch_len: usize,
}

/// Serializes one record. `body` must be a JSON object.
pub(crate) fn line(
    kind: &str,
    body: Value,
    segment: u32,
    time: &str,
    batch: (u64, usize, usize),
    max_text_chars: usize,
) -> String {
    let mut body = match body {
        Value::Object(map) => map,
        other => {
            let mut map = Map::new();
            map.insert("value".into(), other);
            map
        }
    };
    if max_text_chars > 0 {
        for (key, value) in &mut body {
            cut_value(key, value, max_text_chars, false);
        }
    }
    let (batch_id, batch_index, batch_len) = batch;
    let mut record = Map::new();
    record.insert("v".into(), SCHEMA_VERSION.into());
    record.insert("type".into(), kind.into());
    record.insert("segment".into(), segment.into());
    record.insert("time".into(), time.into());
    record.insert("batch_id".into(), batch_id.into());
    record.insert("batch_index".into(), batch_index.into());
    record.insert("batch_len".into(), batch_len.into());
    record.extend(body);
    Value::Object(record).to_string()
}

fn cut_value(key: &str, value: &mut Value, max: usize, nested: bool) {
    let nested = nested || NESTED_TEXT_KEYS.contains(&key);
    match value {
        Value::String(text) if nested || TEXT_KEYS.contains(&key) => cut(text, max),
        Value::Array(items) => {
            for item in items {
                cut_value(key, item, max, nested);
            }
        }
        Value::Object(map) => {
            for (key, value) in map {
                cut_value(key, value, max, nested);
            }
        }
        _ => {}
    }
}

pub(crate) fn cut(text: &mut String, max: usize) {
    let Some((at, _)) = text.char_indices().nth(max) else {
        return;
    };
    let rest = text[at..].chars().count();
    text.truncate(at);
    text.push_str(&format!("…[+{rest} chars]"));
}

/// Current time, RFC 3339 in UTC with milliseconds.
pub(crate) fn utc_now() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format_utc(now.as_secs(), now.subsec_millis())
}

fn format_utc(seconds: u64, millis: u32) -> String {
    // Below 2^47 for any u64 input, so the day count fits an i64.
    let (year, month, day) = civil_from_days((seconds / 86_400) as i64);
    let rem = seconds % 86_400;
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}Z",
        rem / 3600,
        rem / 60 % 60,
        rem % 60
    )
}

/// Proleptic Gregorian date of a day count since 1970-01-01 (Howard
/// Hinnant, "chrono-Compatible Low-Level Date Algorithms").
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    // In [1, 31] and [1, 12] by construction.
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_are_utc_and_rfc3339() {
        assert_eq!(format_utc(0, 0), "1970-01-01T00:00:00.000Z");
        assert_eq!(format_utc(951_782_400, 5), "2000-02-29T00:00:00.005Z");
        assert_eq!(format_utc(1_789_000_000, 999), "2026-09-10T00:26:40.999Z");
    }

    #[test]
    fn a_cut_counts_characters_and_says_how_many_it_removed() {
        let mut text = "héllo world".to_string();
        cut(&mut text, 5);
        assert_eq!(text, "héllo…[+6 chars]");
        let mut short = "abc".to_string();
        cut(&mut short, 3);
        assert_eq!(short, "abc");
    }

    #[test]
    fn only_texts_are_cut_and_the_structure_survives() {
        let body = serde_json::json!({
            "prompt": "p:123456",
            "completion": "abcdef",
            "messages": [{"role": "assistant", "content": "abcdef",
                          "tool_calls": [{"id": "call-123456", "name": "search_all",
                                          "arguments": {"query": "abcdef", "limit": 3}}]}],
        });
        let line = line("rollout", body, 0, "t", (1, 0, 1), 3);
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["prompt"], "p:123456");
        assert_eq!(value["completion"], "abc…[+3 chars]");
        let message = &value["messages"][0];
        assert_eq!(message["content"], "abc…[+3 chars]");
        assert_eq!(message["tool_calls"][0]["id"], "call-123456");
        assert_eq!(message["tool_calls"][0]["name"], "search_all");
        assert_eq!(
            message["tool_calls"][0]["arguments"]["query"],
            "abc…[+3 chars]"
        );
        assert_eq!(message["tool_calls"][0]["arguments"]["limit"], 3);
        assert_eq!(value["v"], 1);
        assert_eq!(value["type"], "rollout");
    }
}
