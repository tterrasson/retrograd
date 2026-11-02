//! Texts written the way Python writes them: `json.dumps`, `repr` of a float,
//! `str.lstrip` and `str.rstrip`. The `h2o-lightning` reference renders its
//! inputs with Python, and its model was trained on those bytes.

use crate::DecisionError;
use crate::json::Json;
use crate::template::{RECURSION_LIMIT, too_deep};

/// The options of a `json.dumps` call.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Dumps {
    /// `None` writes everything on one line.
    pub indent: Option<usize>,
    pub item_separator: &'static str,
    pub key_separator: &'static str,
    pub ensure_ascii: bool,
}

/// `json.dumps(value, ...)`, keys in their order.
pub(crate) fn dumps(value: &Json, options: Dumps) -> Result<String, DecisionError> {
    let mut out = String::new();
    dump(&mut out, value, options, 0)?;
    Ok(out)
}

fn newline(out: &mut String, options: Dumps, level: usize) {
    if let Some(indent) = options.indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', indent * level));
    }
}

fn dump(out: &mut String, value: &Json, options: Dumps, level: usize) -> Result<(), DecisionError> {
    if level >= RECURSION_LIMIT {
        return Err(too_deep());
    }
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(true) => out.push_str("true"),
        Json::Bool(false) => out.push_str("false"),
        Json::Number(number) => match number.as_f64() {
            Some(float) if number.is_f64() => out.push_str(&float_repr(float)),
            _ => out.push_str(&number.to_string()),
        },
        Json::String(text) => string(out, text, options.ensure_ascii),
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(options.item_separator);
                }
                newline(out, options, level + 1);
                dump(out, item, options, level + 1)?;
            }
            newline(out, options, level);
            out.push(']');
        }
        Json::Object(members) if members.is_empty() => out.push_str("{}"),
        Json::Object(members) => {
            out.push('{');
            for (i, (key, item)) in members.iter().enumerate() {
                if i > 0 {
                    out.push_str(options.item_separator);
                }
                newline(out, options, level + 1);
                string(out, key, options.ensure_ascii);
                out.push_str(options.key_separator);
                dump(out, item, options, level + 1)?;
            }
            newline(out, options, level);
            out.push('}');
        }
    }
    Ok(())
}

/// A JSON string as `json.dumps` writes it: the short escapes, `\uXXXX` for
/// the other control characters and, with `ensure_ascii`, for DEL and every
/// non-ASCII character (a surrogate pair above the BMP).
fn string(out: &mut String, text: &str, ensure_ascii: bool) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 || (ensure_ascii && u32::from(c) == 0x7f) => {
                out.push_str(&format!("\\u{:04x}", u32::from(c)));
            }
            c if c.is_ascii() || !ensure_ascii => out.push(c),
            c => {
                let mut units = [0_u16; 2];
                for unit in c.encode_utf16(&mut units) {
                    out.push_str(&format!("\\u{unit:04x}"));
                }
            }
        }
    }
    out.push('"');
}

/// `repr()` of a Python float: the shortest digits that read back the same,
/// in exponent form below 1e-4 and from 1e16.
pub(crate) fn float_repr(value: f64) -> String {
    // The shortest round-trip digits, in exponent form: `-1.25e3`.
    let scientific = format!("{value:e}");
    let (mantissa, exponent) = scientific
        .split_once('e')
        .unwrap_or((scientific.as_str(), "0"));
    let exponent: i64 = exponent.parse().unwrap_or(0);
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    // Widening: a float has at most 17 significant digits.
    let n = digits.len() as i64;
    if !(-4..16).contains(&exponent) {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        let exponent_sign = if exponent < 0 { '-' } else { '+' };
        return format!(
            "{sign}{}{fraction}e{exponent_sign}{:02}",
            &digits[..1],
            exponent.abs()
        );
    }
    if exponent >= n - 1 {
        // Narrowing: `exponent - (n - 1)` is in 0..16 here.
        let zeros = "0".repeat((exponent - (n - 1)) as usize);
        return format!("{sign}{digits}{zeros}.0");
    }
    if exponent >= 0 {
        // Narrowing: `exponent` is in 0..16 here.
        let point = exponent as usize + 1;
        return format!("{sign}{}.{}", &digits[..point], &digits[point..]);
    }
    // Narrowing: `-exponent - 1` is in 0..4 here.
    let zeros = "0".repeat((-exponent - 1) as usize);
    format!("{sign}0.{zeros}{digits}")
}

/// `str.isspace()` of Python, for one character.
fn is_space(c: char) -> bool {
    matches!(u32::from(c),
        0x09..=0x0d | 0x1c..=0x20 | 0x85 | 0xa0 | 0x1680 | 0x2000..=0x200a
        | 0x2028 | 0x2029 | 0x202f | 0x205f | 0x3000)
}

/// `str.lstrip()`.
pub(crate) fn lstrip(text: &str) -> &str {
    text.trim_start_matches(is_space)
}

/// `str.rstrip()`.
pub(crate) fn rstrip(text: &str) -> &str {
    text.trim_end_matches(is_space)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(text: &str) -> Json {
        crate::ordered(text)
    }

    #[test]
    fn floats_are_written_as_python_repr_writes_them() {
        for (value, expected) in [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (1e16, "1e+16"),
            (1.2345e16, "1.2345e+16"),
            (123456789012345.6, "123456789012345.6"),
            (-2.5e-7, "-2.5e-07"),
            (1e100, "1e+100"),
        ] {
            assert_eq!(float_repr(value), expected, "{value}");
        }
    }

    #[test]
    fn dumps_follows_the_separators_the_indent_and_the_ascii_option() {
        let value =
            parse(r#"{"b": [1, 2.0, {"x": null}], "a": "é\u0001\u007f😀", "e": {}, "f": []}"#);
        let compact = Dumps {
            indent: None,
            item_separator: ",",
            key_separator: ":",
            ensure_ascii: false,
        };
        assert_eq!(
            dumps(&value, compact).expect("dumps"),
            "{\"b\":[1,2.0,{\"x\":null}],\"a\":\"é\\u0001\u{7f}😀\",\"e\":{},\"f\":[]}"
        );
        let indented = Dumps {
            indent: Some(1),
            item_separator: ",",
            key_separator: ": ",
            ensure_ascii: true,
        };
        assert_eq!(
            dumps(&parse(r#"{"a": ["é😀\u007f"], "b": 1}"#), indented).expect("dumps"),
            "{\n \"a\": [\n  \"\\u00e9\\ud83d\\ude00\\u007f\"\n ],\n \"b\": 1\n}"
        );
    }

    #[test]
    fn strips_python_whitespace() {
        assert_eq!(lstrip("\u{3000} \tx y \u{85}"), "x y \u{85}");
        assert_eq!(rstrip("\u{3000} x y \u{2029}\u{1c}"), "\u{3000} x y");
        assert_eq!(rstrip("\u{200b}"), "\u{200b}");
    }
}
