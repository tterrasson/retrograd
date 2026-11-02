//! A standalone `systemone` template for a causal model's letter logits.

use std::fmt::Display;

use crate::DecisionError;

/// The content of the user message. Labels follow the upstream `openjev`
/// order, independently of option keys and of the tokenizer's token IDs.
pub const QUESTION_TEMPLATE: &str = r#"State:
{{ state if state is string else state | tojson }}
Question ({{ type }}):
{{ instructions if instructions is string else instructions | tojson }}
Options:
{% for option in options %}[{{ 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'[loop.index0] }}] {{ option.key }}: {{ option.description if option.description is string else option.description | tojson }}
{% endfor %}Reply with the letter of the best option only."#;

/// Freeze a model's chat wrapper, including its generation prompt and template
/// variables, around the dynamic question. Literal text is emitted as quoted
/// Jinja expressions so template syntax in the wrapper cannot execute.
///
/// Two different sentinels must survive exactly once with the same wrapper:
/// templates that transform content or choose a content-dependent wrapper
/// cannot be safely flattened and are refused.
pub fn standalone<E: Display>(
    render_chat: impl Fn(&str) -> Result<String, E>,
) -> Result<String, DecisionError> {
    const SENTINELS: [&str; 2] = [
        "RETRO_DECISION_CONTENT_93f27c1a\n[option] \"quoted\" <tag> {{literal}} A",
        "RETRO_DECISION_CONTENT_b8e046d2\n[state] \"other\" & value B",
    ];
    let mut wrapper = None;
    for sentinel in SENTINELS {
        let rendered =
            render_chat(sentinel).map_err(|error| DecisionError::Render(error.to_string()))?;
        let mut parts = rendered.split(sentinel);
        let prefix = parts.next().unwrap_or_default();
        let suffix = parts.next().ok_or_else(|| {
            DecisionError::InvalidModel("chat template did not preserve the user content".into())
        })?;
        if parts.next().is_some() {
            return Err(DecisionError::InvalidModel(
                "chat template repeats the user content".into(),
            ));
        }
        let current = (prefix.to_owned(), suffix.to_owned());
        if wrapper
            .as_ref()
            .is_some_and(|previous| previous != &current)
        {
            return Err(DecisionError::InvalidModel(
                "chat template has a content-dependent wrapper".into(),
            ));
        }
        wrapper = Some(current);
    }
    let (prefix, suffix) = wrapper.expect("two sentinels were rendered");
    let literal = |text: &str| -> Result<String, DecisionError> {
        serde_json::to_string(text).map_err(|error| DecisionError::Render(error.to_string()))
    };
    Ok(format!(
        "{{{{ {} }}}}{QUESTION_TEMPLATE}{{{{ {} }}}}",
        literal(&prefix)?,
        literal(&suffix)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapper_is_quoted_and_question_stays_dynamic() {
        let source =
            standalone(|s| Ok::<_, String>(format!("<bos>{{{{literal}}}}{s}<assistant>"))).unwrap();
        assert!(source.starts_with("{{ \"<bos>{{literal}}\" }}State:"));
        assert!(source.contains("loop.index0"));
        assert!(source.ends_with("{{ \"<assistant>\" }}"));
    }

    #[test]
    fn lost_repeated_and_content_dependent_sentinels_are_refused() {
        assert!(standalone(|_| Ok::<_, String>("lost".into())).is_err());
        assert!(standalone(|s| Ok::<_, String>(format!("{s}{s}"))).is_err());
        assert!(standalone(|s| Ok::<_, String>(format!("{}{s}", &s[s.len() - 1..]))).is_err());
    }
}
