//! A flattened decision template agrees with the original chat renderer.
use retrograd_decision::Json;
use retrograd_decision::causal::{QUESTION_TEMPLATE, standalone};
use retrograd_engine::{render_chat_template_source, render_decision_template_source};

#[test]
fn standalone_matches_chat_wrapping_for_each_template_and_thinking_mode() {
    let context: Json = serde_json::from_str(r#"{"state":{"x":"{{ escape }}\n"},"type":"choice","instructions":"pick\n\"one\"","options":[{"key":"z","description":null},{"key":"a","description":{"n":2}}],"images":[],"id":"q"}"#).unwrap();
    let question = render_decision_template_source(QUESTION_TEMPLATE, &context).unwrap();
    assert!(question.contains("[A] z: null"));
    assert!(question.contains("[B] a:"));
    for name in ["qwen3.5-0.8b", "qwen3-0.6b", "gemma-3-270m", "lfm2.5-230m"] {
        let source = std::fs::read_to_string(format!(
            "{}/tests/fixtures/chat-templates/{name}.jinja",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap();
        for thinking in [false, true] {
            let variables = serde_json::json!({"enable_thinking":thinking}).to_string();
            let render = |content: &str| {
                render_chat_template_source(
                    &source,
                    &serde_json::json!([{"role":"user","content":content}]).to_string(),
                    None,
                    Some(&variables),
                    true,
                )
            };
            let standalone = standalone(render).unwrap();
            let actual = render_decision_template_source(&standalone, &context).unwrap();
            assert_eq!(
                actual,
                render(&question).unwrap(),
                "{name}, thinking={thinking}"
            );
        }
    }
}
