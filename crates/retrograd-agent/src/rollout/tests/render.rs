//! Tool rendering: a native chat template gets the catalog, a blind one gets
//! the prompt-written fallback.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use super::*;
use crate::trajectory::Role;

#[tokio::test]
async fn a_native_template_gets_the_catalog_instead_of_a_hand_written_prompt() {
    let policy = Arc::new(NativeToolPolicy::default());
    let engine = engine_with(policy.clone(), group_limits());
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();

    assert_eq!(
        policy.prompt_renders.load(Ordering::Relaxed),
        0,
        "a template that renders tools must never be rendered without them"
    );
    let renders = policy.native_renders.lock().unwrap();
    let (messages, tools) = renders.first().expect("the opening prompt is rendered");
    assert_eq!(
        tools
            .iter()
            .map(|tool| tool.name.as_str())
            .collect::<Vec<_>>(),
        ["test__echo"],
        "the catalog must reach the template"
    );
    let system = messages
        .iter()
        .find(|message| message.role == Role::System)
        .expect("the scenario has a system turn");
    assert_eq!(
        system.content, "be useful",
        "the tool catalog must not also be pasted into the system turn"
    );

    // The observation is the tool's own output: the call id travels as
    // `tool_call_id` for the template to frame, not inside the text.
    let observation = trajectory
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .expect("the policy called a tool");
    assert_eq!(observation.content, "{\"text\":\"hi\"}");
    assert!(observation.tool_call_id.is_some());
}

#[tokio::test]
async fn a_template_blind_to_tools_still_gets_the_prompt_written_catalog() {
    let engine = engine_with(Arc::new(MultiTurnPolicy), group_limits());
    let trajectory = engine.rollout(&scenario(), 7).await.unwrap();
    let system = trajectory
        .messages
        .iter()
        .find(|message| message.role == Role::System)
        .expect("the scenario has a system turn");
    assert!(
        system.content.contains("Available tools (JSON)"),
        "the fallback is the only path that works for every template: {}",
        system.content
    );
    let observation = trajectory
        .messages
        .iter()
        .find(|message| message.role == Role::Tool)
        .expect("the policy called a tool");
    assert!(
        observation.content.ends_with(": {\"text\":\"hi\"}"),
        "without a tool role the call id has to survive inside the text: {}",
        observation.content
    );
}
