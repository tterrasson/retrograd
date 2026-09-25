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

/// A scenario may carry its own toolset, so the rendering is decided per tool
/// list: each scenario's template call gets exactly its own tools.
#[tokio::test]
async fn each_scenario_is_rendered_with_its_own_tool_list() {
    let policy = Arc::new(NativeToolPolicy::default());
    let engine = RolloutEngine::with_environments(
        policy.clone(),
        Arc::new(CountingFactory::default()),
        Arc::new(crate::tools::HermesToolCallParser),
        group_limits(),
    )
    .unwrap();
    let mut extended = scenario();
    extended
        .metadata
        .insert("extra_tool".into(), serde_json::json!("lookup"));
    engine.rollout(&scenario(), 1).await.unwrap();
    engine.rollout(&extended, 2).await.unwrap();
    engine.rollout(&scenario(), 3).await.unwrap();

    let renders = policy.native_renders.lock().unwrap();
    let opening_tools = renders
        .iter()
        .filter(|(messages, _)| messages.len() == 2)
        .map(|(_, tools)| tools.len())
        .collect::<Vec<_>>();
    assert_eq!(opening_tools, [1, 2, 1]);
    assert_eq!(
        engine.declared_tools(),
        1,
        "the first rendering is the one reported"
    );
}
