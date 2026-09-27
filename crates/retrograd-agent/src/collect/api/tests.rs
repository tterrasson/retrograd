//! The API loop against a local endpoint that answers like a chat-completions
//! server: one call, then an answer once the call's result is in the
//! conversation.

use std::sync::atomic::Ordering;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use super::*;
use crate::collect::tests::{ScriptedFactory, config, plan, scenario};

type Requests = Arc<std::sync::Mutex<Vec<Value>>>;

/// Serves `/chat/completions` until the test ends, recording every request
/// body it was sent.
async fn endpoint() -> (String, Requests) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}/v1", listener.local_addr().unwrap());
    let requests = Requests::default();
    let seen = requests.clone();
    tokio::spawn(async move {
        loop {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let seen = seen.clone();
            tokio::spawn(async move {
                let mut buffer = Vec::new();
                let body = loop {
                    let mut chunk = [0_u8; 4096];
                    let read = socket.read(&mut chunk).await.unwrap();
                    buffer.extend_from_slice(&chunk[..read]);
                    let text = String::from_utf8_lossy(&buffer).into_owned();
                    if let Some((head, body)) = text.split_once("\r\n\r\n") {
                        let length = head
                            .lines()
                            .find_map(|line| {
                                line.to_ascii_lowercase()
                                    .strip_prefix("content-length:")
                                    .map(|value| value.trim().parse::<usize>().unwrap())
                            })
                            .unwrap_or(0);
                        if body.len() >= length {
                            break body[..length].to_owned();
                        }
                    }
                };
                let request: Value = serde_json::from_str(&body).unwrap();
                let answered = request["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|message| message["role"] == "tool");
                seen.lock().unwrap().push(request);
                let message = match answered {
                    true => json!({"role": "assistant", "content": "It said hi."}),
                    false => json!({"role": "assistant", "content": "Calling.", "tool_calls": [{
                        "id": "call_x", "type": "function",
                        "function": {"name": "echo", "arguments": "{\"text\":\"hi\"}"},
                    }]}),
                };
                let reply =
                    json!({"choices": [{"message": message, "finish_reason": "stop"}]}).to_string();
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\n\
                     connection: close\r\n\r\n{reply}",
                    reply.len()
                );
                socket.write_all(response.as_bytes()).await.unwrap();
            });
        }
    });
    (url, requests)
}

fn generator(url: &str) -> ApiGenerator {
    ApiGenerator::connect(
        url,
        "test-key".into(),
        Duration::from_secs(10),
        "teacher",
        Some(0.7),
    )
    .unwrap()
}

fn limits() -> RolloutLimits {
    RolloutLimits {
        max_turns: 4,
        ..Default::default()
    }
}

fn factory() -> Arc<ScriptedFactory> {
    Arc::new(ScriptedFactory {
        plans: vec![plan("call", Some(1.0), "hi")],
        ..Default::default()
    })
}

#[tokio::test]
async fn a_remote_model_is_collected_into_structured_records() {
    let (url, requests) = endpoint().await;
    let factory = factory();
    let mut records = Vec::new();
    let stats = collect_from_api(
        &generator(&url),
        factory.clone(),
        &[scenario()],
        limits(),
        &config(2, 2),
        &mut records,
    )
    .await
    .unwrap();

    assert_eq!(stats.attempted, 2);
    assert_eq!(factory.created.load(Ordering::Relaxed), 2);
    // Two attempts, one trace: the second is a duplicate of the first.
    assert_eq!(stats.kept, 1);
    assert_eq!(stats.rejected.duplicate, 1);
    let record = &records[0];
    record.validate().unwrap();
    let roles = record
        .messages
        .iter()
        .map(|message| message.role.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        roles,
        ["system", "user", "user", "assistant", "tool", "assistant"]
    );
    assert_eq!(
        record.messages[2].content, "mode:call",
        "the opening is a user turn"
    );
    let calling = &record.messages[3];
    assert_eq!(calling.content, "Calling.");
    assert_eq!(calling.tool_calls[0].id.as_deref(), Some("call_x"));
    assert_eq!(calling.tool_calls[0].arguments, json!({"text": "hi"}));
    assert_eq!(record.messages[4].content, "hi");
    assert_eq!(record.metadata["reward"], 1.0);
    assert_eq!(record.metadata["verification"], "passed");
    assert_eq!(record.tools[0].name, "echo");

    // The request is the OpenAI shape: the catalog as functions, arguments as
    // a string, the observation under its call id.
    let requests = requests.lock().unwrap();
    let second = requests
        .iter()
        .find(|request| request["messages"].as_array().unwrap().len() == 5)
        .expect("a request after the call");
    assert_eq!(second["model"], "teacher");
    assert_eq!(second["temperature"], json!(0.7_f32));
    assert_eq!(second["tools"][0]["function"]["name"], "echo");
    assert_eq!(
        second["messages"][3]["tool_calls"][0]["function"]["arguments"],
        "{\"text\":\"hi\"}"
    );
    assert_eq!(second["messages"][4]["tool_call_id"], "call_x");
}

#[tokio::test]
async fn an_api_collection_writes_no_raw_turns_and_needs_a_grading_environment() {
    let (url, _) = endpoint().await;
    let factory = factory();
    let mut raw = config(2, 1);
    raw.form = AssistantForm::Raw;
    let error = collect_from_api(
        &generator(&url),
        factory.clone(),
        &[scenario()],
        limits(),
        &raw,
        &mut Vec::new(),
    )
    .await
    .unwrap_err();
    assert!(
        error.to_string().contains("structured turns only"),
        "{error}"
    );

    let mut ungraded = config(2, 1);
    ungraded.environment_grades = false;
    let error = collect_from_api(
        &generator(&url),
        factory.clone(),
        &[scenario()],
        limits(),
        &ungraded,
        &mut Vec::new(),
    )
    .await
    .unwrap_err();
    assert!(error.to_string().contains("nothing grades"), "{error}");
    assert_eq!(factory.created.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn an_episode_that_runs_out_of_turns_is_truncated() {
    let (url, _) = endpoint().await;
    let mut records = Vec::new();
    let stats = collect_from_api(
        &generator(&url),
        factory(),
        &[scenario()],
        RolloutLimits {
            max_turns: 1,
            ..Default::default()
        },
        &config(2, 2),
        &mut records,
    )
    .await
    .unwrap();
    assert!(records.is_empty());
    assert_eq!(stats.rejected.truncated, 2);
}
