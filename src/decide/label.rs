//! Teacher distributions in the native decision JSONL format.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::PathBuf;
use std::time::Duration;

use retrograd::decision::Json;
use retrograd::decision::answer::Answer;
use retrograd::decision::dataset::{self, DataFormat};
use retrograd::decision::readout::{DecisionType, Head, Readout};
use retrograd::decision::request::{Question, QuestionKind, SystemOneRequest};
use retrograd::{Device, Engine, EngineConfig, Error, Result};
use serde_json::{Value, json};

use crate::args::Args;

pub(crate) const FLAGS: &[&str] = &[
    "--teacher",
    "--teacher-model",
    "--api-key-env",
    "--data",
    "--out",
    "--device",
    "--adapter",
    "--template-variables",
];

#[derive(Debug)]
struct LabelArgs {
    teacher: String,
    teacher_model: Option<String>,
    api_key_env: Option<String>,
    data: PathBuf,
    out: PathBuf,
    device: Device,
    adapter: Option<PathBuf>,
    variables: Value,
}

fn parse(args: &[String]) -> Result<LabelArgs> {
    let mut teacher = None;
    let mut teacher_model = None;
    let mut api_key_env = None;
    let mut data = None;
    let mut out = None;
    let mut device = Device::Auto;
    let mut adapter = None;
    let mut variables = json!({"enable_thinking": false});
    let mut seen = [false; 8];
    let mut args = Args::new("decide label", args);
    while let Some(flag) = args.next_arg() {
        let index = FLAGS
            .iter()
            .position(|known| *known == flag)
            .ok_or_else(|| args.unknown(flag))?;
        args.once(&mut seen[index], flag)?;
        match flag {
            "--teacher" => teacher = Some(args.value(flag)?.to_owned()),
            "--teacher-model" => teacher_model = Some(args.value(flag)?.to_owned()),
            "--api-key-env" => api_key_env = Some(args.value(flag)?.to_owned()),
            "--data" => data = Some(PathBuf::from(args.value(flag)?)),
            "--out" => out = Some(PathBuf::from(args.value(flag)?)),
            "--device" => device = args.value(flag)?.parse()?,
            "--adapter" => adapter = Some(PathBuf::from(args.value(flag)?)),
            "--template-variables" => {
                variables = serde_json::from_str(args.value(flag)?).map_err(|error| {
                    Error::invalid(format!("invalid --template-variables JSON: {error}"))
                })?;
                if !variables.is_object() {
                    return Err(Error::invalid("--template-variables must be a JSON object"));
                }
            }
            _ => unreachable!("flag is in FLAGS"),
        }
    }
    let teacher = teacher.ok_or_else(|| Error::invalid("--teacher is required"))?;
    let remote = teacher.starts_with("http://") || teacher.starts_with("https://");
    if remote && (teacher_model.is_none() || adapter.is_some()) {
        return Err(Error::invalid(
            "an HTTP teacher needs --teacher-model and does not accept --adapter",
        ));
    }
    if !remote && (teacher_model.is_some() || api_key_env.is_some()) {
        return Err(Error::invalid(
            "--teacher-model and --api-key-env require an HTTP teacher",
        ));
    }
    Ok(LabelArgs {
        teacher,
        teacher_model,
        api_key_env,
        data: data.ok_or_else(|| Error::invalid("--data is required"))?,
        out: out.ok_or_else(|| Error::invalid("--out is required"))?,
        device,
        adapter,
        variables,
    })
}

/// One input line is either a service request or a single native question.
/// Existing labels are ignored; output is always freshly validated native data.
fn requests(text: &str, readout: &Readout, path: &str) -> Result<Vec<SystemOneRequest>> {
    let mut requests = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parse = || -> Result<SystemOneRequest> {
            let mut body: Json =
                serde_json::from_str(line).map_err(|error| Error::invalid(error.to_string()))?;
            if let Some(question) = body.get("question").cloned() {
                if body.get("questions").is_some() {
                    return Err(Error::invalid("question and questions are exclusive"));
                }
                let mut questions = Json::Object(Default::default());
                questions.set("question", question);
                body.set("questions", questions);
            }
            Ok(SystemOneRequest::parse(&body, readout)?)
        };
        requests.push(parse().map_err(|error| Error::dataset(path, index + 1, error.to_string()))?);
    }
    if requests.is_empty() {
        return Err(Error::dataset(path, 0, "dataset contains no questions"));
    }
    Ok(requests)
}

fn question_json(question: &Question) -> Json {
    let mut criteria = match question.kind {
        QuestionKind::Score => Json::Array(Vec::new()),
        _ => Json::Object(Default::default()),
    };
    for option in &question.options {
        if let Json::Array(items) = &mut criteria {
            items.push(option.description.clone());
        } else {
            criteria.set(&option.key, option.description.clone());
        }
    }
    let mut out = Json::Object(Default::default());
    out.set("type", question.kind.name().into());
    out.set("instructions", question.instructions.clone());
    out.set("criteria", criteria);
    out
}

fn record(state: &Json, id: &str, question: &Question, probabilities: Value) -> Result<String> {
    let mut question = question_json(question);
    question.set("probabilities", Json::from(probabilities));
    let mut questions = Json::Object(Default::default());
    questions.set(id, question);
    let mut row = Json::Object(Default::default());
    row.set("state", state.clone());
    row.set("questions", questions);
    let text = serde_json::to_string(&row).map_err(|error| Error::runtime(error.to_string()))?;
    // Exact key coverage, finite nonnegative values and normalized mass are
    // checked by the same parser that will consume the student's dataset.
    dataset::parse_jsonl(&text, "teacher output", DataFormat::Native)?;
    Ok(text)
}

fn probabilities(answer: Answer) -> Value {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => {
            json!(probabilities)
        }
        Answer::Noul { noul } => json!({"false": 1.0 - noul, "true": noul}),
    }
}

pub(super) fn label(args: &[String]) -> Result<()> {
    let args = parse(args)?;
    if args.out.exists() {
        return Err(Error::invalid(format!(
            "{} already exists",
            args.out.display()
        )));
    }
    let text = std::fs::read_to_string(&args.data)?;
    let mut output = Vec::new();
    if let Some(model) = &args.teacher_model {
        let readout = Readout::new(
            DecisionType::named("openjev").expect("openjev is a decision type"),
            Head::LabelLogits {
                labels: (0..52).collect(),
                label_texts: Vec::new(),
            },
        );
        let requests = requests(&text, &readout, &args.data.display().to_string())?;
        let key = args
            .api_key_env
            .as_ref()
            .map(|name| {
                std::env::var(name).map_err(|_| {
                    Error::invalid(format!("API key environment variable {name} is unset"))
                })
            })
            .transpose()?
            .unwrap_or_default();
        let client = retrograd_llm_client::OpenAiClient::new(
            &args.teacher,
            key,
            Duration::from_secs(120),
            1024 * 1024,
            retrograd_llm_client::Purpose::Judge,
        )?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        for request in requests {
            if !request.images.is_empty() {
                return Err(Error::invalid("decision labelling supports text inputs"));
            }
            for (id, question) in &request.questions {
                let payload = json!({
                    "model": model, "temperature": 0, "max_tokens": 4096,
                    "messages": [
                        {"role": "system", "content": "Evaluate the typed decision question. Return only a JSON object with a probabilities object containing exactly every option key and a nonnegative probability, summing to one. For score, keys are zero-based level indices; for noul, keys are false and true. Do not use Markdown."},
                        {"role": "user", "content": json!({"state": request.state, "question": question_json(question)}).to_string()}
                    ]
                });
                let reply = runtime.block_on(client.chat_completion(&payload))?;
                let parsed: Value = serde_json::from_str(&reply.content).map_err(|error| {
                    Error::invalid(format!(
                        "teacher returned invalid probability JSON: {error}"
                    ))
                })?;
                let probabilities = parsed.get("probabilities").cloned().ok_or_else(|| {
                    Error::invalid("teacher response has no probabilities object")
                })?;
                output.push(record(&request.state, id, question, probabilities)?);
            }
        }
    } else {
        let mut engine = Engine::new(
            &args.teacher,
            &EngineConfig {
                device: args.device,
                ..EngineConfig::default()
            },
        )?;
        engine.set_chat_template_variables(&args.variables.to_string())?;
        let mut info = if engine.decision_metadata()?.decision_type.is_none() {
            engine.configure_causal_decision()?.0
        } else {
            engine.decision_info()?
        };
        if let Some(adapter) = &args.adapter {
            engine.load_lora(adapter)?;
            info = info.with_adapter_temperatures(
                &retrograd::trainer::adapter_decision_metadata(adapter)?,
            )?;
        }
        for request in requests(&text, &info.readout, &args.data.display().to_string())? {
            let decision = engine.decide(&info, &request)?;
            let mut answers: BTreeMap<_, _> = decision.answers.into_iter().collect();
            for (id, question) in &request.questions {
                let answer = answers
                    .remove(id)
                    .ok_or_else(|| Error::runtime("teacher omitted an answer"))?;
                output.push(record(&request.state, id, question, probabilities(answer))?);
            }
        }
    }
    // No output is created until every teacher answer has passed validation.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.out)?;
    for line in &output {
        writeln!(file, "{line}")?;
    }
    file.sync_all()?;
    eprintln!(
        "{} labelled questions written to {}",
        output.len(),
        args.out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn http_teacher_validates_targets_before_creating_output() {
        use std::io::{BufRead, Read};
        use std::net::TcpListener;
        let server = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = server.local_addr().unwrap();
        let worker = std::thread::spawn(move || {
            for content in [
                r#"{"probabilities":{"z":0.8,"a":0.2}}"#,
                r#"{"probabilities":{"z":1.0}}"#,
            ] {
                let (mut stream, _) = server.accept().unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .unwrap();
                let mut reader = std::io::BufReader::new(&stream);
                let mut request_line = String::new();
                reader.read_line(&mut request_line).unwrap();
                assert!(request_line.starts_with("POST /v1/chat/completions "));
                let mut length = 0;
                loop {
                    let mut line = String::new();
                    reader.read_line(&mut line).unwrap();
                    if line == "\r\n" {
                        break;
                    }
                    if let Some(value) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        length = value.trim().parse::<usize>().unwrap();
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let body: Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(body["model"], "judge");
                let body =
                    json!({"choices":[{"message":{"content":content},"finish_reason":"stop"}]})
                        .to_string();
                write!(stream, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).unwrap();
            }
        });
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("questions.jsonl");
        std::fs::write(&input, r#"{"state":"s","question":{"type":"choice","instructions":"pick","criteria":{"z":null,"a":null}}}"#).unwrap();
        let args = |out: &str| {
            vec![
                "--teacher".into(),
                format!("http://{address}/v1"),
                "--teacher-model".into(),
                "judge".into(),
                "--data".into(),
                input.display().to_string(),
                "--out".into(),
                dir.path().join(out).display().to_string(),
            ]
        };
        label(&args("good.jsonl")).unwrap();
        let rows = dataset::read_jsonl(&dir.path().join("good.jsonl"), DataFormat::Native).unwrap();
        assert_eq!(rows[0].target.distribution(2), [0.8, 0.2]);
        assert!(label(&args("bad.jsonl")).is_err());
        assert!(!dir.path().join("bad.jsonl").exists());
        worker.join().unwrap();
    }

    #[test]
    fn teacher_targets_preserve_order_and_require_every_key() {
        let readout = Readout::new(
            DecisionType::named("openjev").expect("openjev is a decision type"),
            Head::LabelLogits {
                labels: (0..52).collect(),
                label_texts: Vec::new(),
            },
        );
        let request = requests(r#"{"state":{"x":1},"question":{"type":"choice","instructions":"pick","criteria":{"z":null,"a":"two"}}}"#, &readout, "in").unwrap().remove(0);
        let q = &request.questions["question"];
        let line = record(&request.state, "question", q, json!({"z":0.7,"a":0.3})).unwrap();
        let data = dataset::parse_jsonl(&line, "out", DataFormat::Native).unwrap();
        assert_eq!(data[0].question.options[0].key, "z");
        assert_eq!(data[0].target.distribution(2), [0.7, 0.3]);
        assert!(record(&request.state, "question", q, json!({"z":1.0})).is_err());
        assert!(record(&request.state, "question", q, json!({"z":0.8,"a":0.8})).is_err());
    }
}
