//! `retrograd decide`: one `/v1/systemone` request against a decision model,
//! without a server. The body is read from a file (or `-` for stdin), the
//! answer printed as llama-server writes it: `model`, `answers` in the order
//! of the questions, `usage`.

use std::io::Read as _;
use std::path::PathBuf;

use retrograd::decision::Json;
use retrograd::decision::request::SystemOneRequest;
use retrograd::{Device, Engine, EngineConfig, Error, Result};

use crate::args::Args;

pub(crate) mod label;

pub(crate) const FLAGS: &[&str] = &[
    "--model",
    "--adapter",
    "--request",
    "--device",
    "--template",
    "--max-head-tokens",
    "--pack",
];

#[derive(Debug, PartialEq)]
struct DecideArgs {
    model: PathBuf,
    adapter: Option<PathBuf>,
    /// `None` reads the body from stdin.
    request: Option<PathBuf>,
    device: Device,
    template: Option<PathBuf>,
    max_head_tokens: Option<usize>,
    pack: Option<usize>,
}

fn parse_decide_args(args: &[String]) -> Result<DecideArgs> {
    let mut model = None;
    let mut adapter = None;
    let mut request = None;
    let mut device = Device::Auto;
    let mut template = None;
    let mut max_head_tokens = None;
    let mut pack = None;
    let mut seen = [false; 7];
    let mut args = Args::new("decide", args);
    while let Some(flag) = args.next_arg() {
        let Some(index) = FLAGS.iter().position(|known| *known == flag) else {
            return Err(args.unknown(flag));
        };
        args.once(&mut seen[index], flag)?;
        match flag {
            "--model" => model = Some(PathBuf::from(args.value(flag)?)),
            "--adapter" => adapter = Some(PathBuf::from(args.value(flag)?)),
            "--request" => request = Some(args.value(flag)?),
            "--device" => device = args.value(flag)?.parse()?,
            "--template" => template = Some(PathBuf::from(args.value(flag)?)),
            "--max-head-tokens" => {
                max_head_tokens = Some(args.parse(flag, "a positive integer")?);
            }
            "--pack" => pack = Some(args.parse(flag, "a positive integer")?),
            unknown => return Err(args.unknown(unknown)),
        }
    }
    let request = request.ok_or_else(|| Error::invalid("--request is required"))?;
    Ok(DecideArgs {
        model: model.ok_or_else(|| Error::invalid("--model is required"))?,
        adapter,
        request: (request != "-").then(|| PathBuf::from(request)),
        device,
        template,
        max_head_tokens,
        pack,
    })
}

pub(crate) fn decide(args: Vec<String>) -> Result<()> {
    if args.first().is_some_and(|arg| arg == "label") {
        return label::label(&args[1..]);
    }
    let args = parse_decide_args(&args)?;
    let body = match &args.request {
        Some(path) => std::fs::read(path)
            .map_err(|error| Error::invalid(format!("cannot read {}: {error}", path.display())))?,
        None => {
            let mut body = Vec::new();
            std::io::stdin().read_to_end(&mut body)?;
            body
        }
    };
    let n_seq_max = EngineConfig::default().n_seq_max;
    let mut engine = Engine::new(
        &args.model,
        &EngineConfig {
            device: args.device,
            ..EngineConfig::default()
        },
    )?;
    let source = args
        .template
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path)
                .map_err(|error| Error::invalid(format!("cannot read {}: {error}", path.display())))
        })
        .transpose()?;
    let mut info = if engine.decision_metadata()?.decision_type.is_none() {
        engine.set_chat_template_variables(r#"{"enable_thinking":false}"#)?;
        engine.configure_causal_decision()?.0
    } else {
        if let Some(source) = &source {
            engine.set_decision_template(Some(source))?;
        }
        engine.decision_info()?
    };
    if let Some(source) = &source {
        engine.set_decision_template(Some(source))?;
    }
    if let Some(adapter) = &args.adapter {
        engine.load_lora(adapter)?;
    }
    // An adapter trained here carries the temperatures fitted on its weights.
    if let Some(adapter) = &args.adapter {
        info = info
            .with_adapter_temperatures(&retrograd::trainer::adapter_decision_metadata(adapter)?)?;
    }
    if let Some(max_head_tokens) = args.max_head_tokens {
        info = info.with_max_head_tokens(max_head_tokens)?;
    }
    if let Some(pack) = args.pack {
        // Widening u32 -> usize.
        let slots = n_seq_max as usize;
        if pack == 0 || pack > slots {
            return Err(Error::invalid(format!(
                "--pack must be between 1 and {slots}; got {pack}"
            )));
        }
        info = info.with_packing(pack);
    }
    let request = SystemOneRequest::from_slice(&body, &info.readout)?;
    let decision = engine.decide(&info, &request)?;
    // The answer document of `/v1/systemone`, its keys in that order: the
    // answers go through the ordered `Json`, a `Value` would sort them.
    let unserializable = |error: serde_json::Error| {
        Error::runtime(format!("the answer does not serialize: {error}"))
    };
    let answers: Json =
        serde_json::from_str(&serde_json::to_string(&decision.answers).map_err(unserializable)?)
            .map_err(unserializable)?;
    let mut usage = Json::Object(Default::default());
    // Widening usize -> u64.
    usage.set(
        "input_tokens",
        Json::from(serde_json::Value::from(decision.input_tokens as u64)),
    );
    usage.set("output_tokens", Json::from(serde_json::Value::from(0_u64)));
    let mut document = Json::Object(Default::default());
    document.set("model", Json::from(args.model.display().to_string()));
    document.set("answers", answers);
    document.set("usage", usage);
    println!(
        "{}",
        serde_json::to_string_pretty(&document).map_err(unserializable)?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn the_flags_fill_the_arguments_and_dash_reads_stdin() {
        let args = parse_decide_args(&strings(&[
            "--model",
            "m.gguf",
            "--adapter",
            "a.gguf",
            "--request",
            "-",
            "--device",
            "cpu",
            "--max-head-tokens",
            "512",
            "--pack",
            "2",
        ]))
        .unwrap();
        assert_eq!(
            args,
            DecideArgs {
                model: "m.gguf".into(),
                adapter: Some("a.gguf".into()),
                request: None,
                device: Device::Cpu,
                template: None,
                max_head_tokens: Some(512),
                pack: Some(2),
            }
        );
    }

    #[test]
    fn the_model_and_the_request_are_required_once() {
        let error = parse_decide_args(&strings(&["--request", "r.json"])).unwrap_err();
        assert!(error.to_string().contains("--model is required"));
        let error = parse_decide_args(&strings(&["--model", "m.gguf"])).unwrap_err();
        assert!(error.to_string().contains("--request is required"));
        let error =
            parse_decide_args(&strings(&["--model", "m.gguf", "--model", "n.gguf"])).unwrap_err();
        assert!(error.to_string().contains("at most once"));
        let error = parse_decide_args(&strings(&["--temp", "1"])).unwrap_err();
        assert!(error.to_string().contains("unknown decide flag"));
    }
}
