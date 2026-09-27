//! `retrograd serve` - the OpenAI chat-completions contract over one trained
//! model, without a run server.
//!
//! The weights are chosen the way `retrograd chat` chooses them - the adapter
//! the configuration trains unless told otherwise - and loaded before the
//! socket answers anything, so a model that does not load fails here rather
//! than on a client's first request. The base model is served beside it under
//! the id `base`, which is what an A/B through any OpenAI client needs.

use std::net::{IpAddr, SocketAddr};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use indicatif::HumanDuration;
use retrograd::config::{self, OutputKind};
use retrograd::{Device, Error, Result};
use retrograd_cli_ui::CliUi;
use retrograd_openai::{
    Endpoint, OpenAiError, Session, SessionOptions, SingleModel, TrainerLoader, Unshared,
    WeightsSpec,
};

use crate::args::{Args, parse_value};

pub(crate) const FLAGS: &[&str] = &[
    "--adapter",
    "--checkpoint",
    "--base-only",
    "--model",
    "--device",
    "--ctx",
    "--host",
    "--port",
    "--api-key",
    "--model-name",
];

/// Longest a request may wait for its answer. Generous: a CPU answering 512
/// tokens is slow, and a client that wants less says so with its own timeout.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15 * 60);

#[derive(Debug, PartialEq)]
struct ServeArgs {
    config: PathBuf,
    model: Option<PathBuf>,
    adapter: Option<PathBuf>,
    checkpoint: Option<PathBuf>,
    base_only: bool,
    device: Option<Device>,
    ctx: Option<u32>,
    host: String,
    port: u16,
    api_key: Option<String>,
    model_name: Option<String>,
}

pub(crate) fn serve(args: Vec<String>) -> Result<()> {
    let args = parse_serve_args(&args)?;
    let mut run_config = config::load_with(
        &args.config,
        config::ModelOverride {
            path: args.model.clone(),
            device: args.device,
        },
    )?;
    if let Some(ctx) = args.ctx {
        run_config.training.n_ctx = ctx;
    }
    let source = served_models(&args, &run_config)?;
    let address = SocketAddr::new(parse_host(&args.host)?, args.port);

    let ui = CliUi::new();
    ui.section("serve");
    ui.info(format!("config: {}", args.config.display()));
    ui.info(format!(
        "model `{}`: {}{}",
        source.name,
        source.spec.model.display(),
        source
            .spec
            .adapter
            .as_ref()
            .map(|adapter| format!(" + {}", adapter.display()))
            .unwrap_or_default()
    ));
    if source.base.is_some() {
        ui.info(format!(
            "model `{}`: the same base, no adapter",
            retrograd_openai::BASE_MODEL_ID
        ));
    }

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(address).await?;
        let session = Session::new(
            Arc::new(TrainerLoader),
            Arc::new(Unshared),
            SessionOptions {
                // One process, one model: nothing else wants the device back.
                idle: None,
                ..SessionOptions::default()
            },
        )
        .map_err(openai_error)?;

        let started = Instant::now();
        let load = ui.spinner("loading model");
        session
            .preload(source.spec.clone())
            .await
            .map_err(openai_error)
            .inspect_err(|_| ui.fail_spinner(load.clone(), "model loading failed"))?;
        ui.finish_spinner(
            load,
            format!("model loaded in {}", HumanDuration(started.elapsed())),
        );

        let mut v1 = retrograd_openai::router(Endpoint {
            source: Arc::new(source),
            session: Arc::new(session),
            timeout: REQUEST_TIMEOUT,
            enabled: true,
        });
        if let Some(key) = args.api_key.as_deref() {
            v1 = v1.layer(axum::middleware::from_fn_with_state(
                Arc::<str>::from(key),
                retrograd_openai::auth::require_api_key,
            ));
        }
        let router = axum::Router::new().nest("/v1", v1);

        let base = format!("http://{}/v1", listener.local_addr()?);
        ui.info(format!("serving on {base}"));
        ui.info(format!(
            "try: OPENAI_BASE_URL={base} OPENAI_API_KEY={} <any OpenAI client>",
            args.api_key.as_deref().map_or("unused", |_| "<your key>")
        ));
        ui.info("Ctrl-C stops the server");
        axum::serve(listener, router)
            .with_graceful_shutdown(async {
                let _ = tokio::signal::ctrl_c().await;
            })
            .await?;
        Ok(())
    })
}

/// What the ids of this process name, read off the arguments and the
/// configuration the way `retrograd chat` reads them.
fn served_models(args: &ServeArgs, run_config: &config::RunConfig) -> Result<SingleModel> {
    let variables = run_config.chat_template_variables_json();
    let spec = |model: &Path, adapter: Option<PathBuf>| WeightsSpec {
        model: model.to_path_buf(),
        adapter,
        n_ctx: run_config.training.n_ctx,
        device: run_config.training.device,
        chat_template_variables: variables.clone(),
    };
    let base = spec(&run_config.model, None);
    let served = if args.base_only {
        base.clone()
    } else if let Some(checkpoint) = &args.checkpoint {
        // A checkpoint's adapter is written beside its `.state` directory.
        spec(&run_config.model, Some(checkpoint.with_extension("gguf")))
    } else if let Some(adapter) = &args.adapter {
        spec(&run_config.model, Some(adapter.clone()))
    } else {
        match run_config.output.kind {
            OutputKind::Adapter => spec(&run_config.model, Some(run_config.output.path.clone())),
            OutputKind::Model => spec(&run_config.output.path, None),
            OutputKind::Trainable => {
                return Err(Error::invalid(format!(
                    "serve cannot load {}: a trainable bundle is not a standalone model; pass \
                     --adapter, --checkpoint or --base-only",
                    run_config.output.path.display()
                )));
            }
        }
    };
    for path in std::iter::once(&served.model).chain(served.adapter.as_ref()) {
        if !path.is_file() {
            return Err(Error::invalid(format!(
                "serve: {} does not exist",
                path.display()
            )));
        }
    }
    let name = match &args.model_name {
        Some(name) => name.clone(),
        None => served
            .adapter
            .as_deref()
            .unwrap_or(&served.model)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("retrograd")
            .to_owned(),
    };
    let base = (served != base).then_some(base);
    if base.is_some() && name == retrograd_openai::BASE_MODEL_ID {
        return Err(Error::invalid(format!(
            "serve --model-name cannot be `{}`: that id serves the base model",
            retrograd_openai::BASE_MODEL_ID
        )));
    }
    Ok(SingleModel {
        name,
        spec: served,
        base,
        created: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|elapsed| elapsed.as_secs())
            .unwrap_or(0),
    })
}

fn parse_host(host: &str) -> Result<IpAddr> {
    match host {
        "localhost" => Ok(IpAddr::from([127, 0, 0, 1])),
        host => host.parse().map_err(|_| {
            Error::invalid(format!(
                "invalid --host value '{host}'; expected an IP address"
            ))
        }),
    }
}

fn is_loopback(host: &str) -> bool {
    parse_host(host).is_ok_and(|ip| ip.is_loopback())
}

/// The failures of the serving session, as the CLI reports every other one.
fn openai_error(error: OpenAiError) -> Error {
    match error.status.is_client_error() {
        true => Error::invalid(error.message),
        false => Error::runtime(error.message),
    }
}

fn parse_serve_args(args: &[String]) -> Result<ServeArgs> {
    let mut parsed = ServeArgs {
        config: PathBuf::new(),
        model: None,
        adapter: None,
        checkpoint: None,
        base_only: false,
        device: None,
        ctx: None,
        host: "127.0.0.1".into(),
        port: 8000,
        api_key: None,
        model_name: None,
    };
    let mut config = None;
    let mut seen = std::collections::BTreeSet::new();
    let mut args = Args::new("serve", args);
    while let Some(flag) = args.next_arg() {
        if !flag.starts_with('-') {
            if config.replace(PathBuf::from(flag)).is_some() {
                return Err(Error::invalid("serve accepts exactly one config TOML path"));
            }
            continue;
        }
        if !FLAGS.contains(&flag) {
            return Err(args.unknown(flag));
        }
        let mut repeated = !seen.insert(flag);
        args.once(&mut repeated, flag)?;
        if flag == "--base-only" {
            parsed.base_only = true;
            continue;
        }
        let value = args.value(flag)?;
        match flag {
            "--adapter" => parsed.adapter = Some(PathBuf::from(value)),
            "--checkpoint" => parsed.checkpoint = Some(PathBuf::from(value)),
            "--model" => parsed.model = Some(PathBuf::from(value)),
            "--device" => parsed.device = Some(value.parse()?),
            "--ctx" => {
                let ctx: u32 = parse_value("--ctx", value, "an integer")?;
                if ctx == 0 {
                    return Err(Error::invalid("serve --ctx must be greater than zero"));
                }
                parsed.ctx = Some(ctx);
            }
            "--host" => parsed.host = value.to_owned(),
            "--port" => parsed.port = parse_value("--port", value, "a port number")?,
            "--api-key" => {
                if value.trim().is_empty() {
                    return Err(Error::invalid("serve --api-key must not be empty"));
                }
                parsed.api_key = Some(value.to_owned());
            }
            "--model-name" => {
                if value.trim().is_empty() {
                    return Err(Error::invalid("serve --model-name must not be empty"));
                }
                parsed.model_name = Some(value.to_owned());
            }
            unknown => return Err(args.unknown(unknown)),
        }
    }
    let chosen = [
        parsed.adapter.is_some(),
        parsed.checkpoint.is_some(),
        parsed.base_only,
    ];
    if chosen.iter().filter(|chosen| **chosen).count() > 1 {
        return Err(Error::invalid(
            "serve --adapter, --checkpoint and --base-only are mutually exclusive",
        ));
    }
    if let Some(checkpoint) = &parsed.checkpoint
        && checkpoint
            .extension()
            .and_then(|extension| extension.to_str())
            != Some("state")
    {
        return Err(Error::invalid(format!(
            "serve --checkpoint takes a checkpoint directory ending in .state, not {}",
            checkpoint.display()
        )));
    }
    parse_host(&parsed.host)?;
    // A model served without a key on an address other machines reach should
    // be a decision, never a typo.
    if !is_loopback(&parsed.host) && parsed.api_key.is_none() {
        return Err(Error::invalid(format!(
            "refusing to serve on {} without --api-key: the model would be open to the network",
            parsed.host
        )));
    }
    parsed.config = config.ok_or_else(|| Error::invalid("serve requires a config TOML path"))?;
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn serve_parser_covers_defaults_values_and_conflicts() {
        let parsed = parse_serve_args(&strings(&["run.toml"])).unwrap();
        assert_eq!(parsed.host, "127.0.0.1");
        assert_eq!(parsed.port, 8000);
        assert!(parsed.api_key.is_none());

        let parsed = parse_serve_args(&strings(&[
            "run.toml",
            "--checkpoint",
            "ckpt/best.state",
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--api-key",
            "sk-local",
            "--model-name",
            "tuned",
            "--ctx",
            "2048",
            "--device",
            "cpu",
        ]))
        .unwrap();
        assert_eq!(parsed.checkpoint, Some(PathBuf::from("ckpt/best.state")));
        assert_eq!(parsed.port, 9000);
        assert_eq!(parsed.ctx, Some(2048));
        assert_eq!(parsed.device, Some(Device::Cpu));
        assert_eq!(parsed.model_name.as_deref(), Some("tuned"));

        for (args, expected) in [
            (vec!["run.toml", "--host", "0.0.0.0"], "without --api-key"),
            (
                vec!["run.toml", "--adapter", "a.gguf", "--base-only"],
                "mutually exclusive",
            ),
            (
                vec!["run.toml", "--adapter", "a", "--checkpoint", "b.state"],
                "mutually exclusive",
            ),
            (
                vec!["run.toml", "--checkpoint", "b.gguf"],
                "ending in .state",
            ),
            (vec!["run.toml", "--port", "70000"], "a port number"),
            (vec!["run.toml", "--ctx", "0"], "greater than zero"),
            (vec!["run.toml", "--host", "example.org"], "an IP address"),
            (
                vec!["run.toml", "--port", "1", "--port", "2"],
                "--port at most once",
            ),
            (
                vec!["run.toml", "--base-only", "--base-only"],
                "--base-only at most once",
            ),
            (vec!["run.toml", "--api-key", " "], "must not be empty"),
            (vec!["run.toml", "--nope"], "unknown serve flag"),
            (vec![], "requires a config TOML path"),
        ] {
            let error = parse_serve_args(&strings(&args)).unwrap_err().to_string();
            assert!(error.contains(expected), "{args:?} -> {error}");
        }
    }
}
