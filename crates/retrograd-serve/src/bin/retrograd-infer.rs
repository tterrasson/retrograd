//! `retrograd-infer` - the standalone inference server: OpenAI chat completions
//! over one base model and its LoRA checkpoints, served by the step-wise
//! inference engine.
//!
//! Usage: `retrograd-infer <config.toml> [--host H] [--port P] [--api-key K]`
//! The configuration is minimal (model, adapters, KV, device, auth, queue) and
//! parsed by this crate alone; see [`retrograd_serve::ServeConfig`].

use std::path::PathBuf;
use std::time::Duration;

use retrograd_openai::SessionOptions;
use retrograd_serve::server::REQUEST_TIMEOUT;
use retrograd_serve::{AdapterSource, Serve, ServeConfig, ServeError};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(error) = run(&args) {
        eprintln!("retrograd-infer: {error}");
        std::process::exit(1);
    }
}

fn usage(message: &str) -> ServeError {
    ServeError::Usage(format!(
        "{message}\n\nusage: retrograd-infer <config.toml> [--host H] [--port P] [--api-key K]"
    ))
}

fn run(args: &[String]) -> Result<(), ServeError> {
    let mut config_path: Option<PathBuf> = None;
    let mut host: Option<String> = None;
    let mut port: Option<u16> = None;
    let mut api_key: Option<String> = None;
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        let mut value = |flag: &str| -> Result<&str, ServeError> {
            index += 1;
            args.get(index)
                .map(String::as_str)
                .ok_or_else(|| usage(&format!("{flag} needs a value")))
        };
        match flag {
            "--host" => host = Some(value(flag)?.to_owned()),
            "--port" => {
                let raw = value(flag)?;
                port = Some(
                    raw.parse()
                        .map_err(|_| usage(&format!("--port takes a port number, got `{raw}`")))?,
                );
            }
            "--api-key" => api_key = Some(value(flag)?.to_owned()),
            "--help" | "-h" => return Err(usage("retrograd-infer serves one model over /v1")),
            other if other.starts_with('-') => {
                return Err(usage(&format!("unknown flag `{other}`")));
            }
            other => {
                if config_path.replace(PathBuf::from(other)).is_some() {
                    return Err(usage("exactly one config TOML path"));
                }
            }
        }
        index += 1;
    }
    let mut config =
        ServeConfig::load(config_path.ok_or_else(|| usage("a config TOML path is required"))?)?;
    if let Some(host) = host {
        config.host = host;
    }
    if let Some(port) = port {
        config.port = port;
    }
    if let Some(api_key) = api_key {
        config.api_key = Some(api_key);
    }
    // The flag overrides are in memory, so re-check the rules they can break
    // (the network exposure one) without re-reading the file.
    config.check_network_rule()?;

    let source = AdapterSource::from_config(&config)?;
    let preload = source.preload();
    let server = Serve {
        source: std::sync::Arc::new(source),
        preload,
        session: SessionOptions {
            idle: Some(Duration::from_secs(config.idle_seconds.unwrap_or(300))),
            queue: config.queue.unwrap_or(8),
            admission: config.admission()?,
        },
        address: config.address()?,
        api_key: config.api_key.clone(),
        timeout: REQUEST_TIMEOUT,
        max_body_bytes: 4 * 1024 * 1024,
        on_loaded: None,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(retrograd_serve::serve(server))?;
    Ok(())
}
