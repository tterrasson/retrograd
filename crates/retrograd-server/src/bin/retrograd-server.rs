//! `retrograd-server [config.toml] [--bind ADDR] [--state-dir DIR]`
//! `retrograd-server openapi`
//! `retrograd-server config-schema`
//! `retrograd-server view <observe-dir> [--bind ADDR] [--open]`
//!
//! `openapi` writes the API's OpenAPI document to stdout and exits, without
//! reading a configuration or opening a socket: it is how a client snapshots the
//! contract. `config-schema` writes the run configuration's JSON Schema the
//! same way, for an editor to complete a run's TOML with.
//!
//! `view` serves the web interface over one `[observe]` directory - a run
//! started from the command line - read-only, on loopback, and nothing else.
//!
//! Reads the operator's configuration, connects to every declared MCP server,
//! so the catalogue it publishes is verified rather than declarative - and
//! serves the `/v1` router, on loopback by default and wherever `--bind`
//! points otherwise.

use std::path::PathBuf;
use std::sync::Arc;

use retrograd_core::{Error, Result};
use retrograd_server::state::{BudgetRequest, EngineProbe};
use retrograd_server::{AppState, Catalog, ServerConfig, build_router};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outcome = match args.first().map(String::as_str) {
        Some("openapi") => print_document(&args[1..], retrograd_server::openapi::render()),
        Some("config-schema") => config_schema(&args[1..]),
        Some("view") => view(&args[1..]).await,
        _ => run(&args).await,
    };
    if let Err(error) = outcome {
        eprintln!("error {error}");
        std::process::exit(1);
    }
}

/// Writes a generated document to stdout: no configuration, no socket.
fn print_document(args: &[String], document: serde_json::Value) -> Result<()> {
    if !args.is_empty() {
        return Err(Error::invalid("this subcommand takes no argument"));
    }
    let text = serde_json::to_string_pretty(&document)
        .map_err(|error| Error::runtime(format!("could not render the document: {error}")))?;
    println!("{text}");
    Ok(())
}

#[cfg(feature = "openapi")]
fn config_schema(args: &[String]) -> Result<()> {
    print_document(args, retrograd_server::config_schema::document())
}

#[cfg(not(feature = "openapi"))]
fn config_schema(_: &[String]) -> Result<()> {
    Err(Error::invalid(
        "this retrograd-server was built without the `openapi` feature, so it has no \
         configuration schema",
    ))
}

struct ViewArgs {
    directory: PathBuf,
    bind: String,
    open: bool,
}

fn parse_view_args(args: &[String]) -> Result<ViewArgs> {
    let mut directory = None;
    let mut bind = "127.0.0.1:0".to_string();
    let mut open = false;
    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--open" => open = true,
            "--bind" => {
                index += 1;
                bind = args
                    .get(index)
                    .ok_or_else(|| Error::invalid("missing value for --bind"))?
                    .clone();
            }
            flag if flag.starts_with('-') => {
                return Err(Error::invalid(format!("unknown view flag '{flag}'")));
            }
            path => {
                if directory.replace(PathBuf::from(path)).is_some() {
                    return Err(Error::invalid("view takes exactly one observe directory"));
                }
            }
        }
        index += 1;
    }
    let directory = directory.ok_or_else(|| {
        Error::invalid("usage: retrograd-server view <observe-dir> [--bind ADDR] [--open]")
    })?;
    // Parsed rather than string-matched, like the server's own bind: a host
    // name that does not parse is not known to be loopback.
    let loopback = bind
        .parse::<std::net::SocketAddr>()
        .is_ok_and(|address| address.ip().is_loopback());
    if !loopback {
        return Err(Error::invalid(format!(
            "view serves without a token and binds loopback only; '{bind}' is not a loopback \
             address"
        )));
    }
    Ok(ViewArgs {
        directory,
        bind,
        open,
    })
}

async fn view(args: &[String]) -> Result<()> {
    if !cfg!(feature = "ui") {
        return Err(Error::invalid(
            "this retrograd-server was built without the web interface; rebuild it with \
             `--features ui` to use `view`",
        ));
    }
    let args = parse_view_args(args)?;
    let directory = args.directory.canonicalize().map_err(|error| {
        Error::invalid(format!("cannot open {}: {error}", args.directory.display()))
    })?;
    if !directory.is_dir() {
        return Err(Error::invalid(format!(
            "{} is not a directory",
            directory.display()
        )));
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "retrograd_server=warn".into()),
        )
        .init();
    let listener = tokio::net::TcpListener::bind(&args.bind)
        .await
        .map_err(|error| Error::runtime(format!("could not bind {}: {error}", args.bind)))?;
    let address = listener
        .local_addr()
        .map_err(|error| Error::runtime(format!("could not read the bound address: {error}")))?;
    let url = format!("http://{address}/runs/local/trajectories");
    println!("viewing {} at {url}", directory.display());
    if args.open {
        open_browser(&url);
    }
    axum::serve(listener, retrograd_server::build_viewer_router(directory))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| Error::runtime(format!("server failed: {error}")))
}

/// Best effort: the URL is printed either way.
fn open_browser(url: &str) {
    let command = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    if let Err(error) = std::process::Command::new(command).arg(url).spawn() {
        eprintln!("could not open a browser ({error}); open {url}");
    }
}

struct Args {
    config: Option<PathBuf>,
    bind: Option<String>,
    state_dir: Option<PathBuf>,
    vram_budget: Option<BudgetRequest>,
    ram_budget: Option<BudgetRequest>,
}

fn parse_args(args: &[String]) -> Result<Args> {
    let mut parsed = Args {
        config: None,
        bind: None,
        state_dir: None,
        vram_budget: None,
        ram_budget: None,
    };
    let mut index = 0;
    while index < args.len() {
        let flag = args[index].as_str();
        if !flag.starts_with('-') {
            if parsed.config.replace(PathBuf::from(flag)).is_some() {
                return Err(Error::invalid(
                    "retrograd-server accepts exactly one config TOML path",
                ));
            }
            index += 1;
            continue;
        }
        let value = args
            .get(index + 1)
            .ok_or_else(|| Error::invalid(format!("missing value for {flag}")))?;
        match flag {
            "--bind" => parsed.bind = Some(value.clone()),
            "--state-dir" => parsed.state_dir = Some(PathBuf::from(value)),
            "--vram-budget" => parsed.vram_budget = Some(BudgetRequest::parse(value)?),
            "--ram-budget" => parsed.ram_budget = Some(BudgetRequest::parse(value)?),
            unknown => {
                return Err(Error::invalid(format!(
                    "unknown retrograd-server flag '{unknown}'"
                )));
            }
        }
        index += 2;
    }
    Ok(parsed)
}

async fn run(args: &[String]) -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "retrograd_server=info,tower_http=info".into()),
        )
        .init();

    let args = parse_args(args)?;
    let mut config = match &args.config {
        Some(path) => ServerConfig::load(path)?,
        None => ServerConfig::default(),
    };
    // The command line wins over the file: it is the more immediate intent.
    if let Some(bind) = args.bind {
        config.bind = Some(bind);
    }
    if let Some(state_dir) = args.state_dir {
        config.state_dir = Some(state_dir);
    }
    if let Some(budget) = args.vram_budget {
        config.vram_budget = budget;
    }
    if let Some(budget) = args.ram_budget {
        config.ram_budget = budget;
    }
    // Before `validate`: whether a token is set is what decides whether a
    // non-loopback bind is allowed at all, and a token in the environment counts.
    let config = config.with_token_from_env();
    config.validate()?;
    if config.auth_token.is_none() {
        tracing::warn!(
            "no auth_token: this server is unauthenticated and is therefore bound to loopback"
        );
    }
    if config.path_roots.is_empty() {
        tracing::warn!(
            "no path_roots: every model, dataset and output path the client sends is accepted"
        );
    }

    // Connecting here, before the listener exists, is what makes the published
    // tool lists verified. A required MCP server that is down therefore stops
    // startup - intentional, and documented as a deployment consequence.
    let catalog = Catalog::connect(
        &config.rewards,
        &config.judges,
        &config.mcp_servers,
        &config.environments,
    )
    .await?;

    let address = config.bind_address().to_string();
    let state_dir = config.state_dir();
    std::fs::create_dir_all(&state_dir)?;
    // Opening the state directory is what reads the history back and marks any
    // run left `running` by a previous process as `interrupted`. Logged
    // because it is a fact about this machine an operator wants at startup, not
    // after a client notices.
    let state = AppState::new(config, catalog, Arc::new(EngineProbe));
    tracing::info!(
        restored = state.registry.len(),
        interrupted = state
            .registry
            .count_with(retrograd_server::dto::RunStatus::Interrupted),
        calibrations = state
            .calibration
            .read()
            .map(|store| store.entries.len())
            .unwrap_or(0),
        "state directory read"
    );
    let router = build_router(state.clone());

    let listener = tokio::net::TcpListener::bind(&address)
        .await
        .map_err(|error| Error::runtime(format!("could not bind {address}: {error}")))?;
    tracing::info!(
        address = %address,
        state_dir = %state_dir.display(),
        authenticated = state.config.auth_token.is_some(),
        path_roots = state.config.path_roots.len(),
        "retrograd-server listening"
    );
    let outcome = axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|error| Error::runtime(format!("server failed: {error}")));

    // After the listener is closed and before the process exits: every live run is
    // asked to checkpoint and stop at its next boundary, so a restart resumes
    // instead of repeating.
    tokio::select! {
        _ = retrograd_server::shutdown::drain(&state, DRAIN_BOUND) => {}
        // The operator's override. A patient shutdown must not make an impatient
        // one unavailable.
        _ = tokio::signal::ctrl_c() => tracing::warn!(
            "second shutdown signal: leaving the runs where they are; the next start reads \
             them as interrupted"
        ),
    }
    outcome
}

/// How long a shutdown waits for the runs to reach a boundary.
///
/// Ten minutes is one long rollout update, which is the unit being waited for. A
/// second Ctrl-C is the operator's override: the signal handler returns
/// immediately the second time, so an impatient shutdown is always available
/// without making the patient one unavailable.
const DRAIN_BOUND: std::time::Duration = std::time::Duration::from_secs(600);

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    tracing::info!("shutdown signal received; no longer accepting requests");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn argument_parsing_accepts_the_documented_flags_and_rejects_the_rest() {
        let parsed = parse_args(&strings(&[
            "server.toml",
            "--bind",
            "0.0.0.0:9000",
            "--state-dir",
            "/tmp/runs",
            "--vram-budget",
            "6GiB",
            "--ram-budget",
            "0.5",
        ]))
        .expect("parse");
        assert_eq!(parsed.config, Some(PathBuf::from("server.toml")));
        assert_eq!(parsed.bind.as_deref(), Some("0.0.0.0:9000"));
        assert_eq!(parsed.state_dir, Some(PathBuf::from("/tmp/runs")));
        assert_eq!(
            parsed.vram_budget,
            Some(BudgetRequest::Bytes(6 * 1024 * 1024 * 1024))
        );
        assert_eq!(parsed.ram_budget, Some(BudgetRequest::Fraction(0.5)));

        assert!(parse_args(&strings(&["a.toml", "b.toml"])).is_err());
        assert!(parse_args(&strings(&["--nope", "1"])).is_err());
        assert!(parse_args(&strings(&["--bind"])).is_err());
        assert!(parse_args(&strings(&["--vram-budget", "banana"])).is_err());
        // No config file at all is valid: the defaults are a usable server with
        // an empty catalogue.
        assert!(parse_args(&[]).expect("parse").config.is_none());
    }

    #[test]
    fn view_binds_loopback_only() {
        let parsed = parse_view_args(&strings(&["runs/1/observe", "--open"])).expect("parse");
        assert_eq!(parsed.directory, PathBuf::from("runs/1/observe"));
        assert_eq!(parsed.bind, "127.0.0.1:0");
        assert!(parsed.open);
        assert!(parse_view_args(&strings(&["dir", "--bind", "[::1]:9000"])).is_ok());
        assert!(parse_view_args(&strings(&["dir", "--bind", "0.0.0.0:9000"])).is_err());
        assert!(parse_view_args(&strings(&["dir", "--bind", "localhost:9000"])).is_err());
        assert!(parse_view_args(&strings(&[])).is_err());
        assert!(parse_view_args(&strings(&["a", "b"])).is_err());
    }
}
