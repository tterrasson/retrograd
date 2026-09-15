use std::env;
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;

use indicatif::HumanDuration;
use std::time::Instant;

use retrograd::config;
use retrograd::{Device, Error, Result, SamplingParams, Trainer};

use crate::args::{Args, parse_value};
use crate::bench::load_bench_adapter;
use retrograd_cli_ui::{CliUi, paint};

pub(crate) const FLAGS: &[&str] = &[
    "--compare",
    "--base_only",
    "--base-only",
    "--model",
    "--adapter",
    "--device",
    "--system",
    "--ctx",
    "--temp",
    "--temperature",
    "--top-p",
    "--top_p",
    "--max-new-tokens",
    "--max_new_tokens",
    "--max-tokens",
    "--seed",
];

/// Which weights answer each turn of an interactive chat session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ChatMode {
    /// Base model plus the loaded LoRA adapter (default).
    Adapter,
    /// Base model alone, no adapter loaded.
    BaseOnly,
    /// Both, side by side: every turn generates from the base then the adapter.
    Compare,
}

#[derive(Debug)]
struct ChatArgs {
    config: PathBuf,
    model: Option<PathBuf>,
    adapter: Option<PathBuf>,
    device: Option<Device>,
    ctx: Option<u32>,
    system: Option<String>,
    sampling: SamplingParams,
    mode: ChatMode,
}

pub(crate) fn chat_model(args: Vec<String>) -> Result<()> {
    let args = parse_chat_args(&args)?;
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
    // Generation decodes the prompt in n_batch chunks inside a single context,
    // so the batch sizes only have to stay within the context window.
    run_config.training.n_batch = run_config.training.n_batch.min(run_config.training.n_ctx);
    run_config.training.n_ubatch = run_config
        .training
        .n_ubatch
        .min(run_config.training.n_batch);

    let ui = CliUi::new();
    ui.section("chat");
    ui.info(format!("config: {}", args.config.display()));
    ui.info(format!("model: {}", run_config.model.display()));
    ui.info(format!(
        "mode: {}",
        match args.mode {
            ChatMode::Adapter => "base + adapter",
            ChatMode::BaseOnly => "base only",
            ChatMode::Compare => "compare (base vs adapter)",
        }
    ));

    let started = Instant::now();
    let load = ui.spinner("loading model");
    let mut trainer =
        Trainer::new(&run_config.model, run_config.training.clone()).inspect_err(|_| {
            ui.fail_spinner(load.clone(), "model loading failed");
        })?;
    ui.finish_spinner(
        load,
        format!("model loaded in {}", HumanDuration(started.elapsed())),
    );

    // Every mode except base-only chats through the adapter, defaulting to the
    // adapter this config trains (lora.output) when none is given explicitly.
    if args.mode != ChatMode::BaseOnly {
        let adapter = args
            .adapter
            .clone()
            .unwrap_or_else(|| run_config.lora.output.clone());
        load_bench_adapter(&mut trainer, &adapter, &ui)?;
    }

    run_chat_loop(&mut trainer, &args, run_config.training.n_ctx, &ui)
}

fn parse_chat_args(args: &[String]) -> Result<ChatArgs> {
    let mut config = None;
    let mut model = None;
    let mut adapter = None;
    let mut device = None;
    let mut ctx = None;
    let mut system = None;
    let mut compare = false;
    let mut base_only = false;
    let mut model_seen = false;
    let mut adapter_seen = false;
    let mut device_seen = false;
    let mut ctx_seen = false;
    let mut system_seen = false;
    let mut temperature_seen = false;
    let mut top_p_seen = false;
    let mut max_tokens_seen = false;
    let mut seed_seen = false;
    let mut compare_seen = false;
    let mut base_only_seen = false;
    // Conversational defaults: a little exploration, room for a full answer.
    let mut sampling = SamplingParams {
        temperature: 0.7,
        top_p: 0.95,
        max_new_tokens: 512,
        seed: 0,
    };
    let mut args = Args::new("chat", args);

    while let Some(flag) = args.next_arg() {
        if !flag.starts_with('-') {
            if config.replace(PathBuf::from(flag)).is_some() {
                return Err(Error::invalid("chat accepts exactly one config TOML path"));
            }
            continue;
        }
        if !FLAGS.contains(&flag) {
            return Err(args.unknown(flag));
        }
        match flag {
            "--compare" => {
                args.once(&mut compare_seen, "--compare")?;
                compare = true;
                continue;
            }
            "--base_only" | "--base-only" => {
                args.once(&mut base_only_seen, "--base_only")?;
                base_only = true;
                continue;
            }
            _ => {}
        }
        let value = args.value(flag)?;
        match flag {
            "--model" => {
                args.once(&mut model_seen, "--model")?;
                model = Some(PathBuf::from(value));
            }
            "--adapter" => {
                args.once(&mut adapter_seen, "--adapter")?;
                adapter = Some(PathBuf::from(value));
            }
            "--device" => {
                args.once(&mut device_seen, "--device")?;
                device = Some(value.parse()?);
            }
            "--system" => {
                args.once(&mut system_seen, "--system")?;
                system = Some(value.to_owned());
            }
            "--ctx" => {
                args.once(&mut ctx_seen, "--ctx")?;
                let parsed: u32 = parse_value("--ctx", value, "an integer")?;
                if parsed == 0 {
                    return Err(Error::invalid("chat --ctx must be greater than zero"));
                }
                ctx = Some(parsed);
            }
            "--temp" | "--temperature" => {
                args.once(&mut temperature_seen, "temperature")?;
                let parsed: f32 = parse_value("--temp", value, "a number")?;
                if !(parsed > 0.0 && parsed.is_finite()) {
                    return Err(Error::invalid(
                        "chat --temp must be finite and greater than zero",
                    ));
                }
                sampling.temperature = parsed;
            }
            "--top-p" | "--top_p" => {
                args.once(&mut top_p_seen, "top-p")?;
                let parsed: f32 = parse_value("--top-p", value, "a number")?;
                if !(parsed > 0.0 && parsed <= 1.0 && parsed.is_finite()) {
                    return Err(Error::invalid("chat --top-p must be in (0, 1]"));
                }
                sampling.top_p = parsed;
            }
            "--max-new-tokens" | "--max_new_tokens" | "--max-tokens" => {
                args.once(&mut max_tokens_seen, "max-new-tokens")?;
                let parsed: u32 = parse_value("--max-new-tokens", value, "an integer")?;
                if parsed == 0 {
                    return Err(Error::invalid(
                        "chat --max-new-tokens must be greater than zero",
                    ));
                }
                sampling.max_new_tokens = parsed;
            }
            "--seed" => {
                args.once(&mut seed_seen, "--seed")?;
                sampling.seed = parse_value("--seed", value, "an integer")?;
            }
            unknown => return Err(args.unknown(unknown)),
        }
    }

    if compare && base_only {
        return Err(Error::invalid(
            "chat --compare and --base_only are mutually exclusive",
        ));
    }
    if base_only && adapter.is_some() {
        return Err(Error::invalid(
            "chat --base_only cannot be combined with --adapter",
        ));
    }
    let mode = if base_only {
        ChatMode::BaseOnly
    } else if compare {
        ChatMode::Compare
    } else {
        ChatMode::Adapter
    };

    Ok(ChatArgs {
        config: config.ok_or_else(|| Error::invalid("chat requires a config TOML path"))?,
        model,
        adapter,
        device,
        ctx,
        system,
        sampling,
        mode,
    })
}

/// Reads user turns from stdin and prints model replies until end of input or a
/// `/exit` command. Conversation history is threaded through the model's own
/// chat template so multi-turn context is preserved.
fn run_chat_loop(trainer: &mut Trainer, args: &ChatArgs, n_ctx: u32, ui: &CliUi) -> Result<()> {
    let color = std::io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    let context = trainer.context_size().unwrap_or(n_ctx as usize);
    ui.info(format!(
        "context: {context} tokens, temperature={}, top_p={}, max_new_tokens={}",
        args.sampling.temperature, args.sampling.top_p, args.sampling.max_new_tokens
    ));
    ui.info("commands: /reset clears the history, /exit or Ctrl-D quits");

    let mut messages: Vec<(String, String)> = Vec::new();
    if let Some(system) = &args.system {
        messages.push(("system".to_string(), system.clone()));
    }

    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    let mut turn = 0_u32;
    loop {
        print!("{} ", paint(color, "36;1", "you>"));
        std::io::stdout().flush().ok();
        let Some(line) = lines.next() else {
            println!();
            break;
        };
        let line = line?;
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        match input {
            "/exit" | "/quit" => break,
            "/reset" => {
                messages.retain(|(role, _)| role == "system");
                ui.info("history cleared");
                continue;
            }
            _ => {}
        }
        messages.push(("user".to_string(), input.to_string()));

        let borrowed: Vec<(&str, &str)> = messages
            .iter()
            .map(|(role, content)| (role.as_str(), content.as_str()))
            .collect();
        let prompt = trainer.format_chat(&borrowed, true)?;
        let tokens = trainer.tokenize_text(&prompt)?;
        // Share the seed between the base and adapter passes of a compare turn so
        // the two completions differ only by the adapter, and vary it per turn so
        // repeated prompts are not answered identically.
        let sampling = SamplingParams {
            seed: args.sampling.seed.wrapping_add(turn),
            ..args.sampling
        };

        let reply = match args.mode {
            ChatMode::BaseOnly | ChatMode::Adapter => {
                let reply = generate_reply(trainer, &tokens, &sampling, false)?;
                print_chat_reply(color, "bot", "38;5;40", &reply);
                reply
            }
            ChatMode::Compare => {
                let base = generate_reply(trainer, &tokens, &sampling, true)?;
                print_chat_reply(color, "base", "38;5;245", &base);
                let adapted = generate_reply(trainer, &tokens, &sampling, false)?;
                print_chat_reply(color, "adapter", "38;5;40", &adapted);
                // The adapter answer is the one under test, so it carries the
                // shared history forward.
                adapted
            }
        };
        messages.push(("assistant".to_string(), reply));
        turn = turn.wrapping_add(1);
    }
    Ok(())
}

/// Generates one completion for `prompt`, either with the adapter (`base =
/// false`) or under the frozen base model (`base = true`), and returns the
/// decoded text with end-of-generation markers stripped.
fn generate_reply(
    trainer: &mut Trainer,
    prompt: &[i32],
    sampling: &SamplingParams,
    base: bool,
) -> Result<String> {
    let generation = if base {
        trainer.generate_base(prompt, sampling)?
    } else {
        trainer.generate(prompt, sampling)?
    };
    let mut tokens = generation.tokens;
    while tokens
        .last()
        .is_some_and(|&token| trainer.is_eog_token(token).unwrap_or(false))
    {
        tokens.pop();
    }
    Ok(trainer.detokenize(&tokens, false)?.trim().to_string())
}

fn print_chat_reply(color: bool, label: &str, code: &str, text: &str) {
    println!("{} {text}", paint(color, code, &format!("{label}>")));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn chat_parser_covers_modes_aliases_defaults_and_conflicts() {
        let parsed = parse_chat_args(&strings(&[
            "run.toml",
            "--model",
            "base.gguf",
            "--system",
            "Be concise",
            "--device",
            "gpu",
            "--ctx",
            "128",
            "--temperature",
            "0.5",
            "--top_p",
            "0.8",
            "--max_tokens",
            "32",
            "--seed",
            "9",
            "--compare",
        ]));
        // `--max_tokens` is intentionally not a documented alias: this
        // should fail instead of silently accepting a typo.
        assert!(parsed.is_err());

        let parsed = parse_chat_args(&strings(&[
            "run.toml",
            "--model",
            "base.gguf",
            "--system",
            "Be concise",
            "--device",
            "gpu",
            "--ctx",
            "128",
            "--temperature",
            "0.5",
            "--top_p",
            "0.8",
            "--max-new-tokens",
            "32",
            "--seed",
            "9",
            "--compare",
        ]))
        .unwrap();
        assert_eq!(parsed.mode, ChatMode::Compare);
        assert_eq!(parsed.device, Some(Device::Gpu));
        assert_eq!(parsed.ctx, Some(128));
        assert_eq!(parsed.sampling.temperature, 0.5);
        assert_eq!(parsed.sampling.top_p, 0.8);
        assert_eq!(parsed.sampling.max_new_tokens, 32);
        assert_eq!(parsed.sampling.seed, 9);

        for (args, expected) in [
            (
                vec!["run.toml", "--temp", "0"],
                "must be finite and greater than zero",
            ),
            (
                vec!["run.toml", "--top-p", "1.1"],
                "top-p must be in (0, 1]",
            ),
            (
                vec!["run.toml", "--max-new-tokens", "0"],
                "max-new-tokens must be",
            ),
            (
                vec!["run.toml", "--compare", "--base-only"],
                "mutually exclusive",
            ),
            (
                vec!["run.toml", "--base-only", "--adapter", "a.gguf"],
                "cannot be combined with --adapter",
            ),
            (vec!["run.toml", "--seed"], "missing value"),
            (
                vec!["run.toml", "--top-p", "0.9", "--top_p", "0.8"],
                "top-p at most once",
            ),
            (
                vec!["run.toml", "--model", "a", "--model", "b"],
                "--model at most once",
            ),
            (
                vec!["run.toml", "--adapter", "a", "--adapter", "b"],
                "--adapter at most once",
            ),
            (
                vec!["run.toml", "--device", "cpu", "--device", "cpu"],
                "--device at most once",
            ),
            (
                vec!["run.toml", "--system", "a", "--system", "b"],
                "--system at most once",
            ),
            (
                vec!["run.toml", "--ctx", "8", "--ctx", "16"],
                "--ctx at most once",
            ),
            (
                vec![
                    "run.toml",
                    "--max-new-tokens",
                    "8",
                    "--max_new_tokens",
                    "16",
                ],
                "max-new-tokens at most once",
            ),
            (
                vec!["run.toml", "--seed", "1", "--seed", "2"],
                "--seed at most once",
            ),
            (
                vec!["run.toml", "--compare", "--compare"],
                "--compare at most once",
            ),
            (
                vec!["run.toml", "--base_only", "--base-only"],
                "--base_only at most once",
            ),
        ] {
            let error = parse_chat_args(&strings(&args)).unwrap_err().to_string();
            assert!(error.contains(expected), "{args:?} -> {error}");
        }
    }
}
