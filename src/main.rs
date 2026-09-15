mod args;
mod bench;
mod chat;
mod distill_teacher;
mod inspect;
#[cfg(feature = "agent")]
mod judge;
mod preflight;
#[cfg(feature = "agent")]
mod scenarios;
#[cfg(feature = "agent")]
mod tools;
mod train;

use std::env;

use retrograd::Error;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    if let Err(err) = run() {
        eprintln!("error {err}");
        std::process::exit(1);
    }
}

fn run() -> retrograd::Result<()> {
    let mut args = env::args().skip(1);
    match args.next().as_deref() {
        Some("train") => {
            let rest: Vec<String> = args.collect();
            train::train(train::parse_train_args(&rest)?)
        }
        Some("bench") => bench::bench_model(args.collect()),
        Some("chat") => chat::chat_model(args.collect()),
        Some("inspect") => inspect::inspect_model(args.collect()),
        Some("distill-teacher") => distill_teacher::distill_teacher(args.collect()),
        #[cfg(feature = "agent")]
        Some("judge") => judge::judge(args.collect()),
        #[cfg(feature = "agent")]
        Some("tools") => tools::tools(args.collect()),
        #[cfg(feature = "agent")]
        Some("scenarios") => scenarios::scenarios(args.collect()),
        Some("preflight") => preflight::preflight_model(args.collect()),
        Some("-h") | Some("--help") | None => {
            print_help();
            Ok(())
        }
        Some(command) => Err(Error::invalid(format!("unknown command '{command}'"))),
    }
}

fn print_help() {
    println!("{HELP}");
}

const HELP: &str = "retrograd

Commands:
  train <config.toml> [--resume [checkpoint.state]] [--model base.gguf]
        [--device auto|cpu|gpu]
  bench <config.toml> [--data eval.jsonl] [--model base.gguf]
        [--adapter adapter.gguf] [--device auto|cpu|gpu]
        [--format auto|text|jsonl] [--ctx N] [--limit N]
  chat <config.toml> [--compare] [--base_only] [--adapter adapter.gguf]
       [--model base.gguf] [--device auto|cpu|gpu] [--ctx N] [--system PROMPT]
       [--temp T] [--top-p P] [--max-new-tokens N] [--seed N]
  inspect --model base.gguf [--device auto|cpu|gpu]
  distill-teacher <config.toml> [--data corpus.jsonl] [--out corpus.topk]
        [--k N] [--model base.gguf] [--device auto|cpu|gpu] [--ctx N]
  judge eval <config.toml> --fixtures fixtures.jsonl
  tools list <config.toml> [--json] [--no-connect]
  scenarios generate <config.toml> [--force] [--dry-run]
  preflight --model base.gguf [--device auto|cpu|gpu] [--targets a,b] [--strict]

Compatibility aliases:
  bench: --dataset --eval-data
  chat: --base-only --temperature --top_p --max_new_tokens --max-tokens";

#[cfg(test)]
mod help_tests {
    use super::*;

    #[test]
    fn help_names_every_accepted_subcommand_flag() {
        let groups: &[&[&str]] = &[
            train::FLAGS,
            bench::FLAGS,
            chat::FLAGS,
            inspect::FLAGS,
            distill_teacher::FLAGS,
            preflight::FLAGS,
            #[cfg(feature = "agent")]
            judge::FLAGS,
            #[cfg(feature = "agent")]
            tools::FLAGS,
            #[cfg(feature = "agent")]
            scenarios::FLAGS,
        ];
        for flag in groups.iter().flat_map(|flags| flags.iter()) {
            assert!(
                HELP.contains(flag),
                "accepted flag {flag} is absent from --help"
            );
        }
    }
}
