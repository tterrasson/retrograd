//! `retrograd collect` - roll an agentic configuration out, keep the traces
//! that succeeded, and write them as an SFT dataset.
//!
//! The first half of "generate, filter, SFT, GRPO": the dataset it writes is
//! the warm-start that gets a policy calling its tools validly before GRPO has
//! any signal to learn from.

use std::path::PathBuf;

use indicatif::{HumanDuration, ProgressBar};
use retrograd::config::{self};
use retrograd::run::{self, CollectOptions, RunObserver};
use retrograd::{Device, Error, Result};

use retrograd_cli_ui::CliUi;

pub(crate) const FLAGS: &[&str] = &[
    "--out",
    "--k",
    "--keep",
    "--min-reward",
    "--require-verified",
    "--allow-unverified",
    "--raw",
    "--limit",
    "--seed",
    "--model",
    "--device",
    "--report",
    "--force",
    "--api",
];

#[derive(Debug)]
struct Args {
    config: PathBuf,
    model: Option<PathBuf>,
    device: Option<Device>,
    options: CollectOptions,
}

fn parse(args: &[String]) -> Result<Args> {
    let mut parsed = Args {
        config: PathBuf::new(),
        model: None,
        device: None,
        options: CollectOptions {
            keep: 1,
            ..CollectOptions::default()
        },
    };
    let mut out = None;
    let mut positional = None;
    let mut line = crate::args::Args::new("collect", args);
    while let Some(argument) = line.next_arg() {
        let options = &mut parsed.options;
        match argument {
            "--out" => out = Some(PathBuf::from(line.value(argument)?)),
            "--k" => options.k = Some(line.parse(argument, "an integer")?),
            "--keep" => options.keep = line.parse(argument, "an integer")?,
            "--min-reward" => options.min_reward = Some(line.parse(argument, "a number")?),
            "--require-verified" => options.require_verified = Some(true),
            "--allow-unverified" => options.require_verified = Some(false),
            "--raw" => options.raw = true,
            "--limit" => options.limit = Some(line.parse(argument, "an integer")?),
            "--seed" => options.seed = Some(line.parse(argument, "an integer")?),
            "--model" => parsed.model = Some(PathBuf::from(line.value(argument)?)),
            "--device" => parsed.device = Some(line.value(argument)?.parse()?),
            "--report" => options.report = Some(PathBuf::from(line.value(argument)?)),
            "--force" => options.force = true,
            "--api" => options.api = true,
            other if other.starts_with("--") && !FLAGS.contains(&other) => {
                return Err(line.unknown(other));
            }
            other if other.starts_with("--") => {
                // In FLAGS but not matched above: the two lists have drifted.
                return Err(Error::invalid(format!(
                    "collect flag {other} is declared but not handled"
                )));
            }
            other => {
                if positional.replace(PathBuf::from(other)).is_some() {
                    return Err(Error::invalid("collect takes one configuration file"));
                }
            }
        }
    }
    parsed.config =
        positional.ok_or_else(|| Error::invalid("collect requires a configuration file"))?;
    parsed.options.out = out.ok_or_else(|| Error::invalid("collect requires --out PATH"))?;
    if parsed.options.k == Some(0) || parsed.options.keep == 0 {
        return Err(Error::invalid("--k and --keep must be at least 1"));
    }
    if parsed.options.api && (parsed.model.is_some() || parsed.options.raw) {
        return Err(Error::invalid(
            "--api generates with [agent.collect_api]: it takes neither --model, which names a \
             local generator, nor --raw, since the calls come back already parsed",
        ));
    }
    Ok(parsed)
}

pub(crate) fn collect(args: Vec<String>) -> Result<()> {
    let args = parse(&args)?;
    let run_config = config::load_with(
        &args.config,
        config::ModelOverride {
            path: args.model.clone(),
            device: args.device,
        },
    )?;
    let ui = CliUi::new();
    ui.section("collect");
    ui.info(format!("config: {}", args.config.display()));
    if !args.options.api {
        ui.info(format!("generator: {}", run_config.model.display()));
    }
    if args.options.raw && args.model.is_some() {
        // A raw turn is the generator's own markup; a student of another family
        // cannot read it back, and preparing the dataset will say so.
        ui.diagnostic(
            "raw turns from another model",
            "--raw writes the generator's call markup verbatim. Train on it only with a student \
             of the generator's own family; otherwise drop --raw for structured turns",
        );
    }
    let mut observer = CollectObserver {
        ui: CliUi::new(),
        spinner: None,
    };
    let outcome = run::collect(&run_config, &args.options, &mut observer)?;
    let stats = &outcome.stats;
    match &outcome.written {
        Some(path) => ui.info(format!(
            "kept {} of {} attempts in {}",
            stats.kept,
            stats.attempted,
            path.display()
        )),
        None => ui.info(format!(
            "kept none of {} attempts: nothing written",
            stats.attempted
        )),
    }
    let rejected = &stats.rejected;
    ui.info(format!(
        "rejected: {} truncated, {} unscored, {} with an invalid turn, {} below min-reward, \
         {} unverified, {} duplicates, {} past --keep, {} unexportable",
        rejected.truncated,
        rejected.unscored,
        rejected.invalid_turns,
        rejected.below_min_reward,
        rejected.unverified,
        rejected.duplicate,
        rejected.over_keep,
        rejected.unexportable,
    ));
    if stats.failures.total() > 0 {
        ui.info(format!(
            "failed rollouts: {} tool, {} policy, {} other",
            stats.failures.tool, stats.failures.policy, stats.failures.other
        ));
    }
    let unsolved = stats
        .unsolved()
        .map(|report| report.id.as_str())
        .collect::<Vec<_>>();
    if !unsolved.is_empty() {
        ui.info(format!(
            "no attempt passed on {} scenarios, candidates for a stronger generator: {}",
            unsolved.len(),
            unsolved.join(", ")
        ));
    }
    if let Some(report) = &args.options.report {
        ui.info(format!("report: {}", report.display()));
    }
    Ok(())
}

struct CollectObserver {
    ui: CliUi,
    spinner: Option<ProgressBar>,
}

impl RunObserver for CollectObserver {
    fn info(&mut self, message: &str) {
        self.ui.info(message);
    }

    fn diagnostic(&mut self, title: &str, body: &str) {
        self.ui.diagnostic(title, body);
    }

    fn model_load_started(&mut self) {
        self.spinner = Some(self.ui.spinner("loading model"));
    }

    fn model_load_failed(&mut self) {
        if let Some(spinner) = self.spinner.take() {
            self.ui.fail_spinner(spinner, "model loading failed");
        }
    }

    fn model_load_finished(&mut self, elapsed: std::time::Duration) {
        if let Some(spinner) = self.spinner.take() {
            self.ui.finish_spinner(
                spinner,
                format!("model loaded in {}", HumanDuration(elapsed)),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(args: &[&str]) -> Vec<String> {
        args.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn every_flag_reaches_the_options() {
        let args = parse(&strings(&[
            "run.toml",
            "--out",
            "traces.jsonl",
            "--k",
            "4",
            "--keep",
            "2",
            "--min-reward",
            "0.5",
            "--allow-unverified",
            "--raw",
            "--limit",
            "10",
            "--seed",
            "7",
            "--model",
            "teacher.gguf",
            "--device",
            "cpu",
            "--report",
            "report.json",
            "--force",
        ]))
        .unwrap();
        assert_eq!(args.config, PathBuf::from("run.toml"));
        assert_eq!(args.model, Some(PathBuf::from("teacher.gguf")));
        assert_eq!(args.device, Some(Device::Cpu));
        let options = args.options;
        assert_eq!(options.out, PathBuf::from("traces.jsonl"));
        assert_eq!(options.k, Some(4));
        assert_eq!(options.keep, 2);
        assert_eq!(options.min_reward, Some(0.5));
        assert_eq!(options.require_verified, Some(false));
        assert!(options.raw);
        assert_eq!(options.limit, Some(10));
        assert_eq!(options.seed, Some(7));
        assert_eq!(options.report, Some(PathBuf::from("report.json")));
        assert!(options.force);
    }

    #[test]
    fn the_defaults_keep_the_best_trace_and_leave_the_rest_to_the_configuration() {
        let options = parse(&strings(&["run.toml", "--out", "traces.jsonl"]))
            .unwrap()
            .options;
        assert_eq!(options.keep, 1);
        assert_eq!(options.k, None, "the configuration's group_size");
        assert_eq!(
            options.require_verified, None,
            "implicit on verified scenarios"
        );
        assert!(!options.raw && !options.force && !options.api);
        let options = parse(&strings(&["run.toml", "--out", "traces.jsonl", "--api"]))
            .unwrap()
            .options;
        assert!(options.api);
    }

    #[test]
    fn a_collection_needs_somewhere_to_go_and_something_to_keep() {
        for (args, error) in [
            (&["run.toml"][..], "--out"),
            (&["--out", "x.jsonl"][..], "configuration file"),
            (
                &["run.toml", "--out", "x.jsonl", "--keep", "0"][..],
                "at least 1",
            ),
            (
                &["run.toml", "--out", "x.jsonl", "--bogus"][..],
                "unknown collect flag",
            ),
            (
                &["run.toml", "--out", "x.jsonl", "--api", "--raw"][..],
                "--api generates",
            ),
            (
                &["run.toml", "--out", "x.jsonl", "--api", "--model", "m.gguf"][..],
                "--api generates",
            ),
        ] {
            let message = parse(&strings(args)).unwrap_err().to_string();
            assert!(message.contains(error), "{args:?}: {message}");
        }
    }
}
