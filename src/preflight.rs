use std::path::PathBuf;

use retrograd::config;
use retrograd::{Device, Error, LoraConfig, Result, TargetSet, TrainConfig, Trainer};

use crate::args::Args;

pub(crate) const FLAGS: &[&str] = &["--model", "--device", "--targets", "--strict"];

#[derive(Debug, PartialEq)]
struct PreflightArgs {
    model: PathBuf,
    device: Device,
    targets: TargetSet,
    /// Fail instead of just reporting when the active device hands nodes back to
    /// the CPU. Meant for CI and for the moment before a long run: a scheduler
    /// split costs a device/host round trip per occurrence per step, which is
    /// worth discovering now rather than after some hours of training.
    strict: bool,
}

fn parse_preflight_args(args: &[String]) -> Result<PreflightArgs> {
    let mut model = None;
    let mut device = Device::Auto;
    let mut targets = TargetSet::Auto;
    let mut model_seen = false;
    let mut device_seen = false;
    let mut targets_seen = false;
    let mut strict = false;
    let mut args = Args::new("preflight", args);
    while let Some(flag) = args.next_arg() {
        if !FLAGS.contains(&flag) {
            return Err(args.unknown(flag));
        }
        // --strict takes no value, so it is handled before the value lookup that
        // every other flag requires.
        if flag == "--strict" {
            args.once(&mut strict, "--strict")?;
            continue;
        }
        let value = args.value(flag)?;
        match flag {
            "--model" => {
                args.once(&mut model_seen, "--model")?;
                model = Some(PathBuf::from(value));
            }
            "--device" => {
                args.once(&mut device_seen, "--device")?;
                device = value.parse()?;
            }
            "--targets" => {
                args.once(&mut targets_seen, "--targets")?;
                let values: Vec<String> = value.split(',').map(str::to_string).collect();
                targets = config::parse_targets(&values)?;
            }
            unknown => return Err(args.unknown(unknown)),
        }
    }
    Ok(PreflightArgs {
        model: model.ok_or_else(|| Error::invalid("--model is required"))?,
        device,
        targets,
        strict,
    })
}

/// The `active_device_fallback_nodes` count from a preflight report, or `None` when
/// the report has no such line (a CPU-only run, or a graph that does not build).
fn fallback_node_count(report: &str) -> Option<u64> {
    report
        .lines()
        .find_map(|line| line.trim().strip_prefix("active_device_fallback_nodes:"))
        .and_then(|value| value.trim().parse().ok())
}

pub(crate) fn preflight_model(args: Vec<String>) -> Result<()> {
    let args = parse_preflight_args(&args)?;
    let mut trainer = Trainer::new(
        args.model,
        TrainConfig {
            device: args.device,
            ..TrainConfig::default()
        },
    )?;
    // The backward graph is defined by the trainable parameters, so the
    // preflight needs a (throwaway) LoRA adapter.
    let mut lora = LoraConfig::auto(2, 4.0);
    lora.targets = args.targets;
    trainer.create_lora(&lora)?;
    let report = trainer.train_preflight()?;
    print!("{report}");
    if args.strict {
        match fallback_node_count(&report) {
            Some(0) | None => {}
            Some(n) => {
                return Err(Error::invalid(format!(
                    "--strict: the active device hands {n} node(s) back to the CPU; \
                     see the devices section above for the ops and, when the cause is \
                     a quantization type, the undecodable_types line"
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn preflight_parser_parses_targets_and_rejects_bad_input() {
        let parsed = parse_preflight_args(&strings(&[
            "--model",
            "model.gguf",
            "--device",
            "gpu",
            "--targets",
            "q,v",
        ]))
        .unwrap();
        assert_eq!(parsed.model, PathBuf::from("model.gguf"));
        assert_eq!(parsed.device, Device::Gpu);
        assert_eq!(
            parsed.targets,
            TargetSet::Patterns(vec![
                "blk.*.attn_q.weight".into(),
                "blk.*.attn_v.weight".into(),
            ])
        );
        assert!(!parsed.strict);
        assert!(parse_preflight_args(&strings(&["--model", "m", "--targets", "bad"])).is_err());
    }

    #[test]
    fn preflight_parser_accepts_strict_as_a_valueless_flag() {
        // Trailing, so it also covers the value lookup not running for it.
        let parsed = parse_preflight_args(&strings(&["--model", "m.gguf", "--strict"])).unwrap();
        assert!(parsed.strict);
        let parsed = parse_preflight_args(&strings(&["--strict", "--model", "m.gguf"])).unwrap();
        assert!(parsed.strict);
        assert!(parse_preflight_args(&strings(&["--model", "m", "--strict", "--strict"])).is_err());
    }

    #[test]
    fn fallback_node_count_reads_the_report_line() {
        let report = "training preflight\n  devices:\n    MTL0: training graph ready\n  \
                      active_device_fallback_nodes: 38\n  status: ok\n";
        assert_eq!(fallback_node_count(report), Some(38));
        assert_eq!(
            fallback_node_count("  active_device_fallback_nodes: 0\n"),
            Some(0)
        );
        // A CPU-only run omits the line entirely; that must not read as a failure.
        assert_eq!(
            fallback_node_count("training preflight\n  status: ok\n"),
            None
        );
    }
}
