use std::path::PathBuf;

use retrograd::{Device, Error, Result, TrainConfig, Trainer};

use crate::args::Args;

pub(crate) const FLAGS: &[&str] = &["--model", "--device"];

#[derive(Debug, PartialEq)]
struct InspectArgs {
    model: PathBuf,
    device: Device,
}

fn parse_inspect_args(args: &[String]) -> Result<InspectArgs> {
    let mut model = None;
    let mut device = Device::Auto;
    let mut model_seen = false;
    let mut device_seen = false;
    let mut args = Args::new("inspect", args);
    while let Some(flag) = args.next_arg() {
        if !FLAGS.contains(&flag) {
            return Err(args.unknown(flag));
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
            unknown => return Err(args.unknown(unknown)),
        }
    }
    Ok(InspectArgs {
        model: model.ok_or_else(|| Error::invalid("--model is required"))?,
        device,
    })
}

pub(crate) fn inspect_model(args: Vec<String>) -> Result<()> {
    let args = parse_inspect_args(&args)?;
    print!(
        "{}",
        Trainer::new(
            args.model,
            TrainConfig {
                device: args.device,
                ..TrainConfig::default()
            }
        )?
        .capability_report()?
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn inspect_parser_requires_model_and_rejects_duplicates() {
        assert_eq!(
            parse_inspect_args(&strings(&["--model", "model.gguf", "--device", "cpu"])).unwrap(),
            InspectArgs {
                model: PathBuf::from("model.gguf"),
                device: Device::Cpu,
            }
        );
        assert!(
            parse_inspect_args(&strings(&["--device", "cpu"]))
                .unwrap_err()
                .to_string()
                .contains("--model is required")
        );
        assert!(
            parse_inspect_args(&strings(&["--model"]))
                .unwrap_err()
                .to_string()
                .contains("missing value")
        );
    }
}
