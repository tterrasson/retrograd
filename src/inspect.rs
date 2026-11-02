use std::fmt::Write as _;
use std::path::PathBuf;

use retrograd::decision::readout::{Head, HeadKind, PromptScope};
use retrograd::trainer::{DecisionInfo, TemplateSource, decision_metadata};
use retrograd::{Device, Engine, EngineConfig, Error, Result, TrainConfig, Trainer};

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
    let report = Trainer::new(
        &args.model,
        TrainConfig {
            device: args.device,
            ..TrainConfig::default()
        },
    )?
    .capability_report()?;
    print!("{report}");
    // Read from the metadata alone first: a model without a decision head is
    // not loaded a second time.
    if decision_metadata(&args.model)?.decision_type.is_some() {
        let engine = Engine::new(
            &args.model,
            &EngineConfig {
                device: args.device,
                ..EngineConfig::default()
            },
        )?;
        print!("{}", decision_section(engine.decision_info()));
    }
    Ok(())
}

/// The `decision:` section of the report, under `model capabilities`. A head
/// the engine cannot serve is reported, not fatal: inspecting is how one finds
/// out why.
fn decision_section(info: std::result::Result<DecisionInfo, retrograd::EngineError>) -> String {
    let mut out = String::from("  decision:\n");
    let info = match info {
        Ok(info) => info,
        Err(error) => {
            let _ = writeln!(out, "    status: unsupported ({error})");
            return out;
        }
    };
    let _ = writeln!(out, "    status: ok");
    let _ = writeln!(out, "    type: {}", info.decision_type);
    let readout = match &info.readout.head {
        Head::MarkerColumn { marker, sep, .. } => {
            format!("marker_column (marker token {marker}, separator token {sep})")
        }
        Head::D1Omni { marker } => {
            format!("marker_column (marker token {marker}, pieces cut to their budgets)")
        }
        Head::LabelLogits { labels, .. } => format!(
            "label_logits ({} {} labels)",
            labels.len(),
            match info.readout.kind.head {
                HeadKind::Letters => "letter",
                HeadKind::MetadataLabels => "metadata",
                _ => "code",
            }
        ),
        Head::LabelGroups => "label_groups (labels chosen per question)".to_owned(),
        Head::Pointer { marker } => format!("pointer (option end token {marker})"),
        Head::Joint if info.readout.kind.prompt == PromptScope::Joint => {
            "joint (every question in one prompt)".to_owned()
        }
        Head::Joint => "joint (one prompt per question)".to_owned(),
        Head::Pooled { n_classes } => format!("pooled ({n_classes} classes)"),
    };
    let _ = writeln!(out, "    readout: {readout}");
    let template = match info.template {
        TemplateSource::Model => "systemone (model)",
        TemplateSource::Override => "systemone (override)",
    };
    let _ = writeln!(out, "    template: {template}");
    if let Some(budget) = info.budget {
        let _ = writeln!(out, "    max_head_tokens: {}", budget.max_head_tokens());
    }
    let _ = writeln!(out, "    n_options_max: {}", info.n_options_max);
    if info.temperatures.is_empty() {
        let _ = writeln!(out, "    temperatures: none (1.0)");
    } else {
        let _ = writeln!(out, "    temperatures:");
        for (key, value) in &info.temperatures {
            let name = key.strip_prefix("temperature.").unwrap_or(key);
            let _ = writeln!(out, "      {name}: {value}");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn decision_section_names_the_readout_budget_and_temperatures() {
        use retrograd::decision::answer::Calibration;
        use retrograd::decision::layout::Budget;

        let info = DecisionInfo {
            decision_type: "laya".into(),
            readout: retrograd::decision::readout::Readout::new(
                retrograd::decision::readout::DecisionType::named("laya").unwrap(),
                Head::MarkerColumn {
                    marker: 4,
                    sep: 1,
                    marker_text: "<mask>".into(),
                },
            ),
            calibration: Calibration::default(),
            temperatures: [("temperature.choice".to_string(), "1.500000".to_string())].into(),
            budget: Some(Budget::new(256).unwrap()),
            model_max_head_tokens: Some(256),
            template: TemplateSource::Model,
            n_options_max: 255,
            pack: 1,
        };
        assert_eq!(
            decision_section(Ok(info)),
            "  decision:\n    status: ok\n    type: laya\n    readout: marker_column (marker token 4, separator token 1)\n    \
             template: systemone (model)\n    max_head_tokens: 256\n    n_options_max: 255\n    \
             temperatures:\n      choice: 1.500000\n"
        );
        let refused = decision_section(Err(retrograd::EngineError::invalid("no head")));
        assert_eq!(
            refused,
            "  decision:\n    status: unsupported (invalid argument: no head)\n"
        );
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
