//! A decision run: the supervised epoch loop over grouped steps, then the
//! temperatures fitted on the held-out questions and written into what the
//! run exports.
//!
//! The temperatures of a decision model only spread its probabilities - they
//! never change which option wins - but they are what the confidence a caller
//! thresholds on is computed from, and the ones a model shipped with were
//! fitted on other weights. A trained model is therefore exported with its
//! own: under `<arch>.decision.temperature.*` in a model GGUF, under
//! `adapter.decision.temperature.*` in an adapter (read by retrograd, ignored
//! by llama.cpp), and on request in a copy of the base model whose tensors are
//! untouched, which is what `llama-server --lora` reads them from.

use std::collections::BTreeMap;
use std::path::Path;

use retrograd_checkpoint as checkpoint;
use retrograd_config::{DecisionConfig, OutputKind, RunConfig};
use retrograd_core::{Error, Result, TrainMetrics};
use retrograd_engine::{
    ADAPTER_DECISION_PREFIX, Trainer, read_gguf_metadata, rewrite_gguf_metadata,
};
use retrograd_metrics::MetricValue;
use retrograd_training as training;
use serde_json::{Map, Value};

use crate::algorithm::{
    Context, EvalDetail, SupervisedEval, SupervisedEvaluator, run_supervised_epochs,
};
use crate::observer::RunObserver;
use crate::signature::checked_total_steps;

/// The relative gap between a fitted temperature and the base model's above
/// which an adapter export says that `llama-server --lora` would serve other
/// confidences than retrograd.
const TEMPERATURE_GAP_WARN: f32 = 0.05;

/// What a finished decision run writes besides its weights.
#[derive(Clone, Debug)]
pub(crate) struct DecisionExport {
    /// `general.architecture`, the prefix of the model's decision keys.
    architecture: String,
    /// The temperatures fitted on the held-out questions, when the run
    /// calibrated.
    fitted: Option<training::decision::FittedTemperatures>,
    /// The base model's `temperature.*` entries, as its metadata prints them.
    base_temperatures: BTreeMap<String, String>,
    /// The `systemone` template the run rendered with, when it replaced the
    /// model's.
    template: Option<String>,
    /// The head budget the run laid the questions out with, when it replaced
    /// the model's.
    max_head_tokens: Option<u32>,
}

impl DecisionExport {
    /// The entries a model GGUF (an export, or a copy of the base) carries:
    /// the fitted temperatures replacing the model's whole set, the effective
    /// template and head budget.
    fn model_entries(&self, source: &Path) -> Result<Map<String, Value>> {
        let prefix = format!("{}.decision.", self.architecture);
        let mut entries = Map::new();
        if let Some(fitted) = &self.fitted {
            // A bucket the new set lacks must fall back to its type, not to a
            // temperature fitted on other weights.
            for key in self.base_temperatures.keys() {
                entries.insert(format!("{prefix}{key}"), Value::Null);
            }
            for (key, temperature) in &fitted.temperatures {
                entries.insert(
                    format!("{prefix}temperature.{key}"),
                    Value::from(f64::from(*temperature)),
                );
            }
        }
        if let Some(max_head_tokens) = self.max_head_tokens {
            entries.insert(
                format!("{prefix}max_head_tokens"),
                Value::from(max_head_tokens),
            );
        }
        if let Some(template) = &self.template {
            entries.insert(
                format!(
                    "tokenizer.chat_template.{}",
                    retrograd_engine::DECISION_TEMPLATE
                ),
                Value::from(template.clone()),
            );
            // The list of named templates llama.cpp reads names from.
            let listed = read_gguf_metadata(source, "tokenizer.chat_templates")?;
            if let Some(text) = listed.get("tokenizer.chat_templates") {
                let mut names: Vec<String> = serde_json::from_str(text).unwrap_or_default();
                if !names
                    .iter()
                    .any(|name| name == retrograd_engine::DECISION_TEMPLATE)
                {
                    names.push(retrograd_engine::DECISION_TEMPLATE.to_owned());
                    entries.insert("tokenizer.chat_templates".into(), Value::from(names));
                }
            }
        }
        Ok(entries)
    }

    /// The entries an adapter carries: the fitted temperatures.
    fn adapter_entries(&self) -> Map<String, Value> {
        let mut entries = Map::new();
        if let Some(fitted) = &self.fitted {
            for (key, temperature) in &fitted.temperatures {
                entries.insert(
                    format!("{ADAPTER_DECISION_PREFIX}temperature.{key}"),
                    Value::from(f64::from(*temperature)),
                );
            }
        }
        entries
    }

    /// The fitted temperatures that the base model does not carry, or carries
    /// at a value more than [`TEMPERATURE_GAP_WARN`] away.
    fn departures_from_base(&self) -> Vec<String> {
        let Some(fitted) = &self.fitted else {
            return Vec::new();
        };
        fitted
            .temperatures
            .iter()
            .filter(|(key, fitted)| {
                let base = self
                    .base_temperatures
                    .get(&format!("temperature.{key}"))
                    .and_then(|text| text.trim().parse::<f32>().ok());
                base.is_none_or(|base| ((*fitted - base) / base).abs() > TEMPERATURE_GAP_WARN)
            })
            .map(|(key, fitted)| format!("{key}={fitted:.4}"))
            .collect()
    }

    /// Writes the metadata into what the run saved, and the copy of the base
    /// model `[decision].export_base` asks for.
    pub(crate) fn write(&self, config: &RunConfig, observer: &mut dyn RunObserver) -> Result<()> {
        let decision = match &config.algorithm {
            retrograd_config::Algorithm::Decision(decision) => decision,
            _ => return Ok(()),
        };
        let output = &config.output.path;
        match config.output.kind {
            OutputKind::Model => {
                let entries = self.model_entries(&config.model)?;
                if !entries.is_empty() {
                    rewrite_gguf_metadata(output, output, &entries)?;
                }
            }
            OutputKind::Adapter | OutputKind::Trainable => {
                let adapter = match config.output.kind {
                    OutputKind::Adapter => output.clone(),
                    _ if config.lora.is_some() => crate::trainable_adapter_sibling(output),
                    _ => {
                        if self.fitted.is_some() && decision.export_base.is_none() {
                            observer.info(
                                "decision: a trainable bundle carries no metadata, so the fitted \
                                 temperatures are not exported; use output.kind = 'model'",
                            );
                        }
                        return self.write_base(decision, config, observer);
                    }
                };
                let entries = self.adapter_entries();
                if !entries.is_empty() {
                    rewrite_gguf_metadata(&adapter, &adapter, &entries)?;
                }
                if self.template.is_some() || self.max_head_tokens.is_some() {
                    observer.info(
                        "decision: an adapter carries its temperatures, not the template or head \
                         budget this run laid its questions out with; serve it with the same \
                         [decision] options, or from the base copy decision.export_base writes",
                    );
                }
                let departures = self.departures_from_base();
                if !departures.is_empty() && decision.export_base.is_none() {
                    observer.info(&format!(
                        "decision: the fitted temperatures differ from the base model's ({}); \
                         retrograd serves the adapter with them, llama-server --lora with the \
                         base's - set decision.export_base for a base copy that carries them",
                        departures.join(", ")
                    ));
                }
            }
        }
        self.write_base(decision, config, observer)
    }

    fn write_base(
        &self,
        decision: &DecisionConfig,
        config: &RunConfig,
        observer: &mut dyn RunObserver,
    ) -> Result<()> {
        let Some(path) = &decision.export_base else {
            return Ok(());
        };
        let entries = self.model_entries(&config.model)?;
        rewrite_gguf_metadata(&config.model, path, &entries)?;
        observer.info(&format!(
            "decision: base model with the run's temperatures written to {}",
            path.display()
        ));
        Ok(())
    }
}

/// Trains `decision`, then fits the temperatures when it calibrates.
pub(crate) fn run_decision(
    trainer: &mut Trainer,
    decision: &DecisionConfig,
    ctx: &mut Context<'_>,
) -> Result<(TrainMetrics, DecisionExport)> {
    let config = ctx.config;
    let epochs = config.training.epochs;
    let prepared = training::decision::prepare(
        trainer,
        decision,
        &config.training,
        epochs,
        config
            .evaluation
            .as_ref()
            .map(|evaluation| (evaluation.data.as_path(), evaluation.max_examples)),
    )?;
    let info = prepared.info();
    ctx.observer.info(&format!(
        "decision: {} questions, {} held out, {} updates per epoch, readout {}, head budget {}",
        prepared.examples(),
        prepared.eval_examples(),
        prepared.steps_per_epoch(),
        info.decision_type,
        info.budget.map_or("none".to_string(), |budget| budget
            .max_head_tokens()
            .to_string()),
    ));
    let resume = ctx.controller.begin(
        trainer,
        checkpoint::Dataset {
            version: checkpoint::FORMAT_VERSION,
            path: decision.data.display().to_string(),
            fingerprint: checkpoint::fingerprint(prepared.fingerprint().as_bytes()),
            examples: prepared.examples() as u64,
            row_width: u64::from(config.training.n_ubatch),
            format: format!("decision_{:?}", decision.data_format).to_lowercase(),
            permutation: Vec::new(),
            cursor: 0,
        },
        checked_total_steps("decision", &[prepared.total_steps()])?,
    )?;
    ctx.datasets_prepared(
        prepared.examples(),
        prepared.tokens(),
        vec![MetricValue {
            name: "data/decision_eval_questions".into(),
            value: prepared.eval_examples() as f32,
        }],
    )?;
    let mut evaluate = (prepared.eval_examples() > 0).then_some(
        |trainer: &mut Trainer| -> Result<SupervisedEval> {
            let eval = training::decision::evaluate(trainer, &prepared)?;
            Ok(SupervisedEval {
                loss: eval.loss,
                examples: prepared.eval_examples() as u64,
                detail: EvalDetail::Decision(Box::new(eval)),
            })
        },
    );
    let metrics = run_supervised_epochs(
        trainer,
        ctx,
        evaluate
            .as_mut()
            .map(|evaluate| evaluate as &mut SupervisedEvaluator<'_>),
        epochs,
        prepared.steps_per_epoch(),
        |trainer, on_progress| {
            training::decision::run_resumed(trainer, &prepared, resume, on_progress)
        },
    )?;

    let fitted = if decision.calibrate {
        let eval = training::decision::evaluate(trainer, &prepared)?;
        let fitted = training::decision::fit_temperatures(&eval)?;
        let listed: Vec<String> = fitted
            .temperatures
            .iter()
            .map(|(key, temperature)| format!("{key}={temperature:.4}"))
            .collect();
        ctx.observer.info(&format!(
            "decision: temperatures fitted on {} held-out questions, NLL {:.4} -> {:.4}: {}",
            prepared.eval_examples(),
            fitted.nll_before,
            fitted.nll_after,
            listed.join(", ")
        ));
        Some(fitted)
    } else {
        None
    };
    let metadata = trainer.decision_metadata()?;
    let template = decision
        .template
        .as_ref()
        .map(|path| {
            std::fs::read_to_string(path).map_err(|error| {
                Error::config(format!(
                    "decision.template {} cannot be read: {error}",
                    path.display()
                ))
            })
        })
        .transpose()?;
    Ok((
        metrics,
        DecisionExport {
            architecture: metadata.architecture.unwrap_or_default(),
            fitted,
            base_temperatures: info.temperatures.clone(),
            template,
            max_head_tokens: decision.max_head_tokens,
        },
    ))
}
