//! Measure a model on an evaluation set, with and without an adapter.
//!
//! This module parses arguments, prepares data, runs the passes, and collects
//! distributions. [`report`] owns presentation; shared display primitives live
//! in `retrograd-cli-ui`.

mod report;

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use indicatif::HumanDuration;
use retrograd::config::{self, Algorithm};
use retrograd::dataset::{self, DataFormat, PreparedDataset};
use retrograd::decision::answer::Answer;
use retrograd::decision::dataset::{self as decision_dataset, Example};
use retrograd::decision::metrics::{Metrics, Outcome};
use retrograd::decision::request::{Question, SystemOneRequest};
use retrograd::trainer::{DecisionInfo, adapter_decision_metadata};
use retrograd::{Device, Engine, EngineConfig, Error, EvalMetrics, Result, TrainConfig, Trainer};

use retrograd_cli_ui::CliUi;

use crate::args::{Args, parse_value};

use report::{
    print_decision_bench_report, print_distill_bench_report, print_reward_bench_report,
    print_sft_bench_report,
};

pub(crate) const FLAGS: &[&str] = &[
    "--model",
    "--data",
    "--adapter",
    "--device",
    "--format",
    "--ctx",
    "--limit",
];

#[derive(Debug)]
pub(crate) struct BenchArgs {
    config: PathBuf,
    model: Option<PathBuf>,
    data: Option<PathBuf>,
    adapter: Option<PathBuf>,
    device: Option<Device>,
    format: Option<DataFormat>,
    ctx: Option<u32>,
    limit: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct ExampleBenchScore {
    logprob: f64,
    supervised_tokens: u64,
}

#[derive(Clone, Debug)]
struct BenchResult {
    metrics: EvalMetrics,
    examples: Vec<ExampleBenchScore>,
    elapsed: Duration,
    input_tokens: usize,
}

impl BenchResult {
    fn input_tokens_per_second(&self) -> f64 {
        self.input_tokens as f64 / self.elapsed.as_secs_f64()
    }

    fn scored_tokens_per_second(&self) -> f64 {
        self.metrics.supervised_tokens as f64 / self.elapsed.as_secs_f64()
    }

    fn logprob_distribution(&self) -> Distribution {
        Distribution::new(self.examples.iter().map(|example| example.logprob))
    }

    fn supervised_token_distribution(&self) -> Distribution {
        Distribution::new(
            self.examples
                .iter()
                .map(|example| example.supervised_tokens as f64),
        )
    }
}

/// One distillation bench pass: the three `DistillBench` figures over a held-out set.
#[derive(Clone, Debug)]
struct DistillBenchResult {
    bench: retrograd::training::distill::DistillBench,
    elapsed: Duration,
}

impl DistillBenchResult {
    fn divergence(&self) -> Distribution {
        Distribution::new(self.bench.teacher_kl.iter().map(|&kl| f64::from(kl)))
    }

    fn examples_per_second(&self) -> f64 {
        self.bench.teacher_kl.len() as f64 / self.elapsed.as_secs_f64()
    }
}

#[derive(Clone, Debug)]
struct RewardBenchResult {
    rewards: Vec<f64>,
    elapsed: Duration,
}

impl RewardBenchResult {
    fn distribution(&self) -> Distribution {
        Distribution::new(self.rewards.iter().copied())
    }

    fn examples_per_second(&self) -> f64 {
        self.rewards.len() as f64 / self.elapsed.as_secs_f64()
    }
}

/// One decision bench pass: the served probabilities of every held-out
/// question measured against its answer.
#[derive(Clone, Debug)]
struct DecisionBenchResult {
    metrics: Metrics,
    elapsed: Duration,
    input_tokens: usize,
}

impl DecisionBenchResult {
    fn questions_per_second(&self) -> f64 {
        self.metrics.overall.n as f64 / self.elapsed.as_secs_f64()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Distribution {
    mean: f64,
    low: f64,
    p10: f64,
    median: f64,
    p90: f64,
    high: f64,
    stddev: f64,
}

impl Distribution {
    fn new(values: impl IntoIterator<Item = f64>) -> Self {
        let mut values: Vec<f64> = values.into_iter().collect();
        debug_assert!(!values.is_empty());
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64;
        Self {
            mean,
            low: values[0],
            p10: percentile(&values, 0.10),
            median: percentile(&values, 0.50),
            p90: percentile(&values, 0.90),
            high: values[values.len() - 1],
            stddev: variance.sqrt(),
        }
    }
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    let position = quantile * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f64;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

pub(crate) fn bench_model(args: Vec<String>) -> Result<()> {
    let args = parse_bench_args(&args)?;
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
    configure_bench_batches(&mut run_config.training);
    let data_path = args
        .data
        .clone()
        .or_else(|| {
            run_config
                .evaluation
                .as_ref()
                .map(|evaluation| evaluation.data.clone())
        })
        .ok_or_else(|| {
            Error::invalid("bench requires [evaluation].data or an explicit --data path")
        })?;

    let ui = CliUi::new();
    ui.section("bench");
    ui.info(format!("config: {}", args.config.display()));
    ui.info(format!("model: {}", run_config.model.display()));
    ui.info(format!("dataset: {}", data_path.display()));
    ui.info(format!(
        "objective: {}",
        match &run_config.algorithm {
            Algorithm::Sft(_) => "sft loss",
            Algorithm::Ppo(_) | Algorithm::Grpo(_) => "rollout reward",
            Algorithm::Distill(_) => "teacher divergence",
            Algorithm::AgentGrpo(_) => "judge score",
            Algorithm::Preference(_) => "preference accuracy",
            Algorithm::Decision(_) => "decision accuracy and calibration",
        }
    ));
    if args.format.is_some() && !matches!(&run_config.algorithm, Algorithm::Sft(_)) {
        return Err(Error::invalid(
            "bench --format is only valid for an SFT objective",
        ));
    }

    let started = Instant::now();
    match &run_config.algorithm {
        Algorithm::Sft(_) => {
            // The SFT pass is forward-only - tokenize, render, score labels -
            // so it runs on the lightweight engine instead of a full trainer,
            // and scores one row at a time: a single `n_ctx` slot is the
            // whole pool it needs, not the run's rollout concurrency.
            let scoring = TrainConfig {
                generation_concurrency: 1,
                ..run_config.training.clone()
            };
            let mut engine = load_model(&ui, started, || {
                Engine::from_train_config(&run_config.model, &scoring)
            })?;
            let format = match args.format {
                Some(format) => format,
                None => DataFormat::infer(&data_path)?,
            };
            let eval_ctx = run_config.training.n_ctx as usize;
            let prepare_started = Instant::now();
            let prepare = ui.spinner("preparing dataset");
            let mut data =
                dataset::prepare(&engine, &data_path, format, eval_ctx).inspect_err(|_| {
                    ui.fail_spinner(prepare.clone(), "dataset preparation failed");
                })?;
            if let Some(limit) = args.limit {
                truncate_dataset(&mut data, limit);
            }
            ui.finish_spinner(
                prepare,
                format!(
                    "dataset prepared in {} (format={}, examples={}, supervised_tokens={}, ctx={})",
                    HumanDuration(prepare_started.elapsed()),
                    data_format_name(format),
                    data.examples,
                    data.supervised_tokens,
                    data.n_ctx,
                ),
            );
            let base = run_bench_pass(&mut engine, &data, "base", &ui)?;
            let adapted = if let Some(adapter) = &args.adapter {
                load_bench_adapter(&mut engine, adapter, &ui)?;
                Some(run_bench_pass(&mut engine, &data, "adapter", &ui)?)
            } else {
                None
            };
            print_sft_bench_report(&base, adapted.as_ref());
        }
        Algorithm::Ppo(ppo) => {
            let mut trainer = load_model(&ui, started, || {
                Trainer::new(&run_config.model, run_config.training.clone())
            })?;
            run_reward_bench(&mut trainer, &args, &ui, |trainer| {
                retrograd::training::ppo::benchmark_rewards(
                    trainer,
                    ppo,
                    &run_config.training,
                    &data_path,
                    args.limit,
                )
            })?
        }
        Algorithm::Grpo(grpo) => {
            let mut trainer = load_model(&ui, started, || {
                Trainer::new(&run_config.model, run_config.training.clone())
            })?;
            run_reward_bench(&mut trainer, &args, &ui, |trainer| {
                retrograd::training::grpo::benchmark_rewards(
                    trainer,
                    grpo,
                    &run_config.training,
                    &data_path,
                    args.limit,
                )
            })?
        }
        Algorithm::Distill(distill) => {
            // One teacher for the pass, and for both passes when an adapter is
            // supplied: the base and the adapted student are graded against the
            // *same* model, so the second load would be both a doubled memory
            // cost and a chance to compare two different targets.
            let mut trainer = load_model(&ui, started, || {
                Trainer::new(&run_config.model, run_config.training.clone())
            })?;
            let teacher = retrograd::training::distill::SharedTeacher::new();
            let base = run_distill_bench_pass("base", &ui, || {
                retrograd::training::distill::benchmark(
                    &mut trainer,
                    distill,
                    &run_config.training,
                    &teacher,
                    &data_path,
                    args.limit,
                )
            })?;
            let adapted = if let Some(adapter) = &args.adapter {
                load_bench_adapter(&mut trainer, adapter, &ui)?;
                Some(run_distill_bench_pass("adapter", &ui, || {
                    retrograd::training::distill::benchmark(
                        &mut trainer,
                        distill,
                        &run_config.training,
                        &teacher,
                        &data_path,
                        args.limit,
                    )
                })?)
            } else {
                None
            };
            print_distill_bench_report(&base, adapted.as_ref());
        }
        // `bench` measures a reward over a dataset of prompts. An agentic run
        // scores whole trajectories through a judge and a live environment,
        // which is a run, not a measurement pass.
        Algorithm::AgentGrpo(_) => {
            return Err(Error::invalid(
                "bench does not support run.algorithm = 'agent_grpo': an agentic score needs \
                 rollouts against a live environment, so run `train` instead",
            ));
        }
        // Served, not trained: the engine answers each question with the
        // model's temperatures, or the adapter's when it carries its own, so
        // the confidences measured are the ones a caller thresholds on.
        Algorithm::Decision(decision) => {
            let mut examples = decision_dataset::read_jsonl(&data_path, decision.data_format)?;
            if let Some(limit) = args.limit {
                examples.truncate(limit);
            }
            let mut engine = load_model(&ui, started, || {
                Ok(Engine::new(
                    &run_config.model,
                    &EngineConfig {
                        device: run_config.training.device,
                        ..EngineConfig::default()
                    },
                )?)
            })?;
            engine.set_chat_template_variables(
                &serde_json::Value::Object(decision.template_variables.clone()).to_string(),
            )?;
            let source = decision
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
            let mut info = if engine.decision_metadata()?.decision_type.is_none() {
                engine.configure_causal_decision()?.0
            } else {
                if let Some(source) = &source {
                    engine.set_decision_template(Some(source))?;
                }
                engine.decision_info()?
            };
            if let Some(source) = &source {
                engine.set_decision_template(Some(source))?;
            }
            examples = examples
                .into_iter()
                .map(|example| example.for_readout(&info.readout))
                .collect();
            if let Some(max_head_tokens) = decision.max_head_tokens {
                info = info.with_max_head_tokens(max_head_tokens as usize)?;
            }
            ui.info(format!("questions: {}", examples.len()));
            let base = run_decision_bench_pass(&mut engine, &info, &examples, "base", &ui)?;
            let adapted = if let Some(adapter) = &args.adapter {
                load_bench_adapter(&mut engine, adapter, &ui)?;
                let info = info
                    .clone()
                    .with_adapter_temperatures(&adapter_decision_metadata(adapter)?)?;
                Some(run_decision_bench_pass(
                    &mut engine,
                    &info,
                    &examples,
                    "adapter",
                    &ui,
                )?)
            } else {
                None
            };
            print_decision_bench_report(&base, adapted.as_ref());
        }
        Algorithm::Preference(_) => {
            return Err(Error::invalid(
                "bench does not support run.algorithm = 'preference' yet: set \
                 [evaluation].data to a preference file, and the run reports its accuracy \
                 at every evaluation",
            ));
        }
    };
    Ok(())
}

/// The probability the answer gives each option of `question`, in the
/// question's order.
fn answer_probabilities(answer: &Answer, question: &Question) -> Result<Vec<f64>> {
    match answer {
        Answer::Choice { probabilities, .. } | Answer::Score { probabilities, .. } => question
            .options
            .iter()
            .map(|option| {
                probabilities.get(&option.key).copied().ok_or_else(|| {
                    Error::runtime(format!(
                        "the answer has no probability for '{}'",
                        option.key
                    ))
                })
            })
            .collect(),
        Answer::Noul { noul } => Ok(question
            .options
            .iter()
            .map(|option| {
                if option.key == "true" {
                    *noul
                } else {
                    1.0 - noul
                }
            })
            .collect()),
    }
}

/// Asks every question of `examples`, one request each, and measures the
/// answers against the targets.
fn run_decision_bench_pass(
    engine: &mut Engine,
    info: &DecisionInfo,
    examples: &[Example],
    label: &str,
    ui: &CliUi,
) -> Result<DecisionBenchResult> {
    let started = Instant::now();
    let progress = ui.progress_steps(examples.len() as u64, label);
    let mut outcomes = Vec::with_capacity(examples.len());
    let mut input_tokens = 0_usize;
    for example in examples {
        let request = SystemOneRequest::single(
            example.state.clone(),
            example.question_id.clone(),
            example.question.clone(),
        );
        let decision = engine.decide(info, &request)?;
        input_tokens += decision.input_tokens;
        let answer = decision
            .answers
            .get(&example.question_id)
            .ok_or_else(|| Error::runtime("the decision returned no answer"))?;
        let probabilities = answer_probabilities(answer, &example.question)?;
        outcomes.push(Outcome::new(
            example.question.kind,
            probabilities,
            example.target.distribution(example.question.options.len()),
        )?);
        progress.inc(1);
    }
    progress.finish_and_clear();
    Ok(DecisionBenchResult {
        metrics: Metrics::of(&outcomes),
        elapsed: started.elapsed(),
        input_tokens,
    })
}

/// Loads the model for a bench pass under the standard spinner, whichever
/// handle the pass runs on.
fn load_model<T>(ui: &CliUi, started: Instant, load: impl FnOnce() -> Result<T>) -> Result<T> {
    let spinner = ui.spinner("loading model");
    let model = load().inspect_err(|_| {
        ui.fail_spinner(spinner.clone(), "model loading failed");
    })?;
    ui.finish_spinner(
        spinner,
        format!("model loaded in {}", HumanDuration(started.elapsed())),
    );
    Ok(model)
}

fn run_reward_bench(
    trainer: &mut Trainer,
    args: &BenchArgs,
    ui: &CliUi,
    mut evaluate: impl FnMut(&mut Trainer) -> Result<Vec<f32>>,
) -> Result<()> {
    let base = run_reward_bench_pass("base", ui, || evaluate(trainer))?;
    let adapted = if let Some(adapter) = &args.adapter {
        load_bench_adapter(trainer, adapter, ui)?;
        Some(run_reward_bench_pass("adapter", ui, || evaluate(trainer))?)
    } else {
        None
    };
    print_reward_bench_report(&base, adapted.as_ref());
    Ok(())
}

fn parse_bench_args(args: &[String]) -> Result<BenchArgs> {
    let mut config = None;
    let mut model = None;
    let mut data = None;
    let mut adapter = None;
    let mut device = None;
    let mut format = None;
    let mut ctx = None;
    let mut limit = None;
    let mut model_seen = false;
    let mut adapter_seen = false;
    let mut device_seen = false;
    let mut format_seen = false;
    let mut ctx_seen = false;
    let mut limit_seen = false;
    let mut args = Args::new("bench", args);

    while let Some(flag) = args.next_arg() {
        if !flag.starts_with('-') {
            if config.replace(PathBuf::from(flag)).is_some() {
                return Err(Error::invalid("bench accepts exactly one config TOML path"));
            }
            continue;
        }
        if !FLAGS.contains(&flag) {
            return Err(args.unknown(flag));
        }
        let value = args.value(flag)?;
        match flag {
            "--model" => {
                args.once(&mut model_seen, "--model")?;
                model = Some(PathBuf::from(value));
            }
            "--data" => {
                if data.replace(PathBuf::from(value)).is_some() {
                    return Err(Error::invalid(
                        "bench accepts exactly one evaluation dataset",
                    ));
                }
            }
            "--adapter" => {
                args.once(&mut adapter_seen, "--adapter")?;
                adapter = Some(PathBuf::from(value));
            }
            "--device" => {
                args.once(&mut device_seen, "--device")?;
                device = Some(value.parse()?);
            }
            "--format" => {
                args.once(&mut format_seen, "--format")?;
                format = match value.trim().to_ascii_lowercase().as_str() {
                    "auto" => None,
                    "text" | "txt" => Some(DataFormat::Text),
                    "jsonl" | "chat" | "chat-jsonl" => Some(DataFormat::ChatJsonl),
                    _ => {
                        return Err(Error::invalid(
                            "bench --format must be auto, text, or jsonl",
                        ));
                    }
                };
            }
            "--ctx" => {
                args.once(&mut ctx_seen, "--ctx")?;
                let parsed: u32 = parse_value("--ctx", value, "an integer")?;
                if parsed == 0 {
                    return Err(Error::invalid("bench --ctx must be greater than zero"));
                }
                ctx = Some(parsed);
            }
            "--limit" => {
                args.once(&mut limit_seen, "--limit")?;
                let parsed: usize = parse_value("--limit", value, "an integer")?;
                if parsed == 0 {
                    return Err(Error::invalid("bench --limit must be greater than zero"));
                }
                limit = Some(parsed);
            }
            unknown => return Err(args.unknown(unknown)),
        }
    }

    Ok(BenchArgs {
        config: config.ok_or_else(|| Error::invalid("bench requires a config TOML path"))?,
        model,
        data,
        adapter,
        device,
        format,
        ctx,
        limit,
    })
}

fn configure_bench_batches(config: &mut TrainConfig) {
    config.n_batch = greatest_divisor(config.n_ctx, config.n_ctx.min(128));
    config.n_ubatch = greatest_divisor(config.n_batch, config.n_batch.min(32));
}

/// What `load_bench_adapter` needs from a loaded model: attach a LoRA adapter
/// from a GGUF file. Implemented by both handles a bench pass can run on.
pub(crate) trait AdapterLoader {
    fn load_adapter(&mut self, adapter: &Path) -> Result<()>;
}

impl AdapterLoader for Engine {
    fn load_adapter(&mut self, adapter: &Path) -> Result<()> {
        self.load_lora(adapter).map_err(Error::from)
    }
}

impl AdapterLoader for Trainer {
    fn load_adapter(&mut self, adapter: &Path) -> Result<()> {
        self.load_lora(adapter)
    }
}

pub(crate) fn load_bench_adapter(
    model: &mut impl AdapterLoader,
    adapter: &Path,
    ui: &CliUi,
) -> Result<()> {
    let started = Instant::now();
    let load = ui.spinner("loading adapter");
    model.load_adapter(adapter).inspect_err(|_| {
        ui.fail_spinner(load.clone(), "adapter loading failed");
    })?;
    ui.finish_spinner(
        load,
        format!(
            "adapter loaded in {}: {}",
            HumanDuration(started.elapsed()),
            adapter.display()
        ),
    );
    Ok(())
}

fn greatest_divisor(value: u32, maximum: u32) -> u32 {
    (1..=maximum)
        .rev()
        .find(|candidate| value.is_multiple_of(*candidate))
        .unwrap_or(1)
}

fn truncate_dataset(data: &mut PreparedDataset, limit: usize) {
    let examples = data.examples.min(limit);
    let values = examples * data.n_ctx;
    data.tokens.truncate(values);
    data.labels.truncate(values);
    data.examples = examples;
    data.supervised_tokens = data
        .labels
        .iter()
        .filter(|&&label| label != dataset::IGNORE_LABEL)
        .count();
}

fn bench_row(data: &PreparedDataset, row: usize) -> PreparedDataset {
    let start = row * data.n_ctx;
    let end = start + data.n_ctx;
    PreparedDataset {
        n_ctx: data.n_ctx,
        tokens: data.tokens[start..end].to_vec(),
        labels: data.labels[start..end].to_vec(),
        examples: 1,
        supervised_tokens: data.labels[start..end]
            .iter()
            .filter(|&&label| label != dataset::IGNORE_LABEL)
            .count(),
    }
}

fn run_bench_pass(
    engine: &mut Engine,
    data: &PreparedDataset,
    name: &str,
    ui: &CliUi,
) -> Result<BenchResult> {
    // Build/cache the decode graph outside the timed pass so base and adapter
    // throughput are comparable even for very small datasets.
    engine.eval_sft(&bench_row(data, 0))?;
    let spinner = ui.spinner(format!("evaluating {name}"));
    let started = Instant::now();
    let mut metrics = EvalMetrics::default();
    let mut examples = Vec::with_capacity(data.examples);
    for row in 0..data.examples {
        let row_metrics = engine.eval_sft(&bench_row(data, row)).inspect_err(|_| {
            ui.fail_spinner(spinner.clone(), format!("{name} evaluation failed"));
        })?;
        metrics.negative_log_likelihood += row_metrics.negative_log_likelihood;
        metrics.supervised_tokens += row_metrics.supervised_tokens;
        examples.push(ExampleBenchScore {
            logprob: -row_metrics.loss(),
            supervised_tokens: row_metrics.supervised_tokens,
        });
    }
    let elapsed = started.elapsed();
    ui.finish_spinner(
        spinner,
        format!("{name} evaluated in {}", HumanDuration(elapsed)),
    );
    Ok(BenchResult {
        metrics,
        examples,
        elapsed,
        input_tokens: data.examples * data.n_ctx,
    })
}

fn run_reward_bench_pass(
    name: &str,
    ui: &CliUi,
    evaluate: impl FnOnce() -> Result<Vec<f32>>,
) -> Result<RewardBenchResult> {
    let spinner = ui.spinner(format!("evaluating {name}"));
    let started = Instant::now();
    let rewards = evaluate().inspect_err(|_| {
        ui.fail_spinner(spinner.clone(), format!("{name} evaluation failed"));
    })?;
    let elapsed = started.elapsed();
    ui.finish_spinner(
        spinner,
        format!("{name} evaluated in {}", HumanDuration(elapsed)),
    );
    Ok(RewardBenchResult {
        rewards: rewards.into_iter().map(f64::from).collect(),
        elapsed,
    })
}

fn run_distill_bench_pass(
    name: &str,
    ui: &CliUi,
    measure: impl FnOnce() -> Result<retrograd::training::distill::DistillBench>,
) -> Result<DistillBenchResult> {
    let spinner = ui.spinner(format!("measuring {name} against the teacher"));
    let started = Instant::now();
    let bench = measure().inspect_err(|_| {
        ui.fail_spinner(spinner.clone(), format!("{name} measurement failed"));
    })?;
    let elapsed = started.elapsed();
    ui.finish_spinner(
        spinner,
        format!("{name} measured in {}", HumanDuration(elapsed)),
    );
    Ok(DistillBenchResult { bench, elapsed })
}

pub(crate) fn data_format_name(format: DataFormat) -> &'static str {
    match format {
        DataFormat::Text => "text",
        DataFormat::ChatJsonl => "jsonl",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn distribution_reports_interpolated_percentiles_and_population_stddev() {
        let distribution = Distribution::new([1.0, 2.0, 3.0, 4.0, 5.0]);

        assert_eq!(distribution.low, 1.0);
        assert_eq!(distribution.high, 5.0);
        assert_eq!(distribution.mean, 3.0);
        assert_eq!(distribution.p10, 1.4);
        assert_eq!(distribution.median, 3.0);
        assert_eq!(distribution.p90, 4.6);
        assert!((distribution.stddev - 2.0_f64.sqrt()).abs() < 1.0e-12);
    }

    #[test]
    fn distribution_handles_a_single_example() {
        let distribution = Distribution::new([-2.5]);

        assert_eq!(
            distribution,
            Distribution {
                mean: -2.5,
                low: -2.5,
                p10: -2.5,
                median: -2.5,
                p90: -2.5,
                high: -2.5,
                stddev: 0.0,
            }
        );
    }

    #[test]
    fn bench_parser_covers_aliases_defaults_and_numeric_boundaries() {
        let parsed = parse_bench_args(&strings(&[
            "run.toml",
            "--model",
            "base.gguf",
            "--data",
            "eval.jsonl",
            "--adapter",
            "adapter.gguf",
            "--device",
            "cpu",
            "--format",
            "chat-jsonl",
            "--ctx",
            "64",
            "--limit",
            "3",
        ]))
        .unwrap();
        assert_eq!(parsed.config, PathBuf::from("run.toml"));
        assert_eq!(parsed.model, Some(PathBuf::from("base.gguf")));
        assert_eq!(parsed.data, Some(PathBuf::from("eval.jsonl")));
        assert_eq!(parsed.device, Some(Device::Cpu));
        assert_eq!(parsed.format, Some(DataFormat::ChatJsonl));
        assert_eq!(parsed.ctx, Some(64));
        assert_eq!(parsed.limit, Some(3));

        for (args, expected) in [
            (vec!["run.toml", "--ctx", "0"], "must be greater than zero"),
            (
                vec!["run.toml", "--limit", "0"],
                "must be greater than zero",
            ),
            (vec!["run.toml", "--format", "yaml"], "auto, text, or jsonl"),
            (vec!["run.toml", "--data"], "missing value"),
            (
                vec!["run.toml", "--model", "a", "--model", "b"],
                "--model at most once",
            ),
            (
                vec!["run.toml", "--data", "a", "--data", "b"],
                "exactly one evaluation dataset",
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
                vec!["run.toml", "--format", "text", "--format", "text"],
                "--format at most once",
            ),
            (
                vec!["run.toml", "--ctx", "8", "--ctx", "16"],
                "--ctx at most once",
            ),
            (
                vec!["run.toml", "--limit", "1", "--limit", "2"],
                "--limit at most once",
            ),
        ] {
            let error = parse_bench_args(&strings(&args)).unwrap_err().to_string();
            assert!(error.contains(expected), "{args:?} -> {error}");
        }
    }

    #[test]
    fn data_format_name_and_greatest_divisor_are_stable() {
        assert_eq!(data_format_name(DataFormat::Text), "text");
        assert_eq!(greatest_divisor(64, 32), 32);
        assert_eq!(greatest_divisor(63, 32), 21);
    }
}
