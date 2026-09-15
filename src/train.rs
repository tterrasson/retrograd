use std::path::PathBuf;

use indicatif::{HumanDuration, ProgressBar};
use retrograd::config::{self, Algorithm, RunConfig};
use retrograd::run::{
    self, EvaluationReport, LoopPlan, RolloutEpoch, RunObserver, RunOutcome, SftEpoch, SftStep,
};
use retrograd::{Device, Error, Result};

use retrograd_cli_ui::{
    Better, CliUi, Column, DIM, EVAL_ROW, StreamTable, budget_ansi, cell, fmt_signed,
    magnitude_ansi, throughput_cell, trend_ansi,
};

use crate::args::Args;

/// `best 0.4531 ↑ (saved best.gguf)` or `best 0.4531, stale 3/10`: what an
/// evaluation did to the early-stopping state, appended to its report row.
///
/// It reads an `EvalOutcome`, so it stays here rather than in
/// `retrograd-cli-ui`, which is deliberately free of domain types
fn eval_status(outcome: &run::EvalOutcome, precision: usize) -> String {
    let best = format!("best {:.precision$}", outcome.best);
    if outcome.improved {
        match &outcome.saved {
            Some(path) => {
                let name = path
                    .file_name()
                    .map(|name| name.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.display().to_string());
                format!("{best} ↑ (saved {name})")
            }
            None => format!("{best} ↑"),
        }
    } else {
        match outcome.patience {
            Some(patience) => format!("{best}, stale {}/{patience}", outcome.stale),
            None => format!("{best}, stale {}", outcome.stale),
        }
    }
}

/// Mid-epoch rows printed per SFT epoch. Few enough that an epoch stays one
/// readable block, often enough that a long epoch shows a loss trend rather
/// than a single figure at its end.
const ROWS_PER_EPOCH: u64 = 8;
pub(crate) const FLAGS: &[&str] = &["--resume", "--model", "--device"];

/// Decimal width of a counter, so a column is sized for the run it reports.
fn digits(value: u64) -> usize {
    value.max(1).ilog10() as usize + 1
}

/// Parses `train <config.toml> [--resume [checkpoint.state]] [--model <path>]
/// [--device <auto|cpu|gpu>]`. `--resume` wires the resulting checkpoint into
/// the loaded config so a stopped run can be restarted without editing its
/// TOML; `--model`/`--device` override the matching `[model]` fields so a
/// model can be swapped without editing the TOML either.
pub(crate) fn parse_train_args(args: &[String]) -> Result<RunConfig> {
    let mut path = None;
    let mut resume: Option<Option<PathBuf>> = None;
    let mut model = None;
    let mut device = None;
    let mut resume_seen = false;
    let mut model_seen = false;
    let mut device_seen = false;
    let mut args = Args::new("train", args);
    while let Some(flag) = args.next_arg() {
        if flag.starts_with('-') && !FLAGS.contains(&flag) {
            return Err(args.unknown(flag));
        }
        match flag {
            "--resume" => {
                args.once(&mut resume_seen, "--resume")?;
                resume = Some(args.optional_value().map(PathBuf::from));
            }
            "--model" => {
                let value = args.value(flag)?;
                args.once(&mut model_seen, "--model")?;
                model = Some(PathBuf::from(value));
            }
            "--device" => {
                let value = args.value(flag)?;
                args.once(&mut device_seen, "--device")?;
                device = Some(value.parse::<Device>()?);
            }
            unknown if unknown.starts_with('-') => return Err(args.unknown(unknown)),
            positional => {
                if path.replace(PathBuf::from(positional)).is_some() {
                    return Err(Error::invalid("train accepts exactly one config TOML path"));
                }
            }
        }
    }
    let path = path.ok_or_else(|| Error::invalid("train requires a config TOML path"))?;
    let mut config = config::load_with(
        path,
        config::ModelOverride {
            path: model,
            device,
        },
    )?;
    if let Some(explicit) = resume {
        run::apply_resume_override(&mut config, explicit)?;
    }
    Ok(config)
}

pub(crate) fn train(config: RunConfig) -> Result<()> {
    let ui = CliUi::new();
    ui.section("train");
    ui.info(format!("model: {}", config.model.display()));
    ui.info(format!("output: {}", config.lora.output.display()));
    let mut observer = TerminalObserver::new(ui);
    let outcome = run::execute(&config, &mut observer)?;
    print_done(&outcome, &config);
    Ok(())
}

fn print_done(outcome: &RunOutcome, config: &RunConfig) {
    let metrics = &outcome.metrics;
    match &config.algorithm {
        Algorithm::Sft(_) => {
            let eval_loss = if metrics.eval_loss.is_finite() {
                format!(" eval_loss={:.6}", metrics.eval_loss)
            } else {
                String::new()
            };
            println!(
                "done epoch={} step={} train_loss={:.6}{eval_loss} tok/s={:.1} out={}",
                metrics.epoch,
                metrics.global_step,
                metrics.train_loss,
                metrics.tokens_per_second,
                config.lora.output.display()
            );
        }
        Algorithm::Ppo(ppo) => {
            print_rollout_done(metrics, config, "ppo", ppo.updates, ppo.ppo_epochs)
        }
        Algorithm::Grpo(grpo) => {
            print_rollout_done(metrics, config, "grpo", grpo.updates, grpo.grpo_epochs)
        }
        // The offline mode counts epochs over a corpus, not updates over
        // rollouts, and printing "update=1/0" would be the wrong sentence rather
        // than a missing one.
        Algorithm::Distill(distill) if !distill.mode.is_rollout() => {
            let epochs = distill
                .mode
                .offline()
                .map(|offline| offline.epochs)
                .unwrap_or(1);
            println!(
                "done epoch={}/{epochs} distill_mode=topk_offline step={} train_loss={:.6} tok/s={:.1} out={}",
                metrics.epoch,
                metrics.global_step,
                metrics.train_loss,
                metrics.tokens_per_second,
                config.lora.output.display()
            );
        }
        Algorithm::Distill(distill) => print_rollout_done(
            metrics,
            config,
            "distill",
            distill.updates,
            distill.distill_epochs,
        ),
        Algorithm::AgentGrpo(agent) => print_rollout_done(
            metrics,
            config,
            "agent_grpo",
            agent.config.updates,
            agent.config.epochs,
        ),
    }
}

fn print_rollout_done(
    metrics: &retrograd::TrainMetrics,
    config: &RunConfig,
    algorithm: &str,
    updates: u32,
    epochs_per_update: u32,
) {
    println!(
        "done update={}/{updates} {algorithm}_epochs_per_update={epochs_per_update} step={} train_loss={:.6} tok/s={:.1} out={}",
        metrics.epoch,
        metrics.global_step,
        metrics.train_loss,
        metrics.tokens_per_second,
        config.lora.output.display()
    );
}

/// CLI display state: a model-loading spinner followed by a streaming table and
/// progress bar. The state is presentation-only and is not part of the run.
struct TerminalObserver {
    ui: CliUi,
    spinner: Option<ProgressBar>,
    /// Live progress bar and table of the current loop. `None` outside a loop:
    /// the model-loading phase prints through the spinner instead.
    loop_state: Option<LoopState>,
}

struct LoopState {
    bar: ProgressBar,
    table: StreamTable,
    has_eval: bool,
    /// Steps between two mid-epoch rows, and the step the last one was printed
    /// at. An SFT epoch is hundreds to thousands of optimizer steps, so the
    /// table shows a fixed number of them per epoch instead of every one.
    row_interval: u64,
    last_row_step: u64,
    /// Previous epoch's training loss, for the trend colour of the loss cell.
    previous_loss: f64,
    /// Reward of the last *freshly sampled* update, for the Δreward cell.
    previous_reward: f64,
    /// Last evaluation reward, shown on the bar between evaluations.
    last_eval_reward: f64,
    /// Reward and loss of the epoch currently being reported, so the message
    /// rewritten after an evaluation carries the same figures as before it.
    current_reward: f32,
    current_train_loss: f32,
}

impl TerminalObserver {
    fn new(ui: CliUi) -> Self {
        Self {
            ui,
            spinner: None,
            loop_state: None,
        }
    }

    fn state(&mut self) -> &mut LoopState {
        self.loop_state
            .as_mut()
            .expect("a loop event arrives between loop_started and loop_finished")
    }
}

impl RunObserver for TerminalObserver {
    /// Inside a loop an `info` line is a table row: prefixing it with `info`
    /// and printing it at the left margin would leave the frame open around it.
    fn info(&mut self, message: &str) {
        match &self.loop_state {
            Some(state) => {
                let row = state.table.span_row(message, DIM);
                let bar = state.bar.clone();
                self.ui.progress_table(&bar, row);
            }
            None => self.ui.info(message),
        }
    }

    fn diagnostic(&mut self, title: &str, body: &str) {
        match &self.loop_state {
            Some(state) => self.ui.progress_diagnostic(&state.bar, title, body),
            None => self.ui.diagnostic(title, body),
        }
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

    fn loop_started(&mut self, plan: &LoopPlan) {
        let (bar, table, has_eval, row_interval) = match plan {
            LoopPlan::Sft {
                epochs,
                has_eval,
                steps_per_epoch,
            } => {
                let total_steps = *steps_per_epoch * *epochs as u64;
                self.ui.info(format!(
                    "sft: epochs={epochs}, steps_per_epoch={steps_per_epoch}, steps={total_steps}"
                ));
                let table = StreamTable::new(
                    self.ui.color,
                    [
                        // Wide enough for `3/3  42%`, the mid-epoch form of the
                        // cell, so an intermediate row keeps the frame.
                        Column::new("epoch", 2 * digits(*epochs as u64) + 6),
                        Column::new("step", digits(total_steps).max(6)),
                        Column::new("train_loss", 10),
                        Column::new("eval_loss", 10),
                        Column::new("lr", 9),
                        Column::new("tok/s", 7),
                    ]
                    .into_iter()
                    .filter(|column| *has_eval || column.title != "eval_loss")
                    .collect(),
                );
                (
                    // Optimizer steps, not epochs: an epoch-grained bar spends
                    // the whole first epoch at 0/3 with no elapsed-to-go
                    // estimate, which is the report this display exists to give.
                    self.ui.progress_steps(total_steps, "steps"),
                    table,
                    *has_eval,
                    (steps_per_epoch / ROWS_PER_EPOCH).max(1),
                )
            }
            LoopPlan::Rollout {
                algorithm,
                updates,
                epochs_per_update,
                total_epochs,
            } => {
                self.ui.info(format!(
                    "{algorithm}: updates={updates}, {algorithm}_epochs_per_update={epochs_per_update}, optimizer_epochs={total_epochs}"
                ));
                let table = StreamTable::new(
                    self.ui.color,
                    vec![
                        Column::new("update", 7),
                        Column::new("epoch", 5),
                        Column::new("step", 6),
                        Column::new("loss", 10),
                        Column::new("reward", 8),
                        Column::new("Δreward", 8),
                        Column::new("kl", 8),
                        Column::new("clip", 6),
                        Column::new("lr", 9),
                        Column::new("tok/s", 6),
                    ],
                );
                (
                    self.ui.progress_steps(*total_epochs, "optimizer epochs"),
                    table,
                    true,
                    // A rollout algorithm already reports one row per optimizer
                    // epoch, so it has no mid-iteration rows to throttle.
                    u64::MAX,
                )
            }
        };
        self.ui.progress_table(&bar, table.open());
        self.loop_state = Some(LoopState {
            bar,
            table,
            has_eval,
            row_interval,
            last_row_step: 0,
            previous_loss: f64::NAN,
            previous_reward: f64::NAN,
            last_eval_reward: f64::NAN,
            current_reward: f32::NAN,
            current_train_loss: f32::NAN,
        });
    }

    fn sft_step(&mut self, step: &SftStep) {
        let state = self.state();
        state.bar.set_position(step.global_step);
        state.bar.set_message(format!(
            "epoch {}/{} loss={:.6} {:.0} tok/s",
            step.epoch, step.total_epochs, step.train_loss, step.tokens_per_second
        ));
        // The last step of an epoch is followed immediately by the epoch's own
        // row, which reports the same figures with the evaluation added.
        let due = step.global_step >= state.last_row_step.saturating_add(state.row_interval);
        if !due || step.epoch_step >= step.steps_per_epoch {
            return;
        }
        state.last_row_step = step.global_step;
        let train_loss = step.train_loss as f64;
        let percent = 100 * step.epoch_step / step.steps_per_epoch.max(1);
        let mut cells = vec![
            cell(
                format!("{}/{} {percent:>3}%", step.epoch, step.total_epochs),
                DIM,
            ),
            cell(step.global_step.to_string(), DIM),
            cell(
                format!("{train_loss:.6}"),
                trend_ansi(
                    train_loss - state.previous_loss,
                    state.previous_loss,
                    Better::Lower,
                ),
            ),
        ];
        if state.has_eval {
            // Evaluation only runs at an epoch boundary, so the column has
            // nothing to say here - an em dash would read as a measured value.
            cells.push(cell("", None));
        }
        cells.push(cell(format!("{:.2e}", step.learning_rate), DIM));
        cells.push(throughput_cell(step.tokens_per_second as f64));
        let row = state.table.row(&cells);
        let bar = state.bar.clone();
        self.ui.progress_table(&bar, row);
    }

    fn sft_epoch(&mut self, epoch: &SftEpoch) {
        let state = self.state();
        state.bar.set_position(epoch.global_step);
        state.last_row_step = epoch.global_step;
        let eval_loss = if epoch.eval_loss.is_finite() {
            format!(" eval_loss={:.6}", epoch.eval_loss)
        } else {
            String::new()
        };
        state.bar.set_message(format!(
            "epoch {}/{} done train_loss={:.6}{eval_loss}",
            epoch.epoch, epoch.total_epochs, epoch.train_loss
        ));
        let train_loss = epoch.train_loss as f64;
        let mut cells = vec![
            cell(format!("{}/{}", epoch.epoch, epoch.total_epochs), None),
            cell(epoch.global_step.to_string(), None),
            cell(
                format!("{train_loss:.6}"),
                trend_ansi(
                    train_loss - state.previous_loss,
                    state.previous_loss,
                    Better::Lower,
                ),
            ),
        ];
        if state.has_eval {
            let value = epoch.eval_loss as f64;
            cells.push(cell(
                if value.is_finite() {
                    format!("{value:.6}")
                } else {
                    "-".into()
                },
                None,
            ));
        }
        cells.push(cell(format!("{:.2e}", epoch.learning_rate), DIM));
        cells.push(throughput_cell(epoch.tokens_per_second as f64));
        let row = state.table.row(&cells);
        state.previous_loss = train_loss;
        let bar = state.bar.clone();
        self.ui.progress_table(&bar, row);
    }

    fn rollout_epoch(&mut self, epoch: &RolloutEpoch) {
        let state = self.state();
        state.bar.set_position(epoch.absolute_epoch);
        state.current_reward = epoch.reward;
        state.current_train_loss = epoch.train_loss;
        let reward = epoch.reward;
        let eval_message = if state.last_eval_reward.is_finite() {
            format!(" eval={:.4}", state.last_eval_reward)
        } else {
            String::new()
        };
        state.bar.set_message(format!(
            "reward={reward:.4}{eval_message} loss={:.6}",
            epoch.train_loss
        ));
        // Only the first optimizer epoch of an update sees a freshly sampled
        // batch, so the reward delta is meaningful once per update.
        let reward = reward as f64;
        let reward_delta = reward - state.previous_reward;
        let show_delta = epoch.policy_epoch == 1 && reward_delta.is_finite();
        let train_loss = epoch.train_loss as f64;
        let kl = epoch.kl;
        let clip_fraction = epoch.clip_fraction;
        let row = state.table.row(&[
            cell(format!("{}/{}", epoch.update, epoch.updates), None),
            cell(
                format!("{}/{}", epoch.policy_epoch, epoch.epochs_per_update),
                DIM,
            ),
            cell(epoch.global_step.to_string(), None),
            // A negative policy-gradient loss is the surrogate improving.
            cell(
                fmt_signed(train_loss, 6),
                magnitude_ansi(train_loss, [1.0e-6, 1.0e-3, 1.0e-2], Better::Lower),
            ),
            cell(format!("{reward:.4}"), None),
            cell(
                if show_delta {
                    fmt_signed(reward_delta, 4)
                } else {
                    String::new()
                },
                show_delta
                    .then(|| trend_ansi(reward_delta, state.previous_reward, Better::Higher))
                    .flatten(),
            ),
            // KL and clip fraction are budgets: healthy while small, a
            // warning as they climb toward an off-policy update.
            cell(
                format!("{kl:.5}"),
                budget_ansi(kl as f64, [0.01, 0.05, 0.10]),
            ),
            cell(
                format!("{clip_fraction:.3}"),
                budget_ansi(clip_fraction as f64, [0.05, 0.20, 0.40]),
            ),
            cell(format!("{:.2e}", epoch.learning_rate), DIM),
            throughput_cell(epoch.tokens_per_second as f64),
        ]);
        if epoch.policy_epoch == 1 {
            state.previous_reward = reward;
        }
        let bar = state.bar.clone();
        self.ui.progress_table(&bar, row);
    }

    fn evaluation_started(&mut self, update: u64, updates: u32) {
        let state = self.state();
        state
            .bar
            .set_message(format!("evaluating update {update}/{updates}"));
    }

    fn evaluation(&mut self, report: &EvaluationReport) {
        let state = self.state();
        let line = match report {
            EvaluationReport::Sft {
                epoch,
                loss,
                perplexity,
                outcome,
            } => format!(
                "eval epoch {epoch}: loss {loss:.6}  perplexity {perplexity:.4}  {}",
                eval_status(outcome, 6),
            ),
            EvaluationReport::Rollout {
                update,
                updates,
                mean_reward,
                reward_min,
                reward_max,
                examples,
                outcome,
            } => {
                state.last_eval_reward = *mean_reward as f64;
                let reward = state.current_reward as f64;
                state.bar.set_message(format!(
                    "reward={reward:.4} eval={:.4} loss={:.6}",
                    state.last_eval_reward, state.current_train_loss
                ));
                format!(
                    "eval update {update}/{updates}: mean {mean_reward:.4}  range [{reward_min:.4}, {reward_max:.4}]  n={examples}  {}",
                    eval_status(outcome, 4),
                )
            }
        };
        let row = state.table.span_row(&line, EVAL_ROW);
        let bar = state.bar.clone();
        self.ui.progress_table(&bar, row);
    }

    fn checkpoint_written(&mut self, path: &std::path::Path) {
        let state = self.state();
        let row = state
            .table
            .span_row(&format!("checkpoint {}", path.display()), DIM);
        let bar = state.bar.clone();
        self.ui.progress_table(&bar, row);
    }

    fn memory_note(&mut self, note: &str) {
        let state = self.state();
        let row = state.table.span_row(note, DIM);
        let bar = state.bar.clone();
        self.ui.progress_table(&bar, row);
    }

    fn loop_finished(&mut self) {
        if let Some(state) = self.loop_state.take() {
            self.ui.progress_table(&state.bar, state.table.close());
            state.bar.finish_and_clear();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    #[test]
    fn train_parser_rejects_missing_duplicate_and_unknown_arguments() {
        assert!(
            parse_train_args(&[])
                .unwrap_err()
                .to_string()
                .contains("requires")
        );
        assert!(
            parse_train_args(&strings(&["a.toml", "b.toml"]))
                .unwrap_err()
                .to_string()
                .contains("exactly one")
        );
        assert!(
            parse_train_args(&strings(&["a.toml", "--unknown"]))
                .unwrap_err()
                .to_string()
                .contains("unknown train flag")
        );
        assert!(
            parse_train_args(&strings(&["a.toml", "--resume", "--resume"]))
                .unwrap_err()
                .to_string()
                .contains("at most once")
        );
    }
}
