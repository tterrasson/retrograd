//! Render the measurements collected by `bench_model`.
//!
//! Shared table and formatting primitives come from `retrograd-cli-ui`; this
//! module defines the bench-specific grouping and verdict.

use std::env;
use std::io::IsTerminal;

use comfy_table::{Cell, CellAlignment, Table};
use retrograd_cli_ui::{
    Better, delta_ansi, delta_color, fmt_value, group_row, header_cell, paint, rounded_table,
    styled_cell,
};

use super::{BenchResult, DistillBenchResult, Distribution, RewardBenchResult};

pub(super) fn print_sft_bench_report(base: &BenchResult, adapter: Option<&BenchResult>) {
    let color = std::io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    let has_adapter = adapter.is_some();
    let columns = if has_adapter { 5 } else { 2 };

    let base_logprobs = base.logprob_distribution();
    let base_tokens = base.supervised_token_distribution();
    let adapter_logprobs = adapter.map(BenchResult::logprob_distribution);
    let adapter_tokens = adapter.map(BenchResult::supervised_token_distribution);
    let supervision =
        |r: &BenchResult| 100.0 * r.metrics.supervised_tokens as f64 / r.input_tokens as f64;

    let mut table = rounded_table();

    let mut header = vec![header_cell(color, "metric"), header_cell(color, "base")];
    if has_adapter {
        header.push(header_cell(color, "adapter"));
        header.push(header_cell(color, "Δ"));
        header.push(header_cell(color, "Δ%"));
    }
    table.set_header(header);

    macro_rules! row {
        ($label:expr, $base:expr, $adapter:expr, $prec:expr, $unit:expr, $signed:expr, $better:expr, $pct:expr) => {
            metric_row(
                &mut table,
                color,
                has_adapter,
                $label,
                $base,
                $adapter,
                $prec,
                $unit,
                $signed,
                $better,
                $pct,
            )
        };
    }

    group_row(&mut table, color, columns, "quality");
    row!(
        "loss",
        base.metrics.loss(),
        adapter.map(|a| a.metrics.loss()),
        4,
        "",
        false,
        Better::Lower,
        true
    );
    row!(
        "perplexity",
        base.metrics.perplexity(),
        adapter.map(|a| a.metrics.perplexity()),
        2,
        "",
        false,
        Better::Lower,
        true
    );
    row!(
        "mean logprob",
        -base.metrics.loss(),
        adapter.map(|a| -a.metrics.loss()),
        4,
        "",
        true,
        Better::Higher,
        false
    );

    group_row(&mut table, color, columns, "mean logprob / example");
    row!(
        "mean",
        base_logprobs.mean,
        adapter_logprobs.map(|r| r.mean),
        4,
        "",
        true,
        Better::Higher,
        false
    );
    row!(
        "low",
        base_logprobs.low,
        adapter_logprobs.map(|r| r.low),
        4,
        "",
        true,
        Better::Higher,
        false
    );
    row!(
        "p10",
        base_logprobs.p10,
        adapter_logprobs.map(|r| r.p10),
        4,
        "",
        true,
        Better::Higher,
        false
    );
    row!(
        "p50",
        base_logprobs.median,
        adapter_logprobs.map(|r| r.median),
        4,
        "",
        true,
        Better::Higher,
        false
    );
    row!(
        "p90",
        base_logprobs.p90,
        adapter_logprobs.map(|r| r.p90),
        4,
        "",
        true,
        Better::Higher,
        false
    );
    row!(
        "high",
        base_logprobs.high,
        adapter_logprobs.map(|r| r.high),
        4,
        "",
        true,
        Better::Higher,
        false
    );
    row!(
        "stddev",
        base_logprobs.stddev,
        adapter_logprobs.map(|r| r.stddev),
        4,
        "",
        false,
        Better::Neutral,
        false
    );

    group_row(&mut table, color, columns, "data");
    row!(
        "examples",
        base.examples.len() as f64,
        adapter.map(|a| a.examples.len() as f64),
        0,
        "",
        false,
        Better::Neutral,
        false
    );
    row!(
        "scored tokens",
        base.metrics.supervised_tokens as f64,
        adapter.map(|a| a.metrics.supervised_tokens as f64),
        0,
        "",
        false,
        Better::Neutral,
        false
    );
    row!(
        "tokens/ex",
        base_tokens.mean,
        adapter_tokens.map(|t| t.mean),
        1,
        "",
        false,
        Better::Neutral,
        false
    );
    row!(
        "supervision",
        supervision(base),
        adapter.map(supervision),
        2,
        "%",
        false,
        Better::Neutral,
        false
    );

    group_row(&mut table, color, columns, "performance");
    row!(
        "input tok/s",
        base.input_tokens_per_second(),
        adapter.map(BenchResult::input_tokens_per_second),
        1,
        "",
        false,
        Better::Neutral,
        false
    );
    row!(
        "scored tok/s",
        base.scored_tokens_per_second(),
        adapter.map(BenchResult::scored_tokens_per_second),
        1,
        "",
        false,
        Better::Neutral,
        false
    );
    row!(
        "elapsed",
        base.elapsed.as_secs_f64(),
        adapter.map(|a| a.elapsed.as_secs_f64()),
        3,
        "s",
        false,
        Better::Neutral,
        false
    );

    println!("{table}");

    if let Some(adapter) = adapter {
        print_sft_bench_verdict(base, adapter, color);
    }
}

pub(super) fn print_reward_bench_report(
    base: &RewardBenchResult,
    adapter: Option<&RewardBenchResult>,
) {
    let color = std::io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    let has_adapter = adapter.is_some();
    let columns = if has_adapter { 5 } else { 2 };
    let base_rewards = base.distribution();
    let adapter_rewards = adapter.map(RewardBenchResult::distribution);

    let mut table = rounded_table();
    let mut header = vec![header_cell(color, "metric"), header_cell(color, "base")];
    if has_adapter {
        header.push(header_cell(color, "adapter"));
        header.push(header_cell(color, "Δ"));
        header.push(header_cell(color, "Δ%"));
    }
    table.set_header(header);

    macro_rules! reward_row {
        ($label:expr, $base:expr, $adapter:expr, $prec:expr, $better:expr) => {
            metric_row(
                &mut table,
                color,
                has_adapter,
                $label,
                $base,
                $adapter,
                $prec,
                "",
                false,
                $better,
                false,
            )
        };
    }

    group_row(&mut table, color, columns, "reward / example");
    reward_row!(
        "mean",
        base_rewards.mean,
        adapter_rewards.map(|r| r.mean),
        4,
        Better::Higher
    );
    reward_row!(
        "low",
        base_rewards.low,
        adapter_rewards.map(|r| r.low),
        4,
        Better::Higher
    );
    reward_row!(
        "p10",
        base_rewards.p10,
        adapter_rewards.map(|r| r.p10),
        4,
        Better::Higher
    );
    reward_row!(
        "p50",
        base_rewards.median,
        adapter_rewards.map(|r| r.median),
        4,
        Better::Higher
    );
    reward_row!(
        "p90",
        base_rewards.p90,
        adapter_rewards.map(|r| r.p90),
        4,
        Better::Higher
    );
    reward_row!(
        "high",
        base_rewards.high,
        adapter_rewards.map(|r| r.high),
        4,
        Better::Higher
    );
    reward_row!(
        "stddev",
        base_rewards.stddev,
        adapter_rewards.map(|r| r.stddev),
        4,
        Better::Neutral
    );

    group_row(&mut table, color, columns, "data");
    reward_row!(
        "examples",
        base.rewards.len() as f64,
        adapter.map(|result| result.rewards.len() as f64),
        0,
        Better::Neutral
    );
    group_row(&mut table, color, columns, "performance");
    reward_row!(
        "examples/s",
        base.examples_per_second(),
        adapter.map(RewardBenchResult::examples_per_second),
        1,
        Better::Neutral
    );
    metric_row(
        &mut table,
        color,
        has_adapter,
        "elapsed",
        base.elapsed.as_secs_f64(),
        adapter.map(|result| result.elapsed.as_secs_f64()),
        3,
        "s",
        false,
        Better::Neutral,
        false,
    );
    println!("{table}");

    if let Some(adapter) = adapter {
        print_reward_bench_verdict(base, adapter, color);
    }
}

/// The three figures of a distillation bench, and what they each say.
///
/// The order is deliberate: divergence first because it is what the run
/// minimizes, agreement second because it is the exact figure the divergence
/// cannot supply, perplexity last because it is the one comparable to an SFT
/// bench on the same corpus - and therefore the one that answers "was
/// on-policy distillation worth it".
pub(super) fn print_distill_bench_report(
    base: &DistillBenchResult,
    adapter: Option<&DistillBenchResult>,
) {
    let color = std::io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    let has_adapter = adapter.is_some();
    let columns = if has_adapter { 5 } else { 2 };
    let base_kl = base.divergence();
    let adapter_kl = adapter.map(DistillBenchResult::divergence);

    let mut table = rounded_table();
    let mut header = vec![header_cell(color, "metric"), header_cell(color, "base")];
    if has_adapter {
        header.push(header_cell(color, "adapter"));
        header.push(header_cell(color, "Δ"));
        header.push(header_cell(color, "Δ%"));
    }
    table.set_header(header);

    macro_rules! distill_row {
        ($label:expr, $base:expr, $adapter:expr, $prec:expr, $unit:expr, $better:expr) => {
            metric_row(
                &mut table,
                color,
                has_adapter,
                $label,
                $base,
                $adapter,
                $prec,
                $unit,
                false,
                $better,
                false,
            )
        };
    }

    // Lower is better throughout this group: it is a divergence, and the sign
    // flip the run controller applies lives in the training crate, not here.
    group_row(&mut table, color, columns, "KL(student‖teacher) / token");
    distill_row!(
        "mean",
        base_kl.mean,
        adapter_kl.map(|kl| kl.mean),
        4,
        "",
        Better::Lower
    );
    distill_row!(
        "p50",
        base_kl.median,
        adapter_kl.map(|kl| kl.median),
        4,
        "",
        Better::Lower
    );
    distill_row!(
        "p90",
        base_kl.p90,
        adapter_kl.map(|kl| kl.p90),
        4,
        "",
        Better::Lower
    );
    distill_row!(
        "high",
        base_kl.high,
        adapter_kl.map(|kl| kl.high),
        4,
        "",
        Better::Lower
    );

    group_row(&mut table, color, columns, "top-1 agreement");
    distill_row!(
        "positions",
        f64::from(base.bench.top1_agreement().unwrap_or(0.0)) * 100.0,
        adapter.map(|result| f64::from(result.bench.top1_agreement().unwrap_or(0.0)) * 100.0),
        2,
        "%",
        Better::Higher
    );

    // Absent, not zero: a prompt-only held-out file carries no written answer,
    // and printing 1.0 there would read as a perfect student.
    match base.bench.reference_perplexity() {
        Some(perplexity) => {
            group_row(&mut table, color, columns, "teacher-forced perplexity");
            distill_row!(
                "reference answers",
                perplexity,
                adapter.and_then(|result| result.bench.reference_perplexity()),
                4,
                "",
                Better::Lower
            );
            distill_row!(
                "examples",
                base.bench.reference_examples as f64,
                adapter.map(|result| result.bench.reference_examples as f64),
                0,
                "",
                Better::Neutral
            );
        }
        None => println!(
            "note       teacher-forced perplexity needs an assistant turn on the held-out \
             lines; this dataset carries none"
        ),
    }

    group_row(&mut table, color, columns, "data");
    distill_row!(
        "examples",
        base.bench.teacher_kl.len() as f64,
        adapter.map(|result| result.bench.teacher_kl.len() as f64),
        0,
        "",
        Better::Neutral
    );
    distill_row!(
        "budget clamped",
        f64::from(base.bench.budget_clamped_fraction) * 100.0,
        adapter.map(|result| f64::from(result.bench.budget_clamped_fraction) * 100.0),
        1,
        "%",
        Better::Neutral
    );

    group_row(&mut table, color, columns, "performance");
    distill_row!(
        "examples/s",
        base.examples_per_second(),
        adapter.map(DistillBenchResult::examples_per_second),
        1,
        "",
        Better::Neutral
    );
    distill_row!(
        "elapsed",
        base.elapsed.as_secs_f64(),
        adapter.map(|result| result.elapsed.as_secs_f64()),
        3,
        "s",
        Better::Neutral
    );
    println!("{table}");

    if let Some(adapter) = adapter {
        print_distill_bench_verdict(base, adapter, color);
    }
}

/// The verdict of a distillation bench, on the mean divergence.
///
/// It reads the mean and not the per-prompt deltas the reward verdict reads:
/// the two passes resample, so a prompt's two completions are two different
/// token sequences and their divergences are not paired measurements. Counting
/// "prompts improved" over unpaired samples would put a number on noise.
fn print_distill_bench_verdict(
    base: &DistillBenchResult,
    adapter: &DistillBenchResult,
    color: bool,
) {
    let delta = adapter.divergence().mean - base.divergence().mean;
    let (verdict, code) = if delta < -1.0e-9 {
        ("improved", "38;5;40;1")
    } else if delta > 1.0e-9 {
        ("degraded", "38;5;196;1")
    } else {
        ("unchanged", "38;5;245;1")
    };
    println!(
        "verdict    {}   mean KL Δ{delta:+.4} nats/token",
        paint(color, code, verdict),
    );
    if let (Some(base_top1), Some(adapter_top1)) =
        (base.bench.top1_agreement(), adapter.bench.top1_agreement())
    {
        let delta = 100.0 * f64::from(adapter_top1 - base_top1);
        println!(
            "top-1      {:.2}% → {:.2}%   {}",
            100.0 * f64::from(base_top1),
            100.0 * f64::from(adapter_top1),
            paint(color, delta_ansi(delta), &format!("{delta:+.2} pt")),
        );
    }
    if let (Some(base_ppl), Some(adapter_ppl)) = (
        base.bench.reference_perplexity(),
        adapter.bench.reference_perplexity(),
    ) {
        let delta = adapter_ppl - base_ppl;
        println!(
            "perplexity {base_ppl:.4} → {adapter_ppl:.4}   {}",
            paint(color, delta_ansi(-delta), &format!("{delta:+.4}")),
        );
    }
}

#[expect(clippy::too_many_arguments)]
fn metric_row(
    table: &mut Table,
    color: bool,
    has_adapter: bool,
    label: &str,
    base: f64,
    adapter: Option<f64>,
    precision: usize,
    unit: &str,
    signed: bool,
    better: Better,
    show_pct: bool,
) {
    let mut cells = vec![Cell::new(label)];
    cells.push(styled_cell(
        fmt_value(base, precision, unit, signed),
        CellAlignment::Right,
        None,
    ));
    if has_adapter {
        let adapter = adapter.unwrap_or(base);
        let delta = adapter - base;
        let style = if color {
            Some(delta_color(delta, base, better))
        } else {
            None
        };
        cells.push(styled_cell(
            fmt_value(adapter, precision, unit, signed),
            CellAlignment::Right,
            None,
        ));
        cells.push(styled_cell(
            fmt_value(delta, precision, unit, true),
            CellAlignment::Right,
            style,
        ));
        let pct = if show_pct && base.abs() > 1.0e-9 {
            format!("{:+.2}%", 100.0 * delta / base)
        } else {
            String::new()
        };
        cells.push(styled_cell(pct, CellAlignment::Right, style));
    }
    table.add_row(cells);
}

fn print_sft_bench_verdict(base: &BenchResult, adapter: &BenchResult, color: bool) {
    let loss_delta = adapter.metrics.loss() - base.metrics.loss();
    let reward_delta = -loss_delta;
    let loss_percent = if base.metrics.loss().abs() > 1.0e-9 {
        100.0 * loss_delta / base.metrics.loss()
    } else {
        0.0
    };
    let (verdict, code) = if reward_delta > 1.0e-9 {
        ("improved", "38;5;40;1")
    } else if reward_delta < -1.0e-9 {
        ("degraded", "38;5;196;1")
    } else {
        ("unchanged", "38;5;245;1")
    };

    let example_deltas: Vec<f64> = base
        .examples
        .iter()
        .zip(&adapter.examples)
        .map(|(base, adapter)| adapter.logprob - base.logprob)
        .collect();
    let deltas = Distribution::new(example_deltas.iter().copied());
    let improved = example_deltas.iter().filter(|&&d| d > 1.0e-9).count();
    let degraded = example_deltas.iter().filter(|&&d| d < -1.0e-9).count();
    let unchanged = example_deltas.len() - improved - degraded;
    let improved_pct = 100.0 * improved as f64 / example_deltas.len() as f64;

    println!(
        "verdict    {}   mean logprob Δ{reward_delta:+.4}   loss {loss_percent:+.2}%",
        paint(color, code, verdict),
    );
    println!(
        "examples   {}   {}   {}",
        paint(
            color,
            "38;5;40",
            &format!(
                "↑ {improved}/{} improved ({improved_pct:.1}%)",
                example_deltas.len()
            ),
        ),
        paint(color, "38;5;196", &format!("↓ {degraded} degraded")),
        paint(color, "38;5;245", &format!("= {unchanged} unchanged")),
    );
    println!(
        "change/ex  mean Δ{:+.4}   stddev {:.4}   best Δ{}   worst Δ{}",
        deltas.mean,
        deltas.stddev,
        paint(
            color,
            delta_ansi(deltas.high),
            &format!("{:+.4}", deltas.high)
        ),
        paint(
            color,
            delta_ansi(deltas.low),
            &format!("{:+.4}", deltas.low)
        ),
    );
}

fn print_reward_bench_verdict(base: &RewardBenchResult, adapter: &RewardBenchResult, color: bool) {
    let example_deltas: Vec<f64> = base
        .rewards
        .iter()
        .zip(&adapter.rewards)
        .map(|(base, adapter)| adapter - base)
        .collect();
    let deltas = Distribution::new(example_deltas.iter().copied());
    let mean_delta = adapter.distribution().mean - base.distribution().mean;
    let (verdict, code) = if mean_delta > 1.0e-9 {
        ("improved", "38;5;40;1")
    } else if mean_delta < -1.0e-9 {
        ("degraded", "38;5;196;1")
    } else {
        ("unchanged", "38;5;245;1")
    };
    let improved = example_deltas
        .iter()
        .filter(|&&delta| delta > 1.0e-9)
        .count();
    let degraded = example_deltas
        .iter()
        .filter(|&&delta| delta < -1.0e-9)
        .count();
    let unchanged = example_deltas.len() - improved - degraded;
    let improved_pct = 100.0 * improved as f64 / example_deltas.len() as f64;

    println!(
        "verdict    {}   mean reward Δ{mean_delta:+.4}",
        paint(color, code, verdict),
    );
    println!(
        "examples   {}   {}   {}",
        paint(
            color,
            "38;5;40",
            &format!(
                "↑ {improved}/{} improved ({improved_pct:.1}%)",
                example_deltas.len()
            ),
        ),
        paint(color, "38;5;196", &format!("↓ {degraded} degraded")),
        paint(color, "38;5;245", &format!("= {unchanged} unchanged")),
    );
    println!(
        "change/ex  mean Δ{:+.4}   stddev {:.4}   best Δ{}   worst Δ{}",
        deltas.mean,
        deltas.stddev,
        paint(
            color,
            delta_ansi(deltas.high),
            &format!("{:+.4}", deltas.high)
        ),
        paint(
            color,
            delta_ansi(deltas.low),
            &format!("{:+.4}", deltas.low)
        ),
    );
}
