//! The profiler's report: nine tables and a footer.
//!
//! What lives here is the shape of the profiler's own tables. The rendering
//! helpers they share with the other binaries (`base_table`, `secs`, `bar`) come
//! from `retrograd-cli-ui`.

use std::time::Duration;

use comfy_table::{Cell, CellAlignment, Color};
use retrograd::{DutyCycleStats, MemoryReport, TransferRates, config, memory};
use retrograd_cli_ui::{bar, base_table, right_cell, secs};

use super::{Options, PhaseTotals, UpdateRow, VramTrack};

/// The handful of workload fields [`print_header`] shows, read from whichever
/// algorithm's config: `GrpoConfig` and `AgentGrpoConfig` name the same shape
/// differently (`prompts_per_update` vs `scenarios_per_update`,
/// `sampling.max_new_tokens` vs `limits.max_new_tokens_per_turn`), so the
/// header takes the shape rather than either config type.
pub(super) struct Workload {
    pub updates: u32,
    pub per_update: usize,
    /// "prompts/upd" or "scenarios/upd", matching the algorithm's own vocabulary.
    pub per_update_label: &'static str,
    pub group_size: usize,
    pub epochs_per_update: u32,
    pub max_new_tokens: u32,
}

/// Formats an optional device reading for a table cell.
pub(super) fn vram_cell(value: Option<u64>) -> String {
    value.map_or_else(|| "-".to_string(), memory::format_bytes)
}

pub(super) fn signed(delta: i64) -> String {
    let sign = if delta < 0 { "-" } else { "+" };
    format!("{sign}{}", memory::format_bytes(delta.unsigned_abs()))
}

// --- Report sections. ---

pub(super) fn print_header(
    options: &Options,
    config: &config::RunConfig,
    workload: &Workload,
    backend: &str,
) {
    let field = |name: &str| {
        backend
            .lines()
            .find_map(|line| line.trim().strip_prefix(&format!("{name}: ")))
            .unwrap_or("?")
            .to_string()
    };
    println!("\n\x1b[1m═══ Retrograd training profiler ═══\x1b[0m");
    println!("  config           {}", options.config_path);
    println!("  model            {}", config.model.display());
    println!(
        "  device           {:?}  (backend: {}, gpu_active: {})",
        config.training.device,
        field("backend"),
        field("gpu_active"),
    );
    println!(
        "  workload         updates={} {}={} group={} epochs/upd={} max_new_tokens={}",
        workload.updates,
        workload.per_update_label,
        workload.per_update,
        workload.group_size,
        workload.epochs_per_update,
        workload.max_new_tokens,
    );
    println!(
        "  context          ctx={} step_tokens={} micro_batch={} optimizer_seq_max={} generation_concurrency={} fast_sampling={}",
        config.training.n_ctx,
        config.training.n_batch,
        config.training.n_ubatch,
        config.training.n_seq_max,
        config.training.generation_concurrency,
        field("fast_sampling_context"),
    );
    println!(
        "  generation       batch={} ubatch={} (decoupled from training.micro_batch)",
        field("generation_batch"),
        field("generation_ubatch"),
    );
    println!(
        "  lora             dtype={} optimizer_f16={}",
        field("lora_dtype"),
        field("optimizer_f16"),
    );
    println!(
        "  kv cache         requested={} effective={} ({}) [cap_flash_attn_back={}]",
        field("training_kv_requested_dtype"),
        field("training_kv_dtype"),
        field("training_kv_f16"),
        field("cap_flash_attn_back"),
    );
}

pub(super) fn print_init_table(init: &super::Init, mem_start: u64, vram: &VramTrack) {
    let mut table = base_table();
    table.set_header(vec![
        Cell::new("Initialization"),
        right_cell("Time"),
        right_cell("Δ host RSS"),
        right_cell("Δ device VRAM"),
    ]);
    // Device readings are cumulative relative to the baseline; differencing them
    // shows per-phase growth in both memory columns.
    let vram_delta = |current: Option<u64>, previous: Option<u64>| match (current, previous) {
        (Some(current), Some(previous)) => signed(current as i64 - previous as i64),
        _ => "-".to_string(),
    };
    let zero = vram.available().then_some(0);
    let rows = [
        (
            "Model + context load",
            secs(init.load_time),
            signed(init.mem_after_model as i64 - mem_start as i64),
            vram_delta(init.vram_after_model, zero),
        ),
        (
            "LoRA adapter creation",
            secs(init.lora_time),
            signed(init.mem_after_lora as i64 - init.mem_after_model as i64),
            vram_delta(init.vram_after_lora, init.vram_after_model),
        ),
        (
            "Preflight (backward graph build)",
            secs(init.preflight_time),
            "-".to_string(),
            vram_delta(init.vram_after_preflight, init.vram_after_lora),
        ),
    ];
    for (name, time, mem, gpu) in rows {
        table.add_row(vec![
            Cell::new(name),
            right_cell(time),
            right_cell(mem),
            right_cell(gpu),
        ]);
    }
    println!("\n{table}");
}

/// Split optimizer time into graph build, allocation, and kernel execution using
/// `llama_opt_timing`. These counters are process-lifetime totals, not per-update
/// values, so they need not equal `timing/optimizer_seconds`.
pub(super) fn print_optimizer_timing_table(phases: &PhaseTotals) {
    let build = phases.opt_graph_build;
    let alloc = phases.opt_allocation;
    let exec = phases.opt_execution;
    let total = build + alloc + exec;
    if total <= 0.0 {
        return;
    }

    let mut table = base_table();
    table.set_header(vec![
        Cell::new("Optimizer step (llama_opt_timing)"),
        right_cell("Time"),
        right_cell("Share"),
        Cell::new("Breakdown"),
    ]);
    for (name, value, color) in [
        ("Execution (kernels)", exec, Color::Red),
        ("Graph build", build, Color::Yellow),
        ("Backend allocation", alloc, Color::Blue),
    ] {
        let pct = 100.0 * value / total;
        table.add_row(vec![
            Cell::new(name).fg(color),
            right_cell(secs(Duration::from_secs_f32(value))),
            right_cell(format!("{pct:5.1} %")),
            Cell::new(bar(pct)).fg(color),
        ]);
    }
    println!("\n{table}");
    let overhead = 100.0 * (build + alloc) / total;
    println!(
        "  \x1b[2mbuild + allocation = {overhead:.1} % of the optimizer step \
         (paid per evaluation; above ~40 % it outweighs faster kernels)\x1b[0m"
    );
}

pub(super) fn print_phase_table(phases: &PhaseTotals, train_time: Duration) {
    let total = train_time.as_secs_f32().max(1e-6);
    let accounted = phases.accounted();
    let other = (train_time.as_secs_f32() - accounted).max(0.0);

    let mut table = base_table();
    table.set_header(vec![
        Cell::new("Phase (all updates)"),
        right_cell("Time"),
        right_cell("Share"),
        Cell::new("Breakdown"),
    ]);
    let entries = [
        ("Optimizer (fwd+bwd+step)", phases.optimizer, Color::Red),
        (
            "Sampling (generation + scoring)",
            phases.sampling,
            Color::Yellow,
        ),
        ("  ├─ generation (decode)", phases.generation, Color::Yellow),
        (
            "  └─ behavior scoring (teacher forcing)",
            phases.behavior_scoring,
            Color::Yellow,
        ),
        ("Reward (subprocess)", phases.reward, Color::Green),
        ("KL reference", phases.reference, Color::Blue),
        (
            "Other (prep, tokenization, overhead)",
            other,
            Color::DarkGrey,
        ),
    ];
    // Highlight the dominant phase, so the bottleneck is obvious at a glance.
    for (name, value, color) in entries {
        let pct = 100.0 * value / total;
        table.add_row(vec![
            Cell::new(name).fg(color),
            right_cell(secs(Duration::from_secs_f32(value))),
            right_cell(format!("{pct:5.1} %")),
            Cell::new(bar(pct)).fg(color),
        ]);
    }
    table.add_row(vec![
        Cell::new("TOTAL training").add_attribute(comfy_table::Attribute::Bold),
        right_cell(secs(train_time)),
        right_cell("100.0 %"),
        Cell::new(""),
    ]);
    println!("\n{table}");
    if phases.scoring_updates > 0.0 {
        // The two numbers that say *why* behavior scoring costs what it costs,
        // rather than leaving it to be inferred from the clock.
        let prefills = phases.scoring_prefix_decodes_per_group / phases.scoring_updates;
        let device = 100.0 * phases.scoring_device_logprob_fraction / phases.scoring_updates;
        println!(
            "  \x1b[2mbehavior scoring: {prefills:.2} prefix prefill(s) per group \
             (1.00 = shared prefix), {device:.0} % of logprobs gathered on the device\x1b[0m"
        );
    }
}

/// The duty-cycle limiter's line, when one was requested. The seconds move at
/// every GPU boundary, so they are read when the report is printed rather than
/// folded into the backend report, which is cached.
///
/// The two shares answer two different questions: `observed` is compute over
/// the accounted windows only and mostly echoes the setting back; `wall share`
/// divides by the trainer's whole wall clock, so the gap between the two *is*
/// the unaccounted host time - data loading, judging, tokenization, checkpoint
/// I/O.
pub(super) fn print_duty_cycle_line(stats: &DutyCycleStats) {
    if stats.requested_fraction >= 1.0 {
        return;
    }
    if !stats.active {
        println!(
            "\n  \x1b[33m⚠ max_gpu_duty_cycle={:.2} requested but inactive: the active \
             backend is the CPU, where training.threads remains the control.\x1b[0m",
            stats.requested_fraction,
        );
        return;
    }
    let Some(observed) = stats.observed() else {
        println!(
            "\n  \x1b[2mmax_gpu_duty_cycle={:.2} active, no accounted window yet\x1b[0m",
            stats.requested_fraction,
        );
        return;
    };
    print!(
        "\n  \x1b[2mduty cycle         requested={:.2} observed={:.2} \
         (compute {:.1} s / idle {:.1} s)",
        stats.requested_fraction, observed, stats.compute_seconds, stats.idle_seconds,
    );
    match stats.wall_share() {
        Some(share) => println!(
            "  wall share={share:.2} over {:.1} s\x1b[0m",
            stats.wall_seconds
        ),
        None => println!("\x1b[0m"),
    }
}

pub(super) fn print_updates_table(updates: &[UpdateRow]) {
    if updates.is_empty() {
        return;
    }
    let mut table = base_table();
    table.set_header(vec![
        right_cell("Upd"),
        right_cell("step"),
        right_cell("loss"),
        right_cell("tok/s"),
        right_cell("optim"),
        right_cell("sampling"),
        right_cell("reward"),
    ]);
    for (i, row) in updates.iter().enumerate() {
        table.add_row(vec![
            right_cell(i + 1),
            right_cell(row.step),
            right_cell(format!("{:.4}", row.loss)),
            right_cell(format!("{:.1}", row.tok_s)),
            right_cell(format!("{:.2}s", row.optimizer)),
            right_cell(format!("{:.2}s", row.sampling)),
            right_cell(format!("{:.3}s", row.reward)),
        ]);
    }
    println!("\n{table}");
}

pub(super) fn print_memory_table(
    mem_start: u64,
    mem_after_model: u64,
    mem_after_lora: u64,
    mem_peak: u64,
) {
    let mut table = base_table();
    table.set_header(vec![Cell::new("Memory (host RSS)"), right_cell("Value")]);
    let rows = [
        ("Baseline (before load)", memory::format_bytes(mem_start)),
        (
            "After model + context",
            memory::format_bytes(mem_after_model),
        ),
        ("After LoRA adapter", memory::format_bytes(mem_after_lora)),
        (
            "Peak (VmHWM over the whole run)",
            memory::format_bytes(mem_peak),
        ),
    ];
    for (name, value) in rows {
        table.add_row(vec![Cell::new(name), right_cell(value)]);
    }
    println!("\n{table}");
    if cfg!(not(target_os = "macos")) {
        println!(
            "  \x1b[2mNote: on this system VRAM is not counted in the host RSS.\n  \
             GPU buffers (model, LoRA, KV, activations) appear in the\n  \
             \"Device memory\" table below.\x1b[0m"
        );
    }
}

/// The table the `n_ubatch` sweep reads: activations kept for the backward grow
/// linearly with the physical micro-batch, and they land here, not in the RSS.
pub(super) fn print_vram_table(vram: &VramTrack, n_ubatch: u32) {
    if !vram.available() {
        println!(
            "\n  \x1b[2mDevice memory: no GPU backend registered in this build,\n  \
             VRAM peak not measurable.\x1b[0m"
        );
        return;
    }
    let mut table = base_table();
    table.set_header(vec![Cell::new("Device memory (VRAM)"), right_cell("Value")]);
    let rows = [
        (
            "Device baseline (before load)".to_string(),
            vram_cell(vram.baseline),
        ),
        ("Device total".to_string(), memory::format_bytes(vram.total)),
        (
            format!("Peak above baseline (ubatch={n_ubatch})"),
            memory::format_bytes(vram.peak_above_baseline()),
        ),
        (
            "Absolute device peak".to_string(),
            memory::format_bytes(vram.peak),
        ),
    ];
    let mut rows = rows.to_vec();
    if vram.runtime_peak > 0 {
        // Measured inside the optimizer step; phase rows are sampled between
        // steps and cannot contain the allocation and backward transient.
        rows.push((
            "Peak measured inside the step (runtime)".to_string(),
            memory::format_bytes(vram.runtime_peak),
        ));
        rows.push((
            "  of which backend scratch (CUDA pool / Vulkan prealloc)".to_string(),
            memory::format_bytes(vram.runtime_scratch_peak),
        ));
    }
    for (name, value) in rows {
        table.add_row(vec![Cell::new(name), right_cell(value)]);
    }
    println!("\n{table}");
    println!(
        "  \x1b[2mBudget reported by the device: the absolute peak includes other\n  \
         processes; compare the peaks above baseline across ubatch values.\x1b[0m"
    );
}

/// Report the measurements used by the activation-offload gate: retained
/// checkpoint size, device peak, and host round-trip time.
pub(super) fn print_checkpoint_table(
    memory: &MemoryReport,
    rates: Option<&TransferRates>,
    backward_seconds: f64,
) {
    if !memory.has_checkpoint_profile() {
        println!(
            "\n  \x1b[2mActivation checkpoints: none retained (gradient checkpointing\n  \
             disabled, or no backward graph built). The offload gate has nothing to measure.\x1b[0m"
        );
        return;
    }
    let mut table = base_table();
    table.set_header(vec![
        Cell::new("Activation checkpoints"),
        right_cell("Value"),
    ]);
    let mut rows = vec![
        (
            "Retained count".to_string(),
            memory.checkpoint_count.to_string(),
        ),
        (
            "Retained bytes (as held)".to_string(),
            memory::format_bytes(memory.checkpoint_retained_bytes),
        ),
        (
            format!(
                "Simultaneous peak ({} live)",
                memory.checkpoint_live_peak_count
            ),
            memory::format_bytes(memory.checkpoint_live_peak_bytes),
        ),
        (
            format!(
                "Long-lived ({} ≥ half the backward)",
                memory.checkpoint_long_lived_count
            ),
            memory::format_bytes(memory.checkpoint_long_lived_bytes),
        ),
    ];
    if let Some(fraction) = memory.checkpoint_mean_span_fraction() {
        rows.push((
            "Mean lifetime (fraction of the graph)".to_string(),
            format!("{:.0} %", fraction * 100.0),
        ));
    }
    // The trigger itself: numerator over the measured device peak.
    match memory.checkpoint_share_of_device_peak() {
        Some(share) => rows.push((
            "── Share of the measured device peak (offload threshold: 20 %)".to_string(),
            format!("{:.1} %", share * 100.0),
        )),
        None => rows.push((
            "── Share of the measured device peak".to_string(),
            "peak not measured".to_string(),
        )),
    }
    for (name, value) in rows {
        table.add_row(vec![Cell::new(name), right_cell(value)]);
    }
    println!("\n{table}");

    let Some(rates) = rates else {
        println!(
            "  \x1b[2mHost↔device rates not measured (no GPU device in this build):\n  \
             the offload gate's second condition stays undecided.\x1b[0m"
        );
        return;
    };
    let mut transfer = base_table();
    transfer.set_header(vec![
        Cell::new(format!(
            "Host↔device transfers ({} × {})",
            rates.iterations,
            memory::format_bytes(rates.bytes_per_transfer)
        )),
        right_cell("Rate"),
    ]);
    let rate_row = |label: &str, bytes_per_second: f64| {
        (
            label.to_string(),
            format!("{:.1} GiB/s", bytes_per_second / (1024.0 * 1024.0 * 1024.0)),
        )
    };
    let mut transfer_rows = vec![
        rate_row(
            if rates.pinned_is_pageable {
                "H2D pinned (unavailable: pageable)"
            } else {
                "H2D pinned"
            },
            rates.pinned_h2d_bytes_per_second,
        ),
        rate_row(
            if rates.pinned_is_pageable {
                "D2H pinned (unavailable: pageable)"
            } else {
                "D2H pinned"
            },
            rates.pinned_d2h_bytes_per_second,
        ),
    ];
    if !rates.pinned_is_pageable {
        transfer_rows.push(rate_row(
            "H2D pageable",
            rates.pageable_h2d_bytes_per_second,
        ));
        transfer_rows.push(rate_row(
            "D2H pageable",
            rates.pageable_d2h_bytes_per_second,
        ));
    }
    // The offloadable traffic is the long-lived subset, not everything retained:
    // offload only moves the activations whose lifetime has compute to hide the
    // copy behind, and a ratio computed over all of them would overstate both the
    // gain and the cost.
    let offloadable = memory.checkpoint_long_lived_bytes;
    if let Some(seconds) = rates.round_trip_seconds(offloadable) {
        transfer_rows.push((
            "Round trip of the long-lived subset".to_string(),
            // Preserve sub-millisecond transfers as microseconds so they do not
            // round to an ambiguous `0.0 ms`.
            if seconds < 1.0e-3 {
                format!("{:.0} µs", seconds * 1.0e6)
            } else {
                format!("{:.1} ms", seconds * 1000.0)
            },
        ));
        if backward_seconds > 0.0 {
            let share = seconds / backward_seconds;
            transfer_rows.push((
                "── Share of optimizer time (offload refused above 30 %)".to_string(),
                format!("{:.2} %", share * 100.0),
            ));
        }
    }
    for (name, value) in transfer_rows {
        transfer.add_row(vec![Cell::new(name), right_cell(value)]);
    }
    println!("\n{transfer}");
    println!(
        "  \x1b[2mEach copy is followed by a synchronization: nothing overlaps.\n  \
         A real ring would do better, which is the right direction of error for a gate.\x1b[0m"
    );
    // Do not infer unified memory from `device_buffer_is_host`: Metal reports
    // false for that flag even though its device memory is unified.
    println!(
        "  \x1b[2mCompare with the machine's interconnect (PCIe 4.0 x16 ≈ 25 GiB/s\n  \
         in theory, ~20 in practice): a rate of the same order *and symmetric*, with no\n  \
         pinned buffer type, describes shared memory, not a bus - and an\n  \
         offload ring would move nothing there.\x1b[0m"
    );
}

// --- Backend-report parsing. ---
//
// Only two things are still read from the text: the effective dtype *labels*
// (display-only) and the per-(context, buffer) breakdown, which is the report's
// finest granularity and has no structured counterpart. Every byte total comes
// from `Trainer::memory_report`.

/// Reads a `  name: value` line from the backend report.
fn report_field<'a>(report: &'a str, name: &str) -> Option<&'a str> {
    report
        .lines()
        .find_map(|line| line.trim().strip_prefix(&format!("{name}: ")))
        .map(str::trim)
}

/// One `memory_by_buffer` entry: a (context, buffer type) pair with its bytes.
struct BufferRow {
    context: String,
    label: String,
    host: bool,
    model: u64,
    kv: u64,
    compute: u64,
}

/// Parses the `memory_by_buffer (bytes):` block. Each line reads
/// `<optimizer|generation> <label> host=<0|1> model=<n> kv=<n> compute=<n>`.
fn parse_buffer_rows(report: &str) -> Vec<BufferRow> {
    let mut rows = Vec::new();
    for line in report.lines() {
        let trimmed = line.trim();
        let context = if let Some(rest) = trimmed.strip_prefix("optimizer ") {
            ("optimizer", rest)
        } else if let Some(rest) = trimmed.strip_prefix("generation ") {
            ("generation", rest)
        } else {
            continue;
        };
        let (ctx, rest) = context;
        if !rest.contains("host=") {
            continue;
        }
        let mut tokens = rest.split_whitespace();
        let Some(label) = tokens.next() else { continue };
        let kv_field = |key: &str| -> u64 {
            rest.split_whitespace()
                .find_map(|token| token.strip_prefix(key))
                .and_then(|value| value.parse().ok())
                .unwrap_or(0)
        };
        rows.push(BufferRow {
            context: ctx.to_string(),
            label: label.to_string(),
            host: kv_field("host=") != 0,
            model: kv_field("model="),
            kv: kv_field("kv="),
            compute: kv_field("compute="),
        });
    }
    rows
}

/// Show static allocations by component alongside the measured VRAM peak.
pub(super) fn print_component_memory_table(memory: &MemoryReport, report: &str) {
    // Byte figures come from the structured report; dtype labels come from text
    // because they describe effective precision after runtime fallbacks.
    let model = memory.model_weight_bytes;
    let opt_kv = memory.optimizer_kv_bytes;
    let opt_compute = memory.optimizer_compute_bytes;
    let (gen_kv, gen_compute) = if memory.has_generation_context {
        (
            Some(memory.generation_kv_bytes),
            Some(memory.generation_compute_bytes),
        )
    } else {
        (None, None)
    };
    let lora_params = memory.lora_parameter_bytes;
    let lora_grad = memory.lora_gradient_bytes;
    let momenta = memory.adamw_momenta_bytes;

    let model_dtype = report_field(report, "model_weight_dtype")
        .unwrap_or("?")
        .to_string();
    let training_kv_dtype = report_field(report, "training_kv_dtype").unwrap_or("?");
    // The generation context always trades in F16 KV when it runs its own
    // (fast/forward-only) context; otherwise it shares the optimizer's context
    // and therefore its dtype. See retro_backend.cpp's `fast_generation` branch.
    let fast_sampling = report_field(report, "fast_sampling_context") == Some("true");
    let generation_kv_dtype = if fast_sampling {
        "F16"
    } else {
        training_kv_dtype
    };
    let lora_dtype = report_field(report, "lora_dtype")
        .unwrap_or("?")
        .to_string();

    let mut table = base_table();
    table.set_header(vec![
        Cell::new("Memory component (static alloc.)"),
        right_cell("Type"),
        right_cell("Size"),
    ]);
    let mut rows: Vec<(String, String, u64, Color)> = vec![
        (
            "Model weights (base, shared)".to_string(),
            model_dtype,
            model,
            Color::Cyan,
        ),
        (
            "Optimizer context - KV cache".to_string(),
            training_kv_dtype.to_string(),
            opt_kv,
            Color::Red,
        ),
        (
            "Optimizer context - compute (activations)".to_string(),
            "F32".to_string(),
            opt_compute,
            Color::Red,
        ),
    ];
    if let Some(bytes) = gen_kv {
        rows.push((
            "Generation context - KV cache".to_string(),
            generation_kv_dtype.to_string(),
            bytes,
            Color::Yellow,
        ));
    }
    if let Some(bytes) = gen_compute {
        rows.push((
            "Generation context - compute".to_string(),
            "F32".to_string(),
            bytes,
            Color::Yellow,
        ));
    }
    rows.push((
        "LoRA - parameters".to_string(),
        lora_dtype,
        lora_params,
        Color::Green,
    ));
    rows.push((
        "LoRA - gradients".to_string(),
        "F32".to_string(),
        lora_grad,
        Color::Green,
    ));
    rows.push((
        "AdamW - moments m/v".to_string(),
        "F32".to_string(),
        momenta,
        Color::Green,
    ));
    for (name, dtype, bytes, color) in &rows {
        table.add_row(vec![
            Cell::new(name).fg(*color),
            right_cell(dtype).fg(*color),
            right_cell(memory::format_bytes(*bytes)),
        ]);
    }
    // Device vs host split as reported by the runtime (authoritative on which
    // budget each buffer draws from), then a grand total.
    for (label, bytes) in [
        ("── Device total (VRAM)", memory.device_bytes),
        ("── Host total (RAM)", memory.host_bytes),
    ] {
        table.add_row(vec![
            Cell::new(label).add_attribute(comfy_table::Attribute::Bold),
            Cell::new(""),
            right_cell(memory::format_bytes(bytes)).add_attribute(comfy_table::Attribute::Bold),
        ]);
    }
    println!("\n{table}");
    println!(
        "  \x1b[2mStatic allocations known to the runtime. \"compute\" = activation peak\n  \
         reserved by the scheduler (measured after the preflight). The real device\n  \
         peak also includes transients: see the VRAM table.\x1b[0m"
    );
}

/// The finest view: every (context, backend buffer) pair with its model / KV /
/// compute bytes, so it is obvious which device physically holds each part.
pub(super) fn print_buffer_memory_table(report: &str) {
    let rows = parse_buffer_rows(report);
    if rows.is_empty() {
        return;
    }
    let mut table = base_table();
    table.set_header(vec![
        Cell::new("Context"),
        Cell::new("Buffer"),
        Cell::new("Place").set_alignment(CellAlignment::Center),
        right_cell("Model"),
        right_cell("KV"),
        right_cell("Compute"),
    ]);
    for row in &rows {
        let color = if row.context == "optimizer" {
            Color::Red
        } else {
            Color::Yellow
        };
        table.add_row(vec![
            Cell::new(&row.context).fg(color),
            Cell::new(&row.label),
            Cell::new(if row.host { "host" } else { "device" })
                .set_alignment(CellAlignment::Center),
            right_cell(memory::format_bytes(row.model)),
            right_cell(memory::format_bytes(row.kv)),
            right_cell(memory::format_bytes(row.compute)),
        ]);
    }
    println!("\n{table}");
    println!(
        "  \x1b[2mThe \"Model\" weights of the \"generation\" rows repeat the shared\n  \
         weights (counted once in the device total).\x1b[0m"
    );
}

pub(super) fn print_footer(
    metrics: &retrograd::TrainMetrics,
    train_time: Duration,
    wall: Duration,
) {
    println!(
        "\n\x1b[1mSummary\x1b[0m  loss={:.4}  tok/s={:.1}  step={}  training={}  total={}",
        metrics.train_loss,
        metrics.tokens_per_second,
        metrics.global_step,
        secs(train_time),
        secs(wall),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_rows_are_read_from_the_backend_report_and_the_rest_is_skipped() {
        let report = "\
memory_by_buffer (bytes):
  optimizer CPU host=1 model=100 kv=20 compute=5
  generation Metal host=0 model=0 kv=64 compute=128
  optimizer summary line without fields
  optimizer_f16=1
unrelated line host=1
";
        let rows: Vec<_> = parse_buffer_rows(report)
            .iter()
            .map(|row| {
                (
                    row.context.clone(),
                    row.label.clone(),
                    row.host,
                    row.model,
                    row.kv,
                    row.compute,
                )
            })
            .collect();
        assert_eq!(
            rows,
            [
                ("optimizer".to_string(), "CPU".to_string(), true, 100, 20, 5),
                (
                    "generation".to_string(),
                    "Metal".to_string(),
                    false,
                    0,
                    64,
                    128
                ),
            ]
        );
    }

    #[test]
    fn a_missing_or_malformed_figure_reads_as_zero() {
        let rows = parse_buffer_rows("optimizer CUDA0 host=0 model=many compute=7");
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].model, rows[0].kv, rows[0].compute), (0, 0, 7));
    }
}
