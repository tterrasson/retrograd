//! What an optimizer costs on a real model: persistent state, measured
//! optimizer-step time, and the device high-water the step reaches.
//!
//! A FLOP ratio is not a wall-time multiplier and the plan says so: Newton-
//! Schulz's arithmetic is a prediction, and what decides whether Muon is
//! affordable is a complete update including graph build, allocation, memory
//! traffic and the scratch the iteration needs. Every figure here is measured
//! on the same selection, the same corpus and the same device, so the only
//! thing that differs between two rows is the optimizer.

use std::path::Path;
use std::time::Instant;

use retrograd::{
    Device, GefenLayout, GefenVariant, OptimizerKind, TrainConfig, TrainablePolicy,
    TrainableRunConfig, TrainableSelector, TrainableSet, Trainer, memory, resolve_base,
    tensor_inventory,
};

/// The optimizers the campaign compares, in the order the report reads best:
/// the baseline first, then the two candidates.
fn candidates() -> Vec<(String, OptimizerKind)> {
    let mut kinds = vec![
        ("adamw".to_string(), OptimizerKind::AdamW),
        ("muon".to_string(), OptimizerKind::Muon),
    ];
    for variant in [GefenVariant::SharedV, GefenVariant::QuantizedM] {
        for block_size in [256_u64, 1024] {
            kinds.push((
                format!("gefen/{}/{block_size}", variant.as_str()),
                OptimizerKind::Gefen(GefenLayout {
                    variant,
                    block_size,
                    min_numel: 1,
                }),
            ));
        }
    }
    kinds
}

/// Every matrix and every norm of the model: a real parameter count rather than
/// a fixture's three matrices, which is the only count at which Muon's graph
/// size is a question and not an arithmetic exercise.
fn selector() -> TrainableSelector {
    TrainableSelector {
        modules: [
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_up",
            "ffn_down",
            "ffn_gate",
        ]
        .iter()
        .map(|module| (*module).to_string())
        .collect(),
        norms: true,
        ..Default::default()
    }
}

fn config(optimizer: OptimizerKind, device: Device) -> TrainConfig {
    TrainConfig {
        n_ctx: 512,
        n_batch: 512,
        n_ubatch: 128,
        epochs: 1,
        learning_rate: 1.0e-4,
        weight_decay: 0.0,
        max_grad_norm: 1.0,
        device,
        trainable: TrainableRunConfig {
            policy: TrainablePolicy::Partial,
            selector: selector(),
            optimizer,
        },
        optimizer_hyperparameters: optimizer.declared_hyperparameters(),
        ..TrainConfig::default()
    }
}

/// The corpus the timed runs train on. A file when one is given, otherwise a
/// fixed paragraph repeated: the loss it produces is meaningless either way,
/// and what is being timed is the step.
fn corpus(path: Option<&Path>) -> String {
    match path {
        Some(path) => std::fs::read_to_string(path).unwrap_or_else(|error| {
            eprintln!("optim-report: {}: {error}", path.display());
            std::process::exit(2);
        }),
        None => concat!(
            "The quick brown fox jumps over the lazy dog. ",
            "Pack my box with five dozen liquor jugs. ",
            "How vexingly quick daft zebras jump! ",
            "Sphinx of black quartz, judge my vow. ",
        )
        .repeat(64),
    }
}

pub fn run(model: &Path, device: Device, steps: u32) {
    println!("\n=== Optimizer cost on {} ===", model.display());
    let inventory = match tensor_inventory(model, device) {
        Ok(inventory) => inventory,
        Err(error) => {
            println!("  cost: skipped, {error}");
            return;
        }
    };
    let set = match resolve_base(&inventory, TrainablePolicy::Partial, &selector()) {
        Ok(set) => set,
        Err(error) => {
            println!("  cost: skipped, {error}");
            return;
        }
    };
    let elements: u64 = set.entries.iter().map(|entry| entry.n_elements).sum();
    println!(
        "  {} trainable tensors, {} parameters, device {device:?}, {steps} steps",
        set.entries.len(),
        elements
    );

    println!(
        "\n  {:<22} {:>12} {:>10} {:>12} {:>12} {:>12} {:>10}",
        "optimizer", "state", "bytes/par", "step (exec)", "step (build)", "peak", "matrices"
    );
    let text = corpus(None);
    for (name, kind) in candidates() {
        match measure(model, device, kind, &set, &text, steps) {
            Ok(row) => println!(
                "  {name:<22} {:>12} {:>10.4} {:>12.4} {:>12.4} {:>12} {:>10}",
                memory::format_bytes(row.state_bytes),
                row.state_bytes as f64 / elements as f64,
                row.execution_seconds,
                row.build_seconds,
                row.peak
                    .map_or_else(|| "-".to_string(), memory::format_bytes),
                row.eligible_matrices
            ),
            Err(error) => println!("  {name:<22} refused: {error}"),
        }
    }
    println!(
        "\n  state         the persistent table the runtime allocated, AdamW fallback \
         included\n  \
           step (exec)   seconds of optimizer kernels per step, from the runtime's own \
         counters\n  \
           step (build)  seconds of graph build and backend allocation per step - Muon \
         builds a\n                graph per eligible matrix, so this is where its node \
         count shows\n  \
           peak          device high-water measured inside the optimizer steps; '-' on a \
         backend\n                that reports no budget\n  \
           matrices      tensors the optimizer owns rather than falling back to AdamW on"
    );
}

struct Row {
    state_bytes: u64,
    execution_seconds: f64,
    build_seconds: f64,
    peak: Option<u64>,
    eligible_matrices: usize,
}

fn measure(
    model: &Path,
    device: Device,
    kind: OptimizerKind,
    set: &TrainableSet,
    text: &str,
    steps: u32,
) -> Result<Row, String> {
    let mut trainer =
        Trainer::new(model, config(kind, device)).map_err(|error| error.to_string())?;
    trainer
        .declare_trainable_set(set)
        .map_err(|error| error.to_string())?;
    // The graph build is where a parameter count too large for one graph fails,
    // and it fails as an allocation error rather than as a wrong number.
    trainer
        .prepare_optimizer()
        .map_err(|error| error.to_string())?;

    let marked = trainer
        .marked_trainable_set()
        .map_err(|error| error.to_string())?;
    let eligible_matrices = marked
        .entries
        .iter()
        .filter(|entry| kind.assign(entry) == Some(kind))
        .count();

    let tokens = trainer
        .tokenize_text(text)
        .map_err(|error| error.to_string())?;
    // One row per step, each the context plus the label shift, so `steps`
    // tokens groups are `steps` optimizer steps and not one.
    let per_step = 513.min(tokens.len());
    if per_step < 2 {
        return Err("the corpus is shorter than one context".to_string());
    }

    // One untimed step first: the first one pays for the graph build and the
    // backend allocation that every later step reuses, and averaging it in
    // would report a warm step's cost as if it were cold.
    trainer
        .train_tokens(&tokens[..per_step])
        .map_err(|error| error.to_string())?;
    let before = trainer
        .optimizer_timing()
        .map_err(|error| error.to_string())?;

    let started = Instant::now();
    for index in 0..steps {
        let offset = (index as usize * per_step) % tokens.len().max(1);
        let end = (offset + per_step).min(tokens.len());
        let window = if end - offset >= 2 {
            &tokens[offset..end]
        } else {
            &tokens[..per_step]
        };
        trainer
            .train_tokens(window)
            .map_err(|error| error.to_string())?;
    }
    let wall = started.elapsed().as_secs_f64();

    let after = trainer
        .optimizer_timing()
        .map_err(|error| error.to_string())?;
    let memory = trainer
        .optimizer_memory()
        .map_err(|error| error.to_string())?;
    let report = trainer.memory_report().map_err(|error| error.to_string())?;

    let divisor = f64::from(steps.max(1));
    let _ = wall;
    Ok(Row {
        state_bytes: report.optimizer_state_bytes,
        execution_seconds: (after.execution_seconds - before.execution_seconds) / divisor,
        build_seconds: ((after.graph_build_seconds - before.graph_build_seconds)
            + (after.allocation_seconds - before.allocation_seconds))
            / divisor,
        peak: memory
            .is_measured()
            .then_some(memory.device_peak_used_bytes),
        eligible_matrices,
    })
}

/// Loss over a fixed number of steps, per optimizer, repeated across seeds.
///
/// The only section that can say whether an optimizer *helps*. It is also the
/// one whose answer depends on a learning rate: Muon's scale is not AdamW's, so
/// a single shared rate compares an optimizer with a badly tuned one. The rate
/// each row ran at is printed beside its curve for that reason.
pub fn curves(model: &Path, device: Device, steps: u32, seeds: u32, corpus_path: Option<&Path>) {
    println!("\n=== Loss curves ===");
    let inventory = match tensor_inventory(model, device) {
        Ok(inventory) => inventory,
        Err(error) => {
            println!("  curves: skipped, {error}");
            return;
        }
    };
    let set = match resolve_base(&inventory, TrainablePolicy::Partial, &selector()) {
        Ok(set) => set,
        Err(error) => {
            println!("  curves: skipped, {error}");
            return;
        }
    };
    let text = corpus(corpus_path);
    println!(
        "  {steps} steps, {seeds} seeds, device {device:?}, corpus {}",
        corpus_path.map_or_else(
            || "(built-in)".to_string(),
            |path| path.display().to_string()
        )
    );

    println!(
        "\n  {:<22} {:>10} {:>12} {:>12} {:>12}",
        "optimizer", "lr", "loss@0", "loss@last", "spread"
    );
    for (name, kind) in candidates() {
        let learning_rate = match kind {
            // Muon's update is orthogonalized, so its natural rate is not
            // AdamW's and sharing one would compare a tuned optimizer with an
            // untuned one. This is the ratio the plan asks to state explicitly.
            OptimizerKind::Muon => 2.0e-2,
            _ => 1.0e-4,
        };
        let mut firsts = Vec::new();
        let mut lasts = Vec::new();
        let mut failure = None;
        for seed in 0..seeds {
            match curve(model, device, kind, &set, &text, steps, learning_rate, seed) {
                Ok(losses) if losses.len() >= 2 => {
                    firsts.push(losses[0]);
                    lasts.push(*losses.last().expect("a last loss"));
                }
                Ok(_) => failure = Some("the run produced fewer than two losses".to_string()),
                Err(error) => failure = Some(error),
            }
        }
        match failure {
            Some(error) => println!("  {name:<22} refused: {error}"),
            None => {
                let first = mean(&firsts);
                let last = mean(&lasts);
                let spread = lasts
                    .iter()
                    .fold(f64::MIN, |worst, value| worst.max(*value))
                    - lasts.iter().fold(f64::MAX, |best, value| best.min(*value));
                println!(
                    "  {name:<22} {learning_rate:>10.1e} {first:>12.5} {last:>12.5} {spread:>12.5}"
                );
            }
        }
    }
    println!(
        "\n  loss@0        the loss of the first optimizer step, averaged over the seeds\n  \
           loss@last     the loss of the last one, same average\n  \
           spread        the range of loss@last across seeds, each seed a different \
         starting\n                offset in the corpus: a difference between two \
         optimizers smaller than\n                this says nothing. The weights come \
         from a file, so nothing else varies.\n\n  \
           The learning rates are not one rate. Muon's update is orthogonalized \
         and its\n  scale is not AdamW's, so the column is there to be read: a \
         row that reached a\n  lower loss at a different rate has not been shown \
         to be the better optimizer."
    );
}

#[allow(clippy::too_many_arguments)]
fn curve(
    model: &Path,
    device: Device,
    kind: OptimizerKind,
    set: &TrainableSet,
    text: &str,
    steps: u32,
    learning_rate: f32,
    seed: u32,
) -> Result<Vec<f64>, String> {
    let mut run = config(kind, device);
    run.learning_rate = learning_rate;
    run.shuffle_seed = u64::from(seed);
    // The model's weights come out of a file, so there is no initialization for
    // a seed to vary. What it can vary is the data: each seed starts at a
    // different offset in the corpus and sees the windows in a different order,
    // which is the only independent axis a fixed checkpoint leaves.
    let mut trainer = Trainer::new(model, run).map_err(|error| error.to_string())?;
    trainer
        .declare_trainable_set(set)
        .map_err(|error| error.to_string())?;
    trainer
        .prepare_optimizer()
        .map_err(|error| error.to_string())?;

    let tokens = trainer
        .tokenize_text(text)
        .map_err(|error| error.to_string())?;
    let per_step = 513.min(tokens.len());
    if per_step < 2 {
        return Err("the corpus is shorter than one context".to_string());
    }

    let mut losses = Vec::new();
    for index in 0..steps {
        let offset = ((index as usize + seed as usize * 7) * per_step) % tokens.len().max(1);
        let end = (offset + per_step).min(tokens.len());
        let window = if end - offset >= 2 {
            &tokens[offset..end]
        } else {
            &tokens[..per_step]
        };
        let progress = trainer
            .train_tokens(window)
            .map_err(|error| error.to_string())?;
        losses.push(f64::from(progress.train_loss));
    }
    Ok(losses)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}
