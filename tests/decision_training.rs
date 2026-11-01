//! Fine-tuning a decision model, end to end, on the tiny decision fixture.
//!
//! The questions are rendered through the model's `systemone` template, packed
//! several to a micro-batch and trained through the grouped softmax
//! cross-entropy. What has to hold:
//!
//! - the grouped steps learn: on a mapping the untrained model cannot know,
//!   the held-out NLL falls and the accuracy rises;
//! - a run fits the temperatures on the held-out questions and writes them
//!   where each export is read from - `adapter.decision.temperature.*` in an
//!   adapter, `<arch>.decision.temperature.*` in a model GGUF and in the copy
//!   of the base `decision.export_base` asks for, whose tensors are untouched;
//! - a model GGUF written by a `full` run is served back by the engine with
//!   the scores the trainer computed for it, which is what `exports_model`
//!   promises for `modern-bert`.
//!
//! The fixture is F32 and generated, so the file runs wherever the CPU lane
//! does.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use retrograd::config::{self, DecisionConfig};
use retrograd::decision::answer::softmax;
use retrograd::decision::dataset::{self, DataFormat};
use retrograd::decision::request::SystemOneRequest;
use retrograd::run::{self, SilentObserver};
use retrograd::trainer::{adapter_decision_metadata, read_gguf_metadata};
use retrograd::training::decision;
use retrograd::{Device, Engine, EngineConfig, LoraConfig, TargetSet, TrainConfig, Trainer};

macro_rules! decision_fixture {
    () => {
        match common::tiny_decision_model_path_if_available() {
            Some(model) => model,
            None => {
                eprintln!("skipping: tiny decision fixture not available");
                return;
            }
        }
    };
}

/// The colour in the state decides the option, through keys that say
/// nothing about it: an untrained model is at chance.
const COLOURS: [(&str, &str); 3] = [("red", "k1"), ("green", "k2"), ("blue", "k3")];

fn write_dataset(path: &Path, n: usize, offset: usize) {
    let mut lines = Vec::new();
    for i in 0..n {
        let (colour, key) = COLOURS[(i + offset) % COLOURS.len()];
        let mut keys: Vec<&str> = COLOURS.iter().map(|(_, key)| *key).collect();
        keys.rotate_left((i / 3 + offset) % 3);
        let criteria: serde_json::Map<String, serde_json::Value> = keys
            .iter()
            .map(|key| ((*key).to_owned(), serde_json::Value::Null))
            .collect();
        lines.push(
            serde_json::json!({
                "state": format!("{colour} {}", i % 7),
                "question": {"type": "choice", "instructions": "pick", "criteria": criteria},
                "answer": key,
            })
            .to_string(),
        );
    }
    fs::write(path, lines.join("\n")).expect("write the dataset");
}

fn scratch(label: &str) -> tempfile::TempDir {
    tempfile::Builder::new()
        .prefix(&format!("retrograd-decision-{label}-"))
        .tempdir()
        .expect("create a scratch directory")
}

fn training_config() -> TrainConfig {
    TrainConfig {
        n_ctx: 256,
        n_batch: 256,
        n_ubatch: 256,
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1e-2,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn decision_config(data: PathBuf) -> DecisionConfig {
    DecisionConfig {
        data,
        data_format: DataFormat::Native,
        template: None,
        max_head_tokens: None,
        option_shuffle: true,
        shuffle: true,
        seed: 7,
        calibrate: true,
        questions_per_batch: 4,
        gradient_accumulation: 2,
        export_base: None,
    }
}

#[test]
fn grouped_steps_lower_the_held_out_loss() {
    let model = decision_fixture!();
    let _lock = common::serialize_models();
    let dir = scratch("learn");
    let train = dir.path().join("train.jsonl");
    let eval = dir.path().join("eval.jsonl");
    write_dataset(&train, 48, 0);
    write_dataset(&eval, 12, 1);

    let epochs = 6;
    let mut trainer = Trainer::new(&model, training_config()).expect("load the fixture");
    let mut lora = LoraConfig::auto(4, 8.0);
    lora.targets = TargetSet::Patterns(vec![
        "blk.*.attn_qkv.weight".into(),
        "blk.*.attn_output.weight".into(),
        "blk.*.ffn_up.weight".into(),
        "blk.*.ffn_down.weight".into(),
    ]);
    trainer.create_lora(&lora).expect("create the adapter");
    let prepared = decision::prepare(
        &mut trainer,
        &decision_config(train),
        &training_config(),
        epochs,
        Some((&eval, None)),
    )
    .expect("prepare the questions");
    assert_eq!(prepared.examples(), 48);
    assert_eq!(prepared.eval_examples(), 12);
    // Four questions a micro-batch, two micro-batches an update.
    assert_eq!(prepared.steps_per_epoch(), 6);
    assert_eq!(prepared.total_steps(), 6 * u64::from(epochs));

    let before = decision::evaluate(&mut trainer, &prepared).expect("evaluate the base");
    let mut updates = 0;
    let metrics = decision::run_resumed(&mut trainer, &prepared, None, &mut |_, progress| {
        assert!(progress.metrics.train_loss.is_finite());
        updates += 1;
        Ok(true)
    })
    .expect("train");
    assert_eq!(updates, prepared.total_steps());
    assert_eq!(metrics.global_step, prepared.total_steps());
    let after = decision::evaluate(&mut trainer, &prepared).expect("evaluate the adapter");
    eprintln!(
        "held-out NLL {:.4} -> {:.4}, accuracy {:.3} -> {:.3}",
        before.loss, after.loss, before.metrics.overall.accuracy, after.metrics.overall.accuracy
    );
    assert!(
        after.loss < before.loss,
        "the held-out NLL did not fall: {} -> {}",
        before.loss,
        after.loss
    );
    assert!(after.metrics.overall.accuracy >= before.metrics.overall.accuracy);

    let fitted = decision::fit_temperatures(&after).expect("fit the temperatures");
    assert!(fitted.temperatures.contains_key("choice"));
    assert!(fitted.temperatures.contains_key("choice.3_5"));
    assert!(fitted.nll_after <= fitted.nll_before + 1e-9);
}

/// A decision run on `model`: `training` is appended to `[training]`, `body`
/// after `[evaluation]`.
fn write_toml(dir: &Path, model: &Path, training: &str, body: &str) -> PathBuf {
    let path = dir.join("run.toml");
    fs::write(
        &path,
        format!(
            "[run]\nalgorithm = 'decision'\n[model]\npath = '{}'\ndevice = 'cpu'\n\
             [training]\nctx = 256\nmicro_batch = 256\ngradient_accumulation = 2\nepochs = 2\n\
             {training}\n[evaluation]\ndata = 'eval.jsonl'\n{body}",
            model.display()
        ),
    )
    .expect("write the run document");
    path
}

#[test]
fn an_adapter_run_exports_its_temperatures_and_a_calibrated_base() {
    let model = decision_fixture!();
    let _lock = common::serialize_models();
    let dir = scratch("adapter");
    write_dataset(&dir.path().join("train.jsonl"), 24, 0);
    write_dataset(&dir.path().join("eval.jsonl"), 9, 1);
    let document = write_toml(
        dir.path(),
        &model,
        "lr = 1e-2",
        "[output]\npath = 'adapter.gguf'\n[lora]\nrank = 4\nalpha = 8\n\
         targets = ['qkv', 'o', 'ffn_up', 'ffn_down']\ndtype = 'f32'\n\
         [decision]\ndata = 'train.jsonl'\nquestions_per_batch = 4\nexport_base = 'base.gguf'\n",
    );
    let config = config::load(&document).expect("the document builds");
    run::execute(&config, &mut SilentObserver).expect("the run completes");

    let adapter = adapter_decision_metadata(dir.path().join("adapter.gguf")).expect("read");
    let choice = adapter
        .get("temperature.choice")
        .expect("the adapter carries the fitted choice temperature")
        .clone();
    assert!(adapter.contains_key("temperature.choice.3_5"));

    // The calibrated base: the same tensors, the adapter's temperatures in
    // the model's own keys, served as the model's.
    let base = dir.path().join("base.gguf");
    let source_len = fs::metadata(&model).expect("stat").len();
    let base_len = fs::metadata(&base).expect("stat").len();
    assert!(
        base_len >= source_len && base_len < source_len + 4096,
        "the base copy changed more than its header: {source_len} -> {base_len}"
    );
    let entries =
        read_gguf_metadata(&base, "modern-bert.decision.temperature.").expect("read the base");
    assert_eq!(
        entries.get("modern-bert.decision.temperature.choice"),
        Some(&choice)
    );
    let engine = Engine::new(
        &base,
        &EngineConfig {
            device: Device::Cpu,
            ..EngineConfig::default()
        },
    )
    .expect("serve the base copy");
    let info = engine.decision_info().expect("a decision head");
    assert_eq!(info.temperatures.get("temperature.choice"), Some(&choice));

    // The engine serving the adapter on the original base reads the same.
    let served = Engine::new(
        &model,
        &EngineConfig {
            device: Device::Cpu,
            ..EngineConfig::default()
        },
    )
    .expect("serve the base")
    .decision_info()
    .expect("a decision head")
    .with_adapter_temperatures(&adapter)
    .expect("the adapter's temperatures");
    assert_eq!(served.temperatures, info.temperatures);
}

#[test]
fn a_full_run_exports_a_model_the_engine_serves_as_trained() {
    let model = decision_fixture!();
    let _lock = common::serialize_models();
    let dir = scratch("model");
    write_dataset(&dir.path().join("train.jsonl"), 24, 0);
    write_dataset(&dir.path().join("eval.jsonl"), 9, 1);
    let document = write_toml(
        dir.path(),
        &model,
        "lr = 1e-3\ntrainable = 'full'",
        "[output]\npath = 'model.gguf'\nkind = 'model'\n\
         [decision]\ndata = 'train.jsonl'\nquestions_per_batch = 4\n",
    );
    let config = config::load(&document).expect("the document builds");
    run::execute(&config, &mut SilentObserver).expect("the run completes");

    let exported = dir.path().join("model.gguf");
    let entries = read_gguf_metadata(&exported, "modern-bert.decision.temperature.")
        .expect("read the export");
    assert!(entries.contains_key("modern-bert.decision.temperature.choice"));

    // The trainer and the engine read the export alike.
    let examples = dataset::read_jsonl(&dir.path().join("eval.jsonl"), DataFormat::Native)
        .expect("read the questions");
    let mut trainer = Trainer::new(&exported, training_config()).expect("load the export");
    let info = trainer.decision_info().expect("a decision head");
    let passes: Vec<_> = examples
        .iter()
        .map(|example| {
            trainer
                .decision_pass(
                    &info,
                    &example.state,
                    &example.question_id,
                    &example.question,
                    0,
                )
                .expect("render")
        })
        .collect();
    let scores = trainer
        .decide_passes(&info, &passes, 1)
        .expect("score the export");
    let mut engine = Engine::new(
        &exported,
        &EngineConfig {
            device: Device::Cpu,
            ..EngineConfig::default()
        },
    )
    .expect("serve the export");
    let served_info = engine.decision_info().expect("a decision head");
    assert_eq!(served_info.calibration, info.calibration);
    for (example, scores) in examples.iter().zip(scores) {
        let temperature = f64::from(served_info.calibration.temperature(
            &served_info.readout,
            example.question.kind,
            example.question.options.len(),
        ));
        let expected = softmax(
            &scores
                .iter()
                .map(|&score| f64::from(score))
                .collect::<Vec<_>>(),
            temperature,
        );
        let request = SystemOneRequest::single(
            example.state.clone(),
            example.question_id.clone(),
            example.question.clone(),
        );
        let decision = engine.decide(&served_info, &request).expect("decide");
        let answer = serde_json::to_value(&decision.answers[&example.question_id])
            .expect("serialize the answer");
        for (option, expected) in example.question.options.iter().zip(expected) {
            let served = answer["probabilities"][&option.key]
                .as_f64()
                .expect("a probability per option");
            assert!(
                (served - expected).abs() < 1e-4,
                "option {}: served {served}, trained {expected}",
                option.key
            );
        }
    }
}
