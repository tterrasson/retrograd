//! Preference optimization on a real model.
//!
//! Covered properties:
//!
//! - One step lowers the loss it takes the gradient of, laid out as rows or
//!   packed over the shared prompt.
//! - Scored against the initial policy, the first step sees a margin of exactly
//!   zero: policy and reference are the same weights.
//! - A small set of pairs is learned: the held-out accuracy rises past chance
//!   and the margin grows.
//! - A run stopped between two steps and resumed from its checkpoint ends
//!   where the uninterrupted run ends.
//!
//! The downloaded fixture, because a pair is rendered through a chat template.
//! Its CPU matmuls quantize the activations they read, so a step far below a
//! practical rate moves the scores by rounding rather than by its gradient -
//! a plain cross-entropy step does the same - and a step is only checked at a
//! rate its signal dominates. That the coefficients are the gradient is proven
//! numerically in `preference::loss`.

mod common;

use std::path::{Path, PathBuf};

use retrograd::config::{self, PreferenceConfig, PreferenceLoss, ReferenceSource};
use retrograd::run::{self, ControlPoint, Flow, RunControl, RunControls, SilentObserver};
use retrograd::training::preference;
use retrograd::{
    Device, LoraConfig, LoraDtype, SharedPrefixFanout, TargetSet, TrainConfig, Trainer,
};

macro_rules! cpu_fixture {
    () => {
        match common::model_path_if_available() {
            Some(model) => model,
            None => {
                eprintln!("skipping: CPU fixture not available");
                return;
            }
        }
    };
}

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "retrograd-preference-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("create scratch directory");
    path
}

const WORDS: [&str; 16] = [
    "hello there",
    "good morning",
    "thank you",
    "see you soon",
    "well done",
    "nice to meet you",
    "have a nice day",
    "all is well",
    "welcome back",
    "take care",
    "good night",
    "happy birthday",
    "safe travels",
    "good luck",
    "congratulations",
    "cheers mate",
];

/// One line per phrase: asked to shout it, the uppercase answer is preferred.
fn write_pairs(path: &Path, phrases: &[&str]) {
    let lines = phrases
        .iter()
        .map(|phrase| {
            serde_json::json!({
                "prompt": [{"role": "user", "content": format!("Shout: {phrase}")}],
                "chosen": [{"role": "assistant", "content": phrase.to_uppercase()}],
                "rejected": [{"role": "assistant", "content": phrase.to_string()}],
            })
            .to_string()
        })
        .collect::<Vec<_>>();
    std::fs::write(path, lines.join("\n") + "\n").expect("write the pairs");
}

fn training(learning_rate: f32, fanout: SharedPrefixFanout) -> TrainConfig {
    TrainConfig {
        n_ctx: 128,
        n_batch: 128,
        n_ubatch: 64,
        n_seq_max: 2,
        epochs: 1,
        learning_rate,
        shared_prefix_fanout: fanout,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    LoraConfig {
        rank: 4,
        alpha: 8.0,
        dropout: 0.0,
        seed: 7,
        targets: TargetSet::Auto,
        dtype: LoraDtype::F32,
    }
}

fn dpo(data: &Path) -> PreferenceConfig {
    PreferenceConfig {
        data: data.to_path_buf(),
        loss: PreferenceLoss::Dpo {
            beta: 0.1,
            label_smoothing: 0.0,
        },
        reference: Some(ReferenceSource::Initial),
        shuffle: false,
        seed: 7,
        pairs_per_step: None,
        logps_drop_warn: config::DEFAULT_LOGPS_DROP_WARN,
    }
}

fn trainer(model: &Path, training: &TrainConfig) -> Trainer {
    let mut trainer = Trainer::new(model, training.clone()).expect("load the model");
    trainer.create_lora(&lora()).expect("create the adapter");
    trainer
}

/// The loss of one pair before and after one step, with the pair laid out as
/// two rows or packed over its shared prompt.
fn one_step_losses(model: &Path, data: &Path, fanout: SharedPrefixFanout) -> (f64, f64) {
    let training = training(1.0e-3, fanout);
    let mut trainer = trainer(model, &training);
    // Against the initial policy the reference-relative loss starts at
    // exactly `log 2`; a reference-free loss starts wherever the pair is.
    let config = PreferenceConfig {
        loss: PreferenceLoss::Simpo {
            beta: 2.0,
            gamma_beta_ratio: 0.5,
        },
        reference: None,
        ..dpo(data)
    };
    let prepared = preference::prepare(&mut trainer, &config, &training, 1, Some((data, None)))
        .expect("prepare the pair");
    assert_eq!(prepared.total_steps(), 1, "one pair is one step");
    let before = preference::evaluate(&mut trainer, &prepared, None, &config.loss)
        .expect("evaluate before")
        .loss;
    let metrics =
        preference::run_resumed(&mut trainer, &prepared, None, &config, None, &mut |_, _| {
            Ok(true)
        })
        .expect("train one step");
    assert_eq!(metrics.global_step, 1);
    let after = preference::evaluate(&mut trainer, &prepared, None, &config.loss)
        .expect("evaluate after")
        .loss;
    (before, after)
}

#[test]
fn one_step_lowers_its_loss_in_either_layout() {
    let model = cpu_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("one-step");
    let data = root.join("pair.jsonl");
    write_pairs(&data, &WORDS[..1]);

    for fanout in [SharedPrefixFanout::Off, SharedPrefixFanout::Auto] {
        let (before, after) = one_step_losses(&model, &data, fanout);
        assert!(
            after < before,
            "{fanout:?}: one step raised its own loss, {before} -> {after}"
        );
    }
}

#[test]
fn the_initial_policy_is_the_reference_of_the_first_step() {
    let model = cpu_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("initial");
    let data = root.join("pairs.jsonl");
    write_pairs(&data, &WORDS[..4]);
    let training = training(1.0e-3, SharedPrefixFanout::Auto);
    let mut trainer = trainer(&model, &training);
    let config = dpo(&data);
    let prepared =
        preference::prepare(&mut trainer, &config, &training, 1, None).expect("prepare the pairs");
    let reference = preference::reference_table(&mut trainer, &prepared, &config)
        .expect("score the reference")
        .expect("DPO has a reference");

    let mut first = None;
    preference::run_resumed(
        &mut trainer,
        &prepared,
        Some(&reference),
        &config,
        None,
        &mut |_, progress| {
            first.get_or_insert(progress.values);
            Ok(false)
        },
    )
    .expect("train one step");
    let first = first.expect("one step reported");
    let value = |name: &str| {
        first
            .iter()
            .find(|value| value.name == name)
            .unwrap_or_else(|| panic!("{name} is reported"))
            .value
    };
    assert_eq!(value("preference/margin"), 0.0);
    assert_eq!(value("preference/accuracy"), 0.0);
    assert!((value("preference/loss") - std::f32::consts::LN_2).abs() < 1e-6);
}

#[test]
fn a_small_set_of_pairs_is_learned() {
    let model = cpu_fixture!();
    let _guard = common::serialize_models();
    let root = scratch("learning");
    let data = root.join("train.jsonl");
    let eval = root.join("eval.jsonl");
    write_pairs(&data, &WORDS[..12]);
    write_pairs(&eval, &WORDS[12..]);
    let training = training(2.0e-3, SharedPrefixFanout::Auto);
    let mut trainer = trainer(&model, &training);
    let config = PreferenceConfig {
        shuffle: true,
        pairs_per_step: Some(2),
        ..dpo(&data)
    };
    let prepared = preference::prepare(&mut trainer, &config, &training, 3, Some((&eval, None)))
        .expect("prepare the pairs");
    let reference =
        preference::reference_table(&mut trainer, &prepared, &config).expect("score the reference");
    let before = preference::evaluate(&mut trainer, &prepared, reference.as_ref(), &config.loss)
        .expect("evaluate before");
    let mut steps = 0_u64;
    let metrics = preference::run_resumed(
        &mut trainer,
        &prepared,
        reference.as_ref(),
        &config,
        None,
        &mut |_, progress| {
            steps += u64::from(!progress.metrics.epoch_complete);
            Ok(true)
        },
    )
    .expect("train");
    assert_eq!(
        steps,
        prepared.total_steps(),
        "the plan's step count is the run's"
    );
    assert_eq!(metrics.global_step, prepared.total_steps());
    let after = preference::evaluate(&mut trainer, &prepared, reference.as_ref(), &config.loss)
        .expect("evaluate after");
    assert!(
        after.margin > before.margin,
        "the margin did not grow: {} -> {}",
        before.margin,
        after.margin
    );
    assert!(
        after.accuracy > 0.9,
        "held-out accuracy {} after training",
        after.accuracy
    );
}

/// Stops the run after its `at`-th optimizer step.
struct StopAt {
    at: u64,
}

impl RunControl for StopAt {
    fn poll(&mut self, _: &mut dyn RunControls, at: ControlPoint) -> retrograd::Result<Flow> {
        Ok(match at.global_step >= self.at && at.at_boundary {
            true => Flow::Stop,
            false => Flow::Continue,
        })
    }
}

fn run_document(model: &Path, root: &Path, output: &str) -> config::RunConfig {
    let document = format!(
        r#"
[run]
algorithm = "preference"

[model]
path = "{model}"
device = "cpu"

[output]
path = "{output}"

[lora]
rank = 4
alpha = 8.0
seed = 7
dtype = "f32"

[training]
ctx = 128
micro_batch = 64
epochs = 2
lr = 1e-3
lr_scheduler = "linear"

[preference]
data = "pairs.jsonl"
pairs_per_step = 2

[checkpoint]
directory = "checkpoints"
mode = "steps"
every_steps = 1
"#,
        model = model.display(),
    );
    let document = config::parse_toml(&document, "preference-resume").expect("parse the run");
    config::build(document, root).expect("build the run")
}

fn adapter_scores(model: &Path, adapter: &Path, tokens: &[i32]) -> Vec<f32> {
    let mut trainer =
        Trainer::new(model, training(1.0e-3, SharedPrefixFanout::Auto)).expect("load the model");
    trainer.load_lora(adapter).expect("load the adapter");
    trainer.score_tokens(tokens).expect("score")
}

#[test]
fn a_resumed_run_ends_where_the_uninterrupted_one_does() {
    let model = cpu_fixture!();
    let _guard = common::serialize_models();
    let continuous_root = scratch("continuous");
    let resumed_root = scratch("resumed");
    for root in [&continuous_root, &resumed_root] {
        write_pairs(&root.join("pairs.jsonl"), &WORDS[..6]);
    }

    let continuous = run_document(&model, &continuous_root, "out.gguf");
    let outcome = run::execute(&continuous, &mut SilentObserver).expect("the continuous run");
    let total = outcome.metrics.global_step;
    assert!(total >= 4, "two epochs of three steps: {total}");

    let interrupted = run_document(&model, &resumed_root, "interrupted.gguf");
    let stopped = run::execute_controlled(
        &interrupted,
        &mut SilentObserver,
        &mut StopAt { at: 2 },
        Vec::new(),
    )
    .expect("the interrupted run");
    assert_eq!(stopped.metrics.global_step, 2, "stopped between two steps");
    assert!(
        resumed_root
            .join("checkpoints")
            .join(preference::reference::CACHE_FILE)
            .exists(),
        "the initial policy's scores are kept with the checkpoints"
    );

    let mut resumed = run_document(&model, &resumed_root, "out.gguf");
    run::apply_resume_override(&mut resumed, None).expect("resume from the latest checkpoint");
    let outcome = run::execute(&resumed, &mut SilentObserver).expect("the resumed run");
    assert_eq!(outcome.metrics.global_step, total);

    let trainer =
        Trainer::new(&model, training(1.0e-3, SharedPrefixFanout::Auto)).expect("load the model");
    let tokens = trainer
        .tokenize_text("Shout: hello there. HELLO THERE")
        .expect("tokenize");
    drop(trainer);
    let expected = adapter_scores(&model, &continuous_root.join("out.gguf"), &tokens);
    let actual = adapter_scores(&model, &resumed_root.join("out.gguf"), &tokens);
    let deviation = expected
        .iter()
        .zip(&actual)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        deviation <= 1e-5,
        "the resumed run diverged from the uninterrupted one by {deviation}"
    );
}
