use std::fs;
use std::path::Path;

use retrograd_core::{
    CheckpointDtype, DEFAULT_REWARD_TIMEOUT_SECONDS, Device, FeatureDtype, KvDtype, LoraDtype,
    LrScheduler, RewardMode, RewardProtocol, SamplingParams, SharedPrefixFanout, TargetSet,
};

use retrograd_dataset::DataFormat;
use serde::Deserialize as _;

use crate::common::{parse_data_format, parse_scheduler, sampling};
use crate::grpo::build_grpo;
use crate::ppo::{build_ppo, critic};
use crate::*;

use std::time::{SystemTime, UNIX_EPOCH};

fn write_config(source: &str) -> PathBuf {
    // A timestamp alone is not unique among parallel tests; the pid and
    // counter make the directory unique across processes and tests.
    static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "retrograd-config-{}-{}-{}",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    let file = path.join("run.toml");
    fs::write(&file, source).unwrap();
    file
}

fn remove_config(file: &Path) {
    fs::remove_dir_all(file.parent().unwrap()).unwrap();
}

/// The shipped examples are the documentation a reader copies first, so a
/// rename of the `[training]` surface has to reach them or they teach a
/// spelling the loader rejects. `load` does not touch the filesystem beyond
/// the document itself, so this is a validation pass, not a run.
#[test]
fn every_shipped_example_still_loads() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .join("examples");
    let mut checked = 0;
    let mut pending = vec![examples];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(&dir).expect("read examples") {
            let path = entry.expect("dir entry").path();
            if path.is_dir() {
                pending.push(path);
                continue;
            }
            if path.extension().is_none_or(|ext| ext != "toml") {
                continue;
            }
            // A params-only document has no `[run]`: it is a partial tree for
            // the server's resolver, not a configuration.
            let source = fs::read_to_string(&path).expect("read example");
            if !source.lines().any(|line| line.trim_end() == "[run]") {
                continue;
            }
            // The shipped examples point at the CPU fixture, which this lane
            // does not fetch, so it supplies a path: what is under test is
            // the rest of the document, not where the GGUF lives.
            let overrides = ModelOverride {
                path: Some(PathBuf::from("model.gguf")),
                device: None,
            };
            load_with(&path, overrides)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            checked += 1;
        }
    }
    // The three smoke configurations: SFT, PPO and GRPO.
    assert!(checked >= 3, "only {checked} examples were checked");
}

fn valid_sampling() -> SamplingToml {
    SamplingToml {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 8,
        seed: 7,
    }
}

fn valid_grpo() -> GrpoConfig {
    GrpoConfig {
        prompts: "prompts.jsonl".into(),
        reward_command: vec!["reward".into()],
        reward_protocol: RewardProtocol::default(),
        updates: 1,
        prompts_per_update: 2,
        group_size: 2,
        grpo_epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        mask_truncated: false,
        baseline: AdvantageBaseline::Mean,
        prompt_order: PromptOrder::Sequential,
        overlong_penalty: None,
        kl_schedule: None,
        dynamic_sampling: None,
        judge: None,
        max_stalled_updates: DEFAULT_MAX_STALLED_UPDATES,
        sampling: SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: 7,
        },
    }
}

fn valid_ppo_toml() -> PpoToml {
    PpoToml {
        prompts: "prompts.jsonl".into(),
        reward_command: vec!["reward".into()],
        reward_mode: None,
        reward_timeout_seconds: None,
        updates: 1,
        rollout_batch_size: 2,
        ppo_epochs: 1,
        clip_range: 0.2,
        kl_coefficient: 0.1,
        critic: CriticToml::default(),
        sampling: valid_sampling(),
    }
}

#[test]
fn targets_expand_aliases() {
    let TargetSet::Patterns(patterns) = parse_targets(&[
        "q".into(),
        "v".into(),
        "ffn_gate".into(),
        "blk.0.ffn_up.weight".into(),
    ])
    .unwrap() else {
        panic!("explicit targets should produce patterns")
    };
    assert_eq!(
        patterns,
        [
            "blk.*.attn_q.weight",
            "blk.*.attn_v.weight",
            "blk.*.ffn_gate.weight",
            "blk.0.ffn_up.weight",
        ]
    );
    assert!(matches!(parse_targets(&[]).unwrap(), TargetSet::Auto));
    assert!(parse_targets(&["auto".into(), "q".into()]).is_err());
    assert!(parse_targets(&["not-a-target".into()]).is_err());
}

/// An omitted `lora.targets` trains every projection; only an explicit
/// `['auto']` hands the choice back to the runtime.
#[test]
fn omitted_targets_default_to_every_projection() {
    let base = "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n";
    let default_file = write_config(&format!("{base}[sft]\ndata='data.txt'\n"));
    assert_eq!(
        load(&default_file).unwrap().lora.config.targets,
        TargetSet::Patterns(vec![
            "blk.*.attn_q.weight".into(),
            "blk.*.attn_k.weight".into(),
            "blk.*.attn_v.weight".into(),
            "blk.*.attn_output.weight".into(),
            "blk.*.ffn_up.weight".into(),
            "blk.*.ffn_down.weight".into(),
            "blk.*.ffn_gate.weight".into(),
        ])
    );
    remove_config(&default_file);

    let auto_file = write_config(&format!("{base}targets=['auto']\n[sft]\ndata='data.txt'\n"));
    assert_eq!(
        load(&auto_file).unwrap().lora.config.targets,
        TargetSet::Auto
    );
    remove_config(&auto_file);
}
#[test]
fn sampling_validates_every_numeric_boundary() {
    let parsed = sampling(valid_sampling()).unwrap();
    assert_eq!(parsed.temperature, 1.0);
    assert_eq!(parsed.top_p, 1.0);
    assert_eq!(parsed.max_new_tokens, 8);
    assert_eq!(parsed.seed, 7);

    for invalid in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut value = valid_sampling();
        value.temperature = invalid;
        assert!(sampling(value).is_err(), "temperature={invalid}");
    }
    for invalid in [0.0, -0.1, 1.1, f32::NAN, f32::INFINITY] {
        let mut value = valid_sampling();
        value.top_p = invalid;
        assert!(sampling(value).is_err(), "top_p={invalid}");
    }
    let mut value = valid_sampling();
    value.max_new_tokens = 0;
    assert!(sampling(value).is_err());
}
#[test]
fn resolves_paths_from_the_config_directory() {
    let file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
    );
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.model, file.parent().unwrap().join("model.gguf"));
    assert_eq!(loaded.training.max_grad_norm, 1.0);
    assert!(
        matches!(loaded.algorithm, Algorithm::Sft(SftConfig { data, .. }) if data == file.parent().unwrap().join("data.txt"))
    );
    remove_config(&file);
}

/// A document may leave `[model]` out entirely when the frontend supplies
/// it - that is what `train --model` is. The override is an input to the
/// build, so the document that has no path still has to build; and the
/// override's path is the caller's, not resolved against the document's
/// directory the way a written one is.
#[test]
fn the_model_override_stands_in_for_a_missing_model_section() {
    let file =
        write_config("[run]\nalgorithm='sft'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n");
    let error = load(&file).unwrap_err().to_string();
    assert!(error.contains("[model].path is missing"), "{error}");

    let loaded = load_with(
        &file,
        ModelOverride {
            path: Some(PathBuf::from("elsewhere/model.gguf")),
            device: Some(Device::Cpu),
        },
    )
    .unwrap();
    assert_eq!(loaded.model, PathBuf::from("elsewhere/model.gguf"));
    assert_eq!(loaded.training.device, Device::Cpu);
    remove_config(&file);
}

/// `--model` wins over a written `[model].path`, and `--device` over
/// `[model].device`: the flag exists to swap a model without editing the
/// TOML, which it would not do if the document had the last word.
#[test]
fn the_model_override_outranks_a_written_model_section() {
    let file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\ndevice='gpu'\n\
         [lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
    );
    let loaded = load_with(
        &file,
        ModelOverride {
            path: Some(PathBuf::from("/other/model.gguf")),
            device: Some(Device::Cpu),
        },
    )
    .unwrap();
    assert_eq!(loaded.model, PathBuf::from("/other/model.gguf"));
    assert_eq!(loaded.training.device, Device::Cpu);
    remove_config(&file);
}

#[test]
fn sft_shuffles_by_default_seeded_from_the_lora_seed() {
    let default_file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\nseed=7\n[sft]\ndata='data.txt'\n",
    );
    let loaded = load(&default_file).unwrap();
    assert!(matches!(loaded.algorithm, Algorithm::Sft(SftConfig { shuffle, .. }) if shuffle));
    assert!(loaded.training.shuffle_dataset);
    // The runtime reads the seed from `training`, so the two must agree: a
    // shuffle silently running on 42 while the config names 7 is exactly
    // the class of bug this pair exists to prevent.
    assert_eq!(loaded.training.shuffle_seed, 7);
    remove_config(&default_file);

    let off_file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\nshuffle=false\n",
    );
    let loaded = load(&off_file).unwrap();
    assert!(matches!(loaded.algorithm, Algorithm::Sft(SftConfig { shuffle, .. }) if !shuffle));
    assert!(!loaded.training.shuffle_dataset);
    assert_eq!(loaded.training.shuffle_seed, 42, "the lora.seed default");
    remove_config(&off_file);
}

#[test]
fn lora_dtype_defaults_to_f16_and_accepts_only_f16_or_f32() {
    let default_file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
    );
    assert_eq!(
        load(&default_file).unwrap().lora.config.dtype,
        LoraDtype::F16
    );
    remove_config(&default_file);

    let f32_file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ndtype='f32'\n[sft]\ndata='data.txt'\n",
    );
    assert_eq!(load(&f32_file).unwrap().lora.config.dtype, LoraDtype::F32);
    remove_config(&f32_file);

    let invalid_file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ndtype='bf16'\n[sft]\ndata='data.txt'\n",
    );
    assert!(
        load(&invalid_file)
            .unwrap_err()
            .to_string()
            .contains("dtype")
    );
    remove_config(&invalid_file);
}

#[test]
fn evaluation_and_checkpoint_policy_are_shared_and_resolve_paths() {
    let file = write_config(concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[evaluation]\ndata='eval.txt'\nevery_iterations=2\npatience=3\nmin_delta=0.01\nmax_examples=16\n",
        "[checkpoint]\ndirectory='checkpoints'\nmode='steps_and_best_eval'\nevery_steps=10\n",
        "[sft]\ndata='train.txt'\n",
    ));
    let loaded = load(&file).unwrap();
    let evaluation = loaded.evaluation.unwrap();
    assert_eq!(evaluation.data, file.parent().unwrap().join("eval.txt"));
    assert_eq!(evaluation.every_iterations, 2);
    assert_eq!(evaluation.patience, Some(3));
    assert_eq!(evaluation.min_delta, 0.01);
    assert_eq!(evaluation.max_examples, Some(16));
    let checkpoint = loaded.checkpoint.unwrap();
    assert_eq!(
        checkpoint.directory,
        file.parent().unwrap().join("checkpoints")
    );
    assert_eq!(checkpoint.mode, CheckpointMode::StepsAndBestEval);
    assert_eq!(checkpoint.every_steps, Some(10));
    remove_config(&file);
}

#[test]
fn resume_from_resolves_and_excludes_a_cold_adapter_load() {
    let base = "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[sft]\ndata='train.txt'\n";
    let checkpoint = "[checkpoint]\ndirectory='ckpt'\nmode='steps'\nevery_steps=2\nresume_from='ckpt/step-000000000010.state'\n";

    let file = write_config(&format!("{base}[lora]\noutput='out.gguf'\n{checkpoint}"));
    let loaded = load(&file).unwrap();
    assert_eq!(
        loaded.checkpoint.unwrap().resume_from,
        Some(file.parent().unwrap().join("ckpt/step-000000000010.state"))
    );
    remove_config(&file);

    // A resume restores its own adapter, so pairing it with a cold adapter
    // load would leave which weights actually train ambiguous.
    let file = write_config(&format!(
        "{base}[lora]\noutput='out.gguf'\ninit_adapter='adapter.gguf'\n{checkpoint}"
    ));
    let error = load(&file).unwrap_err().to_string();
    assert!(error.contains("mutually exclusive"), "{error}");
    remove_config(&file);
}

#[test]
fn evaluation_and_checkpoint_defaults_and_dependencies_are_validated() {
    let base = "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='train.txt'\n";

    let file = write_config(&format!("{base}[evaluation]\ndata='eval.txt'\n"));
    let evaluation = load(&file).unwrap().evaluation.unwrap();
    assert_eq!(evaluation.every_iterations, 1);
    assert_eq!(evaluation.patience, None);
    assert_eq!(evaluation.min_delta, 0.0);
    assert_eq!(evaluation.max_examples, None);
    remove_config(&file);

    for section in [
        "[evaluation]\ndata='eval.txt'\nevery_iterations=0\n",
        "[evaluation]\ndata='eval.txt'\npatience=0\n",
        "[evaluation]\ndata='eval.txt'\nmin_delta=-0.1\n",
        "[evaluation]\ndata='eval.txt'\nmax_examples=0\n",
        "[checkpoint]\ndirectory='ckpt'\nmode='steps'\n",
        "[checkpoint]\ndirectory='ckpt'\nmode='best_eval'\n",
        "[checkpoint]\ndirectory='ckpt'\nmode='best_eval'\nevery_steps=2\n[evaluation]\ndata='eval.txt'\n",
        "[checkpoint]\ndirectory='ckpt'\nmode='sometimes'\n",
    ] {
        let file = write_config(&format!("{base}{section}"));
        assert!(
            load(&file).is_err(),
            "section should be rejected: {section}"
        );
        remove_config(&file);
    }
}

#[test]
fn algorithm_sections_no_longer_accept_evaluation_data() {
    let file = write_config(concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[sft]\ndata='train.txt'\neval_data='eval.txt'\n",
    ));
    assert!(
        load(&file)
            .unwrap_err()
            .to_string()
            .contains("invalid TOML")
    );
    remove_config(&file);
}

#[test]
fn an_omitted_or_unit_duty_cycle_normalizes_to_the_unthrottled_path() {
    // The two say the same thing - no limit - and collapsing them here is
    // what keeps a single disabled path in the runtime rather than one that
    // installs a limiter and then never sleeps.
    for training in ["", "[training]\nmax_gpu_duty_cycle=1.0\n"] {
        let file = write_config(&format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n\
             [lora]\noutput='out.gguf'\n{training}[sft]\ndata='data.txt'\n"
        ));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.max_gpu_duty_cycle, None, "{training:?}");
        remove_config(&file);
    }

    let file = write_config(concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nmax_gpu_duty_cycle=0.5\n",
        "[sft]\ndata='data.txt'\n",
    ));
    assert_eq!(load(&file).unwrap().training.max_gpu_duty_cycle, Some(0.5));
    remove_config(&file);
}

#[test]
fn a_cpu_device_accepts_a_duty_cycle_it_cannot_honour() {
    // `auto` can resolve to CPU too, so rejecting the document at parse
    // time would be wrong. The runtime keeps the requested value and its
    // report says `active: false, reason: cpu_backend` instead.
    let file = write_config(concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\ndevice='cpu'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nmax_gpu_duty_cycle=0.25\n",
        "[sft]\ndata='data.txt'\n",
    ));
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.training.max_gpu_duty_cycle, Some(0.25));
    assert_eq!(loaded.training.device, Device::Cpu);
    remove_config(&file);
}

#[test]
fn training_gradient_clip_is_configurable() {
    let file = write_config(concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nmax_grad_norm=0.5\nthreads=6\n",
        "[sft]\ndata='data.txt'\n",
    ));
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.training.max_grad_norm, 0.5);
    assert_eq!(loaded.training.threads, 6);
    remove_config(&file);
}

/// The fused cross-entropy is on with a bounded token chunk unless the file
/// says otherwise, and both knobs have to survive the TOML round trip: a
/// silently dropped key would look exactly like a feature that does not
/// work.
#[test]
fn chunked_ce_is_on_by_default_and_round_trips() {
    let source = |training: &str| {
        format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
        )
    };

    let defaults = write_config(&source(""));
    let loaded = load(&defaults).unwrap();
    assert!(loaded.training.chunked_cross_entropy);
    assert_eq!(
        loaded.training.chunked_ce_seq_chunk,
        retrograd_core::DEFAULT_CE_SEQ_CHUNK
    );
    remove_config(&defaults);

    let tuned = write_config(&source(concat!(
        "chunked_cross_entropy=false\nchunked_ce_tiles=4\n",
        "chunked_ce_seq_chunk=256\n",
    )));
    let loaded = load(&tuned).unwrap();
    assert!(!loaded.training.chunked_cross_entropy);
    assert_eq!(loaded.training.chunked_ce_tiles, 4);
    assert_eq!(loaded.training.chunked_ce_seq_chunk, 256);
    remove_config(&tuned);
}

/// The in-place `grad_h` write follows `chunked_ce_seq_chunk` now, so the
/// key is rejected rather than accepted and ignored.
#[test]
fn the_offload_logsoftmax_key_is_rejected() {
    let file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n\
         [training]\nchunked_ce_offload_logsoftmax=true\n[sft]\ndata='data.txt'\n",
    );
    let error = load(&file).unwrap_err().to_string();
    assert!(error.contains("chunked_ce_offload_logsoftmax"), "{error}");
    remove_config(&file);
}

/// A 16-bit checkpoint in a run that retains no checkpoints is a request for
/// less precision that nothing would honour, and no artifact of the run
/// would show it was dropped.
#[test]
fn a_16bit_checkpoint_dtype_needs_checkpointing() {
    let source = |training: &str| {
        format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
        )
    };

    let orphan = write_config(&source("checkpoint_dtype='f16'\n"));
    let error = load(&orphan).unwrap_err().to_string();
    assert!(error.contains("checkpoint_dtype"), "{error}");
    remove_config(&orphan);

    let paired = write_config(&source(
        "gradient_checkpointing=true\ncheckpoint_dtype='f16'\n",
    ));
    assert_eq!(
        load(&paired).unwrap().training.checkpoint_dtype,
        CheckpointDtype::F16
    );
    remove_config(&paired);

    // F32 is the default, so naming it explicitly is not a request for
    // anything and must not depend on checkpointing.
    let explicit = write_config(&source("checkpoint_dtype='f32'\n"));
    assert_eq!(
        load(&explicit).unwrap().training.checkpoint_dtype,
        CheckpointDtype::F32
    );
    remove_config(&explicit);
}

#[test]
fn gradient_checkpointing_is_opt_in_and_validates_its_interval() {
    let source = |training: &str| {
        format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
        )
    };

    let defaults = write_config(&source(""));
    let loaded = load(&defaults).unwrap();
    assert!(!loaded.training.gradient_checkpointing);
    // Not 1: a checkpoint on every layer is the largest retained term
    // checkpointing can produce, for the same extra forward.
    assert_eq!(
        loaded.training.checkpoint_every_n_layers,
        retrograd_core::DEFAULT_CHECKPOINT_STRIDE
    );
    remove_config(&defaults);

    let enabled = write_config(&source(
        "gradient_checkpointing=true\ncheckpoint_every_n_layers=3\n",
    ));
    let loaded = load(&enabled).unwrap();
    assert!(loaded.training.gradient_checkpointing);
    assert_eq!(loaded.training.checkpoint_every_n_layers, 3);
    remove_config(&enabled);

    let invalid = write_config(&source("checkpoint_every_n_layers=0\n"));
    assert!(
        load(&invalid)
            .unwrap_err()
            .to_string()
            .contains("checkpoint_every_n_layers")
    );
    remove_config(&invalid);
}

#[test]
fn fast_sampling_context_defaults_to_fast_and_can_be_disabled() {
    let source = |training: &str| {
        format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n{training}[sft]\ndata='data.txt'\n"
        )
    };
    for (training, expected) in [
        ("", true),
        ("[training]\nfast_sampling_context=false\n", false),
        ("[training]\nfast_sampling_context=true\n", true),
    ] {
        let file = write_config(&source(training));
        let loaded = load(&file).unwrap();
        assert_eq!(
            loaded.training.fast_generation_context, expected,
            "{training:?}"
        );
        remove_config(&file);
    }
}

#[test]
fn training_kv_dtype_is_f16_by_default_and_accepts_f32() {
    let source = |training: &str| {
        format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
        )
    };

    // The half-precision cache is the default: it halves the one term that
    // grows with the context, and the runtime probes the device before
    // applying it rather than assuming.
    let defaults = write_config(&source(""));
    assert_eq!(load(&defaults).unwrap().training.kv_dtype, KvDtype::F16);
    remove_config(&defaults);

    let f32 = write_config(&source("kv_dtype='f32'\n"));
    assert_eq!(load(&f32).unwrap().training.kv_dtype, KvDtype::F32);
    remove_config(&f32);

    let invalid = write_config(&source("kv_dtype='bf16'\n"));
    assert!(load(&invalid).unwrap_err().to_string().contains("kv_dtype"));
    remove_config(&invalid);
}

#[test]
fn training_checkpoint_dtype_defaults_to_f32_and_rejects_unknown_precisions() {
    let source = |training: &str| {
        format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}[sft]\ndata='data.txt'\n"
        )
    };

    // The bit-exact recompute stays the default: narrowing costs real gradient
    // fidelity, so it must be asked for.
    let defaults = write_config(&source(""));
    assert_eq!(
        load(&defaults).unwrap().training.checkpoint_dtype,
        CheckpointDtype::F32
    );
    remove_config(&defaults);

    for (spelling, expected) in [
        ("f16", CheckpointDtype::F16),
        ("bf16", CheckpointDtype::Bf16),
    ] {
        let file = write_config(&source(&format!(
            "gradient_checkpointing=true\ncheckpoint_dtype='{spelling}'\n"
        )));
        assert_eq!(load(&file).unwrap().training.checkpoint_dtype, expected);
        remove_config(&file);
    }

    // A silently ignored precision would read as a memory win that never
    // happened, so an unknown spelling has to fail loudly.
    let invalid = write_config(&source(
        "gradient_checkpointing=true\ncheckpoint_dtype='q8_0'\n",
    ));
    assert!(
        load(&invalid)
            .unwrap_err()
            .to_string()
            .contains("checkpoint_dtype")
    );
    remove_config(&invalid);
}

#[test]
fn scientific_notation_is_accepted_for_float_settings() {
    let file = write_config(concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\nalpha=1.6e1\n",
        "[training]\nlr=5e-6\nweight_decay=1E-2\nmax_grad_norm=1.5e0\n",
        "[sft]\ndata='data.txt'\n",
    ));
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.training.learning_rate, 5e-6);
    assert_eq!(loaded.training.weight_decay, 1e-2);
    assert_eq!(loaded.training.max_grad_norm, 1.5);
    assert_eq!(loaded.lora.config.alpha, 16.0);
    remove_config(&file);
}

#[test]
fn init_adapter_resolves_and_rejects_creation_keys() {
    let file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ninit_adapter='adapter.gguf'\n[sft]\ndata='data.txt'\n",
    );
    let loaded = load(&file).unwrap();
    assert_eq!(
        loaded.lora.init_adapter,
        Some(file.parent().unwrap().join("adapter.gguf"))
    );
    remove_config(&file);

    for conflicting in [
        "rank=4",
        "alpha=8.0",
        "seed=1",
        "dtype='f16'",
        "targets=['q']",
    ] {
        let source = format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\ninit_adapter='adapter.gguf'\n{conflicting}\n[sft]\ndata='data.txt'\n"
        );
        let file = write_config(&source);
        let error = load(&file).unwrap_err();
        assert!(
            error.to_string().contains("init_adapter"),
            "{conflicting}: {error}"
        );
        remove_config(&file);
    }
}

#[test]
fn rejects_unknown_keys_before_model_loading() {
    let file = write_config(
        "[run]\nalgorithm='sft'\nunknown=true\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[sft]\ndata='data.txt'\n",
    );
    assert!(
        load(&file)
            .unwrap_err()
            .to_string()
            .contains("invalid TOML")
    );
    remove_config(&file);
}

#[test]
fn training_and_lora_values_are_rejected_before_model_loading() {
    let cases = [
        ("ctx=0", "training.ctx must be greater than zero"),
        (
            "micro_batch=0",
            "training.micro_batch must be greater than zero",
        ),
        (
            "gradient_accumulation=0",
            "training.gradient_accumulation must be greater than zero",
        ),
        // 96 tokens per step does not tile a 128-token context.
        (
            "micro_batch=48\ngradient_accumulation=2",
            "must be a multiple of the optimizer window",
        ),
        ("lr=0.0", "lr must be finite"),
        ("lr=nan", "lr must be finite"),
        ("weight_decay=-0.1", "weight_decay"),
        ("weight_decay=nan", "weight_decay"),
        ("max_grad_norm=0.0", "max_grad_norm"),
        ("max_grad_norm=-1.0", "max_grad_norm"),
        ("max_grad_norm=nan", "max_grad_norm"),
        // Zero is not a spelling for pause: the run-control pause operation
        // is already the safe way to stop a live run.
        ("max_gpu_duty_cycle=0.0", "max_gpu_duty_cycle"),
        ("max_gpu_duty_cycle=-0.5", "max_gpu_duty_cycle"),
        ("max_gpu_duty_cycle=1.5", "max_gpu_duty_cycle"),
        ("max_gpu_duty_cycle=nan", "max_gpu_duty_cycle"),
        ("max_gpu_duty_cycle=inf", "max_gpu_duty_cycle"),
    ];
    for (training, expected) in cases {
        let source = format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n[training]\n{training}\n[sft]\ndata='data.txt'\n"
        );
        let file = write_config(&source);
        let error = load(&file).unwrap_err();
        assert!(error.to_string().contains(expected), "{training}: {error}");
        remove_config(&file);
    }

    for alpha in ["0.0", "-1.0", "nan"] {
        let source = format!(
            "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\nalpha={alpha}\n[sft]\ndata='data.txt'\n"
        );
        let file = write_config(&source);
        assert!(load(&file).unwrap_err().to_string().contains("lora.alpha"));
        remove_config(&file);
    }
}

#[test]
fn critic_defaults_and_boundaries_are_validated() {
    let defaults = critic(CriticToml::default()).unwrap();
    assert!(defaults.enabled);
    assert_eq!(defaults.gamma, 1.0);
    assert_eq!(defaults.gae_lambda, 0.95);
    assert_eq!(defaults.value_lr, 1.0e-2);
    assert_eq!(defaults.value_epochs, 8);
    // O8 is opt-in: the default keeps the feature matrix in F32, so enabling
    // a critic never silently rounds what the value head regresses on.
    assert_eq!(defaults.feature_dtype, FeatureDtype::F32);
    for (spelling, expected) in [
        ("f32", FeatureDtype::F32),
        ("f16", FeatureDtype::F16),
        ("bf16", FeatureDtype::Bf16),
    ] {
        let parsed: FeatureDtype = toml::from_str(&format!("v='{spelling}'"))
            .map(|table: toml::Value| table["v"].clone())
            .and_then(FeatureDtype::deserialize)
            .expect("a documented spelling parses");
        assert_eq!(parsed, expected);
    }
    // A narrowing asked for on a run with no critic has no matrix to narrow;
    // silently ignoring it would leave the document uncontradicted.
    assert!(
        critic(CriticToml {
            enabled: Some(false),
            feature_dtype: Some(FeatureDtype::F16),
            ..CriticToml::default()
        })
        .is_err()
    );
    assert!(
        critic(CriticToml {
            enabled: Some(false),
            ..CriticToml::default()
        })
        .is_ok()
    );

    for invalid in [0.0, -0.1, 1.1, f32::NAN] {
        assert!(
            critic(CriticToml {
                gamma: Some(invalid),
                ..CriticToml::default()
            })
            .is_err()
        );
    }
    for invalid in [-0.1, 1.1, f32::NAN] {
        assert!(
            critic(CriticToml {
                gae_lambda: Some(invalid),
                ..CriticToml::default()
            })
            .is_err()
        );
    }
    assert!(
        critic(CriticToml {
            value_epochs: Some(0),
            ..CriticToml::default()
        })
        .is_err()
    );
    for invalid in [0.0, -0.1, f32::NAN, f32::INFINITY] {
        assert!(
            critic(CriticToml {
                value_lr: Some(invalid),
                ..CriticToml::default()
            })
            .is_err()
        );
    }
}

#[test]
fn ppo_validation_rejects_invalid_geometry_and_objective_values() {
    build_ppo(valid_ppo_toml(), Path::new("/config")).unwrap();

    let mut value = valid_ppo_toml();
    value.reward_command.clear();
    assert!(build_ppo(value, Path::new("/config")).is_err());

    let mut value = valid_ppo_toml();
    value.rollout_batch_size = 0;
    assert!(build_ppo(value, Path::new("/config")).is_err());

    for invalid in [0.0, -0.1, 1.0, f32::NAN, f32::INFINITY] {
        let mut value = valid_ppo_toml();
        value.clip_range = invalid;
        assert!(
            build_ppo(value, Path::new("/config")).is_err(),
            "clip={invalid}"
        );
    }
    for invalid in [-0.1, f32::NAN, f32::INFINITY] {
        let mut value = valid_ppo_toml();
        value.kl_coefficient = invalid;
        assert!(
            build_ppo(value, Path::new("/config")).is_err(),
            "kl={invalid}"
        );
    }
}

/// The transport of the reward command: defaulted when the document says
/// nothing, spelled the same way in both sections, and refused at zero -
/// a deadline of zero would fail every batch on its first millisecond.
#[test]
fn the_reward_transport_defaults_to_one_persistent_worker() {
    let ppo = build_ppo(valid_ppo_toml(), Path::new("/config")).unwrap();
    assert_eq!(ppo.reward_protocol, RewardProtocol::default());
    assert_eq!(ppo.reward_protocol.mode, RewardMode::Persistent);

    let mut value = valid_ppo_toml();
    value.reward_mode = Some(RewardMode::OneShot);
    value.reward_timeout_seconds = Some(12);
    let ppo = build_ppo(value, Path::new("/config")).unwrap();
    assert_eq!(ppo.reward_protocol.mode, RewardMode::OneShot);
    assert_eq!(
        ppo.reward_protocol.timeout,
        std::time::Duration::from_secs(12)
    );

    let mut value = valid_ppo_toml();
    value.reward_timeout_seconds = Some(0);
    let error = build_ppo(value, Path::new("/config")).unwrap_err();
    assert!(
        error.to_string().contains("ppo.reward_timeout_seconds"),
        "{error}"
    );

    // The wire spelling is the one the documents use, and an unknown one is
    // refused by name rather than defaulted to.
    let document: GrpoToml = toml::from_str(
        "prompts = 'p.jsonl'\nreward_command = ['r']\nreward_mode = 'oneshot'\n\
         updates = 1\nprompts_per_update = 1\ngroup_size = 2\ngrpo_epochs = 1\n\
         clip_range_low = 0.2\nclip_range_high = 0.2\nkl_coefficient = 0.0\n\
         [sampling]\ntemperature = 1.0\ntop_p = 1.0\nmax_new_tokens = 8\nseed = 1\n",
    )
    .unwrap();
    assert_eq!(document.reward_mode, Some(RewardMode::OneShot));
    let mut value = document.clone();
    value.reward_timeout_seconds = Some(0);
    let error = build_grpo(value, Path::new("/config")).unwrap_err();
    assert!(
        error.to_string().contains("grpo.reward_timeout_seconds"),
        "{error}"
    );
    let grpo = build_grpo(document, Path::new("/config")).unwrap();
    assert_eq!(grpo.reward_protocol.mode, RewardMode::OneShot);
    assert_eq!(
        grpo.reward_protocol.timeout,
        std::time::Duration::from_secs(DEFAULT_REWARD_TIMEOUT_SECONDS)
    );
    assert!(
        toml::from_str::<GrpoToml>(
            "prompts = 'p.jsonl'\nreward_command = ['r']\nreward_mode = 'socket'\n\
             updates = 1\nprompts_per_update = 1\ngroup_size = 2\ngrpo_epochs = 1\n\
             clip_range_low = 0.2\nclip_range_high = 0.2\nkl_coefficient = 0.0\n\
             [sampling]\ntemperature = 1.0\ntop_p = 1.0\nmax_new_tokens = 8\nseed = 1\n",
        )
        .is_err()
    );
}

#[test]
fn parsers_accept_documented_aliases_and_reject_unknown_values() {
    assert_eq!(
        parse_scheduler(" CONSTANT ").unwrap(),
        LrScheduler::Constant
    );
    assert_eq!(parse_scheduler("linear").unwrap(), LrScheduler::Linear);
    assert_eq!(parse_scheduler("Cosine").unwrap(), LrScheduler::Cosine);
    assert!(parse_scheduler("cyclic").is_err());

    for alias in ["text", "txt"] {
        assert_eq!(parse_data_format(alias).unwrap(), DataFormat::Text);
    }
    for alias in ["jsonl", "chat", "chat-jsonl"] {
        assert_eq!(parse_data_format(alias).unwrap(), DataFormat::ChatJsonl);
    }
    assert!(parse_data_format("csv").is_err());
}

#[test]
fn selected_algorithm_requires_exactly_its_own_section() {
    for (algorithm, section, expected) in [
        ("sft", "", "[sft] is required"),
        ("ppo", "", "[ppo] is required"),
        ("grpo", "", "[grpo] is required"),
        ("agent_grpo", "", "[agent] is required"),
        ("unknown", "", "must be one of"),
        (
            "sft",
            "[sft]\ndata='data.txt'\n[ppo]\nprompts='p.jsonl'\nreward_command=['r']\nupdates=1\nrollout_batch_size=1\nppo_epochs=1\nclip_range=0.2\nkl_coefficient=0.0\n[ppo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=1\nseed=1\n",
            "only the section",
        ),
    ] {
        let source = format!(
            "[run]\nalgorithm='{algorithm}'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n{section}"
        );
        let file = write_config(&source);
        let error = load(&file).unwrap_err();
        assert!(error.to_string().contains(expected), "{algorithm}: {error}");
        remove_config(&file);
    }
}

/// The agentic loop shares one document with the other algorithms. Shared
/// sections use the common schema, while sections it cannot honor are
/// rejected explicitly.
#[test]
fn an_agentic_document_shares_every_section_it_can_and_refuses_the_rest() {
    let base = concat!(
        "[run]\nalgorithm='agent_grpo'\n",
        "[model]\npath='model.gguf'\ndevice='cpu'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nctx=2048\nmicro_batch=64\nlr=1e-5\nlr_scheduler='constant'\n",
        "[metrics]\ntensorboard_dir='tb'\n",
        "[agent]\nscenarios='s.jsonl'\nupdates=3\nscenarios_per_update=2\n",
        "group_size=4\nepochs_per_update=2\nmax_new_tokens_per_turn=128\n",
        "system_suffix='Reply with one tool call.'\n",
        "template_variables={ enable_thinking = false }\n",
        "[agent.judge]\ntype='command'\ncommand=['python','judge.py']\n",
    );
    let file = write_config(base);
    let config = load(&file).expect("the agentic document must load");
    remove_config(&file);
    let Algorithm::AgentGrpo(agent) = &config.algorithm else {
        panic!("expected an agentic algorithm");
    };
    assert_eq!(agent.config.updates, 3);
    assert_eq!(agent.config.epochs, 2);
    assert_eq!(agent.config.limits.max_new_tokens_per_turn, 128);
    assert_eq!(agent.system_suffix, "Reply with one tool call.");
    // A TOML table crosses into the JSON object the runtime hands the template; `false`
    // must stay a boolean, not become the string "false", because a template branching on
    // it would read any string as truthy.
    assert_eq!(
        agent.template_variables_json(),
        r#"{"enable_thinking":false}"#
    );
    assert!(agent.scenarios.ends_with("s.jsonl"));
    // The shared sections are read by the same code as every other
    // algorithm.
    assert_eq!(config.training.lr_scheduler, LrScheduler::Constant);
    assert_eq!(config.training.device, retrograd_core::Device::Cpu);
    assert!(config.metrics.tensorboard_dir.is_some());
    // A rollout objective: one optimizer step spans the whole context.
    assert_eq!(config.training.n_batch, config.training.n_ctx);
    // Agent GRPO uses the same batched generation geometry as single-turn
    // GRPO: both scenarios' four-member groups fit in one decode wave.
    assert_eq!(config.training.n_seq_max, 4);
    assert_eq!(config.training.generation_concurrency, 8);
    // Unset means "the whole context", decided against the loaded model.
    assert_eq!(agent.trajectory_limit(2048).unwrap(), 2048);

    for (extra, expected) in [
        // Nothing grades a trajectory here, so there is no evaluation to be
        // had - see the loader's comment on why the judge cannot stand in.
        (
            "[evaluation]\ndata='eval.jsonl'\n",
            "needs [agent.environment]",
        ),
        (
            "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\nupdates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\nclip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=1\nseed=1\n",
            "only the section",
        ),
        // An unknown key is rejected rather than treated as a second dialect.
        (
            "[agent.environment]\ntype='local'\nallow_unsandboxd=true\n",
            "invalid TOML",
        ),
    ] {
        let file = write_config(&format!("{base}{extra}"));
        let error = load(&file).unwrap_err().to_string();
        remove_config(&file);
        assert!(error.contains(expected), "expected {expected}, got {error}");
    }
}

/// Evaluation and checkpointing are shared machinery, and an agentic run
/// uses the same sections for them as every other algorithm - as long as
/// something grades its trajectories.
#[test]
fn an_agentic_run_evaluates_and_checkpoints_like_any_other() {
    let source = concat!(
        "[run]\nalgorithm='agent_grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nctx=2048\nmicro_batch=64\n",
        "[agent]\nscenarios='s.jsonl'\ngroup_size=4\n",
        "[agent.judge]\ntype='command'\ncommand=['judge']\n",
        "[agent.environment]\ntype='local'\nallow_unsandboxed=true\n",
        "[evaluation]\ndata='eval.jsonl'\nevery_iterations=2\npatience=3\n",
        "[checkpoint]\ndirectory='ckpt'\nmode='steps_and_best_eval'\nevery_steps=10\n",
    );
    let file = write_config(source);
    let config = load(&file).expect("an environment-graded agentic run may be evaluated");
    remove_config(&file);
    let evaluation = config.evaluation.expect("[evaluation] is kept");
    assert_eq!(evaluation.every_iterations, 2);
    assert_eq!(evaluation.patience, Some(3));
    let checkpoint = config.checkpoint.expect("[checkpoint] is kept");
    assert!(checkpoint.mode.includes_steps() && checkpoint.mode.includes_best_eval());
}

/// A trajectory budget the model cannot hold is a configuration to fix, not
/// a number to clamp: the loss denominator is derived from it.
#[test]
fn an_explicit_trajectory_budget_is_checked_against_the_model_context() {
    let source = concat!(
        "[run]\nalgorithm='agent_grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nctx=2048\nmicro_batch=64\n",
        "[agent]\nscenarios='s.jsonl'\nmax_trajectory_tokens=4096\n",
        "[agent.judge]\ntype='command'\ncommand=['judge']\n",
    );
    let file = write_config(source);
    let config = load(&file).expect("the document itself is valid");
    remove_config(&file);
    let Algorithm::AgentGrpo(agent) = &config.algorithm else {
        panic!("expected an agentic algorithm");
    };
    assert_eq!(agent.trajectory_limit(8192).unwrap(), 4096);
    assert!(
        agent
            .trajectory_limit(2048)
            .unwrap_err()
            .to_string()
            .contains("exceeds the model context")
    );
}

/// Scenarios and *a source of reward* are the two things a rollout cannot
/// invent, and `[agent]`'s defaults would otherwise stand in for both. The
/// second is a pair: a judge, an environment that grades its own steps, or
/// both - never neither.
#[test]
fn an_agentic_run_without_scenarios_or_any_reward_is_refused() {
    for (section, expected) in [
        ("[agent]\nupdates=1\n", "agent.scenarios is required"),
        (
            "[agent]\nscenarios='s.jsonl'\n",
            "neither a judge nor an environment",
        ),
        // Declared and empty is a mistake, not a way of asking for no
        // judge: the way of asking is not to declare the table.
        (
            "[agent]\nscenarios='s.jsonl'\n[agent.judge]\ntype='command'\ncommand=[]\n",
            "declares an empty command",
        ),
    ] {
        let source = format!(
            "[run]\nalgorithm='agent_grpo'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n{section}"
        );
        let file = write_config(&source);
        let error = load(&file).unwrap_err().to_string();
        remove_config(&file);
        assert!(error.contains(expected), "expected {expected}, got {error}");
    }
}

/// The point of the change: a verifiable task declares an environment and
/// no judge, and that is a complete document.
#[test]
fn an_environment_graded_agentic_run_needs_no_judge() {
    let source = concat!(
        "[run]\nalgorithm='agent_grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[training]\nctx=2048\nmicro_batch=64\n",
        "[agent]\nscenarios='s.jsonl'\ngroup_size=4\n",
        "[agent.environment]\ntype='http'\nbase_url='http://127.0.0.1:8099'\n",
    );
    let file = write_config(source);
    let config = load(&file).expect("an environment grades this run on its own");
    remove_config(&file);
    let Algorithm::AgentGrpo(agent) = &config.algorithm else {
        panic!("expected an agentic algorithm");
    };
    assert!(agent.judge.is_none(), "no [agent.judge] was declared");
}

#[test]
fn grpo_optional_features_parse_and_validate() {
    let base = concat!(
        "[run]\nalgorithm='grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
    );
    let grpo = concat!(
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=4\nprompts_per_update=2\ngroup_size=4\ngrpo_epochs=2\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.02\n",
        "baseline='rloo'\nprompt_order='shuffled'\n",
        "overlong_penalty={buffer_tokens=4,max_penalty=1.0}\n",
        "kl_schedule={warmup_updates=2,target=0.05}\n",
        "dynamic_sampling={max_resample_factor=3}\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=16\nseed=1\n",
    );
    let file = write_config(&format!("{base}{grpo}"));
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.training.n_seq_max, 4);
    assert_eq!(loaded.training.generation_concurrency, 8);
    let Algorithm::Grpo(config) = loaded.algorithm else {
        panic!("expected a grpo config")
    };
    assert_eq!(config.baseline, AdvantageBaseline::LeaveOneOut);
    assert_eq!(config.prompt_order, PromptOrder::Shuffled);
    let penalty = config.overlong_penalty.unwrap();
    assert_eq!(penalty.buffer_tokens, 4);
    assert_eq!(penalty.max_penalty, 1.0);
    let schedule = config.kl_schedule.unwrap();
    assert_eq!(schedule.warmup_updates, 2);
    assert_eq!(schedule.target, Some(0.05));
    assert_eq!(config.dynamic_sampling.unwrap().max_resample_factor, 3);
    remove_config(&file);
}

#[test]
fn observe_resolves_its_directory_and_defaults() {
    let source = concat!(
        "[run]\nalgorithm='grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=16\nseed=1\n",
        "[observe]\ndirectory='observe'\n",
    );
    let file = write_config(source);
    let config = load(&file).unwrap();
    let observe = config.observe.expect("[observe] was declared");
    assert_eq!(observe.directory, file.parent().unwrap().join("observe"));
    assert_eq!(observe.every, 1);
    assert_eq!(observe.max_text_chars, 0);
    remove_config(&file);

    let file = write_config(&source.replace("'observe'\n", "'observe'\nevery=0\n"));
    let error = load(&file).unwrap_err().to_string();
    remove_config(&file);
    assert!(error.contains("observe.every"), "{error}");
}

#[test]
fn observe_is_refused_where_there_is_no_rollout() {
    let source = concat!(
        "[run]\nalgorithm='sft'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[sft]\ndata='data.txt'\n",
        "[observe]\ndirectory='observe'\n",
    );
    let file = write_config(source);
    let error = load(&file).unwrap_err().to_string();
    remove_config(&file);
    assert!(error.contains("[observe]"), "{error}");
}

/// `[grpo.log_completions]` was removed without an alias: the standard
/// serde error has to keep naming the key.
#[test]
fn the_removed_completion_log_is_named_in_the_error() {
    let source = concat!(
        "[run]\nalgorithm='grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=16\nseed=1\n",
        "[grpo.log_completions]\nevery=1\npath='c.jsonl'\n",
    );
    let error = parse_toml(source, "old.toml").unwrap_err().to_string();
    assert!(error.contains("log_completions"), "{error}");
}

/// `[grpo.judge]` is `[agent.judge]`, in the section of the other loop: the
/// same table, the same spelling, and the cache path rebased against the
/// document like every other relative path it declares.
#[test]
fn grpo_judge_parses_next_to_the_reward_command() {
    let base = concat!(
        "[run]\nalgorithm='grpo'\n",
        "[model]\npath='model.gguf'\n",
        "[lora]\noutput='out.gguf'\n",
    );
    let grpo = concat!(
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
        "judge_weight=0.3\njudge_failure='fail'\nmax_judge_dropped_fraction=0.25\n",
        "[grpo.judge]\ntype='ruler'\nbase_url='http://x/v1'\nmodel='lite'\n",
        "cache_path='judge.cache'\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
    );
    let file = write_config(&format!("{base}{grpo}"));
    let loaded = load(&file).unwrap();
    let Algorithm::Grpo(config) = loaded.algorithm else {
        panic!("expected a grpo config")
    };
    let judge = config.judge.expect("a judge was declared");
    assert_eq!(judge.weight, 0.3);
    assert_eq!(judge.max_dropped_fraction, 0.25);
    assert!(matches!(
        judge.failure,
        retrograd_agent_core::config::JudgeFailurePolicy::Fail
    ));
    let retrograd_spec::judge::JudgeConfig::Ruler { config } = judge.config else {
        panic!("expected a RULER judge")
    };
    assert_eq!(
        config.cache_path.as_deref(),
        Some(file.parent().unwrap().join("judge.cache").as_path())
    );
    remove_config(&file);

    // The weight is what the verdict is worth, and there is no honest
    // default for it; and none of the three keys means anything without the
    // section they govern.
    for (source, expected) in [
        (
            concat!(
                "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
                "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
                "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
                "[grpo.judge]\ntype='ruler'\nbase_url='http://x/v1'\nmodel='lite'\n",
                "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
            ),
            "grpo.judge_weight is required",
        ),
        (
            concat!(
                "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
                "updates=1\nprompts_per_update=1\ngroup_size=2\ngrpo_epochs=1\n",
                "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
                "judge_weight=0.3\n",
                "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
            ),
            "need a [grpo.judge] section",
        ),
    ] {
        let file = write_config(&format!("{base}{source}"));
        let error = load(&file).unwrap_err().to_string();
        remove_config(&file);
        assert!(error.contains(expected), "expected {expected}, got {error}");
    }
}

#[test]
fn grpo_shared_prefix_fanout_parses_and_respects_group_size() {
    let grpo = concat!(
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nprompts_per_update=1\ngroup_size=4\ngrpo_epochs=1\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
    );
    let source = |fanout: &str| {
        format!(
            "[run]\nalgorithm='grpo'\n[model]\npath='model.gguf'\n\
         [lora]\noutput='out.gguf'\n[training]\nctx=256\nmicro_batch=32\n\
         shared_prefix_fanout={fanout}\n{grpo}"
        )
    };

    for (value, expected) in [
        ("'auto'", SharedPrefixFanout::Auto),
        ("'off'", SharedPrefixFanout::Off),
        ("'max'", SharedPrefixFanout::Max),
        ("3", SharedPrefixFanout::Exact(3)),
    ] {
        let file = write_config(&source(value));
        let loaded = load(&file).unwrap();
        assert_eq!(loaded.training.shared_prefix_fanout, expected);
        remove_config(&file);
    }

    for (value, expected) in [
        ("1", "at least 2"),
        ("5", "exceeds grpo.group_size"),
        ("'fast'", "must be 'auto', 'off', 'max'"),
    ] {
        let file = write_config(&source(value));
        let error = load(&file).unwrap_err().to_string();
        assert!(error.contains(expected), "expected {expected}, got {error}");
        remove_config(&file);
    }
}

#[test]
fn grpo_generation_concurrency_is_independent_and_validated() {
    let grpo = concat!(
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nprompts_per_update=2\ngroup_size=4\ngrpo_epochs=1\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
    );
    let source = |concurrency: u32, batch: u32| {
        format!(
            "[run]\nalgorithm='grpo'\n[model]\npath='model.gguf'\n\
             [lora]\noutput='out.gguf'\n[training]\nctx={batch}\n\
             micro_batch=4\ngeneration_concurrency={concurrency}\n{grpo}"
        )
    };

    let file = write_config(&source(1, 256));
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.training.n_seq_max, 4);
    assert_eq!(loaded.training.generation_concurrency, 1);
    remove_config(&file);

    for (concurrency, batch, expected) in [
        (0, 256, "must be greater than zero"),
        (9, 256, "must not exceed the 8 grpo rollouts"),
        (8, 4, "must not exceed the optimizer window"),
        (257, 512, "must not exceed 256"),
    ] {
        let file = write_config(&source(concurrency, batch));
        let error = load(&file).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
        remove_config(&file);
    }
}

/// An optimizer window below `ctx` turns one rollout into several steps,
/// most of them over prompt positions carrying no label, and the policy
/// leaves its trust region inside the first update. So a rollout algorithm
/// pins `gradient_accumulation` to `ctx / micro_batch`. SFT is unaffected:
/// there a row is a document, and stepping through it several times is the
/// point.
#[test]
fn a_rollout_algorithm_pins_the_optimizer_step_to_the_trained_window() {
    let grpo = concat!(
        "[grpo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nprompts_per_update=2\ngroup_size=4\ngrpo_epochs=1\n",
        "clip_range_low=0.2\nclip_range_high=0.28\nkl_coefficient=0.0\n",
        "[grpo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
    );
    let ppo = concat!(
        "[ppo]\nprompts='p.jsonl'\nreward_command=['r']\n",
        "updates=1\nrollout_batch_size=2\nppo_epochs=1\n",
        "clip_range=0.2\nkl_coefficient=0.0\n",
        "[ppo.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=1\n",
    );
    let source = |algorithm: &str, section: &str, accumulation: &str| {
        format!(
            "[run]\nalgorithm='{algorithm}'\n[model]\npath='model.gguf'\n\
             [lora]\noutput='out.gguf'\n[training]\nctx=256\n\
             micro_batch=4\n{accumulation}{section}"
        )
    };

    for (algorithm, section) in [("grpo", grpo), ("ppo", ppo)] {
        let file = write_config(&source(algorithm, section, "gradient_accumulation=16\n"));
        let error = load(&file).unwrap_err().to_string();
        assert!(error.contains("one optimizer step per rollout"), "{error}");
        assert!(
            error.contains("pinned to ctx / micro_batch = 64"),
            "{error}"
        );
        remove_config(&file);

        // Omitted, it resolves to the only admissible value.
        let file = write_config(&source(algorithm, section, ""));
        let loaded = load(&file).unwrap_or_else(|error| panic!("{algorithm}: {error}"));
        assert_eq!(loaded.training.n_batch, 256);
        assert_eq!(loaded.training.gradient_accumulation(), 64);
        remove_config(&file);
    }

    // SFT keeps its accumulation window as a free parameter.
    let file = write_config(
        "[run]\nalgorithm='sft'\n[model]\npath='model.gguf'\n[lora]\noutput='out.gguf'\n\
         [training]\nctx=256\nmicro_batch=4\ngradient_accumulation=16\n\
         [sft]\ndata='data.txt'\n",
    );
    let loaded = load(&file).unwrap();
    assert_eq!(loaded.training.n_batch, 64);
    remove_config(&file);
}

#[test]
fn grpo_optional_feature_boundaries_are_rejected() {
    let mut config = valid_grpo();
    config.overlong_penalty = Some(OverlongPenalty {
        buffer_tokens: 8, // == max_new_tokens
        max_penalty: 1.0,
    });
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("buffer_tokens")
    );

    let mut config = valid_grpo();
    config.overlong_penalty = Some(OverlongPenalty {
        buffer_tokens: 2,
        max_penalty: 0.0,
    });
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_penalty")
    );

    // A KL schedule with a zero base coefficient is contradictory.
    let mut config = valid_grpo();
    config.kl_coefficient = 0.0;
    config.kl_schedule = Some(KlSchedule {
        warmup_updates: 1,
        target: None,
    });
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("kl_schedule requires")
    );

    // A resample factor of 1 permits no resampling and is rejected.
    let mut config = valid_grpo();
    config.dynamic_sampling = Some(DynamicSampling {
        max_resample_factor: 1,
    });
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("max_resample_factor")
    );
}

#[test]
fn grpo_validation_enforces_group_geometry_and_on_policy_sampling() {
    valid_grpo().validate().unwrap();

    let mut config = valid_grpo();
    config.group_size = 1;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("at least 2")
    );

    let mut config = valid_grpo();
    config.clip_range_high = 0.1;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("Clip-Higher")
    );

    let mut config = valid_grpo();
    config.sampling.temperature = 0.9;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("on-policy")
    );

    let mut config = valid_grpo();
    config.sampling.top_p = 0.9;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("on-policy")
    );

    let mut config = valid_grpo();
    config.reward_command.clear();
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("executable")
    );

    let mut config = valid_grpo();
    config.updates = 0;
    assert!(
        config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("greater than zero")
    );

    for invalid in [-0.1, f32::NAN, f32::INFINITY] {
        let mut config = valid_grpo();
        config.kl_coefficient = invalid;
        assert!(config.validate().is_err(), "kl={invalid}");
    }
}
