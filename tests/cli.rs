//! End-to-end coverage of the `retrograd` binary's command handling.
//!
//! The parsing cases exit before loading a model and stay fast. The optional
//! final smoke test covers successful CLI orchestration when the local GGUF
//! fixture is available.

mod common;

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_retrograd"))
        .args(args)
        // Keep diagnostics plain so assertions are colour-code independent.
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn retrograd binary")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn no_arguments_prints_help_and_succeeds() {
    let output = run(&[]);
    assert!(output.status.success(), "help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("train <config.toml>"),
        "help should list the command"
    );
    assert!(stdout.contains("inspect"), "help should list inspect");
    assert!(stdout.contains("preflight"), "help should list preflight");
    assert!(
        stdout.contains("bench <config.toml>"),
        "help should list bench"
    );
    assert!(
        stdout.contains("chat <config.toml>"),
        "help should list chat"
    );
}

#[test]
fn bench_validates_arguments_before_model_loading() {
    let output = run(&["bench"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("config TOML path"));

    let output = run(&["bench", "run.toml", "--unknown", "x"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown bench flag '--unknown'"));

    let output = run(&["bench", "run.toml", "--model"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("missing value for --model"));

    let output = run(&["bench", "run.toml", "--format", "csv"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--format must be auto, text, or jsonl"));

    let output = run(&["bench", "run.toml", "--ctx", "0"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--ctx must be greater than zero"));

    let output = run(&["bench", "one.toml", "two.toml"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("exactly one config TOML path"));

    let output = run(&[
        "bench",
        "run.toml",
        "--data",
        "one.jsonl",
        "--eval-data",
        "two.jsonl",
    ]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("exactly one evaluation dataset"));
}

#[test]
fn inspect_requires_a_model_before_loading_one() {
    let output = run(&["inspect"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--model is required"));

    let output = run(&["inspect", "--unknown", "value"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown inspect flag '--unknown'"));

    let output = run(&["inspect", "--device"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("missing value for --device"));

    let output = run(&["inspect", "--device", "tpu"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown device 'tpu'"));
}

#[test]
fn chat_validates_flags_before_model_loading() {
    let output = run(&["chat"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("config TOML path"));

    let output = run(&["chat", "a.toml", "b.toml"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("exactly one config TOML path"));

    let output = run(&["chat", "run.toml", "--unknown", "x"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown chat flag '--unknown'"));

    let output = run(&["chat", "run.toml", "--system"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("missing value for --system"));

    let output = run(&["chat", "run.toml", "--compare", "--base_only"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("mutually exclusive"));

    let output = run(&["chat", "run.toml", "--base_only", "--adapter", "a.gguf"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--base_only cannot be combined with --adapter"));

    for (flag, value) in [
        ("--ctx", "0"),
        ("--temp", "0"),
        ("--top-p", "1.5"),
        ("--max-new-tokens", "0"),
    ] {
        let output = run(&["chat", "run.toml", flag, value]);
        assert!(
            !output.status.success(),
            "{flag}={value} should be rejected"
        );
        assert!(stderr(&output).contains(flag.trim_start_matches('-')));
    }

    // Range checks must run before the config is opened, not after.
    let output = run(&["chat", "run.toml", "--seed", "x"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("invalid --seed value 'x'"));
}

#[test]
fn preflight_validates_flags_and_targets_before_model_loading() {
    let output = run(&["preflight"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("--model is required"));

    let output = run(&["preflight", "--unknown", "value"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown preflight flag '--unknown'"));

    let output = run(&["preflight", "--targets"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("missing value for --targets"));

    let output = run(&["preflight", "--targets", "auto,q"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("cannot be combined"));
}

#[test]
fn help_flag_prints_help_and_succeeds() {
    for flag in ["-h", "--help"] {
        let output = run(&[flag]);
        assert!(output.status.success(), "{flag} should exit 0");
        assert!(String::from_utf8_lossy(&output.stdout).contains("train <config.toml>"));
    }
}

#[test]
fn unknown_command_fails_with_message() {
    let output = run(&["frobnicate"]);
    assert!(
        !output.status.success(),
        "unknown command must exit non-zero"
    );
    assert!(stderr(&output).contains("unknown command 'frobnicate'"));
}

#[test]
fn train_requires_exactly_one_config_path() {
    let missing = run(&["train"]);
    assert!(!missing.status.success());
    assert!(stderr(&missing).contains("train requires a config TOML path"));

    let extra = run(&["train", "first.toml", "second.toml"]);
    assert!(!extra.status.success());
    assert!(stderr(&extra).contains("exactly one config TOML path"));
}

#[test]
fn train_reports_missing_and_invalid_config() {
    let output = run(&["train", "nope.toml"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("io error"));
}

#[test]
fn train_lora_is_no_longer_a_command() {
    let output = run(&["train-lora"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown command 'train-lora'"));
}

#[test]
fn train_sft_runs_through_the_cli_and_writes_metrics_and_adapter() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();
    let dir = std::env::temp_dir().join(format!("retrograd-cli-sft-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("train.txt"),
        "The quick brown fox jumps over the lazy dog. ".repeat(36),
    )
    .unwrap();
    let config = format!(
        "[run]\nalgorithm='sft'\nverbose=true\n[model]\npath='{}'\ndevice='cpu'\n[lora]\noutput='adapter.gguf'\nrank=2\nalpha=4.0\ndtype='f16'\ntargets=['blk.2.attn_q.weight']\n[training]\nctx=32\nmicro_batch=16\ngradient_accumulation=2\nepochs=3\nlr=0.001\n[metrics]\ntensorboard_dir='tensorboard'\nwandb_export_dir='wandb'\n[sft]\ndata='train.txt'\ndata_format='text'\n[evaluation]\ndata='train.txt'\nevery_iterations=1\npatience=1\nmin_delta=100.0\n[checkpoint]\ndirectory='checkpoints'\nmode='steps_and_best_eval'\nevery_steps=1\n",
        model.display()
    );
    let config_path = dir.join("run.toml");
    std::fs::write(&config_path, config).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_retrograd"))
        .args(["train", config_path.to_str().unwrap()])
        .env("NO_COLOR", "1")
        .output()
        .expect("run SFT CLI");
    assert!(output.status.success(), "{}", stderr(&output));
    let diagnostics = stderr(&output);
    assert!(
        diagnostics.contains("backend report"),
        "stderr: {diagnostics}"
    );
    assert!(
        diagnostics.contains("lora_dtype: F16"),
        "stderr: {diagnostics}"
    );
    assert!(
        diagnostics.contains("optimizer_f16: supported"),
        "stderr: {diagnostics}"
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains("done epoch=2"),
        "stdout: {}",
        String::from_utf8_lossy(&output.stdout)
    );
    assert!(dir.join("adapter.gguf").is_file());
    assert!(dir.join("checkpoints/best.gguf").is_file());
    assert!(
        std::fs::read_dir(dir.join("checkpoints"))
            .unwrap()
            .any(|entry| entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("step-"))
    );
    assert!(dir.join("wandb/run.json").is_file());
    let events = std::fs::read_to_string(dir.join("wandb/metrics.jsonl")).unwrap();
    assert!(events.contains("\"event\":\"run_started\""));
    assert!(events.contains("\"event\":\"step\""));
    assert!(events.contains("\"name\":\"eval/loss\""));
    assert!(events.contains("\"event\":\"run_finished\""));
    let tensorboard_runs: Vec<_> = std::fs::read_dir(dir.join("tensorboard"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(tensorboard_runs.len(), 1);
    assert!(tensorboard_runs[0].is_dir());
    assert!(
        std::fs::read_dir(&tensorboard_runs[0])
            .unwrap()
            .any(|entry| entry.unwrap().path().is_file())
    );

    let bench = Command::new(env!("CARGO_BIN_EXE_retrograd"))
        .arg("bench")
        .arg(&config_path)
        .arg("--adapter")
        .arg(dir.join("adapter.gguf"))
        .args(["--device", "cpu", "--ctx", "32", "--limit", "2"])
        .env("NO_COLOR", "1")
        .output()
        .expect("run base + adapter bench CLI");
    assert!(bench.status.success(), "{}", stderr(&bench));
    let bench_stdout = String::from_utf8_lossy(&bench.stdout);
    assert!(bench_stdout.contains("base"), "{bench_stdout}");
    assert!(bench_stdout.contains("adapter"), "{bench_stdout}");
    assert!(bench_stdout.contains("quality"), "{bench_stdout}");
    assert!(bench_stdout.contains("mean logprob"), "{bench_stdout}");
    assert!(bench_stdout.contains("p10"), "{bench_stdout}");
    assert!(bench_stdout.contains("p90"), "{bench_stdout}");
    assert!(bench_stdout.contains("verdict"), "{bench_stdout}");
    assert!(bench_stdout.contains("improved"), "{bench_stdout}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn train_failure_is_exported_and_does_not_write_an_adapter() {
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let _guard = common::serialize_models();
    let dir = std::env::temp_dir().join(format!("retrograd-cli-failure-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("train.txt"), "too short").unwrap();
    let config = format!(
        "[run]\nalgorithm='sft'\n[model]\npath='{}'\ndevice='cpu'\n[lora]\noutput='adapter.gguf'\nrank=2\nalpha=4.0\ntargets=['blk.2.attn_q.weight']\n[training]\nctx=32\nmicro_batch=16\ngradient_accumulation=2\nepochs=1\n[metrics]\nwandb_export_dir='wandb'\n[sft]\ndata='train.txt'\ndata_format='text'\n",
        model.display()
    );
    let config_path = dir.join("run.toml");
    std::fs::write(&config_path, config).unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_retrograd"))
        .args(["train", config_path.to_str().unwrap()])
        .env("NO_COLOR", "1")
        .output()
        .expect("run failing SFT CLI");
    assert!(!output.status.success());
    assert!(stderr(&output).contains("text dataset requires more than"));
    assert!(!dir.join("adapter.gguf").exists());
    let events = std::fs::read_to_string(dir.join("wandb/metrics.jsonl")).unwrap();
    assert!(events.contains("\"event\":\"run_started\""));
    assert!(events.contains("\"event\":\"run_failed\""));
    assert!(!events.contains("\"event\":\"run_finished\""));
    std::fs::remove_dir_all(dir).unwrap();
}
