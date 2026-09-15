//! Argument handling of the `profile` binary.
//!
//! These cases exit before any model is loaded, so they stay fast and run
//! everywhere. They guard the `--micro-batch` override the VRAM sweep drives: the
//! runtime only rejects a bad micro-batch once the context is being created,
//! which would waste a model load per sweep point.

use std::process::{Command, Output};

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_profile"))
        .args(args)
        .env("NO_COLOR", "1")
        .output()
        .expect("spawn profile binary")
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

#[test]
fn help_documents_the_micro_batch_override() {
    let output = run(&["--help"]);
    assert!(output.status.success(), "--help should exit 0");
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("--micro-batch"),
        "help should list --micro-batch"
    );
}

#[test]
fn the_micro_batch_must_divide_the_optimizer_window() {
    // examples/smoke_tiny_grpo.toml trains a 64-token window per step.
    for value in ["48", "96", "0"] {
        let output = run(&[SMOKE_GRPO, "--micro-batch", value]);
        assert!(
            !output.status.success(),
            "--micro-batch {value} should be rejected"
        );
        let message = stderr(&output);
        assert!(
            message.contains("divisor of the optimizer window") && message.contains("64"),
            "the error should name the constraint and the window, got: {message}"
        );
    }
}

const SMOKE_GRPO: &str = "examples/smoke_tiny_grpo.toml";

/// Copy of the smoke GRPO config pointing at a model that cannot exist, so the
/// run stops at model loading instead of training the fixture that may well be
/// present on the machine running the tests. The device is pinned to `cpu`
/// so the run reaches that model-loading error on any machine, GPU or not.
fn config_without_a_model() -> std::path::PathBuf {
    let source = std::fs::read_to_string(SMOKE_GRPO).expect("read smoke GRPO config");
    let mut text = String::new();
    for line in source.lines() {
        if line.starts_with("path = ") {
            text.push_str("path = \"/nonexistent/retrograd-profile-test.gguf\"\n");
        } else if line.starts_with("device = ") {
            text.push_str("device = \"cpu\"\n");
        } else {
            text.push_str(line);
            text.push('\n');
        }
    }
    let path = std::env::temp_dir().join(format!(
        "retrograd-profile-{}-{}.toml",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos(),
    ));
    std::fs::write(&path, text).expect("write temp config");
    path
}

#[test]
fn the_micro_batch_accepts_the_sweep_values() {
    // The campaign sweeps divisors of the 64-token step window; none of these
    // must trip the validation. Reaching the model load proves they were accepted.
    let config = config_without_a_model();
    for value in ["8", "16", "32", "64"] {
        let output = run(&[config.to_str().expect("utf-8 path"), "--micro-batch", value]);
        let message = stderr(&output);
        assert!(
            !message.contains("divisor of the optimizer window"),
            "--micro-batch {value} divides 64 and must be accepted, got: {message}"
        );
        assert!(
            message.contains("retrograd-profile-test.gguf"),
            "the run should have reached model loading, got: {message}"
        );
    }
    let _ = std::fs::remove_file(&config);
}

#[test]
fn unknown_flags_are_reported() {
    let output = run(&["--nope"]);
    assert!(!output.status.success());
    assert!(stderr(&output).contains("unknown profile flag '--nope'"));
}
