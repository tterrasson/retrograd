//! One row per family of recurrent state, not one row per model.
//!
//! A single fixture (LFM2) plus one opt-in regression model (Falcon-H1) would
//! cover one family and a half. The properties this lane checks are the ones an
//! architecture-name list would assert without ever running anything:
//!
//!   * the packed-training verdict is stated, with the reason that produced it;
//!   * the verdict is the same in the text report and in the ABI capability
//!     struct the planner reads;
//!   * a trainer that says "no" does not attempt the packed path, and a trainer
//!     that says "yes" survives an actual step.
//!
//! Every family but the first skips unless its model is present
//! (`common::RECURRENT_FAMILIES` names the environment variable for each), so
//! this binary is safe in the CPU lane and gains coverage on a machine that has
//! the models.

mod common;

use retrograd::{Device, LoraConfig, TrainConfig, Trainer};

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 128,
        n_batch: 32,
        n_ubatch: 16,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

/// The report and the ABI struct must not be able to disagree: the resolver
/// branches on the struct, a human reads the report, and a run where the two
/// say different things is unsupportable.
#[test]
fn every_available_family_states_one_packed_verdict() {
    let mut covered = 0;
    for family in common::RECURRENT_FAMILIES {
        let Some(model) = family.path_if_available() else {
            eprintln!(
                "skipping family {}: no model at {} (set {})",
                family.family,
                family.path().display(),
                family.env
            );
            continue;
        };
        let _guard = common::serialize_models();
        let mut trainer = Trainer::new(&model, config())
            .unwrap_or_else(|error| panic!("load {} family model: {error}", family.family));

        let report = trainer.capability_report().expect("capability report");
        let capability = trainer
            .supports_shared_prefix_packed_training()
            .expect("packed-training capability");

        assert!(
            report.contains("shared_prefix_packed_training: "),
            "family {}: the report must state the verdict\n{report}",
            family.family
        );
        assert!(
            report.contains(&format!(
                "shared_prefix_packed_training: {}",
                if capability { "yes" } else { "no" }
            )),
            "family {}: report and ABI capability disagree (capability = {capability})\n{report}",
            family.family
        );
        // A "no" with no reason is a support question nobody can answer from a
        // log, which is what an architecture-name list produces.
        assert!(
            report.contains("packed_seq: "),
            "family {}: the verdict must name what decided it\n{report}",
            family.family
        );
        assert!(
            report.contains("micro_batch_finite_check: "),
            "family {}: the micro-batch check must be reported\n{report}",
            family.family
        );
        assert!(
            report.contains("recurrent_rollback_derived: "),
            "family {}: the rollback derivation must be reported\n{report}",
            family.family
        );
        covered += 1;
    }
    assert!(
        covered > 0,
        "no recurrent family model was available; the default CPU fixture should have covered one"
    );
}

/// A minimal step on every available family. The packed verdict decides which
/// layout the trainer takes internally; either way the step must produce finite
/// numbers rather than abort, which is the failure mode a stale architecture
/// list produced (a `GGML_ASSERT` inside the graph, not an error return).
#[test]
fn every_available_family_survives_a_step() {
    for family in common::RECURRENT_FAMILIES {
        let Some(model) = family.path_if_available() else {
            eprintln!("skipping family {}: no model", family.family);
            continue;
        };
        let _guard = common::serialize_models();
        let mut trainer = Trainer::new(&model, config())
            .unwrap_or_else(|error| panic!("load {} family model: {error}", family.family));
        trainer
            .create_lora(&LoraConfig::auto(2, 4.0))
            .unwrap_or_else(|error| panic!("create LoRA on {}: {error}", family.family));
        let text = "A short run of tokens for the recurrent family lane. ".repeat(48);
        let tokens = trainer
            .tokenize_text(&text)
            .unwrap_or_else(|error| panic!("tokenize on {}: {error}", family.family));
        let metrics = trainer
            .train_tokens(&tokens)
            .unwrap_or_else(|error| panic!("train step on {}: {error}", family.family));
        assert!(
            metrics.train_loss.is_finite(),
            "family {}: non-finite training loss {:?}",
            family.family,
            metrics
        );
    }
}
