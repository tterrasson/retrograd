//! Exhaustive coverage of the TOML surface, and of the defaults behind it.
//!
//! The 38 tests next door check what a config *rejects*. What none of them
//! checks is that a field written in a document actually reaches [`RunConfig`]:
//! `build` copies the two shapes field by field, by hand, and a line missing
//! there is silent - the run trains with the default and the file that asked
//! for something else is never contradicted. That is the failure this file is
//! here to make loud, because it is the one that makes the `*Toml` doubling
//! dangerous to touch.
//!
//! Two properties, and one construction that carries both:
//!
//! - **Coverage is a compile error, not a claim.** [`exhaustive_document`]
//!   builds every `*Toml` value as a struct literal with no
//!   `..Default::default()` tail. A field added to any of the fifteen types
//!   stops this file compiling, which is what keeps the word "exhaustive"
//!   true after the fact.
//! - **Every key survives the trip.** [`KEYS`] freezes the flattened key set a
//!   full document serializes to. A field that fails to serialize disappears
//!   from it; a field that fails to deserialize disappears from the second
//!   pass. Both are diffed against the frozen list rather than against each
//!   other, so a symmetric loss cannot cancel out.
//!
//! `[agent]` is deliberately absent: `AgentToml` is `#[serde(default)]` and
//! carries the whole agentic stack's declarations, which are typed and tested
//! in their own crates. The sixteen types here are the ones `build` and its
//! `build_*` helpers copy by hand.

use std::path::Path;

use retrograd_core::{
    CheckpointDtype, FeatureDtype, GefenLayout, GefenVariant, KvDtype, LayerRange, LoraDtype,
    LrScheduler, OptimizerKind, RewardMode, SharedPrefixFanout, TrainablePolicy, TrainableSelector,
};
use retrograd_dataset::DataFormat;

use super::*;
use crate::grpo::build_grpo;

/// Every key a fully-populated document serializes to, in dotted form and
/// sorted. Frozen: adding a field to a `*Toml` type breaks
/// [`exhaustive_document`] at compile time, and updating this list is how the
/// author says the new field really does make the round trip.
const KEYS: &[&str] = &[
    "checkpoint.directory",
    "checkpoint.every_steps",
    "checkpoint.mode",
    "checkpoint.resume_from",
    "evaluation.data",
    "evaluation.every_iterations",
    "evaluation.max_examples",
    "evaluation.min_delta",
    "evaluation.patience",
    "grpo.baseline",
    "grpo.clip_range_high",
    "grpo.clip_range_low",
    "grpo.dynamic_sampling.max_resample_factor",
    "grpo.group_size",
    "grpo.grpo_epochs",
    "grpo.judge.command",
    "grpo.judge.timeout_secs",
    "grpo.judge.type",
    "grpo.judge_failure",
    "grpo.judge_weight",
    "grpo.kl_coefficient",
    "grpo.kl_schedule.target",
    "grpo.kl_schedule.warmup_updates",
    "grpo.mask_truncated",
    "grpo.max_judge_dropped_fraction",
    "grpo.max_stalled_updates",
    "grpo.overlong_penalty.buffer_tokens",
    "grpo.overlong_penalty.max_penalty",
    "grpo.prompt_order",
    "grpo.prompts",
    "grpo.prompts_per_update",
    "grpo.reward_command",
    "grpo.reward_mode",
    "grpo.reward_timeout_seconds",
    "grpo.sampling.max_new_tokens",
    "grpo.sampling.seed",
    "grpo.sampling.temperature",
    "grpo.sampling.top_p",
    "grpo.updates",
    "lora.alpha",
    "lora.dtype",
    "lora.rank",
    "lora.seed",
    "lora.targets",
    "metrics.tensorboard_dir",
    "metrics.wandb_export_dir",
    "model.device",
    "model.path",
    "observe.directory",
    "observe.every",
    "observe.max_text_chars",
    "optimizer.gefen.beta1",
    "optimizer.gefen.beta2",
    "optimizer.gefen.block_size",
    "optimizer.gefen.codebook",
    "optimizer.gefen.codebook_levels",
    "optimizer.gefen.eps",
    "optimizer.gefen.min_numel",
    "optimizer.gefen.partition",
    "optimizer.gefen.variant",
    "optimizer.muon.fallback_learning_rate",
    "optimizer.muon.momentum",
    "optimizer.muon.nesterov",
    "optimizer.muon.ns_epsilon",
    "optimizer.muon.ns_steps",
    "output.kind",
    "output.path",
    "reference.ctx",
    "reference.model",
    "run.algorithm",
    "run.verbose",
    "trainable.biases",
    "trainable.layers",
    "trainable.modules",
    "trainable.norms",
    "trainable.output_head",
    "training.checkpoint_dtype",
    "training.checkpoint_every_n_layers",
    "training.chunked_ce_seq_chunk",
    "training.chunked_ce_tiles",
    "training.chunked_cross_entropy",
    "training.ctx",
    "training.epochs",
    "training.fast_sampling_context",
    "training.generation_batch",
    "training.generation_concurrency",
    "training.gradient_accumulation",
    "training.gradient_checkpointing",
    "training.kv_dtype",
    "training.lr",
    "training.lr_scheduler",
    "training.max_gpu_duty_cycle",
    "training.max_grad_norm",
    "training.micro_batch",
    "training.optimizer",
    "training.require_gpu_resident",
    "training.shared_prefix_fanout",
    "training.threads",
    "training.trainable",
    "training.warmup_steps",
    "training.weight_decay",
];

/// The three `*Toml` types the GRPO document cannot reach, because only one
/// algorithm section may be present at a time. Same construction rule: no
/// `..Default::default()` tail.
fn exhaustive_sft() -> SftToml {
    SftToml {
        data: PathBuf::from("data/sft.jsonl"),
        data_format: Some("jsonl".to_string()),
        shuffle: Some(false),
    }
}

fn exhaustive_ppo() -> PpoToml {
    PpoToml {
        prompts: PathBuf::from("data/prompts.txt"),
        reward_command: vec!["score".to_string(), "--ppo".to_string()],
        reward_mode: Some(RewardMode::OneShot),
        reward_timeout_seconds: Some(45),
        updates: 7,
        rollout_batch_size: 4,
        ppo_epochs: 2,
        clip_range: 0.25,
        kl_coefficient: 0.05,
        critic: CriticToml {
            enabled: Some(true),
            gamma: Some(0.97),
            gae_lambda: Some(0.9),
            value_lr: Some(3.0e-4),
            value_epochs: Some(3),
            feature_dtype: Some(FeatureDtype::Bf16),
        },
        sampling: ppo_sampling(),
    }
}

/// Distillation is on-policy for the same reason GRPO is - the behaviour
/// log-probabilities the sampler records are what the advantage is a difference
/// against - so it reuses the GRPO sampling literal.
fn exhaustive_distill() -> DistillToml {
    DistillToml {
        // The on-policy document. `mode` is written explicitly rather than
        // defaulted: this literal exists to name every key the schema has, and
        // the offline three are covered by `exhaustive_distill_offline`.
        mode: Some("on_policy".to_string()),
        data: None,
        sidecar: None,
        offline_epochs: None,
        teacher_path: PathBuf::from("models/teacher.gguf"),
        prompts: Some(PathBuf::from("data/prompts.txt")),
        updates: Some(11),
        prompts_per_update: Some(2),
        samples_per_prompt: Some(3),
        distill_epochs: Some(2),
        clip_range_low: Some(0.15),
        clip_range_high: Some(0.3),
        weight_clip: Some(4.5),
        kl_coefficient: Some(0.03),
        mask_truncated: true,
        prompt_order: Some("shuffled".to_string()),
        sampling: Some(exhaustive_sampling()),
    }
}

/// The offline top-k document: a teacher, a corpus, a
/// sidecar, and none of the keys a sampler would read.
fn exhaustive_distill_offline() -> DistillToml {
    DistillToml {
        mode: Some("topk_offline".to_string()),
        data: Some(PathBuf::from("data/corpus.jsonl")),
        sidecar: Some(PathBuf::from("data/corpus.topk")),
        offline_epochs: Some(4),
        teacher_path: PathBuf::from("models/teacher.gguf"),
        prompts: None,
        updates: None,
        prompts_per_update: None,
        samples_per_prompt: None,
        distill_epochs: None,
        clip_range_low: None,
        clip_range_high: None,
        weight_clip: Some(4.5),
        kl_coefficient: None,
        mask_truncated: false,
        prompt_order: None,
        sampling: None,
    }
}

/// Dr. GRPO refuses anything but strictly on-policy sampling, so the two
/// fields that could vary are covered by the PPO document instead.
fn exhaustive_sampling() -> SamplingToml {
    SamplingToml {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 48,
        seed: 1234,
    }
}

fn ppo_sampling() -> SamplingToml {
    SamplingToml {
        temperature: 0.8,
        top_p: 0.95,
        max_new_tokens: 48,
        seed: 1234,
    }
}

/// A document with **every** field of every `*Toml` type set, each to a value
/// distinguishable from its default, so a `build_*` line that drops one is
/// visible as a wrong value rather than as an absent one.
///
/// The literals carry no `..Default::default()`: that is the compile-time half
/// of the coverage claim.
fn exhaustive_document() -> ConfigDocument {
    ConfigDocument {
        run: RunToml {
            algorithm: "grpo".to_string(),
            verbose: true,
        },
        model: ModelToml {
            path: Some(PathBuf::from("model.gguf")),
            device: Some("cpu".to_string()),
        },
        output: Some(OutputToml {
            path: PathBuf::from("out/adapter.gguf"),
            kind: Some("adapter".to_string()),
        }),
        lora: Some(LoraToml {
            rank: Some(16),
            alpha: Some(32.0),
            seed: Some(7),
            targets: vec!["q".to_string(), "v".to_string()],
            // Mutually exclusive with every field above it, so it is covered by
            // its own test rather than by this document.
            init_adapter: None,
            dtype: Some(LoraDtype::F16),
        }),
        // A `partial` policy with a fully-written selector, because that is the
        // combination that exercises every key. The "reaches RunConfig" test
        // below builds the LoRA normalization of this document, since a LoRA
        // run refuses the section; the base normalization has its own test.
        trainable: Some(TrainableToml {
            layers: Some("1..3".to_string()),
            modules: vec!["attn".to_string(), "ffn_up".to_string()],
            norms: Some(true),
            biases: Some(true),
            output_head: Some(false),
        }),
        training: TrainingToml {
            trainable: Some("partial".to_string()),
            optimizer: Some("adamw".to_string()),
            ctx: Some(256),
            micro_batch: Some(64),
            shared_prefix_fanout: Some(SharedPrefixFanoutToml::Exact(2)),
            gradient_accumulation: Some(4),
            threads: Some(3),
            epochs: Some(2),
            lr: Some(5.0e-5),
            weight_decay: Some(0.02),
            max_grad_norm: Some(0.5),
            lr_scheduler: Some("cosine".to_string()),
            warmup_steps: Some(11),
            fast_sampling_context: Some(false),
            kv_dtype: Some(KvDtype::F32),
            generation_concurrency: Some(6),
            generation_batch: Some(128),
            chunked_cross_entropy: Some(true),
            chunked_ce_tiles: Some(5),
            chunked_ce_seq_chunk: Some(17),
            gradient_checkpointing: Some(true),
            checkpoint_every_n_layers: Some(3),
            checkpoint_dtype: Some(CheckpointDtype::F16),
            require_gpu_resident: Some(true),
            max_gpu_duty_cycle: Some(0.5),
        },
        metrics: MetricsToml {
            tensorboard_dir: Some(PathBuf::from("runs/tb")),
            wandb_export_dir: Some(PathBuf::from("runs/wandb")),
        },
        evaluation: Some(EvaluationToml {
            data: PathBuf::from("data/eval.jsonl"),
            every_iterations: Some(2),
            patience: Some(4),
            min_delta: Some(0.125),
            max_examples: Some(9),
        }),
        checkpoint: Some(CheckpointToml {
            directory: PathBuf::from("ckpt"),
            mode: "steps_and_best_eval".to_string(),
            every_steps: Some(13),
            resume_from: Some(PathBuf::from("ckpt/step-13.state")),
        }),
        sft: None,
        ppo: None,
        distill: None,
        grpo: Some(GrpoToml {
            prompts: PathBuf::from("data/prompts.txt"),
            reward_command: vec!["score".to_string(), "--grpo".to_string()],
            reward_mode: Some(RewardMode::Persistent),
            reward_timeout_seconds: Some(90),
            updates: 5,
            prompts_per_update: 2,
            group_size: 3,
            grpo_epochs: 2,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.01,
            mask_truncated: true,
            baseline: Some("leave_one_out".to_string()),
            prompt_order: Some("shuffled".to_string()),
            overlong_penalty: Some(OverlongPenalty {
                buffer_tokens: 12,
                max_penalty: 0.75,
            }),
            kl_schedule: Some(KlScheduleToml {
                warmup_updates: Some(4),
                target: Some(0.02),
            }),
            dynamic_sampling: Some(DynamicSampling {
                max_resample_factor: 3,
            }),
            judge: Some(retrograd_spec::judge::JudgeConfig::Command {
                command: vec!["judge".to_string()],
                timeout_secs: 45,
            }),
            judge_weight: Some(0.4),
            judge_failure: Some(retrograd_agent_core::config::JudgeFailurePolicy::Fail),
            max_judge_dropped_fraction: Some(0.25),
            max_stalled_updates: Some(9),
            sampling: exhaustive_sampling(),
        }),
        agent: None,
        observe: Some(ObserveToml {
            directory: PathBuf::from("out/observe"),
            every: Some(3),
            max_text_chars: Some(2000),
        }),
        // Legal here only because the GRPO section above carries a positive KL
        // coefficient.
        reference: Some(ReferenceToml {
            model: PathBuf::from("original.gguf"),
            ctx: Some(512),
        }),
        // Both tables, for the key coverage above. No document may carry both
        // and build, which is why the normalizers drop them.
        optimizer: Some(OptimizerToml {
            muon: Some(MuonToml {
                momentum: Some(0.9),
                nesterov: Some(false),
                ns_steps: Some(3),
                ns_epsilon: Some(1.0e-6),
                fallback_learning_rate: Some(3.0e-4),
            }),
            gefen: Some(GefenToml {
                variant: Some("quantized_m".to_string()),
                block_size: Some(512),
                min_numel: Some(8192),
                codebook: Some("uniform".to_string()),
                codebook_levels: Some(256),
                partition: Some("fixed".to_string()),
                beta1: Some(0.8),
                beta2: Some(0.99),
                eps: Some(1.0e-7),
            }),
        }),
    }
}

/// The exhaustive document with its base-training selection removed, i.e. the
/// LoRA run every other field of it describes.
///
/// `build` refuses `trainable = "partial"` until the runtime can honour it, so
/// a document that exercises the whole `[trainable]` schema cannot also be the
/// one that proves the other fields reach `RunConfig`. One helper rather than a
/// second literal: a second literal would drift.
fn lora_normalized(mut document: ConfigDocument) -> ConfigDocument {
    document.trainable = None;
    document.training.trainable = Some("lora".to_string());
    // Two tables for two optimizers, and the document names one: the pair is
    // covered by the cross-field tests rather than by every build here.
    document.optimizer = None;
    document
}

/// The same document as a `partial` base-weight run: no adapter section, and
/// an output kind the policy actually produces.
fn base_normalized(mut document: ConfigDocument) -> ConfigDocument {
    document.lora = None;
    document.optimizer = None;
    document.output = Some(OutputToml {
        path: PathBuf::from("out/adapter.gguf"),
        kind: None,
    });
    document
}

/// A base-weight policy is a run a document can now name, and the whole
/// `[trainable]` selector reaches the configuration rather than being parsed
/// and then refused.
#[test]
fn a_partial_document_reaches_the_run_config_with_its_selector() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = base_normalized(exhaustive_document());
    // No KL term, so no anchor.
    document.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    document.grpo.as_mut().expect("section").kl_schedule = None;
    document.reference = None;
    let config = build(document, root).expect("partial base training builds");

    assert_eq!(config.training.trainable.policy, TrainablePolicy::Partial);
    assert!(config.lora.is_none(), "a partial run creates no adapter");
    assert_eq!(config.output.kind, OutputKind::Trainable);
    assert_eq!(config.output.path, root.join("out/adapter.gguf"));
    let selector = &config.training.trainable.selector;
    assert_eq!(selector.layers, LayerRange::Inclusive { first: 1, last: 3 });
    assert_eq!(selector.modules, ["attn".to_string(), "ffn_up".to_string()]);
    assert!(selector.norms);
    assert!(selector.biases);
    assert!(!selector.output_head);

    // A malformed selector is reported as such.
    let mut broken = base_normalized(exhaustive_document());
    broken.trainable.as_mut().expect("section").layers = Some("3..1".to_string());
    let error = build(broken, root).expect_err("an inverted range is refused");
    assert!(error.to_string().contains("inclusive"), "{error}");
}

/// `[lora]` and `[output].kind` are the two halves of "what does this run
/// produce", and every pairing that would lose half of it is refused.
#[test]
fn the_adapter_section_and_the_output_kind_follow_the_policy() {
    let root = Path::new("/tmp/retrograd-round-trip");
    // A section describing an adapter the run never creates.
    let mut with_adapter = base_normalized(exhaustive_document());
    with_adapter.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    with_adapter.grpo.as_mut().expect("section").kl_schedule = None;
    with_adapter.lora = Some(LoraToml {
        rank: Some(8),
        alpha: Some(16.0),
        seed: None,
        targets: Vec::new(),
        init_adapter: None,
        dtype: None,
    });
    let error = build(with_adapter, root).expect_err("a partial run has no adapter");
    assert!(error.to_string().contains("remove [lora]"), "{error}");

    // ... and the mirror image: hybrid trains one and requires the section.
    let mut hybrid = base_normalized(exhaustive_document());
    hybrid.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    hybrid.grpo.as_mut().expect("section").kl_schedule = None;
    hybrid.training.trainable = Some("hybrid".to_string());
    hybrid.trainable = Some(TrainableToml {
        norms: Some(true),
        ..Default::default()
    });
    let error = build(hybrid.clone(), root).expect_err("hybrid trains an adapter too");
    assert!(error.to_string().contains("requires a [lora]"), "{error}");

    // An adapter-only export of a hybrid run drops the base half.
    hybrid.lora = Some(LoraToml {
        rank: Some(8),
        alpha: Some(16.0),
        seed: None,
        targets: Vec::new(),
        init_adapter: None,
        dtype: Some(LoraDtype::F32),
    });
    hybrid.output = Some(OutputToml {
        path: PathBuf::from("out/adapter.gguf"),
        kind: Some("adapter".to_string()),
    });
    let error = build(hybrid, root).expect_err("an adapter is half of a hybrid run");
    assert!(
        error.to_string().contains("drop the trained base"),
        "{error}"
    );

    // And a LoRA run has no bundle to write.
    let mut lora = lora_normalized(exhaustive_document());
    lora.output = Some(OutputToml {
        path: PathBuf::from("out/adapter.gguf"),
        kind: Some("trainable".to_string()),
    });
    let error = build(lora, root).expect_err("a lora run trains no base tensor");
    assert!(error.to_string().contains("portable result"), "{error}");
}

/// A standalone model GGUF is what a base-weight run publishes: `lora` and
/// `hybrid` are refused, since merging an adapter into the weights is not
/// covered here.
#[test]
fn a_model_export_belongs_to_the_policies_that_carry_no_adapter() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = base_normalized(exhaustive_document());
    document.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    document.grpo.as_mut().expect("section").kl_schedule = None;
    document.reference = None;
    document.output = Some(OutputToml {
        path: PathBuf::from("out/model.gguf"),
        kind: Some("model".to_string()),
    });
    let config = build(document, root).expect("a partial run writes its weights back out");
    assert_eq!(config.output.kind, OutputKind::Model);
    assert_eq!(config.output.path, root.join("out/model.gguf"));

    let mut lora = lora_normalized(exhaustive_document());
    lora.output = Some(OutputToml {
        path: PathBuf::from("out/model.gguf"),
        kind: Some("model".to_string()),
    });
    let error = build(lora, root).expect_err("a lora run's result is not in the weights");
    assert!(error.to_string().contains("parity coverage"), "{error}");

    let mut hybrid = base_normalized(exhaustive_document());
    hybrid.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    hybrid.grpo.as_mut().expect("section").kl_schedule = None;
    hybrid.training.trainable = Some("hybrid".to_string());
    hybrid.trainable = Some(TrainableToml {
        norms: Some(true),
        ..Default::default()
    });
    hybrid.lora = Some(LoraToml {
        rank: Some(8),
        alpha: Some(16.0),
        seed: None,
        targets: Vec::new(),
        init_adapter: None,
        dtype: Some(LoraDtype::F32),
    });
    hybrid.output = Some(OutputToml {
        path: PathBuf::from("out/model.gguf"),
        kind: Some("model".to_string()),
    });
    let error = build(hybrid, root).expect_err("a hybrid run would lose its adapter");
    assert!(error.to_string().contains("drop the adapter"), "{error}");

    let mut unknown = lora_normalized(exhaustive_document());
    unknown.output = Some(OutputToml {
        path: PathBuf::from("out/model.gguf"),
        kind: Some("merged".to_string()),
    });
    let error = build(unknown, root).expect_err("an unknown kind is refused");
    assert!(error.to_string().contains("must be adapter"), "{error}");
}

/// A KL penalty is taken against "the model with its adapter disabled", which
/// is the original policy only while the base weights are frozen. A run that
/// updates them has no reference, and the configuration says so rather than
/// reporting a divergence from a moving target.
#[test]
fn a_fixed_reference_consumer_is_refused_beside_base_training() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = base_normalized(exhaustive_document());
    document.reference = None;
    assert!(
        document.grpo.as_ref().expect("section").kl_coefficient > 0.0
            || document
                .grpo
                .as_ref()
                .expect("section")
                .kl_schedule
                .is_some()
    );
    let error = build(document, root).expect_err("base training has no fixed reference");
    let message = error.to_string();
    assert!(message.contains("grpo.kl_coefficient"), "{message}");
    assert!(message.contains("separate anchor"), "{message}");

    // Zero and unscheduled is not a reference: the term is skipped entirely.
    // (A schedule without a positive coefficient is refused by `[grpo]` itself,
    // which is why the schedule half of the rule is exercised above rather than
    // on its own.)
    let mut without = base_normalized(exhaustive_document());
    without.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    without.grpo.as_mut().expect("section").kl_schedule = None;
    without.reference = None;
    build(without, root).expect("a zero KL needs no anchor");
}

/// The other half of the anchor rule: a declared anchor with no KL term would
/// be loaded and never read.
#[test]
fn ppo_uses_its_rollout_policy_without_a_fixed_reference() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = base_normalized(exhaustive_document());
    document.run.algorithm = "ppo".to_string();
    document.grpo = None;
    document.ppo = Some(exhaustive_ppo());
    document.training.generation_concurrency = None;
    document.reference = None;
    build(document.clone(), root).expect("PPO's old policy does not require frozen base weights");
    document.reference = exhaustive_document().reference;
    let error = build(document, root).expect_err("PPO does not read a fixed reference");
    assert!(
        error
            .to_string()
            .contains("no enabled fixed-reference term"),
        "{error}"
    );
}

#[test]
fn an_anchor_no_term_scores_against_is_refused() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.grpo.as_mut().expect("section").kl_coefficient = 0.0;
    document.grpo.as_mut().expect("section").kl_schedule = None;
    let error = build(document, root).expect_err("nothing scores against this anchor");
    let message = error.to_string();
    assert!(message.contains("[reference]"), "{message}");
}

/// A declared anchor lifts the base-weight refusal, and the path and width
/// both reach the configuration.
#[test]
fn a_declared_anchor_admits_a_kl_penalty_beside_base_training() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let document = base_normalized(exhaustive_document());
    assert!(document.grpo.as_ref().expect("section").kl_coefficient > 0.0);
    let config = build(document, root).expect("a declared anchor is a fixed reference");
    let reference = config.reference.expect("[reference] was declared");
    assert_eq!(reference.model, root.join("original.gguf"));
    assert_eq!(reference.n_ctx, Some(512));

    // The width is optional.
    let mut default_width = base_normalized(exhaustive_document());
    default_width.reference.as_mut().expect("section").ctx = None;
    let config = build(default_width, root).expect("the anchor follows the training width");
    assert_eq!(config.reference.expect("section").n_ctx, None);

    // Zero is not a width.
    let mut zero = base_normalized(exhaustive_document());
    zero.reference.as_mut().expect("section").ctx = Some(0);
    let error = build(zero, root).expect_err("zero is not a context");
    assert!(error.to_string().contains("reference.ctx"), "{error}");

    // Nor a width the run's own sequences would overflow.
    let mut narrow = base_normalized(exhaustive_document());
    let training_ctx = narrow.training.ctx.expect("the document names a width");
    narrow.reference.as_mut().expect("section").ctx = Some(training_ctx - 1);
    let error = build(narrow, root).expect_err("the anchor cannot hold the rollouts");
    assert!(error.to_string().contains("narrower"), "{error}");

    let mut empty = base_normalized(exhaustive_document());
    empty.reference.as_mut().expect("section").model = PathBuf::new();
    let error = build(empty, root).expect_err("an empty path is not the document directory");
    assert!(error.to_string().contains("reference.model"), "{error}");
}

/// A selector a policy ignores is a selector the user believes is in effect.
#[test]
fn a_lora_run_refuses_a_base_selector_instead_of_ignoring_it() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.trainable = Some(TrainableToml {
        norms: Some(true),
        ..Default::default()
    });
    let error = build(document, root).expect_err("lora trains no base tensor");
    assert!(error.to_string().contains("trains none"), "{error}");
}

/// A name no optimizer answers to is refused rather than accepted and silently
/// replaced by AdamW.
#[test]
fn an_unknown_optimizer_name_is_refused_rather_than_substituted() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.training.optimizer = Some("lion".to_string());
    let error = build(document, root).expect_err("an unknown name is refused");
    assert!(error.to_string().contains("must be adamw"), "{error}");
}

/// Every name the runtime can build reaches the training configuration, which
/// is what carries it to `llama_opt_init`. A name that parsed and then arrived
/// as AdamW would be indistinguishable from a run nobody configured.
#[test]
fn a_selectable_optimizer_reaches_the_training_configuration() {
    let root = Path::new("/tmp/retrograd-round-trip");
    for (name, expected) in [
        ("adamw", OptimizerKind::AdamW),
        ("sgd", OptimizerKind::Sgd),
        ("muon", OptimizerKind::Muon),
        // "gefen" alone is the variant available first.
        ("gefen", OptimizerKind::Gefen(GefenLayout::default())),
    ] {
        let mut document = lora_normalized(exhaustive_document());
        document.training.optimizer = Some(name.to_string());
        // Only AdamW's update step writes an F16 parameter, and F16 is the
        // default adapter storage, so every other document names the dtype it
        // can write.
        document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F32);
        let config = build(document, root).expect("the optimizer is selectable");
        assert_eq!(config.training.trainable.optimizer, expected);
        // The recorded vector belongs to the optimizer that was chosen, and
        // carries the run's three universal scalars.
        assert_eq!(
            config.training.optimizer_hyperparameters.optimizer(),
            expected
        );
        assert_eq!(
            config.training.optimizer_scalar("learning_rate"),
            config.training.learning_rate
        );
    }
}

/// `[optimizer.<name>]` is wired to the optimizer that reads it: the variant
/// selects a layout, the keys land on declared rows, and a table for an
/// optimizer the run did not choose is refused rather than ignored.
#[test]
fn an_optimizer_section_is_validated_against_the_optimizer_that_reads_it() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let gefen = |section: GefenToml| {
        let mut document = lora_normalized(exhaustive_document());
        document.training.optimizer = Some("gefen".to_string());
        document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F32);
        document.optimizer = Some(OptimizerToml {
            muon: None,
            gefen: Some(section),
        });
        document
    };

    let config = build(
        gefen(GefenToml {
            variant: Some("quantized_m".to_string()),
            block_size: Some(512),
            min_numel: Some(8192),
            beta1: Some(0.8),
            ..Default::default()
        }),
        root,
    )
    .expect("a gefen section on a gefen run");
    assert_eq!(
        config.training.trainable.optimizer,
        OptimizerKind::Gefen(GefenLayout {
            variant: GefenVariant::QuantizedM,
            block_size: 512,
            min_numel: 8192,
        })
    );
    // The variant is what moves the layout version, and the slot table with it.
    assert_eq!(config.training.trainable.optimizer.layout_version(), 2);
    assert_eq!(
        config.training.optimizer_scalar("beta1"),
        0.8,
        "a declared row did not reach the vector"
    );

    // Shared-v declares no codebook at all, so a codebook-only key is refused
    // by the layout rather than by a second copy of the variant rule.
    let error = build(
        gefen(GefenToml {
            codebook_levels: Some(256),
            ..Default::default()
        }),
        root,
    )
    .expect_err("shared_v has no codebook");
    assert!(error.to_string().contains("codebook_levels"), "{error}");

    // A research option is rejected, not accepted and ignored.
    for (section, needle) in [
        (
            GefenToml {
                codebook: Some("learned".to_string()),
                ..Default::default()
            },
            "codebook",
        ),
        (
            GefenToml {
                partition: Some("discovered".to_string()),
                ..Default::default()
            },
            "partition",
        ),
        (
            GefenToml {
                block_size: Some(1000),
                ..Default::default()
            },
            "power of two",
        ),
        (
            GefenToml {
                variant: Some("shared".to_string()),
                ..Default::default()
            },
            "variant",
        ),
    ] {
        let error = build(gefen(section), root).expect_err("refused");
        assert!(error.to_string().contains(needle), "{error}");
    }

    // A table whose optimizer the run did not choose.
    let mut document = lora_normalized(exhaustive_document());
    document.optimizer = Some(OptimizerToml {
        muon: Some(MuonToml::default()),
        gefen: None,
    });
    let error = build(document, root).expect_err("adamw reads no muon keys");
    assert!(error.to_string().contains("does not use"), "{error}");
}

/// Muon's keys land on its declared rows, and its fallback rate is its own
/// value rather than the run's.
#[test]
fn muons_section_reaches_its_declared_rows() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.training.optimizer = Some("muon".to_string());
    document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F32);
    document.optimizer = Some(OptimizerToml {
        muon: Some(MuonToml {
            momentum: Some(0.9),
            nesterov: Some(false),
            ns_steps: Some(3),
            ns_epsilon: Some(1.0e-6),
            fallback_learning_rate: Some(3.0e-4),
        }),
        gefen: None,
    });
    let config = build(document, root).expect("a muon section on a muon run");
    assert_eq!(config.training.optimizer_scalar("momentum"), 0.9);
    assert!(!config.training.optimizer_toggle("nesterov", true));
    assert_eq!(config.training.optimizer_structural("ns_steps"), 3);
    assert_eq!(
        config.training.optimizer_scalar("fallback_learning_rate"),
        3.0e-4
    );

    let mut document = lora_normalized(exhaustive_document());
    document.training.optimizer = Some("muon".to_string());
    document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F32);
    document.optimizer = Some(OptimizerToml {
        muon: Some(MuonToml {
            ns_steps: Some(0),
            ..Default::default()
        }),
        gefen: None,
    });
    let error = build(document, root).expect_err("zero iterations is not an iteration count");
    assert!(error.to_string().contains("ns_steps"), "{error}");
}

/// The pair a user reaches by writing one line: the default adapter dtype is
/// F16, and the Muon kernel is F32-only. Refused at load time rather than by
/// `GGML_ABORT` in the middle of the first step.
#[test]
fn an_f32_only_optimizer_is_refused_against_the_default_f16_adapter() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.checkpoint.as_mut().unwrap().resume_from = None;
    document.training.optimizer = Some("muon".to_string());
    document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F16);
    let error = build(document, root).expect_err("the muon kernel is F32-only");
    assert!(error.is_user_error(), "{error}");
    assert!(
        error.to_string().contains("cannot write a F16 adapter"),
        "{error}"
    );
}

/// The counterpart: SGD writes F16, so the default adapter needs no override.
#[test]
fn sgd_accepts_the_default_f16_adapter() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.checkpoint.as_mut().unwrap().resume_from = None;
    document.training.optimizer = Some("sgd".to_string());
    document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F16);
    build(document, root).expect("sgd writes an F16 adapter");
}

#[test]
fn sgd_resume_uses_the_checkpoint_adapter_dtype() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let mut document = lora_normalized(exhaustive_document());
    document.training.optimizer = Some("sgd".to_string());
    document.lora.as_mut().expect("section").dtype = Some(LoraDtype::F16);
    assert!(document.checkpoint.as_ref().unwrap().resume_from.is_some());
    build(document, root).expect("the runtime checks the restored adapter's dtype");
}

/// Flattened, sorted key paths of a serialized document. Arrays are one key,
/// their contents are values, not schema.
fn keys(document: &ConfigDocument) -> Vec<String> {
    fn walk(prefix: &str, value: &toml::Value, out: &mut Vec<String>) {
        match value {
            toml::Value::Table(table) => {
                for (key, value) in table {
                    let path = if prefix.is_empty() {
                        key.clone()
                    } else {
                        format!("{prefix}.{key}")
                    };
                    walk(&path, value, out);
                }
            }
            _ => out.push(prefix.to_string()),
        }
    }
    let value = toml::Value::try_from(document).expect("a document serializes");
    let mut out = Vec::new();
    walk("", &value, &mut out);
    out.sort();
    out
}

#[test]
fn every_toml_field_survives_a_document_round_trip() {
    let written = keys(&exhaustive_document());
    assert_eq!(
        written,
        KEYS.iter().map(|key| key.to_string()).collect::<Vec<_>>(),
        "the document does not serialize to the frozen key set: a field either \
         stopped serializing, or was added without being listed"
    );
    let text = toml::to_string(&exhaustive_document()).expect("a document serializes");
    let parsed = parse_toml(&text, "round-trip").expect("a serialized document parses back");
    assert_eq!(
        keys(&parsed),
        written,
        "a key was lost on the way back in: it serializes but does not deserialize"
    );
}

/// The other half: the keys reach [`RunConfig`], not just the document.
///
/// One assertion per field of the exhaustive document, on the *value* - a
/// `build` line that copies the wrong field, or none at all, fails here with
/// the default it silently used.
#[test]
fn every_toml_field_reaches_the_run_config() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let config = build(lora_normalized(exhaustive_document()), root)
        .expect("the exhaustive document builds");
    // The two keys this normalization clears have their value coverage in
    // `the_trainable_section_is_parsed_before_the_mode_is_refused` below.
    assert_eq!(config.training.trainable.policy, TrainablePolicy::Lora);
    assert_eq!(config.training.trainable.optimizer, OptimizerKind::AdamW);
    assert_eq!(
        config.training.trainable.selector,
        TrainableSelector::default()
    );

    assert_eq!(config.model, root.join("model.gguf"));
    assert_eq!(config.output.path, root.join("out/adapter.gguf"));
    assert_eq!(config.output.kind, OutputKind::Adapter);
    let lora = config.lora.as_ref().expect("a lora run has an adapter");
    assert_eq!(lora.init_adapter, None);
    assert_eq!(lora.config.rank, 16);
    assert_eq!(lora.config.alpha, 32.0);
    assert_eq!(lora.config.seed, 7);
    assert_eq!(lora.config.dtype, LoraDtype::F16);
    assert_eq!(
        lora.config.targets,
        parse_targets(&["q".to_string(), "v".to_string()]).unwrap()
    );

    let training = &config.training;
    assert!(training.verbose);
    assert_eq!(training.n_ctx, 256);
    assert_eq!(training.n_ubatch, 64);
    // Not spelled in the file: `micro_batch * gradient_accumulation`.
    assert_eq!(training.n_batch, 256);
    assert_eq!(training.shared_prefix_fanout, SharedPrefixFanout::Exact(2));
    assert_eq!(training.threads, 3);
    assert_eq!(training.epochs, 2);
    assert_eq!(training.learning_rate, 5.0e-5);
    assert_eq!(training.weight_decay, 0.02);
    assert_eq!(training.max_grad_norm, 0.5);
    assert_eq!(training.lr_scheduler, LrScheduler::Cosine);
    assert_eq!(training.warmup_steps, 11);
    assert!(!training.fast_generation_context);
    assert_eq!(training.kv_dtype, KvDtype::F32);
    assert_eq!(training.generation_concurrency, 6);
    assert_eq!(training.generation_batch, 128);
    assert!(training.chunked_cross_entropy);
    assert_eq!(training.chunked_ce_tiles, 5);
    assert_eq!(training.chunked_ce_seq_chunk, 17);
    assert!(training.gradient_checkpointing);
    assert_eq!(training.checkpoint_every_n_layers, 3);
    assert_eq!(training.checkpoint_dtype, CheckpointDtype::F16);
    assert!(training.require_gpu_resident);
    assert_eq!(training.max_gpu_duty_cycle, Some(0.5));
    assert_eq!(training.device, "cpu".parse().unwrap());
    // Pinned by the GRPO group size, not by the file.
    assert_eq!(training.n_seq_max, 3);

    let metrics = &config.metrics;
    assert_eq!(metrics.tensorboard_dir, Some(root.join("runs/tb")));
    assert_eq!(metrics.wandb_export_dir, Some(root.join("runs/wandb")));

    let evaluation = config.evaluation.as_ref().expect("[evaluation] builds");
    assert_eq!(evaluation.data, root.join("data/eval.jsonl"));
    assert_eq!(evaluation.every_iterations, 2);
    assert_eq!(evaluation.patience, Some(4));
    assert_eq!(evaluation.min_delta, 0.125);
    assert_eq!(evaluation.max_examples, Some(9));

    let observe = config.observe.as_ref().expect("[observe] builds");
    assert_eq!(observe.directory, root.join("out/observe"));
    assert_eq!(observe.every, 3);
    assert_eq!(observe.max_text_chars, 2000);

    let checkpoint = config.checkpoint.as_ref().expect("[checkpoint] builds");
    assert_eq!(checkpoint.directory, root.join("ckpt"));
    assert_eq!(checkpoint.mode, CheckpointMode::StepsAndBestEval);
    assert_eq!(checkpoint.every_steps, Some(13));
    assert_eq!(
        checkpoint.resume_from,
        Some(root.join("ckpt/step-13.state"))
    );

    let Algorithm::Grpo(grpo) = &config.algorithm else {
        panic!("run.algorithm = 'grpo' builds a GRPO run");
    };
    assert_eq!(grpo.prompts, root.join("data/prompts.txt"));
    assert_eq!(grpo.reward_command, ["score", "--grpo"]);
    assert_eq!(grpo.reward_protocol.mode, RewardMode::Persistent);
    assert_eq!(
        grpo.reward_protocol.timeout,
        std::time::Duration::from_secs(90)
    );
    assert_eq!(grpo.updates, 5);
    assert_eq!(grpo.prompts_per_update, 2);
    assert_eq!(grpo.group_size, 3);
    assert_eq!(grpo.grpo_epochs, 2);
    assert_eq!(grpo.clip_range_low, 0.2);
    assert_eq!(grpo.clip_range_high, 0.28);
    assert_eq!(grpo.kl_coefficient, 0.01);
    assert!(grpo.mask_truncated);
    assert_eq!(grpo.baseline, AdvantageBaseline::LeaveOneOut);
    assert_eq!(grpo.prompt_order, PromptOrder::Shuffled);
    assert_eq!(grpo.max_stalled_updates, 9);
    let penalty = grpo.overlong_penalty.expect("[grpo.overlong_penalty]");
    assert_eq!(penalty.buffer_tokens, 12);
    assert_eq!(penalty.max_penalty, 0.75);
    let schedule = grpo.kl_schedule.expect("[grpo.kl_schedule]");
    assert_eq!(schedule.warmup_updates, 4);
    assert_eq!(schedule.target, Some(0.02));
    let dynamic = grpo.dynamic_sampling.expect("[grpo.dynamic_sampling]");
    assert_eq!(dynamic.max_resample_factor, 3);
    let judge = grpo.judge.as_ref().expect("[grpo.judge]");
    assert_eq!(judge.weight, 0.4);
    assert_eq!(judge.max_dropped_fraction, 0.25);
    assert!(matches!(
        judge.failure,
        retrograd_agent_core::config::JudgeFailurePolicy::Fail
    ));
    let retrograd_spec::judge::JudgeConfig::Command {
        command,
        timeout_secs,
    } = &judge.config
    else {
        panic!("the judge keeps the command backend it was given");
    };
    assert_eq!(command, &["judge".to_string()]);
    assert_eq!(*timeout_secs, 45);
    assert_eq!(grpo.sampling.temperature, 1.0);
    assert_eq!(grpo.sampling.top_p, 1.0);
    assert_eq!(grpo.sampling.max_new_tokens, 48);
    assert_eq!(grpo.sampling.seed, 1234);
}

/// `[ppo]` and `[sft]` cannot share a document with `[grpo]`, so they get the
/// same treatment on their own.
#[test]
fn the_other_algorithm_sections_reach_the_run_config() {
    let root = Path::new("/tmp/retrograd-round-trip");

    let mut document = lora_normalized(exhaustive_document());
    document.run.algorithm = "ppo".to_string();
    document.grpo = None;
    document.ppo = Some(exhaustive_ppo());
    document.reference = None;
    // `generation_concurrency` is GRPO-only, and PPO has no group size to pin
    // `n_seq_max` from.
    document.training.generation_concurrency = None;
    let config = build(document, root).expect("the PPO document builds");
    let Algorithm::Ppo(ppo) = &config.algorithm else {
        panic!("run.algorithm = 'ppo' builds a PPO run");
    };
    assert_eq!(ppo.prompts, root.join("data/prompts.txt"));
    assert_eq!(ppo.reward_command, ["score", "--ppo"]);
    assert_eq!(ppo.reward_protocol.mode, RewardMode::OneShot);
    assert_eq!(
        ppo.reward_protocol.timeout,
        std::time::Duration::from_secs(45)
    );
    assert_eq!(ppo.updates, 7);
    assert_eq!(ppo.rollout_batch_size, 4);
    assert_eq!(ppo.ppo_epochs, 2);
    assert_eq!(ppo.clip_range, 0.25);
    assert_eq!(ppo.kl_coefficient, 0.05);
    assert!(ppo.critic.enabled);
    assert_eq!(ppo.critic.gamma, 0.97);
    assert_eq!(ppo.critic.gae_lambda, 0.9);
    assert_eq!(ppo.critic.value_lr, 3.0e-4);
    assert_eq!(ppo.critic.value_epochs, 3);
    assert_eq!(ppo.critic.feature_dtype, FeatureDtype::Bf16);
    assert_eq!(ppo.sampling.temperature, 0.8);
    assert_eq!(ppo.sampling.top_p, 0.95);
    assert_eq!(ppo.sampling.max_new_tokens, 48);
    assert_eq!(ppo.sampling.seed, 1234);

    let mut document = lora_normalized(exhaustive_document());
    document.run.algorithm = "distill".to_string();
    // `[observe]` is refused where nothing is rolled out.
    document.observe = None;
    document.grpo = None;
    document.distill = Some(exhaustive_distill());
    // 3 samples over 2 prompts is 6 rollouts an update, and the exhaustive
    // document asks for a concurrency of 6.
    let config = build(document, root).expect("the distillation document builds");
    let Algorithm::Distill(distill) = &config.algorithm else {
        panic!("run.algorithm = 'distill' builds a distillation run");
    };
    assert_eq!(distill.teacher_path, root.join("models/teacher.gguf"));
    assert_eq!(distill.prompts, root.join("data/prompts.txt"));
    assert_eq!(distill.updates, 11);
    assert_eq!(distill.prompts_per_update, 2);
    assert_eq!(distill.samples_per_prompt, 3);
    assert_eq!(distill.distill_epochs, 2);
    assert_eq!(distill.clip_range_low, 0.15);
    assert_eq!(distill.clip_range_high, 0.3);
    assert_eq!(distill.weight_clip, 4.5);
    assert_eq!(distill.kl_coefficient, 0.03);
    assert!(distill.mask_truncated);
    assert_eq!(distill.prompt_order, PromptOrder::Shuffled);
    assert_eq!(distill.sampling.temperature, 1.0);
    assert_eq!(distill.sampling.top_p, 1.0);
    assert_eq!(distill.sampling.max_new_tokens, 48);
    assert_eq!(distill.sampling.seed, 1234);
    // `samples_per_prompt` pins the same geometry `group_size` does.
    assert_eq!(config.training.n_seq_max, 3);
    assert_eq!(distill.mode, DistillMode::OnPolicy);

    // The offline mode of the same section.
    let mut document = lora_normalized(exhaustive_document());
    document.run.algorithm = "distill".to_string();
    document.observe = None;
    document.grpo = None;
    document.distill = Some(exhaustive_distill_offline());
    // Offline distillation does not generate, so the rollout-only geometry the
    // exhaustive document pins would be describing a sampler that never runs.
    document.training.generation_concurrency = None;
    // Offline distillation carries no KL term, so no anchor.
    document.reference = None;
    let config = build(document, root).expect("the offline distillation document builds");
    let Algorithm::Distill(distill) = &config.algorithm else {
        panic!("run.algorithm = 'distill' builds a distillation run");
    };
    let DistillMode::TopkOffline(offline) = &distill.mode else {
        panic!("distill.mode = 'topk_offline' builds an offline run");
    };
    assert_eq!(offline.data, root.join("data/corpus.jsonl"));
    assert_eq!(offline.sidecar, root.join("data/corpus.topk"));
    assert_eq!(offline.epochs, 4);
    assert_eq!(distill.teacher_path, root.join("models/teacher.gguf"));

    // The three offline keys next to `mode = "on_policy"` are a document whose
    // author expects a sidecar to be read, and it would not be.
    let mut document = lora_normalized(exhaustive_document());
    document.run.algorithm = "distill".to_string();
    document.observe = None;
    document.grpo = None;
    let mut mixed = exhaustive_distill();
    mixed.sidecar = Some(PathBuf::from("data/corpus.topk"));
    document.distill = Some(mixed);
    let error = build(document, root)
        .expect_err("offline keys under on_policy are refused")
        .to_string();
    assert!(error.contains("topk_offline"), "{error}");

    // And an offline document without its sidecar is refused by name rather
    // than falling back to a mode nobody asked for.
    let mut document = lora_normalized(exhaustive_document());
    document.run.algorithm = "distill".to_string();
    document.observe = None;
    document.grpo = None;
    document.training.generation_concurrency = None;
    let mut incomplete = exhaustive_distill_offline();
    incomplete.sidecar = None;
    document.distill = Some(incomplete);
    let error = build(document, root)
        .expect_err("an offline document without a sidecar is refused")
        .to_string();
    assert!(error.contains("distill.sidecar"), "{error}");

    let mut document = lora_normalized(exhaustive_document());
    document.run.algorithm = "sft".to_string();
    document.observe = None;
    document.grpo = None;
    document.sft = Some(exhaustive_sft());
    document.training.generation_concurrency = None;
    // SFT carries no KL term, so no anchor either.
    document.reference = None;
    let config = build(document, root).expect("the SFT document builds");
    let Algorithm::Sft(sft) = &config.algorithm else {
        panic!("run.algorithm = 'sft' builds an SFT run");
    };
    assert_eq!(sft.data, root.join("data/sft.jsonl"));
    assert_eq!(sft.data_format, DataFormat::ChatJsonl);
    assert!(!sft.shuffle);
    // The row order lives on `TrainConfig`, where the runtime reads it.
    assert!(!config.training.shuffle_dataset);
    assert_eq!(config.training.shuffle_seed, 7);
}

/// The other side of the same coin: what a document that says nothing gets.
///
/// A default that moves is a silent change in every run that did not spell the
/// field, which is most of them. This freezes the ones `build` and its helpers
/// choose themselves - `TrainConfig::default` and `CriticConfig::default` own
/// the rest and are frozen where they live.
#[test]
fn the_defaults_behind_an_absent_field_are_frozen() {
    let root = Path::new("/tmp/retrograd-round-trip");
    let document = parse_toml(
        r#"
[run]
algorithm = "grpo"

[model]
path = "model.gguf"

[output]
path = "adapter.gguf"

[lora]

[training]
ctx = 256
micro_batch = 64

[grpo]
prompts = "prompts.txt"
reward_command = ["score"]
updates = 5
prompts_per_update = 2
group_size = 3
grpo_epochs = 1
clip_range_low = 0.2
clip_range_high = 0.2
kl_coefficient = 0.0

[grpo.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 32
seed = 0
"#,
        "defaults",
    )
    .expect("the minimal document parses");
    let config = build(document, root).expect("the minimal document builds");

    assert!(!config.training.verbose);
    let lora = config.lora.as_ref().expect("a lora run has an adapter");
    assert_eq!(lora.config.rank, 8);
    assert_eq!(lora.config.alpha, 16.0);
    assert_eq!(lora.config.seed, DEFAULT_SEED);
    assert_eq!(lora.config.dtype, LoraDtype::default());
    assert_eq!(
        lora.config.targets,
        parse_targets(&DEFAULT_TARGETS.map(String::from)).unwrap()
    );
    // `[output]` names no kind, so the policy's default is what it gets.
    assert_eq!(config.output.kind, OutputKind::Adapter);
    // A rollout algorithm pins the optimizer window to the whole context.
    assert_eq!(config.training.n_batch, config.training.n_ctx);
    // `min(prompts_per_update * group_size, n_batch, 256)`.
    assert_eq!(config.training.generation_concurrency, 6);
    assert!(config.evaluation.is_none());
    assert!(config.checkpoint.is_none());
    assert!(config.observe.is_none());
    assert!(config.metrics.tensorboard_dir.is_none());
    assert!(config.metrics.wandb_export_dir.is_none());

    let Algorithm::Grpo(grpo) = &config.algorithm else {
        panic!("run.algorithm = 'grpo' builds a GRPO run");
    };
    assert!(!grpo.mask_truncated);
    assert_eq!(grpo.baseline, AdvantageBaseline::Mean);
    assert_eq!(grpo.prompt_order, PromptOrder::Sequential);
    assert_eq!(grpo.max_stalled_updates, DEFAULT_MAX_STALLED_UPDATES);
    assert!(grpo.overlong_penalty.is_none());
    assert!(grpo.kl_schedule.is_none());
    assert!(grpo.dynamic_sampling.is_none());
    assert!(grpo.judge.is_none());
}

/// `[grpo.kl_schedule]` is the one sub-table whose absent field has a default
/// that is not `None`, and it is spelled in `build_grpo` rather than on the
/// type.
#[test]
fn an_empty_kl_schedule_warms_up_over_zero_updates() {
    let schedule = build_grpo(
        GrpoToml {
            kl_schedule: Some(KlScheduleToml {
                warmup_updates: None,
                target: None,
            }),
            ..exhaustive_document().grpo.expect("the GRPO section")
        },
        Path::new("/tmp/retrograd-round-trip"),
    )
    .expect("the GRPO section builds")
    .kl_schedule
    .expect("[grpo.kl_schedule]");
    assert_eq!(schedule.warmup_updates, 0);
    assert_eq!(schedule.target, None);
}

/// **Why the `*Toml` doubling is not a doubling to remove.**
///
/// The file form and the validated form differ by paths, enums and defaults,
/// so the copy is the price of validation. That leaves the handful of types
/// whose `build_*` is nothing but `unwrap_or(default)`, `CriticToml` above all,
/// and which therefore look collapsible into the runtime type with
/// `#[serde(default)]`.
///
/// They are not, and the reason is here: an `Option` field in a `*Toml` type
/// does not carry "a default in waiting", it carries **presence**. Paired with
/// `skip_serializing_if`, that is what keeps a serialized document down to what
/// its author actually wrote. Collapse `CriticToml` into `CriticConfig` and the
/// one line below becomes five, `gae_lambda` among them printed as
/// `0.949999988079071` - an `f32` default rendered at `f64` precision, in a
/// document the server publishes as `effective_config`.
///
/// So this is asserted on the **serialized text**, not on the parsed value: the
/// parsed value is identical either way, which is exactly why the regression
/// would otherwise go through.
#[test]
fn a_field_the_author_did_not_write_does_not_appear_in_the_document() {
    let only_enabled = CriticToml {
        enabled: Some(true),
        ..CriticToml::default()
    };
    let rendered = toml::to_string(&only_enabled).expect("a critic table serializes");
    assert_eq!(
        rendered, "enabled = true\n",
        "a `*Toml` type must serialize only the fields it was given; see this \
         test's documentation before making one of them non-optional"
    );

    // And the empty table disappears entirely rather than rendering five
    // defaults - the same property one level up.
    let empty = CriticToml::default();
    assert_eq!(toml::to_string(&empty).expect("an empty critic table"), "");
}
