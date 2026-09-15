//! The distillation teacher against a real GGUF: two models loaded at once in
//! one process, the compatibility gate, and the null test the whole plan rests
//! on. Skipped when no local model exists (override with `RETRO_CPU_FIXTURE`).
//!
//! Both models are the same fixture. That is not a shortcut around a missing
//! second model: teacher == student is the one configuration whose correct
//! answer is known in advance, and every sign error, mask misalignment or
//! off-by-one index in the objective breaks it.

mod common;

use common::serialize_models;
use retrograd::config::{DistillConfig, PromptOrder};
use retrograd::dataset::topk::{TopKHeader, TopKSidecar};
use retrograd::training::batch::TrainSequence;
use retrograd::training::distill::{SharedTeacher, Teacher, token_advantages};
use retrograd::{
    Device, LoraConfig, SamplingParams, TargetSet, TrainConfig, Trainer, WeightedBatch,
};

const PROMPT: &str = "The quick brown fox";

/// Two sequences per group, so the teacher's scoring call exercises the
/// shared-prefix branch path rather than a degenerate single row.
fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 32,
        n_seq_max: 2,
        epochs: 1,
        learning_rate: 1.0e-4,
        // The null test asserts the adapter does not move. Decoupled weight
        // decay moves it on a zero gradient, so the claim is only about the
        // objective when the run does not also shrink the weights on its own.
        weight_decay: 0.0,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    cfg.targets = TargetSet::Patterns(vec!["blk.13.shortconv.out_proj.weight".to_string()]);
    cfg
}

/// One prompt, two completions the student actually sampled: the shape the
/// on-policy loop produces.
fn sampled_group(student: &mut Trainer) -> (usize, Vec<TrainSequence>) {
    let prompt = student.tokenize_text(PROMPT).expect("tokenize");
    let sequences = [11_u32, 23]
        .into_iter()
        .map(|seed| {
            let generation = student
                .generate(
                    &prompt,
                    &SamplingParams {
                        temperature: 1.0,
                        top_p: 1.0,
                        max_new_tokens: 8,
                        seed,
                    },
                )
                .expect("generate");
            let mut tokens = prompt.clone();
            tokens.extend_from_slice(&generation.tokens);
            let mut train_mask = vec![false; prompt.len()];
            train_mask.resize(tokens.len(), true);
            TrainSequence {
                tokens,
                old_logprobs: generation.logprobs,
                train_mask,
                reward: 0.0,
                group_id: 0,
                intermediate_returns: Vec::new(),
            }
        })
        .collect::<Vec<_>>();
    for sequence in &sequences {
        sequence.validate().expect("valid sampled sequence");
    }
    (prompt.len(), sequences)
}

/// Teacher-forced log-probabilities of the student's own tokens, read through
/// the same batched call the teacher uses. The sampler's `old_logprobs` are not
/// interchangeable with these: generation runs in the fast F16 context, and the
/// ulp-level difference that follows is a property of the sampler, not of the
/// objective under test.
fn scored_behaviour(student: &mut Trainer, sequences: &[TrainSequence]) -> Vec<Vec<f32>> {
    let inputs = sequences
        .iter()
        .map(|sequence| {
            let first = sequence
                .train_mask
                .iter()
                .position(|&train| train)
                .expect("a supervised token");
            (sequence.tokens.as_slice(), first)
        })
        .collect::<Vec<_>>();
    student
        .score_token_suffix_batch(&inputs)
        .expect("student suffix scores")
}

/// Two `Trainer`s on one device, and what the second one costs.
#[test]
fn a_teacher_coexists_with_the_student_and_scores_the_same_tokens() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut student = Trainer::new(&model, config()).expect("load student");
    let student_alone = student.memory_report().expect("student memory");

    // The teacher is created *after* the student and through the plain
    // constructor, so it never tries to change the runtime policy the first
    // context locked in.
    let mut teacher = Teacher::open(&model, &config()).expect("load teacher");
    teacher
        .compatibility(&student)
        .expect("the fixture is compatible with itself");

    let teacher_memory = teacher.memory_report().expect("teacher memory");
    let student_after = student.memory_report().expect("student memory");
    eprintln!(
        "student alone: weights={} optimizer_kv={} lora={} adamw={} host={} device={}\n\
         teacher:       weights={} optimizer_kv={} lora={} adamw={} host={} device={}\n\
         student after: host={} device={}",
        student_alone.model_weight_bytes,
        student_alone.optimizer_kv_bytes,
        student_alone.lora_parameter_bytes,
        student_alone.adamw_momenta_bytes,
        student_alone.host_bytes,
        student_alone.device_bytes,
        teacher_memory.model_weight_bytes,
        teacher_memory.optimizer_kv_bytes,
        teacher_memory.lora_parameter_bytes,
        teacher_memory.adamw_momenta_bytes,
        teacher_memory.host_bytes,
        teacher_memory.device_bytes,
        student_after.host_bytes,
        student_after.device_bytes,
    );

    // The invariant the plan loads a second model under: weights and KV, and
    // nothing an optimizer would need.
    assert_eq!(teacher_memory.lora_parameter_bytes, 0);
    assert_eq!(teacher_memory.lora_gradient_bytes, 0);
    assert_eq!(teacher_memory.adamw_momenta_bytes, 0);
    assert!(teacher_memory.model_weight_bytes > 0);

    let (_, sequences) = sampled_group(&mut student);
    let expected = scored_behaviour(&mut student, &sequences);
    let members = sequences.iter().collect::<Vec<_>>();
    let scored = teacher.logprobs_group(&members).expect("teacher scores");

    assert_eq!(scored.len(), sequences.len());
    for (row, (teacher_row, student_row)) in scored.iter().zip(&expected).enumerate() {
        assert_eq!(
            teacher_row.len(),
            sequences[row].old_logprobs.len(),
            "row {row}: the teacher scored a different number of tokens"
        );
        assert_eq!(
            teacher_row, student_row,
            "row {row}: the same model scored the same tokens differently"
        );
    }
}

/// The truncated distribution, against the scalar scorer, which only returns
/// the value of a token the caller already has.
///
/// The two are compared on the *same* conditioning, which is the whole
/// difficulty: rescoring a sequence of argmax tokens would change every context
/// past the first position. So the sampled sequence is scored both ways, and
/// wherever the sampled token appears in its own top `k` the two values must be
/// equal - the same expression, spelled once in each path. The argmax bound
/// holds everywhere and is what says the ordering is not merely internally
/// consistent.
#[test]
fn the_truncated_distribution_agrees_with_the_scalar_scorer() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, config()).expect("load trainer");
    let (n_prompt, sequences) = sampled_group(&mut trainer);
    let tokens = sequences[0].tokens.clone();

    const K: usize = 5;
    let top = trainer
        .top_logprobs_suffix(&tokens, n_prompt, K)
        .expect("top-k suffix");
    assert_eq!(top.k(), K);
    assert_eq!(top.rows(), tokens.len() - n_prompt);

    for (row, (ids, logprobs)) in top.iter().enumerate() {
        // Decreasing, and totally ordered by (log-probability, id): the
        // contract the sidecar and the top-1 metric both read column zero under.
        for j in 1..K {
            assert!(
                logprobs[j - 1] > logprobs[j]
                    || (logprobs[j - 1] == logprobs[j] && ids[j - 1] < ids[j]),
                "row {row}: entry {j} is not below entry {} ({logprobs:?} / {ids:?})",
                j - 1
            );
        }
        // Log-probabilities, not logits: k of them cannot carry more than the
        // whole distribution.
        let mass: f64 = logprobs.iter().map(|&value| f64::from(value).exp()).sum();
        assert!(
            mass > 0.0 && mass <= 1.0 + 1e-4,
            "row {row}: the top {K} carry {mass} of the distribution"
        );
    }

    // The value, against the only other path that produces it. Both reductions
    // run on the host here (CPU device), so this is an equality and not a
    // tolerance: `top_logprobs_suffix` spells the same expression the scalar
    // scorer does, deliberately.
    let scalar = trainer
        .score_token_suffix(&tokens, n_prompt)
        .expect("scalar suffix scores");
    assert_eq!(scalar.len(), top.rows());
    let mut matched = 0_usize;
    for (row, ((ids, logprobs), sampled)) in top.iter().zip(&scalar).enumerate() {
        assert!(
            logprobs[0] >= *sampled,
            "row {row}: the argmax is below the sampled token ({} < {sampled})",
            logprobs[0]
        );
        if let Some(position) = ids.iter().position(|&id| id == tokens[n_prompt + row]) {
            assert_eq!(
                logprobs[position], *sampled,
                "row {row}: the two paths disagree on the sampled token's log-probability"
            );
            matched += 1;
        }
    }
    // A temperature-1 sample lands in its own top 5 most of the time; zero
    // matches would mean the two paths were never actually compared.
    assert!(
        matched > 0,
        "no sampled token appeared in its own top {K}, so nothing was compared"
    );

    // `k = 1` is the same row truncated, not a different reduction.
    let single = trainer
        .top_logprobs_suffix(&tokens, n_prompt, 1)
        .expect("top-1 suffix");
    assert_eq!(
        single.argmax().collect::<Vec<_>>(),
        top.argmax().collect::<Vec<_>>(),
        "k = 1 and k = 5 disagree on the leading entry"
    );

    // The refusals, at the boundary rather than inside a row.
    assert!(trainer.top_logprobs_suffix(&tokens, n_prompt, 0).is_err());
    assert!(
        trainer
            .top_logprobs_suffix(&tokens, n_prompt, usize::MAX)
            .is_err()
    );
    assert!(trainer.top_logprobs_suffix(&tokens, 0, 1).is_err());
}

/// What the gate refuses, and how loudly.
#[test]
fn an_unusable_teacher_is_refused_by_a_gate_that_names_it() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let missing = std::env::temp_dir().join("retrograd-distill-absent-teacher.gguf");
    let error = Teacher::open(&missing, &config()).expect_err("a missing teacher is refused");
    assert!(error.is_user_error(), "{error}");
    assert!(
        error
            .to_string()
            .contains("retrograd-distill-absent-teacher"),
        "{error}"
    );

    let student = Trainer::new(&model, config()).expect("load student");
    for sentence in retrograd::training::distill::WITNESS_SENTENCES {
        let witness = student.tokenize_text(sentence).expect("tokenize witness");
        assert!(
            witness.len() > 1,
            "a witness sentence that tokenizes to one id would compare nothing: {sentence:?}"
        );
    }

    // The real negative case needs a second architecture, which this lane does
    // not ship. When a developer has one, it is the case that matters: a
    // teacher whose vocabulary differs must be refused before it scores
    // anything, because what it would return are the log-probabilities of
    // other tokens.
    let Some(other) = common::falcon_h1_model_path_if_available() else {
        eprintln!("skipping the mismatched pair: no second architecture available");
        return;
    };
    let stranger = Teacher::open(&other, &config()).expect("load the other model");
    let error = stranger
        .compatibility(&student)
        .expect_err("two architectures must not pass the gate");
    assert!(error.is_user_error(), "{error}");
    assert!(error.to_string().contains("tokenizer"), "{error}");
}

/// The null test. Teacher and student are the same model, and the
/// adapter is fresh, so `B = 0` and the student's policy *is* the base model's.
/// Then every advantage is exactly zero, every token weight is zero, and one
/// full update must leave the adapter unchanged - not close, unchanged.
#[test]
fn an_identical_teacher_leaves_the_adapter_untouched() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut student = Trainer::new(&model, config()).expect("load student");
    student.create_lora(&lora()).expect("create lora");
    let mut teacher = Teacher::open(&model, &config()).expect("load teacher");
    teacher.compatibility(&student).expect("compatible pair");

    let (n_prompt, sequences) = sampled_group(&mut student);
    let behaviour = scored_behaviour(&mut student, &sequences);
    let members = sequences.iter().collect::<Vec<_>>();
    let scored = teacher.logprobs_group(&members).expect("teacher scores");

    let advantages = scored
        .iter()
        .zip(&behaviour)
        .map(|(teacher_row, student_row)| {
            token_advantages(teacher_row, student_row, 5.0).expect("advantages")
        })
        .collect::<Vec<_>>();
    for (row, values) in advantages.iter().enumerate() {
        assert!(
            values.iter().all(|&value| value == 0.0),
            "row {row}: an identical teacher produced non-zero advantages {values:?}"
        );
    }

    // One update over the whole group, with those weights and nothing else.
    let n_ctx = student.context_size().expect("context");
    let pad = student.eos_token().expect("eos");
    let mut tokens = Vec::new();
    let mut labels = Vec::new();
    let mut weights = Vec::new();
    for (sequence, values) in sequences.iter().zip(&advantages) {
        assert!(
            sequence.tokens.len() <= n_ctx,
            "sequence exceeds the window"
        );
        let mut row_weights = vec![0.0_f32; n_ctx];
        let mut row_labels = vec![-1_i32; n_ctx];
        for (offset, &advantage) in values.iter().enumerate() {
            // Position `n_prompt - 1 + offset` predicts the completion token at
            // `n_prompt + offset`: the label of a position is the next token.
            let position = n_prompt - 1 + offset;
            row_labels[position] = sequence.tokens[position + 1];
            row_weights[position] = advantage;
        }
        let mut row_tokens = sequence.tokens.clone();
        row_tokens.resize(n_ctx, pad);
        tokens.extend(row_tokens);
        labels.extend(row_labels);
        weights.extend(row_weights);
    }
    let batch = WeightedBatch {
        tokens,
        labels,
        weights,
        n_rows: sequences.len(),
        n_ctx,
        n_topk: 1,
    };

    let before = std::env::temp_dir().join("retrograd-distill-null-before.gguf");
    let after = std::env::temp_dir().join("retrograd-distill-null-after.gguf");
    student.save_lora(&before).expect("save adapter");
    let metrics = student.train_weighted(&batch, 0).expect("weighted step");
    student.save_lora(&after).expect("save adapter");

    assert_eq!(
        metrics.train_loss, 0.0,
        "zero weights produced a non-zero weighted loss"
    );
    let before_bytes = std::fs::read(&before).expect("read adapter");
    let after_bytes = std::fs::read(&after).expect("read adapter");
    assert_eq!(
        before_bytes, after_bytes,
        "a zero-advantage update moved the adapter"
    );
    let _ = std::fs::remove_file(&before);
    let _ = std::fs::remove_file(&after);
}

/// The null test of the *measurement* path, symmetric to the null test of the
/// objective.
///
/// Teacher == student is again the one configuration whose answer is known in
/// advance, and it pins both halves of a distillation report at once: a model
/// cannot diverge from itself, so the held-out KL is exactly zero, and it cannot
/// disagree with itself about an argmax, so top-1 agreement is exactly one. Any
/// sign error, mask misalignment or off-by-one between the two scoring calls
/// breaks one of the two.
#[test]
fn a_measurement_against_an_identical_teacher_reports_no_divergence() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("retrograd-distill-eval-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the measurement directory");
    let held_out = dir.join("held-out.jsonl");
    std::fs::write(
        &held_out,
        format!(
            "{{\"messages\":[{{\"role\":\"user\",\"content\":\"{PROMPT}\"}}]}}\n\
             {{\"messages\":[{{\"role\":\"user\",\"content\":\"Name three colours\"}},\
             {{\"role\":\"assistant\",\"content\":\"red, green, blue\"}}]}}\n"
        ),
    )
    .expect("write the held-out file");

    let mut student = Trainer::new(&model, config()).expect("load student");
    let config_toml = DistillConfig {
        mode: retrograd::config::DistillMode::OnPolicy,
        teacher_path: model.clone(),
        prompts: held_out.clone(),
        updates: 1,
        prompts_per_update: 1,
        samples_per_prompt: 1,
        distill_epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        weight_clip: 5.0,
        kl_coefficient: 0.0,
        mask_truncated: true,
        prompt_order: PromptOrder::Sequential,
        sampling: SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: 11,
        },
    };
    let teacher = SharedTeacher::new();
    let training = config();

    let metrics = retrograd::training::distill::evaluate(
        &mut student,
        &config_toml,
        &training,
        &teacher,
        &held_out,
        None,
    )
    .expect("evaluate against an identical teacher");
    // Negated once, in `evaluate`: the controller compares "larger is better",
    // and zero divergence is the best a run can reach. `-0.0` and `0.0` both
    // satisfy this, and neither is a tolerance.
    assert_eq!(metrics.mean_reward, 0.0);
    assert_eq!(metrics.reward_min, 0.0);
    assert_eq!(metrics.reward_max, 0.0);
    assert_eq!(metrics.examples, 2);

    // The teacher was loaded by the evaluation and is the one the bench reuses:
    // a second `Teacher` would be a second model resident beside the student.
    assert!(teacher.is_open());

    let bench = retrograd::training::distill::benchmark(
        &mut student,
        &config_toml,
        &training,
        &teacher,
        &held_out,
        None,
    )
    .expect("bench against an identical teacher");
    assert_eq!(bench.teacher_kl_mean(), Some(0.0));
    assert_eq!(bench.top1_agreement(), Some(1.0));
    // One of the two held-out lines carries an assistant turn, so the third
    // figure exists - and it is a perplexity of a real model on real text, so
    // the only thing worth asserting about it is that it is a number above one.
    assert_eq!(bench.reference_examples, 1);
    let perplexity = bench.reference_perplexity().expect("a reference answer");
    assert!(perplexity > 1.0 && perplexity.is_finite(), "{perplexity}");

    let _ = std::fs::remove_dir_all(&dir);
}

/// The null test of the objective again, but end to end through the CLI: the
/// config document, the variant wiring, the sampler, the teacher, the advantage
/// and the optimizer epoch, in one process the test does not drive itself.
///
/// The claim is the same one and it is checked the same way: teacher == student
/// and a fresh adapter mean `B = 0`, so every advantage is exactly zero and the
/// exported adapter must be the one `create_lora` would have produced from the
/// same seed - byte for byte, not close. `weight_decay` is pinned to zero for
/// the reason the first null test gives: decoupled decay moves a weight whose gradient is
/// zero, and inheriting the default would make this test depend on a setting it
/// does not name.
#[test]
fn a_distillation_update_against_an_identical_teacher_writes_an_untouched_adapter() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let dir = std::env::temp_dir().join(format!("retrograd-distill-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create the run directory");
    std::fs::write(
        dir.join("prompts.jsonl"),
        format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"{PROMPT}\"}}]}}\n"),
    )
    .expect("write the prompt file");
    let document = format!(
        "[run]\nalgorithm='distill'\n\
         [model]\npath='{model}'\ndevice='cpu'\n\
         [lora]\noutput='adapter.gguf'\nrank=2\nalpha=4.0\nseed=7\ntargets=['{target}']\n\
         [training]\nctx=64\nmicro_batch=64\nepochs=1\nlr=0.001\nweight_decay=0.0\n\
         [distill]\nteacher_path='{model}'\nprompts='prompts.jsonl'\n\
         updates=1\nprompts_per_update=1\nsamples_per_prompt=2\n\
         [distill.sampling]\ntemperature=1.0\ntop_p=1.0\nmax_new_tokens=8\nseed=11\n",
        model = model.display(),
        target = "blk.13.shortconv.out_proj.weight",
    );
    let config_path = dir.join("run.toml");
    std::fs::write(&config_path, document).expect("write the config");

    let output = std::process::Command::new(env!("CARGO_BIN_EXE_retrograd"))
        .args(["train", config_path.to_str().unwrap()])
        .env("NO_COLOR", "1")
        .output()
        .expect("run the distillation CLI");
    let diagnostics = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(output.status.success(), "{diagnostics}");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        stdout.contains("done update=1/1 distill_epochs_per_update=1"),
        "{stdout}"
    );
    // The loss the weighted objective reports is the surrogate over zero
    // advantages, so it is zero *before* the adapter is even looked at.
    assert!(stdout.contains("train_loss=0.000000"), "{stdout}");

    // The reference: what the same LoRA declaration produces before any step.
    let reference = dir.join("fresh.gguf");
    {
        let mut fresh = Trainer::new(&model, config()).expect("load the reference student");
        fresh.create_lora(&lora()).expect("create lora");
        fresh.save_lora(&reference).expect("save the fresh adapter");
    }
    let trained = std::fs::read(dir.join("adapter.gguf")).expect("read the trained adapter");
    let fresh = std::fs::read(&reference).expect("read the fresh adapter");
    assert_eq!(
        trained, fresh,
        "an identical teacher moved the adapter: the update is not a no-op"
    );
    std::fs::remove_dir_all(dir).expect("clean up");
}

/// The null test of the offline top-k path, symmetric to the objective's.
///
/// A sidecar whose `k = 1` entry is the corpus's own reference token with all
/// the mass is *the SFT objective written as a distribution*. So the run it
/// produces must be the SFT run: the same loss, and - since the gradient is what
/// moves the adapter and nothing else does - the same adapter bytes, step for
/// step. Any sign error, off-by-one between a sidecar block and a position, or
/// mistaken normalization (dividing by `k` instead of by the active positions)
/// breaks it.
///
/// Run on both optimizer paths. They are different code: the dense one writes
/// `k` entries into a `[n_vocab, n_tokens]` label row, the fused one hands the
/// `[K, n_tokens]` targets straight to the operator, and each was generalized
/// to `k` targets separately.
#[test]
fn a_unit_sidecar_reproduces_the_sft_run_on_both_optimizer_paths() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    for chunked in [false, true] {
        let training = TrainConfig {
            chunked_cross_entropy: chunked,
            ..config()
        };

        // The corpus: two rows of the model's own tokens, labels shifted by one,
        // which is exactly what `retrograd-dataset` produces for a text file.
        let reference = Trainer::new(&model, training.clone()).expect("load reference");
        let n_ctx = reference.context_size().expect("context");
        let text = "Distillation reproduces supervised fine-tuning when the teacher's \
                    distribution is a point mass on the reference token. "
            .repeat(24);
        let encoded = reference.tokenize_text(&text).expect("tokenize");
        assert!(
            encoded.len() > 2 * n_ctx + 1,
            "the fixture tokenized {} tokens, fewer than the {} the two rows need",
            encoded.len(),
            2 * n_ctx + 1
        );
        drop(reference);
        let mut tokens = Vec::with_capacity(2 * n_ctx);
        let mut labels = Vec::with_capacity(2 * n_ctx);
        for row in 0..2 {
            let start = row * n_ctx;
            tokens.extend_from_slice(&encoded[start..start + n_ctx]);
            labels.extend_from_slice(&encoded[start + 1..start + n_ctx + 1]);
        }
        let prepared = retrograd::dataset::PreparedDataset {
            tokens: tokens.clone(),
            labels: labels.clone(),
            n_ctx,
            examples: 2,
            supervised_tokens: 2 * n_ctx,
        };

        // The sidecar: one entry per position, the reference token, log p = 0.
        // Through `weighted_targets` rather than hand-built weights, so the
        // renormalization the real path applies is the one under test.
        let sidecar = TopKSidecar::new(
            TopKHeader {
                version: retrograd::dataset::topk::VERSION,
                k: 1,
                n_rows: prepared.tokens.len() as u64,
                vocab_size: 0,
                tokenizer_hash: 0,
                source_hash: 0,
            },
            labels.clone(),
            vec![0.0_f32; labels.len()],
        )
        .expect("sidecar shape");
        let (topk_labels, topk_weights) = sidecar
            .weighted_targets(&prepared)
            .expect("weighted targets");
        assert!(
            topk_weights.iter().all(|&weight| weight == 1.0),
            "a single entry carrying all the mass must renormalize to exactly 1"
        );
        assert_eq!(topk_labels, labels, "the entry is the reference token");

        let mut sft_trainer = Trainer::new(&model, training.clone()).expect("load sft trainer");
        sft_trainer.create_lora(&lora()).expect("create lora");
        let sft = sft_trainer
            .train_sft_with_progress(&prepared, None, |_| {})
            .expect("sft step");
        let sft_adapter = std::env::temp_dir().join("retrograd-distill-null-sft.gguf");
        sft_trainer.save_lora(&sft_adapter).expect("save adapter");
        drop(sft_trainer);

        let mut kd_trainer = Trainer::new(&model, training).expect("load kd trainer");
        kd_trainer.create_lora(&lora()).expect("create lora");
        let kd = kd_trainer
            .train_weighted(
                &WeightedBatch {
                    tokens,
                    labels: topk_labels,
                    weights: topk_weights,
                    n_rows: 2,
                    n_ctx,
                    n_topk: 1,
                },
                0,
            )
            .expect("top-k step");
        let kd_adapter = std::env::temp_dir().join("retrograd-distill-null-kd.gguf");
        kd_trainer.save_lora(&kd_adapter).expect("save adapter");
        drop(kd_trainer);

        // A masked-out batch would compare 0 == 0 and prove nothing.
        assert!(
            sft.train_loss > 0.1,
            "chunked={chunked}: the SFT loss {} is suspiciously low",
            sft.train_loss
        );
        assert!(
            (sft.train_loss - kd.train_loss).abs() < 1.0e-5,
            "chunked={chunked}: SFT loss {} != unit-sidecar loss {}",
            sft.train_loss,
            kd.train_loss
        );
        let sft_bytes = std::fs::read(&sft_adapter).expect("read sft adapter");
        let kd_bytes = std::fs::read(&kd_adapter).expect("read kd adapter");
        assert_eq!(
            sft_bytes, kd_bytes,
            "chunked={chunked}: a unit sidecar moved the adapter differently from SFT"
        );
        let _ = std::fs::remove_file(&sft_adapter);
        let _ = std::fs::remove_file(&kd_adapter);
    }
}
