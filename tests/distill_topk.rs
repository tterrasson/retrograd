//! The offline top-k path end to end against a real GGUF: produce a sidecar,
//! read it back, and train on it.
//!
//! One model plays both parts, as in `distill_runtime.rs` and for the same
//! reason: teacher == student is the configuration whose answers are known in
//! advance. Here it pins the one thing a sidecar cannot check about itself - the
//! *alignment* between a block and the position it describes - because the
//! teacher's log-probability of the reference token has to equal the value the
//! scalar scorer returns for that same position, and a shift of one token
//! anywhere in the producer breaks that equality while leaving every file
//! well-formed.
//!
//! Skipped when no local model exists (override with `RETRO_CPU_FIXTURE`).

mod common;

use common::serialize_models;
use retrograd::config::OfflineDistillConfig;
use retrograd::dataset::topk::{
    TopKHeader, TopKSidecar, VERSION, corpus_fingerprint, tokenizer_fingerprint,
};
use retrograd::dataset::{IGNORE_LABEL, PreparedDataset};
use retrograd::training::distill::offline::{score_corpus, witness_ids};
use retrograd::{Device, LoraConfig, TargetSet, TrainConfig, Trainer};

const K: usize = 4;

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 32,
        n_seq_max: 1,
        epochs: 1,
        learning_rate: 1.0e-4,
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

/// Two rows of the model's own tokens with the labels shifted by one - what
/// `retrograd-dataset` produces for a text corpus, built here so the test owns
/// both halves of the alignment it is checking.
fn corpus(trainer: &Trainer, n_ctx: usize) -> PreparedDataset {
    let text = "Offline distillation reads a distribution the teacher wrote down once. ".repeat(96);
    let encoded = trainer.tokenize_text(&text).expect("tokenize");
    assert!(
        encoded.len() > 2 * n_ctx + 1,
        "the fixture tokenized {} tokens, fewer than the {} two rows need",
        encoded.len(),
        2 * n_ctx + 1
    );
    let mut tokens = Vec::with_capacity(2 * n_ctx);
    let mut labels = Vec::with_capacity(2 * n_ctx);
    for row in 0..2 {
        let start = row * n_ctx;
        tokens.extend_from_slice(&encoded[start..start + n_ctx]);
        labels.extend_from_slice(&encoded[start + 1..start + n_ctx + 1]);
    }
    PreparedDataset {
        tokens,
        labels,
        n_ctx,
        examples: 2,
        supervised_tokens: 2 * n_ctx,
    }
}

fn sidecar_for(trainer: &mut Trainer, prepared: &PreparedDataset) -> TopKSidecar {
    let (ids, logprobs) = score_corpus(trainer, prepared, K).expect("score the corpus");
    let vocab_size = trainer.vocab_size().expect("vocab") as u32;
    let witnesses = witness_ids(trainer).expect("witnesses");
    TopKSidecar::new(
        TopKHeader {
            version: VERSION,
            k: K as u32,
            n_rows: prepared.tokens.len() as u64,
            vocab_size,
            tokenizer_hash: tokenizer_fingerprint(vocab_size, &witnesses),
            source_hash: corpus_fingerprint(prepared),
        },
        ids,
        logprobs,
    )
    .expect("sidecar shape")
}

/// The sidecar's alignment gate. Column `j = 0` is the teacher's argmax, so it is not the
/// reference token in general; what must coincide is the *value* the sidecar
/// gives the reference token, wherever in the block it landed, and the value
/// `score_token_suffix` returns for that position. Those are the same
/// log-probability computed by two different runtime entry points, and only an
/// alignment error can separate them.
#[test]
fn a_sidecar_block_describes_the_position_it_is_indexed_by() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, config()).expect("load model");
    let n_ctx = trainer.context_size().expect("context");
    let prepared = corpus(&trainer, n_ctx);
    let sidecar = sidecar_for(&mut trainer, &prepared);

    // The scalar scorer over the same rows: `score_token_suffix(tokens, 1)`
    // returns the log-probability of each token from index 1 on, i.e. of the
    // label of position 0, 1, ... - the same indexing the producer used.
    let mut checked = 0_usize;
    for row in 0..prepared.examples {
        let start = row * n_ctx;
        let row_tokens = &prepared.tokens[start..start + n_ctx];
        let scalar = trainer
            .score_token_suffix(row_tokens, 1)
            .expect("scalar scoring");
        for (position, &reference) in prepared.labels[start..start + n_ctx - 1].iter().enumerate() {
            if reference < 0 {
                continue;
            }
            let (ids, logprobs) = sidecar
                .row(start + position)
                .expect("every position has a block");
            let Some(entry) = ids.iter().position(|&id| id == reference) else {
                // The reference token is outside the teacher's top k. Common,
                // and not an alignment failure - it is what the truncation is.
                continue;
            };
            let expected = scalar[position];
            assert!(
                (logprobs[entry] - expected).abs() < 1.0e-4,
                "row {row} position {position}: the sidecar gives token {reference} a \
                 log-probability of {} where the scalar scorer says {expected}",
                logprobs[entry],
            );
            checked += 1;
        }
    }
    assert!(
        checked >= n_ctx / 2,
        "only {checked} positions had their reference token inside the top {K}: too few to \
         call the alignment checked"
    );

    // The blocks themselves: sorted descending, column 0 the argmax, and every
    // entry a real vocabulary row. A producer that emitted them in another
    // order would still train, on a distribution the renormalization silently
    // reshuffles.
    for position in 0..prepared.tokens.len() {
        let (ids, logprobs) = sidecar.row(position).expect("block");
        if ids[0] == IGNORE_LABEL {
            continue;
        }
        for entry in 1..K {
            assert!(
                logprobs[entry] <= logprobs[entry - 1] + 1.0e-6,
                "position {position}: entry {entry} ({}) outranks entry {} ({})",
                logprobs[entry],
                entry - 1,
                logprobs[entry - 1]
            );
        }
    }
}

/// A top-k run of `k > 1` trains end to end, on the dense path and on the
/// fused one, and moves the adapter.
///
/// "Moves the adapter" is the whole assertion, and it is not a weak one: the
/// preceding null test pins *what* the update is when the distribution is a
/// point mass, and the operator parity tests pin the gradient itself. What is
/// left to check here is that the plumbing between a file on disk and the
/// optimizer exists at `k > 1` at all - the step the two ends cannot prove.
#[test]
fn a_topk_run_trains_end_to_end_on_both_optimizer_paths() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let directory = std::env::temp_dir().join("retrograd-distill-topk");
    std::fs::create_dir_all(&directory).expect("scratch directory");

    for chunked in [false, true] {
        let training = TrainConfig {
            chunked_cross_entropy: chunked,
            ..config()
        };
        let mut teacher = Trainer::new(&model, training.clone()).expect("load teacher");
        let n_ctx = teacher.context_size().expect("context");
        let prepared = corpus(&teacher, n_ctx);
        let sidecar = sidecar_for(&mut teacher, &prepared);
        drop(teacher);

        let path = directory.join(format!("corpus-chunked-{chunked}.topk"));
        sidecar.write(&path).expect("write sidecar");
        let read = TopKSidecar::read(&path).expect("read sidecar");
        assert_eq!(read.k(), K);
        assert_eq!(read.rows(), prepared.tokens.len());
        read.check_against(
            &prepared,
            read.header().vocab_size,
            read.header().tokenizer_hash,
            corpus_fingerprint(&prepared),
        )
        .expect("the sidecar belongs to this corpus");

        let (labels, weights) = read.weighted_targets(&prepared).expect("targets");
        let batch = retrograd::WeightedBatch {
            tokens: prepared.tokens.clone(),
            labels,
            weights,
            n_rows: prepared.examples,
            n_ctx,
            n_topk: K,
        };
        batch.validate().expect("valid batch");

        let mut student = Trainer::new(&model, training).expect("load student");
        student.create_lora(&lora()).expect("create lora");
        let before = directory.join(format!("before-chunked-{chunked}.gguf"));
        let after = directory.join(format!("after-chunked-{chunked}.gguf"));
        student.save_lora(&before).expect("save adapter");
        let metrics = student.train_weighted(&batch, 0).expect("top-k step");
        student.save_lora(&after).expect("save adapter");

        assert!(
            metrics.train_loss.is_finite() && metrics.train_loss > 0.0,
            "chunked={chunked}: a top-k step produced a loss of {}",
            metrics.train_loss
        );
        assert_ne!(
            std::fs::read(&before).expect("read adapter"),
            std::fs::read(&after).expect("read adapter"),
            "chunked={chunked}: a top-k update left the adapter untouched"
        );
    }
    let _ = std::fs::remove_dir_all(&directory);
}

/// The refusals of `distill::offline::prepare`, without loading a model: a
/// sidecar is a list of numbers, and the header is the only thing that ties it
/// to a corpus.
#[test]
fn a_sidecar_from_another_corpus_is_refused_before_a_single_step() {
    let one = PreparedDataset {
        tokens: vec![1, 2, 3, 4],
        labels: vec![2, 3, 4, IGNORE_LABEL],
        n_ctx: 4,
        examples: 1,
        supervised_tokens: 3,
    };
    let other = PreparedDataset {
        tokens: vec![1, 2, 3, 5],
        ..one.clone()
    };
    let sidecar = TopKSidecar::new(
        TopKHeader {
            version: VERSION,
            k: 1,
            n_rows: 4,
            vocab_size: 32,
            tokenizer_hash: 11,
            source_hash: corpus_fingerprint(&one),
        },
        vec![2, 3, 4, IGNORE_LABEL],
        vec![0.0, 0.0, 0.0, f32::NEG_INFINITY],
    )
    .expect("sidecar shape");
    sidecar
        .check_against(&one, 32, 11, corpus_fingerprint(&one))
        .expect("its own corpus");
    let error = sidecar
        .check_against(&other, 32, 11, corpus_fingerprint(&other))
        .unwrap_err()
        .to_string();
    assert!(error.contains("another corpus"), "{error}");
}

/// The offline configuration is what a `[distill] mode = "topk_offline"`
/// document builds into, and the two paths it names are what `prepare` opens.
#[test]
fn the_offline_configuration_names_the_two_files_it_reads() {
    let config = OfflineDistillConfig {
        data: "corpus.jsonl".into(),
        sidecar: "corpus.topk".into(),
        epochs: 3,
    };
    assert_eq!(config.epochs, 3);
    assert_eq!(
        config.data.extension().and_then(|e| e.to_str()),
        Some("jsonl")
    );
    assert_eq!(
        config.sidecar.extension().and_then(|e| e.to_str()),
        Some("topk")
    );
}
