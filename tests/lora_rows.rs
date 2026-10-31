//! Per-row LoRA adapters on the CPU fixture. One decode mixing rows of two
//! adapters of different ranks, the base model and the context's own adapter
//! must give each row what that row's adapter gives alone, in a batch of the
//! same shape (the kernels are not batch-invariant: on this fixture the second
//! of two identical base-model rows already differs from the first in the
//! second decimal, adapter or not, so a row is compared at its own position
//! of an identical batch run under its adapter alone). A trainer's
//! per-row snapshot of its adapter must stay frozen while the adapter trains
//! on. Skipped when no local model exists (override with `RETRO_CPU_FIXTURE`).
//!
//! The tolerance is a backend fact. The per-row path gathers zero-padded F32
//! stacks through `mul_mat_id`; the adapter alone runs two plain products.
//! On CPU both reduce in the same order while the rank stays below the SIMD
//! width the CPU GEMM tiles on (ranks 1 and 2 here: the results agree to the
//! bit), so the parity below is checked at 1e-3 on a log-probability. From
//! rank 4 the plain product switches to the tiled GEMM, and the quantized
//! base re-quantizes every activation it reads, which turns that rounding
//! difference into up to ~0.15 of log-probability on this fixture at ranks 4
//! and 8 - the same argmax, a different last digit, measured rather than a
//! defect of either path.

mod common;

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use common::serialize_models;
use retrograd::trainer::{JudgeWeights, SnapshotSource};
use retrograd::{
    Device, Engine, EngineConfig, EngineKvDtype, LoraConfig, RowAdapter, SamplingParams,
    StepOutput, StepRow, TargetSet, TrainConfig, Trainer,
};

/// Log-probability tolerance between the per-row path and the adapter alone,
/// F32 on CPU.
const CPU_TOLERANCE: f32 = 1.0e-3;
const TOP: u32 = 8;

fn train_config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1.0e-2,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn engine_config() -> EngineConfig {
    EngineConfig {
        n_ctx: 256,
        n_seq_max: 4,
        n_batch: 64,
        n_ubatch: 64,
        kv_dtype: EngineKvDtype::F32,
        flash_attn: false,
        kv_unified: true,
        device: Device::Cpu,
        threads: 0,
        verbose: false,
    }
}

fn sampling(seed: u32) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 6,
        seed,
    }
}

/// Trains a q/v adapter of `rank` for a few steps (a fresh adapter has a zero
/// B factor, so its delta would be zero) and saves it.
fn trained_adapter(model: &Path, rank: u32, seed: u32, out: &Path) -> PathBuf {
    let mut trainer = Trainer::new(model, train_config()).expect("load trainer");
    let mut lora = LoraConfig::qv(rank, 2.0 * rank as f32);
    lora.seed = seed;
    lora.targets = TargetSet::QV;
    trainer.create_lora(&lora).expect("create lora");
    let mut row = trainer
        .tokenize_text(
            "The quick brown fox jumps over the lazy dog. "
                .repeat(40)
                .as_str(),
        )
        .expect("tokenize");
    // The training context is wider than `n_ctx`; one row more than it.
    row.truncate(257);
    for _ in 0..3 {
        trainer.train_tokens(&row).expect("train");
    }
    let path = out.join(format!("adapter-r{rank}-s{seed}.gguf"));
    trainer.save_lora(&path).expect("save lora");
    path
}

/// One prefill row over `prompt` into `slot`, sampling its last token with
/// the top-`TOP` log-probabilities reported.
fn scoring_row(request: i32, slot: i32, prompt: &[i32], adapter: RowAdapter) -> StepRow<'_> {
    StepRow {
        request,
        slot,
        tokens: Cow::Borrowed(prompt),
        pos0: 0,
        sample: true,
        sampling: sampling(9).into(),
        n_top_logprobs: TOP,
        want_logprob: false,
        adapter,
    }
}

fn top(out: &StepOutput, row: usize) -> Vec<(i32, f32)> {
    out.top(row).collect()
}

/// The next-token top-k of `prompt` for each of `n` identical rows of one
/// step (slots `0..n`), on a fresh engine with `adapter` (or none) as its
/// context adapter.
fn alone(model: &Path, adapter: Option<&Path>, prompt: &[i32], n: i32) -> Vec<Vec<(i32, f32)>> {
    let mut engine = Engine::new(model, &engine_config()).expect("load engine");
    if let Some(adapter) = adapter {
        engine.load_lora(adapter).expect("load lora");
    }
    let rows: Vec<StepRow> = (0..n)
        .map(|slot| scoring_row(slot + 1, slot, prompt, RowAdapter::Context))
        .collect();
    let mut out = StepOutput::with_capacity(rows.len(), TOP as usize);
    engine.step(&rows, &mut out).expect("step");
    (0..rows.len()).map(|row| top(&out, row)).collect()
}

fn assert_close(label: &str, got: &[(i32, f32)], want: &[(i32, f32)]) {
    assert_eq!(got.len(), want.len(), "{label}: top-k width");
    assert_eq!(got[0].0, want[0].0, "{label}: the argmax token");
    for ((got_id, got_lp), (want_id, want_lp)) in got.iter().zip(want) {
        assert_eq!(got_id, want_id, "{label}: top-k token identity");
        assert!(
            (got_lp - want_lp).abs() <= CPU_TOLERANCE,
            "{label}: token {got_id} logprob {got_lp} vs {want_lp}"
        );
    }
}

#[test]
fn per_row_adapters_match_each_adapter_applied_alone() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let tmp = common::scratch("lora-rows-parity");
    // Two ranks, so the smaller adapter is zero-padded in the stacks.
    let a = trained_adapter(&model, 1, 7, tmp.path());
    let b = trained_adapter(&model, 2, 11, tmp.path());

    let mut engine = Engine::new(&model, &engine_config()).expect("load engine");
    let prompt = engine.tokenize_text("Pack my box with five dozen").unwrap();
    // Each row of the mixed step below is compared with the same row of a
    // four-row step under its adapter alone.
    let base = alone(&model, None, &prompt, 4);
    let want_a = alone(&model, Some(&a), &prompt, 4);
    let want_b = alone(&model, Some(&b), &prompt, 4);
    assert_ne!(
        want_a, base,
        "the trained adapter must move the distribution, or the parity is vacuous"
    );

    let id_a = engine.adapter_load(&a, 1.0).expect("load a per row");
    let id_b = engine.adapter_load(&b, 1.0).expect("load b per row");
    assert_ne!(id_a, id_b);
    let info = engine.adapter_rows_info().unwrap();
    assert_eq!((info.n_adapters, info.max_rank), (2, 2));
    assert!(
        info.stacked_bytes > info.unpadded_bytes,
        "rank 1 is padded to 2"
    );

    // The context's own adapter is `b`: route 0 rows follow it, the others
    // ignore it.
    engine.load_lora(&b).expect("context adapter");
    let mut out = StepOutput::with_capacity(4, TOP as usize);
    engine
        .step(
            &[
                scoring_row(1, 0, &prompt, RowAdapter::Loaded(id_a)),
                scoring_row(2, 1, &prompt, RowAdapter::Loaded(id_b)),
                scoring_row(3, 2, &prompt, RowAdapter::Base),
                scoring_row(4, 3, &prompt, RowAdapter::Context),
            ],
            &mut out,
        )
        .expect("mixed step");
    assert_close("per-row a", &top(&out, 0), &want_a[0]);
    assert_close("per-row b", &top(&out, 1), &want_b[1]);
    assert_close("base row", &top(&out, 2), &base[2]);
    assert_close("context row", &top(&out, 3), &want_b[3]);

    // Unloading `a` restacks `b` alone; its rows still match.
    engine.adapter_unload(id_a).expect("unload a");
    assert!(engine.adapter_unload(id_a).is_err(), "ids are not reused");
    for slot in 0..4 {
        engine.seq_rm(slot, -1, -1).unwrap();
    }
    engine
        .step(
            &[scoring_row(5, 0, &prompt, RowAdapter::Loaded(id_b))],
            &mut out,
        )
        .expect("step after unload");
    let single_b = alone(&model, Some(&b), &prompt, 1);
    assert_close("per-row b after unload", &top(&out, 0), &single_b[0]);
    let unknown = engine.step(
        &[scoring_row(6, 1, &prompt, RowAdapter::Loaded(id_a))],
        &mut out,
    );
    assert!(unknown.is_err(), "an unloaded id is refused");

    // An empty routed set must still support bypassing the context adapter.
    engine.adapter_unload(id_b).unwrap();
    for slot in 0..4 {
        engine.seq_rm(slot, -1, -1).unwrap();
    }
    let rows: Vec<_> = (0..4)
        .map(|slot| scoring_row(slot + 7, slot, &prompt, RowAdapter::Base))
        .collect();
    engine
        .step(&rows, &mut out)
        .expect("base rows without routed adapters");
    for (row, expected) in base.iter().enumerate() {
        assert_close("empty routed set", &top(&out, row), expected);
    }
}

#[test]
fn a_trainer_snapshot_stays_frozen_while_the_adapter_trains() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, train_config()).expect("load trainer");
    trainer
        .create_lora(&LoraConfig::qv(2, 4.0))
        .expect("create lora");
    let mut row = trainer
        .tokenize_text("Sphinx of black quartz, judge my vow. ".repeat(40).as_str())
        .expect("tokenize");
    // The training context is wider than `n_ctx`; one row more than it.
    row.truncate(257);
    trainer.train_tokens(&row).expect("train");
    let prompt = trainer.tokenize_text("Judge my").unwrap();

    let memory_before_snapshot = trainer.memory_report().unwrap();
    let snapshot = trainer
        .generation_adapter_snapshot(None, 1.0)
        .expect("take a snapshot");
    let memory_after_snapshot = trainer.memory_report().unwrap();
    let stacks = trainer
        .generation_adapter_rows_info()
        .unwrap()
        .stacked_bytes;
    assert!(memory_after_snapshot.host_bytes >= memory_before_snapshot.host_bytes + stacks);
    assert_eq!(
        memory_after_snapshot.optimizer_state_bytes,
        memory_before_snapshot.optimizer_state_bytes
    );
    let judge = |trainer: &mut Trainer| {
        trainer
            .with_weights(
                JudgeWeights::Snapshot,
                Some(SnapshotSource::Adapter(snapshot)),
                |model| model.generate(&prompt, &sampling(3)),
            )
            .expect("generate under the snapshot")
    };
    let before = judge(&mut trainer);
    // Both capability probing and an inner selection must preserve the outer
    // snapshot, including when the inner operation fails.
    trainer
        .with_generation_adapter(RowAdapter::Loaded(snapshot), |model| {
            assert!(model.supports_generation_adapters());
            let inner: retrograd::Result<()> = model
                .with_generation_adapter(RowAdapter::Base, |_| {
                    Err(retrograd::Error::invalid("inner operation failed"))
                });
            assert!(inner.is_err());
            let restored = model.generate(&prompt, &sampling(3))?;
            assert_eq!(restored.logprobs, before.logprobs);
            Ok(())
        })
        .expect("nested adapter selections restore the snapshot");
    trainer.train_tokens(&row).expect("train on");
    let after = judge(&mut trainer);
    assert_eq!(before.tokens, after.tokens, "the snapshot did not move");
    assert_eq!(before.logprobs, after.logprobs, "the snapshot did not move");
    let policy = trainer.generate(&prompt, &sampling(3)).expect("policy");
    assert_ne!(
        policy.logprobs, after.logprobs,
        "the policy moved, so the snapshot is not the live adapter"
    );

    // A refresh replaces the snapshot in place: one adapter resident, new id.
    let refreshed = trainer
        .generation_adapter_snapshot(Some(snapshot), 1.0)
        .expect("refresh");
    assert_ne!(refreshed, snapshot);
    assert_eq!(
        trainer.generation_adapter_rows_info().unwrap().n_adapters,
        1
    );
    assert!(trainer.generation_adapter_unload(snapshot).is_err());

    // The base model selected per row is the adapter-free model.
    let base = trainer
        .with_generation_adapter(RowAdapter::Base, |model| {
            model.generate(&prompt, &sampling(3))
        })
        .expect("base rows");
    let mut fresh = Trainer::new(&model, train_config()).expect("load a base trainer");
    let reference = fresh.generate(&prompt, &sampling(3)).expect("base");
    assert_eq!(
        base.tokens, reference.tokens,
        "base rows are the base model"
    );
}

/// Decodes `n` single-token steps in slot 0 after `prompt`'s prefill, from
/// position `pos`, and returns the next position and the last token.
fn decode(
    engine: &mut Engine,
    mut token: i32,
    mut pos: i32,
    n: usize,
    adapter: RowAdapter,
) -> (i32, i32) {
    let mut out = StepOutput::with_capacity(1, 0);
    for _ in 0..n {
        let tokens = [token];
        let row = StepRow {
            request: 1,
            slot: 0,
            tokens: Cow::Borrowed(&tokens),
            pos0: pos,
            sample: true,
            sampling: sampling(3).into(),
            n_top_logprobs: 0,
            want_logprob: false,
            adapter,
        };
        engine.step(&[row], &mut out).expect("decode step");
        token = out.row(0).token;
        pos += 1;
    }
    (pos, token)
}

/// Graph reuse is the fork's (`can_reuse`/`n_reused`), and what is checked
/// here is that the runtime's decode path actually gets it: single-token steps
/// of one shape reuse the previous graph, installing a per-row adapter stack
/// changes the topology and rebuilds it once, and the steps after reuse the
/// new one.
#[test]
fn a_steady_decode_reuses_its_graph_and_an_adapter_load_rebuilds_it() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let tmp = common::scratch("lora-rows-graph-reuse");
    let adapter = trained_adapter(&model, 1, 5, tmp.path());

    let mut engine = Engine::new(&model, &engine_config()).expect("load engine");
    let prompt = engine.tokenize_text("Pack my box with five dozen").unwrap();
    let mut out = StepOutput::with_capacity(1, TOP as usize);
    engine
        .step(&[scoring_row(1, 0, &prompt, RowAdapter::Context)], &mut out)
        .expect("prefill");
    let (pos, token) = (prompt.len() as i32, out.row(0).token);

    // The first decode after the prefill has another shape and builds its
    // graph; every one after it reuses that graph.
    let (pos, token) = decode(&mut engine, token, pos, 1, RowAdapter::Context);
    let before = engine.graphs_reused().unwrap();
    let (pos, token) = decode(&mut engine, token, pos, 6, RowAdapter::Context);
    assert_eq!(
        engine.graphs_reused().unwrap() - before,
        6,
        "six decodes of one shape must each reuse the previous graph"
    );

    // A per-row stack is a new topology: the next step builds, then reuse
    // resumes on the new graph.
    let id = engine.adapter_load(&adapter, 1.0).expect("load per row");
    let before = engine.graphs_reused().unwrap();
    let (pos, token) = decode(&mut engine, token, pos, 1, RowAdapter::Loaded(id));
    assert_eq!(
        engine.graphs_reused().unwrap(),
        before,
        "the step after an adapter load must not run on the stale graph"
    );
    let before = engine.graphs_reused().unwrap();
    let _ = decode(&mut engine, token, pos, 4, RowAdapter::Loaded(id));
    assert_eq!(engine.graphs_reused().unwrap() - before, 4);
}
