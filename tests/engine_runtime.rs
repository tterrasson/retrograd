//! Engine vs trainer parity on the CPU fixture: the inference engine runs the
//! same generation core as the trainer's `retro_trainer_generate_*`, so at
//! equal geometry, seeds and weights its output must be bit-identical. The
//! engine's memory report must also show no training state (zero optimizer
//! state, zero gradient), which is the property that distinguishes the handle
//! from a trainer. Skipped when no local model exists (override with
//! `RETRO_CPU_FIXTURE`).

mod common;

use std::borrow::Cow;

use common::serialize_models;
use retrograd::trainer::SampledRow;
use retrograd::{
    Device, Engine, EngineConfig, EngineError, EngineKvDtype, LoraConfig, SamplingParams,
    StepOutput, StepRow, TargetSet, TrainConfig, Trainer,
};

/// The trainer config the parity tests run with: four sequence slots, so the
/// dedicated generation context exists with a known geometry.
fn train_config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        n_ubatch: 64,
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

/// The engine geometry that reproduces the trainer's dedicated generation
/// context exactly: `n_ctx * effective_generation_concurrency` cells, one slot
/// per concurrent sequence, the generation batch, and the fast context's F16
/// KV with flash attention (the `TrainConfig` defaults here).
fn engine_config(train: &TrainConfig) -> EngineConfig {
    let concurrency = train.effective_generation_concurrency();
    let generation_batch = if train.generation_batch != 0 {
        train.generation_batch
    } else {
        train.n_ctx.min(512)
    };
    let generation_batch = generation_batch.max(concurrency);
    EngineConfig {
        n_ctx: train.n_ctx * concurrency,
        n_seq_max: concurrency,
        n_batch: generation_batch,
        n_ubatch: generation_batch,
        kv_dtype: if train.fast_generation_context {
            match train.generation_kv_dtype {
                retrograd::GenerationKvDtype::F16 => EngineKvDtype::F16,
                retrograd::GenerationKvDtype::Q8_0 => EngineKvDtype::Q8_0,
                retrograd::GenerationKvDtype::Q4_0 => EngineKvDtype::Q4_0,
            }
        } else {
            EngineKvDtype::F32
        },
        flash_attn: train.fast_generation_context,
        kv_unified: true,
        device: Device::Cpu,
        threads: 0,
        verbose: false,
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

fn params(seed: u32) -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 6,
        seed,
    }
}

/// The full parity check: shared-prompt batches, single generation and
/// heterogeneous continuous waves must all agree bit for bit, tokens and
/// logprobs alike.
fn assert_parity(trainer: &mut Trainer, engine: &mut Engine, prompt: &[i32], rows: &[&[i32]]) {
    // Shared-prompt batch, logprobs included. `Generation` has no PartialEq,
    // so compare the fields: tokens exactly, logprobs bit for bit.
    let sampling = [params(31), params(32), params(33)];
    let trainer_batch = trainer
        .generate_batch(prompt, &sampling)
        .expect("trainer generate_batch");
    let engine_batch = engine
        .generate_batch(prompt, &sampling)
        .expect("engine generate_batch");
    assert_eq!(trainer_batch.len(), engine_batch.len());
    for (left, right) in trainer_batch.iter().zip(&engine_batch) {
        assert_eq!(left.tokens, right.tokens, "shared-prompt batch tokens");
        assert_eq!(
            left.logprobs, right.logprobs,
            "shared-prompt batch logprobs"
        );
    }

    // Single generation.
    let trainer_one = trainer
        .generate(prompt, &params(41))
        .expect("trainer generate");
    let engine_one = engine
        .generate(prompt, &params(41))
        .expect("engine generate");
    assert_eq!(
        trainer_one.tokens, engine_one.tokens,
        "single generation tokens"
    );
    assert_eq!(
        trainer_one.logprobs, engine_one.logprobs,
        "single generation logprobs"
    );

    // Heterogeneous continuous wave, tokens and forced masks.
    let wave: Vec<(&[i32], SamplingParams)> = rows
        .iter()
        .enumerate()
        .map(|(i, row)| (*row, params(50 + i as u32)))
        .collect();
    let trainer_rows = trainer
        .generate_sampled_continuous(&wave)
        .expect("trainer continuous");
    let engine_rows = engine
        .generate_sampled_continuous(&wave)
        .expect("engine continuous");
    assert_eq!(trainer_rows, engine_rows, "continuous wave parity");

    // Logprob-free paths agree too.
    let trainer_tokens = trainer
        .generate_tokens_batch(prompt, &sampling)
        .expect("trainer tokens batch");
    let engine_tokens = engine
        .generate_tokens_batch(prompt, &sampling)
        .expect("engine tokens batch");
    assert_eq!(trainer_tokens, engine_tokens);
    let trainer_wave = trainer
        .generate_tokens_continuous(&wave)
        .expect("trainer tokens continuous");
    let engine_wave = engine
        .generate_tokens_continuous(&wave)
        .expect("engine tokens continuous");
    assert_eq!(trainer_wave, engine_wave);
}

#[test]
fn engine_generate_matches_trainer_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let mut trainer = Trainer::new(&model, train.clone()).expect("load trainer");
    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");

    // Tokenization agrees, so the prompts below are the same token stream.
    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    let second = trainer.tokenize_text("Pack my box with").unwrap();
    assert_eq!(first, engine.tokenize_text("The quick brown fox").unwrap());
    assert_eq!(second, engine.tokenize_text("Pack my box with").unwrap());
    assert_eq!(trainer.eos_token().unwrap(), engine.eos_token().unwrap());
    assert_eq!(
        trainer.vocab_size().unwrap(),
        engine.vocab_size().unwrap() as usize
    );
    assert_eq!(
        engine.context_size().unwrap() as u32,
        train.n_ctx * train.effective_generation_concurrency(),
        "the engine context must match the trainer's generation context width"
    );

    // Base weights first.
    assert_parity(&mut trainer, &mut engine, &first, &[&first, &second]);

    // Then with a LoRA adapter attached on both sides: the trainer creates and
    // saves one, the engine loads the same file.
    trainer.create_lora(&lora()).expect("create lora");
    let tmp = common::scratch("engine-parity-lora");
    let adapter = tmp.path().join("adapter.gguf");
    trainer.save_lora(&adapter).expect("save lora");
    engine.load_lora(&adapter).expect("engine load lora");
    assert_parity(&mut trainer, &mut engine, &first, &[&first, &second]);

    // Disabling the adapter on the engine must restore base-model parity
    // against a fresh adapter-free engine.
    let mut engine_base = Engine::new(&model, &engine_config(&train)).expect("load base engine");
    engine.set_lora_enabled(false).expect("engine disable lora");
    let trainer_side = engine_base
        .generate(&second, &params(41))
        .expect("base engine");
    let disabled = engine
        .generate(&second, &params(41))
        .expect("disabled engine");
    assert_eq!(trainer_side.tokens, disabled.tokens);
    assert_eq!(trainer_side.logprobs, disabled.logprobs);

    // The engine is deterministic across repeated calls.
    let again = engine
        .generate(&first, &params(41))
        .expect("engine generate");
    let once = engine
        .generate(&first, &params(41))
        .expect("engine generate again");
    assert_eq!(again.tokens, once.tokens);
    assert_eq!(again.logprobs, once.logprobs);
}

#[test]
fn engine_generation_matches_exact_and_quantized_trainer_contexts() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        return;
    };
    for dtype in [EngineKvDtype::F32, EngineKvDtype::Q8_0, EngineKvDtype::Q4_0] {
        let mut train = train_config();
        train.fast_generation_context = dtype != EngineKvDtype::F32;
        train.generation_kv_dtype = match dtype {
            EngineKvDtype::Q8_0 => retrograd::GenerationKvDtype::Q8_0,
            EngineKvDtype::Q4_0 => retrograd::GenerationKvDtype::Q4_0,
            _ => retrograd::GenerationKvDtype::F16,
        };
        let mut trainer = Trainer::new(&model, train.clone()).unwrap();
        let mut engine = Engine::new(&model, &engine_config(&train)).unwrap();
        let first = trainer.tokenize_text("The quick brown fox").unwrap();
        let second = trainer.tokenize_text("Pack my box with").unwrap();
        assert_parity(&mut trainer, &mut engine, &first, &[&first, &second]);
    }
}

#[test]
fn engine_memory_report_carries_no_training_state() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let mut trainer = Trainer::new(&model, train.clone()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    // Configured to train for real: the optimizer graph exists.
    trainer.prepare_optimizer().expect("prepare optimizer");
    let trainer_report = trainer.memory_report().expect("trainer memory report");

    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");
    let tmp = common::scratch("engine-memory-lora");
    let adapter = tmp.path().join("adapter.gguf");
    trainer.save_lora(&adapter).expect("save lora");
    engine.load_lora(&adapter).expect("engine load lora");
    let engine_report = engine.memory_report().expect("engine memory report");

    // The engine has no optimizer and no backward graph: those fields are a
    // fact, not a gap.
    assert_eq!(engine_report.optimizer_state_bytes, 0);
    assert_eq!(engine_report.trainable_gradient_bytes, 0);
    assert_eq!(engine_report.optimizer_kv_bytes, 0);
    assert_eq!(engine_report.optimizer_compute_bytes, 0);
    assert_eq!(engine_report.checkpoint_count, 0);
    assert!(engine_report.has_generation_context);
    assert!(engine_report.model_weight_bytes > 0);
    assert!(engine_report.generation_kv_bytes > 0);
    // The same adapter is attached on both sides.
    assert_eq!(
        engine_report.trainable_parameter_bytes,
        trainer_report.trainable_parameter_bytes
    );
    assert!(engine_report.trainable_parameter_bytes > 0);

    // Against a trainer actually configured to train, the engine's footprint is
    // strictly smaller: no optimizer context KV, no compute buffers, no
    // optimizer state, no gradient.
    assert!(trainer_report.optimizer_state_bytes > 0);
    assert!(trainer_report.trainable_gradient_bytes > 0);
    let engine_total = engine_report.device_bytes + engine_report.host_bytes;
    let trainer_total = trainer_report.device_bytes + trainer_report.host_bytes;
    assert!(
        engine_total < trainer_total,
        "engine {engine_total} bytes should be below the training-configured trainer's \
         {trainer_total} bytes"
    );
    eprintln!(
        "memory: engine total {engine_total} B vs trainer {trainer_total} B \
         (optimizer_state {} B, gradient {} B, optimizer kv {} B)",
        trainer_report.optimizer_state_bytes,
        trainer_report.trainable_gradient_bytes,
        trainer_report.optimizer_kv_bytes,
    );

    // Generation counters are reported on the engine too; the continuous path
    // is the one that counts calls.
    let stats = engine.generation_stats().expect("engine generation stats");
    assert_eq!(stats.calls, 0);
    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    engine
        .generate_tokens_continuous(&[(&prompt, params(7))])
        .expect("engine continuous");
    let stats = engine.generation_stats().expect("engine generation stats");
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.sequences, 1);
}

#[test]
fn engine_refuses_a_missing_model_with_a_clean_error() {
    let _guard = serialize_models();
    let config = engine_config(&train_config());
    let Err(error) = Engine::new("/nonexistent/model.gguf", &config) else {
        panic!("a missing model path must fail");
    };
    // Converted through the From into the core facade, it stays a runtime error.
    let core: retrograd::Error = error.into();
    assert!(matches!(core, retrograd::Error::Runtime(_)));
    assert!(core.to_string().contains("failed to load model GGUF"));
}

#[test]
fn engine_rejects_sequence_slots_larger_than_the_effective_native_batch() {
    let _guard = serialize_models();
    let Some(model) = common::tiny_model_path_if_available() else {
        return;
    };
    // llama.cpp clamps n_batch to the requested n_ctx before padding the
    // context to 256 cells. These dimensions would produce two sequence slots
    // but a physical batch of one, aborting during the output reservation.
    let mut config = EngineConfig {
        n_ctx: 1,
        n_seq_max: 2,
        n_batch: 2,
        n_ubatch: 2,
        device: Device::Cpu,
        ..EngineConfig::default()
    };
    assert!(matches!(
        Engine::new(&model, &config),
        Err(retrograd::EngineError::InvalidArgument(_))
    ));
    config.n_ctx = 2;
    let mut engine = Engine::new(&model, &config).unwrap();
    engine
        .generate_batch(&[1], &[params(1), params(2)])
        .unwrap();
    assert!(matches!(
        engine.generate_batch(&[1], &[params(1), params(2), params(3)]),
        Err(retrograd::EngineError::InvalidArgument(_))
    ));
    assert!(matches!(
        engine.generate_sampled_continuous(&[
            (&[1], params(1)),
            (&[1], params(2)),
            (&[1], params(3))
        ]),
        Err(retrograd::EngineError::InvalidArgument(_))
    ));
}

#[test]
fn engine_large_token_budgets_are_bounded_by_the_context() {
    let _guard = serialize_models();
    let Some(model) = common::tiny_model_path_if_available() else {
        return;
    };
    let config = EngineConfig {
        n_ctx: 256,
        n_seq_max: 2,
        n_batch: 64,
        n_ubatch: 64,
        device: Device::Cpu,
        ..EngineConfig::default()
    };
    let mut engine = Engine::new(&model, &config).unwrap();
    let prompt = vec![1; 252];
    let mut unlimited = params(2);
    unlimited.max_new_tokens = u32::MAX;
    let mut bounded = unlimited.clone();
    bounded.max_new_tokens = 4;
    let expected = engine.generate(&prompt, &bounded).unwrap();
    let actual = engine.generate(&prompt, &unlimited).unwrap();
    assert_eq!(actual.tokens, expected.tokens);
    assert_eq!(actual.logprobs, expected.logprobs);
    for length in [252, 250] {
        let prompt = &prompt[..length];
        let mut bounded = unlimited.clone();
        // The test prompt lengths are below the 256-cell context.
        bounded.max_new_tokens = u32::try_from(256 - length).unwrap();
        let expected = engine
            .generate_sampled_continuous(&[(prompt, bounded)])
            .unwrap();
        let actual = engine
            .generate_sampled_continuous(&[(prompt, unlimited.clone())])
            .unwrap();
        assert_eq!(actual, expected);
    }
    assert!(
        engine.generation_stats().unwrap().evictions > 0,
        "dormant near-full prefixes must yield their KV cells to the next call"
    );
}

// ---------------------------------------------------------------------------
// The step-wise API: a trivial caller-side loop over `step` must reproduce
// `generate_sampled_continuous` bit for bit in canonical order, and the raw
// sequence operations must behave as documented.

/// The step-wise surface, shared by `Engine` and `Trainer`, so the loop below
/// runs against both handles unchanged.
trait StepApi {
    type Error: std::fmt::Debug;
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), Self::Error>;
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), Self::Error>;
    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), Self::Error>;
    fn request_free(&mut self, request: i32) -> Result<(), Self::Error>;
    fn is_eog_token(&self, token: i32) -> Result<bool, Self::Error>;
    fn context_size(&self) -> Result<usize, Self::Error>;
}

impl StepApi for Engine {
    type Error = EngineError;
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), Self::Error> {
        Engine::step(self, rows, out)
    }
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), Self::Error> {
        Engine::seq_rm(self, slot, p0, p1)
    }
    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), Self::Error> {
        Engine::seq_cp(self, src, dst, p0, p1)
    }
    fn request_free(&mut self, request: i32) -> Result<(), Self::Error> {
        Engine::request_free(self, request)
    }
    fn is_eog_token(&self, token: i32) -> Result<bool, Self::Error> {
        Engine::is_eog_token(self, token)
    }
    fn context_size(&self) -> Result<usize, Self::Error> {
        let slots = self.memory()?.n_slots as usize;
        Engine::context_size(self).map(|n| n as usize / slots)
    }
}

impl StepApi for Trainer {
    type Error = retrograd::Error;
    fn step(&mut self, rows: &[StepRow<'_>], out: &mut StepOutput) -> Result<(), Self::Error> {
        Trainer::step(self, rows, out)
    }
    fn seq_rm(&mut self, slot: i32, p0: i32, p1: i32) -> Result<(), Self::Error> {
        Trainer::seq_rm(self, slot, p0, p1)
    }
    fn seq_cp(&mut self, src: i32, dst: i32, p0: i32, p1: i32) -> Result<(), Self::Error> {
        Trainer::seq_cp(self, src, dst, p0, p1)
    }
    fn request_free(&mut self, request: i32) -> Result<(), Self::Error> {
        Trainer::request_free(self, request)
    }
    fn is_eog_token(&self, token: i32) -> Result<bool, Self::Error> {
        Trainer::is_eog_token(self, token)
    }
    fn context_size(&self) -> Result<usize, Self::Error> {
        Trainer::context_size(self)
    }
}

/// The canonical order of the continuous path: prompt length, prompt tokens,
/// then the sampling parameters. A step loop composing its steps in this order
/// reproduces the batched continuous generator exactly.
fn canonical_order(sequences: &[(&[i32], SamplingParams)]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..sequences.len()).collect();
    order.sort_by(|&a, &b| {
        let (pa, sa) = &sequences[a];
        let (pb, sb) = &sequences[b];
        pa.len()
            .cmp(&pb.len())
            .then_with(|| pa.cmp(pb))
            .then_with(|| sa.seed.cmp(&sb.seed))
            .then_with(|| sa.max_new_tokens.cmp(&sb.max_new_tokens))
            .then_with(|| sa.temperature.total_cmp(&sb.temperature))
            .then_with(|| sa.top_p.total_cmp(&sb.top_p))
    });
    order
}

struct LoopRow {
    slot: i32,
    request: i32,
    prompt_len: usize,
    sampling: SamplingParams,
    tokens: Vec<i32>,
    logprobs: Vec<f32>,
    forced: Vec<bool>,
    live: bool,
    limit: usize,
    next_token: i32,
}

impl LoopRow {
    fn new(
        slot: i32,
        seq: usize,
        prompt_len: usize,
        sampling: SamplingParams,
        limit: usize,
    ) -> Self {
        Self {
            slot,
            request: i32::try_from(seq).expect("sequence count fits i32") + 1,
            prompt_len,
            sampling,
            tokens: Vec::new(),
            logprobs: Vec::new(),
            forced: Vec::new(),
            live: true,
            limit,
            next_token: 0,
        }
    }

    fn emit(&mut self, produced: retrograd::StepOutRow) {
        self.tokens.push(produced.token);
        self.logprobs.push(produced.logprob);
        self.forced.push(produced.forced);
        self.next_token = produced.token;
    }
}

/// The trivial loop: prefill every prompt through `step` (chunked by `n_batch`,
/// shared prompts decoded once and copied with `seq_cp`), then one step per
/// wave over all live rows in canonical order, until EOG, budget or context.
/// Returns the sampled rows and their per-token logprobs, in caller order.
fn step_generate<M: StepApi>(
    model: &mut M,
    sequences: &[(&[i32], SamplingParams)],
    n_batch: usize,
) -> Result<(Vec<SampledRow>, Vec<Vec<f32>>), M::Error> {
    let n_ctx_seq = model.context_size()?;
    let order = canonical_order(sequences);
    let mut out = StepOutput::with_capacity(sequences.len(), 0);
    let mut rows: Vec<LoopRow> = order
        .iter()
        .enumerate()
        .map(|(slot, &seq)| {
            let (prompt, sampling) = &sequences[seq];
            LoopRow::new(
                i32::try_from(slot).expect("slot count fits i32"),
                seq,
                prompt.len(),
                *sampling,
                (sampling.max_new_tokens as usize).min(n_ctx_seq - prompt.len()),
            )
        })
        .collect();
    // Fresh slots: the loop manages placement itself, so it starts every slot
    // empty, like the batched path's placement does for a cold cache.
    for row in &rows {
        model.seq_rm(row.slot, -1, -1)?;
    }

    // Prefill, group by group in canonical order: identical prompts sit
    // adjacent, the leader is decoded, the members get a seq_cp.
    let mut group_start = 0;
    while group_start < rows.len() {
        let prompt: &[i32] = sequences[order[group_start]].0;
        let mut group_end = group_start + 1;
        while group_end < rows.len() && sequences[order[group_end]].0 == prompt {
            group_end += 1;
        }
        let leader = group_start;
        let shared = group_end > group_start + 1;
        // The leader's prefix: positions [0, n_prompt-1), and for a single
        // member the whole prompt, chunked by n_batch.
        let prefix_end = if shared {
            prompt.len() - 1
        } else {
            prompt.len()
        };
        let mut pos = 0;
        while pos < prefix_end {
            let end = (pos + n_batch).min(prefix_end);
            let last = !shared && end == prompt.len();
            let row = StepRow {
                request: rows[leader].request,
                slot: rows[leader].slot,
                tokens: Cow::Borrowed(&prompt[pos..end]),
                pos0: i32::try_from(pos).expect("position fits i32"),
                sample: last,
                sampling: rows[leader].sampling.into(),
                n_top_logprobs: 0,
                want_logprob: false,
            };
            model.step(&[row], &mut out)?;
            if last {
                let produced = out.row(0);
                rows[leader].emit(produced);
            }
            pos = end;
        }
        if shared {
            for member in group_start + 1..group_end {
                model.seq_cp(rows[leader].slot, rows[member].slot, -1, -1)?;
            }
            // One step carrying every member's terminal row, sampling enabled.
            let terminal = prompt.len() - 1;
            let wave: Vec<StepRow> = (group_start..group_end)
                .map(|member| StepRow {
                    request: rows[member].request,
                    slot: rows[member].slot,
                    tokens: Cow::Borrowed(&prompt[terminal..]),
                    pos0: i32::try_from(terminal).expect("position fits i32"),
                    sample: true,
                    sampling: rows[member].sampling.into(),
                    n_top_logprobs: 0,
                    want_logprob: false,
                })
                .collect();
            model.step(&wave, &mut out)?;
            for (offset, member) in (group_start..group_end).enumerate() {
                let produced = out.row(offset);
                rows[member].emit(produced);
            }
        }
        group_start = group_end;
    }

    // Stop criteria of the batched path: EOG, per-request token budget, full
    // context - checked right after each emitted token.
    for row in &mut rows {
        row.live = !(row.tokens.len() == row.limit || model.is_eog_token(row.next_token)?);
    }
    while rows.iter().any(|row| row.live) {
        let active: Vec<usize> = (0..rows.len()).filter(|&i| rows[i].live).collect();
        let wave: Vec<StepRow> = active
            .iter()
            .map(|&i| {
                let row = &rows[i];
                let pos = row.prompt_len + row.tokens.len() - 1;
                StepRow::decode(
                    row.request,
                    row.slot,
                    row.next_token,
                    i32::try_from(pos).expect("position fits i32"),
                    &row.sampling,
                )
            })
            .collect();
        model.step(&wave, &mut out)?;
        for (offset, &i) in active.iter().enumerate() {
            let produced = out.row(offset);
            let row = &mut rows[i];
            row.emit(produced);
            row.live = !(row.tokens.len() == row.limit || model.is_eog_token(produced.token)?);
        }
    }
    for row in &rows {
        model.request_free(row.request)?;
    }
    let mut sampled = vec![SampledRow::default(); sequences.len()];
    let mut logprobs = vec![Vec::new(); sequences.len()];
    for (index, &seq) in order.iter().enumerate() {
        sampled[seq] = SampledRow {
            tokens: rows[index].tokens.clone(),
            forced: rows[index].forced.clone(),
        };
        logprobs[seq] = rows[index].logprobs.clone();
    }
    Ok((sampled, logprobs))
}

#[test]
fn step_loop_reproduces_generate_continuous_bit_for_bit() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let mut trainer = Trainer::new(&model, train.clone()).expect("load trainer");
    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");

    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    let second = trainer.tokenize_text("Pack my box with").unwrap();
    // Two rows share `first` so the loop exercises the seq_cp copy groups, and
    // the prompts differ so the canonical order is not the caller order.
    let wave: Vec<(&[i32], SamplingParams)> = vec![
        (&second, params(50)),
        (&first, params(51)),
        (&first, params(52)),
    ];
    let n_batch = train.generation_batch.max(1) as usize;

    let engine_reference = engine
        .generate_sampled_continuous(&wave)
        .expect("engine continuous reference");
    let (engine_stepped, engine_lps) =
        step_generate(&mut engine, &wave, n_batch).expect("engine step loop");
    assert_eq!(engine_stepped, engine_reference, "engine: rows bit for bit");

    let trainer_reference = trainer
        .generate_sampled_continuous(&wave)
        .expect("trainer continuous reference");
    let (trainer_stepped, _) =
        step_generate(&mut trainer, &wave, n_batch).expect("trainer step loop");
    assert_eq!(
        trainer_stepped, trainer_reference,
        "trainer: rows bit for bit"
    );

    // The per-token logprobs of the step loop match the single-prompt batched
    // generator bit for bit as well.
    for (index, (prompt, sampling)) in wave.iter().enumerate() {
        let reference = engine.generate(prompt, sampling).expect("engine generate");
        assert_eq!(
            engine_lps[index], reference.logprobs,
            "logprobs bit for bit for prompt {index}"
        );
    }
}

#[test]
fn step_loop_reproduces_the_reasoning_budget_forced_tokens() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let trainer = Trainer::new(&model, train.clone()).expect("load trainer");
    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");

    let rendered = trainer
        .format_chat_messages(r#"[{"role":"user","content":"Hi."}]"#, None, true)
        .expect("render the prompt");
    let end = trainer
        .tokenize_fragment("</think>")
        .expect("tokenize the end tag");
    let mut sampling = params(7);
    sampling.max_new_tokens = u32::try_from(end.len() + 4).unwrap();
    // The prompt opens the reasoning block, so a zero budget forces the end
    // tag into the first sampled positions.
    let opening = "<think>";
    let prompt = trainer
        .tokenize_text(&format!("{rendered}{opening}"))
        .expect("tokenize the prompt");
    let wave: Vec<(&[i32], SamplingParams)> = vec![(&prompt, sampling)];

    engine
        .set_reasoning_budget(0)
        .expect("the template declares a reasoning block");
    let reference = engine
        .generate_sampled_continuous(&wave)
        .expect("reference under the budget");
    let (stepped, _) = step_generate(&mut engine, &wave, 64).expect("step loop under the budget");
    assert_eq!(
        stepped[0].tokens, reference[0].tokens,
        "forced tokens bit for bit"
    );
    assert_eq!(
        stepped[0].forced, reference[0].forced,
        "forced mask bit for bit"
    );
    assert!(
        stepped[0].forced[..end.len()].iter().all(|&forced| forced),
        "the budget forces the end tag: {:?}",
        stepped[0]
    );
    engine.set_reasoning_budget(-1).expect("lift the budget");
}

#[test]
fn step_seq_ops_manage_the_kv_pool_as_documented() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let train = train_config();
    let mut engine = Engine::new(&model, &engine_config(&train)).expect("load engine");
    let prompt = engine.tokenize_text("The quick brown fox").unwrap();
    let sampling = params(11);
    let n_batch = 64;

    // An empty pool reports empty slots.
    let memory = engine.memory().expect("memory");
    assert_eq!(
        memory.n_slots as usize,
        train.effective_generation_concurrency() as usize
    );
    assert_eq!(memory.cells_total, engine.context_size().unwrap() as u64);
    assert_eq!(memory.cells_used, 0);
    assert!(memory.kv_bytes > 0);
    assert_eq!(engine.seq_pos(0).unwrap(), (-1, -1));

    // Prefill slot 0 through step calls.
    let mut out = StepOutput::with_capacity(1, 0);
    let mut pos = 0;
    while pos < prompt.len() {
        let end = (pos + n_batch).min(prompt.len());
        let last = end == prompt.len();
        engine
            .step(
                &[StepRow {
                    request: 1,
                    slot: 0,
                    tokens: Cow::Borrowed(&prompt[pos..end]),
                    pos0: i32::try_from(pos).expect("position fits i32"),
                    sample: last,
                    sampling: sampling.into(),
                    n_top_logprobs: 0,
                    want_logprob: false,
                }],
                &mut out,
            )
            .expect("prefill step");
        pos = end;
    }
    let produced = out.row(0);
    assert!(produced.sampled);
    // With a sliding window, pos_min is the window start, not necessarily 0;
    // pos_max is the last decoded position.
    let (prefill_min, prefill_max) = engine.seq_pos(0).unwrap();
    let prompt_pos = i32::try_from(prompt.len()).expect("prompt length fits i32");
    assert_eq!(prefill_max, prompt_pos - 1);
    assert!((0..=prefill_max).contains(&prefill_min));
    let span0 = (prefill_max - prefill_min + 1) as u64;
    let memory = engine.memory().expect("memory");
    assert_eq!(memory.cells_used, span0);

    // seq_cp into slot 1, then a one-token step in each slot must sample the
    // same token (same KV, same seed).
    engine.seq_cp(0, 1, -1, -1).expect("seq_cp");
    assert_eq!(engine.seq_pos(1).unwrap(), (prefill_min, prefill_max));
    engine.request_free(1).unwrap();
    let mut out = StepOutput::with_capacity(2, 0);
    engine
        .step(
            &[
                StepRow::decode(1, 0, produced.token, prompt_pos, &sampling),
                StepRow::decode(2, 1, produced.token, prompt_pos, &sampling),
            ],
            &mut out,
        )
        .expect("decode step on both slots");
    assert!(out.row(0).sampled && out.row(1).sampled);
    assert_eq!(out.row(0).token, out.row(1).token);
    assert_eq!(out.row(0).logprob.to_bits(), out.row(1).logprob.to_bits());
    // The token sampled there; both slots now hold positions 0..=5, so the
    // continuation feeds it at position 6.
    let next_token = out.row(0).token;
    let next_pos = prompt_pos + 1;
    // The decode above slid the sliding window: re-read slot 0's range.
    let (slot0_min, slot0_max) = engine.seq_pos(0).unwrap();

    // seq_save / seq_load round-trip: the snapshot of slot 0 restores into
    // slot 2 and the continuation is bit-identical.
    let len = engine.seq_save_len(0, 0).expect("save size");
    assert!(len > 0);
    let mut snapshot = vec![0_u8; len];
    let written = engine.seq_save_into(0, 0, &mut snapshot).expect("save");
    assert_eq!(written, len);
    engine.seq_rm(2, -1, -1).expect("clear slot 2");
    let read = engine.seq_load(2, 0, &snapshot).expect("load");
    assert_eq!(read, len);
    assert_eq!(engine.seq_pos(2).unwrap(), (slot0_min, slot0_max));
    let mut out = StepOutput::with_capacity(2, 0);
    engine
        .step(
            &[
                StepRow::decode(3, 0, next_token, next_pos, &sampling),
                StepRow::decode(4, 2, next_token, next_pos, &sampling),
            ],
            &mut out,
        )
        .expect("decode after load");
    assert_eq!(out.row(0).token, out.row(1).token);
    assert_eq!(out.row(0).logprob.to_bits(), out.row(1).logprob.to_bits());

    // A too-small save buffer is refused with the required size, and a
    // truncated snapshot is a clean error, not a crash.
    let mut small = vec![0_u8; len / 2];
    assert!(engine.seq_save_into(0, 0, &mut small).is_err());
    assert!(engine.seq_save_into(0, 0, &mut []).is_err());
    // Device snapshots require internal storage lifetimes a byte slice cannot
    // represent. Refuse the flag before inspecting untrusted snapshot bytes.
    assert!(matches!(
        engine.seq_load(3, retrograd_ffi::RETRO_STATE_SEQ_FLAGS_ON_DEVICE, &[0; 8]),
        Err(EngineError::InvalidArgument(_))
    ));
    engine.seq_rm(3, -1, -1).expect("clear slot 3");
    assert!(engine.seq_load(3, 0, &snapshot[..len / 2]).is_err());

    // seq_rm frees the cells: the used count is exactly the sum of the
    // remaining slots' resident spans.
    engine.seq_rm(0, -1, -1).expect("rm slot 0");
    assert_eq!(engine.seq_pos(0).unwrap(), (-1, -1));
    let span = |slot: i32| {
        let (min, max) = engine.seq_pos(slot).unwrap();
        if min < 0 { 0 } else { (max - min + 1) as u64 }
    };
    assert_eq!(
        engine.memory().unwrap().cells_used,
        span(1) + span(2) + span(3)
    );

    // Validation: a sampling row without a request, a bad slot and an
    // over-batch step are refused before anything is decoded.
    let mut out = StepOutput::with_capacity(1, 0);
    let no_request = StepRow {
        request: -1,
        slot: 0,
        tokens: Cow::Owned(vec![produced.token]),
        pos0: prompt_pos,
        sample: true,
        sampling: sampling.into(),
        n_top_logprobs: 0,
        want_logprob: false,
    };
    let err = engine.step(&[no_request], &mut out).unwrap_err();
    assert!(matches!(err, EngineError::InvalidArgument(_)), "{err}");
    let bad_slot = StepRow::prefill(9, 99, &prompt, 0);
    assert!(matches!(
        engine.step(&[bad_slot], &mut out),
        Err(EngineError::InvalidArgument(_))
    ));
    let overlapping = [
        StepRow::prefill(10, 0, &prompt, 0),
        StepRow::prefill(11, 0, &prompt, 0),
    ];
    assert!(matches!(
        engine.step(&overlapping, &mut out),
        Err(EngineError::InvalidArgument(_))
    ));
    assert_eq!(engine.seq_pos(0).unwrap(), (-1, -1));
    // The engine's n_batch is 64 here, so 20 copies of the 5-token prompt
    // exceed one step's physical batch.
    let wide_prompt = prompt.repeat(20);
    let too_wide = StepRow::prefill(9, 0, &wide_prompt, 0);
    let err = engine.step(&[too_wide], &mut out).unwrap_err();
    assert!(err.to_string().contains("n_batch"), "{err}");

    for request in [1, 2, 3, 4, 9] {
        engine.request_free(request).expect("request_free");
    }
}

#[test]
fn step_budget_handles_chunked_prompt_tails_and_configuration_changes() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        return;
    };
    let train = train_config();
    let trainer = Trainer::new(&model, train.clone()).unwrap();
    let mut engine = Engine::new(&model, &engine_config(&train)).unwrap();
    let rendered = trainer
        .format_chat_messages(r#"[{"role":"user","content":"Hi."}]"#, None, true)
        .unwrap();
    let prompt = trainer
        .tokenize_text(&format!("{rendered}<think>\n"))
        .unwrap();
    let mut sampling = params(7);
    sampling.max_new_tokens = 8;
    let wave = [(&prompt[..], sampling)];
    engine.set_reasoning_budget(0).unwrap();
    let expected = engine.generate_sampled_continuous(&wave).unwrap();
    assert!(expected[0].forced[0]);
    for chunk in 1..=5 {
        let (actual, _) = step_generate(&mut engine, &wave, chunk).unwrap();
        assert_eq!(actual, expected, "prompt chunks of {chunk} tokens");
    }

    engine.seq_rm(0, -1, -1).unwrap();
    let mut out = StepOutput::with_capacity(1, 4);
    engine
        .step(
            &[StepRow {
                request: 50,
                slot: 0,
                tokens: Cow::Borrowed(&prompt),
                pos0: 0,
                sample: true,
                sampling: sampling.into(),
                n_top_logprobs: 4,
                want_logprob: false,
            }],
            &mut out,
        )
        .unwrap();
    let first = out.row(0);
    assert!(first.forced);
    let top: Vec<_> = out.top(0).collect();
    assert_eq!(first.n_top, 4);
    assert_eq!(top.len(), 4);
    assert!(top.iter().all(|(_, lp)| lp.is_finite() && *lp <= 0.0));
    assert!(top.windows(2).all(|pair| pair[0].1 >= pair[1].1));

    // Disabling an active budget must reset its machine before it consults
    // the replaced plan's forced-token vector.
    engine.set_reasoning_budget(-1).unwrap();
    let caller_managed_sampling = SamplingParams {
        max_new_tokens: 0,
        ..sampling
    };
    engine
        .step(
            &[StepRow::decode(
                50,
                0,
                first.token,
                i32::try_from(prompt.len()).unwrap(),
                &caller_managed_sampling,
            )],
            &mut out,
        )
        .unwrap();
    assert!(!out.row(0).forced);
    assert_eq!(out.row(0).n_top, 0);
    assert_eq!(out.top(0).count(), 0);
    engine.request_free(50).unwrap();
}

#[test]
fn v2_step_reports_the_truncated_distribution_and_releases_grammars() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        return;
    };
    let mut engine = Engine::new(&model, &engine_config(&train_config())).unwrap();
    let prompt = engine.tokenize_text("Hello").unwrap();
    let mut sampling = retrograd_core::SamplingParamsV2::from(params(17));
    sampling.top_p = 1e-6;
    let mut row = StepRow::prefill(70, 0, &prompt, 0);
    row.sample = true;
    row.sampling = sampling.clone();
    row.n_top_logprobs = 4;
    row.want_logprob = true;
    let mut out = StepOutput::with_capacity(1, 4);

    row.sampling.grammar_gbnf = Some("root ::= missing".into());
    assert!(engine.step(&[row.clone()], &mut out).is_err());
    assert_eq!(
        engine.seq_pos(0).unwrap(),
        (-1, -1),
        "invalid grammar must not decode a prompt"
    );

    row.sampling.grammar_gbnf = None;
    engine.step(&[row], &mut out).unwrap();
    let first = out.row(0);
    assert_eq!(first.sampled_logprob, 0.0);
    let top: Vec<_> = out.top(0).collect();
    assert_eq!(top[0], (first.token, 0.0));
    assert!(
        top[1..].iter().all(|(_, lp)| *lp == f32::NEG_INFINITY),
        "tokens outside the nucleus have no mass: {top:?}"
    );

    // Constrain one generated position to a token different from the previous
    // one, then remove the grammar without freeing the request.
    let forced = (first.token + 1) % i32::try_from(engine.vocab_size().unwrap()).unwrap();
    sampling.top_p = 1.0;
    sampling.grammar_gbnf = Some(format!("root ::= <[{forced}]>").into());
    let pos = i32::try_from(prompt.len()).unwrap();
    let mut constrained = StepRow::decode(70, 0, first.token, pos, &sampling);
    constrained.want_logprob = true;
    engine.step(&[constrained], &mut out).unwrap();
    assert_eq!(out.row(0).token, forced);
    assert_eq!(out.row(0).sampled_logprob, 0.0);

    sampling.grammar_gbnf = None;
    sampling.top_k = 1;
    sampling.logit_bias = vec![(first.token, 100.0)];
    engine
        .step(
            &[StepRow::decode(70, 0, forced, pos + 1, &sampling)],
            &mut out,
        )
        .unwrap();
    assert_eq!(
        out.row(0).token,
        first.token,
        "removing the grammar must restore unconstrained sampling"
    );
    engine.request_free(70).unwrap();
}
