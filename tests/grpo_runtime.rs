//! End-to-end validation of the GRPO loop against a real GGUF model. The
//! runtime building blocks (generation, scoring, weighted step) are already
//! pinned by `ppo_runtime.rs`; this suite exercises what GRPO adds on top:
//! group sampling and the group-relative baseline. Skipped when no local
//! model exists (override with `RETRO_TEST_MODEL`).

mod common;

use common::serialize_models;
use retrograd::training::batch::train_grpo_batch;
use retrograd::{
    Device, GrpoBatchParams, LoraConfig, LoraDtype, SamplingParams, TargetSet, TrainConfig,
    TrainSequence, Trainer,
};
use retrograd_engine::PackedSequenceBatch;

/// A reward script in the shape the default (persistent) mode asks for: answer
/// the handshake once, then one flushed line per request, for as many batches
/// as the loop sends. `printf` in `sh` flushes on every call.
fn reward_script_with_setup(setup: &str, body: &str) -> String {
    format!(
        r#"IFS= read -r hello || exit 1
printf '{{"protocol":"{}"}}\n'
reply() {{
  value=$(printf '%s\n' "$1" | sed 's/}}$//')
  printf '%s,"_retrograd_batch":%s,"_retrograd_index":%s}}\n' "$value" "$batch" "$index"
}}
{setup}
while IFS= read -r line; do
  case "$line" in
    *'"_retrograd_batch_end"'*) printf '%s\n' "$line";;
    *)
      batch=$(printf '%s\n' "$line" | sed -n 's/.*"_retrograd_batch":\([0-9][0-9]*\).*/\1/p')
      index=$(printf '%s\n' "$line" | sed -n 's/.*"_retrograd_index":\([0-9][0-9]*\).*/\1/p')
      {body}
      ;;
  esac
done
"#,
        retrograd::training::REWARD_PROTOCOL_VERSION
    )
}

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 64,
        n_batch: 64,
        // A full physical batch exercises GRPO's differentiable shared-prefix
        // graph when both group members fit in the full physical window.
        n_ubatch: 64,
        // Two prompt groups of two members exercise the continuous,
        // heterogeneous generation path in the end-to-end GRPO tests.
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        ..TrainConfig::default()
    }
}

fn lora() -> LoraConfig {
    let mut cfg = LoraConfig::qv(2, 4.0);
    cfg.seed = 7;
    cfg.targets = TargetSet::Patterns(vec!["blk.2.attn_q.weight".to_string()]);
    cfg
}

#[test]
fn packed_fanout_transaction_publishes_only_on_the_final_pass() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create LoRA");
    let token = *trainer
        .tokenize_text("shared prefix transaction")
        .unwrap()
        .last()
        .unwrap();

    // Forty physical prefix tokens belong to both logical sequences. The two
    // twelve-token suffixes reuse logical positions 40..52, forcing LFM2's
    // indexed ShortConv path while keeping the physical graph at n_ubatch=64.
    let tokens = vec![token; 64];
    let mut labels = vec![-1; 64];
    let mut weights = vec![0.0; 64];
    let mut positions = Vec::with_capacity(64);
    let mut seq_offsets = Vec::with_capacity(65);
    let mut seq_ids = Vec::with_capacity(104);
    seq_offsets.push(0);
    for physical in 0..64 {
        if physical < 40 {
            positions.push(physical as i32);
            seq_ids.extend([0, 1]);
        } else {
            positions.push(if physical < 52 {
                physical as i32
            } else {
                (physical - 12) as i32
            });
            seq_ids.push(if physical < 52 { 0 } else { 1 });
            labels[physical] = token;
            weights[physical] = 2.0; // cancel ggml's 1 / accumulation_steps
        }
        seq_offsets.push(seq_ids.len());
    }
    let batch = PackedSequenceBatch {
        tokens,
        labels,
        weights,
        positions,
        seq_offsets,
        seq_ids,
        n_sequences: 2,
        n_topk: 1,
    };

    let mut callbacks = 0;
    let first = trainer
        .train_packed_sequences_controlled(&batch, 2, 2, |_, _| {
            callbacks += 1;
            Ok(true)
        })
        .expect("accumulate first physical pass")
        .0;
    assert_eq!(first.global_step, 0);
    assert_eq!(callbacks, 0, "an accumulation pass published an update");

    let second = trainer
        .train_packed_sequences_controlled(&batch, 2, 2, |_, _| {
            callbacks += 1;
            Ok(true)
        })
        .expect("finish logical update")
        .0;
    assert_eq!(second.global_step, 1);
    assert_eq!(callbacks, 1, "the logical update was not published once");
}

#[test]
fn auto_packing_trains_unequal_passes_in_one_optimizer_update() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create LoRA");
    assert!(trainer.supports_shared_prefix_packed_training().unwrap());
    let token = *trainer
        .tokenize_text("shared prefix")
        .unwrap()
        .last()
        .unwrap();
    let mut sequences = Vec::new();
    for completion in [50, 8, 8, 8] {
        let tokens = vec![token; 8 + completion];
        let mut train_mask = vec![false; 8];
        train_mask.extend(vec![true; completion]);
        let old_logprobs = trainer.score_masked_tokens(&tokens, &train_mask).unwrap();
        sequences.push(TrainSequence {
            tokens,
            train_mask,
            old_logprobs,
            reward: if completion == 50 { 1.0 } else { 0.0 },
            group_id: 1,
            intermediate_returns: Vec::new(),
        });
    }
    // The long completion fits alone (7 + 50 + 3 = 60), but never
    // beside a short one (7 + 50 + 8 + 2 = 67 > micro_batch=64).
    let mut progress = Vec::new();
    let metrics = train_grpo_batch(
        &mut trainer,
        &sequences,
        &GrpoBatchParams {
            epochs: 1,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.0,
            loss_denominator: 50,
            seed: 42,
            scheduler_total_rollouts: None,
        },
        &config(),
        &mut |event| progress.push(event),
    )
    .expect("train unequal packed passes");
    assert_eq!(metrics.global_step, 1);
    assert!(metrics.train_loss.is_finite());
    assert_eq!(progress.len(), 1);
    let notes = &progress[0].notes;
    assert!(
        notes
            .iter()
            .any(|note| note.starts_with("shared-prefix packing: ")),
        "{notes:?}"
    );
    assert!(
        !notes.iter().any(|note| note.contains("packing disabled")),
        "{notes:?}"
    );
}

#[test]
fn batched_generation_and_suffix_scoring_share_one_loaded_model() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 17,
    };
    let generations = trainer
        .generate_batch(&prompt, &[sampling, sampling])
        .expect("generate two sequences");
    assert_eq!(generations.len(), 2);
    assert_eq!(generations[0].tokens, generations[1].tokens);
    assert_eq!(generations[0].logprobs.len(), generations[1].logprobs.len());
    for (&left, &right) in generations[0].logprobs.iter().zip(&generations[1].logprobs) {
        assert!(
            (left - right).abs() < 0.01,
            "identical seeded rows diverged: {left} vs {right}"
        );
    }

    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 23,
    };
    let completions = trainer
        .generate_batch(
            &prompt,
            &[
                sampling,
                SamplingParams {
                    seed: 24,
                    ..sampling
                },
            ],
        )
        .expect("generate suffixes");
    let rows = completions
        .iter()
        .map(|completion| {
            let mut tokens = prompt.clone();
            tokens.extend_from_slice(&completion.tokens);
            tokens
        })
        .collect::<Vec<_>>();
    let scalar = rows
        .iter()
        .map(|tokens| {
            trainer
                .score_masked_tokens(
                    tokens,
                    &(0..tokens.len())
                        .map(|index| index >= prompt.len())
                        .collect::<Vec<_>>(),
                )
                .expect("scalar score")
        })
        .collect::<Vec<_>>();
    let inputs = rows
        .iter()
        .map(|tokens| (tokens.as_slice(), prompt.len()))
        .collect::<Vec<_>>();
    let batched = trainer
        .score_token_suffix_batch(&inputs)
        .expect("batched score");
    assert_eq!(batched, scalar);
}

/// The shared prefix must be decoded once per call, not once per completion.
/// It is decoded into sequence zero and every branch takes a copy of it; the
/// mid-sequence rollback this replaced is refused by a recurrent cache without
/// rollback snapshots, and the only recovery from that refusal is a full prompt
/// prefill per branch. Both paths must produce the
/// same scores, so the counters are the only thing that can catch the
/// regression.
#[test]
fn shared_prefix_scoring_decodes_the_prefix_once_per_call() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    let rows = [5, 9]
        .map(|extra| {
            let mut tokens = prompt.clone();
            tokens.extend(prompt.iter().cycle().take(extra));
            tokens
        })
        .to_vec();
    let inputs = rows
        .iter()
        .map(|tokens| (tokens.as_slice(), prompt.len()))
        .collect::<Vec<_>>();

    let before = trainer.scoring_stats().expect("scoring stats");
    let isolated = trainer
        .score_token_suffix_batch(&inputs)
        .expect("batched score");
    let stats = trainer
        .scoring_stats()
        .expect("scoring stats")
        .delta_since(before);
    assert_eq!(stats.calls, 1);
    assert_eq!(stats.prefix_decodes, 1, "the prefix was re-decoded");
    assert_eq!(stats.prefix_reprefills, 0);
    assert_eq!(stats.branch_evictions_refused, 0);
    assert_eq!(
        stats.scored_positions,
        rows.iter()
            .map(|row| row.len() - prompt.len())
            .sum::<usize>() as u64
    );

    // Same scores through the single-sequence rollback path, whatever the
    // rollback costs there.
    let _env = common::EnvGuard::set("RETRO_SCORING_BRANCH_ISOLATION", "0");
    let rolled_back = trainer
        .score_token_suffix_batch(&inputs)
        .expect("batched score without branch isolation");
    assert_eq!(isolated, rolled_back);
}

/// The physical decode batch (`llama_decode`) is not numerically
/// batch-invariant: mixing rows from unrelated prompts into one physical
/// decode changes the GEMM tiling ggml selects, which can perturb the
/// generated tokens versus decoding each prompt group alone. That leaves two
/// guarantees this call *can* make: it is deterministic for a fixed input,
/// and each row's output only depends on that row's own prompt and seed, not
/// on the physical position other rows occupy in the same call.
#[test]
fn continuous_generation_preserves_per_prompt_seeded_rows() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    let second = trainer.tokenize_text("Pack my box with").unwrap();
    let params = |seed| SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed,
    };
    let rows: [(&[i32], SamplingParams); 4] = [
        (&first, params(31)),
        (&first, params(32)),
        (&second, params(33)),
        (&second, params(34)),
    ];
    let baseline = trainer
        .generate_tokens_continuous(&rows)
        .expect("continuous generation");
    let repeated = trainer
        .generate_tokens_continuous(&rows)
        .expect("continuous generation");
    assert_eq!(
        repeated, baseline,
        "same input must generate deterministically"
    );

    let reordered = [rows[2], rows[3], rows[0], rows[1]];
    let swapped = trainer
        .generate_tokens_continuous(&reordered)
        .expect("continuous generation");
    assert_eq!(
        swapped,
        vec![
            baseline[2].clone(),
            baseline[3].clone(),
            baseline[0].clone(),
            baseline[1].clone(),
        ],
        "each row's output must follow its own prompt and seed, not its physical slot"
    );
}

/// Held-out evaluation generates its prompts in continuous waves of
/// `generation_concurrency`, so a dataset that does not divide by the wave width
/// has to survive a partial final wave with every prompt scored exactly once.
/// Seeds derive from a prompt's index in the dataset, not from its slot in a
/// wave, so two evaluations of the same policy agree.
#[test]
fn reward_evaluation_covers_every_prompt_across_partial_waves() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let training = config();
    let mut trainer = Trainer::new(model, training.clone()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");

    let dir = std::env::temp_dir().join(format!("retrograd-grpo-eval-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // Five prompts against a four-sequence wave: one full wave, one holding a
    // single prompt.
    let data = dir.join("eval.jsonl");
    let prompts = [
        "The quick brown fox",
        "Pack my box with",
        "How razorback jumping",
        "Sphinx of black quartz",
        "Jackdaws love my",
    ];
    std::fs::write(
        &data,
        prompts
            .iter()
            .map(|prompt| {
                format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"{prompt}\"}}]}}\n")
            })
            .collect::<String>(),
    )
    .unwrap();
    // One reward per request, increasing with the request's position, so a
    // missing or duplicated completion moves the aggregate.
    let reward = dir.join("reward.sh");
    std::fs::write(
        &reward,
        reward_script_with_setup("i=0", "reply \"{\\\"reward\\\": $i}\"; i=$((i + 1))"),
    )
    .unwrap();

    let mut grpo_config = valid_eval_config(&data, &reward);
    let metrics = retrograd::training::grpo::evaluate(
        &mut trainer,
        &grpo_config,
        &training,
        &data,
        /* max_examples = */ None,
    )
    .expect("evaluate rewards");
    assert_eq!(
        metrics.examples,
        prompts.len(),
        "every prompt must be generated and scored exactly once"
    );
    assert_eq!(metrics.reward_min, 0.0);
    assert_eq!(metrics.reward_max, prompts.len() as f32 - 1.0);
    assert_eq!(metrics.mean_reward, 2.0, "0..4 averages to 2");

    // Same policy, same seeds: the evaluation is comparable across updates.
    let repeated =
        retrograd::training::grpo::evaluate(&mut trainer, &grpo_config, &training, &data, None)
            .expect("evaluate rewards again");
    assert_eq!(repeated.examples, metrics.examples);
    assert_eq!(repeated.mean_reward, metrics.mean_reward);

    // A capped evaluation keeps the dataset's spread, still one wave at a time.
    let capped =
        retrograd::training::grpo::evaluate(&mut trainer, &grpo_config, &training, &data, Some(3))
            .expect("evaluate a capped subset");
    assert_eq!(capped.examples, 3);

    // Wave width changes the physical decode, not which prompts are scored nor
    // the order they reach the reward command in: the position-derived rewards
    // pin exactly that, at one sequence per call.
    grpo_config.sampling.seed = 42;
    let serial = TrainConfig {
        generation_concurrency: 1,
        ..training.clone()
    };
    let one_at_a_time =
        retrograd::training::grpo::evaluate(&mut trainer, &grpo_config, &serial, &data, None)
            .expect("evaluate without batching");
    assert_eq!(one_at_a_time.examples, prompts.len());
    assert_eq!(one_at_a_time.mean_reward, metrics.mean_reward);

    std::fs::remove_dir_all(&dir).ok();
}

/// A GRPO config whose only exercised part is the reward command and the
/// sampling parameters: the evaluation path never looks at the update schedule.
fn valid_eval_config(
    prompts: &std::path::Path,
    reward: &std::path::Path,
) -> retrograd::config::GrpoConfig {
    retrograd::config::GrpoConfig {
        prompts: prompts.to_path_buf(),
        reward_command: vec!["/bin/sh".into(), reward.display().to_string()],
        // The default: one worker for the whole loop. `reward_script` writes
        // the handshake that goes with it.
        reward_protocol: retrograd::RewardProtocol::default(),
        updates: 1,
        prompts_per_update: 1,
        group_size: 2,
        grpo_epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        mask_truncated: false,
        baseline: retrograd::config::AdvantageBaseline::Mean,
        prompt_order: retrograd::config::PromptOrder::Sequential,
        overlong_penalty: None,
        kl_schedule: None,
        dynamic_sampling: None,
        judge: None,
        max_stalled_updates: retrograd::config::DEFAULT_MAX_STALLED_UPDATES,
        sampling: retrograd::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: 7,
        },
    }
}

/// Full GRPO loop: prompts -> groups of rollouts -> external rewards ->
/// group-relative advantages -> weighted differentiable updates, with the
/// base model frozen.
#[test]
fn grpo_runs_end_to_end() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    // Exercise the lowest-VRAM path: each two-member GRPO group spans two
    // physical generation waves and must be reassembled before scoring.
    let low_vram = TrainConfig {
        generation_concurrency: 1,
        ..config()
    };
    let mut trainer = Trainer::new(model, low_vram.clone()).expect("load trainer");
    let mut adapter = lora();
    adapter.dtype = LoraDtype::F16;
    trainer.create_lora(&adapter).expect("create F16 lora");
    let packed_supported = trainer
        .supports_shared_prefix_packed_training()
        .expect("packed-training capability");
    let dir = std::env::temp_dir().join(format!(
        "retrograd-grpo-e2e-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let prompts = dir.join("prompts.jsonl");
    std::fs::write(
        &prompts,
        concat!(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox\"}]}\n",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Pack my box with\"}]}\n"
        ),
    )
    .unwrap();
    // Deterministic rewards by input position: the first group is uniform and
    // must be filtered, while the second gets [0, 1] and must train.
    let reward = dir.join("reward.sh");
    std::fs::write(
        &reward,
        reward_script_with_setup(
            "i=0",
            "case $i in 3) reward=1 ;; *) reward=0 ;; esac; reply \"{\\\"reward\\\": $reward}\"; i=$((i + 1))",
        ),
    )
    .unwrap();

    let grpo_config = retrograd::config::GrpoConfig {
        prompts,
        reward_command: vec!["/bin/sh".into(), reward.display().to_string()],
        // The default: one worker for the whole loop. `reward_script` writes
        // the handshake that goes with it.
        reward_protocol: retrograd::RewardProtocol::default(),
        updates: 1,
        prompts_per_update: 2,
        group_size: 2,
        grpo_epochs: 2,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.1,
        mask_truncated: false,
        baseline: retrograd::config::AdvantageBaseline::Mean,
        prompt_order: retrograd::config::PromptOrder::Sequential,
        overlong_penalty: None,
        kl_schedule: None,
        dynamic_sampling: None,
        judge: None,
        // Not what this test is about: one update, and both groups carry
        // signal, so the stall counter never leaves zero.
        max_stalled_updates: retrograd::config::DEFAULT_MAX_STALLED_UPDATES,
        // Dr. GRPO is strictly on-policy: temperature and top_p must be 1.
        sampling: retrograd::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: 42,
        },
    };

    let mut progress_events = 0;
    let mut trained_fractions = Vec::new();
    let metrics =
        retrograd::training::grpo::run(&mut trainer, &grpo_config, &low_vram, &mut |event| {
            progress_events += 1;
            for name in [
                "reward/mean",
                "reward/min",
                "reward/max",
                "reward/std",
                "reward/group_std",
                "batch/zero_std_group_fraction",
                "batch/advantage_abs_mean",
                "completions/length_mean",
                "completions/length_min",
                "completions/length_max",
                "completions/truncation_fraction",
                "batch/trained_fraction",
                "policy/surrogate_loss",
                "policy/kl",
                "policy/clip_fraction",
                "policy/total_loss",
                "timing/sampling_seconds",
                "timing/generation_seconds",
                "timing/behavior_scoring_seconds",
                "timing/optimizer_seconds",
                "timing/optimizer_graph_build_seconds",
                "timing/optimizer_allocation_seconds",
                "timing/optimizer_execution_seconds",
            ] {
                let value = event
                    .values
                    .iter()
                    .find(|value| value.name == name)
                    .unwrap_or_else(|| panic!("missing metric {name}"));
                assert!(value.value.is_finite(), "{name} is not finite");
                if name == "batch/trained_fraction" {
                    trained_fractions.push(value.value);
                }
            }
            let timing = |name| {
                event
                    .values
                    .iter()
                    .find(|value| value.name == name)
                    .unwrap_or_else(|| panic!("missing metric {name}"))
                    .value
            };
            let sampling = timing("timing/sampling_seconds");
            let sampling_parts = timing("timing/generation_seconds")
                + timing("timing/behavior_scoring_seconds");
            assert!(
                (sampling - sampling_parts).abs() <= 1.0e-6 * sampling.abs().max(1.0),
                "sampling total {sampling} does not match generation + behavior scoring {sampling_parts}"
            );
            let optimizer = timing("timing/optimizer_seconds");
            let optimizer_parts = timing("timing/optimizer_graph_build_seconds")
                + timing("timing/optimizer_allocation_seconds")
                + timing("timing/optimizer_execution_seconds");
            assert!(
                (optimizer - optimizer_parts).abs() <= 1.0e-6 * optimizer.abs().max(1.0),
                "optimizer total {optimizer} does not match its runtime components {optimizer_parts}"
            );
        })
        .expect("grpo run");

    assert_eq!(progress_events, 2, "one progress event per grpo epoch");
    assert_eq!(trained_fractions, vec![0.5, 0.5]);
    // Attention-only models share the two live rows in one packed step.
    // Recurrent models deliberately keep them isolated because their state
    // cannot branch across packed sequence ids. Four filtered slots still
    // advance the scheduler by one steps_per_row each.
    assert_eq!(
        metrics.global_step,
        if packed_supported { 6 } else { 8 },
        "trained and filtered scheduler steps must match the selected layout"
    );
    assert!(metrics.train_loss.is_finite());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn dynamic_sampling_resamples_zero_signal_groups() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let dir = std::env::temp_dir().join(format!(
        "retrograd-grpo-dyn-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let prompts = dir.join("prompts.jsonl");
    std::fs::write(
        &prompts,
        concat!(
            "{\"messages\":[{\"role\":\"user\",\"content\":\"The quick brown fox\"}]}\n",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"Pack my box with\"}]}\n"
        ),
    )
    .unwrap();
    // A reward keyed on a persistent global rollout counter: only the second
    // rollout ever sampled gets reward 1, every other 0. So the first group is
    // informative ([0, 1]) and every later group is uniform ([0, 0]). With a
    // resample factor of 2 the loop draws the full budget of 4 candidate groups
    // (one informative, three uniform), then pads to two groups - half live.
    let counter = dir.join("counter");
    let reward = dir.join("reward.sh");
    std::fs::write(
        &reward,
        reward_script_with_setup(
            &format!("cf='{}'", counter.display()),
            "n=$(cat \"$cf\" 2>/dev/null || echo 0); if [ \"$n\" -eq 1 ]; then r=1; else r=0; fi; reply \"{\\\"reward\\\": $r}\"; echo $((n + 1)) > \"$cf\"",
        ),
    )
    .unwrap();

    let grpo_config = retrograd::config::GrpoConfig {
        prompts,
        reward_command: vec!["/bin/sh".into(), reward.display().to_string()],
        // The default: one worker for the whole loop. `reward_script` writes
        // the handshake that goes with it.
        reward_protocol: retrograd::RewardProtocol::default(),
        updates: 1,
        prompts_per_update: 2,
        group_size: 2,
        grpo_epochs: 1,
        clip_range_low: 0.2,
        clip_range_high: 0.28,
        kl_coefficient: 0.0,
        mask_truncated: false,
        baseline: retrograd::config::AdvantageBaseline::Mean,
        prompt_order: retrograd::config::PromptOrder::Sequential,
        overlong_penalty: None,
        kl_schedule: None,
        judge: None,
        // The update here ends with one live group out of two, so it is not a
        // stalled update and the counter stays at zero whatever this holds.
        max_stalled_updates: retrograd::config::DEFAULT_MAX_STALLED_UPDATES,
        dynamic_sampling: Some(retrograd::config::DynamicSampling {
            max_resample_factor: 2,
        }),
        sampling: retrograd::SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_new_tokens: 8,
            seed: 42,
        },
    };

    let mut trained_fraction = None;
    let mut groups_sampled_fraction = None;
    retrograd::training::grpo::run(&mut trainer, &grpo_config, &config(), &mut |event| {
        for value in &event.values {
            if value.name == "batch/trained_fraction" {
                trained_fraction = Some(value.value);
            } else if value.name == "batch/groups_sampled_fraction" {
                groups_sampled_fraction = Some(value.value);
            }
        }
    })
    .expect("grpo run");

    // One informative group out of two trained slots.
    assert_eq!(trained_fraction, Some(0.5));
    // Four candidate groups drawn for two update slots: the cap was reached.
    assert_eq!(groups_sampled_fraction, Some(2.0));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn pre_generated_batch_trains_disjoint_policy_segments() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let packed_supported = trainer
        .supports_shared_prefix_packed_training()
        .expect("packed-training capability");

    let mut sequences = Vec::new();
    for (text, reward) in [
        ("This is a deliberately longer first training answer.", 0.0),
        (
            "This is a deliberately longer second training response.",
            1.0,
        ),
    ] {
        let tokens = trainer.tokenize_text(text).expect("tokenize batch row");
        assert!(tokens.len() >= 4 && tokens.len() <= 64);
        let mut train_mask = vec![false; tokens.len()];
        // Two trained positions separated by an untrained observation token.
        let last = tokens.len() - 1;
        train_mask[last - 2] = true;
        train_mask[last] = true;
        let old_logprobs = trainer
            .score_masked_tokens(&tokens, &train_mask)
            .expect("score masked row");
        sequences.push(TrainSequence {
            tokens,
            old_logprobs,
            train_mask,
            reward,
            group_id: 99,
            intermediate_returns: vec![1.0, 0.0],
        });
    }
    let mut progress = Vec::new();
    let metrics = train_grpo_batch(
        &mut trainer,
        &sequences,
        &GrpoBatchParams {
            epochs: 1,
            clip_range_low: 0.2,
            clip_range_high: 0.28,
            kl_coefficient: 0.0,
            loss_denominator: 8,
            seed: 42,
            scheduler_total_rollouts: None,
        },
        &config(),
        &mut |event| progress.push(event),
    )
    .expect("train pre-generated GRPO batch");
    // Attention-only models share one packed step; recurrent models use two
    // isolated fixed-width rows.
    assert_eq!(metrics.global_step, if packed_supported { 1 } else { 2 });
    assert_eq!(progress.len(), 1);
    assert!(metrics.train_loss.is_finite());
}

/// The token-level design rests on one convention: `score_masked_tokens`
/// returns the mask-selected targets in increasing index order. If that
/// alignment ever slipped, rewards would land on the wrong tokens with nothing
/// else failing.
///
/// Note it is checked against the full pass with a tolerance, not exactly.
/// llama.cpp splits a sequence into micro-batches, and which positions request
/// logits changes that split, so the same target scored from a different start
/// offset lands within ~1e-1 rather than bit-for-bit. A one-position misalign-
/// ment moves logprobs by whole units, so the tolerance still catches it.
/// This is also why collection and training must score from the *same* start
/// offset - otherwise the first-step GRPO ratio is silently off by ~0.3%.
#[test]
fn masked_scoring_selects_the_right_targets_in_order() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");

    let tokens = trainer
        .tokenize_text("A somewhat longer sentence used to exercise masked scoring.")
        .expect("tokenize");
    assert!(tokens.len() >= 6);
    let all_scores = trainer.score_tokens(&tokens).expect("score full sequence");
    assert_eq!(all_scores.len(), tokens.len() - 1);

    // Two disjoint policy segments separated by untrained observation tokens.
    let mut train_mask = vec![false; tokens.len()];
    let last = tokens.len() - 1;
    train_mask[2] = true;
    train_mask[last - 1] = true;
    train_mask[last] = true;

    let masked = trainer
        .score_masked_tokens(&tokens, &train_mask)
        .expect("score masked");
    let expected = (1..tokens.len())
        .zip(&all_scores)
        .filter(|(target, _)| train_mask[*target])
        .map(|(_, &score)| score)
        .collect::<Vec<_>>();
    assert_eq!(masked.len(), 3);
    for (index, (&got, &want)) in masked.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 0.5,
            "masked score {index} is misaligned: {got} vs {want} (all: {masked:?} vs {expected:?})"
        );
    }

    // Repeated calls must agree exactly: collection and the optimizer's
    // re-scoring share this path, and the first-step ratio depends on it.
    let repeat = trainer
        .score_masked_tokens(&tokens, &train_mask)
        .expect("re-score masked");
    assert_eq!(masked, repeat, "masked scoring must be reproducible");

    // A mask covering a contiguous completion is the single-turn GRPO case.
    let mut suffix_mask = vec![false; tokens.len()];
    suffix_mask[3..].fill(true);
    let suffix = trainer
        .score_masked_tokens(&tokens, &suffix_mask)
        .expect("score suffix mask");
    assert_eq!(suffix.len(), tokens.len() - 3);
    for (&got, &want) in suffix.iter().zip(&all_scores[2..]) {
        assert!((got - want).abs() < 0.5, "suffix mask is misaligned");
    }
}

#[test]
fn masked_scoring_rejects_unusable_masks() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let tokens = trainer
        .tokenize_text("short sentence here")
        .expect("tokenize");

    // Length mismatch, empty selection, and a trainable first token: each has
    // no valid teacher-forced reading and must fail rather than be coerced.
    assert!(
        trainer
            .score_masked_tokens(&tokens, &vec![false; tokens.len() - 1])
            .is_err()
    );
    assert!(
        trainer
            .score_masked_tokens(&tokens, &vec![false; tokens.len()])
            .is_err()
    );
    let mut first_trainable = vec![false; tokens.len()];
    first_trainable[0] = true;
    assert!(
        trainer
            .score_masked_tokens(&tokens, &first_trainable)
            .is_err()
    );
}

/// The opt-in fast sampling context (F16 KV + flash-attention) must stay
/// confined to generation: it is allowed to change the sampled tokens, but the
/// optimizer context - which produces `old_logprobs` and every training
/// gradient - must score bit-identically to a run without the flag. That is
/// what keeps all ratios exactly 1 before the first optimizer step.
#[test]
fn fast_sampling_context_leaves_the_optimizer_context_exact() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let fast = TrainConfig {
        fast_generation_context: true,
        ..config()
    };

    let mut exact_trainer = Trainer::new(model.clone(), config()).expect("load exact trainer");
    let tokens = exact_trainer
        .tokenize_text("The quick brown fox jumps")
        .expect("tokenize");
    let exact_scores = exact_trainer.score_tokens(&tokens).expect("exact scoring");
    drop(exact_trainer);

    let mut fast_trainer = Trainer::new(model, fast).expect("load fast trainer");
    assert_eq!(
        fast_trainer.score_tokens(&tokens).expect("fast scoring"),
        exact_scores,
        "the fast sampling context must not touch the optimizer context"
    );

    // The fast context still generates, batched and deterministically.
    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 17,
    };
    let generations = fast_trainer
        .generate_batch(&tokens, &[sampling, sampling])
        .expect("generate two sequences on the fast context");
    assert_eq!(generations.len(), 2);
    assert_eq!(generations[0].tokens.len(), 4);
    assert_eq!(generations[0].tokens, generations[1].tokens);
}

/// The cost model of a multi-turn rollout: a turn whose prompt extends the
/// previous turn's tokens must decode only what it added.
///
/// Without this, `generate_tokens_continuous` clears the context and re-decodes
/// the whole prefix at every call, so a trajectory's prefill grows with the
/// *square* of its turn count - invisible at the one or two turns a
/// question-answering run takes, and the dominant cost of an agentic one.
#[test]
fn a_continuation_prefills_only_the_tokens_it_added() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 11,
    };

    let before = trainer.generation_stats().expect("generation stats");
    let opening = trainer
        .generate_tokens_continuous(&[(prompt.as_slice(), params)])
        .expect("continuous generation");
    let cold = trainer
        .generation_stats()
        .expect("generation stats")
        .delta_since(before);
    assert_eq!(cold.hits, 0, "an empty context can reuse nothing");
    assert_eq!(cold.reused_tokens, 0);
    assert_eq!(cold.prefilled_tokens, prompt.len() as u64);

    // What the next turn of a rollout looks like: the prompt it was given, the
    // tokens it sampled, and the framing the environment's answer added.
    let mut turn = prompt.clone();
    turn.extend_from_slice(&opening[0]);
    // Every sampled token but the last: a token is decoded to produce the next
    // one, so the token generation stopped on was emitted and never fed back.
    // Its position is not in the cache, and this call has to decode it.
    let resident = (turn.len() - 1) as u64;
    turn.extend_from_slice(&trainer.tokenize_fragment(" and then").unwrap());

    let before = trainer.generation_stats().expect("generation stats");
    trainer
        .generate_tokens_continuous(&[(turn.as_slice(), params)])
        .expect("continuous generation");
    let warm = trainer
        .generation_stats()
        .expect("generation stats")
        .delta_since(before);
    assert_eq!(
        warm.hits, 1,
        "the prompt extends what the slot already holds"
    );
    assert_eq!(warm.reused_tokens, resident);
    assert_eq!(
        warm.prefilled_tokens,
        turn.len() as u64 - resident,
        "only the tokens the turn added may be decoded"
    );
    assert_eq!(warm.evictions, 0);
}

/// A prompt that diverges from what the slot holds takes the whole slot with
/// it. The alternative - dropping the diverged tail and keeping the head,
/// would mean removing a range from the middle of a sequence, which is not
/// available for the recurrent half of a hybrid model.
#[test]
fn a_divergent_prompt_costs_its_slot_rather_than_reusing_a_wrong_prefix() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 12,
    };
    let first = trainer.tokenize_text("The quick brown fox").unwrap();
    trainer
        .generate_tokens_continuous(&[(first.as_slice(), params)])
        .expect("continuous generation");

    let second = trainer.tokenize_text("Pack my box with five").unwrap();
    let before = trainer.generation_stats().expect("generation stats");
    let diverged = trainer
        .generate_tokens_continuous(&[(second.as_slice(), params)])
        .expect("continuous generation");
    let stats = trainer
        .generation_stats()
        .expect("generation stats")
        .delta_since(before);
    assert_eq!(stats.hits, 0);
    assert_eq!(stats.prefilled_tokens, second.len() as u64);

    // And the answer is the one an empty context would have given: a slot that
    // was dropped rather than partially reused leaves nothing behind.
    let mut fresh = Trainer::new(common::model_path_if_available().expect("model"), config())
        .expect("load trainer");
    let reference = fresh
        .generate_tokens_continuous(&[(second.as_slice(), params)])
        .expect("continuous generation");
    assert_eq!(diverged, reference);
}

/// The invalidation that matters most, because nothing downstream would notice
/// it: a prefix decoded before an optimizer step describes the weights of
/// before, and reusing it would sample from a model that no longer exists.
#[test]
fn a_training_step_forgets_every_resident_prefix() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(model, config()).expect("load trainer");
    trainer.create_lora(&lora()).expect("create lora");
    let params = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 4,
        seed: 13,
    };
    let prompt = trainer.tokenize_text("The quick brown fox").unwrap();
    let opening = trainer
        .generate_tokens_continuous(&[(prompt.as_slice(), params)])
        .expect("continuous generation");
    let mut turn = prompt.clone();
    turn.extend_from_slice(&opening[0]);
    turn.extend_from_slice(&trainer.tokenize_fragment(" and then").unwrap());

    // `train_tokens` wants strictly more than one context of material; the
    // content is irrelevant here, only that the weights move.
    let corpus = trainer
        .tokenize_text(&"the quick brown fox jumps over the lazy dog. ".repeat(40))
        .unwrap();
    trainer.train_tokens(&corpus).expect("one training step");

    let before = trainer.generation_stats().expect("generation stats");
    trainer
        .generate_tokens_continuous(&[(turn.as_slice(), params)])
        .expect("continuous generation");
    let stats = trainer
        .generation_stats()
        .expect("generation stats")
        .delta_since(before);
    assert_eq!(
        stats.hits, 0,
        "a prefix decoded under the previous weights must not be reused"
    );
    assert_eq!(stats.prefilled_tokens, turn.len() as u64);
}

#[test]
fn packing_geometry_benchmarks_complete_updates_on_disposable_adapters() {
    use retrograd::training::packing_benchmark::PackingBenchmark;

    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    for (width, fanout) in [(64, 4), (32, 2)] {
        let mut training = config();
        training.n_ubatch = width;
        training.shared_prefix_fanout = retrograd_core::SharedPrefixFanout::Exact(fanout);
        let result =
            retrograd::training::packing_benchmark::benchmark(&model, &training, &lora(), 8, 4, 4)
                .expect("benchmark private packed trainer");
        let PackingBenchmark::Measured { seconds, memory } = result else {
            panic!("ubatch={width}, fanout={fanout} was refused: {result:?}");
        };
        assert_eq!(seconds.len(), 3);
        assert!(seconds.iter().all(|s| s.is_finite() && *s > 0.0));
        // The planner charges `device_bytes`; on this CPU fixture the runtime's
        // buffers are host buffers, so the report is checked as a whole.
        assert!(memory.device_bytes + memory.host_bytes > 0);
        eprintln!("packed geometry ubatch={width}, fanout={fanout}: {seconds:?} seconds/update");
    }
    // A locked fanout the width cannot hold is a refusal, not an error.
    let mut training = config();
    training.n_ubatch = 16;
    training.shared_prefix_fanout = retrograd_core::SharedPrefixFanout::Exact(4);
    let result =
        retrograd::training::packing_benchmark::benchmark(&model, &training, &lora(), 8, 4, 4)
            .expect("a refused geometry is not an error");
    assert!(matches!(result, PackingBenchmark::Refused(_)), "{result:?}");
}
