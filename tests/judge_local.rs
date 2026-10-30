//! The in-process judge on the CPU fixture: the J3 local transport and the
//! J4 pointwise strategy end to end, on a real model. Runs in the CPU
//! integration lane (`scripts/test-cpu-integration.sh judge_local`); skipped
//! without the fixture.

mod common;

use common::serialize_models;
use retrograd::training::rollout::local_judge::LocalJudge;
use retrograd::{Device, SamplingParams, TrainConfig, Trainer};
use retrograd_agent_core::config::JudgeFailurePolicy;
use retrograd_agent_core::trajectory::{Message, Role, Trajectory, TrajectoryGroup};
use retrograd_core::SamplingParamsV2;
use retrograd_engine::JudgeWeights;
use retrograd_judge::{JudgeStrategy, RulerConfig};
use retrograd_spec::judge::{LocalJudgeConfig, LocalJudgeWeights};

fn config() -> TrainConfig {
    TrainConfig {
        n_ctx: 512,
        n_batch: 128,
        n_ubatch: 64,
        n_seq_max: 4,
        epochs: 1,
        learning_rate: 1.0e-3,
        device: Device::Cpu,
        generation_concurrency: 2,
        ..TrainConfig::default()
    }
}

fn sampling() -> SamplingParams {
    SamplingParams {
        temperature: 0.01,
        top_p: 1.0,
        max_new_tokens: 16,
        seed: 7,
    }
}

fn group() -> TrajectoryGroup {
    let member = |answer: &str| Trajectory {
        scenario_id: "s".into(),
        messages: vec![
            Message::text(Role::User, "Say ok"),
            Message::text(Role::Assistant, answer),
        ],
        tokens: vec![1, 2],
        old_logprobs: vec![-1.0],
        train_mask: vec![false, true],
        steps: Vec::new(),
        reward: None,
        truncated: false,
        metadata: Default::default(),
        provenance: None,
    };
    TrajectoryGroup {
        group_id: 1,
        scenario_id: "s".into(),
        trajectories: vec![member("ok"), member("no")],
    }
}

fn local_ruler(strategy: JudgeStrategy) -> RulerConfig {
    RulerConfig {
        local: Some(LocalJudgeConfig {
            weights: LocalJudgeWeights::Base,
            refresh_every: None,
            snapshot_dir: None,
            max_new_tokens: Some(128),
        }),
        strategy,
        max_retries: 0,
        ..Default::default()
    }
}

/// The grammar-constrained v2 generation the local transport uses for a
/// structured verdict: the output must be a complete JSON object, which the
/// unconstrained v1 path does not guarantee on this fixture.
#[test]
fn grammar_constrained_v2_generation_emits_a_json_object() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, config()).expect("load model");
    let rendered = trainer
        .format_chat_messages(
            &serde_json::json!([{"role": "user", "content": "Answer with a JSON object."}])
                .to_string(),
            None,
            true,
        )
        .expect("render");
    let prompt = trainer.tokenize_text(&rendered).expect("tokenize");
    let mut params = SamplingParamsV2::from(sampling());
    params.max_new_tokens = 64;
    params.grammar_gbnf = Some(
        retrograd::trainer::grammar_from_json_schema(
            r#"{"type":"object","properties":{"score":{"type":"number"}},"required":["score"],"additionalProperties":false}"#,
        )
        .expect("grammar")
        .into(),
    );
    let rows = trainer
        .generate_sampled_continuous_v2(&[(&prompt, params)])
        .expect("grammar-constrained generation");
    assert_eq!(rows.len(), 1);
    let text = trainer
        .detokenize(&rows[0].tokens, false)
        .expect("detokenize");
    let parsed: serde_json::Value =
        serde_json::from_str(text.trim()).expect("the grammar-constrained output is JSON");
    assert!(
        parsed.get("score").is_some(),
        "the schema's field is there: {text}"
    );

    // The listwise verdict schema, the one the local judge actually constrains
    // its generations with: the output must carry `scores`, nothing else.
    let listwise = retrograd_judge::render::prompt::listwise_schema();
    let mut params = SamplingParamsV2::from(sampling());
    params.max_new_tokens = 128;
    params.grammar_gbnf = Some(
        retrograd::trainer::grammar_from_json_schema(&listwise.to_string())
            .expect("listwise grammar")
            .into(),
    );
    let rows = trainer
        .generate_sampled_continuous_v2(&[(&prompt, params)])
        .expect("listwise-constrained generation");
    let text = trainer
        .detokenize(&rows[0].tokens, false)
        .expect("detokenize");
    let parsed: serde_json::Value =
        serde_json::from_str(text.trim()).expect("the listwise-constrained output is JSON");
    assert!(
        parsed
            .get("scores")
            .and_then(|scores| scores.as_array())
            .is_some(),
        "the verdict shape is there: {text}"
    );
}

/// The pointwise scoring primitive: the two label suffixes branch off one
/// decoded prefix, and each branch's logprob is exactly what the single
/// suffix scorer returns for the same target.
#[test]
fn branched_label_scoring_matches_the_single_suffix_scorer() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let mut trainer = Trainer::new(&model, config()).expect("load model");
    let prompt = trainer
        .tokenize_text("Is this answer correct? Answer yes or no.")
        .expect("tokenize");
    let yes = trainer.tokenize_text(" yes").expect("tokenize label");
    let no = trainer.tokenize_text(" no").expect("tokenize label");
    let (yes_id, no_id) = match (yes.as_slice(), no.as_slice()) {
        ([a], [b]) => (*a, *b),
        _ => {
            eprintln!("skipping: the fixture tokenizes a label to more than one token");
            return;
        }
    };
    let mut yes_tokens = prompt.clone();
    yes_tokens.push(yes_id);
    let mut no_tokens = prompt.clone();
    no_tokens.push(no_id);
    let branched = trainer
        .score_token_suffix_batch(&[
            (yes_tokens.as_slice(), prompt.len()),
            (no_tokens.as_slice(), prompt.len()),
        ])
        .expect("branched scoring");
    let single_yes = trainer
        .score_token_suffix(&yes_tokens, prompt.len())
        .expect("single scoring");
    let single_no = trainer
        .score_token_suffix(&no_tokens, prompt.len())
        .expect("single scoring");
    assert_eq!(branched[0][0], single_yes[0]);
    assert_eq!(branched[1][0], single_no[0]);
}

/// A local judge scores a real group on the fixture, in generation mode
/// (grammar-constrained listwise) and in pointwise mode.
#[test]
fn a_local_judge_scores_a_group_on_the_cpu_fixture() {
    let _guard = serialize_models();
    let Some(model) = common::model_path_if_available() else {
        eprintln!("skipping: no local test model");
        return;
    };
    let training = config();
    let mut trainer = Trainer::new(&model, training.clone()).expect("load model");
    // `weights = "base"` disables and re-enables the adapter around each
    // judge call, which requires one to exist.
    let mut lora = retrograd::LoraConfig::qv(2, 4.0);
    lora.seed = 7;
    trainer.create_lora(&lora).expect("create the adapter");

    for strategy in [JudgeStrategy::Listwise, JudgeStrategy::PointwiseLogprob] {
        let mut judge = LocalJudge::new(
            local_ruler(strategy),
            JudgeWeights::Base,
            sampling(),
            128,
            2,
            &model,
            &training,
            None,
            20,
            None,
        )
        .expect("build the local judge");
        let mut groups = vec![group()];
        let metrics = judge
            .score_groups(
                &mut trainer,
                &mut groups,
                JudgeFailurePolicy::DropGroup,
                1.0,
                false,
            )
            .expect("score the group");
        assert_eq!(
            metrics.counts.scored_groups, 1,
            "{strategy:?}; last error: {:?}; dropped: {}",
            metrics.last_error, metrics.counts.dropped_groups
        );
        let rewards: Vec<Option<f32>> = groups[0]
            .trajectories
            .iter()
            .map(|member| member.reward)
            .collect();
        assert_eq!(rewards.len(), 2);
        for reward in &rewards {
            let value = reward.expect("a scored member");
            assert!(
                (0.0..=1.0).contains(&value),
                "verdict in [0,1]: {value} ({strategy:?})"
            );
        }
    }
}
