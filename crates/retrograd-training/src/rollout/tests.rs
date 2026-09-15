//! Unit tests for the rollout building blocks. Kept in one module: most of
//! them cross two of the sibling files (weights packed into a batch, a sampled
//! rollout scored then stepped), so splitting them per file would only move
//! the imports around.

use super::packing::*;
use super::prompts::*;
use super::reward::*;
use super::sampling::*;
use super::step::*;
use super::weights::*;
use retrograd_core::{RewardMode, RewardProtocol, SharedPrefixFanout};

/// Builds the four objective values by name so call sites state which value is
/// the clip range, KL coefficient, or loss denominator.
fn objective(
    clip_range_low: f32,
    clip_range_high: f32,
    kl_coefficient: f32,
    loss_denominator: usize,
) -> GrpoObjective {
    GrpoObjective {
        clip_range_low,
        clip_range_high,
        kl_coefficient,
        loss_denominator,
    }
}

#[test]
fn evenly_spaced_subset_keeps_ends_and_spread() {
    let values: Vec<usize> = (0..45).collect();
    let subset = evenly_spaced_subset(values.clone(), 5);
    assert_eq!(subset, vec![0, 11, 22, 33, 44]);
    assert_eq!(evenly_spaced_subset(values.clone(), 45), values);
    assert_eq!(evenly_spaced_subset(values.clone(), 100), values);
    assert_eq!(evenly_spaced_subset(values.clone(), 1), vec![0]);
    assert_eq!(evenly_spaced_subset(values, 0), (0..45).collect::<Vec<_>>());
    assert_eq!(
        evenly_spaced_subset(Vec::<usize>::new(), 3),
        Vec::<usize>::new()
    );
}

fn temp_prompts(name: &str, source: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "retrograd-prompts-{name}-{}.jsonl",
        std::process::id()
    ));
    std::fs::write(&path, source).unwrap();
    path
}

fn shell(script: &str) -> Vec<String> {
    vec!["/bin/sh".into(), "-c".into(), script.into()]
}

#[test]
fn prompt_reader_accepts_strict_non_empty_jsonl() {
    let path = temp_prompts(
        "valid",
        concat!(
            "{\"messages\":[{\"role\":\"system\",\"content\":\"concise\"},{\"role\":\"user\",\"content\":\"first\"}]}\n",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"second\"}]}\n"
        ),
    );
    let prompts = read_prompts(&path).unwrap();
    assert_eq!(
        prompts.iter().map(Prompt::reward_text).collect::<Vec<_>>(),
        ["first", "second"]
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn eval_prompt_reader_detaches_the_assistant_reference() {
    let path = temp_prompts(
        "eval-reference",
        "{\"messages\":[{\"role\":\"system\",\"content\":\"concise\"},{\"role\":\"user\",\"content\":\"2+2\"},{\"role\":\"assistant\",\"content\":\"4\"}]}\n",
    );
    let prompts = read_eval_prompts(&path).unwrap();
    assert_eq!(prompts.len(), 1);
    assert_eq!(prompts[0].reward_text(), "2+2");
    assert_eq!(prompts[0].reference(), Some("4"));
    assert_eq!(prompts[0].conversation.messages.len(), 2);
    assert_eq!(
        prompts[0].conversation.messages.last().unwrap().role,
        "user"
    );
    std::fs::remove_file(path).unwrap();
}

/// A training line may carry a `rubric` - criteria for the judge - where it may
/// not carry an assistant turn. The two are not the same thing: one is a target
/// the policy would be trained on, the other is only ever read by the judge.
#[test]
fn a_training_prompt_carries_its_rubric_into_the_judged_group() {
    let path = temp_prompts(
        "rubric",
        "{\"messages\":[{\"role\":\"system\",\"content\":\"concise\"},{\"role\":\"user\",\"content\":\"list the ports\"}],\"rubric\":\"expected: ss -tulpn\"}\n{\"messages\":[{\"role\":\"user\",\"content\":\"no rubric\"}],\"rubric\":\"   \"}\n",
    );
    let prompts = read_prompts(&path).unwrap();
    assert_eq!(prompts[0].rubric(), Some("expected: ss -tulpn"));
    // Blank is not a rubric: it would replace the run-wide one with nothing.
    assert_eq!(prompts[1].rubric(), None);

    let completions = ["ss -tulpn".to_string(), "netstat -tulpn".to_string()];
    let group = super::judge::trajectory_group(&super::JudgeGroup {
        group_id: 3,
        prompt: &prompts[0],
        completions: &completions,
    })
    .unwrap();
    assert_eq!(group.group_id, 3);
    assert_eq!(group.trajectories.len(), 2);
    // The prompt is shared and the completion is what differs - which is what
    // lets the judge render the group's context once.
    let first = &group.trajectories[0];
    assert_eq!(first.messages.len(), 3);
    assert_eq!(first.messages[2].content, "ss -tulpn");
    assert_eq!(group.trajectories[1].messages[2].content, "netstat -tulpn");
    assert_eq!(
        first
            .metadata
            .get("rubric")
            .and_then(|value| value.as_str()),
        Some("expected: ss -tulpn")
    );
    std::fs::remove_file(path).unwrap();
}

#[test]
fn prompt_reader_reports_record_level_errors() {
    for (name, source, expected) in [
        ("empty-file", "", "contains no JSONL records"),
        (
            "blank-line",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}]}\n\n",
            ":2: empty JSONL record",
        ),
        (
            "empty-prompt",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"\"}]}\n",
            ":1: message content must not be empty",
        ),
        (
            "unknown-field",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"ok\"}],\"extra\":1}\n",
            ":1: invalid JSON",
        ),
        (
            "assistant-last",
            "{\"messages\":[{\"role\":\"user\",\"content\":\"ok\"},{\"role\":\"assistant\",\"content\":\"done\"}]}\n",
            ":1: prompt messages must end with a user message",
        ),
    ] {
        let path = temp_prompts(name, source);
        let error = read_prompts(&path).unwrap_err();
        assert!(error.to_string().contains(expected), "{name}: {error}");
        std::fs::remove_file(path).unwrap();
    }
}

/// A reward worker in the shape the persistent mode asks for: the handshake,
/// then `body`. `sh` flushes on every `printf`, which is exactly what the
/// protocol requires of a worker in any language.
fn reward_worker_with_setup(setup: &str, body: &str) -> retrograd_judge::RewardProcess {
    let command = shell(&format!(
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
done"#,
        retrograd_judge::REWARD_PROTOCOL_VERSION,
    ));
    reward_process(&command, RewardProtocol::default()).unwrap()
}

fn reward_worker(body: &str) -> retrograd_judge::RewardProcess {
    reward_worker_with_setup("", body)
}

/// The same worker, one batch, no handshake - what `reward_mode = "oneshot"`
/// spells and what every reward command spoke before the persistent mode.
fn one_shot_reward(body: &str) -> retrograd_judge::RewardProcess {
    reward_process(
        &shell(body),
        RewardProtocol {
            mode: RewardMode::OneShot,
            ..RewardProtocol::default()
        },
    )
    .unwrap()
}

#[test]
fn reward_command_round_trips_one_finite_score_per_rollout() {
    let mut process = reward_worker(r#"reply '{"reward":1.5}'"#);
    assert_eq!(
        score(&mut process, [("p1", "c1"), ("p2", "c2")]).unwrap(),
        [1.5, 1.5]
    );
    // The second batch is the point of the persistent mode: same process, same
    // contract, and the count is per call rather than per run.
    assert_eq!(score(&mut process, [("p3", "c3")]).unwrap(), [1.5]);

    let mut one_shot =
        one_shot_reward(r#"while IFS= read -r line; do printf '{"reward":1.5}\n'; done"#);
    assert_eq!(
        score(&mut one_shot, [("p1", "c1"), ("p2", "c2")]).unwrap(),
        [1.5, 1.5]
    );
}

#[test]
fn evaluation_reward_request_includes_the_optional_reference() {
    let mut process = reward_worker(
        r#"case "$line" in *'"reference":"4"'*) reply '{"reward":2.0}';; *) exit 9;; esac"#,
    );
    let rewards = rows_with_references(&mut process, [("2+2", "3", Some("4"))]).unwrap();
    assert_eq!(rewards[0].reward, 2.0);
}

/// `judge_weight` is how a reward process says how much of a verdict this one
/// completion should carry. Only the GRPO path has a verdict to weigh, so the
/// scalar path refuses the field instead of dropping it silently.
#[test]
fn a_judge_weight_reaches_the_grpo_path_and_stops_the_others() {
    let mut process = reward_worker(r#"reply '{"reward":0.5,"judge_weight":0.3}'"#);
    let rows = score_rows(&mut process, [("p", "c")]).unwrap();
    assert_eq!(rows[0].reward, 0.5);
    assert_eq!(rows[0].judge_weight, Some(0.3));

    let error = score(&mut process, [("p", "c")]).unwrap_err();
    assert!(error.to_string().contains("judge_weight"), "{error}");

    let mut negative = reward_worker(r#"reply '{"reward":0.5,"judge_weight":-1}'"#);
    let error = score_rows(&mut negative, [("p", "c")]).unwrap_err();
    assert!(error.to_string().contains("non-negative"), "{error}");
    assert!(error.is_user_error(), "{error}");
}

/// The two refusals this layer owns - a reward that is not a number the
/// gradient can carry - and the transport failures it forwards, both classed
/// as what the caller wired up rather than as a broken machine.
#[test]
fn reward_command_surfaces_process_and_protocol_failures() {
    let mut exits = one_shot_reward("printf 'reward failed\\n' >&2; exit 7");
    let error = score(&mut exits, [("p", "c")]).unwrap_err();
    assert!(error.to_string().contains("reward failed"), "{error}");
    assert!(!error.is_user_error(), "{error}");

    let mut short = one_shot_reward(r#"cat > /dev/null; printf '{"reward":1}\n'"#);
    let error = score(&mut short, [("p1", "c1"), ("p2", "c2")]).unwrap_err();
    assert!(
        error.to_string().contains("1 responses for 2 rollouts"),
        "{error}"
    );

    let mut unknown_field = reward_worker(r#"reply '{"reward":1,"extra":true}'"#);
    let error = score(&mut unknown_field, [("p", "c")]).unwrap_err();
    assert!(error.to_string().contains("reward response 1"), "{error}");
    assert!(error.is_user_error(), "{error}");

    // Beyond f32::MAX: serde_json parses it into an infinity rather than
    // failing, so the finiteness check is the only thing standing between it
    // and an advantage.
    let mut infinite = reward_worker(r#"reply '{"reward":1e39}'"#);
    let error = score(&mut infinite, [("p", "c")]).unwrap_err();
    assert!(error.to_string().contains("is not finite"), "{error}");
    assert!(error.is_user_error(), "{error}");
}

/// A command written for the one-shot protocol fails on the handshake with the
/// line of TOML that fixes it, instead of hanging until the batch deadline.
#[test]
fn a_command_that_cannot_be_persistent_says_which_setting_to_flip() {
    let mut process = reward_process(
        &shell(r#"body=$(cat); printf '{"reward":1}\n'"#),
        RewardProtocol {
            mode: RewardMode::Persistent,
            timeout: std::time::Duration::from_millis(300),
        },
    )
    .unwrap();
    let error = score(&mut process, [("p", "c")]).unwrap_err();
    assert!(error.to_string().contains("handshake"), "{error}");
    assert!(
        error.to_string().contains("reward_mode = \"oneshot\""),
        "{error}"
    );
    assert!(error.is_user_error(), "{error}");
}

#[test]
fn mean_and_standard_deviation_handle_empty_and_extreme_inputs() {
    assert_eq!(mean_std(std::iter::empty()), (0.0, 0.0, 0));
    let (mean, std, count) = mean_std([f32::MAX, -f32::MAX].into_iter());
    assert_eq!(mean, 0.0);
    assert!(std.is_finite() && std >= f32::MAX as f64);
    assert_eq!(count, 2);
}

#[test]
fn unclipped_weight_is_advantage_times_ratio() {
    // ratio = e^0.1, inside the clip band; no KL term.
    let (weights, stats) = ppo_token_weights(&[2.0], &[-1.0], &[-0.9], 0.2, 0.0);
    let ratio = 0.1_f32.exp();
    assert!((weights[0] - 2.0 * ratio).abs() < 1e-5);
    assert_eq!(stats.clip_fraction, 0.0);
}

#[test]
fn clipping_zeroes_the_surrogate_gradient() {
    // Positive advantage with ratio e^1 > 1.2: the clip binds, gradient 0.
    let (weights, stats) = ppo_token_weights(&[1.0], &[-2.0], &[-1.0], 0.2, 0.0);
    assert_eq!(weights[0], 0.0);
    assert_eq!(stats.clip_fraction, 1.0);

    // Negative advantage with ratio e^-1 < 0.8 also binds.
    let (weights, stats) = ppo_token_weights(&[-1.0], &[-1.0], &[-2.0], 0.2, 0.0);
    assert_eq!(weights[0], 0.0);
    assert_eq!(stats.clip_fraction, 1.0);

    // Negative advantage with a large ratio is NOT clipped (min picks it).
    let (weights, _) = ppo_token_weights(&[-1.0], &[-2.0], &[-1.0], 0.2, 0.0);
    assert!(weights[0] < 0.0);
}

#[test]
fn sampling_waves_keep_groups_whole_and_cover_every_member() {
    // Capacity is a multiple of the group size: one group per wave.
    assert_eq!(
        continuous_waves(&[2, 2], 2),
        vec![vec![(0, 0), (0, 1)], vec![(1, 0), (1, 1)]]
    );
    // Ragged capacity: the third group would straddle, so it starts a wave of
    // its own instead of paying for its prompt in both.
    let waves = continuous_waves(&[2, 2, 2], 5);
    assert_eq!(
        waves,
        vec![vec![(0, 0), (0, 1), (1, 0), (1, 1)], vec![(2, 0), (2, 1)],]
    );
    // A group wider than the capacity is the one case that splits.
    assert_eq!(
        continuous_waves(&[3], 2),
        vec![vec![(0, 0), (0, 1)], vec![(0, 2)]]
    );
    // Whatever the shape: every member exactly once, in order, and no wave
    // above the capacity.
    for capacity in 1..=9 {
        let sizes = [4, 1, 3, 8];
        let waves = continuous_waves(&sizes, capacity);
        assert!(waves.iter().all(|wave| wave.len() <= capacity));
        let flat = waves.concat();
        let mut expected = Vec::new();
        for (group, &size) in sizes.iter().enumerate() {
            expected.extend((0..size).map(|member| (group, member)));
        }
        let mut sorted = flat.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, expected, "capacity {capacity}");
        // Per group, members stay in increasing order: that is what lets the
        // caller reassemble a split group by pushing in wave order.
        for (group, &size) in sizes.iter().enumerate() {
            let members = flat
                .iter()
                .filter(|(candidate, _)| *candidate == group)
                .map(|(_, member)| *member)
                .collect::<Vec<_>>();
            assert_eq!(
                members,
                (0..size).collect::<Vec<_>>(),
                "capacity {capacity}"
            );
        }
    }
}

#[test]
fn dual_clip_bounds_a_negative_advantage_whose_ratio_grew() {
    // The quadrant no clip band covers: A < 0 with a ratio the policy pushed
    // *up*. `min` picks the unclipped branch there, so without the dual clip
    // the weight is A*r and grows without bound.
    let log_ratio = (DUAL_CLIP_C + 1.0).ln();
    let (weights, stats) = ppo_token_weights(&[-1.0], &[-1.0], &[-1.0 + log_ratio], 0.2, 0.0);
    assert_eq!(weights[0], 0.0, "beyond c the objective is flat");
    assert_eq!(stats.clip_fraction, 1.0);
    // The reported objective saturates at c*A rather than r*A.
    assert!(
        (stats.surrogate_loss - DUAL_CLIP_C).abs() < 1e-5,
        "{stats:?}"
    );

    // Just inside c the token still trains, and with its exact ratio.
    let inside = (DUAL_CLIP_C - 0.5).ln();
    let (weights, stats) = ppo_token_weights(&[-1.0], &[-1.0], &[-1.0 + inside], 0.2, 0.0);
    assert!(
        (weights[0] + (DUAL_CLIP_C - 0.5)).abs() < 1e-5,
        "{weights:?}"
    );
    assert_eq!(stats.clip_fraction, 0.0);

    // Same bound on the GRPO objective.
    let (weights, _) = grpo_token_weights(
        -2.0,
        &[-1.0],
        &[-1.0 + log_ratio],
        &[-1.0],
        &objective(0.2, 0.28, 0.0, 1),
    );
    assert_eq!(weights[0], 0.0);
}

#[test]
fn the_k3_coefficient_stays_bounded_when_the_policy_diverges() {
    // A reference 30 nats above the policy makes the unclamped k3 coefficient
    // enormous; the configured bound must keep it from dominating every token.
    let (weights, stats) = grpo_token_weights(
        0.0,
        &[-1.0],
        &[-31.0],
        &[-1.0],
        &objective(0.2, 0.28, 1.0, 1),
    );
    let limit = REFERENCE_LOG_RATIO_LIMIT.exp();
    assert!((weights[0] - (limit - 1.0)).abs() < 1e-2, "{weights:?}");
    assert!(
        (stats.kl - (limit - REFERENCE_LOG_RATIO_LIMIT - 1.0)).abs() < 1e-2,
        "{stats:?}"
    );
    // PPO's k3 anchor differs but its coefficient is bounded the same way.
    let (weights, _) = ppo_token_weights(&[0.0], &[-31.0], &[-1.0], 0.2, 1.0);
    assert!((weights[0] - (1.0 - limit)).abs() < 1e-2, "{weights:?}");
}

#[test]
fn divergence_is_refused_only_once_the_trust_region_is_gone() {
    // Healthy epoch: a small KL and a handful of clipped tokens.
    assert!(check_policy_divergence(1, 1, 0.01, 0.05, 1.2).is_ok());
    // A fully saturated clip, and a KL past the bound, each stop the run.
    let error = check_policy_divergence(3, 2, 0.01, 0.99, 4.0).unwrap_err();
    assert!(error.to_string().contains("clip band"), "{error}");
    let error = check_policy_divergence(3, 2, 1.0e6, 0.1, 1.0e4).unwrap_err();
    assert!(error.to_string().contains("k3 KL"), "{error}");
    assert!(
        check_policy_divergence(1, 1, f32::NAN, 0.0, 1.0).is_err(),
        "a non-finite KL is divergence, not a healthy epoch"
    );
    // The message has to name the geometry that causes it in practice.
    let error = check_policy_divergence(1, 1, 0.0, 1.0, 1.0).unwrap_err();
    assert!(
        error.to_string().contains("training.gradient_accumulation"),
        "{error}"
    );
}

#[test]
fn generation_room_refuses_a_prompt_that_fills_the_window() {
    let layout = RowLayout {
        row_width: 32,
        window: 16,
        pad_token: 0,
        batch: 16,
        ubatch: 16,
        n_seq_max: 1,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 1,
    };
    assert_eq!(generation_room(&layout, 4).unwrap(), 12);
    assert!(generation_room(&layout, 16).is_err());
    assert!(
        generation_room(&layout, 17).is_err(),
        "no wrapping subtraction"
    );
}

#[test]
fn kl_weight_pushes_the_policy_back_toward_the_rollout() {
    // ratio > 1: the k3 gradient k*(1-r) is negative (push logprob down).
    let (weights, stats) = ppo_token_weights(&[0.0], &[-1.0], &[-0.5], 0.2, 0.5);
    assert!(weights[0] < 0.0, "{weights:?}");
    assert!(stats.kl > 0.0);
    // ratio < 1: positive (pull logprob back up); k3 stays non-negative.
    let (weights, stats) = ppo_token_weights(&[0.0], &[-0.5], &[-1.0], 0.2, 0.5);
    assert!(weights[0] > 0.0, "{weights:?}");
    assert!(stats.kl > 0.0);
    // ratio = 1: no penalty at all.
    let (weights, stats) = ppo_token_weights(&[0.0], &[-0.7], &[-0.7], 0.2, 0.5);
    assert_eq!(weights[0], 0.0);
    assert_eq!(stats.kl, 0.0);
}

#[test]
fn grpo_kl_uses_the_fixed_reference_not_the_rollout_policy() {
    // The current and rollout policies match, but the fixed reference is
    // lower: the KL term must still push the current logprob down.
    let (weights, stats) =
        grpo_token_weights(0.0, &[-0.5], &[-0.5], &[-1.0], &objective(0.2, 0.2, 0.5, 1));
    assert!(weights[0] < 0.0, "{weights:?}");
    assert!(stats.kl > 0.0);

    // At the reference policy the fixed-reference KL is exactly zero,
    // even when the rollout policy differs.
    let (weights, stats) =
        grpo_token_weights(0.0, &[-0.5], &[-1.0], &[-1.0], &objective(0.2, 0.2, 0.5, 1));
    assert_eq!(weights[0], 0.0);
    assert_eq!(stats.kl, 0.0);
}

#[test]
fn intermediate_returns_can_redistribute_advantage_per_token() {
    let mut weights = Vec::new();
    let stats = grpo_token_weights_into(
        &mut weights,
        0.0,
        Some(&[2.0, -1.0]),
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &objective(0.2, 0.28, 0.0, 2),
    );
    assert_eq!(weights, [2.0, -1.0]);
    assert!((stats.surrogate_loss + 0.5).abs() < 1e-6);
}

#[test]
fn clip_higher_decouples_the_upper_and_lower_ranges() {
    // ratio = e^0.2 ~ 1.221: outside a symmetric 0.2 band, inside the
    // Clip-Higher upper range 0.28 - the promotion must survive.
    let ratio = 0.2_f32.exp();
    let (weights, stats) = grpo_token_weights(
        1.0,
        &[-1.0],
        &[-0.8],
        &[-1.0],
        &objective(0.2, 0.28, 0.0, 1),
    );
    assert!((weights[0] - ratio).abs() < 1e-5, "{weights:?}");
    assert_eq!(stats.clip_fraction, 0.0);

    // Beyond 1 + 0.28 the upper clip still binds.
    let (weights, stats) = grpo_token_weights(
        1.0,
        &[-1.0],
        &[-0.6],
        &[-1.0],
        &objective(0.2, 0.28, 0.0, 1),
    );
    assert_eq!(weights[0], 0.0);
    assert_eq!(stats.clip_fraction, 1.0);

    // The lower range is unchanged: negative advantage with
    // ratio e^-0.25 < 0.8 is clipped out regardless of the upper range.
    let (weights, stats) = grpo_token_weights(
        -1.0,
        &[-1.0],
        &[-1.25],
        &[-1.0],
        &objective(0.2, 0.28, 0.0, 1),
    );
    assert_eq!(weights[0], 0.0);
    assert_eq!(stats.clip_fraction, 1.0);
}

#[test]
fn pack_row_masks_prompt_and_padding() {
    // Sequence: prompt [10, 11], completion [12, 13]; n_ctx 6, pad 99.
    let (tokens, labels, weights) = pack_row(
        &[10, 11, 12, 13],
        &[false, false, true, true],
        &[0.5, -0.25],
        6,
        99,
    )
    .unwrap();
    assert_eq!(tokens, vec![10, 11, 12, 13, 99, 99]);
    // Position 1 predicts token 12, position 2 predicts token 13.
    assert_eq!(labels, vec![-1, 12, 13, -1, -1, -1]);
    assert_eq!(weights, vec![0.0, 0.5, -0.25, 0.0, 0.0, 0.0]);
}

#[test]
fn pack_row_rejects_overflow_and_shape_mismatch() {
    assert!(pack_row(&[1, 2, 3], &[false, true, true], &[1.0, 1.0], 2, 0).is_err());
    assert!(pack_row(&[1, 2, 3], &[false, true, true], &[1.0], 4, 0).is_err());
}

#[test]
fn pack_row_supports_disjoint_policy_segments() {
    let (_, labels, weights) = pack_row(
        &[10, 11, 12, 13, 14, 15],
        &[false, false, true, false, false, true],
        &[0.5, -0.25],
        8,
        99,
    )
    .unwrap();
    assert_eq!(labels, [-1, 12, -1, -1, 15, -1, -1, -1]);
    assert_eq!(weights, [0.0, 0.5, 0.0, 0.0, -0.25, 0.0, 0.0, 0.0]);
}

#[test]
fn reusable_row_clears_only_the_previous_live_ranges() {
    let layout = RowLayout {
        row_width: 8,
        window: 8,
        pad_token: 99,
        batch: 4,
        ubatch: 2,
        n_seq_max: 1,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 2,
    };
    let mut scratch = WeightedStepScratch::new(&layout);
    scratch.begin(1);
    scratch.token_weights.extend([1.0, 2.0, 3.0]);
    scratch
        .pack_row(
            0,
            &Rollout {
                tokens: vec![10, 11, 12, 13, 14],
                train_mask: vec![false, false, true, true, true],
                old_logprobs: Vec::new(),
            },
            &layout,
            3,
        )
        .unwrap();

    scratch.begin(1);
    scratch.token_weights.clear();
    scratch.token_weights.push(4.0);
    scratch
        .pack_row(
            0,
            &Rollout {
                tokens: vec![20, 21, 22],
                train_mask: vec![false, false, true],
                old_logprobs: Vec::new(),
            },
            &layout,
            1,
        )
        .unwrap();

    assert_eq!(scratch.batch.tokens, [20, 21, 22, 99, 99, 99, 99, 99]);
    assert_eq!(scratch.batch.labels, [-1, 22, -1, -1, -1, -1, -1, -1]);
    assert_eq!(
        scratch.batch.weights,
        [0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
}

#[test]
fn multi_row_scratch_isolates_slots_and_survives_resizing() {
    let layout = RowLayout {
        row_width: 4,
        window: 4,
        pad_token: 99,
        batch: 4,
        ubatch: 2,
        n_seq_max: 1,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 1,
    };
    let short = Rollout {
        tokens: vec![20, 21],
        train_mask: vec![false, true],
        old_logprobs: Vec::new(),
    };
    let long = Rollout {
        tokens: vec![10, 11, 12, 13],
        train_mask: vec![false, false, true, true],
        old_logprobs: Vec::new(),
    };
    let mut scratch = WeightedStepScratch::new(&layout);
    scratch.begin(2);
    scratch.token_weights.clear();
    scratch.token_weights.extend([1.0, 1.0]);
    scratch.pack_row(0, &long, &layout, 2).unwrap();
    scratch.token_weights.clear();
    scratch.token_weights.push(1.0);
    scratch.pack_row(1, &short, &layout, 1).unwrap();
    assert_eq!(scratch.batch.n_rows, 2);
    assert_eq!(scratch.batch.tokens, [10, 11, 12, 13, 20, 21, 99, 99]);
    assert_eq!(scratch.batch.labels, [-1, 12, 13, -1, 21, -1, -1, -1]);

    // Shrinking drops the second slot; growing back restores clean rows.
    scratch.begin(1);
    assert_eq!(scratch.batch.tokens.len(), 4);
    scratch.begin(2);
    assert_eq!(scratch.batch.tokens[4..], [99, 99, 99, 99]);
    assert_eq!(scratch.batch.labels[4..], [-1, -1, -1, -1]);
    assert_eq!(scratch.packed_len[1], 0);
}

#[test]
fn chunks_fill_one_accumulation_period_of_real_evals() {
    // Rollout evals: 0 → 2, 1 → 1, 2 → 6 (over budget, own chunk), 3 → 1.
    let evals = vec![2, 1, 6, 1];
    let order = vec![0, 1, 2, 3];
    let chunks = chunk_by_period(&order, &evals, 4);
    assert_eq!(chunks, vec![0..2, 2..3, 3..4]);
    // A single over-budget rollout still trains, alone.
    assert_eq!(chunk_by_period(&[2], &evals, 4), vec![0..1]);
    assert!(chunk_by_period(&[], &evals, 4).is_empty());
}

#[test]
fn shared_prefix_chunks_keep_groups_together_and_split_at_capacity() {
    let rollout = || Rollout {
        tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
        train_mask: vec![false, false, false, false, false, true, true, true],
        old_logprobs: vec![-1.0; 3],
    };
    let rollouts = vec![rollout(), rollout(), rollout(), rollout()];
    let layout = RowLayout {
        row_width: 12,
        window: 12,
        pad_token: 0,
        batch: 12,
        ubatch: 12,
        n_seq_max: 4,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 1,
    };
    // Shared prefix costs four tokens and each member three: two members
    // consume ten real tokens plus two unused-sequence padding slots.
    let chunks =
        grouped_optimizer_chunks(&[2, 0, 3, 1], &[7, 7, 9, 9], &rollouts, &[1; 4], &layout)
            .unwrap();
    assert_eq!(chunks, vec![vec![2, 3], vec![0, 1]]);

    let short = || Rollout {
        tokens: vec![1, 2, 3],
        train_mask: vec![false, false, true],
        old_logprobs: vec![-1.0],
    };
    let short_rollouts = vec![short(), short(), short(), short()];
    let chunks = grouped_optimizer_chunks(
        &[2, 0, 3, 1],
        &[7, 7, 9, 9],
        &short_rollouts,
        &[1; 4],
        &layout,
    )
    .unwrap();
    assert_eq!(chunks, vec![vec![2, 3], vec![0, 1]]);
}

#[test]
fn packed_sequences_share_each_prompt_and_isolate_groups() {
    let a0 = Rollout {
        tokens: vec![1, 2, 3, 4],
        train_mask: vec![false, false, true, true],
        old_logprobs: vec![-1.0; 2],
    };
    let a1 = a0.clone();
    let b0 = Rollout {
        tokens: vec![5, 6, 7],
        train_mask: vec![false, false, true],
        old_logprobs: vec![-1.0],
    };
    let chunk = [
        ChunkMember {
            group_id: 10,
            rollout: &a0,
            advantage: 0.0,
            token_advantages: None,
            reference_logprobs: &[],
        },
        ChunkMember {
            group_id: 10,
            rollout: &a1,
            advantage: 0.0,
            token_advantages: None,
            reference_logprobs: &[],
        },
        ChunkMember {
            group_id: 20,
            rollout: &b0,
            advantage: 0.0,
            token_advantages: None,
            reference_logprobs: &[],
        },
    ];
    let layout = RowLayout {
        row_width: 10,
        window: 10,
        pad_token: 99,
        batch: 10,
        ubatch: 10,
        n_seq_max: 4,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 1,
    };
    let mut scratch = WeightedStepScratch::new(&layout);
    scratch.member_weights = vec![vec![1.0; 2], vec![1.0; 2], vec![1.0]];
    assert!(matches!(
        scratch.pack_sequences(&chunk, 0, &layout, 5).unwrap(),
        PackOutcome::Packed { .. }
    ));
    let batch = &scratch.packed_sequence_batch;

    assert_eq!(batch.tokens, [1, 2, 3, 2, 3, 5, 6, 99, 99, 99]);
    assert_eq!(batch.labels, [-1, 3, 4, 3, 4, -1, 7, -1, -1, -1]);
    assert_eq!(batch.positions, [0, 1, 2, 1, 2, 0, 1, 0, 3, 4]);
    let memberships = batch
        .seq_offsets
        .windows(2)
        .map(|range| batch.seq_ids[range[0]..range[1]].to_vec())
        .collect::<Vec<_>>();
    assert_eq!(
        memberships,
        vec![
            vec![0, 1],
            vec![0],
            vec![0],
            vec![1],
            vec![1],
            vec![2],
            vec![2],
            vec![3],
            vec![0],
            vec![0],
        ]
    );
}

#[test]
fn packed_sequences_stay_continuous_when_the_last_token_is_not_trained() {
    // A multi-turn trajectory ending on an untrained tool observation: the
    // tail padding must continue sequence 0 from the last position it
    // actually owns, not from `tokens.len() - 2`.
    let rollout = Rollout {
        tokens: vec![1, 2, 3, 4, 5],
        train_mask: vec![false, false, true, true, false],
        old_logprobs: vec![-1.0; 2],
    };
    let chunk = [ChunkMember {
        group_id: 1,
        rollout: &rollout,
        advantage: 0.0,
        token_advantages: None,
        reference_logprobs: &[],
    }];
    let layout = RowLayout {
        row_width: 10,
        window: 10,
        pad_token: 99,
        batch: 10,
        ubatch: 10,
        n_seq_max: 2,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 1,
    };
    let mut scratch = WeightedStepScratch::new(&layout);
    scratch.member_weights = vec![vec![1.0; 2]];
    assert!(matches!(
        scratch.pack_sequences(&chunk, 0, &layout, 5).unwrap(),
        PackOutcome::Packed { .. }
    ));
    let batch = &scratch.packed_sequence_batch;

    let mut sequence_zero = batch
        .seq_offsets
        .windows(2)
        .enumerate()
        .filter(|(_, range)| batch.seq_ids[range[0]..range[1]].contains(&0))
        .map(|(token, _)| batch.positions[token])
        .collect::<Vec<_>>();
    sequence_zero.sort_unstable();
    sequence_zero.dedup();
    assert_eq!(
        sequence_zero,
        (0..sequence_zero.len() as i32).collect::<Vec<_>>(),
        "positions {:?}",
        batch.positions
    );
}

#[test]
fn the_scheduler_horizon_shrinks_to_the_steps_actually_taken() {
    // Nominal: 4 updates × 8 slots. Grouping makes each update cost 2.
    let mut horizon = SchedulerHorizon::new(32, 0);
    assert_eq!(horizon.steps(), 32);
    horizon.observe(2, 3, 2);
    assert_eq!(horizon.steps(), 8);
    horizon.observe(2, 2, 4);
    assert_eq!(horizon.steps(), 8);
    horizon.observe(2, 1, 6);
    horizon.observe(2, 0, 8);
    // The last update ends exactly at the horizon: the decay reached zero.
    assert_eq!(horizon.steps(), 9);
}

#[test]
fn the_scheduler_horizon_never_exceeds_its_bound_nor_undercuts_the_warmup() {
    let mut horizon = SchedulerHorizon::new(10, 6);
    // An update costing more than the nominal bound cannot raise it.
    horizon.observe(50, 4, 50);
    assert_eq!(horizon.steps(), 10);
    // And a run that finishes early keeps the warm-up length reachable.
    let mut short = SchedulerHorizon::new(100, 40);
    short.observe(1, 0, 1);
    assert_eq!(short.steps(), 40);
}

#[test]
fn rollout_evals_round_the_last_label_up_to_a_ubatch() {
    let layout = RowLayout {
        row_width: 16,
        window: 16,
        pad_token: 0,
        batch: 8,
        ubatch: 4,
        n_seq_max: 1,
        shared_prefix_fanout: SharedPrefixFanout::Auto,
        steps_per_row: 2,
    };
    // Last trainable target index 5 → label at 4 → ubatches 0..=1 → 2.
    let rollout = Rollout {
        tokens: vec![1; 6],
        train_mask: vec![false, false, true, true, true, true],
        old_logprobs: Vec::new(),
    };
    assert_eq!(layout.rollout_evals(&rollout).unwrap(), 2);
    // A label in the first ubatch costs exactly one eval.
    let tiny = Rollout {
        tokens: vec![1, 2],
        train_mask: vec![false, true],
        old_logprobs: Vec::new(),
    };
    assert_eq!(layout.rollout_evals(&tiny).unwrap(), 1);
    assert_eq!(layout.accumulation_period(), 2);
}

#[test]
fn runtime_weight_normalization_keeps_clipped_tokens_in_the_denominator() {
    // One logical batch has two ubatches. The first has two live weights,
    // the second has one live and one clipped token. After rescaling, the
    // runtime's mean-of-ubatch-means equals sum(original_weights) / 4.
    let labels = vec![1, 2, 3, 4];
    let mut weights = vec![2.0, 4.0, 0.0, 8.0];
    normalize_runtime_weights(&labels, &mut weights, 4, 4, 2).unwrap();
    let runtime_reduction = ((weights[0] + weights[1]) / 2.0 + weights[3]) / 2.0;
    assert!((runtime_reduction - 14.0 / 4.0).abs() < 1e-6, "{weights:?}");
}

#[test]
fn runtime_weight_normalization_rejects_invalid_geometry() {
    for (labels, mut weights, denominator, batch, ubatch) in [
        (vec![1], vec![], 1, 1, 1),
        (vec![1], vec![1.0], 0, 1, 1),
        (vec![1], vec![1.0], 1, 0, 1),
        (vec![1], vec![1.0], 1, 1, 0),
        (vec![1, 2, 3], vec![1.0; 3], 3, 3, 2),
    ] {
        assert!(
            normalize_runtime_weights(&labels, &mut weights, denominator, batch, ubatch).is_err()
        );
    }
}

#[test]
fn dr_grpo_uses_a_constant_generation_budget_denominator() {
    let (short_weights, short_stats) =
        grpo_token_weights(1.0, &[-1.0], &[-1.0], &[-1.0], &objective(0.2, 0.2, 0.0, 4));
    let (long_weights, long_stats) = grpo_token_weights(
        1.0,
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &[-1.0, -1.0],
        &objective(0.2, 0.2, 0.0, 4),
    );
    assert_eq!(short_weights, vec![1.0]);
    assert_eq!(long_weights, vec![1.0, 1.0]);
    assert!((short_stats.surrogate_loss + 0.25).abs() < 1e-6);
    assert!((long_stats.surrogate_loss + 0.5).abs() < 1e-6);

    let labels = vec![1, 2, -1, -1];
    let mut short_runtime = vec![1.0, 0.0, 0.0, 0.0];
    normalize_runtime_weights(&labels, &mut short_runtime, 4, 4, 2).unwrap();
    let short_reduction = short_runtime[0] / 2.0;
    assert!((short_reduction - 0.25).abs() < 1e-6);
}

#[allow(clippy::too_many_arguments)]
fn exact_grpo_loss(
    new: f32,
    old: f32,
    reference: f32,
    advantage: f32,
    clip_range_low: f32,
    clip_range_high: f32,
    beta: f32,
) -> f32 {
    let ratio = (new - old).exp();
    let clipped = ratio.clamp(1.0 - clip_range_low, 1.0 + clip_range_high);
    let surrogate = -(advantage * ratio).min(advantage * clipped);
    let reference_log_ratio = reference - new;
    let kl = reference_log_ratio.exp() - reference_log_ratio - 1.0;
    surrogate + beta * kl
}

#[test]
fn grpo_weight_matches_the_numeric_objective_derivative() {
    for (advantage, new) in [(0.7_f32, -0.95_f32), (-0.7, -1.05), (0.7, -0.5)] {
        let old = -1.0;
        let reference = -1.2;
        let beta = 0.3;
        let epsilon = 1.0e-3;
        let (low, high) = (0.15_f32, 0.28_f32);
        let (weights, _) = grpo_token_weights(
            advantage,
            &[old],
            &[new],
            &[reference],
            &objective(low, high, beta, 1),
        );
        let numeric = (exact_grpo_loss(new + epsilon, old, reference, advantage, low, high, beta)
            - exact_grpo_loss(new - epsilon, old, reference, advantage, low, high, beta))
            / (2.0 * epsilon);
        assert!(
            (numeric + weights[0]).abs() < 2.0e-3,
            "{numeric} {weights:?}"
        );
    }
}
