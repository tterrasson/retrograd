//! Judge prompt construction and response parsing.
//!
//! Two comparison shapes are supported. A *listwise* prompt shows several
//! trajectories at once and asks for one score each: cheap (one request per
//! group) but its discrimination degrades as the list grows and the model has to
//! hold every candidate in view. A *pairwise* prompt shows exactly two and asks
//! which is better: far more reliable per judgement, at `O(n)`-to-`O(n²)`
//! requests, and it needs an aggregation step to become the scalar reward GRPO
//! consumes.

use serde::Deserialize;

use retrograd_agent_core::trajectory::TrajectoryGroup;
use retrograd_agent_core::{Error, Result};

use super::{self as context, JudgeContext, JudgeMessage, RenderStats};
use crate::Score;

pub const DEFAULT_RUBRIC: &str = "Rank each trajectory relative to the others. Score goal achievement, efficiency, proportional granularity, and partial credit. Return one score in [0,1] per trajectory.";

pub const DEFAULT_PAIRWISE_RUBRIC: &str = "Two trajectories attempted the same task. Decide which one is better on goal achievement, efficiency, and proportional granularity. Answer 'a', 'b', or 'tie'.";

/// One trajectory as it will appear in a prompt, with the cost that chunk
/// planning needs.
#[derive(Clone, Debug)]
pub struct RenderedTrajectory {
    pub id: usize,
    pub messages: Vec<JudgeMessage>,
    /// What the environment reports about the terminal state - typically the
    /// diff produced on the workspace. For a code task this, and not the
    /// dialogue, is what the judge should be reading.
    pub env_summary: Option<String>,
    pub chars: usize,
}

/// The shared opening of a group plus one rendering per trajectory. The shared
/// prefix is sent once; each trajectory carries only what makes it different.
#[derive(Clone, Debug)]
pub struct RenderedGroup {
    /// Half of the presentation-permutation seed; see [`presentation_order`].
    pub group_id: u64,
    pub context: Vec<JudgeMessage>,
    pub trajectories: Vec<RenderedTrajectory>,
    /// Scenario-specific rubric, when the scenario declared one.
    pub rubric: Option<String>,
    pub stats: RenderStats,
}

/// Scenario rubric carried by the group, if every member agrees on it.
///
/// `Trajectory::metadata` is a copy of `Scenario::metadata`, so all members of a
/// group normally carry the same rubric; a group whose members disagree is not a
/// group, and falling back to the global rubric is the honest answer.
fn group_rubric(group: &TrajectoryGroup) -> Option<String> {
    let first = group
        .trajectories
        .first()?
        .metadata
        .get("rubric")?
        .as_str()?
        .to_owned();
    group
        .trajectories
        .iter()
        .all(|trajectory| {
            trajectory
                .metadata
                .get("rubric")
                .and_then(|value| value.as_str())
                == Some(&first)
        })
        .then_some(first)
}

/// Environment summary a trajectory carries, written by the rollout engine into
/// `metadata.env_state.summary`.
fn env_summary(trajectory: &retrograd_agent_core::trajectory::Trajectory) -> Option<String> {
    let summary = trajectory
        .metadata
        .get("env_state")?
        .get("summary")?
        .as_str()?
        .trim();
    (!summary.is_empty()).then(|| summary.to_owned())
}

pub fn render_group(group: &TrajectoryGroup, budget: &JudgeContext) -> Result<RenderedGroup> {
    if group.trajectories.is_empty() {
        return Err(Error::invalid("judge group must not be empty"));
    }
    let rows = group
        .trajectories
        .iter()
        .map(|trajectory| trajectory.messages.as_slice())
        .collect::<Vec<_>>();
    let prefix_len = context::common_prefix_len(&rows);
    let (shared, mut stats) =
        context::render_messages(&group.trajectories[0].messages[..prefix_len], budget);
    let mut trajectories = Vec::with_capacity(group.trajectories.len());
    for (id, trajectory) in group.trajectories.iter().enumerate() {
        let (messages, trajectory_stats) =
            context::render_messages(&trajectory.messages[prefix_len..], budget);
        stats.merge_from(trajectory_stats);
        // The summary goes through the same per-message cap as everything else:
        // an unbounded `git diff` would otherwise walk straight past the budget
        // the rest of the pipeline exists to enforce.
        let env_summary = budget
            .include_env_state
            .then(|| env_summary(trajectory))
            .flatten()
            .map(|summary| {
                let (content, elided) =
                    context::elide(&summary, budget.max_message_chars, budget.head_ratio);
                stats.rendered_chars += content.chars().count();
                stats.elided_chars += elided;
                content
            });
        let chars = messages
            .iter()
            .map(|message| message.content.chars().count())
            .sum::<usize>()
            + env_summary
                .as_ref()
                .map_or(0, |summary| summary.chars().count());
        trajectories.push(RenderedTrajectory {
            id,
            messages,
            env_summary,
            chars,
        });
    }
    Ok(RenderedGroup {
        group_id: group.group_id,
        context: shared,
        trajectories,
        rubric: group_rubric(group),
        stats,
    })
}

/// Cost every request pays regardless of how many trajectories it carries.
pub fn request_overhead(rubric: &str, rendered: &RenderedGroup) -> usize {
    rubric.chars().count()
        + rendered
            .context
            .iter()
            .map(|message| message.content.chars().count())
            .sum::<usize>()
        + 256
}

fn envelope(rubric: &str, context: &[JudgeMessage], body: &serde_json::Value) -> Result<String> {
    Ok(format!(
        "{rubric}\n\n<context>\n{}\n</context>\n\n<trajectories>\n{}\n</trajectories>",
        serde_json::to_string(context)
            .map_err(|error| Error::Reward(format!("serialize judge context: {error}")))?,
        serde_json::to_string(body)
            .map_err(|error| Error::Reward(format!("serialize judge trajectories: {error}")))?
    ))
}

/// Listwise prompt over `selection`, a subset of the rendered group. Prompt ids
/// are positional within the selection, so a chunk always asks for ids
/// `0..selection.len()`; the caller maps them back to trajectory indices.
pub fn listwise_prompt(
    rubric: &str,
    rendered: &RenderedGroup,
    selection: &[usize],
) -> Result<String> {
    if selection.len() < 2 {
        return Err(Error::invalid(
            "a listwise judge request needs at least two trajectories",
        ));
    }
    let body = selection
        .iter()
        .enumerate()
        .map(|(position, &index)| trajectory_body(rendered, index, Some(position)))
        .collect::<Vec<_>>();
    envelope(rubric, &rendered.context, &serde_json::json!(body))
}

/// One trajectory as the judge sees it. The environment summary is a sibling
/// field of the messages rather than an extra message: it is not something the
/// policy said, and presenting it as one would invite the judge to read it as a
/// turn the trajectory earned.
fn trajectory_body(rendered: &RenderedGroup, index: usize, id: Option<usize>) -> serde_json::Value {
    let trajectory = &rendered.trajectories[index];
    let mut body = serde_json::Map::new();
    if let Some(id) = id {
        body.insert("trajectory_id".into(), serde_json::json!(id));
    }
    body.insert("messages".into(), serde_json::json!(trajectory.messages));
    if let Some(summary) = &trajectory.env_summary {
        body.insert("final_state".into(), serde_json::json!(summary));
    }
    serde_json::Value::Object(body)
}

pub fn pairwise_prompt(
    rubric: &str,
    rendered: &RenderedGroup,
    left: usize,
    right: usize,
) -> Result<String> {
    let body = serde_json::json!({
        "a": trajectory_body(rendered, left, None),
        "b": trajectory_body(rendered, right, None),
    });
    envelope(rubric, &rendered.context, &body)
}

/// Deterministic presentation permutation for one request.
///
/// Position bias is the first bias of an LLM-as-a-judge: the same trajectory
/// scores differently depending on where it sits in the list. Shuffling the
/// presentation spreads that bias over the group instead of letting it track
/// rollout order, which is arbitrary but *stable* across updates - the worst
/// case, since a systematic offset on member 0 survives the GRPO group baseline.
///
/// The permutation is derived from `(group_id, request_index)` and nothing else,
/// so it is reproducible from a run's seed and the response cache keeps hitting.
pub fn presentation_order(group_id: u64, request_index: u64, len: usize) -> Vec<usize> {
    let mut order = (0..len).collect::<Vec<_>>();
    let mut state = group_id
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(request_index.wrapping_mul(0xbf58_476d_1ce4_e5b9))
        .wrapping_add(0x94d0_49bb_1331_11eb);
    // Fisher-Yates over a splitmix64 stream: uniform, and it needs no rng crate.
    for index in (1..len).rev() {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut draw = state;
        draw = (draw ^ (draw >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        draw = (draw ^ (draw >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        draw ^= draw >> 31;
        order.swap(index, (draw % (index as u64 + 1)) as usize);
    }
    order
}

pub fn listwise_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["scores"],
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["trajectory_id", "explanation", "score"],
                    "properties": {
                        "trajectory_id": {"type": "integer", "minimum": 0},
                        "explanation": {"type": "string"},
                        "score": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    })
}

pub fn pairwise_schema() -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false,
        "required": ["winner", "explanation"],
        "properties": {
            "winner": {"type": "string", "enum": ["a", "b", "tie"]},
            "explanation": {"type": "string"}
        }
    })
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListwiseResponse {
    scores: Vec<ListwiseScore>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ListwiseScore {
    trajectory_id: usize,
    explanation: String,
    score: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Winner {
    A,
    B,
    Tie,
}

/// Unknown fields are tolerated here, unlike the listwise response: a verdict is
/// one enum and one sentence, and a judge that volunteers a `analysis` or
/// `reasoning` field alongside them has still answered the question. Rejecting
/// it costs the whole group - a member left without a comparison is unscored,
/// and `apply_group_scores` then drops every member it had. The listwise schema
/// keeps `deny_unknown_fields` because there a stray field usually means the
/// model scored something other than the trajectories it was given.
#[derive(Deserialize)]
struct PairwiseResponse {
    winner: Winner,
    explanation: String,
}

/// Extracts the JSON object from a model response that may be wrapped in prose
/// or a Markdown fence.
pub(super) fn json_body(content: &str) -> Result<&str> {
    let trimmed = content.trim();
    if trimmed.starts_with('{') && trimmed.ends_with('}') {
        return Ok(trimmed);
    }
    let start = trimmed
        .find('{')
        .ok_or_else(|| Error::Reward("judge content contains no JSON object".into()))?;
    let end = trimmed
        .rfind('}')
        .ok_or_else(|| Error::Reward("judge content contains no complete JSON object".into()))?;
    if end <= start {
        return Err(Error::Reward(
            "judge content contains no complete JSON object".into(),
        ));
    }
    Ok(&trimmed[start..=end])
}

pub fn parse_listwise(content: &str, expected: usize) -> Result<Vec<Score>> {
    let response: ListwiseResponse = serde_json::from_str(json_body(content)?)
        .map_err(|error| Error::Reward(format!("invalid judge score JSON: {error}")))?;
    if response.scores.len() != expected {
        return Err(Error::Reward(format!(
            "judge returned {} scores for {expected} trajectories",
            response.scores.len()
        )));
    }
    let mut output = vec![None; expected];
    for score in response.scores {
        if score.trajectory_id >= expected || output[score.trajectory_id].is_some() {
            return Err(Error::Reward(
                "judge trajectory ids must cover the request exactly once".into(),
            ));
        }
        if !score.score.is_finite() || !(0.0..=1.0).contains(&score.score) {
            return Err(Error::Reward(
                "judge score must be finite and in [0,1]".into(),
            ));
        }
        output[score.trajectory_id] = Some(Score {
            value: score.score,
            valid: true,
            explanation: Some(score.explanation),
            error: None,
        });
    }
    output
        .into_iter()
        .map(|score| score.ok_or_else(|| Error::Reward("judge score id is missing".into())))
        .collect()
}

pub fn parse_pairwise(content: &str) -> Result<(Winner, String)> {
    let response: PairwiseResponse = serde_json::from_str(json_body(content)?)
        .map_err(|error| Error::Reward(format!("invalid judge verdict JSON: {error}")))?;
    Ok((response.winner, response.explanation))
}

/// Deterministic comparison schedule for a group of `size` trajectories.
///
/// The ring `(0,1), (1,2), … (n-1,0)` comes first: it is the cheapest schedule
/// that gives every trajectory at least one comparison, which is what the
/// aggregation below requires. Remaining pairs follow in lexicographic order, so
/// raising `max_pairs` only ever *adds* comparisons to an existing schedule and
/// never reshuffles the ones already cached.
pub fn pair_schedule(size: usize, max_pairs: Option<usize>) -> Vec<(usize, usize)> {
    if size < 2 {
        return Vec::new();
    }
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    if size > 2 {
        for index in 0..size {
            pairs.push((index, (index + 1) % size));
        }
    } else {
        pairs.push((0, 1));
    }
    for left in 0..size {
        for right in (left + 1)..size {
            if !pairs.contains(&(left, right)) && !pairs.contains(&(right, left)) {
                pairs.push((left, right));
            }
        }
    }
    // Fewer than `size` comparisons would leave a trajectory unrewarded, so the
    // floor is the ring itself.
    let limit = max_pairs.map_or(pairs.len(), |max| max.max(size.min(pairs.len())));
    pairs.truncate(limit);
    pairs
}

/// One completed comparison.
#[derive(Clone, Debug, PartialEq)]
pub struct PairOutcome {
    pub left: usize,
    pub right: usize,
    pub winner: Winner,
    pub explanation: String,
}

/// Turns comparisons into one score per trajectory: the win rate, with ties
/// counting as half a win. Win rate keeps scores in `[0,1]` and is invariant to
/// how many comparisons a trajectory happened to get, which matters because a
/// truncated schedule does not give everyone the same number.
///
/// A trajectory with no completed comparison cannot be scored. Rather than
/// inventing a neutral 0.5 - which would hand GRPO a fabricated baseline - it is
/// reported invalid and the caller's failure policy decides.
pub fn scores_from_pairs(size: usize, outcomes: &[PairOutcome]) -> Vec<Score> {
    let mut wins = vec![0.0_f32; size];
    let mut matches = vec![0_u32; size];
    let mut notes: Vec<Vec<&str>> = vec![Vec::new(); size];
    for outcome in outcomes {
        if outcome.left >= size || outcome.right >= size {
            continue;
        }
        matches[outcome.left] += 1;
        matches[outcome.right] += 1;
        match outcome.winner {
            Winner::A => wins[outcome.left] += 1.0,
            Winner::B => wins[outcome.right] += 1.0,
            Winner::Tie => {
                wins[outcome.left] += 0.5;
                wins[outcome.right] += 0.5;
            }
        }
        notes[outcome.left].push(&outcome.explanation);
        notes[outcome.right].push(&outcome.explanation);
    }
    (0..size)
        .map(|index| {
            if matches[index] == 0 {
                return Score {
                    value: 0.0,
                    valid: false,
                    explanation: None,
                    error: Some("no completed pairwise comparison".into()),
                };
            }
            Score {
                value: wins[index] / matches[index] as f32,
                valid: true,
                explanation: Some(notes[index].join(" | ")),
                error: None,
            }
        })
        .collect()
}

pub use retrograd_spec::judge::Aggregation;

/// Bradley-Terry strengths from the same comparisons, projected into `[0,1]`.
///
/// Maximises `Σ w_ij · log σ(θ_i − θ_j)` by gradient ascent - a concave problem
/// once the small ridge term is added, so a fixed iteration count converges from
/// any start and no line search is needed. The ridge also pins the otherwise
/// free additive constant.
///
/// The absolute scale is meaningless (GRPO centres each group anyway), so the
/// strengths are min-max projected; a group whose members are indistinguishable
/// collapses to a constant 0.5, which is the honest reading of "no signal" and
/// is what [`crate::JudgeBatchMetrics::degenerate_group_fraction`] counts.
pub fn bradley_terry_scores(size: usize, outcomes: &[PairOutcome]) -> Vec<Score> {
    let mut wins = vec![vec![0.0_f64; size]; size];
    let mut matches = vec![0_u32; size];
    let mut notes: Vec<Vec<&str>> = vec![Vec::new(); size];
    for outcome in outcomes {
        if outcome.left >= size || outcome.right >= size || outcome.left == outcome.right {
            continue;
        }
        matches[outcome.left] += 1;
        matches[outcome.right] += 1;
        match outcome.winner {
            Winner::A => wins[outcome.left][outcome.right] += 1.0,
            Winner::B => wins[outcome.right][outcome.left] += 1.0,
            Winner::Tie => {
                wins[outcome.left][outcome.right] += 0.5;
                wins[outcome.right][outcome.left] += 0.5;
            }
        }
        notes[outcome.left].push(&outcome.explanation);
        notes[outcome.right].push(&outcome.explanation);
    }

    let mut theta = vec![0.0_f64; size];
    let ridge = 1e-3;
    for _ in 0..200 {
        let mut gradient = vec![0.0_f64; size];
        for i in 0..size {
            for j in 0..size {
                if i == j || (wins[i][j] == 0.0 && wins[j][i] == 0.0) {
                    continue;
                }
                let probability = 1.0 / (1.0 + (theta[j] - theta[i]).exp());
                gradient[i] += wins[i][j] - (wins[i][j] + wins[j][i]) * probability;
            }
            gradient[i] -= ridge * theta[i];
        }
        for (value, step) in theta.iter_mut().zip(&gradient) {
            *value += 0.1 * step;
        }
    }

    let scored = (0..size)
        .filter(|&index| matches[index] > 0)
        .collect::<Vec<_>>();
    let low = scored
        .iter()
        .map(|&index| theta[index])
        .fold(f64::INFINITY, f64::min);
    let high = scored
        .iter()
        .map(|&index| theta[index])
        .fold(f64::NEG_INFINITY, f64::max);
    let span = high - low;
    (0..size)
        .map(|index| {
            if matches[index] == 0 {
                return Score {
                    value: 0.0,
                    valid: false,
                    explanation: None,
                    error: Some("no completed pairwise comparison".into()),
                };
            }
            Score {
                value: if span > 1e-9 {
                    ((theta[index] - low) / span) as f32
                } else {
                    0.5
                },
                valid: true,
                explanation: Some(notes[index].join(" | ")),
                error: None,
            }
        })
        .collect()
}

/// Combines the two verdicts on one pair judged in both presentation orders.
///
/// A judge that answers "a" in both directions is answering about the position,
/// not the trajectory. Only an agreement counts as a verdict; a disagreement is
/// recorded as a tie - the one reading that adds no signal in either direction,
/// and reported through `judge/position_disagreement`.
pub fn combine_orders(forward: Winner, swapped: Winner) -> (Winner, bool) {
    // `swapped` was asked with the trajectories exchanged, so its answer is read
    // back through the same exchange.
    let swapped = match swapped {
        Winner::A => Winner::B,
        Winner::B => Winner::A,
        Winner::Tie => Winner::Tie,
    };
    if forward == swapped {
        (forward, false)
    } else {
        (Winner::Tie, true)
    }
}

/// Realigns chunk scores using a trajectory judged in every chunk.
///
/// Independent requests have no common scale: a chunk of weak trajectories gets
/// generous scores, a chunk of strong ones harsh scores, and GRPO would read the
/// difference as signal. Shifting each chunk so the shared anchor lands on the
/// same value removes that offset. Only the offset is corrected - rescaling
/// would need at least two shared trajectories per chunk, which costs another
/// comparison slot per chunk and is left to `Pairwise`.
pub fn align_to_anchor(scores: &mut [f32], anchor_value: f32, anchor_reference: f32) {
    let shift = anchor_reference - anchor_value;
    if shift == 0.0 {
        return;
    }
    for score in scores.iter_mut() {
        *score = (*score + shift).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::trajectory::{Message, Role, Trajectory};

    fn trajectory(answer: &str) -> Trajectory {
        Trajectory {
            scenario_id: "s".into(),
            messages: vec![
                Message::text(Role::System, "common"),
                Message::text(Role::User, "question"),
                Message::text(Role::Assistant, answer),
            ],
            tokens: vec![1, 2],
            old_logprobs: vec![-1.0],
            train_mask: vec![false, true],
            steps: vec![],
            reward: None,
            truncated: false,
            metadata: Default::default(),
            provenance: None,
        }
    }

    fn group(answers: &[&str]) -> TrajectoryGroup {
        TrajectoryGroup {
            group_id: 1,
            scenario_id: "s".into(),
            trajectories: answers.iter().map(|answer| trajectory(answer)).collect(),
        }
    }

    /// A `{"scores":[...]}` listwise judge response, built from
    /// `(trajectory_id, explanation, score)` triples.
    fn scores_json(entries: &[(usize, &str, f64)]) -> String {
        let scores: Vec<_> = entries
            .iter()
            .map(|&(trajectory_id, explanation, score)| {
                serde_json::json!({
                    "trajectory_id": trajectory_id,
                    "explanation": explanation,
                    "score": score,
                })
            })
            .collect();
        serde_json::json!({ "scores": scores }).to_string()
    }

    #[test]
    fn rendering_sends_the_shared_prefix_once() {
        let rendered = render_group(&group(&["a", "b"]), &JudgeContext::default()).unwrap();
        assert_eq!(rendered.context.len(), 2);
        assert_eq!(rendered.trajectories.len(), 2);
        assert_eq!(rendered.trajectories[0].messages.len(), 1);
        let prompt = listwise_prompt("rubric", &rendered, &[0, 1]).unwrap();
        assert_eq!(prompt.matches("common").count(), 1);
        assert_eq!(prompt.matches("question").count(), 1);
        assert!(prompt.contains("\"a\"") && prompt.contains("\"b\""));
    }

    #[test]
    fn the_scenario_rubric_and_the_environment_state_reach_the_prompt() {
        let mut group = group(&["a", "b"]);
        for (index, trajectory) in group.trajectories.iter_mut().enumerate() {
            trajectory
                .metadata
                .insert("rubric".into(), serde_json::json!("prefer fewer edits"));
            trajectory.metadata.insert(
                "env_state".into(),
                serde_json::json!({"summary": format!("diff-{index}")}),
            );
        }
        let rendered = render_group(&group, &JudgeContext::default()).unwrap();
        assert_eq!(rendered.rubric.as_deref(), Some("prefer fewer edits"));
        let prompt = listwise_prompt("rubric", &rendered, &[0, 1]).unwrap();
        assert!(
            prompt.contains("diff-0") && prompt.contains("diff-1"),
            "{prompt}"
        );
        // The state is a sibling of the messages, not an extra turn: the judge
        // must not read it as something the policy said.
        assert!(prompt.contains("\"final_state\""));
        // A pairwise prompt shows it too - that mode is where a code task
        // benefits most from grading the result rather than the dialogue.
        assert!(
            pairwise_prompt("rubric", &rendered, 0, 1)
                .unwrap()
                .contains("diff-1")
        );

        // Members that disagree on the rubric are not a group; fall back to the
        // run-wide one rather than picking a member's.
        group.trajectories[1]
            .metadata
            .insert("rubric".into(), serde_json::json!("other"));
        assert!(
            render_group(&group, &JudgeContext::default())
                .unwrap()
                .rubric
                .is_none()
        );

        // Switched off, the state does not reach the prompt at all.
        let without = render_group(
            &group,
            &JudgeContext {
                include_env_state: false,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(
            without
                .trajectories
                .iter()
                .all(|trajectory| trajectory.env_summary.is_none())
        );
    }

    #[test]
    fn a_chunk_prompt_renumbers_ids_from_zero() {
        let rendered =
            render_group(&group(&["a", "b", "c", "d"]), &JudgeContext::default()).unwrap();
        let prompt = listwise_prompt("rubric", &rendered, &[2, 3]).unwrap();
        assert!(prompt.contains("\"trajectory_id\":0"));
        assert!(prompt.contains("\"trajectory_id\":1"));
        assert!(!prompt.contains("\"trajectory_id\":2"));
        assert!(prompt.contains("\"c\"") && prompt.contains("\"d\"") && !prompt.contains("\"a\""));
    }

    #[test]
    fn a_single_trajectory_request_is_refused() {
        let rendered = render_group(&group(&["a", "b"]), &JudgeContext::default()).unwrap();
        assert!(listwise_prompt("rubric", &rendered, &[0]).is_err());
    }

    #[test]
    fn listwise_parsing_reorders_ids_and_rejects_out_of_range_scores() {
        let scores = parse_listwise(&scores_json(&[(1, "b", 0.8), (0, "a", 0.2)]), 2).unwrap();
        assert_eq!(scores[0].value, 0.2);
        assert_eq!(scores[1].value, 0.8);
        assert!(parse_listwise(&scores_json(&[(0, "a", 2.0)]), 1).is_err());
        assert!(parse_listwise(&scores_json(&[(0, "a", 0.5), (0, "a", 0.5)]), 2).is_err());
    }

    #[test]
    fn pairwise_parsing_accepts_fenced_json_and_every_verdict() {
        let (winner, explanation) = parse_pairwise(
            "Here you go:\n```json\n{\"winner\":\"b\",\"explanation\":\"clearer\"}\n```",
        )
        .unwrap();
        assert_eq!(winner, Winner::B);
        assert_eq!(explanation, "clearer");
        assert_eq!(
            parse_pairwise(r#"{"winner":"tie","explanation":"same"}"#)
                .unwrap()
                .0,
            Winner::Tie
        );
        assert!(parse_pairwise(r#"{"winner":"c","explanation":"?"}"#).is_err());
    }

    #[test]
    fn a_pairwise_verdict_survives_fields_the_judge_added_itself() {
        // What an endpoint that renders `response_format` into prompt text
        // produced once the schema stopped being sent: the verdict is there,
        // wrapped in the model's own commentary fields.
        let (winner, explanation) = parse_pairwise(
            r#"{"analysis":{"a":"fails"},"winner":"b","explanation":"less broken","reasoning":"…"}"#,
        )
        .unwrap();
        assert_eq!(winner, Winner::B);
        assert_eq!(explanation, "less broken");
        // A missing verdict is still a failure: tolerance is about extra fields,
        // not about inventing an answer.
        assert!(parse_pairwise(r#"{"decision":"b","reasoning":"…"}"#).is_err());
    }

    #[test]
    fn the_pair_schedule_covers_everyone_and_only_grows() {
        let ring = pair_schedule(4, Some(4));
        assert_eq!(ring, vec![(0, 1), (1, 2), (2, 3), (3, 0)]);
        for index in 0..4 {
            assert!(ring.iter().any(|&(l, r)| l == index || r == index));
        }
        let full = pair_schedule(4, None);
        assert_eq!(full.len(), 6);
        assert!(full.starts_with(&ring));
        // A budget below the ring still yields full coverage.
        assert_eq!(pair_schedule(4, Some(1)).len(), 4);
        assert_eq!(pair_schedule(2, None), vec![(0, 1)]);
        assert!(pair_schedule(1, None).is_empty());
    }

    #[test]
    fn win_rate_aggregation_handles_ties_and_missing_comparisons() {
        let outcomes = vec![
            PairOutcome {
                left: 0,
                right: 1,
                winner: Winner::A,
                explanation: "0 beats 1".into(),
            },
            PairOutcome {
                left: 1,
                right: 2,
                winner: Winner::Tie,
                explanation: "1 ties 2".into(),
            },
        ];
        let scores = scores_from_pairs(4, &outcomes);
        assert_eq!(scores[0].value, 1.0);
        assert_eq!(scores[1].value, 0.25); // 0 + 0.5 over two matches
        assert_eq!(scores[2].value, 0.5);
        assert!(!scores[3].valid);
        assert!(scores[3].error.is_some());
        assert!(scores[0].explanation.as_deref().unwrap().contains("beats"));
    }

    #[test]
    fn the_presentation_order_is_a_permutation_and_depends_only_on_its_seed() {
        for len in 0..10_usize {
            for (group_id, request) in [(0, 0), (1, 0), (1, 1), (u64::MAX, 7)] {
                let order = presentation_order(group_id, request, len);
                let mut sorted = order.clone();
                sorted.sort_unstable();
                assert_eq!(sorted, (0..len).collect::<Vec<_>>(), "{order:?}");
                // Reproducible: the cache key is the prompt, and the prompt
                // carries this order.
                assert_eq!(order, presentation_order(group_id, request, len));
            }
        }
        // Two requests of the same group, and two groups, get different orders,
        // otherwise position bias would still track rollout order.
        assert_ne!(
            presentation_order(1, 0, 8),
            presentation_order(1, 1, 8),
            "chunks of one group share a presentation order"
        );
        assert_ne!(presentation_order(1, 0, 8), presentation_order(2, 0, 8));
        // It does shuffle: an 8-element identity would be one draw in 40320.
        assert_ne!(presentation_order(1, 0, 8), (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn a_swapped_verdict_is_read_back_swapped_and_a_contradiction_is_a_tie() {
        // The judge preferred the trajectory shown first in both directions:
        // that is an answer about the position, not about the trajectory.
        assert_eq!(combine_orders(Winner::A, Winner::A), (Winner::Tie, true));
        // Consistent: "a" wins, and after the exchange it is "b" that wins.
        assert_eq!(combine_orders(Winner::A, Winner::B), (Winner::A, false));
        assert_eq!(combine_orders(Winner::B, Winner::A), (Winner::B, false));
        assert_eq!(
            combine_orders(Winner::Tie, Winner::Tie),
            (Winner::Tie, false)
        );
        assert_eq!(combine_orders(Winner::Tie, Winner::A), (Winner::Tie, true));
    }

    #[test]
    fn bradley_terry_reads_a_truncated_schedule_that_win_rate_cannot() {
        // Ring schedule over four, each trajectory judged twice: 0 beat 1, 1
        // beat 2, 2 beat 3, and 3 lost to 0. Every one of them is 1-1 except the
        // extremes, so win rate calls 1 and 2 exactly equal - even though 1 beat
        // the trajectory that beat 3, and 2 only beat 3.
        let win = |left: usize, right: usize| PairOutcome {
            left,
            right,
            winner: Winner::A,
            explanation: format!("{left}>{right}"),
        };
        let outcomes = vec![win(0, 1), win(1, 2), win(2, 3), win(0, 3)];
        let rates = scores_from_pairs(4, &outcomes);
        assert_eq!(rates[1].value, rates[2].value);

        let strengths = bradley_terry_scores(4, &outcomes);
        assert!(strengths.iter().all(|score| score.valid));
        assert!(
            strengths[1].value > strengths[2].value,
            "beating a stronger opponent must count for more: {strengths:?}"
        );
        // 0 won both of its matches, 3 lost both: the projection pins them.
        assert!((strengths[0].value - 1.0).abs() < 1e-5, "{strengths:?}");
        assert!((strengths[3].value - 0.0).abs() < 1e-5, "{strengths:?}");
        assert!(
            strengths
                .iter()
                .all(|score| (0.0..=1.0).contains(&score.value))
        );
    }

    #[test]
    fn bradley_terry_reports_the_unjudged_and_collapses_the_indistinguishable() {
        // No comparison at all: inventing a neutral 0.5 would hand GRPO a
        // fabricated baseline, so the score is reported invalid instead.
        let scores = bradley_terry_scores(
            3,
            &[PairOutcome {
                left: 0,
                right: 1,
                winner: Winner::Tie,
                explanation: "same".into(),
            }],
        );
        assert!(!scores[2].valid);
        // Two trajectories nothing distinguishes: a constant, which is what
        // `degenerate_group_fraction` is there to count.
        assert_eq!(scores[0].value, 0.5);
        assert_eq!(scores[1].value, 0.5);
    }

    #[test]
    fn anchor_alignment_removes_the_chunk_offset_and_stays_in_range() {
        let mut scores = [0.2, 0.4, 0.6];
        align_to_anchor(&mut scores, 0.2, 0.5);
        for (value, expected) in scores.iter().zip([0.5, 0.7, 0.9]) {
            assert!((value - expected).abs() < 1e-6, "{scores:?}");
        }
        let mut clamped = [0.9, 1.0];
        align_to_anchor(&mut clamped, 0.5, 1.0);
        assert_eq!(clamped, [1.0, 1.0]);
    }
}
