//! Context budgeting for judge prompts.
//!
//! A judge request carries whole trajectories, and a trajectory can be as long
//! as `max_trajectory_tokens` allows. Multiplied by a group size, the naive
//! prompt overruns the judge's own context window, which surfaces as an API
//! error and - under `DropGroup` - as a silently discarded update. Everything
//! here bounds that prompt *before* it is sent, and does so deterministically:
//! the same trajectories must always produce the same prompt, otherwise the
//! response cache and run reproducibility both break.
//!
//! Budgets are counted in characters rather than tokens on purpose. The judge
//! is a remote model whose tokenizer is unknown to us; characters are the only
//! measure available on both sides, and a conservative character budget bounds
//! the token count for any tokenizer.

use serde::Serialize;

use retrograd_agent_core::trajectory::Message;

pub mod compact;
pub mod prompt;

pub use retrograd_spec::judge::JudgeContext;

/// A message as the judge sees it: the role, the (possibly elided) content, and
/// whether it reported a tool failure. Tool errors are part of what
/// distinguishes a good trajectory from a lucky one, so the flag is kept
/// explicit instead of being folded into the text.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct JudgeMessage {
    pub role: &'static str,
    pub content: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub is_error: bool,
}

/// Rendering statistics, reported as metrics so a run can tell whether its
/// judge is reading whole trajectories or heavily elided ones.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RenderStats {
    pub rendered_chars: usize,
    pub elided_chars: usize,
    pub elided_messages: usize,
}

impl RenderStats {
    pub fn merge_from(&mut self, other: Self) {
        self.rendered_chars += other.rendered_chars;
        self.elided_chars += other.elided_chars;
        self.elided_messages += other.elided_messages;
    }
}

/// Shortens `text` to at most `max_chars` characters by keeping both ends and
/// replacing the middle with a marker. Byte-safe on multi-byte characters:
/// cut points always land on a `char` boundary.
pub fn elide(text: &str, max_chars: usize, head_ratio: f32) -> (String, usize) {
    if text.chars().count() <= max_chars {
        return (text.to_owned(), 0);
    }
    let total = text.chars().count();
    // The marker itself must fit, otherwise eliding could grow the text.
    let marker_budget = 32;
    let keep = max_chars.saturating_sub(marker_budget);
    if keep == 0 {
        return (format!("[{total} chars elided]"), total);
    }
    let head = ((keep as f32 * head_ratio).round() as usize).min(keep);
    let tail = keep - head;
    let head_end = char_offset(text, head);
    let tail_start = char_offset(text, total - tail);
    let elided = total - keep;
    (
        format!(
            "{}\n[{elided} chars elided]\n{}",
            &text[..head_end],
            &text[tail_start..]
        ),
        elided,
    )
}

/// Byte offset of the `index`-th character.
fn char_offset(text: &str, index: usize) -> usize {
    text.char_indices()
        .nth(index)
        .map(|(offset, _)| offset)
        .unwrap_or(text.len())
}

/// Renders a conversation slice under the per-message and per-trajectory
/// budgets. When the whole slice is still too long, messages are dropped from
/// the middle: the opening messages set up the task and the closing ones carry
/// the outcome, so both ends are what a judge actually needs.
pub fn render_messages(
    messages: &[Message],
    context: &JudgeContext,
) -> (Vec<JudgeMessage>, RenderStats) {
    let mut stats = RenderStats::default();
    let mut rendered = Vec::with_capacity(messages.len());
    for message in messages {
        let (content, elided) = elide(
            &message.content,
            context.max_message_chars,
            context.head_ratio,
        );
        if elided > 0 {
            stats.elided_chars += elided;
            stats.elided_messages += 1;
        }
        stats.rendered_chars += content.chars().count();
        rendered.push(JudgeMessage {
            role: message.role.as_str(),
            content,
            is_error: message.is_error,
        });
    }
    if stats.rendered_chars <= context.max_trajectory_chars {
        return (rendered, stats);
    }

    // Drop from the middle until the trajectory fits, keeping the first and
    // last message whatever happens.
    let mut sizes: Vec<usize> = rendered
        .iter()
        .map(|message| message.content.chars().count())
        .collect();
    let mut total = stats.rendered_chars;
    let mut dropped = 0_usize;
    while total > context.max_trajectory_chars && rendered.len() > 2 {
        let victim = rendered.len() / 2;
        total -= sizes[victim];
        stats.elided_chars += sizes[victim];
        stats.elided_messages += 1;
        rendered.remove(victim);
        sizes.remove(victim);
        dropped += 1;
    }
    if dropped > 0 {
        let marker = JudgeMessage {
            role: "system",
            content: format!("[{dropped} messages elided]"),
            is_error: false,
        };
        total += marker.content.chars().count();
        rendered.insert(rendered.len().saturating_sub(1).max(1), marker);
    }
    stats.rendered_chars = total;
    (rendered, stats)
}

/// Number of leading messages identical across every trajectory of a group.
/// Sending that prefix once instead of `group_size` times is the single
/// largest saving available on a judge prompt.
pub fn common_prefix_len(rows: &[&[Message]]) -> usize {
    let shortest = rows.iter().map(|row| row.len()).min().unwrap_or(0);
    (0..shortest)
        .take_while(|&index| rows[1..].iter().all(|row| row[index] == rows[0][index]))
        .count()
}

/// Packs items into chunks that each stay under `budget`, preserving order.
///
/// `overhead` is the fixed cost every request pays (rubric, shared context,
/// JSON scaffolding). `min_items` wins over the budget: a relative judge cannot
/// rank a trajectory against nothing, so a chunk is never closed before it holds
/// `min_items`, even when that overruns `budget`. An item larger than the whole
/// budget therefore still gets judged - against a sibling - rather than being
/// dropped or sent alone.
pub fn plan_chunks(
    costs: &[usize],
    budget: usize,
    overhead: usize,
    min_items: usize,
) -> Vec<Vec<usize>> {
    if costs.is_empty() {
        return Vec::new();
    }
    let room = budget.saturating_sub(overhead).max(1);
    let min_items = min_items.max(1);
    let mut chunks: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut used = 0_usize;
    for (index, &cost) in costs.iter().enumerate() {
        if current.len() >= min_items && used + cost > room {
            chunks.push(std::mem::take(&mut current));
            used = 0;
        }
        used += cost;
        current.push(index);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    // The tail can still fall short when the item count is not a multiple of the
    // chunk size; fold it back into its predecessor.
    while chunks.len() > 1 && chunks.last().is_some_and(|chunk| chunk.len() < min_items) {
        let orphan = chunks.pop().expect("checked above");
        chunks
            .last_mut()
            .expect("more than one chunk")
            .extend(orphan);
    }
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_agent_core::trajectory::Role;

    #[test]
    fn elision_keeps_both_ends_and_lands_on_char_boundaries() {
        let text = "é".repeat(500);
        let (elided, dropped) = elide(&text, 100, 0.4);
        assert!(elided.is_char_boundary(elided.len()));
        assert!(dropped > 0);
        assert!(elided.chars().count() <= 100);
        assert!(elided.contains("chars elided"));
        assert!(elided.starts_with('é') && elided.ends_with('é'));

        // Below the cap nothing is touched.
        let (kept, dropped) = elide("short", 100, 0.4);
        assert_eq!((kept.as_str(), dropped), ("short", 0));
    }

    #[test]
    fn head_ratio_moves_the_kept_window() {
        let text: String = ('a'..='z').cycle().take(400).collect();
        let (head_heavy, _) = elide(&text, 100, 1.0);
        let (tail_heavy, _) = elide(&text, 100, 0.0);
        assert!(head_heavy.starts_with(&text[..40]));
        assert!(tail_heavy.ends_with(&text[text.len() - 40..]));
    }

    #[test]
    fn a_long_conversation_is_elided_from_the_middle_keeping_the_ends() {
        let messages: Vec<Message> = (0..10)
            .map(|index| {
                Message::text(
                    Role::Assistant,
                    format!("message-{index} {}", "x".repeat(100)),
                )
            })
            .collect();
        let context = JudgeContext {
            max_request_chars: 10_000,
            max_trajectory_chars: 400,
            max_message_chars: 200,
            head_ratio: 0.5,
            include_env_state: true,
        };
        let (rendered, stats) = render_messages(&messages, &context);
        assert!(stats.rendered_chars <= context.max_trajectory_chars + 32);
        assert!(rendered.first().unwrap().content.contains("message-0"));
        assert!(rendered.last().unwrap().content.contains("message-9"));
        assert!(
            rendered
                .iter()
                .any(|message| message.content.contains("messages elided"))
        );
        assert!(stats.elided_messages > 0);
    }

    #[test]
    fn tool_errors_survive_rendering() {
        let mut message = Message::text(Role::Tool, "boom");
        message.is_error = true;
        let (rendered, _) = render_messages(&[message], &JudgeContext::default());
        assert!(rendered[0].is_error);
        assert_eq!(rendered[0].role, "tool");
    }

    #[test]
    fn chunk_planning_respects_the_budget_and_never_leaves_a_singleton() {
        // Four items of 30, budget 100, overhead 10: room is 90, so three fit and
        // the fourth would be alone - it folds back into the previous chunk.
        assert_eq!(
            plan_chunks(&[30, 30, 30, 30], 100, 10, 2),
            vec![vec![0, 1, 2, 3]]
        );
        // Six items: two chunks of three, both valid.
        assert_eq!(
            plan_chunks(&[30, 30, 30, 30, 30, 30], 100, 10, 2),
            vec![vec![0, 1, 2], vec![3, 4, 5]]
        );
        // Items that each exceed the room are still paired rather than sent
        // alone: min_items wins over the budget.
        assert_eq!(
            plan_chunks(&[400, 400, 400, 400], 1_200, 860, 2),
            vec![vec![0, 1], vec![2, 3]]
        );
        assert_eq!(plan_chunks(&[500, 10], 100, 10, 2), vec![vec![0, 1]]);
        // Every chunk of every plan holds at least min_items.
        for count in 2..12_usize {
            let costs = vec![400; count];
            let chunks = plan_chunks(&costs, 1_200, 860, 2);
            assert!(
                chunks.iter().all(|chunk| chunk.len() >= 2),
                "singleton chunk for {count} items: {chunks:?}"
            );
            let flattened: Vec<usize> = chunks.into_iter().flatten().collect();
            assert_eq!(flattened, (0..count).collect::<Vec<_>>());
        }
        assert!(plan_chunks(&[], 100, 10, 2).is_empty());
    }

    #[test]
    fn context_validation_rejects_inconsistent_budgets() {
        assert!(JudgeContext::default().validate().is_ok());
        for context in [
            JudgeContext {
                max_request_chars: 0,
                ..Default::default()
            },
            JudgeContext {
                max_message_chars: 9_000,
                max_trajectory_chars: 8_000,
                ..Default::default()
            },
            JudgeContext {
                max_trajectory_chars: 70_000,
                max_request_chars: 60_000,
                max_message_chars: 100,
                ..Default::default()
            },
            JudgeContext {
                head_ratio: 1.5,
                ..Default::default()
            },
        ] {
            assert!(context.validate().is_err(), "accepted {context:?}");
        }
    }
}
