use std::collections::HashSet;

use retrograd_core::hex_lower;
use retrograd_tools::ToolCatalog;
use sha2::{Digest, Sha256};

use crate::{RejectionCounts, ScenarioDraft};

pub const NEAR_DUPLICATE_THRESHOLD: f32 = 0.82;

pub fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase()
}

pub fn text_hash(text: &str) -> String {
    hex_lower(&Sha256::digest(normalized(text).as_bytes()))
}

pub fn validate_draft(
    draft: &ScenarioDraft,
    catalog: &ToolCatalog,
    min_difficulty: u8,
    max_difficulty: u8,
    accepted: &[ScenarioDraft],
    rejected: &mut RejectionCounts,
) -> bool {
    if draft.task.trim().is_empty()
        || draft.difficulty < min_difficulty
        || draft.difficulty > max_difficulty
        || draft.success_criteria.is_empty()
        || draft.success_criteria.len() > 5
        || draft
            .success_criteria
            .iter()
            .any(|criterion| criterion.trim().is_empty())
        || (draft.expected_tools.is_empty() && draft.expected_resources.is_empty())
    {
        rejected.invalid += 1;
        return false;
    }
    let tools = catalog
        .tools
        .iter()
        .map(|entry| entry.spec.name.as_str())
        .collect::<HashSet<_>>();
    if draft
        .expected_tools
        .iter()
        .any(|name| !tools.contains(name.as_str()))
    {
        rejected.unknown_tool += 1;
        return false;
    }
    let resources = catalog
        .resources
        .iter()
        .map(|resource| resource.uri.as_str())
        .collect::<HashSet<_>>();
    if draft
        .expected_resources
        .iter()
        .any(|uri| !resources.contains(uri.as_str()))
    {
        rejected.unknown_resource += 1;
        return false;
    }
    let candidate = normalized(&draft.task);
    for previous in accepted {
        let existing = normalized(&previous.task);
        if existing == candidate {
            rejected.duplicate += 1;
            return false;
        }
        if jaccard(&candidate, &existing) >= NEAR_DUPLICATE_THRESHOLD {
            rejected.near_duplicate += 1;
            return false;
        }
    }
    true
}

fn jaccard(a: &str, b: &str) -> f32 {
    let grams = |text: &str| {
        let chars = text.chars().collect::<Vec<_>>();
        if chars.len() < 3 {
            return [text.to_owned()].into_iter().collect::<HashSet<_>>();
        }
        chars
            .windows(3)
            .map(|window| window.iter().collect())
            .collect::<HashSet<String>>()
    };
    let a = grams(a);
    let b = grams(b);
    let union = a.union(&b).count();
    if union == 0 {
        1.0
    } else {
        a.intersection(&b).count() as f32 / union as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_tools::{CatalogEntry, ToolSource};

    #[test]
    fn normalization_and_similarity_are_deterministic() {
        assert_eq!(normalized("  Hello  WORLD\n"), "hello world");
        assert_eq!(text_hash("A  b"), text_hash("a b"));
        assert!(jaccard("compare the two documents", "compare the two document") > 0.8);
    }

    #[test]
    fn validation_accepts_session_tools_and_rejects_unknown_references() {
        let catalog = ToolCatalog {
            tools: vec![CatalogEntry {
                id: "write_file".into(),
                version: Some(1),
                spec: retrograd_agent_core::ToolSpec {
                    name: "write_file".into(),
                    description: "write".into(),
                    input_schema: serde_json::json!({"type": "object"}),
                },
                source: ToolSource::Builtin {
                    factory: "write_file".into(),
                    params: String::new(),
                },
                stateful: true,
            }],
            ..Default::default()
        };
        let mut draft = ScenarioDraft {
            task: "Write the report".into(),
            difficulty: 2,
            expected_tools: vec!["write_file".into()],
            expected_resources: vec![],
            success_criteria: vec!["report exists".into()],
        };
        assert!(validate_draft(
            &draft,
            &catalog,
            1,
            5,
            &[],
            &mut RejectionCounts::default()
        ));
        draft.expected_tools[0] = "missing".into();
        let mut rejected = RejectionCounts::default();
        assert!(!validate_draft(&draft, &catalog, 1, 5, &[], &mut rejected));
        assert_eq!(rejected.unknown_tool, 1);
    }
}
