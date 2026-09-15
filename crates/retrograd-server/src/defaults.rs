//! `GET /v1/defaults`: what this server decides when `params` stays silent.
//!
//! It replaced `GET /v1/presets`. A preset was a word - `fast`, `balanced`,
//! `thorough` - that a client selected and the server interpreted; nothing about
//! the word said what it would do to *this* model on *this* machine, and the
//! interpretation was three tables of constants compiled into the binary.
//!
//! What a client actually needs in order to build its own profile is the
//! interpretation: for each field, the rule, its thresholds, and whether it can
//! set the field itself. So that is what this renders - **generated from the
//! resolver's own tables**, [`retrograd_plan::DERIVATIONS`] and
//! [`retrograd_plan::ACTIVE_DEFAULTS`], never recopied. A rule that moves in the
//! resolver moves here, and `crates/retrograd-plan/tests/derivation_table.rs`
//! fails if either table stops describing what a resolution really does.
//!
//! Nothing here is selectable. There is no `?profile=`, no id to post back: it is
//! documentation with numbers in it.

use retrograd_plan::tuning::Scope;

use crate::dto;

pub fn listing() -> dto::Defaults {
    dto::Defaults {
        derived: retrograd_plan::DERIVATIONS
            .iter()
            .map(|derivation| dto::DerivedField {
                path: derivation.path,
                rule: derivation.rule,
                thresholds: derivation.thresholds.iter().copied().collect(),
                applies_to: scope_id(derivation.scope),
                // Every derivable field is a `params` field. Stated rather than
                // implied, because "the server decides this" and "you may not"
                // are exactly the two things a client must not confuse.
                client_can_set: true,
            })
            .collect(),
        active_defaults: retrograd_plan::ACTIVE_DEFAULTS
            .iter()
            .map(|entry| dto::ActiveDefault {
                id: entry.id,
                condition: entry.condition,
                cost: entry.cost,
                note: entry.note,
                paths: entry.touches.to_vec(),
                degrades: entry.degrades,
                warning: entry.warning,
                client_can_disable: true,
            })
            .collect(),
    }
}

/// The wire spelling of a rule's scope.
fn scope_id(scope: Scope) -> &'static str {
    match scope {
        Scope::Any => "any",
        Scope::Sft => "sft",
        Scope::Grpo => "grpo",
        Scope::Ppo => "ppo",
        Scope::Rollout => "rollout",
        Scope::Evaluation => "evaluation",
        Scope::Checkpoint => "checkpoint",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_listing_is_the_resolvers_own_tables() {
        let listing = listing();
        assert_eq!(listing.derived.len(), retrograd_plan::DERIVATIONS.len());
        assert_eq!(
            listing.active_defaults.len(),
            retrograd_plan::ACTIVE_DEFAULTS.len()
        );
        assert!(!listing.derived.is_empty() && !listing.active_defaults.is_empty());
    }

    #[test]
    fn every_rule_names_a_dotted_path_a_sentence_and_a_scope() {
        const SCOPES: [&str; 7] = [
            "any",
            "sft",
            "grpo",
            "ppo",
            "rollout",
            "evaluation",
            "checkpoint",
        ];
        for field in listing().derived {
            // One path grammar across `params`, `provenance` and the PATCH
            // whitelist: dotted, no leading or trailing separator.
            assert!(
                field.path.contains('.'),
                "'{}' is not a dotted field path",
                field.path
            );
            assert!(
                !field.path.starts_with('.') && !field.path.ends_with('.'),
                "{}",
                field.path
            );
            assert!(!field.rule.is_empty(), "{}", field.path);
            assert!(SCOPES.contains(&field.applies_to), "{}", field.applies_to);
            assert!(field.client_can_set, "{}", field.path);
            for (name, value) in &field.thresholds {
                assert!(!name.is_empty() && value.is_finite(), "{}", field.path);
            }
        }
    }

    /// The listing is the *whole* of what invariant 4 now permits without an
    /// opt-in, so a degrading default that did not announce a warning code would
    /// be a silent degradation with paperwork.
    #[test]
    fn every_degrading_default_publishes_the_code_it_raises() {
        let mut degrading = 0;
        for entry in listing().active_defaults {
            assert!(!entry.paths.is_empty(), "{}", entry.id);
            assert!(!entry.condition.is_empty(), "{}", entry.id);
            assert!(entry.client_can_disable, "{}", entry.id);
            if entry.degrades {
                degrading += 1;
                let code = entry
                    .warning
                    .expect("a degrading default must be detectable");
                assert!(!code.is_empty(), "{}", entry.id);
            }
        }
        assert_eq!(
            degrading, 1,
            "only fast_sampling_context is a degradation by default (D4)"
        );
    }

    /// A rule is only useful to a client building its own profile if the
    /// numbers in its sentence come out as numbers, not only as prose.
    #[test]
    fn a_rule_publishes_the_thresholds_its_sentence_names() {
        let rank = listing()
            .derived
            .into_iter()
            .find(|field| field.path == "lora.rank")
            .expect("the rank rule is documented");
        assert_eq!(rank.thresholds.get("examples_rank_8"), Some(&2_000.0));
        assert_eq!(rank.thresholds.get("parameter_budget_percent"), Some(&2.0));
    }
}
