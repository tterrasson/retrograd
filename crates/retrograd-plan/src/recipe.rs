//! A recipe describes intent rather than a complete configuration.
//!
//! It lives here rather than in the server because the resolver is what consumes
//! it, and because a CLI subcommand that says "train this file, in this much
//! VRAM" wants exactly this type. Everything the operator owns - reward
//! commands, judge endpoints, MCP servers - is referenced by `id` only; the
//! server refuses the rest before this type is ever built (`guard.rs`).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result};

use crate::budget::BudgetRequest;

/// What the run is for. Selects the algorithm and the shape of phase 1.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Objective {
    /// Supervised fine-tuning on demonstrations.
    #[default]
    InstructionTuning,
    /// GRPO against a verifiable reward.
    ReasoningRl,
    /// PPO against a learned or judged preference.
    PreferenceRl,
    /// GRPO over tool-using trajectories.
    Agentic,
}

impl Objective {
    /// The `run.algorithm` spelling this objective resolves to.
    pub fn algorithm(self) -> &'static str {
        match self {
            Self::InstructionTuning => "sft",
            Self::ReasoningRl | Self::Agentic => "grpo",
            Self::PreferenceRl => "ppo",
        }
    }

    pub fn is_rollout(self) -> bool {
        !matches!(self, Self::InstructionTuning)
    }
}

/// How much training to do. Exactly one key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct TrainingBudget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub epochs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updates: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub minutes: Option<u32>,
}

impl TrainingBudget {
    fn validate(&self) -> Result<()> {
        let given = [
            self.epochs.map(|_| "epochs"),
            self.updates.map(|_| "updates"),
            self.minutes.map(|_| "minutes"),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
        match given.len() {
            0 | 1 => {}
            _ => {
                return Err(Error::invalid(format!(
                    "budget takes exactly one of epochs, updates or minutes; got {}",
                    given.join(", ")
                )));
            }
        }
        for (name, value) in [
            ("epochs", self.epochs),
            ("updates", self.updates),
            ("minutes", self.minutes),
        ] {
            if value == Some(0) {
                return Err(Error::invalid(format!(
                    "budget.{name} must be greater than zero"
                )));
            }
        }
        Ok(())
    }
}

/// A degradation the caller explicitly accepts.
///
/// Each of these changes *what* is computed, not only how fast, and neither is
/// recoverable from a warning: both lose data outright. The resolver never
/// reaches for one on its own; without the opt-in it fails and says what it
/// would have done.
///
/// `fast_generation` and `kv_f16` are handled as defaults rather than
/// degradations: neither discards data, and the runtime may fall back to F32
/// after probing the device. `kv_f16_may_fall_back` records that the estimate
/// may therefore overstate the saving.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Allow {
    /// Lower `n_ctx` below the dataset's chosen percentile. Examples are then
    /// truncated; the rate is reported.
    TruncateContext,
    /// `n_ctx` above the model's trained context.
    ExceedTrainContext,
}

impl Allow {
    pub fn id(self) -> &'static str {
        match self {
            Self::TruncateContext => "truncate_context",
            Self::ExceedTrainContext => "exceed_train_context",
        }
    }
}

/// Where the training data is: a literal path, or a dataset id from
/// `POST /v1/datasets` - mutually exclusive.
///
/// `dataset` is an id and nothing else, the same shape as [`RewardRef`] and
/// [`JudgeRef`]: this crate never touches a filesystem or a store, so
/// resolving it to bytes is entirely the server's job (`resolve::plan_recipe`
/// turns it into a real `path` before this recipe ever reaches [`crate::resolve`]).
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct DataSpec {
    /// A literal filesystem path. The server accepts it only when its
    /// `allow_local_paths` setting is on (default: loopback only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>))]
    pub path: Option<PathBuf>,
    /// A dataset id from `POST /v1/datasets` (`ds_…`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset: Option<String>,
    /// `auto` | `text` | `jsonl`. Absent is `auto`. Ignored when `dataset` is
    /// set - a stored dataset already knows its own format.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

impl DataSpec {
    /// Exactly one of `path` or `dataset` - checkable without touching either
    /// one, which is why it lives beside [`Recipe::validate`] rather than in
    /// the server.
    ///
    /// Returns the facade error, like [`TrainingBudget::validate`] above: what
    /// a request body says is an argument, and a `String` here would be the
    /// error type the facade exists to replace.
    fn validate(&self) -> Result<()> {
        match (&self.path, &self.dataset) {
            (Some(_), Some(_)) => Err(Error::invalid("send either `path` or `dataset`, not both")),
            (None, None) => Err(Error::invalid("either `path` or `dataset` is required")),
            _ => Ok(()),
        }
    }
}

/// Per-run memory limits, capped by the server-wide budgets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Limits {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vram: Option<BudgetRequest>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ram: Option<BudgetRequest>,
}

/// A catalogue reference: an id and nothing else.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RewardRef {
    pub id: String,
}

/// A judge reference plus the method settings a client may choose. The
/// endpoint, the model and the credential are the operator's and never appear.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct JudgeRef {
    pub id: String,
    /// `auto` | `listwise` | `chunked` | `pairwise`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pairs: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub anchor: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rubric: Option<String>,
}

/// The intent.
#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Recipe {
    #[serde(default)]
    pub objective: Objective,
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub model: PathBuf,
    pub data: DataSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval: Option<DataSpec>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub budget: Option<TrainingBudget>,
    #[serde(default)]
    pub limits: Limits,
    /// Opt-ins for the degradations of invariant 4. Deduplicated and sorted on
    /// validation, so two spellings of the same set resolve identically.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allow: Vec<Allow>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<RewardRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge: Option<JudgeRef>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<String>,
    /// Where the adapter and the checkpoints go. Defaulted by the caller (the
    /// server derives them from its state directory), never guessed here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>))]
    pub output: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(value_type = Option<String>))]
    pub checkpoint_dir: Option<PathBuf>,
    /// Seed for the LoRA initialisation and the samplers. A run created without
    /// one must still get one recorded, or a fork is not reproducible.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
}

/// One independent problem found by [`Recipe::validate`], located by a JSON
/// Pointer into the request body.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FieldIssue {
    pub pointer: &'static str,
    pub message: String,
}

impl Recipe {
    /// Everything checkable without a model, a dataset or a device.
    ///
    /// Every check here is independent of every other - none reads a value a
    /// previous check might have rejected - so all of them run and every
    /// failure is reported together: a body with three
    /// problems is worth three lines, not three round trips.
    pub fn validate(&mut self) -> std::result::Result<(), Vec<FieldIssue>> {
        let mut issues = Vec::new();
        if self.model.as_os_str().is_empty() {
            issues.push(FieldIssue {
                pointer: "/recipe/model",
                message: "recipe.model is required".to_string(),
            });
        }
        if let Err(error) = self.data.validate() {
            issues.push(FieldIssue {
                pointer: "/recipe/data",
                message: error.to_string(),
            });
        }
        if let Some(eval) = &self.eval
            && let Err(error) = eval.validate()
        {
            issues.push(FieldIssue {
                pointer: "/recipe/eval",
                message: error.to_string(),
            });
        }
        if let Some(budget) = &self.budget
            && let Err(error) = budget.validate()
        {
            issues.push(FieldIssue {
                pointer: "/recipe/budget",
                message: error.to_string(),
            });
        }
        if self.objective.is_rollout() && self.reward.is_none() && self.judge.is_none() {
            issues.push(FieldIssue {
                pointer: "/recipe",
                message: "a reinforcement objective needs a reward or a judge; declare one \
                          server-side and reference its id"
                    .to_string(),
            });
        }
        if !self.objective.is_rollout() && (self.reward.is_some() || self.judge.is_some()) {
            issues.push(FieldIssue {
                pointer: "/recipe/reward",
                message: "reward and judge only apply to a reinforcement objective".to_string(),
            });
        }
        if !self.tools.is_empty() && self.objective != Objective::Agentic {
            issues.push(FieldIssue {
                pointer: "/recipe/tools",
                message: "tools only apply to the agentic objective".to_string(),
            });
        }
        if !issues.is_empty() {
            return Err(issues);
        }
        self.allow.sort_unstable();
        self.allow.dedup();
        Ok(())
    }

    pub fn allows(&self, allow: Allow) -> bool {
        self.allow.contains(&allow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sft() -> Recipe {
        Recipe {
            objective: Objective::InstructionTuning,
            model: PathBuf::from("model.gguf"),
            data: DataSpec {
                path: Some(PathBuf::from("data.jsonl")),
                dataset: None,
                format: None,
            },
            eval: None,
            budget: None,
            limits: Limits::default(),
            allow: Vec::new(),
            reward: None,
            judge: None,
            tools: Vec::new(),
            output: None,
            checkpoint_dir: None,
            seed: None,
        }
    }

    #[test]
    fn a_budget_takes_exactly_one_key() {
        let mut recipe = sft();
        recipe.budget = Some(TrainingBudget {
            epochs: Some(3),
            updates: None,
            minutes: None,
        });
        recipe.validate().expect("one key is fine");

        recipe.budget = Some(TrainingBudget {
            epochs: Some(3),
            updates: Some(10),
            minutes: None,
        });
        let issues = recipe.validate().expect_err("two keys must be refused");
        assert!(issues[0].message.contains("exactly one"), "{issues:?}");

        recipe.budget = Some(TrainingBudget {
            epochs: Some(0),
            updates: None,
            minutes: None,
        });
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn data_takes_exactly_one_of_path_or_dataset() {
        let mut neither = sft();
        neither.data.path = None;
        let issues = neither.validate().expect_err("neither is not enough");
        assert_eq!(issues[0].pointer, "/recipe/data");
        assert!(issues[0].message.contains("required"), "{issues:?}");

        let mut both = sft();
        both.data.dataset = Some("ds_abc123".to_string());
        let issues = both.validate().expect_err("both is ambiguous");
        assert_eq!(issues[0].pointer, "/recipe/data");
        assert!(issues[0].message.contains("not both"), "{issues:?}");

        let mut dataset_only = sft();
        dataset_only.data.path = None;
        dataset_only.data.dataset = Some("ds_abc123".to_string());
        dataset_only.validate().expect("a dataset id is enough");
    }

    #[test]
    fn eval_takes_exactly_one_of_path_or_dataset_too() {
        let mut recipe = sft();
        recipe.eval = Some(DataSpec {
            path: None,
            dataset: None,
            format: None,
        });
        let issues = recipe.validate().expect_err("neither is not enough");
        assert_eq!(issues[0].pointer, "/recipe/eval");
    }

    #[test]
    fn a_reinforcement_objective_needs_something_to_optimize() {
        let mut recipe = Recipe {
            objective: Objective::ReasoningRl,
            ..sft()
        };
        assert!(recipe.validate().is_err());
        recipe.reward = Some(RewardRef {
            id: "sql-exec".to_string(),
        });
        recipe.validate().expect("a reward is enough");
    }

    #[test]
    fn independent_problems_are_all_reported_at_once() {
        let mut recipe = Recipe {
            objective: Objective::ReasoningRl,
            model: PathBuf::new(),
            ..sft()
        };
        recipe.tools = vec!["shell".to_string()];
        let issues = recipe.validate().expect_err("three independent problems");
        let pointers: Vec<_> = issues.iter().map(|issue| issue.pointer).collect();
        assert!(pointers.contains(&"/recipe/model"), "{pointers:?}");
        assert!(pointers.contains(&"/recipe"), "{pointers:?}");
        assert!(pointers.contains(&"/recipe/tools"), "{pointers:?}");
    }

    #[test]
    fn a_supervised_objective_refuses_a_reward() {
        let mut recipe = sft();
        recipe.judge = Some(JudgeRef {
            id: "ruler-mini".to_string(),
            mode: None,
            max_pairs: None,
            anchor: None,
            rubric: None,
        });
        assert!(recipe.validate().is_err());
    }

    #[test]
    fn opt_ins_are_normalized_so_two_spellings_resolve_alike() {
        let mut recipe = sft();
        recipe.allow = vec![
            Allow::ExceedTrainContext,
            Allow::TruncateContext,
            Allow::TruncateContext,
        ];
        recipe.validate().unwrap();
        assert_eq!(
            recipe.allow,
            vec![Allow::TruncateContext, Allow::ExceedTrainContext]
        );
        assert!(recipe.allows(Allow::TruncateContext));
    }

    /// `fast_generation` and `kv_f16` left the vocabulary when they became
    /// defaults. A recipe still asking for one is refused rather than
    /// ignored: silently dropping it would tell a client its opt-in was
    /// honoured.
    #[test]
    fn an_opt_in_that_became_a_default_is_no_longer_accepted() {
        for retired in ["fast_generation", "kv_f16"] {
            let error = serde_json::from_str::<Recipe>(&format!(
                r#"{{"model":"m.gguf","data":{{"path":"d.jsonl"}},"allow":["{retired}"]}}"#
            ))
            .expect_err("a retired opt-in must not be silently dropped");
            assert!(error.to_string().contains(retired), "{error}");
        }
    }

    #[test]
    fn the_wire_spelling_of_every_enum_is_the_documented_one() {
        assert_eq!(
            serde_json::to_string(&Objective::InstructionTuning).unwrap(),
            r#""instruction-tuning""#
        );
        assert_eq!(
            serde_json::to_string(&Objective::ReasoningRl).unwrap(),
            r#""reasoning-rl""#
        );
        assert_eq!(
            serde_json::to_string(&Allow::TruncateContext).unwrap(),
            r#""truncate_context""#
        );
        assert_eq!(
            serde_json::to_string(&Allow::ExceedTrainContext).unwrap(),
            r#""exceed_train_context""#
        );
    }

    #[test]
    fn an_unknown_field_is_refused_rather_than_ignored() {
        let error = serde_json::from_str::<Recipe>(
            r#"{"model":"m.gguf","data":{"path":"d.jsonl"},"profil":"fast"}"#,
        )
        .expect_err("a typo must not be silently dropped");
        assert!(error.to_string().contains("profil"), "{error}");
    }
}
