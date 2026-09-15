//! The language preset a run selects its tools with.

use serde::{Deserialize, Serialize};

/// A language preset: the tools it makes sense to hand a model, and - read by
/// `retrograd-env` rather than here - the image and package cache that go with
/// them.
///
/// A profile is only a way to fill two lists in one word. There is no per-language
/// code path to maintain anywhere in the stack.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Profile {
    #[default]
    Python,
    Typescript,
    /// Nothing preselected: the configuration lists the tools itself.
    Custom,
}

impl Profile {
    /// Maps public selection names to the registry entry appropriate for this
    /// profile. Both test runners intentionally expose the public name
    /// `run_tests` while using different implementations.
    pub fn registry_names(self, selected: &[String]) -> Vec<String> {
        selected
            .iter()
            .map(|name| {
                if name == "run_tests" {
                    match self {
                        Self::Python => "pytest".to_owned(),
                        Self::Typescript => "npm_test".to_owned(),
                        Self::Custom => name.clone(),
                    }
                } else {
                    name.clone()
                }
            })
            .collect()
    }

    /// Registry names selected by this preset, in prompt-stable order.
    pub fn tool_names(self) -> Vec<String> {
        let mut names = vec!["shell".to_owned()];
        match self {
            Self::Python => names.extend(["python".to_owned(), "pytest".to_owned()]),
            Self::Typescript => names.extend(["node".to_owned(), "npm_test".to_owned()]),
            Self::Custom => return Vec::new(),
        }
        names.extend(
            [
                "read_file",
                "write_file",
                "edit_file",
                "list_dir",
                "grep",
                "submit",
            ]
            .into_iter()
            .map(str::to_owned),
        );
        names
    }
}
