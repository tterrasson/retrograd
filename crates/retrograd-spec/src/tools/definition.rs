//! Tool and toolset definitions, as a file writes them.
//!
//! A tool is identified by `id@version`, not by the name the model reads: its
//! description and schema are tokenized and trained on, so changing either is a
//! new version, and two versions of one tool can be compared in two runs - or
//! in two toolsets of the same run - without either overwriting the other.
//!
//! ```toml
//! [[tool]]
//! id = "run_tests"
//! version = 2
//! description = "Run the test suite and report failures."
//! exec = { argv = ["python3", "-c"], script = "tools/run_tests.py", protocol = "json" }
//!
//! [toolset.python-lite]
//! include = ["python"]
//! tools = ["run_tests@2"]
//! deny = ["write_file"]
//! ```

use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use retrograd_agent_core::{Error, Result};

/// `id` or `id@version`. Without a version the reference resolves to the latest
/// one registered, and the catalog records which one that was.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ToolRef {
    pub id: String,
    pub version: Option<u32>,
}

impl ToolRef {
    pub fn exact(id: impl Into<String>, version: u32) -> Self {
        Self {
            id: id.into(),
            version: Some(version),
        }
    }
}

impl fmt::Display for ToolRef {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.version {
            Some(version) => write!(formatter, "{}@{version}", self.id),
            None => formatter.write_str(&self.id),
        }
    }
}

impl FromStr for ToolRef {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        let (id, version) = match value.split_once('@') {
            Some((id, version)) => {
                let version = version.parse::<u32>().map_err(|_| {
                    Error::invalid(format!(
                        "tool reference '{value}': the version must be a positive integer"
                    ))
                })?;
                (id, Some(version))
            }
            None => (value, None),
        };
        validate_id(id)?;
        if version == Some(0) {
            return Err(Error::invalid(format!(
                "tool reference '{value}': versions start at 1"
            )));
        }
        Ok(Self {
            id: id.to_owned(),
            version,
        })
    }
}

impl Serialize for ToolRef {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for ToolRef {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let value = String::deserialize(deserializer)?;
        value.parse().map_err(serde::de::Error::custom)
    }
}

fn validate_id(id: &str) -> Result<()> {
    let valid = !id.is_empty()
        && id
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_-.".contains(character));
    if !valid {
        return Err(Error::invalid(format!(
            "tool id '{id}' must be non-empty and use only ASCII letters, digits, '_', '-' or '.'"
        )));
    }
    Ok(())
}

/// One tool: who it is, what the model reads, and what runs it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToolDefinition {
    pub id: String,
    pub version: u32,
    /// The name the model calls. Defaults to `id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Required for an exec tool. A builtin brings its own and may be given a
    /// different one here.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Exec tools only: a builtin reads its own arguments, so its schema is not
    /// the declaration's to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_schema: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub builtin: Option<BuiltinImpl>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exec: Option<ExecImpl>,
}

impl ToolDefinition {
    pub fn key(&self) -> ToolRef {
        ToolRef::exact(&self.id, self.version)
    }

    pub fn exposed_name(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.id)
    }

    pub fn implementation(&self) -> Result<ToolImpl<'_>> {
        match (&self.builtin, &self.exec) {
            (Some(builtin), None) => Ok(ToolImpl::Builtin(builtin)),
            (None, Some(exec)) => Ok(ToolImpl::Exec(exec)),
            _ => Err(Error::invalid(format!(
                "tool '{}' must declare exactly one of `builtin` or `exec`",
                self.key()
            ))),
        }
    }

    /// Everything checkable without a registry or a filesystem.
    pub fn validate(&self) -> Result<()> {
        let key = self.key();
        validate_id(&self.id)?;
        if self.version == 0 {
            return Err(Error::invalid(format!("tool '{key}': versions start at 1")));
        }
        if self
            .name
            .as_deref()
            .is_some_and(|name| name.trim().is_empty())
        {
            return Err(Error::invalid(format!(
                "tool '{key}': name must not be empty"
            )));
        }
        match self.implementation()? {
            ToolImpl::Builtin(_) => {
                if self.input_schema.is_some() {
                    return Err(Error::invalid(format!(
                        "tool '{key}': input_schema is declared for exec tools only; a builtin \
                         reads its own arguments"
                    )));
                }
            }
            ToolImpl::Exec(exec) => {
                if self
                    .description
                    .as_deref()
                    .is_none_or(|text| text.trim().is_empty())
                {
                    return Err(Error::invalid(format!(
                        "tool '{key}': an exec tool needs a description"
                    )));
                }
                if self
                    .input_schema
                    .as_ref()
                    .is_some_and(|schema| !schema.is_object())
                {
                    return Err(Error::invalid(format!(
                        "tool '{key}': input_schema must be a JSON object"
                    )));
                }
                exec.validate(&key)?;
            }
        }
        Ok(())
    }
}

/// Which of the two implementations a definition declared.
#[derive(Clone, Copy, Debug)]
pub enum ToolImpl<'a> {
    Builtin(&'a BuiltinImpl),
    Exec(&'a ExecImpl),
}

/// A Rust implementation from the registry, with its parameters.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuiltinImpl {
    pub factory: String,
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub params: serde_json::Value,
}

/// A program run inside the trajectory's sandbox, in any language.
///
/// The arguments the model passed arrive as one JSON object on stdin. With
/// `protocol = "text"` the command's result is rendered like every other
/// command; with `protocol = "json"` stdout is an [`ExecReply`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecImpl {
    pub argv: Vec<String>,
    /// A host file whose contents are appended to `argv` as its last element:
    /// `argv = ["python3", "-c"]` and a script ship a tool without building it
    /// into the image. Relative to the file that declares it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub script: Option<PathBuf>,
    #[serde(default)]
    pub protocol: ExecProtocol,
    /// Falls back to the sandbox's own exec timeout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

impl ExecImpl {
    fn validate(&self, key: &ToolRef) -> Result<()> {
        if self
            .argv
            .first()
            .is_none_or(|program| program.trim().is_empty())
        {
            return Err(Error::invalid(format!(
                "tool '{key}': exec.argv must name a program"
            )));
        }
        if self.timeout_secs == Some(0) {
            return Err(Error::invalid(format!(
                "tool '{key}': exec.timeout_secs must be positive"
            )));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecProtocol {
    /// Exit code, stdout and stderr, rendered as the model reads any command.
    #[default]
    Text,
    /// Stdout is one [`ExecReply`] object.
    Json,
}

/// What a `protocol = "json"` tool prints on stdout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecReply {
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reward: Option<f32>,
    #[serde(default)]
    pub done: bool,
}

/// A named selection of tools.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsetDefinition {
    /// Toolsets whose tools come first. A tool listed in `tools` replaces an
    /// included one with the same id or the same exposed name.
    pub include: Vec<String>,
    pub tools: Vec<ToolRef>,
    /// Exposed-name patterns removed last, `*` as a wildcard.
    pub deny: Vec<String>,
}

/// The shape of a definition file: tools and toolsets, nothing else.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolDefinitions {
    #[serde(rename = "tool", skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(rename = "toolset", skip_serializing_if = "BTreeMap::is_empty")]
    pub toolsets: BTreeMap<String, ToolsetDefinition>,
}

impl ToolDefinitions {
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty() && self.toolsets.is_empty()
    }

    /// Resolves every `exec.script` against `base`, the directory of the
    /// document that declared it.
    pub fn resolve_paths(&mut self, base: &Path) {
        for tool in &mut self.tools {
            if let Some(script) = tool.exec.as_mut().and_then(|exec| exec.script.as_mut())
                && script.is_relative()
            {
                *script = base.join(&*script);
            }
        }
    }
}

/// The `tools` table of a sandbox environment: which toolset a trajectory
/// gets, which others a scenario may ask for, and where the definitions are.
///
/// ```toml
/// [agent.environment.tools]
/// default = "python"
/// scenario_toolsets = ["python-lite"]
/// files = ["tools.toml"]
/// ```
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    /// The toolset of a scenario that names none. Required.
    pub default: Option<String>,
    /// Other toolsets a scenario may select with `metadata.env.toolset`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub scenario_toolsets: Vec<String>,
    /// Definition files, each shaped like [`ToolDefinitions`].
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub files: Vec<PathBuf>,
    #[serde(rename = "tool", skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<ToolDefinition>,
    #[serde(rename = "toolset", skip_serializing_if = "BTreeMap::is_empty")]
    pub toolsets: BTreeMap<String, ToolsetDefinition>,
}

impl ToolsConfig {
    /// The default toolset. Refused when absent: a sandbox with no tools gives
    /// the policy nothing to do, and no profile picks them any more.
    pub fn default_toolset(&self) -> Result<&str> {
        self.default.as_deref().ok_or_else(|| {
            Error::invalid(
                "a sandbox environment needs a toolset: set tools.default (e.g. \"python\")",
            )
        })
    }

    /// Every toolset a trajectory of this environment may get, default first.
    pub fn selectable(&self) -> Result<Vec<&str>> {
        let mut names = vec![self.default_toolset()?];
        for name in &self.scenario_toolsets {
            if !names.contains(&name.as_str()) {
                names.push(name);
            }
        }
        Ok(names)
    }

    /// The inline definitions, in the shape of a file.
    pub fn inline(&self) -> ToolDefinitions {
        ToolDefinitions {
            tools: self.tools.clone(),
            toolsets: self.toolsets.clone(),
        }
    }

    /// Makes `files` and inline `exec.script` paths absolute against `root`,
    /// the directory of the configuration.
    pub fn resolve_paths(&mut self, root: &Path) {
        for file in &mut self.files {
            if file.is_relative() {
                *file = root.join(&*file);
            }
        }
        let mut inline = self.inline();
        inline.resolve_paths(root);
        self.tools = inline.tools;
    }

    pub fn validate_declaration(&self) -> Result<()> {
        self.default_toolset()?;
        for tool in &self.tools {
            tool.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_is_an_id_with_an_optional_positive_version() {
        assert_eq!(
            "bash".parse::<ToolRef>().unwrap(),
            ToolRef {
                id: "bash".into(),
                version: None
            }
        );
        assert_eq!(
            "run_tests@2".parse::<ToolRef>().unwrap(),
            ToolRef::exact("run_tests", 2)
        );
        for bad in ["", "@1", "bash@", "bash@0", "bash@x", "a b"] {
            assert!(bad.parse::<ToolRef>().is_err(), "{bad}");
        }
    }

    #[test]
    fn a_definition_declares_exactly_one_implementation() {
        let parse = |source: &str| toml::from_str::<ToolDefinitions>(source).unwrap().tools;
        let tools = parse(
            r#"
            [[tool]]
            id = "lint"
            version = 1
            description = "Lint the workspace."
            exec = { argv = ["ruff", "check"] }
            "#,
        );
        tools[0].validate().unwrap();
        assert_eq!(tools[0].exposed_name(), "lint");

        for (body, expected) in [
            ("", "exactly one"),
            (
                "builtin = { factory = \"shell\" }\nexec = { argv = [\"x\"] }",
                "exactly one",
            ),
            ("exec = { argv = [\"x\"] }", "description"),
            (
                "description = \"d\"\nexec = { argv = [] }",
                "must name a program",
            ),
            (
                "builtin = { factory = \"shell\" }\ninput_schema = { type = \"object\" }",
                "exec tools only",
            ),
        ] {
            let source = format!("[[tool]]\nid = \"t\"\nversion = 1\n{body}\n");
            let error = parse(&source)[0].validate().unwrap_err().to_string();
            assert!(error.contains(expected), "{body}: {error}");
        }
    }

    #[test]
    fn paths_are_resolved_against_the_declaring_document() {
        let mut config: ToolsConfig = toml::from_str(
            r#"
            default = "python"
            files = ["defs.toml", "/abs/defs.toml"]
            [[tool]]
            id = "t"
            version = 1
            description = "d"
            exec = { argv = ["python3", "-c"], script = "t.py" }
            "#,
        )
        .unwrap();
        config.resolve_paths(Path::new("/cfg"));
        assert_eq!(
            config.files,
            [
                PathBuf::from("/cfg/defs.toml"),
                PathBuf::from("/abs/defs.toml")
            ]
        );
        assert_eq!(
            config.tools[0].exec.as_ref().unwrap().script.as_deref(),
            Some(Path::new("/cfg/t.py"))
        );
        assert_eq!(config.selectable().unwrap(), ["python"]);
        assert!(ToolsConfig::default().validate_declaration().is_err());
    }
}
