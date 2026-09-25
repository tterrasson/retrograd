//! The registry: Rust implementations, the versioned tools declared over them,
//! and the toolsets that select those tools.
//!
//! Three layers, because they change for different reasons:
//!
//! - a **factory** is code (`shell`, `run_tests`, an embedder's lookup) and
//!   takes parameters;
//! - a **tool** is a [`ToolDefinition`] - `id@version`, the name and
//!   description the model reads, and either a factory with its parameters or
//!   a program to exec in the sandbox;
//! - a **toolset** is a named selection of tools.
//!
//! The built-in tools and toolsets are declared in `builtin.toml`, in the same
//! format a user writes: nothing about them is special beyond being loaded
//! first.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use retrograd_agent_core::{Error, Result, Sandbox, ToolSpec};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::exec_tool::ExecTool;
use crate::{
    CatalogEntry, LocalTool, SessionTool, ToolDefinition, ToolDefinitions, ToolImpl, ToolOutcome,
    ToolRef, ToolSource, ToolsConfig, ToolsetDefinition, glob_match,
};

/// A built tool, kept in the shape its caller needs it in: bound to a sandbox,
/// or shared as-is.
#[derive(Clone)]
pub enum RegisteredTool {
    Session(Arc<dyn SessionTool>),
    Shared(Arc<dyn LocalTool>),
}

/// Builds one implementation from the `params` a definition gave it
/// (`Value::Null` when it gave none).
pub trait ToolFactory: Send + Sync {
    fn build(&self, params: &Value) -> Result<RegisteredTool>;
}

impl<F> ToolFactory for F
where
    F: Fn(&Value) -> Result<RegisteredTool> + Send + Sync,
{
    fn build(&self, params: &Value) -> Result<RegisteredTool> {
        self(params)
    }
}

/// One tool built from its definition, with the catalog entry that says what
/// it is.
#[derive(Clone)]
pub struct BuiltTool {
    pub entry: CatalogEntry,
    pub tool: RegisteredTool,
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    factories: BTreeMap<String, Arc<dyn ToolFactory>>,
    tools: BTreeMap<(String, u32), ToolDefinition>,
    toolsets: BTreeMap<String, ToolsetDefinition>,
}

impl ToolRegistry {
    /// The built-in factories, tools and toolsets.
    pub fn builtin() -> Self {
        crate::builtin::builtin_registry()
    }

    /// Adds an implementation. Rejects an empty name and one already taken -
    /// checked before insertion, so a caller that treats the error as non-fatal
    /// never ends up with the original silently replaced.
    pub fn register_factory(
        &mut self,
        name: impl Into<String>,
        factory: Arc<dyn ToolFactory>,
    ) -> Result<()> {
        let name = name.into();
        if name.trim().is_empty() {
            return Err(Error::invalid("a tool factory name must not be empty"));
        }
        if self.factories.contains_key(&name) {
            return Err(Error::invalid(format!("duplicate tool factory '{name}'")));
        }
        self.factories.insert(name, factory);
        Ok(())
    }

    /// Adds tools and toolsets. A tool `id@version` or a toolset name that is
    /// already registered is refused: a new behaviour is a new version, never
    /// a replacement of the one a previous run was trained with.
    pub fn define(&mut self, definitions: ToolDefinitions) -> Result<()> {
        let mut seen = BTreeSet::new();
        for tool in &definitions.tools {
            tool.validate()?;
            if let ToolImpl::Builtin(builtin) = tool.implementation()?
                && !self.factories.contains_key(&builtin.factory)
            {
                return Err(Error::invalid(format!(
                    "tool '{}' names unknown factory '{}'; known: {}",
                    tool.key(),
                    builtin.factory,
                    self.factories
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                )));
            }
            let key = (tool.id.clone(), tool.version);
            if self.tools.contains_key(&key) || !seen.insert(key) {
                return Err(Error::invalid(format!(
                    "tool '{}' is defined more than once",
                    tool.key()
                )));
            }
        }
        for name in definitions.toolsets.keys() {
            if name.trim().is_empty() {
                return Err(Error::invalid("a toolset name must not be empty"));
            }
            if self.toolsets.contains_key(name) {
                return Err(Error::invalid(format!(
                    "toolset '{name}' is defined more than once"
                )));
            }
        }
        for tool in definitions.tools {
            self.tools.insert((tool.id.clone(), tool.version), tool);
        }
        self.toolsets.extend(definitions.toolsets);
        Ok(())
    }

    /// Reads one definition file. Its `exec.script` paths are relative to it.
    pub fn load_file(&mut self, path: &Path) -> Result<()> {
        let text = std::fs::read_to_string(path).map_err(|error| {
            Error::invalid(format!("read tool definitions {}: {error}", path.display()))
        })?;
        let mut definitions: ToolDefinitions = toml::from_str(&text).map_err(|error| {
            Error::invalid(format!(
                "{}: invalid tool definitions: {error}",
                path.display()
            ))
        })?;
        definitions.resolve_paths(path.parent().unwrap_or_else(|| Path::new(".")));
        self.define(definitions)
    }

    /// This registry plus what an environment's `tools` table declares: its
    /// files first, then its inline definitions.
    pub fn with_config(&self, config: &ToolsConfig) -> Result<Self> {
        let mut registry = self.clone();
        for file in &config.files {
            registry.load_file(file)?;
        }
        registry.define(config.inline())?;
        Ok(registry)
    }

    /// The definition a reference names: the exact version, or the latest one.
    pub fn lookup(&self, reference: &ToolRef) -> Result<&ToolDefinition> {
        let found = match reference.version {
            Some(version) => self.tools.get(&(reference.id.clone(), version)),
            None => self
                .tools
                .range((reference.id.clone(), 0)..=(reference.id.clone(), u32::MAX))
                .next_back()
                .map(|(_, definition)| definition),
        };
        found.ok_or_else(|| {
            let versions = self
                .tools
                .keys()
                .filter(|(id, _)| *id == reference.id)
                .map(|(id, version)| format!("{id}@{version}"))
                .collect::<Vec<_>>();
            Error::invalid(if versions.is_empty() {
                format!("unknown tool '{reference}'")
            } else {
                format!(
                    "unknown tool '{reference}'; registered: {}",
                    versions.join(", ")
                )
            })
        })
    }

    /// Builds the tool a reference names.
    pub fn build(&self, reference: &ToolRef) -> Result<BuiltTool> {
        let definition = self.lookup(reference)?;
        let key = definition.key();
        let (tool, source) = match definition.implementation()? {
            ToolImpl::Builtin(builtin) => {
                let factory = self.factories.get(&builtin.factory).ok_or_else(|| {
                    Error::invalid(format!(
                        "tool '{key}' names unknown factory '{}'",
                        builtin.factory
                    ))
                })?;
                let tool = factory.build(&builtin.params)?;
                let params = if builtin.params.is_null() {
                    String::new()
                } else {
                    builtin.params.to_string()
                };
                (
                    present(tool, definition),
                    ToolSource::Builtin {
                        factory: builtin.factory.clone(),
                        params,
                    },
                )
            }
            ToolImpl::Exec(exec) => {
                let tool = ExecTool::from_definition(definition, exec)?;
                let sha256 = tool.fingerprint();
                (
                    RegisteredTool::Session(Arc::new(tool)),
                    ToolSource::Exec { sha256 },
                )
            }
        };
        let (spec, stateful) = match &tool {
            RegisteredTool::Session(tool) => (tool.spec(), true),
            RegisteredTool::Shared(tool) => (tool.spec(), false),
        };
        if !spec.input_schema.is_object() {
            return Err(Error::invalid(format!(
                "tool '{key}' has a non-object input schema"
            )));
        }
        Ok(BuiltTool {
            entry: CatalogEntry {
                id: definition.id.clone(),
                version: Some(definition.version),
                spec,
                source,
                stateful,
            },
            tool,
        })
    }

    /// The definitions a toolset selects, in declaration order: its includes
    /// first, then its own tools - each replacing an earlier one with the same
    /// id or the same exposed name - and finally its `deny` patterns.
    pub fn toolset(&self, name: &str) -> Result<Vec<&ToolDefinition>> {
        let tools = self.expand(name, &mut Vec::new())?;
        if tools.is_empty() {
            return Err(Error::invalid(format!("toolset '{name}' selects no tools")));
        }
        Ok(tools)
    }

    /// Every registered tool definition, ordered by id then version.
    pub fn definitions(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools.values()
    }

    pub fn toolset_names(&self) -> impl Iterator<Item = &str> {
        self.toolsets.keys().map(String::as_str)
    }

    fn expand<'a>(
        &'a self,
        name: &str,
        stack: &mut Vec<String>,
    ) -> Result<Vec<&'a ToolDefinition>> {
        if stack.iter().any(|seen| seen == name) {
            stack.push(name.to_owned());
            return Err(Error::invalid(format!(
                "toolsets include each other in a cycle: {}",
                stack.join(" -> ")
            )));
        }
        let definition = self.toolsets.get(name).ok_or_else(|| {
            Error::invalid(format!(
                "unknown toolset '{name}'; known: {}",
                self.toolsets.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;
        stack.push(name.to_owned());
        let mut tools = Vec::new();
        for include in &definition.include {
            for tool in self.expand(include, stack)? {
                replace_or_push(&mut tools, tool);
            }
        }
        for reference in &definition.tools {
            replace_or_push(&mut tools, self.lookup(reference)?);
        }
        stack.pop();
        tools.retain(|tool| {
            !definition
                .deny
                .iter()
                .any(|pattern| glob_match(pattern, tool.exposed_name()))
        });
        Ok(tools)
    }
}

fn replace_or_push<'a>(tools: &mut Vec<&'a ToolDefinition>, tool: &'a ToolDefinition) {
    tools.retain(|existing| {
        existing.id != tool.id && existing.exposed_name() != tool.exposed_name()
    });
    tools.push(tool);
}

/// Hex SHA-256 of `bytes`, the one fingerprint the catalog uses.
pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    retrograd_core::hex_lower(&Sha256::digest(bytes))
}

/// Puts the definition's name - and its description, when it gave one - on a
/// builtin, whose behaviour and schema stay its own.
fn present(tool: RegisteredTool, definition: &ToolDefinition) -> RegisteredTool {
    let spec = |inner: ToolSpec| ToolSpec {
        name: definition.exposed_name().to_owned(),
        description: definition.description.clone().unwrap_or(inner.description),
        input_schema: inner.input_schema,
    };
    match tool {
        RegisteredTool::Session(inner) => RegisteredTool::Session(Arc::new(Presented {
            spec: spec(inner.spec()),
            inner,
        })),
        RegisteredTool::Shared(inner) => RegisteredTool::Shared(Arc::new(Presented {
            spec: spec(inner.spec()),
            inner,
        })),
    }
}

struct Presented<T: ?Sized> {
    spec: ToolSpec,
    inner: Arc<T>,
}

#[async_trait]
impl SessionTool for Presented<dyn SessionTool> {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, sandbox: &dyn Sandbox, arguments: Value) -> Result<ToolOutcome> {
        self.inner.call(sandbox, arguments).await
    }
}

#[async_trait]
impl LocalTool for Presented<dyn LocalTool> {
    fn spec(&self) -> ToolSpec {
        self.spec.clone()
    }

    async fn call(&self, arguments: Value) -> Result<String> {
        self.inner.call(arguments).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definitions(source: &str) -> ToolDefinitions {
        toml::from_str(source).unwrap()
    }

    #[test]
    fn an_unversioned_reference_is_the_latest_version() {
        let mut registry = ToolRegistry::builtin();
        registry
            .define(definitions(
                r#"
                [[tool]]
                id = "bash"
                version = 2
                description = "Run a command."
                builtin = { factory = "shell" }
                "#,
            ))
            .unwrap();
        assert_eq!(
            registry.lookup(&"bash".parse().unwrap()).unwrap().version,
            2
        );
        assert_eq!(
            registry.lookup(&"bash@1".parse().unwrap()).unwrap().version,
            1
        );
        let error = registry
            .lookup(&"bash@3".parse().unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("bash@1, bash@2"), "{error}");

        let built = registry.build(&"bash@2".parse().unwrap()).unwrap();
        assert_eq!(built.entry.spec.name, "bash");
        assert_eq!(built.entry.spec.description, "Run a command.");
        assert_eq!(built.entry.reference(), "bash@2");
    }

    #[test]
    fn a_version_is_never_redefined_and_a_factory_must_exist() {
        let mut registry = ToolRegistry::builtin();
        let error = registry
            .define(definitions(
                "[[tool]]\nid = \"bash\"\nversion = 1\nbuiltin = { factory = \"shell\" }\n",
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("more than once"), "{error}");
        let error = registry
            .define(definitions(
                "[[tool]]\nid = \"x\"\nversion = 1\nbuiltin = { factory = \"nope\" }\n",
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("unknown factory 'nope'"), "{error}");
    }

    #[test]
    fn duplicate_versions_in_one_definition_are_refused_without_inserting_any_tools() {
        let mut registry = ToolRegistry::builtin();
        let error = registry
            .define(definitions(
                r#"
                [[tool]]
                id = "custom"
                version = 1
                description = "First definition."
                builtin = { factory = "shell" }

                [[tool]]
                id = "custom"
                version = 1
                description = "Second definition."
                builtin = { factory = "shell" }
                "#,
            ))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("custom@1") && error.contains("more than once"),
            "{error}"
        );
        assert!(registry.lookup(&"custom@1".parse().unwrap()).is_err());
    }

    #[test]
    fn params_reach_the_factory_and_bad_ones_are_refused() {
        let mut registry = ToolRegistry::builtin();
        registry
            .define(definitions(
                r#"
                [[tool]]
                id = "cargo_test"
                version = 1
                name = "run_tests"
                builtin = { factory = "run_tests", params = { argv = ["cargo", "test"] } }

                [[tool]]
                id = "broken"
                version = 1
                builtin = { factory = "read_file", params = { path = "x" } }
                "#,
            ))
            .unwrap();
        let built = registry.build(&"cargo_test".parse().unwrap()).unwrap();
        assert!(built.entry.spec.description.contains("cargo test"));
        assert_eq!(
            built.entry.source,
            ToolSource::Builtin {
                factory: "run_tests".into(),
                params: r#"{"argv":["cargo","test"]}"#.into(),
            }
        );
        let error = registry
            .build(&"broken".parse().unwrap())
            .err()
            .expect("unknown params are refused")
            .to_string();
        assert!(error.contains("'read_file' takes no params"), "{error}");
    }

    #[test]
    fn a_toolset_includes_replaces_and_denies() {
        let mut registry = ToolRegistry::builtin();
        registry
            .define(definitions(
                r#"
                [[tool]]
                id = "my_tests"
                version = 1
                name = "run_tests"
                description = "Run the tests."
                exec = { argv = ["make", "test"] }

                [toolset.lite]
                include = ["python"]
                tools = ["my_tests"]
                deny = ["write_*", "edit_file"]
                "#,
            ))
            .unwrap();
        let names = registry
            .toolset("lite")
            .unwrap()
            .iter()
            .map(|tool| tool.key().to_string())
            .collect::<Vec<_>>();
        assert!(names.contains(&"my_tests@1".to_owned()), "{names:?}");
        assert!(!names.contains(&"pytest@1".to_owned()), "{names:?}");
        assert!(!names.iter().any(|name| name.starts_with("write_file")));
        assert!(names.contains(&"read_file@1".to_owned()));
    }

    #[test]
    fn a_toolset_cycle_and_an_empty_toolset_are_refused() {
        let mut registry = ToolRegistry::builtin();
        registry
            .define(definitions(
                r#"
                [toolset.a]
                include = ["b"]
                [toolset.b]
                include = ["a"]
                [toolset.none]
                include = ["python"]
                deny = ["*"]
                "#,
            ))
            .unwrap();
        let error = registry.toolset("a").unwrap_err().to_string();
        assert!(error.contains("a -> b -> a"), "{error}");
        let error = registry.toolset("none").unwrap_err().to_string();
        assert!(error.contains("selects no tools"), "{error}");
        assert!(registry.toolset("missing").is_err());
    }
}
