//! Extensible registry for session-bound and shared in-process tools.

use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

use retrograd_agent_core::{Error, Result, ToolSpec};

use crate::{LocalTool, SessionTool};

/// A tool built from the registry, kept in the shape its caller needs it in:
/// bound to a sandbox, or shared as-is.
#[derive(Clone)]
pub enum RegisteredTool {
    Session(Arc<dyn SessionTool>),
    Shared(Arc<dyn LocalTool>),
}

/// Builds one named tool on demand, so the registry can hand out fresh
/// instances without knowing their concrete type.
pub trait ToolFactory: Send + Sync {
    fn name(&self) -> &str;
    fn describe(&self) -> ToolSpec;
    fn build(&self, params: &serde_json::Value) -> Result<RegisteredTool>;
}

/// Maps tool names to the [`ToolFactory`] that builds them.
#[derive(Clone, Default)]
pub struct ToolRegistry {
    factories: BTreeMap<String, Arc<dyn ToolFactory>>,
}

impl ToolRegistry {
    /// The registry of built-in tools (`bash`, `read_file`, `submit`, ...).
    pub fn builtin() -> Self {
        crate::builtin::builtin_registry()
    }

    /// Adds a factory under its own name. Rejects an empty name, a schema
    /// that is not a JSON object, and a name already registered - checked
    /// before insertion, so a caller that treats the error as non-fatal never
    /// ends up with the original tool silently replaced.
    pub fn register(&mut self, factory: Arc<dyn ToolFactory>) -> Result<()> {
        let name = factory.name().trim().to_owned();
        if name.is_empty() {
            return Err(Error::invalid("a registered tool name must not be empty"));
        }
        let spec = factory.describe();
        if !spec.input_schema.is_object() {
            return Err(Error::invalid(format!(
                "registered tool '{name}' has a non-object schema"
            )));
        }
        // Checked before inserting: reporting the duplicate *after* the
        // overwrite leaves a caller that treats the error as non-fatal with a
        // registry whose original tool has silently been replaced.
        if self.factories.contains_key(&name) {
            return Err(Error::invalid(format!(
                "duplicate registered tool '{name}'"
            )));
        }
        self.factories.insert(name, factory);
        Ok(())
    }

    /// Builds each named tool, in the order given. Rejects a name requested
    /// twice or one no factory was registered under. Always builds with
    /// `Value::Null`: a factory that needs parameters must be built by
    /// calling [`ToolFactory::build`] directly, not through this resolver.
    pub fn resolve(&self, names: &[String]) -> Result<Vec<(String, RegisteredTool)>> {
        let mut seen = HashSet::new();
        names
            .iter()
            .map(|name| {
                if !seen.insert(name.clone()) {
                    return Err(Error::invalid(format!("duplicate requested tool '{name}'")));
                }
                let factory = self
                    .factories
                    .get(name)
                    .ok_or_else(|| Error::invalid(format!("unknown registered tool '{name}'")))?;
                Ok((name.clone(), factory.build(&serde_json::Value::Null)?))
            })
            .collect()
    }
}
