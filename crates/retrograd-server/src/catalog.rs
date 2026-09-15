//! The operator-declared catalogue: rewards, judges and MCP servers.
//!
//! The client never supplies a command, an endpoint or a key - it references an
//! `id`. The catalogue is therefore both the security boundary and the server's
//! extension point: adding a tool to the agent is a `[[mcp_server]]` entry,
//! adding a judge is a `[[judge]]` entry, and there is no code to write for
//! either.
//!
//! Declarations reuse `retrograd_agent`'s own config types (`JudgeConfig`,
//! `McpServerConfig`) rather than redeclaring them: the server is one more
//! frontend for those types and has no schema of its own.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use retrograd_agent::judge::JudgeConfig;
use retrograd_agent::tools::{McpServerConfig, McpToolProvider, ToolProvider};
use retrograd_agent::{EnvironmentConfig, Environments};
use retrograd_core::{Error, Result as CoreResult, RewardMode, RewardProtocol};
use serde::Deserialize;

use crate::dto;

/// One `[[reward]]` / `[[judge]]` / `[[mcp_server]]` table.
///
/// The `id` and `description` are the catalogue's own fields; everything else is
/// captured verbatim and handed to the matching `retrograd_agent` type. Keeping
/// the rest untyped here is what lets the agent's schema evolve without this
/// file changing.
#[derive(Clone, Debug, Deserialize)]
pub struct CatalogDeclaration {
    pub id: String,
    #[serde(default)]
    pub description: String,
    #[serde(flatten)]
    pub fields: BTreeMap<String, toml::Value>,
}

impl CatalogDeclaration {
    fn require_id(&self) -> CoreResult<()> {
        if self.id.trim().is_empty() {
            return Err(Error::config("a catalog entry needs a non-empty id"));
        }
        Ok(())
    }

    /// Converts the captured fields into JSON so they can be fed to the agent's
    /// `serde` implementations, which are written against JSON shapes.
    fn as_json(&self) -> CoreResult<serde_json::Map<String, serde_json::Value>> {
        let mut object = serde_json::Map::new();
        for (key, value) in &self.fields {
            let json = serde_json::to_value(value).map_err(|error| {
                Error::config(format!(
                    "catalog entry '{}' has an unrepresentable value for '{key}': {error}",
                    self.id
                ))
            })?;
            object.insert(key.clone(), json);
        }
        Ok(object)
    }
}

/// A declared reward command, ready to be substituted into a `PpoConfig` or
/// `GrpoConfig` by the resolver.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RewardFields {
    command: Vec<String>,
    #[serde(default)]
    cwd: Option<PathBuf>,
    #[serde(default = "default_reward_timeout")]
    timeout_seconds: u64,
    /// How the command is spoken to. Declared here because the client never
    /// sees the command and therefore cannot say: a script that reads its stdin
    /// to the end needs `mode = "oneshot"`, and only the operator knows which
    /// one it is.
    #[serde(default)]
    mode: RewardMode,
}

fn default_reward_timeout() -> u64 {
    30
}

#[derive(Clone, Debug)]
pub struct RewardEntry {
    pub id: String,
    pub description: String,
    pub command: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub protocol: RewardProtocol,
}

#[derive(Clone, Debug)]
pub struct JudgeEntry {
    pub id: String,
    pub description: String,
    pub config: JudgeConfig,
}

impl JudgeEntry {
    /// `ruler` or `command`, the only two shapes `JudgeConfig` has.
    pub fn kind(&self) -> &'static str {
        match self.config {
            JudgeConfig::Ruler { .. } => "ruler",
            JudgeConfig::Command { .. } => "command",
        }
    }

    /// What a recipe may set on this judge. Choices of *method* only: none of
    /// them has an effect outside the process, which is the criterion.
    pub fn client_settings(&self) -> Vec<&'static str> {
        match self.config {
            JudgeConfig::Ruler { .. } => vec![
                "mode",
                "max_pairs",
                "anchor",
                "rubric",
                "pairwise_rubric",
                "context",
            ],
            // A command judge runs an operator-declared subprocess and takes no
            // method parameters at all.
            JudgeConfig::Command { .. } => Vec::new(),
        }
    }
}

/// A declared MCP server plus what connecting to it actually revealed.
#[derive(Clone, Debug)]
pub struct McpEntry {
    pub id: String,
    pub description: String,
    pub config: McpServerConfig,
    /// `true` once the server answered and its tools were listed.
    pub connected: bool,
    /// Tool names as the policy sees them, already filtered by
    /// `allowed_tools`/`denied_tools`.
    pub tools: Vec<dto::ToolEntry>,
    pub warnings: Vec<String>,
}

/// One `[[environment]]` entry: what a trajectory acts on, declared by the
/// operator and referenced by the client with an id only.
///
/// This asymmetry is a security boundary rather than a convention: letting a
/// client pick its own image, mount, or
/// network mode is letting it have the machine.
#[derive(Debug)]
pub struct EnvironmentEntry {
    pub id: String,
    pub description: String,
    pub config: EnvironmentConfig,
}

impl EnvironmentEntry {
    pub fn kind(&self) -> &'static str {
        match self.config {
            EnvironmentConfig::Http(_) => "http",
            EnvironmentConfig::Container(_) => "container",
            EnvironmentConfig::Local(_) => "local",
        }
    }

    /// The tool names a rollout will see, when the kind is one whose tools are
    /// known without contacting anything. An HTTP environment's list belongs to
    /// its server, so it is reported empty rather than invented.
    pub fn tool_names(&self) -> Vec<String> {
        let tools = match &self.config {
            EnvironmentConfig::Local(local) => retrograd_agent::tools::ToolSet::builder()
                .with_profile(local.profile)
                .deny(local.deny_tools.clone())
                .build(),
            #[cfg(feature = "container")]
            EnvironmentConfig::Container(container) => {
                container.tools(retrograd_agent::tools::ToolSet::builder())
            }
            _ => return Vec::new(),
        };
        tools
            .map(|set| set.specs().iter().map(|spec| spec.name.clone()).collect())
            .unwrap_or_default()
    }
}

/// Everything the operator declared, indexed by id.
///
/// `BTreeMap` rather than `HashMap`: every listing endpoint is then ordered by
/// id without sorting at the edge, which is what makes the responses byte-stable.
#[derive(Debug, Default)]
pub struct Catalog {
    rewards: BTreeMap<String, RewardEntry>,
    judges: BTreeMap<String, JudgeEntry>,
    mcp_servers: BTreeMap<String, McpEntry>,
    environments: BTreeMap<String, EnvironmentEntry>,
}

impl Catalog {
    /// Builds the catalogue from declarations, without contacting anything.
    ///
    /// Used by the tests and by `connect`, which adds the MCP round trip on top.
    pub fn declare(
        rewards: &[CatalogDeclaration],
        judges: &[CatalogDeclaration],
        mcp_servers: &[CatalogDeclaration],
        environments: &[CatalogDeclaration],
    ) -> CoreResult<Self> {
        let mut catalog = Self::default();
        for declaration in rewards {
            declaration.require_id()?;
            let fields: RewardFields = deserialize_entry("reward", declaration)?;
            if fields.command.is_empty() {
                return Err(Error::config(format!(
                    "reward '{}' has an empty command",
                    declaration.id
                )));
            }
            if fields.timeout_seconds == 0 {
                return Err(Error::config(format!(
                    "reward '{}' has a zero timeout",
                    declaration.id
                )));
            }
            let entry = RewardEntry {
                id: declaration.id.clone(),
                description: declaration.description.clone(),
                command: fields.command,
                cwd: fields.cwd,
                protocol: RewardProtocol {
                    mode: fields.mode,
                    timeout: Duration::from_secs(fields.timeout_seconds),
                },
            };
            insert_unique(&mut catalog.rewards, "reward", entry.id.clone(), entry)?;
        }
        for declaration in judges {
            declaration.require_id()?;
            let mut object = declaration.as_json()?;
            // `[[judge]]` entries name their endpoint and model directly, the way
            // the TOML example in the plan does. `JudgeConfig` needs a `type`, and
            // `ruler` is the only shape that has an endpoint at all, so an absent
            // type means RULER rather than an error.
            object
                .entry("type")
                .or_insert_with(|| serde_json::Value::String("ruler".to_string()));
            let config: JudgeConfig = serde_json::from_value(serde_json::Value::Object(object))
                .map_err(|error| Error::config(format!("judge '{}': {error}", declaration.id)))?;
            let entry = JudgeEntry {
                id: declaration.id.clone(),
                description: declaration.description.clone(),
                config,
            };
            insert_unique(&mut catalog.judges, "judge", entry.id.clone(), entry)?;
        }
        for declaration in mcp_servers {
            declaration.require_id()?;
            let mut object = declaration.as_json()?;
            // `McpServerConfig` keys a server by `name`; the catalogue keys it by
            // `id`. They are the same thing, and the tool names the policy sees
            // are prefixed with it, so they must not diverge.
            object.insert(
                "name".to_string(),
                serde_json::Value::String(declaration.id.clone()),
            );
            let config: McpServerConfig = serde_json::from_value(serde_json::Value::Object(object))
                .map_err(|error| {
                    Error::config(format!("mcp_server '{}': {error}", declaration.id))
                })?;
            config.validate().map_err(|error| {
                Error::config(format!("mcp_server '{}': {error}", declaration.id))
            })?;
            let entry = McpEntry {
                id: declaration.id.clone(),
                description: declaration.description.clone(),
                config,
                connected: false,
                tools: Vec::new(),
                warnings: Vec::new(),
            };
            insert_unique(
                &mut catalog.mcp_servers,
                "mcp_server",
                entry.id.clone(),
                entry,
            )?;
        }
        for declaration in environments {
            declaration.require_id()?;
            let config: EnvironmentConfig = deserialize_entry("environment", declaration)?;
            // Everything an operator can get wrong without a daemon - an image
            // the build cannot run, a pool that cannot converge, a local sandbox
            // asked for by omission - fails server startup. The client only ever
            // sends an id, so this is the *only* place those values are read.
            config.validate().map_err(|error| {
                Error::config(format!("environment '{}': {error}", declaration.id))
            })?;
            let entry = EnvironmentEntry {
                id: declaration.id.clone(),
                description: declaration.description.clone(),
                config,
            };
            insert_unique(
                &mut catalog.environments,
                "environment",
                entry.id.clone(),
                entry,
            )?;
        }
        Ok(catalog)
    }

    /// Declares the catalogue and connects to every MCP server, so the tool
    /// lists it reports are the ones a rollout will actually get.
    ///
    /// A `required` server that cannot be reached fails here, and therefore fails
    /// server startup: the availability of the training API depends on the
    /// availability of its required tools. That is deliberate.
    pub async fn connect(
        rewards: &[CatalogDeclaration],
        judges: &[CatalogDeclaration],
        mcp_servers: &[CatalogDeclaration],
        environments: &[CatalogDeclaration],
    ) -> CoreResult<Self> {
        let mut catalog = Self::declare(rewards, judges, mcp_servers, environments)?;
        if catalog.mcp_servers.is_empty() {
            return Ok(catalog);
        }
        let configs: Vec<McpServerConfig> = catalog
            .mcp_servers
            .values()
            .map(|entry| entry.config.clone())
            .collect();
        let provider = McpToolProvider::connect(configs)
            .await
            .map_err(|error| Error::runtime(format!("MCP catalog: {error}")))?;
        let listed = provider
            .list_tools()
            .await
            .map_err(|error| Error::runtime(format!("MCP catalog: {error}")))?;
        // `McpToolProvider` exposes one flat, prefixed namespace
        // (`server__tool`), which is exactly how the policy sees it. Splitting on
        // that prefix attributes each tool back to its server without asking the
        // provider for a per-server view it does not have.
        for spec in listed {
            let Some((server, tool)) = spec.name.split_once("__") else {
                continue;
            };
            if let Some(entry) = catalog.mcp_servers.get_mut(server) {
                entry.tools.push(dto::ToolEntry {
                    name: tool.to_string(),
                    description: spec.description,
                });
            }
        }
        let warnings = provider.warnings().to_vec();
        for entry in catalog.mcp_servers.values_mut() {
            // A server with no tool listed is one the provider skipped: it was
            // optional and unreachable, or its filters exposed nothing.
            entry.connected = !entry.tools.is_empty();
            entry.warnings = warnings
                .iter()
                .filter(|warning| warning.contains(&format!("'{}'", entry.id)))
                .cloned()
                .collect();
        }
        // The provider owns child processes and HTTP sessions; the catalogue only
        // needed the tool list, so close them rather than hold them for the
        // server's lifetime.
        provider.shutdown().await;
        Ok(catalog)
    }

    pub fn reward(&self, id: &str) -> CoreResult<&RewardEntry> {
        self.rewards
            .get(id)
            .ok_or_else(|| unknown_id("reward", id, self.rewards.keys()))
    }

    pub fn judge(&self, id: &str) -> CoreResult<&JudgeEntry> {
        self.judges
            .get(id)
            .ok_or_else(|| unknown_id("judge", id, self.judges.keys()))
    }

    pub fn mcp_server(&self, id: &str) -> CoreResult<&McpEntry> {
        self.mcp_servers
            .get(id)
            .ok_or_else(|| unknown_id("mcp_server", id, self.mcp_servers.keys()))
    }

    pub fn environment(&self, id: &str) -> CoreResult<&EnvironmentEntry> {
        self.environments
            .get(id)
            .ok_or_else(|| unknown_id("environment", id, self.environments.keys()))
    }

    /// The listing for `GET /v1/environments`.
    ///
    /// Deliberately thin: the kind and the tools, never the image, the mounts
    /// or the limits. A client that could read them would be one step from
    /// choosing them, and choosing an image is choosing what runs on the
    /// operator's machine.
    pub fn environment_listing(&self) -> dto::Environments {
        dto::Environments {
            environments: self
                .environments
                .values()
                .map(|entry| dto::EnvironmentEntry {
                    id: entry.id.clone(),
                    description: entry.description.clone(),
                    kind: entry.kind(),
                    tools: entry.tool_names(),
                })
                .collect(),
        }
    }

    /// The listing for `GET /v1/rewards`. Commands and working directories are
    /// absent: the catalogue publishes what a client may reference, not what the
    /// server will run.
    pub fn reward_listing(&self) -> dto::Rewards {
        dto::Rewards {
            rewards: self
                .rewards
                .values()
                .map(|entry| dto::RewardEntry {
                    id: entry.id.clone(),
                    description: entry.description.clone(),
                    timeout_seconds: entry.protocol.timeout.as_secs(),
                    mode: entry.protocol.mode.as_str(),
                })
                .collect(),
        }
    }

    /// The listing for `GET /v1/judges`. No endpoint, no model, no key name.
    pub fn judge_listing(&self) -> dto::Judges {
        dto::Judges {
            judges: self
                .judges
                .values()
                .map(|entry| dto::JudgeEntry {
                    id: entry.id.clone(),
                    description: entry.description.clone(),
                    kind: entry.kind(),
                    client_settings: entry.client_settings(),
                })
                .collect(),
        }
    }

    /// The listing for `GET /v1/mcp-servers`, including the verified tool list.
    pub fn mcp_listing(&self) -> dto::McpServers {
        dto::McpServers {
            mcp_servers: self
                .mcp_servers
                .values()
                .map(|entry| dto::McpServerEntry {
                    id: entry.id.clone(),
                    description: entry.description.clone(),
                    required: entry.config.required,
                    status: if entry.connected {
                        "connected"
                    } else {
                        "skipped"
                    },
                    tools: entry.tools.clone(),
                    warnings: entry.warnings.clone(),
                })
                .collect(),
        }
    }

    /// Whether any RL or agentic run is possible at all. `POST /v1/plan` says so
    /// explicitly rather than letting a recipe fail deep in resolution.
    pub fn is_empty(&self) -> bool {
        self.rewards.is_empty()
            && self.judges.is_empty()
            && self.mcp_servers.is_empty()
            && self.environments.is_empty()
    }
}

/// Shared by the three families so an unknown id always answers with what *is*
/// available - a 422 that lists nothing is a dead end for the caller.
fn unknown_id<'a>(family: &str, id: &str, available: impl Iterator<Item = &'a String>) -> Error {
    let available: Vec<&str> = available.map(String::as_str).collect();
    if available.is_empty() {
        Error::invalid(format!("unknown {family} '{id}': the server declares none"))
    } else {
        Error::invalid(format!(
            "unknown {family} '{id}'; declared: {}",
            available.join(", ")
        ))
    }
}

fn insert_unique<T>(
    target: &mut BTreeMap<String, T>,
    family: &str,
    id: String,
    entry: T,
) -> CoreResult<()> {
    if target.insert(id.clone(), entry).is_some() {
        return Err(Error::config(format!("duplicate {family} id '{id}'")));
    }
    Ok(())
}

fn deserialize_entry<T: serde::de::DeserializeOwned>(
    family: &str,
    declaration: &CatalogDeclaration,
) -> CoreResult<T> {
    let object = declaration.as_json()?;
    serde_json::from_value(serde_json::Value::Object(object))
        .map_err(|error| Error::config(format!("{family} '{}': {error}", declaration.id)))
}

/// Arc-wrapping helper so callers can share one catalogue across handlers.
pub fn shared(catalog: Catalog) -> Arc<Catalog> {
    Arc::new(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ServerConfig;

    fn config(text: &str) -> ServerConfig {
        toml::from_str(text).expect("parse server config")
    }

    fn declared(text: &str) -> CoreResult<Catalog> {
        let config = config(text);
        Catalog::declare(
            &config.rewards,
            &config.judges,
            &config.mcp_servers,
            &config.environments,
        )
    }

    /// The operator declares the image, the limits and the pool; the client
    /// gets an id, a kind and a tool list. That asymmetry is the point of the
    /// family, so the listing is asserted to *not* carry the rest.
    #[test]
    fn an_environment_is_declared_by_the_operator_and_referenced_by_id() {
        let catalog = declared(
            r#"
[[environment]]
id = "py-sandbox"
description = "Python tasks, no network"
type = "local"
profile = "python"
allow_unsandboxed = true
deny_tools = ["bash"]
"#,
        )
        .expect("declare catalog");

        let entry = catalog.environment("py-sandbox").expect("environment");
        assert_eq!(entry.kind(), "local");
        assert!(entry.tool_names().contains(&"submit".to_owned()));
        assert!(!entry.tool_names().contains(&"bash".to_owned()));

        let listing = catalog.environment_listing();
        assert_eq!(listing.environments.len(), 1);
        let published = serde_json::to_string(&listing).unwrap();
        for secret in ["allow_unsandboxed", "profile", "image", "limits"] {
            assert!(!published.contains(secret), "{secret} leaked: {published}");
        }

        // An unknown id answers with what *is* declared rather than a dead end.
        let error = catalog.environment("nope").unwrap_err().to_string();
        assert!(error.contains("py-sandbox"), "{error}");
    }

    /// Every way an environment declaration can be wrong fails server startup.
    /// The client cannot fix any of them, and discovering one when a run starts
    /// would mean the API was up while advertising something it cannot serve.
    #[test]
    fn a_broken_environment_declaration_fails_at_startup() {
        for (body, expected) in [
            // A local sandbox is never obtained by omission.
            ("type = \"local\"\n", "allow_unsandboxed"),
            // A typo is a typo.
            (
                "type = \"local\"\nallow_unsandboxd = true\n",
                "unknown field",
            ),
            ("type = \"nowhere\"\n", "unknown variant"),
        ] {
            let source = format!("[[environment]]\nid = \"e\"\n{body}");
            let error = declared(&source)
                .expect_err(&format!("accepted {body}"))
                .to_string();
            assert!(
                error.contains("environment 'e'") || error.contains(expected),
                "{error}"
            );
            assert!(error.contains(expected), "{error}");
        }

        // Two entries with one id would make a reference ambiguous.
        let duplicate = concat!(
            "[[environment]]\nid = \"e\"\ntype = \"local\"\nallow_unsandboxed = true\n",
            "[[environment]]\nid = \"e\"\ntype = \"local\"\nallow_unsandboxed = true\n",
        );
        assert!(matches!(declared(duplicate), Err(Error::Config(_))));
    }

    #[test]
    fn the_plan_example_declares_one_of_each_family() {
        // Verbatim from the plan example: if that example stops parsing, the
        // documentation is wrong and so is the operator following it.
        let catalog = declared(
            r#"
[[reward]]
id = "sql-exec"
description = "Runs the generated query against the test database"
command = ["python", "rewards/sql_exec.py"]
cwd = "."
timeout_seconds = 30

[[judge]]
id = "ruler-mini"
description = "gpt-5-mini, default rubric"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[[mcp_server]]
id = "calc"
command = ["python", "rewards/calculator_server.py"]
denied_tools = ["write_*"]
required = true
"#,
        )
        .expect("declare catalog");

        let reward = catalog.reward("sql-exec").expect("reward");
        assert_eq!(reward.command, ["python", "rewards/sql_exec.py"]);
        assert_eq!(reward.protocol.timeout, Duration::from_secs(30));
        assert_eq!(reward.protocol.mode, RewardMode::Persistent);

        let judge = catalog.judge("ruler-mini").expect("judge");
        assert_eq!(judge.kind(), "ruler");
        assert!(judge.client_settings().contains(&"max_pairs"));

        let mcp = catalog.mcp_server("calc").expect("mcp server");
        assert_eq!(mcp.config.name, "calc");
        assert!(mcp.config.required);
        assert_eq!(
            mcp.config.denied_tools.as_deref(),
            Some(&["write_*".to_string()][..])
        );
        assert!(!catalog.is_empty());
    }

    #[test]
    fn listings_never_carry_a_command_an_endpoint_or_a_key() {
        let catalog = declared(
            r#"
[[reward]]
id = "sql-exec"
command = ["python", "secret_script.py"]

[[judge]]
id = "ruler-mini"
base_url = "https://internal.example/v1"
model = "gpt-5-mini"
api_key_env = "SECRET_KEY_VAR"

[[mcp_server]]
id = "calc"
command = ["python", "another_secret.py"]
"#,
        )
        .expect("declare catalog");

        // One serialized blob covering all three listings: the property is that
        // *nothing* leaks, so testing them separately would invite a gap.
        let rendered = [
            serde_json::to_string(&catalog.reward_listing()).unwrap(),
            serde_json::to_string(&catalog.judge_listing()).unwrap(),
            serde_json::to_string(&catalog.mcp_listing()).unwrap(),
        ]
        .join("");
        for secret in [
            "secret_script.py",
            "another_secret.py",
            "internal.example",
            "SECRET_KEY_VAR",
            "gpt-5-mini",
            "command",
            "base_url",
            "api_key_env",
        ] {
            assert!(
                !rendered.contains(secret),
                "listing leaked '{secret}': {rendered}"
            );
        }
        // What it does carry: the ids a client references.
        assert!(rendered.contains("sql-exec"));
        assert!(rendered.contains("ruler-mini"));
        assert!(rendered.contains("calc"));
    }

    #[test]
    fn declarations_are_validated_and_ids_are_unique() {
        // An empty command would only fail when a run tried to use it.
        assert!(matches!(
            declared("[[reward]]\nid = 'r'\ncommand = []\n"),
            Err(Error::Config(_))
        ));
        assert!(declared("[[reward]]\nid = 'r'\ncommand = ['x']\ntimeout_seconds = 0\n").is_err());
        assert!(declared("[[reward]]\nid = ' '\ncommand = ['x']\n").is_err());
        assert!(
            declared(
                "[[reward]]\nid = 'r'\ncommand = ['x']\n[[reward]]\nid = 'r'\ncommand = ['y']\n"
            )
            .is_err()
        );
        // A typo in a field name must not be absorbed silently.
        assert!(declared("[[reward]]\nid = 'r'\ncommand = ['x']\ntimeout = 5\n").is_err());
        // A RULER judge without an endpoint is not usable.
        assert!(declared("[[judge]]\nid = 'j'\nmodel = 'm'\n").is_err());
        // An MCP server needs a transport.
        assert!(declared("[[mcp_server]]\nid = 'm'\n").is_err());
    }

    #[test]
    fn an_unknown_id_names_what_is_available() {
        let catalog = declared("[[reward]]\nid = 'sql-exec'\ncommand = ['x']\n").unwrap();
        let error = catalog.reward("nope").unwrap_err().to_string();
        assert!(error.contains("unknown reward 'nope'"), "{error}");
        assert!(error.contains("sql-exec"), "{error}");

        let empty = Catalog::default();
        assert!(empty.is_empty());
        assert!(
            empty
                .judge("nope")
                .unwrap_err()
                .to_string()
                .contains("declares none")
        );
    }

    #[test]
    fn a_command_judge_is_declared_with_an_explicit_type() {
        let catalog =
            declared("[[judge]]\nid = 'j'\ntype = 'command'\ncommand = ['python', 'judge.py']\n")
                .expect("declare catalog");
        let judge = catalog.judge("j").expect("judge");
        assert_eq!(judge.kind(), "command");
        // Nothing about a subprocess judge is a client-side method choice.
        assert!(judge.client_settings().is_empty());
    }
}
