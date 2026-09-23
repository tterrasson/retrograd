//! What a configuration file says about the world a trajectory acts on.
//!
//! ```toml
//! [agent.environment]
//! type = "container"        # "http" | "container" | "local"
//! profile = "python"        # or image = "…@sha256:…"
//!
//! [agent.environment.pool]
//! max_live = 8
//! reuse = "workspace"
//! ```
//!
//! Instantiating one is `retrograd-env`'s business - it connects, pulls an
//! image, spawns a process - and so is saying "this binary was built without
//! container support". Reading a document must not require being able to run it:
//! the planner and the server read configurations for machines that are not
//! theirs.

use serde::{Deserialize, Serialize};

use retrograd_agent_core::{Error, Result};

use crate::tools::Profile;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnvironmentConfig {
    /// The world lives behind `reset`/`step`/`state`/`close` over HTTP - also
    /// how an existing OpenEnv server plugs in.
    Http(HttpEnvironmentConfig),
    /// A pool of containers on the local Docker daemon.
    Container(ContainerConfig),
    /// The host, with no isolation. Refused unless `allow_unsandboxed` is set.
    Local(LocalConfig),
}

/// The `container` table, whether or not the binary can honour it.
///
/// Parsed even without the feature, so the error is "not compiled in" rather
/// than "unknown variant" - the operator's configuration is right, the build is
/// not, and only one of those two sentences says which.
#[cfg(feature = "container")]
pub type ContainerConfig = crate::env::ContainerEnvironmentConfig;

#[cfg(not(feature = "container"))]
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContainerConfig {
    #[serde(flatten)]
    ignored: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LocalConfig {
    pub profile: Profile,
    pub tools: Option<Vec<String>>,
    /// Must be typed by someone. Model-generated code runs on the training
    /// machine with the training process's privileges and network.
    pub allow_unsandboxed: bool,
    pub deny_tools: Vec<String>,
    pub setup_timeout_secs: u64,
    pub verify_timeout_secs: Option<u64>,
}

impl Default for LocalConfig {
    fn default() -> Self {
        Self {
            profile: Profile::default(),
            tools: None,
            allow_unsandboxed: false,
            deny_tools: Vec::new(),
            setup_timeout_secs: 300,
            verify_timeout_secs: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HttpEnvironmentConfig {
    /// Base URL the four routes hang off, e.g. `http://127.0.0.1:8080`.
    pub base_url: String,
    /// Per-request budget. Distinct from `RolloutLimits::max_rollout_secs`,
    /// which bounds the whole trajectory: one hung `step` must not consume it.
    #[serde(default = "default_request_timeout")]
    pub request_timeout_secs: u64,
    #[serde(default = "default_connect_timeout")]
    pub connect_timeout_secs: u64,
    /// Extra headers on every request - an authorization token, a tenant id.
    #[serde(default)]
    pub headers: std::collections::BTreeMap<String, String>,
    /// Cap on an observation's size, applied before it enters the prompt. Same
    /// reason as the MCP provider's: an unbounded tool result silently eats the
    /// trajectory's whole token budget.
    #[serde(default = "default_max_result_bytes")]
    pub max_result_bytes: usize,
    /// Concurrent connections kept alive to the environment server. A group of
    /// N runs N environments at once, so a pool below the group size serializes
    /// the turn that batching just made parallel.
    #[serde(default = "default_pool_size")]
    pub pool_size: usize,
}

fn default_request_timeout() -> u64 {
    60
}

fn default_connect_timeout() -> u64 {
    10
}

fn default_max_result_bytes() -> usize {
    64 * 1024
}

fn default_pool_size() -> usize {
    16
}

impl HttpEnvironmentConfig {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            request_timeout_secs: default_request_timeout(),
            connect_timeout_secs: default_connect_timeout(),
            headers: Default::default(),
            max_result_bytes: default_max_result_bytes(),
            pool_size: default_pool_size(),
        }
    }

    pub fn with_request_timeout(mut self, seconds: u64) -> Self {
        self.request_timeout_secs = seconds;
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.base_url.trim().is_empty() {
            return Err(Error::invalid("environment base_url must not be empty"));
        }
        let Some(authority) = self
            .base_url
            .strip_prefix("http://")
            .or_else(|| self.base_url.strip_prefix("https://"))
        else {
            return Err(Error::invalid(
                "environment base_url must be an http:// or https:// URL",
            ));
        };
        // The scheme alone is not a URL: `http:///step` would build a request
        // against no host and fail at the first rollout instead of here.
        if authority.split('/').next().unwrap_or_default().is_empty() {
            return Err(Error::invalid(
                "environment base_url must name a host after its scheme",
            ));
        }
        if self.request_timeout_secs == 0
            || self.connect_timeout_secs == 0
            || self.max_result_bytes == 0
            || self.pool_size == 0
        {
            return Err(Error::invalid(
                "environment timeouts, result cap and pool size must be greater than zero",
            ));
        }
        Ok(())
    }
}

// The container declaration is the one part of the schema that cannot be
// feature-free: its limits *are* `retrograd_container::spec::SpecLimits`, so
// typing the table costs the crate that defines them. Without the feature the
// table parses into an opaque map, exactly as it did in `retrograd-env` before
// this move - the operator's configuration is right, the build is not.
#[cfg(feature = "container")]
mod container {
    use serde::{Deserialize, Serialize};

    use retrograd_container::{PoolConfig, ReusePolicy};

    use crate::tools::Profile;

    /// The serialized shape shared by the TOML frontend, the server catalogue and
    /// the Python binding.
    #[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    pub struct ContainerEnvironmentConfig {
        pub profile: Profile,
        pub tools: Option<Vec<String>>,
        /// Overrides the profile's image. Prefer `name@sha256:…`: a run resumed
        /// three weeks later on a moving tag is not the same environment.
        pub image: Option<String>,
        /// Off by default. A task that needs the network says so, and says it in the
        /// operator's configuration rather than in a scenario.
        pub allow_network: bool,
        pub limits: retrograd_container::spec::SpecLimits,
        pub pool: ContainerPoolConfig,
        /// Extra tools beyond the profile's are not configurable - that is the
        /// programmatic extension point. Removing one is.
        pub deny_tools: Vec<String>,
        /// Mounted read-only, keyed by target path. Defaults to the profile's
        /// package cache: one `pip install` per run instead of one per episode,
        /// which is the real payoff of container reuse.
        pub cache_volume: Option<String>,
        pub setup_timeout_secs: u64,
        pub verify_timeout_secs: Option<u64>,
        /// Identifies this run's containers, so a crash leaves something the reaper
        /// can recognize. Defaults to a value derived from the process id.
        pub run_id: Option<String>,
    }

    impl Default for ContainerEnvironmentConfig {
        fn default() -> Self {
            Self {
                profile: Profile::default(),
                tools: None,
                image: None,
                allow_network: false,
                limits: retrograd_container::spec::SpecLimits::default(),
                pool: ContainerPoolConfig::default(),
                deny_tools: Vec::new(),
                cache_volume: None,
                setup_timeout_secs: 300,
                verify_timeout_secs: None,
                run_id: None,
            }
        }
    }

    /// [`PoolConfig`] in its serialized form, so the TOML table maps onto
    /// one type instead of being re-declared per frontend.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[serde(default, deny_unknown_fields)]
    pub struct ContainerPoolConfig {
        pub max_live: usize,
        pub min_idle: usize,
        pub reuse: ReusePolicy,
        pub max_leases_per_container: u32,
    }

    impl Default for ContainerPoolConfig {
        fn default() -> Self {
            let defaults = PoolConfig::default();
            Self {
                max_live: defaults.max_live,
                min_idle: defaults.min_idle,
                reuse: defaults.reuse,
                max_leases_per_container: defaults.max_leases_per_container,
            }
        }
    }

    impl From<ContainerPoolConfig> for PoolConfig {
        fn from(config: ContainerPoolConfig) -> Self {
            Self {
                max_live: config.max_live,
                min_idle: config.min_idle,
                reuse: config.reuse,
                max_leases_per_container: config.max_leases_per_container,
            }
        }
    }
}

#[cfg(feature = "container")]
pub use container::{ContainerEnvironmentConfig, ContainerPoolConfig};

#[cfg(feature = "container")]
impl ContainerEnvironmentConfig {
    /// The budgets a container declaration has to respect. What it *cannot*
    /// check here is the image and the tool set, which need the container spec
    /// and the built-in registry: `retrograd_env::ContainerEnvironment::validate`
    /// adds those.
    pub fn validate_declaration(&self) -> Result<()> {
        if self.setup_timeout_secs == 0 {
            return Err(Error::invalid("setup_timeout_secs must be positive"));
        }
        if self.pool.max_live == 0 {
            return Err(Error::invalid("pool.max_live must be positive"));
        }
        if self.pool.min_idle > self.pool.max_live {
            return Err(Error::invalid(
                "pool.min_idle cannot exceed pool.max_live: the pool would warm containers it \
                 is not allowed to hold",
            ));
        }
        if self.pool.max_leases_per_container == 0 {
            return Err(Error::invalid(
                "pool.max_leases_per_container must be positive",
            ));
        }
        Ok(())
    }
}

impl EnvironmentConfig {
    /// The tool selection this environment declares: the profile, the explicit
    /// list, and what the operator denied.
    pub fn tool_selection(&self) -> (Profile, Option<Vec<String>>, crate::tools::ToolFilter) {
        match self {
            Self::Local(config) => (
                config.profile,
                config.tools.clone(),
                crate::tools::ToolFilter {
                    allow: None,
                    deny: config.deny_tools.clone(),
                },
            ),
            #[cfg(feature = "container")]
            Self::Container(config) => (
                config.profile,
                config.tools.clone(),
                crate::tools::ToolFilter {
                    allow: None,
                    deny: config.deny_tools.clone(),
                },
            ),
            _ => (Profile::Custom, Some(Vec::new()), Default::default()),
        }
    }

    /// Is this a well-formed *declaration*?
    ///
    /// This is as far as this crate goes, deliberately: reading a document must
    /// not require being able to run it, since the planner and the server read
    /// configurations for machines that are not theirs. Whether this build has a
    /// container backend, and whether the tools named exist in its registry, are
    /// `retrograd_env::Environments::validate`'s sentences to say.
    pub fn validate_declaration(&self) -> Result<()> {
        match self {
            Self::Http(config) => config.validate(),
            Self::Container(config) => validate_container(config),
            Self::Local(config) => {
                if !config.allow_unsandboxed {
                    return Err(Error::invalid(
                        "a local environment executes model-generated code on the host with no \
                         isolation; set allow_unsandboxed to run without a container",
                    ));
                }
                Ok(())
            }
        }
    }
}

// Without the feature the container table parsed into an opaque map, so there
// is nothing left to check and nothing to complain about yet.
#[cfg(feature = "container")]
fn validate_container(config: &ContainerConfig) -> Result<()> {
    config.validate_declaration()
}

#[cfg(not(feature = "container"))]
fn validate_container(_config: &ContainerConfig) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_base_url_needs_a_scheme_and_a_host() {
        let with = HttpEnvironmentConfig::new;
        assert!(with("http://localhost:8000").validate().is_ok());
        assert!(with("https://env.example/prefix").validate().is_ok());
        for bad in ["", "  ", "ftp://env.example", "env.example", "http:///step"] {
            assert!(with(bad).validate().is_err(), "{bad}");
        }
    }
}
