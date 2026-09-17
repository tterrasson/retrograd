//! One declaration of what a trajectory acts on, for all three frontends.
//!
//! The TOML binary, the server catalogue and the Python binding share
//! [`EnvironmentConfig`], so `type = "container"` means the same thing - and
//! validates the same way - wherever it is written:
//!
//! ```toml
//! [grpo.agent.environment]
//! type = "container"        # "http" | "container" | "local"
//! profile = "python"        # or image = "…@sha256:…"
//!
//! [grpo.agent.environment.pool]
//! max_live = 8
//! reuse = "workspace"
//! ```
//!
//! `container` is behind a compile-time feature. A binary built without it
//! **says so** when the configuration asks for one, at load time: an
//! unsupported environment must not be a panic, and must not be a silent
//! fallback to something weaker.

use std::sync::Arc;

use retrograd_agent_core::Result;
use retrograd_agent_core::env::EnvironmentFactory;
// Only `unsupported` below names this type, and it is compiled out of a build
// that has every backing. Same `cfg` on the import, so the build that drops the
// function drops its import with it rather than warning about it.
#[cfg(not(all(feature = "http-env", feature = "container", feature = "local-sandbox")))]
use retrograd_agent_core::Error;

pub use retrograd_spec::env::{
    ContainerConfig, EnvironmentConfig, HttpEnvironmentConfig, LocalConfig,
};

/// What a declared environment can do in *this* build: check itself against the
/// features compiled in, and instantiate a factory.
///
/// The declaration is `retrograd-spec`'s. Everything
/// below needs either a tool registry, a daemon or a socket, which is why it is
/// here - and why a binary that only reads a configuration links none of them.
pub trait Environments {
    /// The tool profile this declaration selects, the registry names an
    /// explicit `tools` list resolves to (`None` when the profile's own set is
    /// used unfiltered), and the `deny_tools` filter. `Http` declarations carry
    /// no tool selection of their own, so they answer with `Profile::Custom`
    /// and an empty list.
    fn tool_selection(
        &self,
    ) -> (
        retrograd_tools::Profile,
        Option<Vec<String>>,
        retrograd_tools::ToolFilter,
    );
    fn validate(&self) -> Result<()>;
    fn validate_declaration(&self) -> Result<()>;
    fn build(&self) -> impl std::future::Future<Output = Result<Arc<dyn EnvironmentFactory>>>;
}

impl Environments for EnvironmentConfig {
    fn tool_selection(
        &self,
    ) -> (
        retrograd_tools::Profile,
        Option<Vec<String>>,
        retrograd_tools::ToolFilter,
    ) {
        match self {
            Self::Local(config) => (
                config.profile,
                config
                    .tools
                    .as_ref()
                    .map(|tools| config.profile.registry_names(tools)),
                retrograd_tools::ToolFilter {
                    allow: None,
                    deny: config.deny_tools.clone(),
                },
            ),
            #[cfg(feature = "container")]
            Self::Container(config) => (
                config.profile,
                config
                    .tools
                    .as_ref()
                    .map(|tools| config.profile.registry_names(tools)),
                retrograd_tools::ToolFilter {
                    allow: None,
                    deny: config.deny_tools.clone(),
                },
            ),
            _ => (
                retrograd_tools::Profile::Custom,
                Some(Vec::new()),
                retrograd_tools::ToolFilter::default(),
            ),
        }
    }

    /// Everything that can be decided without touching the network: the shape
    /// of the configuration, and whether this binary can honour it at all.
    ///
    /// Separate from [`build`](Self::build) because a catalogue validates what
    /// an operator declared at startup, long before anything runs - and because
    /// "this binary has no container support" is a sentence that must not wait
    /// for a daemon connection to be said.
    fn validate(&self) -> Result<()> {
        self.validate_declaration()?;
        match self {
            Self::Http(_) => supports_http(),
            Self::Container(_) => supports_container(),
            Self::Local(_) => supports_local(),
        }
    }

    /// The declaration's own checks, plus the ones that need a registry: an
    /// environment naming a tool this build does not carry is a configuration
    /// error, and it is *this* crate that knows the registry.
    fn validate_declaration(&self) -> Result<()> {
        EnvironmentConfig::validate_declaration(self)?;
        match self {
            Self::Local(config) => selected_builder(config.profile, config.tools.as_ref())?
                .deny(config.deny_tools.clone())
                .build()
                .map(|_| ()),
            Self::Container(config) => validate_container(config),
            Self::Http(_) => Ok(()),
        }
    }

    /// Builds the factory. Async because a container pool connects to the
    /// daemon, pulls its image and reaps leftovers here - before the first
    /// rollout, on purpose.
    async fn build(&self) -> Result<Arc<dyn EnvironmentFactory>> {
        self.validate()?;
        match self {
            Self::Http(config) => build_http(config),
            Self::Container(config) => build_container(config).await,
            Self::Local(config) => build_local(config),
        }
    }
}

// The half of a container declaration that needs the container spec and the
// built-in registry. Without the feature the table parsed into an opaque map, so
// there is nothing left to check and nothing to complain about yet.
#[cfg(feature = "container")]
fn validate_container(config: &ContainerConfig) -> Result<()> {
    crate::container::ContainerEnvironment::validate(config)
}

#[cfg(not(feature = "container"))]
fn validate_container(_config: &ContainerConfig) -> Result<()> {
    Ok(())
}

// Capability checks: what this build can actually instantiate.
fn supports_http() -> Result<()> {
    #[cfg(feature = "http-env")]
    return Ok(());
    #[cfg(not(feature = "http-env"))]
    return Err(unsupported("http", "http-env"));
}

fn supports_container() -> Result<()> {
    #[cfg(feature = "container")]
    return Ok(());
    #[cfg(not(feature = "container"))]
    return Err(unsupported("container", "container"));
}

fn supports_local() -> Result<()> {
    #[cfg(feature = "local-sandbox")]
    return Ok(());
    #[cfg(not(feature = "local-sandbox"))]
    return Err(unsupported("local", "local-sandbox"));
}

#[cfg(feature = "http-env")]
fn build_http(config: &HttpEnvironmentConfig) -> Result<Arc<dyn EnvironmentFactory>> {
    Ok(Arc::new(crate::HttpEnvironmentFactory::new(
        config.clone(),
    )?))
}

#[cfg(not(feature = "http-env"))]
fn build_http(_config: &HttpEnvironmentConfig) -> Result<Arc<dyn EnvironmentFactory>> {
    Err(unsupported("http", "http-env"))
}

#[cfg(feature = "container")]
async fn build_container(config: &ContainerConfig) -> Result<Arc<dyn EnvironmentFactory>> {
    let (factory, _metrics) =
        crate::container::build_factory(config, retrograd_tools::ToolSet::builder()).await?;
    Ok(Arc::new(factory))
}

#[cfg(not(feature = "container"))]
async fn build_container(_config: &ContainerConfig) -> Result<Arc<dyn EnvironmentFactory>> {
    Err(unsupported("container", "container"))
}

#[cfg(feature = "local-sandbox")]
fn build_local(config: &LocalConfig) -> Result<Arc<dyn EnvironmentFactory>> {
    let tools = selected_builder(config.profile, config.tools.as_ref())?
        .deny(config.deny_tools.clone())
        .build()?;
    let provider = crate::LocalSandboxProvider::new(crate::LocalSandboxConfig {
        allow_unsandboxed: config.allow_unsandboxed,
        ..Default::default()
    })?;
    Ok(Arc::new(crate::SandboxEnvironmentFactory::new(
        provider,
        tools,
        crate::SandboxEnvironmentConfig {
            image: None,
            setup_timeout: std::time::Duration::from_secs(config.setup_timeout_secs),
            verify_timeout: config
                .verify_timeout_secs
                .map(std::time::Duration::from_secs),
            ..Default::default()
        },
    )?))
}

#[cfg(not(feature = "local-sandbox"))]
fn build_local(_config: &LocalConfig) -> Result<Arc<dyn EnvironmentFactory>> {
    Err(unsupported("local", "local-sandbox"))
}

fn selected_builder(
    profile: retrograd_tools::Profile,
    tools: Option<&Vec<String>>,
) -> Result<retrograd_tools::ToolSetBuilder> {
    match tools {
        Some(names) => retrograd_tools::ToolSet::builder().with_registry(
            &retrograd_tools::ToolRegistry::builtin(),
            &profile.registry_names(names),
        ),
        None => Ok(retrograd_tools::ToolSet::builder().with_profile(profile)),
    }
}

// Called only from the `#[cfg(not(feature = …))]` arms of the three capability
// checks above, so a build that compiles every backing in has no caller left.
// The `cfg` says which builds those are; `allow(dead_code)` said only "trust
// me", and would have gone on saying it after the last caller disappeared.
#[cfg(not(all(feature = "http-env", feature = "container", feature = "local-sandbox")))]
fn unsupported(kind: &str, feature: &str) -> Error {
    Error::invalid(format!(
        "environment type '{kind}' is not compiled into this binary: rebuild with \
         `--features {feature}`"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> std::result::Result<EnvironmentConfig, String> {
        toml::from_str(source).map_err(|error| error.to_string())
    }

    #[test]
    fn the_documented_container_table_parses_whatever_the_build_supports() {
        let config = parse(concat!(
            "type = 'container'\n",
            "profile = 'python'\n",
            "allow_network = false\n",
            "[limits]\ncpus = 1.0\nmemory_mb = 1024\npids = 256\n",
            "exec_timeout_secs = 30\nmax_output_bytes = 65536\n",
            "[pool]\nmax_live = 8\nmin_idle = 2\nreuse = 'workspace'\n",
            "max_leases_per_container = 32\n",
        ))
        .expect("the documented table must parse");
        assert!(matches!(config, EnvironmentConfig::Container(_)));
    }

    /// Selecting an environment the binary cannot provide is a configuration
    /// error with a sentence that says what to do, checked at load time.
    #[cfg(not(feature = "container"))]
    #[tokio::test]
    async fn a_container_environment_without_the_feature_says_so() {
        let config = parse("type = 'container'\nprofile = 'python'\n").unwrap();
        let error = match config.build().await {
            Ok(_) => panic!("must not build"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("--features container"), "{error}");
    }

    #[cfg(feature = "local-sandbox")]
    #[tokio::test]
    async fn a_local_environment_is_never_obtained_by_omission() {
        let config = parse("type = 'local'\n").unwrap();
        let error = match config.build().await {
            Ok(_) => panic!("must not build"),
            Err(error) => error.to_string(),
        };
        assert!(error.contains("allow_unsandboxed"), "{error}");

        let config =
            parse("type = 'local'\nallow_unsandboxed = true\ndeny_tools = ['bash']\n").unwrap();
        let factory = config.build().await.expect("an explicit local sandbox");
        let environment = factory.create().await.unwrap();
        let names = environment
            .tools()
            .await
            .unwrap()
            .into_iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>();
        assert!(!names.contains(&"bash".to_owned()) && names.contains(&"submit".to_owned()));
    }

    /// The Python binding serializes its dataclasses into exactly this, so the
    /// two shapes are pinned against each other here rather than discovered to
    /// disagree at the first `fit_agentic_grpo`.
    #[test]
    fn the_python_binding_s_json_is_the_same_configuration() {
        let container = serde_json::json!({
            "type": "container", "profile": "python", "image": null,
            "allow_network": false,
            "limits": {"cpus": 1.0, "memory_mb": 1024, "pids": 256,
                       "exec_timeout_secs": 30, "max_output_bytes": 65536},
            "pool": {"max_live": 8, "min_idle": 0, "reuse": "never",
                     "max_leases_per_container": 32},
            "deny_tools": ["bash"], "cache_volume": null,
            "setup_timeout_secs": 300, "verify_timeout_secs": null, "run_id": null
        });
        let config: EnvironmentConfig = serde_json::from_value(container).expect("container JSON");
        assert!(matches!(config, EnvironmentConfig::Container(_)));

        let local = serde_json::json!({
            "type": "local", "profile": "python", "allow_unsandboxed": true,
            "deny_tools": [], "setup_timeout_secs": 300, "verify_timeout_secs": null
        });
        let config: EnvironmentConfig = serde_json::from_value(local).expect("local JSON");
        config
            .validate()
            .expect("an explicit local sandbox is valid");

        let http = serde_json::json!({
            "type": "http", "base_url": "http://127.0.0.1:8099",
            "request_timeout_secs": 120, "pool_size": 8, "max_result_bytes": 65536
        });
        let config: EnvironmentConfig = serde_json::from_value(http).expect("http JSON");
        config
            .validate()
            .expect("a well-formed http environment is valid");
    }

    /// The repo's convention: an unknown key is a typo, not an option that gets
    /// quietly dropped.
    #[test]
    fn unknown_keys_and_unknown_types_are_refused() {
        assert!(parse("type = 'local'\nallow_unsandboxd = true\n").is_err());
        assert!(parse("type = 'quantum'\n").is_err());
    }
}
