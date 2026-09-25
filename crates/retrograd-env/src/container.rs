//! Turning a language profile into a container pool.
//!
//! The only module of the stack that names `retrograd-container`, and it is
//! behind the `container` feature: that is what makes bollard absent from a run
//! whose tools are all read-only, rather than merely unused.
//!
//! Nothing here decides policy. A profile fills in an image and a package
//! cache; both are overridable, and the security defaults are the container
//! crate's own - this module never widens them. The tools are not the
//! profile's: they arrive resolved, as [`Toolsets`].

use std::sync::Arc;

use retrograd_agent_core::{Error, Result, SandboxProvider};
use retrograd_container::{
    ContainerSource, DaemonLocality, DockerClient, MountSpec, NetworkMode, SandboxMetrics,
    SandboxPool, reap,
};
use retrograd_spec::env::Profile;
use retrograd_tools::Toolsets;

use crate::sandbox::{SandboxEnvironmentConfig, SandboxEnvironmentFactory};

pub use retrograd_spec::env::{ContainerEnvironmentConfig, ContainerPoolConfig};

/// What a declared container environment can do once there is a daemon.
///
/// The declaration itself is `retrograd-spec`'s: building a `ContainerSpec`
/// from it is this crate's, and the split is what lets a configuration be
/// *read* without linking bollard.
pub trait ContainerEnvironment {
    fn image(&self) -> Result<String>;
    fn validate(&self) -> Result<()>;
}

impl ContainerEnvironment for ContainerEnvironmentConfig {
    /// The image this configuration runs: the explicit one, or the profile's.
    fn image(&self) -> Result<String> {
        match (&self.image, self.profile) {
            (Some(image), _) => Ok(image.clone()),
            (None, Profile::Python) => Ok("python:3.12-slim".into()),
            (None, Profile::Typescript) => Ok("node:22-slim".into()),
            (None, Profile::Custom) => Err(Error::invalid(
                "environment profile 'custom' has no default image: set `image`",
            )),
        }
    }

    /// Everything checkable without a daemon: the image and the container
    /// spec's own refusals (root, the Docker socket, absurd limits).
    ///
    /// It exists so a catalogue or a config loader can reject a declaration at
    /// startup. What it cannot check - that the image exists, that the daemon
    /// answers - is [`connect_pool`]'s job, and that one still runs before the
    /// first rollout.
    fn validate(&self) -> Result<()> {
        self.validate_declaration()?;
        container_spec(self)?;
        Ok(())
    }
}

/// Where the profile's package manager keeps its cache, when it has one.
fn cache_target(config: &ContainerEnvironmentConfig) -> Option<&'static str> {
    match config.profile {
        Profile::Python => Some("/opt/cache/pip"),
        Profile::Typescript => Some("/opt/cache/npm"),
        Profile::Custom => None,
    }
}

fn container_spec(
    config: &ContainerEnvironmentConfig,
) -> Result<retrograd_container::ContainerSpec> {
    let mut spec = retrograd_container::ContainerSpec::new(config.image()?);
    spec.limits = config.limits;
    if config.allow_network {
        spec.network = NetworkMode::Bridge;
    }
    if let (Some(volume), Some(target)) = (&config.cache_volume, cache_target(config)) {
        spec.mounts.push(MountSpec {
            source: volume.clone(),
            target: target.into(),
            read_only: true,
            volume: true,
        });
        // Pointing the package manager at the cache is the whole reason the
        // mount exists; leaving it to the task would make the mount silent.
        match config.profile {
            Profile::Python => {
                spec.env.insert("PIP_CACHE_DIR".into(), target.into());
            }
            Profile::Typescript => {
                spec.env.insert("npm_config_cache".into(), target.into());
            }
            Profile::Custom => {}
        }
    }
    spec.validate()?;
    Ok(spec)
}

/// Connects to the daemon, pulls and pins the image, reaps leftovers, and
/// returns the pool.
///
/// Every failure here is a configuration failure, and it happens before the
/// first rollout on purpose: an unreachable daemon discovered on the 700th
/// trajectory is a lost run.
pub async fn connect_pool(
    config: &ContainerEnvironmentConfig,
) -> Result<(Arc<SandboxPool>, Arc<SandboxMetrics>, String)> {
    let spec = container_spec(config)?;
    let client = DockerClient::connect().await?;
    let run_id = config
        .run_id
        .clone()
        .unwrap_or_else(|| format!("retrograd-{}", std::process::id()));
    let metrics = Arc::new(SandboxMetrics::default());
    let source =
        ContainerSource::with_metrics(client.clone(), spec, run_id, metrics.clone()).await?;
    let image = source.image().to_owned();
    // Containers from a run that is gone are removed before we add ours, so a
    // machine that has crashed a few times does not silently run out.
    match reap(&client, source.labels(), DaemonLocality::from_env()).await {
        Ok(removed) if removed > 0 => tracing::info!(removed, "reaped containers of dead runs"),
        Err(error) => tracing::warn!(%error, "could not reap leftover containers"),
        _ => {}
    }
    let pool = SandboxPool::with_metrics(Arc::new(source), config.pool.into(), metrics.clone());
    Ok((pool, metrics, image))
}

/// The whole environment side of a container run, in one call.
pub async fn build_factory(
    config: &ContainerEnvironmentConfig,
    toolsets: Toolsets,
) -> Result<(SandboxEnvironmentFactory, Arc<SandboxMetrics>)> {
    let (pool, metrics, image) = connect_pool(config).await?;
    tracing::info!(%image, "container environment pinned to image digest");
    let factory = SandboxEnvironmentFactory::new(
        pool as Arc<dyn SandboxProvider>,
        toolsets,
        SandboxEnvironmentConfig {
            // Both: the reference the operator configured, and what it resolved
            // to. Keeping only the first would compare scenarios against a tag
            // that no longer points where it did, and would leave the digest
            // nowhere but in a log line - see `SandboxEnvironmentConfig`.
            image: Some(config.image()?),
            pinned_image: Some(image),
            setup_timeout: std::time::Duration::from_secs(config.setup_timeout_secs),
            verify_timeout: config
                .verify_timeout_secs
                .map(std::time::Duration::from_secs),
            ..SandboxEnvironmentConfig::default()
        },
    )?;
    Ok((factory, metrics))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_profile_fills_in_an_image_and_a_cache() {
        let python = ContainerEnvironmentConfig {
            cache_volume: Some("retrograd-pip".into()),
            ..Default::default()
        };
        assert_eq!(python.image().unwrap(), "python:3.12-slim");
        let spec = container_spec(&python).unwrap();
        assert_eq!(spec.mounts[0].target, "/opt/cache/pip");
        assert!(spec.mounts[0].read_only);
        assert_eq!(spec.env.get("PIP_CACHE_DIR").unwrap(), "/opt/cache/pip");
        assert_eq!(spec.network, NetworkMode::None);

        let typescript = ContainerEnvironmentConfig {
            profile: Profile::Typescript,
            ..Default::default()
        };
        assert_eq!(typescript.image().unwrap(), "node:22-slim");
    }

    /// `custom` exists so an operator can bring their own image; forgetting to
    /// name it must fail at configuration time, not at the first acquisition.
    /// A pool that must keep more warm than it may hold never converges, and
    /// the shape of the mistake is a hang rather than an error - so it is
    /// refused where an operator can still read the sentence.
    #[test]
    fn an_impossible_pool_is_refused_at_declaration() {
        let config = ContainerEnvironmentConfig {
            tools: python_tools(),
            pool: ContainerPoolConfig {
                max_live: 2,
                min_idle: 8,
                ..Default::default()
            },
            ..Default::default()
        };
        let error = config.validate().unwrap_err().to_string();
        assert!(error.contains("min_idle"), "{error}");
    }

    #[test]
    fn a_custom_profile_without_an_image_is_refused() {
        let config = ContainerEnvironmentConfig {
            profile: Profile::Custom,
            ..Default::default()
        };
        assert!(config.image().is_err());
        let config = ContainerEnvironmentConfig {
            image: Some("ghcr.io/me/tasks@sha256:abc".into()),
            ..config
        };
        assert_eq!(
            container_spec(&config).unwrap().image,
            "ghcr.io/me/tasks@sha256:abc"
        );
    }

    #[test]
    fn the_network_stays_a_decision_and_a_toolset_is_required() {
        let config = ContainerEnvironmentConfig {
            tools: python_tools(),
            allow_network: true,
            ..Default::default()
        };
        config.validate().unwrap();
        assert_eq!(
            container_spec(&config).unwrap().network,
            NetworkMode::Bridge
        );
        let error = ContainerEnvironmentConfig::default()
            .validate()
            .unwrap_err()
            .to_string();
        assert!(error.contains("tools.default"), "{error}");
    }

    fn python_tools() -> retrograd_tools::ToolsConfig {
        retrograd_tools::ToolsConfig {
            default: Some("python".into()),
            ..Default::default()
        }
    }
}
