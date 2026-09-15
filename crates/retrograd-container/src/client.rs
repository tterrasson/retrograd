//! Connecting to the daemon, and turning its errors into ours.

use std::sync::Arc;

use bollard::Docker;
use bollard::errors::Error as BollardError;
use retrograd_agent_core::{Error, Result};

/// A connected daemon.
///
/// Cloning is cheap and shares the connection pool, which is what lets one
/// client serve a whole [`SandboxPool`](crate::SandboxPool).
#[derive(Clone)]
pub struct DockerClient {
    docker: Arc<Docker>,
    version: String,
}

impl DockerClient {
    /// Connects and immediately asks for the daemon version.
    ///
    /// The ping is the point: an unreachable socket has to fail here, at build
    /// time, with a message naming `DOCKER_HOST`. Discovering it on the first
    /// rollout means the failure arrives as a lost trajectory instead of as a
    /// configuration error.
    ///
    /// `DOCKER_HOST` covers Docker Desktop, colima and a Podman socket in
    /// Docker-compatible mode alike.
    pub async fn connect() -> Result<Self> {
        let docker = Docker::connect_with_defaults().map_err(|error| {
            Error::Tool(format!(
                "connect to the Docker daemon: {error}; set DOCKER_HOST if it is not at the \
                 default socket"
            ))
        })?;
        let version = docker
            .version()
            .await
            .map_err(|error| {
                Error::Tool(format!(
                    "reach the Docker daemon: {error}; is it running? (DOCKER_HOST={})",
                    std::env::var("DOCKER_HOST").unwrap_or_else(|_| "unset".into())
                ))
            })?
            .version
            .unwrap_or_else(|| "unknown".into());
        tracing::info!(version = %version, "connected to the Docker daemon");
        Ok(Self {
            docker: Arc::new(docker),
            version,
        })
    }

    pub fn docker(&self) -> &Docker {
        &self.docker
    }

    pub fn version(&self) -> &str {
        &self.version
    }
}

/// Maps a daemon error to [`Error::Sandbox`].
///
/// Everything the daemon reports is a broken world rather than a failed action:
/// a container that will not start, an exec that cannot be created, a transport
/// that died in the middle of an upload. The variant is what tells a tool it may
/// *not* render this as an observation - see
/// [`Error::is_sandbox_failure`](retrograd_agent_core::Error::is_sandbox_failure).
///
/// The two cases that are *not* this: a command exiting non-zero, which comes
/// back as an [`ExecOutput`](retrograd_agent_core::ExecOutput), and a file the
/// container does not have, which the callers below word as
/// [`Error::Tool`](retrograd_agent_core::Error::Tool).
pub(crate) fn docker_error(context: &str, error: BollardError) -> Error {
    Error::Sandbox(format!("{context}: {error}"))
}

pub(crate) fn is_not_found(error: &BollardError) -> bool {
    matches!(
        error,
        BollardError::DockerResponseServerError {
            status_code: 404,
            ..
        }
    )
}
