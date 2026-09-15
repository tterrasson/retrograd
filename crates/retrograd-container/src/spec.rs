//! What a container is allowed to be.

use std::collections::BTreeMap;

use bollard::models::{ContainerCreateBody, HostConfig, Mount, MountType};
use retrograd_agent_core::{Error, Result, SandboxLimits};
use serde::{Deserialize, Serialize};

/// Every episode gets the *same* working directory.
///
/// Not cosmetic: a path is a deterministic function of the action only if it is
/// the same path in every container of the group. A per-episode directory name
/// (a uuid, a temp dir) leaks into `pwd`, into compiler diagnostics and into
/// stack traces, and from there into training tokens that differ between
/// members for no reason the policy caused.
pub const WORKDIR: &str = "/work";

/// Every container of every group answers `hostname` with this.
///
/// Docker and Podman default to the container's own id, which is exactly the
/// kind of value a group must not be able to tell apart: `hostname`,
/// `uname -n`, Python's `socket.gethostname()`, or a shell prompt would hand
/// two members of one group different tokens for the same command, and the
/// group-relative baseline would measure the difference.
pub const HOSTNAME: &str = "sandbox";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NetworkMode {
    /// The default. A task that needs the network says so.
    #[default]
    None,
    /// Full container networking, for a task that explicitly asked for it.
    Bridge,
}

impl NetworkMode {
    fn as_docker(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Bridge => "bridge",
        }
    }
}

/// A named volume made visible inside the container.
///
/// Its purpose is the package cache (`/opt/cache/pip`, `/opt/cache/npm`): one
/// `pip install` per run instead of one per episode, which is the real payoff of
/// container reuse - far more than the `docker create` it saves.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MountSpec {
    /// Volume name. Host paths remain representable for clear validation errors
    /// but are refused by [`ContainerSpec::validate`].
    pub source: String,
    pub target: String,
    #[serde(default = "default_true")]
    pub read_only: bool,
    /// Must remain true: a bind mount hands the container a replaceable piece
    /// of the training machine and cannot uphold the runtime-socket refusal.
    #[serde(default = "default_true")]
    pub volume: bool,
}

fn default_true() -> bool {
    true
}

/// Configuration for one sandbox container - image, hardening defaults,
/// resource limits - validated by [`ContainerSpec::validate`] before it ever
/// reaches the daemon.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContainerSpec {
    /// Tag or, preferably, `name@sha256:…`. A run resumed three weeks later on
    /// `python:3.12` is not the same environment; see [`crate::ensure_image`].
    pub image: String,
    #[serde(default)]
    pub network: NetworkMode,
    #[serde(default = "default_true")]
    pub read_only_rootfs: bool,
    /// Writable tmpfs mounts, size in bytes. The workdir is one of them, so a
    /// read-only rootfs still leaves the episode somewhere to work.
    #[serde(default = "default_tmpfs")]
    pub tmpfs: BTreeMap<String, u64>,
    /// `uid:gid`. Never root.
    #[serde(default = "default_user")]
    pub user: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub mounts: Vec<MountSpec>,
    /// Alternative OCI runtime - `"runsc"` for gVisor, `"kata-runtime"` for
    /// Kata. The whole point of the field is that hardening the isolation is a
    /// configuration change and not a code change.
    #[serde(default)]
    pub runtime: Option<String>,
    #[serde(default)]
    pub limits: SpecLimits,
    /// Wraps every command in coreutils `timeout`, so a runaway process is
    /// killed *inside* the container instead of being merely abandoned by us.
    /// Requires `timeout` on the image's PATH, which every Debian-based image
    /// has. Turning it off makes a timed-out exec retire its container.
    #[serde(default = "default_true")]
    pub timeout_wrapper: bool,
}

/// [`SandboxLimits`] in its serialized form: durations and sizes in the units an
/// operator writes them in.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpecLimits {
    pub cpus: f32,
    pub memory_mb: u64,
    pub pids: u32,
    pub exec_timeout_secs: u64,
    pub max_output_bytes: usize,
}

impl Default for SpecLimits {
    fn default() -> Self {
        Self {
            cpus: 1.0,
            memory_mb: 1024,
            pids: 256,
            exec_timeout_secs: 30,
            max_output_bytes: 64 * 1024,
        }
    }
}

fn default_user() -> String {
    "10001:10001".into()
}

fn default_tmpfs() -> BTreeMap<String, u64> {
    BTreeMap::from([
        ("/tmp".into(), 64 * 1024 * 1024),
        (WORKDIR.into(), 512 * 1024 * 1024),
    ])
}

/// Accepts only numeric `uid[:gid]`: resolving a name requires the image's
/// passwd database, where an innocent-looking name may map to uid zero.
fn validate_user(user: &str) -> Result<()> {
    let mut parts = user.split(':');
    let uid = parts
        .next()
        .filter(|uid| !uid.is_empty() && uid.trim() == *uid)
        .and_then(|uid| uid.parse::<u64>().ok())
        .ok_or_else(|| {
            Error::invalid(
                "sandbox user must be a numeric non-zero uid, optionally followed by :gid",
            )
        })?;
    if uid == 0 {
        return Err(Error::invalid(
            "refusing to run a sandbox as root; set user to a non-zero uid:gid",
        ));
    }
    if u32::try_from(uid).is_err() {
        return Err(Error::invalid("sandbox uid is out of range"));
    }
    if let Some(gid) = parts.next()
        && (gid.is_empty()
            || gid.trim() != gid
            || gid
                .parse::<u64>()
                .ok()
                .and_then(|gid| u32::try_from(gid).ok())
                .is_none())
    {
        return Err(Error::invalid("sandbox group must be a numeric gid"));
    }
    if parts.next().is_some() {
        return Err(Error::invalid("sandbox user must have the form uid[:gid]"));
    }
    Ok(())
}

/// Whether a path is a container runtime socket.
///
/// Podman's `podman.sock` in Docker-compatible mode grants exactly what
/// `docker.sock` grants, and so does talking to containerd or CRI-O directly.
/// Matching on the file name rather than the full path is deliberate: the socket
/// moves between `/var/run`, `/run`, `$XDG_RUNTIME_DIR` and a user's own
/// directory, and it is the same socket in all of them.
fn is_container_socket(path: &str) -> bool {
    let name = path
        .trim_end_matches('/')
        .rsplit('/')
        .next()
        .unwrap_or_default();
    matches!(
        name,
        "docker.sock" | "podman.sock" | "containerd.sock" | "crio.sock" | "docker.socket"
    )
}

impl ContainerSpec {
    /// Every default is restrictive. Widening one is a decision someone types.
    pub fn new(image: impl Into<String>) -> Self {
        Self {
            image: image.into(),
            network: NetworkMode::None,
            read_only_rootfs: true,
            tmpfs: default_tmpfs(),
            user: default_user(),
            env: BTreeMap::new(),
            mounts: Vec::new(),
            runtime: None,
            limits: SpecLimits::default(),
            timeout_wrapper: true,
        }
    }

    pub fn limits(&self) -> SandboxLimits {
        let memory_bytes = self.limits.memory_mb.saturating_mul(1024 * 1024);
        SandboxLimits {
            cpus: self.limits.cpus,
            memory_bytes,
            pids: self.limits.pids,
            exec_timeout: std::time::Duration::from_secs(self.limits.exec_timeout_secs),
            max_output_bytes: self.limits.max_output_bytes,
            network: self.network != NetworkMode::None,
        }
    }

    /// Refuses what no configuration may ask for.
    ///
    /// Runtime sockets, host bind mounts, root and `privileged` are refused here
    /// rather than left to operator judgement because they can hand the machine
    /// to code the model wrote.
    pub fn validate(&self) -> Result<()> {
        if self.image.trim().is_empty() {
            return Err(Error::invalid("container spec needs an image"));
        }
        validate_user(&self.user)?;
        for mount in &self.mounts {
            if is_container_socket(&mount.source) || is_container_socket(&mount.target) {
                return Err(Error::invalid(
                    "refusing to mount a container runtime socket into a sandbox: it is \
                     equivalent to giving the machine away",
                ));
            }
            if mount.target.trim().is_empty() || !mount.target.starts_with('/') {
                return Err(Error::invalid(format!(
                    "mount target '{}' must be an absolute path",
                    mount.target
                )));
            }
            if !mount.volume {
                // A path checked here can be replaced before the daemon resolves
                // it (including with a socket or symlink). Named volumes are the
                // only mount source whose contents the host filesystem cannot
                // swap underneath this security decision.
                return Err(Error::invalid(
                    "refusing host bind mounts in a model sandbox; use a named volume",
                ));
            }
        }
        if !self.limits.cpus.is_finite()
            || self.limits.cpus <= 0.0
            || self.limits.memory_mb == 0
            || self.limits.pids == 0
        {
            return Err(Error::invalid("container limits must all be positive"));
        }
        let memory_bytes = self
            .limits
            .memory_mb
            .checked_mul(1024 * 1024)
            .ok_or_else(|| Error::invalid("container memory limit is too large"))?;
        if i64::try_from(memory_bytes).is_err() {
            return Err(Error::invalid("container memory limit is too large"));
        }
        let nano_cpus = f64::from(self.limits.cpus) * 1e9;
        if nano_cpus < 1.0 || nano_cpus > i64::MAX as f64 {
            return Err(Error::invalid("container CPU limit is out of range"));
        }
        if self.limits.exec_timeout_secs == 0 || self.limits.max_output_bytes == 0 {
            return Err(Error::invalid(
                "exec_timeout_secs and max_output_bytes must be positive",
            ));
        }
        if self.limits.max_output_bytes.checked_mul(2).is_none() {
            return Err(Error::invalid("max_output_bytes is too large"));
        }
        Ok(())
    }

    /// Pool key: two specs with the same fingerprint produce interchangeable
    /// containers, so one may be recycled for the other.
    ///
    /// FNV-1a over the serialized spec rather than `DefaultHasher`, whose output
    /// is not stable across processes - this value goes into a container label
    /// that a later process reads back.
    pub fn fingerprint(&self) -> String {
        let json = serde_json::to_string(self).unwrap_or_default();
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in json.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{hash:016x}")
    }

    /// Builds the create request. The container runs an infinite sleep loop as
    /// pid 1 and every tool call is an exec into it: the episode's lifetime is
    /// ours to decide, not the entrypoint's.
    pub(crate) fn to_create_body(
        &self,
        image: &str,
        labels: BTreeMap<String, String>,
    ) -> ContainerCreateBody {
        let memory_bytes = self
            .limits
            .memory_mb
            .checked_mul(1024 * 1024)
            .and_then(|bytes| i64::try_from(bytes).ok())
            .expect("validated container memory limit");
        let nano_cpus = (f64::from(self.limits.cpus) * 1e9).round() as i64;
        let host_config = HostConfig {
            network_mode: Some(self.network.as_docker().to_string()),
            readonly_rootfs: Some(self.read_only_rootfs),
            tmpfs: Some(
                self.tmpfs
                    .iter()
                    .map(|(path, size)| {
                        (
                            path.clone(),
                            format!("rw,exec,nosuid,size={size},mode=1777"),
                        )
                    })
                    .collect(),
            ),
            memory: Some(memory_bytes),
            // Denying swap as well: memory_bytes is meant to be a wall, and a
            // container that swaps instead of failing turns a bounded failure
            // into an unbounded slowdown of the whole machine.
            memory_swap: Some(memory_bytes),
            nano_cpus: Some(nano_cpus),
            pids_limit: Some(i64::from(self.limits.pids)),
            cap_drop: Some(vec!["ALL".to_string()]),
            security_opt: Some(vec!["no-new-privileges".to_string()]),
            auto_remove: Some(true),
            runtime: self.runtime.clone(),
            mounts: Some(
                self.mounts
                    .iter()
                    .map(|mount| Mount {
                        source: Some(mount.source.clone()),
                        target: Some(mount.target.clone()),
                        typ: Some(if mount.volume {
                            MountType::VOLUME
                        } else {
                            MountType::BIND
                        }),
                        read_only: Some(mount.read_only),
                        ..Default::default()
                    })
                    .collect(),
            ),
            privileged: Some(false),
            ..Default::default()
        };
        ContainerCreateBody {
            image: Some(image.to_string()),
            user: Some(self.user.clone()),
            // Fixed, not derived from the container id: see [`HOSTNAME`].
            hostname: Some(HOSTNAME.to_string()),
            working_dir: Some(WORKDIR.to_string()),
            env: Some(
                self.env
                    .iter()
                    .map(|(key, value)| format!("{key}={value}"))
                    .collect(),
            ),
            // No shell: an image whose entrypoint is a REPL or a server would
            // otherwise decide what "running" means for us.
            entrypoint: Some(vec!["/bin/sh".into(), "-c".into()]),
            cmd: Some(vec!["while :; do sleep 3600; done".into()]),
            labels: Some(labels.into_iter().collect()),
            network_disabled: Some(self.network == NetworkMode::None),
            host_config: Some(host_config),
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_restrictive_ones() {
        let spec = ContainerSpec::new("python:3.12-slim");
        spec.validate().unwrap();
        assert_eq!(spec.network, NetworkMode::None);
        assert!(spec.read_only_rootfs);
        assert_ne!(spec.user, "0:0");
        assert!(spec.tmpfs.contains_key(WORKDIR));
        assert!(!spec.limits().network);
    }

    /// Neither of these can be reached from a configuration file, whatever it
    /// says: both are equivalent to handing over the training machine.
    ///
    /// Every spelling is listed rather than one of each, because the refusal is
    /// worth exactly as much as its weakest spelling: `root:root` is as much
    /// root as `0:0`, and Podman's socket grants what Docker's grants.
    #[test]
    fn the_runtime_socket_and_root_are_refused_outright_however_they_are_spelled() {
        for path in [
            "/var/run/docker.sock",
            "/run/podman/podman.sock",
            "/run/user/1000/podman/podman.sock",
            "/run/containerd/containerd.sock",
        ] {
            let mut spec = ContainerSpec::new("python:3.12-slim");
            spec.mounts.push(MountSpec {
                source: path.into(),
                target: "/var/run/docker.sock".into(),
                read_only: true,
                volume: false,
            });
            let error = spec.validate().unwrap_err().to_string();
            assert!(error.contains("socket"), "{path}: {error}");

            // Also when it is only the target that names it.
            let mut spec = ContainerSpec::new("python:3.12-slim");
            spec.mounts.push(MountSpec {
                source: "some-volume".into(),
                target: path.into(),
                read_only: true,
                volume: true,
            });
            assert!(spec.validate().is_err(), "{path} as a target");
        }

        for user in [
            "0",
            "0:0",
            "root",
            "root:root",
            "root:1000",
            "0:1000",
            "admin",
            "nobody:0",
        ] {
            let mut spec = ContainerSpec::new("python:3.12-slim");
            spec.user = user.into();
            assert!(spec.validate().is_err(), "accepted user '{user}'");
        }

        // Only numeric identities are unambiguous without resolving the image's
        // passwd database. Group zero does not change the process uid.
        for user in ["10001", "10001:10001", "10001:0"] {
            let mut spec = ContainerSpec::new("python:3.12-slim");
            spec.user = user.into();
            spec.validate()
                .unwrap_or_else(|error| panic!("refused user '{user}': {error}"));
        }
    }

    #[test]
    fn bind_mounts_cannot_smuggle_a_socket_through_a_parent_or_symlink() {
        for source in ["/var/run", "/tmp/innocent-link", "/tmp/fixture.txt"] {
            let mut spec = ContainerSpec::new("python:3.12-slim");
            spec.mounts.push(MountSpec {
                source: source.into(),
                target: "/host".into(),
                read_only: true,
                volume: false,
            });
            assert!(spec.validate().unwrap_err().to_string().contains("bind"));
        }
    }

    #[test]
    fn non_finite_and_overflowing_limits_are_refused() {
        for cpus in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0] {
            let mut spec = ContainerSpec::new("python:3.12-slim");
            spec.limits.cpus = cpus;
            assert!(spec.validate().is_err(), "accepted cpus={cpus}");
        }
        let mut spec = ContainerSpec::new("python:3.12-slim");
        spec.limits.memory_mb = u64::MAX;
        assert!(spec.validate().is_err());
        let mut spec = ContainerSpec::new("python:3.12-slim");
        spec.limits.max_output_bytes = usize::MAX;
        assert!(spec.validate().is_err());
    }

    #[test]
    fn the_fingerprint_keys_the_pool_on_what_makes_containers_interchangeable() {
        let spec = ContainerSpec::new("python:3.12-slim");
        let same = ContainerSpec::new("python:3.12-slim");
        assert_eq!(spec.fingerprint(), same.fingerprint());

        let mut other = spec.clone();
        other.limits.memory_mb *= 2;
        assert_ne!(spec.fingerprint(), other.fingerprint());

        let mut network = spec.clone();
        network.network = NetworkMode::Bridge;
        assert_ne!(spec.fingerprint(), network.fingerprint());
    }

    /// The create body is where a default silently stops applying, so the four
    /// that matter are asserted rather than trusted.
    #[test]
    fn the_create_body_carries_the_hardening() {
        let spec = ContainerSpec::new("python:3.12-slim");
        let body = spec.to_create_body("python@sha256:abc", BTreeMap::new());
        let host = body.host_config.unwrap();
        assert_eq!(host.network_mode.as_deref(), Some("none"));
        assert_eq!(host.cap_drop, Some(vec!["ALL".to_string()]));
        assert_eq!(host.privileged, Some(false));
        assert_eq!(host.auto_remove, Some(true));
        assert_eq!(host.memory, host.memory_swap);
        assert_eq!(body.working_dir.as_deref(), Some(WORKDIR));
        assert_eq!(body.image.as_deref(), Some("python@sha256:abc"));
        // The hostname is ours and not the container id, or `hostname` alone
        // would make two members of a group read different bytes.
        assert_eq!(body.hostname.as_deref(), Some(HOSTNAME));
    }
}
