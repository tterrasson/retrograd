//! Docker-backed sandboxes.
//!
//! This crate knows nothing about agents, trajectories or rewards. It starts a
//! container, runs a command in it, moves files in and out, and recycles it,
//! which is why a judge with a sandboxed command or a server-side evaluation job
//! can reuse it unchanged, and why it is the only place in the workspace where
//! bollard appears.
//!
//! ```ignore
//! let client = DockerClient::connect().await?;
//! let source = ContainerSource::new(client, ContainerSpec::new("python:3.12-slim"), labels);
//! let pool = SandboxPool::new(Arc::new(source), PoolConfig::default());
//! let lease = pool.acquire().await?;
//! let output = lease.exec(ExecRequest::new(["python", "-c", "print(1)"])).await?;
//! ```
//!
//! # Two things this crate is not
//!
//! **Docker is not a security boundary** against hostile code. The defaults here
//! (no network, read-only rootfs, no capabilities, non-root, `no-new-privileges`)
//! raise the cost of an escape; they do not make one impossible. Training on code
//! from an untrusted source at scale calls for an isolating runtime - gVisor,
//! Kata, Firecracker - which is what [`ContainerSpec::runtime`] exists for, so
//! that switching costs a config line rather than a rewrite.
//!
//! **Nothing here builds an image.** [`ensure_image`] pulls and pins by digest;
//! building is a tooling step that belongs outside a training run.

pub mod client;
pub mod image;
pub mod metrics;
pub mod pool;
pub mod reaper;
pub mod sandbox;
pub mod spec;

pub use client::DockerClient;
pub use image::ensure_image;
pub use metrics::SandboxMetrics;
pub use pool::{ManagedSandbox, PoolConfig, ReusePolicy, SandboxPool, SandboxSource};
pub use reaper::{DaemonLocality, RunLabels, reap, reap_run};
pub use sandbox::{ContainerSandbox, ContainerSource};
pub use spec::{ContainerSpec, MountSpec, NetworkMode, WORKDIR};
