//! Not leaving containers behind.
//!
//! A run killed with SIGKILL runs no `Drop`, closes no environment and, without
//! this, leaves as many containers as it had trajectories in flight. Five
//! overlapping defences, because each one has a hole the next covers:
//! `AutoRemove` on the container, `Drop` on the lease, `shutdown` on the pool,
//! [`reap_run`] from the interrupt handler for the signals a process can still
//! catch, and [`reap`] - a sweep at startup for what the four before it missed,
//! which is the only defence left against a `SIGKILL`.

use std::collections::HashMap;

use bollard::query_parameters::{ListContainersOptionsBuilder, RemoveContainerOptionsBuilder};
use futures::StreamExt;
use retrograd_agent_core::Result;

use crate::client::{DockerClient, docker_error};

pub const LABEL_RUN: &str = "retrograd.run";
pub const LABEL_POOL: &str = "retrograd.pool";
pub const LABEL_OWNER: &str = "retrograd.owner";

/// How a container says who it belongs to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunLabels {
    /// Identifies the run. Ends up in the run metadata too, so a leftover
    /// container can be traced back to what created it.
    pub run: String,
    /// Pool fingerprint - the [`ContainerSpec`](crate::ContainerSpec) hash, so
    /// containers of two differently configured pools are never confused.
    pub pool: String,
    /// `<pid>` of the training process, used for liveness. Deliberately not a
    /// hostname: see [`reap`].
    pub owner: String,
}

impl RunLabels {
    pub fn new(run: impl Into<String>, pool: impl Into<String>) -> Self {
        Self {
            run: run.into(),
            pool: pool.into(),
            owner: std::process::id().to_string(),
        }
    }

    pub fn to_map(&self) -> HashMap<String, String> {
        HashMap::from([
            (LABEL_RUN.to_string(), self.run.clone()),
            (LABEL_POOL.to_string(), self.pool.clone()),
            (LABEL_OWNER.to_string(), self.owner.clone()),
        ])
    }
}

/// Whether the daemon is on this machine, which decides what the reaper may
/// conclude from a pid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DaemonLocality {
    Local,
    Remote,
}

impl DaemonLocality {
    /// A unix socket is this machine; a `tcp://`/`ssh://` `DOCKER_HOST` is not.
    pub fn from_env() -> Self {
        match std::env::var("DOCKER_HOST") {
            Ok(host) if !host.starts_with("unix://") && !host.is_empty() => Self::Remote,
            _ => Self::Local,
        }
    }
}

/// Removes containers left by runs that are gone.
///
/// The decision is deliberately conservative, because the failure mode of being
/// wrong is killing a *concurrent* run's containers mid-rollout. On a local
/// daemon, a container is reaped when its owner pid is no longer alive. On a
/// remote one, a pid means nothing here, so only our own run's leftovers - from
/// a previous crash with the same run id - are removed, and the rest is left
/// with a warning rather than guessed at.
pub async fn reap(
    client: &DockerClient,
    labels: &RunLabels,
    locality: DaemonLocality,
) -> Result<usize> {
    let filters = HashMap::from([("label".to_string(), vec![LABEL_POOL.to_string()])]);
    let options = ListContainersOptionsBuilder::default()
        .all(true)
        .filters(&filters)
        .build();
    let containers = client
        .docker()
        .list_containers(Some(options))
        .await
        .map_err(|error| docker_error("list sandbox containers", error))?;

    let mut doomed = Vec::new();
    let mut skipped = 0;
    for container in containers {
        let container_labels = container.labels.unwrap_or_default();
        let Some(id) = container.id else { continue };
        if !should_reap(&container_labels, labels, locality, pid_is_alive) {
            skipped += 1;
            continue;
        }
        doomed.push(id);
    }
    let removed = remove_all(client, doomed).await;
    if removed > 0 {
        tracing::info!(removed, "reaped sandbox containers from runs that are gone");
    }
    if skipped > 0 && locality == DaemonLocality::Remote {
        tracing::warn!(
            skipped,
            "left sandbox containers of other runs alone: a remote daemon makes their owner pid \
             meaningless from here"
        );
    }
    Ok(removed)
}

/// Removes every container of one run, whatever its owner pid still looks like.
///
/// This is the interrupt path, and it is not the same decision as [`reap`].
/// `reap` guesses about *other people's* runs, so it is conservative and asks
/// whether the owner is still alive. Here the caller is the run being named:
/// there is nothing to infer, no concurrent run to damage, and the owner pid is
/// deliberately ignored - it is our own, and it is still alive, which is exactly
/// why `reap`'s rule would spare every one of these.
///
/// Never returns an error for a container it could not remove: a signal handler
/// has no one to report to and a partial cleanup beats none. Only failing to
/// *list* is an error, since that means nothing was even attempted.
pub async fn reap_run(client: &DockerClient, run: &str) -> Result<usize> {
    let filters = HashMap::from([("label".to_string(), vec![format!("{LABEL_RUN}={run}")])]);
    let options = ListContainersOptionsBuilder::default()
        .all(true)
        .filters(&filters)
        .build();
    let containers = client
        .docker()
        .list_containers(Some(options))
        .await
        .map_err(|error| docker_error("list this run's containers", error))?;
    let doomed = containers.into_iter().filter_map(|c| c.id).collect();
    Ok(remove_all(client, doomed).await)
}

/// How many removals are in flight at once.
///
/// One removal is a daemon round trip that ends in a `SIGKILL`, a wait and an
/// unmount, and it is independent of every other one. Doing them in sequence
/// made the interrupt path cost the width of the pool times a container's death:
/// the operator sees a process that does not answer, and reaches for
/// `kill -9`, which is the one thing this module exists to avoid. Bounded
/// rather than unbounded, because the daemon is one server and a wide pool
/// would open one connection per container against it.
const REMOVE_CONCURRENCY: usize = 8;

/// Force-removes each id, counting what went. A failure is one container, not
/// the sweep: it usually means someone else - a racing reaper, `AutoRemove`,
/// got there first, which is the outcome we wanted anyway.
async fn remove_all(client: &DockerClient, ids: Vec<String>) -> usize {
    futures::stream::iter(ids)
        .map(|id| async move {
            let options = RemoveContainerOptionsBuilder::default()
                .force(true)
                .v(true)
                .build();
            match client.docker().remove_container(&id, Some(options)).await {
                Ok(()) => 1,
                Err(error) => {
                    tracing::debug!(error = %error, "removing a container failed");
                    0
                }
            }
        })
        .buffer_unordered(REMOVE_CONCURRENCY)
        .fold(0, |total, removed| async move { total + removed })
        .await
}

/// The whole decision, as a pure function - which is what lets the rule be
/// tested without a daemon and without leaving anything to reap.
fn should_reap(
    container: &HashMap<String, String>,
    ours: &RunLabels,
    locality: DaemonLocality,
    alive: impl Fn(u32) -> bool,
) -> bool {
    // Never touch a container that is not one of ours to begin with.
    if !container.contains_key(LABEL_POOL) {
        return false;
    }
    if container.get(LABEL_OWNER) == Some(&ours.owner) {
        // Ours, right now, in flight.
        return false;
    }
    match locality {
        DaemonLocality::Remote => false,
        DaemonLocality::Local => {
            match container.get(LABEL_OWNER).and_then(|pid| pid.parse().ok()) {
                Some(pid) => !alive(pid),
                // Labelled as a pool container but with no owner: nobody will ever
                // claim it.
                None => true,
            }
        }
    }
}

/// `kill(pid, 0)` without a libc dependency. Startup-only, so the cost of a
/// process spawn does not matter.
fn pid_is_alive(pid: u32) -> bool {
    std::process::Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(run: &str, owner: &str) -> HashMap<String, String> {
        HashMap::from([
            (LABEL_POOL.to_string(), "fp".to_string()),
            (LABEL_RUN.to_string(), run.to_string()),
            (LABEL_OWNER.to_string(), owner.to_string()),
        ])
    }

    fn ours() -> RunLabels {
        RunLabels {
            run: "run-a".into(),
            pool: "fp".into(),
            owner: "100".into(),
        }
    }

    #[test]
    fn our_own_live_containers_are_never_reaped() {
        assert!(!should_reap(
            &labels("run-a", "100"),
            &ours(),
            DaemonLocality::Local,
            |_| true
        ));
    }

    #[test]
    fn a_dead_owner_leaves_containers_to_collect() {
        assert!(should_reap(
            &labels("run-b", "200"),
            &ours(),
            DaemonLocality::Local,
            |_| false
        ));
        // A repeated run id does not prove the other process is dead.
        assert!(!should_reap(
            &labels("run-a", "200"),
            &ours(),
            DaemonLocality::Local,
            |_| true
        ));
        assert!(should_reap(
            &labels("run-a", "200"),
            &ours(),
            DaemonLocality::Local,
            |_| false
        ));
    }

    /// The expensive mistake is not leaving debris, it is killing the
    /// containers of a training run that is still going.
    #[test]
    fn a_live_concurrent_run_is_left_alone() {
        assert!(!should_reap(
            &labels("run-b", "200"),
            &ours(),
            DaemonLocality::Local,
            |_| true
        ));
        // A remote daemon makes a pid meaningless, so nothing is concluded.
        assert!(!should_reap(
            &labels("run-b", "200"),
            &ours(),
            DaemonLocality::Remote,
            |_| false
        ));
    }

    #[test]
    fn containers_that_are_not_ours_at_all_are_out_of_scope() {
        let foreign = HashMap::from([("com.example".to_string(), "x".to_string())]);
        assert!(!should_reap(
            &foreign,
            &ours(),
            DaemonLocality::Local,
            |_| false
        ));
        // Labelled as a pool container, but nobody claims it.
        let orphan = HashMap::from([(LABEL_POOL.to_string(), "fp".to_string())]);
        assert!(should_reap(&orphan, &ours(), DaemonLocality::Local, |_| {
            true
        }));
    }
}
