//! Bounding, reusing and cleaning up sandboxes.
//!
//! One update is `scenarios_per_update × group_size` trajectories: eight
//! containers with the defaults, sixty-four at `4 × 16`. Creating one costs
//! 100 ms to 1 s, and sixty-four at once on a machine that is also holding a
//! model is not a plan. The pool answers three separate problems, and it is
//! worth keeping them separate because only one of them is about speed:
//!
//! - **backpressure** - [`PoolConfig::max_live`] bounds how many containers
//!   exist at once, whatever the caller asks for. A group larger than the bound
//!   serializes instead of flattening the machine, the same policy as chunking
//!   decode by `sequence_capacity`;
//! - **creation cost** - idle containers are kept warm, ideally created during
//!   the optimizer step, when the environment side is doing nothing;
//! - **contamination** - and this one is statistical, not operational. Two
//!   members of a group that share state are no longer independent samples, and
//!   the group-relative baseline stops measuring the policy. This is why a
//!   doubtful cleanup destroys the container instead of returning it: the cost
//!   is one `docker create`, and the alternative shows up in no metric at all.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use retrograd_agent_core::{Error, Lease, Result, Sandbox, SandboxOwner, SandboxProvider};
use retrograd_metrics::MetricValue;
use serde::{Deserialize, Serialize};
use tokio::sync::{Notify, mpsc};

use crate::metrics::SandboxMetrics;

/// How long `shutdown` waits for outstanding leases before giving up on them.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(30);

/// What happens to a sandbox between two episodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReusePolicy {
    /// One container per episode, destroyed at the end. The default, because it
    /// is the only policy where isolation is a property of the runtime rather
    /// than of our cleanup code being right.
    #[default]
    Never,
    /// The container is recycled with a wiped workspace and its leftover
    /// processes killed. What still survives, and what one accepts by choosing
    /// this: the image layers, the read-only package caches, and any damage a
    /// previous episode did outside the workspace.
    Workspace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConfig {
    /// Hard bound on live containers. Defaults to a typical `group_size`.
    pub max_live: usize,
    /// How many to keep warm ahead of the next update.
    pub min_idle: usize,
    /// What happens to a container between two episodes; see [`ReusePolicy`].
    pub reuse: ReusePolicy,
    /// A recycled container accumulates whatever escaped the workspace; retiring
    /// it after this many episodes bounds that accumulation.
    pub max_leases_per_container: u32,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            max_live: 8,
            min_idle: 0,
            reuse: ReusePolicy::Never,
            max_leases_per_container: 32,
        }
    }
}

/// One reusable backing instance.
///
/// Not a supertrait of [`Sandbox`]: the pool needs to hand out an
/// `Arc<dyn Sandbox>` while keeping the management handle, and going through a
/// method rather than a trait upcast keeps that working on every toolchain.
#[async_trait]
pub trait ManagedSandbox: Send + Sync {
    fn sandbox(&self) -> Arc<dyn Sandbox>;

    /// Makes the instance fit for a *new* episode: empty workspace, no leftover
    /// processes. `Err` means the pool must destroy it rather than reuse it.
    async fn recycle(&self) -> Result<()>;

    async fn destroy(&self);

    /// False once anything has made the instance untrustworthy.
    fn healthy(&self) -> bool;
}

#[async_trait]
pub trait SandboxSource: Send + Sync {
    /// Creates one instance, ready to be leased out immediately.
    async fn create(&self) -> Result<Arc<dyn ManagedSandbox>>;

    /// Destroys everything this source ever created, including instances the
    /// pool has leased out and will never get back, and says how many.
    ///
    /// The pool cannot do this itself: it only knows the instances it is
    /// currently holding, and the whole point of the interrupt path is the ones
    /// it is *not* holding - leased to a rollout that is about to be abandoned
    /// mid-flight. Only the source knows how to enumerate them.
    async fn force_cleanup(&self) -> usize {
        0
    }
}

struct Entry {
    instance: Arc<dyn ManagedSandbox>,
    leases: u32,
}

#[derive(Default)]
struct PoolState {
    idle: Vec<Entry>,
    leased: HashMap<u64, Entry>,
    /// Containers that exist right now, idle or leased.
    live: usize,
    next_token: u64,
    closed: bool,
}

pub struct SandboxPool {
    /// Weak reference to itself: a [`Lease`] owns its pool for as long as it is
    /// out, so handing one over needs a strong reference. Taking it from the
    /// inside is what lets `SandboxPool` implement [`SandboxProvider`] directly,
    /// rather than through a wrapper the caller would have to know about.
    me: Weak<SandboxPool>,
    source: Arc<dyn SandboxSource>,
    config: PoolConfig,
    state: Mutex<PoolState>,
    /// Woken whenever a slot frees up, so a caller blocked on `max_live` does
    /// not poll.
    available: Notify,
    returns: mpsc::UnboundedSender<(u64, bool)>,
    metrics: Arc<SandboxMetrics>,
}

impl SandboxPool {
    /// Must be called inside a Tokio runtime: the pool spawns the task that does
    /// the slow half of a release.
    ///
    /// That split is what makes [`Lease`] droppable anywhere - including on a
    /// deadline or an unwind, where there is no `.await` to be had - while the
    /// container is still wiped before anyone else sees it.
    pub fn new(source: Arc<dyn SandboxSource>, config: PoolConfig) -> Arc<Self> {
        Self::with_metrics(source, config, Arc::new(SandboxMetrics::default()))
    }

    pub fn with_metrics(
        source: Arc<dyn SandboxSource>,
        config: PoolConfig,
        metrics: Arc<SandboxMetrics>,
    ) -> Arc<Self> {
        let (returns, mut incoming) = mpsc::unbounded_channel();
        let pool = Arc::new_cyclic(|me| Self {
            me: me.clone(),
            source,
            config,
            state: Mutex::new(PoolState::default()),
            available: Notify::new(),
            returns,
            metrics,
        });
        // Weak, so an idle pool nobody holds any more is not kept alive by its
        // own janitor. Outstanding leases hold a strong reference of their own.
        let weak = Arc::downgrade(&pool);
        tokio::spawn(async move {
            while let Some((token, healthy)) = incoming.recv().await {
                let Some(pool) = Weak::upgrade(&weak) else {
                    break;
                };
                pool.reclaim(token, healthy).await;
            }
        });
        pool
    }

    pub fn metrics(&self) -> Arc<SandboxMetrics> {
        self.metrics.clone()
    }

    pub fn metric_values(&self) -> Vec<MetricValue> {
        self.metrics.metric_values()
    }

    /// Containers alive right now. Test and diagnostic surface - nothing in a
    /// rollout is allowed to branch on it.
    pub fn live(&self) -> usize {
        self.state.lock().expect("pool state").live
    }

    /// Sandboxes sitting idle, ready to be reused. Same diagnostic-only caveat
    /// as [`Self::live`].
    pub fn idle(&self) -> usize {
        self.state.lock().expect("pool state").idle.len()
    }

    fn hand_out(&self, entry: Entry, waited_since: Instant, reused: bool) -> Result<Lease> {
        let owner = self
            .me
            .upgrade()
            .ok_or_else(|| Error::Tool("the sandbox pool is gone".into()))?;
        let sandbox = entry.instance.sandbox();
        let mut state = self.state.lock().expect("pool state");
        let token = state.next_token;
        state.next_token += 1;
        let live = state.live;
        state.leased.insert(
            token,
            Entry {
                instance: entry.instance,
                leases: entry.leases + 1,
            },
        );
        drop(state);
        self.metrics.record_live(live);
        self.metrics
            .record_acquire(waited_since.elapsed().as_micros() as u64, reused);
        Ok(Lease::new(sandbox, owner, token))
    }

    /// The slow half of a release: wipe, decide, and only then make the slot
    /// available again. A recycled container becomes acquirable *after* it is
    /// clean, never before.
    async fn reclaim(self: Arc<Self>, token: u64, healthy: bool) {
        let Some(entry) = self.state.lock().expect("pool state").leased.remove(&token) else {
            return;
        };
        let closed = self.state.lock().expect("pool state").closed;
        let reusable = healthy
            && !closed
            && entry.instance.healthy()
            && self.config.reuse == ReusePolicy::Workspace
            && entry.leases < self.config.max_leases_per_container
            && match entry.instance.recycle().await {
                Ok(()) => true,
                Err(error) => {
                    tracing::warn!(error = %error, "recycling a sandbox failed; destroying it");
                    false
                }
            };
        if reusable {
            self.state.lock().expect("pool state").idle.push(entry);
        } else {
            entry.instance.destroy().await;
            self.state.lock().expect("pool state").live -= 1;
        }
        self.available.notify_one();
    }
}

impl SandboxOwner for SandboxPool {
    fn release(&self, token: u64, healthy: bool) {
        // Send and return: the caller may be a `Drop` on a failure path.
        let _ = self.returns.send((token, healthy));
    }
}

#[async_trait]
impl SandboxProvider for SandboxPool {
    fn metric_values(&self) -> Vec<MetricValue> {
        self.metrics.metric_values()
    }

    async fn acquire(&self) -> Result<Lease> {
        let waiting_since = Instant::now();
        loop {
            // Registered before the state is inspected, so a release happening
            // in between is not a lost wake-up.
            let available = self.available.notified();
            enum Next {
                Reuse(Entry),
                Create,
                Wait,
            }
            let next = {
                let mut state = self.state.lock().expect("pool state");
                if state.closed {
                    return Err(Error::Tool("the sandbox pool is shut down".into()));
                }
                match state.idle.pop() {
                    Some(entry) => Next::Reuse(entry),
                    None if state.live < self.config.max_live => {
                        // Counted before the create so two concurrent callers
                        // cannot both decide there is room for the last slot.
                        state.live += 1;
                        Next::Create
                    }
                    None => Next::Wait,
                }
            };
            match next {
                Next::Reuse(entry) => return self.hand_out(entry, waiting_since, true),
                Next::Create => {
                    let started = Instant::now();
                    match self.source.create().await {
                        Ok(instance) => {
                            self.metrics
                                .record_create(started.elapsed().as_micros() as u64);
                            return self.hand_out(
                                Entry {
                                    instance,
                                    leases: 0,
                                },
                                waiting_since,
                                false,
                            );
                        }
                        Err(error) => {
                            self.state.lock().expect("pool state").live -= 1;
                            self.available.notify_one();
                            return Err(error);
                        }
                    }
                }
                Next::Wait => available.await,
            }
        }
    }

    /// Fills the idle set up to `count`, never past `max_live`.
    ///
    /// Best effort by design: it runs while the optimizer step does, and a
    /// warm-up that failed must cost a slower `acquire`, not a failed update.
    async fn prewarm(&self, count: usize) -> Result<()> {
        // `min_idle` is the floor a caller cannot warm below: it is the
        // operator's statement about how much of the creation cost they want
        // paid off the critical path.
        let count = count.max(self.config.min_idle);
        while {
            let mut state = self.state.lock().expect("pool state");
            if state.closed || state.idle.len() >= count || state.live >= self.config.max_live {
                false
            } else {
                state.live += 1;
                true
            }
        } {
            let started = Instant::now();
            match self.source.create().await {
                Ok(instance) => {
                    self.metrics
                        .record_create(started.elapsed().as_micros() as u64);
                    let (destroy, live) = {
                        let mut state = self.state.lock().expect("pool state");
                        if state.closed {
                            state.live -= 1;
                            (true, state.live)
                        } else {
                            state.idle.push(Entry {
                                instance: instance.clone(),
                                leases: 0,
                            });
                            (false, state.live)
                        }
                    };
                    if destroy {
                        instance.destroy().await;
                        self.metrics.record_live(live);
                        self.available.notify_waiters();
                    } else {
                        self.metrics.record_live(live);
                        self.available.notify_one();
                    }
                }
                Err(error) => {
                    self.state.lock().expect("pool state").live -= 1;
                    tracing::warn!(error = %error, "prewarming a sandbox failed");
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Destroys everything idle, refuses further acquisitions, and waits for the
    /// leases still out to come back and be destroyed.
    ///
    /// The wait is the point: releasing is asynchronous, so a `shutdown` that
    /// returned immediately would leave containers being destroyed after the
    /// process decided it was done - which on a crash-free exit is exactly the
    /// debris the reaper exists to clean up after a crash. It is bounded, since
    /// a caller still *holding* a lease would otherwise deadlock a run's exit;
    /// past the bound, `AutoRemove` and the reaper are the backstop.
    async fn shutdown(&self) {
        let idle = {
            let mut state = self.state.lock().expect("pool state");
            state.closed = true;
            std::mem::take(&mut state.idle)
        };
        for entry in idle {
            entry.instance.destroy().await;
            self.state.lock().expect("pool state").live -= 1;
        }
        self.available.notify_waiters();

        let deadline = Instant::now() + DRAIN_TIMEOUT;
        loop {
            let drained = self.available.notified();
            if self.live() == 0 {
                return;
            }
            if Instant::now() >= deadline {
                tracing::warn!(
                    live = self.live(),
                    "gave up waiting for sandboxes to come back; the reaper will collect them"
                );
                return;
            }
            let _ = tokio::time::timeout(Duration::from_millis(100), drained).await;
        }
    }

    /// Closes the pool and hands the teardown to the source, which is the only
    /// one that can see the leased instances too.
    ///
    /// No drain, deliberately: this is called from a signal handler, the leases
    /// that are out belong to rollouts that will never resume, and the
    /// `DRAIN_TIMEOUT` wait `shutdown` does would just be thirty seconds of
    /// delay before the same teardown.
    async fn force_cleanup(&self) -> usize {
        {
            let mut state = self.state.lock().expect("pool state");
            state.closed = true;
        }
        self.available.notify_waiters();
        self.source.force_cleanup().await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use retrograd_agent_core::{DirEntry, ExecOutput, ExecRequest, SandboxLimits};

    use super::*;

    /// A sandbox that does nothing but be counted. Everything the pool decides
    /// is decided without a daemon, which is why these tests live in the fast
    /// lane and the container lane only has to prove that a real container
    /// behaves like this one.
    struct FakeSandbox {
        id: usize,
        recycles: AtomicUsize,
        destroyed: Arc<AtomicUsize>,
        recycle_fails: bool,
        healthy: bool,
    }

    #[async_trait]
    impl Sandbox for FakeSandbox {
        async fn exec(&self, _request: ExecRequest) -> Result<ExecOutput> {
            Ok(ExecOutput {
                exit_code: Some(0),
                stdout: self.id.to_string(),
                stderr: String::new(),
                timed_out: false,
                truncated: false,
            })
        }
        async fn read_file(&self, _path: &str) -> Result<Vec<u8>> {
            Ok(Vec::new())
        }
        async fn write_file(&self, _path: &str, _bytes: &[u8]) -> Result<()> {
            Ok(())
        }
        async fn list_dir(&self, _path: &str) -> Result<Vec<DirEntry>> {
            Ok(Vec::new())
        }
        async fn remove(&self, _path: &str) -> Result<()> {
            Ok(())
        }
        fn workdir(&self) -> &str {
            "/work"
        }
        fn limits(&self) -> SandboxLimits {
            SandboxLimits::default()
        }
    }

    struct FakeInstance {
        sandbox: Arc<FakeSandbox>,
    }

    #[async_trait]
    impl ManagedSandbox for FakeInstance {
        fn sandbox(&self) -> Arc<dyn Sandbox> {
            self.sandbox.clone()
        }
        async fn recycle(&self) -> Result<()> {
            self.sandbox.recycles.fetch_add(1, Ordering::SeqCst);
            if self.sandbox.recycle_fails {
                Err(Error::Tool("cleanup failed".into()))
            } else {
                Ok(())
            }
        }
        async fn destroy(&self) {
            self.sandbox.destroyed.fetch_add(1, Ordering::SeqCst);
        }
        fn healthy(&self) -> bool {
            self.sandbox.healthy
        }
    }

    #[derive(Default)]
    struct FakeSource {
        created: AtomicUsize,
        destroyed: Arc<AtomicUsize>,
        recycle_fails: bool,
        healthy_instances: bool,
        /// Stands in for "every container carrying this run's label": the real
        /// source counts them at the daemon, not in the pool's bookkeeping.
        swept: Arc<AtomicUsize>,
    }

    impl FakeSource {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                healthy_instances: true,
                ..Default::default()
            })
        }
    }

    #[async_trait]
    impl SandboxSource for FakeSource {
        async fn create(&self) -> Result<Arc<dyn ManagedSandbox>> {
            let id = self.created.fetch_add(1, Ordering::SeqCst);
            Ok(Arc::new(FakeInstance {
                sandbox: Arc::new(FakeSandbox {
                    id,
                    recycles: AtomicUsize::new(0),
                    destroyed: self.destroyed.clone(),
                    recycle_fails: self.recycle_fails,
                    healthy: self.healthy_instances,
                }),
            }))
        }

        async fn force_cleanup(&self) -> usize {
            let swept = self.created.load(Ordering::SeqCst);
            self.swept.store(swept, Ordering::SeqCst);
            swept
        }
    }

    struct BlockingSource {
        started: Notify,
        release: Notify,
        destroyed: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl SandboxSource for BlockingSource {
        async fn create(&self) -> Result<Arc<dyn ManagedSandbox>> {
            self.started.notify_one();
            self.release.notified().await;
            Ok(Arc::new(FakeInstance {
                sandbox: Arc::new(FakeSandbox {
                    id: 0,
                    recycles: AtomicUsize::new(0),
                    destroyed: self.destroyed.clone(),
                    recycle_fails: false,
                    healthy: true,
                }),
            }))
        }
    }

    async fn settle() {
        // The janitor runs on its own task; a release is visible once it has had
        // a turn.
        for _ in 0..50 {
            tokio::task::yield_now().await;
        }
    }

    #[tokio::test]
    async fn never_reusing_destroys_every_sandbox_at_the_end_of_its_episode() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(source.clone(), PoolConfig::default());

        drop(pool.acquire().await.unwrap());
        drop(pool.acquire().await.unwrap());
        settle().await;

        assert_eq!(source.created.load(Ordering::SeqCst), 2);
        assert_eq!(source.destroyed.load(Ordering::SeqCst), 2);
        assert_eq!(pool.live(), 0);
    }

    #[tokio::test]
    async fn recycling_hands_the_same_container_back_once_it_is_clean() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(
            source.clone(),
            PoolConfig {
                reuse: ReusePolicy::Workspace,
                ..PoolConfig::default()
            },
        );

        drop(pool.acquire().await.unwrap());
        settle().await;
        assert_eq!(pool.idle(), 1);

        let second = pool.acquire().await.unwrap();
        // One creation for two episodes, and the wipe happened in between.
        assert_eq!(source.created.load(Ordering::SeqCst), 1);
        assert_eq!(source.destroyed.load(Ordering::SeqCst), 0);
        drop(second);
    }

    /// The rule that protects the GRPO baseline: a cleanup that did not clearly
    /// succeed must cost a container, never produce a reused one.
    #[tokio::test]
    async fn a_doubtful_cleanup_destroys_instead_of_recycling() {
        for (recycle_fails, healthy, poisoned) in [
            (true, true, false),
            (false, false, false),
            (false, true, true),
        ] {
            let source = Arc::new(FakeSource {
                recycle_fails,
                healthy_instances: healthy,
                ..Default::default()
            });
            let pool = SandboxPool::new(
                source.clone(),
                PoolConfig {
                    reuse: ReusePolicy::Workspace,
                    ..PoolConfig::default()
                },
            );
            let mut lease = pool.acquire().await.unwrap();
            if poisoned {
                lease.poison();
            }
            drop(lease);
            settle().await;

            assert_eq!(pool.idle(), 0, "{recycle_fails} {healthy} {poisoned}");
            assert_eq!(source.destroyed.load(Ordering::SeqCst), 1);
            assert_eq!(pool.live(), 0);
        }
    }

    #[tokio::test]
    async fn a_container_is_retired_after_enough_episodes() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(
            source.clone(),
            PoolConfig {
                reuse: ReusePolicy::Workspace,
                max_leases_per_container: 2,
                ..PoolConfig::default()
            },
        );
        for _ in 0..3 {
            drop(pool.acquire().await.unwrap());
            settle().await;
        }
        // Two episodes on the first container, then it retires and a second one
        // takes over.
        assert_eq!(source.created.load(Ordering::SeqCst), 2);
        assert_eq!(source.destroyed.load(Ordering::SeqCst), 1);
    }

    /// A group larger than the bound must serialize, not open one container per
    /// member on a machine that is also holding a model.
    #[tokio::test]
    async fn backpressure_bounds_live_containers_whatever_the_caller_asks() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(
            source.clone(),
            PoolConfig {
                max_live: 2,
                ..PoolConfig::default()
            },
        );

        let first = pool.acquire().await.unwrap();
        let second = pool.acquire().await.unwrap();
        assert_eq!(pool.live(), 2);

        let waiting = {
            let pool = pool.clone();
            tokio::spawn(async move {
                pool.acquire()
                    .await
                    .map(|lease| lease.workdir().to_string())
            })
        };
        // Nothing has been released, so the third caller cannot have been served.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiting.is_finished());

        drop(first);
        let third = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("the waiting caller is served once a slot frees")
            .unwrap();
        assert!(third.is_ok());
        drop(second);
        settle().await;
        assert!(pool.live() <= 2);
    }

    #[tokio::test]
    async fn prewarming_creates_ahead_and_makes_the_next_acquire_a_reuse() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(
            source.clone(),
            PoolConfig {
                max_live: 4,
                reuse: ReusePolicy::Workspace,
                ..PoolConfig::default()
            },
        );
        pool.prewarm(3).await.unwrap();
        assert_eq!(pool.idle(), 3);
        assert_eq!(source.created.load(Ordering::SeqCst), 3);

        let lease = pool.acquire().await.unwrap();
        assert_eq!(source.created.load(Ordering::SeqCst), 3, "acquire reused");
        drop(lease);
        settle().await;

        // Prewarming never exceeds the live bound.
        pool.prewarm(99).await.unwrap();
        assert!(pool.live() <= 4);
    }

    #[tokio::test]
    async fn shutdown_destroys_the_idle_set_and_refuses_new_leases() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(
            source.clone(),
            PoolConfig {
                reuse: ReusePolicy::Workspace,
                ..PoolConfig::default()
            },
        );
        pool.prewarm(2).await.unwrap();
        pool.shutdown().await;

        assert_eq!(source.destroyed.load(Ordering::SeqCst), 2);
        assert_eq!(pool.live(), 0);
        assert!(pool.acquire().await.is_err());
    }

    #[tokio::test]
    async fn shutdown_destroys_a_prewarm_that_finishes_after_close() {
        let source = Arc::new(BlockingSource {
            started: Notify::new(),
            release: Notify::new(),
            destroyed: Arc::new(AtomicUsize::new(0)),
        });
        let pool = Arc::new(SandboxPool::new(
            source.clone(),
            PoolConfig {
                reuse: ReusePolicy::Workspace,
                ..PoolConfig::default()
            },
        ));
        let warming = tokio::spawn({
            let pool = pool.clone();
            async move { pool.prewarm(1).await }
        });
        source.started.notified().await;
        let closing = tokio::spawn({
            let pool = pool.clone();
            async move { pool.shutdown().await }
        });
        while !pool.state.lock().unwrap().closed {
            tokio::task::yield_now().await;
        }
        source.release.notify_one();
        warming.await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(1), closing)
            .await
            .expect("shutdown must not wait for the drain timeout")
            .unwrap();
        assert_eq!(source.destroyed.load(Ordering::SeqCst), 1);
        assert_eq!(pool.live(), 0);
        assert_eq!(pool.idle(), 0);
    }

    /// The case `shutdown` cannot serve, and the reason `force_cleanup` goes
    /// through the source instead of the pool's own bookkeeping: a lease that is
    /// still out. `shutdown` would wait `DRAIN_TIMEOUT` for it to come back,
    /// which it never will, the process is exiting - and leave its container
    /// behind. The source counts it because it enumerates by label, not by what
    /// the pool happens to be holding.
    #[tokio::test]
    async fn force_cleanup_takes_down_a_leased_sandbox_too_and_closes_the_pool() {
        let source = FakeSource::new();
        let pool = SandboxPool::new(
            source.clone(),
            PoolConfig {
                reuse: ReusePolicy::Workspace,
                ..PoolConfig::default()
            },
        );
        pool.prewarm(2).await.unwrap();
        let _held = pool.acquire().await.unwrap();
        assert_eq!(pool.live(), 2);

        assert_eq!(pool.force_cleanup().await, 2);
        assert_eq!(source.swept.load(Ordering::SeqCst), 2);
        assert!(pool.acquire().await.is_err());
    }
}
