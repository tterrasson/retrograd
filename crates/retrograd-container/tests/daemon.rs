//! What only a real daemon can prove.
//!
//! Everything decidable without one - the pool's policy, the spec's refusals,
//! the reaper's rule, path handling - is a unit test in the fast lane. What is
//! left here is the part where being wrong is invisible in code review: that the
//! limits are actually applied by the runtime, that two episodes on a recycled
//! container really cannot see each other, and that the same command in two
//! different containers produces the same bytes.
//!
//! Run through `scripts/test-container.sh`, which announces itself skipped
//! rather than green when no daemon answers.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use retrograd_agent_core::{ExecRequest, SandboxProvider};
use retrograd_container::{
    ContainerSource, ContainerSpec, DaemonLocality, DockerClient, NetworkMode, PoolConfig,
    ReusePolicy, RunLabels, SandboxPool, reap,
};

/// Debian-based, because the built-in listing uses GNU `find -printf` and the
/// timeout wrapper uses coreutils `timeout`.
fn image() -> String {
    std::env::var("RETRO_CONTAINER_TEST_IMAGE").unwrap_or_else(|_| "debian:bookworm-slim".into())
}

fn run_id() -> String {
    format!("test-{}-{}", std::process::id(), rand_suffix())
}

/// No `rand` dependency for a label suffix.
fn rand_suffix() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| since.subsec_nanos() as u64)
        .unwrap_or(0)
}

async fn client() -> DockerClient {
    DockerClient::connect()
        .await
        .expect("a daemon: run this through scripts/test-container.sh")
}

async fn pool(reuse: ReusePolicy, spec: ContainerSpec, run: &str) -> Arc<SandboxPool> {
    let source = ContainerSource::new(client().await, spec, run.to_string())
        .await
        .expect("create a container source");
    SandboxPool::new(
        Arc::new(source),
        PoolConfig {
            max_live: 2,
            reuse,
            ..PoolConfig::default()
        },
    )
}

/// Counts what is still labelled as ours. Every test ends by asserting this is
/// zero, including the ones that failed on the way - the pool's shutdown runs
/// regardless.
async fn live_containers(client: &DockerClient, run: &str) -> usize {
    use bollard::query_parameters::ListContainersOptionsBuilder;
    let filters = HashMap::from([("label".to_string(), vec![format!("retrograd.run={run}")])]);
    let options = ListContainersOptionsBuilder::default()
        .all(true)
        .filters(&filters)
        .build();
    client
        .docker()
        .list_containers(Some(options))
        .await
        .expect("list containers")
        .len()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_container_round_trips_commands_and_files() {
    let run = run_id();
    let pool = pool(ReusePolicy::Never, ContainerSpec::new(image()), &run).await;
    let lease = pool.acquire().await.expect("acquire a sandbox");

    let output = lease
        .exec(ExecRequest::new(["echo", "hello"]))
        .await
        .unwrap();
    assert_eq!(output.stdout.trim(), "hello");
    assert!(output.succeeded());

    // A non-zero exit is an observation, not an error: this is the split the
    // whole stack is built on.
    let failed = lease
        .exec(ExecRequest::new(["sh", "-c", "echo oops >&2; exit 3"]))
        .await
        .expect("a failing command is not a broken sandbox");
    assert_eq!(failed.exit_code, Some(3));
    assert_eq!(failed.stderr.trim(), "oops");

    lease.write_file("src/a.py", b"print(1)\n").await.unwrap();
    assert_eq!(lease.read_file("src/a.py").await.unwrap(), b"print(1)\n");

    lease.write_file("b.txt", b"b").await.unwrap();
    let entries = lease.list_dir(".").await.unwrap();
    let names: Vec<_> = entries.iter().map(|entry| entry.path.as_str()).collect();
    assert_eq!(names, ["b.txt", "src"]);

    lease.remove("src").await.unwrap();
    assert!(lease.read_file("src/a.py").await.is_err());

    let stdin = lease
        .exec(ExecRequest::new(["cat"]).with_stdin("piped"))
        .await
        .unwrap();
    assert_eq!(stdin.stdout.trim(), "piped");

    drop(lease);
    pool.shutdown().await;
    assert_eq!(live_containers(&client().await, &run).await, 0);
}

/// The test that protects the GRPO baseline. If it ever fails, a group's members
/// are no longer independent samples and the relative advantage is measuring
/// leftovers instead of the policy.
#[tokio::test(flavor = "multi_thread")]
async fn a_recycled_container_shows_the_next_episode_nothing() {
    let run = run_id();
    let pool = pool(ReusePolicy::Workspace, ContainerSpec::new(image()), &run).await;

    let first = pool.acquire().await.unwrap();
    first
        .write_file("marker.txt", b"episode one")
        .await
        .unwrap();
    first
        .exec(ExecRequest::new([
            "sh",
            "-c",
            // The background process is what `recycle` must kill; the marker in
            // `/tmp` is what identifies the container further down.
            "echo one > /tmp/same-container; nohup sleep 600 >/dev/null 2>&1 &",
        ]))
        .await
        .unwrap();
    drop(first);

    // The wipe happens on the janitor task, between the release and the next
    // acquire. Waiting for the container to reappear *idle* is what makes this
    // test about recycling: an `acquire` issued while the release is still in
    // flight is served by a brand-new container - legitimately, since the live
    // bound has room - and the reuse path would never be exercised at all.
    // Reaching this point at all is already the first assertion: a cleanup that
    // fails destroys the container instead, and the idle set stays empty.
    let idle = tokio::time::timeout(Duration::from_secs(60), async {
        while pool.idle() == 0 {
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    })
    .await;
    assert!(
        idle.is_ok(),
        "the container never came back to the idle set: recycling failed and it was destroyed"
    );
    let second = pool.acquire().await.unwrap();
    assert!(
        second.read_file("marker.txt").await.is_err(),
        "the previous episode's files must be gone"
    );
    assert!(second.list_dir(".").await.unwrap().is_empty());
    // Straight from /proc rather than through pgrep, which a slim image has no
    // reason to carry.
    let leftovers = second
        .exec(ExecRequest::new([
            "sh",
            "-c",
            "for p in /proc/[0-9]*; do tr '\\0' ' ' < $p/cmdline 2>/dev/null; echo; done",
        ]))
        .await
        .unwrap();
    assert!(
        !leftovers.stdout.contains("sleep 600"),
        "the previous episode's processes must be gone: {}",
        leftovers.stdout
    );

    // A brand-new container satisfies every assertion above, which is how a
    // cleanup that quietly failed - and made the pool destroy instead of recycle
    // - could leave `ReusePolicy::Workspace`, the prewarming and
    // `env/pool_reuse_fraction` doing nothing at all while this test stayed
    // green. So the reuse itself is asserted, from both sides.
    //
    // `/tmp` is the container's own tmpfs and is not what `recycle` wipes: it
    // survives exactly one recycling and not one re-creation, which makes it the
    // cheapest possible proof that this is the same container. It is also, spelt
    // out, the leak `ReusePolicy::Workspace` documents accepting.
    let marker = second
        .exec(ExecRequest::new(["cat", "/tmp/same-container"]))
        .await
        .unwrap();
    assert_eq!(
        marker.stdout.trim(),
        "one",
        "the second episode ran in a different container: the pool recreated instead of recycling"
    );
    let reuse = pool
        .metric_values()
        .into_iter()
        .find(|metric| metric.name == "env/pool_reuse_fraction")
        .expect("the pool reports its reuse fraction")
        .value;
    assert_eq!(reuse, 0.5, "one of the two acquires was a reuse");

    drop(second);
    pool.shutdown().await;
    assert_eq!(live_containers(&client().await, &run).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn the_limits_are_applied_by_the_runtime_and_not_only_by_us() {
    let run = run_id();
    let mut spec = ContainerSpec::new(image());
    spec.limits.exec_timeout_secs = 2;
    spec.limits.memory_mb = 64;
    let pool = pool(ReusePolicy::Never, spec, &run).await;
    let lease = pool.acquire().await.unwrap();

    let timed_out = lease
        .exec(ExecRequest::new(["sleep", "60"]).with_timeout(Duration::from_secs(2)))
        .await
        .expect("a timeout is an observation");
    assert!(timed_out.timed_out);
    assert_eq!(timed_out.stderr, "command timed out after 2 seconds");

    let truncated = lease
        .exec(
            ExecRequest::new(["sh", "-c", "printf 'é%.0s' $(seq 1 5000)"])
                .with_max_output_bytes(64),
        )
        .await
        .unwrap();
    assert!(truncated.truncated);
    // Cut on a character boundary: half a code point in the prompt is a
    // tokenizer problem later.
    assert!(truncated.stdout.is_char_boundary(truncated.stdout.len()));

    // A greedy allocation must fail as a readable observation, not as a broken
    // sandbox - the policy has to be able to react to it.
    let greedy = lease
        .exec(
            // `dd` really allocates its block, unlike a pipeline that streams
            // the same bytes through a small buffer.
            ExecRequest::new(["dd", "if=/dev/zero", "of=/dev/null", "bs=200M", "count=1"])
                .with_timeout(Duration::from_secs(20)),
        )
        .await
        .expect("hitting the memory limit is not a broken sandbox");
    assert!(!greedy.succeeded());

    drop(lease);
    pool.shutdown().await;
    assert_eq!(live_containers(&client().await, &run).await, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn there_is_no_network_unless_someone_asked_for_one() {
    let run = run_id();
    let pool = pool(ReusePolicy::Never, ContainerSpec::new(image()), &run).await;
    let lease = pool.acquire().await.unwrap();
    assert!(!lease.limits().network);

    let reached = lease
        .exec(
            ExecRequest::new(["getent", "hosts", "example.com"])
                .with_timeout(Duration::from_secs(10)),
        )
        .await
        .unwrap();
    assert!(!reached.succeeded(), "the sandbox reached the network");

    drop(lease);
    pool.shutdown().await;
    assert_eq!(live_containers(&client().await, &run).await, 0);
}

/// Observations become training tokens, so the same action in two different
/// containers has to produce the same bytes. Anything that differs - a hostname,
/// a container id, a path, a duration - is noise the group-relative baseline
/// would happily measure.
///
/// What this can and cannot prove: the guarantee is that *the sandbox* adds no
/// per-container difference, not that every command is deterministic. With a
/// `bash` tool the model can always run `date` or `echo $$`, and no wrapper here
/// changes that; the answer to those is the task author's tool list (`deny`),
/// not the runtime. So the commands below are the ones whose answer the runtime
/// picks for us - the working directory, the hostname, a missing file's wording,
/// a listing's order - and `hostname` is here because it is the one Docker and
/// Podman would otherwise fill in with the container's own id.
#[tokio::test(flavor = "multi_thread")]
async fn the_same_command_in_two_containers_produces_the_same_bytes() {
    let run = run_id();
    let pool = pool(ReusePolicy::Never, ContainerSpec::new(image()), &run).await;

    let first = pool.acquire().await.unwrap();
    let second = pool.acquire().await.unwrap();
    for request in [
        ExecRequest::new(["pwd"]),
        ExecRequest::new(["sh", "-c", "cd /work && ls -a"]),
        ExecRequest::new(["python3", "-c", "import os; print(os.getcwd())"]),
        ExecRequest::new(["cat", "/nope"]),
        // The one the runtime decides for us if we let it: Docker and Podman
        // both default a container's hostname to its own id.
        ExecRequest::new(["hostname"]),
        ExecRequest::new(["sh", "-c", "echo \"$HOSTNAME\"; uname -n"]),
    ] {
        let left = first.exec(request.clone()).await.unwrap();
        let right = second.exec(request.clone()).await.unwrap();
        assert_eq!(left.stdout, right.stdout, "{:?}", request.argv);
        assert_eq!(left.stderr, right.stderr, "{:?}", request.argv);
        assert_eq!(left.exit_code, right.exit_code, "{:?}", request.argv);
    }

    // Not merely equal to each other: equal to a value nothing per-container
    // could have produced, so a future default that happens to agree between
    // two containers of the same image would still be caught.
    let name = first.exec(ExecRequest::new(["hostname"])).await.unwrap();
    assert_eq!(name.stdout.trim(), retrograd_container::spec::HOSTNAME);

    // Same for a listing, whose order must not come from the filesystem.
    for name in ["c.txt", "a.txt", "b.txt"] {
        first.write_file(name, b"x").await.unwrap();
    }
    for name in ["b.txt", "c.txt", "a.txt"] {
        second.write_file(name, b"x").await.unwrap();
    }
    assert_eq!(
        first.list_dir(".").await.unwrap(),
        second.list_dir(".").await.unwrap()
    );

    drop(first);
    drop(second);
    pool.shutdown().await;
    assert_eq!(live_containers(&client().await, &run).await, 0);
}

/// A run killed with SIGKILL runs no `Drop` and closes no pool. The sweep at
/// startup is what keeps the next run from inheriting sixty-four containers.
#[tokio::test(flavor = "multi_thread")]
async fn the_reaper_collects_what_a_dead_run_left_behind() {
    use bollard::models::ContainerCreateBody;
    use bollard::query_parameters::{CreateContainerOptions, StartContainerOptions};

    let client = client().await;
    let dead_run = run_id();

    // A container labelled by a run that is over: its run id, and an owner pid
    // that is not alive. Created by hand, because a `ContainerSource` labels
    // with this process's pid - alive, so the reaper would rightly spare it.
    let mut dead = RunLabels::new(dead_run.clone(), "fp");
    dead.owner = "999999".to_string(); // a pid nobody has
    let image = retrograd_container::ensure_image(&client, &image())
        .await
        .unwrap();
    let body = ContainerCreateBody {
        image: Some(image),
        entrypoint: Some(vec!["/bin/sh".into(), "-c".into()]),
        cmd: Some(vec!["while :; do sleep 3600; done".into()]),
        labels: Some(dead.to_map()),
        ..Default::default()
    };
    let created = client
        .docker()
        .create_container(None::<CreateContainerOptions>, body)
        .await
        .unwrap();
    client
        .docker()
        .start_container(&created.id, None::<StartContainerOptions>)
        .await
        .unwrap();
    assert_eq!(live_containers(&client, &dead_run).await, 1);

    // The sweep, as the next run's startup does it: our own pid, alive.
    let ours = RunLabels::new(run_id(), "fp");
    let removed = reap(&client, &ours, DaemonLocality::Local).await.unwrap();
    assert!(removed >= 1);
    assert_eq!(live_containers(&client, &dead_run).await, 0);
}

/// The interrupt path, which is the exact inverse of the sweep above: a live
/// pid, a lease still out, and everything goes anyway.
///
/// Worth a daemon test rather than a unit test because the two rules are one
/// mistaken argument apart. `reap` spares these containers on purpose - a live
/// owner is a run in flight, and killing it would be the worst thing this crate
/// could do. `force_cleanup` must take exactly the same containers down, and
/// only a real listing proves it selects by run label instead of inheriting
/// `reap`'s liveness check.
#[tokio::test(flavor = "multi_thread")]
async fn a_forced_cleanup_removes_this_run_leases_and_live_pid_included() {
    let client = client().await;
    let run = run_id();

    let source = ContainerSource::new(client.clone(), ContainerSpec::new(image()), run.clone())
        .await
        .unwrap();
    let pool = SandboxPool::new(Arc::new(source), PoolConfig::default());
    let held = pool.acquire().await.unwrap();
    pool.prewarm(1).await.unwrap();
    assert_eq!(live_containers(&client, &run).await, 2);

    // What the ordinary sweep would do with them: nothing. Our pid is alive, so
    // every one of these looks like a run in flight - which it is. Asserted on
    // this run's containers rather than on the sweep's return value, which also
    // counts whatever previous crashes left on the machine.
    reap(
        &client,
        &RunLabels::new(run.clone(), "fp"),
        DaemonLocality::Local,
    )
    .await
    .unwrap();
    assert_eq!(
        live_containers(&client, &run).await,
        2,
        "the startup sweep must never touch a live run"
    );

    assert_eq!(pool.force_cleanup().await, 2);
    assert_eq!(live_containers(&client, &run).await, 0);
    // The lease outlived its container, which is the whole point: on this path
    // nobody is coming back to return it.
    drop(held);
}

/// The spec refuses what would hand the machine away, whatever a configuration
/// file says - checked here too, because it must hold on the path that actually
/// talks to the daemon.
#[tokio::test(flavor = "multi_thread")]
async fn a_source_refuses_an_unsafe_spec_before_touching_the_daemon() {
    let mut spec = ContainerSpec::new(image());
    spec.user = "0:0".into();
    let refused = ContainerSource::new(client().await, spec, run_id()).await;
    let error = match refused {
        Ok(_) => panic!("running as root must be refused"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("root"), "{error}");

    let mut spec = ContainerSpec::new(image());
    spec.network = NetworkMode::Bridge;
    // Widening is allowed - it just has to be asked for.
    ContainerSource::new(client().await, spec, run_id())
        .await
        .expect("an explicit network is a decision, not a refusal");
}
