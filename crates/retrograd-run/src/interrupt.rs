//! Turning Ctrl+C into something that cleans up after itself.
//!
//! Without this, `SIGINT` takes the default action: the process dies, no `Drop`
//! runs, `EnvironmentFactory::shutdown` is never reached, and an agentic run
//! leaves one container per trajectory in flight. They do not exit on their own
//! (a sandbox container idles on a `sleep` loop), so `AutoRemove` does not fire
//! either, and the only thing that ever collects them is the startup sweep of
//! whatever run comes next.
//!
//! What the two stages mean is in [`retrograd_agent::interrupt`]. This module is
//! the half that owns the process: which signal escalates, and what running the
//! cleanup actually does.
//!
//! # Why a thread of its own
//!
//! The agentic trainer is pinned to a `LocalSet`, and the optimizer step inside
//! it is a long synchronous call into the engine. Cleanup must not depend on
//! that local driver reaching its next await: a Ctrl+C landing during a GPU step
//! has to be observed immediately, before an operator gives up and reaches for
//! `kill -9`. The listener therefore owns a runtime that nothing else blocks.
//!
//! Registering the handler is also what stops the default action, so from the
//! call to [`install`] on, no interrupt kills the process on its own. That is a
//! debt: having taken the default action away, the forced path owes an exit
//! under every outcome, including the ones where the cleanup does not answer.
//! It pays it by racing the cleanup against [`FORCE_CLEANUP_GRACE`] and against
//! any further signal, and by exiting on all three - never by awaiting the
//! cleanup and hoping it returns.

use std::sync::Arc;

use retrograd_agent::EnvironmentFactory;
use retrograd_agent::interrupt::{self, Level};

/// The shell's `128 + signal` convention, kept exactly because a supervisor
/// reads the difference: 130 is "the operator gave up", 143 is "we asked it to
/// stop". Reporting one as the other turns a normal shutdown into an incident.
const EXIT_INTERRUPTED: i32 = 130;
const EXIT_TERMINATED: i32 = 143;

/// How long a forced interrupt waits for the environments before leaving
/// without them.
///
/// Wide on purpose: the removals run concurrently, but each one is a `SIGKILL`,
/// a wait and an unmount inside a daemon that may itself be under the memory
/// pressure the run just created, and a cleanup that would have finished in
/// forty seconds is worth waiting for - the alternative is containers left for
/// the next run's startup sweep. It is a bound, not a budget: the normal case
/// returns in well under a second and never sees it.
const FORCE_CLEANUP_GRACE: std::time::Duration = std::time::Duration::from_secs(60);

/// Why a forced interrupt stopped waiting, which is the only thing that changes
/// between the three ways out - the exit itself is the same on all of them.
#[cfg(unix)]
enum Forced {
    /// The cleanup finished, with the number of environments it removed.
    Removed(usize),
    /// The grace ran out.
    Grace,
    /// A further signal arrived while the cleanup was still running. The
    /// operator asked twice; waiting any longer is what makes a `kill -9`
    /// look like the only option left.
    Signal,
}

/// Ends the process without running anything on the way out.
///
/// Not `std::process::exit`: that one runs `atexit` handlers and static
/// destructors, and the ones registered by the CUDA runtime and by llama.cpp
/// tear down state the *training* thread is still inside - it is mid-kernel, it
/// was never told to stop, and nothing here can tell it to. The observed result
/// is a segfault after the cleanup has already succeeded, so the operator sees
/// the containers go and then a crash.
///
/// `_exit` skips all of it and hands the process straight to the kernel, which
/// is the correct semantics anyway: the containers are already gone, the
/// adapter is not being saved on this path, and there is nothing left worth
/// flushing.
#[cfg(unix)]
fn exit_now(code: i32) -> ! {
    // The messages above are the only output on this path, and both are
    // unbuffered stderr; nothing is waiting in a buffer that `_exit` would drop.
    unsafe { libc::_exit(code) }
}

/// Installs the interrupt handler for the rest of the process.
///
/// `environment` is what a forced interrupt tears down; a run without one still
/// installs the handler, so Ctrl+C keeps its two stages and the first one still
/// buys a checkpoint.
///
/// Best-effort by construction: if the thread or its runtime cannot be created,
/// the run proceeds with the default signal behaviour rather than refusing to
/// start over a cleanup nicety.
pub(crate) fn install(environment: Option<Arc<dyn EnvironmentFactory>>) {
    #[cfg(unix)]
    {
        let _ = std::thread::Builder::new()
            .name("retrograd-interrupt".into())
            .spawn(move || listen(environment));
    }
    #[cfg(not(unix))]
    {
        let _ = environment;
    }
}

#[cfg(unix)]
fn listen(environment: Option<Arc<dyn EnvironmentFactory>>) {
    use tokio::signal::unix::{SignalKind, signal};

    let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
    else {
        return;
    };
    runtime.block_on(async move {
        let (Ok(mut sigint), Ok(mut sigterm)) = (
            signal(SignalKind::interrupt()),
            signal(SignalKind::terminate()),
        ) else {
            return;
        };
        loop {
            // A SIGTERM skips the graceful stage. Whoever sent it - an init
            // system, an orchestrator, a `kill` - has already decided, and
            // waiting out an update only delays the SIGKILL that follows it.
            let (level, code) = tokio::select! {
                _ = sigint.recv() => (interrupt::escalate(), EXIT_INTERRUPTED),
                _ = sigterm.recv() => (interrupt::force(), EXIT_TERMINATED),
            };
            match level {
                Level::Graceful => {
                    // eprintln, not the observer: the observer belongs to the
                    // training thread, which is mid-update and holds it. Nor
                    // `tracing`: these lines answer a key press, and the CLI's
                    // subscriber would prefix them with a timestamp and a target.
                    eprintln!(
                        "\ninterrupt: stopping at the end of the current update - its rollouts, \
                         evaluation and checkpoint still complete, so this can take a few \
                         minutes. Press Ctrl+C again to drop the update and tear the \
                         environments down now."
                    );
                }
                Level::Running | Level::Force => {
                    // Nothing here may wait without a way out. The cleanup is a
                    // conversation with a daemon that can be slow or wedged,
                    // and the branch it lives in is the one that ends the
                    // process: an unbounded await inside it means every later
                    // signal is queued and never consumed - the handler is
                    // installed, so the default action is gone too, and the
                    // process becomes unkillable short of `SIGKILL`. That is
                    // strictly worse than not installing a handler at all,
                    // which is why the grace and the further signals race the
                    // cleanup rather than following it.
                    let forced = match &environment {
                        Some(environment) => tokio::select! {
                            removed = environment.force_cleanup() => Forced::Removed(removed),
                            _ = tokio::time::sleep(FORCE_CLEANUP_GRACE) => Forced::Grace,
                            _ = sigint.recv()  => Forced::Signal,
                            _ = sigterm.recv() => Forced::Signal,
                        },
                        None => Forced::Removed(0),
                    };
                    match forced {
                        Forced::Removed(removed) => {
                            eprintln!("interrupt: forced, {removed} environment(s) removed");
                        }
                        // Said plainly, because it is the one outcome that
                        // leaves work for someone: the containers are still
                        // there, and what collects them is the next run's
                        // startup sweep.
                        Forced::Grace => eprintln!(
                            "interrupt: forced, but the environments did not go in \
                             {}s - leaving them behind; the next run's startup sweep \
                             removes them, or `docker ps` shows them now",
                            FORCE_CLEANUP_GRACE.as_secs()
                        ),
                        Forced::Signal => eprintln!(
                            "interrupt: leaving now, the environment cleanup was still \
                             running - the next run's startup sweep removes what is left"
                        ),
                    }
                    exit_now(code);
                }
            }
        }
    });
}
