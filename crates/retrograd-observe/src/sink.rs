//! The producer side: a handle the loops call from their own thread, which
//! never waits for the disk.

use std::fs::{self, File, TryLockError};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender, TrySendError};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::batch::{ObserveBatch, RunInfo, UpdateStatus, UpdateSummary};
use crate::error::ObserveError;
use crate::{TrajectoryObserver, writer};

const CHANNEL_CAPACITY: usize = 64;
const FINISH_TIMEOUT: Duration = Duration::from_secs(2);
const LOCK_FILE: &str = ".observe.lock";

/// What the sink needs from `[observe]`, without depending on the
/// configuration crate.
#[derive(Clone, Debug)]
pub struct SinkConfig {
    pub directory: PathBuf,
    pub every: u32,
    pub max_text_chars: usize,
}

/// State shared by the owner, every handle and the writer thread.
pub(crate) struct Shared {
    sender: Mutex<Option<SyncSender<ObserveBatch>>>,
    every: u32,
    accepting: AtomicBool,
    dropped: AtomicU64,
    warned_full: AtomicBool,
    warnings: Mutex<Vec<String>>,
    /// Held by a test to stall the writer between two batches.
    #[cfg(test)]
    pub(crate) gate: Mutex<()>,
}

fn locked<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // Nothing held under these locks can be left half-updated.
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

impl Shared {
    pub(crate) fn warn(&self, message: String) {
        locked(&self.warnings).push(message);
    }

    /// Stops accepting batches for good and reports why, once.
    pub(crate) fn fail(&self, message: String) {
        self.accepting.store(false, Ordering::SeqCst);
        self.warn(message);
    }

    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    fn close(&self) {
        self.accepting.store(false, Ordering::SeqCst);
        locked(&self.sender).take();
    }
}

impl TrajectoryObserver for Shared {
    fn wants(&self, update: u32) -> bool {
        self.accepting.load(Ordering::Relaxed) && update.is_multiple_of(self.every)
    }

    fn observe(&self, batch: ObserveBatch) {
        if !self.accepting.load(Ordering::Relaxed) {
            return;
        }
        let sender = locked(&self.sender);
        let Some(sender) = sender.as_ref() else {
            return;
        };
        match sender.try_send(batch) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                if !self.warned_full.swap(true, Ordering::Relaxed) {
                    self.warn(
                        "observe: the export writer is behind, batches are being dropped \
                         (count in observe/dropped_batches)"
                            .into(),
                    );
                }
            }
            // The writer is gone; it already said why.
            Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                self.accepting.store(false, Ordering::Relaxed);
            }
        }
    }
}

/// Owner of the export: the only value that can close it. Handles from
/// [`ObserveSink::observer`] only send.
pub struct ObserveSink {
    shared: Arc<Shared>,
    writer: Option<JoinHandle<()>>,
}

impl ObserveSink {
    /// Creates the directory and starts the writer. A directory that cannot be
    /// created, or a writer thread that cannot start, returns an error. If the
    /// directory cannot be locked, yields a disabled sink and a warning.
    pub fn open(config: &SinkConfig, run: RunInfo) -> Result<Self, ObserveError> {
        Self::open_with_capacity(config, run, CHANNEL_CAPACITY)
    }

    pub(crate) fn open_with_capacity(
        config: &SinkConfig,
        run: RunInfo,
        capacity: usize,
    ) -> Result<Self, ObserveError> {
        let directory_error = |source| ObserveError::Directory {
            path: config.directory.clone(),
            source,
        };
        fs::create_dir_all(&config.directory).map_err(directory_error)?;
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let shared = Arc::new(Shared {
            sender: Mutex::new(Some(sender)),
            every: config.every.max(1),
            accepting: AtomicBool::new(true),
            dropped: AtomicU64::new(0),
            warned_full: AtomicBool::new(false),
            warnings: Mutex::new(Vec::new()),
            #[cfg(test)]
            gate: Mutex::new(()),
        });
        let refusal = match lock_directory(&config.directory).unwrap_or_else(Lock::Unsupported) {
            Lock::Held(file) => Ok(file),
            Lock::Busy => Err("another run is exporting there".to_string()),
            Lock::Unsupported(error) => Err(format!("the directory cannot be locked ({error})")),
        };
        let lock = match refusal {
            Ok(file) => file,
            Err(reason) => {
                shared.close();
                shared.warn(format!(
                    "observe: {}: {reason}; this run exports nothing",
                    config.directory.display()
                ));
                return Ok(Self {
                    shared,
                    writer: None,
                });
            }
        };
        let thread = {
            let shared = shared.clone();
            let directory = config.directory.clone();
            let max_text_chars = config.max_text_chars;
            std::thread::Builder::new()
                .name("retrograd-observe".into())
                .spawn(move || {
                    let _lock = lock;
                    writer::run(&directory, max_text_chars, &shared, run, receiver);
                })
                .map_err(ObserveError::Thread)?
        };
        Ok(Self {
            shared,
            writer: Some(thread),
        })
    }

    #[cfg(test)]
    pub(crate) fn shared(&self) -> Arc<Shared> {
        self.shared.clone()
    }

    /// A handle for the training loop.
    pub fn observer(&self) -> Arc<dyn TrajectoryObserver> {
        self.shared.clone()
    }

    /// Queues the `update` record of a completed update.
    pub fn update_summary<'a>(
        &self,
        update: u32,
        values: impl IntoIterator<Item = (&'a str, f32)>,
    ) {
        self.shared.observe(ObserveBatch::Update(UpdateSummary::new(
            update,
            UpdateStatus::Completed,
            values,
        )));
    }

    /// Warnings accumulated since the last call.
    pub fn take_warnings(&self) -> Vec<String> {
        std::mem::take(&mut *locked(&self.shared.warnings))
    }

    /// Batches dropped since the sink opened.
    pub fn dropped_batches(&self) -> u64 {
        self.shared.dropped()
    }

    /// Stops accepting batches and waits a bounded time for the queue to be
    /// written. A writer still busy after that is detached with a warning.
    pub fn finish(&mut self) {
        self.shared.close();
        let Some(thread) = self.writer.take() else {
            return;
        };
        let deadline = Instant::now() + FINISH_TIMEOUT;
        while !thread.is_finished() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        if thread.is_finished() {
            let _ = thread.join();
        } else {
            self.shared.warn(format!(
                "observe: the export writer did not finish within {}s; the batches still \
                 queued may not be written",
                FINISH_TIMEOUT.as_secs()
            ));
        }
    }
}

impl TrajectoryObserver for ObserveSink {
    fn wants(&self, update: u32) -> bool {
        self.shared.wants(update)
    }

    fn observe(&self, batch: ObserveBatch) {
        self.shared.observe(batch);
    }
}

impl Drop for ObserveSink {
    /// Never waits: [`ObserveSink::finish`] is the path that drains.
    fn drop(&mut self) {
        self.shared.close();
    }
}

enum Lock {
    Held(File),
    Busy,
    Unsupported(std::io::Error),
}

fn lock_directory(directory: &Path) -> std::io::Result<Lock> {
    let file = File::options()
        .create(true)
        .truncate(false)
        .write(true)
        .open(directory.join(LOCK_FILE))?;
    Ok(match file.try_lock() {
        Ok(()) => Lock::Held(file),
        Err(TryLockError::WouldBlock) => Lock::Busy,
        Err(TryLockError::Error(error)) => Lock::Unsupported(error),
    })
}
