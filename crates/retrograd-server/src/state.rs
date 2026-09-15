//! Server configuration, budgets, the operator-declared catalogue, and the
//! shared state every handler sees.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use retrograd_config::RunConfig;
use retrograd_core::{
    Device, ExecutionProfile, LoraConfig, MemoryReport, ModelCapabilities, ModelInfo,
    PreflightReport, Result as CoreResult, TargetSet,
};
use retrograd_engine::Trainer;
use retrograd_plan::CalibrationStore;
use serde::Deserialize;
use tokio::sync::Semaphore;

use crate::catalog::{Catalog, CatalogDeclaration};
use crate::error::{ApiError, ApiResult, ErrorCode, ProblemKind};
use crate::lock::Recover as _;
use crate::runtime::{RunEngine, RunRegistry, TrainingEngine};

/// Budgets, margins and the memory baseline now live in `retrograd-plan`, so the
/// cost model and the CLI can resolve a budget without an HTTP server. Re-exported
/// here because the server's configuration file is written in these types.
pub use retrograd_plan::budget::{BudgetRequest, MarginPolicy, MemoryBaseline};

/// The operator's configuration file, read once at startup.
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ServerConfig {
    /// Loopback by default: exposing a training API to a network is a decision,
    /// not a default.
    pub bind: Option<String>,
    pub state_dir: Option<PathBuf>,
    pub vram_budget: BudgetRequest,
    pub ram_budget: BudgetRequest,
    pub margin: MarginPolicy,
    /// A single run per process unless the operator has several devices: two
    /// trainers mean two sets of weights and KV caches on one device.
    pub max_concurrent_runs: Option<usize>,
    /// Measure a candidate configuration before creating a run.
    /// On by default: the model is about to be loaded anyway, so the measurement
    /// costs the load the run would pay, and it is what keeps the cost model
    /// honest on this machine. Turning it off trades that for a faster `201`.
    pub calibrate_runs: Option<bool>,
    /// Directory roots every client-supplied path must resolve inside. Empty
    /// means unrestricted, which turns the API into an arbitrary file reader; it
    /// is therefore allowed only on loopback, where the caller can already read
    /// the filesystem (see [`ServerConfig::validate`]).
    pub path_roots: Vec<PathBuf>,
    /// Whether `recipe.data.path` / `recipe.eval.path` - a literal filesystem
    /// path, as opposed to an uploaded dataset's id - is accepted at all.
    /// Default: on for a loopback bind, off otherwise:
    /// the same asymmetry `path_roots` and `auth_token` already draw. A caller
    /// on loopback can already read the filesystem, so the field costs
    /// nothing there; the moment the socket leaves the machine it is the
    /// difference between a training API and a way to name any file on disk.
    pub allow_local_paths: Option<bool>,
    /// Shared bearer token. Absent disables authentication, which is only
    /// accepted on a loopback bind.
    ///
    /// Read from `RETROGRAD_SERVER_TOKEN` when the file does not set it, so a
    /// deployment does not have to write its secret into a config file.
    pub auth_token: Option<String>,
    /// How long `evaluate` and `generate` may wait for the run's next progress
    /// callback. They are the only routes that wait at all.
    pub command_timeout_seconds: Option<u64>,
    /// How long any other request may take. The event streams are exempt: an SSE
    /// connection is meant to stay open, and a timeout layer over one would cut it
    /// every minute.
    pub request_timeout_seconds: Option<u64>,
    /// Largest request body accepted. A recipe or a configuration is kilobytes.
    pub max_body_bytes: Option<usize>,
    /// Requests served at once. Bounds what an unauthenticated caller on loopback
    /// can make the process do concurrently; runs have their own, much smaller,
    /// limit (`max_concurrent_runs`).
    pub max_concurrent_requests: Option<usize>,
    /// Trim absolute paths out of error responses, keeping them in the logs.
    /// Defaults to on for any bind that is not loopback: what leaves the
    /// machine is redacted, and a shared bearer token is not an identity.
    pub redact_error_paths: Option<bool>,
    /// Largest single dataset `POST /v1/datasets` accepts, in bytes. Default
    /// 1 GiB.
    pub max_dataset_bytes: Option<u64>,
    /// Total bytes every stored dataset may occupy together. Default 20 GiB.
    pub max_datasets_bytes: Option<u64>,
    /// Maximum pause between two chunks of a dataset upload. This is distinct
    /// from the total request timeout: a large upload may take hours while a
    /// stalled connection must eventually release its concurrency slot.
    pub upload_idle_timeout_seconds: Option<u64>,
    #[serde(rename = "reward")]
    pub rewards: Vec<CatalogDeclaration>,
    #[serde(rename = "judge")]
    pub judges: Vec<CatalogDeclaration>,
    #[serde(rename = "mcp_server")]
    pub mcp_servers: Vec<CatalogDeclaration>,
    /// `[[environment]]`: what a trajectory acts on. The operator declares the
    /// image, the limits and the pool; a client only ever names an id.
    #[serde(rename = "environment")]
    pub environments: Vec<CatalogDeclaration>,
}

impl ServerConfig {
    pub fn load(path: impl AsRef<Path>) -> CoreResult<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|error| {
            retrograd_core::Error::invalid(format!(
                "could not read server config {}: {error}",
                path.display()
            ))
        })?;
        let config: Self = toml::from_str(&text).map_err(|error| {
            retrograd_core::Error::invalid(format!(
                "invalid server config {}: {error}",
                path.display()
            ))
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Reads `RETROGRAD_SERVER_TOKEN` when the file left `auth_token` unset.
    ///
    /// Separate from [`Self::load`] so a test can build a configuration without
    /// the ambient environment deciding whether it is authenticated.
    pub fn with_token_from_env(mut self) -> Self {
        if self.auth_token.is_none() {
            self.auth_token = std::env::var("RETROGRAD_SERVER_TOKEN")
                .ok()
                .map(|token| token.trim().to_string())
                .filter(|token| !token.is_empty());
        }
        self
    }

    pub fn validate(&self) -> CoreResult<()> {
        self.margin.validate()?;
        if self.max_concurrent_runs == Some(0) {
            return Err(retrograd_core::Error::invalid(
                "max_concurrent_runs must be greater than zero",
            ));
        }
        for (name, value) in [
            ("max_concurrent_requests", self.max_concurrent_requests),
            ("max_body_bytes", self.max_body_bytes),
        ] {
            if value == Some(0) {
                return Err(retrograd_core::Error::invalid(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        for (name, value) in [
            ("command_timeout_seconds", self.command_timeout_seconds),
            ("request_timeout_seconds", self.request_timeout_seconds),
            (
                "upload_idle_timeout_seconds",
                self.upload_idle_timeout_seconds,
            ),
        ] {
            if value == Some(0) {
                return Err(retrograd_core::Error::invalid(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        // The two refusals that make the defaults safe rather than merely
        // conservative. On loopback, a caller can already read the filesystem and
        // run processes, so neither a token nor a path root adds anything; the
        // moment the socket leaves the machine, both are the difference between a
        // training API and a remote file-and-process primitive.
        if !self.binds_loopback() {
            if self.auth_token.is_none() {
                return Err(retrograd_core::Error::invalid(format!(
                    "refusing to serve {} without auth_token: a training API reachable off \
                     this machine can read files and start processes",
                    self.bind_address()
                )));
            }
            if self.path_roots.is_empty() {
                return Err(retrograd_core::Error::invalid(format!(
                    "refusing to serve {} without path_roots: every model, dataset and output \
                     path comes from the client, so an empty root list is an arbitrary file \
                     reader",
                    self.bind_address()
                )));
            }
        }
        Ok(())
    }

    /// Whether the bind address is reachable only from this machine.
    ///
    /// Parsed rather than string-matched, so `[::1]:8471` and `127.0.0.2:0` are
    /// recognised as loopback and `0.0.0.0` is not. A host name that does not
    /// parse as an address is treated as **not** loopback: the safe reading of
    /// something we cannot confirm.
    pub fn binds_loopback(&self) -> bool {
        let address = self.bind_address();
        match address.parse::<std::net::SocketAddr>() {
            Ok(socket) => socket.ip().is_loopback(),
            Err(_) => address
                .rsplit_once(':')
                .map(|(host, _)| host.trim_matches(['[', ']']))
                .and_then(|host| host.parse::<std::net::IpAddr>().ok())
                .is_some_and(|ip| ip.is_loopback()),
        }
    }

    pub fn bind_address(&self) -> &str {
        self.bind.as_deref().unwrap_or("127.0.0.1:8471")
    }

    pub fn state_dir(&self) -> PathBuf {
        self.state_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("runs"))
    }

    pub fn concurrency(&self) -> usize {
        self.max_concurrent_runs.unwrap_or(1)
    }

    pub fn calibrate_runs(&self) -> bool {
        self.calibrate_runs.unwrap_or(true)
    }

    /// Default five minutes: long enough for a rollout update on a small model,
    /// short enough that a stuck run does not hold a client connection until it
    /// gives up on its own.
    pub fn command_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.command_timeout_seconds.unwrap_or(300))
    }

    /// Applies to everything except the event streams. Sixty seconds is generous
    /// for a plan (a geometry read and a cost model) and for a preflight behind
    /// the device queue it is not - which is the point: a preflight that waits for
    /// a training run to finish should fail, not hang.
    pub fn request_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.request_timeout_seconds.unwrap_or(60))
    }

    /// One minute without a body chunk is a stalled upload, independently of
    /// how long the complete transfer is allowed to take.
    pub fn upload_idle_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.upload_idle_timeout_seconds.unwrap_or(60))
    }

    pub fn max_body_bytes(&self) -> usize {
        self.max_body_bytes.unwrap_or(1024 * 1024)
    }

    pub fn max_concurrent_requests(&self) -> usize {
        self.max_concurrent_requests.unwrap_or(256)
    }

    pub fn redact_error_paths(&self) -> bool {
        self.redact_error_paths
            .unwrap_or_else(|| !self.binds_loopback())
    }

    pub fn allow_local_paths(&self) -> bool {
        self.allow_local_paths
            .unwrap_or_else(|| self.binds_loopback())
    }

    pub fn max_dataset_bytes(&self) -> u64 {
        self.max_dataset_bytes.unwrap_or(1024 * 1024 * 1024)
    }

    pub fn max_datasets_bytes(&self) -> u64 {
        self.max_datasets_bytes.unwrap_or(20 * 1024 * 1024 * 1024)
    }
}

/// Measures the machine, once, before this process has allocated anything.
///
/// The type itself is plain data in `retrograd-plan`; reading a device is what
/// needs the FFI, so it lives here. Taken at startup on purpose:
/// `DeviceMemory::used` is a device-wide reading, so it contains the compositor
/// and every other process. Measuring it later would fold our own allocations
/// into the baseline and double-count them.
pub fn measure_baseline() -> MemoryBaseline {
    let host_used = retrograd_memory::snapshot()
        .map(|snapshot| snapshot.footprint)
        .unwrap_or(0);
    baseline_from(
        retrograd_memory::device_snapshot(),
        retrograd_memory::system_memory_bytes(),
        host_used,
    )
}

/// The rule the three readings are folded into, without the readings - so the
/// CPU-only case is testable on a machine that has a GPU, and the reverse.
fn baseline_from(
    device: Option<retrograd_memory::DeviceMemory>,
    host_total: Option<u64>,
    host_used: u64,
) -> MemoryBaseline {
    // A device whose reported total is the system's RAM is not a separate pool.
    // This is the observable signal: `ggml_backend_dev_memory` on a unified-memory
    // backend reports the machine's memory, and treating it as independent VRAM
    // would let the resolver promise the model twice.
    //
    // No device at all (a CPU-only build, or a machine with no GPU) is the same
    // situation taken to its limit: the run's "device" *is* the CPU, so its
    // footprint is drawn from RAM. Reporting no device total would leave the VRAM
    // budget at zero - and the planner charges the weights, the KV cache and the compute
    // buffers to the device side, so *every* configuration would be refused,
    // however small the model.
    let unified = match (device.map(|device| device.total), host_total) {
        (Some(device_total), Some(host_total)) => {
            device_total.abs_diff(host_total) < host_total / 8
        }
        (None, Some(_)) => true,
        _ => false,
    };
    MemoryBaseline {
        // Host figures on the CPU-only path, so the single unified budget is the
        // machine's, measured the same way on both sides.
        device_total: device.map(|device| device.total).or(host_total),
        device_used: device
            .map(retrograd_memory::DeviceMemory::used)
            .unwrap_or(host_used),
        host_total,
        host_used,
        unified,
    }
}

/// Loads a model just far enough to answer a question about it, on the caller's
/// thread.
///
/// A trait rather than a direct call because a `Trainer` needs a real GGUF and
/// several seconds, and the HTTP surface has to be testable without either. The
/// fake lives in the tests.
pub trait ModelProbe: Send + Sync {
    /// Builds the training graph for a throwaway adapter and returns the
    /// runtime's versioned, machine-readable preflight report.
    fn preflight(
        &self,
        model: &Path,
        device: Device,
        targets: TargetSet,
        profile_fingerprint: String,
    ) -> CoreResult<PreflightReport>;

    /// Exact candidate preflight used by planning. Test probes can reuse their
    /// request-oriented implementation; production overrides this to preserve
    /// every graph-changing training option.
    fn preflight_config(
        &self,
        model: &Path,
        config: &RunConfig,
        profile_fingerprint: String,
    ) -> CoreResult<PreflightReport> {
        self.preflight(
            model,
            config.training.device,
            config.lora.config.targets.clone(),
            profile_fingerprint,
        )
    }

    /// Reads the model's geometry without building a context.
    fn geometry(&self, model: &Path, device: Device) -> CoreResult<ModelInfo>;

    /// Structured model/device capabilities. Synthetic probes may keep the
    /// conservative default; production never parses the human report.
    fn model_capabilities(&self, _model: &Path, _device: Device) -> CoreResult<ModelCapabilities> {
        Ok(ModelCapabilities::default())
    }

    /// Builds the candidate configuration for real and reports what it cost,
    /// phase 4 of the resolver.
    ///
    /// This is the expensive one: it loads the model, creates the adapter and
    /// reserves the optimizer graph, i.e. everything a run does before its first
    /// step. That is the point - the numbers analysis cannot predict are exactly
    /// the ones only an allocation produces.
    fn measure(&self, model: &Path, config: &RunConfig) -> CoreResult<MemoryReport>;

    /// Real per-example token lengths of `path`, using `model`'s own
    /// tokenizer and chat template rather than the character heuristic.
    /// Host-only: never touches a device.
    fn tokenize_lengths(
        &self,
        model: &Path,
        path: &Path,
        format: retrograd_dataset::DataFormat,
    ) -> CoreResult<Vec<u32>>;
}

/// The real probe: what the `retrograd preflight` CLI command does.
pub struct EngineProbe;

impl ModelProbe for EngineProbe {
    fn preflight(
        &self,
        model: &Path,
        device: Device,
        targets: TargetSet,
        profile_fingerprint: String,
    ) -> CoreResult<PreflightReport> {
        let mut trainer = Trainer::new(
            model,
            retrograd_core::TrainConfig {
                device,
                ..Default::default()
            },
        )?;
        // The backward graph is defined by the trainable parameters, so the
        // preflight needs a (throwaway) LoRA adapter.
        let mut lora = LoraConfig::auto(2, 4.0);
        lora.targets = targets;
        trainer.create_lora(&lora)?;
        trainer.structured_preflight(profile_fingerprint)
    }

    fn geometry(&self, model: &Path, device: Device) -> CoreResult<ModelInfo> {
        retrograd_engine::model_info(model, device)
    }

    fn model_capabilities(&self, model: &Path, device: Device) -> CoreResult<ModelCapabilities> {
        let mut trainer = Trainer::new(
            model,
            retrograd_core::TrainConfig {
                device,
                ..Default::default()
            },
        )?;
        trainer.model_capabilities()
    }

    fn preflight_config(
        &self,
        model: &Path,
        config: &RunConfig,
        profile_fingerprint: String,
    ) -> CoreResult<PreflightReport> {
        let mut trainer = Trainer::new(model, config.training.clone())?;
        trainer.create_lora(&config.lora.config)?;
        trainer.structured_preflight(profile_fingerprint)
    }

    fn measure(&self, model: &Path, config: &RunConfig) -> CoreResult<MemoryReport> {
        let mut trainer = Trainer::new(model, config.training.clone())?;
        trainer.create_lora(&config.lora.config)?;
        trainer.prepare_optimizer()?;
        // The compute buffers only exist once the backward graph has been
        // reserved, and the preflight is what reserves it. Reading the report
        // before this call would measure a run that has not allocated the
        // activations yet, and report them as free.
        trainer.train_preflight()?;
        trainer.memory_report()
    }

    fn tokenize_lengths(
        &self,
        model: &Path,
        path: &Path,
        format: retrograd_dataset::DataFormat,
    ) -> CoreResult<Vec<u32>> {
        // CPU: the tokenizer and the chat template do not depend on where the
        // weights end up, and loading straight to the device this call
        // happens to run on would reserve VRAM for no reason.
        let trainer = Trainer::new(
            model,
            retrograd_core::TrainConfig {
                device: Device::Cpu,
                ..Default::default()
            },
        )?;
        retrograd_dataset::measured_lengths(&trainer, path, format)
    }
}

/// How many model geometries may be read at once. Four because each one is an
/// mmap and a header parse: enough that a handful of clients planning at the same
/// time do not queue behind each other, few enough that a burst cannot pin an
/// unbounded number of file mappings.
const CONCURRENT_GEOMETRY_READS: usize = 4;

/// Everything a handler needs, cheap to clone.
#[derive(Clone)]
pub struct AppState {
    pub config: Arc<ServerConfig>,
    pub catalog: Arc<Catalog>,
    pub baseline: MemoryBaseline,
    pub probe: Arc<dyn ModelProbe>,
    /// One permit per concurrent device user. Resolutions and runs share it: two
    /// concurrent resolutions each loading a 4 GiB model would cause exactly the
    /// overflow the resolver exists to avoid.
    pub device: Arc<Semaphore>,
    /// Permits for reading a model's *geometry*, which is a host-only mmap and
    /// a header parse - no device allocation at all.
    ///
    /// Deliberately not the device semaphore: calibration really does build the
    /// training graph and must be serialized with runs, but a geometry read
    /// does not. Sharing one semaphore would mean no plan could be made while a
    /// run trains - hours of a `/v1/plan` blocking on a job it has nothing to
    /// do with. A separate, small pool bounds a burst of plans without
    /// coupling them to a run's lifetime.
    pub probes: Arc<Semaphore>,
    pub backends: Arc<Vec<String>>,
    /// Every run this process knows about, including the ones it read back from
    /// the state directory at startup.
    pub registry: Arc<RunRegistry>,
    /// What executes a configuration. Swapped for a fake in the fast lane, which
    /// is what lets the whole runtime be tested without a model.
    pub engine: Arc<dyn RunEngine>,
    /// The per-machine correction factors, read at startup and written
    /// back after every measurement. A `std` lock, not tokio's: it is never held
    /// across an await.
    pub calibration: Arc<RwLock<CalibrationStore>>,
    /// Immutable snapshots keyed by model file identity and requested device.
    /// A queued run must not wait for the active run merely to rediscover an
    /// identical capability set.
    pub(crate) execution_profiles: Arc<RwLock<BTreeMap<String, ExecutionProfile>>>,
    /// Exact graph preflights keyed by the V2 model/config/profile identity.
    /// Entries are invalidated naturally by engine, catalogue, model or shape
    /// changes because all of those facts are part of the key.
    pub(crate) preflights: Arc<RwLock<BTreeMap<String, PreflightReport>>>,
    /// Full model-content fingerprints, cached only while size and mtime agree.
    /// The tokenizer cache and resume contract need content identity; hashing a
    /// multi-gigabyte GGUF on every plan would make that correctness unusable.
    model_fingerprints: Arc<RwLock<FingerprintCache>>,
    /// Every dataset this server has stored, rebuilt from `state_dir/datasets/`
    /// at startup.
    pub datasets: Arc<crate::datasets::DatasetStore>,
}

/// What a cached fingerprint stays valid for: the file's size and modification
/// time. A model rewritten in place changes at least one of the two, and the
/// entry is recomputed rather than trusted.
type FileIdentity = (u64, Option<std::time::SystemTime>);

type FingerprintCache = BTreeMap<PathBuf, (FileIdentity, String)>;

impl AppState {
    pub fn new(config: ServerConfig, catalog: Catalog, probe: Arc<dyn ModelProbe>) -> Self {
        let concurrency = config.concurrency();
        let state_dir = config.state_dir();
        let calibration = CalibrationStore::load(state_dir.join("calibration.json"));
        let datasets = crate::datasets::DatasetStore::open(&state_dir);
        Self {
            catalog: Arc::new(catalog),
            baseline: measure_baseline(),
            probe,
            device: Arc::new(Semaphore::new(concurrency)),
            probes: Arc::new(Semaphore::new(CONCURRENT_GEOMETRY_READS)),
            backends: Arc::new(compiled_backends()),
            registry: Arc::new(RunRegistry::open(state_dir)),
            engine: Arc::new(TrainingEngine),
            calibration: Arc::new(RwLock::new(calibration)),
            execution_profiles: Arc::new(RwLock::new(BTreeMap::new())),
            preflights: Arc::new(RwLock::new(BTreeMap::new())),
            model_fingerprints: Arc::new(RwLock::new(BTreeMap::new())),
            datasets: Arc::new(datasets),
            config: Arc::new(config),
        }
    }

    /// Replaces the engine. The one seam the tests need, and the only way to
    /// drive the runtime without a GGUF.
    pub fn with_engine(mut self, engine: Arc<dyn RunEngine>) -> Self {
        self.engine = engine;
        self
    }

    /// Where the correction table lives: one file for the whole state
    /// directory, not one per run - it describes the machine, not the job.
    pub fn calibration_path(&self) -> PathBuf {
        self.config.state_dir().join("calibration.json")
    }

    /// The server-wide budgets, before any per-run limit narrows them.
    pub fn budgets(&self) -> retrograd_plan::Budgets {
        retrograd_plan::Budgets::resolve(
            self.baseline,
            self.server_budgets(),
            (None, None),
            self.config.margin,
        )
    }

    pub fn server_budgets(&self) -> (BudgetRequest, BudgetRequest) {
        (self.config.vram_budget, self.config.ram_budget)
    }

    /// Stable identity of the complete GGUF bytes, computed off the async
    /// executor and reused while the filesystem identity remains unchanged.
    pub async fn model_fingerprint(&self, path: &Path) -> ApiResult<String> {
        let metadata = std::fs::metadata(path)
            .map_err(retrograd_core::Error::from)
            .map_err(ApiError::from)?;
        let identity = (metadata.len(), metadata.modified().ok());
        if let Some((_, fingerprint)) = self
            .model_fingerprints
            .read()
            .recover()
            .get(path)
            .filter(|(cached, _)| *cached == identity)
        {
            return Ok(fingerprint.clone());
        }
        let owned = path.to_path_buf();
        let fingerprint =
            tokio::task::spawn_blocking(move || retrograd_checkpoint::fingerprint_file(&owned))
                .await
                .map_err(|error| {
                    ApiError::internal(format!("model fingerprint task failed: {error}"))
                })?
                .map_err(ApiError::from)?;
        let after = std::fs::metadata(path)
            .map_err(retrograd_core::Error::from)
            .map_err(ApiError::from)?;
        if (after.len(), after.modified().ok()) != identity {
            return Err(ApiError::new(
                ProblemKind::Conflict,
                "model file changed while its content identity was being computed",
            ));
        }
        self.model_fingerprints
            .write()
            .recover()
            .insert(path.to_path_buf(), (identity, fingerprint.clone()));
        Ok(fingerprint)
    }

    /// What the resolver needs to know about this machine beyond its memory
    /// totals.
    ///
    /// The backend is read from the device list this build actually registered,
    /// not from a compile-time feature: `chunked_cross_entropy` is skipped by
    /// default on Metal because two of its nodes fall back to the CPU there, and
    /// a decision like that has to follow the hardware that will run the job. A
    /// run pinned to the CPU reports `cpu` whatever accelerators are present.
    pub fn hardware(&self, device: Device) -> retrograd_plan::HardwareFacts {
        let backend = if device == Device::Cpu {
            retrograd_plan::Backend::Cpu
        } else {
            // The first non-CPU family in the list, which is the one the runtime
            // itself picks (`retro_device_memory` reads the first GPU device).
            self.backends
                .iter()
                .filter(|name| name.as_str() != "cpu")
                .map(|name| retrograd_plan::Backend::from_family(name))
                .find(|backend| *backend != retrograd_plan::Backend::Unknown)
                // Not `Cpu`: the run is not pinned to the CPU, so claiming that
                // backend would hand every rule that keys on one a fact that is
                // wrong. `Unknown` is what an unrecognised accelerator is, and
                // it is the branch the rules treat conservatively.
                .unwrap_or_default()
        };
        retrograd_plan::HardwareFacts {
            backend,
            unified_memory: self.baseline.unified,
            device_total_bytes: self.baseline.device_total,
        }
    }

    /// Validates a client-supplied path against the configured roots.
    ///
    /// Canonicalizes first, so `..` segments and symlinks are resolved before the
    /// containment check - comparing the strings as given would let
    /// `roots/../etc/passwd` through.
    pub fn resolve_path(&self, raw: &str, pointer: &str) -> ApiResult<PathBuf> {
        let candidate = PathBuf::from(raw);
        let canonical = candidate.canonicalize().map_err(|error| {
            ApiError::invalid(format!("path '{raw}' cannot be resolved: {error}"))
                .with_field(
                    pointer,
                    ErrorCode::PathNotFound,
                    "no such file or directory",
                )
                .with_hint(
                    "if this path is not reachable from the server's filesystem, \
                     upload it with POST /v1/datasets instead",
                )
        })?;
        if self.config.path_roots.is_empty() {
            return Ok(canonical);
        }
        let allowed = self.config.path_roots.iter().any(|root| {
            root.canonicalize()
                .map(|root| canonical.starts_with(root))
                .unwrap_or(false)
        });
        if allowed {
            Ok(canonical)
        } else {
            // The message names neither the roots nor the resolved path: both
            // would describe the server's filesystem to a caller who is being
            // told they may not read it.
            Err(ApiError::new(
                ProblemKind::ForbiddenPath,
                "path is outside every allowed root",
            )
            .with_field(
                pointer,
                ErrorCode::ForbiddenPath,
                "not under an allowed root",
            )
            .with_hint("upload the dataset instead"))
        }
    }

    /// Resolves a path that may not exist yet by canonicalizing its nearest
    /// existing ancestor. This applies the same component-aware root policy as
    /// [`Self::resolve_path`] without requiring an output file to pre-exist.
    pub fn resolve_output_path(&self, path: &Path, pointer: &str) -> ApiResult<PathBuf> {
        let absolute = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .map_err(|error| {
                    ApiError::internal(format!("working directory is unavailable: {error}"))
                })?
                .join(path)
        };
        let mut ancestor = absolute.clone();
        let mut suffix = Vec::new();
        while !ancestor.exists() {
            let name = ancestor.file_name().ok_or_else(|| {
                ApiError::invalid("output path has no existing ancestor").with_field(
                    pointer,
                    ErrorCode::PathNotFound,
                    "no existing parent directory",
                )
            })?;
            suffix.push(name.to_os_string());
            if !ancestor.pop() {
                break;
            }
        }
        let mut canonical = ancestor.canonicalize().map_err(|error| {
            ApiError::invalid(format!("output parent cannot be resolved: {error}")).with_field(
                pointer,
                ErrorCode::PathNotFound,
                "parent directory cannot be resolved",
            )
        })?;
        for component in suffix.into_iter().rev() {
            canonical.push(component);
        }
        if self.config.path_roots.is_empty()
            || self.config.path_roots.iter().any(|root| {
                root.canonicalize()
                    .map(|root| canonical.starts_with(root))
                    .unwrap_or(false)
            })
        {
            Ok(canonical)
        } else {
            Err(ApiError::new(
                ProblemKind::ForbiddenPath,
                "output path is outside every allowed root",
            )
            .with_field(
                pointer,
                ErrorCode::ForbiddenPath,
                "not under an allowed root",
            ))
        }
    }

    /// Applies `path_roots` to every filesystem path a complete configuration
    /// can make the engine read or write. `managed_adapter` skips the recipe's
    /// placeholder: run creation replaces it with a server-owned run path.
    pub fn validate_run_paths(
        &self,
        config: &retrograd_config::RunConfig,
        managed_adapter: bool,
    ) -> ApiResult<()> {
        self.resolve_path(&config.model.to_string_lossy(), "/config/model/path")?;
        match &config.algorithm {
            retrograd_config::Algorithm::Sft(value) => {
                self.resolve_path(&value.data.to_string_lossy(), "/config/sft/data")?;
            }
            retrograd_config::Algorithm::Ppo(value) => {
                self.resolve_path(&value.prompts.to_string_lossy(), "/config/ppo/prompts")?;
            }
            retrograd_config::Algorithm::Grpo(value) => {
                self.resolve_path(&value.prompts.to_string_lossy(), "/config/grpo/prompts")?;
                if let Some(log) = &value.log_completions {
                    self.resolve_output_path(&log.path, "/config/grpo/log_completions/path")?;
                }
            }
            retrograd_config::Algorithm::Distill(value) => {
                // The two modes read different files, and the sandbox has to
                // reach exactly the ones the run will open - a prompt list on one
                // path, a corpus and its sidecar on the other.
                match value.mode.offline() {
                    Some(offline) => {
                        self.resolve_path(&offline.data.to_string_lossy(), "/config/distill/data")?;
                        self.resolve_path(
                            &offline.sidecar.to_string_lossy(),
                            "/config/distill/sidecar",
                        )?;
                    }
                    None => {
                        self.resolve_path(
                            &value.prompts.to_string_lossy(),
                            "/config/distill/prompts",
                        )?;
                    }
                }
                // The teacher is an input the server must be able to reach, and
                // it is the one path of this section that is not a dataset: a
                // run whose teacher sits outside the sandbox fails here rather
                // than after the student is already loaded.
                self.resolve_path(
                    &value.teacher_path.to_string_lossy(),
                    "/config/distill/teacher_path",
                )?;
            }
            retrograd_config::Algorithm::AgentGrpo(value) => {
                self.resolve_path(
                    &value.scenarios.to_string_lossy(),
                    "/config/agent/scenarios",
                )?;
                for (index, path) in value.tool_plan.mcp_config_files.iter().enumerate() {
                    self.resolve_path(
                        &path.to_string_lossy(),
                        &format!("/config/agent/mcp_config/{index}"),
                    )?;
                }
            }
        }
        if !managed_adapter {
            self.resolve_output_path(&config.lora.output, "/config/lora/output")?;
        }
        if let Some(path) = &config.lora.init_adapter {
            self.resolve_path(&path.to_string_lossy(), "/config/lora/init_adapter")?;
        }
        if let Some(evaluation) = &config.evaluation {
            self.resolve_path(
                &evaluation.data.to_string_lossy(),
                "/config/evaluation/data",
            )?;
        }
        if let Some(checkpoint) = &config.checkpoint {
            self.resolve_output_path(&checkpoint.directory, "/config/checkpoint/directory")?;
            if let Some(path) = &checkpoint.resume_from {
                self.resolve_path(&path.to_string_lossy(), "/config/checkpoint/resume_from")?;
            }
        }
        if let Some(path) = &config.metrics.tensorboard_dir {
            self.resolve_output_path(path, "/config/metrics/tensorboard_dir")?;
        }
        if let Some(path) = &config.metrics.wandb_export_dir {
            self.resolve_output_path(path, "/config/metrics/wandb_export_dir")?;
        }
        Ok(())
    }
}

/// Backends compiled into this build, derived from the device list rather than
/// from build-time cfgs, so it reports what is actually registered.
fn compiled_backends() -> Vec<String> {
    let Ok(list) = retrograd_engine::backend_list() else {
        return Vec::new();
    };
    // A `BTreeMap` keyed by lowercase name: stable order, and one entry per
    // backend even when a machine registers several devices of the same kind.
    let mut seen: BTreeMap<String, ()> = BTreeMap::new();
    for line in list.lines() {
        let mut fields = line.split('\t');
        let Some(kind) = fields.next() else { continue };
        let name = fields.next().unwrap_or("");
        let label = if kind == "cpu" {
            "cpu".to_string()
        } else {
            backend_family(name)
        };
        if !label.is_empty() {
            seen.insert(label, ());
        }
    }
    seen.into_keys().collect()
}

/// Reduces a device name (`"MTL0"`, `"CUDA0"`, `"Vulkan0"`) to its backend
/// family, so `/v1/health` lists backends and not device instances.
fn backend_family(name: &str) -> String {
    let lower = name.to_ascii_lowercase();
    for (needle, family) in [
        ("mtl", "metal"),
        ("metal", "metal"),
        ("cuda", "cuda"),
        ("vulkan", "vulkan"),
        ("blas", "blas"),
    ] {
        if lower.contains(needle) {
            return family.to_string();
        }
    }
    lower
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A CPU-only build reports no device, and the planner still charges the weights,
    /// the KV cache and the compute buffers to the device side. So the baseline
    /// has to hand that side the machine's RAM, as one pool: a `None` total
    /// leaves the VRAM budget at zero, and the resolver then refuses every
    /// configuration - the failure `tests/e2e_cpu.rs` catches.
    #[test]
    fn a_machine_with_no_device_draws_the_device_side_from_ram() {
        const GIB: u64 = 1024 * 1024 * 1024;
        let baseline = baseline_from(None, Some(16 * GIB), 512 * 1024 * 1024);
        assert_eq!(baseline.device_total, Some(16 * GIB));
        assert_eq!(baseline.device_used, 512 * 1024 * 1024);
        assert!(baseline.unified, "one pool, because there is only one");

        // A real GPU whose total is the system's RAM is the same pool by the
        // 1/8 test; a discrete card is not, and keeps its own budget.
        let unified = baseline_from(
            Some(retrograd_memory::DeviceMemory {
                free: 8 * GIB,
                total: 16 * GIB,
            }),
            Some(16 * GIB),
            GIB,
        );
        assert!(unified.unified);
        assert_eq!(
            unified.device_used,
            8 * GIB,
            "device-wide, not this process"
        );
        let discrete = baseline_from(
            Some(retrograd_memory::DeviceMemory {
                free: 20 * GIB,
                total: 24 * GIB,
            }),
            Some(64 * GIB),
            GIB,
        );
        assert!(!discrete.unified);
        assert_eq!(discrete.device_total, Some(24 * GIB));

        // No reading at all on either side is not a licence to invent one.
        let blind = baseline_from(None, None, 0);
        assert_eq!(blind.device_total, None);
        assert!(!blind.unified);
    }

    #[test]
    fn budgets_deserialize_from_every_toml_spelling() {
        #[derive(Deserialize)]
        struct Holder {
            a: BudgetRequest,
            b: BudgetRequest,
            c: BudgetRequest,
            d: BudgetRequest,
        }
        let holder: Holder =
            toml::from_str("a = 'all'\nb = '6GiB'\nc = 0.8\nd = 4096\n").expect("parse");
        assert_eq!(holder.a, BudgetRequest::All);
        assert_eq!(holder.b, BudgetRequest::Bytes(6 * 1024 * 1024 * 1024));
        assert_eq!(holder.c, BudgetRequest::Fraction(0.8));
        assert_eq!(holder.d, BudgetRequest::Bytes(4096));

        assert!(toml::from_str::<Holder>("a = 1.5\nb = 1\nc = 1\nd = 1\n").is_err());
    }

    #[test]
    fn the_server_budgets_are_what_a_resolution_is_narrowed_from() {
        let config = ServerConfig {
            vram_budget: BudgetRequest::Bytes(6 * 1024 * 1024 * 1024),
            ..Default::default()
        };
        assert_eq!(
            config.vram_budget,
            BudgetRequest::Bytes(6 * 1024 * 1024 * 1024)
        );
        assert_eq!(config.ram_budget, BudgetRequest::All);
    }

    #[test]
    fn server_config_defaults_are_conservative() {
        let config = ServerConfig::default();
        assert_eq!(config.bind_address(), "127.0.0.1:8471");
        assert_eq!(config.concurrency(), 1);
        assert_eq!(config.state_dir(), PathBuf::from("runs"));
        assert_eq!(config.vram_budget, BudgetRequest::All);
        assert!(
            config.calibrate_runs(),
            "a created run measures what it is about to load (§5.6 phase 4)"
        );
        assert!(config.rewards.is_empty());
        config.validate().unwrap();

        let mut invalid = ServerConfig {
            max_concurrent_runs: Some(0),
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
        invalid.max_concurrent_runs = Some(2);
        invalid.margin.fraction = 1.0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn backend_families_collapse_device_instances() {
        assert_eq!(backend_family("MTL0"), "metal");
        assert_eq!(backend_family("CUDA1"), "cuda");
        assert_eq!(backend_family("Vulkan0"), "vulkan");
        assert_eq!(backend_family("weird"), "weird");
    }
}
