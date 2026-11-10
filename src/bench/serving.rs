//! Serving-throughput mode of `bench`: prefill and decode measured
//! separately over a grid of micro-batch widths and active sequence counts,
//! with the page cache cold and hot, per-pass disk reads, and engine buffer sizes.
//!
//! This is the measurement mode a placement or caching change is judged
//! against: it reports what the hardware already does, before any policy
//! moves an expert anywhere. The numbers it prints go to stdout as a table
//! and, with `--out`, to a markdown file a reference table can be copied
//! from; the file records the exact command, the machine, the commit and the
//! model digest beside the rows.

use std::fs::File;
use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use comfy_table::Cell;
use retrograd::{
    Device, Engine, EngineConfig, EngineKvDtype, Error, MemoryReport, Result, SamplingParams,
    StepOutput, StepRow, gpu_runtime_available, model_info,
};
use retrograd_cli_ui::{CliUi, header_cell, right_cell, rounded_table};

use crate::args::Args;

/// Which page-cache state a measured pass runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CacheMode {
    Cold,
    Hot,
    Both,
}

impl CacheMode {
    fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "cold" => Ok(Self::Cold),
            "hot" => Ok(Self::Hot),
            "both" => Ok(Self::Both),
            _ => Err(Error::invalid("bench --cache must be cold, hot, or both")),
        }
    }

    fn passes(self) -> &'static [CacheMode] {
        match self {
            Self::Cold => &[Self::Cold],
            Self::Hot => &[Self::Hot],
            Self::Both => &[Self::Cold, Self::Hot],
        }
    }
}

/// The raw serving flags as the shared argument loop collects them, before
/// defaults are applied.
#[derive(Default, Debug)]
pub(crate) struct ServingFlags {
    pub serving: bool,
    pub prompt_tokens: Option<usize>,
    pub decode_tokens: Option<usize>,
    pub ubatches: Option<Vec<u32>>,
    pub seqs: Option<Vec<u32>>,
    pub repeats: Option<usize>,
    pub cache: Option<String>,
    pub devices: Option<String>,
    pub out: Option<PathBuf>,
}

impl ServingFlags {
    pub(crate) fn first_option(&self) -> Option<&'static str> {
        [
            ("--prompt-tokens", self.prompt_tokens.is_some()),
            ("--decode-tokens", self.decode_tokens.is_some()),
            ("--ubatches", self.ubatches.is_some()),
            ("--seqs", self.seqs.is_some()),
            ("--repeats", self.repeats.is_some()),
            ("--cache", self.cache.is_some()),
            ("--devices", self.devices.is_some()),
            ("--out", self.out.is_some()),
        ]
        .into_iter()
        .find_map(|(flag, present)| present.then_some(flag))
    }
}

/// The serving-mode arguments, with defaults applied.
#[derive(Clone, Debug)]
pub(crate) struct ServingArgs {
    pub prompt_tokens: usize,
    pub decode_tokens: usize,
    /// Micro-batch widths to measure.
    pub ubatches: Vec<u32>,
    /// Active sequence counts to measure.
    pub seqs: Vec<u32>,
    pub repeats: usize,
    pub cache: CacheMode,
    /// Raw `--devices` value: `auto` or a comma list of `cpu`/`gpu`/`auto`.
    pub devices: String,
    pub out: Option<PathBuf>,
}

impl ServingArgs {
    /// Builds the serving arguments from the collected flags, applying
    /// defaults. `None` values mean the flag was absent.
    pub(crate) fn from_flags(flags: &ServingFlags, devices: Option<&str>) -> Result<Self> {
        let prompt_tokens = flags.prompt_tokens.unwrap_or(512);
        let decode_tokens = flags.decode_tokens.unwrap_or(64);
        let repeats = flags.repeats.unwrap_or(3);
        for (flag, value) in [
            ("bench --prompt-tokens", prompt_tokens),
            ("bench --decode-tokens", decode_tokens),
            ("bench --repeats", repeats),
        ] {
            if value == 0 {
                return Err(Error::invalid(format!("{flag} must be greater than zero")));
            }
        }
        let ubatches = flags
            .ubatches
            .clone()
            .unwrap_or_else(|| vec![1, 8, 32, 512]);
        let seqs = flags.seqs.clone().unwrap_or_else(|| vec![1, 4]);
        if ubatches.is_empty() || seqs.is_empty() {
            return Err(Error::invalid(
                "bench --ubatches and --seqs require at least one entry",
            ));
        }
        for &width in &ubatches {
            if width == 0 || width > 4096 {
                return Err(Error::invalid(
                    "bench --ubatches entries must be in 1..=4096",
                ));
            }
        }
        for &seq in &seqs {
            if seq == 0 || seq > 256 {
                return Err(Error::invalid("bench --seqs entries must be in 1..=256"));
            }
        }
        Ok(Self {
            prompt_tokens,
            decode_tokens,
            ubatches,
            seqs,
            repeats,
            cache: flags
                .cache
                .as_deref()
                .map(CacheMode::parse)
                .transpose()?
                .unwrap_or(CacheMode::Both),
            devices: devices
                .or(flags.devices.as_deref())
                .unwrap_or("auto")
                .to_string(),
            out: flags.out.clone(),
        })
    }

    /// Parses a comma-separated list of positive integers (`1,8,32,512`).
    pub(crate) fn parse_u32_list(flag: &str, value: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for part in value.split(',') {
            let parsed = crate::args::parse_value(flag, part.trim(), "an integer")?;
            out.push(parsed);
        }
        if out.is_empty() {
            return Err(Error::invalid(format!(
                "{flag} requires at least one value"
            )));
        }
        Ok(out)
    }

    /// Parses a positive integer flag value, refusing zero.
    pub(crate) fn parse_positive(flag: &str, value: &str) -> Result<usize> {
        let parsed: usize = crate::args::parse_value(flag, value, "an integer")?;
        if parsed == 0 {
            return Err(Error::invalid(format!("{flag} must be greater than zero")));
        }
        Ok(parsed)
    }

    /// The device list to measure: `auto` resolves to GPU plus CPU when a GPU
    /// backend is compiled in, CPU alone otherwise.
    fn resolve_devices(&self) -> Result<Vec<Device>> {
        let mut out = Vec::new();
        for part in self.devices.split(',') {
            match part.trim().to_ascii_lowercase().as_str() {
                "auto" => {
                    if gpu_runtime_available() {
                        out.push(Device::Gpu);
                    }
                    out.push(Device::Cpu);
                }
                "cpu" => out.push(Device::Cpu),
                "gpu" => out.push(Device::Gpu),
                other => {
                    return Err(Error::invalid(format!(
                        "unknown device '{other}' in --devices; use auto, cpu, or gpu"
                    )));
                }
            }
        }
        out.sort_by_key(|device| match device {
            Device::Gpu => 0,
            Device::Auto | Device::Cpu => 1,
        });
        out.dedup();
        if out.is_empty() {
            return Err(Error::invalid(
                "bench --devices requires at least one device",
            ));
        }
        Ok(out)
    }
}

/// One measured cell of the grid: a device, a micro-batch width, an active
/// sequence count, and a page-cache state, each at its median over the
/// repeats.
#[derive(Debug)]
struct ServingRow {
    device: &'static str,
    ubatch: u32,
    n_seq: u32,
    cache: &'static str,
    prefill_tok_s: f64,
    decode_tok_s: f64,
    prefill: Duration,
    decode: Duration,
    disk_read_bytes: Option<u64>,
    /// Whether every cold repeat recorded disk reads.
    cold_effective: bool,
    buffer_bytes: u64,
    kv_bytes: u64,
    repeats: Vec<ServingRepeat>,
}

#[derive(Debug)]
struct ServingRepeat {
    prefill: Duration,
    decode: Duration,
    disk_read_bytes: Option<u64>,
}

/// Runs the serving grid and prints the table. `model` is the already
/// override-resolved model path; `threads` comes from the run config.
pub(crate) fn bench_serving(
    model: &Path,
    threads: u32,
    ctx: Option<u32>,
    args: &ServingArgs,
    ui: &CliUi,
) -> Result<()> {
    let devices = args.resolve_devices()?;
    let n_seq_max = *args.seqs.iter().max().expect("seqs is non-empty");
    let needed = args
        .prompt_tokens
        .checked_add(args.decode_tokens)
        .ok_or_else(|| Error::overflow("serving context does not fit usize"))?;
    let needed =
        u32::try_from(needed).map_err(|_| Error::overflow("serving context does not fit u32"))?;
    let per_seq_ctx = match ctx {
        Some(configured) if configured < needed => {
            return Err(Error::invalid(
                "bench --ctx must cover --prompt-tokens plus --decode-tokens",
            ));
        }
        Some(configured) => configured,
        None => needed.saturating_add(15),
    };
    let n_ctx = per_seq_ctx
        .checked_mul(n_seq_max)
        .filter(|&n_ctx| i32::try_from(n_ctx).is_ok())
        .ok_or_else(|| Error::overflow("serving n_ctx does not fit i32"))?;

    if let Some(out) = &args.out
        && out.symlink_metadata().is_ok()
    {
        return Err(Error::invalid(format!(
            "bench --out already exists: {}",
            out.display()
        )));
    }
    let models = model_files(model)?;
    let info = model_info(model, Device::Cpu)?;
    if info.has_encoder {
        return Err(Error::invalid(
            "bench --serving requires a decoder-only model",
        ));
    }
    if info.n_expert > 0 {
        ui.info(format!(
            "model is a MoE: {} experts, {} used per token",
            info.n_expert, info.n_expert_used
        ));
    }

    let mut ubatches = Vec::new();
    for &requested in &args.ubatches {
        // The runtime clamps the batch before padding the KV pool.
        let ubatch = requested.min(n_ctx);
        if ubatch != requested {
            ui.info(format!(
                "ubatch {requested} clamped to context size {ubatch}"
            ));
        }
        if !ubatches.contains(&ubatch) {
            ubatches.push(ubatch);
        }
    }
    let mut rows: Vec<ServingRow> = Vec::new();
    for &device in &devices {
        let label = match device {
            Device::Gpu => "gpu",
            Device::Cpu => "cpu",
            Device::Auto => "auto",
        };
        for &ubatch in &ubatches {
            let n_batch = ubatch.max(n_seq_max);
            let config = EngineConfig {
                n_ctx,
                n_seq_max,
                n_batch,
                n_ubatch: ubatch,
                kv_dtype: EngineKvDtype::F32,
                flash_attn: device != Device::Cpu,
                kv_unified: true,
                device,
                threads,
                verbose: false,
            };
            let spinner = ui.spinner(format!(
                "loading model for serving bench (device={label}, ubatch={ubatch})"
            ));
            // Request eviction before load so its reads can be reported separately.
            if args.cache.passes().contains(&CacheMode::Cold) {
                for path in &models {
                    evict_page_cache(path)?;
                }
            }
            let load_disk_before = disk_read_bytes();
            let mut engine = Engine::new(model, &config).map_err(Error::from)?;
            let load_disk = disk_read_delta(load_disk_before);
            ui.finish_spinner(spinner, "model loaded");
            let load_mem = engine.memory_report().map_err(Error::from)?;
            ui.info(format!(
                "device={label} ubatch={ubatch}: weights {} MiB, buffers {} MiB (device {} + host {}), load read {} MiB from disk",
                load_mem.model_weight_bytes / (1 << 20),
                buffer_bytes(&load_mem) / (1 << 20),
                load_mem.device_bytes / (1 << 20),
                load_mem.host_bytes / (1 << 20),
                disk_mib(load_disk),
            ));
            let prompt = make_prompt(&engine, args.prompt_tokens)?;
            for &n_seq in &args.seqs {
                for &cache in args.cache.passes() {
                    let cache_label = match cache {
                        CacheMode::Cold => "cold",
                        CacheMode::Hot => "hot",
                        CacheMode::Both => unreachable!("passes() never yields Both"),
                    };
                    let mut repeats = Vec::new();
                    repeats.try_reserve_exact(args.repeats).map_err(|error| {
                        Error::invalid(format!("cannot allocate --repeats samples: {error}"))
                    })?;
                    let mut buffers = 0_u64;
                    let mut out = StepOutput::with_capacity(n_seq as usize, 0);
                    if cache == CacheMode::Hot {
                        clear_slots(&mut engine, n_seq_max)?;
                        run_pass(
                            &mut engine,
                            &prompt,
                            n_seq,
                            n_batch,
                            args.decode_tokens,
                            &mut out,
                        )?;
                    }
                    for _ in 0..args.repeats {
                        clear_slots(&mut engine, n_seq_max)?;
                        if cache == CacheMode::Cold {
                            for path in &models {
                                evict_page_cache(path)?;
                            }
                        }
                        repeats.push(run_pass(
                            &mut engine,
                            &prompt,
                            n_seq,
                            n_batch,
                            args.decode_tokens,
                            &mut out,
                        )?);
                        let memory = engine.memory_report().map_err(Error::from)?;
                        buffers = buffer_bytes(&memory).max(buffers);
                    }
                    let prefill_tokens = f64::from(n_seq) * args.prompt_tokens as f64;
                    let decode_tokens = f64::from(n_seq) * args.decode_tokens as f64;
                    let (prefill_tok_s, decode_tok_s) = (
                        median(repeats.iter().map(|r| rate(prefill_tokens, r.prefill))),
                        median(repeats.iter().map(|r| rate(decode_tokens, r.decode))),
                    );
                    let cold_effective = cache != CacheMode::Cold
                        || repeats
                            .iter()
                            .all(|r| r.disk_read_bytes.is_some_and(|bytes| bytes > 0));
                    rows.push(ServingRow {
                        device: label,
                        ubatch,
                        n_seq,
                        cache: cache_label,
                        prefill_tok_s,
                        decode_tok_s,
                        prefill: Duration::from_secs_f64(median(
                            repeats.iter().map(|r| r.prefill.as_secs_f64()),
                        )),
                        decode: Duration::from_secs_f64(median(
                            repeats.iter().map(|r| r.decode.as_secs_f64()),
                        )),
                        disk_read_bytes: repeats.iter().filter_map(|r| r.disk_read_bytes).max(),
                        cold_effective,
                        buffer_bytes: buffers,
                        kv_bytes: engine.memory().map_err(Error::from)?.kv_bytes,
                        repeats,
                    });
                }
            }
        }
    }

    print_table(&rows);
    if let Some(out) = &args.out {
        write_report(out, &models, args, &rows)?;
        ui.info(format!("serving report written to {}", out.display()));
    }
    Ok(())
}

/// Median of a non-empty iterator of values.
fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values: Vec<f64> = values.into_iter().collect();
    debug_assert!(!values.is_empty());
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

fn rate(tokens: f64, elapsed: Duration) -> f64 {
    tokens / elapsed.as_secs_f64().max(1e-9)
}

fn run_pass(
    engine: &mut Engine,
    prompt: &[i32],
    n_seq: u32,
    n_batch: u32,
    n_decode: usize,
    out: &mut StepOutput,
) -> Result<ServingRepeat> {
    let disk_before = disk_read_bytes();
    let (prefill, first_tokens) = run_prefill(engine, prompt, n_seq, n_batch, out)?;
    let decode = run_decode(engine, prompt.len(), n_decode, &first_tokens, out)?;
    Ok(ServingRepeat {
        prefill,
        decode,
        disk_read_bytes: disk_read_delta(disk_before),
    })
}

/// The sizes of the engine's ggml buffers, including file-backed mappings.
fn buffer_bytes(report: &MemoryReport) -> u64 {
    report.device_bytes.saturating_add(report.host_bytes)
}

/// Empties every slot's KV and releases its per-request state, so a pass
/// starts from an empty context.
fn clear_slots(engine: &mut Engine, n_seq: u32) -> Result<()> {
    for slot in 0..n_seq {
        engine.seq_rm(slot as i32, -1, -1).map_err(Error::from)?;
        engine.request_free(slot as i32).map_err(Error::from)?;
    }
    Ok(())
}

/// A deterministic prompt with the tokenizer's initial special tokens and a
/// repeated sentence fragment.
fn make_prompt(engine: &Engine, len: usize) -> Result<Vec<i32>> {
    let text = "the quick brown fox jumps over the lazy dog.";
    let initial = engine.tokenize_text(text).map_err(Error::from)?;
    let continuation = engine
        .tokenize_fragment(&format!(" {text}"))
        .map_err(Error::from)?;
    if initial.is_empty() || continuation.is_empty() {
        return Err(Error::invalid(
            "the tokenizer returned an empty prompt or continuation",
        ));
    }
    Ok(initial
        .into_iter()
        .chain(continuation.into_iter().cycle())
        .take(len)
        .collect())
}

/// Prefills `n_seq` copies of `prompt` in interleaved chunks, sampling the
/// first generated token on each slot's last chunk. Returns the wall time
/// and, per slot, that first sampled token.
fn run_prefill(
    engine: &mut Engine,
    prompt: &[i32],
    n_seq: u32,
    n_batch: u32,
    out: &mut StepOutput,
) -> Result<(Duration, Vec<i32>)> {
    // The runtime refuses temperature 0, so the pass samples at a fixed seed
    // instead of greedily; the throughput measured is the same either way.
    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 1,
        seed: 0x5EED,
    };
    // Context validation bounds all positions by i32::MAX, and slots by 256.
    let n = prompt.len();
    let mut pos = vec![0_usize; n_seq as usize];
    let mut first_tokens = vec![0_i32; n_seq as usize];
    let started = Instant::now();
    while pos.iter().any(|&p| p < n) {
        let active: Vec<usize> = (0..n_seq as usize).filter(|&i| pos[i] < n).collect();
        let per_row = (n_batch as usize / active.len()).max(1);
        let mut rows: Vec<StepRow> = Vec::with_capacity(active.len());
        for &slot in &active {
            let end = (pos[slot] + per_row).min(n);
            let mut row = StepRow::prefill(
                slot as i32,
                slot as i32,
                &prompt[pos[slot]..end],
                pos[slot] as i32,
            );
            if end == n {
                row.sample = true;
                row.sampling = sampling.into();
            }
            pos[slot] = end;
            rows.push(row);
        }
        engine.step(&rows, out).map_err(Error::from)?;
        for (index, row) in rows.iter().enumerate() {
            if row.sample {
                let slot = row.slot as usize;
                first_tokens[slot] = out.row(index).token;
            }
        }
    }
    Ok((started.elapsed(), first_tokens))
}

/// Decodes `n_decode` tokens per slot, one row per slot per step, feeding
/// back each sampled token. `first_tokens` is what the prefill sampled.
/// Returns the wall time.
fn run_decode(
    engine: &mut Engine,
    prompt_len: usize,
    n_decode: usize,
    first_tokens: &[i32],
    out: &mut StepOutput,
) -> Result<Duration> {
    let sampling = SamplingParams {
        temperature: 1.0,
        top_p: 1.0,
        max_new_tokens: 1,
        seed: 0x5EED,
    };
    let mut last: Vec<i32> = first_tokens.to_vec();
    let started = Instant::now();
    for pos in prompt_len..prompt_len + n_decode {
        // The validated context and sequence count bound these positions and slots.
        let rows: Vec<StepRow> = last
            .iter()
            .enumerate()
            .map(|(slot, &token)| {
                StepRow::decode(slot as i32, slot as i32, token, pos as i32, &sampling)
            })
            .collect();
        engine.step(&rows, out).map_err(Error::from)?;
        for (slot, token) in last.iter_mut().enumerate() {
            *token = out.row(slot).token;
        }
    }
    Ok(started.elapsed())
}

/// Process disk bytes read so far, when the platform reports them. Linux
/// reads `read_bytes` from `/proc/self/io`; macOS counts them in
/// `proc_pid_rusage`'s `ri_diskio_bytesread` (mmap page faults included, which
/// `getrusage`'s `ru_inblock` misses). `None` on anything else.
fn disk_read_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let io = std::fs::read_to_string("/proc/self/io").ok()?;
        io.lines()
            .find_map(|line| line.strip_prefix("read_bytes: "))
            .and_then(|value| value.trim().parse().ok())
    }
    #[cfg(target_os = "macos")]
    {
        // Flavor 4 (`RUSAGE_INFO_V4`): the newest `rusage_info` the libc crate
        // names, and it already carries `ri_diskio_bytesread`.
        const RUSAGE_INFO_V4: libc::c_int = 4;
        // SAFETY: `info` is a valid out-buffer of the struct the flavor names.
        // The C signature takes `rusage_info_t` (a `void *`) pointing straight
        // at that struct; the libc binding spells the parameter `*mut
        // rusage_info_t`, so the struct pointer is reinterpreted once, exactly
        // like the C cast `(rusage_info_t)&info`.
        let mut info = unsafe { std::mem::zeroed::<libc::rusage_info_v4>() };
        let code = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                RUSAGE_INFO_V4,
                (&mut info as *mut libc::rusage_info_v4 as *mut libc::c_void)
                    as *mut libc::rusage_info_t,
            )
        };
        if code != 0 {
            return None;
        }
        Some(info.ri_diskio_bytesread)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        None
    }
}

fn disk_read_delta(before: Option<u64>) -> Option<u64> {
    disk_read_bytes()?.checked_sub(before?)
}

fn disk_mib(bytes: Option<u64>) -> String {
    bytes.map_or_else(
        || "n/a".into(),
        |bytes| format!("{:.1}", bytes as f64 / (1 << 20) as f64),
    )
}

fn model_files(model: &Path) -> Result<Vec<PathBuf>> {
    let metadata = retrograd::trainer::read_gguf_metadata(model, "split.count")?;
    let Some(count) = metadata.get("split.count") else {
        return Ok(vec![model.to_path_buf()]);
    };
    let count: u16 = count
        .parse()
        .map_err(|_| Error::invalid("invalid GGUF split.count"))?;
    if count <= 1 {
        return Ok(vec![model.to_path_buf()]);
    }
    let suffix = format!("-00001-of-{count:05}.gguf");
    let prefix = model
        .file_name()
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_suffix(&suffix))
        .filter(|prefix| !prefix.is_empty())
        .ok_or_else(|| Error::invalid(format!("bench requires the first GGUF shard ({suffix})")))?;
    Ok((1..=count)
        .map(|index| model.with_file_name(format!("{prefix}-{index:05}-of-{count:05}.gguf")))
        .collect())
}

/// Drops a file's clean pages from the page cache, so the next pass that
/// faults them in reads from disk. Linux calls `posix_fadvise(DONTNEED)`
/// directly; macOS has no `posix_fadvise`, so it maps the file read-only and
/// invalidates the mapping with `msync(MS_INVALIDATE)`, which discards the
/// clean pages from the unified buffer cache. A kernel that refuses either
/// leaves the cache warm, which the cold pass then reports as not effective.
fn evict_page_cache(path: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;
        let file = File::open(path)
            .map_err(|error| Error::runtime(format!("cannot open {}: {error}", path.display())))?;
        // SAFETY: `fd` is a live open file; the call only advises the kernel.
        let code =
            unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
        if code != 0 {
            return Err(Error::runtime(format!(
                "posix_fadvise(DONTNEED) on {} failed: {}",
                path.display(),
                std::io::Error::from_raw_os_error(code)
            )));
        }
        Ok(())
    }
    #[cfg(target_os = "macos")]
    {
        use std::os::unix::io::AsRawFd;
        let file = File::open(path)
            .map_err(|error| Error::runtime(format!("cannot open {}: {error}", path.display())))?;
        let len = file
            .metadata()
            .map_err(|error| Error::runtime(format!("cannot stat {}: {error}", path.display())))?
            .len();
        let len =
            usize::try_from(len).map_err(|_| Error::overflow("GGUF length does not fit usize"))?;
        if len == 0 {
            return Ok(());
        }
        // SAFETY: `fd` is a live open file and `len` its exact length; the
        // mapping is read-only and released before returning.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                file.as_raw_fd(),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(Error::runtime(format!(
                "mmap of {} failed: {}",
                path.display(),
                std::io::Error::last_os_error()
            )));
        }
        // SAFETY: `map` covers `len` bytes of a live mapping.
        let code = unsafe { libc::msync(map, len, libc::MS_INVALIDATE) };
        let invalidation_error = (code != 0).then(std::io::Error::last_os_error);
        // SAFETY: the mapping was created above and is released exactly once.
        unsafe { libc::munmap(map, len) };
        if let Some(error) = invalidation_error {
            return Err(Error::runtime(format!(
                "msync(MS_INVALIDATE) on {} failed: {}",
                path.display(),
                error
            )));
        }
        Ok(())
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = path;
        Ok(())
    }
}

const COLD_NOTE: &str = "* cold row includes a repeat without confirmed disk reads";

fn print_table(rows: &[ServingRow]) {
    let color = std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none();
    let mut table = rounded_table();
    table.set_header(vec![
        header_cell(color, "device"),
        header_cell(color, "ubatch"),
        header_cell(color, "n_seq"),
        header_cell(color, "cache"),
        header_cell(color, "prefill tok/s"),
        header_cell(color, "decode tok/s"),
        header_cell(color, "prefill ms"),
        header_cell(color, "decode ms"),
        header_cell(color, "max disk read MiB"),
        header_cell(color, "buffer MiB"),
        header_cell(color, "kv MiB"),
    ]);
    for row in rows {
        let cache = if row.cold_effective {
            row.cache.to_string()
        } else {
            format!("{}*", row.cache)
        };
        table.add_row(vec![
            Cell::new(row.device),
            Cell::new(row.ubatch),
            Cell::new(row.n_seq),
            Cell::new(cache),
            right_cell(format!("{:.1}", row.prefill_tok_s)),
            right_cell(format!("{:.1}", row.decode_tok_s)),
            right_cell(format!("{:.1}", row.prefill.as_secs_f64() * 1e3)),
            right_cell(format!("{:.1}", row.decode.as_secs_f64() * 1e3)),
            right_cell(disk_mib(row.disk_read_bytes)),
            right_cell(format!("{:.1}", row.buffer_bytes as f64 / (1 << 20) as f64)),
            right_cell(format!("{:.1}", row.kv_bytes as f64 / (1 << 20) as f64)),
        ]);
    }
    println!("{table}");
    if rows.iter().any(|row| !row.cold_effective) {
        println!("{COLD_NOTE}");
    }
}

/// The markdown report written by `--out`: provenance header, the measured
/// table, and the raw per-repeat rates.
fn write_report(
    path: &Path,
    models: &[PathBuf],
    args: &ServingArgs,
    rows: &[ServingRow],
) -> Result<()> {
    let mut text = String::new();
    text.push_str("# serving bench\n\n");
    text.push_str(&format!("- command: `{}`\n", command_line()));
    text.push_str(&format!(
        "- date (UTC): {}\n",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| {
                let secs = d.as_secs();
                let days = secs / 86_400;
                let tod = secs % 86_400;
                // Civil-from-days (Howard Hinnant's algorithm), enough for a stamp.
                let z = days + 719_468;
                let era = z.div_euclid(146_097);
                let doe = z.rem_euclid(146_097);
                let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
                let y = yoe + era * 400;
                let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
                let mp = (5 * doy + 2) / 153;
                let day = doy - (153 * mp + 2) / 5 + 1;
                let month = if mp < 10 { mp + 3 } else { mp - 9 };
                let year = if month <= 2 { y + 1 } else { y };
                format!(
                    "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
                    tod / 3600,
                    (tod % 3600) / 60,
                    tod % 60
                )
            })
            .unwrap_or_else(|_| "unknown".to_string())
    ));
    text.push_str(&format!("- machine: {}\n", machine_string()));
    let commit = option_env!("RETROGRAD_SOURCE_COMMIT").unwrap_or("unknown");
    text.push_str(&format!("- commit: {commit}\n"));
    for model in models {
        let (size, sha) = file_digest(model)?;
        text.push_str(&format!(
            "- model file: `{}` ({} bytes, sha256 `{}`)\n",
            model.display(),
            size,
            sha
        ));
    }
    text.push_str(&format!(
        "- grid: prompt_tokens={} decode_tokens={} ubatches={:?} seqs={:?} repeats={} cache={:?}\n",
        args.prompt_tokens, args.decode_tokens, args.ubatches, args.seqs, args.repeats, args.cache
    ));
    text.push('\n');
    text.push_str("| device | ubatch | n_seq | cache | prefill tok/s | decode tok/s | prefill ms | decode ms | max disk read MiB | buffer MiB | kv MiB |\n");
    text.push_str("|---|---|---|---|---|---|---|---|---|---|---|\n");
    for row in rows {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {:.1} | {:.1} | {:.1} | {:.1} | {} | {:.1} | {:.1} |\n",
            row.device,
            row.ubatch,
            row.n_seq,
            if row.cold_effective {
                row.cache.to_string()
            } else {
                format!("{}*", row.cache)
            },
            row.prefill_tok_s,
            row.decode_tok_s,
            row.prefill.as_secs_f64() * 1e3,
            row.decode.as_secs_f64() * 1e3,
            disk_mib(row.disk_read_bytes),
            row.buffer_bytes as f64 / (1 << 20) as f64,
            row.kv_bytes as f64 / (1 << 20) as f64,
        ));
    }
    if rows.iter().any(|row| !row.cold_effective) {
        text.push_str(&format!("\n\\{COLD_NOTE}\n"));
    }
    text.push_str("\n## raw repeats (prefill tok/s, decode tok/s, disk read bytes)\n\n");
    for row in rows {
        let prefill_tokens = f64::from(row.n_seq) * args.prompt_tokens as f64;
        let decode_tokens = f64::from(row.n_seq) * args.decode_tokens as f64;
        text.push_str(&format!(
            "- device={} ubatch={} n_seq={} cache={}: {}\n",
            row.device,
            row.ubatch,
            row.n_seq,
            row.cache,
            row.repeats
                .iter()
                .map(|repeat| {
                    let prefill = rate(prefill_tokens, repeat.prefill);
                    let decode = rate(decode_tokens, repeat.decode);
                    let disk = repeat
                        .disk_read_bytes
                        .map_or_else(|| "n/a".into(), |bytes| bytes.to_string());
                    format!("({prefill:.1}, {decode:.1}, {disk})")
                })
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let write_error = |error| Error::runtime(format!("cannot write {}: {error}", path.display()));
    std::fs::create_dir_all(parent).map_err(write_error)?;
    let mut file = tempfile::NamedTempFile::new_in(parent).map_err(write_error)?;
    file.write_all(text.as_bytes()).map_err(write_error)?;
    file.persist_noclobber(path)
        .map_err(|error| write_error(error.error))?;
    Ok(())
}

fn command_line() -> String {
    format!("{:?}", std::env::args().collect::<Vec<_>>())
}

/// `uname`-style machine description, for the report header.
fn machine_string() -> String {
    #[cfg(unix)]
    {
        // SAFETY: `info` is a valid out-pointer the call fills.
        let mut info = unsafe { std::mem::zeroed::<libc::utsname>() };
        // SAFETY: a valid buffer; uname only fails on a bad pointer.
        if unsafe { libc::uname(&mut info) } == 0 {
            let field = |bytes: &[libc::c_char]| -> String {
                let end = bytes.iter().position(|&c| c == 0).unwrap_or(bytes.len());
                let raw: Vec<u8> = bytes[..end].iter().map(|&c| c as u8).collect();
                String::from_utf8_lossy(&raw).into_owned()
            };
            return format!(
                "{} {} {} {}",
                field(&info.sysname),
                field(&info.release),
                field(&info.machine),
                field(&info.version),
            );
        }
    }
    format!("{} {}", std::env::consts::OS, std::env::consts::ARCH)
}

/// File size and SHA-256, so a reference row names the exact bytes measured.
fn file_digest(path: &Path) -> Result<(u64, String)> {
    use sha2::{Digest, Sha256};
    let mut file = File::open(path)
        .map_err(|error| Error::runtime(format!("cannot open {}: {error}", path.display())))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    let mut size = 0_u64;
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)
            .map_err(|error| Error::runtime(format!("cannot read {}: {error}", path.display())))?;
        if read == 0 {
            break;
        }
        size += read as u64;
        hasher.update(&buffer[..read]);
    }
    let digest = hasher.finalize();
    let hex: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
    Ok((size, hex))
}

/// Parses one serving flag in the shared argument loop into `flags`.
/// Returns `true` when `flag` is a serving flag (and consumed its value,
/// except `--serving`), `false` when it is not one.
pub(crate) fn parse_serving_flag(
    flag: &str,
    args: &mut Args,
    flags: &mut ServingFlags,
) -> Result<bool> {
    let once = |slot: &mut bool, what: &str| {
        if std::mem::replace(slot, true) {
            return Err(Error::invalid(format!("bench accepts {what} at most once")));
        }
        Ok(())
    };
    match flag {
        "--serving" => {
            once(&mut flags.serving, "--serving")?;
            Ok(true)
        }
        "--prompt-tokens" => {
            if flags
                .prompt_tokens
                .replace(ServingArgs::parse_positive(flag, args.value(flag)?)?)
                .is_some()
            {
                return Err(Error::invalid("bench accepts --prompt-tokens at most once"));
            }
            Ok(true)
        }
        "--decode-tokens" => {
            if flags
                .decode_tokens
                .replace(ServingArgs::parse_positive(flag, args.value(flag)?)?)
                .is_some()
            {
                return Err(Error::invalid("bench accepts --decode-tokens at most once"));
            }
            Ok(true)
        }
        "--ubatches" => {
            let parsed = ServingArgs::parse_u32_list(flag, args.value(flag)?)?;
            if parsed.contains(&0) {
                return Err(Error::invalid(
                    "bench --ubatches entries must be greater than zero",
                ));
            }
            if flags.ubatches.replace(parsed).is_some() {
                return Err(Error::invalid("bench accepts --ubatches at most once"));
            }
            Ok(true)
        }
        "--seqs" => {
            let parsed = ServingArgs::parse_u32_list(flag, args.value(flag)?)?;
            if parsed.contains(&0) {
                return Err(Error::invalid(
                    "bench --seqs entries must be greater than zero",
                ));
            }
            if flags.seqs.replace(parsed).is_some() {
                return Err(Error::invalid("bench accepts --seqs at most once"));
            }
            Ok(true)
        }
        "--repeats" => {
            if flags
                .repeats
                .replace(ServingArgs::parse_positive(flag, args.value(flag)?)?)
                .is_some()
            {
                return Err(Error::invalid("bench accepts --repeats at most once"));
            }
            Ok(true)
        }
        "--cache" => {
            if flags.cache.replace(args.value(flag)?.to_string()).is_some() {
                return Err(Error::invalid("bench accepts --cache at most once"));
            }
            Ok(true)
        }
        "--devices" => {
            if flags
                .devices
                .replace(args.value(flag)?.to_string())
                .is_some()
            {
                return Err(Error::invalid("bench accepts --devices at most once"));
            }
            Ok(true)
        }
        "--out" => {
            if flags
                .out
                .replace(PathBuf::from(args.value(flag)?))
                .is_some()
            {
                return Err(Error::invalid("bench accepts --out at most once"));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serving_args_apply_defaults_and_reject_bad_values() {
        let args = ServingArgs::from_flags(&ServingFlags::default(), None).unwrap();
        assert_eq!(args.prompt_tokens, 512);
        assert_eq!(args.decode_tokens, 64);
        assert_eq!(args.ubatches, vec![1, 8, 32, 512]);
        assert_eq!(args.seqs, vec![1, 4]);
        assert_eq!(args.repeats, 3);
        assert_eq!(args.cache, CacheMode::Both);
        assert_eq!(args.devices, "auto");

        let flags = ServingFlags {
            prompt_tokens: Some(128),
            decode_tokens: Some(16),
            ubatches: Some(vec![8, 32]),
            seqs: Some(vec![2]),
            repeats: Some(1),
            cache: Some("cold".to_string()),
            devices: Some("cpu".to_string()),
            out: Some(PathBuf::from("/tmp/report.md")),
            ..ServingFlags::default()
        };
        let args = ServingArgs::from_flags(&flags, None).unwrap();
        assert_eq!(args.ubatches, vec![8, 32]);
        assert_eq!(args.cache, CacheMode::Cold);
        assert_eq!(args.out, Some(PathBuf::from("/tmp/report.md")));

        let zero_prompt = ServingFlags {
            prompt_tokens: Some(0),
            ..ServingFlags::default()
        };
        assert!(ServingArgs::from_flags(&zero_prompt, None).is_err());
        let empty_ubatches = ServingFlags {
            ubatches: Some(vec![]),
            ..ServingFlags::default()
        };
        assert!(ServingArgs::from_flags(&empty_ubatches, None).is_err());
        let bad_cache = ServingFlags {
            cache: Some("warm".to_string()),
            ..ServingFlags::default()
        };
        assert!(ServingArgs::from_flags(&bad_cache, None).is_err());
        assert!(ServingArgs::parse_u32_list("--ubatches", "1,,8").is_err());
        assert!(ServingArgs::parse_u32_list("--ubatches", "x").is_err());
        assert_eq!(
            ServingArgs::parse_u32_list("--seqs", "1, 4,32").unwrap(),
            vec![1, 4, 32]
        );
        assert!(ServingArgs::parse_positive("--repeats", "0").is_err());
        assert_eq!(ServingArgs::parse_positive("--repeats", "5").unwrap(), 5);
    }

    #[test]
    fn cache_mode_parses_and_lists_passes() {
        assert_eq!(CacheMode::parse("COLD").unwrap(), CacheMode::Cold);
        assert_eq!(CacheMode::parse("hot").unwrap(), CacheMode::Hot);
        assert_eq!(CacheMode::parse("both").unwrap(), CacheMode::Both);
        assert!(CacheMode::parse("warm").is_err());
        assert_eq!(CacheMode::Both.passes().len(), 2);
        assert_eq!(CacheMode::Cold.passes(), &[CacheMode::Cold]);
    }

    #[test]
    fn median_interpolates_even_counts() {
        assert_eq!(median([3.0]), 3.0);
        assert_eq!(median([1.0, 2.0, 3.0, 4.0]), 2.5);
        assert_eq!(median([5.0, 1.0, 3.0]), 3.0);
    }

    #[test]
    fn buffer_bytes_counts_each_buffer_once_across_device_and_host() {
        let host_only = MemoryReport {
            host_bytes: 1000,
            ..MemoryReport::default()
        };
        assert_eq!(buffer_bytes(&host_only), 1000);
        // On a unified-memory device the device and host buffers share one
        // pool but are distinct allocations: the sum counts each once.
        let both = MemoryReport {
            host_bytes: 1000,
            device_bytes: 4000,
            ..MemoryReport::default()
        };
        assert_eq!(buffer_bytes(&both), 5000);
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn disk_read_bytes_is_available_and_monotonic() {
        let first = disk_read_bytes();
        assert!(first.is_some(), "unix reports disk reads");
        let second = disk_read_bytes().unwrap();
        assert!(second >= first.unwrap());
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn evict_page_cache_accepts_a_real_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.gguf");
        std::fs::write(&path, b"weights").unwrap();
        evict_page_cache(&path).unwrap();
        assert!(evict_page_cache(&dir.path().join("missing.gguf")).is_err());
    }

    #[test]
    fn report_markdown_has_provenance_and_rows() {
        let row = ServingRow {
            device: "cpu",
            ubatch: 8,
            n_seq: 1,
            cache: "cold",
            prefill_tok_s: 12.5,
            decode_tok_s: 3.5,
            prefill: Duration::from_secs_f64(4.0),
            decode: Duration::from_secs_f64(2.0),
            disk_read_bytes: Some(1 << 20),
            cold_effective: true,
            buffer_bytes: 100 << 20,
            kv_bytes: 2 << 20,
            repeats: vec![ServingRepeat {
                prefill: Duration::from_secs(4),
                decode: Duration::from_secs(2),
                disk_read_bytes: Some(1 << 20),
            }],
        };
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.gguf");
        std::fs::write(&model, b"tiny").unwrap();
        let out = dir.path().join("report.md");
        let args = ServingArgs::from_flags(&ServingFlags::default(), None).unwrap();
        write_report(&out, &[model.clone()], &args, &[row]).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.contains("# serving bench"), "{text}");
        assert!(text.contains("- command: `"), "{text}");
        assert!(text.contains("- machine: "), "{text}");
        assert!(text.contains("- commit: "), "{text}");
        assert!(text.contains("sha256"), "{text}");
        assert!(
            text.contains("| cpu | 8 | 1 | cold | 12.5 | 3.5 |"),
            "{text}"
        );
        assert!(text.contains("## raw repeats"), "{text}");
        assert!(write_report(&out, &[model], &args, &[]).is_err());
        assert_eq!(std::fs::read_to_string(&out).unwrap(), text);
    }
}
