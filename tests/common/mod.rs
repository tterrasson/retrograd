//! Shared helpers for the targeted Metal/CPU integration tests.
//!
//! Model-dependent CPU tests use the declared GGUF fixtures, each materialized
//! and checksum-verified by `scripts/fetch-cpu-fixture.sh` (downloaded or
//! generated, depending on its manifest). Override a fixture's location with
//! the variable its manifest names (`RETRO_CPU_FIXTURE`, `RETRO_TINY_FIXTURE`).
//!
//! Not every test binary uses every helper, so silence dead-code warnings.
#![allow(dead_code)]

/// The fixed-block Gefen algorithm in slow F64, shared by the model-driven and
/// op-driven Gefen tests.
pub mod gefen;

use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    process::Command,
    thread,
    time::{Duration, Instant},
};

/// Repository-relative location of the CPU integration fixture.
pub const CPU_FIXTURE: &str = "tests/fixtures/LFM2.5-230M-Q4_K_M.gguf";

/// The generated CPU fixture: F32 throughout, with an untied projection head.
pub const TINY_FIXTURE: &str = "tests/fixtures/retrograd-tiny-qwen2-f32.gguf";

/// The same generated model with its matrices stored as F16. Same numbers,
/// different storage precision, so runs of the two are comparable.
pub const TINY_F16_FIXTURE: &str = "tests/fixtures/retrograd-tiny-qwen2-f16.gguf";

/// The same generated model again, matrices stored as Q8_0. Its use is the
/// quantized-anchor measurement: the same numbers as the F32 fixture, so a
/// score that differs differs because of quantization.
pub const TINY_Q8_FIXTURE: &str = "tests/fixtures/retrograd-tiny-qwen2-q8_0.gguf";

/// Repository-relative default location for the Vulkan integration model.
/// Nothing ships at this path - `RETRO_VULKAN_TEST_MODEL` is how a developer
/// points at wherever they keep it, and the `_if_available` accessors skip
/// cleanly when neither is set.
pub const DEFAULT_VULKAN_MODEL: &str = "tests/fixtures/gemma-3-270m-it-Q4_K_M.gguf";

/// Repository-relative default location for the Falcon-H1 MoltenVK
/// regression model. Override with `RETRO_FALCON_H1_TEST_MODEL`.
pub const DEFAULT_FALCON_H1_MODEL: &str = "tests/fixtures/Falcon-H1-Tiny-90M-Instruct-Q5_K_M.gguf";

/// Resolves the CPU fixture path, honouring an explicit test override.
pub fn model_path() -> PathBuf {
    std::env::var("RETRO_CPU_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CPU_FIXTURE))
}

/// Returns the CPU fixture when available. The dedicated CPU lane sets
/// `RETRO_REQUIRE_CPU_FIXTURE=1`, converting a missing fixture into a useful
/// failure instead of allowing the lane to pass by skipping model coverage.
pub fn model_path_if_available() -> Option<PathBuf> {
    let path = model_path();
    if path.exists() {
        Some(path)
    } else if std::env::var_os("RETRO_REQUIRE_CPU_FIXTURE").is_some() {
        panic!(
            "CPU integration fixture missing at {}; run scripts/fetch-cpu-fixture.sh",
            path.display()
        );
    } else {
        None
    }
}

/// Resolves the tiny fixture path, honouring an explicit test override.
pub fn tiny_model_path() -> PathBuf {
    std::env::var("RETRO_TINY_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TINY_FIXTURE))
}

/// Returns the tiny fixture when available. It is generated, not downloaded,
/// so `RETRO_REQUIRE_CPU_FIXTURE=1` covers it too.
pub fn tiny_model_path_if_available() -> Option<PathBuf> {
    let path = tiny_model_path();
    if path.exists() {
        Some(path)
    } else if std::env::var_os("RETRO_REQUIRE_CPU_FIXTURE").is_some() {
        panic!(
            "tiny CPU fixture missing at {}; run scripts/fetch-cpu-fixture.sh",
            path.display()
        );
    } else {
        None
    }
}

/// Resolves the F16 tiny fixture, honouring an explicit test override.
pub fn tiny_f16_model_path() -> PathBuf {
    std::env::var("RETRO_TINY_F16_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TINY_F16_FIXTURE))
}

/// Returns the F16 tiny fixture when available, required by the dedicated CPU
/// lane like its F32 twin.
pub fn tiny_f16_model_path_if_available() -> Option<PathBuf> {
    let path = tiny_f16_model_path();
    if path.exists() {
        Some(path)
    } else if std::env::var_os("RETRO_REQUIRE_CPU_FIXTURE").is_some() {
        panic!(
            "F16 tiny CPU fixture missing at {}; run scripts/fetch-cpu-fixture.sh",
            path.display()
        );
    } else {
        None
    }
}

/// Resolves the Q8_0 tiny fixture, honouring an explicit test override.
pub fn tiny_q8_model_path() -> PathBuf {
    std::env::var("RETRO_TINY_Q8_FIXTURE")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(TINY_Q8_FIXTURE))
}

/// Returns the Q8_0 tiny fixture when available, required by the dedicated CPU
/// lane like its two twins.
pub fn tiny_q8_model_path_if_available() -> Option<PathBuf> {
    let path = tiny_q8_model_path();
    if path.exists() {
        Some(path)
    } else if std::env::var_os("RETRO_REQUIRE_CPU_FIXTURE").is_some() {
        panic!(
            "Q8_0 tiny CPU fixture missing at {}; run scripts/fetch-cpu-fixture.sh",
            path.display()
        );
    } else {
        None
    }
}

/// Resolves the Vulkan integration model, independently from the existing
/// Q8_0 Metal reference model. Override with `RETRO_VULKAN_TEST_MODEL`.
pub fn vulkan_model_path() -> PathBuf {
    std::env::var("RETRO_VULKAN_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_VULKAN_MODEL))
}

pub fn vulkan_model_path_if_available() -> Option<PathBuf> {
    let path = vulkan_model_path();
    path.exists().then_some(path)
}

/// Resolves the Falcon-H1 regression model, honouring
/// `RETRO_FALCON_H1_TEST_MODEL`.
pub fn falcon_h1_model_path() -> PathBuf {
    std::env::var("RETRO_FALCON_H1_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_FALCON_H1_MODEL))
}

pub fn falcon_h1_model_path_if_available() -> Option<PathBuf> {
    let path = falcon_h1_model_path();
    path.exists().then_some(path)
}

/// One family of recurrent state, and the model that stands for it.
///
/// "Recurrent support" is not one claim: the graphs differ by how they carry
/// state, and so does what a backend has to get right. Three families exist
/// today and each needs its own representative, because a lane that only ever
/// loads the LFM2 fixture proves nothing about a Mamba2 conv+scan or a gated
/// delta net.
pub struct RecurrentFamily {
    /// How this family carries state, not the name of one model.
    pub family: &'static str,
    /// Environment variable supplying a GGUF for it.
    pub env: &'static str,
    /// Repository-relative fallback, used when the variable is unset.
    pub default_path: &'static str,
}

/// The families, in the order a lane should report them.
pub const RECURRENT_FAMILIES: &[RecurrentFamily] = &[
    // Indexed ShortConv. The default CPU fixture, so this row is always
    // covered; it is also the only family llama.cpp declares packable today.
    RecurrentFamily {
        family: "shortconv",
        env: "RETRO_LFM2_TEST_MODEL",
        default_path: CPU_FIXTURE,
    },
    // Convolution + selective scan (Mamba2-style), the family behind
    // falcon-h1, granite-hybrid, jamba and nemotron-h.
    RecurrentFamily {
        family: "conv_ssm",
        env: "RETRO_FALCON_H1_TEST_MODEL",
        default_path: DEFAULT_FALCON_H1_MODEL,
    },
    // Gated delta net (Qwen3-Next / Qwen3.5), the family with a fused kernel
    // and a separate differentiable chunking graph.
    RecurrentFamily {
        family: "gated_delta_net",
        env: "RETRO_QWEN3NEXT_TEST_MODEL",
        default_path: "tests/fixtures/qwen3next-tiny.gguf",
    },
];

impl RecurrentFamily {
    pub fn path(&self) -> PathBuf {
        std::env::var(self.env)
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(self.default_path))
    }

    pub fn path_if_available(&self) -> Option<PathBuf> {
        let path = self.path();
        path.exists().then_some(path)
    }
}

/// The llama.cpp training runtime shares process-global state (ggml thread
/// pools, backend registry), and concurrent trainers in one process can trip
/// KV-cache asserts. Model-loading tests take this lock so a test binary
/// behaves like `--test-threads=1` without requiring callers to know that.
static MODEL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub struct ModelLock {
    _thread: std::sync::MutexGuard<'static, ()>,
    process: ProcessLock,
}

impl Drop for ModelLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.process.path);
    }
}

struct ProcessLock {
    path: PathBuf,
}

fn stale_process_lock(path: &std::path::Path) -> bool {
    let Ok(contents) = fs::read_to_string(path) else {
        return false;
    };
    let Some(pid) = contents.trim().strip_prefix("pid=") else {
        return false;
    };
    let Ok(pid) = pid.parse::<u32>() else {
        return false;
    };
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .status()
        .map(|status| !status.success())
        // If the probe itself cannot run, preserve the lock instead of
        // risking concurrent runtime initialization.
        .unwrap_or(false)
}

/// Serializes the runtime across both threads and separately spawned test
/// binaries. A `Mutex` alone cannot protect the latter: cargo runs integration
/// test binaries as distinct processes.
pub fn serialize_models() -> ModelLock {
    let started = Instant::now();
    let thread = MODEL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let path = std::env::var_os("RETRO_RUNTIME_LOCK_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("retrograd-runtime.lock"));
    let deadline = Instant::now() + Duration::from_secs(300);
    loop {
        match OpenOptions::new().write(true).create_new(true).open(&path) {
            Ok(mut file) => {
                let _ = writeln!(file, "pid={}", std::process::id());
                if std::env::var_os("RETRO_TEST_TIMING").is_some() {
                    eprintln!(
                        "test_phase=runtime_lock_wait duration_ms={}",
                        started.elapsed().as_millis()
                    );
                }
                return ModelLock {
                    _thread: thread,
                    process: ProcessLock { path },
                };
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                // A killed test binary cannot run `Drop`; reclaim only a lock
                // whose recorded owner PID is no longer alive.
                if stale_process_lock(&path) {
                    let _ = fs::remove_file(&path);
                    continue;
                }
                assert!(
                    Instant::now() < deadline,
                    "timed out waiting for runtime lock {}; remove it only after confirming no CPU integration test is running",
                    path.display()
                );
                thread::sleep(Duration::from_millis(50));
            }
            Err(error) => panic!("create runtime lock {}: {error}", path.display()),
        }
    }
}

/// Resolves the lowest block index whose LoRA-eligible tensors include every
/// named family, e.g. `["attn_k", "attn_v"]` -> the first attention block.
///
/// Hard-coding `blk.0` only holds on an architecture whose
/// blocks are all alike. On a hybrid model (`lfm2` interleaves shortconv and
/// attention blocks, and it is the CPU fixture) `blk.0` carries no attention
/// projection at all and target resolution fails outright. Asking the loaded
/// model keeps the test's intent - one block, minimal adapter - without
/// assuming a layout.
pub fn first_block_with(trainer: &retrograd::Trainer, families: &[&str]) -> Option<u32> {
    let names = trainer.lora_candidate_targets().ok()?;
    let mut blocks: Vec<u32> = names
        .iter()
        .filter_map(|name| {
            let rest = name.strip_prefix("blk.")?;
            let (index, _) = rest.split_once('.')?;
            index.parse::<u32>().ok()
        })
        .collect();
    blocks.sort_unstable();
    blocks.dedup();
    // Matched on the full name the caller will pass to the resolver, not on a
    // prefix: `attn_q` must not be satisfied by an `attn_q_norm` entry.
    blocks.into_iter().find(|index| {
        families
            .iter()
            .all(|family| names.contains(&block_target(*index, family)))
    })
}

fn block_target(index: u32, family: &str) -> String {
    format!("blk.{index}.{family}.weight")
}

/// [`first_block_with`] rendered as concrete `blk.N.<family>.weight` patterns,
/// in the order the families were given. `None` when the model has no block
/// carrying all of them.
pub fn block_targets(trainer: &retrograd::Trainer, families: &[&str]) -> Option<Vec<String>> {
    let index = first_block_with(trainer, families)?;
    Some(
        families
            .iter()
            .map(|family| block_target(index, family))
            .collect(),
    )
}

/// Whether this build compiled the Metal backend in.
pub fn metal_compiled() -> bool {
    cfg!(retro_metal)
}

/// Whether a registered GPU can initialize a real device context and queue.
///
/// A backend-list entry only proves registration. In particular, Metal can be
/// registered in a headless process while command-queue creation fails. GPU
/// tests must use this probe before executing kernels or loading a model.
pub fn gpu_device_present() -> bool {
    retrograd::gpu_runtime_available()
}

/// Whether RIR may encode in this process, read from `RETRO_RIR_MODE`.
///
/// An **unset** variable means `prefer`, not `off`:
/// the promoted pairs are the build's dispatch path. Reading unset as `off`
/// would make the RIR tests skip on exactly the configuration they exist to
/// cover - and `rir_request_without_the_mode_is_a_clean_error` would fail
/// instead, since it asserts the *refusal* of a RIR request, which does not
/// happen when RIR is on.
///
/// Shared rather than duplicated per binary: `rir_probe.rs` and
/// `rir_graph_coverage.rs` each carried their own copy, and a copy is a place
/// where the next contract change lands on one reader but not the other.
pub fn rir_encoding_enabled() -> bool {
    matches!(
        std::env::var("RETRO_RIR_MODE").as_deref(),
        Ok("prefer") | Ok("require") | Err(_)
    )
}

/// Parses a `retro_trainer_backend_report` / `describe_lora` style report into a
/// flat key/value-ish view: returns the count for a `"<section>"` indented block
/// entry, e.g. buffer -> count. We keep it dead simple: return the whole block
/// following a section header line.
pub fn section_lines<'a>(report: &'a str, header: &str) -> Vec<&'a str> {
    let mut out = Vec::new();
    let mut in_section = false;
    for line in report.lines() {
        let trimmed_indent = line.trim_start();
        if line.trim_end().ends_with(&format!("{header}:"))
            && line.trim_start() == format!("{header}:")
        {
            in_section = true;
            continue;
        }
        if in_section {
            // Section entries are indented deeper than the header (4 spaces).
            let indent = line.len() - trimmed_indent.len();
            if indent >= 4 && !trimmed_indent.is_empty() {
                out.push(trimmed_indent);
            } else {
                in_section = false;
            }
        }
    }
    out
}

/// Returns true if any line in the given section names a Metal buffer. ggml
/// labels the Metal buffer `MTL<n>` (e.g. `MTL0`); some versions use `Metal`.
pub fn section_has_metal(report: &str, header: &str) -> bool {
    section_lines(report, header).iter().any(|line| {
        let lower = line.to_lowercase();
        lower.contains("mtl") || lower.contains("metal")
    })
}

/// Returns true if a report section contains a Vulkan device buffer.
pub fn section_has_vulkan(report: &str, header: &str) -> bool {
    section_lines(report, header)
        .iter()
        .any(|line| line.to_lowercase().contains("vulkan"))
}

/// Returns true if a report section contains a CUDA device buffer. ggml labels
/// CUDA buffers `CUDA<n>` (e.g. `CUDA0`), sometimes with a `_Host` pinned-memory
/// variant, so match the `cuda` substring case-insensitively.
pub fn section_has_cuda(report: &str, header: &str) -> bool {
    section_lines(report, header)
        .iter()
        .any(|line| line.to_lowercase().contains("cuda"))
}

/// Whether this build compiled the CUDA backend in.
pub fn cuda_compiled() -> bool {
    cfg!(retro_cuda)
}

/// Whether a CUDA device is registered in the ggml backend registry. Only
/// meaningful in a `retro_cuda` build; registration alone does not prove the
/// device can initialize a context/stream - pair it with `gpu_device_present`.
pub fn cuda_registered() -> bool {
    retrograd::backend_list()
        .map(|list| {
            list.lines()
                .any(|line| line.starts_with("gpu\t") && line.contains("CUDA"))
        })
        .unwrap_or(false)
}

/// Resolves the CUDA integration model. Defaults to the redistributable in-repo
/// CPU fixture (a small Qwen3 GGUF); override with `RETRO_CUDA_TEST_MODEL` to
/// exercise a different architecture. Never hard-codes a developer path.
pub fn cuda_model_path() -> PathBuf {
    std::env::var("RETRO_CUDA_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(CPU_FIXTURE))
}

pub fn cuda_model_path_if_available() -> Option<PathBuf> {
    let path = cuda_model_path();
    path.exists().then_some(path)
}

/// Resolves the Falcon-H1 CUDA regression model, honouring
/// `RETRO_CUDA_FALCON_H1_TEST_MODEL` and falling back to the shared
/// Falcon-H1 fixture variable.
pub fn cuda_falcon_h1_model_path() -> PathBuf {
    std::env::var("RETRO_CUDA_FALCON_H1_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| falcon_h1_model_path())
}

pub fn cuda_falcon_h1_model_path_if_available() -> Option<PathBuf> {
    let path = cuda_falcon_h1_model_path();
    path.exists().then_some(path)
}

/// Restores an environment variable when the test leaves, so a runtime knob
/// forced for one case cannot leak into the rest of the suite.
///
/// This is the single place in the test tree allowed to mutate the process
/// environment, so that the audit `set_var` demands since edition 2024 has to
/// be done once instead of per copy.
///
/// Every lane that reaches a `EnvGuard` runs its binary with
/// `--test-threads=1` (`scripts/test-cpu-integration.sh`, `test-abi.sh`, and
/// the GPU invocations in `docs/engineering/tests/notice.md`): libtest
/// then runs the cases one after another on the main thread, so no sibling
/// test is reading the environment while this writes to it. The runtime's own
/// worker threads are not a second reader either -- the knobs guarded here are
/// read when the graph or the backend is built, on the thread that constructed
/// the guard, and the pools are joined before it drops.
///
/// Adding a case to a binary that uses this therefore comes with an
/// obligation: the lane that runs it must keep `--test-threads=1`.
pub struct EnvGuard(&'static str, Option<String>);

impl EnvGuard {
    pub fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        // SAFETY: single-threaded by lane contract, see the type's documentation.
        unsafe { std::env::set_var(key, value) };
        Self(key, previous)
    }
}

impl Drop for EnvGuard {
    fn drop(&mut self) {
        // SAFETY: as in `set` -- same thread, same single-threaded lane.
        match self.1.take() {
            Some(value) => unsafe { std::env::set_var(self.0, value) },
            None => unsafe { std::env::remove_var(self.0) },
        }
    }
}

/// Asserts that every type in `GGML_RETRO_DEQUANT_TYPES` produces the CPU's
/// `out_prod` on the compiled GPU backend, named by `backend` in failure messages.
///
/// The type list comes from `retrograd::dequant_types()`, i.e. straight from the
/// fork's table, so this grows with the table rather than needing a case per type.
/// That matters because the drift it replaced had exactly this shape: Metal shipped
/// Q5_0 but not Q4_0 and no IQ type, Vulkan shipped Q4_0 but no IQ type, CUDA
/// shipped every quant but not F16 -- and nothing failed, because no test asked for
/// the types that were missing.
///
/// This is not vacuous: `probe_op` computes on a single backend with no scheduler,
/// and ggml's backends abort on an op they do not support (verified: an id absent
/// from the table aborts in `ggml_metal_op_encode`). A type the backend rejects
/// therefore fails loudly instead of yielding a CPU result that passes.
pub fn assert_out_prod_all_dequant_types_match_cpu(backend: &str) {
    let types = retrograd::dequant_types();
    assert!(
        types.len() >= 19,
        "dequant_types() returned {} entries, expected the full table",
        types.len()
    );

    // ne_src0[0] of 256 and 512 is a whole number of blocks for every type in the
    // table (the largest block is 256) and a multiple of the 16-value chunk the
    // in-kernel decoders emit. The second case adds two blocks per row and batched
    // planes; ne_src1[0] of 11 and 5 leaves the 16-column output tile partly
    // inactive, which is what the training graph actually dispatches (ne_src1[0] is
    // the ubatch).
    let cases = [
        ([256_i64, 7, 1, 1], [11_i64, 7, 1, 1]),
        ([512, 9, 2, 1], [5, 9, 2, 1]),
    ];

    for (type_index, (type_id, type_name)) in types.iter().enumerate() {
        for (case_index, (ne_src0, ne_src1)) in cases.into_iter().enumerate() {
            let seed_base = 31 * type_index as u64 + case_index as u64;
            let src0 =
                deterministic_f32s(ne_src0.iter().product::<i64>() as usize, 0x5150 + seed_base);
            let src1 =
                deterministic_f32s(ne_src1.iter().product::<i64>() as usize, 0x6260 + seed_base);
            let out_len = (ne_src0[0] * ne_src1[0] * ne_src1[2] * ne_src1[3]) as usize;
            let params = [*type_id as f32, 0.0];
            let op = retrograd::ProbeOp::OutProdQuant;
            let cpu = retrograd::probe_op(
                op,
                false,
                retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
                params,
                out_len,
            )
            .unwrap_or_else(|e| panic!("CPU out_prod for {type_name}: {e}"));
            let gpu = retrograd::probe_op(
                op,
                true,
                retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
                params,
                out_len,
            )
            .unwrap_or_else(|e| panic!("{backend} out_prod for {type_name}: {e}"));

            assert!(
                cpu.iter().any(|v| v.abs() > 1e-6),
                "{type_name} case {case_index}: CPU output is all ~0, \
                 so the comparison would prove nothing"
            );
            // The CPU reference decodes the very same bytes, so this bound is about
            // summation order, not quantization error: a coarse type is no licence
            // for a looser tolerance.
            for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
                let d = (c - g).abs();
                assert!(
                    d <= 2.0e-3,
                    "out_prod {type_name} case {case_index} mismatch at {i}: \
                     cpu={c} {backend}={g} diff={d}"
                );
            }
        }
    }
}

/// Exercises the CPU implementation for every cross-backend training type.
/// GPU parity tests use CPU as their oracle, but this separate contract keeps
/// the whole table in the model-free CPU lane as well.
pub fn assert_out_prod_all_dequant_types_run_on_cpu() {
    let types = retrograd::dequant_types();
    assert!(types.len() >= 19, "incomplete dequant type table");
    let ne_src0 = [256_i64, 7, 1, 1];
    let ne_src1 = [11_i64, 7, 1, 1];
    let src0 = deterministic_f32s((256 * 7) as usize, 0x7150);
    let src1 = deterministic_f32s((11 * 7) as usize, 0x7260);

    for (type_id, type_name) in types {
        let output = retrograd::probe_op(
            retrograd::ProbeOp::OutProdQuant,
            false,
            retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
            [type_id as f32, 0.0],
            256 * 11,
        )
        .unwrap_or_else(|e| panic!("CPU out_prod for {type_name}: {e}"));
        assert!(
            output.iter().all(|v| v.is_finite()),
            "{type_name}: non-finite output"
        );
        assert!(
            output.iter().any(|v| v.abs() > 1e-6),
            "{type_name}: vacuous all-zero output"
        );
    }
}

/// Rarer formats `out_prod` decodes in place on CPU/CUDA/Vulkan but intentionally excluded
/// from the Metal/CE-oriented `retro_dequant_types()` ABI table.  The numeric ids
/// are stable ggml ABI values and are named here so a mismatch is diagnosable.
pub const OUT_PROD_EXTRA_TYPES: &[(i32, &str)] = &[
    (41, "Q1_0"),
    (42, "Q2_0"),
    (19, "IQ1_S"),
    (29, "IQ1_M"),
    (40, "NVFP4"),
];

pub fn assert_out_prod_extra_types_match_cpu(use_gpu: bool, backend: &str) {
    let cases = [
        ([256_i64, 7, 1, 1], [11_i64, 7, 1, 1]),
        ([512_i64, 9, 2, 1], [5_i64, 9, 2, 1]),
    ];
    for (type_index, (type_id, type_name)) in OUT_PROD_EXTRA_TYPES.iter().enumerate() {
        for (case_index, (ne_src0, ne_src1)) in cases.into_iter().enumerate() {
            let seed = type_index as u64 * 17 + case_index as u64;
            let src0 = deterministic_f32s(ne_src0.iter().product::<i64>() as usize, 0x9150 + seed);
            let src1 = deterministic_f32s(ne_src1.iter().product::<i64>() as usize, 0x9260 + seed);
            let out_len = (ne_src0[0] * ne_src1[0] * ne_src1[2] * ne_src1[3]) as usize;
            let params = [*type_id as f32, 0.0];
            let cpu = retrograd::probe_op(
                retrograd::ProbeOp::OutProdQuant,
                false,
                retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
                params,
                out_len,
            )
            .unwrap_or_else(|e| panic!("CPU out_prod for {type_name}: {e}"));
            let actual = retrograd::probe_op(
                retrograd::ProbeOp::OutProdQuant,
                use_gpu,
                retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
                params,
                out_len,
            )
            .unwrap_or_else(|e| panic!("{backend} out_prod for {type_name}: {e}"));
            assert!(
                actual.iter().all(|v| v.is_finite()),
                "{backend} {type_name}: non-finite output"
            );
            for (i, (c, a)) in cpu.iter().zip(&actual).enumerate() {
                let diff = (c - a).abs();
                assert!(
                    diff <= 2.0e-3,
                    "{backend} out_prod {type_name} case {case_index} mismatch at {i}: cpu={c} actual={a} diff={diff}"
                );
            }
        }
    }
}

/// In-place decoding does not consult the CUDA dequantization budget on CPU,
/// Vulkan, or the native CUDA path. Use a shape for which the compatibility CUDA
/// fallback really would split at 1 MiB, so this catches accidental routing back through
/// the materialized-F32 path rather than merely checking a dormant setting.
pub fn assert_out_prod_quant_budget_independent(use_gpu: bool, backend: &str) {
    const NE00: i64 = 65536;
    const NE01: i64 = 8;
    let ne_src0 = [NE00, NE01, 1, 1];
    let ne_src1 = [5, NE01, 1, 1];
    let src0 = deterministic_f32s((NE00 * NE01) as usize, 0x811c);
    let src1 = deterministic_f32s((5 * NE01) as usize, 0x8112);
    let out_len = (NE00 * 5) as usize;

    let run = |budget: &str| {
        let _env = EnvGuard::set("GGML_CUDA_DEQUANT_BUDGET_MB", budget);
        retrograd::probe_op(
            retrograd::ProbeOp::OutProdQ4K,
            use_gpu,
            retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
            [0.0, 0.0],
            out_len,
        )
        .unwrap_or_else(|e| panic!("{backend} quantized out_prod: {e}"))
    };

    let unlimited = run("0");
    let one_mib = run("1");
    assert_eq!(
        unlimited, one_mib,
        "{backend}: GGML_CUDA_DEQUANT_BUDGET_MB changed native Q2 output"
    );
}

/// Deterministic pseudo-random f32s in [-1, 1] (xorshift64*, no rand dependency).
/// Each backend test file has its own copy for its local cases; this one exists so
/// the shared sweep above does not depend on which file calls it.
pub fn deterministic_f32s(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545F4914F6CDD1D) >> 40;
            let unit = (bits as f32) / ((1u32 << 24) as f32);
            unit * 2.0 - 1.0
        })
        .collect()
}

/// A type outside `GGML_RETRO_OUT_PROD_TYPES` must come back as an error, not take
/// the process down.
///
/// The probe computes on a single backend with no scheduler, so an unlisted type
/// would reach `ggml_out_prod` directly: the CPU reference `GGML_ABORT`s on BF16, and
/// a GPU backend aborts on the missing pipeline (verified: an unlisted id aborts in
/// `ggml_metal_op_encode`). Validating only "is a valid ggml_type id" was therefore
/// not enough -- it turned a bad argument into SIGABRT.
///
/// BF16 (id 30) is the sharpest case: it is a real, common type that Metal and Vulkan
/// can both decode, so nothing about it looks wrong until the CPU oracle aborts.
pub fn assert_out_prod_rejects_unlisted_type(use_gpu: bool) {
    const GGML_TYPE_BF16: i32 = 30;
    let types = retrograd::dequant_types();
    assert!(
        !types.iter().any(|(id, _)| *id == GGML_TYPE_BF16),
        "this test assumes BF16 is absent from the table; if it was added, pick \
         another unlisted id"
    );

    let ne_src0 = [256_i64, 4, 1, 1];
    let ne_src1 = [8_i64, 4, 1, 1];
    let src0 = deterministic_f32s(256 * 4, 0x7f7f);
    let src1 = deterministic_f32s(8 * 4, 0x8e8e);
    let result = retrograd::probe_op(
        retrograd::ProbeOp::OutProdQuant,
        use_gpu,
        retrograd::ProbeInputs::pair(ne_src0, &src0, ne_src1, &src1),
        [GGML_TYPE_BF16 as f32, 0.0],
        256 * 8,
    );
    let err = result.expect_err("BF16 is not in the table, so the probe must refuse it");
    let message = err.to_string();
    assert!(
        message.contains("GGML_RETRO_OUT_PROD_TYPES"),
        "the refusal should name the table so the caller knows what to enumerate, \
         got: {message}"
    );
}
