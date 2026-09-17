//! Synthetic fixtures for the resolver's own tests and for the server's.
//!
//! Real `ModelInfo` values come out of a GGUF; nothing here does. A table of
//! made-up geometries is what the bulk of the coverage runs on:
//! "no model, no GPU, fast lane".

use std::path::PathBuf;

use retrograd_core::ModelInfo;

use crate::budget::{BudgetRequest, MarginPolicy, MemoryBaseline};
use crate::cost::{Workload, WorkloadKind};
use crate::dataset::DatasetStats;
use crate::recipe::{DataSpec, Limits, Objective, Recipe};
use crate::resolver::ResolveInput;
use crate::{Backend, HardwareFacts};

/// A small dense transformer with a large vocabulary - the shape that makes the
/// logits post matter.
pub fn tiny_model() -> ModelInfo {
    ModelInfo {
        n_layer: 24,
        n_embd: 1024,
        n_ff: 4096,
        n_head: 16,
        n_head_kv: 8,
        n_embd_head_k: 64,
        n_embd_head_v: 64,
        n_embd_k_gqa: 512,
        n_embd_v_gqa: 512,
        n_embd_r: 0,
        n_embd_s: 0,
        n_vocab: 151_936,
        n_ctx_train: 32_768,
        n_expert: 0,
        n_expert_used: 0,
        n_params: 600_000_000,
        model_size_bytes: 400 * 1024 * 1024,
        file_size_bytes: 400 * 1024 * 1024,
        dominant_weight_bytes: 350 * 1024 * 1024,
        tied_embeddings: true,
        is_recurrent: false,
        has_encoder: false,
        architecture: "qwen3".to_string(),
        dominant_weight_type: "Q4_K".to_string(),
    }
}

/// The same model, an order of magnitude larger: what a small budget has to
/// refuse.
pub fn large_model() -> ModelInfo {
    ModelInfo {
        n_layer: 48,
        n_embd: 4096,
        n_head: 32,
        n_head_kv: 8,
        n_embd_k_gqa: 1024,
        n_embd_v_gqa: 1024,
        n_params: 13_000_000_000,
        model_size_bytes: 8 * 1024 * 1024 * 1024,
        file_size_bytes: 8 * 1024 * 1024 * 1024,
        architecture: "llama".to_string(),
        ..tiny_model()
    }
}

pub fn sft_workload() -> Workload {
    Workload {
        kind: WorkloadKind::Sft,
        examples: 4_000,
        co_resident_bytes: 0,
    }
}

/// A dataset of `examples` conversations, each `tokens` long. Deterministic, so
/// a snapshot of a resolution stays a snapshot.
pub fn uniform_dataset(examples: u64, tokens: u32) -> DatasetStats {
    DatasetStats::measured(vec![tokens; examples as usize])
}

/// A long-tailed dataset: `examples` conversations, nine tenths of them `short`
/// tokens and the rest `long`. A uniform corpus has P50 = P99, which makes the
/// truncation lever unreachable by construction - a spread is what a real
/// dataset looks like.
pub fn mixed_dataset(examples: u64, short: u32, long: u32) -> DatasetStats {
    let tail = (examples / 10).max(1);
    let mut lengths = vec![short; (examples - tail) as usize];
    lengths.extend(std::iter::repeat_n(long, tail as usize));
    DatasetStats::measured(lengths)
}

/// A machine with `device` bytes of VRAM and `host` bytes of RAM, nothing else
/// running.
pub fn machine(device: u64, host: u64) -> MemoryBaseline {
    MemoryBaseline {
        device_total: Some(device),
        device_used: 0,
        host_total: Some(host),
        host_used: 0,
        unified: false,
    }
}

/// A minimal supervised recipe against `model`.
pub fn sft_recipe() -> Recipe {
    Recipe {
        objective: Objective::InstructionTuning,
        model: PathBuf::from("model.gguf"),
        data: DataSpec {
            path: Some(PathBuf::from("data.jsonl")),
            dataset: None,
            format: Some("jsonl".to_string()),
        },
        eval: None,
        budget: None,
        limits: Limits::default(),
        allow: Vec::new(),
        reward: None,
        judge: None,
        tools: Vec::new(),
        output: Some(PathBuf::from("adapter.gguf")),
        checkpoint_dir: None,
        seed: Some(7),
    }
}

/// A machine whose backend runs everything the resolver may turn on.
///
/// Any named backend does; CUDA remains the fixture
/// because it is the discrete-GPU case, where device memory is its own budget;
/// [`Backend::Unknown`] is the branch a test has to ask for explicitly, since
/// that is the one every rule still treats conservatively.
pub fn hardware(device_total: u64) -> HardwareFacts {
    HardwareFacts {
        backend: Backend::Cuda,
        unified_memory: false,
        device_total_bytes: Some(device_total),
    }
}

/// A profile in the shape the server builds one, with no measured kernel row:
/// what the planner needs is the published capability set, and every rule that
/// keys on one has to be exercised against a profile rather than against the
/// `None` that only pure analytical callers pass.
pub fn execution_profile(shared_prefix_packed_training: bool) -> retrograd_core::ExecutionProfile {
    retrograd_core::ExecutionProfile {
        schema_version: retrograd_core::EXECUTION_PROFILE_VERSION,
        engine_fingerprint: "retrograd-plan-fixture".to_string(),
        kernel_catalog_fingerprint: "sha256:fixture".to_string(),
        device: retrograd_core::DeviceProfile {
            stable_id: "cuda:fixture".to_string(),
            backend: "cuda".to_string(),
            ..Default::default()
        },
        capabilities: retrograd_core::ModelCapabilities {
            shared_prefix_packed_training,
            ..Default::default()
        },
        ..Default::default()
    }
}

/// Assembles a [`ResolveInput`] over the fixtures above. The borrowed pieces
/// have to outlive the input, so the caller owns them.
#[allow(clippy::too_many_arguments)]
pub fn input<'a>(
    recipe: &'a Recipe,
    params: &'a serde_json::Value,
    model: &'a ModelInfo,
    data: &'a DatasetStats,
    baseline: MemoryBaseline,
) -> ResolveInput<'a> {
    input_with_eval(recipe, params, model, data, None, baseline)
}

/// Same as [`input`], with an eval dataset wired in - what `resolve::plan_recipe`
/// does once `recipe.eval` is set, and what most fixtures do not need to
/// bother with.
#[allow(clippy::too_many_arguments)]
pub fn input_with_eval<'a>(
    recipe: &'a Recipe,
    params: &'a serde_json::Value,
    model: &'a ModelInfo,
    data: &'a DatasetStats,
    eval: Option<&'a DatasetStats>,
    baseline: MemoryBaseline,
) -> ResolveInput<'a> {
    ResolveInput {
        // No fixture resolves a `distill` document; the one test that does
        // supplies its own teacher geometry.
        teacher: None,
        recipe,
        params,
        model,
        data,
        eval,
        data_format: retrograd_dataset::DataFormat::ChatJsonl,
        baseline,
        hardware: hardware(baseline.device_total.unwrap_or(0)),
        execution_profile: None,
        preflight: None,
        reward_protocol: None,
        server_budgets: (BudgetRequest::All, BudgetRequest::All),
        margin: MarginPolicy {
            fraction: 0.05,
            // No floor: a fixture's budget is an exact number, and a 256 MiB
            // floor would silently eat most of a deliberately small one.
            floor_bytes: 0,
        },
        calibrations: None,
        packing_measurements: None,
        reward_command: Vec::new(),
        root: PathBuf::from("."),
    }
}
