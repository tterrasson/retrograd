//! The resolved **trainable set**: what one run actually trains, resolved once
//! from a tensor inventory and read afterwards by selection, optimizer
//! allocation, checkpointing, reporting and the planner.
//!
//! Changing which parameters carry a gradient is a filter change; agreeing on
//! *which tensors those are* is a contract. Resolution happens here, in the
//! bottom crate, on an inventory read from GGUF metadata alone - no weights
//! allocated, no context created - so a frontend can answer "what would this
//! configuration train, and what would it cost" before anything is loaded.
//!
//! What this module does **not** do: mark anything. A resolved set is a
//! proposal about tensor identity; the runtime still has to confirm that every
//! entry it names came back carrying a gradient.

use std::collections::BTreeMap;
use std::fmt;

use crate::error::{Error, Result};

/// Schema version of [`TensorInventory`]. A reader that keys on a field must be
/// able to tell a v1 inventory from a later one rather than silently miss a
/// tensor family, so the version travels with the data instead of being implied
/// by the binary that produced it.
pub const TENSOR_INVENTORY_VERSION: u32 = 1;

/// Tensors that are never trainable, whatever the policy asks for.
///
/// Input embeddings: `llama_set_param` refuses them and `opt_init` never visits
/// them, so "selected" would mean "flagged and silently not trained".
/// Rotary constants: precomputed frequencies, not parameters - a gradient on
/// them is meaningless rather than expensive.
pub const ALWAYS_FROZEN: [&str; 2] = ["token_embd.weight", "rope_freqs.weight"];

/// The canonical name of the vocabulary projection, when the model carries one
/// of its own.
pub const OUTPUT_HEAD: &str = "output.weight";

/// The model-wide final norm. It follows `norms` rather than the block range,
/// and so does every other norm outside a block - `token_embd_norm.weight` is
/// one real models carry, and "the norms except the ones nobody listed" is not
/// a selection anyone asked for.
pub const OUTPUT_NORM: &str = "output_norm.weight";

// --- dtypes ------------------------------------------------------------------

/// The dtype of a stored tensor, to the precision selection needs.
///
/// Only the three float types a training path could plausibly carry are named;
/// everything else - every quantization, every index type - is `Other`, carried
/// by its ggml name so a refusal can quote it. Selection never needs to know
/// *which* quantization a tensor uses, only that it is not one it can train.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorDtype {
    F32,
    F16,
    BF16,
    Other(String),
}

impl TensorDtype {
    /// Parses a ggml type name (`"F32"`, `"Q4_K"`, ...) as it appears in
    /// `ggml_type_name`. Unknown names are preserved rather than rejected: an
    /// inventory records what the file says, and refusing is the resolver's job.
    pub fn from_ggml_name(name: &str) -> Self {
        match name.trim().to_ascii_uppercase().as_str() {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            _ => Self::Other(name.trim().to_string()),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
            Self::Other(name) => name,
        }
    }

    /// Whether a base tensor of this dtype may be marked trainable **today**.
    ///
    /// F32 only. F16 base training is gated on measured parity *and* long-run
    /// stability per backend, not on the existence of an F16 AdamW kernel;
    /// BF16 is separate work across loading, backward and update kernels.
    pub fn is_trainable_base(&self) -> bool {
        matches!(self, Self::F32)
    }
}

impl fmt::Display for TensorDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

// --- inventory ---------------------------------------------------------------

/// One tensor as the GGUF declares it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub name: String,
    /// ggml dimension order, `ne[0]` fastest-varying. Trailing dimensions are 1.
    pub ne: [i64; 4],
    pub dtype: TensorDtype,
    pub n_elements: u64,
    pub n_bytes: u64,
    /// Opaque identity of the allocation backing this tensor: two names for one
    /// allocation carry the same value. Deduplication keys on it rather than on
    /// the name, because "two updates to one allocation" is a property of the
    /// storage and not of how many ways it can be spelled.
    ///
    /// Comparable *within one inventory only* - the producer fills it from
    /// whatever identifies storage on its side, and nothing promises the same
    /// tensor gets the same value in another process. Never part of a manifest
    /// or a signature for that reason.
    pub storage_id: u64,
}

impl TensorDesc {
    /// `blk.<N>.` prefix, if this is a per-layer tensor.
    pub fn layer(&self) -> Option<u32> {
        let rest = self.name.strip_prefix("blk.")?;
        let (digits, _) = rest.split_once('.')?;
        digits.parse().ok()
    }

    /// The module stem: what is left after the `blk.<N>.` prefix and the
    /// `.weight` / `.bias` suffix. `blk.3.attn_q.weight` -> `attn_q`.
    /// A tensor with neither suffix has no stem.
    pub fn stem(&self) -> Option<&str> {
        let rest = match self.name.strip_prefix("blk.") {
            Some(rest) => rest.split_once('.').map(|(_, rest)| rest)?,
            None => self.name.as_str(),
        };
        rest.strip_suffix(".weight")
            .or_else(|| rest.strip_suffix(".bias"))
    }

    pub fn is_bias(&self) -> bool {
        self.name.ends_with(".bias")
    }

    /// Whether the stem names a normalization. Keys on the suffix rather than on
    /// a table, the same way LoRA target detection keys on the tensors present:
    /// a new architecture spelling its norms `*_norm` works without an edit.
    pub fn is_norm(&self) -> bool {
        self.stem()
            .is_some_and(|stem| stem == "norm" || stem.ends_with("_norm"))
    }
}

/// Every tensor a GGUF declares, read from its metadata without allocating a
/// single weight.
///
/// Versioned because it is a data contract: the planner sizes optimizer state
/// from it and a checkpoint records a signature derived from it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInventory {
    pub version: u32,
    pub architecture: String,
    /// Number of transformer blocks, from the `blk.<N>.` indices present.
    pub n_layer: u32,
    /// Whether the vocabulary projection reuses the input-embedding storage.
    /// When it does the file carries no `output.weight` of its own, so the head
    /// cannot be told from "absent" by looking at the tensor list alone.
    pub tied_embeddings: bool,
    pub tensors: Vec<TensorDesc>,
}

impl TensorInventory {
    /// Builds an inventory from descriptors, deriving `n_layer` from the block
    /// indices actually present rather than trusting a separate count.
    pub fn new(
        architecture: impl Into<String>,
        tied_embeddings: bool,
        tensors: Vec<TensorDesc>,
    ) -> Self {
        let n_layer = tensors
            .iter()
            .filter_map(TensorDesc::layer)
            .max()
            .map_or(0, |last| last + 1);
        Self {
            version: TENSOR_INVENTORY_VERSION,
            architecture: architecture.into(),
            n_layer,
            tied_embeddings,
            tensors,
        }
    }

    pub fn get(&self, name: &str) -> Option<&TensorDesc> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Refuses an inventory this build does not know how to read. Called before
    /// resolution rather than at construction, so an inventory can be carried
    /// across a boundary and validated where it is used.
    pub fn check_version(&self) -> Result<()> {
        if self.version == TENSOR_INVENTORY_VERSION {
            return Ok(());
        }
        Err(Error::runtime(format!(
            "tensor inventory has schema version {}, expected {TENSOR_INVENTORY_VERSION}",
            self.version
        )))
    }
}

// --- policy and selectors ----------------------------------------------------

/// Which family of parameters a run trains.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TrainablePolicy {
    /// LoRA A/B pairs, no base parameter. The existing trajectory, unchanged.
    #[default]
    Lora,
    /// Every supported eligible base tensor, with explicit exclusions.
    Full,
    /// Base modules/norms/biases intersected with a layer range.
    Partial,
    /// The A/B pairs plus explicitly selected base norms/biases.
    Hybrid,
}

impl TrainablePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Lora => "lora",
            Self::Full => "full",
            Self::Partial => "partial",
            Self::Hybrid => "hybrid",
        }
    }

    /// Whether this policy marks base tensors trainable. The predicate the
    /// fixed-reference guard branches on: any base mutation invalidates
    /// "disable the adapter and you have the original policy", including a
    /// norms-only hybrid.
    pub fn trains_base_weights(self) -> bool {
        !matches!(self, Self::Lora)
    }

    /// The integer the C `retro_train_config.trainable` carries. The runtime
    /// branches on it before a single tensor is named: it decides whether the
    /// weights are mapped read-only or loaded into owned writable buffers, and
    /// that decision is taken at model load, long before a resolved set exists.
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::Lora => 0,
            Self::Full => 1,
            Self::Partial => 2,
            Self::Hybrid => 3,
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "lora" => Ok(Self::Lora),
            "full" => Ok(Self::Full),
            "partial" => Ok(Self::Partial),
            "hybrid" => Ok(Self::Hybrid),
            other => Err(Error::config(format!(
                "training.trainable must be lora, full, partial or hybrid; got '{other}'"
            ))),
        }
    }
}

impl fmt::Display for TrainablePolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Which transformer blocks a partial selection covers.
///
/// `blk.{12,13}.*` is explanatory notation and not a supported wildcard: a range
/// resolves to individual layer indices here, and those indices are what the
/// expansion below matches against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LayerRange {
    #[default]
    All,
    /// The last `n` blocks. Bounds-checked against the model's own count.
    Last(u32),
    /// Inclusive on both ends, as `12..15` reads in the document.
    Inclusive { first: u32, last: u32 },
}

impl LayerRange {
    /// Parses `"all"`, `"last:4"` or `"12..15"`.
    pub fn parse(value: &str) -> Result<Self> {
        let value = value.trim();
        if value.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        if let Some(count) = value.strip_prefix("last:") {
            let count: u32 = count.trim().parse().map_err(|_| {
                Error::config(format!("trainable.layers: '{value}' is not 'last:<count>'"))
            })?;
            if count == 0 {
                return Err(Error::config(
                    "trainable.layers = 'last:0' selects no layer; omit the section or widen it",
                ));
            }
            return Ok(Self::Last(count));
        }
        if let Some((first, last)) = value.split_once("..") {
            let parse = |part: &str, which: &str| -> Result<u32> {
                part.trim().parse().map_err(|_| {
                    Error::config(format!(
                        "trainable.layers: '{value}' has a non-numeric {which} bound"
                    ))
                })
            };
            let first = parse(first, "lower")?;
            let last = parse(last, "upper")?;
            if first > last {
                return Err(Error::config(format!(
                    "trainable.layers = '{value}' is empty: the range is inclusive, \
                     so the lower bound must not exceed the upper one"
                )));
            }
            return Ok(Self::Inclusive { first, last });
        }
        Err(Error::config(format!(
            "trainable.layers must be 'all', 'last:<count>' or '<first>..<last>'; got '{value}'"
        )))
    }

    /// The selected indices for a model of `n_layer` blocks, bounds-checked.
    pub fn resolve(self, n_layer: u32) -> Result<Vec<u32>> {
        if n_layer == 0 {
            return Err(Error::config(
                "the model declares no transformer block, so no layer range can be selected",
            ));
        }
        let (first, last) = match self {
            Self::All => (0, n_layer - 1),
            Self::Last(count) => (n_layer.saturating_sub(count), n_layer - 1),
            Self::Inclusive { first, last } => {
                if last >= n_layer {
                    return Err(Error::config(format!(
                        "trainable.layers = '{first}..{last}' is out of range: \
                         the model has {n_layer} blocks, so the last index is {}",
                        n_layer - 1
                    )));
                }
                (first, last)
            }
        };
        Ok((first..=last).collect())
    }
}

impl fmt::Display for LayerRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => f.write_str("all"),
            Self::Last(count) => write!(f, "last:{count}"),
            Self::Inclusive { first, last } => write!(f, "{first}..{last}"),
        }
    }
}

/// What a `partial` or `hybrid` document asks for inside the selected blocks.
///
/// Every field is off by default: an omitted selector selects nothing, and a
/// partial policy with nothing selected is refused rather than silently
/// resolved to the empty set.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainableSelector {
    pub layers: LayerRange,
    /// Module aliases (`attn`, `ffn`), individual stems (`attn_q`, `ffn_up`) or
    /// explicit tensor patterns. Never norms: those follow `norms`.
    pub modules: Vec<String>,
    /// Every normalization: the ones inside the selected blocks, and the
    /// model-wide ones - `output_norm`, `token_embd_norm` - which are not in
    /// any block and therefore do not follow the layer range.
    pub norms: bool,
    /// Every `.bias` inside the selected blocks, as its own family.
    pub biases: bool,
    /// The vocabulary projection, independent of the block range. Refused when
    /// the model ties it to the input embedding.
    pub output_head: bool,
}

impl TrainableSelector {
    pub fn is_empty(&self) -> bool {
        self.modules.is_empty() && !self.norms && !self.biases && !self.output_head
    }
}

/// The resolved answer to "what does this run train, and with which optimizer".
///
/// A *policy and a selection*, not a resolved tensor list: turning it into
/// actual tensors needs the model's inventory, which a document does not have.
/// The engine resolves [`resolve_base`] against this once the model is open.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainableRunConfig {
    pub policy: TrainablePolicy,
    /// Empty for `lora`, where a base selector is refused rather than ignored.
    pub selector: TrainableSelector,
    pub optimizer: crate::optimizer::OptimizerKind,
}

/// Module aliases, expanded to the stems a GGUF actually spells.
///
/// Keyed on the tensors present rather than on the architecture name, the same
/// way LoRA target detection is: a model spelling its attention `attn_qkv`
/// resolves under `attn` without a per-architecture table.
fn expand_module_alias(alias: &str) -> Option<&'static [&'static str]> {
    match alias {
        "attn" | "attention" => Some(&["attn_q", "attn_k", "attn_v", "attn_qkv", "attn_output"]),
        "ffn" | "mlp" => Some(&["ffn_up", "ffn_down", "ffn_gate"]),
        _ => None,
    }
}

// --- resolved set ------------------------------------------------------------

/// What a resolved entry is, in the optimizer's terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorRole {
    LoraA,
    LoraB,
    Base,
}

impl TensorRole {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::LoraA => "lora_a",
            Self::LoraB => "lora_b",
            Self::Base => "base",
        }
    }
}

/// One resolved trainable tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainableEntry {
    pub name: String,
    pub role: TensorRole,
    pub ne: [i64; 4],
    pub dtype: TensorDtype,
    pub n_elements: u64,
    pub n_bytes: u64,
    pub storage_id: u64,
}

/// Why an otherwise eligible tensor is not in the set.
///
/// Only tensors the policy *would* have taken appear here. A tensor outside a
/// partial selection is simply not selected, and listing every one of those
/// would bury the four that matter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExclusionReason {
    /// `token_embd.weight`: the runtime refuses to mark it and never visits it.
    InputEmbedding,
    /// `rope_freqs.weight`: a precomputed constant, not a parameter.
    RotaryConstant,
    /// The head shares the input-embedding storage; training it would train the
    /// embedding through an alias.
    TiedOutputHead,
    /// A dtype no base training path supports yet.
    UnsupportedDtype(TensorDtype),
    /// A second name for storage already in the set.
    DuplicateStorage { of: String },
}

impl fmt::Display for ExclusionReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputEmbedding => f.write_str("input embedding, never trainable"),
            Self::RotaryConstant => f.write_str("rotary constant, not a parameter"),
            Self::TiedOutputHead => {
                f.write_str("output head tied to the input embedding, kept frozen")
            }
            Self::UnsupportedDtype(dtype) => {
                write!(f, "dtype {dtype} is not supported for base training")
            }
            Self::DuplicateStorage { of } => write!(f, "shares storage with '{of}'"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrainableExclusion {
    pub name: String,
    pub reason: ExclusionReason,
}

/// The resolved trainable set: one answer, read by selection, optimizer
/// allocation, checkpointing, reporting and the planner.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrainableSet {
    pub policy: TrainablePolicy,
    /// LoRA entries first, in their registration order, then base entries in
    /// canonical order. The LoRA half must not be reordered: its optimizer
    /// updates are what the existing trajectory parity tests compare.
    pub entries: Vec<TrainableEntry>,
    /// Eligible tensors the policy did not take, with the reason, so a user can
    /// see that `full` means "all *supported* eligible tensors".
    pub exclusions: Vec<TrainableExclusion>,
}

impl TrainableSet {
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn base_entries(&self) -> impl Iterator<Item = &TrainableEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.role == TensorRole::Base)
    }

    pub fn n_parameters(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.n_elements)
            .fold(0, u64::saturating_add)
    }

    /// Bytes the trainable parameters occupy *as stored*.
    ///
    /// For base entries this is a subset of the loaded model weights and must
    /// not be added to them; [`Self::parameter_bytes_on_top`] is the figure a
    /// budget adds. Keeping both is the point: one number answers two
    /// different questions.
    pub fn parameter_bytes(&self) -> u64 {
        self.entries
            .iter()
            .map(|entry| entry.n_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Parameter bytes a device budget must *add* to the loaded model weights:
    /// adapter factors, which are allocated on top, and never base tensors,
    /// which are already counted inside them.
    pub fn parameter_bytes_on_top(&self) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.role != TensorRole::Base)
            .map(|entry| entry.n_bytes)
            .fold(0, u64::saturating_add)
    }

    /// Persistent F32 gradient bytes.
    ///
    /// Always F32 and always persistent: the llama training path uses dynamic
    /// graphs, so `ggml-opt`'s accumulator condition holds even at
    /// `opt_period == 1` and one micro-batch does not remove this term.
    pub fn gradient_bytes(&self) -> u64 {
        self.n_parameters().saturating_mul(4)
    }

    /// The canonical manifest: one line per entry, stable across runs, and the
    /// text a checkpoint signature is taken over.
    ///
    /// Separately canonicalized on purpose - the `entries` order carries the
    /// optimizer's update order, which must not be disturbed, while identity
    /// comparison needs an order that does not depend on it.
    pub fn manifest_lines(&self) -> Vec<String> {
        let mut lines: Vec<String> = self
            .entries
            .iter()
            .map(|entry| {
                format!(
                    "{}\t{}\t{}\t{}x{}x{}x{}\t{}",
                    entry.name,
                    entry.role.as_str(),
                    entry.dtype,
                    entry.ne[0],
                    entry.ne[1],
                    entry.ne[2],
                    entry.ne[3],
                    entry.n_elements,
                )
            })
            .collect();
        lines.sort();
        lines
    }
}

// --- resolution --------------------------------------------------------------

/// Resolves the base half of the trainable set.
///
/// LoRA entries are not produced here. The A/B factors do not exist in the
/// GGUF: they are created or loaded by the runtime, which appends them in its
/// own registration order. This function answers the question the inventory
/// *can* answer, and answers it deterministically.
pub fn resolve_base(
    inventory: &TensorInventory,
    policy: TrainablePolicy,
    selector: &TrainableSelector,
) -> Result<TrainableSet> {
    inventory.check_version()?;
    let mut set = TrainableSet {
        policy,
        ..Default::default()
    };
    match policy {
        TrainablePolicy::Lora => return Ok(set),
        TrainablePolicy::Full => resolve_full(inventory, &mut set)?,
        TrainablePolicy::Partial | TrainablePolicy::Hybrid => {
            resolve_selected(inventory, selector, &mut set)?;
        }
    }
    if set.entries.is_empty() {
        return Err(Error::config(format!(
            "trainable = '{policy}' resolved to no tensor; \
             widen the selection or check the model's tensor names"
        )));
    }
    Ok(set)
}

/// Every supported eligible base tensor.
///
/// A dtype refusal here is loud on purpose: silently keeping the F32 norms of a
/// Q4 model and calling that "full" would report a trainable set two orders of
/// magnitude smaller than the word promises.
fn resolve_full(inventory: &TensorInventory, set: &mut TrainableSet) -> Result<()> {
    let mut unsupported: Vec<(&str, &TensorDtype)> = Vec::new();
    let mut candidates: Vec<&TensorDesc> = Vec::new();
    for tensor in &inventory.tensors {
        match frozen_reason(inventory, tensor) {
            Some(reason) => set.exclusions.push(TrainableExclusion {
                name: tensor.name.clone(),
                reason,
            }),
            None if !tensor.dtype.is_trainable_base() => {
                unsupported.push((&tensor.name, &tensor.dtype));
            }
            None => candidates.push(tensor),
        }
    }
    if !unsupported.is_empty() {
        return Err(unsupported_dtype_error("full", &unsupported));
    }
    push_deduplicated(candidates, set);
    Ok(())
}

/// The union of the requested modules, norms and biases, intersected with the
/// layer range, plus the two tensors that live outside every block.
fn resolve_selected(
    inventory: &TensorInventory,
    selector: &TrainableSelector,
    set: &mut TrainableSet,
) -> Result<()> {
    if selector.is_empty() {
        return Err(Error::config(
            "trainable = 'partial' needs an explicit selection: \
             set at least one of [trainable].modules, .norms, .biases or .output_head",
        ));
    }
    let layers = selector.layers.resolve(inventory.n_layer)?;
    let in_range = |tensor: &TensorDesc| match tensor.layer() {
        Some(layer) => layers.contains(&layer),
        None => false,
    };

    // The head is checked before the scan, not during it: "you asked for a
    // tied head" is a different sentence from "this tensor is frozen", and the
    // user needs the one that explains the alias.
    if selector.output_head {
        head_is_selectable(inventory)?;
    }

    // One request, one verdict. An alias expands to the stems several
    // architectures spell their attention with, and a model carrying `attn_q`
    // and `attn_v` but no `attn_qkv` is the normal case - so the "matched
    // nothing" check belongs to what the user wrote, never to the expansion.
    // An explicit stem is its own one-element expansion, which is what keeps
    // `modules = ["attn_nope"]` a typo rather than a silently smaller set.
    struct Request {
        written: String,
        stems: Vec<String>,
        patterns: Vec<String>,
        matched: bool,
    }
    let mut requests: Vec<Request> = Vec::new();
    for module in &selector.modules {
        let module = module.trim();
        if module.is_empty() {
            return Err(Error::config("trainable.modules contains an empty entry"));
        }
        let (stems, patterns) =
            if module.contains('*') || module.contains(".weight") || module.contains(".bias") {
                (Vec::new(), vec![module.to_string()])
            } else {
                let stems = match expand_module_alias(module) {
                    Some(expanded) => expanded.iter().map(|stem| (*stem).to_string()).collect(),
                    None => vec![module.to_string()],
                };
                (stems, Vec::new())
            };
        requests.push(Request {
            written: module.to_string(),
            stems,
            patterns,
            matched: false,
        });
    }

    let mut selected: Vec<&TensorDesc> = Vec::new();
    for tensor in &inventory.tensors {
        let mut wanted = false;
        let in_block_range = in_range(tensor);

        for request in &mut requests {
            let hit = (in_block_range
                && !tensor.is_bias()
                && !tensor.is_norm()
                && tensor
                    .stem()
                    .is_some_and(|stem| request.stems.iter().any(|want| want == stem)))
                || (tensor.layer().is_none() || in_block_range)
                    && request
                        .patterns
                        .iter()
                        .any(|pattern| wildcard_match(pattern, &tensor.name));
            if hit {
                request.matched = true;
                wanted = true;
            }
        }
        if in_block_range {
            if selector.norms && tensor.is_norm() && !tensor.is_bias() {
                wanted = true;
            }
            if selector.biases && tensor.is_bias() {
                wanted = true;
            }
        }
        // Outside every block, and therefore outside the layer range: the
        // model-wide norms follow `norms`, the head follows `output_head`.
        // Keyed on the shape of the name, not on a list of two: a model that
        // normalizes its embeddings carries `token_embd_norm.weight`, and a
        // `norms = true` that quietly skipped it would be selecting "the norms
        // someone remembered to enumerate".
        // `layer().is_none()`, not `!in_block_range`: a block norm the range
        // excluded is excluded, not promoted to model-wide.
        if selector.norms && tensor.layer().is_none() && tensor.is_norm() && !tensor.is_bias() {
            wanted = true;
        }
        if selector.output_head && tensor.name == OUTPUT_HEAD {
            wanted = true;
        }
        if !wanted {
            continue;
        }
        match frozen_reason(inventory, tensor) {
            Some(reason) => {
                return Err(Error::config(format!(
                    "trainable selection names '{}', which cannot be trained: {reason}",
                    tensor.name
                )));
            }
            None => selected.push(tensor),
        }
    }

    let unmatched: Vec<&str> = requests
        .iter()
        .filter(|request| !request.matched)
        .map(|request| request.written.as_str())
        .collect();
    if !unmatched.is_empty() {
        return Err(Error::config(format!(
            "trainable.modules selects nothing in the chosen layers: {}. \
             Check the spelling against the model's own tensor names",
            unmatched.join(", ")
        )));
    }

    let unsupported: Vec<(&str, &TensorDtype)> = selected
        .iter()
        .filter(|tensor| !tensor.dtype.is_trainable_base())
        .map(|tensor| (tensor.name.as_str(), &tensor.dtype))
        .collect();
    if !unsupported.is_empty() {
        // Only the *selected* tensors are checked: a Q4 model may still carry
        // trainable F32 norms, and refusing it on its dominant dtype would
        // refuse exactly the run partial training exists for.
        return Err(unsupported_dtype_error(set.policy.as_str(), &unsupported));
    }
    push_deduplicated(selected, set);
    Ok(())
}

/// Why this tensor can never be trained, or `None` if it can.
fn frozen_reason(inventory: &TensorInventory, tensor: &TensorDesc) -> Option<ExclusionReason> {
    // Freezing is a property of storage, not only of its canonical name.
    // Unknown identities (zero) must never make unrelated tensors aliases.
    let aliases = |name: &str| {
        tensor.name == name
            || (tensor.storage_id != 0
                && inventory
                    .get(name)
                    .is_some_and(|frozen| frozen.storage_id == tensor.storage_id))
    };
    if tensor.name == OUTPUT_HEAD && (inventory.tied_embeddings || aliases(ALWAYS_FROZEN[0])) {
        return Some(ExclusionReason::TiedOutputHead);
    }
    if aliases(ALWAYS_FROZEN[0]) {
        return Some(ExclusionReason::InputEmbedding);
    }
    if aliases(ALWAYS_FROZEN[1]) {
        return Some(ExclusionReason::RotaryConstant);
    }
    None
}

/// A tied head has no storage of its own; asking for it is asking to train the
/// input embedding through an alias, which is an error rather than a silent
/// no-op.
fn head_is_selectable(inventory: &TensorInventory) -> Result<()> {
    if inventory.tied_embeddings {
        return Err(Error::config(
            "trainable.output_head = true, but this model ties its vocabulary \
             projection to token_embd.weight: training the head would train the \
             input embedding through an alias. Keep the head frozen",
        ));
    }
    if inventory.get(OUTPUT_HEAD).is_none() {
        return Err(Error::config(format!(
            "trainable.output_head = true, but the model declares no '{OUTPUT_HEAD}'"
        )));
    }
    Ok(())
}

/// Appends candidates in canonical order, dropping any whose storage is already
/// in the set.
///
/// Two updates to one allocation is not a slow set, it is a wrong one: the
/// second update would read the weights the first had already moved.
fn push_deduplicated(mut candidates: Vec<&TensorDesc>, set: &mut TrainableSet) {
    candidates.sort_by_key(|tensor| canonical_key(tensor));
    let mut seen: BTreeMap<u64, String> = BTreeMap::new();
    for tensor in candidates {
        // Zero is what a descriptor carries when the producer records no
        // storage identity at all; treating every such tensor as sharing one
        // allocation would collapse the whole set into its first entry.
        if tensor.storage_id != 0
            && let Some(owner) = seen.get(&tensor.storage_id)
        {
            set.exclusions.push(TrainableExclusion {
                name: tensor.name.clone(),
                reason: ExclusionReason::DuplicateStorage { of: owner.clone() },
            });
            continue;
        }
        seen.insert(tensor.storage_id, tensor.name.clone());
        set.entries.push(TrainableEntry {
            name: tensor.name.clone(),
            role: TensorRole::Base,
            ne: tensor.ne,
            dtype: tensor.dtype.clone(),
            n_elements: tensor.n_elements,
            n_bytes: tensor.n_bytes,
            storage_id: tensor.storage_id,
        });
    }
}

/// Layer-major, then name. Sorting on the raw name alone would order `blk.10`
/// before `blk.2`, which makes a golden test read as if the set had changed
/// whenever a model gained a layer.
fn canonical_key(tensor: &TensorDesc) -> (u32, u32, String) {
    match tensor.layer() {
        Some(layer) => (0, layer, tensor.name.clone()),
        None => (1, 0, tensor.name.clone()),
    }
}

/// Names the tensors and their dtypes, capped: a full-policy refusal on a
/// quantized model would otherwise list every matmul in the file.
fn unsupported_dtype_error(policy: &str, unsupported: &[(&str, &TensorDtype)]) -> Error {
    const SHOWN: usize = 6;
    let listed: Vec<String> = unsupported
        .iter()
        .take(SHOWN)
        .map(|(name, dtype)| format!("{name} ({dtype})"))
        .collect();
    let rest = unsupported.len().saturating_sub(listed.len());
    let tail = if rest > 0 {
        format!(" and {rest} more")
    } else {
        String::new()
    };
    Error::config(format!(
        "trainable = '{policy}' selects {} tensor(s) whose dtype base training does not \
         support yet: {}{tail}. Base training is F32-only today; quantized weights are \
         out of scope",
        unsupported.len(),
        listed.join(", "),
    ))
}

/// The wildcard grammar the LoRA targets already use: `*` matches any run of
/// characters, everything else is literal.
fn wildcard_match(pattern: &str, value: &str) -> bool {
    let pattern: Vec<char> = pattern.chars().collect();
    let value: Vec<char> = value.chars().collect();
    fn walk(pattern: &[char], value: &[char]) -> bool {
        match pattern.first() {
            None => value.is_empty(),
            Some('*') => {
                let rest = &pattern[1..];
                (0..=value.len()).any(|split| walk(rest, &value[split..]))
            }
            Some(head) => match value.first() {
                Some(first) if first == head => walk(&pattern[1..], &value[1..]),
                _ => false,
            },
        }
    }
    walk(&pattern, &value)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, ne: [i64; 4], dtype: TensorDtype, offset: u64) -> TensorDesc {
        let n_elements = ne.iter().product::<i64>() as u64;
        let element_bytes = match dtype {
            TensorDtype::F32 => 4,
            TensorDtype::F16 | TensorDtype::BF16 => 2,
            TensorDtype::Other(_) => 1,
        };
        TensorDesc {
            name: name.to_string(),
            ne,
            dtype,
            n_elements,
            n_bytes: n_elements * element_bytes,
            storage_id: offset,
        }
    }

    /// Two blocks of a plausible transformer, with a norm, a bias and an
    /// untied head. Offsets are distinct, as a real GGUF's are.
    fn inventory() -> TensorInventory {
        let mut tensors = vec![
            tensor("token_embd.weight", [8, 32, 1, 1], TensorDtype::F16, 100),
            tensor("rope_freqs.weight", [4, 1, 1, 1], TensorDtype::F32, 200),
        ];
        let mut offset = 300;
        for layer in 0..2 {
            for stem in ["attn_q", "attn_v", "ffn_up", "ffn_down"] {
                tensors.push(tensor(
                    &format!("blk.{layer}.{stem}.weight"),
                    [8, 8, 1, 1],
                    TensorDtype::F32,
                    offset,
                ));
                offset += 100;
            }
            tensors.push(tensor(
                &format!("blk.{layer}.attn_norm.weight"),
                [8, 1, 1, 1],
                TensorDtype::F32,
                offset,
            ));
            offset += 100;
            tensors.push(tensor(
                &format!("blk.{layer}.attn_q.bias"),
                [8, 1, 1, 1],
                TensorDtype::F32,
                offset,
            ));
            offset += 100;
        }
        tensors.push(tensor(
            OUTPUT_NORM,
            [8, 1, 1, 1],
            TensorDtype::F32,
            offset + 100,
        ));
        // A real hybrid carries this one (lfm2 does); it is a norm and it is in
        // no block, which is exactly the case the rule above exists for.
        tensors.push(tensor(
            "token_embd_norm.weight",
            [8, 1, 1, 1],
            TensorDtype::F32,
            offset + 150,
        ));
        tensors.push(tensor(
            OUTPUT_HEAD,
            [8, 32, 1, 1],
            TensorDtype::F32,
            offset + 200,
        ));
        TensorInventory::new("llama", false, tensors)
    }

    fn names(set: &TrainableSet) -> Vec<&str> {
        set.entries.iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn the_layer_count_comes_from_the_blocks_actually_present() {
        assert_eq!(inventory().n_layer, 2);
    }

    #[test]
    fn lora_resolves_to_no_base_tensor() {
        let set = resolve_base(
            &inventory(),
            TrainablePolicy::Lora,
            &TrainableSelector::default(),
        )
        .expect("lora resolves");
        assert!(set.is_empty());
        assert!(set.exclusions.is_empty());
    }

    #[test]
    fn full_takes_every_supported_tensor_and_names_what_it_left_out() {
        let set = resolve_base(
            &inventory(),
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .expect("full resolves");
        // The two always-frozen tensors are excluded, and visibly so.
        assert_eq!(
            set.exclusions
                .iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>(),
            ["token_embd.weight", "rope_freqs.weight"]
        );
        assert!(!names(&set).contains(&"token_embd.weight"));
        assert!(!names(&set).contains(&"rope_freqs.weight"));
        assert!(names(&set).contains(&OUTPUT_HEAD));
        // Layer-major: blk.0 before blk.1, model-wide tensors last.
        assert_eq!(names(&set)[0], "blk.0.attn_norm.weight");
        assert_eq!(names(&set).last(), Some(&"token_embd_norm.weight"));
    }

    #[test]
    fn full_refuses_a_quantized_model_rather_than_silently_training_its_norms() {
        let mut inventory = inventory();
        for tensor in &mut inventory.tensors {
            if tensor.name.ends_with("attn_q.weight") {
                tensor.dtype = TensorDtype::Other("Q4_K".into());
            }
        }
        let error = resolve_base(
            &inventory,
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .expect_err("a quantized weight is not trainable");
        assert!(error.is_user_error());
        let message = error.to_string();
        assert!(message.contains("Q4_K"), "{message}");
        assert!(message.contains("blk.0.attn_q.weight"), "{message}");
    }

    #[test]
    fn partial_validates_the_selected_tensors_not_the_dominant_dtype() {
        // A Q4 model whose norms are still F32: norms-only must resolve.
        let mut inventory = inventory();
        for tensor in &mut inventory.tensors {
            if tensor.name.ends_with(".weight") && !tensor.is_norm() {
                tensor.dtype = TensorDtype::Other("Q4_K".into());
            }
        }
        let selector = TrainableSelector {
            norms: true,
            ..Default::default()
        };
        let set = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
            .expect("F32 norms of a quantized model are trainable");
        assert_eq!(
            names(&set),
            [
                "blk.0.attn_norm.weight",
                "blk.1.attn_norm.weight",
                OUTPUT_NORM,
                "token_embd_norm.weight",
            ]
        );
    }

    #[test]
    fn the_layer_range_is_inclusive_and_bounds_checked() {
        let selector = TrainableSelector {
            layers: LayerRange::Inclusive { first: 1, last: 1 },
            modules: vec!["attn".into()],
            ..Default::default()
        };
        let set = resolve_base(&inventory(), TrainablePolicy::Partial, &selector)
            .expect("one layer resolves");
        assert_eq!(names(&set), ["blk.1.attn_q.weight", "blk.1.attn_v.weight"]);

        let out_of_range = TrainableSelector {
            layers: LayerRange::Inclusive { first: 0, last: 7 },
            modules: vec!["attn".into()],
            ..Default::default()
        };
        let error = resolve_base(&inventory(), TrainablePolicy::Partial, &out_of_range)
            .expect_err("layer 7 does not exist");
        assert!(error.to_string().contains("the last index is 1"));
    }

    #[test]
    fn last_k_clamps_instead_of_underflowing() {
        assert_eq!(LayerRange::Last(9).resolve(2).unwrap(), vec![0, 1]);
        assert_eq!(LayerRange::Last(1).resolve(2).unwrap(), vec![1]);
    }

    #[test]
    fn the_global_norm_follows_norms_and_ignores_the_block_range() {
        let selector = TrainableSelector {
            layers: LayerRange::Last(1),
            norms: true,
            ..Default::default()
        };
        let set = resolve_base(&inventory(), TrainablePolicy::Partial, &selector).unwrap();
        assert_eq!(
            names(&set),
            [
                "blk.1.attn_norm.weight",
                OUTPUT_NORM,
                "token_embd_norm.weight"
            ]
        );
    }

    #[test]
    fn biases_are_their_own_family_and_resolve_real_tensor_names() {
        let selector = TrainableSelector {
            biases: true,
            ..Default::default()
        };
        let set = resolve_base(&inventory(), TrainablePolicy::Partial, &selector).unwrap();
        assert_eq!(names(&set), ["blk.0.attn_q.bias", "blk.1.attn_q.bias"]);
    }

    #[test]
    fn a_module_stem_that_matches_nothing_is_a_typo_rather_than_a_smaller_set() {
        let selector = TrainableSelector {
            modules: vec!["attn".into(), "attn_nope".into()],
            ..Default::default()
        };
        let error = resolve_base(&inventory(), TrainablePolicy::Partial, &selector)
            .expect_err("an unmatched stem is refused");
        assert!(error.to_string().contains("attn_nope"), "{error}");
    }

    #[test]
    fn an_empty_partial_selection_is_refused() {
        let error = resolve_base(
            &inventory(),
            TrainablePolicy::Partial,
            &TrainableSelector::default(),
        )
        .expect_err("partial needs an explicit selection");
        assert!(error.to_string().contains("explicit selection"), "{error}");
    }

    #[test]
    fn a_tied_head_stays_frozen_and_asking_for_it_is_an_error() {
        let mut inventory = inventory();
        inventory.tied_embeddings = true;
        let full = resolve_base(
            &inventory,
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .unwrap();
        assert!(!names(&full).contains(&OUTPUT_HEAD));
        assert!(
            full.exclusions
                .iter()
                .any(|e| e.reason == ExclusionReason::TiedOutputHead)
        );

        let selector = TrainableSelector {
            output_head: true,
            ..Default::default()
        };
        let error = resolve_base(&inventory, TrainablePolicy::Partial, &selector)
            .expect_err("an explicit tied head is an error");
        assert!(error.to_string().contains("alias"), "{error}");
    }

    #[test]
    fn two_names_for_one_allocation_produce_one_update() {
        let mut inventory = inventory();
        let shared = inventory
            .get("blk.0.attn_q.weight")
            .expect("fixture")
            .storage_id;
        inventory.tensors.push(tensor(
            "blk.0.attn_q_alias.weight",
            [8, 8, 1, 1],
            TensorDtype::F32,
            shared,
        ));
        let set = resolve_base(
            &inventory,
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .unwrap();
        assert_eq!(
            set.entries
                .iter()
                .filter(|entry| entry.storage_id == shared)
                .count(),
            1
        );
        assert!(set.exclusions.iter().any(|exclusion| matches!(
            &exclusion.reason,
            ExclusionReason::DuplicateStorage { of } if of == "blk.0.attn_q.weight"
        )));
    }

    #[test]
    fn a_pattern_is_accepted_alongside_the_aliases() {
        let selector = TrainableSelector {
            modules: vec!["blk.*.ffn_down.weight".into()],
            ..Default::default()
        };
        let set = resolve_base(&inventory(), TrainablePolicy::Partial, &selector).unwrap();
        assert_eq!(
            names(&set),
            ["blk.0.ffn_down.weight", "blk.1.ffn_down.weight"]
        );
    }

    #[test]
    fn explicit_patterns_respect_the_layer_range() {
        let mut selector = TrainableSelector {
            layers: LayerRange::Last(1),
            modules: vec!["blk.*.ffn_down.weight".into()],
            ..Default::default()
        };
        let set = resolve_base(&inventory(), TrainablePolicy::Partial, &selector).unwrap();
        assert_eq!(names(&set), ["blk.1.ffn_down.weight"]);

        selector.modules = vec!["blk.0.ffn_down.weight".into()];
        let error = resolve_base(&inventory(), TrainablePolicy::Partial, &selector).unwrap_err();
        assert!(error.to_string().contains("selects nothing"), "{error}");
    }

    #[test]
    fn aliases_of_frozen_storage_stay_frozen() {
        let mut inventory = inventory();
        for (source, alias) in [
            ("token_embd.weight", "blk.0.embedding_alias.weight"),
            ("rope_freqs.weight", "blk.0.rotary_alias.weight"),
        ] {
            let mut tensor = inventory.get(source).unwrap().clone();
            tensor.name = alias.into();
            inventory.tensors.push(tensor);
        }
        let set = resolve_base(
            &inventory,
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .expect("even an unsupported embedding dtype is excluded through its alias");
        assert!(!names(&set).iter().any(|name| name.contains("alias")));
        for alias in ["blk.0.embedding_alias.weight", "blk.0.rotary_alias.weight"] {
            let selector = TrainableSelector {
                modules: vec![alias.into()],
                ..Default::default()
            };
            let error = resolve_base(&inventory, TrainablePolicy::Partial, &selector).unwrap_err();
            assert!(error.to_string().contains("cannot be trained"), "{error}");
        }
    }

    #[test]
    fn the_manifest_is_stable_and_independent_of_the_update_order() {
        let set = resolve_base(
            &inventory(),
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .unwrap();
        let first = set.manifest_lines();
        let mut reordered = set.clone();
        reordered.entries.reverse();
        assert_eq!(first, reordered.manifest_lines());
        assert!(first[0].contains('\t'));
    }

    #[test]
    fn base_parameter_bytes_are_never_added_on_top_of_the_model_weights() {
        let set = resolve_base(
            &inventory(),
            TrainablePolicy::Full,
            &TrainableSelector::default(),
        )
        .unwrap();
        assert!(set.parameter_bytes() > 0);
        assert_eq!(set.parameter_bytes_on_top(), 0);
        assert_eq!(set.gradient_bytes(), set.n_parameters() * 4);
    }

    #[test]
    fn layer_ranges_parse_the_three_documented_spellings() {
        assert_eq!(LayerRange::parse("all").unwrap(), LayerRange::All);
        assert_eq!(LayerRange::parse(" last:4 ").unwrap(), LayerRange::Last(4));
        assert_eq!(
            LayerRange::parse("12..15").unwrap(),
            LayerRange::Inclusive {
                first: 12,
                last: 15
            }
        );
        assert!(LayerRange::parse("15..12").is_err());
        assert!(LayerRange::parse("last:0").is_err());
        assert!(LayerRange::parse("blk.{12,13}.*").is_err());
    }

    #[test]
    fn policies_round_trip_through_their_document_spelling() {
        for policy in [
            TrainablePolicy::Lora,
            TrainablePolicy::Full,
            TrainablePolicy::Partial,
            TrainablePolicy::Hybrid,
        ] {
            assert_eq!(TrainablePolicy::parse(policy.as_str()).unwrap(), policy);
        }
        assert!(TrainablePolicy::parse("everything").is_err());
        assert!(!TrainablePolicy::Lora.trains_base_weights());
        assert!(TrainablePolicy::Hybrid.trains_base_weights());
    }
}
