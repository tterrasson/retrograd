//! Which optimizer a run uses, and what persistent state that costs.
//!
//! The half of the optimizer contract that can be decided before a graph
//! exists: the choice itself, its per-parameter policy, its **slot
//! definitions**, and the state-bytes formula the planner needs. The
//! executable half (the slot initializer and `build_step`) lives in the
//! runtime, because it needs a kernel.
//!
//! [`OptimizerPlan`] declares the slot table from the resolved trainable set
//! before the graph exists, so it can be compared against the table the
//! runtime actually allocates after `llama_opt_init`.
//!
//! Nothing here allocates. `state_bytes` is a storage formula, not a fit
//! guarantee: backend alignment, padding, the AdamW fallback of an ineligible
//! tensor and the optimizer's own scratch are added by the caller that knows
//! them.

use std::fmt;

use crate::error::{Error, Result};
use crate::trainable::{TensorDtype, TensorRole, TrainableEntry, TrainableSet};

/// The optimizers a document may name.
///
/// `AdamW` is the default and stays the default: an experimental optimizer does
/// not become a recommended fine-tuning default because its memory formula is
/// attractive.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizerKind {
    #[default]
    AdamW,
    /// llama.cpp's existing SGD semantics: no persistent state, but still a
    /// step counter and a schedule. "No slots" and "not initialized" are
    /// different states, and a resume must not confuse them.
    Sgd,
    /// Orthogonalized momentum on eligible hidden base matrices, AdamW
    /// everywhere else. Not implemented yet.
    Muon,
    /// Fixed-block, uniform-codebook Gefen. Not implemented yet.
    Gefen,
}

impl OptimizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
            Self::Muon => "muon",
            Self::Gefen => "gefen",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "adamw" | "adam_w" => Ok(Self::AdamW),
            "sgd" => Ok(Self::Sgd),
            "muon" => Ok(Self::Muon),
            "gefen" => Ok(Self::Gefen),
            other => Err(Error::config(format!(
                "training.optimizer must be adamw, sgd, muon or gefen; got '{other}'"
            ))),
        }
    }

    /// Whether this build can actually run it.
    ///
    /// A name the schema parses but the runtime cannot honour must be refused
    /// at load time: accepting a name and silently running AdamW would publish
    /// a trajectory nobody asked for, and a checkpoint that records the wrong
    /// optimizer.
    ///
    /// Muon and Gefen have no kernels here at all; AdamW and SGD are the two
    /// the update step can build, and the choice reaches it through
    /// [`Self::as_ffi`].
    pub fn is_implemented(self) -> bool {
        matches!(self, Self::AdamW | Self::Sgd)
    }

    /// The integer the C `retro_train_config.optimizer` carries, and with it
    /// `ggml_opt_optimizer_type`: `0` AdamW, `1` SGD. Refusing to widen past
    /// what [`Self::is_implemented`] admits is deliberate - a value the runtime
    /// would read as AdamW is exactly the silent substitution that predicate
    /// exists to prevent.
    pub fn as_ffi(self) -> Result<i32> {
        match self {
            Self::AdamW => Ok(0),
            Self::Sgd => Ok(1),
            Self::Muon | Self::Gefen => Err(Error::invalid(format!(
                "optimizer {self} is not available in this build; use adamw or sgd"
            ))),
        }
    }

    /// Reads back what a runtime or a checkpoint recorded. An unknown value is
    /// an error rather than a default: "the optimizer this run used" is not a
    /// field that may be guessed.
    pub fn from_ffi(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::AdamW),
            1 => Ok(Self::Sgd),
            other => Err(Error::runtime(format!(
                "the runtime reported optimizer {other}, which this build does not know"
            ))),
        }
    }

    /// The version of this optimizer's layout: its slot table and the
    /// arithmetic of its update. Recorded beside the optimizer's name in the
    /// checkpoint and compared on resume, because the name alone does not pin
    /// what a slot payload means.
    ///
    /// All layouts are `1` today. The first fork is the pending Gefen `variant`
    /// key, which should move this number rather than the optimizer name.
    pub fn layout_version(self) -> u32 {
        match self {
            Self::AdamW | Self::Sgd | Self::Muon | Self::Gefen => 1,
        }
    }

    /// The hyperparameters this optimizer's update reads, with their defaults
    /// and the range each one is valid in.
    ///
    /// Every layout declares `learning_rate`, `weight_decay` and
    /// `max_grad_norm`; the rest are the optimizer's own. Structural values
    /// (block size, iteration count, codebook width) stay integers: a power of
    /// two is not a property a float has.
    pub fn hyperparameters(self) -> &'static [HyperparameterDefinition] {
        match self {
            Self::AdamW => &ADAMW_HYPERPARAMETERS,
            Self::Sgd => &SGD_HYPERPARAMETERS,
            Self::Muon => &MUON_HYPERPARAMETERS,
            Self::Gefen => &GEFEN_HYPERPARAMETERS,
        }
    }

    /// This optimizer's hyperparameters at their declared defaults, ready to
    /// be overwritten with the values a run configures.
    pub fn declared_hyperparameters(self) -> HyperparameterVector {
        HyperparameterVector {
            optimizer: self,
            values: self
                .hyperparameters()
                .iter()
                .map(|declared| (declared.name, declared.default))
                .collect(),
        }
    }

    /// The optimizer's declarative description: identity, layout version,
    /// slot tables and hyperparameters. No graphs, so it exists before any
    /// kernel does.
    pub fn descriptor(self) -> OptimizerDescriptor {
        OptimizerDescriptor {
            id: self.as_str(),
            layout_version: self.layout_version(),
            slots: self.slot_definitions(),
            shared_slots: self.shared_slot_definitions(),
            hyperparameters: self.hyperparameters(),
        }
    }

    /// The persistent per-parameter slots this optimizer keeps, in the order the
    /// state API enumerates them.
    ///
    /// SGD's empty list is the case worth naming: *no slot* and *no optimizer*
    /// are different states. An SGD run still has a step counter, a schedule
    /// and an RNG state, and a resume that concluded "no slots, so nothing was
    /// initialized" would silently restart the schedule from zero.
    ///
    /// Gefen's rows are declared with their block shape and byte widths even
    /// though nothing can allocate them yet.
    pub fn slot_definitions(self) -> &'static [SlotDefinition] {
        match self {
            Self::AdamW => &[
                SlotDefinition {
                    name: "m",
                    dtype: SlotDtype::F32,
                    shape: SlotShape::Parameter,
                    init: SlotInit::Zero,
                },
                SlotDefinition {
                    name: "v",
                    dtype: SlotDtype::F32,
                    shape: SlotShape::Parameter,
                    init: SlotInit::Zero,
                },
            ],
            Self::Sgd => &[],
            Self::Muon => &[SlotDefinition {
                name: "momentum",
                dtype: SlotDtype::F32,
                shape: SlotShape::Parameter,
                init: SlotInit::Zero,
            }],
            Self::Gefen => &[
                SlotDefinition {
                    name: "indices",
                    dtype: SlotDtype::I8,
                    shape: SlotShape::Parameter,
                    // The codebook has no exact zero, so an empty state is a
                    // zero scale and a canonical index.
                    init: SlotInit::Code(GEFEN_ZERO_BLOCK_INDEX),
                },
                SlotDefinition {
                    name: "scales",
                    dtype: SlotDtype::F32,
                    shape: SlotShape::Blocks(GEFEN_DEFAULT_BLOCK_SIZE),
                    init: SlotInit::Zero,
                },
                SlotDefinition {
                    name: "second_moments",
                    dtype: SlotDtype::F32,
                    shape: SlotShape::Blocks(GEFEN_DEFAULT_BLOCK_SIZE),
                    init: SlotInit::Zero,
                },
            ],
        }
    }

    /// Slot names alone, for a caller that compares a table without allocating one.
    pub fn slot_names(self) -> Vec<&'static str> {
        self.slot_definitions()
            .iter()
            .map(|slot| slot.name)
            .collect()
    }

    /// State kept once per *owner* rather than per parameter, e.g. a codebook.
    ///
    /// Empty for everything this build can run; the scope exists in the
    /// checkpoint at length zero.
    pub fn shared_slot_definitions(self) -> &'static [SlotDefinition] {
        match self {
            Self::AdamW | Self::Sgd | Self::Muon => &[],
            Self::Gefen => &[SlotDefinition {
                name: "codebook",
                dtype: SlotDtype::F32,
                shape: SlotShape::Fixed(GEFEN_CODEBOOK_LEVELS),
                init: SlotInit::UniformCodebook,
            }],
        }
    }

    /// Whether this optimizer's update step can write a parameter of this dtype.
    ///
    /// Beside [`Self::is_eligible`] and a different question: eligibility asks
    /// whether this optimizer is the right one for the parameter, this asks
    /// whether its kernel can touch it at all. The SGD step is F32-only while
    /// AdamW writes F32 and F16, and F16 is the default adapter storage.
    ///
    /// A `false` is a refusal in a single-optimizer run and a fallback in a
    /// mixed one; see [`Self::assign`].
    pub fn supports_dtype(self, dtype: &TensorDtype) -> bool {
        match self {
            Self::AdamW => matches!(dtype, TensorDtype::F32 | TensorDtype::F16),
            // F32 first; an F16 writeback is separate work.
            Self::Sgd | Self::Muon | Self::Gefen => matches!(dtype, TensorDtype::F32),
        }
    }

    /// The optimizer a run under this choice falls back to for a parameter it
    /// does not own.
    ///
    /// AdamW: widest dtype table, no eligibility rule. AdamW and SGD have no
    /// fallback of their own: they are chosen for the whole run, and silently
    /// running part of it on another optimizer would publish a trajectory the
    /// document did not ask for.
    pub fn fallback(self) -> Option<Self> {
        match self {
            Self::AdamW | Self::Sgd => None,
            Self::Muon | Self::Gefen => Some(Self::AdamW),
        }
    }

    /// The optimizer that actually owns this parameter, or `None` when nothing
    /// in this run's policy can write it.
    ///
    /// Checks eligibility first, then dtype: a tensor can be eligible by role
    /// but still one this kernel cannot touch, and answering with this
    /// optimizer there would assign an update that aborts inside the step.
    ///
    /// `None` is a refusal the caller raises by name; never substituted.
    pub fn assign(self, entry: &TrainableEntry) -> Option<Self> {
        if self.is_eligible(entry) && self.supports_dtype(&entry.dtype) {
            return Some(self);
        }
        let fallback = self.fallback()?;
        fallback.supports_dtype(&entry.dtype).then_some(fallback)
    }

    /// Persistent state bytes for one parameter of `n_elements`, under this
    /// optimizer, assuming the parameter is eligible for it.
    ///
    /// AdamW keeps two F32 tensors (`8N`), SGD keeps none, Muon one F32
    /// momentum (`4N`). Gefen is shaped per block and has no single-parameter
    /// closed form here; it is refused rather than approximated.
    fn eligible_state_bytes(self, n_elements: u64) -> u64 {
        match self {
            Self::AdamW => n_elements.saturating_mul(8),
            Self::Sgd => 0,
            Self::Muon => n_elements.saturating_mul(4),
            // Unreachable while `is_implemented` gates the choice; the fallback
            // is AdamW's because that is what an ineligible tensor gets.
            Self::Gefen => n_elements.saturating_mul(8),
        }
    }

    /// Persistent state bytes for `n_elements` of *adapter* factors, through
    /// the same eligibility rule [`Self::state_bytes`] applies.
    ///
    /// The entry point for the planner, which sizes an adapter analytically -
    /// from the model's geometry and the target set - and never materializes a
    /// factor to hand to [`Self::state_bytes`]. Sharing the arithmetic is what
    /// keeps its budget and the runtime's report from drifting apart, and
    /// without it the planner would spell AdamW's `8N` a second time and
    /// over-budget every SGD run by the whole optimizer.
    ///
    /// Adapter-only because a *base* parameter's eligibility depends on its
    /// shape, not only on its family: Muon takes hidden matrices, and an
    /// element count cannot say whether one is. A caller that has the resolved
    /// set uses [`Self::state_bytes`], which does.
    pub fn adapter_state_bytes(self, n_elements: u64) -> u64 {
        // The eligibility predicates read only `role`, `name` and the shape,
        // and for a LoRA factor the first two already decide. A synthetic entry
        // rather than a second copy of those rules: two spellings of "is this
        // parameter eligible" is the drift this method exists to stop.
        let factor = TrainableEntry {
            name: "adapter.lora_a".to_string(),
            role: TensorRole::LoraA,
            // Synthetic geometry only; budget arithmetic retains the full u64.
            ne: [i64::try_from(n_elements).unwrap_or(i64::MAX), 1, 1, 1],
            dtype: crate::trainable::TensorDtype::F32,
            n_elements,
            n_bytes: n_elements.saturating_mul(4),
            storage_id: 0,
        };
        self.assign(&factor)
            .unwrap_or(Self::AdamW)
            .eligible_state_bytes(n_elements)
    }

    /// Whether this entry is eligible for the optimizer, or falls back to AdamW.
    ///
    /// Muon's v1 policy is *role*, not rank: hidden base matrices with exactly
    /// two non-trivial logical dimensions. Embeddings, the LM head, norms and
    /// biases are excluded because of what they are, and LoRA factors stay on
    /// AdamW - orthogonalizing `A` and `B` separately is not orthogonalizing
    /// `BA`, so it is a separate experiment rather than a free extension.
    pub fn is_eligible(self, entry: &TrainableEntry) -> bool {
        match self {
            Self::AdamW | Self::Sgd => true,
            Self::Muon => {
                entry.role == TensorRole::Base
                    && !entry.name.ends_with(".bias")
                    && !entry.name.contains("norm")
                    && entry.name != crate::trainable::OUTPUT_HEAD
                    && entry.ne[0] > 1
                    && entry.ne[1] > 1
                    && entry.ne[2] <= 1
                    && entry.ne[3] <= 1
            }
            Self::Gefen => entry.n_elements >= GEFEN_DEFAULT_MIN_NUMEL,
        }
    }

    /// Persistent optimizer state for a whole resolved set, with the AdamW
    /// fallback of every ineligible tensor included.
    ///
    /// The fallback is part of the total on purpose: an optimizer whose ratio
    /// only holds for the tensors it accepts reports a figure no run ever pays.
    pub fn state_bytes(self, set: &TrainableSet) -> u64 {
        set.entries
            .iter()
            .map(|entry| {
                // An entry nothing can write is priced as AdamW, not free.
                self.assign(entry)
                    .unwrap_or(Self::AdamW)
                    .eligible_state_bytes(entry.n_elements)
            })
            .fold(0, u64::saturating_add)
    }

    /// The persistent state table this run would allocate, declared from the
    /// resolved set alone, to be compared against the table the runtime
    /// allocates after `llama_opt_init`.
    pub fn plan(self, set: &TrainableSet) -> OptimizerPlan {
        let mut parameters = Vec::with_capacity(set.entries.len());
        for entry in &set.entries {
            let owner = self.assign(entry);
            let slots = owner
                .map(|owner| {
                    owner
                        .slot_definitions()
                        .iter()
                        .map(|slot| slot.resolve(entry))
                        .collect()
                })
                .unwrap_or_default();
            parameters.push(PlannedParameter {
                name: entry.name.clone(),
                optimizer: owner,
                slots,
            });
        }
        // One shared row per owner that owns something, in first-seen order.
        let mut owners: Vec<OptimizerKind> = Vec::new();
        for parameter in &parameters {
            if let Some(owner) = parameter.optimizer
                && !owners.contains(&owner)
            {
                owners.push(owner);
            }
        }
        let shared = owners
            .iter()
            .flat_map(|owner| {
                owner
                    .shared_slot_definitions()
                    .iter()
                    .map(|slot| SharedSlot {
                        owner: *owner,
                        slot: slot.name,
                        dtype: slot.dtype,
                        n_elements: slot.shared_elements(),
                        n_bytes: slot.shared_elements().saturating_mul(slot.dtype.bytes()),
                    })
            })
            .collect();
        OptimizerPlan {
            chosen: self,
            parameters,
            shared,
        }
    }
}

/// The declarative description of one optimizer, assembled from the tables
/// above rather than stored: one definition, one reader.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizerDescriptor {
    /// The name a document writes and a checkpoint records.
    pub id: &'static str,
    /// The slot layout version, see [`OptimizerKind::layout_version`].
    pub layout_version: u32,
    pub slots: &'static [SlotDefinition],
    pub shared_slots: &'static [SlotDefinition],
    pub hyperparameters: &'static [HyperparameterDefinition],
}

/// One hyperparameter value, typed by what it is. `f32` for scalars: these
/// are the numbers the update kernel reads, and matching widths keep a
/// recorded value equal to a declared default bit for bit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HyperparameterValue {
    /// A rate, a decay, an epsilon.
    Scalar(f32),
    /// A block size, an iteration count, a codebook width.
    Structural(i64),
    /// A binary choice, e.g. Nesterov momentum.
    Toggle(bool),
}

impl HyperparameterValue {
    /// The variant name, for errors that say what was expected.
    pub fn type_name(self) -> &'static str {
        match self {
            Self::Scalar(_) => "scalar",
            Self::Structural(_) => "structural",
            Self::Toggle(_) => "toggle",
        }
    }

    /// Canonical rendering, the same way for the same value. This is what a
    /// checkpoint stores and a resume compares.
    pub fn render(self) -> String {
        match self {
            Self::Scalar(value) => format!("{value:?}"),
            Self::Structural(value) => value.to_string(),
            Self::Toggle(value) => value.to_string(),
        }
    }
}

impl fmt::Display for HyperparameterValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

/// The valid range of one hyperparameter. Checked when the value is set, not
/// when the step runs: an out of range beta aborts inside ggml instead of
/// surfacing as an error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HyperparameterBound {
    /// No range check; the type is the whole constraint (toggles).
    Free,
    /// Finite and strictly greater than zero.
    Positive,
    /// Finite and not negative.
    NonNegative,
    /// `0 <= x <= 1`.
    UnitInterval,
    /// A positive power of two.
    PowerOfTwo,
    /// An integer of at least this value.
    AtLeast(i64),
}

impl HyperparameterBound {
    fn check(self, name: &str, value: HyperparameterValue) -> Result<()> {
        let refuse = |requirement: &str| {
            Err(Error::invalid(format!(
                "optimizer hyperparameter '{name}' must be {requirement}; got {value}"
            )))
        };
        match (self, value) {
            (Self::Free, _) => Ok(()),
            (Self::Positive, HyperparameterValue::Scalar(scalar)) => {
                if scalar.is_finite() && scalar > 0.0 {
                    Ok(())
                } else {
                    refuse("finite and greater than zero")
                }
            }
            (Self::NonNegative, HyperparameterValue::Scalar(scalar)) => {
                if scalar.is_finite() && scalar >= 0.0 {
                    Ok(())
                } else {
                    refuse("finite and not negative")
                }
            }
            (Self::UnitInterval, HyperparameterValue::Scalar(scalar)) => {
                if scalar.is_finite() && (0.0..=1.0).contains(&scalar) {
                    Ok(())
                } else {
                    refuse("between zero and one")
                }
            }
            (Self::PowerOfTwo, HyperparameterValue::Structural(integer)) => {
                if integer > 0 && integer.count_ones() == 1 {
                    Ok(())
                } else {
                    refuse("a positive power of two")
                }
            }
            (Self::AtLeast(least), HyperparameterValue::Structural(integer)) => {
                if integer >= least {
                    Ok(())
                } else {
                    refuse(&format!("at least {least}"))
                }
            }
            // A bound and a value of different shapes: the definition is wrong.
            (bound, value) => Err(Error::invalid(format!(
                "optimizer hyperparameter '{name}' is declared {bound:?} and cannot hold a {} value",
                value.type_name()
            ))),
        }
    }
}

/// One declared hyperparameter: its name, its default and its valid range.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HyperparameterDefinition {
    pub name: &'static str,
    pub default: HyperparameterValue,
    pub bound: HyperparameterBound,
}

impl HyperparameterDefinition {
    const fn new(
        name: &'static str,
        default: HyperparameterValue,
        bound: HyperparameterBound,
    ) -> Self {
        Self {
            name,
            default,
            bound,
        }
    }

    const fn toggle(name: &'static str, default: bool) -> Self {
        Self {
            name,
            default: HyperparameterValue::Toggle(default),
            bound: HyperparameterBound::Free,
        }
    }
}

/// One optimizer's hyperparameters, filled in for one run. Built from the
/// declaration and overwritten with the run's values; a name the optimizer
/// does not declare is refused. Order is the declaration's, which makes
/// [`Self::lines`] canonical.
#[derive(Clone, Debug, PartialEq)]
pub struct HyperparameterVector {
    optimizer: OptimizerKind,
    values: Vec<(&'static str, HyperparameterValue)>,
}

impl HyperparameterVector {
    pub fn optimizer(&self) -> OptimizerKind {
        self.optimizer
    }

    /// Set one declared value. Refuses an unknown name, a value of the wrong
    /// shape and a value outside its bound.
    pub fn set(&mut self, name: &str, value: HyperparameterValue) -> Result<()> {
        let declared = self
            .optimizer
            .hyperparameters()
            .iter()
            .find(|declared| declared.name == name)
            .ok_or_else(|| {
                Error::invalid(format!(
                    "optimizer {} has no hyperparameter '{name}'",
                    self.optimizer
                ))
            })?;
        if declared.default.type_name() != value.type_name() {
            return Err(Error::invalid(format!(
                "optimizer hyperparameter '{name}' is a {} and was given a {} value",
                declared.default.type_name(),
                value.type_name()
            )));
        }
        declared.bound.check(name, value)?;
        for entry in &mut self.values {
            if entry.0 == name {
                entry.1 = value;
                return Ok(());
            }
        }
        self.values.push((declared.name, value));
        Ok(())
    }

    /// Convenience for setting a scalar value.
    pub fn set_scalar(&mut self, name: &str, value: f32) -> Result<()> {
        self.set(name, HyperparameterValue::Scalar(value))
    }

    pub fn get(&self, name: &str) -> Option<HyperparameterValue> {
        self.values
            .iter()
            .find(|(declared, _)| *declared == name)
            .map(|(_, value)| *value)
    }

    /// Name and rendering pairs, in declaration order.
    pub fn rows(&self) -> impl Iterator<Item = (&'static str, String)> + '_ {
        self.values
            .iter()
            .map(|(name, value)| (*name, value.render()))
    }

    /// The canonical `name=value` lines a checkpoint stores.
    pub fn lines(&self) -> Vec<String> {
        self.rows()
            .map(|(name, value)| format!("{name}={value}"))
            .collect()
    }
}

impl fmt::Display for HyperparameterVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.lines().join(" "))
    }
}

/// The dtype of one persistent slot.
///
/// Its own enumeration rather than [`TensorDtype`]: a slot's dtype is chosen
/// by the optimizer, not read off a file, and the two sets do not coincide
/// (Gefen stores byte indices).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotDtype {
    F32,
    /// ggml `I8` storage, read as unsigned `0..=255` bit patterns.
    I8,
}

impl SlotDtype {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::I8 => "i8",
        }
    }

    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::I8 => 1,
        }
    }
}

impl fmt::Display for SlotDtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How large one slot is, relative to the parameter it belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotShape {
    /// The parameter's own shape, element for element.
    Parameter,
    /// One element per quantization block of the given size: `ceil(N / block)`;
    /// a partial trailing block still counts.
    Blocks(u64),
    /// A fixed element count, independent of any parameter. Shared state.
    Fixed(u64),
}

/// What a slot holds before the first update. An enumeration rather than a
/// scalar fill because a codebook is generated, not filled.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotInit {
    /// Every element zero, in the slot's own dtype.
    Zero,
    /// Every element the same code, an unsigned bit pattern.
    Code(u8),
    /// `c[k] = -1 + 2k/255` over [`GEFEN_CODEBOOK_LEVELS`] entries. Generated,
    /// not filled.
    UniformCodebook,
}

/// One persistent slot an optimizer keeps, as the optimizer declares it.
/// The `build_step` and `fill_params` that use it live in the runtime.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotDefinition {
    pub name: &'static str,
    pub dtype: SlotDtype,
    pub shape: SlotShape,
    pub init: SlotInit,
}

impl SlotDefinition {
    /// This definition against one parameter: the element count and the byte
    /// count it would allocate.
    pub fn resolve(&self, entry: &TrainableEntry) -> PlannedSlot {
        let n_elements = match self.shape {
            SlotShape::Parameter => entry.n_elements,
            SlotShape::Blocks(size) => entry.n_elements.div_ceil(size.max(1)),
            SlotShape::Fixed(count) => count,
        };
        PlannedSlot {
            slot: self.name,
            dtype: self.dtype,
            n_elements,
            n_bytes: n_elements.saturating_mul(self.dtype.bytes()),
        }
    }

    /// The element count of a shared slot; only `Fixed` is a shared shape.
    fn shared_elements(&self) -> u64 {
        match self.shape {
            SlotShape::Fixed(count) => count,
            SlotShape::Parameter | SlotShape::Blocks(_) => 0,
        }
    }
}

/// One resolved slot: what the runtime is expected to have allocated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedSlot {
    pub slot: &'static str,
    pub dtype: SlotDtype,
    pub n_elements: u64,
    pub n_bytes: u64,
}

/// One marked parameter, the optimizer that owns it, and the slots that
/// optimizer keeps for it. An empty slot list still records ownership (SGD owns
/// everything and keeps no state).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedParameter {
    pub name: String,
    /// `None` when no optimizer in this run's policy can write it.
    pub optimizer: Option<OptimizerKind>,
    pub slots: Vec<PlannedSlot>,
}

impl PlannedParameter {
    /// The layout version of the optimizer that owns this parameter. Derived
    /// from [`OptimizerKind::layout_version`] so a row cannot disagree with it.
    pub fn layout_version(&self) -> Option<u32> {
        self.optimizer.map(OptimizerKind::layout_version)
    }
}

/// One resolved shared slot, allocated once per owning optimizer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedSlot {
    pub owner: OptimizerKind,
    pub slot: &'static str,
    pub dtype: SlotDtype,
    pub n_elements: u64,
    pub n_bytes: u64,
}

/// The persistent state one run would allocate, declared before the graph
/// exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizerPlan {
    /// What the document asked for; parameters may be assigned to its fallback,
    /// so the choice is recorded beside the table rather than inferred from it.
    pub chosen: OptimizerKind,
    pub parameters: Vec<PlannedParameter>,
    pub shared: Vec<SharedSlot>,
}

impl OptimizerPlan {
    /// Total persistent bytes, parameters and shared state together.
    pub fn state_bytes(&self) -> u64 {
        let parameters = self
            .parameters
            .iter()
            .flat_map(|parameter| parameter.slots.iter())
            .map(|slot| slot.n_bytes);
        let shared = self.shared.iter().map(|slot| slot.n_bytes);
        parameters.chain(shared).fold(0, u64::saturating_add)
    }

    /// Parameters no optimizer in this run can write, by name.
    pub fn unwritable(&self) -> Vec<&str> {
        self.parameters
            .iter()
            .filter(|parameter| parameter.optimizer.is_none())
            .map(|parameter| parameter.name.as_str())
            .collect()
    }

    /// The `(parameter, slot)` rows this plan declares, the shape the runtime's
    /// live table is compared against.
    pub fn slot_rows(&self) -> Vec<(&str, &PlannedSlot)> {
        self.parameters
            .iter()
            .flat_map(|parameter| {
                parameter
                    .slots
                    .iter()
                    .map(move |slot| (parameter.name.as_str(), slot))
            })
            .collect()
    }

    /// The `(owner, slot)` rows of the shared scope, in the same shape.
    pub fn shared_rows(&self) -> Vec<(&str, &SharedSlot)> {
        self.shared
            .iter()
            .map(|slot| (slot.owner.as_str(), slot))
            .collect()
    }

    /// Refuses a live shared table that is not the declared one.
    pub fn check_live_shared(&self, live: &[(String, String, u64)]) -> Result<()> {
        let declared = self.shared_rows();
        if declared.len() != live.len() {
            return Err(Error::runtime(format!(
                "optimizer {} declares {} shared slot(s), but the runtime allocated {}",
                self.chosen,
                declared.len(),
                live.len()
            )));
        }
        for ((owner, slot), (live_owner, live_slot, live_bytes)) in declared.iter().zip(live) {
            if owner != live_owner || slot.slot != live_slot || slot.n_bytes != *live_bytes {
                return Err(Error::runtime(format!(
                    "shared optimizer slot {}/{} ({} bytes) was declared where the runtime \
                     allocated {live_owner}/{live_slot} ({live_bytes} bytes)",
                    owner, slot.slot, slot.n_bytes
                )));
            }
        }
        Ok(())
    }

    /// Refuses a live parameter table that is not the declared one, naming the
    /// first row that disagrees.
    ///
    /// `live` is `(owner, slot, n_bytes)`. Compared by identity, never by
    /// index: the two tables follow different node orders, and restore matches
    /// on the same keys.
    pub fn check_live(&self, live: &[(String, String, u64)]) -> Result<()> {
        let declared = self.slot_rows();
        for (owner, slot) in &declared {
            let found = live
                .iter()
                .find(|(live_owner, live_slot, _)| live_owner == owner && live_slot == slot.slot);
            let Some((_, _, live_bytes)) = found else {
                return Err(Error::runtime(format!(
                    "optimizer {} declares a '{}' slot for '{owner}' that the runtime did not \
                     allocate",
                    self.chosen, slot.slot
                )));
            };
            if slot.n_bytes != *live_bytes {
                return Err(Error::runtime(format!(
                    "optimizer slot {owner}/{} was declared as {} byte(s) and allocated as \
                     {live_bytes}",
                    slot.slot, slot.n_bytes
                )));
            }
        }
        for (owner, slot, _) in live {
            if !declared
                .iter()
                .any(|(declared_owner, declared)| declared_owner == owner && declared.slot == slot)
            {
                return Err(Error::runtime(format!(
                    "the runtime allocated a '{slot}' slot for '{owner}' that optimizer {} \
                     does not declare",
                    self.chosen
                )));
            }
        }
        Ok(())
    }
}

impl fmt::Display for OptimizerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// AdamW. `beta1`, `beta2` and `eps` come from ggml's defaults; declaring
/// them lets a checkpoint record the values the update actually read.
const ADAMW_HYPERPARAMETERS: [HyperparameterDefinition; 6] = [
    HyperparameterDefinition::new(
        "learning_rate",
        HyperparameterValue::Scalar(1.0e-3),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "beta1",
        HyperparameterValue::Scalar(0.9),
        HyperparameterBound::UnitInterval,
    ),
    HyperparameterDefinition::new(
        "beta2",
        HyperparameterValue::Scalar(0.999),
        HyperparameterBound::UnitInterval,
    ),
    HyperparameterDefinition::new(
        "eps",
        HyperparameterValue::Scalar(1.0e-8),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "weight_decay",
        HyperparameterValue::Scalar(0.0),
        HyperparameterBound::NonNegative,
    ),
    HyperparameterDefinition::new(
        "max_grad_norm",
        HyperparameterValue::Scalar(1.0),
        HyperparameterBound::Positive,
    ),
];

/// SGD: the universal three and nothing else.
const SGD_HYPERPARAMETERS: [HyperparameterDefinition; 3] = [
    HyperparameterDefinition::new(
        "learning_rate",
        HyperparameterValue::Scalar(1.0e-3),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "weight_decay",
        HyperparameterValue::Scalar(0.0),
        HyperparameterBound::NonNegative,
    ),
    HyperparameterDefinition::new(
        "max_grad_norm",
        HyperparameterValue::Scalar(1.0),
        HyperparameterBound::Positive,
    ),
];

/// Muon v1. The fallback rate is its own value, not a ratio of `learning_rate`:
/// an orthogonalized update and an AdamW one are not in the same units.
const MUON_HYPERPARAMETERS: [HyperparameterDefinition; 8] = [
    HyperparameterDefinition::new(
        "learning_rate",
        HyperparameterValue::Scalar(0.02),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "momentum",
        HyperparameterValue::Scalar(0.95),
        HyperparameterBound::UnitInterval,
    ),
    HyperparameterDefinition::toggle("nesterov", true),
    HyperparameterDefinition::new(
        "ns_steps",
        HyperparameterValue::Structural(5),
        HyperparameterBound::AtLeast(1),
    ),
    HyperparameterDefinition::new(
        "ns_epsilon",
        HyperparameterValue::Scalar(1.0e-7),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "fallback_learning_rate",
        HyperparameterValue::Scalar(1.0e-3),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "weight_decay",
        HyperparameterValue::Scalar(0.0),
        HyperparameterBound::NonNegative,
    ),
    HyperparameterDefinition::new(
        "max_grad_norm",
        HyperparameterValue::Scalar(1.0),
        HyperparameterBound::Positive,
    ),
];

/// Gefen. `block_size` and `codebook_levels` describe shapes, so they are
/// integers; `block_size` is additionally a power of two.
const GEFEN_HYPERPARAMETERS: [HyperparameterDefinition; 9] = [
    HyperparameterDefinition::new(
        "learning_rate",
        HyperparameterValue::Scalar(1.0e-3),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "beta1",
        HyperparameterValue::Scalar(0.9),
        HyperparameterBound::UnitInterval,
    ),
    HyperparameterDefinition::new(
        "beta2",
        HyperparameterValue::Scalar(0.999),
        HyperparameterBound::UnitInterval,
    ),
    HyperparameterDefinition::new(
        "eps",
        HyperparameterValue::Scalar(1.0e-8),
        HyperparameterBound::Positive,
    ),
    HyperparameterDefinition::new(
        "block_size",
        HyperparameterValue::Structural(GEFEN_DEFAULT_BLOCK_SIZE as i64),
        HyperparameterBound::PowerOfTwo,
    ),
    HyperparameterDefinition::new(
        "min_numel",
        HyperparameterValue::Structural(GEFEN_DEFAULT_MIN_NUMEL as i64),
        HyperparameterBound::AtLeast(1),
    ),
    HyperparameterDefinition::new(
        "codebook_levels",
        HyperparameterValue::Structural(GEFEN_CODEBOOK_LEVELS as i64),
        HyperparameterBound::AtLeast(2),
    ),
    HyperparameterDefinition::new(
        "weight_decay",
        HyperparameterValue::Scalar(0.0),
        HyperparameterBound::NonNegative,
    ),
    HyperparameterDefinition::new(
        "max_grad_norm",
        HyperparameterValue::Scalar(1.0),
        HyperparameterBound::Positive,
    ),
];

/// Default `min_numel` below which a Gefen-selected parameter falls back to
/// AdamW. Named rather than inlined because the plan's own warning depends on
/// it: a rank-16 LoRA factor on a 1024-wide projection has 16384 elements, so
/// "most LoRA factors are below the threshold" is false.
pub const GEFEN_DEFAULT_MIN_NUMEL: u64 = 4096;

/// Default elements per fixed quantization block, the divisor of
/// `ceil(N / B)` in [`SlotShape::Blocks`].
pub const GEFEN_DEFAULT_BLOCK_SIZE: u64 = 1024;

/// Entries of the uniform codebook, `c[k] = -1 + 2k/255`.
pub const GEFEN_CODEBOOK_LEVELS: u64 = 256;

/// The index a zero block stores. The codebook has no exact zero (a zero block
/// decodes through its zero scale), so this value only has to be canonical.
pub const GEFEN_ZERO_BLOCK_INDEX: u8 = 127;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainable::TensorDtype;

    fn entry(name: &str, role: TensorRole, ne: [i64; 4]) -> TrainableEntry {
        let n_elements = ne.iter().product::<i64>() as u64;
        TrainableEntry {
            name: name.to_string(),
            role,
            ne,
            dtype: TensorDtype::F32,
            n_elements,
            n_bytes: n_elements * 4,
            storage_id: 0,
        }
    }

    fn set(entries: Vec<TrainableEntry>) -> TrainableSet {
        TrainableSet {
            entries,
            ..Default::default()
        }
    }

    #[test]
    fn the_two_with_an_update_step_are_selectable_and_the_others_are_not() {
        assert!(OptimizerKind::AdamW.is_implemented());
        assert!(OptimizerKind::Sgd.is_implemented());
        // No kernel, no descriptor: accepting either would be a checkpoint
        // that records an optimizer the run never ran.
        assert!(!OptimizerKind::Muon.is_implemented());
        assert!(!OptimizerKind::Gefen.is_implemented());
        assert_eq!(OptimizerKind::default(), OptimizerKind::AdamW);
    }

    #[test]
    fn the_wire_value_round_trips_for_what_the_runtime_can_build() {
        for kind in [OptimizerKind::AdamW, OptimizerKind::Sgd] {
            assert_eq!(
                OptimizerKind::from_ffi(kind.as_ffi().unwrap()).unwrap(),
                kind
            );
        }
        assert!(OptimizerKind::Muon.as_ffi().is_err());
        assert!(OptimizerKind::Gefen.as_ffi().is_err());
        assert!(OptimizerKind::from_ffi(7).is_err());
    }

    /// The planner has no resolved entry to hand `state_bytes`, so it goes
    /// through this - and the two must agree, or a budget and a report
    /// describing the same run disagree by the whole optimizer.
    #[test]
    fn the_adapter_formula_agrees_with_the_resolved_set_it_cannot_build() {
        let factors = set(vec![entry(
            "blk.0.attn_q.weight.lora_a",
            TensorRole::LoraA,
            [1024, 16, 1, 1],
        )]);
        let n = 1024 * 16;
        for kind in [
            OptimizerKind::AdamW,
            OptimizerKind::Sgd,
            OptimizerKind::Muon,
        ] {
            assert_eq!(
                kind.adapter_state_bytes(n),
                kind.state_bytes(&factors),
                "{kind}"
            );
        }
        // And the figure itself: SGD keeps none, which is what the planner was
        // spelling as AdamW's 8N.
        assert_eq!(OptimizerKind::Sgd.adapter_state_bytes(n), 0);
        assert_eq!(OptimizerKind::AdamW.adapter_state_bytes(n), n * 8);
    }

    #[test]
    fn sgd_keeps_no_slot_and_adamw_keeps_two() {
        assert!(OptimizerKind::Sgd.slot_names().is_empty());
        assert_eq!(OptimizerKind::AdamW.slot_names(), ["m", "v"]);
    }

    /// AdamW's two slots per parameter, declared from the resolved set in the
    /// order the state API enumerates them.
    #[test]
    fn the_declared_table_is_two_f32_slots_per_parameter_under_adamw() {
        let two = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.0.attn_norm.weight", TensorRole::Base, [64, 1, 1, 1]),
        ]);
        let plan = OptimizerKind::AdamW.plan(&two);
        let rows: Vec<(&str, &str, u64)> = plan
            .slot_rows()
            .iter()
            .map(|(owner, slot)| (*owner, slot.slot, slot.n_bytes))
            .collect();
        assert_eq!(
            rows,
            [
                ("blk.0.attn_q.weight", "m", 64 * 64 * 4),
                ("blk.0.attn_q.weight", "v", 64 * 64 * 4),
                ("blk.0.attn_norm.weight", "m", 64 * 4),
                ("blk.0.attn_norm.weight", "v", 64 * 4),
            ]
        );
        assert!(plan.shared.is_empty());
        assert_eq!(plan.state_bytes(), OptimizerKind::AdamW.state_bytes(&two));
        assert!(plan.unwritable().is_empty());
    }

    /// Every parameter owned, no slot kept.
    #[test]
    fn an_sgd_plan_owns_every_parameter_and_declares_no_slot() {
        let two = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.0.attn_norm.weight", TensorRole::Base, [64, 1, 1, 1]),
        ]);
        let plan = OptimizerKind::Sgd.plan(&two);
        assert_eq!(plan.parameters.len(), 2);
        assert!(plan.slot_rows().is_empty());
        assert_eq!(plan.state_bytes(), 0);
        assert!(
            plan.parameters
                .iter()
                .all(|parameter| parameter.optimizer == Some(OptimizerKind::Sgd))
        );
    }

    /// Declared and live tables are compared by identity, never by index.
    #[test]
    fn a_live_table_that_is_not_the_declared_one_is_refused_by_the_row_that_differs() {
        let one = set(vec![entry(
            "blk.0.attn_q.weight",
            TensorRole::Base,
            [8, 8, 1, 1],
        )]);
        let plan = OptimizerKind::AdamW.plan(&one);
        let live = |rows: &[(&str, &str, u64)]| -> Vec<(String, String, u64)> {
            rows.iter()
                .map(|(owner, slot, bytes)| (owner.to_string(), slot.to_string(), *bytes))
                .collect()
        };
        plan.check_live(&live(&[
            ("blk.0.attn_q.weight", "m", 256),
            ("blk.0.attn_q.weight", "v", 256),
        ]))
        .unwrap();
        // A declared slot the runtime did not allocate.
        let error = plan
            .check_live(&live(&[("blk.0.attn_q.weight", "m", 256)]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not allocate"), "{error}");
        // A slot the runtime allocated that this optimizer does not declare.
        let error = plan
            .check_live(&live(&[
                ("blk.0.attn_q.weight", "m", 256),
                ("blk.0.attn_q.weight", "v", 256),
                ("blk.0.attn_q.weight", "momentum", 256),
            ]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not declare"), "{error}");
        // The two orders need not coincide.
        plan.check_live(&live(&[
            ("blk.0.attn_q.weight", "v", 256),
            ("blk.0.attn_q.weight", "m", 256),
        ]))
        .unwrap();
        // The right rows, the wrong width.
        let error = plan
            .check_live(&live(&[
                ("blk.0.attn_q.weight", "m", 256),
                ("blk.0.attn_q.weight", "v", 128),
            ]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("128"), "{error}");
    }

    /// Under SGD an F16 factor is a refusal; under Muon it falls back to the
    /// optimizer that writes F16.
    #[test]
    fn the_dtype_predicate_refuses_under_a_chosen_optimizer_and_falls_back_under_a_mixed_one() {
        let mut factor = entry(
            "blk.0.attn_q.weight.lora_a",
            TensorRole::LoraA,
            [1024, 16, 1, 1],
        );
        factor.dtype = TensorDtype::F16;

        assert_eq!(
            OptimizerKind::AdamW.assign(&factor),
            Some(OptimizerKind::AdamW)
        );
        // SGD has no fallback.
        assert_eq!(OptimizerKind::Sgd.assign(&factor), None);
        assert_eq!(OptimizerKind::Sgd.fallback(), None);
        // Muon falls back; AdamW writes F16.
        assert_eq!(
            OptimizerKind::Muon.assign(&factor),
            Some(OptimizerKind::AdamW)
        );

        // A dtype nothing writes is recorded by name, not substituted.
        let mut quantized = entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]);
        quantized.dtype = TensorDtype::Other("Q4_K".to_string());
        assert_eq!(OptimizerKind::AdamW.assign(&quantized), None);
        let plan = OptimizerKind::AdamW.plan(&set(vec![quantized]));
        assert_eq!(plan.unwritable(), ["blk.0.attn_q.weight"]);
    }

    /// Gefen's rows are declared per block, its codebook once per owner.
    #[test]
    fn gefens_rows_are_declared_per_block_and_its_codebook_once_per_owner() {
        // 4096 elements at the default block size: four blocks.
        let one = set(vec![entry(
            "blk.0.attn_q.weight",
            TensorRole::Base,
            [64, 64, 1, 1],
        )]);
        let plan = OptimizerKind::Gefen.plan(&one);
        let rows: Vec<(&str, u64)> = plan
            .slot_rows()
            .iter()
            .map(|(_, slot)| (slot.slot, slot.n_bytes))
            .collect();
        let blocks = 4096u64.div_ceil(GEFEN_DEFAULT_BLOCK_SIZE);
        assert_eq!(
            rows,
            [
                ("indices", 4096),
                ("scales", blocks * 4),
                ("second_moments", blocks * 4),
            ]
        );
        assert_eq!(plan.shared.len(), 1);
        assert_eq!(plan.shared[0].slot, "codebook");
        assert_eq!(plan.shared[0].n_bytes, GEFEN_CODEBOOK_LEVELS * 4);
        // A partial trailing block is a block: 4097 elements is five, not four.
        let odd = set(vec![entry(
            "blk.0.ssm_a",
            TensorRole::Base,
            [4097, 1, 1, 1],
        )]);
        let plan = OptimizerKind::Gefen.plan(&odd);
        assert_eq!(plan.slot_rows()[1].1.n_elements, 5);
    }

    #[test]
    fn state_bytes_follow_the_published_table() {
        let one = set(vec![entry(
            "blk.0.attn_q.weight",
            TensorRole::Base,
            [64, 64, 1, 1],
        )]);
        let n = 64 * 64;
        assert_eq!(OptimizerKind::AdamW.state_bytes(&one), n * 8);
        assert_eq!(OptimizerKind::Sgd.state_bytes(&one), 0);
        assert_eq!(OptimizerKind::Muon.state_bytes(&one), n * 4);
    }

    #[test]
    fn muon_excludes_by_role_and_falls_back_to_adamw_rather_than_skipping() {
        let mixed = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.0.attn_norm.weight", TensorRole::Base, [64, 1, 1, 1]),
            entry("blk.0.attn_q.bias", TensorRole::Base, [64, 1, 1, 1]),
            entry("output.weight", TensorRole::Base, [64, 32, 1, 1]),
            entry(
                "blk.0.attn_q.weight.lora_a",
                TensorRole::LoraA,
                [64, 8, 1, 1],
            ),
        ]);
        // 4 bytes per element only on the hidden matrix; 8 everywhere else.
        let expected = 64 * 64 * 4 + 64 * 8 + 64 * 8 + 64 * 32 * 8 + 64 * 8 * 8;
        assert_eq!(OptimizerKind::Muon.state_bytes(&mixed), expected);
        // Worth saying out loud: on a LoRA-only set Muon costs exactly AdamW.
        let lora_only = set(vec![entry(
            "blk.0.attn_q.weight.lora_a",
            TensorRole::LoraA,
            [1024, 16, 1, 1],
        )]);
        assert_eq!(
            OptimizerKind::Muon.state_bytes(&lora_only),
            OptimizerKind::AdamW.state_bytes(&lora_only)
        );
    }

    #[test]
    fn a_rank_16_lora_factor_is_above_the_gefen_fallback_threshold() {
        // The plan's own correction: 16 x 1024 is 16384, not "below 4096".
        let factor = entry("a.lora_a", TensorRole::LoraA, [1024, 16, 1, 1]);
        assert!(OptimizerKind::Gefen.is_eligible(&factor));
    }

    const ALL: [OptimizerKind; 4] = [
        OptimizerKind::AdamW,
        OptimizerKind::Sgd,
        OptimizerKind::Muon,
        OptimizerKind::Gefen,
    ];

    /// Every layout declares the three scalars a run configures, so the
    /// schedule and the gradient-norm ceiling stay per-run rather than
    /// per-owner in a mixed run.
    #[test]
    fn every_layout_declares_the_three_scalars_a_run_configures() {
        for kind in ALL {
            let vector = kind.declared_hyperparameters();
            for name in ["learning_rate", "weight_decay", "max_grad_norm"] {
                assert!(
                    matches!(vector.get(name), Some(HyperparameterValue::Scalar(_))),
                    "{kind} declares no scalar '{name}'"
                );
            }
        }
    }

    #[test]
    fn a_descriptor_is_the_tables_its_optimizer_declares() {
        for kind in ALL {
            let descriptor = kind.descriptor();
            assert_eq!(descriptor.id, kind.as_str());
            assert_eq!(descriptor.layout_version, kind.layout_version());
            assert_eq!(descriptor.slots, kind.slot_definitions());
            assert_eq!(descriptor.shared_slots, kind.shared_slot_definitions());
            assert_eq!(descriptor.hyperparameters, kind.hyperparameters());
            // One layout each, so far. The first fork is Gefen's `variant`.
            assert_eq!(descriptor.layout_version, 1);
        }
    }

    #[test]
    fn a_value_outside_its_bound_is_refused_by_name() {
        let mut adamw = OptimizerKind::AdamW.declared_hyperparameters();
        let refused = adamw.set_scalar("beta1", 1.5).expect_err("out of range");
        assert!(refused.to_string().contains("beta1"), "{refused}");
        assert!(adamw.set_scalar("eps", 0.0).is_err());
        assert!(adamw.set_scalar("learning_rate", f32::NAN).is_err());
        assert!(adamw.set_scalar("weight_decay", 0.0).is_ok());

        let mut gefen = OptimizerKind::Gefen.declared_hyperparameters();
        assert!(
            gefen
                .set("block_size", HyperparameterValue::Structural(1000))
                .is_err(),
            "a block size that is not a power of two was accepted"
        );
        assert!(
            gefen
                .set("block_size", HyperparameterValue::Structural(512))
                .is_ok()
        );
        let mut muon = OptimizerKind::Muon.declared_hyperparameters();
        assert!(
            muon.set("ns_steps", HyperparameterValue::Structural(0))
                .is_err()
        );
    }

    #[test]
    fn a_knob_the_layout_does_not_declare_is_refused() {
        let mut sgd = OptimizerKind::Sgd.declared_hyperparameters();
        let refused = sgd.set_scalar("beta1", 0.9).expect_err("sgd has no beta1");
        assert!(refused.to_string().contains("beta1"), "{refused}");
        // A value of the wrong shape is refused too.
        let mut gefen = OptimizerKind::Gefen.declared_hyperparameters();
        let refused = gefen
            .set("block_size", HyperparameterValue::Scalar(1024.0))
            .expect_err("a structural knob is not a float");
        assert!(refused.to_string().contains("block_size"), "{refused}");
    }

    /// The rendering is stable and in declaration order.
    #[test]
    fn the_vector_renders_in_declaration_order() {
        let mut adamw = OptimizerKind::AdamW.declared_hyperparameters();
        adamw.set_scalar("learning_rate", 2.0e-4).unwrap();
        adamw.set_scalar("weight_decay", 0.01).unwrap();
        assert_eq!(
            adamw.lines(),
            vec![
                "learning_rate=0.0002",
                "beta1=0.9",
                "beta2=0.999",
                "eps=1e-8",
                "weight_decay=0.01",
                "max_grad_norm=1.0",
            ]
        );
        // Setting twice replaces rather than appends.
        adamw.set_scalar("weight_decay", 0.02).unwrap();
        assert_eq!(adamw.lines().len(), 6);
        assert_eq!(
            adamw.get("weight_decay"),
            Some(HyperparameterValue::Scalar(0.02))
        );
    }

    /// In a mixed run each parameter carries the layout of its own optimizer.
    #[test]
    fn a_planned_parameter_carries_the_layout_of_its_owner() {
        let mixed = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.0.attn_norm.weight", TensorRole::Base, [64, 1, 1, 1]),
        ]);
        let plan = OptimizerKind::Muon.plan(&mixed);
        assert_eq!(plan.parameters[0].optimizer, Some(OptimizerKind::Muon));
        assert_eq!(plan.parameters[1].optimizer, Some(OptimizerKind::AdamW));
        for parameter in &plan.parameters {
            assert_eq!(
                parameter.layout_version(),
                parameter.optimizer.map(OptimizerKind::layout_version)
            );
        }
        // A parameter nothing can write has no layout.
        let unwritable = TrainableEntry {
            dtype: TensorDtype::Other("Q4_K".to_string()),
            ..entry("blk.0.ffn_up.weight", TensorRole::Base, [64, 64, 1, 1])
        };
        let plan = OptimizerKind::Sgd.plan(&set(vec![unwritable]));
        assert_eq!(plan.parameters[0].layout_version(), None);
    }

    #[test]
    fn names_round_trip_through_their_document_spelling() {
        for kind in [
            OptimizerKind::AdamW,
            OptimizerKind::Sgd,
            OptimizerKind::Muon,
            OptimizerKind::Gefen,
        ] {
            assert_eq!(OptimizerKind::parse(kind.as_str()).unwrap(), kind);
        }
        assert!(OptimizerKind::parse("lion").is_err());
    }
}
