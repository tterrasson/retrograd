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
    /// everywhere else.
    Muon,
    /// Fixed-block Gefen, under the layout the run declared. The variant is
    /// part of the value because it selects a slot table rather than scaling
    /// an update: two runs that differ only in it keep different state.
    Gefen(GefenLayout),
}

/// Which fixed-block state a Gefen run keeps.
///
/// Not a hyperparameter: it does not parameterize a layout, it selects one, so
/// it moves [`OptimizerKind::layout_version`] instead of appearing in the
/// declared vector.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GefenVariant {
    /// An F32 first moment per element and an F32 second moment per block:
    /// `4N + 4K`, which is approximately half of AdamW rather than exactly
    /// half. Available first, and not a fine-tuning recommendation on arrival.
    #[default]
    SharedV,
    /// A byte-indexed first moment against a shared uniform codebook, with an
    /// F32 scale and second moment per block: `N + 8K`.
    QuantizedM,
}

impl GefenVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::SharedV => "shared_v",
            Self::QuantizedM => "quantized_m",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "shared_v" | "shared-v" => Ok(Self::SharedV),
            "quantized_m" | "quantized-m" => Ok(Self::QuantizedM),
            other => Err(Error::config(format!(
                "optimizer.gefen.variant must be shared_v or quantized_m; got '{other}'"
            ))),
        }
    }

    /// The integer `retro_train_config.gefen_variant` carries.
    pub fn as_ffi(self) -> i32 {
        match self {
            Self::SharedV => 0,
            Self::QuantizedM => 1,
        }
    }

    pub fn from_ffi(value: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::SharedV),
            1 => Ok(Self::QuantizedM),
            other => Err(Error::runtime(format!(
                "the runtime reported gefen variant {other}, which this build does not know"
            ))),
        }
    }
}

impl fmt::Display for GefenVariant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Gefen's structural parameters: the ones that decide a slot's shape or which
/// parameters the optimizer owns, rather than scaling an update.
///
/// Carried by the optimizer value itself because the slot table is a function
/// of them: a plan built from one layout and a runtime allocated under another
/// disagree by construction, which is what [`OptimizerPlan::check_live`] is
/// there to catch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GefenLayout {
    pub variant: GefenVariant,
    /// Elements per quantization block, the divisor of `ceil(N / B)`.
    pub block_size: u64,
    /// Below this element count a selected parameter falls back to AdamW.
    pub min_numel: u64,
}

impl Default for GefenLayout {
    fn default() -> Self {
        Self {
            variant: GefenVariant::default(),
            block_size: GEFEN_DEFAULT_BLOCK_SIZE,
            min_numel: GEFEN_DEFAULT_MIN_NUMEL,
        }
    }
}

impl OptimizerKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
            Self::Muon => "muon",
            Self::Gefen(_) => "gefen",
        }
    }

    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "adamw" | "adam_w" => Ok(Self::AdamW),
            "sgd" => Ok(Self::Sgd),
            "muon" => Ok(Self::Muon),
            // The variant is a separate key: shared_v is what a document that
            // names only "gefen" gets, and it is the one available first.
            "gefen" => Ok(Self::Gefen(GefenLayout::default())),
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
    /// All four have an update step: AdamW's and SGD's kernels, Muon's graph
    /// and Gefen's two-phase pair. What is still refused is a *device* - the
    /// Gefen phases are written for the CPU and for Metal; a state mutation
    /// answered on a fallback backend would update a copy and leave the real
    /// slot stale, so the runtime asks the device at preflight.
    pub fn is_implemented(self) -> bool {
        true
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
            Self::Muon => Ok(2),
            Self::Gefen(_) => Ok(3),
        }
    }

    /// Reads back what a runtime or a checkpoint recorded. An unknown value is
    /// an error rather than a default: "the optimizer this run used" is not a
    /// field that may be guessed.
    /// `gefen` is read together with its variant, because the name alone does
    /// not say which slot table the runtime allocated - and building a plan
    /// against the other one would compare two different layouts row by row.
    /// The value is ignored for every other optimizer.
    pub fn from_ffi(value: i32, gefen_variant: i32) -> Result<Self> {
        match value {
            0 => Ok(Self::AdamW),
            1 => Ok(Self::Sgd),
            2 => Ok(Self::Muon),
            3 => Ok(Self::Gefen(GefenLayout {
                variant: GefenVariant::from_ffi(gefen_variant)?,
                ..GefenLayout::default()
            })),
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
    /// Gefen's two variants are the one fork: they are two slot tables under
    /// one name, so the variant moves this number and a `quantized_m` payload
    /// cannot be restored into a `shared_v` run by accident.
    pub fn layout_version(self) -> u32 {
        match self {
            Self::AdamW | Self::Sgd | Self::Muon => 1,
            Self::Gefen(layout) => match layout.variant {
                GefenVariant::SharedV => 1,
                GefenVariant::QuantizedM => 2,
            },
        }
    }

    /// The same number for a parameter that keeps an F32 master copy.
    ///
    /// A master copy adds a slot, so it is a different slot table under the
    /// same optimizer name and moves the version the same way. The numbering
    /// is per optimizer, so this never collides with another optimizer's.
    pub fn layout_version_with_master(self, master_weights: bool) -> u32 {
        self.layout_version() + u32::from(master_weights)
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
            // A first moment nothing quantizes has no codebook to size, so
            // shared-v does not declare the key: an option one variant ignores
            // is refused rather than accepted and dropped.
            Self::Gefen(layout) => match layout.variant {
                GefenVariant::SharedV => &GEFEN_SHARED_V_HYPERPARAMETERS,
                GefenVariant::QuantizedM => &GEFEN_QUANTIZED_M_HYPERPARAMETERS,
            },
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

    /// The Gefen layout this choice declares, for a caller that has to spell
    /// the structural values on the wire. `None` for every other optimizer,
    /// which is what "these keys belong to one section" means at the type.
    pub fn gefen_layout(self) -> Option<GefenLayout> {
        match self {
            Self::Gefen(layout) => Some(layout),
            _ => None,
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
    pub fn slot_definitions(self) -> Vec<SlotDefinition> {
        match self {
            Self::AdamW => vec![
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
            Self::Sgd => Vec::new(),
            // The Newton-Schulz workspace is graph scratch and not state: it
            // survives no step, so it is not a slot.
            Self::Muon => vec![SlotDefinition {
                name: "momentum",
                dtype: SlotDtype::F32,
                shape: SlotShape::Parameter,
                init: SlotInit::Zero,
            }],
            Self::Gefen(layout) => match layout.variant {
                GefenVariant::SharedV => vec![
                    SlotDefinition {
                        name: "m",
                        dtype: SlotDtype::F32,
                        shape: SlotShape::Parameter,
                        init: SlotInit::Zero,
                    },
                    SlotDefinition {
                        name: "v",
                        dtype: SlotDtype::F32,
                        shape: SlotShape::Blocks(layout.block_size),
                        init: SlotInit::Zero,
                    },
                ],
                GefenVariant::QuantizedM => vec![
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
                        shape: SlotShape::Blocks(layout.block_size),
                        init: SlotInit::Zero,
                    },
                    SlotDefinition {
                        name: "second_moments",
                        dtype: SlotDtype::F32,
                        shape: SlotShape::Blocks(layout.block_size),
                        init: SlotInit::Zero,
                    },
                ],
            },
        }
    }

    /// Slot names alone, for a caller that compares a table without allocating one.
    pub fn slot_names(self) -> Vec<&'static str> {
        self.slot_definitions()
            .into_iter()
            .map(|slot| slot.name)
            .collect()
    }

    /// State kept once per *owner* rather than per parameter, e.g. a codebook.
    ///
    /// Empty for everything this build can run; the scope exists in the
    /// checkpoint at length zero.
    pub fn shared_slot_definitions(self) -> Vec<SlotDefinition> {
        match self {
            Self::AdamW | Self::Sgd | Self::Muon => Vec::new(),
            // Only the quantized variant has anything to look up; shared-v's
            // first moment is the value itself.
            Self::Gefen(layout) => match layout.variant {
                GefenVariant::SharedV => Vec::new(),
                GefenVariant::QuantizedM => vec![SlotDefinition {
                    name: "codebook",
                    dtype: SlotDtype::F32,
                    shape: SlotShape::Fixed(GEFEN_CODEBOOK_LEVELS),
                    init: SlotInit::UniformCodebook,
                }],
            },
        }
    }

    /// Whether this optimizer's update step can write a parameter of this dtype.
    ///
    /// Beside [`Self::is_eligible`] and a different question: eligibility asks
    /// whether this optimizer is the right one for the parameter, this asks
    /// whether its kernel can touch it at all. The Muon and Gefen steps are
    /// F32-only while AdamW and SGD write F32, F16 and BF16, and F16 is the
    /// default adapter storage. Which backend carries a kernel is
    /// [`crate::base_dtype::BASE_DTYPE_TABLE`]'s job, not this one.
    ///
    /// A `false` is a refusal in a single-optimizer run and a fallback in a
    /// mixed one; see [`Self::assign`].
    pub fn supports_dtype(self, dtype: &TensorDtype) -> bool {
        match self {
            // Both store their update with rounding, so both take the two
            // half-precision grids beside F32.
            Self::AdamW | Self::Sgd => matches!(
                dtype,
                TensorDtype::F32 | TensorDtype::F16 | TensorDtype::BF16
            ),
            // F32 only: Muon's Newton-Schulz orthogonalization and Gefen's
            // first-moment estimate are already approximations, and a rounded
            // store would stack a second on them.
            Self::Muon | Self::Gefen(_) => matches!(dtype, TensorDtype::F32),
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
            Self::Muon | Self::Gefen(_) => Some(Self::AdamW),
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
            // Per block, and the trailing partial block still costs a row:
            // `4N + 4K` under shared-v, `N + 8K` under quantized-m. The shared
            // codebook is counted once per owner by the plan, not here.
            Self::Gefen(layout) => {
                let blocks = n_elements.div_ceil(layout.block_size.max(1));
                match layout.variant {
                    GefenVariant::SharedV => n_elements
                        .saturating_mul(4)
                        .saturating_add(blocks.saturating_mul(4)),
                    GefenVariant::QuantizedM => n_elements.saturating_add(blocks.saturating_mul(8)),
                }
            }
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
            // Below the threshold the per-block state costs more than the
            // dense pair it replaces, and the fallback is counted in the total.
            Self::Gefen(layout) => entry.n_elements >= layout.min_numel,
        }
    }

    /// Persistent optimizer state for a whole resolved set, with the AdamW
    /// fallback of every ineligible tensor included.
    ///
    /// The fallback is part of the total on purpose: an optimizer whose ratio
    /// only holds for the tensors it accepts reports a figure no run ever pays.
    pub fn state_bytes(self, set: &TrainableSet) -> u64 {
        self.state_bytes_with_master(set, false)
    }

    /// The same total, counting the F32 master copy of every half-precision
    /// parameter when the run keeps one: four bytes per element, on top of the
    /// optimizer's own slots.
    ///
    /// A separate entry point rather than a flag in [`Self::state_bytes`]: the
    /// master copy belongs to the run's layout, not to the optimizer.
    pub fn state_bytes_with_master(self, set: &TrainableSet, master_weights: bool) -> u64 {
        set.entries
            .iter()
            .map(|entry| {
                // An entry nothing can write is priced as AdamW, not free.
                let slots = self
                    .assign(entry)
                    .unwrap_or(Self::AdamW)
                    .eligible_state_bytes(entry.n_elements);
                let master = master_weights
                    .then(|| master_slot_definition(&entry.dtype))
                    .flatten()
                    .map_or(0, |slot| slot.resolve(entry).n_bytes);
                slots.saturating_add(master)
            })
            .fold(0, u64::saturating_add)
    }

    /// The persistent state table this run would allocate, declared from the
    /// resolved set alone, to be compared against the table the runtime
    /// allocates after `llama_opt_init`.
    pub fn plan(self, set: &TrainableSet) -> OptimizerPlan {
        self.plan_with(set, |entry| self.assign(entry))
    }

    /// The same table for a run that keeps an F32 master copy of every
    /// half-precision parameter. The master slot rides each parameter's slot
    /// list, last, where the allocator puts it, so the plan and the live
    /// table stay row-for-row comparable.
    pub fn plan_with_master(
        self,
        set: &TrainableSet,
        master_weights: bool,
        owner_of: impl Fn(&TrainableEntry) -> Option<Self>,
    ) -> OptimizerPlan {
        let mut plan = self.plan_with(set, owner_of);
        if !master_weights {
            return plan;
        }
        for (parameter, entry) in plan.parameters.iter_mut().zip(&set.entries) {
            // `plan_with` emits one row per entry, in order; the pairing is
            // checked, not assumed, since a mismatch would give one parameter
            // another's dtype.
            debug_assert_eq!(parameter.name, entry.name);
            // A parameter no optimizer can write gets no slots at all, master
            // copy included.
            if parameter.optimizer.is_none() {
                continue;
            }
            if let Some(master) = master_slot_definition(&entry.dtype) {
                parameter.slots.push(master.resolve(entry));
            }
        }
        plan
    }

    /// The same table, but with the owner of each parameter given explicitly
    /// instead of derived from this optimizer's own policy.
    pub fn plan_with(
        self,
        set: &TrainableSet,
        owner_of: impl Fn(&TrainableEntry) -> Option<Self>,
    ) -> OptimizerPlan {
        let mut parameters = Vec::with_capacity(set.entries.len());
        for entry in &set.entries {
            let owner = owner_of(entry);
            let slots = owner
                .map(|owner| {
                    owner
                        .slot_definitions()
                        .into_iter()
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
                    .into_iter()
                    .map(|slot| SharedSlot {
                        owner: *owner,
                        slot: slot.name,
                        dtype: slot.dtype,
                        n_elements: slot.shared_elements(),
                        n_bytes: slot.shared_elements().saturating_mul(slot.dtype.bytes()),
                    })
                    .collect::<Vec<_>>()
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
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizerDescriptor {
    /// The name a document writes and a checkpoint records.
    pub id: &'static str,
    /// The slot layout version, see [`OptimizerKind::layout_version`].
    pub layout_version: u32,
    pub slots: Vec<SlotDefinition>,
    pub shared_slots: Vec<SlotDefinition>,
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
    /// The parameter's own values, widened to the slot's dtype. Declared here
    /// but filled by the allocator: the definition has no bytes to give.
    Parameter,
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

/// The F32 master copy one half-precision parameter keeps, or `None` for a
/// parameter that is already its own master.
///
/// Not part of any optimizer's [`OptimizerKind::slot_definitions`]: a slot
/// table belongs to an optimizer, and this slot's existence depends on the
/// parameter's dtype. Mirrors `ggml_opt_master_slot`.
pub fn master_slot_definition(dtype: &TensorDtype) -> Option<SlotDefinition> {
    matches!(dtype, TensorDtype::F16 | TensorDtype::BF16).then_some(SlotDefinition {
        name: MASTER_SLOT,
        dtype: SlotDtype::F32,
        shape: SlotShape::Parameter,
        init: SlotInit::Parameter,
    })
}

/// The name the master copy is allocated, checkpointed and restored under, on
/// both sides of the boundary.
pub const MASTER_SLOT: &str = "master";

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
    /// Whether this parameter keeps an F32 master copy: a `master` row in its
    /// own slot list.
    pub fn keeps_master_copy(&self) -> bool {
        self.slots.iter().any(|slot| slot.slot == MASTER_SLOT)
    }

    /// The layout version of the slot table this parameter is allocated under,
    /// derived from [`OptimizerKind::layout_version_with_master`] so a row
    /// cannot disagree with it.
    pub fn layout_version(&self) -> Option<u32> {
        self.optimizer
            .map(|optimizer| optimizer.layout_version_with_master(self.keeps_master_copy()))
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
    /// Whether this run keeps an F32 master copy at all: true as soon as one
    /// parameter has one.
    pub fn keeps_master_copy(&self) -> bool {
        self.parameters
            .iter()
            .any(PlannedParameter::keeps_master_copy)
    }

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
    ///
    /// `live` is `(owner, slot, n_bytes)`, matched by identity in both
    /// directions like [`Self::check_live`]: the shared scope is allocated per
    /// owner in the optimizer graph's order, and restore matches on the same
    /// keys.
    pub fn check_live_shared(&self, live: &[(String, String, u64)]) -> Result<()> {
        let declared = self.shared_rows();
        for (owner, slot) in &declared {
            let found = live
                .iter()
                .find(|(live_owner, live_slot, _)| live_owner == owner && live_slot == slot.slot);
            let Some((_, _, live_bytes)) = found else {
                return Err(Error::runtime(format!(
                    "optimizer {} declares a shared '{}' slot for '{owner}' that the runtime \
                     did not allocate",
                    self.chosen, slot.slot
                )));
            };
            if slot.n_bytes != *live_bytes {
                return Err(Error::runtime(format!(
                    "shared optimizer slot {owner}/{} was declared as {} byte(s) and \
                     allocated as {live_bytes}",
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
                    "the runtime allocated a shared '{slot}' slot for '{owner}' that \
                     optimizer {} does not declare",
                    self.chosen
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

/// Gefen's shared rows. `block_size` and `min_numel` describe shapes and
/// thresholds, so they are integers; `block_size` is additionally a power of
/// two. `variant` is deliberately absent: it selects a layout rather than
/// parameterizing one, and [`OptimizerKind::layout_version`] is where it lands.
const GEFEN_SHARED_V_HYPERPARAMETERS: [HyperparameterDefinition; 8] = [
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

/// The quantized variant's rows: shared-v's, plus the codebook width, which is
/// the one key that describes state only this variant keeps.
const GEFEN_QUANTIZED_M_HYPERPARAMETERS: [HyperparameterDefinition; 9] = [
    GEFEN_SHARED_V_HYPERPARAMETERS[0],
    GEFEN_SHARED_V_HYPERPARAMETERS[1],
    GEFEN_SHARED_V_HYPERPARAMETERS[2],
    GEFEN_SHARED_V_HYPERPARAMETERS[3],
    GEFEN_SHARED_V_HYPERPARAMETERS[4],
    GEFEN_SHARED_V_HYPERPARAMETERS[5],
    HyperparameterDefinition::new(
        "codebook_levels",
        HyperparameterValue::Structural(GEFEN_CODEBOOK_LEVELS as i64),
        HyperparameterBound::AtLeast(2),
    ),
    GEFEN_SHARED_V_HYPERPARAMETERS[6],
    GEFEN_SHARED_V_HYPERPARAMETERS[7],
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
    fn every_declared_optimizer_has_an_update_step() {
        for kind in ALL {
            assert!(kind.is_implemented(), "{kind}");
        }
        assert_eq!(OptimizerKind::default(), OptimizerKind::AdamW);
    }

    #[test]
    fn the_wire_value_round_trips_for_what_the_runtime_can_build() {
        for kind in ALL {
            let variant = kind
                .gefen_layout()
                .map(|layout| layout.variant.as_ffi())
                .unwrap_or(0);
            assert_eq!(
                OptimizerKind::from_ffi(kind.as_ffi().unwrap(), variant).unwrap(),
                kind
            );
        }
        // The name alone does not say which slot table was allocated, so the
        // variant travels beside it rather than being guessed.
        assert_eq!(
            OptimizerKind::from_ffi(3, 1).unwrap(),
            OptimizerKind::Gefen(GefenLayout {
                variant: GefenVariant::QuantizedM,
                ..GefenLayout::default()
            })
        );
        assert!(OptimizerKind::from_ffi(3, 9).is_err());
        assert!(OptimizerKind::from_ffi(7, 0).is_err());
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

    /// The shared scope at a length above zero. Only Gefen declares a shared
    /// row, so its plan is the fixture.
    #[test]
    fn a_shared_table_is_compared_by_identity_like_the_parameter_one() {
        let two = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.1.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
        ]);
        let plan = quantized_m().plan(&two);
        // One codebook per owner, not per parameter.
        assert_eq!(plan.shared_rows().len(), 1);
        let live = |rows: &[(&str, &str, u64)]| -> Vec<(String, String, u64)> {
            rows.iter()
                .map(|(owner, slot, bytes)| (owner.to_string(), slot.to_string(), *bytes))
                .collect()
        };
        let codebook_bytes = GEFEN_CODEBOOK_LEVELS * 4;
        plan.check_live_shared(&live(&[("gefen", "codebook", codebook_bytes)]))
            .unwrap();
        let error = plan.check_live_shared(&[]).unwrap_err().to_string();
        assert!(error.contains("did not allocate"), "{error}");
        let error = plan
            .check_live_shared(&live(&[
                ("gefen", "codebook", codebook_bytes),
                ("gefen", "histogram", 64),
            ]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not declare"), "{error}");
        let error = plan
            .check_live_shared(&live(&[("gefen", "codebook", 4)]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("allocated as 4"), "{error}");
        // The owner is part of the key: another optimizer's row is a miss and
        // a surprise, never a match.
        let error = plan
            .check_live_shared(&live(&[("adamw", "codebook", codebook_bytes)]))
            .unwrap_err()
            .to_string();
        assert!(error.contains("did not allocate"), "{error}");
    }

    /// A mixed run's shared scope: the table can have two owners, so the
    /// comparison matches by name, not by position.
    #[test]
    fn a_shared_row_belongs_to_its_owner_and_not_to_the_chosen_optimizer() {
        // Below `min_numel`, so AdamW takes this one; AdamW declares no shared
        // row, leaving one owner with one.
        let mixed = set(vec![
            entry("blk.0.attn_q.weight", TensorRole::Base, [64, 64, 1, 1]),
            entry("blk.0.attn_norm.weight", TensorRole::Base, [64, 1, 1, 1]),
        ]);
        let plan = quantized_m().plan(&mixed);
        let owners: Vec<&str> = plan
            .parameters
            .iter()
            .filter_map(|parameter| parameter.optimizer.map(OptimizerKind::as_str))
            .collect();
        assert_eq!(owners, ["gefen", "adamw"]);
        let shared: Vec<(&str, &str)> = plan
            .shared_rows()
            .iter()
            .map(|(owner, slot)| (*owner, slot.slot))
            .collect();
        assert_eq!(shared, [("gefen", "codebook")]);
        // The shared bytes are in the total once, not once per parameter.
        let per_parameter: u64 = plan
            .parameters
            .iter()
            .flat_map(|parameter| parameter.slots.iter())
            .map(|slot| slot.n_bytes)
            .sum();
        assert_eq!(
            plan.state_bytes(),
            per_parameter + GEFEN_CODEBOOK_LEVELS * 4
        );
    }

    /// Under Muon an F16 factor falls back to an optimizer that writes F16;
    /// a dtype nothing writes is refused by name rather than substituted.
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
        // SGD writes F16, so it assigns and needs no fallback.
        assert_eq!(OptimizerKind::Sgd.assign(&factor), Some(OptimizerKind::Sgd));
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
        let plan = quantized_m().plan(&one);
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
        let plan = quantized_m().plan(&odd);
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
        assert!(shared_v().is_eligible(&factor));
    }

    /// Gefen under each of its two variants, spelled once.
    fn shared_v() -> OptimizerKind {
        OptimizerKind::Gefen(GefenLayout::default())
    }

    fn quantized_m() -> OptimizerKind {
        OptimizerKind::Gefen(GefenLayout {
            variant: GefenVariant::QuantizedM,
            ..GefenLayout::default()
        })
    }

    const ALL: [OptimizerKind; 5] = [
        OptimizerKind::AdamW,
        OptimizerKind::Sgd,
        OptimizerKind::Muon,
        OptimizerKind::Gefen(GefenLayout {
            variant: GefenVariant::SharedV,
            block_size: GEFEN_DEFAULT_BLOCK_SIZE,
            min_numel: GEFEN_DEFAULT_MIN_NUMEL,
        }),
        OptimizerKind::Gefen(GefenLayout {
            variant: GefenVariant::QuantizedM,
            block_size: GEFEN_DEFAULT_BLOCK_SIZE,
            min_numel: GEFEN_DEFAULT_MIN_NUMEL,
        }),
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
            // One layout each except Gefen's two, which is the whole point of
            // the variant: the same name over two slot tables.
            let expected = match kind.gefen_layout().map(|layout| layout.variant) {
                Some(GefenVariant::QuantizedM) => 2,
                _ => 1,
            };
            assert_eq!(descriptor.layout_version, expected, "{kind}");
        }
        // And the two Gefen descriptors differ in more than their version.
        assert_ne!(
            shared_v().descriptor().slots,
            quantized_m().descriptor().slots
        );
        assert!(shared_v().descriptor().shared_slots.is_empty());
    }

    #[test]
    fn a_value_outside_its_bound_is_refused_by_name() {
        let mut adamw = OptimizerKind::AdamW.declared_hyperparameters();
        let refused = adamw.set_scalar("beta1", 1.5).expect_err("out of range");
        assert!(refused.to_string().contains("beta1"), "{refused}");
        assert!(adamw.set_scalar("eps", 0.0).is_err());
        assert!(adamw.set_scalar("learning_rate", f32::NAN).is_err());
        assert!(adamw.set_scalar("weight_decay", 0.0).is_ok());

        let mut gefen = quantized_m().declared_hyperparameters();
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
        let mut gefen = quantized_m().declared_hyperparameters();
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
            shared_v(),
        ] {
            assert_eq!(OptimizerKind::parse(kind.as_str()).unwrap(), kind);
        }
        // "gefen" alone is the variant available first; the other is a key.
        assert_eq!(OptimizerKind::parse("gefen").unwrap(), shared_v());
        for variant in [GefenVariant::SharedV, GefenVariant::QuantizedM] {
            assert_eq!(GefenVariant::parse(variant.as_str()).unwrap(), variant);
        }
        assert!(GefenVariant::parse("learned").is_err());
        assert!(OptimizerKind::parse("lion").is_err());
    }
}
