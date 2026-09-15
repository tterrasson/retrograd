//! The **dispatch plan**: one ggml op served by several kernels in order, with
//! a scratch buffer between them.
//!
//! Everything the RIR chain publishes so far describes *one* dispatch: a
//! manifest names one entrypoint, one grid, one binding list, and the runtime
//! executes it once. That is enough for every kernel in the registry, and it is
//! not enough for one shape of problem - a scan, or any global reduction, over
//! a space too small to occupy the device in one workgroup and too sequential to
//! cut without recombining. The answer is not a fourth scan strategy: it is a
//! sequence of dispatches with a barrier between them and a place to leave the
//! partial results.
//!
//! Such a design has to state what it publishes before any of it is written, and
//! the four items are the four fields below:
//!
//! - **ordered dispatches** - `passes`, executed in order, each naming the
//!   generated artifact it dispatches;
//! - **shape and lifetime of the scratch** - `Scratch`, whose extent is an
//!   expression over the op's own axes and whose `produced_by` / `last_read_by`
//!   bound the interval it is live over;
//! - **inter-dispatch barriers** - `Pass::barrier`, stated per pass rather than
//!   assumed, because "the runtime happens to serialize" is not a contract;
//! - **memory budget** - [`DispatchPlan::peak_scratch_bytes`], which is what a
//!   caller has to be able to compute *before* dispatching, from the shape.
//!
//! It is a **separate artifact from the manifest**, and deliberately: a plan
//! adds nothing to a kernel that does not have one, and folding these fields
//! into `Manifest` would bump the schema of the 234 generated manifests that
//! will never carry them. A pass is an ordinary generated kernel with an
//! ordinary manifest; the plan is what says in which order, over which shapes,
//! and through which buffers they run.

use serde::{Deserialize, Serialize};

use crate::backend::Backend;
use crate::manifest::DomainRestriction;
use crate::types::DType;

/// Schema version of the plan artifact, independent of the manifest's: the two
/// are read by the same runtime and written by the same emitter, but a change
/// to one is not a change to the other.
pub const PLAN_SCHEMA_VERSION: u32 = 1;

/// One factor of an extent or a stride, in the op's own vocabulary.
///
/// A plan cannot carry numbers: it is written once and dispatched on every
/// shape, so "the number of tiles" is `ceil(n_col / 256)` and not 4. What it
/// *can* carry is a constant - a tile width is a property of the plan, not of
/// the node.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Term {
    Const(u32),
    /// `ceil(n_<axis> / divide_by)`, the axis being one of the **op's** axes.
    Axis {
        axis: String,
        divide_by: u32,
    },
}

/// A product of terms: an axis extent of a pass, a stride in elements, or the
/// element count of a scratch buffer. One type for the three because they are
/// the same arithmetic, and because a reader who has understood one has
/// understood the others.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Extent {
    pub terms: Vec<Term>,
}

impl Extent {
    pub fn constant(n: u32) -> Extent {
        Extent {
            terms: vec![Term::Const(n)],
        }
    }

    pub fn axis(axis: &str) -> Extent {
        Extent {
            terms: vec![Term::Axis {
                axis: axis.to_string(),
                divide_by: 1,
            }],
        }
    }

    pub fn axis_tiles(axis: &str, tile: u32) -> Extent {
        Extent {
            terms: vec![Term::Axis {
                axis: axis.to_string(),
                divide_by: tile.max(1),
            }],
        }
    }

    pub fn times(mut self, other: Extent) -> Extent {
        self.terms.extend(other.terms);
        self
    }

    /// The value of this product for a node, given the op's axis extents.
    ///
    /// An unknown axis and an overflowing product are different contract
    /// failures, so they stay distinguishable all the way to the caller.
    pub fn eval(&self, axes: &dyn Fn(&str) -> Option<u64>) -> Result<u64, ExtentError> {
        let mut product: u64 = 1;
        for t in &self.terms {
            let v = match t {
                Term::Const(n) => u64::from(*n),
                Term::Axis { axis, divide_by } => axes(axis)
                    .ok_or_else(|| ExtentError::UnknownAxis(axis.clone()))?
                    .div_ceil(u64::from((*divide_by).max(1))),
            };
            product = product.checked_mul(v).ok_or(ExtentError::Overflow)?;
        }
        Ok(product)
    }
}

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ExtentError {
    #[error("unknown axis '{0}'")]
    UnknownAxis(String),
    #[error("extent product overflows u64")]
    Overflow,
}

/// Where a pass's binding gets its buffer.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// An argument of the op itself, by the name the plan gives it (`x`, `y`).
    Arg(String),
    /// A scratch buffer of this plan, by name.
    Scratch(String),
}

/// One binding of one pass: which buffer it reads or writes, and the strides
/// the pass sees it through.
///
/// The strides are the plan's, not the node's, and that is the whole trick a
/// multi-dispatch plan turns: the first pass reads the *same bytes* as the op's
/// input through a different shape - a row of `n_col` seen as `n_col / tile`
/// rows of `tile`. Nothing is copied and no new tensor exists; a stride list is
/// what a view is.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Binding {
    /// The binding's name **in the pass's own manifest**.
    pub name: String,
    pub source: Source,
    /// Strides in **elements**, innermost first, one per dimension the pass
    /// indexes. In elements and not in bytes because a plan is written once for
    /// a dtype it also declares; the runtime multiplies.
    pub strides: Vec<Extent>,
}

/// What must have completed before a pass starts.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Barrier {
    /// Nothing: the pass reads nothing a previous pass of this plan wrote.
    /// Only legitimate on the first pass, which is why `check` says so.
    None,
    /// Every write of every previous pass is visible to this one. The only
    /// other value today, and the honest one: the passes of a plan exist
    /// because each consumes what the last produced.
    Full,
}

/// One dispatch of the plan.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Pass {
    /// The generated artifact this pass dispatches - the name its manifest and
    /// its entrypoint carry.
    pub artifact: String,
    /// Extents of the pass's **own** axes, as expressions over the op's axes.
    /// The pass's manifest turns these into a grid; this is what makes the
    /// grid a function of the node's shape rather than of a number written
    /// twice.
    pub axes: Vec<(String, Extent)>,
    pub bindings: Vec<Binding>,
    pub barrier: Barrier,
}

/// Several dispatches serving one op.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DispatchPlan {
    pub plan_schema_version: u32,
    /// The plan's own name, distinct from any of its passes'.
    pub name: String,
    pub ggml_op: Option<String>,
    pub backend: Backend,
    /// What this plan does **not** claim. Never empty in practice: a plan that
    /// tiles an axis claims a relation between that axis and its tile, and a
    /// node that breaks it must be refused rather than over-read.
    pub assumed_domain: Vec<DomainRestriction>,
    pub scratch: Vec<Scratch>,
    pub passes: Vec<Pass>,
}

/// A buffer that exists only between two passes.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Scratch {
    pub name: String,
    pub dtype: DType,
    /// Elements, as an expression over the op's axes.
    pub elements: Extent,
    /// Index of the pass that writes it, and of the last pass that reads it.
    /// Together they are its **lifetime**, which is what makes a budget a peak
    /// rather than a sum.
    pub produced_by: usize,
    pub last_read_by: usize,
}

/// What is wrong with a plan, structurally - before any device is involved.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum PlanError {
    #[error("plan '{plan}': schema_version {got}; this build reads {PLAN_SCHEMA_VERSION}")]
    Schema { plan: String, got: u32 },
    #[error("plan '{plan}': no passes")]
    Empty { plan: String },
    #[error("plan '{plan}': pass {pass} binds '{binding}' to unknown {what} '{name}'")]
    UnknownSource {
        plan: String,
        pass: usize,
        binding: String,
        what: &'static str,
        name: String,
    },
    #[error(
        "plan '{plan}': scratch '{scratch}' is read by pass {read} but written by pass {written}"
    )]
    ReadBeforeWritten {
        plan: String,
        scratch: String,
        written: usize,
        read: usize,
    },
    #[error("plan '{plan}': pass {pass} reads what pass {producer} wrote, without a barrier")]
    MissingBarrier {
        plan: String,
        pass: usize,
        producer: usize,
    },
    #[error("plan '{plan}': scratch '{scratch}' has a lifetime outside the pass list")]
    LifetimeOutOfRange { plan: String, scratch: String },
    #[error("plan '{plan}': scratch '{scratch}' is declared more than once")]
    DuplicateScratch { plan: String, scratch: String },
    #[error(
        "plan '{plan}': scratch '{scratch}' is last read by pass {last_read} before it is produced by pass {produced}"
    )]
    InvalidLifetime {
        plan: String,
        scratch: String,
        produced: usize,
        last_read: usize,
    },
    #[error("plan '{plan}': the scratch budget overflows on this shape")]
    BudgetOverflow { plan: String },
    #[error("plan '{plan}': axis '{axis}' is not an axis of the op")]
    UnknownAxis { plan: String, axis: String },
}

impl DispatchPlan {
    /// Every structural property a plan must have for the runtime to execute it
    /// at all: passes exist, sources resolve, a scratch is written before it is
    /// read, and a pass that reads a previous pass's output carries a barrier.
    ///
    /// The last one is the property this whole artifact exists for, and it is
    /// checked rather than assumed: a plan whose second pass reads the first
    /// pass's scratch without a barrier is a race, and a race is not something
    /// a reader should have to notice.
    pub fn check(&self) -> Result<(), PlanError> {
        if self.plan_schema_version != PLAN_SCHEMA_VERSION {
            return Err(PlanError::Schema {
                plan: self.name.clone(),
                got: self.plan_schema_version,
            });
        }
        if self.passes.is_empty() {
            return Err(PlanError::Empty {
                plan: self.name.clone(),
            });
        }
        for (index, s) in self.scratch.iter().enumerate() {
            if self.scratch[..index]
                .iter()
                .any(|other| other.name == s.name)
            {
                return Err(PlanError::DuplicateScratch {
                    plan: self.name.clone(),
                    scratch: s.name.clone(),
                });
            }
            if s.produced_by >= self.passes.len() || s.last_read_by >= self.passes.len() {
                return Err(PlanError::LifetimeOutOfRange {
                    plan: self.name.clone(),
                    scratch: s.name.clone(),
                });
            }
            if s.last_read_by < s.produced_by {
                return Err(PlanError::InvalidLifetime {
                    plan: self.name.clone(),
                    scratch: s.name.clone(),
                    produced: s.produced_by,
                    last_read: s.last_read_by,
                });
            }
        }
        for (i, pass) in self.passes.iter().enumerate() {
            for b in &pass.bindings {
                match &b.source {
                    Source::Scratch(name) => {
                        let Some(s) = self.scratch.iter().find(|s| s.name == *name) else {
                            return Err(PlanError::UnknownSource {
                                plan: self.name.clone(),
                                pass: i,
                                binding: b.name.clone(),
                                what: "scratch",
                                name: name.clone(),
                            });
                        };
                        if i > s.produced_by && pass.barrier == Barrier::None {
                            return Err(PlanError::MissingBarrier {
                                plan: self.name.clone(),
                                pass: i,
                                producer: s.produced_by,
                            });
                        }
                        if i < s.produced_by {
                            return Err(PlanError::ReadBeforeWritten {
                                plan: self.name.clone(),
                                scratch: s.name.clone(),
                                written: s.produced_by,
                                read: i,
                            });
                        }
                    }
                    Source::Arg(_) => {}
                }
            }
        }
        Ok(())
    }

    /// Bytes of scratch live at the busiest point of the plan, for a node whose
    /// axes `axes` resolves.
    ///
    /// A **peak** and not a total, which is the entire reason `Scratch` carries
    /// a lifetime: two buffers whose intervals do not overlap can share bytes,
    /// and a budget that summed them would refuse plans that fit. The interval
    /// sweep is the same one the checkpoint profile performs over a backward
    /// graph.
    pub fn peak_scratch_bytes(&self, axes: &dyn Fn(&str) -> Option<u64>) -> Result<u64, PlanError> {
        let mut peak = 0u64;
        for at in 0..self.passes.len() {
            let mut live = 0u64;
            for s in &self.scratch {
                if at < s.produced_by || at > s.last_read_by {
                    continue;
                }
                let elements = match s.elements.eval(axes) {
                    Ok(elements) => elements,
                    Err(ExtentError::UnknownAxis(axis)) => {
                        return Err(PlanError::UnknownAxis {
                            plan: self.name.clone(),
                            axis,
                        });
                    }
                    Err(ExtentError::Overflow) => {
                        return Err(PlanError::BudgetOverflow {
                            plan: self.name.clone(),
                        });
                    }
                };
                let bytes = elements
                    .checked_mul(s.dtype.size_bytes() as u64)
                    .ok_or_else(|| PlanError::BudgetOverflow {
                        plan: self.name.clone(),
                    })?;
                live = live
                    .checked_add(bytes)
                    .ok_or_else(|| PlanError::BudgetOverflow {
                        plan: self.name.clone(),
                    })?;
            }
            peak = peak.max(live);
        }
        Ok(peak)
    }
}
