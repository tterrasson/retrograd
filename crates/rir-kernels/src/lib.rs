//! RIR kernels described once in the DSL.
//!
//! Each file defines one kernel's mathematics, axes, and contract. Schedules
//! live in `rir-lower` and emitters elsewhere; what lives **here** is the
//! registry the whole pipeline reads: `registry()` keeps each kernel's identity,
//! family, schedules and integration spec together, in one entry. A kernel
//! without a family cannot be registered.

pub mod cumsum;
pub mod elementwise;
mod integration;
pub mod l2_norm_back;
pub mod l2_norm_fwd;
pub mod mat_mul_naive;
pub mod out_prod;
pub mod rms_norm;
pub mod rms_norm_back;
pub mod scan_plan;
pub mod sum_rows_quant;
pub mod unary;

use rir_core::{QuantType, ValidatedKernel};
use rir_emit::{
    ArgSource, BackendPolicy, DomainAssumption, DomainRestriction, GgmlBackend, IntegrationSpec,
};
use rir_lower::{Family, Schedule};

/// The identity of a generatable kernel: a closed type, not a name.
///
/// Families generated from a canonical table carry their member - a quantized
/// format, a `UNARY` op, an elementwise band member - so `sum_rows_q4_K` is a
/// *value* of this type rather than a string that happens to start with
/// `sum_rows_`. Adding a format to `QUANT_FORMATS` still adds kernels without a
/// line here, and adding a **family** now requires naming its schedule arm.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelId {
    L2NormBack,
    /// One per lowerable quantized format.
    SumRowsQuant(QuantType),
    L2NormFwd,
    /// Derived from `L2NormFwd` by transposition (stage D), never handwritten.
    L2NormFwdGrad,
    Cumsum,
    MatMulNaive,
    /// One per `src0` dtype the compiler can read: `None` is F32.
    OutProd(Option<QuantType>),
    RmsNormBack,
    RmsNorm,
    Unary(unary::Unary),
    Elementwise(elementwise::Band),
}

/// Every kernel identity, in the order generation walks them. That order
/// controls emission order - file layout, the Metal aggregate, the Vulkan
/// artifact list - not contracts.
pub fn ids() -> Vec<KernelId> {
    let mut v = vec![KernelId::L2NormBack];
    // One quantized kernel per lowerable format, straight from the canonical
    // table - adding a format adds its kernel.
    v.extend(
        sum_rows_quant::variants()
            .into_iter()
            .map(KernelId::SumRowsQuant),
    );
    v.extend([
        KernelId::L2NormFwd,
        KernelId::L2NormFwdGrad,
        KernelId::Cumsum,
        KernelId::MatMulNaive,
    ]);
    // One `out_prod` per `src0` dtype the compiler can read, straight from the
    // canonical table - F32 first, then the quantized weights of a LoRA backward.
    v.extend(out_prod::variants().into_iter().map(KernelId::OutProd));
    v.extend([
        KernelId::RmsNormBack,
        // The **forward** counterpart of `RmsNormBack`.
        // It shares the schedule table of its backward counterpart, whose exact
        // problem shape it has.
        KernelId::RmsNorm,
    ]);
    // The `UNARY` family: one kernel per canonical-table member, like the
    // elementwise strip and quantized reductions. The association is explicit.
    v.extend(unary::variants().into_iter().map(KernelId::Unary));
    // The elementwise band, one kernel per ggml op it covers, straight from
    // its canonical table - same shape as the quantized reductions above:
    // adding a member adds its kernel.
    v.extend(
        elementwise::variants()
            .into_iter()
            .map(KernelId::Elementwise),
    );
    v
}

impl KernelId {
    /// The emitted name of this kernel: generated directory, artifact prefix,
    /// entrypoint. One function, so the name is a *rendering* of the identity
    /// and never the other way round.
    pub fn name(self) -> String {
        match self {
            KernelId::L2NormBack => "l2_norm_back".to_string(),
            KernelId::SumRowsQuant(f) => sum_rows_quant::kernel_name(f),
            KernelId::L2NormFwd => "l2_norm_fwd".to_string(),
            KernelId::L2NormFwdGrad => "l2_norm_fwd_grad".to_string(),
            KernelId::Cumsum => "cumsum".to_string(),
            KernelId::MatMulNaive => "mat_mul_naive".to_string(),
            KernelId::OutProd(f) => out_prod::kernel_name(f),
            KernelId::RmsNormBack => "rms_norm_back".to_string(),
            KernelId::RmsNorm => "rms_norm".to_string(),
            KernelId::Unary(m) => m.kernel_name(),
            KernelId::Elementwise(b) => b.kernel_name(),
        }
    }

    /// The scheduling family this kernel belongs to - the arm of
    /// `rir_lower::schedules_for` that decides its lowerings.
    ///
    /// This is the declaration that replaces three name prefixes and a default
    /// arm. It is written per identity, so a new kernel cannot acquire a
    /// schedule table by being named like its neighbour, and cannot lose one by
    /// being named unlike them.
    pub fn family(self) -> Family {
        match self {
            // One row per workgroup with subgroup reduction: one parallel axis
            // and an inner reduction. The quantized reductions qualify since
            // both GPU emitters can load F16 and bytes.
            KernelId::L2NormBack
            | KernelId::L2NormFwd
            | KernelId::L2NormFwdGrad
            | KernelId::SumRowsQuant(_) => Family::RowReduce,
            KernelId::RmsNorm | KernelId::RmsNormBack => Family::RmsNorm,
            KernelId::Cumsum => Family::Cumsum,
            KernelId::MatMulNaive => Family::MatMulNaive,
            KernelId::OutProd(_) => Family::OutProd,
            KernelId::Unary(_) => Family::Unary,
            KernelId::Elementwise(b) => match b.op {
                elementwise::BandOp::Scale => Family::Scale,
                elementwise::BandOp::AddRepeat | elementwise::BandOp::MulRepeat => {
                    Family::ElementwiseRepeat
                }
                elementwise::BandOp::Add | elementwise::BandOp::Mul => Family::Elementwise,
            },
        }
    }

    /// Builds the kernel. `l2_norm_fwd_grad` is **derived** by transposition
    /// (stage D) and validated against the handwritten kernel and numerical
    /// gradients in the `l2_norm_fwd` tests.
    ///
    /// # Panics
    ///
    /// If a kernel of the registry does not validate. That is a table defect,
    /// not an input: every caller here is the generator building its own
    /// declarations.
    pub fn build(self) -> ValidatedKernel {
        let named = |r: Result<ValidatedKernel, rir_core::ValidateError>| {
            r.unwrap_or_else(|e| panic!("{} : {e}", self.name()))
        };
        match self {
            KernelId::L2NormBack => named(l2_norm_back::build()),
            KernelId::SumRowsQuant(f) => named(sum_rows_quant::build(f)),
            KernelId::L2NormFwd => named(l2_norm_fwd::build()),
            KernelId::L2NormFwdGrad => {
                let fwd = named(l2_norm_fwd::build());
                rir_core::derive_backward(&fwd).expect("autodiff l2_norm_fwd")
            }
            KernelId::Cumsum => named(cumsum::build()),
            KernelId::MatMulNaive => named(mat_mul_naive::build()),
            KernelId::OutProd(f) => named(out_prod::build_for(f)),
            KernelId::RmsNormBack => named(rms_norm_back::build()),
            KernelId::RmsNorm => named(rms_norm::build()),
            KernelId::Unary(m) => named(unary::build(m)),
            KernelId::Elementwise(b) => named(elementwise::build(b)),
        }
    }
}

/// One kernel, complete: its identity, its graph, its scheduling family, the
/// schedules that family decides, and its integration spec.
///
/// The registration is one value, so `registry()` cannot produce a kernel with
/// a missing schedule or integration specification.
pub struct KernelRegistration {
    pub id: KernelId,
    pub kernel: ValidatedKernel,
    pub family: Family,
    /// The family's schedules: CPU plus one per GPU target per lowering.
    pub schedules: Vec<Schedule>,
    pub integration: IntegrationSpec,
}

/// The canonical registry: one entry per kernel, in generation order.
pub fn registry() -> Vec<KernelRegistration> {
    ids()
        .into_iter()
        .map(|id| {
            let family = id.family();
            KernelRegistration {
                id,
                kernel: id.build(),
                family,
                schedules: rir_lower::schedules_for(family),
                integration: id.integration(),
            }
        })
        .collect()
}

/// Registry of generatable kernels, in generation order. A projection of
/// `registry()`, kept because most consumers want the graphs alone.
pub fn all() -> Vec<ValidatedKernel> {
    registry().into_iter().map(|r| r.kernel).collect()
}

/// The integration table: one explicit entry per kernel
/// of `all()`, **in the same order**, because both are projections of one
/// registry. `ggml_op: None` marks an oracle-only kernel - the
/// association is never derived from the kernel's name, because that derivation
/// produced ops that do not exist in ggml (`GGML_OP_L2_NORM_FWD`,
/// `GGML_OP_SUM_ROWS_Q8`).
pub fn integrations() -> Vec<IntegrationSpec> {
    registry().into_iter().map(|r| r.integration).collect()
}

/// The spec for one kernel, if declared.
pub fn integration_for(kernel: &str) -> Option<IntegrationSpec> {
    integrations().into_iter().find(|s| s.kernel == kernel)
}

/// The shape restriction shared by the non-repeating `ADD` and `MUL` members.
///
/// `ggml_add`/`ggml_mul` allow `src1` to be *repeated* over `src0`
/// (`ggml_can_repeat`). Expressing it takes index arithmetic - a modulo to
/// replay a row - which the DSL has (`RepeatIndex`, `read_repeat`) and which
/// `out_prod`'s broadcast below still does not use. The restriction is not
/// written by hand anywhere: on *this* kernel `a`, `b` and `dst` are indexed by
/// the *same* axes, and the axis agreement of `supports_op` derives the equality
/// of the three shapes from that.
///
/// This is therefore a restriction of **one kernel of the pair**, not
/// of the op: `add_repeat`/`mul_repeat` claim `ggml_can_repeat` and this one
/// does not, and the op's policy row carries only what *none* of its kernels
/// claims. Kept declared because it is true of this
/// kernel, and because the vectorized lowering it protects is the one carrying
/// the totality of the measured traffic.
const BROADCAST_BAND: DomainAssumption = DomainAssumption {
    restriction: DomainRestriction::Shape,
    // Shared by ADD and MUL, so the figure names its op: on the measured
    // backward graph the restriction cost MUL 280 of 920 nodes and ADD nothing
    // at all. The same declaration, two very different bills - and the 280 are
    // what `mul_repeat` serves.
    why: "ggml_can_repeat on src1: served by add_repeat/mul_repeat, not this \
          variant - selection uses node shape. Real graph: \
          280/920 nodes on MUL, none on ADD",
};

/// A spec with everything an oracle-only kernel leaves empty. `GGML_OP_SUM_ROWS`
/// keeps the input dtype and these kernels sum a quantized tensor into F32, so
/// no exact ggml mapping exists - but each one is still declared, because the
/// rule is that no kernel escapes the table, not that every kernel maps to an
/// op.
fn oracle_only(kernel: String, args: Vec<ArgSource>) -> IntegrationSpec {
    IntegrationSpec {
        kernel,
        ggml_op_variant: None,
        ggml_op: None,
        args,
        params: vec![],
        production: false,
        backend_policy: vec![],
        assumed_domain: vec![],
        retired_native: vec![],
        native_exception: None,
    }
}

/// The four-row policy every promoted pair carries: CPU stays native in this
/// slice, both GPU backends run the generated variant, and CUDA keeps its
/// native kernel unless a second table admits it (`CUDA_ADMITTED`, in
/// `rir-gen`).
fn gpu_policy(gpu: BackendPolicy) -> Vec<(GgmlBackend, BackendPolicy)> {
    cuda_policy(gpu, BackendPolicy::NativeOnly)
}

/// The same four rows with CUDA stated rather than assumed.
///
/// It is a separate constructor and not a third argument to `gpu_policy`
/// because the asymmetry is the point: sixteen of the eighteen production specs
/// have nothing to say about CUDA and must keep saying nothing, while the two
/// pairs CUDA promotes say it in one place each. A default parameter would have made
/// the sixteen carry a value they never chose.
fn cuda_policy(gpu: BackendPolicy, cuda: BackendPolicy) -> Vec<(GgmlBackend, BackendPolicy)> {
    vec![
        (GgmlBackend::Cuda, cuda),
        (GgmlBackend::Vulkan, gpu),
        (GgmlBackend::Metal, gpu),
        (GgmlBackend::Cpu, BackendPolicy::NativeOnly),
    ]
}

impl KernelId {
    /// This kernel's integration spec, written next to its identity rather than
    /// in a second table joined by name.
    ///
    /// **`integer_range` is deliberately declared nowhere below**, and that is
    /// the decision, not an omission. Generated shaders
    /// address bytes in `index_bits = 32`, so a binding whose largest byte
    /// offset exceeds 4 GiB leaves through the portable contract. Widening the
    /// field would cost every push constant of every kernel; *declaring* the
    /// restriction would excuse the rejection forever, including the day it is a
    /// real addressing bug. Leaving it undeclared turns 32 bits into a claim and
    /// makes both lanes fail the day it stops holding - the only reading of "no
    /// silent rejection" that produces evidence rather than a note. Neither the
    /// matrix nor the measured backward graph produces one today.
    pub fn integration(self) -> IntegrationSpec {
        use ArgSource::{Dst, Src};

        let name = self.name();
        match self {
            KernelId::L2NormBack => integration::l2_norm_back(name),
            // Non-equivalent to GGML_OP_L2_NORM: the native kernel scales by
            // 1/max(sqrt(Σx²), eps); this one has no epsilon floor. Autodiff
            // bench only.
            KernelId::L2NormFwd => oracle_only(name, vec![Src(0), Dst]),
            // Derived without the epsilon branch: no safe direct mapping.
            KernelId::L2NormFwdGrad => oracle_only(name, vec![Src(0), Src(1), Dst]),
            KernelId::SumRowsQuant(_) => oracle_only(name, vec![Src(0), Dst]),
            KernelId::Cumsum => integration::cumsum(name),
            KernelId::MatMulNaive => integration::mat_mul_naive(name),
            KernelId::OutProd(format) => integration::out_prod(name, format),
            KernelId::RmsNormBack => integration::rms_norm_back(name),
            KernelId::RmsNorm => integration::rms_norm(name),
            KernelId::Unary(member) => integration::unary(name, member),
            KernelId::Elementwise(band) => integration::elementwise(name, band),
        }
    }
}
