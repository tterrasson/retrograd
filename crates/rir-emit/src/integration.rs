//! Integration contract between a RIR kernel and a ggml op (ADR-4 section 3).
//!
//! The semantic IR knows nothing about ggml; this metadata is what binds a
//! kernel to a `GGML_OP_*`, its `src[i]`/`dst` argument mapping, its
//! `op_params` scalars, and the per-backend variant policy. It deliberately
//! lives **outside** `rir_core::Kernel`: the same computation may back several
//! integrations, or none (oracle-only kernels).
//!
//! The table itself is declared in `rir-kernels` (`rir_kernels::integrations`)
//! next to the kernels; `rir-gen` zips both by kernel name and feeds this
//! struct to the manifest emitter (`schema_version` 9) and to the C++ registry
//! emitter (`RIR_REGISTRY_SCHEMA` 18).

/// Where a binding's buffer comes from at dispatch time.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArgSource {
    /// `ggml_tensor.src[i]`.
    Src(u8),
    /// The destination tensor itself.
    Dst,
}

impl ArgSource {
    /// Manifest spelling: `src[0]`, `src[1]`, … or `dst`.
    pub fn manifest_name(self) -> String {
        match self {
            ArgSource::Src(i) => format!("src[{i}]"),
            ArgSource::Dst => "dst".to_string(),
        }
    }
}

/// A scalar parameter read from `ggml_tensor.op_params` at a byte offset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ParamSpec {
    pub name: &'static str,
    pub op_params_offset: u32,
}

/// The ggml policy vocabulary, defined in the leaf crate and re-exported here.
///
/// `GgmlBackend` and `BackendPolicy` are what the **catalogue** publishes, and
/// the catalogue is read by a crate that shares nothing with this one but
/// `rir-core`. Defining them there and re-exporting them
/// here is the move `Backend`/`GpuBackend` already made: no call site
/// changes, and the generator stops translating one enum into another by hand.
/// The *tables* stay here - which backend a spec claims, what a policy prints
/// into the C registry - because that is emission, not vocabulary.
pub use rir_core::catalog::{BackendPolicy, GgmlBackend};

/// A rejection of the **portable contract** - the half of `supports_op` the
/// registry publishes and every backend evaluates identically.
///
/// The values are the `ggml_rir_reject` enum, which is ABI: the mask this
/// enum builds is compared against the counters a dispatch site records, and
/// two spellings of the same taxonomy is exactly what
/// `ggml_rir_reject_name` exists to prevent.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DomainRestriction {
    DType = 2,
    Rank = 3,
    Shape = 4,
    Stride = 5,
    QuantBlock = 6,
    IntegerRange = 7,
    /// The node names a **member of an op family** this kernel does not write
    /// (ADR-4 section 7).
    ///
    /// It exists because `GGML_OP_UNARY` is not one op: it is twenty-two
    /// functions behind one `ggml_op`, chosen by an integer in `op_params`. A
    /// kernel therefore claims a member the way it already claims a dtype, and
    /// a node whose member no kernel writes has to be refused for a reason a
    /// reader can act on. Filing it under `dtype` would have been a lie the lane
    /// could not catch; leaving it undeclared would have sent the node to the
    /// pair's shape-blind fallback, which computes **a different function**.
    ///
    /// Its value is 13 and not 8, and the gap is the ABI speaking: the fork's
    /// `ggml_rir_reject` interleaves the portable reasons (2–7) with the device
    /// ones (8–12), so a new contract reason appends rather than inserts.
    /// `rir-gen` checks the two tables against each other name by name.
    OpVariant = 13,
}

impl DomainRestriction {
    pub fn name(self) -> &'static str {
        match self {
            DomainRestriction::DType => "dtype",
            DomainRestriction::Rank => "rank",
            DomainRestriction::Shape => "shape",
            DomainRestriction::Stride => "stride",
            DomainRestriction::QuantBlock => "quant_block",
            DomainRestriction::IntegerRange => "integer_range",
            DomainRestriction::OpVariant => "op_variant",
        }
    }

    pub fn bit(self) -> u32 {
        1u32 << (self as u32)
    }
}

/// One part of the ggml domain of an op that the RIR kernel knowingly leaves
/// to the native kernel (ADR-4 section 6).
///
/// Declaring it is what turns a fallback from *observed* into *published*.
/// Without it, a kernel that started refusing on `stride` would look exactly
/// like one whose domain had always excluded strided views. With the
/// declaration, the two are different: a rejection in a category the registry names is the restriction
/// working, and a rejection in a category it does not name fails both lanes.
///
/// **The granularity is the category, and that is a real limit.** `DType`
/// declared for F16 also excuses a kernel that stops serving an F32 node;
/// `Shape` declared for broadcast also excuses a new shape rejection. This
/// enum names *which reason* is legitimate, never *which nodes*, so it cannot
/// by itself detect a narrowing inside a category it already lists. What
/// detects that is the claimed-node count, recorded per pair in
/// `scripts/rir-domain-baseline.tsv` and compared on every lane run: the matrix
/// is a fixed case list, so a narrowing of any kind makes the count drop. The
/// two checks are complementary and neither is sufficient - the mask says *why*
/// a fallback is legitimate, the baseline notices *how many* stopped being
/// served.
///
/// `why` is published, not decorative: a native kernel may be *kept* for a
/// restriction only if the restriction is stated, and a mask alone states
/// nothing a reader can act on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DomainAssumption {
    pub restriction: DomainRestriction,
    pub why: &'static str,
}

/// The explicit association between one RIR kernel and one ggml op.
#[derive(Clone, Debug)]
pub struct IntegrationSpec {
    /// `Kernel::name` this spec describes. Owned rather than `&'static str`
    /// because some kernels are generated per quantized format, so their names
    /// only exist once the canonical table has been read.
    pub kernel: String,
    /// Explicit ggml op. `None` marks an oracle-only kernel: no production
    /// mapping exists (or would be semantically wrong), and the manifest says
    /// so instead of deriving a nonexistent `GGML_OP_*` from the name.
    pub ggml_op: Option<&'static str>,
    /// The **member** of the op's family this kernel writes, spelled as its ggml
    /// enumerator (`"GGML_UNARY_OP_SILU"`), or `None` for an op that is not a
    /// family (ADR-4 section 7).
    ///
    /// Published as a spelling and not as a number, for the reason `ggml_op`
    /// already is: the numeric value belongs to ggml's header, and copying it
    /// here would create a second source of truth that compiles either way. The
    /// fork resolves it once against `ggml_unary_op_name`.
    pub ggml_op_variant: Option<&'static str>,
    /// Dispatch source of each kernel argument, in argument order.
    pub args: Vec<ArgSource>,
    /// Scalar params, in kernel param order.
    pub params: Vec<ParamSpec>,
    /// Whether any backend may run this kernel in production.
    pub production: bool,
    /// Per-backend policy. A backend absent from the list is `NativeOnly`.
    pub backend_policy: Vec<(GgmlBackend, BackendPolicy)>,
    /// The parts of the op's ggml domain this kernel does not claim, each with
    /// the reason it does not. Empty means the kernel claims the op entirely,
    /// and a single contract rejection at a dispatch site then contradicts the
    /// declaration - which is the point (ADR-4 section 6).
    ///
    /// It is a property of the *integration*, not of a backend: the portable
    /// contract is evaluated from the registry row before any device is asked,
    /// so Metal and Vulkan cannot legitimately differ on it.
    pub assumed_domain: Vec<DomainAssumption>,
    /// Backends whose **native kernel for this op no longer exists in the fork**
    /// (ADR-5 section 5).
    ///
    /// It is the second half of a promotion, and it is a different claim from
    /// `PreferGenerated`: preferring says the generated kernel runs when the
    /// contract matches, retiring says there is nothing else to run when it does
    /// not. So it may only be declared for a pair whose `assumed_domain` is
    /// empty - a pair with a published restriction keeps its native kernel *for
    /// that restriction*, which is exactly what is allowed, and what
    /// two pairs out of seven have. `rir-gen` refuses the combination rather than
    /// trusting the reader.
    ///
    /// What the fact changes for a consumer is that `off` is no longer a
    /// complete mode for this pair: `supports_op` answers on the RIR contract
    /// alone, so a node RIR declines leaves the backend for the CPU instead of
    /// reaching a native kernel that is gone. The lane reads the same field and
    /// stops asking for a differential it can no longer measure.
    pub retired_native: Vec<GgmlBackend>,
    /// Why the native kernel is **kept** on a pair that has nothing left to
    /// close (ADR-5 section 5).
    ///
    /// `assumed_domain` says which part of the op RIR declines; this says why
    /// the native survives when RIR declines *nothing*. The two are the same
    /// kind of statement - something RIR does not claim, published rather than
    /// observed - and they are separate fields because they answer to different
    /// checks: a domain restriction is compared against a dispatch site's
    /// rejection counters, and there is no counter for "the site never offered
    /// this node".
    ///
    /// That is exactly the case it exists for. Metal's `ADD` native serves the
    /// fused chains RIR cannot express: a generated variant computes one node in
    /// one dispatch, and no rejection is recorded when the encoder never asks.
    /// Without this field, such a pair is indistinguishable from a retirement
    /// nobody got round to - the forbidden third state: promoted, retirable, not
    /// retired, with no line to explain it. `rir-gen` refuses that state instead
    /// of leaving it to a reader.
    ///
    /// `Some` on a pair that also retires its native is a contradiction, and so
    /// is `Some` on a pair whose declared domain is non-empty: there the native
    /// is kept for the restriction, which is already published.
    pub native_exception: Option<&'static str>,
}

impl IntegrationSpec {
    pub fn policy_for(&self, backend: GgmlBackend) -> BackendPolicy {
        self.backend_policy
            .iter()
            .find(|(b, _)| *b == backend)
            .map(|(_, p)| *p)
            .unwrap_or(BackendPolicy::NativeOnly)
    }

    /// A backend carries this variant in its production build - i.e. it gets a
    /// registry row, a pipeline and a dispatch site - when the spec is
    /// production and its policy is not `NativeOnly`. Whether that site may
    /// *encode* the variant is the policy's second question, `dispatches_on`:
    /// an `ObserveGenerated` pair is registered and measured without ever
    /// replacing the native kernel.
    pub fn production_on(&self, backend: GgmlBackend) -> bool {
        self.production
            && self.ggml_op.is_some()
            && self.policy_for(backend) != BackendPolicy::NativeOnly
    }

    /// Whether the registered variant may actually run on `backend`.
    pub fn dispatches_on(&self, backend: GgmlBackend) -> bool {
        self.production_on(backend) && self.policy_for(backend).dispatches()
    }

    /// The declared domain as the bitmask the registry publishes and a
    /// dispatch site's `reject_by_reason` is checked against.
    pub fn assumed_domain_mask(&self) -> u32 {
        self.assumed_domain
            .iter()
            .fold(0, |m, a| m | a.restriction.bit())
    }

    /// Whether the native kernel of this op is gone from the fork on `backend`.
    pub fn native_retired_on(&self, backend: GgmlBackend) -> bool {
        self.retired_native.contains(&backend)
    }
}
