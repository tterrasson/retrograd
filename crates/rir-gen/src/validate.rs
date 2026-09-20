//! What the declared tables must satisfy before a byte is written.
//!
//! `generate_all` calls `validate_registry` first: an invalid table is a typed
//! error before emission, not a panic in the middle of it.

use crate::GenError;

/// The CUDA pairs the registry admits past `native_only`, each with the policy
/// it is admitted **up to** and the reason.
///
/// This list is the CUDA lock. Its purpose is that a promotion be a change to a
/// table someone has to read rather than a field edited in passing: two pairs
/// out of seven are promoted, and the five that are not stay refused by the
/// same mechanism.
///
/// A ceiling, not a value: the spec still declares its own policy, and this
/// says how far it may go. So admitting a pair for measurement
/// (`observe_generated`) does not admit it for dispatch, and raising it later is
/// a second, separate edit here.
///
/// Nothing else in the pipeline reads this list - it is not a policy, it is the
/// permission to hold one - which is why it lives with the validation and not
/// with the specs.
const CUDA_ADMITTED: &[(&str, rir_emit::BackendPolicy, &str)] = &[
    (
        "GGML_OP_L2_NORM_BACK",
        rir_emit::BackendPolicy::PreferGenerated,
        "the census gives it a single shape, [128,16,16,1], where the native CUDA kernel takes \
         its 32-thread branch - one row per warp, grid (16,16,1) - which is the geometry \
         `f32_4d_subgroup_tree` launches exactly. It is also one of the \
         two pairs whose native is already retired on Metal and Vulkan, so the code that \
         disappears here is code this project maintains alone",
    ),
    (
        "GGML_OP_RMS_NORM_BACK",
        rir_emit::BackendPolicy::PreferGenerated,
        "the other half of the same argument, with one difference the census found and \
         answered: the native switches on `ncols`, not on the row count, so the pair is admitted \
         with a CUDA-specific shape rule and a 1 024-lane variant rather than with Metal's rule",
    ),
    // The elementwise band and the unary family. They
    // entered this list at `ObserveGenerated` - a ceiling and not a value,
    // because the native CUDA kernels of this band are flat loops with
    // no address algebra to amortize and nothing here could be encoded before
    // the lane had published a ratio - and one of the four came out of the
    // measurement above it.
    (
        "GGML_OP_ADD",
        rir_emit::BackendPolicy::ObserveGenerated,
        "measured and **refused**, on one shape out of eight: `ne=[4096,1,1,1] nr=[1,512,1,1]` \
         - a row replayed 512 times - comes in at 1,10 while the seven non-repeating shapes are \
         0,84 to 1,03 (session `wisp-20260815T142310Z`). What loses is the repeat path, not the \
         flattening, so the ceiling stays where it is until that path is answered",
    ),
    (
        "GGML_OP_MUL",
        rir_emit::BackendPolicy::ObserveGenerated,
        "eight shapes out of eight between 0,86 and 0,98 in the same session, and refused all \
         the same: the shape that refuses `ADD` is a broadcast its twin `mul_repeat` serves \
         identically, and `MUL`'s own matrix does not contain one. Promoting here would certify \
         a path this lane never exercised, the false positive this list refuses",
    ),
    (
        "GGML_OP_SCALE",
        rir_emit::BackendPolicy::PreferGenerated,
        "the hardest case the census named, and the one that held: `scale_f32` is a flat \
         grid-stride loop of `dst[i] = scale·x[i] + bias`, block 256, with zero address algebra \
         to amortize - and the flattened dispatch measures 0,92 / 0,89 / 0,99 against it on the \
         three shapes of its matrix (session `wisp-20260815T142310Z`). The ceiling is raised \
         because the measurement raised it; the pair keeps its native kernel for the F16 \
         restriction it publishes",
    ),
    (
        "GGML_OP_UNARY",
        rir_emit::BackendPolicy::ObserveGenerated,
        "admitted on the **redundancy** and not on the traffic (0,39 % / 1,75 %): fourteen \
         native `unary_op_kernel` instantiations against one generated kernel. Measured and \
         refused - 1,03 to 1,12 over nine shapes, and the flat lowering already carries the \
         geometry that wins (`unary_flat_width_on_cuda`): what remains is the address, four \
         stride products where the native indexes a linear array",
    ),
];

/// Checks, **before any emission**, the rules linking three separately declared
/// tables: kernels (`rir_kernels::all`), their integration specs
/// (`rir_kernels::integrations`), and per-backend policies.
///
/// These properties once existed only under `#[test]`, so
/// `cargo run -p rir-gen` lacked the lane's guarantees: an invalid table caused
/// an indexing panic halfway through emission, or worse, a written artifact.
/// The binary and tests now call the same phase, whose rejections are
/// `GenError`s.
///
/// What it cannot cover stays elsewhere for a reason: lowering-dependent rules
/// (variant identity, POD capacities, grid geometry) are checked by
/// `rir_emit::emit_registry`, while actual existence of a `GGML_OP_*` in the
/// fork remains a test because it reads a header that may be absent.
///
/// # Errors
///
/// `GenError::Registry` on the first violation, naming the kernel at fault.
pub fn validate_registry() -> Result<(), GenError> {
    let entries = rir_kernels::registry();
    let kernels: Vec<rir_core::ValidatedKernel> =
        entries.iter().map(|e| e.kernel.clone()).collect();
    let specs: Vec<rir_emit::IntegrationSpec> =
        entries.iter().map(|e| e.integration.clone()).collect();
    validate_tables(&kernels, &specs)?;
    check_gpu_coverage(&entries)
}

/// The body of `validate_registry`, operating on supplied tables: this makes its
/// rejections testable without breaking the real tables.
///
/// # Errors
///
/// `GenError::Registry` on the first violation, naming the kernel at fault.
pub fn validate_tables(
    kernels: &[rir_core::ValidatedKernel],
    specs: &[rir_emit::IntegrationSpec],
) -> Result<(), GenError> {
    use rir_emit::{BackendPolicy, GgmlBackend, RegistryError};

    let err = |subject: &str, detail: String| {
        GenError::Registry(RegistryError {
            subject: subject.to_string(),
            detail,
        })
    };

    for spec in specs {
        if !kernels.iter().any(|k| k.name() == spec.kernel) {
            return Err(err(
                &spec.kernel,
                "orphan spec: no kernel with this name".into(),
            ));
        }
    }
    for kernel in kernels {
        let matching: Vec<&rir_emit::IntegrationSpec> =
            specs.iter().filter(|s| s.kernel == kernel.name()).collect();
        let [spec] = matching[..] else {
            return Err(err(
                kernel.name(),
                format!(
                    "{} IntegrationSpec entries, expected exactly one",
                    matching.len()
                ),
            ));
        };
        // A spec and its kernel are declarations joined by name: their arity and
        // parameter order are therefore a property to check, not an indexing
        // assumption.
        if spec.args.len() != kernel.args().len() {
            return Err(err(
                kernel.name(),
                format!(
                    "spec: {} arguments for {} in the kernel",
                    spec.args.len(),
                    kernel.args().len()
                ),
            ));
        }
        if spec.params.len() != kernel.params().len() {
            return Err(err(
                kernel.name(),
                format!(
                    "spec: {} params for {} in the kernel",
                    spec.params.len(),
                    kernel.params().len()
                ),
            ));
        }
        for (i, p) in kernel.params().iter().enumerate() {
            if spec.params[i].name != p.name {
                return Err(err(
                    kernel.name(),
                    format!(
                        "spec parameter order != kernel: {} instead of {}",
                        spec.params[i].name, p.name
                    ),
                ));
            }
        }
        if !spec.production {
            continue;
        }
        // A production variant without an op would be a registry row no
        // dispatcher can select.
        if spec.ggml_op.is_none() {
            return Err(err(
                kernel.name(),
                "production without an explicit ggml_op".into(),
            ));
        }
        // CPU remains native in this slice: promoting a RIR variant there must
        // be a visible table change, never a side effect.
        if spec.policy_for(GgmlBackend::Cpu) != BackendPolicy::NativeOnly {
            return Err(err(kernel.name(), "cpu must remain NativeOnly".into()));
        }
        // CUDA has the same lock, and it **enumerates
        // instead of forbidding**. What the blanket refusal guaranteed was that
        // a CUDA promotion is a visible table change; a list of admitted pairs
        // guarantees the same thing one pair at a time, which is what the plan
        // asked for and what a phase promoting two pairs out of seven needs.
        if spec.policy_for(GgmlBackend::Cuda) != BackendPolicy::NativeOnly {
            let op = spec.ggml_op.unwrap_or("");
            let Some((_, ceiling, _)) = CUDA_ADMITTED.iter().find(|(o, _, _)| *o == op) else {
                return Err(err(
                    kernel.name(),
                    format!(
                        "{op} is not in CUDA_ADMITTED: a CUDA policy above native_only is a \
                         line in that list, not a field edited on its own"
                    ),
                ));
            };
            if spec.policy_for(GgmlBackend::Cuda) > *ceiling {
                return Err(err(
                    kernel.name(),
                    format!(
                        "{op} is admitted on CUDA up to {}, not {}",
                        ceiling.name(),
                        spec.policy_for(GgmlBackend::Cuda).name()
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// The check prevents impossible backend coverage from reaching emission.
///
/// The schedule table and the integration table are two declarations joined by
/// a kernel name, and this compares their **backend coverage**. Without it,
/// three things go wrong, all silent:
///
/// 1. a kernel reaching the default arm of `schedules_for` got the CPU schedule
///    alone. Legitimate for an oracle-only kernel, and for a production one it
///    meant a policy row saying `prefer_generated` with no artifact to prefer;
/// 2. a kernel scheduled on one GPU backend and not on its neighbour - the
///    fourteen `push` pairs disagreeing - kept CPU-only coverage there;
/// 3. a schedule on a backend `GPU_REFUSED` names would reach an emitter that
///    does not exist, as a `GenError::NoEmitter` halfway through emission.
///
/// None of the three broke a test: they made a test tautological, which is the
/// regression mode this check exists for. They are generation errors now,
/// raised before the first byte is written.
///
/// The input is the **registration**, so each kernel names its family directly.
/// The remaining coverage failure is a family arm that schedules only the CPU,
/// and that is what the check reports.
///
/// # Errors
///
/// `GenError::Registry` naming the kernel and the backend at fault.
pub fn check_gpu_coverage(entries: &[rir_kernels::KernelRegistration]) -> Result<(), GenError> {
    let err = |subject: &str, detail: String| {
        GenError::Registry(rir_emit::RegistryError {
            subject: subject.to_string(),
            detail,
        })
    };

    for entry in entries {
        let (kernel, schedules) = (&entry.kernel, &entry.schedules);
        let on =
            |gpu: rir_lower::GpuBackend| schedules.iter().any(|s| s.backend() == gpu.backend());

        // (3) A refused backend carries nothing. Checked first: it is the one
        // failure that would otherwise surface as an emitter that is missing
        // rather than a table that is wrong.
        for (gpu, why) in rir_lower::GPU_REFUSED {
            if on(*gpu) {
                return Err(err(
                    kernel.name(),
                    format!("schedule on {}, which the table refuses: {why}", gpu.name()),
                ));
            }
        }

        // (3b) The same, per family: a refusal this family wrote down is a
        // refusal, not a preference. `targets_for` already skips the backend,
        // so a schedule on it could only come from an arm that names the
        // backend itself.
        for gpu in rir_lower::GPU_TARGETS {
            if let Some(why) = rir_lower::family_refusal(entry.family, *gpu)
                && on(*gpu)
            {
                return Err(err(
                    kernel.name(),
                    format!(
                        "schedule on {}, which the {} family refuses: {why}",
                        gpu.name(),
                        entry.family.name()
                    ),
                ));
            }
        }

        // (2) Coverage is all or nothing across the backends this family is
        // scheduled on. A kernel may be GPU-less, but a family may not be
        // GPU-less on one backend by omission: selective refusal belongs in
        // `FAMILY_REFUSED`, which is already reflected by `targets`.
        let targets: Vec<rir_lower::GpuBackend> = rir_lower::targets_for(entry.family).collect();
        let covered: Vec<rir_lower::GpuBackend> =
            targets.iter().copied().filter(|g| on(*g)).collect();
        if !covered.is_empty() && covered.len() != targets.len() {
            let missing: Vec<&str> = targets
                .iter()
                .filter(|g| !on(**g))
                .map(|g| g.name())
                .collect();
            return Err(err(
                kernel.name(),
                format!(
                    "scheduled on {} but not on {} - a GPU backend is covered by \
                     every arm of the table, refused in writing by the family, or \
                     by none",
                    covered
                        .iter()
                        .map(|g| g.name())
                        .collect::<Vec<_>>()
                        .join(", "),
                    missing.join(", ")
                ),
            ));
        }

        // (1) And the join with the policy table, in both directions the pair
        // admits: a backend that carries this variant in production needs a
        // lowering for it, and a lowering claimed by no spec at all is already
        // refused above as an orphan.
        let spec = &entry.integration;
        for gpu in rir_lower::GPU_TARGETS {
            let ggml = rir_emit::manifest::ggml_backend_of(gpu.backend());
            if spec.production_on(ggml) && !on(*gpu) {
                return Err(err(
                    kernel.name(),
                    format!(
                        "policy {} on {} but no schedule for that backend - the \
                         registry row would name an artifact the table never emits",
                        spec.policy_for(ggml).name(),
                        gpu.name()
                    ),
                ));
            }
        }
    }
    Ok(())
}
