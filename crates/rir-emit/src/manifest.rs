//! Manifest JSON: the contract shared by generated code and the C++ wrapper
//! (ADR-4 section 4). Binding order, push-constant layout, and schema version are
//! **stable**; changing them must fail the regeneration-diff test.
//!
//! What this file holds is the *derivation* - entrypoint, variant id, push
//! layout, grid geometry, required features - read off the Loop IR. The schema
//! itself is `rir_core::manifest`, shared with the runtime that reads it, and
//! the JSON is `serde_json`'s: field order is the declaration order of
//! `rir_core::manifest::Manifest`, so it stays a contract without a `push_str`
//! per field.

use rir_core::manifest::{self, Determinism, Manifest};
use rir_core::{Constraint, ScalarType};
use rir_lower::{Backend, HwLevel, LoopKernel, MemType, Stmt};

use crate::integration::{GgmlBackend, IntegrationSpec};

/// The symbol a backend asks its compiler/loader for. GLSL modules expose a
/// single `main`; MSL entrypoints share ggml's Metal library namespace, so
/// they carry a `rir_` prefix to make a collision with a native kernel
/// impossible; the CPU function is the kernel name itself.
pub fn entrypoint(k: &LoopKernel) -> String {
    match k.schedule.backend() {
        Backend::Vulkan => "main".to_string(),
        Backend::Metal => format!("rir_{}", artifact_name(k)),
        // A CUDA `__global__` is not launched by name:
        // what the fork resolves is the generated `extern "C"` stub, so the
        // entrypoint the manifest and the registry publish is that stub.
        Backend::Cuda => format!("rir_launch_{}", artifact_name(k)),
        Backend::Cpu => artifact_name(k),
    }
}

/// The name every per-variant artifact of this lowering is built from: the
/// source file, the fork's shader file, the SPIR-V symbol, the MSL entrypoint.
/// The kernel name for a pair's fallback variant, `kernel_variant` otherwise.
///
/// One function so those four names cannot drift: adding a variant renames
/// nothing that already exists, because the fallback keeps the bare kernel name.
pub fn artifact_name(k: &LoopKernel) -> String {
    k.schedule.artifact_name(&k.name)
}

/// The two file names this lowering occupies under `generated/rir/<kernel>/`:
/// its source, and its manifest.
///
/// The fallback keeps the variant-free names - `kernel.comp`,
/// `manifest.vulkan.json` - so adding a variant renames nothing on disk or in
/// the fork. A named variant inserts its name, which is
/// what stops two lowerings of one kernel from overwriting each other: the
/// concrete failure that kept the blocked scan out of the AOT pipeline.
pub fn artifact_files(k: &LoopKernel) -> (String, String) {
    // `""` for the fallback, `".<variant>"` otherwise: one infix, inserted at
    // the same place in both names.
    let infix = k
        .schedule
        .variant()
        .map(|v| format!(".{v}"))
        .unwrap_or_default();
    let source = match k.schedule.backend() {
        Backend::Cpu => format!("cpu{infix}.rs"),
        Backend::Cuda => format!("kernel{infix}.cu"),
        Backend::Vulkan => format!("kernel{infix}.comp"),
        Backend::Metal => format!("kernel{infix}.metal"),
    };
    (
        source,
        format!("manifest{infix}.{}.json", k.schedule.backend().name()),
    )
}

/// Stable variant identity, distinct from the dtype: the same kernel emitted
/// under a different schedule (a different reduction strategy, a different scan
/// strategy, a wider domain) must produce a different id, so registries and
/// telemetry can tell them apart without parsing names.
pub fn variant_id(k: &LoopKernel) -> String {
    let rank = k.args.iter().map(|a| a.ty.rank).max().unwrap_or(0);
    let dtype = k.args.first().map(|a| a.ty.dtype.name()).unwrap_or("f32");
    // The kernel name leads. A `variant_id` identifies a variant *within its
    // ggml op*, and an op can have
    // carries several kernels: two for a repeat, two for a dtype, fifteen for
    // the unary family. Fifteen members of `GGML_OP_UNARY` share a dtype, a
    // rank and a schedule, so without the name they would share an id - and the
    // lane's per-variant counters, which are exactly what proves the
    // arbitration ran on a device, would count them as one.
    let base = format!(
        "{}_{dtype}_{rank}d_{}",
        k.name,
        k.schedule.reduction().name()
    );
    match k.schedule.variant() {
        // A named variant already spells what distinguishes it - `vec4` says
        // the width in the name the artifact carries.
        Some(v) => format!("{base}_{v}"),
        // The pair's fallback has no name, so a width above one has to reach
        // the id some other way: the lane prints these ids to say *which*
        // lowering ran, and a vectorized fallback reading as `..._serial` would
        // be indistinguishable from the scalar one it replaced.
        None if k.vector_width() > 1 => format!("{base}_w{}", k.vector_width()),
        None => base,
    }
}

/// The ggml backend a schedule's code runs on, for policy lookups.
pub fn ggml_backend(k: &LoopKernel) -> GgmlBackend {
    ggml_backend_of(k.schedule.backend())
}

/// The same mapping on a bare `Backend`, for callers that have a schedule
/// target rather than a lowered kernel - `rir_gen::check_gpu_coverage` compares
/// the schedule table against the policy table before anything is lowered.
///
/// Written once because it is the whole content of "two enums for the same four
/// devices": one is a scheduling domain, the other a policy domain, and a second
/// copy of the correspondence is exactly the duplication this guards against.
pub fn ggml_backend_of(backend: Backend) -> GgmlBackend {
    match backend {
        Backend::Vulkan => GgmlBackend::Vulkan,
        Backend::Metal => GgmlBackend::Metal,
        Backend::Cuda => GgmlBackend::Cuda,
        Backend::Cpu => GgmlBackend::Cpu,
    }
}

/// GPU parameter layout, shared by GLSL, MSL, and their manifests: kernel
/// parameters, then axis extents, then each argument's byte strides. Every
/// entry occupies four bytes at sequential offsets.
pub fn shader_params_layout(k: &LoopKernel) -> Vec<(String, &'static str)> {
    let mut out = Vec::new();
    for p in &k.params {
        let ty = match p.ty {
            ScalarType::F32 => "float",
        };
        out.push((p.name.clone(), ty));
    }
    for a in &k.axes {
        out.push((format!("n_{}", a.name), "uint"));
    }
    for arg in &k.args {
        for d in 0..arg.ty.rank {
            out.push((format!("{}_nb{}", arg.name, d), "uint"));
        }
    }
    // The flattened dispatch's own block, last so that nothing before it moves.
    // It exists only on a flattened lowering, which is
    // why the header this feeds is emitted per **artifact** and not per kernel:
    // `add` and `add_flat` are one kernel and two constant layouts.
    let flat = k.flat_axes();
    if !flat.is_empty() {
        out.push((FLAT_TOTAL.to_string(), "uint"));
        // One triple per divisor, and there is one fewer divisor than axes: the
        // last index is the quotient itself, so nothing divides by its extent.
        //
        // None at all when the shader does not decompose:
        // a reciprocal nobody divides by would be a constant the host computes,
        // the buffer carries and no invocation reads - the same silent field the
        // rest of this pipeline refuses.
        if !k.linear_addr() {
            for i in 0..flat.len() - 1 {
                for (name, _) in flat_divisor_names(i) {
                    out.push((name, "uint"));
                }
            }
        }
    }
    out
}

/// Name of the constant carrying the number of points a flattened dispatch
/// covers - `Π ceil(extent / per_index)` - which is also the bound an
/// invocation past the end tests against.
pub const FLAT_TOTAL: &str = "rir_flat_total";

/// The three constants of one flattened divisor: the divisor itself, the magic
/// multiplier, and the shift. Named in one place
/// because five consumers spell them - the layout above, three emitters, and
/// the host that fills them.
pub fn flat_divisor_names(i: usize) -> [(String, &'static str); 3] {
    [
        (format!("rir_flat{i}_div"), "divisor"),
        (format!("rir_flat{i}_mp"), "magic multiplier"),
        (format!("rir_flat{i}_sh"), "shift"),
    ]
}

/// The three names alone, in layout order.
pub fn flat_divisor(i: usize) -> [String; 3] {
    let n = flat_divisor_names(i);
    [n[0].0.clone(), n[1].0.clone(), n[2].0.clone()]
}

fn backend_features(k: &LoopKernel) -> Option<Vec<String>> {
    let needs = kernel_needs(k);
    match k.schedule.backend() {
        Backend::Vulkan => Some(needs.vulkan()),
        Backend::Metal => Some(needs.metal()),
        Backend::Cuda => Some(needs.cuda()),
        Backend::Cpu => None,
    }
}

/// What a lowered kernel **needs from a device**, in no backend's vocabulary.
///
/// It is derived from the Loop IR - the collective it carries, the memory types
/// it reads, the shared storage and the block it declares - and each backend
/// *renders* it in its own words: Vulkan names extensions, Metal a language
/// version, CUDA a compute capability and a shared-memory budget. One source,
/// three renderings.
///
/// The structure contains kernel requirements rather than backend vocabulary:
/// Vulkan, Metal, and CUDA render the same facts as their own capabilities.
/// Keeping this distinction here prevents a backend-specific list from becoming
/// a second source of truth.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KernelNeeds {
    /// Reads or writes 16-bit floating point in memory (quantized scales, F16
    /// tensors).
    pub f16_mem: bool,
    /// Reads bytes in memory (quantized payloads).
    pub i8_mem: bool,
    /// Uses a lane collective - the reduction or the scan primitive the
    /// hardware offers over a subgroup/warp/SIMD-group.
    pub collective: bool,
    /// Lanes the collective spans, or 0 when there is none.
    pub lanes: u32,
    /// Bytes of workgroup-shared storage the kernel declares.
    pub shared_bytes: u32,
    /// Threads in one workgroup.
    pub block_threads: u32,
}

pub fn kernel_needs(k: &LoopKernel) -> KernelNeeds {
    let mut needs = KernelNeeds {
        block_threads: k.schedule.block().iter().product(),
        // Four bytes per element: every shared array the Loop IR declares is a
        // `float` array, on all three backends.
        shared_bytes: k.shared.iter().map(|(_, len)| 4 * len).sum(),
        ..KernelNeeds::default()
    };
    // The width the nest needs, not the block it runs on: the two coincide on
    // every lowering but the hierarchical reduction, which runs a 256-lane
    // workgroup on 32-lane subgroups.
    if let Some(width) = k.subgroup_width() {
        needs.collective = true;
        needs.lanes = width;
    }
    for tys in k.mem_types() {
        needs.f16_mem |= tys.contains(&MemType::F16);
        // Both byte access types are the same need: reading a byte.
        needs.i8_mem |= tys.contains(&MemType::I8) || tys.contains(&MemType::U8);
    }
    needs
}

impl KernelNeeds {
    /// Vulkan's rendering: extension and version names, in manifest order.
    pub fn vulkan(self) -> Vec<String> {
        VulkanCaps::from(self).features()
    }

    /// Metal's rendering. `simd_sum`/`simd_max` are macOS Metal 2.1 and later
    /// (iOS 2.3), per the MSL specification; everything else the emitter prints
    /// compiles under Metal 2.0. Same shape as the Vulkan 1.0/1.1 split, and for
    /// the same reason: the collective is the cost.
    pub fn metal(self) -> Vec<String> {
        let mut features = vec![
            if self.collective {
                "metal>=2.1"
            } else {
                "metal>=2.0"
            }
            .to_string(),
        ];
        if self.collective {
            features.push("simdgroup_reduction".to_string());
            features.push(format!("simdgroup_size>={}", self.lanes));
        }
        features
    }

    /// CUDA's rendering, and it is short on purpose:
    /// there is no extension to negotiate. `__half` and `int8_t` are types of
    /// the language - half arithmetic is `sm_53`, byte access is universal - and
    /// `__shfl_xor_sync` is `sm_30` with `_sync` since CUDA 9. What a device can
    /// still fail to provide is a *budget*: shared memory per block and threads
    /// per block, which is what the device half of the contract checks.
    pub fn cuda(self) -> Vec<String> {
        let cc = if self.f16_mem { "5.3" } else { "5.0" };
        let mut features = vec![format!("cuda>={cc}")];
        if self.shared_bytes > 0 {
            features.push(format!("shared_bytes>={}", self.shared_bytes));
        }
        features.push(format!("block_threads>={}", self.block_threads));
        features
    }
}

impl From<KernelNeeds> for VulkanCaps {
    fn from(n: KernelNeeds) -> Self {
        VulkanCaps {
            subgroup: n.collective,
            min_subgroup: n.lanes,
            f16: n.f16_mem,
            i8: n.i8_mem,
        }
    }
}

/// Vulkan's rendering of `KernelNeeds`: what the GLSL emitter turns into
/// `#extension` directives and the manifest into features checked before
/// pipeline creation. It is a **view**, not a second
/// derivation - `kernel_needs` is the one that reads the Loop IR.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct VulkanCaps {
    /// Subgroup collectives (`subgroupAdd`) require SPIR-V 1.3, hence Vulkan
    /// 1.1, and a subgroup at least as large as the block.
    pub subgroup: bool,
    /// Minimum required subgroup size, or 0 when no collective is used.
    pub min_subgroup: u32,
    /// 16-bit memory access for quantization scales.
    pub f16: bool,
    /// 8-bit memory access for quantized data bytes.
    pub i8: bool,
}

pub fn vulkan_caps(k: &LoopKernel) -> VulkanCaps {
    kernel_needs(k).into()
}

impl VulkanCaps {
    /// Features in manifest order.
    pub fn features(self) -> Vec<String> {
        let mut v = vec![
            if self.subgroup {
                "vulkan>=1.1"
            } else {
                "vulkan>=1.0"
            }
            .to_string(),
        ];
        if self.subgroup {
            v.push("subgroup_arithmetic".to_string());
            v.push(format!("subgroup_size>={}", self.min_subgroup));
        }
        if self.f16 {
            v.push("storage_buffer_16bit".to_string());
            v.push("shader_float16".to_string());
        }
        if self.i8 {
            v.push("storage_buffer_8bit".to_string());
            v.push("shader_int8".to_string());
        }
        v
    }
}

/// Grid-mapped axes in x, y, z order, with the number of indices covered by a
/// workgroup on each axis. Read **from the loop nest**, not reconstructed from
/// the schedule: the manifest describes what the shader does, not what was
/// requested.
pub(crate) fn grid_axes(k: &LoopKernel) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let mut stmts = &k.body;
    while let [
        Stmt::Parallel {
            axis,
            level,
            vector,
            body,
            ..
        },
    ] = stmts.as_slice()
    {
        let d = match level {
            HwLevel::Grid(d) | HwLevel::Global(d) => *d as usize,
            HwLevel::Lane => break,
        };
        // An invocation covering `vector` consecutive indices covers that many
        // more per workgroup, so the grid is that much smaller. Read from the
        // nest for the same reason as the rest of this function: the dispatcher
        // must launch what the shader was printed for, not what was requested.
        let per = k.schedule.par_map().per_workgroup(k.schedule.block(), d) * (*vector).max(1);
        out.push((k.axes[axis.0 as usize].name.clone(), per));
        stmts = body;
    }
    out
}

/// The flattened axes in decomposition order, with the number of indices one
/// invocation covers on each. The named twin of `grid_axes`, read from the same
/// place and for the same reason.
pub(crate) fn flat_axes(k: &LoopKernel) -> Vec<(String, u32)> {
    k.flat_axes()
        .into_iter()
        .map(|(ax, per)| (k.axes[ax.0 as usize].name.clone(), per))
        .collect()
}

fn constraint_str(k: &LoopKernel, c: &Constraint) -> String {
    match c {
        Constraint::DType { arg, allowed } => {
            let names: Vec<&str> = allowed.iter().map(|d| d.name()).collect();
            format!("dtype({})={}", k.args[arg.0 as usize].name, names.join("|"))
        }
        // A maximum, not an equality: trailing extents of 1 lower the
        // effective rank (see `Constraint::Rank`).
        Constraint::Rank { arg, max } => {
            format!("rank({})<={}", k.args[arg.0 as usize].name, max)
        }
        Constraint::Contiguous { arg } => {
            format!("contiguous({})", k.args[arg.0 as usize].name)
        }
    }
}

/// The manifest of one lowered kernel, as a value.
///
/// Building the value and printing it are two steps since:
/// the schema is `rir_core::manifest`, shared with the runtime that reads it, so
/// a field added here is a field the reader sees at compile time. The value is
/// built as a struct literal, leaving serialization and escaping to the schema
/// type.
pub fn build_manifest(k: &LoopKernel, spec: Option<&IntegrationSpec>) -> Manifest {
    if let Some(s) = spec {
        assert_eq!(s.kernel, k.name, "IntegrationSpec for the wrong kernel");
        assert_eq!(
            s.args.len(),
            k.args.len(),
            "{}: incomplete src/dst mapping ({} arguments declared, {} mapped)",
            k.name,
            k.args.len(),
            s.args.len()
        );
        assert_eq!(
            s.params.len(),
            k.params.len(),
            "{}: incomplete op_params mapping",
            k.name
        );
    }

    let bindings = k
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| manifest::Binding {
            name: a.name.clone(),
            binding: i as u32,
            // `source` is the dispatch-time origin of the buffer: src[i]/dst per
            // the integration spec, or `None` for an oracle-only kernel.
            source: spec.map(|s| s.args[i].manifest_name().to_string()),
            dtype: a.ty.dtype,
            access: a.access,
            // `extents` names the axis indexing each ggml dimension of the
            // binding, in order. With it, and the strides already in the push
            // constants, a caller knows exactly which byte range a dispatch
            // addresses on this buffer - which is what turns "the kernel ran"
            // into "the kernel stayed in bounds".
            extents: k.arg_axes[i]
                .iter()
                .map(|slot| slot.map(|ax| k.axes[ax.0 as usize].name.clone()))
                .collect(),
        })
        .collect();

    let params = k
        .params
        .iter()
        .enumerate()
        .map(|(i, p)| manifest::Param {
            name: p.name.clone(),
            ty: p.ty,
            source: spec.map(|_| manifest::ParamSource::OpParams),
            source_offset: spec.map(|s| s.params[i].op_params_offset),
        })
        .collect();

    // CPU passes scalars as function arguments. Shader backends record the
    // complete constant-buffer layout: parameters, extents, and strides.
    //
    // CUDA joins the two shader backends here rather than the CPU, and that is
    // the whole of it: its params are a launch argument
    // instead of a push constant, but the **layout** is the one
    // `rir_kernel_params.h` already publishes. No new ABI is introduced by CUDA.
    let push: Vec<(String, &'static str)> = match k.schedule.backend() {
        Backend::Vulkan | Backend::Metal | Backend::Cuda => shader_params_layout(k),
        Backend::Cpu => k
            .params
            .iter()
            .map(|p| {
                let ty = match p.ty {
                    ScalarType::F32 => "float",
                };
                (p.name.clone(), ty)
            })
            .collect(),
    };
    let push_constants = push
        .iter()
        .enumerate()
        .map(|(i, (name, ty))| manifest::PushConstant {
            name: name.clone(),
            ty: match *ty {
                "float" => manifest::PushType::F32,
                "int" => manifest::PushType::I32,
                _ => manifest::PushType::U32,
            },
            offset: 4 * i as u32,
        })
        .collect();

    // Dispatch metadata computes workgroup counts without replaying lowering.
    // There is one axis per grid dimension in x, y, z order; `per_workgroup`
    // is the number of axis indices covered by one workgroup, so
    // grid[d] = ceil(n_<axis> / per_workgroup).
    let dispatch = grid_axes(k)
        .into_iter()
        .map(|(axis, per_workgroup)| manifest::GridAxis {
            axis,
            per_workgroup,
        })
        .collect();
    // The flattened dispatch, when there is one. It and
    // `dispatch` are exclusive by construction - `grid_axes` reads the nest,
    // and a flattened nest has no `Stmt::Parallel` - so a reader that honours
    // both fields cannot compute two grids, and one that honours neither
    // dispatches nothing rather than dispatching a wrong geometry.
    //
    // `per_index` is what one *invocation* covers on that axis, so the divisor
    // is `ceil(extent / per_index)`: the vector width on the contiguous axis,
    // one everywhere else. The grid is then `ceil(Π divisors / workgroup[0])`.
    let flat = flat_axes(k)
        .into_iter()
        .map(|(axis, per_index)| manifest::FlatAxis { axis, per_index })
        .collect();

    Manifest {
        schema_version: manifest::SCHEMA_VERSION,
        name: k.name.clone(),
        entrypoint: entrypoint(k),
        // Explicit, never derived from the kernel name: `None` marks an
        // oracle-only kernel with no production ggml mapping.
        ggml_op: spec.and_then(|s| s.ggml_op).map(str::to_string),
        variant_id: variant_id(k),
        production: spec.is_some_and(|s| s.production_on(ggml_backend(k))),
        // The parts of the ggml op this kernel does not claim, with the reason
        // for each. Empty means it claims the op entirely - and then a single
        // contract rejection at a dispatch site contradicts this manifest
        // (ADR-4 section 6).
        assumed_domain: spec
            .map(|s| {
                s.assumed_domain
                    .iter()
                    .map(|a| manifest::DomainRestriction {
                        reject: a.restriction.name().to_string(),
                        // The reasons are written across source lines; the
                        // manifest is one string per reason, not a transcript of
                        // the indentation that produced it.
                        why: a.why.split_whitespace().collect::<Vec<_>>().join(" "),
                    })
                    .collect()
            })
            .unwrap_or_default(),
        backend: k.schedule.backend(),
        bindings,
        params,
        push_constants,
        packed_output: None,
        constraints: k.constraints.iter().map(|c| constraint_str(k, c)).collect(),
        determinism: Determinism::strictest(k.reduction_semantics.iter().copied()),
        reduction: k.schedule.reduction().published(),
        scan: k.schedule.scan().published(),
        // Which of the pair's variants this is, and the shapes it claims. `None`
        // with an empty rule list is the fallback: it accepts everything, which
        // is what makes per-shape selection total.
        variant: k.schedule.variant().map(str::to_string),
        priority: k.schedule.priority(),
        vector_width: k.vector_width(),
        linear_addr: k.linear_addr(),
        eligible_when: k
            .schedule
            .eligible_when()
            .iter()
            .map(|r| manifest::ShapeRule {
                axes: r.axes.iter().map(|a| (*a).to_string()).collect(),
                min: r.min,
                max: r.max,
            })
            .collect(),
        features: backend_features(k).unwrap_or_default(),
        parallel_mapping: k.schedule.par_map().published(),
        dispatch,
        flat,
        workgroup: k.schedule.block(),
    }
}

/// The manifest as the generated file carries it: pretty JSON, one trailing
/// newline, fields in the declaration order of `rir_core::manifest::Manifest`,
/// which is the contract (ADR-4 section 4).
pub fn emit_manifest(k: &LoopKernel, spec: Option<&IntegrationSpec>) -> String {
    let m = build_manifest(k, spec);
    let mut out = serde_json::to_string_pretty(&m).expect("a manifest is plain data");
    out.push('\n');
    out
}
