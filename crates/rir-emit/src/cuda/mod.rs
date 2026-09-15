//! CUDA emitter: Loop IR to a `.cu` translation unit.
//!
//! Like the other emitters, this is a pure printer. What CUDA adds to the two
//! that came before is not a decision but a **second half of the file**: a
//! `__global__` kernel cannot be launched by its name from another translation
//! unit, so the generator also prints the `extern "C"` launch stub that owns the
//! `<<<>>>`. The stub is generated from the same `LoopKernel` as the
//! body, so the geometry it launches is the geometry the manifest publishes, by
//! construction.
//!
//! Two things it does **not** introduce, and both were paid for earlier: the
//! parameter ABI is the one `rir_kernel_params.h` already publishes - passed by
//! value as a launch argument instead of a push constant - and the
//! capability vocabulary is `KernelNeeds`, not a third list.
//!
//! The statement subset includes the collectives: `ParallelLane`, `LaneReduce`,
//! `LaneScan`, `LaneZero`, `WorkgroupReduce` and `Barrier` all print, the
//! two lane primitives through the warp prelude below. Nothing is approximated:
//! `__shfl_xor_sync` is the same butterfly topology as `subgroupAdd`, and
//! the shared-memory tree is the expansion the two other emitters already print.

pub mod addr;
pub mod expr;
pub mod stmts;
pub mod types;

#[cfg(test)]
mod tests;

use rir_core::Access;
use rir_lower::{LoopKernel, VarKind};

use crate::EmitError;
use crate::manifest::{artifact_name, kernel_needs, shader_params_layout};

/// Componentwise register vectors, printed only when the lowering uses them.
///
/// CUDA has `float4`, and no arithmetic on it: the vector operators GLSL and MSL
/// give for free do not exist here. The width is a **lowering** decision
/// (`vector_width`), so what the printer needs is a type it can instantiate at
/// any width rather than four hand-written ones - hence a template, and hence
/// the fact that this prelude names no width at all.
///
/// It is a header, not a decision: every operator below is the componentwise
/// meaning the oracle already gives these expressions, and nothing here chooses
/// between two ways of computing anything.
const VECTOR_PRELUDE: &str = r#"// Componentwise register vectors. CUDA's own `float4`
// carries no arithmetic, so the operators the two other backends get from their
// shading language are spelled out once, here, for any width.
template <int N> struct rir_vf { float c[N]; };
template <int N> struct rir_vb { bool  c[N]; };

template <int N> __device__ __forceinline__ rir_vf<N> rir_vsplat(float x) {
    rir_vf<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) { r.c[i] = x; }
    return r;
}
template <int N> __device__ __forceinline__ rir_vb<N> rir_vbsplat(bool x) {
    rir_vb<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) { r.c[i] = x; }
    return r;
}

#define RIR_VF_OP(op)                                                               \
    template <int N> __device__ __forceinline__                                     \
    rir_vf<N> operator op(rir_vf<N> a, rir_vf<N> b) {                               \
        rir_vf<N> r;                                                                \
        _Pragma("unroll") for (int i = 0; i < N; ++i) { r.c[i] = a.c[i] op b.c[i]; } \
        return r;                                                                   \
    }                                                                               \
    template <int N> __device__ __forceinline__                                     \
    rir_vf<N> operator op(rir_vf<N> a, float b) { return a op rir_vsplat<N>(b); }    \
    template <int N> __device__ __forceinline__                                     \
    rir_vf<N> operator op(float a, rir_vf<N> b) { return rir_vsplat<N>(a) op b; }
RIR_VF_OP(+)
RIR_VF_OP(-)
RIR_VF_OP(*)
RIR_VF_OP(/)
#undef RIR_VF_OP

#define RIR_VF_CMP(name, op)                                                        \
    template <int N> __device__ __forceinline__                                     \
    rir_vb<N> name(rir_vf<N> a, rir_vf<N> b) {                                      \
        rir_vb<N> r;                                                                \
        _Pragma("unroll") for (int i = 0; i < N; ++i) { r.c[i] = a.c[i] op b.c[i]; } \
        return r;                                                                   \
    }                                                                               \
    template <int N> __device__ __forceinline__                                     \
    rir_vb<N> name(rir_vf<N> a, float b) { return name(a, rir_vsplat<N>(b)); }      \
    template <int N> __device__ __forceinline__                                     \
    rir_vb<N> name(float a, rir_vf<N> b) { return name(rir_vsplat<N>(a), b); }
RIR_VF_CMP(rir_vgt, >)
RIR_VF_CMP(rir_vge, >=)
RIR_VF_CMP(rir_vlt, <)
RIR_VF_CMP(rir_vle, <=)
RIR_VF_CMP(rir_veq, ==)
RIR_VF_CMP(rir_vne, !=)
#undef RIR_VF_CMP

#define RIR_VF_MAP(name, fn)                                                  \
    template <int N> __device__ __forceinline__ rir_vf<N> name(rir_vf<N> a) { \
        rir_vf<N> r;                                                          \
        _Pragma("unroll") for (int i = 0; i < N; ++i) { r.c[i] = fn(a.c[i]); } \
        return r;                                                             \
    }
RIR_VF_MAP(rir_vsqrt, sqrtf)
RIR_VF_MAP(rir_vexp, expf)
RIR_VF_MAP(rir_vtanh, tanhf)
#undef RIR_VF_MAP

template <int N> __device__ __forceinline__ rir_vf<N> rir_vmax(rir_vf<N> a, rir_vf<N> b) {
    rir_vf<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) { r.c[i] = fmaxf(a.c[i], b.c[i]); }
    return r;
}
template <int N> __device__ __forceinline__
rir_vf<N> rir_vselect(rir_vf<N> f, rir_vf<N> t, rir_vb<N> c) {
    rir_vf<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) { r.c[i] = c.c[i] ? t.c[i] : f.c[i]; }
    return r;
}
"#;

/// The warp collectives, printed only when the lowering uses one.
///
/// The same reason the vector prelude exists: what CUDA offers is an
/// *instruction*, not an operator, so the one-line form the two other emitters
/// get from their shading language (`subgroupAdd`, `simd_sum`) has to be spelled
/// out once. Spelling it here rather than at each statement is what keeps
/// `Stmt::LaneReduce` printing as a single line on all three backends.
///
/// The width is 32 and it is written as a constant, not read from the schedule.
/// That is the point exactly: the two collective schedules assume a 32-lane
/// subgroup, NVIDIA's warp *is* 32, and the assumption is true here without
/// being justified by anything this file could check. A backend where it is
/// false - AMD in wave64 - would need the width to become a parameter, and that
/// is a different change from this one.
///
/// The reduction is a butterfly (`__shfl_xor_sync`), which is the topology
/// `subgroupAdd` has: every lane ends with the complete result, so the lowering's
/// assumption that the next dependency level sees the reduction in *every* lane
/// holds without a broadcast.
const WARP_PRELUDE: &str = r#"// Warp collectives. A butterfly over the 32 lanes
// of a warp: same topology as `subgroupAdd`/`simd_sum`, so every lane ends with
// the complete result and the accumulation order matches theirs.
__device__ __forceinline__ float rir_lane_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = v + __shfl_xor_sync(0xffffffffu, v, o, 32); }
    return v;
}
__device__ __forceinline__ float rir_lane_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) { v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o, 32)); }
    return v;
}

// The exclusive prefix, as `subgroupExclusiveAdd` defines it: lane `l` receives
// the combination of lanes `0..l`, and lane 0 the identity. Inclusive first -
// the Hillis-Steele ladder of `__shfl_up_sync` - then shifted up by one lane.
__device__ __forceinline__ float rir_lane_exclusive_sum(float v) {
    const uint32_t lane = threadIdx.x & 31u;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float n = __shfl_up_sync(0xffffffffu, v, o, 32);
        if (lane >= (uint32_t) o) { v = v + n; }
    }
    const float s = __shfl_up_sync(0xffffffffu, v, 1, 32);
    return lane == 0u ? 0.0f : s;
}
__device__ __forceinline__ float rir_lane_exclusive_max(float v) {
    const uint32_t lane = threadIdx.x & 31u;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const float n = __shfl_up_sync(0xffffffffu, v, o, 32);
        if (lane >= (uint32_t) o) { v = fmaxf(v, n); }
    }
    const float s = __shfl_up_sync(0xffffffffu, v, 1, 32);
    return lane == 0u ? __int_as_float(0xff800000) : s;
}
"#;

/// Unsigned division by a runtime divisor, printed only when the lowering
/// flattens its grid.
///
/// The same two lines `ggml-cuda/common.cuh` carries, and deliberately the same:
/// The native `k_bin_bcast_unravel` decomposes its linear index
/// with a host-precomputed multiplier, and a flattening measured against it must
/// pay the arithmetic it pays. The multiplier and the shift are constants of the
/// buffer, so the only device cost is `__umulhi`, an add and a shift.
///
/// The addition is 32-bit and it wraps for a large enough `n`; the contract
/// bounds the flattened total below 2^31, which is what makes that exact rather
/// than probable (`ggml_rir_flat_total`).
const FASTDIV_PRELUDE: &str = r#"// Magic-number division: the divisor's reciprocal is precomputed host side
// and passed in the constant buffer, so the decomposition of a flattened
// index costs a multiply-high instead of an integer division.
__device__ __forceinline__ uint32_t rir_fastdiv(uint32_t n, uint32_t mp, uint32_t sh) {
    return (__umulhi(n, mp) + n) >> sh;
}
__device__ __forceinline__ uint32_t rir_fastmod(uint32_t n, uint32_t mp, uint32_t sh, uint32_t d) {
    return n - rir_fastdiv(n, mp, sh) * d;
}
"#;

/// The CUDA printer: the shared skeleton, with CUDA's lexicon
/// (`crate::printer`, `crate::dialect`).
type CudaPrinter<'k> = crate::printer::Printer<'k, crate::dialect::Cuda>;

impl<'k> CudaPrinter<'k> {
    /// Whether any register of this lowering is a vector, which is what decides
    /// if the prelude above is printed. Read off the Loop IR rather than off the
    /// schedule's `vector_width`: what must be declared is what the body uses.
    fn uses_vectors(&self) -> bool {
        self.k
            .var_kinds
            .iter()
            .any(|k| matches!(k, VarKind::Vec(_) | VarKind::VecBool(_)))
    }
}

pub fn emit_cuda(k: &LoopKernel) -> Result<String, EmitError> {
    let mut p = CudaPrinter::new(k);
    let artifact = artifact_name(k);
    let needs = kernel_needs(k);
    let block = k.schedule.block();
    let threads: u32 = block.iter().product();

    p.line("// Generated by rir-gen - DO NOT EDIT MANUALLY.");
    p.line(&format!(
        "// Kernel: {} (backend cuda, reduction {}, scan {}, variante {}).",
        k.name,
        k.schedule.reduction().name(),
        k.schedule.scan().name(),
        k.schedule.variant().unwrap_or("repli")
    ));
    p.line(&format!(
        "// Strides are bytes in ggml order; parallel axes are mapped by {}.",
        k.schedule.par_map().name()
    ));
    p.line("#include <cstdint>");
    p.line("#include <cstring>");
    if needs.f16_mem {
        p.line("#include <cuda_fp16.h>");
    }
    // The params struct, from the header the registry already publishes
    //  The path is the one this file has in the fork,
    // `ggml/src/ggml-cuda/rir/` - the same relative spelling `norm.cu` uses one
    // directory up.
    p.line("#include \"../../ggml-rir/rir_kernel_params.h\"");
    p.line("");
    if p.uses_vectors() {
        p.out.push_str(VECTOR_PRELUDE);
        p.line("");
    }
    // Same rule again: the nest says whether the kernel divides, so a
    // non-flattened artifact carries none of this.
    // Not `flat_axes` alone: a linear-addressing variant is flattened and
    // divides nothing, so the prelude would be two
    // functions no line of the kernel calls.
    if !k.flat_axes().is_empty() && !k.linear_addr() {
        p.out.push_str(FASTDIV_PRELUDE);
        p.line("");
    }
    // Same rule as the vector prelude: read off the Loop IR, not off the
    // schedule. What has to be declared is what the body uses.
    if k.uses_subgroup() {
        p.out.push_str(WARP_PRELUDE);
        p.line("");
    }
    // Constant tables at module scope, prefixed by the artifact for the reason
    // Metal's are (metal/types.rs): nothing guarantees that a translation unit
    // will not one day see two artifacts.
    for lut in k.luts() {
        let values: Vec<String> = lut.values().iter().map(|v| format!("{v}.0f")).collect();
        p.line(&format!(
            "__device__ const float rir_{artifact}_{}[{}] = {{{}}};",
            lut.symbol(),
            values.len(),
            values.join(", ")
        ));
        p.line("");
    }

    // `__launch_bounds__` publishes to the compiler the block the stub below
    // launches: the two come from one `LoopKernel`, so a register allocation
    // tuned for another occupancy is not representable.
    p.line(&format!("__global__ __launch_bounds__({threads})"));
    p.line(&format!("static void rir_k_{artifact}("));
    p.indent = 2;
    for arg in k.args.iter() {
        let qualifier = match arg.access {
            Access::Read => "const uint8_t * __restrict__",
            Access::Write => "uint8_t * __restrict__",
        };
        p.line(&format!("{qualifier} {},", arg.name));
    }
    p.line(&format!("const rir_{artifact}_params p)"));
    p.indent = 0;
    p.line("{");
    p.indent = 1;
    // Shared storage, as the kernel declares it. It
    // is declared inside the function because that is where CUDA puts static
    // `__shared__`, which is also where Metal puts its threadgroup arrays.
    for (array, len) in &k.shared {
        let name = p.shared(*array);
        p.line(&format!("__shared__ float {name}[{len}];"));
    }
    let body = k.body.clone();
    p.stmts(&body)?;
    p.indent = 0;
    p.line("}");
    p.line("");

    // The launch stub. Three properties it gives for free: the geometry
    // launched is the manifest's, the params size is checked **at compile time**
    // of the fork rather than at runtime as on Vulkan, and the adapter only ever
    // holds a function pointer - so it cannot name a kernel.
    let params_bytes = 4 * shader_params_layout(k).len();
    p.line(&format!("extern \"C\" void rir_launch_{artifact}("));
    p.line("        void * const * bufs, const void * params, const uint32_t grid[3],");
    p.line("        cudaStream_t stream) {");
    p.indent = 1;
    p.line(&format!(
        "static_assert(sizeof(rir_{artifact}_params) == {params_bytes}, \
         \"registry: push_constant_bytes\");"
    ));
    p.line(&format!("rir_{artifact}_params p;"));
    p.line("std::memcpy(&p, params, sizeof p);");
    p.line(&format!(
        "rir_k_{artifact}<<<dim3(grid[0], grid[1], grid[2]), dim3({}, {}, {}), 0, stream>>>(",
        block[0], block[1], block[2]
    ));
    let bufs: Vec<String> = k
        .args
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let ty = match a.access {
                Access::Read => "const uint8_t *",
                Access::Write => "uint8_t *",
            };
            format!("({ty}) bufs[{i}]")
        })
        .collect();
    p.line(&format!("        {}, p);", bufs.join(", ")));
    p.indent = 0;
    p.line("}");

    Ok(p.out)
}
