//! What each family of `Stmt` prints as, in CUDA.
//!
//! The expectation of every test below is the emitter's **exact** output for the
//! same hand-built nests the two other emitters are pinned on, so a change in
//! the printed form is a diff in this file - and a divergence between the three
//! printings of one statement is three diffs in three files rather than a
//! fixture to compare by eye.
//!
//! One thing is tested here that has no equivalent next door: the launch stub,
//! which is the half of a `.cu` that has no counterpart in a `.comp` or a
//! `.metal`. The launch stub is the only CUDA-specific surface covered here;
//! the remaining statements share the same pinned printer cases as the other
//! backends.

use crate::pin_probes;
use crate::testkit::*;

/// The warp prelude follows what the body uses, exactly as the vector one does:
/// a nest whose only collective is the shared-memory tree does not pull in a
/// single `__shfl_*_sync`.
#[test]
fn the_warp_prelude_follows_what_the_body_uses() {
    assert!(lanes().cu_unit().contains("rir_lane_exclusive_sum"));
    assert!(!collectives().cu_unit().contains("__shfl_xor_sync"));
    assert!(!loops().cu_unit().contains("__shfl_xor_sync"));
}

/// The half of the file that has no counterpart on the two other backends:
/// the launch stub.
///
/// Three properties are asserted here because all three are what makes the stub
/// worth generating. The geometry it launches is the schedule's, so it cannot
/// disagree with the manifest. The params size is checked by a `static_assert`,
/// so an ABI drift is a compilation error of the fork rather than a runtime
/// check as on Vulkan. And it is `extern "C"`, so the adapter holds a pointer
/// and never a name.
#[test]
fn the_launch_stub_carries_the_schedules_geometry() {
    let unit = loops().cu_unit();
    assert!(
        unit.contains("__global__ __launch_bounds__(64)\nstatic void rir_k_probe("),
        "{unit}"
    );
    assert!(
        unit.contains(
            "extern \"C\" void rir_launch_probe(\n\
             \x20       void * const * bufs, const void * params, const uint32_t grid[3],\n\
             \x20       cudaStream_t stream) {"
        ),
        "{unit}"
    );
    assert!(
        unit.contains(
            "    static_assert(sizeof(rir_probe_params) == 28, \"registry: push_constant_bytes\");"
        ),
        "{unit}"
    );
    assert!(
        unit.contains(
            "    rir_k_probe<<<dim3(grid[0], grid[1], grid[2]), dim3(64, 1, 1), 0, stream>>>(\n\
             \x20           (const uint8_t *) bufs[0], (uint8_t *) bufs[1], p);"
        ),
        "{unit}"
    );
}

/// The vector prelude is printed **only** when the lowering uses vectors, and
/// `cuda_fp16.h` only when a binding is read as F16. Both are read off the Loop
/// IR rather than off the schedule: what has to be declared is what the body
/// uses.
#[test]
fn the_prelude_follows_what_the_body_uses() {
    let scalar = loops().cu_unit();
    assert!(!scalar.contains("template <int N> struct rir_vf"));
    assert!(!scalar.contains("cuda_fp16.h"));

    let vectorized = vectors().cu_unit();
    assert!(vectorized.contains("template <int N> struct rir_vf"));
    // No F16 binding in that nest: the half header is not pulled in by the
    // vectors alone.
    assert!(!vectorized.contains("cuda_fp16.h"));

    // `memory()` reads one binding as F16 and no vector at all - the two
    // conditions are independent, and this is where that shows.
    let half = memory().cu_unit();
    assert!(half.contains("#include <cuda_fp16.h>"));
    assert!(!half.contains("template <int N> struct rir_vf"));
}

pin_probes!(cu, {
    /// Loop forms: the grid mapping, a reversed sequential axis, a constant count
    /// and the tiled step.
    Loops as loops_print_as => r#"
const uint32_t r = blockIdx.x;
if (r >= p.n_row) {
    return;
}
for (uint32_t c = p.n_col; c-- > 0u;) {
    const float t = 0.5f;
}
for (uint32_t k = 0u; k < 4u; ++k) {
}
for (uint32_t k0 = 0u; k0 < p.n_col; k0 += 8u) {
}
"#,
    /// Memory: an F32 load, an F16 load and its conversion, a store under a bounds
    /// check. Byte addresses throughout - a CUDA binding is a `uint8_t *`, so unlike
    /// GLSL there is no element index to divide by.
    Memory as memory_print_as => r#"
const uint32_t row_i = blockIdx.x * blockDim.x + threadIdx.x;
const float val = *(const float *)(x + (row_i * p.x_nb1 + col_i * p.x_nb0 + 4u));
const float half = __half2float(*(const __half *)(x + (col_i * 2u)));
if (row_i < p.n_row) {
    *(float *)(y + (row_i * p.y_nb1)) = val;
}
"#,
    /// The vector lowering: a widened body, its scalar tail, and the per-component
    /// bounded write.
    ///
    /// This is where CUDA differs most from the two other backends, and not by a
    /// decision: `float4` carries no arithmetic, so the componentwise comparison and
    /// select go through the prelude's templates instead of a language operator.
    Vectors as vectors_print_as => r#"
const uint32_t col_i = (blockIdx.x * blockDim.x + threadIdx.x) * 4u;
if (col_i + 4u <= p.n_col) {
    const uint32_t v4_i = col_i * p.x_nb0;
    const rir_vf<4> v4 = {{*(const float *)(x + (v4_i)), *(const float *)(x + (v4_i + 4u)), *(const float *)(x + (v4_i + 8u)), *(const float *)(x + (v4_i + 12u))}};
    const rir_vb<4> p4 = rir_vge(v4, v4);
    const rir_vf<4> w4 = rir_vselect(v4, v4, p4);
    const uint32_t w4_o = col_i * p.y_nb0;
    if (col_i + 4u <= p.n_col) {
        *(float *)(y + (w4_o)) = w4.c[0];
        *(float *)(y + (w4_o + 4u)) = w4.c[1];
        *(float *)(y + (w4_o + 8u)) = w4.c[2];
        *(float *)(y + (w4_o + 12u)) = w4.c[3];
    } else {
        for (uint32_t c = 0u; c + col_i < p.n_col; ++c) {
            *(float *)(y + (w4_o + c * p.y_nb0)) = w4.c[c];
        }
    }
} else {
    for (uint32_t tail = col_i; tail < p.n_col; ++tail) {
        *(float *)(y + (tail * p.y_nb0)) = s;
    }
}
"#,
    /// Expressions: one line per `LExpr` family.
    ///
    /// Every float literal carries its `f`, and every transcendental its single
    /// precision spelling: without them the expression is promoted to `double`,
    /// which is a different computation from the one the oracle ran and a slower one.
    Arithmetic as arithmetic_print_as => r#"
const float e0 = 1.5f;
const float e1 = p.eps;
const float e2 = float(p.n_col);
const float e3 = a + b;
const float e4 = a - b;
const float e5 = a * b;
const float e6 = a / b;
const float e7 = sqrtf(a);
const float e8 = expf(a);
const float e9 = tanhf(a);
const float e10 = a;
const bool e11 = a >= b;
const float e12 = p ? a : b;
const uint32_t e13 = 7u;
const uint32_t e14 = i + j;
const uint32_t e15 = j + 3u;
const uint32_t e16 = i - j;
const uint32_t e17 = j - 1u;
const uint32_t e18 = j * 8u;
const uint32_t e19 = j / 8u;
const uint32_t e20 = j % 8u;
const uint32_t e21 = j & 15u;
const uint32_t e22 = j >> 2u;
const uint32_t e23 = j >> i;
const uint32_t e24 = j | i;
const uint32_t e25 = j % p.n_col;
const uint32_t e26 = p.n_col;
const bool e27 = j < 16u;
const float e28 = float(i);
const float e29 = rir_probe_rir_lut_iq4nl[i];
const float e30 = fmaxf(a, b);
const uint32_t e31 = j * p.x_nb0 + 8u;
"#,
    /// Cooperative staging: one tile loaded once by the whole block. The two
    /// barriers are printed by the statement, exactly as on the two other backends
    /// - they are not `Stmt::Barrier`, which v1 refuses.
    Staging as staging_print_as => r#"
__shared__ float rir_shared_tile[128];
const uint32_t r = blockIdx.x;
__syncthreads();
for (uint32_t l_tile = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)); l_tile < 128u; l_tile += 64u) {
    const uint32_t slot = l_tile;
    const uint32_t row = l_tile % 16u;
    const uint32_t dep = l_tile / 16u;
    const uint32_t row_g = row_o + row;
    const uint32_t dep_g = dep_o + dep;
    if (row_g < p.n_row && dep_g < p.n_col) {
        const float v = *(const float *)(x + (row_g * p.x_nb1 + dep_g * p.x_nb0));
        rir_shared_tile[slot] = v;
    } else {
        rir_shared_tile[l_tile] = 0.0f;
    }
}
__syncthreads();
"#,
    /// The lane forms: a strided walk, an accumulator, and the
    /// two warp primitives.
    ///
    /// Both primitives print as **one line**, which is the property the warp prelude
    /// exists for: `subgroupAdd` is an operator and `__shfl_xor_sync` is a ladder, so
    /// without the prelude this statement would print differently here than next
    /// door and a divergence would be invisible in a diff.
    Lanes as lanes_print_as => r#"
const uint32_t lane = threadIdx.x;
float acc = 0.0f;
for (uint32_t c = lane; c < p.n_col; c += 32u) {
    acc += v;
}
const float red = rir_lane_max(acc);
const float off = rir_lane_exclusive_sum(acc);
const uint32_t chunk_c = (p.n_col + 32u - 1u) / 32u;
const uint32_t beg_c = min(lane * chunk_c, p.n_col);
const uint32_t end_c = min((lane + 1u) * chunk_c, p.n_col);
for (uint32_t c = beg_c; c < end_c; ++c) {
}
if (lane == 0u) {
    const float v = red;
}
"#,
    /// The workgroup reduction and the control forms the lowered scan is written
    /// with. The tree is the two other emitters' expansion with `__syncthreads()`
    /// where they print `barrier()` - nothing about it is a warp assumption, which
    /// is why it is the variant the wide CUDA `rms_norm_back` goes through.
    Collectives as collectives_print_as => r#"
__shared__ float rir_shared_red[32];
const uint32_t lane = threadIdx.x;
rir_shared_red[threadIdx.x] = acc;
__syncthreads();
for (uint32_t stride_red = 16u; stride_red > 0u; stride_red >>= 1u) {
    if (threadIdx.x < stride_red) {
        rir_shared_red[threadIdx.x] = rir_shared_red[threadIdx.x] + rir_shared_red[threadIdx.x + stride_red];
    }
    __syncthreads();
}
const float red = rir_shared_red[0];
__syncthreads();
const bool cond = lane < 16u;
if (cond) {
    v = red;
}
"#,
    /// Shared memory: a declared array, a store, a barrier and a load.
    SharedMemory as shared_memory_print_as => r#"
__shared__ float rir_shared_sh[64];
const uint32_t lane = threadIdx.x;
rir_shared_sh[at] = v;
__syncthreads();
const float got = rir_shared_sh[at];
"#,
    /// The flattened dispatch: the linear index, its bound, and the magic-number
    /// decomposition.
    ///
    /// Pinned on the three backends: only the
    /// CUDA schedules flatten in production, so this arm was the one statement of
    /// the Loop IR whose Metal and Vulkan printings no test named. Everything
    /// separating the three here is lexical - the builtin, the integer type, the
    /// constant-buffer prefix - which is exactly what one Loop IR shared by the
    /// three emitters claims.
    Flat as flat_dispatch_prints_as => r#"
const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= p.rir_flat_total) {
    return;
}
const uint32_t c = (rir_fastmod(i, p.rir_flat0_mp, p.rir_flat0_sh, p.rir_flat0_div)) * 4u;
const uint32_t i_q1 = rir_fastdiv(i, p.rir_flat0_mp, p.rir_flat0_sh);
const uint32_t r = i_q1;
const float t = 0.5f;
"#,
    /// The same dispatch **without** the decomposition.
    ///
    /// What it pins is an absence - no `rir_fastmod`, no `rir_fastdiv`, no axis
    /// index - because that absence *is* the optimisation. A lowering that
    /// published the claim and went on dividing would pass every other test in
    /// this file.
    FlatLinear as flat_linear_dispatch_prints_as => r#"
const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
if (i >= p.rir_flat_total) {
    return;
}
const float t = *(const float *)(x + (i * 4u));
*(float *)(y + (i * 4u)) = t;
"#,
    /// The two-stage workgroup reduction.
    ///
    /// What the text has to show is a **count**: two barriers for 256 lanes,
    /// where the flat tree above spends one per level, and eight words of shared
    /// storage where it spends 256. Both collectives sit under a condition that
    /// is uniform per subgroup, which is what makes the primitive defined inside
    /// them.
    HierarchicalReduce as hierarchical_reduce_prints_as => r#"
__shared__ float rir_shared_red[8];
const uint32_t lane = threadIdx.x;
const float red_sg = rir_lane_sum(acc);
if (threadIdx.x % 32u == 0u) {
    rir_shared_red[threadIdx.x / 32u] = red_sg;
}
__syncthreads();
if (threadIdx.x < 32u) {
    const float red_p = threadIdx.x < 8u ? rir_shared_red[threadIdx.x] : 0.0f;
    const float red_t = rir_lane_sum(red_p);
    if (threadIdx.x == 0u) {
        rir_shared_red[0] = red_t;
    }
}
__syncthreads();
const float red = rir_shared_red[0];
"#,
});
