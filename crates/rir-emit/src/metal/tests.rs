//! What each family of `Stmt` prints as, in MSL.
//!
//! Same probes as `super::super::vulkan::tests`, printed by this emitter. Two
//! expectations of one statement, side by side in two files: what the emitters
//! share is the input, and a divergence between them is a diff in exactly one
//! of the two.
//!
//! `contains` is deliberately not used. Here the object of the test *is* the
//! text, so it is compared whole rather than searched.

use crate::pin_probes;

pin_probes!(msl, {
    /// Loop forms: the grid mapping, a reversed sequential axis, a constant count and the tiled step.
    Loops as loops_print_as => r#"
const uint r = group_id.x;
if (r >= pc.n_row) {
    return;
}
for (uint c = pc.n_col; c-- > 0u;) {
    const float t = 0.5;
}
for (uint k = 0u; k < 4u; ++k) {
}
for (uint k0 = 0u; k0 < pc.n_col; k0 += 8u) {
}
"#,
    /// Lane forms: the strided walk, the accumulator, and the two subgroup primitives.
    Lanes as lanes_print_as => r#"
const uint lane = simd_lane_id;
float acc = 0.0f;
for (uint c = lane; c < pc.n_col; c += 32u) {
    acc += v;
}
const float red = simd_max(acc);
const float off = simd_prefix_exclusive_sum(acc);
const uint chunk_c = (pc.n_col + 32u - 1u) / 32u;
const uint beg_c = min(lane * chunk_c, pc.n_col);
const uint end_c = min((lane + 1u) * chunk_c, pc.n_col);
for (uint c = beg_c; c < end_c; ++c) {
}
if (lane == 0u) {
    const float v = red;
}
"#,
    /// The workgroup reduction and lowered scan control forms: one barrier per
    /// level around the group of accumulators, plus `Barrier`, `If`, and `Set`.
    Collectives as collectives_print_as => r#"
threadgroup float rir_shared_red[32];
const uint lane = threadgroup_lane_id;
rir_shared_red[threadgroup_lane_id] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint stride_red = 16u; stride_red > 0u; stride_red >>= 1u) {
    if (threadgroup_lane_id < stride_red) {
        rir_shared_red[threadgroup_lane_id] = rir_shared_red[threadgroup_lane_id] + rir_shared_red[threadgroup_lane_id + stride_red];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
const float red = rir_shared_red[0];
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool cond = lane < 16u;
if (cond) {
    v = red;
}
"#,
    /// Shared memory: a declared array, a store, a barrier and a load.
    SharedMemory as shared_memory_print_as => r#"
threadgroup float rir_shared_sh[64];
const uint lane = threadgroup_lane_id;
rir_shared_sh[at] = v;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float got = rir_shared_sh[at];
"#,
    /// Memory: an F32 load, an F16 load and its conversion, a store under a bounds check.
    Memory as memory_print_as => r#"
const uint row_i = global_id.x;
const float val = *(device const float *)(x + (row_i * pc.x_nb1 + col_i * pc.x_nb0 + 4u));
const float half = float(*(device const half *)(x + (col_i * 2u)));
if (row_i < pc.n_row) {
    *(device float *)(y + (row_i * pc.y_nb1)) = val;
}
"#,
    /// The vector lowering: a widened body, its scalar tail, and the per-component bounded write.
    Vectors as vectors_print_as => r#"
const uint col_i = global_id.x * 4u;
if (col_i + 4u <= pc.n_col) {
    const float4 v4 = *(device const packed_float4 *)(x + (col_i * pc.x_nb0));
    const bool4 p4 = v4 >= v4;
    const float4 w4 = select(v4, v4, p4);
    if (col_i + 4u <= pc.n_col) {
        *(device packed_float4 *)(y + (col_i * pc.y_nb0)) = packed_float4(w4);
    } else {
        for (uint c = 0u; c + col_i < pc.n_col; ++c) {
            *(device float *)(y + (col_i * pc.y_nb0) + c * pc.y_nb0) = w4[c];
        }
    }
} else {
    for (uint tail = col_i; tail < pc.n_col; ++tail) {
        *(device float *)(y + (tail * pc.y_nb0)) = s;
    }
}
"#,
    /// Expressions: one line per `LExpr` family.
    Arithmetic as arithmetic_print_as => r#"
const float e0 = 1.5;
const float e1 = pc.eps;
const float e2 = float(pc.n_col);
const float e3 = a + b;
const float e4 = a - b;
const float e5 = a * b;
const float e6 = a / b;
const float e7 = sqrt(a);
const float e8 = exp(a);
const float e9 = precise::tanh(a);
const float e10 = a;
const bool e11 = a >= b;
const float e12 = p ? a : b;
const uint e13 = 7u;
const uint e14 = i + j;
const uint e15 = j + 3u;
const uint e16 = i - j;
const uint e17 = j - 1u;
const uint e18 = j * 8u;
const uint e19 = j / 8u;
const uint e20 = j % 8u;
const uint e21 = j & 15u;
const uint e22 = j >> 2u;
const uint e23 = j >> i;
const uint e24 = j | i;
const uint e25 = j % pc.n_col;
const uint e26 = pc.n_col;
const bool e27 = j < 16u;
const float e28 = float(i);
const float e29 = rir_probe_rir_lut_iq4nl[i];
const float e30 = max(a, b);
const uint e31 = j * pc.x_nb0 + 8u;
"#,
    /// Cooperative staging: one tile loaded once by the whole workgroup.
    Staging as staging_print_as => r#"
threadgroup float rir_shared_tile[128];
const uint r = group_id.x;
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint l_tile = threadgroup_lane_id; l_tile < 128u; l_tile += 64u) {
    const uint slot = l_tile;
    const uint row = l_tile % 16u;
    const uint dep = l_tile / 16u;
    const uint row_g = row_o + row;
    const uint dep_g = dep_o + dep;
    if (row_g < pc.n_row && dep_g < pc.n_col) {
        const float v = *(device const float *)(x + (row_g * pc.x_nb1 + dep_g * pc.x_nb0));
        rir_shared_tile[slot] = v;
    } else {
        rir_shared_tile[l_tile] = 0.0f;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
"#,
    /// The flattened dispatch: the linear index, its bound, and the magic-number
    /// decomposition.
    ///
    /// Pinned since no Metal schedule flattens in
    /// production, so this arm printed for no test at all - neither here nor in
    /// `generated/`, which only holds what a schedule produces.
    Flat as flat_dispatch_prints_as => r#"
const uint i = global_id.x;
if (i >= pc.rir_flat_total) {
    return;
}
const uint c = (rir_fastmod(i, pc.rir_flat0_mp, pc.rir_flat0_sh, pc.rir_flat0_div)) * 4u;
const uint i_q1 = rir_fastdiv(i, pc.rir_flat0_mp, pc.rir_flat0_sh);
const uint r = i_q1;
const float t = 0.5;
"#,
    /// The same dispatch **without** the decomposition.
    ///
    /// What it pins is an absence - no `rir_fastmod`, no `rir_fastdiv`, no axis
    /// index - because that absence *is* the optimisation. A lowering that
    /// published the claim and went on dividing would pass every other test in
    /// this file.
    FlatLinear as flat_linear_dispatch_prints_as => r#"
const uint i = global_id.x;
if (i >= pc.rir_flat_total) {
    return;
}
const float t = *(device const float *)(x + (i * 4u));
*(device float *)(y + (i * 4u)) = t;
"#,
    /// The two-stage workgroup reduction.
    ///
    /// What the text has to show is a **count**: two barriers for 256 lanes,
    /// where the flat tree above spends one per level, and eight words of shared
    /// storage where it spends 256. Both collectives sit under a condition that
    /// is uniform per subgroup, which is what makes the primitive defined inside
    /// them.
    HierarchicalReduce as hierarchical_reduce_prints_as => r#"
threadgroup float rir_shared_red[8];
const uint lane = threadgroup_lane_id;
const float red_sg = simd_sum(acc);
if (threadgroup_lane_id % 32u == 0u) {
    rir_shared_red[threadgroup_lane_id / 32u] = red_sg;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (threadgroup_lane_id < 32u) {
    const float red_p = threadgroup_lane_id < 8u ? rir_shared_red[threadgroup_lane_id] : 0.0f;
    const float red_t = simd_sum(red_p);
    if (threadgroup_lane_id == 0u) {
        rir_shared_red[0] = red_t;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float red = rir_shared_red[0];
"#,
});
