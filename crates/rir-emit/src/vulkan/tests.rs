//! What each family of `Stmt` prints as, in GLSL.
//!
//! The expectation of every test below is the emitter's **exact** output for a
//! hand-built nest, so a change in the printed form is a diff in this file. That
//! is the property the emitter did not have: parity says the shader computes the
//! right thing, shader compilation says it is valid GLSL, and neither says what
//! a statement looks like.
//!
//! `contains` is deliberately not used. Here the object of the test *is* the
//! text, so it is compared whole rather than searched.

use crate::pin_probes;

pin_probes!(glsl, {
    /// Loop forms: the grid mapping, a reversed sequential axis, a constant count and the tiled step.
    Loops as loops_print_as => r#"
const uint r = gl_WorkGroupID.x;
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
const uint lane = gl_LocalInvocationID.x;
float acc = 0.0;
for (uint c = lane; c < pc.n_col; c += 32u) {
    acc += v;
}
const float red = subgroupMax(acc);
const float off = subgroupExclusiveAdd(acc);
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
const uint lane = gl_LocalInvocationID.x;
rir_shared_red[gl_LocalInvocationID.x] = acc;
barrier();
for (uint stride_red = 16u; stride_red > 0u; stride_red >>= 1u) {
    if (gl_LocalInvocationID.x < stride_red) {
        rir_shared_red[gl_LocalInvocationID.x] = rir_shared_red[gl_LocalInvocationID.x] + rir_shared_red[gl_LocalInvocationID.x + stride_red];
    }
    barrier();
}
const float red = rir_shared_red[0];
barrier();
const bool cond = lane < 16u;
if (cond) {
    v = red;
}
"#,
    /// Shared memory: a declared array, a store, a barrier and a load.
    SharedMemory as shared_memory_print_as => r#"
const uint lane = gl_LocalInvocationID.x;
rir_shared_sh[at] = v;
barrier();
const float got = rir_shared_sh[at];
"#,
    /// Memory: an F32 load, an F16 load and its conversion, a store under a bounds check.
    Memory as memory_print_as => r#"
const uint row_i = gl_GlobalInvocationID.x;
const float val = x[(row_i * pc.x_nb1 + col_i * pc.x_nb0 + 4u) / 4u];
const float half = float(x_f16[(col_i * 2u) / 2u]);
if (row_i < pc.n_row) {
    y[(row_i * pc.y_nb1) / 4u] = val;
}
"#,
    /// The vector lowering: a widened body, its scalar tail, and the per-component bounded write.
    Vectors as vectors_print_as => r#"
const uint col_i = gl_GlobalInvocationID.x * 4u;
if (col_i + 4u <= pc.n_col) {
    const uint v4_i = (col_i * pc.x_nb0) / 4u;
    const vec4 v4 = vec4(x[v4_i], x[v4_i + 1u], x[v4_i + 2u], x[v4_i + 3u]);
    const bvec4 p4 = greaterThanEqual(v4, v4);
    const vec4 w4 = mix(v4, v4, p4);
    const uint w4_o = (col_i * pc.y_nb0) / 4u;
    if (col_i + 4u <= pc.n_col) {
        y[w4_o + 0u] = w4.x;
        y[w4_o + 1u] = w4.y;
        y[w4_o + 2u] = w4.z;
        y[w4_o + 3u] = w4.w;
    } else {
        for (uint c = 0u; c + col_i < pc.n_col; ++c) {
            y[w4_o + c] = w4[c];
        }
    }
} else {
    for (uint tail = col_i; tail < pc.n_col; ++tail) {
        y[(tail * pc.y_nb0) / 4u] = s;
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
const float e9 = tanh(a);
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
const float e29 = rir_lut_iq4nl[i];
const float e30 = max(a, b);
const uint e31 = j * pc.x_nb0 + 8u;
"#,
    /// Cooperative staging: one tile loaded once by the whole workgroup.
    Staging as staging_print_as => r#"
const uint r = gl_WorkGroupID.x;
barrier();
for (uint l_tile = gl_LocalInvocationIndex; l_tile < 128u; l_tile += 64u) {
    const uint slot = l_tile;
    const uint row = l_tile % 16u;
    const uint dep = l_tile / 16u;
    const uint row_g = row_o + row;
    const uint dep_g = dep_o + dep;
    if (row_g < pc.n_row && dep_g < pc.n_col) {
        const float v = x[(row_g * pc.x_nb1 + dep_g * pc.x_nb0) / 4u];
        rir_shared_tile[slot] = v;
    } else {
        rir_shared_tile[l_tile] = 0.0;
    }
}
barrier();
"#,
    /// The flattened dispatch: the linear index, its bound, and the magic-number
    /// decomposition.
    ///
    /// Pinned since no Vulkan schedule flattens in
    /// production, so this arm printed for no test at all - neither here nor in
    /// `generated/`, which only holds what a schedule produces.
    Flat as flat_dispatch_prints_as => r#"
const uint i = gl_GlobalInvocationID.x;
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
const uint i = gl_GlobalInvocationID.x;
if (i >= pc.rir_flat_total) {
    return;
}
const float t = x[(i * 4u) / 4u];
y[(i * 4u) / 4u] = t;
"#,
    /// The two-stage workgroup reduction.
    ///
    /// What the text has to show is a **count**: two barriers for 256 lanes,
    /// where the flat tree above spends one per level, and eight words of shared
    /// storage where it spends 256. Both collectives sit under a condition that
    /// is uniform per subgroup, which is what makes the primitive defined inside
    /// them.
    HierarchicalReduce as hierarchical_reduce_prints_as => r#"
const uint lane = gl_LocalInvocationID.x;
const float red_sg = subgroupAdd(acc);
if (gl_LocalInvocationID.x % 32u == 0u) {
    rir_shared_red[gl_LocalInvocationID.x / 32u] = red_sg;
}
barrier();
if (gl_LocalInvocationID.x < 32u) {
    const float red_p = gl_LocalInvocationID.x < 8u ? rir_shared_red[gl_LocalInvocationID.x] : 0.0;
    const float red_t = subgroupAdd(red_p);
    if (gl_LocalInvocationID.x == 0u) {
        rir_shared_red[0] = red_t;
    }
}
barrier();
const float red = rir_shared_red[0];
"#,
});
