# RIR - kernel families

RIR is a source-to-source kernel compiler: kernels are written **once** in a
Rust DSL as a pure tensor SSA graph, then lowered against a hand-written
*schedule* and printed as backend source (Rust for CPU, GLSL compute for
Vulkan, MSL for Metal) **ahead of time**. The generated files are committed
under `generated/rir/` and a test fails if regenerating produces any diff.

This document lists what is actually generated. It is deliberately narrow -
the full architecture writeup (semantic IR, schedules, lowering, emitters,
autodiff, dispatch) lives closer to the crates themselves and is not part of
this public excerpt yet.

## Kernels currently generated

`rir_kernels::all()` is the registry consumed by `rir-gen`; the order is
stable and determines generation order. A **family** is one kernel shape
instantiated by table - per quantized format, per unary member, per element
type - so a row below is a shape, not a copy: adding a format or a member adds
a kernel without adding a lowering.

The table is not prose: `the_readme_kernel_table_matches_the_registry`
(`crates/rir-gen/tests/tables.rs`) parses it and compares every family and
every count to `rir_kernels::all()`, so a kernel added without a row here
fails the lane.

<!-- rir-gen:families:start -->

| Family | Members | Math | What it exercises |
|---|---:|---|---|
| `l2_norm_back` | 1 | `dx = norm > eps ? (dz − x·(Σx·dz / Σx²))/norm : dz/eps` | two deterministic reductions sharing one pass over `x`, elementwise epilogue with a safety branch |
| `l2_norm_fwd` | 1 | `y = x / sqrt(Σx²)` | the autodiff test bench |
| `l2_norm_fwd_grad` | 1 | derived from `l2_norm_fwd` | autodiff by transposition, **multi-level** reductions - under lanes too |
| `rms_norm` | 1 | `y = x / sqrt(Σx²/n + eps)` | one reduction, `SharedTree` on wide rows |
| `rms_norm_back` | 1 | gradient of the above | two reductions of one level - the shape a fused kernel wants under one set of barriers |
| `cumsum` | 1 | `y[c,r,p,b] = Σ_{c'≤c} x[c',r,p,b]` | inclusive scan over the full ggml rank, and its transpose (a backward scan); three variants |
| `mat_mul_naive` | 1 | `c[i,j] = Σ_k a[k,i]·b[k,j]` (ggml's `C = Aᵀ·B`) | N parallel axes + one inner axis (a contraction) |
| `out_prod` | 1 | `d[i,j,p,b] = Σ_k a[i,k,…]·b[j,k,…]` | contraction over the full ggml rank, staged in shared memory (`TiledStage`) |
| `out_prod_<format>` | 12 | same, with a quantized `src0` | the quantized loader under a *staged* tile, one kernel per format |
| `sum_rows_<format>` | 12 | `y[row] = Σ_col dequant(x[col,row])` | fused quantized loader + row reduction, row-level store hoisting; on GLSL, 8/16-bit storage |
| `unary_<member>` | 14 | `y[i] = f(x[i])` | one shape, one vectorized lowering, one member per ggml unary op |
| `add`, `add_f16`, `mul`, `mul_f16`, `add_repeat`, `add_repeat_f16`, `mul_repeat`, `mul_repeat_f16`, `scale` | 9 | `dst = a ⊕ b`, `dst = a·s + b` | the elementwise band: vec4 lowering with a scalar fallback, F16 members, and index arithmetic for `ggml_can_repeat` |

<!-- rir-gen:families:end -->

The twelve quantized formats are `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`,
`q2_K`, `q3_K`, `q4_K`, `q5_K`, `q6_K`, `iq4_nl`, `iq4_xs`; the table lives in
`rir_core::quant_formats()` and is emitted for the fork as
`generated/rir/quant/ggml-retro-quant.h`.
