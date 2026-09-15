//! The schedule table: one arm per family, each arm iterating `targets_for`
//! - `GPU_TARGETS` less what the family refuses in writing.

use crate::schedule::*;

/// The scheduling **families**: one variant per arm of the table below.
///
/// A family is the set of kernels sharing one lowering decision. It is a
/// *declaration* carried by each kernel's registration
/// (`rir_kernels::KernelRegistration`) rather than inferred from its name. A
/// kernel that reaches this function has said which family it belongs to, and
/// adding a family without
/// scheduling it does not compile.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Family {
    /// One row per single-subgroup workgroup with an inner reduction:
    /// `l2_norm_back`, the two autodiff bench kernels, and the twelve
    /// quantized reductions.
    RowReduce,
    /// The same problem shape with one variant more, arbitrated by row count:
    /// `rms_norm` and `rms_norm_back`.
    RmsNorm,
    /// The `UNARY` family: fourteen members, one vectorized lowering plus a
    /// narrow-workgroup twin.
    Unary,
    /// The elementwise band's non-repeating members (`add`, `mul` and their F16
    /// twins): scalar fallback plus `vec4`.
    Elementwise,
    /// The repeating members of the same band (`add_repeat`, `mul_repeat`, …),
    /// whose `vec4` carries one additional claim.
    ElementwiseRepeat,
    /// `SCALE`: one lowering, guaranteed by the op's own precondition.
    Scale,
    /// The rank-2 oracle contraction.
    MatMulNaive,
    /// The outer product over the full ggml rank: one kernel per `src0` dtype,
    /// all staged the same way.
    OutProd,
    /// The scan: three variants per GPU backend, arbitrated by shape.
    Cumsum,
}

impl Family {
    /// Every family the table schedules. Exists so a test can walk the whole
    /// table without a kernel in hand, the way `GpuBackend::ALL` lets the
    /// coverage test walk every backend.
    pub const ALL: [Family; 9] = [
        Family::RowReduce,
        Family::RmsNorm,
        Family::Unary,
        Family::Elementwise,
        Family::ElementwiseRepeat,
        Family::Scale,
        Family::MatMulNaive,
        Family::OutProd,
        Family::Cumsum,
    ];

    /// Stable name of the family, for diagnostics and for the tests that
    /// enumerate the table.
    pub fn name(self) -> &'static str {
        match self {
            Family::RowReduce => "row_reduce",
            Family::RmsNorm => "rms_norm",
            Family::Unary => "unary",
            Family::Elementwise => "elementwise",
            Family::ElementwiseRepeat => "elementwise_repeat",
            Family::Scale => "scale",
            Family::MatMulNaive => "mat_mul_naive",
            Family::OutProd => "out_prod",
            Family::Cumsum => "cumsum",
        }
    }
}

/// Declarative schedule table (ADR-2 section 2), handwritten per family and backend.
/// The AOT pipeline emits every plausible variant; `supports_op` selects one
/// at dispatch.
///
/// Every arm iterates `targets_for(family)` rather than naming a backend, so a
/// lowering is a decision written once, and the argument is a `Family`
/// rather than a kernel name, so no arm is reached by string matching and none
/// can be missed. The table therefore states every supported target:
/// a family reaches every target it has not refused in `FAMILY_REFUSED`, and
/// "CPU alone" is a value this function returns only for a family that refused
/// both targets by name.
pub fn schedules_for(family: Family) -> Vec<Schedule> {
    let mut v = vec![Schedule::cpu_serial()];
    // The quantized reduction is one kernel per format (`sum_rows_q8_0`,
    // `sum_rows_q4_0`, …) and they all share one schedule: the block shape is a
    // lowering concern, not a scheduling one. Same for the `UNARY` family and
    // for `out_prod`, whose members differ by a member enum or a `src0` dtype,
    // a member that changed schedules would be a member whose formula, or whose
    // decoding, leaked out of the semantic graph ADR-1 section 7,
    // ADR-3).
    match family {
        // One row per workgroup with subgroup reduction: kernels with one
        // parallel axis and an inner reduction. The quantized reductions
        // qualify now that both GPU emitters can load F16 and bytes.
        Family::RowReduce => {
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_subgroup(gpu));
            }
        }
        // A 32-lane subgroup leaves the GPU underfilled when there are few wide
        // rows. Use the shared tree for at most 32 rows of at least 256 columns;
        // the subgroup fallback wins at 64 rows and on shorter rows. Forward and
        // backward RMS normalization share this reduction geometry.
        Family::RmsNorm => {
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_subgroup(gpu));
            }
            let few_wide_rows = || {
                vec![
                    ShapeRule::at_most(&["row", "plane", "batch"], 32),
                    ShapeRule {
                        axes: &["col"],
                        min: 256,
                        max: u32::MAX,
                    },
                ]
            };
            // CUDA measurements confirm the same threshold. A 1,024-lane tree
            // adds barriers and shared storage while losing on every measured
            // shape, so it is intentionally absent.
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_shared_reduce(gpu).claiming(90, few_wide_rows()));
            }
            // Flat-row dispatch covers the complementary many-row case only on
            // backends whose native kernel also flattens rows. Metal and Vulkan
            // already dispatch per row and lose with this variant.
            let many_rows = || {
                vec![ShapeRule {
                    axes: &["row", "plane", "batch"],
                    min: 33,
                    max: u32::MAX,
                }]
            };
            for gpu in flat_targets_for(family) {
                v.push(Schedule::gpu_subgroup_flat_rows(gpu).claiming(85, many_rows()));
            }
        }
        // The elementwise band. No collective at all: four parallel axes, one
        // invocation per output point, and nothing between the reads and the
        // write. `col` takes x because it is the contiguous axis - the one that
        // decides whether consecutive lanes touch consecutive addresses - and
        // `batch` stays a sequential loop, the grid having three dimensions.
        //
        // `vec4` covers four elements per invocation and needs
        // the contiguous stride to be one element to do so; the scalar
        // lowering stays the fallback for everything else - a permuted `src1`,
        // an `nb[0]` the native kernel would also have to handle. Making the
        // vector one the fallback here would have been an ABI simpler and a
        // slice of the domain narrower, and the lane counts that slice at 2 of
        // the 32 claimed nodes.
        //
        // **Repeated** members (ADR-1 section 5): the same kernel shape,
        // mapping, and two lowerings as their contiguous twins.
        //
        // Here `vec4` has **one additional claim**: four consecutive `src1`
        // indices are four consecutive addresses only when repetition of the
        // contiguous axis is the identity - that is, when `src1` and `src0`
        // have the same `ne0` extent. The registry publishes this (`vector_width`
        // plus the repetition relation), dispatch evaluates it, and a node
        // repeated *over the contiguous axis* falls back to scalar rather than
        // reading the same address four times.
        //
        // Without that claim, a repeated contiguous element could be loaded four
        // times from the same address.
        // The `UNARY` family has the `scale` shape - four parallel axes, one
        // read, one write, no collective - and the same single lowering for the
        // same contract reason. `ggml_unary` requires
        // `ggml_is_contiguous_1(src0)`, so the op guarantees
        // `nb[0] == type_size`; a scalar twin would be unreachable, which the
        // lane rejects explicitly rather than carrying indefinitely.
        Family::Unary => {
            for gpu in grid_targets_for(family) {
                v.push(Schedule::gpu_grid_vec4(gpu, [256, 1, 1]).unnamed());
            }
            // Short rows cannot fill a 256-wide vectorized workgroup. The waste
            // depends on column count rather than the number of rows.
            let short_rows = || {
                vec![ShapeRule {
                    axes: &["col"],
                    min: 0,
                    max: 256,
                }]
            };
            for gpu in grid_targets_for(family) {
                v.push(
                    Schedule::gpu_grid_vec4(gpu, [32, 1, 1])
                        .as_variant("narrow")
                        .claiming(90, short_rows()),
                );
            }
            // The flattened dispatch, on the backends
            // `FLAT_TARGETS` names and for the reason it gives. It **replaces**
            // the vectorized grid lowering above rather than sitting on top of
            // it: same width, same body, same claim on `nb[0]`, one grid
            // dimension instead of three. Two lowerings that differ only by a
            // dispatch geometry do not partition a domain, so stacking them
            // would leave one of the two at zero dispatches - which the lane
            // reports and refuses (`grid_targets_for`).
            //
            // And it replaces `narrow` with it, which is the clearest case of
            // the rule above: the narrow twin exists because a 128-column row
            // leaves 224 of 256 lanes idle in a three-dimensional grid, and a
            // flat grid leaves none idle at any row length. One lowering answers
            // what two did, without a shape rule to arbitrate between them.
            //
            // The width is **one**, and it is measured rather than inherited.
            // Four is what was measured against natives that walk a row per
            // threadgroup; `unary_op_kernel` walks one element per thread and is
            // already flat, so the width buys no address algebra here and
            // divides the grid by four - 16 blocks for a 16 384-element tensor
            // on a 128-SM device. `unary_flat_width_on_cuda` ranks the three
            // geometries on the lane's own shapes (µs, RTX 4090):
            //
            // ```text
            //                  [4096,16,1,1]  [1024,16,1,1]  [128,16,16,1]
            //   grid vec4          1.8            1.9            1.8
            //   flat  w4           1.8            1.7            1.7
            //   flat  w1           1.7            1.6            1.7
            // ```
            for gpu in flat_targets_for(family) {
                v.push(Schedule::gpu_grid_flat(gpu, [256, 1, 1], 1).unnamed());
            }
            // The **address** the flattening left open: a claim of layout,
            // that one real - all bindings contiguous and of the same shape,
            // in which case `addr = i · elem_bytes` and there is no
            // decomposition at all.
            //
            // It sits *above* the lowering it specializes rather than replacing
            // it, and that is the difference with the flattening itself. Two
            // dispatch geometries do not partition a domain - hence
            // `grid_targets_for` - but a layout claim does: the pair keeps its
            // decomposing variant for a permuted operand, for a view whose rows
            // have a gap, and for a contiguous extent that is not a whole number
            // of vectors. Each of those is a node this variant declines and that
            // one serves, which is the same partition `vec4` already makes one
            // claim lower down.
            //
            // The width stays **one** here, and deliberately: it is what
            // `unary_flat_width_on_cuda` measured against a native that walks
            // one element per thread, and keeping it is what makes the next
            // measurement read as the address alone rather than as a width and
            // an address at once.
            for gpu in flat_targets_for(family) {
                v.push(
                    Schedule::gpu_grid_flat_linear(gpu, [256, 1, 1], 1).claiming(90, Vec::new()),
                );
            }
        }
        Family::ElementwiseRepeat => {
            for gpu in grid_targets_for(family) {
                v.push(Schedule::gpu_grid(gpu, [256, 1, 1]));
            }
            for gpu in grid_targets_for(family) {
                v.push(Schedule::gpu_grid_vec4(gpu, [256, 1, 1]).claiming(90, Vec::new()));
            }
            // The flattened dispatch, on the backends
            // `FLAT_TARGETS` names and for the reason it gives. It **replaces**
            // the grid lowerings above rather than sitting on top of them: same
            // widths, same bodies, same claims, one grid dimension instead of
            // three. Two lowerings that differ only by a dispatch geometry do
            // not partition a domain, so stacking them would leave one of the
            // two at zero dispatches - which the lane reports and refuses
            // (`grid_targets_for`).
            //
            // Both halves of the pair are flattened, fallback included, and the
            // fallback is where it pays for something other than speed: a
            // three-dimensional grid puts the row count on `gridDim.y`, which
            // stops at 65 535, and the lane found two nodes of its own matrix
            // rejected as `device_grid` for exactly that. A linear space has no
            // such ceiling, so flattening the fallback widens the claimed
            // domain rather than only shortening it.
            for gpu in flat_targets_for(family) {
                v.push(Schedule::gpu_grid_flat(gpu, [256, 1, 1], 1).unnamed());
                v.push(
                    Schedule::gpu_grid_flat(gpu, [256, 1, 1], 4)
                        .as_variant(VEC4)
                        .claiming(90, Vec::new()),
                );
            }
        }
        // F16 members share the table of their F32 twins: element type lives at
        // the memory boundary, not in the schedule
        // (ADR-3 section 6).
        Family::Elementwise => {
            for gpu in grid_targets_for(family) {
                v.push(Schedule::gpu_grid(gpu, [256, 1, 1]));
            }
            for gpu in grid_targets_for(family) {
                v.push(Schedule::gpu_grid_vec4(gpu, [256, 1, 1]).claiming(90, Vec::new()));
            }
            // The flattened dispatch, on the backends
            // `FLAT_TARGETS` names and for the reason it gives. It **replaces**
            // the grid lowerings above rather than sitting on top of them: same
            // widths, same bodies, same claims, one grid dimension instead of
            // three. Two lowerings that differ only by a dispatch geometry do
            // not partition a domain, so stacking them would leave one of the
            // two at zero dispatches - which the lane reports and refuses
            // (`grid_targets_for`).
            //
            // Both halves of the pair are flattened, fallback included, and the
            // fallback is where it pays for something other than speed: a
            // three-dimensional grid puts the row count on `gridDim.y`, which
            // stops at 65 535, and the lane found two nodes of its own matrix
            // rejected as `device_grid` for exactly that. A linear space has no
            // such ceiling, so flattening the fallback widens the claimed
            // domain rather than only shortening it.
            for gpu in flat_targets_for(family) {
                v.push(Schedule::gpu_grid_flat(gpu, [256, 1, 1], 1).unnamed());
                v.push(
                    Schedule::gpu_grid_flat(gpu, [256, 1, 1], 4)
                        .as_variant(VEC4)
                        .claiming(90, Vec::new()),
                );
            }
            // The **address** the flattening left open: a claim of layout,
            // that one real - all bindings contiguous and of the same shape,
            // in which case `addr = i · elem_bytes` and there is no
            // decomposition at all.
            //
            // It sits *above* the lowering it specializes rather than replacing
            // it, and that is the difference with the flattening itself. Two
            // dispatch geometries do not partition a domain - hence
            // `grid_targets_for` - but a layout claim does: the pair keeps its
            // decomposing variant for a permuted operand, for a view whose rows
            // have a gap, and for a contiguous extent that is not a whole number
            // of vectors. Each of those is a node this variant declines and that
            // one serves, which is the same partition `vec4` already makes one
            // claim lower down.
            //
            // Above `vec4` in priority because it claims strictly more and
            // computes strictly less: same four elements per invocation, one
            // multiplication instead of a decomposition and two stride sums.
            //
            // The repeating half of this band has no such variant, and it is not
            // an omission: a repeated operand *is* a second shape, which is the
            // half of the claim no runtime test can rescue. `lower` says so,
            // `LinearAddrUnsupported`, on the `RepeatIndex` - rather than
            // leaving the table to remember it.
            for gpu in flat_targets_for(family) {
                v.push(
                    Schedule::gpu_grid_flat_linear(gpu, [256, 1, 1], 4).claiming(100, Vec::new()),
                );
            }
        }
        // `SCALE` is the same kernel shape with one lowering, and the reason is
        // ggml's own precondition: `ggml_scale_impl` asserts `ggml_is_padded_1d`
        // on its source, i.e. `nb[0] == type_size`, and `dst` is a dup or a view
        // of it. The layout the vector lowering claims is therefore guaranteed
        // by the op, and a scalar twin would be a variant no node can reach,
        // which the lane says out loud ("declared variant never dispatched")
        // rather than letting it sit in the registry forever.
        //
        // The claim is still **published**: the registry carries `vector_width`
        // and the portable contract evaluates it. If that precondition ever
        // stopped holding, the node would leave RIR with a `stride` rejection,
        // a stated refusal, never four elements read from one address.
        // A single name, not `"scale" | "scale_f16"`: `elementwise::variants()`
        // explicitly excludes an F16 SCALE member (no F16 arm in
        // `ggml_compute_forward_scale`), so the second arm could not name any
        // generated kernel.
        Family::Scale => {
            for gpu in grid_targets_for(family) {
                v.push(Schedule::gpu_grid_vec4(gpu, [256, 1, 1]).unnamed());
            }
            // The flattened dispatch, on the backends
            // `FLAT_TARGETS` names and for the reason it gives. It **replaces**
            // the vectorized grid lowering above rather than sitting on top of
            // it: same width, same body, same claim on `nb[0]`, one grid
            // dimension instead of three. Two lowerings that differ only by a
            // dispatch geometry do not partition a domain, so stacking them
            // would leave one of the two at zero dispatches - which the lane
            // reports and refuses (`grid_targets_for`).
            // The pair's **fallback** here, exactly as `vec4` is above it: this
            // family has no scalar twin - `ggml_scale_impl` pins `nb[0]`
            // through `ggml_is_padded_1d`, so a twin would be a variant no node
            // can reach - and the flattening claims no more than `vec4` already
            // did on the layout side.
            for gpu in flat_targets_for(family) {
                v.push(Schedule::gpu_grid_flat(gpu, [256, 1, 1], 4).unnamed());
            }
            // The **address** the flattening left open: a claim of layout,
            // that one real - all bindings contiguous and of the same shape,
            // in which case `addr = i · elem_bytes` and there is no
            // decomposition at all.
            //
            // It sits *above* the lowering it specializes rather than replacing
            // it, and that is the difference with the flattening itself. Two
            // dispatch geometries do not partition a domain - hence
            // `grid_targets_for` - but a layout claim does: the pair keeps its
            // decomposing variant for a permuted operand, for a view whose rows
            // have a gap, and for a contiguous extent that is not a whole number
            // of vectors. Each of those is a node this variant declines and that
            // one serves, which is the same partition `vec4` already makes one
            // claim lower down.
            //
            // `SCALE` is where this variant has the least to win and is written
            // anyway, for this reason: it is the
            // pair whose launch already weighs more than its address, so what it
            // measures is the *floor* of the lever. A ratio that does not move
            // here and moves on `UNARY` is the address; one that moves on both
            // is something else.
            for gpu in flat_targets_for(family) {
                v.push(
                    Schedule::gpu_grid_flat_linear(gpu, [256, 1, 1], 4).claiming(90, Vec::new()),
                );
            }
        }
        // Two parallel axes (i, j): no collective, one invocation per output
        // point, and a sequential contraction within the invocation.
        Family::MatMulNaive => {
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_grid(gpu, [16, 16, 1]));
            }
        }
        // Same shape of problem as `mat_mul_naive` - a contraction with two
        // parallel output axes - but over the full ggml rank, so a third
        // parallel axis (`plane`) takes the z dimension and `batch` stays a
        // sequential loop.
        //
        // A 16 by 16 block preserves occupancy for degenerate LoRA dimensions;
        // widening x leaves most lanes idle when `m = 1`.
        //
        // What those ratios could not fix is the traffic: an untiled invocation
        // re-reads `a[i,k]` for each of its 15 `j` neighbours and `b[j,k]` for
        // each of its 15 `i` neighbours, so a 16×16 workgroup pulls 512 floats
        // per `k` slice for 256 products. `TiledStage` stages both operands once
        // per workgroup instead: 512 floats for 4 096 products, sixteen times
        // less global traffic for the same arithmetic and the same accumulation
        // order.
        //
        // Depth 16 amortizes the two barriers per round. Register width 4 stages
        // 64 contiguous rows and reuses each `b` value across four FMAs.
        //
        // One lowering, not two, and for the same reason as `scale`: the width
        // is a claim on `dst`'s layout, which `ggml_out_prod` gives - the
        // destination is a fresh contiguous tensor. A scalar twin under it
        // would be a variant no node of the bench can reach, which the lane
        // refuses out loud rather than carrying forever.
        Family::OutProd => {
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_grid_tiled(gpu, [16, 16, 1], 16, 4));
            }
        }
        // Three variants per GPU backend, arbitrated by shape.
        //
        // The fallback is one invocation per row scanning it sequentially: it
        // accepts every shape, at a cost linear in the row length. The blocked
        // scan divides that depth by 32, but only pays off while the row count
        // alone leaves the GPU idle - past that it does 32× the invocations for
        // nothing and loses by a factor 2 to 4.
        //
        // Device timing places the blocked-scan crossover between 128 and 256
        // total rows, so the claim uses `row × plane × batch ≤ 128`.
        // The third is the **tiled** scan (ADR-2 section 5), using the same 256 lanes
        // as `SharedTree` with coalesced access. It requires at least 4,096
        // columns; at 1,024, most lanes are idle and the blocked scan ties it.
        //
        // Deliberately left unclaimed, as before: short rows (`n_col ≤ 256`),
        // where the blocked scan also wins at any row count. Both variants are
        // launch-dominated there, the absolute gain is tens of microseconds, and
        // claiming it would need a disjunction in a rule the registry publishes
        // as a conjunction. Not worth an ABI for that.
        Family::Cumsum => {
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_grid(gpu, [64, 1, 1]));
            }
            let few_rows = || vec![ShapeRule::at_most(&["row", "plane", "batch"], 128)];
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_blocked_scan(gpu).claiming(90, few_rows()));
            }
            let very_long_few_rows = || {
                vec![
                    ShapeRule::at_most(&["row", "plane", "batch"], 128),
                    ShapeRule {
                        axes: &["col"],
                        min: 4096,
                        max: u32::MAX,
                    },
                ]
            };
            for gpu in targets_for(family) {
                v.push(Schedule::gpu_tiled_scan(gpu, 256, 16).claiming(100, very_long_few_rows()));
            }
        }
    }
    v
}
