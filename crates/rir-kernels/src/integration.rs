//! One function per kernel family, holding that family's integration spec.
//!
//! `integration()` keeps the dispatch and nothing else. Nothing here may change
//! a byte of the emitted registry - `rir-gen`'s `regenerating_produces_no_diff`
//! is what says so.

use rir_core::QuantType;
use rir_emit::{
    ArgSource, BackendPolicy, DomainAssumption, DomainRestriction, GgmlBackend, IntegrationSpec,
    ParamSpec,
};

use super::{BROADCAST_BAND, cuda_policy, elementwise, gpu_policy, oracle_only, unary};
use ArgSource::{Dst, Src};
use BackendPolicy::{NativeOnly, ObserveGenerated, PreferGenerated};
use DomainRestriction::{DType, OpVariant, Shape};
use GgmlBackend::{Cpu, Cuda, Metal, Vulkan};

/// The production pilot. dz = src[0], x = src[1], dx = dst, eps at
/// op_params offset 0 - verified against
/// ggml_compute_forward_l2_norm_back_f32. Promoted and retired on all
/// three GPU backends, CUDA included (see below); CPU stays native in
/// the first slice.
pub(super) fn l2_norm_back(name: String) -> IntegrationSpec {
    IntegrationSpec {
        kernel: name,
        ggml_op_variant: None,
        ggml_op: Some("GGML_OP_L2_NORM_BACK"),
        args: vec![Src(0), Src(1), Dst],
        params: vec![ParamSpec {
            name: "eps",
            op_params_offset: 0,
        }],
        production: true,
        // **Promoted on CUDA too**, and it is the
        // one pair of the phase whose prognosis held exactly. The census
        // gives the op a single shape, `[128,16,16,1]`, where the native
        // takes its 32-thread branch - one row per warp - which is the
        // geometry `f32_4d_subgroup_tree` launches. The lane measures
        // 0.99 on both variants of that shape over nine passes
        // (1.55 µs native, 1.53 µs RIR), i.e. parity - and equalling a native CUDA kernel is the complete result
        // here, the gain being counted in deleted delta and not in
        // microseconds.
        backend_policy: cuda_policy(PreferGenerated, PreferGenerated),
        // Nothing assumed: both natives gate this op on F32 sources, an
        // F32 destination, `nb[0] == type_size` and three identical
        // shapes - which is exactly what the variant claims. A single
        // contract rejection here therefore *is* a defect, and the lane
        // says so.
        assumed_domain: vec![],
        // And because nothing is assumed, this pair may lose its native kernel
        // entirely: the RIR contract *is* the ggml domain of the op on
        // these two backends, so the native kernel served no node the
        // generated one declines. It is gone from the fork,
        // and what proves the claim is that
        // `supports_op` now answers on the contract alone - a node RIR
        // would refuse leaves for the CPU instead of finding a kernel
        // that no longer exists.
        // And CUDA joins them. It is the same
        // argument and it is not weakened by CUDA being third: the pair
        // declares no restriction, the lane measures it claiming 60/60
        // of the matrix, so the native kernel served no node the
        // generated one declines. The registry is what forced the two
        // halves to move together - a promoted pair with nothing
        // declared, a native still there and no line explaining it is
        // the forbidden state, and it is the right refusal: keeping
        // `l2_norm_back_f32` alive on CUDA would have been the "not got
        // round to it" that never gets round to it.
        retired_native: vec![Vulkan, Metal, Cuda],
        native_exception: None,
    }
}

/// The second production op. Exact for F32
/// over the full ggml rank, but the scan is sequential *within* one
/// invocation, where the native Metal and Vulkan kernels use a
/// blocked scan: the variant is registered and measured, never
/// encoded, until a benchmark says otherwise on a given backend. That
/// is what `ObserveGenerated` states - in the registry, not in a
/// backend `if`.
pub(super) fn cumsum(name: String) -> IntegrationSpec {
    IntegrationSpec {
        kernel: name,
        ggml_op_variant: None,
        ggml_op: Some("GGML_OP_CUMSUM"),
        args: vec![Src(0), Dst],
        params: vec![],
        production: true,
        backend_policy: gpu_policy(ObserveGenerated),
        // The scan is F32 over the full ggml rank and claims every
        // shape; the twenty matrix shapes produce no contract rejection
        // at all.
        assumed_domain: vec![],
        retired_native: vec![],
        native_exception: None,
    }
}

/// A sub-domain of MUL_MAT (C = AᵀB, F32, rank 2); oracle only - it
/// replaces neither BLAS nor MMA.
pub(super) fn mat_mul_naive(name: String) -> IntegrationSpec {
    IntegrationSpec {
        ggml_op: Some("GGML_OP_MUL_MAT"),
        ..oracle_only(name, vec![Src(0), Src(1), Dst])
    }
}

/// The third production op, and the first chosen by measurement
/// rather than by what the compiler happened to be able to express:
/// the census of a real backward graph ranks `OUT_PROD` first among
/// the ops that are both heavy and winnable.
/// It is the LoRA backward's own outer product, its ggml semantics is
/// a plain outer product - no epsilon, no divergent output dtype,
/// and it appears in both a hybrid SSM model and a pure transformer.
///
/// It entered in `observe`, and the ratio table promoted it: the six
/// census shapes are between 0.56 and 1.01 on Vulkan and between 0.74
/// and 0.90 on Metal, once the contraction reads staged tiles rather
/// than global memory.
///
/// One spec per `src0` dtype, in the order `ids()` builds them. They
/// are the **same** ggml op: the policy row the registry emits is
/// merged across them, and the declared domain of `GGML_OP_OUT_PROD`
/// is what *none* of them claims.
pub(super) fn out_prod(name: String, format: Option<QuantType>) -> IntegrationSpec {
    {
        // What each kernel leaves out. Both are real ggml, both are
        // per-kernel, and only their intersection reaches the op's
        // policy row.
        //
        // The dtype row selects between variants of one op. The table contains
        // all portable formats; `NativeIntrinsic` formats remain outside the
        // lowering domain, and this row also excludes F16 for the scalar path.
        let dtype_why: &'static str = match format {
            None => {
                "quantized src0: served by out_prod_<format> variants, not this one - \
             selection uses the node dtype"
            }
            Some(_) => {
                "F32 src0: served by the fallback variant, not this one; plus formats \
                without a portable decoder (native_intrinsic and mxfp4, whose E8M0 \
                scale is rejected with its reason) and F16"
            }
        };
        IntegrationSpec {
            kernel: name,
            ggml_op_variant: None,
            ggml_op: Some("GGML_OP_OUT_PROD"),
            args: vec![Src(0), Src(1), Dst],
            params: vec![],
            production: true,
            backend_policy: gpu_policy(PreferGenerated),
            assumed_domain: vec![
                DomainAssumption {
                    restriction: DType,
                    why: dtype_why,
                },
                DomainAssumption {
                    restriction: Shape,
                    why: "src0 broadcast over ne2/ne3; same lack of index arithmetic as \
                          the elementwise-strip broadcast. No real-graph node",
                },
            ],
            // Two published restrictions, so the native kernel stays,
            // kept *for* them: the census counts 56 of 736 nodes on a real graph
            // that still need it.
            retired_native: vec![],
            native_exception: None,
        }
    }
}

/// The fourth production op, second of the backward-graph census: the exact
/// analogue of `l2_norm_back`, which is already promoted on both GPU
/// backends - two deterministic reductions over one traversal of a
/// row, then an elementwise epilogue. dz = src[0], x = src[1],
/// dx = dst, eps at op_params offset 0, checked against
/// ggml_compute_forward_rms_norm_back_f32.
///
/// It entered in `observe`, and what promoted it was a **schedule**
/// and not a compiler capability: the shape
/// rule arbitrating its two lowerings over-claimed by one shape.
/// Sixty-four rows is where the 32-lane fallback already wins (0.82
/// in V3), and sending it to a 256-lane workgroup with barriers cost
/// it 1.09 on Metal - the single shape that held the pair back. With
/// the ceiling at 32 rows the five census shapes are between 0.50 and
/// 1.03 on Metal and between 0.96 and 1.03 on Vulkan.
pub(super) fn rms_norm_back(name: String) -> IntegrationSpec {
    IntegrationSpec {
        kernel: name,
        ggml_op_variant: None,
        ggml_op: Some("GGML_OP_RMS_NORM_BACK"),
        args: vec![Src(0), Src(1), Dst],
        params: vec![ParamSpec {
            name: "eps",
            op_params_offset: 0,
        }],
        production: true,
        // **Refused on CUDA, and it stays in `observe`**.
        // Not a phase left unfinished: the pair
        // was measured over nine passes on the five shapes of the
        // matrix - 0.97 / 0.99 / 1.03 / 1.15 / 1.17 - and two of the
        // five are past the tolerance.
        //
        // Where the two are is what makes the refusal informative. The
        // shapes the `shared_reduce` variant claims are green, both of
        // them; the two that fail are `[256,8,16,1]` and `[256,4,16,1]`,
        // 128 and 64 rows, which the rule leaves to the 32-lane
        // fallback - and the short loop says the 256-lane tree does not
        // help there either (2.2 against 2.1 µs). So the remaining gap
        // is not a variant that was not written: it is what the fallback
        // pays against a native kernel that flattens its rows into one
        // grid dimension and reduces through `warp_reduce_sum` plus a
        // 32-entry shared stage, where RIR keeps three grid dimensions
        // and expands a full workgroup tree. Closing it needs a
        // two-stage reduction strategy in the Loop IR and the grid
        // flattening - two pieces of work, neither of them a
        // block size.
        //
        // Kept registered rather than dropped, for what it is used
        // for: the pair claims 30/30 of the matrix and produces the
        // coverage a later attempt is measured against.
        backend_policy: cuda_policy(PreferGenerated, ObserveGenerated),
        // Same reading as `l2_norm_back`, whose contract it is the
        // analogue of: F32 throughout, `nb[0] == type_size`, three
        // identical shapes.
        assumed_domain: vec![],
        // And the same consequence: the second of the two pairs allowed
        // to lose its native kernel entirely.
        retired_native: vec![Vulkan, Metal],
        native_exception: None,
    }
}

/// Forward `RMS_NORM`: 1.0% of measured traffic on Qwen3.5 and 1.7% on
/// gemma-3-270m.
///
/// Nothing assumed about the **contract**: both native GPU kernels
/// gate the op on `ggml_is_contiguous_rows(src0)` and F32, and axis
/// agreement derives shape equality. What remains native is not a
/// rejected domain but a node never offered: Metal fuses `RMS_NORM`
/// with the following `MUL` and `ADD`, and dispatch hands control to
/// RIR only when fusion found nothing - the exact elementwise-strip
/// pattern. A `seen` count above `rir` with
/// no contract rejection identifies this fusion, not an undeclared
/// restriction.
pub(super) fn rms_norm(name: String) -> IntegrationSpec {
    IntegrationSpec {
        kernel: name,
        ggml_op_variant: None,
        ggml_op: Some("GGML_OP_RMS_NORM"),
        args: vec![Src(0), Dst],
        params: vec![ParamSpec {
            name: "eps",
            op_params_offset: 0,
        }],
        production: true,
        backend_policy: gpu_policy(ObserveGenerated),
        assumed_domain: vec![],
        // Native remains beyond measurement: it carries
        // `RMS_NORM + MUL + ADD` fusion, which the registry cannot
        // describe and a RIR kernel - one node, one dispatch - cannot
        // express. This is RIR's "honest limit" encountered for the
        // first time on an op otherwise fully served by RIR.
        retired_native: vec![],
        native_exception: None,
    }
}

/// The `UNARY` family. One spec per
/// member, all on the **same** `ggml_op` - the registry merges their
/// policy row, and the declared `GGML_OP_UNARY` domain is what *none*
/// of them claims (the `out_prod` pattern).
pub(super) fn unary(name: String, member: unary::Unary) -> IntegrationSpec {
    IntegrationSpec {
        kernel: name,
        ggml_op_variant: Some(member.ggml_unary_op()),
        ggml_op: Some("GGML_OP_UNARY"),
        args: vec![Src(0), Dst],
        params: vec![],
        production: true,
        // **Promoted on both GPU backends**. The
        // family entered `observe` and the benchmark rejected it once:
        // on `[128,16,16,1]` - 256 rows of 128 columns - a
        // 256-invocation vectorized workgroup covered 1,024 elements for
        // a 128-element row, reflected in ratios of 1.08 on `relu`, 1.15
        // on `silu`, and 1.40 on `tanh`. A second workgroup width,
        // arbitrated by a `col` shape rule, closed exactly this gap: all
        // nine benchmark shapes are 0.18–0.94.
        //
        // Vulkan gives the family's
        // best result: **0.23–0.33** over all nine benchmark shapes,
        // versus 0.18–0.94 on Metal. Native Vulkan runs the unary strip
        // through the generic `ggml_vk_op_f32` path, while native Metal
        // has a dedicated kernel - the gap measures that generic path,
        // not the compiler.
        // CUDA in **observe** for the flattened variant,
        // on the family the census picks for its
        // redundancy rather than for its traffic: fourteen native
        // templates against one generated kernel.
        backend_policy: cuda_policy(PreferGenerated, ObserveGenerated),
        assumed_domain: vec![
            DomainAssumption {
                restriction: DType,
                why: "both native kernels accept F16 for this family; all fourteen \
                      written members are typed F32. Closing it would cost fourteen more \
                      kernels for a dtype produced by neither census on a UNARY",
            },
            DomainAssumption {
                restriction: OpVariant,
                why: "the eight members RIR does not declare: softplus and gelu_erf need \
                      log and erf, floor/ceil/round/trunc need directed rounding, xielu \
                      carries its own scalars in op_params - and gelu, written and \
                      exact, is removed by measurement: on Metal native itself yields \
                      NMSE of 1.00-1.14e-7 against a 1.0e-7 threshold",
            },
        ],
        retired_native: vec![],
        native_exception: None,
    }
}

/// The elementwise band, third of the census and the first entry
/// where one *shape* of kernel covers three ggml ops: `MUL` + `ADD` +
/// `SCALE` are 7.7 % of the traffic on Qwen3.5 and 13.9 % on
/// gemma-3-270m over ~6 700 nodes each, and taken one at a time none
/// of them would be worth the detour.
///
/// `SCALE` is the only one with scalars: `ggml_scale` writes the
/// multiplier at op_params offset 0 and the bias at offset 4. The two
/// binary members take src[0], src[1], dst and no parameter at all.
///
/// **Promoted** on both GPU backends. They
/// entered in `observe` and the lane refused them: the generated
/// kernel loaded one `float` where the two native ones walk their row
/// in `float4`, and the deficit grew with the row length - 1.09 to
/// 1.27 past 4 096 columns on Metal. `vector_width > 1` closed
/// exactly that, and the ratio table is now 0.71 to 0.98 on Metal and
/// 0.89 to 1.01 on Vulkan, on the same shapes, with no shape left
/// above the tolerance.
///
/// The claimed domain is unchanged by the promotion - 32 of 86 nodes
/// on Metal, the rest F16 and broadcast the registry publishes as out
/// of scope. `prefer` encodes what RIR claims; it does not
/// widen the claim.
pub(super) fn elementwise(name: String, band: elementwise::Band) -> IntegrationSpec {
    {
        use elementwise::BandOp;
        let (args, params) = if band.is_binary() {
            (vec![Src(0), Src(1), Dst], vec![])
        } else {
            (
                vec![Src(0), Dst],
                vec![
                    ParamSpec {
                        name: "scale",
                        op_params_offset: 0,
                    },
                    ParamSpec {
                        name: "bias",
                        op_params_offset: 4,
                    },
                ],
            )
        };
        // `SCALE` takes only the dtype restriction: it is unary, so
        // there is no second operand to repeat, and `ggml_scale_impl`
        // pins `nb[0]` through `ggml_is_padded_1d` - the very
        // precondition that made its vec4 lowering the pair's fallback.
        //
        // A **repeating** member takes neither `shape` restriction: it
        // is the one that claims `ggml_can_repeat`, and only its
        // intersection with the non-repeating member reaches the op's
        // policy row - which is how `shape` leaves `GGML_OP_ADD` and
        // `GGML_OP_MUL` altogether. The two
        // kernels together claim the shape domain of the op; neither
        // does alone, and that is exactly the reading `out_prod`'s dtype
        // line already had.
        //
        // The dtype row for `ADD` and `MUL` restricts **one kernel in the
        // pair**, not the op. The F32 member carries it because it rejects F16;
        // the F16 member declares nothing, so the intersection is empty
        // and `dtype` leaves `GGML_OP_ADD` and `GGML_OP_MUL`.
        // Both native kernels accept exactly
        // `F32 | F16` for these ops, so the two generated kernels
        // together claim the whole domain, as 86/86 coverage shows.
        //
        // `SCALE` retains it because `ggml_compute_forward_scale` has no F16
        // arm. The
        // reference backend aborts on such a node, so no benchmark could
        // judge a kernel serving it.
        let dtype_line = DomainAssumption {
            restriction: DType,
            why: "F16: both native GPU kernels accept it, ggml_compute_forward_scale does \
                  not - an F16 member would be a variant no benchmark can judge",
        };
        let mut assumed_domain: Vec<DomainAssumption> = if band.op == BandOp::Scale {
            vec![dtype_line]
        } else {
            Vec::new()
        };
        if !band.repeats() && band.is_binary() {
            assumed_domain.push(BROADCAST_BAND);
        }
        // `ADD` and `MUL` publish no remaining restriction while retaining
        // their native kernels. The reason is written here rather than inferred
        // from silence.
        //
        // The native strip has four clients a RIR kernel cannot
        // serve, all of the same kind - a dispatch covering more than
        // one node:
        //
        //   1. fused ADD chains (`n_fuse` up to 8 on Metal, `multi_add`
        //      on Vulkan);
        //   2. `RMS_NORM + MUL + ADD` fusion in the norm encoder, which
        //      consumes MUL and ADD nodes without presenting them;
        //   3. Metal's "snake" fusion `MUL·SIN·SQR·MUL·ADD`;
        //   4. `do_add_rms_partials` on Vulkan, where ADD also writes
        //      partial sums read by the following RMS_NORM,
        //      partial sums required by the following RMS_NORM.
        //
        // A fifth reason is not fusion: on Metal there is no ADD kernel
        // to remove. `kernel_bin_fuse_impl` is **one** pattern for
        // ADD/SUB/MUL/DIV, selected by a function constant;
        // `ggml_metal_op_acc` also uses it for an op rejected by RIR
        // Removing "native ADD" would mean removing one constant
        // value, which removes no line.
        //
        // `add.comp` and `mul.comp` have no vendored changes. Removing them
        // would add patch lines instead of removing any.
        let native_exception = match band.op {
            BandOp::Scale => None,
            _ => Some(
                "the native strip serves four paths a RIR kernel cannot express - one \
                 node, one dispatch: fused ADD chains, RMS_NORM+MUL+ADD fusion, Metal \
                 snake fusion, and do_add_rms_partials on Vulkan, which is a correctness \
                 path. On Metal there is also no kernel to remove: one pattern serves \
                 ADD/SUB/MUL/DIV and ACC. Deliberate native exception, not a missed \
                 removal",
            ),
        };
        IntegrationSpec {
            kernel: name,
            ggml_op_variant: None,
            ggml_op: Some(band.ggml_op()),
            args,
            params,
            production: true,
            // CUDA, and the three members do not get the same answer.
            // All of them run the flattened
            // dispatch and all of them were measured against the native
            // flat loop in one session; `SCALE` cleared the tolerance on
            // every shape of its matrix (0.92 / 0.89 / 0.99) and is
            // promoted, `ADD` and `MUL` did not and stay observed.
            //
            // What refuses them is **one** shape and it is not the
            // flattening: `ADD ne=[4096,1,1,1] nr=[1,512,1,1]` - a row
            // replayed 512 times - measures 1.10, while the eight
            // non-repeating shapes of the two ops are 0.84 to 1.03. The
            // deficit is on the *repeat* path, which `add_repeat` and
            // `mul_repeat` share. `MUL`'s own matrix has no broadcast
            // shape of that size, so promoting it would be certifying a
            // path this lane did not exercise while its twin fails it,
            // a false positive the lane must refuse.
            backend_policy: vec![
                (
                    Cuda,
                    if band.op == BandOp::Scale {
                        PreferGenerated
                    } else {
                        ObserveGenerated
                    },
                ),
                (Vulkan, PreferGenerated),
                (Metal, PreferGenerated),
                (Cpu, NativeOnly),
            ],
            assumed_domain,
            retired_native: vec![],
            native_exception,
        }
    }
}
