use super::*;
use rir_core::{
    DType, Extent, KernelBuilder, ReduceOp, ReductionSemantics, ScanDirection, ScanOp, TensorType,
};

use crate::lower::dequant::format_span;
use crate::schedule::{ReductionStrategy, ScheduleError, bench};

fn kernel_with_reduce(sem: ReductionSemantics) -> rir_core::ValidatedKernel {
    let mut k = KernelBuilder::new("sum_rows");
    let x = k.input("x", TensorType::f32_2d());
    let y = k.output("y", TensorType::f32_2d());
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = k.read(x, &[col, row]);
    let s = k.reduce(ReduceOp::Sum, col, xv, sem);
    k.write(y, &[col, row], s);
    k.finish().unwrap()
}

fn kernel_cumsum(dir: ScanDirection) -> rir_core::ValidatedKernel {
    let mut k = KernelBuilder::new("cumsum");
    let x = k.input("x", TensorType::f32_2d());
    let y = k.output("y", TensorType::f32_2d());
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = k.read(x, &[col, row]);
    let s = k.scan(ScanOp::Sum, col, dir, xv);
    k.write(y, &[col, row], s);
    k.finish().unwrap()
}

/// The blocked scan is an *algorithm*, not a syntax rewrite: this test runs
/// it - and the strided scan beside it - through the oracle on lengths not divisible by lane count and compares
/// it with the sequential scan. It checks decomposition - block geometry,
/// exclusive prefix, reverse direction - independently of any GPU.
///
/// The tolerance is for regrouped additions, exactly what `Deterministic`
/// permits and `ExactOrder` forbids.
#[test]
fn blocked_scan_matches_the_sequential_scan() {
    use crate::interp::{BoundArg, TensorView, TensorViewMut, run};

    for dir in [ScanDirection::Forward, ScanDirection::Backward] {
        let k = kernel_cumsum(dir);
        let seq = lower(&k, Schedule::cpu_serial()).unwrap();
        for blocked in [
            lower(&k, Schedule::vulkan_blocked_scan()).unwrap(),
            lower(&k, bench::vulkan_shared_scan()).unwrap(),
            // Not blocked, but judged by the same property: a regrouped scan
            // that must agree with the sequential one on every length.
            lower(&k, Schedule::gpu_strided_scan(crate::GpuBackend::Vulkan)).unwrap(),
        ] {
            // Blocked scans give up exact order; sequential keeps it, and
            // the manifest must say so.
            assert_eq!(
                seq.reduction_semantics,
                vec![ReductionSemantics::ExactOrder]
            );
            assert_eq!(
                blocked.reduction_semantics,
                vec![ReductionSemantics::Deterministic]
            );

            // Lengths around 32 lanes and beyond: a block shorter than lane
            // count, an exact block, and a block with remainder.
            for &(n_col, n_row) in &[
                (1usize, 1usize),
                (7, 3),
                (32, 2),
                (33, 2),
                (200, 3),
                (257, 2),
            ] {
                let x: Vec<f32> = (0..n_col * n_row)
                    .map(|i| ((i * 37 % 101) as f32 - 50.0) / 8.0)
                    .collect();
                let mut expected = vec![0f32; x.len()];
                let mut got = vec![0f32; x.len()];

                let mut args = [
                    BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
                    BoundArg::Out(TensorViewMut::contiguous_2d(&mut expected, n_col, n_row)),
                ];
                run(&seq, &mut args, &[]).unwrap();

                let mut args = [
                    BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
                    BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_col, n_row)),
                ];
                run(&blocked, &mut args, &[]).unwrap();

                for (i, (g, e)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                        "{dir:?} {n_col}x{n_row} element {i}: blocked {g} vs sequential {e}"
                    );
                }
            }
        }
    }
}

/// Two reductions at the same level under the shared tree: one collective,
/// hence one series of barriers.
///
/// The test checks **structure** because that is where barriers are decided:
/// an emitter may not move one, so sharing requires Loop IR to present one
/// statement. Value control is the equality test immediately below.
#[test]
fn two_reductions_of_one_level_share_a_single_shared_tree() {
    let k = kernel_two_reductions();
    for schedule in [
        Schedule::vulkan_shared_reduce(),
        Schedule::metal_shared_reduce(),
    ] {
        let lk = lower(&k, schedule).unwrap();
        let Stmt::Parallel { body, .. } = &lk.body[0] else {
            panic!("root")
        };
        let Stmt::ParallelLane { body: lane, .. } = &body[0] else {
            panic!("expected ParallelLane")
        };
        let groups: Vec<usize> = lane
            .iter()
            .filter_map(|s| match s {
                Stmt::WorkgroupReduce { reds, .. } => Some(reds.len()),
                _ => None,
            })
            .collect();
        assert_eq!(
            groups,
            vec![2],
            "expected one collective carrying both accumulators"
        );
        // Shared storage remains per accumulator: the group shares
        // synchronization, not memory.
        assert_eq!(lk.shared.len(), 2);
    }
}

/// Grouping does not change a value: the oracle executes grouped and
/// sequential nests on identical inputs, comparing sums with the tolerance
/// for regrouped additions - the same already allowed by `Deterministic`
/// between lanes.
#[test]
fn the_grouped_tree_computes_what_the_ungrouped_one_computed() {
    use crate::interp::{BoundArg, TensorView, TensorViewMut, run};

    let k = kernel_two_reductions();
    let serial = lower(&k, Schedule::cpu_serial()).unwrap();
    let grouped = lower(&k, Schedule::vulkan_shared_reduce()).unwrap();
    for &(n_col, n_row) in &[(1usize, 1usize), (300, 2), (1024, 3)] {
        let x: Vec<f32> = (0..n_col * n_row)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 16.0)
            .collect();
        let run_one = |lk: &LoopKernel| {
            let mut y = vec![0f32; n_row];
            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
                BoundArg::Out(TensorViewMut::contiguous_1d(&mut y, n_row)),
            ];
            run(lk, &mut args, &[]).unwrap();
            y
        };
        let (a, b) = (run_one(&serial), run_one(&grouped));
        for (r, (g, e)) in b.iter().zip(&a).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                "{n_col}x{n_row} row {r}: grouped {g} vs sequential {e}"
            );
        }
    }
}

/// The two-stage tree computes what the flat one computes, and says what it
/// costs.
///
/// Three assertions, and the last two are the item. The values agree with the
/// sequential reduction within the tolerance `Deterministic` allows - the
/// regrouping is the whole strategy, so this is the bound that applies. The
/// shared storage is one word per **subgroup** and not per lane: 256 lanes are
/// eight words per accumulator, against 256. And the group survives the new
/// stage - two reductions of one level still meet one collective statement, so
/// the barriers stay shared rather than reappearing one per accumulator.
#[test]
fn the_two_stage_tree_computes_what_the_flat_one_computed() {
    use crate::interp::{BoundArg, TensorView, TensorViewMut, run};

    let k = kernel_two_reductions();
    let serial = lower(&k, Schedule::cpu_serial()).unwrap();
    for gpu in [crate::GpuBackend::Vulkan, crate::GpuBackend::Cuda] {
        let hier = lower(&k, Schedule::gpu_hier_reduce(gpu)).unwrap();

        assert_eq!(
            hier.shared.iter().map(|(_, len)| *len).collect::<Vec<_>>(),
            vec![8, 8],
            "one slot per subgroup, one array per accumulator"
        );
        assert_eq!(hier.subgroup_width(), Some(32));
        let Stmt::Parallel { body, .. } = &hier.body[0] else {
            panic!("root")
        };
        let Stmt::ParallelLane { body: lane, .. } = &body[0] else {
            panic!("expected ParallelLane")
        };
        let groups: Vec<usize> = lane
            .iter()
            .filter_map(|s| match s {
                Stmt::WorkgroupReduce {
                    reds,
                    subgroup: Some(32),
                    ..
                } => Some(reds.len()),
                _ => None,
            })
            .collect();
        assert_eq!(groups, vec![2], "one collective carrying both accumulators");

        for &(n_col, n_row) in &[(1usize, 1usize), (300, 2), (1024, 3)] {
            let x: Vec<f32> = (0..n_col * n_row)
                .map(|i| ((i * 31 % 97) as f32 - 48.0) / 16.0)
                .collect();
            let run_one = |lk: &LoopKernel| {
                let mut y = vec![0f32; n_row];
                let mut args = [
                    BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
                    BoundArg::Out(TensorViewMut::contiguous_1d(&mut y, n_row)),
                ];
                run(lk, &mut args, &[]).unwrap();
                y
            };
            let (a, b) = (run_one(&serial), run_one(&hier));
            for (r, (g, e)) in b.iter().zip(&a).enumerate() {
                assert!(
                    (g - e).abs() <= 1e-4 * e.abs().max(1.0),
                    "{n_col}x{n_row} row {r}: two-stage {g} vs sequential {e}"
                );
            }
        }
    }
}

/// A geometry the two stages cannot cover is refused rather than expanded into
/// something else.
#[test]
fn a_hierarchical_geometry_the_two_stages_cannot_cover_is_refused() {
    let k = kernel_two_reductions();
    let gpu = crate::GpuBackend::Vulkan;
    for block in [
        // Not a whole number of subgroups: one partial subgroup's total would be
        // stored by nobody.
        [48u32, 1, 1],
        // More subgroups than one subgroup can reduce: the second stage would
        // need a third.
        [2048, 1, 1],
    ] {
        assert!(
            matches!(
                lower(&k, Schedule::gpu_hier_reduce(gpu).with_block(block)),
                Err(LowerError::HierarchicalTreeGeometry { .. })
            ),
            "block {block:?} should be refused"
        );
    }
}

/// `y[row] = Σx + Σx²`: two independent reductions over the same axis, hence
/// one dependency level - the shape of `rms_norm_back`, without depending
/// on `rir-kernels`, which depends on this crate.
fn kernel_two_reductions() -> rir_core::ValidatedKernel {
    let mut k = KernelBuilder::new("two_sums");
    let x = k.input("x", TensorType::f32_2d());
    let y = k.output("y", TensorType::f32(1));
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = k.read(x, &[col, row]);
    let sq = k.mul(xv, xv);
    let s1 = k.reduce(ReduceOp::Sum, col, xv, ReductionSemantics::Deterministic);
    let s2 = k.reduce(ReduceOp::Sum, col, sq, ReductionSemantics::Deterministic);
    let s = k.add(s1, s2);
    k.write(y, &[row], s);
    k.finish().unwrap()
}

/// `y[row] = Σ_col dequant(x[col, row])`, the shape of `sum_rows_<format>`
/// without depending on `rir-kernels`.
fn kernel_quantized_sum(format: rir_core::QuantType) -> rir_core::ValidatedKernel {
    let mut k = KernelBuilder::new(&format!("sum_rows_{}", format.desc().name));
    let x = k.input(
        "x",
        TensorType {
            dtype: DType::Quant(format),
            rank: 2,
            layout: rir_core::Layout::Ggml,
        },
    );
    let y = k.output("y", TensorType::f32(1));
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = k.read(x, &[col, row]);
    let s = k.reduce(ReduceOp::Sum, col, xv, ReductionSemantics::Deterministic);
    k.write(y, &[row], s);
    k.finish().unwrap()
}

/// Quantized reduction lowers the format **per block**, not per element
/// The header - F16 scale and, for a K quant, the
/// packed scale pair - is read once per segment, while only the payload
/// remains in the element loop.
///
/// The test is structural because sharing is decided there; value is checked
/// format by format against `rir_core::dequantize_row` by `rir-kernels`
/// parity and on device
/// by `sum_rows_quant_on_gpu_against_the_oracle`.
#[test]
fn a_quantized_reduction_reads_its_block_header_once_per_segment() {
    for format in [rir_core::QuantType::Q8_0, rir_core::QuantType::Q4_K] {
        let k = kernel_quantized_sum(format);
        let span = format_span(format);
        assert!(span > 1, "{:?}: one-element segment", format);

        // Under lanes: one segment per lane, aligned to the span.
        let lk = lower(&k, Schedule::vulkan_subgroup()).unwrap();
        let Stmt::Parallel { body, .. } = &lk.body[0] else {
            panic!("root")
        };
        let Stmt::ParallelLane { body: lane, .. } = &body[0] else {
            panic!("expected ParallelLane")
        };
        let Some(Stmt::ForStrided { step, body, .. }) =
            lane.iter().find(|s| matches!(s, Stmt::ForStrided { .. }))
        else {
            panic!("expected segment loop")
        };
        assert_eq!(*step, 32 * span, "one lanes · span step");
        check_segment(body, span, format);

        // Under sequential lowering - the oracle - the same shape from
        // zero: this makes the CPU witness judge the nest executed by the
        // device.
        let lk = lower(&k, Schedule::cpu_serial()).unwrap();
        let Stmt::Parallel { body, .. } = &lk.body[0] else {
            panic!("root")
        };
        let Some(Stmt::ForTiled { step, body, .. }) =
            body.iter().find(|s| matches!(s, Stmt::ForTiled { .. }))
        else {
            panic!("expected segment loop")
        };
        assert_eq!(*step, span);
        check_segment(body, span, format);
    }
}

/// A segment body: header reads outside, a `span`-iteration element loop,
/// and no scale reread within it.
fn check_segment(body: &[Stmt], span: u32, format: rir_core::QuantType) {
    let header_loads = body
        .iter()
        .filter(|s| matches!(s, Stmt::Load { .. }))
        .count();
    assert!(
        header_loads >= 1,
        "{:?}: no header read at segment level",
        format
    );
    let Some(Stmt::ForConst { count, body, .. }) =
        body.iter().find(|s| matches!(s, Stmt::ForConst { .. }))
    else {
        panic!("{:?}: expected element loop", format)
    };
    assert_eq!(*count, span);
    // The scale is an F16 access: none should remain per element. The
    // payload is a byte and correctly remains there.
    assert!(
        !body.iter().any(|s| matches!(
            s,
            Stmt::Load {
                ty: MemType::F16,
                ..
            }
        )),
        "{:?}: an F16 scale is reread per element",
        format
    );
}

#[test]
fn shared_tree_refuses_a_non_power_of_two_workgroup() {
    let k = kernel_cumsum(ScanDirection::Forward);
    let bad = Schedule {
        block: [192, 1, 1],
        ..bench::vulkan_shared_scan()
    };
    assert_eq!(
        lower(&k, bad).err(),
        Some(LowerError::SharedTreeRequiresPowerOfTwo { lanes: 192 })
    );
}

/// The tiled scan requires the **shared** tree, and rejection prevents two
/// distinct faults from passing silently: under `SubgroupTree`, the manifest
/// would publish a subgroup collective unused by every shader line, and the
/// power-of-two check would no longer apply - a Blelloch tree over 192 lanes
/// combines the wrong lanes and returns an incorrect prefix instead of an
/// error.
#[test]
fn the_tiled_scan_refuses_anything_but_the_shared_tree() {
    let k = kernel_cumsum(ScanDirection::Forward);
    let subgroup = Schedule {
        reduction: ReductionStrategy::SubgroupTree,
        ..Schedule::vulkan_tiled_scan(32, 4)
    };
    assert!(matches!(
        lower(&k, subgroup).err(),
        Some(LowerError::TiledScanUnsupported { .. })
    ));

    // Once the shared tree is required, so is its geometry: this is the only
    // reason both rejections belong in one test.
    assert_eq!(
        lower(&k, Schedule::vulkan_tiled_scan(192, 4)).err(),
        Some(LowerError::SharedTreeRequiresPowerOfTwo { lanes: 192 })
    );
}

/// A blocked scan requested without a collective and a sequential scan under
/// a collective strategy are both schedule errors - not
/// silent fallbacks to the other form.
#[test]
fn scan_strategy_and_reduction_strategy_must_agree() {
    let k = kernel_cumsum(ScanDirection::Forward);
    let without_lanes = Schedule {
        scan: ScanStrategy::BlockedLanes,
        ..Schedule::vulkan_grid([64, 1, 1])
    };
    assert_eq!(
        lower(&k, without_lanes).err(),
        Some(LowerError::BlockedScanRequiresLanes)
    );
    assert_eq!(
        lower(&k, Schedule::vulkan_subgroup()).err(),
        Some(LowerError::ScanRequiresSerial)
    );
}

#[test]
fn exact_order_refuses_a_reassociating_schedule() {
    let k = kernel_with_reduce(ReductionSemantics::ExactOrder);
    assert!(matches!(
        lower(&k, Schedule::vulkan_subgroup()),
        Err(LowerError::Schedule(
            ScheduleError::IncompatibleReduction { .. }
        ))
    ));
}

#[test]
fn serial_lowering_produces_parallel_then_two_loops() {
    let k = kernel_with_reduce(ReductionSemantics::Deterministic);
    let lk = lower(&k, Schedule::cpu_serial()).unwrap();
    assert_eq!(lk.body.len(), 1);
    match &lk.body[0] {
        Stmt::Parallel { body, .. } => {
            let fors = body
                .iter()
                .filter(|s| matches!(s, Stmt::For { .. }))
                .count();
            assert_eq!(fors, 2, "phase 1 (accumulation) + phase 2 (writes)");
            assert!(matches!(body[0], Stmt::InitAcc { .. }));
        }
        other => panic!("unexpected root: {other:?}"),
    }
}

#[test]
fn lane_lowering_produces_forstrided_and_lanereduce() {
    let k = kernel_with_reduce(ReductionSemantics::Deterministic);
    let lk = lower(&k, Schedule::vulkan_subgroup()).unwrap();
    let Stmt::Parallel { body, .. } = &lk.body[0] else {
        panic!("root")
    };
    let Stmt::ParallelLane {
        lanes,
        body: lane_body,
        ..
    } = &body[0]
    else {
        panic!("expected ParallelLane")
    };
    assert_eq!(*lanes, 32);
    assert!(
        lane_body
            .iter()
            .any(|s| matches!(s, Stmt::LaneReduce { .. }))
    );
    let strided = lane_body
        .iter()
        .filter(|s| matches!(s, Stmt::ForStrided { .. }))
        .count();
    assert_eq!(strided, 2);
}

/// With a grid schedule, two parallel axes each receive a grid dimension;
/// neither falls back to a sequential loop.
#[test]
fn the_invocation_mapping_carries_every_parallel_axis_to_the_grid() {
    let mut kb = KernelBuilder::new("mat_mul");
    let a = kb.input("a", TensorType::f32_2d());
    let b = kb.input("b", TensorType::f32_2d());
    let c = kb.output("c", TensorType::f32_2d());
    let i = kb.axis("i", Extent::Dim { arg: a, dim: 1 });
    let j = kb.axis("j", Extent::Dim { arg: b, dim: 1 });
    let kk = kb.axis("k", Extent::Dim { arg: a, dim: 0 });
    let av = kb.read(a, &[kk, i]);
    let bv = kb.read(b, &[kk, j]);
    let p = kb.mul(av, bv);
    let s = kb.reduce(ReduceOp::Sum, kk, p, ReductionSemantics::Deterministic);
    kb.write(c, &[i, j], s);
    let k = kb.finish().unwrap();

    let lk = lower(&k, Schedule::vulkan_grid([16, 16, 1])).unwrap();
    let Stmt::Parallel { level, body, .. } = &lk.body[0] else {
        panic!("root")
    };
    assert_eq!(*level, HwLevel::Global(0));
    let Stmt::Parallel { level, .. } = &body[0] else {
        panic!("second parallel axis")
    };
    assert_eq!(*level, HwLevel::Global(1));
}

/// `vector_width` is a *decision*, so a lowering that cannot honour it must
/// say so. The failure mode this prevents:
/// a field declared in `Schedule`, read by nobody, and silently worth 1.
#[test]
fn a_width_the_lowering_cannot_honour_is_an_error() {
    let with_inner = kernel_with_reduce(ReductionSemantics::Deterministic);
    assert!(matches!(
        lower(&with_inner, Schedule::vulkan_grid_vec4([256, 1, 1])),
        Err(LowerError::VectorWidthUnsupported { .. })
    ));

    // Elementwise, but under a workgroup mapping: the lanes of the
    // workgroup already share the index, so there is nothing to widen.
    let mut kb = KernelBuilder::new("scale");
    let x = kb.input("x", TensorType::f32_2d());
    let y = kb.output("y", TensorType::f32_2d());
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = kb.read(x, &[col, row]);
    kb.write(y, &[col, row], xv);
    let elementwise = kb.finish().unwrap();

    // GLSL and MSL expose only native vectors of two to four components.
    // Rejecting width here avoids invalid `vec5`/`float5` and, on Vulkan, a
    // panic while selecting `.x/.y/.z/.w`.
    //
    // The two schedules below are written as struct literals on purpose:
    // No constructor produces either of them
    // - there is no five-wide constructor, and `gpu_grid_vec4` cannot take
    // `Backend::Cpu` - so an external caller can no longer reach these
    // rejections at all. They stay tested from inside the crate because the
    // lowering must keep refusing what it cannot print, not because a
    // caller can still ask.
    assert!(matches!(
        lower(
            &elementwise,
            Schedule {
                vector_width: 5,
                ..Schedule::vulkan_grid([256, 1, 1])
            }
        ),
        Err(LowerError::VectorWidthUnsupported { .. })
    ));

    assert!(matches!(
        lower(
            &elementwise,
            Schedule {
                vector_width: 4,
                ..Schedule::cpu_serial()
            }
        ),
        Err(LowerError::VectorWidthUnsupported { .. })
    ));
}

/// The vectorized nest, checked structurally: the contiguous axis carries
/// the width, the tail branch exists, and the vector half really does load
/// four elements at a time. Without the last assertion the variant could
/// ship as a scalar copy of the fallback under a different name.
#[test]
fn the_vectorized_nest_carries_the_width_and_a_tail() {
    let k = rir_core::KernelBuilder::new("elem");
    let mut kb = k;
    let x = kb.input("x", TensorType::f32_2d());
    let y = kb.output("y", TensorType::f32_2d());
    // `col` first: the first parallel axis is the one that takes grid
    // dimension x, and widening anything but the contiguous axis is
    // refused - the guard that the elementwise band's declaration order
    // exists to satisfy.
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let xv = kb.read(x, &[col, row]);
    let t = kb.mul(xv, xv);
    kb.write(y, &[col, row], t);
    let kernel = kb.finish().unwrap();

    let lk = lower(&kernel, Schedule::vulkan_grid_vec4([256, 1, 1])).unwrap();
    assert_eq!(lk.vector_width(), 4);
    let Stmt::Parallel { vector, body, .. } = &lk.body[0] else {
        panic!("root")
    };
    assert_eq!(*vector, 4, "the contiguous axis carries the width");
    let Stmt::Parallel { vector, body, .. } = &body[0] else {
        panic!("second parallel axis")
    };
    assert_eq!(*vector, 1, "only the contiguous axis is vectorized");
    let Some(Stmt::VecTail {
        width,
        vec_body,
        tail_body,
        ..
    }) = body.iter().find(|s| matches!(s, Stmt::VecTail { .. }))
    else {
        panic!("expected vector branch/tail")
    };
    assert_eq!(*width, 4);
    assert!(
        vec_body
            .iter()
            .any(|s| matches!(s, Stmt::Load { width: 4, .. })),
        "the vector half reads four at a time"
    );
    assert!(
        tail_body
            .iter()
            .all(|s| !matches!(s, Stmt::Load { width: 4, .. })),
        "the tail remains scalar"
    );
}

/// An elementwise kernel has nothing to contribute to a collective, so a
/// lane strategy rejects it instead of panicking over a missing inner axis.
#[test]
fn lanes_refuses_a_kernel_with_no_inner_axis() {
    let mut kb = KernelBuilder::new("scale");
    let x = kb.input("x", TensorType::f32_2d());
    let y = kb.output("y", TensorType::f32_2d());
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = kb.read(x, &[col, row]);
    let t = kb.mul(xv, xv);
    kb.write(y, &[col, row], t);
    let k = kb.finish().unwrap();

    assert!(matches!(
        lower(&k, Schedule::vulkan_subgroup()),
        Err(LowerError::LanesRequireInnerAxis)
    ));
}

/// Reserving a 32-lane workgroup per index without a collective is a
/// schedule error, not merely a slow shader.
#[test]
fn serial_refuses_a_wide_workgroup() {
    let k = kernel_with_reduce(ReductionSemantics::Deterministic);
    let mut bad = Schedule::vulkan_grid([64, 1, 1]);
    bad.par_map = ParallelMapping::Workgroup;
    assert!(matches!(
        lower(&k, bad),
        Err(LowerError::MappingMismatch { .. })
    ));
}

#[test]
fn a_row_level_store_is_hoisted_out_of_the_inner_loop() {
    // y[row] = Σ_col x: the write is independent of col, so emit it once
    // per row rather than inside the inner loop.
    let mut kb = KernelBuilder::new("sum_rows_1d");
    let x = kb.input("x", TensorType::f32_2d());
    let y = kb.output("y", TensorType::f32(1));
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let xv = kb.read(x, &[col, row]);
    let s = kb.reduce(ReduceOp::Sum, col, xv, ReductionSemantics::Deterministic);
    kb.write(y, &[row], s);
    let k = kb.finish().unwrap();

    let lk = lower(&k, Schedule::cpu_serial()).unwrap();
    let Stmt::Parallel { body, .. } = &lk.body[0] else {
        panic!("root")
    };
    let fors = body
        .iter()
        .filter(|s| matches!(s, Stmt::For { .. }))
        .count();
    assert_eq!(fors, 1, "only one inner loop: phase 1");
    assert!(
        body.iter().any(|s| matches!(s, Stmt::Store { .. })),
        "the store is at row level"
    );
}

#[test]
fn a_backward_scan_produces_a_reversed_loop() {
    let k = kernel_cumsum(ScanDirection::Backward);
    let lk = lower(&k, Schedule::cpu_serial()).unwrap();
    let Stmt::Parallel { body, .. } = &lk.body[0] else {
        panic!("root")
    };
    let Some(Stmt::For {
        reverse,
        body: loop_body,
        ..
    }) = body.iter().find(|s| matches!(s, Stmt::For { .. }))
    else {
        panic!("expected scan loop")
    };
    assert!(*reverse);
    assert!(
        loop_body.iter().any(|s| matches!(s, Stmt::Store { .. })),
        "writes live in the scan loop"
    );
}

#[test]
fn a_scan_refuses_the_lane_strategy() {
    let k = kernel_cumsum(ScanDirection::Forward);
    assert!(matches!(
        lower(&k, Schedule::vulkan_subgroup()),
        Err(LowerError::ScanRequiresSerial)
    ));
}

#[test]
fn three_axes_produce_a_loop_nest() {
    // C[i,j] = Σ_k A[k,i]·B[k,j]
    let mut kb = KernelBuilder::new("mm");
    let a = kb.input("a", TensorType::f32_2d());
    let b = kb.input("b", TensorType::f32_2d());
    let c = kb.output("c", TensorType::f32_2d());
    let i = kb.axis("i", Extent::Dim { arg: a, dim: 1 });
    let j = kb.axis("j", Extent::Dim { arg: b, dim: 1 });
    let kk = kb.axis("k", Extent::Dim { arg: a, dim: 0 });
    let av = kb.read(a, &[kk, i]);
    let bv = kb.read(b, &[kk, j]);
    let p = kb.mul(av, bv);
    let s = kb.reduce(ReduceOp::Sum, kk, p, ReductionSemantics::Deterministic);
    kb.write(c, &[i, j], s);
    let k = kb.finish().unwrap();

    let lk = lower(&k, Schedule::cpu_serial()).unwrap();
    let Stmt::Parallel { body, .. } = &lk.body[0] else {
        panic!("root")
    };
    let Stmt::For { body: j_body, .. } = &body[0] else {
        panic!("expected j loop")
    };
    assert!(j_body.iter().any(|s| matches!(s, Stmt::InitAcc { .. })));
    assert!(j_body.iter().any(|s| matches!(s, Stmt::Store { .. })));
}

/// The flattened nest visits **exactly** the points the
/// grid nest visits, on shapes where the decomposition has something to get
/// wrong: a row that is not a whole number of vectors, an axis of extent one,
/// and a batch the grid nest would have walked sequentially.
///
/// It is the oracle-level half of `family_parity`'s device check, and it is
/// worth having separately because it fails for a *different* reason: here the
/// two lowerings are compared to each other, so a decomposition that skips a
/// point or visits one twice shows up as a wrong element rather than as a
/// disagreement with a device the machine may not have.
#[test]
fn a_flattened_nest_covers_the_same_points_as_the_grid_nest() {
    use crate::interp::{BoundArg, TensorView, TensorViewMut, run};

    // dst[c,r] = x[c,r] · 2 - the elementwise shape, two axes, no collective.
    let mut kb = KernelBuilder::new("twice");
    let x = kb.input("x", TensorType::f32_2d());
    let dst = kb.output("dst", TensorType::f32_2d());
    // The contiguous axis first, as every kernel of the elementwise band
    // declares it: it is the one a vector width applies to.
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let v = kb.read(x, &[col, row]);
    let two = kb.const_f32(2.0);
    let d = kb.mul(v, two);
    kb.write(dst, &[col, row], d);
    let k = kb.finish().unwrap();

    let grid = lower(&k, Schedule::vulkan_grid([256, 1, 1])).unwrap();
    for width in [1u32, 4] {
        let flat = lower(
            &k,
            Schedule::gpu_grid_flat(crate::GpuBackend::Cuda, [256, 1, 1], width),
        )
        .unwrap();
        // The flattening is what the manifest and the registry publish, read
        // off the nest and not off the schedule.
        assert_eq!(
            flat.flat_axes().len(),
            2,
            "two parallel axes, one dimension"
        );
        assert_eq!(flat.flat_axes()[0].1, width);
        assert!(
            grid.flat_axes().is_empty(),
            "the grid nest flattens nothing"
        );

        for &(n_col, n_row) in &[(1usize, 1usize), (7, 3), (32, 2), (33, 5), (200, 3)] {
            let src: Vec<f32> = (0..n_col * n_row).map(|i| i as f32 - 3.0).collect();
            // Filled with a value neither lowering can produce, so a point
            // *nobody* visited is as visible as a point visited twice.
            let mut expected = vec![f32::NAN; src.len()];
            let mut got = vec![f32::NAN; src.len()];

            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&src, n_col, n_row)),
                BoundArg::Out(TensorViewMut::contiguous_2d(&mut expected, n_col, n_row)),
            ];
            run(&grid, &mut args, &[]).unwrap();
            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&src, n_col, n_row)),
                BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_col, n_row)),
            ];
            run(&flat, &mut args, &[]).unwrap();
            assert_eq!(got, expected, "w={width} shape=[{n_col},{n_row}]");
        }
    }
}

/// Linear addressing computes the same tensor as the decomposing flat nest, and
/// computes it from **one** term.
///
/// Two assertions, and the second is the point of the item. Agreeing with the
/// grid nest says the address is right; carrying a single `VarConst` address
/// says the four stride products are gone - a lowering that agreed while still
/// summing `nb[]` would pass the first and deliver nothing.
#[test]
fn linear_addressing_computes_the_same_tensor_from_a_single_term() {
    use crate::interp::{BoundArg, TensorView, TensorViewMut, run};

    let mut kb = KernelBuilder::new("twice");
    let x = kb.input("x", TensorType::f32_2d());
    let dst = kb.output("dst", TensorType::f32_2d());
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let v = kb.read(x, &[col, row]);
    let two = kb.const_f32(2.0);
    let d = kb.mul(v, two);
    kb.write(dst, &[col, row], d);
    let k = kb.finish().unwrap();

    let grid = lower(&k, Schedule::vulkan_grid([256, 1, 1])).unwrap();
    for width in [1u32, 4] {
        let lin = lower(
            &k,
            Schedule::gpu_grid_flat_linear(crate::GpuBackend::Cuda, [256, 1, 1], width),
        )
        .unwrap();
        assert!(lin.linear_addr(), "the nest is what publishes the claim");
        assert_eq!(lin.flat_axes().len(), 2, "the host still decomposes");
        assert_eq!(lin.vector_width(), width);

        // Every access is one term, and the term is the linear register scaled
        // by `width · 4` bytes. Nothing else may appear: a `VarNb` here would be
        // a stride product this lowering exists to remove.
        let mut seen = 0;
        walk_addrs(&lin.body, &mut |addr| {
            seen += 1;
            match addr {
                [AddrTerm::VarConst { c, .. }] => assert_eq!(*c, width * 4),
                other => panic!("w={width}: address is not a single scaled term: {other:?}"),
            }
        });
        assert_eq!(seen, 2, "one read and one write");

        // The claim requires the contiguous extent to be a whole number of
        // vectors, so the shapes tried here are the ones a dispatch site would
        // let through.
        for &(n_col, n_row) in &[(4usize, 1usize), (8, 3), (32, 2), (256, 5)] {
            let src: Vec<f32> = (0..n_col * n_row).map(|i| i as f32 - 3.0).collect();
            let mut expected = vec![f32::NAN; src.len()];
            let mut got = vec![f32::NAN; src.len()];
            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&src, n_col, n_row)),
                BoundArg::Out(TensorViewMut::contiguous_2d(&mut expected, n_col, n_row)),
            ];
            run(&grid, &mut args, &[]).unwrap();
            let mut args = [
                BoundArg::In(TensorView::contiguous_2d(&src, n_col, n_row)),
                BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_col, n_row)),
            ];
            run(&lin, &mut args, &[]).unwrap();
            assert_eq!(got, expected, "w={width} shape=[{n_col},{n_row}]");
        }
    }
}

/// Every address of a nest, in program order.
fn walk_addrs(stmts: &[Stmt], f: &mut impl FnMut(&[AddrTerm])) {
    for s in stmts {
        match s {
            Stmt::Load { addr, .. } | Stmt::Store { addr, .. } => f(addr),
            Stmt::Parallel { body, .. }
            | Stmt::ParallelFlat { body, .. }
            | Stmt::ParallelLane { body, .. }
            | Stmt::For { body, .. }
            | Stmt::ForStrided { body, .. }
            | Stmt::ForChunk { body, .. }
            | Stmt::ForConst { body, .. }
            | Stmt::ForTiled { body, .. }
            | Stmt::InBounds { body, .. }
            | Stmt::If { body, .. }
            | Stmt::LaneZero { body, .. } => walk_addrs(body, f),
            Stmt::VecTail {
                vec_body,
                tail_body,
                ..
            } => {
                walk_addrs(vec_body, f);
                walk_addrs(tail_body, f);
            }
            _ => {}
        }
    }
}

/// The *same shape* half of the claim is proved on the graph, and a kernel that
/// breaks it is refused rather than lowered with a claim it does not honour.
///
/// Three shapes of failure, one per way the flattened point stops being the
/// element: a fold (a second shape), an inner axis (a dimension the linear index
/// does not count), and a schedule that asked for the address without the
/// flattening it is computed from.
#[test]
fn a_linear_address_a_kernel_cannot_honour_is_refused() {
    let cuda = crate::GpuBackend::Cuda;

    // A fold: `b` is replayed under `a`, so the two do not have one shape.
    let mut kb = KernelBuilder::new("add_repeat_2d");
    let a = kb.input("a", TensorType::f32_2d());
    let b = kb.input("b", TensorType::f32_2d());
    let dst = kb.output("dst", TensorType::f32_2d());
    let col = kb.axis("col", Extent::Dim { arg: a, dim: 0 });
    let row = kb.axis("row", Extent::Dim { arg: a, dim: 1 });
    let col_b = kb.axis("col_b", Extent::Dim { arg: b, dim: 0 });
    let row_b = kb.axis("row_b", Extent::Dim { arg: b, dim: 1 });
    let av = kb.read(a, &[col, row]);
    let bv = kb.read_repeat(b, &[col, row], &[col_b, row_b]);
    let s = kb.add(av, bv);
    kb.write(dst, &[col, row], s);
    let repeat = kb.finish().unwrap();
    assert!(matches!(
        lower(
            &repeat,
            Schedule::gpu_grid_flat_linear(cuda, [256, 1, 1], 1)
        ),
        Err(LowerError::LinearAddrUnsupported { .. })
    ));

    // An inner axis: the linear index counts the parallel space alone.
    let reduce = kernel_with_reduce(ReductionSemantics::Deterministic);
    assert!(matches!(
        lower(
            &reduce,
            Schedule::gpu_grid_flat_linear(cuda, [256, 1, 1], 1)
        ),
        Err(LowerError::LinearAddrUnsupported { .. })
    ));

    // The address without the flattening: there is no linear index to scale.
    let mut kb = KernelBuilder::new("twice");
    let x = kb.input("x", TensorType::f32_2d());
    let out = kb.output("dst", TensorType::f32_2d());
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let v = kb.read(x, &[col, row]);
    let two = kb.const_f32(2.0);
    let d = kb.mul(v, two);
    kb.write(out, &[col, row], d);
    let plain = kb.finish().unwrap();
    let unflattened = Schedule {
        linear_addr: true,
        ..Schedule::gpu_grid(cuda, [256, 1, 1])
    };
    assert!(matches!(
        lower(&plain, unflattened),
        Err(LowerError::LinearAddrUnsupported { .. })
    ));
}

/// A flattening the rest of the schedule cannot honour is an **error**, not a
/// nest that quietly keeps its three grid dimensions.
///
/// The failure mode this closes is the one of a silent `vector_width`: the
/// field is published - in the manifest's `flat`, in the registry's `flat[]`,
/// in the constant buffer's divisors - so a lowering that ignored it would be
/// dispatched with a grid computed for a linear index it never reads.
///
/// What is refused is the **crossing** and not the collective, and that is the
/// correction this test makes: a collective flattens over
/// workgroups, everything else over invocations, and it is a linear *invocation*
/// index under a collective - or a workgroup one without - that has no meaning.
#[test]
fn a_flattening_the_mapping_contradicts_is_refused() {
    let k = kernel_with_reduce(ReductionSemantics::Deterministic);
    // The workgroup mapping without a collective: one workgroup per point and
    // nothing for its lanes to do together. The trivial block is what gets past
    // the mapping check above, so the refusal measured here is this one.
    let no_collective = Schedule {
        flatten: true,
        block: [1, 1, 1],
        par_map: ParallelMapping::Workgroup,
        ..Schedule::gpu_grid(crate::GpuBackend::Cuda, [1, 1, 1])
    };
    assert!(matches!(
        lower(&k, no_collective),
        Err(LowerError::FlattenUnsupported { .. })
    ));
}

/// A collective **may** flatten, over workgroups, and what it publishes says so.
///
/// The row space is folded into one dimension, the lane collective is untouched,
/// and the dispatcher is told one workgroup per point - the last being the half
/// a wrong reading would get silently right on a single-row shape.
#[test]
fn a_collective_flattens_its_rows_over_workgroups() {
    let k = kernel_with_reduce(ReductionSemantics::Deterministic);
    let flat_rows = lower(
        &k,
        Schedule::gpu_subgroup_flat_rows(crate::GpuBackend::Cuda),
    )
    .unwrap();
    assert_eq!(flat_rows.flat_axes().len(), 1, "one parallel axis, folded");
    assert!(
        !matches!(flat_rows.body.as_slice(), [Stmt::Parallel { .. }]),
        "and no grid nest beside it"
    );
    assert_eq!(flat_rows.subgroup_width(), Some(32), "the collective stays");
    match flat_rows.body.as_slice() {
        [Stmt::ParallelFlat { level, .. }] => {
            assert_eq!(*level, HwLevel::Grid(0), "one workgroup per row")
        }
        other => panic!("expected a single flattened nest: {other:?}"),
    }
}

/// Four parallel axes and a reduction: the fourth axis has no grid dimension
/// left, so it becomes a sequential loop around the lane body. A shared-memory
/// collective read back from `sh[0]` must not be overwritten by the next round,
/// so every re-entered lane body that uses shared storage ends on a barrier -
/// and one that uses none (a subgroup reduction) is left alone.
#[test]
fn a_shared_collective_under_a_sequential_axis_closes_its_round() {
    let mut k = KernelBuilder::new("scaled_sums");
    let x = k.input("x", TensorType::f32(4));
    let w = k.input("w", TensorType::f32(1));
    let y = k.output("y", TensorType::f32(4));
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
    let e = k.axis("e", Extent::Dim { arg: w, dim: 0 });
    let xv = k.read(x, &[col, row, plane, batch]);
    let wv = k.read(w, &[e]);
    let p = k.mul(xv, wv);
    let s = k.reduce(ReduceOp::Sum, col, p, ReductionSemantics::Deterministic);
    k.write(y, &[row, plane, batch, e], s);
    let k = k.finish().unwrap();

    fn lane_body_under_loop(stmts: &[Stmt], in_loop: bool) -> Option<(bool, Vec<Stmt>)> {
        stmts.iter().find_map(|s| match s {
            Stmt::ParallelLane { body, .. } => Some((in_loop, body.clone())),
            Stmt::For { body, .. } => lane_body_under_loop(body, true),
            Stmt::Parallel { body, .. } => lane_body_under_loop(body, in_loop),
            _ => None,
        })
    }
    for (schedule, shared) in [
        (Schedule::vulkan_shared_reduce(), true),
        (Schedule::gpu_hier_reduce(crate::GpuBackend::Vulkan), true),
        (Schedule::vulkan_subgroup(), false),
    ] {
        let lk = lower(&k, schedule.clone()).unwrap();
        let (in_loop, body) = lane_body_under_loop(&lk.body, false).expect("a lane body");
        assert!(
            in_loop,
            "{:?}: the fourth axis is not a loop",
            schedule.reduction
        );
        assert_eq!(
            matches!(body.last(), Some(Stmt::Barrier)),
            shared,
            "{:?}: a re-entered lane body closes its round exactly when it uses shared storage",
            schedule.reduction
        );
    }
}
