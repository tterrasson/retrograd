//! What the interpreter guarantees, on nests written statement by statement.
//!
//! The interpreter is the parity oracle, so it had the worst kind of coverage:
//! it is checked against the Loop IR oracle. Comparing the two interpreters
//! keeps failures local to the lowering or execution semantics. These tests
//! cover the oracle's own **execution model**, which is where the risk moved:
//! shared memory that is genuinely shared, loops that run in lockstep because
//! they contain barriers, and typed register banks.

use rir_core::{ReduceOp, ScalarType};

use crate::interp::{BoundArg, TensorViewMut, run};
use crate::loop_ir::{AddrTerm, Inst, LExpr, MemType, Stmt, VarId, VarKind};
use crate::probe::{I, Y, kernel};
use crate::schedule::{GpuBackend, Schedule, bench};

fn store_lane(value: VarId, at: VarId) -> Stmt {
    Stmt::Store {
        arg: Y,
        ty: MemType::F32,
        addr: vec![AddrTerm::VarNb { var: at, dim: 0 }],
        value,
        width: 1,
        bound: None,
    }
}

/// A store by one lane, read by its neighbour after a barrier.
///
/// Two properties in one nest, and neither could be observed if
/// shared memory were not a declaration. That the shared array is **not** per-lane: it
/// lived in `Regs`, which the interpreter clones per lane, and every lane then
/// read back exactly what it had written. And that `LExpr::Copy` of an index
/// register stays in the **integer** bank: it was evaluated in the F32 bank
/// always, which pinned every copied index at zero - the bug the lowered scan
/// surfaced, on a path no kernel had taken until then.
#[test]
fn a_lane_reads_what_its_neighbour_wrote_through_shared_memory() {
    const LANES: u32 = 4;
    let lk = kernel(
        &[
            ("lane", VarKind::Idx),
            ("sh", VarKind::F32),
            ("mine", VarKind::Idx),
            ("as_f", VarKind::F32),
            ("next", VarKind::Idx),
            ("wrapped", VarKind::Idx),
            ("got", VarKind::F32),
        ],
        vec![(VarId(1), LANES)],
        vec![Stmt::ParallelLane {
            var: VarId(0),
            lanes: LANES,
            body: vec![
                Stmt::Compute(Inst {
                    dst: VarId(2),
                    expr: LExpr::Copy(VarId(0)),
                }),
                Stmt::Compute(Inst {
                    dst: VarId(3),
                    expr: LExpr::IToF(VarId(2)),
                }),
                Stmt::StoreShared {
                    array: VarId(1),
                    index: VarId(2),
                    value: VarId(3),
                },
                Stmt::Barrier,
                Stmt::Compute(Inst {
                    dst: VarId(4),
                    expr: LExpr::IAddC(VarId(0), 1),
                }),
                Stmt::Compute(Inst {
                    dst: VarId(5),
                    expr: LExpr::IModC(VarId(4), LANES),
                }),
                Stmt::LoadShared {
                    dst: VarId(6),
                    array: VarId(1),
                    index: VarId(5),
                    width: 1,
                },
                store_lane(VarId(6), VarId(0)),
            ],
        }],
        bench::vulkan_shared_scan(),
    );

    let mut out = vec![-1f32; LANES as usize];
    let mut args = [BoundArg::Out(TensorViewMut::contiguous_1d(
        &mut out,
        LANES as usize,
    ))];
    run(&lk, &mut args, &[]).unwrap();
    assert_eq!(out, vec![1.0, 2.0, 3.0, 0.0]);
}

/// A loop whose body contains a barrier, run twice.
///
/// This is the property that shapes the interpreter's execution model:
/// a loop over lanes cannot be played lane by lane once its body
/// synchronizes. Each round rotates every lane's value by one position, so two
/// rounds must give `(lane + 2) % 4`. Lane-at-a-time execution gives `lane + 1`
/// twice for lane 0 and garbage afterwards, because lane 0 would finish both
/// iterations before lane 1 wrote anything.
#[test]
fn a_loop_that_contains_a_barrier_runs_in_lockstep() {
    const LANES: u32 = 4;
    let lk = kernel(
        &[
            ("lane", VarKind::Idx),
            ("sh", VarKind::F32),
            ("acc", VarKind::F32),
            ("next", VarKind::Idx),
            ("wrapped", VarKind::Idx),
            ("got", VarKind::F32),
            ("k", VarKind::Idx),
        ],
        vec![(VarId(1), LANES)],
        vec![Stmt::ParallelLane {
            var: VarId(0),
            lanes: LANES,
            body: vec![
                Stmt::Compute(Inst {
                    dst: VarId(2),
                    expr: LExpr::IToF(VarId(0)),
                }),
                Stmt::Compute(Inst {
                    dst: VarId(3),
                    expr: LExpr::IAddC(VarId(0), 1),
                }),
                Stmt::Compute(Inst {
                    dst: VarId(4),
                    expr: LExpr::IModC(VarId(3), LANES),
                }),
                Stmt::ForConst {
                    var: VarId(6),
                    count: 2,
                    body: vec![
                        Stmt::StoreShared {
                            array: VarId(1),
                            index: VarId(0),
                            value: VarId(2),
                        },
                        Stmt::Barrier,
                        Stmt::LoadShared {
                            dst: VarId(5),
                            array: VarId(1),
                            index: VarId(4),
                            width: 1,
                        },
                        Stmt::Barrier,
                        Stmt::Set {
                            var: VarId(2),
                            value: VarId(5),
                        },
                    ],
                },
                store_lane(VarId(2), VarId(0)),
            ],
        }],
        bench::vulkan_shared_scan(),
    );

    let mut out = vec![-1f32; LANES as usize];
    let mut args = [BoundArg::Out(TensorViewMut::contiguous_1d(
        &mut out,
        LANES as usize,
    ))];
    run(&lk, &mut args, &[]).unwrap();
    assert_eq!(out, vec![2.0, 3.0, 0.0, 1.0]);
}

/// `LaneReduce` broadcasts to every lane, and `LaneScan` is **exclusive**: on
/// lanes holding `0, 1, 2, 3` the scan gives `0, 0, 1, 3` - lane `l` receives
/// the total of the lanes *before* it, which is what the blocked scan adds to
/// its chunk.
///
/// Both are backend primitives the emitters print in one call, which is exactly
/// why the oracle's version of them needs a witness of its own: nothing else in
/// the repository states what they mean, and a device comparison would agree
/// with a wrong oracle only by failing.
#[test]
fn lane_reduce_broadcasts_and_lane_scan_is_exclusive() {
    const LANES: u32 = 4;
    for (op, scan, expect) in [
        (ReduceOp::Sum, true, vec![0.0, 0.0, 1.0, 3.0]),
        (ReduceOp::Sum, false, vec![6.0, 6.0, 6.0, 6.0]),
        (ReduceOp::Max, false, vec![3.0, 3.0, 3.0, 3.0]),
    ] {
        let lk = kernel(
            &[
                ("lane", VarKind::Idx),
                ("v", VarKind::F32),
                ("out", VarKind::F32),
            ],
            vec![],
            vec![Stmt::ParallelLane {
                var: VarId(0),
                lanes: LANES,
                body: vec![
                    Stmt::Compute(Inst {
                        dst: VarId(1),
                        expr: LExpr::IToF(VarId(0)),
                    }),
                    if scan {
                        Stmt::LaneScan {
                            op,
                            src: VarId(1),
                            dst: VarId(2),
                        }
                    } else {
                        Stmt::LaneReduce {
                            op,
                            src: VarId(1),
                            dst: VarId(2),
                        }
                    },
                    store_lane(VarId(2), VarId(0)),
                ],
            }],
            Schedule::gpu_subgroup(GpuBackend::Vulkan),
        );
        let mut out = vec![-1f32; LANES as usize];
        let mut args = [BoundArg::Out(TensorViewMut::contiguous_1d(
            &mut out,
            LANES as usize,
        ))];
        run(&lk, &mut args, &[]).unwrap();
        assert_eq!(out, expect, "{op:?} scan={scan}");
    }
}

/// A parallel axis is walked by index, and a store lands where its address says.
///
/// The smallest end-to-end statement the oracle has, and the one every other
/// test here reads its result through: without it a failure above could be a
/// wrong `Store` rather than a wrong collective.
#[test]
fn a_parallel_axis_writes_one_element_per_index() {
    let lk = kernel(
        &[("i", VarKind::Idx), ("v", VarKind::F32)],
        vec![],
        vec![Stmt::Parallel {
            var: VarId(0),
            axis: I,
            level: crate::loop_ir::HwLevel::Global(0),
            vector: 1,
            bounded: false,
            body: vec![
                Stmt::Compute(Inst {
                    dst: VarId(1),
                    expr: LExpr::IToF(VarId(0)),
                }),
                store_lane(VarId(1), VarId(0)),
            ],
        }],
        Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    );
    let mut out = vec![-1f32; 5];
    let mut args = [BoundArg::Out(TensorViewMut::contiguous_1d(&mut out, 5))];
    run(&lk, &mut args, &[]).unwrap();
    assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
}

/// An argument bound to a view whose shape or stride the kernel cannot address
/// is an `InterpError`, not a wrong number.
#[test]
fn a_binding_that_does_not_fit_the_kernel_is_an_error() {
    let lk = kernel(
        &[("i", VarKind::Idx), ("v", VarKind::F32)],
        vec![],
        vec![Stmt::Parallel {
            var: VarId(0),
            axis: I,
            level: crate::loop_ir::HwLevel::Global(0),
            vector: 1,
            bounded: false,
            body: vec![
                Stmt::Compute(Inst {
                    dst: VarId(1),
                    expr: LExpr::ConstF32(1.0),
                }),
                store_lane(VarId(1), VarId(0)),
            ],
        }],
        Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    );
    // Four indices declared, three floats behind the view.
    let mut out = vec![0f32; 3];
    let mut args = [BoundArg::Out(TensorViewMut {
        data: &mut out,
        shape: [4, 1, 1, 1],
        nb: [4, 16, 16, 16],
    })];
    let err = run(&lk, &mut args, &[]).unwrap_err();
    assert!(
        format!("{err}").contains("out of bounds") || format!("{err}").contains("buffer"),
        "unexpected error: {err}"
    );
}

/// A parameter the nest reads is the one the caller passed, by position.
#[test]
fn a_parameter_is_read_by_position() {
    let mut lk = kernel(
        &[("i", VarKind::Idx), ("v", VarKind::F32)],
        vec![],
        vec![Stmt::Parallel {
            var: VarId(0),
            axis: I,
            level: crate::loop_ir::HwLevel::Global(0),
            vector: 1,
            bounded: false,
            body: vec![
                Stmt::Compute(Inst {
                    dst: VarId(1),
                    expr: LExpr::Param(rir_core::ParamId(1)),
                }),
                store_lane(VarId(1), VarId(0)),
            ],
        }],
        Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    );
    lk.params = vec![
        rir_core::ParamDecl {
            name: "eps".into(),
            ty: ScalarType::F32,
        },
        rir_core::ParamDecl {
            name: "scale".into(),
            ty: ScalarType::F32,
        },
    ];
    let mut out = vec![0f32; 2];
    let mut args = [BoundArg::Out(TensorViewMut::contiguous_1d(&mut out, 2))];
    run(&lk, &mut args, &[1e-6, 0.25]).unwrap();
    assert_eq!(out, vec![0.25, 0.25]);
}
