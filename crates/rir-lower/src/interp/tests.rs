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

/// A byte view does not imply a quantized block stride: dense F16 inputs are
/// byte views too. Reading the next element must skip the padding half.
#[test]
fn dense_f16_element_offsets_keep_the_tensor_stride() {
    use crate::interp::{Regs, TensorViewBytes, exec::exec_load, f32_to_f16};

    let mut lk = kernel(
        &[("v", VarKind::F32)],
        vec![],
        vec![],
        Schedule::cpu_serial(),
    );
    lk.args[0].access = rir_core::Access::Read;
    lk.args[0].ty.dtype = rir_core::DType::F16;
    let bytes: Vec<u8> = [1.0, -99.0, 2.0]
        .into_iter()
        .flat_map(|v| f32_to_f16(v).to_le_bytes())
        .collect();
    let args = [BoundArg::InBytes(TensorViewBytes {
        data: &bytes,
        shape: [2, 1, 1, 1],
        nb: [4, 8, 8, 8],
    })];
    let mut regs = Regs {
        f: vec![0.0],
        i: vec![0],
        vlane: 0,
        m: Default::default(),
    };
    exec_load(&lk, &args, &mut regs, VarId(0), Y, MemType::F16, &[], 1).unwrap();
    assert_eq!(regs.f[0], 2.0);
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
            tile: 1,
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
            tile: 1,
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

/// `Schedule::stage_b` moves the shared operand's reads from global memory to
/// a tile the workgroup stages once per reduction round. That is a memory
/// decision, not an arithmetic one: the accumulated terms are the same, in the
/// same order, so the staged lowering is **bit-identical** to the unstaged one.
#[test]
fn a_staged_operand_is_bit_identical_to_the_unstaged_lowering() {
    use rir_core::{
        Constraint, DType, Extent, KernelBuilder, Layout, QuantType, ReduceOp, ReductionSemantics,
        TensorType,
    };

    use crate::interp::{BoundArg, TensorView, TensorViewBytes, TensorViewMut};
    use crate::lower::lower;

    // dst[row, plane] = Σ_col dequant(a[col, row]) · b[col, plane], the decode
    // GEMV's shape, with an additional row-dependent dense operand. Only b
    // can be shared across rows; the additional operand must stay per-row.
    let format = QuantType::Q8_0;
    let mut k = KernelBuilder::new("gemv");
    let a = k.input(
        "a",
        TensorType {
            dtype: DType::Quant(format),
            rank: 2,
            layout: Layout::Ggml,
        },
    );
    let b = k.input("b", TensorType::f32_2d());
    let row_b = k.input("row_b", TensorType::f32_2d());
    let dst = k.output("dst", TensorType::f32_2d());
    let row = k.axis("row", Extent::Dim { arg: a, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: b, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: a, dim: 0 });
    let av = k.read(a, &[col, row]);
    let bv = k.read(b, &[col, plane]);
    let row_bv = k.read(row_b, &[col, row]);
    let bv = k.add(bv, row_bv);
    let p = k.mul(av, bv);
    let s = k.reduce(ReduceOp::Sum, col, p, ReductionSemantics::Deterministic);
    k.write(dst, &[row, plane], s);
    k.constrain(Constraint::DType {
        arg: a,
        allowed: vec![DType::Quant(format)],
    });
    k.constrain(Constraint::DType {
        arg: b,
        allowed: vec![DType::F32],
    });
    let kernel = k.finish().unwrap();

    let (blocks, n_row, n_tok) = (6usize, 5usize, 6usize);
    let d = format.desc();
    let (be, bb) = (d.block_elements as usize, d.block_bytes as usize);
    let n_col = blocks * be;
    let row_bytes = blocks * bb;
    let mut seed = 0x51ed_u64;
    let mut raw = Vec::with_capacity(row_bytes * n_row);
    for _ in 0..n_row {
        raw.extend_from_slice(&rir_core::random_block_bytes(format, blocks, &mut seed).unwrap());
    }
    let x: Vec<f32> = (0..n_col * n_tok)
        .map(|i| (i as f32 * 0.37 - 1.0) * 0.5)
        .collect();
    let row_x: Vec<f32> = (0..n_col * n_row)
        .map(|i| (i as f32 * 0.19 + 0.5) * 0.3)
        .collect();

    let base = || {
        Schedule::gpu_subgroup_cols(GpuBackend::Metal, 4)
            .with_reduce_run(4)
            .with_segment(16)
            .with_row_tile(2, 2)
            .with_factored_scales()
    };
    let run_one = |schedule: Schedule| {
        let lk = lower(&kernel, schedule).unwrap();
        let mut got = vec![0f32; n_row * n_tok];
        let mut args = [
            BoundArg::InBytes(TensorViewBytes {
                data: &raw,
                shape: [n_col, n_row, 1, 1],
                nb: [bb, row_bytes, row_bytes * n_row, row_bytes * n_row],
            }),
            BoundArg::In(TensorView {
                data: &x,
                shape: [n_col, n_tok, 1, 1],
                nb: [4, 4 * n_col, 4 * n_col * n_tok, 4 * n_col * n_tok],
            }),
            BoundArg::In(TensorView {
                data: &row_x,
                shape: [n_col, n_row, 1, 1],
                nb: [4, 4 * n_col, 4 * n_col * n_row, 4 * n_col * n_row],
            }),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut got, n_row, n_tok)),
        ];
        run(&lk, &mut args, &[]).unwrap();
        got.iter().map(|v| v.to_bits()).collect::<Vec<u32>>()
    };
    let plain = run_one(base());
    let staged = run_one(base().with_staged_operand());
    assert_eq!(plain, staged);
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
            tile: 1,
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

/// `erf` in `f64`, independent of the code under test: the Maclaurin series
/// where it converges without cancelling (|x| < 2.5), and the continued
/// fraction of `erfc` beyond, where `1 - erfc` has no cancellation problem.
fn erf_reference(x: f64) -> f64 {
    let a = x.abs();
    let value = if a < 2.5 {
        let (mut term, mut sum, mut n) = (a, a, 0.0f64);
        while term.abs() > 1e-18 {
            n += 1.0;
            term *= -a * a / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * std::f64::consts::FRAC_2_SQRT_PI
    } else {
        // erfc(a) = exp(-a²)/√π · 1/(a + ½/(a + 1/(a + 3/2/(a + …)))), from the tail.
        let mut f = a;
        for k in (1..=80).rev() {
            f = a + (f64::from(k) / 2.0) / f;
        }
        1.0 - (-a * a).exp() / (std::f64::consts::PI.sqrt() * f)
    };
    value.copysign(x)
}

/// The oracle's `erff` against the `f64` reference, over both signs and across
/// each region `erff` switches formula at (0.84375, 1.25, 1/0.35, 6). Four ulps
/// of slack: the point is that a wrong coefficient shows, and one does by
/// orders of magnitude, not that `erff` is correctly rounded.
#[test]
fn the_oracle_erf_is_accurate_to_a_few_ulps_across_every_region() {
    use crate::interp::erff::erf;

    let mut worst = 0.0f64;
    for i in -12000..=12000 {
        let x = i as f32 * 0.0005;
        let want = erf_reference(f64::from(x));
        let got = f64::from(erf(x));
        let w32 = want.abs() as f32;
        let ulp = f64::from(f32::from_bits(w32.to_bits() + 1) - w32);
        let err = (got - want).abs() / ulp.max(1e-45);
        worst = worst.max(err);
        assert!(
            err <= 4.0,
            "erf({x}) = {got}, reference {want}: {err:.1} ulps"
        );
    }
    assert!(
        worst > 0.0,
        "a series that agrees bit for bit is not independent"
    );
    assert_eq!(erf(f32::INFINITY), 1.0);
    assert_eq!(erf(f32::NEG_INFINITY), -1.0);
    assert!(erf(f32::NAN).is_nan());
    assert_eq!(erf(10.0), 1.0);
    assert_eq!(erf(-10.0), -1.0);
}
