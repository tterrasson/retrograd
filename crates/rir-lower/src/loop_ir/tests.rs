//! What the Loop IR answers about a nest, on nests written by hand.
//!
//! These queries are what the emitters and the runtime build a shader's
//! *declarations* from - buffer views, subgroup requirement, shared storage,
//! vector width. They were covered only through whatever the emitters happened
//! to ask them on real kernels, which is coverage of the kernels and not of the
//! queries: a query that answered wrongly for a statement no kernel produces
//! would be found by the next kernel that did.

use rir_core::{ArgId, LutId, ReduceOp};

use crate::loop_ir::{
    AddrTerm, GroupRed, HwLevel, Inst, LExpr, MemType, Stmt, TileStage, VarId, VarKind,
    child_blocks_of, chunk_range,
};
use crate::probe::{I, Y, kernel};
use crate::schedule::{GpuBackend, Schedule};

fn load(dst: VarId, ty: MemType) -> Stmt {
    Stmt::Load {
        dst,
        arg: Y,
        ty,
        addr: vec![AddrTerm::Const(0)],
        width: 1,
    }
}

fn nest(shared: Vec<(VarId, u32)>, body: Vec<Stmt>) -> crate::loop_ir::LoopKernel {
    kernel(
        &[
            ("i", VarKind::Idx),
            ("a", VarKind::F32),
            ("b", VarKind::F32),
            ("v", VarKind::Vec(4)),
        ],
        shared,
        body,
        Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    )
}

/// `chunk_range` splits `n` into `lanes` contiguous chunks, and mirrors them
/// when the scan runs backwards.
///
/// The blocked scan's whole correctness rests on this: the chunks must tile
/// `0..n` exactly once, forwards and backwards, including when `n` is not a
/// multiple of the lane count and the last lanes get nothing.
#[test]
fn chunk_ranges_tile_the_axis_exactly_once() {
    for n in [0usize, 1, 3, 4, 7, 8, 33] {
        for lanes in [1u32, 2, 4, 32] {
            let mut seen = vec![0u8; n];
            let mut last_end = 0;
            for lane in 0..lanes as usize {
                let (b, e) = chunk_range(n, lane, lanes, false);
                assert!(
                    b <= e && e <= n,
                    "n={n} lanes={lanes} lane={lane}: {b}..{e}"
                );
                assert_eq!(b, last_end, "n={n} lanes={lanes}: chunks are contiguous");
                last_end = e;
                for slot in seen[b..e].iter_mut() {
                    *slot += 1;
                }
            }
            assert_eq!(last_end, n, "n={n} lanes={lanes}: chunks cover the axis");
            assert!(
                seen.iter().all(|c| *c == 1),
                "n={n} lanes={lanes}: {seen:?}"
            );
            // Backwards: the same partition, mirrored, so lane 0 owns the end.
            let mut mirrored = vec![0u8; n];
            for lane in 0..lanes as usize {
                let (b, e) = chunk_range(n, lane, lanes, true);
                for slot in mirrored[b..e].iter_mut() {
                    *slot += 1;
                }
            }
            assert!(
                mirrored.iter().all(|c| *c == 1),
                "n={n} lanes={lanes} reverse"
            );
            if n > 0 {
                assert_eq!(chunk_range(n, 0, lanes, true).1, n, "lane 0 owns the tail");
            }
        }
    }
}

/// `child_blocks_of` reaches every nested block, including the ones that are not
/// called `body`.
///
/// Every pass that walks the nest - hoisting, CSE, the barrier count,
/// uses this as its only enumeration of children. A statement whose block it
/// forgot would be invisible to all of them at once, and silently: a pass that
/// does not see a block does not fail, it optimizes less.
#[test]
fn every_nested_block_is_reachable() {
    let cases: Vec<(Stmt, usize)> = vec![
        (
            Stmt::VecTail {
                base: VarId(0),
                axis: I,
                width: 4,
                vec_body: vec![Stmt::Barrier],
                tail_var: VarId(0),
                tail_body: vec![Stmt::Barrier],
            },
            2,
        ),
        (
            Stmt::StageTiles {
                tiles: vec![TileStage {
                    tile: VarId(0),
                    arg: Y,
                    row: VarId(0),
                    depth: VarId(0),
                    row_global: VarId(0),
                    depth_global: VarId(0),
                    row_origin: VarId(0),
                    depth_origin: VarId(0),
                    row_axis: I,
                    depth_axis: I,
                    slot: VarId(0),
                    span: 1,
                    load: vec![Stmt::Barrier],
                    n_rows: 1,
                    n_depth: 1,
                }],
                threads: 1,
                body: vec![Stmt::Barrier],
            },
            2,
        ),
        (
            Stmt::If {
                cond: VarId(0),
                body: vec![Stmt::Barrier],
            },
            1,
        ),
        (
            Stmt::InBounds {
                bounds: vec![(VarId(0), I)],
                body: vec![Stmt::Barrier],
            },
            1,
        ),
        (Stmt::Barrier, 0),
        (
            Stmt::Set {
                var: VarId(0),
                value: VarId(0),
            },
            0,
        ),
    ];
    for (s, expected) in cases {
        assert_eq!(
            child_blocks_of(&s).len(),
            expected,
            "{:?}",
            std::mem::discriminant(&s)
        );
    }
}

/// The buffer views an argument needs are the memory types actually accessed on
/// it - including inside a staged tile's loader.
#[test]
fn mem_types_are_collected_per_argument_through_every_block() {
    let lk = nest(
        vec![],
        vec![Stmt::Parallel {
            var: VarId(0),
            axis: I,
            level: HwLevel::Grid(0),
            vector: 1,
            bounded: false,
            body: vec![
                load(VarId(1), MemType::F32),
                Stmt::If {
                    cond: VarId(0),
                    body: vec![load(VarId(2), MemType::F16)],
                },
            ],
        }],
    );
    let tys = lk.mem_types();
    assert_eq!(tys.len(), 1);
    assert_eq!(
        tys[ArgId(0).0 as usize]
            .iter()
            .copied()
            .collect::<Vec<MemType>>(),
        vec![MemType::F32, MemType::F16]
    );
}

/// A nest needs a subgroup when it prints a subgroup primitive, and shared
/// storage when the kernel declares any - and the two are independent.
///
/// The distinction is a *capability* published in the manifest: a shader that
/// declared `requires_subgroup` for a shared-memory tree would be refused on a
/// device that has no subgroup arithmetic and needs none.
#[test]
fn the_subgroup_and_shared_requirements_are_independent() {
    let lane_only = nest(
        vec![],
        vec![Stmt::LaneReduce {
            op: ReduceOp::Sum,
            src: VarId(1),
            dst: VarId(2),
        }],
    );
    assert!(lane_only.uses_subgroup());
    assert!(!lane_only.uses_shared());

    let shared_only = nest(
        vec![(VarId(1), 32)],
        vec![Stmt::WorkgroupReduce {
            reds: vec![GroupRed {
                op: ReduceOp::Sum,
                src: VarId(1),
                dst: VarId(2),
            }],
            lanes: 32,
            subgroup: None,
        }],
    );
    assert!(!shared_only.uses_subgroup());
    assert!(shared_only.uses_shared());

    // The hierarchical form uses **both**: the subgroup primitive for its first
    // stage and shared memory for the totals between the two.
    // The width it publishes is the subgroup's, not the
    // workgroup's, which is the whole reason `subgroup_width` exists.
    let hierarchical = nest(
        vec![(VarId(1), 8)],
        vec![Stmt::ParallelLane {
            var: VarId(0),
            lanes: 256,
            body: vec![Stmt::WorkgroupReduce {
                reds: vec![GroupRed {
                    op: ReduceOp::Sum,
                    src: VarId(1),
                    dst: VarId(2),
                }],
                lanes: 256,
                subgroup: Some(32),
            }],
        }],
    );
    assert_eq!(hierarchical.subgroup_width(), Some(32));
    assert!(hierarchical.uses_shared());

    let neither = nest(vec![], vec![load(VarId(1), MemType::F32)]);
    assert!(!neither.uses_subgroup());
    assert!(!neither.uses_shared());
}

/// The vector width of a nest is the one its **outermost parallel statement**
/// declares, and 1 when the nest is not a single parallel statement.
///
/// It is published in the registry as a **claim**: a node whose `nb[0]` is
/// not one element is refused for the variant. And the shape of the query is the
/// contract, not an accident - the width belongs to the invocation mapping, so a
/// nest that is not exactly one parallel statement has no width to publish, even
/// if some access inside it is wide.
#[test]
fn the_vector_width_is_the_one_the_parallel_statement_declares() {
    let scalar = nest(vec![], vec![load(VarId(1), MemType::F32)]);
    assert_eq!(scalar.vector_width(), 1);

    let wide = nest(
        vec![],
        vec![Stmt::Parallel {
            var: VarId(0),
            axis: I,
            level: HwLevel::Global(0),
            vector: 4,
            bounded: false,
            body: vec![Stmt::Load {
                dst: VarId(3),
                arg: Y,
                ty: MemType::F32,
                addr: vec![AddrTerm::Const(0)],
                width: 4,
            }],
        }],
    );
    assert_eq!(wide.vector_width(), 4);

    // A wide access under something else than a lone `Parallel` publishes no
    // width: the claim is about the mapping, and there is none to state.
    let buried = nest(
        vec![],
        vec![
            Stmt::Barrier,
            Stmt::Parallel {
                var: VarId(0),
                axis: I,
                level: HwLevel::Global(0),
                vector: 4,
                bounded: false,
                body: vec![],
            },
        ],
    );
    assert_eq!(buried.vector_width(), 1);
}

/// A constant table is declared once per nest that indexes it, however many
/// times it is indexed.
///
/// Both emitters print one declaration per entry of this list; a duplicate would
/// be a shader that declares the same table twice, which no shader
/// may do.
#[test]
fn a_constant_table_is_listed_once_however_often_it_is_read() {
    let lk = nest(
        vec![],
        vec![
            Stmt::Compute(Inst {
                dst: VarId(1),
                expr: LExpr::Lut {
                    table: LutId::Iq4Nl,
                    idx: VarId(0),
                },
            }),
            Stmt::Compute(Inst {
                dst: VarId(2),
                expr: LExpr::Lut {
                    table: LutId::Iq4Nl,
                    idx: VarId(0),
                },
            }),
        ],
    );
    assert_eq!(lk.luts(), vec![LutId::Iq4Nl]);
}

/// The magic-number division of a flattened dispatch
/// agrees with real division on **every** value it can be handed.
///
/// It is the one place in this project where the device and the oracle compute
/// the same quantity by two different formulas, and the divergence would not be
/// loud: a wrong multiplier decomposes a linear index into the wrong axes, which
/// reads and writes the wrong elements without addressing a single byte outside
/// the tensor. So the property is stated here, over the whole range the lowering
/// admits - `n < 2^31`, which is what makes the 32-bit `hi + n` exact - rather
/// than sampled on the shapes a parity harness happens to try.
#[test]
fn the_magic_numbers_divide_exactly() {
    // Divisors an extent can plausibly take, plus the awkward ones: one, a
    // power of two, a prime, and the largest a 31-bit index can be divided by.
    let divisors = [
        1u32,
        2,
        3,
        4,
        5,
        7,
        16,
        17,
        31,
        32,
        33,
        64,
        127,
        128,
        256,
        1000,
        1024,
        4096,
        65535,
        65536,
        0x7fff_ffff,
        0x8000_0000,
    ];
    for d in divisors {
        let (mp, sh) = crate::fastdiv_magic(d);
        // Around each multiple boundary, where an off-by-one multiplier shows,
        // plus the two ends of the admitted range.
        let mut probes: Vec<u32> = vec![0, 1, 0x7fff_ffff];
        for k in [1u64, 2, 3, 1000, 100_000] {
            for delta in [-1i64, 0, 1] {
                let n = k as i64 * d as i64 + delta;
                if (0..=0x7fff_ffff).contains(&n) {
                    probes.push(n as u32);
                }
            }
        }
        for n in probes {
            assert_eq!(
                crate::fastdiv(n, mp, sh),
                n / d,
                "fastdiv({n}, {d}) with mp={mp} sh={sh}"
            );
        }
    }
}
