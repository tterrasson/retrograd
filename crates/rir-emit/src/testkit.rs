//! A hand-built `LoopKernel` for the emitters' unit tests.
//!
//! Why by hand and not through `lower()`: the property these tests buy is that
//! **a given `Stmt` prints as a given text**. Going through lowering would test
//! the pair, and a change in either half could keep the pair consistent while
//! the printed form moved - which is all that parity, shader compilation and
//! string search from `rir-gen` can see, and none of them says what a
//! statement looks like.
//!
//! The kernel is deliberately minimal and always the same: two F32 2-D
//! arguments, two axes, and whatever registers a test names. A test therefore
//! reads as its statement plus its expected text, with no fixture to decode.

use rir_core::IrId;
use rir_core::{Access, Arg, ArgId, AxisDecl, AxisId, Extent, ParamDecl, TensorType};
use rir_lower::{LoopKernel, Schedule, Stmt, VarId, VarKind};

/// The read argument (`x`) and the written one (`y`).
pub const X: ArgId = ArgId(0);
pub const Y: ArgId = ArgId(1);

/// The outer (`row`) and inner (`col`) axes.
pub const ROW: AxisId = AxisId(0);
pub const COL: AxisId = AxisId(1);

/// Builds the fixture kernel.
///
/// `vars` names the registers in `VarId` order, so a test writes `VarId(0)` for
/// the first name it declared. `shared` is the declaration list emitters and
/// interpreter both allocate from (`LoopKernel::shared`).
pub fn kernel(
    vars: &[(&str, VarKind)],
    shared: Vec<(VarId, u32)>,
    body: Vec<Stmt>,
    schedule: Schedule,
) -> LoopKernel {
    LoopKernel {
        name: "probe".into(),
        args: vec![
            Arg {
                name: "x".into(),
                ty: TensorType::f32_2d(),
                access: Access::Read,
            },
            Arg {
                name: "y".into(),
                ty: TensorType::f32_2d(),
                access: Access::Write,
            },
        ],
        params: vec![ParamDecl {
            name: "eps".into(),
            ty: rir_core::ScalarType::F32,
        }],
        axes: vec![
            AxisDecl {
                name: "row".into(),
                extent: Extent::Dim { arg: X, dim: 1 },
            },
            AxisDecl {
                name: "col".into(),
                extent: Extent::Dim { arg: X, dim: 0 },
            },
        ],
        arg_axes: vec![vec![Some(COL), Some(ROW)], vec![Some(COL), Some(ROW)]],
        folds: vec![],
        constraints: vec![],
        reduction_semantics: vec![],
        var_names: vars.iter().map(|(n, _)| (*n).to_string()).collect(),
        var_kinds: vars.iter().map(|(_, k)| *k).collect(),
        shared,
        body,
        schedule,
    }
}

/// The GLSL between `void main() {` and the closing brace, dedented by one
/// level, so an expectation is the statement's own text and nothing else.
pub fn glsl_body(src: &str) -> String {
    body_after(src, "void main() {")
}

/// The MSL between the entrypoint's opening brace and the closing one, dedented
/// by one level. The threadgroup declarations `emit_metal` prints there are part
/// of it on purpose: they are what `LoopKernel::shared` becomes.
pub fn msl_body(src: &str) -> String {
    body_after(src, "{")
}

/// The CUDA between the kernel's opening brace and **its** closing one.
///
/// It cannot reuse `body_after`, and the reason is that a generated `.cu` holds two functions, the kernel and
/// its launch stub, so the last `}` of the file is the stub's. The body ends at
/// the first closing brace at column zero after the kernel opens.
pub fn cu_body(src: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let start = lines
        .iter()
        .position(|l| *l == "{")
        .unwrap_or_else(|| panic!("no kernel opening brace in:\n{src}"))
        + 1;
    let end = start
        + lines[start..]
            .iter()
            .position(|l| *l == "}")
            .unwrap_or_else(|| panic!("no kernel closing brace in:\n{src}"));
    lines[start..end]
        .iter()
        .map(|l| l.strip_prefix("    ").unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
}

fn body_after(src: &str, open: &str) -> String {
    let lines: Vec<&str> = src.lines().collect();
    let start = lines
        .iter()
        .position(|l| *l == open)
        .unwrap_or_else(|| panic!("no `{open}` line in:\n{src}"))
        + 1;
    let end = lines.len()
        - 1
        - lines
            .iter()
            .rev()
            .position(|l| *l == "}")
            .unwrap_or_else(|| panic!("no closing brace in:\n{src}"));
    lines[start..end]
        .iter()
        .map(|l| l.strip_prefix("    ").unwrap_or(l))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Asserts the printed body, and prints both sides readably when it differs.
///
/// `expected` is written as a raw string starting with a newline, which is what
/// makes an expectation in these tests look like the shader it describes.
#[track_caller]
pub fn assert_body(actual: &str, expected: &str) {
    let expected = expected.strip_prefix('\n').unwrap_or(expected).trim_end();
    assert_eq!(
        actual.trim_end(),
        expected,
        "\n--- printed ---\n{actual}\n--- expected ---\n{expected}\n"
    );
}

/// One hand-built nest, printed by both emitters.
///
/// The same `Probe` feeds the Vulkan test and the Metal one: the *input* is
/// stated once and each emitter's expectation is its own text. Two printings of
/// one statement diverging is then a diff in one file, not a fixture to compare
/// by eye.
pub struct Probe {
    vars: Vec<(&'static str, VarKind)>,
    shared: Vec<(VarId, u32)>,
    body: Vec<Stmt>,
    schedule: Schedule,
}

impl Probe {
    fn kernel(&self) -> LoopKernel {
        kernel(
            &self.vars,
            self.shared.clone(),
            self.body.clone(),
            self.schedule.clone(),
        )
    }

    /// The GLSL body this nest prints as.
    pub fn glsl(&self) -> String {
        glsl_body(&crate::emit_vulkan(&self.kernel()).expect("vulkan emitter"))
    }

    /// The MSL body this nest prints as.
    pub fn msl(&self) -> String {
        msl_body(&crate::emit_metal(&self.kernel()).expect("metal emitter"))
    }

    /// The CUDA body this nest prints as.
    pub fn cu(&self) -> String {
        cu_body(&crate::emit_cuda(&self.kernel()).expect("cuda emitter"))
    }

    /// The whole generated translation unit, for the tests whose subject is the
    /// half of the file that is not the body - the includes, the prelude, and
    /// the launch stub.
    pub fn cu_unit(&self) -> String {
        crate::emit_cuda(&self.kernel()).expect("cuda emitter")
    }
}

use rir_core::{CmpOp, LutId, ParamId, ReduceOp};
use rir_lower::{AddrTerm, GpuBackend, GroupRed, HwLevel, Inst, LExpr, MemType, TileStage};

fn nb(var: VarId, dim: usize) -> AddrTerm {
    AddrTerm::VarNb { var, dim }
}

/// Loop forms: the grid mapping, a sequential axis, a constant count, and the
/// tiled step.
pub fn loops() -> Probe {
    Probe {
        vars: vec![
            ("r", VarKind::Idx),
            ("c", VarKind::Idx),
            ("k", VarKind::Idx),
            ("k0", VarKind::Idx),
            ("t", VarKind::F32),
        ],
        shared: vec![],
        body: vec![Stmt::Parallel {
            var: VarId(0),
            axis: ROW,
            level: HwLevel::Grid(0),
            vector: 1,
            bounded: true,
            body: vec![
                Stmt::For {
                    var: VarId(1),
                    axis: COL,
                    reverse: true,
                    body: vec![Stmt::Compute(Inst {
                        dst: VarId(4),
                        expr: LExpr::ConstF32(0.5),
                    })],
                },
                Stmt::ForConst {
                    var: VarId(2),
                    count: 4,
                    body: vec![],
                },
                Stmt::ForTiled {
                    var: VarId(3),
                    axis: COL,
                    step: 8,
                    body: vec![],
                },
            ],
        }],
        schedule: Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    }
}

/// The lane forms: a strided walk, an accumulator, and the two subgroup
/// primitives an emitter is allowed to print in one call.
pub fn lanes() -> Probe {
    Probe {
        vars: vec![
            ("lane", VarKind::Idx),
            ("c", VarKind::Idx),
            ("acc", VarKind::F32),
            ("red", VarKind::F32),
            ("off", VarKind::F32),
            ("v", VarKind::F32),
        ],
        shared: vec![],
        body: vec![Stmt::ParallelLane {
            var: VarId(0),
            lanes: 32,
            body: vec![
                Stmt::InitAcc {
                    acc: VarId(2),
                    op: ReduceOp::Sum,
                },
                Stmt::ForStrided {
                    var: VarId(1),
                    axis: COL,
                    start: VarId(0),
                    step: 32,
                    body: vec![Stmt::Accum {
                        acc: VarId(2),
                        op: ReduceOp::Sum,
                        value: VarId(5),
                    }],
                },
                Stmt::LaneReduce {
                    op: ReduceOp::Max,
                    src: VarId(2),
                    dst: VarId(3),
                },
                Stmt::LaneScan {
                    op: ReduceOp::Sum,
                    src: VarId(2),
                    dst: VarId(4),
                },
                Stmt::ForChunk {
                    var: VarId(1),
                    axis: COL,
                    lane: VarId(0),
                    lanes: 32,
                    reverse: false,
                    body: vec![],
                },
                Stmt::LaneZero {
                    lane: VarId(0),
                    body: vec![Stmt::Compute(Inst {
                        dst: VarId(5),
                        expr: LExpr::Copy(VarId(3)),
                    })],
                },
            ],
        }],
        schedule: Schedule::gpu_subgroup(GpuBackend::Vulkan),
    }
}

/// The two statements that carry a barrier of their own, and the three control
/// forms the lowered scan is written with.
pub fn collectives() -> Probe {
    Probe {
        vars: vec![
            ("lane", VarKind::Idx),
            ("acc", VarKind::F32),
            ("red", VarKind::F32),
            ("cond", VarKind::Bool),
            ("v", VarKind::F32),
        ],
        // The array `WorkgroupReduce` names after its destination register: the
        // statement expands into a tree over it, and the kernel declares the
        // storage.
        shared: vec![(VarId(2), 32)],
        body: vec![Stmt::ParallelLane {
            var: VarId(0),
            lanes: 32,
            body: vec![
                Stmt::WorkgroupReduce {
                    reds: vec![GroupRed {
                        op: ReduceOp::Sum,
                        src: VarId(1),
                        dst: VarId(2),
                    }],
                    lanes: 32,
                    subgroup: None,
                },
                Stmt::Barrier,
                Stmt::Compute(Inst {
                    dst: VarId(3),
                    expr: LExpr::ICmpC {
                        op: CmpOp::Lt,
                        var: VarId(0),
                        c: 16,
                    },
                }),
                Stmt::If {
                    cond: VarId(3),
                    body: vec![Stmt::Set {
                        var: VarId(4),
                        value: VarId(2),
                    }],
                },
            ],
        }],
        schedule: Schedule::gpu_shared_reduce(GpuBackend::Vulkan),
    }
}

/// Shared memory as the lowered scan uses it: a declared array, a store, a
/// barrier, a load.
pub fn shared_memory() -> Probe {
    Probe {
        vars: vec![
            ("lane", VarKind::Idx),
            ("sh", VarKind::F32),
            ("at", VarKind::Idx),
            ("v", VarKind::F32),
            ("got", VarKind::F32),
        ],
        shared: vec![(VarId(1), 64)],
        body: vec![Stmt::ParallelLane {
            var: VarId(0),
            lanes: 32,
            body: vec![
                Stmt::StoreShared {
                    array: VarId(1),
                    index: VarId(2),
                    value: VarId(3),
                },
                Stmt::Barrier,
                Stmt::LoadShared {
                    dst: VarId(4),
                    array: VarId(1),
                    index: VarId(2),
                    width: 1,
                },
            ],
        }],
        schedule: Schedule::gpu_tiled_scan(GpuBackend::Vulkan, 32, 2),
    }
}

/// Memory: an F32 load, an F16 load with its conversion, and a store under the
/// bounds check a partial workgroup needs.
pub fn memory() -> Probe {
    Probe {
        vars: vec![
            ("row_i", VarKind::Idx),
            ("val", VarKind::F32),
            ("half", VarKind::F32),
            ("col_i", VarKind::Idx),
        ],
        shared: vec![],
        body: vec![Stmt::Parallel {
            var: VarId(0),
            axis: ROW,
            level: HwLevel::Global(0),
            vector: 1,
            bounded: false,
            body: vec![
                Stmt::Load {
                    dst: VarId(1),
                    arg: X,
                    ty: MemType::F32,
                    addr: vec![nb(VarId(0), 1), nb(VarId(3), 0), AddrTerm::Const(4)],
                    width: 1,
                },
                Stmt::Load {
                    dst: VarId(2),
                    arg: X,
                    ty: MemType::F16,
                    addr: vec![AddrTerm::VarConst {
                        var: VarId(3),
                        c: 2,
                    }],
                    width: 1,
                },
                Stmt::InBounds {
                    bounds: vec![(VarId(0), ROW)],
                    body: vec![Stmt::Store {
                        arg: Y,
                        ty: MemType::F32,
                        addr: vec![nb(VarId(0), 1)],
                        value: VarId(1),
                        width: 1,
                        bound: None,
                    }],
                },
            ],
        }],
        schedule: Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    }
}

/// The vector lowering: a widened body, the componentwise comparison and select
/// the unary family is written with, the per-component bounded write, and the
/// scalar tail that guards a row which is not a multiple of the width.
pub fn vectors() -> Probe {
    Probe {
        vars: vec![
            ("col_i", VarKind::Idx),
            ("v4", VarKind::Vec(4)),
            ("p4", VarKind::VecBool(4)),
            ("w4", VarKind::Vec(4)),
            ("tail", VarKind::Idx),
            ("s", VarKind::F32),
        ],
        shared: vec![],
        body: vec![Stmt::Parallel {
            var: VarId(0),
            axis: COL,
            level: HwLevel::Global(0),
            vector: 4,
            bounded: false,
            body: vec![Stmt::VecTail {
                base: VarId(0),
                axis: COL,
                width: 4,
                vec_body: vec![
                    Stmt::Load {
                        dst: VarId(1),
                        arg: X,
                        ty: MemType::F32,
                        addr: vec![nb(VarId(0), 0)],
                        width: 4,
                    },
                    Stmt::Compute(Inst {
                        dst: VarId(2),
                        expr: LExpr::Cmp {
                            op: CmpOp::Ge,
                            lhs: VarId(1),
                            rhs: VarId(1),
                        },
                    }),
                    Stmt::Compute(Inst {
                        dst: VarId(3),
                        expr: LExpr::Select {
                            cond: VarId(2),
                            t: VarId(1),
                            f: VarId(1),
                        },
                    }),
                    // The per-component bound a tiled write carries instead of
                    // branching a body that contains barriers.
                    Stmt::Store {
                        arg: Y,
                        ty: MemType::F32,
                        addr: vec![nb(VarId(0), 0)],
                        value: VarId(3),
                        width: 4,
                        bound: Some((VarId(0), COL)),
                    },
                ],
                tail_var: VarId(4),
                tail_body: vec![Stmt::Store {
                    arg: Y,
                    ty: MemType::F32,
                    addr: vec![nb(VarId(4), 0)],
                    value: VarId(5),
                    width: 1,
                    bound: None,
                }],
            }],
        }],
        schedule: Schedule::gpu_grid_vec4(GpuBackend::Vulkan, [64, 1, 1]),
    }
}

/// Expressions: one `Compute` per family of `LExpr`, including the constant
/// table a quantized decoder indexes.
///
/// Each expression gets its **own** destination register, so the printed body is
/// a valid declaration sequence and not a list of redefinitions: what is being
/// pinned here is one line per expression, and a line that could not compile
/// would not be one.
pub fn arithmetic() -> Probe {
    let exprs: Vec<(VarKind, LExpr)> = vec![
        (VarKind::F32, LExpr::ConstF32(1.5)),
        (VarKind::F32, LExpr::Param(ParamId(0))),
        (VarKind::F32, LExpr::AxisExtent(COL)),
        (VarKind::F32, LExpr::Add(VarId(0), VarId(1))),
        (VarKind::F32, LExpr::Sub(VarId(0), VarId(1))),
        (VarKind::F32, LExpr::Mul(VarId(0), VarId(1))),
        (VarKind::F32, LExpr::Div(VarId(0), VarId(1))),
        (VarKind::F32, LExpr::Sqrt(VarId(0))),
        (VarKind::F32, LExpr::Exp(VarId(0))),
        (VarKind::F32, LExpr::Tanh(VarId(0))),
        (VarKind::F32, LExpr::Copy(VarId(0))),
        (
            VarKind::Bool,
            LExpr::Cmp {
                op: CmpOp::Ge,
                lhs: VarId(0),
                rhs: VarId(1),
            },
        ),
        (
            VarKind::F32,
            LExpr::Select {
                cond: VarId(4),
                t: VarId(0),
                f: VarId(1),
            },
        ),
        (VarKind::Idx, LExpr::ConstIdx(7)),
        (VarKind::Idx, LExpr::IAdd(VarId(2), VarId(3))),
        (VarKind::Idx, LExpr::IAddC(VarId(3), 3)),
        (VarKind::Idx, LExpr::ISub(VarId(2), VarId(3))),
        (VarKind::Idx, LExpr::ISubC(VarId(3), 1)),
        (VarKind::Idx, LExpr::IMulC(VarId(3), 8)),
        (VarKind::Idx, LExpr::IDivC(VarId(3), 8)),
        (VarKind::Idx, LExpr::IModC(VarId(3), 8)),
        (VarKind::Idx, LExpr::IAndC(VarId(3), 15)),
        (VarKind::Idx, LExpr::IShrC(VarId(3), 2)),
        (VarKind::Idx, LExpr::IShr(VarId(3), VarId(2))),
        (VarKind::Idx, LExpr::IOr(VarId(3), VarId(2))),
        (
            VarKind::Idx,
            LExpr::IModAxis {
                var: VarId(3),
                axis: COL,
            },
        ),
        (VarKind::Idx, LExpr::AxisExtentIdx(COL)),
        (
            VarKind::Bool,
            LExpr::ICmpC {
                op: CmpOp::Lt,
                var: VarId(3),
                c: 16,
            },
        ),
        (VarKind::F32, LExpr::IToF(VarId(2))),
        (
            VarKind::F32,
            LExpr::Lut {
                table: LutId::Iq4Nl,
                idx: VarId(2),
            },
        ),
        (
            VarKind::F32,
            LExpr::Combine {
                op: ReduceOp::Max,
                lhs: VarId(0),
                rhs: VarId(1),
            },
        ),
        (
            VarKind::Idx,
            LExpr::AddrSum {
                arg: X,
                terms: vec![nb(VarId(3), 0), AddrTerm::Const(8)],
            },
        ),
    ];
    // The four registers the expressions read, then one destination each.
    const NAMES: [&str; 37] = [
        "a", "b", "i", "j", "p", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e10",
        "e11", "e12", "e13", "e14", "e15", "e16", "e17", "e18", "e19", "e20", "e21", "e22", "e23",
        "e24", "e25", "e26", "e27", "e28", "e29", "e30", "e31",
    ];
    assert_eq!(NAMES.len(), 5 + exprs.len(), "one name per register");
    let mut vars: Vec<(&'static str, VarKind)> = vec![
        ("a", VarKind::F32),
        ("b", VarKind::F32),
        ("i", VarKind::Idx),
        ("j", VarKind::Idx),
        ("p", VarKind::Bool),
    ];
    let mut body = Vec::new();
    for (n, (kind, expr)) in exprs.into_iter().enumerate() {
        vars.push((NAMES[5 + n], kind));
        body.push(Stmt::Compute(Inst {
            dst: VarId::at(5 + n),
            expr,
        }));
    }
    Probe {
        vars,
        shared: vec![],
        body,
        schedule: Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    }
}

/// Cooperative staging: one tile of an argument loaded once by the whole
/// workgroup, consumed by every invocation of it.
pub fn staging() -> Probe {
    Probe {
        vars: vec![
            ("r", VarKind::Idx),
            ("tile", VarKind::Idx),
            ("row", VarKind::Idx),
            ("dep", VarKind::Idx),
            ("row_g", VarKind::Idx),
            ("dep_g", VarKind::Idx),
            ("row_o", VarKind::Idx),
            ("dep_o", VarKind::Idx),
            ("slot", VarKind::Idx),
            ("v", VarKind::F32),
        ],
        shared: vec![(VarId(1), 128)],
        body: vec![Stmt::Parallel {
            var: VarId(0),
            axis: ROW,
            level: HwLevel::Grid(0),
            vector: 1,
            bounded: false,
            body: vec![Stmt::StageTiles {
                tiles: vec![TileStage {
                    tile: VarId(1),
                    arg: X,
                    row: VarId(2),
                    depth: VarId(3),
                    row_global: VarId(4),
                    depth_global: VarId(5),
                    row_origin: VarId(6),
                    depth_origin: VarId(7),
                    row_axis: ROW,
                    depth_axis: COL,
                    slot: VarId(8),
                    span: 1,
                    load: vec![
                        Stmt::Load {
                            dst: VarId(9),
                            arg: X,
                            ty: MemType::F32,
                            addr: vec![nb(VarId(4), 1), nb(VarId(5), 0)],
                            width: 1,
                        },
                        Stmt::StoreShared {
                            array: VarId(1),
                            index: VarId(8),
                            value: VarId(9),
                        },
                    ],
                    n_rows: 16,
                    n_depth: 8,
                }],
                threads: 64,
                body: vec![],
            }],
        }],
        schedule: Schedule::gpu_grid_tiled(GpuBackend::Vulkan, [64, 1, 1], 8, 1),
    }
}

/// The flattened dispatch: one linear index, its bound, and the magic-number
/// decomposition back into one index per axis.
///
/// It is the one `Stmt` the registry does not reach on every backend - only the
/// CUDA schedules flatten in production, so `emit_metal` and `emit_vulkan`
/// printed their `ParallelFlat` arm for no test at all until
/// this went looking. A statement pinned on one
/// backend out of three is exactly the divergence the three `tests.rs` exist to
/// prevent.
///
/// `vector` is 4 rather than 1 so both halves of `scaled` are printed by one
/// nest: the contiguous axis counts vectors and is scaled back to elements, the
/// axis above it counts elements already.
pub fn flat() -> Probe {
    Probe {
        vars: vec![
            ("i", VarKind::Idx),
            ("c", VarKind::Idx),
            ("r", VarKind::Idx),
            ("t", VarKind::F32),
        ],
        shared: vec![],
        body: vec![Stmt::ParallelFlat {
            linear: VarId(0),
            level: HwLevel::Global(0),
            axes: vec![(VarId(1), COL), (VarId(2), ROW)],
            decompose: true,
            vector: 4,
            body: vec![Stmt::Compute(Inst {
                dst: VarId(3),
                expr: LExpr::ConstF32(0.5),
            })],
        }],
        schedule: Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    }
}

/// The **two-stage** workgroup reduction: a subgroup
/// collective, one total per subgroup in shared memory, and a second collective
/// over those totals.
///
/// Pinned on all three backends for the reason the flattened dispatch is: it is
/// the only place where a lane collective and a barrier appear in the same
/// statement, and each backend spells both. What the text has to show is the
/// count - two barriers, whatever the lane count - because the count *is* the
/// optimisation.
pub fn hierarchical_reduce() -> Probe {
    Probe {
        vars: vec![
            ("lane", VarKind::Idx),
            ("acc", VarKind::F32),
            ("red", VarKind::F32),
        ],
        // Eight slots for 256 lanes: one per subgroup, not one per lane.
        shared: vec![(VarId(2), 8)],
        body: vec![Stmt::ParallelLane {
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
        schedule: Schedule::gpu_shared_reduce(GpuBackend::Vulkan),
    }
}

/// The **linearly addressed** flattened dispatch: the same
/// linear index and the same bound, and then nothing.
///
/// A probe of its own rather than a flag on [`flat`] above, because what it pins
/// is an *absence*: no `rir_fastmod`, no `rir_fastdiv`, no axis index. That is
/// the whole content of the item on the emitter side, and the failure it guards
/// is the one that would leave the claim published and the shader unchanged.
///
/// The body reads and writes through the linear register directly - the address
/// form `Lowerer::dense_addr` builds - so the three backends also pin that a
/// single scaled term prints as one multiplication and not as a stride sum.
pub fn flat_linear() -> Probe {
    Probe {
        vars: vec![
            ("i", VarKind::Idx),
            ("c", VarKind::Idx),
            ("r", VarKind::Idx),
            ("t", VarKind::F32),
        ],
        shared: vec![],
        body: vec![Stmt::ParallelFlat {
            linear: VarId(0),
            level: HwLevel::Global(0),
            axes: vec![(VarId(1), COL), (VarId(2), ROW)],
            decompose: false,
            vector: 1,
            body: vec![
                Stmt::Load {
                    dst: VarId(3),
                    arg: X,
                    ty: MemType::F32,
                    addr: vec![AddrTerm::VarConst {
                        var: VarId(0),
                        c: 4,
                    }],
                    width: 1,
                },
                Stmt::Store {
                    arg: Y,
                    ty: MemType::F32,
                    addr: vec![AddrTerm::VarConst {
                        var: VarId(0),
                        c: 4,
                    }],
                    value: VarId(3),
                    width: 1,
                    bound: None,
                },
            ],
        }],
        schedule: Schedule::gpu_grid(GpuBackend::Vulkan, [64, 1, 1]),
    }
}

/// The canonical list of probes, as a closed set.
///
/// This is the half of the three backend `tests.rs` that **is** common.
/// The expectations are not and cannot be:
/// each backend's expected text *is* its lexicon, and merging them would
/// delete the coverage. What the three files did share was the list of nests
/// they pin - shared by convention, and by nothing that noticed when the
/// convention broke.
///
/// A list shared by convention lets a nest be pinned on one backend alone and
/// printed by no test on the other two. An enum makes that a compile error
/// instead: [`pin_probes!`] expands a `match` over this type, so a backend
/// that leaves a probe out fails to compile with a non-exhaustive match - the
/// same device the emitters' own wildcard-free `match` uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Probes {
    Loops,
    Lanes,
    Collectives,
    SharedMemory,
    Memory,
    Vectors,
    Arithmetic,
    Staging,
    Flat,
    FlatLinear,
    HierarchicalReduce,
}

impl Probes {
    pub fn build(self) -> Probe {
        match self {
            Probes::Loops => loops(),
            Probes::Lanes => lanes(),
            Probes::Collectives => collectives(),
            Probes::SharedMemory => shared_memory(),
            Probes::Memory => memory(),
            Probes::Vectors => vectors(),
            Probes::Arithmetic => arithmetic(),
            Probes::Staging => staging(),
            Probes::Flat => flat(),
            Probes::FlatLinear => flat_linear(),
            Probes::HierarchicalReduce => hierarchical_reduce(),
        }
    }
}

/// Pins every probe's printed body for one backend.
///
/// `$render` is the `Probe` method that prints for this backend - `glsl`,
/// `msl` or `cu`. Each entry is `Variant as test_name => expected`, so the test
/// names stay the ones a failure has always reported.
///
/// The exhaustiveness check is the point of the macro; the loop over entries is
/// just what makes it worth writing.
#[macro_export]
macro_rules! pin_probes {
    ($render:ident, { $($(#[$meta:meta])* $variant:ident as $name:ident => $expected:expr,)* }) => {
        $(
            $(#[$meta])*
            #[test]
            fn $name() {
                $crate::testkit::assert_body(
                    &$crate::testkit::Probes::$variant.build().$render(),
                    $expected,
                );
            }
        )*

        /// Never called: it exists so that a probe this backend forgot to pin
        /// is a non-exhaustive `match`, at compile time.
        #[expect(dead_code)]
        fn every_probe_is_pinned(p: $crate::testkit::Probes) {
            match p {
                $($crate::testkit::Probes::$variant => ()),*
            }
        }
    };
}
