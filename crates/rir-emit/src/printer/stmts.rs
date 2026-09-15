//! One `Stmt` to one backend's text, for the three GPU backends.
//!
//! The match has no wildcard, so a statement added to the Loop IR still fails
//! to compile rather than falling into a refusal that says the wrong thing,
//! the property the three copies had, kept in one place.
//!
//! Of the 25 variants, six were already byte-identical across the three
//! emitters and seventeen differed only by a word.
//! Those 23 are here. The remaining two, `Load` and `Store`, carry the
//! backend's memory model and go through [`MemoryModel`]; the reason is in
//! `printer/memory.rs` and it is not a byte count.
//!
//! Nothing in this file may ask which backend it is printing for. Every
//! difference below arrives through `D`, and where the shape itself differed,
//! the parenthesis around a CUDA builtin used as a factor, the splat GLSL's
//! vector comparison wants and CUDA's does not - the table grew a column
//! rather than the skeleton growing a branch.

use rir_lower::{Inst, Stmt, VarKind};

use crate::EmitError;
use crate::dialect::Dialect;
use crate::manifest::{FLAT_TOTAL, flat_divisor};

use super::Printer;
use super::memory::{LoadOp, MemoryModel, StoreOp};

impl<'k, D: Dialect + MemoryModel> Printer<'k, D> {
    pub(crate) fn stmts(&mut self, stmts: &[Stmt]) -> Result<(), EmitError> {
        let uint = D::UINT;
        for s in stmts {
            match s {
                Stmt::Parallel {
                    var,
                    axis,
                    level,
                    vector,
                    bounded,
                    body,
                } => {
                    let builtin = D::axis_builtin(*level).ok_or(EmitError::UnsupportedStmt {
                        backend: D::BACKEND,
                        stmt: "Parallel(Lane) - utiliser ParallelLane",
                    })?;
                    let name = self.var(*var).to_string();
                    // A vectorized axis gives each invocation `vector`
                    // consecutive indices; `name` is the first of them, and the
                    // grid the dispatcher asks for is that many times smaller.
                    if *vector > 1 {
                        self.line(&format!(
                            "const {uint} {name} = {} * {vector}u;",
                            builtin.as_factor
                        ));
                    } else {
                        self.line(&format!("const {uint} {name} = {};", builtin.plain));
                    }
                    // An unbounded axis is one whose workgroup holds invocations
                    // both inside and outside the tensor: returning here would
                    // strand the barriers below, so the bound waits for
                    // `Stmt::InBounds`.
                    if *bounded {
                        self.line(&format!("if ({name} >= {}) {{", self.extent(*axis)));
                        self.line("    return;");
                        self.line("}");
                    }
                    self.stmts(body)?;
                }
                // The flattened dispatch. One linear
                // index, one bound, then one magic-number division per axis but
                // the last - the multiplier and the shift come from the constant
                // buffer, so no invocation performs an integer division.
                Stmt::ParallelFlat {
                    linear,
                    level,
                    axes,
                    decompose,
                    vector,
                    body,
                } => {
                    let params = D::PARAMS;
                    let lin = self.var(*linear).to_string();
                    // The same builtin the grid nest reads at this level: an
                    // invocation index for the flattened dispatch, a workgroup
                    // index for the flattened rows. The
                    // bound below is then uniform across the workgroup in the
                    // second case, so the early return cannot strand a barrier.
                    let builtin = D::axis_builtin(*level).ok_or(EmitError::UnsupportedStmt {
                        backend: D::BACKEND,
                        stmt: "ParallelFlat(Lane) - a lane index is not a dispatch",
                    })?;
                    self.line(&format!("const {uint} {lin} = {};", builtin.plain));
                    self.line(&format!("if ({lin} >= {params}{FLAT_TOTAL}) {{"));
                    self.line("    return;");
                    self.line("}");
                    // Linear addressing stops here: the
                    // bound is the only thing the linear index is used for
                    // besides the address itself, and there is no axis index to
                    // recover. The divisors are then not even declared - the
                    // constant layout drops them - so printing a division would
                    // read a name that does not exist.
                    let mut src = lin.clone();
                    for (d, (var, _)) in axes.iter().enumerate().filter(|_| *decompose) {
                        let name = self.var(*var).to_string();
                        // The last axis is the quotient itself: dividing by its
                        // extent would be dividing by the bound already reached.
                        if d + 1 == axes.len() {
                            let scaled = Self::scaled(&src, d, *vector);
                            self.line(&format!("const {uint} {name} = {scaled};"));
                            break;
                        }
                        let [div, mp, sh] = flat_divisor(d);
                        self.line(&format!(
                            "const {uint} {name} = {};",
                            Self::scaled(
                                &format!(
                                    "rir_fastmod({src}, {params}{mp}, {params}{sh}, {params}{div})"
                                ),
                                d,
                                *vector
                            )
                        ));
                        let next = format!("{lin}_q{}", d + 1);
                        self.line(&format!(
                            "const {uint} {next} = rir_fastdiv({src}, {params}{mp}, {params}{sh});"
                        ));
                        src = next;
                    }
                    self.stmts(body)?;
                }
                Stmt::VecTail {
                    base,
                    axis,
                    width,
                    vec_body,
                    tail_var,
                    tail_body,
                } => {
                    let b = self.var(*base).to_string();
                    let n = self.extent(*axis);
                    let t = self.var(*tail_var).to_string();
                    self.line(&format!("if ({b} + {width}u <= {n}) {{"));
                    self.indent += 1;
                    self.stmts(vec_body)?;
                    self.indent -= 1;
                    self.line("} else {");
                    self.indent += 1;
                    self.line(&format!("for ({uint} {t} = {b}; {t} < {n}; ++{t}) {{"));
                    self.indent += 1;
                    self.stmts(tail_body)?;
                    self.indent -= 1;
                    self.line("}");
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::ParallelLane { var, body, .. } => {
                    // Hardware SPMD needs no loop: each invocation is a lane.
                    let name = self.var(*var).to_string();
                    self.line(&format!(
                        "const {uint} {name} = {};",
                        D::lane_builtin(self.k)
                    ));
                    self.stmts(body)?;
                }
                Stmt::For {
                    var,
                    axis,
                    reverse,
                    body,
                } => {
                    let name = self.var(*var).to_string();
                    if *reverse {
                        self.line(&format!(
                            "for ({uint} {name} = {}; {name}-- > 0u;) {{",
                            self.extent(*axis)
                        ));
                    } else {
                        self.line(&format!(
                            "for ({uint} {name} = 0u; {name} < {}; ++{name}) {{",
                            self.extent(*axis)
                        ));
                    }
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::ForStrided {
                    var,
                    axis,
                    start,
                    step,
                    body,
                } => {
                    let name = self.var(*var).to_string();
                    self.line(&format!(
                        "for ({uint} {name} = {}; {name} < {}; {name} += {}u) {{",
                        self.var(*start),
                        self.extent(*axis),
                        step
                    ));
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::ForConst { var, count, body } => {
                    let name = self.var(*var).to_string();
                    self.line(&format!(
                        "for ({uint} {name} = 0u; {name} < {count}u; ++{name}) {{"
                    ));
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::ForTiled {
                    var,
                    axis,
                    step,
                    body,
                } => {
                    let name = self.var(*var).to_string();
                    self.line(&format!(
                        "for ({uint} {name} = 0u; {name} < {}; {name} += {step}u) {{",
                        self.extent(*axis)
                    ));
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::ForChunk {
                    var,
                    axis,
                    lane,
                    lanes,
                    reverse,
                    body,
                } => {
                    // Chunk geometry printed from `rir_lower::chunk_range`: the
                    // oracle computes the same bounds, so a disagreement here
                    // would be a wrong result, not a slow one.
                    let name = self.var(*var).to_string();
                    let n = self.extent(*axis);
                    let lane = self.var(*lane).to_string();
                    let (c, b, e) = (
                        format!("chunk_{name}"),
                        format!("beg_{name}"),
                        format!("end_{name}"),
                    );
                    self.line(&format!(
                        "const {uint} {c} = ({n} + {lanes}u - 1u) / {lanes}u;"
                    ));
                    self.line(&format!("const {uint} {b} = min({lane} * {c}, {n});"));
                    self.line(&format!(
                        "const {uint} {e} = min(({lane} + 1u) * {c}, {n});"
                    ));
                    if *reverse {
                        // Mirrored assignment: lane `l` owns the `l`-th chunk
                        // from the end, traversed descending.
                        self.line(&format!(
                            "for ({uint} {name} = {n} - {b}; {name}-- > {n} - {e};) {{"
                        ));
                    } else {
                        self.line(&format!(
                            "for ({uint} {name} = {b}; {name} < {e}; ++{name}) {{"
                        ));
                    }
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::InitAcc { acc, op } => {
                    // A vector accumulator is register tiling: the invocation
                    // holds one partial sum per output it owns.
                    let ty = self.decl_ty(*acc);
                    let init = match self.kind(*acc) {
                        VarKind::Vec(w) => D::splat(VarKind::F32, w, D::identity(*op)),
                        _ => D::identity(*op).to_string(),
                    };
                    self.line(&format!("{ty} {} = {init};", self.var(*acc)));
                }
                Stmt::Accum { acc, op, value } => {
                    let l = match op {
                        rir_core::ReduceOp::Sum => {
                            format!("{} += {};", self.var(*acc), self.var(*value))
                        }
                        rir_core::ReduceOp::Max => format!(
                            "{} = {};",
                            self.var(*acc),
                            self.combine_vars(*op, *acc, *value)
                        ),
                    };
                    self.line(&l);
                }
                Stmt::If { cond, body } => {
                    self.line(&format!("if ({}) {{", self.var(*cond)));
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::Set { var, value } => {
                    let l = format!("{} = {};", self.var(*var), self.var(*value));
                    self.line(&l);
                }
                Stmt::InBounds { bounds, body } => {
                    let conds: Vec<String> = bounds
                        .iter()
                        .map(|(var, axis)| format!("{} < {}", self.var(*var), self.extent(*axis)))
                        .collect();
                    self.line(&format!("if ({}) {{", conds.join(" && ")));
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::LaneZero { lane, body } => {
                    self.line(&format!("if ({} == 0u) {{", self.var(*lane)));
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::StageTiles {
                    tiles,
                    threads,
                    body,
                } => {
                    // Two barriers a round, whatever the tile count: the first
                    // keeps this round's writers behind the previous round's
                    // readers, the second keeps this round's readers behind its
                    // writers. They are printed here, not lowered as
                    // `Stmt::Barrier`.
                    self.line(D::BARRIER);
                    let local = D::local_index(self.k);
                    for t in tiles {
                        let sh = self.shared(t.tile);
                        let l = format!("l_{}", self.var(t.tile));
                        let (row, dep) =
                            (self.var(t.row).to_string(), self.var(t.depth).to_string());
                        let (m, kg) = (
                            self.var(t.row_global).to_string(),
                            self.var(t.depth_global).to_string(),
                        );
                        let (m0, k0) = (
                            self.var(t.row_origin).to_string(),
                            self.var(t.depth_origin).to_string(),
                        );
                        let (n_row, n_dep) = (self.extent(t.row_axis), self.extent(t.depth_axis));
                        // A span > 1 gives each invocation `span` consecutive
                        // rows instead of one element, so the block header its
                        // loader reads is shared by all of them.
                        let (start, step) = if t.span > 1 {
                            (
                                format!("{local} * {}u", t.span),
                                format!("{}u", threads * t.span),
                            )
                        } else {
                            (local.to_string(), format!("{threads}u"))
                        };
                        self.line(&format!(
                            "for ({uint} {l} = {start}; {l} < {}u; {l} += {step}) {{",
                            t.len()
                        ));
                        self.indent += 1;
                        let slot = self.var(t.slot).to_string();
                        if slot != l {
                            self.line(&format!("const {uint} {slot} = {l};"));
                        }
                        self.line(&format!("const {uint} {row} = {l} % {}u;", t.n_rows));
                        self.line(&format!("const {uint} {dep} = {l} / {}u;", t.n_rows));
                        self.line(&format!("const {uint} {m} = {m0} + {row};"));
                        self.line(&format!("const {uint} {kg} = {k0} + {dep};"));
                        // The loader under the range test, the zeros in the
                        // **else** branch: the loader is a *body* - a plain
                        // read, or a whole fused decoder over a segment - so it
                        // cannot be the arm of a ternary, it writes the tile
                        // itself through `StoreTile`, and a segment outside the
                        // tensor must not run it at all (ADR-3 section 3).
                        //
                        // Zeroing first and letting the loader overwrite was one
                        // shared store too many on every element of every
                        // interior tile, which is nearly all of them.
                        // The branch is not new - the
                        // range test was already there - and it holds no
                        // barrier, so an invocation taking one arm while its
                        // neighbour takes the other is exactly as safe as
                        // before.
                        self.line(&format!("if ({m} < {n_row} && {kg} < {n_dep}) {{"));
                        self.indent += 1;
                        self.stmts(&t.load)?;
                        self.indent -= 1;
                        self.line("} else {");
                        self.indent += 1;
                        for c in 0..t.span {
                            let off = if c == 0 {
                                String::new()
                            } else {
                                format!(" + {c}u")
                            };
                            self.line(&format!("{sh}[{l}{off}] = {};", D::ZERO));
                        }
                        self.indent -= 1;
                        self.line("}");
                        self.indent -= 1;
                        self.line("}");
                    }
                    self.line(D::BARRIER);
                    self.stmts(body)?;
                }
                Stmt::StoreShared {
                    array,
                    index,
                    value,
                } => {
                    let sh = self.shared(*array);
                    self.line(&format!(
                        "{sh}[{}] = {};",
                        self.var(*index),
                        self.var(*value)
                    ));
                }
                Stmt::LoadShared {
                    dst,
                    array,
                    index,
                    width,
                } => {
                    let sh = self.shared(*array);
                    let at = self.var(*index).to_string();
                    let name = self.var(*dst).to_string();
                    if *width > 1 {
                        let comps: Vec<String> = (0..*width)
                            .map(|c| {
                                if c == 0 {
                                    format!("{sh}[{at}]")
                                } else {
                                    format!("{sh}[{at} + {c}u]")
                                }
                            })
                            .collect();
                        self.line(&format!(
                            "const {} {name} = {};",
                            D::decl_ty(VarKind::Vec(*width)),
                            D::vec_ctor(*width, &comps)
                        ));
                        continue;
                    }
                    self.line(&format!("const float {name} = {sh}[{at}];"));
                }
                // The two statements that are not a word: the backend's memory
                // model (`printer/memory.rs`).
                Stmt::Load {
                    dst,
                    arg,
                    ty,
                    addr,
                    width,
                } => D::load(
                    self,
                    &LoadOp {
                        dst: *dst,
                        arg: *arg,
                        ty: *ty,
                        addr,
                        width: *width,
                    },
                ),
                Stmt::Store {
                    arg,
                    ty,
                    addr,
                    value,
                    width,
                    bound,
                } => D::store(
                    self,
                    &StoreOp {
                        arg: *arg,
                        ty: *ty,
                        addr,
                        value: *value,
                        width: *width,
                        bound: *bound,
                    },
                ),
                Stmt::Compute(Inst { dst, expr }) => {
                    let l = format!(
                        "const {} {} = {};",
                        self.decl_ty(*dst),
                        self.var(*dst),
                        self.expr(expr)
                    );
                    self.line(&l);
                }
                Stmt::LaneReduce { op, src, dst } => {
                    self.line(&format!(
                        "const float {} = {}({});",
                        self.var(*dst),
                        D::lane_reduce_fn(*op),
                        self.var(*src)
                    ));
                }
                Stmt::LaneScan { op, src, dst } => {
                    // A backend without the primitive refuses here rather than
                    // printing a function that does not exist, which would fail
                    // at shader compile time, far from the cause.
                    let f = D::lane_scan_fn(*op).ok_or(EmitError::UnsupportedStmt {
                        backend: D::BACKEND,
                        stmt: match op {
                            rir_core::ReduceOp::Sum => "LaneScan(Sum)",
                            rir_core::ReduceOp::Max => "LaneScan(Max)",
                        },
                    })?;
                    let l = format!("const float {} = {f}({});", self.var(*dst), self.var(*src));
                    self.line(&l);
                }
                Stmt::WorkgroupReduce {
                    reds,
                    lanes,
                    subgroup: Some(width),
                } => {
                    // The two-stage reduction. Same group,
                    // same barriers-shared-by-every-accumulator rule as the tree
                    // below, and two stages instead of `log2(lanes)` levels:
                    //
                    //   sg  = lane_reduce(acc)            // one per subgroup
                    //   if (lane % w == 0) sh[lane/w] = sg
                    //   barrier
                    //   if (lane < w) { total = lane_reduce(lane < n ? sh[lane]
                    // identity)
                    //                   if (lane == 0) sh[0] = total }
                    //   barrier
                    //   dst = sh[0]
                    //
                    // Both collectives sit under a condition that is uniform
                    // *per subgroup* - every invocation of subgroup zero enters
                    // the second stage and no invocation of any other does,
                    // which is what makes the primitive defined here. That rests
                    // on a subgroup being a contiguous run of workgroup-local
                    // indices, the same assumption `Stmt::LaneReduce` already
                    // makes on a 32-lane block, and the manifest publishes the
                    // width so a device that disagrees refuses the kernel.
                    let lane = D::WG_LANE_INDEX;
                    let n_sub = lanes / width;
                    for r in reds {
                        let sh = self.shared(r.dst);
                        self.line(&format!(
                            "const float {}_sg = {}({});",
                            self.var(r.dst),
                            D::lane_reduce_fn(r.op),
                            self.var(r.src)
                        ));
                        self.line(&format!("if ({lane} % {width}u == 0u) {{"));
                        self.line(&format!(
                            "    {sh}[{lane} / {width}u] = {}_sg;",
                            self.var(r.dst)
                        ));
                        self.line("}");
                    }
                    self.line(D::BARRIER);
                    self.line(&format!("if ({lane} < {width}u) {{"));
                    self.indent += 1;
                    for r in reds {
                        let sh = self.shared(r.dst);
                        let name = self.var(r.dst).to_string();
                        // A workgroup narrower than one subgroup squared leaves
                        // lanes with no total to contribute; they carry the
                        // combiner's identity, which is what makes the second
                        // stage a reduction over `n_sub` values whatever the
                        // subgroup width.
                        self.line(&format!(
                            "const float {name}_p = {lane} < {n_sub}u ? {sh}[{lane}] : {};",
                            D::identity(r.op)
                        ));
                        self.line(&format!(
                            "const float {name}_t = {}({name}_p);",
                            D::lane_reduce_fn(r.op)
                        ));
                        self.line(&format!("if ({lane} == 0u) {{"));
                        self.line(&format!("    {sh}[0] = {name}_t;"));
                        self.line("}");
                    }
                    self.indent -= 1;
                    self.line("}");
                    self.line(D::BARRIER);
                    for r in reds {
                        let sh = self.shared(r.dst);
                        self.line(&format!("const float {} = {sh}[0];", self.var(r.dst)));
                    }
                }
                Stmt::WorkgroupReduce {
                    reds,
                    lanes,
                    subgroup: None,
                } => {
                    // One tree, every accumulator: each level combines all of
                    // them and *then* meets a single barrier, which is what
                    // makes two reductions of one level cost nine barriers
                    // instead of eighteen. It is a
                    // workgroup tree, so nothing here is a warp assumption.
                    let lane = D::WG_LANE_INDEX;
                    for r in reds {
                        let sh = self.shared(r.dst);
                        self.line(&format!("{sh}[{lane}] = {};", self.var(r.src)));
                    }
                    self.line(D::BARRIER);
                    let stride = format!("stride_{}", self.var(reds[0].dst));
                    self.line(&format!(
                        "for ({uint} {stride} = {}u; {stride} > 0u; {stride} >>= 1u) {{",
                        lanes / 2
                    ));
                    self.indent += 1;
                    self.line(&format!("if ({lane} < {stride}) {{"));
                    self.indent += 1;
                    for r in reds {
                        let sh = self.shared(r.dst);
                        let rhs = format!("{sh}[{lane} + {stride}]");
                        let combined = Self::combine(r.op, &format!("{sh}[{lane}]"), &rhs);
                        self.line(&format!("{sh}[{lane}] = {combined};"));
                    }
                    self.indent -= 1;
                    self.line("}");
                    self.line(D::BARRIER);
                    self.indent -= 1;
                    self.line("}");
                    for r in reds {
                        let sh = self.shared(r.dst);
                        self.line(&format!("const float {} = {sh}[0];", self.var(r.dst)));
                    }
                }
                Stmt::Barrier => self.line(D::BARRIER),
            }
        }
        Ok(())
    }
}
