//! `ReductionStrategy::MatrixTile`: a contraction staged through shared
//! memory, as `TiledStage` stages one, and multiplied by the subgroup matrix
//! product instead of term by term.

use rir_core::{ArgId, AxisId, Op, ValueId};

use crate::loop_ir::*;
use crate::schedule::{
    MATRIX_COLS, MATRIX_ROWS, MATRIX_SUBGROUPS, ParallelMapping, SUBGROUP_LANES, Schedule,
};

use crate::lower::strategies::stage::stage_loader;
use crate::lower::strategies::tiled::collect_contraction_reads;
use crate::lower::{Analysis, LowerError, Lowerer, lower_writes};

/// Fragments one subgroup owns along the rows of the tile: its half of
/// `MATRIX_ROWS`.
const SG_ROW_FRAGS: u32 = MATRIX_ROWS / 2 / FRAG_DIM;

/// Fragments one subgroup owns along the columns of the tile: its half of
/// `MATRIX_COLS`.
const SG_COL_FRAGS: u32 = MATRIX_COLS / 2 / FRAG_DIM;

/// One operand of the product, as the tile round stages it.
struct Operand {
    arg: ArgId,
    quant: Option<rir_core::QuantType>,
    /// The parallel axis its second dimension walks.
    axis: AxisId,
}

/// The matrix-tiled contraction.
///
/// ```text
/// parallel m0 (grid x, MATRIX_ROWS)   parallel n0 (grid y, MATRIX_COLS)
///   subgroup sg: lanes                         // MATRIX_SUBGROUPS × 32
///     c[i][j] = 0                              // the subgroup's quarter
///     for k0 = 0.. n_k step depth:             // one tile round
///       stage A[m][k] ← dequant(a[k0+k, m0+m])  cooperatively
///       stage B[n][k] ← b[k0+k, n0+n]           barrier
///       for kk in 0..depth/8:
///         a_i = A[sg_m + 8i.., 8kk..]           b_j = Bᵀ[8kk.., sg_n + 8j..]
///         c[i][j] += a_i × b_j
///     barrier; C[n][m] ← c (into A's storage); barrier
///     for l = local; l < ROWS·COLS; l += threads:
///       if m0+m < n_m && n0+n < n_n { dst[m0+m, n0+n] = C[l] }
/// ```
///
/// The staged tiles are `TiledStage`'s - the same statement, the same loader,
/// the same zero padding that makes a partial tile neutral to the sum - and
/// what the strategy changes is the consumer: eight multiply-accumulates of
/// 8 × 8 fragments per subgroup and per eight steps of depth, where the
/// staged contraction does one multiply-add per invocation and step. The
/// output leaves through shared memory so that its write is the kernel's own
/// expression, guarded on both axes, rather than a fragment store that would
/// have to assume the destination's layout.
pub(crate) fn lower_matrix(
    lo: &mut Lowerer,
    a: &Analysis,
    par_vars: &[VarId],
    schedule: &Schedule,
) -> Result<Vec<Stmt>, LowerError> {
    let unsupported = |why: &'static str| LowerError::MatrixTileUnsupported { why };
    if schedule.par_map != ParallelMapping::Workgroup {
        return Err(unsupported(
            "workgroup mapping required: the tile is one workgroup's",
        ));
    }
    if schedule.flatten || schedule.vector_width > 1 || schedule.linear_addr {
        return Err(unsupported(
            "a flattened or vectorized grid: the tile is two grid dimensions",
        ));
    }
    if schedule.lane_tile > 1 || schedule.row_tile > 1 || schedule.reduce_run > 1 {
        return Err(unsupported(
            "lane tile, row tile or run: the matrix tile is its own geometry",
        ));
    }
    if schedule.subgroups != MATRIX_SUBGROUPS
        || schedule.block != [MATRIX_SUBGROUPS * SUBGROUP_LANES, 1, 1]
    {
        return Err(unsupported(
            "the workgroup is MATRIX_SUBGROUPS subgroups, two by two over the tile",
        ));
    }
    if schedule.segment > 0 || schedule.factor_scales {
        return Err(unsupported(
            "segment and scale factoring apply to scalar reductions, not matrix staging",
        ));
    }
    let depth = schedule.tile_depth;
    let stride = depth
        .checked_add(schedule.matrix_padding)
        .filter(|s| s.checked_mul((MATRIX_ROWS + MATRIX_COLS) * 4).is_some())
        .ok_or(unsupported(
            "shared operand stride overflows the storage counter",
        ))?;

    if !depth.is_multiple_of(FRAG_DIM) {
        return Err(unsupported("the depth is not a whole number of fragments"));
    }
    if depth < MATRIX_COLS {
        return Err(unsupported(
            "a depth below MATRIX_COLS: the output tile leaves through the row operand's storage",
        ));
    }
    let Some(inner_axis) = a.inner_axis else {
        return Err(unsupported("reduction axis required - nothing to contract"));
    };
    if !a.scans.is_empty() || !a.inner_writes.is_empty() {
        return Err(unsupported(
            "only a contraction closed within the workgroup is multiplied",
        ));
    }
    if a.reduces.len() != 1 {
        return Err(unsupported("only one reduction"));
    }
    let (rv, rop, rin) = a.reduces[0];
    if rop != rir_core::ReduceOp::Sum {
        return Err(unsupported(
            "sum required: a partial tile is completed with the additive identity",
        ));
    }
    if a.par_axes.len() != 2 || schedule.grid_dims < 2 {
        return Err(unsupported(
            "two parallel axes required: the rows and the columns of the product",
        ));
    }

    // ---- the two operands: each indexed by the reduced axis on its
    // contiguous dimension and by one parallel axis on the other.
    let mut reads = Vec::new();
    collect_contraction_reads(lo.k, rin, &mut reads)?;
    if reads.len() != 2 {
        return Err(unsupported("a product of exactly two operands"));
    }
    let k = lo.k;
    let operand = |axis: AxisId| -> Result<Operand, LowerError> {
        let indexed = |idx: &[ValueId]| {
            idx.len() == 2
                && matches!(k.ops()[idx[0].0 as usize], Op::Index(ax) if ax == inner_axis)
                && matches!(k.ops()[idx[1].0 as usize], Op::Index(ax) if ax == axis)
        };
        let (_, arg, _, quant) =
            reads
                .iter()
                .find(|(_, _, idx, _)| indexed(idx))
                .ok_or(unsupported(
                    "each operand must be indexed [reduced axis, one parallel axis]",
                ))?;
        Ok(Operand {
            arg: *arg,
            quant: *quant,
            axis,
        })
    };
    let lhs = operand(a.par_axes[0])?;
    let rhs = operand(a.par_axes[1])?;
    if rhs.quant.is_some() {
        return Err(unsupported(
            "the column operand is staged as it is read: only the row operand is decoded",
        ));
    }

    let threads = MATRIX_SUBGROUPS * SUBGROUP_LANES;
    let (m0, n0) = (par_vars[0], par_vars[1]);
    let sg = lo.new_var("sg", VarKind::Idx);
    let lane = lo.new_var("lane", VarKind::Idx);

    // ---- per lane: its index in the workgroup, and the subgroup's quarter
    // of the tile - rows `sg_m..sg_m + 32`, columns `sg_n..sg_n + 16`.
    lo.begin_scope(true);
    let sg_base = lo.emit("sg_base", VarKind::Idx, LExpr::IMulC(sg, SUBGROUP_LANES));
    let local = lo.emit("local", VarKind::Idx, LExpr::IAdd(sg_base, lane));
    let sg_row = lo.emit("sg_row", VarKind::Idx, LExpr::IAndC(sg, 1));
    let sg_m = lo.emit(
        "sg_m",
        VarKind::Idx,
        LExpr::IMulC(sg_row, SG_ROW_FRAGS * FRAG_DIM),
    );
    let sg_col = lo.emit("sg_col", VarKind::Idx, LExpr::IShrC(sg, 1));
    let sg_n = lo.emit(
        "sg_n",
        VarKind::Idx,
        LExpr::IMulC(sg_col, SG_COL_FRAGS * FRAG_DIM),
    );
    let mut acc = Vec::new();
    for i in 0..SG_ROW_FRAGS {
        let mut row = Vec::new();
        for j in 0..SG_COL_FRAGS {
            let c = lo.new_var(&format!("c{i}{j}"), VarKind::Frag);
            lo.body.push(Stmt::FragZero { dst: c });
            row.push(c);
        }
        acc.push(row);
    }
    let mut lane_body = lo.take_body();

    // ---- the tile round: both operands staged, then the fragments.
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    let k0 = lo.new_var(&format!("{inner_name}0"), VarKind::Idx);
    let tile_lhs = stage(lo, &lhs, k0, m0, inner_axis, depth, stride, MATRIX_ROWS)?;
    let tile_rhs = stage(lo, &rhs, k0, n0, inner_axis, depth, stride, MATRIX_COLS)?;
    let (a_tile, b_tile) = (tile_lhs.tile, tile_rhs.tile);

    let kk = lo.new_var(&format!("{inner_name}k"), VarKind::Idx);
    lo.begin_scope(true);
    let a_base = lo.emit("a_base", VarKind::Idx, LExpr::IMulC(sg_m, stride));
    let b_base = lo.emit("b_base", VarKind::Idx, LExpr::IMulC(sg_n, stride));
    let k_off = lo.emit("k_off", VarKind::Idx, LExpr::IMulC(kk, FRAG_DIM));
    let mut frag_a = Vec::new();
    for i in 0..SG_ROW_FRAGS {
        let row = lo.emit(
            "a_row",
            VarKind::Idx,
            LExpr::IAddC(a_base, i * FRAG_DIM * stride),
        );
        let index = lo.emit("a_at", VarKind::Idx, LExpr::IAdd(row, k_off));
        let dst = lo.new_var(&format!("a{i}"), VarKind::Frag);
        // `A[m][k]` row-major: fragment element `(i, j)` is row `m + i`,
        // depth `k + j`.
        lo.body.push(Stmt::FragLoad {
            dst,
            array: a_tile,
            index,
            stride,
            transpose: false,
        });
        frag_a.push(dst);
    }
    let mut frag_b = Vec::new();
    for j in 0..SG_COL_FRAGS {
        let col = lo.emit(
            "b_col",
            VarKind::Idx,
            LExpr::IAddC(b_base, j * FRAG_DIM * stride),
        );
        let index = lo.emit("b_at", VarKind::Idx, LExpr::IAdd(col, k_off));
        let dst = lo.new_var(&format!("b{j}"), VarKind::Frag);
        // `B[n][k]` read transposed: fragment element `(i, j)` is depth
        // `k + i`, column `n + j`.
        lo.body.push(Stmt::FragLoad {
            dst,
            array: b_tile,
            index,
            stride,
            transpose: true,
        });
        frag_b.push(dst);
    }
    for (i, &fa) in frag_a.iter().enumerate() {
        for (j, &fb) in frag_b.iter().enumerate() {
            lo.body.push(Stmt::FragMma {
                acc: acc[i][j],
                a: fa,
                b: fb,
            });
        }
    }
    let consume = lo.take_body();
    lane_body.push(Stmt::ForTiled {
        var: k0,
        axis: inner_axis,
        step: depth,
        body: vec![Stmt::StageTiles {
            tiles: vec![tile_lhs, tile_rhs],
            threads,
            body: vec![Stmt::ForConst {
                var: kk,
                count: depth / FRAG_DIM,
                body: consume,
            }],
        }],
    });

    // ---- the output, through the row operand's storage: every subgroup has
    // finished reading the last round once the first barrier is passed, and
    // every fragment is stored once the second is.
    lo.begin_scope(true);
    lane_body.push(Stmt::Barrier);
    let c_base = lo.emit("c_base", VarKind::Idx, LExpr::IMulC(sg_n, MATRIX_ROWS));
    let c_first = lo.emit("c_first", VarKind::Idx, LExpr::IAdd(c_base, sg_m));
    for (i, row) in acc.iter().enumerate() {
        for (j, &c) in row.iter().enumerate() {
            let off = (j as u32) * FRAG_DIM * MATRIX_ROWS + (i as u32) * FRAG_DIM;
            let index = lo.emit("c_at", VarKind::Idx, LExpr::IAddC(c_first, off));
            // `C[n][m]`, so that consecutive invocations write consecutive
            // rows of the destination below.
            lo.body.push(Stmt::FragStore {
                src: c,
                array: a_tile,
                index,
                stride: MATRIX_ROWS,
                transpose: true,
            });
        }
    }
    lane_body.extend(lo.take_body());
    lane_body.push(Stmt::Barrier);

    let t = lo.new_var("out_t", VarKind::Idx);
    lo.begin_scope(true);
    let l_base = lo.emit("l_base", VarKind::Idx, LExpr::IMulC(t, threads));
    let l = lo.emit("l", VarKind::Idx, LExpr::IAdd(l_base, local));
    let m = lo.emit("m", VarKind::Idx, LExpr::IModC(l, MATRIX_ROWS));
    let n = lo.emit("n", VarKind::Idx, LExpr::IDivC(l, MATRIX_ROWS));
    let row = lo.emit("row", VarKind::Idx, LExpr::IAdd(m0, m));
    let col = lo.emit("col", VarKind::Idx, LExpr::IAdd(n0, n));
    let value = lo.new_var("c", VarKind::F32);
    lo.body.push(Stmt::LoadShared {
        dst: value,
        array: a_tile,
        index: l,
        width: 1,
    });
    let mut out_body = lo.take_body();
    lo.axis_vars.remove(&inner_axis);
    lo.axis_vars.insert(a.par_axes[0], row);
    lo.axis_vars.insert(a.par_axes[1], col);
    lo.results_env.insert(rv, value);
    let writes = lower_writes(lo, &a.row_writes)?;
    out_body.push(Stmt::InBounds {
        bounds: vec![(row, a.par_axes[0]), (col, a.par_axes[1])],
        body: writes,
    });
    lane_body.push(Stmt::ForConst {
        var: t,
        count: MATRIX_ROWS * MATRIX_COLS / threads,
        body: out_body,
    });

    Ok(vec![Stmt::ParallelSubgroup {
        var: sg,
        count: MATRIX_SUBGROUPS,
        body: vec![Stmt::ParallelLane {
            var: lane,
            lanes: SUBGROUP_LANES,
            body: lane_body,
        }],
    }])
}

/// One operand's tile: `rows` indices of its parallel axis from `origin`, by
/// `depth` indices of the reduced axis from `k0`, laid out `[row][depth]` -
/// the tile's *row* coordinate is the reduced axis, so the quantized operand
/// is staged in segments along its own blocks.
#[expect(clippy::too_many_arguments)]
fn stage(
    lo: &mut Lowerer,
    op: &Operand,
    k0: VarId,
    origin: VarId,
    inner_axis: AxisId,
    depth: u32,
    stride: u32,
    rows: u32,
) -> Result<TileStage, LowerError> {
    let name = lo.k.args()[op.arg.0 as usize].name.clone();
    let tile = lo.new_var(&format!("tile_{name}"), VarKind::Idx);
    lo.shared.push((tile, rows * stride));
    let k_local = lo.new_var(&format!("{name}_r"), VarKind::Idx);
    let r_local = lo.new_var(&format!("{name}_d"), VarKind::Idx);
    let k_global = lo.new_var(&format!("{name}_k"), VarKind::Idx);
    let r_global = lo.new_var(&format!("{name}_m"), VarKind::Idx);
    let slot = lo.new_var(&format!("{name}_at"), VarKind::Idx);
    // The widest segment the three boundaries allow: a block, a sub-block when
    // the format has one, and the depth of a round - so a segment shares one
    // header and never straddles two rows of the tile.
    let span = match op.quant {
        Some(f) => {
            let desc = f.desc();
            let sub = match desc.layout.map(|l| l.scales) {
                Some(rir_core::ScalePlan::SubBlock { sub_elements, .. }) => Some(sub_elements),
                _ => None,
            };
            [16u32, 8, 4, 2, 1]
                .into_iter()
                .find(|s| {
                    desc.block_elements.is_multiple_of(*s)
                        && depth.is_multiple_of(*s)
                        && sub.is_none_or(|e| e.is_multiple_of(*s))
                })
                .unwrap_or(1)
        }
        None => 1,
    };
    let load = stage_loader(
        lo,
        &name,
        op.arg,
        op.quant,
        &[k_global, r_global],
        k_global,
        slot,
        tile,
        span,
    )?;
    Ok(TileStage {
        tile,
        arg: op.arg,
        row: k_local,
        depth: r_local,
        row_global: k_global,
        depth_global: r_global,
        row_origin: k0,
        depth_origin: origin,
        row_axis: inner_axis,
        depth_axis: op.axis,
        slot,
        span,
        load,
        row_stride: stride,
        n_rows: depth,
        n_depth: rows,
    })
}
