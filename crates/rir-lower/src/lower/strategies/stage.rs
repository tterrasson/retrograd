//! The one cooperative loader a staged tile is filled with.
//!
//! `TiledStage` and `MatrixTile` both stage operands through shared memory,
//! and both built their own copy of the loader - the statements that read one
//! segment of the tile and store it through `StoreShared`. This module holds
//! the single copy, so a tile's load is always built by the same code a direct
//! read goes through: a quantized operand is staged by the format's own
//! decoder, and no emitter composes an address or a block layout of its own.

use rir_core::{ArgId, AxisId, Op, ValueId};

use crate::loop_ir::*;
use crate::lower::{LowerError, Lowerer};

/// One dense read served from the staged tile, and the tile's shared array.
#[derive(Clone)]
pub(crate) struct StagedRead {
    /// The `Read` value the consumers lower.
    pub(crate) value: ValueId,
    /// The shared array its tile occupies, declared once in
    /// `LoopKernel::shared`.
    pub(crate) tile: VarId,
}

/// The shared operand a lane reduction stages through workgroup memory
/// (`Schedule::stage_b`), built once by `lower_lanes` and read by the segment
/// lowering of every dependency level.
#[derive(Clone)]
pub(crate) struct StageB {
    /// Invocations in the workgroup, over which the cooperative load spreads.
    pub(crate) threads: u32,
    /// Elements of the reduced axis one round covers: `lanes · span`.
    pub(crate) round: u32,
    /// Columns of the lane tile one round covers.
    pub(crate) cols: u32,
    /// The workgroup's first column: the grid's own index on the lane axis.
    pub(crate) cols_first: VarId,
    /// The lane-tiled axis.
    pub(crate) col_axis: AxisId,
    /// The dense reads served from the tile, each with its array.
    pub(crate) reads: Vec<StagedRead>,
}

/// The tiles one reduction round stages, for `round_var` - the loop variable
/// the round loop walks, which is the tile's origin on the reduced axis.
///
/// A tile is `cols × round`, laid out column-major: element `(c, d)` of
/// column `c` and reduced offset `d` sits at `c · round + d`. The consumers of
/// one column then read consecutive words, and the cooperative load - which
/// walks the reduced offset fastest - reads the operand's contiguous
/// dimension coalesced.
pub(crate) fn stage_b_tiles(
    st: &StageB,
    lo: &mut Lowerer,
    inner_axis: AxisId,
    round_var: VarId,
) -> Result<Vec<TileStage>, LowerError> {
    let k = lo.k;
    let mut tiles = Vec::with_capacity(st.reads.len());
    for read in &st.reads {
        let rv = read.value;
        let Op::Read { tensor: arg, idx } = &k.ops()[rv.0 as usize] else {
            unreachable!("a staged read is a Read value");
        };
        let name = k.args()[arg.0 as usize].name.clone();
        let tile = read.tile;
        let row = lo.new_var(&format!("{name}_r"), VarKind::Idx);
        let depth = lo.new_var(&format!("{name}_d"), VarKind::Idx);
        let row_global = lo.new_var(&format!("{name}_m"), VarKind::Idx);
        let depth_global = lo.new_var(&format!("{name}_k"), VarKind::Idx);
        let slot = lo.new_var(&format!("{name}_at"), VarKind::Idx);
        // The index registers the loader addresses with: the reduced axis and
        // the lane axis are the tile's two coordinates, anything else is an
        // outer loop variable already in scope.
        let mut idx_vars = Vec::with_capacity(idx.len());
        for &iv in idx.iter() {
            match k.ops()[iv.0 as usize] {
                Op::Index(ax) if ax == inner_axis => idx_vars.push(row_global),
                Op::Index(ax) if ax == st.col_axis => idx_vars.push(depth_global),
                Op::Index(ax) => idx_vars.push(
                    *lo.axis_vars
                        .get(&ax)
                        .ok_or(LowerError::AxisOutOfScope { value: rv, axis: ax })?,
                ),
                _ => {
                    return Err(LowerError::StageUnsupported {
                        why: "computed index: the staged tile is an axis rectangle",
                    })
                }
            }
        }
        let load = stage_loader(lo, &name, *arg, None, &idx_vars, row_global, slot, tile, 1)?;
        tiles.push(TileStage {
            tile,
            arg: *arg,
            row,
            depth,
            row_global,
            depth_global,
            row_origin: round_var,
            depth_origin: st.cols_first,
            row_axis: inner_axis,
            depth_axis: st.col_axis,
            slot,
            span: 1,
            load,
            row_stride: st.round,
            n_rows: st.round,
            n_depth: st.cols,
        });
    }
    Ok(tiles)
}

/// The statements staging one segment of a tile: `span` consecutive rows of
/// `arg` from `row_global`, read at `idx_vars` and stored from `slot` on.
///
/// The split the span exists for: a quantized operand's block index, address
/// base and F16 scales are emitted **once**, then the per-element work goes
/// under a `ForConst` the emitters print as a loop of constant trip count. A
/// dense operand is one element per invocation.
#[expect(clippy::too_many_arguments)]
pub(crate) fn stage_loader(
    lo: &mut Lowerer,
    name: &str,
    arg: ArgId,
    quant: Option<rir_core::QuantType>,
    idx_vars: &[VarId],
    row_global: VarId,
    slot: VarId,
    tile: VarId,
    span: u32,
) -> Result<Vec<Stmt>, LowerError> {
    let outer = std::mem::take(&mut lo.body);
    match quant {
        Some(format) if span > 1 => {
            let be = format.desc().block_elements;
            let block = lo.emit("blk", VarKind::Idx, LExpr::IDivC(row_global, be));
            let inb0 = lo.emit("inb0", VarKind::Idx, LExpr::IModC(row_global, be));
            let (base, header) =
                lo.lower_dequant_header(arg, idx_vars, block, Some((inb0, span)), format)?;
            let elem = lo.new_var(&format!("{name}_e"), VarKind::Idx);
            let inner = std::mem::take(&mut lo.body);
            let inb = lo.emit("inb", VarKind::Idx, LExpr::IAdd(inb0, elem));
            let value = lo.lower_dequant_element(arg, format, &base, &header, inb)?;
            let index = lo.emit("at", VarKind::Idx, LExpr::IAdd(slot, elem));
            lo.body.push(Stmt::StoreShared {
                array: tile,
                index,
                value,
            });
            let body = std::mem::replace(&mut lo.body, inner);
            lo.body.push(Stmt::ForConst {
                var: elem,
                count: span,
                body,
            });
        }
        Some(format) => {
            let value = lo.lower_dequant_read(arg, idx_vars, format)?;
            lo.body.push(Stmt::StoreShared {
                array: tile,
                index: slot,
                value,
            });
        }
        None => {
            let value = lo.new_var("t", VarKind::F32);
            lo.body.push(Stmt::Load {
                dst: value,
                arg,
                ty: Lowerer::elem_mem_type(lo.k, arg)?,
                addr: Lowerer::f32_addr(idx_vars),
                width: 1,
            });
            lo.body.push(Stmt::StoreShared {
                array: tile,
                index: slot,
                value,
            });
        }
    }
    Ok(std::mem::replace(&mut lo.body, outer))
}
