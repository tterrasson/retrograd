//! The one cooperative loader a staged tile is filled with.
//!
//! `TiledStage` and `MatrixTile` both stage operands through shared memory,
//! and both built their own copy of the loader - the statements that read one
//! segment of the tile and store it through `StoreShared`. This module holds
//! the single copy, so a tile's load is always built by the same code a direct
//! read goes through: a quantized operand is staged by the format's own
//! decoder, and no emitter composes an address or a block layout of its own.

use rir_core::ArgId;

use crate::loop_ir::*;
use crate::lower::{LowerError, Lowerer};

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
