//! `ReductionStrategy::TiledStage`: a contraction staged through shared
//! memory, one tile of `rows × depth` per cooperative round.

use rir_core::{ArgId, AxisId, Kernel, Op, ValueId};

use crate::loop_ir::*;
use crate::schedule::{ParallelMapping, Schedule};

use crate::lower::{Analysis, LowerError, Lowerer, VecShape, lower_writes, widen_tiled};

/// The reads a tiled contraction stages, with the shape check that says a tile
/// can describe them.
///
/// Only a **product of reads** is accepted, and the reason is the zero padding:
/// a tile element outside the tensor is staged as zero, which is neutral for
/// the sum precisely because it makes its product vanish. Under any other
/// combiner it would not be - `exp(0)` is one, not zero - so an input shape
/// this cannot prove is an error rather than a term silently added to every
/// output of a partial tile.
pub(crate) fn collect_contraction_reads(
    k: &Kernel,
    v: ValueId,
    out: &mut Vec<(ValueId, ArgId, Vec<ValueId>, Option<rir_core::QuantType>)>,
) -> Result<(), LowerError> {
    match &k.ops()[v.0 as usize] {
        Op::Mul(x, y) => {
            collect_contraction_reads(k, *x, out)?;
            collect_contraction_reads(k, *y, out)
        }
        Op::Read { tensor, idx } => {
            out.push((v, *tensor, idx.clone(), None));
            Ok(())
        }
        // A quantized operand reaches the contraction through the explicit
        // `Dequant` the builder inserts, so what a tile stages is the
        // *dequantized* value - the same node the consumer below reads from
        // shared memory. Staging the raw block instead would put a format's
        // layout in the tile, and therefore in three emitters.
        Op::Dequant { value: rv, from } => match &k.ops()[rv.0 as usize] {
            Op::Read { tensor, idx } => {
                out.push((v, *tensor, idx.clone(), Some(*from)));
                Ok(())
            }
            _ => unreachable!("validated Dequant: applies to a Read"),
        },
        _ => Err(LowerError::TilingUnsupportedRead {
            value: v,
            why: "contraction input must be a product of reads: zero-filling a partial \
                  tile is neutral only there",
        }),
    }
}

/// Tiled contraction (ADR-2 section 6).
///
/// ```text
/// parallel i (grid x, unbounded)   parallel j (grid y, unbounded)   …
///   init acc
///   for k0 = 0.. n_k step depth:            // one tile round
///     stage tile_a[depth][BM] ← a[i0.., k0..]  cooperatively
///     stage tile_b[depth][BN] ← b[j0.., k0..]  barrier
///     for kk in 0..depth:                     // constant trip count
///       acc += tile_a[kk·BM + (i % BM)] · tile_b[kk·BN + (j % BN)]
///   if i < n_i && j < n_j { dst[i,j] = acc }
/// ```
///
/// Two properties are worth stating because they are what make it admissible
/// rather than merely faster. The **order is unchanged**: `k` ascends within a
/// round and the rounds ascend, so every output sums the same terms in the same
/// sequence as the untiled lowering - the tiling is a memory decision, not an
/// arithmetic one. And the **bound moves**: an invocation past the output
/// extent may no longer return, because its share of the cooperative loads and
/// its arrivals at the barriers are what the rest of the workgroup waits on.
/// One operand of a staged contraction, and the geometry the tile round reads
/// off it.
///
/// A named record keeps the two `VarId` fields and two integer fields distinct:
/// `rows` versus `grid`, and `origin` versus `local`.
struct StagedOperand {
    /// The `Read` value being staged, for the error that names it.
    value: ValueId,
    /// The argument it reads.
    arg: ArgId,
    /// Its quantized format, when the decoder stages a block instead of a float.
    quant: Option<rir_core::QuantType>,
    /// The axes indexing it, contiguous dimension first.
    axes: Vec<AxisId>,
    /// Which grid dimension walks its contiguous dimension.
    grid: usize,
    /// Rows of the tile: what one workgroup covers on that dimension.
    rows: u32,
    /// Register holding the tile's first row in global coordinates.
    origin: VarId,
    /// Register holding this invocation's row inside the tile.
    local: VarId,
}

pub(crate) fn lower_tiled(
    lo: &mut Lowerer,
    a: &Analysis,
    par_vars: &[VarId],
    schedule: &Schedule,
) -> Result<Vec<Stmt>, LowerError> {
    let unsupported = |why: &'static str| LowerError::TilingUnsupported { why };
    let Staging {
        vector,
        depth,
        inner_axis,
        rv,
        rop,
        rin,
    } = staging_of(a, schedule)?;

    let n_grid = a.par_axes.len().min(schedule.grid_dims);
    let grid_of =
        |axis: AxisId| -> Option<usize> { a.par_axes.iter().take(n_grid).position(|&x| x == axis) };
    let threads: u32 = schedule.block.iter().product();

    let mut reads = Vec::new();
    collect_contraction_reads(lo.k, rin, &mut reads)?;

    // ---- per-invocation prologue: each staged operand's tile origin, the
    // invocation's offset inside it, and the accumulator.
    lo.begin_scope(true);
    let mut staged: Vec<StagedOperand> = Vec::new();
    for (rvalue, arg, idx, quant) in &reads {
        // Every index must be a plain axis: staging addresses a rectangle, and
        // an arithmetic index would not describe one.
        let axes: Vec<AxisId> = idx
            .iter()
            .map(|&i| match lo.k.ops()[i.0 as usize] {
                Op::Index(ax) => Ok(ax),
                _ => Err(LowerError::TilingUnsupportedRead {
                    value: *rvalue,
                    why: "computed index: a tile is an axis rectangle",
                }),
            })
            .collect::<Result<_, _>>()?;
        // Rows walk the argument's **contiguous** dimension, which is what makes
        // the cooperative load read consecutive addresses; the reduction axis is
        // the depth, wherever it sits.
        let Some(grid) = axes.first().copied().and_then(grid_of) else {
            return Err(LowerError::TilingUnsupportedRead {
                value: *rvalue,
                why: "the contiguous dimension must be indexed by a grid axis",
            });
        };
        if !axes.iter().skip(1).any(|&ax| ax == inner_axis) {
            return Err(LowerError::TilingUnsupportedRead {
                value: *rvalue,
                why: "the reduction axis must index a non-contiguous dimension",
            });
        }
        // Grid x is the only vectorized dimension, so its workgroup covers
        // `block · w` indices; the others cover `block`.
        let rows = schedule.block[grid] * if grid == 0 { vector } else { 1 };
        let row_var = par_vars[grid];
        let block = lo.emit("blk", VarKind::Idx, LExpr::IDivC(row_var, rows));
        let origin = lo.emit("org", VarKind::Idx, LExpr::IMulC(block, rows));
        let local = lo.emit("loc", VarKind::Idx, LExpr::IModC(row_var, rows));
        staged.push(StagedOperand {
            value: *rvalue,
            arg: *arg,
            quant: *quant,
            axes,
            grid,
            rows,
            origin,
            local,
        });
    }

    let acc = lo.new_var(&format!("acc{}", rv.0), VarKind::F32);
    lo.body.push(Stmt::InitAcc { acc, op: rop });
    let mut out = lo.take_body();

    // ---- the tile round: what the workgroup stages, then what each invocation
    // accumulates out of it.
    let inner_name = lo.k.axes()[inner_axis.0 as usize].name.clone();
    let k0 = lo.new_var(&format!("{inner_name}0"), VarKind::Idx);

    let (tiles, tile_index) = stage_tiles(lo, &staged, k0, inner_axis, depth)?;
    let (kk, mut consume) = consume_tiles(lo, &tile_index, &inner_name, vector, rin, acc, rop)?;

    // ---- the epilogue, under the bound the early return no longer carries.
    lo.results_env.insert(rv, acc);
    lo.axis_vars.remove(&inner_axis);
    let mut stores = lower_writes(lo, &a.row_writes)?;

    // Register widths, propagated once over the two halves in the order they
    // execute: what the wide tile reads seed, the arithmetic carries, the
    // accumulator inherits, and the write finally has to honour.
    let vec_axis = (vector > 1).then(|| (par_vars[0], a.par_axes[0]));
    if let Some((vec_var, vec_ax)) = vec_axis {
        let mut is_vec: Vec<Option<VecShape>> = vec![None; lo.var_names.len()];
        widen_tiled(lo, &mut consume, &mut is_vec, vector)?;
        widen_tiled(lo, &mut stores, &mut is_vec, vector)?;
        for s in stores.iter_mut() {
            let Stmt::Store {
                addr,
                value,
                width,
                bound,
                ..
            } = s
            else {
                continue;
            };
            if is_vec[value.0 as usize].is_none() {
                return Err(unsupported(
                    "scalar write while the invocation covers multiple indices",
                ));
            }
            // Same address rule as the elementwise widening: `w` consecutive
            // indices are `w` consecutive addresses only at the dimension whose
            // stride the contract pins to one element.
            let indexed =
                matches!(addr.first(), Some(AddrTerm::VarNb { var, dim: 0 }) if *var == vec_var);
            if !indexed {
                return Err(unsupported(
                    "the write is not indexed by the vectorized axis on its contiguous dimension",
                ));
            }
            *width = vector;
            *bound = Some((vec_var, vec_ax));
        }
    }

    out.push(Stmt::ForTiled {
        var: k0,
        axis: inner_axis,
        step: depth,
        body: vec![Stmt::StageTiles {
            tiles,
            threads,
            body: vec![Stmt::ForConst {
                var: kk,
                count: depth,
                body: consume,
            }],
        }],
    });

    // The vectorized axis is **not** listed: its bound is per component, and
    // the store carries it. Listing it here would drop the whole vector as soon
    // as its first index was in range and its last was not.
    let bounds: Vec<(VarId, AxisId)> = (0..n_grid)
        .filter(|&d| schedule.block[d] > 1 && !(d == 0 && vector > 1))
        .map(|d| (par_vars[d], a.par_axes[d]))
        .collect();
    out.push(Stmt::InBounds {
        bounds,
        body: stores,
    });
    Ok(out)
}

/// What the tiled strategy needs out of the analysis and the schedule, once
/// every condition it cannot lower under has been refused.
struct Staging {
    /// Register tiling: one invocation owns `vector` consecutive indices of
    /// grid x, so its tile is that many times taller and its share of the
    /// cooperative load that many times wider - a staged row read in one
    /// transaction instead of four (ADR-2 section 6).
    vector: u32,
    depth: u32,
    inner_axis: AxisId,
    /// The single sum reduction: its value, its operator and the expression it
    /// folds. `rop` is `Sum` by construction and kept so the emitted
    /// `InitAcc`/`Reduce` still name it rather than re-assert it.
    rv: ValueId,
    rop: rir_core::ReduceOp,
    rin: ValueId,
}

fn staging_of(a: &Analysis, schedule: &Schedule) -> Result<Staging, LowerError> {
    let unsupported = |why: &'static str| LowerError::TilingUnsupported { why };

    if schedule.par_map != ParallelMapping::Invocation {
        return Err(unsupported(
            "invocation mapping required: a tile is shared by workgroup invocations",
        ));
    }
    if schedule.vector_width > 4 {
        return Err(unsupported(
            "width exceeds the four native components of GPU emitters",
        ));
    }
    let Some(inner_axis) = a.inner_axis else {
        return Err(unsupported("reduction axis required - nothing to contract"));
    };
    if !a.scans.is_empty() || !a.inner_writes.is_empty() {
        return Err(unsupported(
            "only a contraction closed within the invocation is staged",
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
    Ok(Staging {
        vector: schedule.vector_width,
        depth: schedule.tile_depth,
        inner_axis,
        rv,
        rop,
        rin,
    })
}

/// One staged operand's shared tile, as the consuming loop reads it.
struct TileBinding {
    value: ValueId,
    tile: VarId,
    /// Rows of the tile, and this invocation's row inside it.
    rows: u32,
    local: VarId,
    /// The grid dimension the rows walk; `0` is the vectorized one.
    grid: usize,
}

/// What the workgroup stages: one shared tile per operand, cooperatively
/// filled. Returns the statements and the bindings the consuming loop reads
/// them back through.
fn stage_tiles(
    lo: &mut Lowerer,
    staged: &[StagedOperand],
    k0: VarId,
    inner_axis: AxisId,
    depth: u32,
) -> Result<(Vec<TileStage>, Vec<TileBinding>), LowerError> {
    let mut tiles = Vec::new();
    let mut tile_index = Vec::new();
    for StagedOperand {
        value: rvalue,
        arg,
        quant,
        axes,
        grid,
        rows,
        origin,
        local,
    } in staged
    {
        let name = lo.k.args()[arg.0 as usize].name.clone();
        let tile = lo.new_var(&format!("tile_{name}"), VarKind::Idx);
        // Declared here, where the geometry is decided. The statement defines
        // the tile; this value defines how much shared storage it takes.
        lo.shared.push((tile, rows * depth));
        let row = lo.new_var(&format!("{name}_r"), VarKind::Idx);
        let dep = lo.new_var(&format!("{name}_d"), VarKind::Idx);
        let row_global = lo.new_var(&format!("{name}_m"), VarKind::Idx);
        let depth_global = lo.new_var(&format!("{name}_k"), VarKind::Idx);
        // The index registers the staged element is read at: the two tile
        // coordinates, and whatever outer loop variables are in scope.
        let mut idx_vars = Vec::new();
        for (d, ax) in axes.iter().enumerate() {
            let var = if d == 0 {
                row_global
            } else if *ax == inner_axis {
                depth_global
            } else {
                *lo.axis_vars.get(ax).ok_or(LowerError::AxisOutOfScope {
                    value: *rvalue,
                    axis: *ax,
                })?
            };
            idx_vars.push(var);
        }
        let slot = lo.new_var(&format!("{name}_at"), VarKind::Idx);
        // How many consecutive rows one invocation stages. Above one only for
        // a quantized operand, and only under three conditions lowering can
        // *prove* rather than hope (ADR-3 section 3):
        //
        //   - `span` divides `block_elements`, so a segment never straddles two
        //     blocks and the header it reads is the header of all of them;
        //   - `span` divides `n_rows`, so a segment never straddles two rows of
        //     the tile - which is also what keeps its slots consecutive;
        //   - the segment's first row is a multiple of `span` (the tile origin
        //     is a multiple of `n_rows`, the cooperative loop steps by `span`).
        //
        // Together with the `quant_block` rejection - a row is a
        // whole number of blocks - they make the *segment* the unit of the
        // range test: if its first element is inside the tensor, all of them
        // are, because the boundary is a multiple of `block_elements`.
        let span = match quant {
            Some(f) => {
                let be = f.desc().block_elements;
                [4u32, 2, 1]
                    .into_iter()
                    .find(|s| be % s == 0 && rows % s == 0)
                    .unwrap_or(1)
            }
            None => 1,
        };
        // The loader, built by the **same** code a direct read goes through, so
        // a quantized operand is staged by the format's own decoder and no
        // emitter composes an address or a block layout of its own.
        let outer = std::mem::take(&mut lo.body);
        match quant {
            // The split the span exists for: the block index, the address base
            // and the F16 scales are emitted **once**, then the per-element
            // work goes under a `ForConst` the emitters print as a loop of
            // constant trip count.
            Some(format) if span > 1 => {
                let be = format.desc().block_elements;
                let block = lo.emit("blk", VarKind::Idx, LExpr::IDivC(row_global, be));
                let inb0 = lo.emit("inb0", VarKind::Idx, LExpr::IModC(row_global, be));
                let (base, header) =
                    lo.lower_dequant_header(*arg, &idx_vars, block, Some((inb0, span)), *format)?;
                let elem = lo.new_var(&format!("{name}_e"), VarKind::Idx);
                let inner = std::mem::take(&mut lo.body);
                let inb = lo.emit("inb", VarKind::Idx, LExpr::IAdd(inb0, elem));
                let value = lo.lower_dequant_element(*arg, *format, &base, &header, inb)?;
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
                let value = lo.lower_dequant_read(*arg, &idx_vars, *format)?;
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
                    arg: *arg,
                    ty: Lowerer::elem_mem_type(lo.k, *arg)?,
                    addr: Lowerer::f32_addr(&idx_vars),
                    width: 1,
                });
                lo.body.push(Stmt::StoreShared {
                    array: tile,
                    index: slot,
                    value,
                });
            }
        }
        let load = std::mem::replace(&mut lo.body, outer);
        tiles.push(TileStage {
            tile,
            arg: *arg,
            row,
            depth: dep,
            row_global,
            depth_global,
            row_origin: *origin,
            depth_origin: k0,
            row_axis: axes[0],
            depth_axis: inner_axis,
            slot,
            span,
            load,
            n_rows: *rows,
            n_depth: depth,
        });
        tile_index.push(TileBinding {
            value: *rvalue,
            tile,
            rows: *rows,
            local: *local,
            grid: *grid,
        });
    }
    Ok((tiles, tile_index))
}

/// What each invocation accumulates out of the staged tiles. Seeding `lo.env`
/// with the loaded registers is what makes the body below the *same* expression
/// the untiled lowering builds, with its reads answered from shared memory.
fn consume_tiles(
    lo: &mut Lowerer,
    tile_index: &[TileBinding],
    inner_name: &str,
    vector: u32,
    rin: ValueId,
    acc: VarId,
    rop: rir_core::ReduceOp,
) -> Result<(VarId, Vec<Stmt>), LowerError> {
    let kk = lo.new_var(&format!("{inner_name}k"), VarKind::Idx);
    lo.begin_scope(true);
    for TileBinding {
        value: rvalue,
        tile,
        rows,
        local,
        grid,
    } in tile_index
    {
        let scaled = lo.emit("row", VarKind::Idx, LExpr::IMulC(kk, *rows));
        let index = lo.emit("at", VarKind::Idx, LExpr::IAdd(scaled, *local));
        // Only the tile whose rows walk the vectorized axis is read wide: `w`
        // consecutive tile elements are `w` consecutive outputs of this
        // invocation. The others are scalar and broadcast over them.
        let width = if *grid == 0 { vector } else { 1 };
        let dst = lo.new_var(
            "t",
            if width > 1 {
                VarKind::Vec(width)
            } else {
                VarKind::F32
            },
        );
        lo.body.push(Stmt::LoadShared {
            dst,
            array: *tile,
            index,
            width,
        });
        // Seeding the memo is what makes the body below the *same* expression
        // the untiled lowering builds, with its reads answered from shared
        // memory instead of from the buffer.
        lo.env.insert(*rvalue, dst);
    }
    let term = lo.lower_value(rin)?;
    lo.body.push(Stmt::Accum {
        acc,
        op: rop,
        value: term,
    });
    Ok((kk, lo.take_body()))
}
