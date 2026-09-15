//! The statement walk: loops, memory, quantized loads.
//!
//! It is the bulk of the emitter - 270 lines against 60 for expressions - and
//! it is why `cpu.rs` became a directory. The split is
//! by cohesion, not by symmetry with `metal/`, `vulkan/` and `cuda/`: those
//! three carry the same five file names because they *share*
//! `crate::printer`, and three of their five files are now a doc comment
//! saying so. The CPU shares none of it - it prints Rust, not a C dialect,
//! so it takes the names and not the seam.

use rir_lower::{Inst, MemType, Stmt};

use crate::EmitError;

use super::CpuPrinter;

impl CpuPrinter<'_> {
    pub(super) fn stmts(&mut self, stmts: &[Stmt]) -> Result<(), EmitError> {
        for s in stmts {
            match s {
                Stmt::Parallel {
                    var, axis, body, ..
                } => {
                    let head = format!(
                        "for {} in 0..{} {{",
                        self.k.var_names[var.0 as usize],
                        self.extent_name(*axis)
                    );
                    self.line(&head);
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::For {
                    var,
                    axis,
                    reverse,
                    body,
                } => {
                    let range = format!("0..{}", self.extent_name(*axis));
                    let head = if *reverse {
                        format!(
                            "for {} in ({range}).rev() {{",
                            self.k.var_names[var.0 as usize]
                        )
                    } else {
                        format!("for {} in {range} {{", self.k.var_names[var.0 as usize])
                    };
                    self.line(&head);
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
                    let head = format!(
                        "for {} in ({}..{}).step_by({}) {{",
                        self.k.var_names[var.0 as usize],
                        self.var(*start),
                        self.extent_name(*axis),
                        step
                    );
                    self.line(&head);
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                // The flattened dispatch is a **grid** decision: it exists to fill blocks the hardware
                // would otherwise leave half idle, and the CPU has no blocks. A
                // `cpu_serial` lowering never produces one, so reaching this
                // arm means a GPU schedule reached the wrong emitter - the same
                // thing a lane collective here means, and it is said the same
                // way rather than printed as a nest that would compute the
                // right bytes for the wrong reason.
                Stmt::ParallelFlat { .. } => {
                    return Err(EmitError::UnsupportedStmt {
                        backend: "cpu",
                        stmt: "flattened grid",
                    });
                }
                Stmt::ParallelLane { .. }
                | Stmt::LaneReduce { .. }
                | Stmt::LaneScan { .. }
                | Stmt::WorkgroupReduce { .. }
                | Stmt::Barrier
                | Stmt::LaneZero { .. }
                | Stmt::ForChunk { .. } => {
                    // CPU schedules are Serial. A lane collective indicates
                    // that a GPU schedule reached the wrong emitter.
                    return Err(EmitError::UnsupportedStmt {
                        backend: "cpu",
                        stmt: "lane collective",
                    });
                }
                // Two plain sequential loops, and they are here because a
                // **segment** is not a tile: the
                // quantized reductions walk their axis block by block on every
                // backend, the CPU oracle included, so the pair
                // `ForTiled`/`ForConst` reaches this emitter through
                // `cpu_serial` and not through a stray GPU schedule. Refusing
                // them, which is what this arm did while they meant staging
                // alone, would have made the oracle the one nest nobody could
                // print.
                Stmt::ForTiled {
                    var,
                    axis,
                    step,
                    body,
                } => {
                    let head = format!(
                        "for {} in (0..{}).step_by({step}) {{",
                        self.k.var_names[var.0 as usize],
                        self.extent_name(*axis)
                    );
                    self.line(&head);
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                Stmt::ForConst { var, count, body } => {
                    let head = format!("for {} in 0..{count} {{", self.k.var_names[var.0 as usize]);
                    self.line(&head);
                    self.indent += 1;
                    self.stmts(body)?;
                    self.indent -= 1;
                    self.line("}");
                }
                // Shared memory is a *workgroup* notion - a tile amortized
                // over the invocations that share it, a tree level a barrier
                // separates - and a CPU loop nest has no such thing.
                // `cpu_serial` carries neither `TiledStage` nor a collective, so
                // one of these here means a GPU schedule reached the wrong
                // emitter. `If` and `Set` are in the same list for the same
                // reason and not because they could not be printed: nothing but
                // a lowered collective produces them today, so accepting them
                // would be accepting a nest the oracle is not supposed to see.
                Stmt::StageTiles { .. }
                | Stmt::LoadShared { .. }
                | Stmt::StoreShared { .. }
                | Stmt::If { .. }
                | Stmt::Set { .. }
                | Stmt::InBounds { .. } => {
                    return Err(EmitError::UnsupportedStmt {
                        backend: "cpu",
                        stmt: "shared memory",
                    });
                }
                // The CPU lowering is the oracle, and the oracle is scalar:
                // `cpu_serial` carries `vector_width = 1`, so a vectorized nest
                // here means a GPU schedule reached the wrong emitter. Printing
                // a scalar equivalent would make the oracle stop being the
                // independent reference it exists to be.
                Stmt::VecTail { .. } => {
                    return Err(EmitError::UnsupportedStmt {
                        backend: "cpu",
                        stmt: "vectorized body",
                    });
                }
                Stmt::InitAcc { acc, op } => {
                    let init = match op {
                        rir_core::ReduceOp::Sum => "0.0f32",
                        rir_core::ReduceOp::Max => "f32::NEG_INFINITY",
                    };
                    let l = format!("let mut {} = {};", self.var(*acc), init);
                    self.line(&l);
                }
                Stmt::Accum { acc, op, value } => {
                    let l = match op {
                        rir_core::ReduceOp::Sum => {
                            format!("{} += {};", self.var(*acc), self.var(*value))
                        }
                        rir_core::ReduceOp::Max => {
                            let a = self.var(*acc).to_string();
                            format!("{a} = {a}.max({});", self.var(*value))
                        }
                    };
                    self.line(&l);
                }
                Stmt::Load {
                    dst,
                    arg,
                    ty,
                    addr,
                    width,
                } => {
                    if *width > 1 {
                        return Err(EmitError::UnsupportedStmt {
                            backend: "cpu",
                            stmt: "vector read",
                        });
                    }
                    let name = self.k.args[arg.0 as usize].name.clone();
                    let a = self.addr(*arg, addr);
                    let l = match ty {
                        MemType::F32 => {
                            format!("let {} = {}.data[({}) / 4];", self.var(*dst), name, a)
                        }
                        MemType::F16 => {
                            let addr_var = format!("{}_a", self.var(*dst));
                            self.line(&format!("let {addr_var} = {a};"));
                            format!(
                                "let {} = f16_to_f32(u16::from_le_bytes([{}.data[{addr_var}], {}.data[{addr_var} + 1]]));",
                                self.var(*dst),
                                name,
                                name
                            )
                        }
                        MemType::I8 => format!(
                            "let {} = ({}.data[{}] as i8) as f32;",
                            self.var(*dst),
                            name,
                            a
                        ),
                        // An unsigned byte lands in an integer register: the
                        // reader wants the raw bits, not a converted value.
                        MemType::U8 => {
                            format!("let {} = {}.data[{}] as usize;", self.var(*dst), name, a)
                        }
                    };
                    self.line(&l);
                }
                Stmt::Store {
                    arg,
                    ty,
                    addr,
                    value,
                    width,
                    ..
                } => {
                    if *width > 1 {
                        return Err(EmitError::UnsupportedStmt {
                            backend: "cpu",
                            stmt: "vector write",
                        });
                    }
                    let name = self.k.args[arg.0 as usize].name.clone();
                    let a = self.addr(*arg, addr);
                    let l = match ty {
                        MemType::F32 => {
                            format!("{}.data[({}) / 4] = {};", name, a, self.var(*value))
                        }
                        // The oracle narrows exactly where a shader does, and
                        // with the same rounding: `f32_to_f16` is
                        // round-to-nearest-even, which is what `half(x)` and
                        // `float16_t(x)` are (ADR-3 section 6).
                        MemType::F16 => {
                            let addr_var = format!("{}_a", self.var(*value));
                            self.line(&format!("let {addr_var} = {a};"));
                            self.line(&format!(
                                "let {}_h = f32_to_f16({}).to_le_bytes();",
                                self.var(*value),
                                self.var(*value)
                            ));
                            self.line(&format!(
                                "{name}.data[{addr_var}] = {}_h[0];",
                                self.var(*value)
                            ));
                            format!("{name}.data[{addr_var} + 1] = {}_h[1];", self.var(*value))
                        }
                        MemType::I8 | MemType::U8 => {
                            return Err(EmitError::UnsupportedStmt {
                                backend: "cpu",
                                stmt: "byte write",
                            });
                        }
                    };
                    self.line(&l);
                }
                Stmt::Compute(Inst { dst, expr }) => {
                    let l = format!("let {} = {};", self.var(*dst), self.expr(expr));
                    self.line(&l);
                }
            }
        }
        Ok(())
    }
}
