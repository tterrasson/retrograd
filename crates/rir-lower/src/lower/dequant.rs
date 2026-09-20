//! Fusing `Read` + `Dequant` into Loop IR: block headers, packed bit
//! fields, sub-block scales, and the segmentation that reads a header once
//! per block instead of once per element.

use rir_core::IrId;
use rir_core::{ArgId, AxisId, Kernel, Op, ValueId, depends_on_axis};

use crate::loop_ir::*;

use crate::lower::{LowerError, Lowerer, can_lower_dequant};

/// What a quantized block shares between its elements, resolved once by
/// `Lowerer::lower_dequant_header`.
///
/// A named record rather than a positional `Vec<VarId>`: with five scale
/// packings the slots stopped being "scale then dmin then maybe two more", and
/// a header read by index is exactly how a min gets multiplied by a scale.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DequantHeader {
    /// The block's F16 scale.
    pub(crate) d: VarId,
    /// The block's F16 min (`q4_1`) or `dmin` (K quants), when it has one.
    pub(crate) m: Option<VarId>,
    /// The sub-block (scale, min) pair, when a staged segment shares one.
    pub(crate) sub: Option<(VarId, Option<VarId>)>,
}

impl<'k> Lowerer<'k> {
    /// Fused quantized load: blocks dimension 0 using `QuantFormat`, reads the
    /// block's scale and payload, and produces F32 - with no materialized
    /// dequantized tensor.
    ///
    /// **The expansion is driven by the format's `BlockLayout`**, the
    /// description the canonical table carries beside `BlockShape`.
    /// There is one expansion, not one arm per
    /// format: the index plan says where the bits of element `e` are, the scale
    /// plan says what multiplies them, and the formula says what is subtracted.
    /// A format the table describes no layout for is an explicit error, never
    /// an approximation - `NativeIntrinsicQuant` when RIR deliberately defers to
    /// the backend primitive, `UnsupportedQuantShape` when the table
    /// simply has no description yet.
    ///
    /// Takes the index registers rather than the semantic indices, so the same
    /// decoder serves a direct read and a **staged** one: a tile's coordinates
    /// are registers the emitters' cooperative loop drives, and no `ValueId`
    /// names them.
    ///
    /// It is split in two halves for a measured reason: everything that
    /// depends on the *block* and not on the element - the address base, the
    /// F16 scales, and now the packed sub-block scale - is read once per block,
    /// and a staged segment reads it once for the whole segment. Fused into one
    /// function, it was re-read for every element, which cost the quantized
    /// `out_prod` its promotion.
    pub(crate) fn lower_dequant_read(
        &mut self,
        tensor: ArgId,
        idx_vars: &[VarId],
        from: rir_core::QuantType,
    ) -> Result<VarId, LowerError> {
        let desc = from.desc();
        let block = self.emit(
            "blk",
            VarKind::Idx,
            LExpr::IDivC(idx_vars[0], desc.block_elements),
        );
        let inb = self.emit(
            "inb",
            VarKind::Idx,
            LExpr::IModC(idx_vars[0], desc.block_elements),
        );
        let (base, header) = self.lower_dequant_header(tensor, idx_vars, block, None, from)?;
        self.lower_dequant_element(tensor, from, &base, &header, inb)
    }

    /// The per-**block** half: the address base, and everything an element of
    /// the block does not have to re-read.
    ///
    /// `segment` is the index inside the block of the first element of a staged
    /// segment, together with its length. It buys the second level of sharing:
    /// a K quant carries one packed scale per **sub-block**, so a segment that
    /// stays inside one sub-block resolves it here rather than once per element.
    /// The condition is `sub_elements % span == 0` - read from the description,
    /// not the `32 % span == 0` that spelled out `q4_K`'s sub-block and would
    /// have silently mis-shared `q2_K`'s sixteen.
    pub(crate) fn lower_dequant_header(
        &mut self,
        tensor: ArgId,
        idx_vars: &[VarId],
        block: VarId,
        segment: Option<(VarId, u32)>,
        from: rir_core::QuantType,
    ) -> Result<(Vec<AddrTerm>, DequantHeader), LowerError> {
        let desc = from.desc();
        if !desc.is_portable() {
            return Err(LowerError::NativeIntrinsicQuant { format: from });
        }
        let layout = desc
            .layout
            .ok_or(LowerError::UnsupportedQuantShape { format: from })?;
        // For a quantized tensor, nb[0] is the block size (ggml type_size).
        let mut base = vec![AddrTerm::VarNb { var: block, dim: 0 }];
        for (d, &v) in idx_vars.iter().enumerate().skip(1) {
            base.push(AddrTerm::VarNb { var: v, dim: d });
        }
        let f16_at = |lo: &mut Self, name: &str, off: u32| -> VarId {
            let mut addr = base.clone();
            if off != 0 {
                addr.push(AddrTerm::Const(off));
            }
            let dst = lo.new_var(name, VarKind::F32);
            lo.body.push(Stmt::Load {
                dst,
                arg: tensor,
                ty: MemType::F16,
                addr,
                width: 1,
            });
            dst
        };
        let header = match layout.scales {
            rir_core::ScalePlan::Global { d_off } => DequantHeader {
                d: f16_at(self, "scale", d_off),
                m: None,
                sub: None,
            },
            rir_core::ScalePlan::GlobalMin { d_off, m_off } => DequantHeader {
                d: f16_at(self, "scale", d_off),
                m: Some(f16_at(self, "min", m_off)),
                sub: None,
            },
            rir_core::ScalePlan::SubBlock {
                d_off,
                dmin_off,
                sub_elements,
                ..
            } => {
                let d = f16_at(self, "scale", d_off);
                let m = dmin_off.map(|o| f16_at(self, "dmin", o));
                let sub = match segment {
                    Some((inb_base, span)) if span > 1 && sub_elements % span == 0 => {
                        Some(self.lower_sub_scales(tensor, &layout, &base, inb_base)?)
                    }
                    _ => None,
                };
                DequantHeader { d, m, sub }
            }
        };
        Ok((base, header))
    }

    /// One bit field of the block, as an `Idx` register, from its index plan.
    ///
    /// This is the block layout's index plan turned into three registers and a load:
    ///
    /// ```text
    /// byte  = offset + (e / group)·plane + (e % plane)
    /// shift = bits · ((e % group) / plane)
    /// field = (mem[byte] >> shift) & mask
    /// ```
    ///
    /// Every term whose value the block size makes constant is dropped rather
    /// than emitted and multiplied by one - a plan covering the whole block has
    /// no group term, a plan one byte wide has no in-plane term.
    pub(crate) fn lower_bit_field(
        &mut self,
        tensor: ArgId,
        base: &[AddrTerm],
        plan: rir_core::BitPlan,
        elem: VarId,
        block_elements: u32,
        name: &str,
    ) -> VarId {
        let (group, plane) = (plan.group(), plan.plane);
        let mut addr = base.to_vec();
        if plan.offset != 0 {
            addr.push(AddrTerm::Const(plan.offset));
        }
        if group < block_elements {
            let g = self.emit(
                &format!("{name}_grp"),
                VarKind::Idx,
                LExpr::IDivC(elem, group),
            );
            addr.push(AddrTerm::VarConst { var: g, c: plane });
        }
        if plane > 1 {
            let inp = if plane >= block_elements {
                elem
            } else {
                self.emit(
                    &format!("{name}_inp"),
                    VarKind::Idx,
                    LExpr::IModC(elem, plane),
                )
            };
            addr.push(AddrTerm::VarConst { var: inp, c: 1 });
        }
        let raw = self.new_var(&format!("{name}_byte"), VarKind::Idx);
        self.body.push(Stmt::Load {
            dst: raw,
            arg: tensor,
            ty: MemType::U8,
            addr,
            width: 1,
        });
        if plan.per_byte() == 1 {
            return raw;
        }
        let ingroup = if group >= block_elements {
            elem
        } else {
            self.emit(
                &format!("{name}_ing"),
                VarKind::Idx,
                LExpr::IModC(elem, group),
            )
        };
        let sel = if plane == 1 {
            ingroup
        } else {
            self.emit(
                &format!("{name}_sel"),
                VarKind::Idx,
                LExpr::IDivC(ingroup, plane),
            )
        };
        let shift = if plan.bits == 1 {
            sel
        } else {
            self.emit(
                &format!("{name}_sh"),
                VarKind::Idx,
                LExpr::IMulC(sel, plan.bits),
            )
        };
        let shifted = self.emit(
            &format!("{name}_shr"),
            VarKind::Idx,
            LExpr::IShr(raw, shift),
        );
        self.emit(name, VarKind::Idx, LExpr::IAndC(shifted, plan.mask()))
    }

    /// The per-**element** half: the payload read and the arithmetic, given the
    /// block's base address and what `lower_dequant_header` resolved.
    ///
    /// The whole formula, in the order ggml writes it - the association is not
    /// a detail, it is what lets the parity test demand equality rather than a
    /// tolerance:
    ///
    /// ```text
    /// (d [· sc]) · (LUT[q] | q − offset)  [+ m | − dmin·mm]
    /// ```
    pub(crate) fn lower_dequant_element(
        &mut self,
        tensor: ArgId,
        from: rir_core::QuantType,
        block_base: &[AddrTerm],
        header: &DequantHeader,
        inb: VarId,
    ) -> Result<VarId, LowerError> {
        let desc = from.desc();
        let layout = desc
            .layout
            .ok_or(LowerError::UnsupportedQuantShape { format: from })?;
        let be = desc.block_elements;
        let base = block_base.to_vec();

        // The payload, as the F32 term the scale multiplies.
        let qterm = if layout.signed {
            // A signed byte is read as one: extracting it from an unsigned load
            // would need a sign fix-up per element, and `MemType::I8` is the
            // conversion the memory boundary already owns. `q8_0`, and only it.
            let mut addr = base.clone();
            if layout.payload.offset != 0 {
                addr.push(AddrTerm::Const(layout.payload.offset));
            }
            addr.push(AddrTerm::VarConst { var: inb, c: 1 });
            let qv = self.new_var("q", VarKind::F32);
            self.body.push(Stmt::Load {
                dst: qv,
                arg: tensor,
                ty: MemType::I8,
                addr,
                width: 1,
            });
            qv
        } else {
            let mut q = self.lower_bit_field(tensor, &base, layout.payload, inb, be, "q");
            // The high plane is concatenated **above** the payload, which is
            // exactly what ggml's `qs | (h << 4)` and `qv - (bit ? 0: 4)` both
            // are - the second one read forwards, with the offset carrying the
            // bias the branch hid.
            if let Some(high) = layout.high {
                let h = self.lower_bit_field(tensor, &base, high, inb, be, "qh");
                let shifted = self.emit(
                    "qh_up",
                    VarKind::Idx,
                    LExpr::IMulC(h, 1u32 << layout.payload.bits),
                );
                q = self.emit("q_full", VarKind::Idx, LExpr::IOr(q, shifted));
            }
            match layout.value {
                rir_core::Payload::Lut(table) => {
                    self.emit("q_lut", VarKind::F32, LExpr::Lut { table, idx: q })
                }
                rir_core::Payload::Offset(0) => self.emit("q_f", VarKind::F32, LExpr::IToF(q)),
                rir_core::Payload::Offset(off) => {
                    let qf = self.emit("q_f", VarKind::F32, LExpr::IToF(q));
                    let c = self.emit("q_off", VarKind::F32, LExpr::ConstF32(off as f32));
                    self.emit("q_c", VarKind::F32, LExpr::Sub(qf, c))
                }
            }
        };

        // The scale, block-wide or block times sub-block.
        let (scale, sub_min) = match layout.scales {
            rir_core::ScalePlan::Global { .. } | rir_core::ScalePlan::GlobalMin { .. } => {
                (header.d, None)
            }
            rir_core::ScalePlan::SubBlock { .. } => {
                let (sc, mm) = match header.sub {
                    Some(pair) => pair,
                    None => self.lower_sub_scales(tensor, &layout, &base, inb)?,
                };
                (self.emit("dl", VarKind::F32, LExpr::Mul(header.d, sc)), mm)
            }
        };

        let scaled = self.emit("t", VarKind::F32, LExpr::Mul(scale, qterm));
        Ok(match layout.min {
            rir_core::MinTerm::None => scaled,
            rir_core::MinTerm::GlobalPlus => {
                let m = header
                    .m
                    .ok_or(LowerError::UnsupportedQuantShape { format: from })?;
                self.emit("t", VarKind::F32, LExpr::Add(scaled, m))
            }
            rir_core::MinTerm::SubMinus => {
                let dmin = header
                    .m
                    .ok_or(LowerError::UnsupportedQuantShape { format: from })?;
                let mm = sub_min.ok_or(LowerError::UnsupportedQuantShape { format: from })?;
                let ml = self.emit("ml", VarKind::F32, LExpr::Mul(dmin, mm));
                self.emit("t", VarKind::F32, LExpr::Sub(scaled, ml))
            }
        })
    }

    /// The sub-block scale (and min, when the formula subtracts one) of the
    /// sub-block containing element `inb`, already converted and de-biased.
    ///
    /// One arm per **packing**, not per format: the five here are the five
    /// ways ggml actually packs a sub-block scale, and `q4_K`/`q5_K` share one
    /// where the shared header logic applies to both paths.
    ///
    /// Called with the segment's first element when a segment shares one
    /// sub-block, and with the element itself otherwise. Both are the same
    /// expression; only *where* it is emitted differs, which is the whole point
    /// of the split.
    pub(crate) fn lower_sub_scales(
        &mut self,
        tensor: ArgId,
        layout: &rir_core::BlockLayout,
        base: &[AddrTerm],
        inb: VarId,
    ) -> Result<(VarId, Option<VarId>), LowerError> {
        let rir_core::ScalePlan::SubBlock {
            sub_elements,
            off,
            packing,
            bias,
            ..
        } = layout.scales
        else {
            unreachable!("lower_sub_scales: scale plan without sub-blocks");
        };
        let u8_at = |lo: &mut Self, name: &str, terms: Vec<AddrTerm>| -> VarId {
            let dst = lo.new_var(name, VarKind::Idx);
            lo.body.push(Stmt::Load {
                dst,
                arg: tensor,
                ty: MemType::U8,
                addr: terms,
                width: 1,
            });
            dst
        };
        let at = |o: u32, var: VarId| -> Vec<AddrTerm> {
            let mut a = base.to_vec();
            if o != 0 {
                a.push(AddrTerm::Const(o));
            }
            a.push(AddrTerm::VarConst { var, c: 1 });
            a
        };
        let is = self.emit("sub_is", VarKind::Idx, LExpr::IDivC(inb, sub_elements));

        let (sc_i, mm_i, sc_f_direct) = match packing {
            // `q2_K`: one byte per sub-block, scale in the low nibble, min in
            // the high one.
            rir_core::SubScalePacking::NibblePair => {
                let addr = at(off, is);
                let raw = u8_at(self, "sub_raw", addr);
                let sc = self.emit("sub_sc", VarKind::Idx, LExpr::IAndC(raw, 15));
                let mm = self.emit("sub_mm", VarKind::Idx, LExpr::IShrC(raw, 4));
                (Some(sc), Some(mm), None)
            }
            // `q6_K`: one signed byte per sub-block, read as one for the same
            // reason `q8_0`'s payload is.
            rir_core::SubScalePacking::I8 => {
                let mut addr = base.to_vec();
                if off != 0 {
                    addr.push(AddrTerm::Const(off));
                }
                addr.push(AddrTerm::VarConst { var: is, c: 1 });
                let dst = self.new_var("sub_sc_f", VarKind::F32);
                self.body.push(Stmt::Load {
                    dst,
                    arg: tensor,
                    ty: MemType::I8,
                    addr,
                    width: 1,
                });
                (None, None, Some(dst))
            }
            // `q4_K`/`q5_K`: `get_scale_min_k4`, eight 6-bit (scale, min) pairs
            // in twelve bytes.
            //
            // Written on `j % 4` and `j / 4` rather than on `j`, because the
            // three bytes it can touch are then at three *constant* offsets from
            // the same index: `q[j]` and `q[j+4]` below four, and `q[j-4]`,
            // `q[j]`, `q[j+4]` above - which is `q[lo]`, `q[lo+4]`, `q[lo+8]` in
            // both cases. Three unconditional loads instead of a branch over
            // addresses.
            rir_core::SubScalePacking::SixBitPairs12 => {
                let lo_j = self.emit("sub_jlo", VarKind::Idx, LExpr::IAndC(is, 3));
                let hi_j = self.emit("sub_jhi", VarKind::Idx, LExpr::IShrC(is, 2));
                let a0 = at(off, lo_j);
                let sa = u8_at(self, "sub_sa", a0);
                let a1 = at(off + 4, lo_j);
                let sb = u8_at(self, "sub_sb", a1);
                let a2 = at(off + 8, lo_j);
                let sc_ = u8_at(self, "sub_sc2", a2);

                // j < 4: (q[j] & 63, q[j+4] & 63).
                let sc_low = self.emit("sub_sc_low", VarKind::Idx, LExpr::IAndC(sa, 63));
                let m_low = self.emit("sub_m_low", VarKind::Idx, LExpr::IAndC(sb, 63));
                // j ≥ 4: ((q[j+4] & 15) | (q[j-4] >> 6) << 4,
                //         (q[j+4] >> 4)  | (q[j]   >> 6) << 4).
                let sc_hi_lo = self.emit("sub_sc_hl", VarKind::Idx, LExpr::IAndC(sc_, 15));
                let sa_top = self.emit("sub_sa_top", VarKind::Idx, LExpr::IShrC(sa, 6));
                let sa_top4 = self.emit("sub_sa_t4", VarKind::Idx, LExpr::IMulC(sa_top, 16));
                let sc_high = self.emit("sub_sc_hi", VarKind::Idx, LExpr::IOr(sc_hi_lo, sa_top4));
                let m_hi_lo = self.emit("sub_m_hl", VarKind::Idx, LExpr::IShrC(sc_, 4));
                let sb_top = self.emit("sub_sb_top", VarKind::Idx, LExpr::IShrC(sb, 6));
                let sb_top4 = self.emit("sub_sb_t4", VarKind::Idx, LExpr::IMulC(sb_top, 16));
                let m_high = self.emit("sub_m_hi", VarKind::Idx, LExpr::IOr(m_hi_lo, sb_top4));

                let half_c = self.emit("sub_half_c", VarKind::F32, LExpr::ConstF32(0.5));
                let low_group = self.lower_is_low(hi_j, half_c, "sub_low_group");
                return Ok((
                    self.lower_pick("sub_scale", low_group, sc_low, sc_high),
                    Some(self.lower_pick("sub_min", low_group, m_low, m_high)),
                ));
            }
            // `q3_K`: sixteen 6-bit scales in twelve bytes, four low bits from
            // one of two four-byte words and two high bits from the third.
            // `ggml`'s `kmask1`/`kmask2` shuffle is per byte lane, so the whole
            // thing resolves to two byte reads at computed offsets.
            rir_core::SubScalePacking::Kmask12 => {
                let t = self.emit("sub_t", VarKind::Idx, LExpr::IAndC(is, 3));
                let g = self.emit("sub_g", VarKind::Idx, LExpr::IShrC(is, 2));
                let g_lo = self.emit("sub_glo", VarKind::Idx, LExpr::IAndC(g, 1));
                let g_hi = self.emit("sub_ghi", VarKind::Idx, LExpr::IShrC(g, 1));

                let mut a_lo = base.to_vec();
                if off != 0 {
                    a_lo.push(AddrTerm::Const(off));
                }
                a_lo.push(AddrTerm::VarConst { var: g_lo, c: 4 });
                a_lo.push(AddrTerm::VarConst { var: t, c: 1 });
                let raw_lo = u8_at(self, "sub_lo_raw", a_lo);
                let sh_lo = self.emit("sub_lo_sh", VarKind::Idx, LExpr::IMulC(g_hi, 4));
                let shifted_lo = self.emit("sub_lo_shr", VarKind::Idx, LExpr::IShr(raw_lo, sh_lo));
                let lo4 = self.emit("sub_lo4", VarKind::Idx, LExpr::IAndC(shifted_lo, 15));

                let a_hi = at(off + 8, t);
                let raw_hi = u8_at(self, "sub_hi_raw", a_hi);
                let sh_hi = self.emit("sub_hi_sh", VarKind::Idx, LExpr::IMulC(g, 2));
                let shifted_hi = self.emit("sub_hi_shr", VarKind::Idx, LExpr::IShr(raw_hi, sh_hi));
                let hi2 = self.emit("sub_hi2", VarKind::Idx, LExpr::IAndC(shifted_hi, 3));
                let hi_up = self.emit("sub_hi_up", VarKind::Idx, LExpr::IMulC(hi2, 16));
                let sc = self.emit("sub_sc", VarKind::Idx, LExpr::IOr(lo4, hi_up));
                (Some(sc), None, None)
            }
            // `iq4_xs`: four low bits packed two per byte plus two high bits
            // taken from a 16-bit word - two byte reads, same shape as above.
            rir_core::SubScalePacking::FourPlusTwo { high_off } => {
                let byte_lo = self.emit("sub_lo_b", VarKind::Idx, LExpr::IShrC(is, 1));
                let sel_lo = self.emit("sub_lo_s", VarKind::Idx, LExpr::IAndC(is, 1));
                let a_lo = at(off, byte_lo);
                let raw_lo = u8_at(self, "sub_lo_raw", a_lo);
                let sh_lo = self.emit("sub_lo_sh", VarKind::Idx, LExpr::IMulC(sel_lo, 4));
                let shifted_lo = self.emit("sub_lo_shr", VarKind::Idx, LExpr::IShr(raw_lo, sh_lo));
                let lo4 = self.emit("sub_lo4", VarKind::Idx, LExpr::IAndC(shifted_lo, 15));

                let byte_hi = self.emit("sub_hi_b", VarKind::Idx, LExpr::IShrC(is, 2));
                let sel_hi = self.emit("sub_hi_s", VarKind::Idx, LExpr::IAndC(is, 3));
                let a_hi = at(high_off, byte_hi);
                let raw_hi = u8_at(self, "sub_hi_raw", a_hi);
                let sh_hi = self.emit("sub_hi_sh", VarKind::Idx, LExpr::IMulC(sel_hi, 2));
                let shifted_hi = self.emit("sub_hi_shr", VarKind::Idx, LExpr::IShr(raw_hi, sh_hi));
                let hi2 = self.emit("sub_hi2", VarKind::Idx, LExpr::IAndC(shifted_hi, 3));
                let hi_up = self.emit("sub_hi_up", VarKind::Idx, LExpr::IMulC(hi2, 16));
                let sc = self.emit("sub_sc", VarKind::Idx, LExpr::IOr(lo4, hi_up));
                (Some(sc), None, None)
            }
        };

        // Conversion and bias, in ggml's own order: the integer scale is
        // de-biased *then* converted nowhere - every K quant converts first in
        // this expansion and the two agree because a 6-bit scale minus 32 is
        // exact in both.
        let sc_f = match sc_f_direct {
            Some(v) => v,
            None => {
                let raw = sc_i.expect("packing without an integer scale");
                self.emit("sub_sc_f", VarKind::F32, LExpr::IToF(raw))
            }
        };
        let sc_f = if bias != 0 {
            let c = self.emit("sub_bias", VarKind::F32, LExpr::ConstF32(bias as f32));
            self.emit("sub_sc_b", VarKind::F32, LExpr::Sub(sc_f, c))
        } else {
            sc_f
        };
        let mm_f = mm_i.map(|m| self.emit("sub_mm_f", VarKind::F32, LExpr::IToF(m)));
        Ok((sc_f, mm_f))
    }
}

/// A quantized read an accumulation loop may traverse **by segment** rather
/// than by element.
///
/// The `Dequant` node is what a body reads, so it is what a segment has to
/// answer: its header - the block index, the address base, the F16 scales and,
/// when the format has one, the packed sub-block scale - is resolved once for
/// the whole segment, and the memo (`Lowerer::env`) hands the per-element result
/// to whatever expression the kernel wrote around it.
#[derive(Clone)]
pub(crate) struct QuantRead {
    pub(crate) value: ValueId,
    pub(crate) arg: ArgId,
    pub(crate) idx: Vec<ValueId>,
    pub(crate) format: rir_core::QuantType,
}

/// How many consecutive elements of the reduced axis one iteration covers, and
/// the reads that make it worth more than one.
pub(crate) struct SegmentPlan {
    pub(crate) span: u32,
    pub(crate) reads: Vec<QuantRead>,
}

/// What one segment resolved, carried from the segment scope to the element
/// scope: the address base of the block, its header, and the index inside the
/// block of the segment's first element.
pub(crate) struct SegmentHeader {
    pub(crate) read: QuantRead,
    pub(crate) base: Vec<AddrTerm>,
    pub(crate) header: DequantHeader,
    pub(crate) inb0: VarId,
}

/// The largest span a format shares a header over, among the widths lowering
/// knows how to emit.
///
/// Two divisibility conditions, both read from the description rather than
/// spelled out per format: `block_elements % span == 0` so a segment never
/// straddles two blocks - the header it reads is then the header of all of its
/// elements - and, when the scales are per sub-block, `sub_elements % span == 0`
/// so the packed scale is shared too. Without the second one a K quant would
/// share its F16 pair and re-read its 6-bit scale per element, which is the more
/// expensive half of the two.
pub(crate) fn format_span(format: rir_core::QuantType) -> u32 {
    let desc = format.desc();
    let sub = match desc.layout.map(|l| l.scales) {
        Some(rir_core::ScalePlan::SubBlock { sub_elements, .. }) => Some(sub_elements),
        _ => None,
    };
    [4u32, 2, 1]
        .into_iter()
        .find(|s| {
            desc.block_elements.is_multiple_of(*s) && sub.is_none_or(|e| e.is_multiple_of(*s))
        })
        .unwrap_or(1)
}

/// The segment plan of one reduced axis: which quantized reads can share a
/// header, and over how many elements.
///
/// A read qualifies when three things hold, and each of them is what makes the
/// segment *provably* inside one block and inside the tensor:
///
/// - it indexes the **contiguous** dimension of its own quantized argument with
///   the reduced axis, and that axis nowhere else. `supports_op` refuses a
///   quantized argument whose `ne[0]` is not a whole number of blocks
///   (`RejectReason::QuantBlock`) and makes every argument indexed by an axis
///   agree on its extent, so the axis extent is a multiple of `block_elements`
/// - and therefore of any span dividing it;
/// - the segment's first element is a multiple of the span, which the traversals
///   below guarantee by construction (`lane · span`, stepping by
///   `lanes · span`);
/// - the span divides the block, and the sub-block when there is one.
///
/// Together they remove the range test a segment would otherwise need per
/// element: if the first element of a segment is inside the axis, all of them
/// are. A read that does not qualify is not an error and does not disable the
/// plan - it simply lowers per element inside the segment, exactly as it does
/// today.
pub(crate) fn plan_segments(k: &Kernel, inner_axis: AxisId) -> SegmentPlan {
    let mut reads = Vec::new();
    for (i, op) in k.ops().iter().enumerate() {
        let Op::Dequant { value: rv, from } = op else {
            continue;
        };
        let Op::Read { tensor, idx } = &k.ops()[rv.0 as usize] else {
            continue;
        };
        // The reduced axis on the contiguous dimension, and nowhere else: a
        // second occurrence would make element `seg + e` sit at an address the
        // segment's base does not describe.
        if !matches!(k.ops()[idx[0].0 as usize], Op::Index(a) if a == inner_axis) {
            continue;
        }
        if idx
            .iter()
            .skip(1)
            .any(|&iv| depends_on_axis(k, iv, inner_axis))
        {
            continue;
        }
        if !can_lower_dequant(*from) {
            continue;
        }
        reads.push(QuantRead {
            value: ValueId::at(i),
            arg: *tensor,
            idx: idx.clone(),
            format: *from,
        });
    }
    let span = reads
        .iter()
        .map(|r| format_span(r.format))
        .min()
        .unwrap_or(1);
    SegmentPlan { span, reads }
}
