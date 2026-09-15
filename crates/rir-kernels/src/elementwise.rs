//! The elementwise strip - `MUL`, `ADD`, and `SCALE`, identified together by the
//! real-graph census.
//!
//! The ranking puts them at 7.7% of Qwen3.5 traffic and 13.9% on gemma-3-270m,
//! over roughly 6,700 nodes each. Individually none would justify the effort;
//! together they matter, and they are written together - three instantiations
//! of one kernel shape, just as `sum_rows_quant` is instantiated per quantized
//! format.
//!
//! ```text
//! add: dst[i] = a[i] + b[i]
//! mul: dst[i] = a[i] · b[i]
//! scale: dst[i] = a[i] · scale + bias
//! ```
//!
//! **The claimed domain and what remains native.** ggml allows `src1` to be
//! *repeated* over `src0` (`ggml_can_repeat`) for `ADD` and `MUL`.
//! `add_repeat`/`mul_repeat` claim this since the DSL gained the required index
//! arithmetic (`RepeatIndex`, `read_repeat`); the non-repeating member declines
//! broadcast shapes. The restriction is not hand-written - `a`, `b`, and `dst`
//! are indexed by the **same** axes, and `supports_op` derives equality of all
//! three shapes from axis agreement. Selection therefore uses node shape, and
//! the kernel pair covers the op domain, not either kernel alone.
//!
//! The strip has one member per element type (eight total for `ADD`/`MUL`). Only
//! `SCALE` retains the `dtype` row, for its
//! own reason: `ggml_compute_forward_scale` has no F16 arm, so no benchmark
//! could judge a kernel serving such a node (ADR-3 section 6).
//!
//! **In-place is safe without declaring anything.** ggml calls these three ops
//! with `dst` aliasing `src0` (`ggml_add_inplace`, `ggml_scale_inplace`, …), each
//! invocation reading and writing the **same** index: no ordering between
//! invocations must be preserved, so aliasing does not change the result. This
//! is a property of the kernel shape, not a guarantee for the contract to check.
//!
//! **Vectorization.** Native kernels in this strip are vectorized on both
//! backends: Metal selects a `c4` pipeline processing four elements per thread
//! whenever the shape is contiguous, while Vulkan dispatches flat over
//! `nelements`. `vector_width` is lowered, every member publishes a vec4 variant
//! plus its scalar fallback, and
//! the vector variant claims contiguous layouts. The scalar fallback remains
//! live, serving layouts no vectorization can claim.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ScalarType, TensorType, ValidateError,
    ValidatedKernel,
};

/// A member's ggml op and corresponding kernel shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BandOp {
    Add,
    Mul,
    Scale,
    /// `ADD` with `src1` **repeated** under `src0` (`ggml_can_repeat`).
    AddRepeat,
    /// `MUL` with repeated `src1`.
    MulRepeat,
}

/// Member element type (ADR-3 section 6).
///
/// This is the **only** distinction between otherwise identical members, and
/// the point of F4: both native kernels accept `F32 | F16` for these three ops,
/// while the DSL typed everything as F32. Closing the restriction required a
/// second element type at the memory boundary - not a schedule capability or
/// register bank. An F16 member reads `half`, computes in F32, and writes
/// `half`; the kernel body below is identical.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BandType {
    F32,
    F16,
}

impl BandType {
    fn dtype(self) -> DType {
        match self {
            BandType::F32 => DType::F32,
            BandType::F16 => DType::F16,
        }
    }

    fn tensor(self, rank: usize) -> TensorType {
        match self {
            BandType::F32 => TensorType::f32(rank),
            BandType::F16 => TensorType::f16(rank),
        }
    }

    fn suffix(self) -> &'static str {
        match self {
            BandType::F32 => "",
            BandType::F16 => "_f16",
        }
    }
}

/// A strip member: one ggml op and one element type. Three ggml ops, two types,
/// one kernel pattern - four parallel axes, one read per source, one write.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Band {
    pub op: BandOp,
    pub ty: BandType,
}

/// Canonical strip table in generation order: five F32 members first - their
/// artifacts do not move - followed by five F16 members.
///
/// The two repeated members are distinct kernels, not schedule variants of the
/// first two, for **correctness**, not speed (ADR-1 section 5). A vectorized
/// variant reads four consecutive `src1` elements; under repetition of the
/// contiguous dimension, four consecutive indices are not necessarily four
/// consecutive addresses, and this depends on an extent known at dispatch.
/// Shape selection already arbitrates between two *kernels* - as `out_prod`
/// does for `src0` dtype - but cannot express equality of two extents in a shape
/// rule. Intended result: the contiguous path carrying all measured traffic is
/// unchanged by even one instruction.
///
/// Duplication by type follows exactly the same precedent one level lower: it
/// applies `out_prod`'s **node dtype** selection to a strip whose dtype had been
/// a published restriction until now.
pub fn variants() -> [Band; 9] {
    let f32_ops = [
        BandOp::Add,
        BandOp::Mul,
        BandOp::Scale,
        BandOp::AddRepeat,
        BandOp::MulRepeat,
    ];
    // **`SCALE` has no F16 member, and this is not an omission.** Both native
    // GPU kernels accept it (`supports_op` says yes for `F32 | F16`), but
    // `ggml_compute_forward_scale` has only an F32 arm: the reference backend
    // aborts on an F16 node. A kernel for this case would be a variant no
    // benchmark can judge - "declared and never dispatched," a rejection the
    // lane already states three times in this document - and one no ggml graph
    // can produce without crashing its own CPU. Reopening trigger: when ggml
    // gives `SCALE` an F16 CPU arm (ADR-3 section 6).
    let f16_ops = [
        BandOp::Add,
        BandOp::Mul,
        BandOp::AddRepeat,
        BandOp::MulRepeat,
    ];
    let mut out = [Band {
        op: BandOp::Add,
        ty: BandType::F32,
    }; 9];
    let mut i = 0;
    while i < 5 {
        out[i] = Band {
            op: f32_ops[i],
            ty: BandType::F32,
        };
        i += 1;
    }
    let mut j = 0;
    while j < 4 {
        out[5 + j] = Band {
            op: f16_ops[j],
            ty: BandType::F16,
        };
        j += 1;
    }
    out
}

impl Band {
    /// Kernel name, hence generated directory, `rir_<name>` artifact, and
    /// entrypoint. Deliberately the ggml op name for the F32 member: existing
    /// artifacts do not change names because a second type appeared, keeping
    /// a new member's diff limited to what it adds.
    pub fn kernel_name(self) -> String {
        let base = match self.op {
            BandOp::Add => "add",
            BandOp::Mul => "mul",
            BandOp::Scale => "scale",
            BandOp::AddRepeat => "add_repeat",
            BandOp::MulRepeat => "mul_repeat",
        };
        format!("{base}{}", self.ty.suffix())
    }

    pub fn ggml_op(self) -> &'static str {
        match self.op {
            BandOp::Add | BandOp::AddRepeat => "GGML_OP_ADD",
            BandOp::Mul | BandOp::MulRepeat => "GGML_OP_MUL",
            BandOp::Scale => "GGML_OP_SCALE",
        }
    }

    /// `ADD` and `MUL` take two tensors; `SCALE` takes one tensor and two
    /// scalars. This is the only shape difference among the three.
    pub fn is_binary(self) -> bool {
        !matches!(self.op, BandOp::Scale)
    }

    /// The member repeats `src1` under `src0` instead of requiring equality of
    /// shapes.
    pub fn repeats(self) -> bool {
        matches!(self.op, BandOp::AddRepeat | BandOp::MulRepeat)
    }

    pub fn is_f16(self) -> bool {
        self.ty == BandType::F16
    }
}

pub fn build(band: Band) -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new(&band.kernel_name());

    let a = k.input("a", band.ty.tensor(4));
    let b = band.is_binary().then(|| k.input("b", band.ty.tensor(4)));
    let dst = k.output("dst", band.ty.tensor(4));
    // `ggml_scale` writes `scale` at offset 0 and `bias` at offset 4 in
    // `op_params`; declaration order here matches the integration table, which
    // carries the offsets.
    let scale_params = (band.op == BandOp::Scale).then(|| {
        (
            k.param("scale", ScalarType::F32),
            k.param("bias", ScalarType::F32),
        )
    });

    // `col` first: it is the contiguous axis, therefore determines coalescing,
    // and lowering sends the first parallel axis to x. The next three take y,
    // z, then a sequential loop - the same distribution as `out_prod`, for the
    // same reason.
    let col = k.axis("col", Extent::Dim { arg: a, dim: 0 });
    let row = k.axis("row", Extent::Dim { arg: a, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: a, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: a, dim: 3 });
    let idx = [col, row, plane, batch];

    // The four `src1` axes, declared only for their **extent**: no `Index` names
    // them, so lowering opens no loop over them, and dispatch fills them like
    // any other extent. This gives the shader the repetition divisor without
    // adding a push-constant class (ADR-1 section 5).
    let b_axes = b.filter(|_| band.repeats()).map(|b| {
        [
            k.axis("col_b", Extent::Dim { arg: b, dim: 0 }),
            k.axis("row_b", Extent::Dim { arg: b, dim: 1 }),
            k.axis("plane_b", Extent::Dim { arg: b, dim: 2 }),
            k.axis("batch_b", Extent::Dim { arg: b, dim: 3 }),
        ]
    });

    let av = k.read(a, &idx);
    let read_b = |k: &mut KernelBuilder| {
        // Called only from the binary arms below, where `band.is_binary()` holds,
        // the same condition that bound `b` above.
        let b = b.expect("a binary band declares its src1 input");
        match b_axes {
            Some(over) => k.read_repeat(b, &idx, &over),
            None => k.read(b, &idx),
        }
    };
    let value = match band.op {
        BandOp::Add | BandOp::AddRepeat => {
            let bv = read_b(&mut k);
            k.add(av, bv)
        }
        BandOp::Mul | BandOp::MulRepeat => {
            let bv = read_b(&mut k);
            k.mul(av, bv)
        }
        BandOp::Scale => {
            let (s, bias) = scale_params.expect("BandOp::Scale declares scale and bias params");
            let t = k.mul(av, s);
            k.add(t, bias)
        }
    };
    k.write(dst, &idx, value);

    k.constrain(Constraint::DType {
        arg: a,
        allowed: vec![band.ty.dtype()],
    });
    if let Some(b) = b {
        k.constrain(Constraint::DType {
            arg: b,
            allowed: vec![band.ty.dtype()],
        });
        k.constrain(Constraint::Rank { arg: b, max: 4 });
    }
    k.constrain(Constraint::Rank { arg: a, max: 4 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rir_core::supports::{RejectReason, TensorDesc, supports_op};
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewMut, run};

    fn fill(seed: &mut u64, buf: &mut [f32], scale: f32) {
        for v in buf.iter_mut() {
            *seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((*seed >> 33) as u32) as f32 / u32::MAX as f32;
            *v = (u * 2.0 - 1.0) * scale;
        }
    }

    /// Analytical reference, written from ggml semantics rather than the
    /// kernel: `ggml_compute_forward_add_f32`, `_mul_f32`, `_scale_f32`.
    fn reference(band: Band, a: f32, b: f32, params: (f32, f32)) -> f32 {
        match band.op {
            BandOp::Add | BandOp::AddRepeat => a + b,
            BandOp::Mul | BandOp::MulRepeat => a * b,
            BandOp::Scale => a * params.0 + params.1,
        }
    }

    /// The F32 member of an op - what suites written before F4 exercise and must
    /// continue to exercise identically.
    fn f32_band(op: BandOp) -> Band {
        Band {
            op,
            ty: BandType::F32,
        }
    }

    /// Executes a strided rank-4 shape under `schedule` and compares element by
    /// element. `gap` separates `a` planes: this is the view into a packed
    /// tensor sent by the real graph, which only a strided address can express.
    fn run_case(band: Band, schedule: Schedule, ne: [usize; 4], gap: usize) {
        let kernel = build(band).unwrap();
        let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
        let (n_col, n_row, n_plane, n_batch) = (ne[0], ne[1], ne[2], ne[3]);
        let params = (1.75f32, -0.5f32);

        let packed = |g: usize| {
            [
                4,
                4 * n_col,
                4 * n_col * n_row * g,
                4 * n_col * n_row * g * n_plane,
            ]
        };
        let len = |g: usize| n_col * n_row * n_plane * n_batch * g;

        let mut seed = 0x9e37_2026u64 ^ ((n_col as u64) << 32) ^ (n_row as u64);
        let mut a = vec![0f32; len(gap)];
        let mut b = vec![0f32; len(1)];
        fill(&mut seed, &mut a, 2.0);
        fill(&mut seed, &mut b, 1.5);

        let mut got = vec![0f32; len(1)];
        let shape = [n_col, n_row, n_plane, n_batch];
        let mut args: Vec<BoundArg> = vec![BoundArg::In(TensorView {
            data: &a,
            shape,
            nb: packed(gap),
        })];
        if band.is_binary() {
            args.push(BoundArg::In(TensorView {
                data: &b,
                shape,
                nb: packed(1),
            }));
        }
        args.push(BoundArg::Out(TensorViewMut {
            data: &mut got,
            shape,
            nb: packed(1),
        }));
        let scalars: Vec<f32> = if band.op == BandOp::Scale {
            vec![params.0, params.1]
        } else {
            Vec::new()
        };
        run(&lk, &mut args, &scalars).unwrap();

        for bt in 0..n_batch {
            for p in 0..n_plane {
                for r in 0..n_row {
                    let dst_base = ((bt * n_plane + p) * n_row + r) * n_col;
                    let a_base = ((bt * n_plane * gap + p * gap) * n_row + r) * n_col;
                    for c in 0..n_col {
                        let e = reference(band, a[a_base + c], b[dst_base + c], params);
                        let g = got[dst_base + c];
                        let tol = 1e-6f32.max(1e-6 * e.abs());
                        assert!(
                            (g - e).abs() <= tol,
                            "{band:?} {:?} [{bt},{p},{r},{c}] : {g} vs {e}",
                            schedule.backend()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for band in variants().into_iter().filter(|b| !b.is_f16()) {
            for ne in [[1, 1, 1, 1], [7, 3, 1, 1], [64, 4, 2, 3], [33, 1, 1, 1]] {
                run_case(band, Schedule::cpu_serial(), ne, 1);
            }
        }
    }

    /// The shape class motivating four axes: `a` occupies one plane out of three
    /// in a packed tensor. Under the GPU lowering executed by the fork.
    #[test]
    fn parity_on_a_rank_4_view_with_a_plane_gap() {
        for band in variants().into_iter().filter(|b| !b.is_f16()) {
            for schedule in [Schedule::cpu_serial(), Schedule::vulkan_grid([256, 1, 1])] {
                run_case(band, schedule, [33, 4, 3, 2], 3);
            }
        }
    }

    /// Vectorized lowering (ADR-2 section 6) over decisive row lengths: a
    /// multiple of four, all three possible remainders, then a row **shorter**
    /// than the vector. These are the only shapes exercising the scalar tail,
    /// without them, the shader's `else` branch would be tested nowhere.
    #[test]
    fn parity_of_the_vectorized_lowering_including_its_tail() {
        for band in variants()
            .into_iter()
            .filter(|b| !b.repeats() && !b.is_f16())
        {
            for schedule in [
                Schedule::vulkan_grid_vec4([256, 1, 1]),
                Schedule::metal_grid_vec4([256, 1, 1]),
            ] {
                for n_col in [1usize, 2, 3, 4, 5, 7, 8, 33, 64] {
                    run_case(band, schedule.clone(), [n_col, 3, 2, 2], 2);
                }
            }
        }
    }

    /// A broadcast shape accepted by ggml: one `src1` row repeated over all 16
    /// `src0` rows. The **non-repeated** member must reject it with its reason,
    /// No constraint says it explicitly; axis agreement derives it - and the
    /// repeated member must accept it. The pair performs selection:
    /// `supports_op` arbitrates between two kernels of the same ggml op, just as
    /// it arbitrates `src0` dtype for `out_prod`.
    #[test]
    fn a_repeated_src1_selects_the_repeating_kernel() {
        let desc = |ne: [usize; 4]| TensorDesc {
            dtype: DType::F32,
            ne,
            nb: [4, 4 * ne[0], 4 * ne[0] * ne[1], 4 * ne[0] * ne[1] * ne[2]],
        };
        let full = desc([1024, 16, 1, 1]);
        for band in [f32_band(BandOp::Add), f32_band(BandOp::Mul)] {
            let k = build(band).unwrap();
            assert_eq!(supports_op(&k, &[full, full, full]), Ok(()));
            let one_row = desc([1024, 1, 1, 1]);
            assert!(
                matches!(
                    supports_op(&k, &[full, one_row, full]),
                    Err(RejectReason::AxisExtent { .. })
                ),
                "{band:?}: a broadcast shape must be rejected by the non-repeated member"
            );
        }
        for band in [f32_band(BandOp::AddRepeat), f32_band(BandOp::MulRepeat)] {
            let k = build(band).unwrap();
            // The contiguous shape remains in-domain: repetition is the
            // identity, and priority sends the node to `vec4`.
            assert_eq!(supports_op(&k, &[full, full, full]), Ok(()));
            for repeated in [
                desc([1024, 1, 1, 1]), // one row replayed over sixteen
                desc([1, 16, 1, 1]),   // one column replayed over one thousand
                desc([1024, 8, 1, 1]), // a factor that is neither 1 nor equality
                desc([1, 1, 1, 1]),    // the scalar
            ] {
                assert_eq!(
                    supports_op(&k, &[full, repeated, full]),
                    Ok(()),
                    "{band:?}: {:?} divides the src0 shape",
                    repeated.ne
                );
            }
            // Also what `ggml_can_repeat` rejects: a non-dividing extent. A
            // shape rejection, never modulo that repeats
            // travers.
            let ragged = desc([1024, 5, 1, 1]);
            assert!(
                matches!(
                    supports_op(&k, &[full, ragged, full]),
                    Err(RejectReason::RepeatNotDivisible { .. })
                ),
                "{band:?}: 16 % 5 != 0 must be rejected"
            );
        }
    }

    /// Repetition computes what `ggml_compute_forward_add_f32` computes: every
    /// `src1` dimension is replayed modulo its own extent. Reference written
    /// from ggml semantics, not from the kernel.
    #[test]
    fn a_repeated_src1_is_replayed_dimension_by_dimension() {
        for band in [f32_band(BandOp::AddRepeat), f32_band(BandOp::MulRepeat)] {
            let kernel = build(band).unwrap();
            for schedule in [Schedule::cpu_serial(), Schedule::vulkan_grid([256, 1, 1])] {
                let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
                let ne = [12usize, 6, 4, 2];
                // One repeated extent per dimension, not the same everywhere: a
                // shader applying one dimension's modulo to another would pass
                // a case where they happen to coincide.
                for be in [[12usize, 1, 1, 1], [1, 6, 4, 2], [4, 3, 2, 1], [1, 1, 1, 1]] {
                    let nb = |n: [usize; 4]| [4, 4 * n[0], 4 * n[0] * n[1], 4 * n[0] * n[1] * n[2]];
                    let len = |n: [usize; 4]| n[0] * n[1] * n[2] * n[3];
                    let mut seed = 0x2f_2026u64 ^ (be[0] as u64) << 8 ^ be[1] as u64;
                    let mut a = vec![0f32; len(ne)];
                    let mut b = vec![0f32; len(be)];
                    fill(&mut seed, &mut a, 2.0);
                    fill(&mut seed, &mut b, 1.5);
                    let mut got = vec![0f32; len(ne)];
                    let mut args = vec![
                        BoundArg::In(TensorView {
                            data: &a,
                            shape: ne,
                            nb: nb(ne),
                        }),
                        BoundArg::In(TensorView {
                            data: &b,
                            shape: be,
                            nb: nb(be),
                        }),
                        BoundArg::Out(TensorViewMut {
                            data: &mut got,
                            shape: ne,
                            nb: nb(ne),
                        }),
                    ];
                    run(&lk, &mut args, &[]).unwrap();

                    for i3 in 0..ne[3] {
                        for i2 in 0..ne[2] {
                            for i1 in 0..ne[1] {
                                for i0 in 0..ne[0] {
                                    let ai = ((i3 * ne[2] + i2) * ne[1] + i1) * ne[0] + i0;
                                    let bi = (((i3 % be[3]) * be[2] + i2 % be[2]) * be[1]
                                        + i1 % be[1])
                                        * be[0]
                                        + i0 % be[0];
                                    let e = reference(band, a[ai], b[bi], (0.0, 0.0));
                                    assert!(
                                        (got[ai] - e).abs() <= 1e-6f32.max(1e-6 * e.abs()),
                                        "{band:?} {:?} src1={be:?} [{i3},{i2},{i1},{i0}] : {} vs {e}",
                                        schedule.backend(),
                                        got[ai]
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// The repeated member is vectorized, at the cost of **one additional
    /// claim**, not a branch: four consecutive `src1` indices are consecutive
    /// addresses only if contiguous-axis repetition is the identity. Lowering
    /// produces it, the registry publishes it (`vector_width` plus the
    /// repetition relation), dispatch evaluates it, and a node repeated over
    /// `ne0` falls back to scalar.
    ///
    /// Here the oracle checks the other half: with identity repetition on `ne0`
    /// and real repetition on the other three dimensions, the vectorized body
    /// computes what ggml computes, including the scalar tail.
    #[test]
    fn the_repeating_member_vectorizes_when_the_contiguous_fold_is_the_identity() {
        for band in [f32_band(BandOp::AddRepeat), f32_band(BandOp::MulRepeat)] {
            let kernel = build(band).unwrap();
            for schedule in [
                Schedule::metal_grid_vec4([256, 1, 1]),
                Schedule::vulkan_grid_vec4([256, 1, 1]),
            ] {
                let lk = rir_lower::lower(&kernel, schedule.clone()).expect("vectorized lowering");
                assert_eq!(lk.vector_width(), 4);
                // Decisive lengths: a multiple of four, all three remainders,
                // and a row shorter than the vector.
                for n_col in [1usize, 3, 4, 7, 33] {
                    let ne = [n_col, 6, 4, 2];
                    for be in [[n_col, 1, 1, 1], [n_col, 3, 2, 1], [n_col, 6, 4, 2]] {
                        let nb =
                            |n: [usize; 4]| [4, 4 * n[0], 4 * n[0] * n[1], 4 * n[0] * n[1] * n[2]];
                        let len = |n: [usize; 4]| n[0] * n[1] * n[2] * n[3];
                        let mut seed = 0x4e_2026u64 ^ (n_col as u64) << 8 ^ be[1] as u64;
                        let mut a = vec![0f32; len(ne)];
                        let mut b = vec![0f32; len(be)];
                        fill(&mut seed, &mut a, 2.0);
                        fill(&mut seed, &mut b, 1.5);
                        let mut got = vec![0f32; len(ne)];
                        let mut args = vec![
                            BoundArg::In(TensorView {
                                data: &a,
                                shape: ne,
                                nb: nb(ne),
                            }),
                            BoundArg::In(TensorView {
                                data: &b,
                                shape: be,
                                nb: nb(be),
                            }),
                            BoundArg::Out(TensorViewMut {
                                data: &mut got,
                                shape: ne,
                                nb: nb(ne),
                            }),
                        ];
                        run(&lk, &mut args, &[]).unwrap();
                        for i3 in 0..ne[3] {
                            for i2 in 0..ne[2] {
                                for i1 in 0..ne[1] {
                                    for i0 in 0..ne[0] {
                                        let ai = ((i3 * ne[2] + i2) * ne[1] + i1) * ne[0] + i0;
                                        let bi = (((i3 % be[3]) * be[2] + i2 % be[2]) * be[1]
                                            + i1 % be[1])
                                            * be[0]
                                            + i0 % be[0];
                                        let e = reference(band, a[ai], b[bi], (0.0, 0.0));
                                        assert_eq!(
                                            got[ai], e,
                                            "{band:?} vec4 n_col={n_col} src1={be:?} \
                                             [{i3},{i2},{i1},{i0}]"
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Axes declared only for their extent do not become loops. Otherwise the
    /// repeated kernel would traverse `src1` a second time - visible only in
    /// timing, not the result.
    #[test]
    fn an_extent_only_axis_is_not_a_loop() {
        let plain = rir_lower::lower(
            &build(f32_band(BandOp::Mul)).unwrap(),
            Schedule::cpu_serial(),
        )
        .unwrap();
        let rep = rir_lower::lower(
            &build(f32_band(BandOp::MulRepeat)).unwrap(),
            Schedule::cpu_serial(),
        )
        .unwrap();
        // Eight axes declared for repeated, four for simple - and the same
        // number of nested loops in both nests.
        assert_eq!(plain.axes.len(), 4);
        assert_eq!(rep.axes.len(), 8);
        fn depth(stmts: &[rir_lower::Stmt]) -> usize {
            stmts
                .iter()
                .map(|s| match s {
                    rir_lower::Stmt::For { body, .. } | rir_lower::Stmt::Parallel { body, .. } => {
                        1 + depth(body)
                    }
                    _ => 0,
                })
                .max()
                .unwrap_or(0)
        }
        assert_eq!(depth(&plain.body), depth(&rep.body), "one extra loop");
    }

    /// F16 members against the same analytical reference and under the same
    /// vectorized lowering as their F32 twins (ADR-3 section 6).
    ///
    /// The oracle does not compare F32 values: it **narrows** at the store like
    /// the shader, and compares stored half-floats. This is the only way to see
    /// the sole addition of the F16 members - one rounding - instead of hiding it under a
    /// tolerance.
    #[test]
    fn parity_of_the_f16_members_including_the_rounding_of_their_store() {
        use rir_lower::interp::{TensorViewBytes, TensorViewBytesMut, f32_to_f16};

        let to_f16 = |v: &[f32]| -> Vec<u8> {
            v.iter()
                .flat_map(|x| f32_to_f16(*x).to_le_bytes())
                .collect()
        };
        let from_f16 = |b: &[u8]| -> Vec<f32> {
            b.as_chunks::<2>()
                .0
                .iter()
                .map(|c| {
                    let h = u16::from_le_bytes(*c);
                    half_to_f32(h)
                })
                .collect()
        };

        for band in variants().into_iter().filter(|b| b.is_f16()) {
            for schedule in [
                Schedule::cpu_serial(),
                Schedule::metal_grid_vec4([256, 1, 1]),
                Schedule::vulkan_grid_vec4([256, 1, 1]),
            ] {
                // The repeated member has no `vec4` under arbitrary repetition;
                // these shapes are the identity required for
                // selecting the vectorized variant.
                let lk = match rir_lower::lower(&build(band).unwrap(), schedule.clone()) {
                    Ok(lk) => lk,
                    Err(e) => panic!("{}: {e}", band.kernel_name()),
                };
                let (n_col, n_row) = (33usize, 3usize);
                let n = n_col * n_row;
                let mut seed = 0xf16_2026u64;
                let mut a = vec![0f32; n];
                let mut b = vec![0f32; n];
                fill(&mut seed, &mut a, 2.0);
                fill(&mut seed, &mut b, 1.5);
                // Inputs pass through F16: the reference must use what the
                // kernel reads, otherwise the measured gap would belong to the
                // fixture.
                let a_bytes = to_f16(&a);
                let b_bytes = to_f16(&b);
                let a16 = from_f16(&a_bytes);
                let b16 = from_f16(&b_bytes);

                let params = (1.75f32, -0.5f32);
                let mut got = vec![0u8; n * 2];
                let shape = [n_col, n_row, 1, 1];
                let nb = [2usize, 2 * n_col, 2 * n, 2 * n];
                let mut args: Vec<BoundArg> = vec![BoundArg::InBytes(TensorViewBytes {
                    data: &a_bytes,
                    shape,
                    nb,
                })];
                if band.is_binary() {
                    args.push(BoundArg::InBytes(TensorViewBytes {
                        data: &b_bytes,
                        shape,
                        nb,
                    }));
                }
                args.push(BoundArg::OutBytes(TensorViewBytesMut {
                    data: &mut got,
                    shape,
                    nb,
                }));
                let scalars: Vec<f32> = if band.op == BandOp::Scale {
                    vec![params.0, params.1]
                } else {
                    Vec::new()
                };
                run(&lk, &mut args, &scalars).unwrap();

                let out = from_f16(&got);
                for i in 0..n {
                    let e = half_to_f32(f32_to_f16(reference(band, a16[i], b16[i], params)));
                    assert_eq!(
                        out[i],
                        e,
                        "{} {:?} [{i}]: store rounding",
                        band.kernel_name(),
                        schedule.backend()
                    );
                }
            }
        }
    }

    /// F16 → F32 widening for this file's expectations: the interpreter's own
    /// decoder, and not a fourth hand-written copy.
    ///
    /// RIR keeps exactly three copies of binary16↔binary32, each judged over
    /// its 65 536 inputs by `tests/f16_oracles.rs`.
    /// A private copy here would sit outside that witness while serving to
    /// build the expected values of a test - it could redrift alone, which is
    /// the failure the exhaustive comparison exists to make impossible. The
    /// narrowing beside it already comes from `rir_lower::interp`, and the
    /// oracle this test compares against is the *emitted* kernel: the pair
    /// stays independent of what it judges.
    use rir_lower::interp::f16_to_f32 as half_to_f32;

    #[allow(dead_code)]
    mod generated_add {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/add/cpu.rs"
        ));
    }

    // `too_many_arguments` is the only lint the emitted CPU kernels trip, and it
    // is not a style choice: the arity is the kernel's shape rank plus its
    // buffers, and the emitter cannot be bent to please a lint without moving
    // `generated/`, which must regenerate byte for byte. Named rather than blanketed
    // (`clippy::all`), so the next lint an emitter change trips is reported.
    #[allow(dead_code, clippy::too_many_arguments)]
    mod generated_scale {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/scale/cpu.rs"
        ));
    }

    /// The **generated** CPU, not the interpreter: this is the oracle path and
    /// must return the same result on a strided shape.
    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row, stride) = (33usize, 7usize, 40usize);
        let len = stride * n_row;
        let mut seed = 0xb17e_2027u64;
        let mut a = vec![0f32; len];
        let mut b = vec![0f32; len];
        fill(&mut seed, &mut a, 2.0);
        fill(&mut seed, &mut b, 1.5);
        let nb = [4usize, 4 * stride, 4 * stride * n_row, 4 * stride * n_row];

        let mut got = vec![0f32; len];
        generated_add::add(
            n_col,
            n_row,
            1,
            1,
            generated_add::TensorRef { data: &a, nb },
            generated_add::TensorRef { data: &b, nb },
            generated_add::TensorRefMut { data: &mut got, nb },
        );
        for r in 0..n_row {
            for c in 0..n_col {
                let i = r * stride + c;
                assert_eq!(got[i], a[i] + b[i], "generated add [{r},{c}]");
            }
        }

        let mut got = vec![0f32; len];
        let (s, bias) = (1.75f32, -0.5f32);
        generated_scale::scale(
            n_col,
            n_row,
            1,
            1,
            generated_scale::TensorRef { data: &a, nb },
            generated_scale::TensorRefMut { data: &mut got, nb },
            s,
            bias,
        );
        for r in 0..n_row {
            for c in 0..n_col {
                let i = r * stride + c;
                assert_eq!(got[i], a[i] * s + bias, "generated scale [{r},{c}]");
            }
        }
    }
}
