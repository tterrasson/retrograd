//! `out_prod` - the first candidate identified by the real-graph census:
//! 8–14% of backward memory traffic, over 1,200 nodes
//! on both measured models, regardless of architecture.
//!
//! ```text
//! dst[i,j,p,b] = Σ_k a[i,k,p,b] · b[j,k,p,b]
//! ```
//!
//! This is exactly `ggml_compute_forward_out_prod_f32`, rewritten as an explicit
//! sum rather than accumulation into a pre-zeroed destination: native zeros
//! `dst` and adds one contribution per `i01`; here the contraction is closed
//! within the invocation, so nothing needs clearing first. The result is the
//! same because the sum traverses `k` in the same direction.
//!
//! **A quantized `src0` is the same kernel one table row later**
//! (ADR-3 section 3). Most observed `OUT_PROD` nodes have a quantized `src0`,
//! the frozen LoRA weight. The body below changes by
//! no character: `KernelBuilder::read` inserts explicit `Dequant` for a
//! quantized argument, and lowering fuses it into tile staging. What had to be
//! built was neither kernel nor emitter, but the ability for a tile to carry a
//! *loader* rather than an address.
//!
//! **The domain this kernel does not claim.** ggml allows `src0` broadcast over
//! dimensions 2 and 3 (`ne12 % ne02 == 0`), which is not expressible here: the
//! DSL lacks index arithmetic and therefore index division for broadcasting.
//! The restriction is not hand-written - `supports_op` derives it from `plane`
//! and `batch` indexing *both* sources. A broadcast shape is rejected with its
//! reason and then encoded by native; it is never computed incorrectly. Formats
//! lowering cannot decode also remain outside the domain: `variants()` excludes
//! them by querying `can_lower_dequant`, never by
//! listing names.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, Layout, QUANT_FORMATS, QuantType, ReduceOp,
    ReductionSemantics, TensorType, ValidateError, ValidatedKernel,
};

/// Kernel name for a `src0` dtype: `out_prod` for F32, `out_prod_q4_K` for a
/// quantized weight.
pub fn kernel_name(format: Option<QuantType>) -> String {
    match format {
        None => "out_prod".to_string(),
        Some(q) => format!("out_prod_{}", q.desc().name),
    }
}

/// The `src0` dtypes this kernel is generated for: F32, then every member of
/// the **`out_prod` family** of the canonical table (ADR-3 section 2) whose
/// block shape lowering can expand.
///
/// Two filters, and neither is a list of names. `ops.out_prod` is what ggml's
/// own fork says may reach this op; `can_lower_dequant` is what RIR can decode
/// without a backend primitive. A format that fails either is left to the native
/// kernel - and `assumed_domain` says so - rather than generated wrong.
pub fn variants() -> Vec<Option<QuantType>> {
    let mut v = vec![None];
    v.extend(
        QUANT_FORMATS
            .into_iter()
            .filter(|q| {
                let d = q.desc();
                d.ops.out_prod && d.block_elements > 1 && rir_lower::can_lower_dequant(*q)
            })
            .map(Some),
    );
    v
}

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    build_for(None)
}

pub fn build_for(format: Option<QuantType>) -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new(&kernel_name(format));

    let a_ty = match format {
        None => TensorType::f32(4),
        Some(q) => TensorType {
            dtype: DType::Quant(q),
            rank: 4,
            layout: Layout::Ggml,
        },
    };
    let a = k.input("a", a_ty);
    let b = k.input("b", TensorType::f32(4));
    let dst = k.output("dst", TensorType::f32(4));

    // Declaration order is what lowering sends to the grid: the first three
    // parallel axes take x, y, z, while the fourth remains a sequential loop.
    // `i` first because it is `dst`'s contiguous axis and determines write
    // coalescing.
    let i = k.axis("i", Extent::Dim { arg: a, dim: 0 });
    let j = k.axis("j", Extent::Dim { arg: b, dim: 0 });
    let plane = k.axis("plane", Extent::Dim { arg: b, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: b, dim: 3 });
    let kk = k.axis("k", Extent::Dim { arg: a, dim: 1 });

    let av = k.read(a, &[i, kk, plane, batch]);
    let bv = k.read(b, &[j, kk, plane, batch]);
    let p = k.mul(av, bv);
    let s = k.reduce(ReduceOp::Sum, kk, p, ReductionSemantics::Deterministic);
    k.write(dst, &[i, j, plane, batch], s);

    k.constrain(Constraint::DType {
        arg: a,
        allowed: vec![match format {
            None => DType::F32,
            Some(q) => DType::Quant(q),
        }],
    });
    k.constrain(Constraint::DType {
        arg: b,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::Rank { arg: a, max: 4 });
    k.constrain(Constraint::Rank { arg: b, max: 4 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use rir_core::DType;
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

    /// Case geometry: ggml extents and `nb[]` in bytes, source by source.
    #[derive(Clone, Copy)]
    struct Case {
        n_i: usize,
        n_j: usize,
        n_k: usize,
        n_plane: usize,
        n_batch: usize,
        /// Second-dimension multiplier: simulates a view into a packed tensor,
        /// where rows are farther apart than their logical length.
        gap: usize,
    }

    impl Case {
        fn a_nb(&self) -> [usize; 4] {
            let s1 = 4 * self.n_i * self.gap;
            [4, s1, s1 * self.n_k, s1 * self.n_k * self.n_plane]
        }
        fn b_nb(&self) -> [usize; 4] {
            let s1 = 4 * self.n_j;
            [4, s1, s1 * self.n_k, s1 * self.n_k * self.n_plane]
        }
        fn d_nb(&self) -> [usize; 4] {
            let s1 = 4 * self.n_i;
            [4, s1, s1 * self.n_j, s1 * self.n_j * self.n_plane]
        }
        fn a_len(&self) -> usize {
            self.a_nb()[3] / 4 * self.n_batch
        }
        fn b_len(&self) -> usize {
            self.b_nb()[3] / 4 * self.n_batch
        }
        fn d_len(&self) -> usize {
            self.d_nb()[3] / 4 * self.n_batch
        }
    }

    /// Independent reference: the `ggml_compute_forward_out_prod_f32` loop,
    /// written from `nb[]` rather than an assumed packing.
    fn reference(c: &Case, a: &[f32], b: &[f32]) -> Vec<f32> {
        let (anb, bnb, dnb) = (c.a_nb(), c.b_nb(), c.d_nb());
        let e = |nb: [usize; 4], idx: [usize; 4]| -> usize {
            (0..4).map(|d| idx[d] * nb[d]).sum::<usize>() / 4
        };
        let mut dst = vec![0f32; c.d_len()];
        for bt in 0..c.n_batch {
            for p in 0..c.n_plane {
                for j in 0..c.n_j {
                    for i in 0..c.n_i {
                        let mut acc = 0f32;
                        for k in 0..c.n_k {
                            acc += a[e(anb, [i, k, p, bt])] * b[e(bnb, [j, k, p, bt])];
                        }
                        dst[e(dnb, [i, j, p, bt])] = acc;
                    }
                }
            }
        }
        dst
    }

    fn run_case(schedule: Schedule, c: Case) -> Vec<f32> {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, schedule).unwrap();

        let mut seed =
            0x0a7_2026u64 ^ ((c.n_i as u64) << 24) ^ ((c.n_j as u64) << 12) ^ c.n_k as u64;
        let mut a = vec![0f32; c.a_len()];
        let mut b = vec![0f32; c.b_len()];
        fill(&mut seed, &mut a, 1.0);
        fill(&mut seed, &mut b, 1.0);
        let expected = reference(&c, &a, &b);

        let mut got = vec![0f32; c.d_len()];
        let shape_a = [c.n_i, c.n_k, c.n_plane, c.n_batch];
        let shape_b = [c.n_j, c.n_k, c.n_plane, c.n_batch];
        let shape_d = [c.n_i, c.n_j, c.n_plane, c.n_batch];
        let mut args = [
            BoundArg::In(TensorView {
                data: &a,
                shape: shape_a,
                nb: c.a_nb(),
            }),
            BoundArg::In(TensorView {
                data: &b,
                shape: shape_b,
                nb: c.b_nb(),
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut got,
                shape: shape_d,
                nb: c.d_nb(),
            }),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for (idx, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4f32.max(1e-5 * e.abs()),
                "{}x{}x{} element {idx}: {g} vs {e}",
                c.n_i,
                c.n_j,
                c.n_k
            );
        }
        got
    }

    fn cases() -> Vec<Case> {
        vec![
            Case {
                n_i: 1,
                n_j: 1,
                n_k: 1,
                n_plane: 1,
                n_batch: 1,
                gap: 1,
            },
            Case {
                n_i: 5,
                n_j: 3,
                n_k: 8,
                n_plane: 1,
                n_batch: 1,
                gap: 1,
            },
            Case {
                n_i: 7,
                n_j: 4,
                n_k: 33,
                n_plane: 1,
                n_batch: 1,
                gap: 1,
            },
            Case {
                n_i: 4,
                n_j: 6,
                n_k: 3,
                n_plane: 2,
                n_batch: 3,
                gap: 1,
            },
            // The view: `a` occupies one row out of three in a packed tensor, so
            // its `nb[1..3]` cannot be derived from any product of extents.
            Case {
                n_i: 3,
                n_j: 5,
                n_k: 4,
                n_plane: 2,
                n_batch: 2,
                gap: 3,
            },
            // More than one 16×16×16 tile in every direction, with none of the
            // three extents a tile multiple: tiling has edges on both output
            // axes and the contraction.
            Case {
                n_i: 20,
                n_j: 18,
                n_k: 40,
                n_plane: 1,
                n_batch: 1,
                gap: 1,
            },
        ]
    }

    #[test]
    fn parity_of_the_interpreter_against_the_reference() {
        for c in cases() {
            run_case(Schedule::cpu_serial(), c);
        }
    }

    /// The GPU schedule sends `i`, `j`, and `plane` to the grid and keeps `batch`
    /// as a sequential loop: this is remapping, not reassociation, so the result
    /// must be **bit-for-bit** equal to sequential.
    #[test]
    fn the_grid_mapping_does_not_change_the_result() {
        for c in cases() {
            let serial = run_case(Schedule::cpu_serial(), c);
            let grid = run_case(Schedule::vulkan_grid([16, 16, 1]), c);
            assert_eq!(serial, grid, "the grid changed the result");
        }
    }

    /// Tiling (ADR-2 section 6) is a **memory** decision, not arithmetic:
    /// `k` increases within a tile and tiles increase, so each output sums the
    /// same terms in the same order as sequential. Required equality is thus
    /// **bit-for-bit** - tolerance would hide exactly what this test verifies.
    ///
    /// Shapes cover the three edges staging must zero-fill: a tile wider than
    /// `n_i`, taller than `n_j`, and a contraction not divisible by tile depth.
    #[test]
    fn tiling_does_not_change_the_result() {
        for c in cases() {
            let serial = run_case(Schedule::cpu_serial(), c);
            for tiled in [
                // Delivered geometry: BM = 16 · 4 = 64, BN = 16, BK = 16.
                Schedule::vulkan_grid_tiled([16, 16, 1], 16, 4),
                Schedule::metal_grid_tiled([16, 16, 1], 16, 4),
                // The same staging without register tiling: width changes
                // geometry, never the result.
                Schedule::vulkan_grid_tiled([16, 16, 1], 16, 1),
                // Deliberately degenerate geometry: a tile one row deep shares
                // nothing over `k`, and the result must remain the same.
                Schedule::vulkan_grid_tiled([8, 4, 1], 1, 1),
            ] {
                let got = run_case(tiled, c);
                assert_eq!(serial, got, "tiling changed the result");
            }
        }
    }

    /// Shared-storage size is **derived** from the loop nest, not maintained as
    /// a number beside the schedule: this is the property
    /// `SharedTree` pays for and every schedule has to preserve.
    #[test]
    fn the_shared_storage_is_derived_from_the_loop_nest() {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::metal_grid_tiled([16, 16, 1], 16, 4)).unwrap();
        let tiles = lk.shared.clone();
        // One tile per contracted operand, with vector width included in
        // geometry: `a` over 16 · 4 = 64 `i` rows, `b` over 16 `j` rows, both
        // 16 deep.
        let lens: Vec<u32> = tiles.iter().map(|(_, l)| *l).collect();
        assert_eq!(
            lens,
            vec![64 * 16, 16 * 16],
            "one contracted operand, one tile"
        );
        assert!(lk.uses_shared());

        // Different geometry yields different storage without any emitter
        // needing to know.
        let lk = rir_lower::lower(&kernel, Schedule::metal_grid_tiled([8, 4, 1], 32, 1)).unwrap();
        let lens: Vec<u32> = lk.shared.clone().into_iter().map(|(_, l)| l).collect();
        assert_eq!(lens, vec![8 * 32, 4 * 32]);
    }

    /// Anything tiling cannot stage is an **error**, never a silent fallback to
    /// global reads: the manifest would publish `tiled_stage` for a shader that
    /// tiles nothing.
    #[test]
    fn a_shape_tiling_cannot_stage_is_an_error() {
        use rir_lower::LowerError;
        let kernel = super::build().unwrap();

        // The constructor sets the strategy and depth together. A zero depth is
        // therefore a **stated error**, not a shader whose manifest advertises
        // a tile it does not stage.
        let depthless = Schedule::vulkan_grid_tiled([16, 16, 1], 0, 4);
        assert!(matches!(
            rir_lower::lower(&kernel, depthless),
            Err(LowerError::Schedule(
                rir_lower::ScheduleError::Malformed { .. }
            ))
        ));

        // `mat_mul_naive` contracts over its **contiguous** dimension, so this
        // shape cannot be staged as a tile of consecutive addresses.
        let mm = crate::mat_mul_naive::build().unwrap();
        assert!(matches!(
            rir_lower::lower(&mm, Schedule::vulkan_grid_tiled([16, 16, 1], 16, 1)),
            Err(LowerError::TilingUnsupportedRead { .. })
        ));

        // An elementwise kernel has no contraction to amortize.
        let add = crate::elementwise::build(crate::elementwise::variants()[0]).unwrap();
        assert!(matches!(
            rir_lower::lower(&add, Schedule::vulkan_grid_tiled([16, 16, 1], 16, 1)),
            Err(LowerError::TilingUnsupported { .. })
        ));
    }

    /// Two domains left native by this kernel, checked as **rejections**, not
    /// cases that "do not occur": `src0` broadcast over planes (allowed by
    /// ggml), and quantized `src0` (accepted by the fork). Without this test, the
    /// first becomes out-of-bounds access and the second reinterpreted bytes.
    #[test]
    fn broadcast_and_quantized_shapes_are_rejected_not_miscomputed() {
        let kernel = super::build().unwrap();
        let f32d = |ne: [usize; 4], nb: [usize; 4]| TensorDesc {
            dtype: DType::F32,
            ne,
            nb,
        };
        // ne = [i, k, plane, batch] / [j, k, plane, batch] / [i, j, plane, batch]
        let a = f32d([4, 8, 2, 1], [4, 16, 128, 256]);
        let b = f32d([6, 8, 2, 1], [4, 24, 192, 384]);
        let d = f32d([4, 6, 2, 1], [4, 16, 96, 192]);
        assert_eq!(supports_op(&kernel, &[a, b, d]), Ok(()));

        // Broadcast `a`: one plane for both `b` planes.
        let a_bcast = f32d([4, 8, 1, 1], [4, 16, 128, 128]);
        assert!(matches!(
            supports_op(&kernel, &[a_bcast, b, d]),
            Err(RejectReason::AxisExtent { .. })
        ));

        // Quantized `a`: the dtype contract rejects before any address.
        let a_q = TensorDesc {
            dtype: DType::Quant(rir_core::QuantType::Q8_0),
            ne: [32, 8, 2, 1],
            nb: [34, 34, 272, 544],
        };
        assert!(matches!(
            supports_op(&kernel, &[a_q, b, d]),
            Err(RejectReason::DType { .. })
        ));
    }

    // ---- quantized `src0` (ADR-3 section 3) -------------------------

    /// A quantized `a` tensor and the F32 values it encodes.
    ///
    /// Bytes are **random**: every bit pattern is a legal block, and a fixture
    /// written here would be a disguised second decoder - it would agree with a
    /// wrong kernel exactly where packing is wrong. The reference is therefore
    /// the portable oracle, compared element by element with ggml `to_float` by
    /// `tests/rir_quant_oracle.rs`. The trust chain is ggml → oracle → lowering,
    /// with no link written twice.
    ///
    /// Only leading F16 scales are fixed: a random half-float is infinite or NaN
    /// about once in thirty-two, leaving a contraction with nothing comparable.
    fn quantized_a(
        format: rir_core::QuantType,
        n_i: usize,
        n_k: usize,
        n_plane: usize,
        n_batch: usize,
        seed: &mut u64,
    ) -> (Vec<u8>, Vec<f32>, [usize; 4]) {
        let d = format.desc();
        let (be, bb) = (d.block_elements as usize, d.block_bytes as usize);
        assert_eq!(n_i % be, 0, "{}: ne0 is a multiple of {be}", d.name);
        let row_bytes = n_i / be * bb;
        let rows = n_k * n_plane * n_batch;
        // One fixture derived from the description: it knows where to pin
        // F16 scales for each format. The version pinning "two for q4_K, one
        // otherwise" decoded NaNs at the first format whose `dmin` was not at
        // offset 2.
        let raw = rir_core::random_block_bytes(format, n_i / be * rows, seed)
            .unwrap_or_else(|| panic!("{}: format without a description", d.name));
        let mut logical = Vec::with_capacity(n_i * rows);
        for r in 0..rows {
            logical.extend(
                rir_core::dequantize_row(format, &raw[r * row_bytes..(r + 1) * row_bytes], n_i)
                    .unwrap(),
            );
        }
        let nb = [bb, row_bytes, row_bytes * n_k, row_bytes * n_k * n_plane];
        (raw, logical, nb)
    }

    /// Formats for which an `out_prod` is generated, excluding F32.
    fn quant_formats() -> Vec<rir_core::QuantType> {
        super::variants().into_iter().flatten().collect()
    }

    /// The kernel body changes by no character between F32 and quantized: this
    /// test verifies the result also changes only by format precision loss,
    /// the contraction reads values decoded by the oracle.
    #[test]
    fn a_quantized_src0_contracts_the_values_the_oracle_decodes() {
        for format in quant_formats() {
            let be = format.desc().block_elements as usize;
            let (n_i, n_j, n_k, n_plane, n_batch) = (be * 2, 3, 5, 2, 2);
            let mut seed = 0x9_2026u64 ^ format.desc().id as u64;
            let (raw, a_logical, a_nb) = quantized_a(format, n_i, n_k, n_plane, n_batch, &mut seed);

            let b_nb = [4, 4 * n_j, 4 * n_j * n_k, 4 * n_j * n_k * n_plane];
            let d_nb = [4, 4 * n_i, 4 * n_i * n_j, 4 * n_i * n_j * n_plane];
            let mut b = vec![0f32; b_nb[3] / 4 * n_batch];
            fill(&mut seed, &mut b, 1.0);

            // The reference: ggml's loop over **decoded** values.
            let e = |nb: [usize; 4], idx: [usize; 4], unit: usize| -> usize {
                (0..4).map(|d| idx[d] * nb[d]).sum::<usize>() / unit
            };
            let a_logical_nb = [1, n_i, n_i * n_k, n_i * n_k * n_plane];
            let mut expected = vec![0f32; d_nb[3] / 4 * n_batch];
            for bt in 0..n_batch {
                for p in 0..n_plane {
                    for j in 0..n_j {
                        for i in 0..n_i {
                            let mut acc = 0f32;
                            for k in 0..n_k {
                                acc += a_logical[e(a_logical_nb, [i, k, p, bt], 1)]
                                    * b[e(b_nb, [j, k, p, bt], 4)];
                            }
                            expected[e(d_nb, [i, j, p, bt], 4)] = acc;
                        }
                    }
                }
            }

            let kernel = super::build_for(Some(format)).unwrap();
            let mut results = Vec::new();
            for schedule in [
                Schedule::cpu_serial(),
                Schedule::vulkan_grid_tiled([16, 16, 1], 16, 4),
                Schedule::metal_grid_tiled([16, 16, 1], 16, 4),
            ] {
                let lk = rir_lower::lower(&kernel, schedule).unwrap();
                let mut got = vec![0f32; expected.len()];
                let mut args = [
                    BoundArg::InBytes(rir_lower::interp::TensorViewBytes {
                        data: &raw,
                        shape: [n_i, n_k, n_plane, n_batch],
                        nb: a_nb,
                    }),
                    BoundArg::In(TensorView {
                        data: &b,
                        shape: [n_j, n_k, n_plane, n_batch],
                        nb: b_nb,
                    }),
                    BoundArg::Out(TensorViewMut {
                        data: &mut got,
                        shape: [n_i, n_j, n_plane, n_batch],
                        nb: d_nb,
                    }),
                ];
                run(&lk, &mut args, &[]).unwrap();
                for (idx, (g, x)) in got.iter().zip(&expected).enumerate() {
                    assert!(
                        (g - x).abs() <= 1e-3f32.max(1e-5 * x.abs()),
                        "{} element {idx}: {g} vs {x}",
                        format.desc().name
                    );
                }
                results.push(got);
            }
            // Staging a **quantized** tile remains a memory decision: `k`
            // increases within a tile and tiles increase, so the sum is
            // bit-for-bit equal to sequential. Decoding occurs at the same point
            // in both cases - loading - so no tolerance is justified.
            for got in &results[1..] {
                assert_eq!(
                    &results[0],
                    got,
                    "{}: tiling changed the result",
                    format.desc().name
                );
            }
        }
    }

    /// The table decides, not a name list: a format outside the `out_prod`
    /// family or without a portable decoder has no kernel, and the F32 variant's
    /// contract rejects it - it is never misread.
    #[test]
    fn the_variant_list_follows_the_two_tables_it_asks() {
        let names: Vec<&str> = quant_formats().iter().map(|q| q.desc().name).collect();
        assert!(names.contains(&"q4_K"), "{names:?}");
        assert!(names.contains(&"q8_0"), "{names:?}");
        for q in rir_core::QUANT_FORMATS {
            let d = q.desc();
            if names.contains(&d.name) {
                assert!(d.ops.out_prod, "{}: outside the out_prod family", d.name);
                assert!(
                    rir_lower::can_lower_dequant(q),
                    "{}: without a decoder",
                    d.name
                );
            }
        }
        // Ten standard formats, plus two made nearly free by the LUT table
        // (ADR-3 section 5). `q6_K` is included because the census
        // counted it at 56 of 736 nodes.
        for standard in [
            "q2_K", "q3_K", "q4_0", "q4_1", "q4_K", "q5_0", "q5_1", "q5_K", "q6_K", "q8_0",
        ] {
            assert!(
                names.contains(&standard),
                "{standard} absent from {names:?}"
            );
        }
        // `mxfp4` remains excluded with its reason: its E8M0 scale would require
        // one more scalar conversion in three emitters for a format outside the
        // ten and absent from every census. A rejection, not an omission.
        assert!(!names.contains(&"mxfp4"), "{names:?}");
    }

    /// Dtype selection is a **claim**, like a block layout: every kernel rejects
    /// types it does not decode, and dispatch places the node on the right row.
    #[test]
    fn each_kernel_claims_exactly_its_own_src0_dtype() {
        let f32d = |ne: [usize; 4], nb: [usize; 4]| TensorDesc {
            dtype: DType::F32,
            ne,
            nb,
        };
        let b = f32d([6, 8, 1, 1], [4, 24, 192, 192]);
        for format in quant_formats() {
            let desc = format.desc();
            let (be, bb) = (desc.block_elements as usize, desc.block_bytes as usize);
            let kernel = super::build_for(Some(format)).unwrap();
            // A one-block row: `nb[1]` therefore equals `block_bytes`.
            let a = TensorDesc {
                dtype: DType::Quant(format),
                ne: [be, 8, 1, 1],
                nb: [bb, bb, bb * 8, bb * 8],
            };
            let dq = f32d([be, 6, 1, 1], [4, 4 * be, 24 * be, 24 * be]);
            assert_eq!(supports_op(&kernel, &[a, b, dq]), Ok(()));

            // F32 `a` on a quantized kernel and vice versa: both are dtype
            // rejections, never reinterpreted-byte reads.
            let a_f32 = f32d([be, 8, 1, 1], [4, 4 * be, 32 * be, 32 * be]);
            assert!(matches!(
                supports_op(&kernel, &[a_f32, b, dq]),
                Err(RejectReason::DType { .. })
            ));
            let plain = super::build().unwrap();
            assert!(matches!(
                supports_op(&plain, &[a, b, dq]),
                Err(RejectReason::DType { .. })
            ));

            // A row that is not a whole number of blocks is refused
            // because generated code addresses `i / block_elements`.
            let ragged = TensorDesc {
                dtype: DType::Quant(format),
                ne: [be + 1, 8, 1, 1],
                nb: [bb, bb, bb * 8, bb * 8],
            };
            let d_ragged = f32d([be + 1, 6, 1, 1], [4, 4 * (be + 1), 0, 0]);
            assert!(matches!(
                supports_op(&kernel, &[ragged, b, d_ragged]),
                Err(RejectReason::QuantBlock { .. })
            ));
        }
    }

    // The **generated** CPU compiles and computes the same result as the reference.
    // `too_many_arguments` is the only lint the emitted CPU kernels trip, and it
    // is not a style choice: the arity is the kernel's shape rank plus its
    // buffers, and the emitter cannot be bent to please a lint without moving
    // `generated/`, which must regenerate byte for byte. Named rather than blanketed
    // (`clippy::all`), so the next lint an emitter change trips is reported.
    #[expect(clippy::too_many_arguments)]
    mod generated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/out_prod/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let c = Case {
            n_i: 5,
            n_j: 3,
            n_k: 9,
            n_plane: 2,
            n_batch: 2,
            gap: 1,
        };
        let mut seed = 0x0b7_2026u64;
        let mut a = vec![0f32; c.a_len()];
        let mut b = vec![0f32; c.b_len()];
        fill(&mut seed, &mut a, 1.0);
        fill(&mut seed, &mut b, 1.0);
        let expected = reference(&c, &a, &b);

        let mut got = vec![0f32; c.d_len()];
        generated::out_prod(
            c.n_i,
            c.n_j,
            c.n_plane,
            c.n_batch,
            c.n_k,
            generated::TensorRef {
                data: &a,
                nb: c.a_nb(),
            },
            generated::TensorRef {
                data: &b,
                nb: c.b_nb(),
            },
            generated::TensorRefMut {
                data: &mut got,
                nb: c.d_nb(),
            },
        );

        for (idx, (g, e)) in got.iter().zip(&expected).enumerate() {
            assert!(
                (g - e).abs() <= 1e-4,
                "generated CPU, element {idx}: {g} vs {e}"
            );
        }
    }
}
