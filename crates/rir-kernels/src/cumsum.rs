//! `cumsum` - the stage-E reference kernel: a forward inclusive scan.
//!
//! ```text
//! y[c,r,p,b] = Σ_{c'≤c} x[c',r,p,b]
//! ```
//!
//! The current implementation executes it sequentially; a blocked
//! `ScanStrategy` is provided separately by `scan_plan`.
//! Transposition derives its backward pass as a **backward** scan of `dy`,
//! `dx[j] = Σ_{i≥j} dy[i]`, tested against the analytical form.
//!
//! Like `l2_norm_back`, it carries the **three outer axes** of the ggml rank
//! (`row`, `plane`, `batch`) around the scanned axis `col`. A rank-2 tensor is
//! the same problem with `n_plane = n_batch = 1`; what the outer axes buy is a
//! distinct `nb[2]`/`nb[3]` per argument, i.e. the ability to scan a view
//! inside a packed tensor. `GGML_OP_CUMSUM` is a rank-4 op in ggml, so without
//! them the variant would refuse every node a real graph builds.

use rir_core::{
    Constraint, DType, Extent, KernelBuilder, ScanDirection, ScanOp, TensorType, ValidateError,
    ValidatedKernel,
};

pub fn build() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("cumsum");

    let x = k.input("x", TensorType::f32(4));
    let y = k.output("y", TensorType::f32(4));

    // Declaration order is what lowering maps to the grid: `row` to x,
    // `plane` to y, `batch` to z, leaving `col` as the sequential scan axis.
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let plane = k.axis("plane", Extent::Dim { arg: x, dim: 2 });
    let batch = k.axis("batch", Extent::Dim { arg: x, dim: 3 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });

    let xv = k.read(x, &[col, row, plane, batch]);
    let s = k.scan(ScanOp::Sum, col, ScanDirection::Forward, xv);
    k.write(y, &[col, row, plane, batch], s);

    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::F32],
    });
    k.constrain(Constraint::Rank { arg: x, max: 4 });

    k.finish()
}

#[cfg(test)]
mod tests {
    use rir_core::{
        Constraint, DType, Extent, KernelBuilder, ScanDirection, ScanOp, TensorType,
        derive_backward,
    };
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

    #[test]
    fn parity_against_the_reference_prefixes() {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap();

        for &(n_col, n_row, stride) in &[(1usize, 1usize, 1usize), (7, 3, 7), (33, 2, 40)] {
            let len = stride * n_row;
            let mut seed = 0xc5_2026u64 ^ (n_col as u64);
            let mut x = vec![0f32; len];
            fill(&mut seed, &mut x, 1.0);

            let mut got = vec![0f32; len];
            let mut args = [
                BoundArg::In(TensorView::strided_2d(&x, n_col, n_row, stride)),
                BoundArg::Out(TensorViewMut::strided_2d(&mut got, n_col, n_row, stride)),
            ];
            run(&lk, &mut args, &[]).unwrap();

            for r in 0..n_row {
                let mut acc = 0f32;
                for c in 0..n_col {
                    acc += x[r * stride + c];
                    let g = got[r * stride + c];
                    assert!(
                        (g - acc).abs() <= 1e-5f32.max(1e-5 * acc.abs()),
                        "[{r},{c}] : {g} vs {acc}"
                    );
                }
            }
        }
    }

    /// Vulkan uses one invocation per row and keeps the scan sequential
    /// **within** that invocation. `ExactOrder` requires bit-for-bit equality
    /// with the serial path; a reassociating GPU scan would require a distinct
    /// `ScanStrategy`, not merely a remapping.
    #[test]
    fn the_grid_mapping_does_not_change_the_scan() {
        let kernel = super::build().unwrap();
        let serial = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap();
        let grid = rir_lower::lower(&kernel, Schedule::vulkan_grid([64, 1, 1])).unwrap();

        for &(n_col, n_row, stride) in &[(1usize, 1usize, 1usize), (7, 3, 7), (33, 2, 40)] {
            let len = stride * n_row;
            let mut seed = 0xc5_2026u64 ^ (n_col as u64);
            let mut x = vec![0f32; len];
            fill(&mut seed, &mut x, 1.0);

            let mut out = [vec![0f32; len], vec![0f32; len]];
            for (lk, got) in [&serial, &grid].into_iter().zip(out.iter_mut()) {
                let mut args = [
                    BoundArg::In(TensorView::strided_2d(&x, n_col, n_row, stride)),
                    BoundArg::Out(TensorViewMut::strided_2d(got, n_col, n_row, stride)),
                ];
                run(lk, &mut args, &[]).unwrap();
            }
            assert_eq!(out[0], out[1], "{n_col}x{n_row}: the grid changed the scan");
        }
    }

    /// The coalesced tiled scan (ADR-2 section 5) computes the same
    /// prefixes as the sequential one. Two lane counts and two tile widths,
    /// against row lengths that are a whole number of tiles, a fraction of one,
    /// and one element past a tile - the boundary where the tail slots must
    /// hold the identity for the running carry to stay right.
    #[test]
    fn the_tiled_scan_matches_the_sequential_scan() {
        let kernel = super::build().unwrap();
        let serial = rir_lower::lower(&kernel, Schedule::cpu_serial()).unwrap();

        for (lanes, items) in [(4u32, 2u32), (8, 4)] {
            let tiled =
                rir_lower::lower(&kernel, Schedule::vulkan_tiled_scan(lanes, items)).unwrap();
            let tile = (lanes * items) as usize;
            for &n_col in &[1usize, 3, tile, tile + 1, 3 * tile, 3 * tile - 1] {
                let n_row = 3usize;
                let len = n_col * n_row;
                let mut seed = 0x0007_11ed_2026u64 ^ (n_col as u64);
                let mut x = vec![0f32; len];
                fill(&mut seed, &mut x, 1.0);

                let mut out = [vec![0f32; len], vec![0f32; len]];
                for (lk, got) in [&serial, &tiled].into_iter().zip(out.iter_mut()) {
                    let mut args = [
                        BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
                        BoundArg::Out(TensorViewMut::contiguous_2d(got, n_col, n_row)),
                    ];
                    run(lk, &mut args, &[]).unwrap();
                }
                for (i, (a, b)) in out[0].iter().zip(out[1].iter()).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-5f32.max(1e-5 * a.abs()),
                        "{lanes}×{items}, n_col={n_col} [{i}] : {b} vs {a}"
                    );
                }
            }
        }
    }

    /// The tiled scan carries the rank-4 view too: its `load` and `store` are
    /// the kernel's own indexing, so a plane gap has to survive the tiling.
    #[test]
    fn the_tiled_scan_addresses_a_rank_4_view() {
        let kernel = super::build().unwrap();
        let lk = rir_lower::lower(&kernel, Schedule::metal_tiled_scan(8, 4)).unwrap();
        let (n_col, n_row, n_plane, n_batch, gap) = (37usize, 2usize, 3usize, 2usize, 3usize);

        let packed = |g: usize| {
            [
                4,
                4 * n_col,
                4 * n_col * n_row * g,
                4 * n_col * n_row * g * n_plane,
            ]
        };
        let mut seed = 0x7ca4_2026u64;
        let mut x = vec![0f32; n_col * n_row * n_plane * n_batch * gap];
        fill(&mut seed, &mut x, 1.0);

        let mut got = vec![0f32; n_col * n_row * n_plane * n_batch];
        let shape = [n_col, n_row, n_plane, n_batch];
        let mut args = [
            BoundArg::In(TensorView {
                data: &x,
                shape,
                nb: packed(gap),
            }),
            BoundArg::Out(TensorViewMut {
                data: &mut got,
                shape,
                nb: packed(1),
            }),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for b in 0..n_batch {
            for p in 0..n_plane {
                for r in 0..n_row {
                    let y_base = ((b * n_plane + p) * n_row + r) * n_col;
                    let x_base = ((b * n_plane * gap + p * gap) * n_row + r) * n_col;
                    let mut acc = 0f32;
                    for c in 0..n_col {
                        acc += x[x_base + c];
                        let g = got[y_base + c];
                        assert!(
                            (g - acc).abs() <= 1e-5f32.max(1e-5 * acc.abs()),
                            "[{b},{p},{r},{c}] : {g} vs {acc}"
                        );
                    }
                }
            }
        }
    }

    /// The shape class the outer axes exist for: `x` is a view inside a packed
    /// tensor, so its planes sit `gap` times further apart than the rows they
    /// contain, while `y` is densely packed. Only a distinct `nb[2]`/`nb[3]`
    /// per argument can address that; a row-folded scan would read the wrong
    /// plane.
    #[test]
    fn parity_on_a_rank_4_view_with_a_plane_gap() {
        let kernel = super::build().unwrap();
        for schedule in [Schedule::cpu_serial(), Schedule::vulkan_grid([64, 1, 1])] {
            let lk = rir_lower::lower(&kernel, schedule.clone()).unwrap();
            let (n_col, n_row, n_plane, n_batch, gap) = (17usize, 5usize, 3usize, 2usize, 3usize);

            let packed = |g: usize| {
                [
                    4,
                    4 * n_col,
                    4 * n_col * n_row * g,
                    4 * n_col * n_row * g * n_plane,
                ]
            };
            let len = |g: usize| n_col * n_row * n_plane * n_batch * g;

            let mut seed = 0x5ca4_2026u64;
            let mut x = vec![0f32; len(gap)];
            fill(&mut seed, &mut x, 1.0);

            let mut got = vec![0f32; len(1)];
            let shape = [n_col, n_row, n_plane, n_batch];
            let mut args = [
                BoundArg::In(TensorView {
                    data: &x,
                    shape,
                    nb: packed(gap),
                }),
                BoundArg::Out(TensorViewMut {
                    data: &mut got,
                    shape,
                    nb: packed(1),
                }),
            ];
            run(&lk, &mut args, &[]).unwrap();

            for b in 0..n_batch {
                for p in 0..n_plane {
                    for r in 0..n_row {
                        let y_base = ((b * n_plane + p) * n_row + r) * n_col;
                        let x_base = ((b * n_plane * gap + p * gap) * n_row + r) * n_col;
                        let mut acc = 0f32;
                        for c in 0..n_col {
                            acc += x[x_base + c];
                            let g = got[y_base + c];
                            assert!(
                                (g - acc).abs() <= 1e-5f32.max(1e-5 * acc.abs()),
                                "{:?} [{b},{p},{r},{c}] : {g} vs {acc}",
                                schedule.backend()
                            );
                        }
                    }
                }
            }
        }
    }

    /// The rank-2 form of the same scan. `derive_backward` transposes a kernel
    /// with exactly one parallel axis and one inner axis, so the production
    /// kernel's three outer axes put it out of the transposer's domain. The
    /// transposition rule being tested here is about the *scan*, not about how
    /// many outer axes ride along, so the test states the two-axis form
    /// explicitly rather than pretending the rank-4 kernel is differentiable.
    fn build_rank2() -> rir_core::ValidatedKernel {
        let mut k = KernelBuilder::new("cumsum_2d");
        let x = k.input("x", TensorType::f32_2d());
        let y = k.output("y", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
        let xv = k.read(x, &[col, row]);
        let s = k.scan(ScanOp::Sum, col, ScanDirection::Forward, xv);
        k.write(y, &[col, row], s);
        k.constrain(Constraint::DType {
            arg: x,
            allowed: vec![DType::F32],
        });
        k.constrain(Constraint::Rank { arg: x, max: 2 });
        k.finish().expect("cumsum_2d: invalid kernel")
    }

    /// The transpose of a forward scan is a backward scan:
    /// `dx[j] = Σ_{i≥j} dy[i]`, a suffix sum.
    #[test]
    fn the_derived_gradient_is_the_backward_scan() {
        let fwd = build_rank2();
        let grad = derive_backward(&fwd).expect("autodiff cumsum");
        let lk = rir_lower::lower(&grad, Schedule::cpu_serial()).unwrap();

        let (n_col, n_row) = (11usize, 3usize);
        let len = n_col * n_row;
        let mut seed = 0x5ca_2026u64;
        let mut x = vec![0f32; len];
        let mut dy = vec![0f32; len];
        fill(&mut seed, &mut x, 1.0);
        fill(&mut seed, &mut dy, 1.0);

        let mut dx = vec![0f32; len];
        let mut args = [
            BoundArg::In(TensorView::contiguous_2d(&x, n_col, n_row)),
            BoundArg::In(TensorView::contiguous_2d(&dy, n_col, n_row)),
            BoundArg::Out(TensorViewMut::contiguous_2d(&mut dx, n_col, n_row)),
        ];
        run(&lk, &mut args, &[]).unwrap();

        for r in 0..n_row {
            let mut suffix = 0f32;
            for c in (0..n_col).rev() {
                suffix += dy[r * n_col + c];
                let g = dx[r * n_col + c];
                assert!(
                    (g - suffix).abs() <= 1e-5f32.max(1e-5 * suffix.abs()),
                    "[{r},{c}] : {g} vs {suffix}"
                );
            }
        }
    }

    // The generated CPU scan loop compiles and produces the expected result.
    #[allow(dead_code)]
    mod generated {
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../generated/rir/cumsum/cpu.rs"
        ));
    }

    #[test]
    fn parity_of_the_generated_cpu_against_the_reference() {
        let (n_col, n_row) = (19usize, 2usize);
        let mut seed = 0x9e4_2026u64;
        let mut x = vec![0f32; n_col * n_row];
        fill(&mut seed, &mut x, 1.0);

        let mut got = vec![0f32; n_col * n_row];
        let nb = [4usize, 4 * n_col, 4 * n_col * n_row, 4 * n_col * n_row];
        generated::cumsum(
            n_row,
            1,
            1,
            n_col,
            generated::TensorRef { data: &x, nb },
            generated::TensorRefMut { data: &mut got, nb },
        );

        for r in 0..n_row {
            let mut acc = 0f32;
            for c in 0..n_col {
                acc += x[r * n_col + c];
                assert!((got[r * n_col + c] - acc).abs() <= 1e-5, "[{r},{c}]");
            }
        }
    }
}
