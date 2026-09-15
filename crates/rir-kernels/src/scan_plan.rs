//! The decoupled scan: `CUMSUM` served by three dispatches instead of one.
//!
//! **The problem this exists for, stated as a shape.** A scan of one very long
//! row cannot occupy a GPU with one workgroup: the fallback gives it one
//! invocation, the blocked scan thirty-two lanes, the tiled scan two hundred
//! and fifty-six - and past that there is nothing left to add, because a scan is
//! sequential and a workgroup is the largest thing whose lanes can communicate.
//! The design therefore uses three passes instead of a fourth scan variant.
//!
//! **What it is instead.** The row is cut into tiles of [`TILE`] elements, and
//! three dispatches recombine them:
//!
//! ```text
//!   pass 0  part[t]   = Σ_{i<TILE} x[i, t]                one workgroup per tile
//!   pass 1  offs      = inclusive scan of part            one dispatch, T elements
//!   pass 2  y[i, t]   = scan_{j≤i} x[j, t] + offs[t] − part[t]
//! ```
//!
//! `offs[t] − part[t]` is the **exclusive** prefix of the tile totals, and it is
//! computed that way rather than by an exclusive scan because the DSL's scan is
//! inclusive: subtracting a tile's own total from its inclusive prefix is exact
//! in the same arithmetic, and it needs no second scan operator in the IR.
//!
//! **Not one kernel that dispatches three times.** Each pass is an ordinary
//! kernel of this crate, lowered by the ordinary lowering and printed by the
//! ordinary emitters; pass 1 *is* `cumsum`, unchanged, on a T-element row. What
//! is new is the [`rir_core::plan`] artifact that says in which order they run,
//! through which buffers, with what barriers, and what the scratch costs - which
//! is exactly the list a multi-pass design has to publish.
//!
//! **The view is the trick, and it costs nothing.** No pass reshapes anything:
//! a row of `n_col` elements is read by passes 0 and 2 as `n_col / TILE` rows of
//! `TILE`, through strides `[1, TILE]`. A stride list is what a view is, and the
//! plan carries one per binding.
//!
//! **Out of the production table, on purpose.** A multi-pass plan is promoted
//! only for a measured client - a client representing at least 3% of GPU time, or a second measured
//! op requiring the same capability - and `CUMSUM` has zero nodes in both
//! censused graphs. So no arm of `schedules_for` returns these schedules, no
//! `KernelId` names these kernels, and the AOT pipeline emits none of them:
//! this is the capability, built and measurable, with its policy left untouched.

use rir_core::manifest::DomainRestriction;
use rir_core::plan::{
    Barrier, Binding, DispatchPlan, Extent, PLAN_SCHEMA_VERSION, Pass, Scratch, Source,
};
use rir_core::{
    Backend, Constraint, DType, ReduceOp, ReductionSemantics, ScanDirection, ScanOp, TensorType,
    ValidateError, ValidatedKernel,
};
use rir_core::{Extent as AxisExtent, KernelBuilder};
use rir_lower::{GpuBackend, Schedule};

/// Elements one tile covers.
///
/// 256 and not a swept number: it is the width at which one workgroup's lanes
/// already scan a tile without shared storage in pass 2, and the point of this
/// plan is the *number of tiles*, not their width. Sweeping the width is
/// `rir-sweep`'s job, and needs a client first.
pub const TILE: u32 = 256;

/// Pass 0 - one tile total per tile.
///
/// The reduction is `Deterministic` rather than `ExactOrder`: a subgroup tree
/// regroups the additions inside a tile, which is what `l2_norm_back` and
/// `rms_norm_back` already publish. The plan as a whole is therefore
/// deterministic and not term-for-term identical to a sequential scan - stated
/// here because it is the one numerical difference between this and the single
/// dispatch it replaces.
pub fn tile_sum() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("scan_tile_sum");
    let x = k.input("x", TensorType::f32(2));
    let part = k.output("part", TensorType::f32(1));

    // Parallel axis first, reduced axis last, as every row-reduce kernel of this
    // crate declares them: `tile` takes the grid, `col` is traversed inside.
    let tile = k.axis("tile", AxisExtent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", AxisExtent::Dim { arg: x, dim: 0 });

    let xv = k.read(x, &[col, tile]);
    let total = k.reduce(ReduceOp::Sum, col, xv, ReductionSemantics::Deterministic);
    k.write(part, &[tile], total);

    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::F32],
    });
    k.finish()
}

/// Pass 2 - the scan inside each tile, offset by the exclusive prefix of the
/// tile totals.
pub fn tile_apply() -> Result<ValidatedKernel, ValidateError> {
    let mut k = KernelBuilder::new("scan_tile_apply");
    let x = k.input("x", TensorType::f32(2));
    let offs = k.input("offs", TensorType::f32(1));
    let part = k.input("part", TensorType::f32(1));
    let y = k.output("y", TensorType::f32(2));

    let tile = k.axis("tile", AxisExtent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", AxisExtent::Dim { arg: x, dim: 0 });

    let xv = k.read(x, &[col, tile]);
    let inside = k.scan(ScanOp::Sum, col, ScanDirection::Forward, xv);
    let inclusive = k.read(offs, &[tile]);
    let own = k.read(part, &[tile]);
    // The exclusive prefix of the tile totals. Two reads and a subtraction
    // instead of a second scan operator in the IR.
    let before = k.sub(inclusive, own);
    let out = k.add(inside, before);
    k.write(y, &[col, tile], out);

    k.constrain(Constraint::DType {
        arg: x,
        allowed: vec![DType::F32],
    });
    k.finish()
}

/// Pass 1 - `cumsum` itself, on a row of tile totals. Returned by this module so
/// the three passes are read in one place; it is the registry's kernel, byte for
/// byte.
pub fn partials() -> Result<ValidatedKernel, ValidateError> {
    crate::cumsum::build()
}

/// The three kernels, in pass order.
pub fn kernels() -> Result<[ValidatedKernel; 3], ValidateError> {
    Ok([tile_sum()?, partials()?, tile_apply()?])
}

/// The schedule each pass is lowered with, in pass order.
///
/// One workgroup per tile for the two tile passes - the parallelism this plan
/// buys is the tile count, and a wider workgroup would spend lanes on a 256-long
/// row that already fits an invocation's serial loop. Pass 1 keeps the scan
/// fallback: it is one row of `n_col / 256` elements, and cutting *it* further
/// would be the same problem one level down, which is where a plan of four
/// passes would start.
pub fn schedules(gpu: GpuBackend) -> [Schedule; 3] {
    [
        Schedule::gpu_subgroup(gpu),
        Schedule::gpu_grid(gpu, [64, 1, 1]),
        Schedule::gpu_grid(gpu, [64, 1, 1]),
    ]
}

/// The plan artifact: ordered dispatches, scratch shape and lifetime, barriers,
/// and - through `peak_scratch_bytes` - the budget.
///
/// `artifacts` are the generated names of the three passes in order, which the
/// caller has because it is the caller that emitted them.
pub fn plan(backend: Backend, artifacts: [String; 3]) -> DispatchPlan {
    let tiles = || Extent::axis_tiles("col", TILE);
    let tile_width = || Extent::constant(TILE);
    let [pass0, pass1, pass2] = artifacts;

    DispatchPlan {
        plan_schema_version: PLAN_SCHEMA_VERSION,
        name: "cumsum_decoupled".to_string(),
        ggml_op: Some("GGML_OP_CUMSUM".to_string()),
        backend,
        assumed_domain: vec![
            DomainRestriction {
                reject: "multi_row".to_string(),
                why: "one row: the plan tiles the contiguous axis and addresses tile `t` at \
                      `t · TILE` elements, which is the next row's data as soon as there is a \
                      next row. A multi-row version is the same plan with the row axis carried \
                      through every pass, and it is not written because the client O10 waits \
                      for is a single very long row"
                    .to_string(),
            },
            DomainRestriction {
                reject: "partial_tile".to_string(),
                why: "n_col must be a whole number of tiles: the last workgroup would otherwise \
                      read past the end of the row. A remainder is a fourth pass or a bounds \
                      check in two kernels, and neither is worth writing before a client is \
                      measured"
                    .to_string(),
            },
            DomainRestriction {
                reject: "non_contiguous".to_string(),
                why: "the plan publishes its strides as constants in elements, which is what \
                      makes a tiled view free; a strided source would need stride expressions \
                      over the node's own `nb[]`"
                    .to_string(),
            },
        ],
        scratch: vec![
            Scratch {
                name: "part".to_string(),
                dtype: DType::F32,
                elements: tiles(),
                produced_by: 0,
                last_read_by: 2,
            },
            Scratch {
                name: "offs".to_string(),
                dtype: DType::F32,
                elements: tiles(),
                produced_by: 1,
                last_read_by: 2,
            },
        ],
        passes: vec![
            Pass {
                artifact: pass0,
                axes: vec![
                    ("tile".to_string(), tiles()),
                    ("col".to_string(), tile_width()),
                ],
                bindings: vec![
                    Binding {
                        name: "x".to_string(),
                        source: Source::Arg("x".to_string()),
                        strides: vec![Extent::constant(1), tile_width()],
                    },
                    Binding {
                        name: "part".to_string(),
                        source: Source::Scratch("part".to_string()),
                        strides: vec![Extent::constant(1)],
                    },
                ],
                barrier: Barrier::None,
            },
            // `cumsum` over the tile totals: its own four axes, a row of `T`.
            Pass {
                artifact: pass1,
                axes: vec![
                    ("col".to_string(), tiles()),
                    ("row".to_string(), Extent::constant(1)),
                    ("plane".to_string(), Extent::constant(1)),
                    ("batch".to_string(), Extent::constant(1)),
                ],
                bindings: vec![
                    Binding {
                        name: "x".to_string(),
                        source: Source::Scratch("part".to_string()),
                        strides: vec![Extent::constant(1), tiles(), tiles(), tiles()],
                    },
                    Binding {
                        name: "y".to_string(),
                        source: Source::Scratch("offs".to_string()),
                        strides: vec![Extent::constant(1), tiles(), tiles(), tiles()],
                    },
                ],
                barrier: Barrier::Full,
            },
            Pass {
                artifact: pass2,
                axes: vec![
                    ("tile".to_string(), tiles()),
                    ("col".to_string(), tile_width()),
                ],
                bindings: vec![
                    Binding {
                        name: "x".to_string(),
                        source: Source::Arg("x".to_string()),
                        strides: vec![Extent::constant(1), tile_width()],
                    },
                    Binding {
                        name: "offs".to_string(),
                        source: Source::Scratch("offs".to_string()),
                        strides: vec![Extent::constant(1)],
                    },
                    Binding {
                        name: "part".to_string(),
                        source: Source::Scratch("part".to_string()),
                        strides: vec![Extent::constant(1)],
                    },
                    Binding {
                        name: "y".to_string(),
                        source: Source::Arg("y".to_string()),
                        strides: vec![Extent::constant(1), tile_width()],
                    },
                ],
                barrier: Barrier::Full,
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use rir_lower::Schedule;
    use rir_lower::interp::{BoundArg, TensorView, TensorViewMut, run};

    /// The three passes against the scan they replace, on the CPU oracle: no
    /// device, no plan runtime, just the arithmetic. If this is wrong, every
    /// number the device produces is wrong for a reason that has nothing to do
    /// with dispatching.
    #[test]
    fn the_three_passes_reconstruct_the_scan() {
        let tile = super::TILE as usize;
        let tiles = 5usize;
        let n = tile * tiles;
        let mut x = vec![0f32; n];
        for (i, v) in x.iter_mut().enumerate() {
            *v = ((i % 7) as f32 - 3.0) * 0.25;
        }

        let sum = rir_lower::lower(&super::tile_sum().unwrap(), Schedule::cpu_serial()).unwrap();
        let scan = rir_lower::lower(&super::partials().unwrap(), Schedule::cpu_serial()).unwrap();
        let apply =
            rir_lower::lower(&super::tile_apply().unwrap(), Schedule::cpu_serial()).unwrap();

        let mut part = vec![0f32; tiles];
        run(
            &sum,
            &mut [
                BoundArg::In(TensorView {
                    data: &x,
                    shape: [tile, tiles, 1, 1],
                    nb: [4, 4 * tile, 4 * n, 4 * n],
                }),
                BoundArg::Out(TensorViewMut {
                    data: &mut part,
                    shape: [tiles, 1, 1, 1],
                    nb: [4, 4 * tiles, 4 * tiles, 4 * tiles],
                }),
            ],
            &[],
        )
        .expect("pass 0");

        let mut offs = vec![0f32; tiles];
        run(
            &scan,
            &mut [
                BoundArg::In(TensorView {
                    data: &part,
                    shape: [tiles, 1, 1, 1],
                    nb: [4, 4 * tiles, 4 * tiles, 4 * tiles],
                }),
                BoundArg::Out(TensorViewMut {
                    data: &mut offs,
                    shape: [tiles, 1, 1, 1],
                    nb: [4, 4 * tiles, 4 * tiles, 4 * tiles],
                }),
            ],
            &[],
        )
        .expect("pass 1");

        let mut y = vec![0f32; n];
        run(
            &apply,
            &mut [
                BoundArg::In(TensorView {
                    data: &x,
                    shape: [tile, tiles, 1, 1],
                    nb: [4, 4 * tile, 4 * n, 4 * n],
                }),
                BoundArg::In(TensorView {
                    data: &offs,
                    shape: [tiles, 1, 1, 1],
                    nb: [4, 4 * tiles, 4 * tiles, 4 * tiles],
                }),
                BoundArg::In(TensorView {
                    data: &part,
                    shape: [tiles, 1, 1, 1],
                    nb: [4, 4 * tiles, 4 * tiles, 4 * tiles],
                }),
                BoundArg::Out(TensorViewMut {
                    data: &mut y,
                    shape: [tile, tiles, 1, 1],
                    nb: [4, 4 * tile, 4 * n, 4 * n],
                }),
            ],
            &[],
        )
        .expect("pass 2");

        let mut acc = 0f32;
        for i in 0..n {
            acc += x[i];
            assert!(
                (y[i] - acc).abs() <= 1e-4 * (1.0 + acc.abs()),
                "element {i}: plan {} against the scan {acc}",
                y[i]
            );
        }
    }

    /// The plan's own structure, and the budget it publishes: two live scratch
    /// arrays of one element per tile, which is what a caller has to be able to
    /// compute before dispatching.
    #[test]
    fn the_plan_publishes_its_barriers_and_its_budget() {
        let plan = super::plan(
            rir_core::Backend::Vulkan,
            ["a".to_string(), "b".to_string(), "c".to_string()],
        );
        plan.check().expect("a well-formed plan");

        let n_col = 65_536u64;
        let axes = |name: &str| match name {
            "col" => Some(n_col),
            "row" | "plane" | "batch" => Some(1),
            _ => None,
        };
        let tiles = n_col / u64::from(super::TILE);
        assert_eq!(
            plan.peak_scratch_bytes(&axes).unwrap(),
            2 * tiles * 4,
            "both arrays are live during the last pass"
        );

        // The property the artifact exists for: a pass reading what a previous
        // pass wrote declares a barrier, and a plan where one does not is
        // refused rather than raced.
        let mut raced = plan.clone();
        raced.passes[2].barrier = rir_core::plan::Barrier::None;
        assert!(matches!(
            raced.check(),
            Err(rir_core::plan::PlanError::MissingBarrier { pass: 2, .. })
        ));

        let mut duplicate = plan.clone();
        duplicate.scratch.push(duplicate.scratch[0].clone());
        assert!(matches!(
            duplicate.check(),
            Err(rir_core::plan::PlanError::DuplicateScratch { .. })
        ));

        let mut reversed = plan.clone();
        reversed.scratch[0].produced_by = 2;
        reversed.scratch[0].last_read_by = 1;
        assert!(matches!(
            reversed.check(),
            Err(rir_core::plan::PlanError::InvalidLifetime { .. })
        ));

        let mut overflow = plan.clone();
        overflow.scratch[0].elements = rir_core::plan::Extent::constant(u32::MAX)
            .times(rir_core::plan::Extent::constant(u32::MAX))
            .times(rir_core::plan::Extent::constant(2));
        assert!(matches!(
            overflow.peak_scratch_bytes(&axes),
            Err(rir_core::plan::PlanError::BudgetOverflow { .. })
        ));
    }
}
