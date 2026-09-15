use super::*;
use rir_core::{Extent, KernelBuilder, TensorType};

/// A kernel with two axes, enough to write rules against.
fn kernel() -> rir_core::ValidatedKernel {
    let mut k = KernelBuilder::new("t");
    let x = k.input("x", TensorType::f32_2d());
    let y = k.output("y", TensorType::f32_2d());
    let row = k.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = k.axis("col", Extent::Dim { arg: x, dim: 0 });
    let v = k.read(x, &[col, row]);
    k.write(y, &[col, row], v);
    k.finish().expect("test kernel")
}

fn fallback() -> Schedule {
    Schedule::vulkan_grid([64, 1, 1])
}

fn special(name: &'static str, priority: u8, rules: Vec<ShapeRule>) -> Schedule {
    Schedule::vulkan_grid([64, 1, 1])
        .as_variant(name)
        .claiming(priority, rules)
}

/// `GPU_TARGETS` and `GPU_REFUSED` partition `GpuBackend::ALL`: every GPU
/// backend is either scheduled or refused **in writing**, and never both.
///
/// This is what makes the absence of a `push` stop meaning anything.
/// Adding a `GpuBackend` variant fails here
/// until somebody decides which of the two lists it belongs to, so the
/// decision cannot be made by forgetting.
#[test]
fn gpu_coverage_is_a_partition() {
    for gpu in GpuBackend::ALL {
        let scheduled = GPU_TARGETS.contains(&gpu);
        let refused = GPU_REFUSED.iter().any(|(g, _)| *g == gpu);
        assert!(
            scheduled != refused,
            "{}: scheduled={scheduled}, refused={refused} - a GPU backend is \
             exactly one of the two",
            gpu.name()
        );
    }
    for (_, why) in GPU_REFUSED {
        assert!(!why.trim().is_empty(), "a refusal states its reason");
    }
}

/// Every family reaches the GPU.
///
/// The closed family table makes this property local: a kernel that matches no
/// arm cannot silently keep only its CPU schedule. Every arm schedules every
/// backend it has not refused **in writing**, so:
/// every arm schedules every backend it has not refused **in writing**, so
/// "CPU alone" is not a value `schedules_for` returns by omission.
#[test]
fn every_family_reaches_every_gpu_target() {
    for family in Family::ALL {
        let schedules = schedules_for(family);
        for gpu in targets_for(family) {
            assert!(
                schedules.iter().any(|s| s.backend() == gpu.backend()),
                "{}: no schedule on {}",
                family.name(),
                gpu.name()
            );
        }
        // Both refusals, the table's and the family's, are refusals: nothing
        // is emitted for a backend either of them names.
        let refused = GPU_REFUSED.iter().map(|(g, _)| *g).chain(
            GPU_TARGETS
                .iter()
                .copied()
                .filter(|g| family_refusal(family, *g).is_some()),
        );
        for gpu in refused {
            assert!(
                !schedules.iter().any(|s| s.backend() == gpu.backend()),
                "{}: scheduled on {}, which is refused",
                family.name(),
                gpu.name()
            );
        }
    }
}

/// A family's refusal is a statement, and a statement has a reason and a
/// subject that exists. Refusing a backend the whole table already refuses
/// would be a second voice on one decision - `GPU_REFUSED` is where that is
/// said.
#[test]
fn a_family_refusal_names_a_scheduled_backend_and_its_reason() {
    for (family, gpu, why) in FAMILY_REFUSED {
        assert!(
            GPU_TARGETS.contains(gpu),
            "{}: refuses {}, which the whole table already refuses",
            family.name(),
            gpu.name()
        );
        assert!(!why.trim().is_empty(), "a refusal states its reason");
    }
}

/// Every schedule the production table emits is well-shaped. The point is
/// not that the fourteen constructors are correct - it is that
/// `check_shape` is what says so, on the real table, rather than a
/// `max(1)` scattered over the lowering and the emitters.
#[test]
fn the_production_table_is_well_shaped() {
    for s in Family::ALL.iter().flat_map(|f| schedules_for(*f)) {
        assert_eq!(
            s.check_shape(),
            Ok(()),
            "{} {:?}",
            s.backend().name(),
            s.variant()
        );
        assert!(s.grid_dims() <= 3, "grid_dims never indexes past block[2]");
        assert!(s.block().iter().all(|&b| b >= 1));
    }
}

/// The four numeric arguments a constructor cannot vouch for, each rejected
/// as a `ScheduleError` and not as a panic three layers down.
#[test]
fn a_degenerate_numeric_argument_is_an_error() {
    let cases: Vec<(&str, Schedule)> = vec![
        ("zero block", Schedule::vulkan_grid([0, 1, 1])),
        ("zero block in y", Schedule::vulkan_grid([64, 0, 1])),
        (
            "zero tile depth under TiledStage",
            Schedule::vulkan_grid_tiled([16, 16, 1], 0, 4),
        ),
        (
            "zero register width",
            Schedule::vulkan_grid_tiled([16, 16, 1], 16, 0),
        ),
        ("zero items per lane", Schedule::vulkan_tiled_scan(256, 0)),
        // Both non-zero, both powers of two - so the lane-count guard of the
        // shared tree lets this one through - and their product is not a
        // `u32`. The lowering would compute a tile width of zero and emit a
        // `ForTiled` that steps by nothing.
        (
            "tile width overflows 32 bits",
            Schedule::vulkan_tiled_scan(65_536, 65_536),
        ),
    ];
    for (why, s) in cases {
        assert!(
            matches!(s.check_shape(), Err(ScheduleError::Malformed { .. })),
            "'{why}' should have been rejected at construction"
        );
        // `lower` refuses it before reading a field, so no caller can reach an
        // invalid index.
        assert!(matches!(
            check_schedule(&kernel(), &s),
            Err(ScheduleError::Malformed { .. })
        ));
    }
}

/// The table the pipeline ships: one fallback, one specialization above it.
#[test]
fn a_fallback_plus_one_specialization_is_well_formed() {
    let t = [
        fallback(),
        special("blocked", 90, vec![ShapeRule::at_most(&["row"], 128)]),
    ];
    assert_eq!(check_schedule_table(&kernel(), &t), Ok(()));
}

/// A variant may claim a **layout** instead of a shape. `vec4` carries no
/// extent interval - its condition is `nb[0] == elem_bytes`, which no
/// `ShapeRule` can express - and the table must still see it as a
/// specialization and not as a second fallback.
#[test]
fn a_vector_width_is_a_claim_like_a_shape_rule() {
    let vec4 = Schedule::vulkan_grid_vec4([64, 1, 1]).claiming(90, Vec::new());
    assert!(vec4.claims());
    assert!(!fallback().claims());
    assert_eq!(check_schedule_table(&kernel(), &[fallback(), vec4]), Ok(()));

    // The same lowering left scalar claims nothing, so naming it would
    // shadow the fallback everywhere: still refused.
    let named_but_empty = Schedule::vulkan_grid([64, 1, 1])
        .as_variant("vec4")
        .claiming(90, Vec::new());
    assert!(check_schedule_table(&kernel(), &[fallback(), named_but_empty]).is_err());
}

/// Every way a table can make selection partial, ambiguous, or carry a
/// variant no shape reaches. Each of these once passed generation and would
/// only have shown up as a lowering that silently never ran.
#[test]
fn a_malformed_variant_table_is_a_generation_error() {
    let k = kernel();
    let cases: Vec<(&str, Vec<Schedule>)> = vec![
        (
            "no fallback",
            vec![special("a", 90, vec![ShapeRule::at_most(&["row"], 8)])],
        ),
        ("two fallbacks", vec![fallback(), fallback()]),
        (
            "colliding identities",
            vec![
                fallback(),
                special("a", 90, vec![ShapeRule::at_most(&["row"], 8)]),
                special("a", 91, vec![ShapeRule::at_most(&["col"], 8)]),
            ],
        ),
        (
            "specialization below fallback",
            vec![
                fallback(),
                special("a", 10, vec![ShapeRule::at_most(&["row"], 8)]),
            ],
        ),
        (
            "tie between specializations",
            vec![
                fallback(),
                special("a", 90, vec![ShapeRule::at_most(&["row"], 8)]),
                special("b", 90, vec![ShapeRule::at_most(&["col"], 8)]),
            ],
        ),
        (
            "axis-free rule",
            vec![
                fallback(),
                special(
                    "a",
                    90,
                    vec![ShapeRule {
                        axes: &[],
                        min: 1,
                        max: 8,
                    }],
                ),
            ],
        ),
        (
            "empty interval",
            vec![
                fallback(),
                special(
                    "a",
                    90,
                    vec![ShapeRule {
                        axes: &["row"],
                        min: 9,
                        max: 8,
                    }],
                ),
            ],
        ),
        (
            "unknown axis",
            vec![
                fallback(),
                special("a", 90, vec![ShapeRule::at_most(&["plane"], 8)]),
            ],
        ),
    ];
    for (why, table) in cases {
        assert!(
            check_schedule_table(&k, &table).is_err(),
            "'{why}' should have been rejected at generation"
        );
    }
}

/// Backends are validated independently: a table well-formed on Vulkan and
/// broken on Metal must not pass because the Vulkan half is fine.
#[test]
fn each_backend_is_checked_on_its_own() {
    let t = [
        fallback(),
        special("blocked", 90, vec![ShapeRule::at_most(&["row"], 128)]),
        // Metal: a specialization and no fallback.
        Schedule::metal_grid([64, 1, 1])
            .as_variant("blocked")
            .claiming(90, vec![ShapeRule::at_most(&["row"], 128)]),
    ];
    assert!(matches!(
        check_schedule_table(&kernel(), &t),
        Err(ScheduleError::NoFallback {
            backend: Backend::Metal
        })
    ));
}
