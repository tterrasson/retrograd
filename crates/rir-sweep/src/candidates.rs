//! The candidate product: which schedules a family's sweep puts under
//! arbitration, and on which shapes.
//!
//! **Small on purpose**. The point of an offline search is
//! not to explore a space - the space of `Schedule` is finite and tiny, and
//! walking it exhaustively would produce a table nobody can read - it is to put
//! the two or three geometries a human would otherwise try by hand opposite the
//! lowering that ships, on the shapes the plan already argues about. Each arm
//! below is one such question, and the arm names the item that asked it.
//!
//! Every candidate is **validated by lowering it**: `lower` is what knows that a
//! hierarchical tree needs a whole number of subgroups, that a linear address
//! needs bindings of one shape, that a scan and a reduction do not coexist. A
//! product filtered by anything else here would be a second opinion about the
//! compiler's own rules (`ScheduleError`, `LowerError`).

use rir_core::ValidatedKernel;
use rir_lower::schedule::bench;
use rir_lower::{Family, GpuBackend, Schedule, ShapeRule};

use crate::SweepError;

/// Where a candidate comes from, which is what makes a verdict readable: a row
/// the table already carries is a **control**, and one that beats the fallback
/// is news only if it is not already published.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Origin {
    /// The pair's current fallback on this backend - what a candidate must beat.
    Fallback,
    /// A variant `schedules_for` already carries, with its shape rule.
    Table,
    /// A geometry under arbitration: not in the table on this backend.
    Proposed,
}

/// One lowering under arbitration.
pub struct Candidate {
    /// Short name printed in every row of the report.
    pub tag: String,
    pub schedule: Schedule,
    pub origin: Origin,
    /// The constructor call that builds this schedule, as it would be written in
    /// `schedules_for`. A proposal a reader has to translate back into source is
    /// a proposal nobody commits.
    pub source: String,
}

impl Candidate {
    fn new(tag: impl Into<String>, source: impl Into<String>, schedule: Schedule) -> Self {
        Candidate {
            tag: tag.into(),
            schedule,
            origin: Origin::Proposed,
            source: source.into(),
        }
    }

    fn from_table(tag: impl Into<String>, source: impl Into<String>, schedule: Schedule) -> Self {
        Candidate {
            // The fallback is the row whose `variant` is `None` - the one that
            // accepts every shape the contract accepts - and not the row that
            // claims nothing: a vectorized fallback claims a *layout*
            // (`Schedule::claims`) while still being the pair's fallback, which
            // is exactly the shape `Family::Unary` has.
            origin: if schedule.variant().is_none() {
                Origin::Fallback
            } else {
                Origin::Table
            },
            ..Candidate::new(tag, source, schedule)
        }
    }
}

/// A kernel to sweep, resolved against the registry.
pub struct Subject {
    pub name: String,
    pub kernel: ValidatedKernel,
    pub family: Family,
    /// The schedules `schedules_for` gives this family, unchanged.
    pub table: Vec<Schedule>,
}

impl Subject {
    /// Resolves a kernel name against `rir_kernels::registry()` - the same list
    /// the AOT pipeline generates from, so a name that sweeps is a name that
    /// ships.
    pub fn resolve(name: &str) -> Result<Subject, SweepError> {
        let reg = rir_kernels::registry()
            .into_iter()
            .find(|r| r.kernel.name() == name)
            .ok_or_else(|| SweepError::UnknownKernel(name.to_string()))?;
        Ok(Subject {
            name: name.to_string(),
            kernel: reg.kernel,
            family: reg.family,
            table: reg.schedules,
        })
    }

    /// Every kernel name the registry carries, for `--list`.
    pub fn names() -> Vec<String> {
        rir_kernels::registry()
            .into_iter()
            .map(|r| r.kernel.name().to_string())
            .collect()
    }
}

/// The shape vocabulary of the sweep: the four ggml axes, by the names every
/// kernel in scope gives them.
///
/// A kernel whose axes are named otherwise is refused rather than bound by
/// position: `out_prod` has `i`, `j` and `k`, and mapping `col` onto `i`
/// because both come first would be a shape sweep of something else.
pub const AXES: [&str; 4] = ["col", "row", "plane", "batch"];

/// Axis extents in the kernel's own axis order, from a `[col, row, plane,
/// batch]` shape.
pub fn extents(lk: &rir_lower::LoopKernel, shape: [usize; 4]) -> Result<Vec<usize>, SweepError> {
    lk.axes
        .iter()
        .map(|a| {
            AXES.iter()
                .position(|n| *n == a.name)
                .map(|i| shape[i])
                .ok_or_else(|| SweepError::Unbindable {
                    kernel: lk.name.clone(),
                    why: format!(
                        "axis '{}' is outside the sweep's vocabulary {AXES:?}",
                        a.name
                    ),
                })
        })
        .collect()
}

/// The shapes a family is arbitrated on, `[col, row, plane, batch]`.
///
/// These are not "representative shapes" in the abstract: each of them appears
/// in a table already argued about elsewhere, so a
/// sweep's rows can be read against a number that was published. What decides a
/// rule is the *ranking across* them, which is why the list mixes row counts and
/// row lengths rather than sweeping one axis.
pub fn shapes(family: Family) -> Vec<[usize; 4]> {
    match family {
        // The four reference shapes - the two `shared_reduce` claims and the two it does
        // not - plus a short row, where a wide workgroup has nothing to reduce.
        Family::RowReduce | Family::RmsNorm => vec![
            [1024, 16, 1, 1],
            [640, 16, 1, 1],
            [256, 8, 16, 1],
            [128, 16, 16, 1],
            [128, 1, 1, 1],
        ],
        // The width sweep's three shapes, plus one whose contiguous
        // extent is no whole number of vectors, which is the layout a vectorized
        // variant has to decline.
        Family::Unary | Family::Elementwise | Family::ElementwiseRepeat | Family::Scale => vec![
            [4096, 16, 1, 1],
            [1024, 16, 1, 1],
            [128, 16, 16, 1],
            [33, 256, 1, 1],
        ],
        // The crossover: row length against row count.
        Family::Cumsum => vec![
            [4096, 1, 1, 1],
            [32768, 1, 1, 1],
            [2048, 320, 1, 1],
            [20000, 40, 1, 1],
        ],
        Family::MatMulNaive | Family::OutProd => vec![[512, 512, 1, 1], [64, 64, 16, 1]],
    }
}

/// The candidate product for `family` on `gpu`: the table's own lowerings
/// first - the fallback is the baseline every gain is measured against - then
/// the geometries under arbitration.
///
/// Invalid combinations are **not** filtered here: `plan` lowers each candidate
/// and drops the ones the compiler refuses, with the refusal printed. A product
/// pruned by a second copy of the rules would drift from them.
pub fn candidates(subject: &Subject, gpu: GpuBackend) -> Vec<Candidate> {
    let family = subject.family;
    let backend = gpu.backend();
    let gpu_name = match gpu {
        GpuBackend::Vulkan => "GpuBackend::Vulkan",
        GpuBackend::Metal => "GpuBackend::Metal",
        GpuBackend::Cuda => "GpuBackend::Cuda",
    };

    let mut v: Vec<Candidate> = subject
        .table
        .iter()
        .filter(|s| s.backend() == backend)
        .map(|s| {
            let tag = match s.variant() {
                Some(name) => format!("table/{name}"),
                None => "table/fallback".to_string(),
            };
            Candidate::from_table(tag, "already in schedules_for", s.clone())
        })
        .collect();

    match family {
        // The address, the width and the workgroup, which is the
        // product a hand sweep covers: a flattened dispatch against the grid, a
        // linear address against the decomposition, three widths and three
        // blocks. On a backend outside `FLAT_TARGETS` the flat half is the
        // question rather than the control - the mechanism is backend-neutral
        // and only the *measurement* is CUDA's (`FLAT_TARGETS`).
        Family::Unary | Family::Elementwise | Family::ElementwiseRepeat | Family::Scale => {
            for block in [64u32, 128, 256, 512] {
                for width in [1u32, 4] {
                    v.push(Candidate::new(
                        format!("grid b{block} w{width}"),
                        format!("Schedule::gpu_grid_vec4({gpu_name}, [{block}, 1, 1])"),
                        if width == 1 {
                            Schedule::gpu_grid(gpu, [block, 1, 1])
                        } else {
                            Schedule::gpu_grid_vec4(gpu, [block, 1, 1])
                        },
                    ));
                    v.push(Candidate::new(
                        format!("flat b{block} w{width}"),
                        format!("Schedule::gpu_grid_flat({gpu_name}, [{block}, 1, 1], {width})"),
                        Schedule::gpu_grid_flat(gpu, [block, 1, 1], width),
                    ));
                    v.push(Candidate::new(
                        format!("linear b{block} w{width}"),
                        format!(
                            "Schedule::gpu_grid_flat_linear({gpu_name}, [{block}, 1, 1], {width})"
                        ),
                        Schedule::gpu_grid_flat_linear(gpu, [block, 1, 1], width),
                    ));
                }
            }
        }
        // The reduction topologies, at the widths that separate them:
        // the 32-lane subgroup, the flat tree, the two-stage tree, and the flat
        // row space. The 1 024-lane tree is in the product although a
        // measurement already saw it lose - a sweep that dropped the geometry a table already refused
        // could not reproduce the refusal.
        Family::RowReduce | Family::RmsNorm => {
            v.push(Candidate::new(
                "subgroup 32",
                format!("Schedule::gpu_subgroup({gpu_name})"),
                Schedule::gpu_subgroup(gpu),
            ));
            v.push(Candidate::new(
                "flat_rows 32",
                format!("Schedule::gpu_subgroup_flat_rows({gpu_name})"),
                Schedule::gpu_subgroup_flat_rows(gpu),
            ));
            for lanes in [128u32, 256, 512, 1024] {
                v.push(Candidate::new(
                    format!("shared {lanes}"),
                    format!("Schedule::gpu_shared_reduce({gpu_name}).with_block([{lanes}, 1, 1])"),
                    Schedule::gpu_shared_reduce(gpu).with_block([lanes, 1, 1]),
                ));
                v.push(Candidate::new(
                    format!("hier {lanes}"),
                    format!("Schedule::gpu_hier_reduce({gpu_name}).with_block([{lanes}, 1, 1])"),
                    Schedule::gpu_hier_reduce(gpu).with_block([lanes, 1, 1]),
                ));
            }
        }
        // The scan, with its benchmark predecessor opposite it: the pair
        // `schedule::bench` exists for (ADR-2 section 5).
        Family::Cumsum => {
            v.push(Candidate::new(
                "shared 256",
                format!("bench::gpu_shared_scan({gpu_name})"),
                bench::gpu_shared_scan(gpu),
            ));
            for items in [8u32, 16, 32] {
                v.push(Candidate::new(
                    format!("tiled 256×{items}"),
                    format!("Schedule::gpu_tiled_scan({gpu_name}, 256, {items})"),
                    Schedule::gpu_tiled_scan(gpu, 256, items),
                ));
            }
        }
        // The search stops here: do not extend to `TiledStage` (double
        // buffering, depth, and register tiling) unless `OUT_PROD` returns to
        // the top of the profile. The product is therefore empty
        // rather than absent - the sweep runs, prints the table's own lowering,
        // and says why it has nothing to put opposite it.
        Family::MatMulNaive | Family::OutProd => {}
    }
    v
}

/// The axis groups a shape rule may be written over, in the order the sweep
/// tries them.
///
/// It is the registry's vocabulary and not a superset of it: a published rule is
/// the product of some axes' extents inside `[min, max]`, and the two groupings
/// below are the two the table uses - the row count (`row · plane · batch`,
/// what occupies the device) and the row length (`col`, what a workgroup
/// covers). A domain that needs a third grouping is a domain this tool declines
/// to invent (`verdict::Refusal::NoDistinctDomain`).
pub const RULE_AXES: [&[&str]; 2] = [&["row", "plane", "batch"], &["col"]];

/// The product of a rule's axes on a shape, which is what a `ShapeRule`
/// compares against `[min, max]`.
pub fn rule_product(axes: &[&str], shape: [usize; 4]) -> u64 {
    axes.iter()
        .filter_map(|a| AXES.iter().position(|n| n == a))
        .map(|i| shape[i] as u64)
        // The production dispatcher saturates a rule product at u32::MAX
        // before comparing it to the u32 interval. The sweep must use the same
        // arithmetic or it can derive a different domain on a large shape.
        .fold(1, |product, factor| {
            product.saturating_mul(factor).min(u64::from(u32::MAX))
        })
}

/// A rule as `schedules_for` would write it.
pub fn rule_source(rule: &ShapeRule) -> String {
    let axes = rule
        .axes
        .iter()
        .map(|a| format!("\"{a}\""))
        .collect::<Vec<_>>()
        .join(", ");
    if rule.min == 1 {
        format!("ShapeRule::at_most(&[{axes}], {})", rule.max)
    } else if rule.max == u32::MAX {
        format!(
            "ShapeRule {{ axes: &[{axes}], min: {}, max: u32::MAX }}",
            rule.min
        )
    } else {
        format!(
            "ShapeRule {{ axes: &[{axes}], min: {}, max: {} }}",
            rule.min, rule.max
        )
    }
}
