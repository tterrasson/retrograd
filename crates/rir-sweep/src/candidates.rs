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
use rir_core::catalog::{ModelShape, Regime};
use rir_lower::schedule::bench;
use rir_lower::schedule::spaces;
use rir_lower::{Claim, Family, Geometry, GpuBackend, Pick, Schedule};

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
    /// For a candidate drawn from a geometry space: the pick it would be
    /// committed as, with the space's defaults to derive the literal against.
    /// The pick's claim is a placeholder here - the verdict's claim is printed
    /// in its place.
    pub pick: Option<(Pick, Geometry)>,
}

impl Candidate {
    fn new(tag: impl Into<String>, source: impl Into<String>, schedule: Schedule) -> Self {
        Candidate {
            tag: tag.into(),
            schedule,
            origin: Origin::Proposed,
            source: source.into(),
            pick: None,
        }
    }

    fn from_space(
        tag: impl Into<String>,
        schedule: Schedule,
        pick: Pick,
        defaults: Geometry,
    ) -> Self {
        Candidate {
            tag: tag.into(),
            schedule,
            origin: Origin::Proposed,
            source: String::new(),
            pick: Some((pick, defaults)),
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
/// OUT_PROD uses i/j for the two output dimensions and a separate contraction
/// k. Its reduction depth is supplied separately from [col, row, plane, batch].
pub const AXES: [&str; 4] = ["col", "row", "plane", "batch"];

/// Axis extents in the kernel's own axis order, from a `[col, row, plane,
/// batch]` shape.
pub fn extents(lk: &rir_lower::LoopKernel, shape: [usize; 4]) -> Result<Vec<usize>, SweepError> {
    extents_with_contraction(lk, shape, 64)
}

/// Bind OUT_PROD's contraction independently from its four output dimensions.
pub fn extents_with_contraction(
    lk: &rir_lower::LoopKernel,
    shape: [usize; 4],
    contraction: usize,
) -> Result<Vec<usize>, SweepError> {
    lk.axes
        .iter()
        .map(|a| {
            if lk.name.starts_with("out_prod") {
                match a.name.as_str() {
                    "i" => return Ok(shape[0]),
                    "j" => return Ok(shape[1]),
                    "k" if contraction > 0 => return Ok(contraction),
                    _ => {}
                }
            }
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

/// The decode token counts of the GEMV grid: every step a decode
/// measurement takes, plus two, the smallest the native side serves with its
/// multi-column vector kernel.
pub const GEMV_TOKENS: [usize; 6] = [1, 2, 4, 8, 16, 32];

/// The reduction depths (`k`, the `col` axis) of the GEMV grid.
pub const GEMV_DEPTHS: [usize; 3] = [1024, 4096, 14336];

/// The output row counts (`m`, the `row` axis) of the GEMV grid.
pub const GEMV_ROWS: [usize; 3] = [1024, 4096, 14336];

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
        Family::MatMulNaive => vec![[512, 512, 1, 1], [64, 64, 16, 1]],
        // `[K, M, tokens, 1]`: the decode steps against three depths and three
        // row counts - the Qwen3 0.6B width, the 8B-class width and the
        // 8B-class FFN - so that a rule can be cut along the token count, the
        // depth or the row count, and a claim proposed for one of them is
        // seen on the others.
        Family::MulMatVec => {
            let mut grid = Vec::new();
            for n in GEMV_TOKENS {
                for k in GEMV_DEPTHS {
                    for m in GEMV_ROWS {
                        grid.push([k, m, n, 1]);
                    }
                }
            }
            grid
        }
        // Synthetic coverage; --shape and --contraction reproduce census shapes.
        Family::OutProd => vec![
            [8, 3, 2, 3],
            [33, 17, 2, 2],
            [512, 512, 1, 1],
            [64, 64, 16, 2],
        ],
    }
}

/// The shapes a family runs on in one model profile and one regime,
/// `[col, row, plane, batch]`, at every step size the regime measures.
///
/// They are the forward pass's own shapes for that model, derived from its
/// dimensions rather than listed: the hidden-state norm `[n_embd, t]` and the
/// per-head query/key norms `[head_dim, n_head(_kv), t]`; the gated FFN's
/// `[n_ff, t]` activation and product; the residual `[n_embd, t]`; and the
/// weight products `[K, N, t]` - the Q, K/V, output, gate/up and down
/// projections. A family
/// the forward pass does not run - the backward and training kernels - has
/// no shape in a profile, and `None` says so instead of inventing one.
pub fn profile_shapes(
    family: Family,
    shape: &ModelShape,
    regime: Regime,
) -> Option<Vec<[usize; 4]>> {
    let (embd, ff, head) = (
        shape.n_embd as usize,
        shape.n_ff as usize,
        shape.head_dim as usize,
    );
    let (q, kv) = (
        shape.n_head as usize * head,
        shape.n_head_kv as usize * head,
    );
    let per_step = |t: usize| -> Vec<[usize; 4]> {
        match family {
            Family::RmsNorm => vec![
                [embd, t, 1, 1],
                [head, shape.n_head as usize, t, 1],
                [head, shape.n_head_kv as usize, t, 1],
            ],
            Family::Unary => vec![[ff, t, 1, 1]],
            Family::Elementwise | Family::ElementwiseRepeat | Family::Scale => {
                vec![[ff, t, 1, 1], [embd, t, 1, 1]]
            }
            Family::MulMatVec => vec![
                [embd, q, t, 1],
                [embd, kv, t, 1],
                [q, embd, t, 1],
                [embd, ff, t, 1],
                [ff, embd, t, 1],
            ],
            _ => Vec::new(),
        }
    };
    let mut shapes: Vec<[usize; 4]> = Vec::new();
    for &t in regime.tokens() {
        for s in per_step(t as usize) {
            if !shapes.contains(&s) {
                shapes.push(s);
            }
        }
    }
    (!shapes.is_empty()).then_some(shapes)
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

    // A family with a geometry space draws its candidates from the space the
    // table picks from: `expand` is the cartesian product of the space's
    // dimensions, filtered by the lowering itself - never a copy of its
    // refusal rules. The control is the defaults, which the table's fallback
    // row already carries. The label is the derived variant name, and the
    // proposal is the pick the geometry would be committed as.
    if let Some(space) = spaces::space(family) {
        let defaults = (space.defaults)(gpu);
        for g in space.expand(gpu, &subject.kernel) {
            if g == defaults {
                continue;
            }
            let tag = g
                .variant_name(defaults)
                .map(|n| n.to_string())
                .unwrap_or_else(|| "defaults".to_string());
            let pick = Pick {
                geometry: g,
                claim: Claim::any(),
                priority: 90,
                evidence: "",
                name: None,
            };
            v.push(Candidate::from_space(tag, g.schedule(gpu), pick, defaults));
        }
    }

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
        Family::RowReduce | Family::RmsNorm | Family::MulMatVec => {
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
        // `schedule::bench` exists for.
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
        // Experimental parallel plane/batch space. Flat grid also flattens i/j,
        // so small dimensions can fill a workgroup instead of wasting a tiled grid.
        Family::OutProd => {
            for block in [32u32, 64, 128, 256] {
                v.push(Candidate::new(
                    format!("flat outputs b{block}"),
                    format!("Schedule::gpu_grid_flat({gpu_name}, [{block}, 1, 1], 1)"),
                    Schedule::gpu_grid_flat(gpu, [block, 1, 1], 1),
                ));
            }
        }
        Family::MatMulNaive => {}
    }
    v
}

/// The axis groups a shape rule may be written over, in the order the sweep
/// tries them.
///
/// It is the registry's vocabulary and not a superset of it: a published rule is
/// the product of some axes' extents inside `[min, max]`, and the two groupings
/// below include the table's token domain alongside - the row count (`row · plane · batch`,
/// what occupies the device) and the row length (`col`, what a workgroup
/// covers). A domain that needs a third grouping is a domain this tool declines
/// to invent (`verdict::Refusal::NoDistinctDomain`).
pub const RULE_AXES: [&[&str]; 3] = [&["row", "plane", "batch"], &["col"], &["plane"]];

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

/// A geometry as the table's pick list would write it: a lane geometry as the
/// `map_lanes` diff against the space's `defaults` (so the literal is correct
/// on every backend), a matrix geometry as the full literal.
pub fn geometry_source(g: Geometry, defaults: Geometry) -> String {
    match g {
        Geometry::Matrix {
            rows,
            depth,
            padding,
        } => {
            format!("Geometry::Matrix {{ rows: {rows}, depth: {depth}, padding: {padding} }}")
        }
        Geometry::Lanes { .. } => {
            let l = g.lanes().expect("a lane geometry has lane dimensions");
            let d = defaults
                .lanes()
                .unwrap_or(rir_lower::schedule::geometry::Lanes::NEUTRAL);
            let mut sets: Vec<String> = Vec::new();
            if l.lane_tile != d.lane_tile {
                sets.push(format!("l.lane_tile = {}", l.lane_tile));
            }
            if l.row_tile != d.row_tile {
                sets.push(format!("l.row_tile = {}", l.row_tile));
            }
            if l.subgroups != d.subgroups {
                sets.push(format!("l.subgroups = {}", l.subgroups));
            }
            if l.segment != d.segment {
                sets.push(format!("l.segment = {}", l.segment));
            }
            if l.reduce_run != d.reduce_run {
                sets.push(format!("l.reduce_run = {}", l.reduce_run));
            }
            if l.factor_scales != d.factor_scales {
                sets.push(format!("l.factor_scales = {}", l.factor_scales));
            }
            if l.stage_b != d.stage_b {
                sets.push(format!("l.stage_b = {}", l.stage_b));
            }
            match sets.len() {
                0 => "defaults".to_string(),
                1 => format!("defaults.map_lanes(|l| {})", sets[0]),
                _ => format!("defaults.map_lanes(|l| {{ {} }})", sets.join("; ")),
            }
        }
    }
}

/// A pick as the table's pick list would carry it, with `claim` as the rule
/// the measurement justified - the verdict's claim when the pick is proposed.
pub fn pick_source(pick: &Pick, defaults: Geometry, claim: &Claim) -> String {
    format!(
        "Pick {{ geometry: {}, claim: {}, priority: {}, evidence: {:?}, name: {:?} }}",
        geometry_source(pick.geometry, defaults),
        claim_source(claim),
        pick.priority,
        pick.evidence,
        pick.name
    )
}

/// A claim as `schedules_for` would write it, with the open ends of
/// `ShapeRule::new` left open.
pub fn claim_source(claim: &Claim) -> String {
    let mut source = "Claim::any()".to_string();
    for rule in claim.rules() {
        let range = match (rule.min, rule.max) {
            (1, u32::MAX) => "..".to_string(),
            (1, max) => format!("..={max}"),
            (min, u32::MAX) => format!("{min}.."),
            (min, max) => format!("{min}..={max}"),
        };
        match rule.axes.as_slice() {
            [axis] => source.push_str(&format!(".on(\"{axis}\", {range})")),
            axes => {
                let axes = axes
                    .iter()
                    .map(|a| format!("\"{a}\""))
                    .collect::<Vec<_>>()
                    .join(", ");
                source.push_str(&format!(".product(&[{axes}], {range})"));
            }
        }
    }
    source
}

#[cfg(test)]
mod tests {
    use rir_core::catalog::Regime;
    use rir_lower::Family;

    use super::{Origin, Subject, candidates, profile_shapes};
    use rir_lower::GpuBackend;
    use rir_lower::schedule::spaces;

    /// The GEMV's proposed candidates are the schedule space's geometries
    /// (minus the defaults, which the table's fallback row already carries),
    /// each labelled by its derived variant name and carrying the pick its
    /// proposal would commit.
    #[test]
    fn the_gemv_candidates_come_from_the_schedule_space() {
        let subject = Subject::resolve("mul_mat_vec_q4_K").unwrap();
        let gpu = GpuBackend::Metal;
        let cands = candidates(&subject, gpu);
        let space = spaces::mul_mat_vec::space();
        let defaults = (space.defaults)(gpu);
        let expanded = space.expand(gpu, &subject.kernel);
        let tags: Vec<&str> = cands.iter().map(|c| c.tag.as_str()).collect();
        for g in &expanded {
            if *g == defaults {
                continue;
            }
            let name = g
                .variant_name(defaults)
                .map(|n| n.to_string())
                .expect("a non-default geometry has a name");
            assert!(
                tags.contains(&name.as_str()),
                "the space expands to {name} but no candidate carries it"
            );
        }
        for c in &cands {
            if let Some((pick, d)) = &c.pick {
                assert_eq!(*d, defaults);
                assert!(expanded.contains(&pick.geometry));
            }
        }
        // The control is the table's fallback row, and it is there.
        assert!(cands.iter().any(|c| c.origin == Origin::Fallback));
    }

    /// The forward shapes of one profile follow its dimensions and the
    /// regime's step sizes; a training-only family has none.
    #[test]
    fn a_profile_derives_the_forward_shapes_of_its_model() {
        let p = rir_core::profiles::named("qwen3-0.6b-q8_0").unwrap();
        let decode = profile_shapes(Family::RmsNorm, &p.shape, Regime::Decode).unwrap();
        assert!(decode.contains(&[1024, 1, 1, 1]), "hidden norm, one token");
        assert!(decode.contains(&[128, 16, 32, 1]), "query norm, 32 tokens");
        assert!(decode.contains(&[128, 8, 4, 1]), "key norm, 4 tokens");
        assert!(decode.iter().all(|s| s[1] <= 32 && s[2] <= 32));

        let prefill = profile_shapes(Family::Unary, &p.shape, Regime::Prefill).unwrap();
        assert_eq!(prefill, vec![[3072, 512, 1, 1]]);

        let gemv = profile_shapes(Family::MulMatVec, &p.shape, Regime::Decode).unwrap();
        assert!(gemv.contains(&[1024, 2048, 1, 1]), "query projection");
        assert!(gemv.contains(&[1024, 1024, 8, 1]), "key/value projection");
        assert!(gemv.contains(&[3072, 1024, 32, 1]), "down projection");

        assert!(profile_shapes(Family::OutProd, &p.shape, Regime::Decode).is_none());
        assert!(profile_shapes(Family::Cumsum, &p.shape, Regime::Decode).is_none());
    }
}
