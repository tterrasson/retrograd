//! `rir-gen promote` - attribute a measured ratio session to the variants that
//! served it, and propose the per-variant policy table from it.
//!
//! The promotion lane (`scripts/test-rir.sh --emit-ratios`) writes one row per
//! timed shape: `session  pair  regime  label  nat_us  rir_us  median  min
//! max  passes  verdict`. This subcommand replays the lane's own selection in
//! Rust - same dtype claim, same shape claim, same priority rule - to say
//! which variant each measured shape was served by, and then proposes a
//! `variants.tsv` diff: `prefer_generated` when every shape attributed to a
//! variant holds the tolerance (the lane's own criterion, median of the
//! passes), `observe_generated` otherwise, never above the pair's ceiling.
//!
//! Nothing is decided here: without `--write` the tool prints the proposed
//! diff and the commit stays human. `--check` is the lane's back-stop: it
//! fails when a variant already at `prefer_generated` has an attributed shape
//! beyond the tolerance.
//!
//! Attribution replays the **measurement-time** selection: the lane runs the
//! RIR matrix with `RETRO_RIR_TEST_PREFER`, which raises every row of the
//! measured pair to `prefer`, so the shape a pass timed is the one the
//! highest-priority fitting row serves - policy included, the effective
//! policy of a measured row is `prefer`. Production skipping of non-`prefer`
//! rows is a separate rule and never changes which variant a measurement ran.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rir_core::catalog::BackendPolicy;
use rir_core::claim::Claim;
use rir_emit::{ArgSource, GgmlBackend};

use crate::GenError;

/// What `promote` refuses to do.
#[derive(Debug, thiserror::Error)]
pub enum PromoteError {
    #[error("cannot read {path}: {source}")]
    Read {
        path: String,
        source: std::io::Error,
    },
    #[error("cannot write {path}: {source}")]
    Write {
        path: String,
        source: std::io::Error,
    },
    #[error("ratios file {0}: {1}")]
    MalformedRatio(String, String),
    #[error("override table: {0}")]
    MalformedTable(String),
    #[error("ratios file mixes sessions ({0} and {1}); one file is one lane pass")]
    MixedSessions(String, String),
    #[error("label '{0}' does not parse for {1}, which has a shape adapter")]
    UnparsableLabel(String, String),
    #[error("tolerance must be finite and greater than zero (got {0})")]
    InvalidTolerance(f64),
    #[error("usage: {0}")]
    Usage(String),
}

/// One ratio row, as `scripts/test-rir.sh --emit-ratios` writes it.
#[derive(Clone, Debug)]
pub struct RatioRow {
    pub session: String,
    /// `OP/backend`, e.g. `MUL_MAT/metal`.
    pub pair: String,
    pub regime: String,
    /// The `test-backend-ops` perf label, e.g.
    /// `MUL_MAT(type_a=q8_0,type_b=f32,m=4096,n=8,k=14336,...)`.
    pub label: String,
    pub nat_us: f64,
    pub rir_us: f64,
    /// Median of the per-pass ratios - the column the verdict is made on.
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub passes: u32,
    pub verdict: String,
}

/// Parses a ratios file written by `--emit-ratios`. Lines starting with `#`
/// (the header) are skipped.
pub fn parse_ratios(text: &str) -> Result<Vec<RatioRow>, PromoteError> {
    let mut rows = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        let bad = |what: &str| {
            PromoteError::MalformedRatio(
                format!("line {}", lineno + 1),
                format!("{what}: expected 11 tab-separated fields, got {}", f.len()),
            )
        };
        if f.len() != 11 {
            return Err(bad("field count"));
        }
        let num = |i: usize| -> Result<f64, PromoteError> {
            f[i].parse::<f64>()
                .ok()
                .filter(|value| value.is_finite() && *value > 0.0)
                .ok_or_else(|| {
                    bad(&format!(
                        "field {} ('{}') must be finite and positive",
                        i + 1,
                        f[i]
                    ))
                })
        };
        let row = RatioRow {
            session: f[0].to_string(),
            pair: f[1].to_string(),
            regime: f[2].to_string(),
            label: f[3].to_string(),
            nat_us: num(4)?,
            rir_us: num(5)?,
            median: num(6)?,
            min: num(7)?,
            max: num(8)?,
            passes: f[9]
                .parse::<u32>()
                .map_err(|_| bad("field 10 is not an integer"))?,
            verdict: f[10].to_string(),
        };
        if row.session.is_empty() || split_pair(&row.pair).is_none() {
            return Err(bad("missing session or invalid OP/backend pair"));
        }
        if row.passes == 0 || row.min > row.median || row.median > row.max {
            return Err(bad("expected passes > 0 and min <= median <= max"));
        }
        rows.push(row);
    }
    Ok(rows)
}

/// One production variant row, in the shape the selection rule needs.
#[derive(Clone, Debug)]
pub struct VariantRow {
    pub op: String,
    /// The member a family op claims (`GGML_UNARY_OP_RELU` on `UNARY`), or
    /// `None` for an op that is not a family.
    pub op_variant: Option<String>,
    pub backend: GgmlBackend,
    pub variant_id: String,
    pub priority: u8,
    pub claim: Claim,
    /// Dtype the `src[0]` binding claims; the label's `type_a` selects on it.
    pub src0_dtype: String,
    pub pair_policy: BackendPolicy,
    /// The policy the override table gives this variant (the pair's if absent).
    pub policy: BackendPolicy,
    /// Registry order, the tie-break the C selection uses.
    pub order: usize,
}

/// Every production variant the registry publishes, in registry order.
///
/// # Errors
///
/// `GenError` if a schedule of the table does not lower.
pub fn production_variants() -> Result<Vec<VariantRow>, GenError> {
    let overrides = rir_kernels::policy::variant_policies();
    let mut out = Vec::new();
    for entry in rir_kernels::registry() {
        let (kernel, spec) = (&entry.kernel, &entry.integration);
        for schedule in &entry.schedules {
            if schedule.backend() == rir_lower::Backend::Cpu {
                continue;
            }
            let lk = rir_lower::lower(kernel, schedule.clone())?;
            let backend = rir_emit::manifest::ggml_backend(&lk);
            if !spec.production_on(backend) {
                continue;
            }
            let src0 = spec
                .args
                .iter()
                .position(|a| matches!(a, ArgSource::Src(0)))
                .and_then(|i| lk.args.get(i))
                .map(|a| a.ty.dtype.name().to_string())
                .unwrap_or_default();
            let variant_id = rir_emit::variant_id(&lk);
            let pair_policy = spec.policy_for(backend);
            out.push(VariantRow {
                op: spec.ggml_op.unwrap_or_default().to_string(),
                op_variant: spec.ggml_op_variant.map(str::to_string),
                backend,
                policy: overrides.get(&variant_id, backend, pair_policy),
                variant_id,
                priority: schedule.priority(),
                claim: schedule.eligible_when().clone(),
                src0_dtype: src0,
                pair_policy,
                order: out.len(),
            });
        }
    }
    Ok(out)
}

/// Whether `op` has a shape adapter below: a per-variant policy on an op
/// without one could never be decided from measurements.
pub fn has_shape_adapter(op: &str) -> bool {
    matches!(op, "GGML_OP_MUL_MAT" | "GGML_OP_UNARY")
}

/// What a perf label says about the node it timed, in the terms the selection
/// rule evaluates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelShape {
    /// The `src0` dtype spelling, e.g. `q8_0` or `f32`.
    pub dtype: String,
    /// The member of a family op, spelled as the registry claims it
    /// (`GGML_UNARY_OP_RELU`); `None` for an op that is not a family.
    pub op_variant: Option<String>,
    /// The kernel's axis extents.
    pub extents: BTreeMap<String, u64>,
}

/// Reads one `test-backend-ops` perf label into the [`LabelShape`] the
/// selection rule evaluates. One adapter per op, because the label's parameter
/// names are the op's own.
///
/// `MUL_MAT` spells its case as `m` (output rows), `n` (step columns) and `k`
/// (contraction); the GEMV kernel's axes are `row = m`, `plane = n`,
/// `col = k`.
///
/// `UNARY` is described by its member (`ggml_op_desc`), so the label is
/// `RELU(type=f32,ne_a=[4096,16,1,1],v=0)`: the member is the claim, and
/// `ne_a` gives the four axes `col`, `row`, `plane`, `batch` in ggml order. A
/// view (`v=1`) is refused: which variant serves it depends on a layout claim
/// (the flattened linear address wants contiguous bindings) that a shape replay
/// cannot see.
pub fn label_shape(op: &str, label: &str) -> Option<LabelShape> {
    match op {
        "GGML_OP_MUL_MAT" => {
            let inner = label.strip_prefix("MUL_MAT(")?.strip_suffix(')')?;
            let mut extents = BTreeMap::new();
            let mut dtype = None;
            let mut src1_f32 = false;
            for part in inner.split(',') {
                let Some((key, value)) = part.split_once('=') else {
                    continue;
                };
                match key {
                    "type_a" => dtype = Some(value.to_string()),
                    "type_b" => src1_f32 = value == "f32",
                    "m" => {
                        extents.insert("row".to_string(), value.parse().ok()?);
                    }
                    "n" => {
                        extents.insert("plane".to_string(), value.parse().ok()?);
                    }
                    "k" => {
                        extents.insert("col".to_string(), value.parse().ok()?);
                    }
                    _ => {}
                }
            }
            if !src1_f32
                || ["row", "plane", "col"]
                    .iter()
                    .any(|axis| extents.get(*axis).is_none_or(|extent| *extent == 0))
            {
                return None;
            }
            Some(LabelShape {
                dtype: dtype?,
                op_variant: None,
                extents,
            })
        }
        "GGML_OP_UNARY" => {
            let (member, rest) = label.split_once('(')?;
            let inner = rest.strip_suffix(')')?;
            if member.is_empty()
                || !member
                    .bytes()
                    .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
            {
                return None;
            }
            let (mut dtype, mut ne, mut contiguous) = (None, None, false);
            for (key, value) in label_params(inner) {
                match key {
                    "type" => dtype = Some(value.to_string()),
                    "ne_a" => ne = Some(bracketed_extents(value)?),
                    "v" => contiguous = value == "0",
                    _ => {}
                }
            }
            let ne: [u64; 4] = ne?;
            if !contiguous || ne.contains(&0) {
                return None;
            }
            let extents = ["col", "row", "plane", "batch"]
                .into_iter()
                .zip(ne)
                .map(|(axis, extent)| (axis.to_string(), extent))
                .collect();
            Some(LabelShape {
                dtype: dtype?,
                op_variant: Some(format!("GGML_UNARY_OP_{member}")),
                extents,
            })
        }
        _ => None,
    }
}

/// The `key=value` parameters of a perf label, split at the commas that are not
/// inside a `[...]` list.
fn label_params(inner: &str) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0usize, 0usize);
    for (i, c) in inner.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                out.extend(inner[start..i].split_once('='));
                start = i + 1;
            }
            _ => {}
        }
    }
    out.extend(inner[start..].split_once('='));
    out
}

/// `[a,b,c,d]`, the four extents a label prints for a ggml tensor.
fn bracketed_extents(value: &str) -> Option<[u64; 4]> {
    let inner = value.strip_prefix('[')?.strip_suffix(']')?;
    let parsed: Vec<u64> = inner
        .split(',')
        .map(|v| v.trim().parse().ok())
        .collect::<Option<_>>()?;
    parsed.try_into().ok()
}

/// The variant a measured shape was served by: same rule as the dispatcher -
/// member claim, `src0` dtype claim, every shape rule holds, highest priority,
/// registry order on a tie. Observation rows are elevated for measurement,
/// while native-only rows remain excluded.
pub fn select_for_shape<'a>(
    variants: &'a [VariantRow],
    op: &str,
    backend: GgmlBackend,
    shape: &LabelShape,
) -> Option<&'a VariantRow> {
    variants
        .iter()
        .filter(|v| {
            v.op == op
                && v.backend == backend
                && v.op_variant == shape.op_variant
                && v.src0_dtype == shape.dtype
        })
        // TEST_PREFER raises observe, but cannot enable a native-only row.
        .filter(|v| v.policy.min(v.pair_policy) != BackendPolicy::NativeOnly)
        .filter(|v| v.claim.holds(|axis| shape.extents.get(axis).copied()))
        .max_by_key(|v| (v.priority, std::cmp::Reverse(v.order)))
}

/// `OP/backend` as the ratios file spells a pair, to the registry's spelling.
fn split_pair(pair: &str) -> Option<(String, GgmlBackend)> {
    let (op, backend) = pair.split_once('/')?;
    let backend = match backend {
        "metal" => GgmlBackend::Metal,
        "vulkan" => GgmlBackend::Vulkan,
        "cuda" => GgmlBackend::Cuda,
        "cpu" => GgmlBackend::Cpu,
        _ => return None,
    };
    Some((format!("GGML_OP_{op}"), backend))
}

/// One attributed measurement: a variant and the ratio rows it served.
pub struct Attribution {
    pub variant: VariantRow,
    pub rows: Vec<RatioRow>,
}

/// Attributes every ratio row to the variant that served it. Rows whose pair
/// has no shape adapter, or whose label does not parse, are skipped - except
/// a label of an adapted op that fails to parse, which is an error: silently
/// dropping it would under-count a variant's shapes.
///
/// # Errors
///
/// `PromoteError` on an unparsable label of an adapted op or a mixed-session
/// file.
pub fn attribute(
    rows: &[RatioRow],
    variants: &[VariantRow],
) -> Result<Vec<Attribution>, PromoteError> {
    let mut by_variant: BTreeMap<String, Attribution> = BTreeMap::new();
    let mut sessions = std::collections::BTreeSet::new();
    for row in rows {
        sessions.insert(row.session.clone());
    }
    if sessions.len() > 1 {
        let mut it = sessions.into_iter();
        let a = it.next().unwrap_or_default();
        let b = it.next().unwrap_or_default();
        return Err(PromoteError::MixedSessions(a, b));
    }
    for row in rows {
        let Some((op, backend)) = split_pair(&row.pair) else {
            continue;
        };
        if !has_shape_adapter(&op) {
            continue;
        }
        let Some(shape) = label_shape(&op, &row.label) else {
            return Err(PromoteError::UnparsableLabel(row.label.clone(), op));
        };
        let Some(variant) = select_for_shape(variants, &op, backend, &shape) else {
            continue;
        };
        let key = format!("{}/{}", variant.variant_id, variant.backend.name());
        by_variant
            .entry(key)
            .or_insert_with(|| Attribution {
                variant: variant.clone(),
                rows: Vec::new(),
            })
            .rows
            .push(row.clone());
    }
    Ok(by_variant.into_values().collect())
}

/// The policy a variant should carry, given the shapes attributed to it: the
/// lane's criterion - at least one shape, and the median of every one of them
/// at or under the tolerance - never above the pair's ceiling.
pub fn proposed_policy(attribution: &Attribution, tolerance: f64) -> BackendPolicy {
    let ceiling = attribution.variant.pair_policy;
    if attribution.rows.is_empty() {
        return attribution.variant.policy.min(ceiling);
    }
    let wins = attribution.rows.iter().all(|r| r.median <= tolerance);
    let proposed = if wins {
        BackendPolicy::PreferGenerated
    } else {
        BackendPolicy::ObserveGenerated
    };
    proposed.min(ceiling)
}

/// Where the committed override table lives.
pub fn variants_tsv_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../rir-kernels/policy/variants.tsv")
}

/// One proposed change to the override table.
#[derive(Clone, Debug, PartialEq)]
pub struct Proposal {
    pub variant_id: String,
    pub backend: GgmlBackend,
    pub current: BackendPolicy,
    pub proposed: BackendPolicy,
    pub session: String,
    pub shapes: usize,
    pub worst: f64,
}

/// The proposals for a ratios file: every attributed variant whose proposed
/// policy differs from the table's, plus every attributed variant (for
/// `--check`).
pub fn proposals(rows: &[RatioRow], tolerance: f64) -> Result<Vec<Proposal>, GenError> {
    validate_tolerance(tolerance)?;
    let variants = production_variants()?;
    let attributions = attribute(rows, &variants)?;
    let session = rows.first().map(|r| r.session.clone()).unwrap_or_default();
    let mut out = Vec::new();
    for a in &attributions {
        let proposed = proposed_policy(a, tolerance);
        out.push(Proposal {
            variant_id: a.variant.variant_id.clone(),
            backend: a.variant.backend,
            current: a.variant.policy,
            proposed,
            session: session.clone(),
            shapes: a.rows.len(),
            worst: a.rows.iter().map(|r| r.median).fold(0.0, f64::max),
        });
    }
    Ok(out)
}

/// Renders the proposed `variants.tsv` diff: `-`/`+` lines for every proposal
/// that changes the table, and a summary line per attributed variant.
pub fn render_diff(
    proposals: &[Proposal],
    current: &[rir_kernels::policy::VariantPolicyRow],
) -> String {
    let mut out = String::new();
    let find = |variant_id: &str, backend: GgmlBackend| {
        current
            .iter()
            .find(|r| r.variant_id == variant_id && r.backend == backend)
    };
    for p in proposals {
        let cur = find(&p.variant_id, p.backend);
        let current_policy = cur.map(|r| r.policy).unwrap_or(p.current);
        if current_policy == p.proposed {
            continue;
        }
        let old_line = cur.map(|r| {
            format!(
                "{}\t{}\t{}\t{}\t{}",
                r.variant_id,
                r.backend.name(),
                r.policy.name(),
                r.session,
                r.note
            )
        });
        let new_line = format!(
            "{}\t{}\t{}\t{}\tworst median {:.4} over {} shape(s)",
            p.variant_id,
            p.backend.name(),
            p.proposed.name(),
            p.session,
            p.worst,
            p.shapes
        );
        if let Some(old) = old_line {
            out.push_str(&format!("-{old}\n"));
        } else {
            out.push_str("# (new row)\n");
        }
        out.push_str(&format!("+{new_line}\n"));
    }
    out
}

/// Applies the proposals to the committed `variants.tsv`, preserving its
/// comments and the order of untouched rows. Returns the new content.
pub fn apply_write(proposals: &[Proposal], current_text: &str) -> Result<String, PromoteError> {
    let current = rir_kernels::policy::parse_variant_policies(current_text)
        .map_err(PromoteError::MalformedTable)?;
    let mut lines: Vec<String> = Vec::new();
    let mut appended: Vec<String> = Vec::new();
    for line in current_text.lines() {
        if line.is_empty() || line.starts_with('#') {
            lines.push(line.to_string());
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        let Some(row) = current
            .iter()
            .find(|r| r.variant_id == fields[0] && r.backend.name() == fields[1])
        else {
            lines.push(line.to_string());
            continue;
        };
        let Some(p) = proposals
            .iter()
            .find(|p| p.variant_id == row.variant_id && p.backend == row.backend)
        else {
            lines.push(line.to_string());
            continue;
        };
        if p.proposed == row.policy {
            lines.push(line.to_string());
            continue;
        }
        lines.push(format!(
            "{}\t{}\t{}\t{}\tworst median {:.4} over {} shape(s)",
            row.variant_id,
            row.backend.name(),
            p.proposed.name(),
            p.session,
            p.worst,
            p.shapes
        ));
    }
    // A proposal for a variant the table does not name yet: append a row.
    for p in proposals {
        if current
            .iter()
            .any(|r| r.variant_id == p.variant_id && r.backend == p.backend)
        {
            continue;
        }
        let current_policy = p.current;
        if current_policy == p.proposed {
            continue;
        }
        appended.push(format!(
            "{}\t{}\t{}\t{}\tworst median {:.4} over {} shape(s)",
            p.variant_id,
            p.backend.name(),
            p.proposed.name(),
            p.session,
            p.worst,
            p.shapes
        ));
    }
    let mut out = lines.join("\n");
    if !out.is_empty() {
        out.push('\n');
    }
    for a in appended {
        out.push_str(&a);
        out.push('\n');
    }
    Ok(out)
}

/// `--check`: every variant whose effective policy is `prefer_generated` must
/// hold the tolerance on every shape attributed to it. Returns the offending
/// `(variant, shape, median)` triples.
pub fn check(rows: &[RatioRow], tolerance: f64) -> Result<Vec<(String, String, f64)>, GenError> {
    validate_tolerance(tolerance)?;
    let variants = production_variants()?;
    let attributions = attribute(rows, &variants)?;
    let mut violations = Vec::new();
    for a in &attributions {
        let effective = a.variant.policy.min(a.variant.pair_policy);
        if effective != BackendPolicy::PreferGenerated {
            continue;
        }
        for row in &a.rows {
            if row.median > tolerance {
                violations.push((a.variant.variant_id.clone(), row.label.clone(), row.median));
            }
        }
    }
    Ok(violations)
}

fn validate_tolerance(tolerance: f64) -> Result<(), PromoteError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PromoteError::InvalidTolerance(tolerance));
    }
    Ok(())
}

/// Runs the `promote` subcommand. Returns the process exit code.
pub fn run(args: &[String]) -> Result<std::process::ExitCode, GenError> {
    use std::process::ExitCode;

    let mut ratios_path: Option<PathBuf> = None;
    let mut tolerance: f64 = 1.05;
    let mut write = false;
    let mut check_only = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ratios" => {
                i += 1;
                ratios_path =
                    Some(PathBuf::from(args.get(i).ok_or_else(|| {
                        PromoteError::Usage("--ratios expects a path".into())
                    })?));
            }
            "--tolerance" => {
                i += 1;
                let v = args
                    .get(i)
                    .ok_or_else(|| PromoteError::Usage("--tolerance expects a number".into()))?;
                tolerance = v.parse::<f64>().map_err(|_| {
                    PromoteError::Usage(format!("--tolerance: '{v}' is not a number"))
                })?;
            }
            "--write" => write = true,
            "--check" => check_only = true,
            other => {
                return Err(PromoteError::Usage(format!("unknown argument: {other}")).into());
            }
        }
        i += 1;
    }
    validate_tolerance(tolerance)?;
    let ratios_path =
        ratios_path.ok_or_else(|| PromoteError::Usage("--ratios is required".into()))?;
    let text = std::fs::read_to_string(&ratios_path).map_err(|e| {
        GenError::Promote(PromoteError::Read {
            path: ratios_path.display().to_string(),
            source: e,
        })
    })?;
    let rows = parse_ratios(&text).map_err(GenError::Promote)?;
    if rows.is_empty() {
        eprintln!("promote: {} carries no ratio row", ratios_path.display());
        return Ok(ExitCode::FAILURE);
    }
    let session = rows[0].session.clone();

    if check_only {
        let violations = check(&rows, tolerance)?;
        if violations.is_empty() {
            println!(
                "promote --check: OK - every prefer variant holds {:.2} on all its attributed shapes (session {session})",
                tolerance
            );
            return Ok(ExitCode::SUCCESS);
        }
        for (variant, label, median) in &violations {
            eprintln!("promote --check: {variant} at {median:.2} > {tolerance:.2} on {label}");
        }
        return Ok(ExitCode::FAILURE);
    }

    let proposals = proposals(&rows, tolerance)?;
    let current_text = std::fs::read_to_string(variants_tsv_path()).map_err(|e| {
        GenError::Promote(PromoteError::Read {
            path: variants_tsv_path().display().to_string(),
            source: e,
        })
    })?;
    let current = rir_kernels::policy::parse_variant_policies(&current_text)
        .map_err(PromoteError::MalformedTable)?;
    let diff = render_diff(&proposals, &current);

    println!("promote: session {session}, tolerance {tolerance:.2}");
    for p in &proposals {
        println!(
            "  {}/{}: {} -> {} ({} shape(s), worst median {:.2})",
            p.variant_id,
            p.backend.name(),
            p.current.name(),
            p.proposed.name(),
            p.shapes,
            p.worst
        );
    }
    if diff.is_empty() {
        println!("no change proposed for variants.tsv");
    } else {
        println!("proposed diff of {}:", variants_tsv_path().display());
        print!("{diff}");
    }

    if write {
        if diff.is_empty() {
            return Ok(ExitCode::SUCCESS);
        }
        let new_text = apply_write(&proposals, &current_text).map_err(GenError::Promote)?;
        std::fs::write(variants_tsv_path(), &new_text).map_err(|e| {
            GenError::Promote(PromoteError::Write {
                path: variants_tsv_path().display().to_string(),
                source: e,
            })
        })?;
        println!("wrote {}", variants_tsv_path().display());
    }
    Ok(ExitCode::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real perf label of the reference session: the single-token q8_0 GEMV
    /// shape the decode grid starts with.
    const N1_Q8_0: &str = "MUL_MAT(type_a=q8_0,type_b=f32,m=1024,n=1,k=1024,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0)";
    const N8_Q4_K: &str = "MUL_MAT(type_a=q4_K,type_b=f32,m=4096,n=8,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0)";
    const N32_Q6_K: &str = "MUL_MAT(type_a=q6_K,type_b=f32,m=4096,n=32,k=14336,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0,m_v=0,pad=0)";

    #[test]
    fn the_mul_mat_adapter_reads_the_lane_labels() {
        let LabelShape {
            dtype,
            op_variant,
            extents,
        } = label_shape("GGML_OP_MUL_MAT", N1_Q8_0).unwrap();
        assert_eq!(dtype, "q8_0");
        assert_eq!(op_variant, None);
        assert_eq!(
            extents,
            BTreeMap::from([
                ("row".to_string(), 1024),
                ("plane".to_string(), 1),
                ("col".to_string(), 1024),
            ])
        );
        let shape = label_shape("GGML_OP_MUL_MAT", N8_Q4_K).unwrap();
        assert_eq!(shape.dtype, "q4_K");
        assert_eq!(shape.extents.get("plane"), Some(&8));
        assert_eq!(shape.extents.get("col"), Some(&14336));
        let shape = label_shape("GGML_OP_MUL_MAT", N32_Q6_K).unwrap();
        assert_eq!(shape.dtype, "q6_K");
        assert_eq!(shape.extents.get("plane"), Some(&32));
        // `k_v` and `m_v` must not be read as `k`/`m`.
        assert!(!shape.extents.contains_key("k_v"));
        assert!(label_shape("GGML_OP_CUMSUM", "CUMSUM(...)").is_none());
        assert!(!has_shape_adapter("GGML_OP_CUMSUM"));
        assert!(label_shape("GGML_OP_MUL_MAT", "MUL_MAT(type_a=q8_0,type_b=f32,n=1)").is_none());
        assert!(
            label_shape(
                "GGML_OP_MUL_MAT",
                &N1_Q8_0.replace("type_b=f32", "type_b=f16")
            )
            .is_none()
        );
        assert!(label_shape("GGML_OP_MUL_MAT", &N1_Q8_0.replace("m=1024", "m=0")).is_none());
    }

    #[test]
    fn attribution_selects_the_variant_the_lane_measured() {
        let variants = production_variants().unwrap();
        let metal = |dtype: &str, plane: u64| {
            let mut extents = BTreeMap::new();
            extents.insert("row".to_string(), 4096);
            extents.insert("plane".to_string(), plane);
            extents.insert("col".to_string(), 14336);
            let shape = LabelShape {
                dtype: dtype.to_string(),
                op_variant: None,
                extents,
            };
            select_for_shape(&variants, "GGML_OP_MUL_MAT", GgmlBackend::Metal, &shape)
                .map(|v| v.variant_id.clone())
        };
        // plane = 1: only the n1 pick claims it.
        assert_eq!(
            metal("q8_0", 1).as_deref(),
            Some("mul_mat_vec_q8_0_q8_0_2d_subgroup_tree_n1")
        );
        // plane in [2, 4]: cols4; in [5, 31]: cols8_rt4_seg4_nofs; 32: mma.
        assert!(metal("q8_0", 4).unwrap().ends_with("_cols4"));
        assert!(metal("q4_K", 8).unwrap().ends_with("_cols8_rt4_seg4_nofs"));
        assert!(metal("q6_K", 32).unwrap().ends_with("_mma"));
        // A dtype no metal row claims selects nothing.
        assert!(metal("f32", 8).is_none());
    }

    /// The lane's `UNARY` labels name the member, not the op: the adapter must
    /// turn the member into the claim the registry rows carry, read `ne_a`
    /// despite the commas inside its brackets, and refuse a view.
    #[test]
    fn the_unary_adapter_reads_the_member_and_its_four_axes() {
        let shape =
            label_shape("GGML_OP_UNARY", "GELU_ERF(type=f32,ne_a=[128,16,16,1],v=0)").unwrap();
        assert_eq!(shape.dtype, "f32");
        assert_eq!(shape.op_variant.as_deref(), Some("GGML_UNARY_OP_GELU_ERF"));
        assert_eq!(
            shape.extents,
            BTreeMap::from([
                ("col".to_string(), 128),
                ("row".to_string(), 16),
                ("plane".to_string(), 16),
                ("batch".to_string(), 1),
            ])
        );
        for refused in [
            "RELU(type=f32,ne_a=[128,16,16,1],v=1)",
            "RELU(type=f32,ne_a=[128,16,16],v=0)",
            "RELU(type=f32,ne_a=[0,16,16,1],v=0)",
            "RELU(type=f32,v=0)",
            "relu(type=f32,ne_a=[128,16,16,1],v=0)",
        ] {
            assert!(label_shape("GGML_OP_UNARY", refused).is_none(), "{refused}");
        }
    }

    /// Each member's shapes go to that member's variants only - the claim that
    /// lets `variants.tsv` hold one member of the family - and the shape rule
    /// still arbitrates between a member's lowerings.
    #[test]
    fn unary_attribution_selects_the_measured_member_and_lowering() {
        let variants = production_variants().unwrap();
        let pick = |backend: GgmlBackend, label: &str| {
            let shape = label_shape("GGML_OP_UNARY", label).unwrap();
            select_for_shape(&variants, "GGML_OP_UNARY", backend, &shape)
                .map(|v| v.variant_id.clone())
                .unwrap()
        };
        let wide = "TANH(type=f32,ne_a=[4096,16,1,1],v=0)";
        let short = "TANH(type=f32,ne_a=[128,16,16,1],v=0)";
        for backend in [GgmlBackend::Metal, GgmlBackend::Vulkan] {
            assert!(pick(backend, wide).starts_with("unary_tanh_"));
            assert!(!pick(backend, wide).ends_with("_narrow"));
            // `col <= 256` is the narrow twin's claim.
            assert!(pick(backend, short).ends_with("_narrow"));
            assert!(pick(backend, short).starts_with("unary_tanh_"));
        }
        // On CUDA the linear-address lowering claims every contiguous shape.
        let gelu_erf = pick(
            GgmlBackend::Cuda,
            "GELU_ERF(type=f32,ne_a=[1024,16,1,1],v=0)",
        );
        assert_eq!(gelu_erf, "unary_gelu_erf_f32_4d_serial_flat_linear");
    }

    #[test]
    fn the_ratios_file_parses_and_refuses_mixed_sessions() {
        let text = format!(
            "# session\tpair\tregime\tlabel\tnat_us\trir_us\tmedian\tmin\tmax\tpasses\tverdict\n\
                    s1\tMUL_MAT/metal\tdecode\t{N1_Q8_0}\t10.0\t7.0\t0.70\t0.68\t0.73\t3\tok\n"
        );
        let rows = parse_ratios(&text).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].median, 0.70);
        let mixed = format!(
            "{text}s2\tMUL_MAT/metal\tdecode\t{N1_Q8_0}\t10.0\t7.0\t0.70\t0.68\t0.73\t3\tok\n"
        );
        assert!(matches!(
            attribute(
                &parse_ratios(&mixed).unwrap(),
                &production_variants().unwrap()
            ),
            Err(PromoteError::MixedSessions(..))
        ));
    }

    #[test]
    fn invalid_measurements_and_tolerances_cannot_pass_the_promotion_check() {
        let valid =
            format!("s1\tMUL_MAT/metal\tdecode\t{N1_Q8_0}\t10.0\t7.0\t0.70\t0.68\t0.73\t3\tok\n");
        for invalid in ["NaN", "inf", "-1", "0"] {
            assert!(parse_ratios(&valid.replace("\t0.70\t", &format!("\t{invalid}\t"))).is_err());
        }
        assert!(parse_ratios(&valid.replace("\t3\t", "\t0\t")).is_err());
        assert!(parse_ratios(&valid.replace("\t0.68\t", "\t0.80\t")).is_err());
        for invalid in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert!(check(&[], invalid).is_err());
            assert!(proposals(&[], invalid).is_err());
        }
    }

    #[test]
    fn the_proposal_is_the_lane_criterion_capped_by_the_pair() {
        let variants = production_variants().unwrap();
        let n1 = variants
            .iter()
            .find(|v| v.variant_id == "mul_mat_vec_q8_0_q8_0_2d_subgroup_tree_n1")
            .unwrap()
            .clone();
        let row = |median: f64| RatioRow {
            session: "s".into(),
            pair: "MUL_MAT/metal".into(),
            regime: "decode".into(),
            label: N1_Q8_0.into(),
            nat_us: 10.0,
            rir_us: 10.0 * median,
            median,
            min: median,
            max: median,
            passes: 3,
            verdict: "ok".into(),
        };
        let a = Attribution {
            variant: n1.clone(),
            rows: vec![row(0.70), row(0.73)],
        };
        // Every attributed shape under the tolerance: prefer, capped by the
        // pair's ceiling (prefer on metal).
        assert_eq!(proposed_policy(&a, 1.05), BackendPolicy::PreferGenerated);
        let a = Attribution {
            variant: n1,
            rows: vec![row(0.70), row(1.20)],
        };
        assert_eq!(proposed_policy(&a, 1.05), BackendPolicy::ObserveGenerated);
    }
}
