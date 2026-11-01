//! Rendering a sweep: the measured table, then the proposal.
//!
//! Two blocks and not one, because they are read at different moments. The
//! table is what a reader argues with - it has the same columns as the tables
//! already carries, so a row can be put beside a published
//! one. The proposal is what a reader *commits*: the constructor call and the
//! shape rule, in the vocabulary of `schedules_for`, so that adopting a verdict
//! is a paste and a lane run rather than a translation.

use std::fmt::Write as _;

use crate::candidates::{claim_source, geometry_source, pick_source};
use crate::run::{Arbitrated, SweepReport};
use crate::verdict::{Outcome, Refusal, Verdict};

/// The measured table and the proposal, as printed by the binary.
pub fn render(r: &SweepReport) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "# {} ({}) on {} - {}",
        r.kernel,
        r.family,
        r.backend.name(),
        r.device
    );
    let _ = writeln!(
        out,
        "# baseline: {} - {}",
        r.base_tag,
        footprint(&r.base_footprint)
    );
    let _ = writeln!(
        out,
        "# {} - median of the repetitions; baseline remeasured around each candidate; a gain counts only above the two spreads added, including control drift (floor 2%)",
        r.mode.name()
    );
    if let Some((profile, regime)) = &r.profile {
        let _ = writeln!(
            out,
            "# profile: {profile}, {} regime - shapes derived from the model's dimensions",
            regime.name()
        );
    }
    if let Some(k) = r.contraction {
        let _ = writeln!(
            out,
            "# OUT_PROD: output [i,j,plane,batch], contraction k={k}"
        );
    }
    let _ = writeln!(out);

    let _ = writeln!(
        out,
        "{:<20} {:<22} {:>10} {:>10} {:>8} {:>8}  outcome",
        "candidate", "shape [c,r,p,b]", "base µs", "cand µs", "gain", "noise"
    );
    for a in &r.arbitrated {
        for row in &a.rows {
            let _ = writeln!(
                out,
                "{:<20} {:<22} {:>10.1} {:>10.1} {:>7.1}% {:>7.1}%  {}",
                a.tag,
                format!("{:?}", row.shape),
                row.base.median_us,
                row.candidate.median_us,
                row.gain() * 100.0,
                row.noise() * 100.0,
                match row.outcome() {
                    Outcome::Wins if row.agrees => "wins",
                    Outcome::Loses if row.agrees => "loses",
                    _ if !row.agrees => "DISAGREES",
                    _ => "tied",
                }
            );
        }
    }

    let _ = writeln!(out);
    match &r.profile {
        // A profile identifies the geometry measured in this regime. The
        // proposal changes schedule claims; the promotion lane decides the
        // per-variant policy, and the profile carries no dispatch selection.
        Some((profile, regime)) => {
            let _ = writeln!(out, "## proposal for {profile} ({})", regime.name());
        }
        None => {
            let _ = writeln!(out, "## proposal");
        }
    }
    for a in &r.arbitrated {
        let _ = writeln!(out, "{}", proposal(a));
    }
    for (tag, why) in &r.declined {
        let _ = writeln!(out, "- {tag}: not lowered - {why}");
    }
    out
}

/// One candidate's verdict, with the line to commit when there is one.
pub fn proposal(a: &Arbitrated) -> String {
    let head = format!("- {} [{}] {}", a.tag, origin(a), footprint(&a.footprint));
    // A row the table already carries gets its verdict and no source line: it is
    // a **control**, and printing `v.push(…)` for a schedule already in
    // `schedules_for` would read as a change to make. What its verdict says is
    // whether the rule it already has still matches what the device does.
    if a.origin != crate::candidates::Origin::Proposed {
        return format!("{head}\n    control - {}", control(&a.verdict));
    }
    match &a.verdict {
        Verdict::Replaces { best_gain } => {
            if let Some((pick, defaults)) = &a.pick {
                // A space candidate that wins everywhere becomes the space's
                // defaults rather than a pick above them.
                format!(
                    "{head}\n    REPLACES the fallback: wins by up to {:.0}%, and on no measured \
                     shape does it lose by more than the noise.\n    \
                     // becomes the space's defaults: {}",
                    best_gain * 100.0,
                    geometry_source(pick.geometry, *defaults)
                )
            } else {
                format!(
                    "{head}\n    REPLACES the fallback: wins by up to {:.0}%, and on no measured shape \
                     does it lose by more than the noise.\n    \
                     v.push({}.unnamed());",
                    best_gain * 100.0,
                    a.source
                )
            }
        }
        Verdict::Claims {
            claim,
            best_gain,
            ties_claimed,
        } => {
            let ties = if ties_claimed.is_empty() {
                String::new()
            } else {
                format!(
                    "\n    (the rule also claims {} shape(s) it only ties on: {:?})",
                    ties_claimed.len(),
                    ties_claimed
                )
            };
            if let Some((pick, defaults)) = &a.pick {
                // A candidate from a geometry space is proposed as the pick it
                // would be committed as, with the verdict's rule as its claim.
                format!(
                    "{head}\n    CLAIMS a domain: up to {:.0}% on the shapes the rule covers.{ties}\n    \
                     {}",
                    best_gain * 100.0,
                    pick_source(pick, *defaults, claim)
                )
            } else {
                format!(
                    "{head}\n    CLAIMS a domain: up to {:.0}% on the shapes the rule covers.{ties}\n    \
                     v.push({}.claiming(90, {}));",
                    best_gain * 100.0,
                    a.source,
                    claim_source(claim)
                )
            }
        }
        Verdict::Refused(why) => format!("{head}\n    refused: {}", refusal(why)),
    }
}

/// A table row's verdict, said as what it is: a check on a published rule.
fn control(v: &Verdict) -> String {
    match v {
        Verdict::Replaces { best_gain } => format!(
            "it is ahead by up to {:.0}% and behind nowhere, so the rule it carries claims \
             less than the device does",
            best_gain * 100.0
        ),
        Verdict::Claims {
            claim, best_gain, ..
        } => format!(
            "up to {:.0}%, on the domain {}",
            best_gain * 100.0,
            claim_source(claim)
        ),
        Verdict::Refused(why) => refusal(why),
    }
}

fn origin(a: &Arbitrated) -> &'static str {
    match a.origin {
        crate::candidates::Origin::Fallback => "fallback",
        crate::candidates::Origin::Table => "already in the table",
        crate::candidates::Origin::Proposed => "proposed",
    }
}

fn refusal(r: &Refusal) -> String {
    match r {
        Refusal::Disagrees { shape } => format!(
            "it does not compute what the fallback computes on {shape:?} - a lowering bug, \
             not a candidate"
        ),
        Refusal::NoGain { best_gain, noise } => format!(
            "no gain outside the noise: best {:.1}% against a floor of {:.1}%",
            best_gain * 100.0,
            noise * 100.0
        ),
        Refusal::NoDistinctDomain { overclaimed } => format!(
            "no distinct domain: it loses on {overclaimed:?}, which every rule covering its \
             wins would claim as well"
        ),
        Refusal::NotMeasured => "nothing measured".to_string(),
    }
}

fn footprint(f: &crate::measure::Footprint) -> String {
    let mut s = format!("{} lanes, {} B shared", f.lanes, f.shared_bytes);
    if let Some(limit) = f.shared_limit_bytes {
        let _ = write!(s, ", {limit} B limit per workgroup");
    }
    match (f.registers, f.spill_bytes) {
        (Some(r), Some(sp)) => {
            let _ = write!(s, ", {r} regs, {sp} B spilled");
        }
        _ => s.push_str(", regs/spills not published by this backend"),
    }
    s
}

/// The first line of every proposal file: what the binary checks before it
/// overwrites a path, so that a proposal never lands on a source file.
pub const PROPOSAL_HEADER: &str = "// rir-sweep proposal";

/// The retained rows of `reports`, as a file to read and paste by hand: per
/// sweep, the picks its space candidates earned with the verdict's claim, and
/// for a candidate outside any space the table call it would be committed as.
/// A control and a refusal are not retained; a sweep that retained nothing
/// says so rather than writing an empty block.
pub fn proposal_file(reports: &[SweepReport]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{PROPOSAL_HEADER}");
    let _ = writeln!(
        out,
        "// RIR against RIR: nothing here is decided. A pick's priority is a placeholder and its \
         evidence is the session that measured it, to fill in; a retained row then goes through \
         the promotion lane like any table change."
    );
    for r in reports {
        let _ = writeln!(out);
        let _ = write!(
            out,
            "// {} ({}) on {} - {}",
            r.kernel,
            r.family,
            r.backend.name(),
            r.device
        );
        if let Some((profile, regime)) = &r.profile {
            let _ = write!(out, " - profile {profile}, {} regime", regime.name());
        }
        let _ = writeln!(out);
        let mut retained = 0usize;
        for a in &r.arbitrated {
            if a.origin != crate::candidates::Origin::Proposed {
                continue;
            }
            let line = match (&a.verdict, &a.pick) {
                (Verdict::Claims { claim, .. }, Some((pick, defaults))) => {
                    format!("{},", pick_source(pick, *defaults, claim))
                }
                (Verdict::Claims { claim, .. }, None) => {
                    format!(
                        "v.push({}.claiming(90, {}));",
                        a.source,
                        claim_source(claim)
                    )
                }
                (Verdict::Replaces { .. }, Some((pick, defaults))) => format!(
                    "// becomes the space's defaults: {}",
                    geometry_source(pick.geometry, *defaults)
                ),
                (Verdict::Replaces { .. }, None) => format!("v.push({}.unnamed());", a.source),
                (Verdict::Refused(_), _) => continue,
            };
            let gain = match &a.verdict {
                Verdict::Claims { best_gain, .. } | Verdict::Replaces { best_gain } => *best_gain,
                Verdict::Refused(_) => 0.0,
            };
            let _ = writeln!(out, "// {}: up to {:.0}%", a.tag, gain * 100.0);
            let _ = writeln!(out, "{line}");
            retained += 1;
        }
        if retained == 0 {
            let _ = writeln!(out, "// nothing retained");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use rir_lower::schedule::spaces;
    use rir_lower::{Claim, GpuBackend, Pick};
    use rir_runtime::any::Backend;

    use super::{PROPOSAL_HEADER, proposal_file};
    use crate::candidates::Origin;
    use crate::measure::{Footprint, Mode};
    use crate::run::{Arbitrated, SweepReport};
    use crate::verdict::{Refusal, Verdict};

    fn footprint() -> Footprint {
        Footprint {
            shared_bytes: 0,
            lanes: 64,
            shared_limit_bytes: None,
            registers: None,
            spill_bytes: None,
        }
    }

    fn arbitrated(tag: &str, origin: Origin, pick: Option<Pick>, verdict: Verdict) -> Arbitrated {
        let defaults = (spaces::mul_mat_vec::space().defaults)(GpuBackend::Metal);
        Arbitrated {
            tag: tag.to_string(),
            source: "Schedule::gpu_subgroup(GpuBackend::Metal)".to_string(),
            origin,
            pick: pick.map(|p| (p, defaults)),
            footprint: footprint(),
            rows: Vec::new(),
            verdict,
        }
    }

    /// The file carries the retained picks with the verdict's claim, and
    /// neither the controls nor the refusals.
    #[test]
    fn the_proposal_file_carries_the_retained_picks_only() {
        let defaults = (spaces::mul_mat_vec::space().defaults)(GpuBackend::Metal);
        let cols16 = Pick {
            geometry: defaults.map_lanes(|l| l.lane_tile = 16),
            claim: Claim::any(),
            priority: 90,
            evidence: "",
            name: None,
        };
        let report = SweepReport {
            mode: Mode::Stream,
            kernel: "mul_mat_vec_q4_K".to_string(),
            family: "mul_mat_vec",
            profile: None,
            contraction: None,
            backend: Backend::Metal,
            device: "test".to_string(),
            base_tag: "table/fallback".to_string(),
            base_footprint: footprint(),
            shapes: Vec::new(),
            base: Vec::new(),
            arbitrated: vec![
                arbitrated(
                    "table/cols4",
                    Origin::Table,
                    None,
                    Verdict::Replaces { best_gain: 0.5 },
                ),
                arbitrated(
                    "cols16",
                    Origin::Proposed,
                    Some(cols16),
                    Verdict::Claims {
                        claim: Claim::any().on("plane", 9..=31),
                        best_gain: 0.25,
                        ties_claimed: Vec::new(),
                    },
                ),
                arbitrated(
                    "subgroup 32",
                    Origin::Proposed,
                    None,
                    Verdict::Refused(Refusal::NotMeasured),
                ),
            ],
            declined: Vec::new(),
        };
        let file = proposal_file(&[report]);
        assert!(file.starts_with(PROPOSAL_HEADER));
        assert!(file.contains("// cols16: up to 25%"));
        assert!(file.contains(
            "Pick { geometry: defaults.map_lanes(|l| l.lane_tile = 16), \
             claim: Claim::any().on(\"plane\", 9..=31), priority: 90, evidence: \"\", name: None },"
        ));
        assert!(!file.contains("table/cols4"));
        assert!(!file.contains("subgroup 32"));
        assert!(!file.contains("nothing retained"));
    }
}
