//! Rendering a sweep: the measured table, then the proposal.
//!
//! Two blocks and not one, because they are read at different moments. The
//! table is what a reader argues with - it has the same columns as the tables
//! already carries, so a row can be put beside a published
//! one. The proposal is what a reader *commits*: the constructor call and the
//! shape rule, in the vocabulary of `schedules_for`, so that adopting a verdict
//! is a paste and a lane run rather than a translation.

use std::fmt::Write as _;

use crate::candidates::rule_source;
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
        "# {} - median of the repetitions; a gain counts only above the two spreads added (floor 2%)",
        r.mode.name()
    );
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
    let _ = writeln!(out, "## proposal");
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
        Verdict::Replaces { best_gain } => format!(
            "{head}\n    REPLACES the fallback: wins by up to {:.0}%, and on no measured shape \
             does it lose by more than the noise.\n    \
             v.push({}.unnamed());",
            best_gain * 100.0,
            a.source
        ),
        Verdict::Claims {
            rules,
            best_gain,
            ties_claimed,
        } => {
            let list = rules
                .iter()
                .map(rule_source)
                .collect::<Vec<_>>()
                .join(",\n        ");
            let ties = if ties_claimed.is_empty() {
                String::new()
            } else {
                format!(
                    "\n    (the rule also claims {} shape(s) it only ties on: {:?})",
                    ties_claimed.len(),
                    ties_claimed
                )
            };
            format!(
                "{head}\n    CLAIMS a domain: up to {:.0}% on the shapes the rule covers.{ties}\n    \
                 v.push({}.claiming(90, vec![\n        {list},\n    ]));",
                best_gain * 100.0,
                a.source
            )
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
            rules, best_gain, ..
        } => format!(
            "up to {:.0}%, on the domain {}",
            best_gain * 100.0,
            rules
                .iter()
                .map(rule_source)
                .collect::<Vec<_>>()
                .join(" and ")
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
    if let Some(w) = f.workgroups_by_shared {
        let _ = write!(s, ", ≤{w} workgroups by shared budget");
    }
    match (f.registers, f.spill_bytes) {
        (Some(r), Some(sp)) => {
            let _ = write!(s, ", {r} regs, {sp} B spilled");
        }
        _ => s.push_str(", regs/spills not published by this backend"),
    }
    s
}
