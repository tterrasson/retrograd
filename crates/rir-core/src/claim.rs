//! The shape claim a variant publishes, and the one Rust evaluator of it.
//!
//! A claim is what decides, per node, whether a specialization is a candidate
//! at all: the schedule table writes it, the manifest and the catalogue publish
//! it, the registry hands it to the C dispatcher, the sweep derives it from a
//! measurement and the planner mirrors the dispatcher with it. One type for all
//! of them, so that a claim cannot mean one thing where it is written and
//! another where it is evaluated.
//!
//! The arithmetic is the dispatcher's (`ggml_rir_variant_fits_shape`): extents
//! are positive, a product saturates at `u32::MAX` before it is compared, and
//! an axis that cannot be resolved fails the rule. The registry publishes test
//! vectors computed by [`Claim::holds`] so the C side can check it agrees.

use std::ops::{Bound, RangeBounds};

use serde::{Deserialize, Serialize};

/// `product(extent(axes)) ∈ [min, max]`, inclusive.
///
/// A product rather than a single extent because what decides between two
/// lowerings of the same kernel is rarely one dimension: the blocked scan wins
/// while the *total* number of rows leaves the GPU idle, and that total is
/// `row × plane × batch` - three dimensions the kernel keeps distinct so it can
/// address a view.
///
/// Axes are named, not indexed: the registry emitter resolves the name against
/// the kernel's axis table, so a rule naming an axis the kernel does not have
/// fails generation rather than constraining the wrong dimension.
#[derive(Clone, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShapeRule {
    #[serde(default)]
    pub axes: Vec<String>,
    pub min: u32,
    pub max: u32,
}

impl ShapeRule {
    /// The rule over `axes` whose interval is `range`. An open lower bound is
    /// 1 - an extent is never below it - and an open upper bound `u32::MAX`,
    /// which a saturated product always reaches.
    pub fn new(axes: &[&str], range: impl RangeBounds<u32>) -> Self {
        let (min, max) = interval(range);
        Self {
            axes: axes.iter().map(|a| (*a).to_string()).collect(),
            min,
            max,
        }
    }

    /// The dispatcher's product of this rule's extents: `None` when an axis
    /// does not resolve or an extent is zero, saturated at `u32::MAX`
    /// otherwise.
    fn product(&self, extent: &impl Fn(&str) -> Option<u64>) -> Option<u64> {
        let cap = u64::from(u32::MAX);
        self.axes.iter().try_fold(1u64, |product, axis| {
            let factor = extent(axis).filter(|e| *e > 0)?;
            Some(product.saturating_mul(factor).min(cap))
        })
    }

    fn holds(&self, extent: &impl Fn(&str) -> Option<u64>) -> bool {
        self.product(extent)
            .is_some_and(|p| p >= u64::from(self.min) && p <= u64::from(self.max))
    }
}

/// `[min, max]` from a range, with the open ends of [`ShapeRule::new`]. An
/// excluded end that leaves nothing is spelled as the empty interval `[1, 0]`,
/// which the schedule table refuses by name.
fn interval(range: impl RangeBounds<u32>) -> (u32, u32) {
    const EMPTY: (u32, u32) = (1, 0);
    let min = match range.start_bound() {
        Bound::Included(&lo) => lo,
        Bound::Excluded(&lo) => match lo.checked_add(1) {
            Some(lo) => lo,
            None => return EMPTY,
        },
        Bound::Unbounded => 1,
    };
    let max = match range.end_bound() {
        Bound::Included(&hi) => hi,
        Bound::Excluded(&hi) => match hi.checked_sub(1) {
            Some(hi) => hi,
            None => return EMPTY,
        },
        Bound::Unbounded => u32::MAX,
    };
    (min, max)
}

/// A conjunction of rules: a shape is claimed when every rule holds. Empty
/// claims every shape, which is what the pair's fallback publishes.
///
/// Serialized as the bare rule list, so every document that published
/// `eligible_when: [...]` reads unchanged.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Claim(pub Vec<ShapeRule>);

impl Claim {
    /// Every shape.
    pub fn any() -> Self {
        Self::default()
    }

    /// Adds `extent(axis) ∈ range`.
    #[must_use]
    pub fn on(self, axis: &str, range: impl RangeBounds<u32>) -> Self {
        self.product(&[axis], range)
    }

    /// Adds `Π extent(axes) ∈ range`.
    #[must_use]
    pub fn product(mut self, axes: &[&str], range: impl RangeBounds<u32>) -> Self {
        self.0.push(ShapeRule::new(axes, range));
        self
    }

    pub fn is_any(&self) -> bool {
        self.0.is_empty()
    }

    pub fn rules(&self) -> &[ShapeRule] {
        &self.0
    }

    /// Whether the shape whose extents `extent` answers is claimed.
    ///
    /// `extent` returns `None` for an axis it does not know, and the rule
    /// naming it then fails: an unresolved axis is never read as a match. An
    /// extent of zero fails too, and a product saturates at `u32::MAX`, so a
    /// rule open to `u32::MAX` holds however large the shape - the
    /// dispatcher's arithmetic exactly.
    pub fn holds(&self, extent: impl Fn(&str) -> Option<u64>) -> bool {
        self.0.iter().all(|rule| rule.holds(&extent))
    }

    /// Whether every shape `self` claims is also claimed by one of `others`.
    ///
    /// Exact when the rules involved, across `self` and `others`, group their
    /// axes into disjoint sets - one axis per rule being the usual case - since
    /// each group's product then varies independently of the others. When two
    /// distinct groups share an axis (`row × plane` beside `plane`) the answer
    /// would need the products' joint range, and it is `None`.
    pub fn covered_by(&self, others: &[&Claim]) -> Option<bool> {
        let mut groups: Vec<Vec<&str>> = Vec::new();
        for rule in self.0.iter().chain(others.iter().flat_map(|c| c.0.iter())) {
            let group = group_of(rule);
            if group.is_empty() || groups.contains(&group) {
                continue;
            }
            if groups
                .iter()
                .any(|g| g.iter().any(|axis| group.contains(axis)))
            {
                return None;
            }
            groups.push(group);
        }
        let Some(target) = Cell::of(self, &groups) else {
            return Some(true); // claims nothing, so nothing is left uncovered
        };
        let covers: Vec<Cell> = others.iter().filter_map(|c| Cell::of(c, &groups)).collect();
        Some(target.covered_by(&covers))
    }
}

/// A rule's axes as a set, the coordinate its product is: the same axes in
/// another order are the same product.
fn group_of(rule: &ShapeRule) -> Vec<&str> {
    let mut group: Vec<&str> = rule.axes.iter().map(String::as_str).collect();
    group.sort_unstable();
    group
}

/// A claim as an axis-aligned box over the products of `groups`, each side an
/// inclusive interval inside `[1, u32::MAX]` - the values a saturated product
/// takes.
struct Cell(Vec<(u64, u64)>);

impl Cell {
    /// `None` when the claim holds for no shape.
    fn of(claim: &Claim, groups: &[Vec<&str>]) -> Option<Cell> {
        let mut sides = vec![(1u64, u64::from(u32::MAX)); groups.len()];
        for rule in &claim.0 {
            let (lo, hi) = (u64::from(rule.min), u64::from(rule.max));
            let group = group_of(rule);
            if group.is_empty() {
                // A product of nothing is 1, whatever the shape.
                if !(lo..=hi).contains(&1) {
                    return None;
                }
                continue;
            }
            let at = groups.iter().position(|g| *g == group)?;
            let side = &mut sides[at];
            *side = (side.0.max(lo), side.1.min(hi));
        }
        sides.iter().all(|(lo, hi)| lo <= hi).then_some(Cell(sides))
    }

    fn contains(&self, point: &[u64]) -> bool {
        self.0
            .iter()
            .zip(point)
            .all(|((lo, hi), p)| (lo..=hi).contains(&p))
    }

    /// Whether the union of `covers` contains this box. Each side is cut at
    /// every bound a cover places inside it, so each resulting sub-box lies
    /// wholly inside or wholly outside each cover, and its lowest corner
    /// decides for all of it.
    fn covered_by(&self, covers: &[Cell]) -> bool {
        let cuts: Vec<Vec<u64>> = self
            .0
            .iter()
            .enumerate()
            .map(|(d, &(lo, hi))| {
                let mut starts = vec![lo];
                for c in covers {
                    let (clo, chi) = c.0[d];
                    for s in [clo, chi.saturating_add(1)] {
                        if s > lo && s <= hi {
                            starts.push(s);
                        }
                    }
                }
                starts.sort_unstable();
                starts.dedup();
                starts
            })
            .collect();
        let mut index = vec![0usize; cuts.len()];
        loop {
            let corner: Vec<u64> = index.iter().zip(&cuts).map(|(i, c)| c[*i]).collect();
            if !covers.iter().any(|c| c.contains(&corner)) {
                return false;
            }
            // The next sub-box, odometer order; done when every side wrapped.
            let mut d = 0;
            loop {
                if d == index.len() {
                    return true;
                }
                index[d] += 1;
                if index[d] < cuts[d].len() {
                    break;
                }
                index[d] = 0;
                d += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape<'a>(extents: &'a [(&'a str, u64)]) -> impl Fn(&str) -> Option<u64> + 'a {
        move |axis| extents.iter().find(|(a, _)| *a == axis).map(|(_, e)| *e)
    }

    #[test]
    fn a_rule_holds_on_the_product_of_its_axes() {
        let claim = Claim::any().product(&["row", "plane"], ..=32);
        assert!(claim.holds(shape(&[("row", 8), ("plane", 4)])));
        assert!(!claim.holds(shape(&[("row", 8), ("plane", 5)])));
    }

    #[test]
    fn a_claim_is_a_conjunction() {
        let claim = Claim::any().on("plane", 5..=32).on("col", 8192..);
        assert!(claim.holds(shape(&[("plane", 8), ("col", 8192)])));
        assert!(!claim.holds(shape(&[("plane", 8), ("col", 4096)])));
        assert!(!claim.holds(shape(&[("plane", 4), ("col", 8192)])));
        assert!(Claim::any().holds(shape(&[])));
    }

    #[test]
    fn an_unknown_axis_or_an_empty_extent_never_matches() {
        let claim = Claim::any().on("plane", ..);
        assert!(!claim.holds(shape(&[("row", 1)])));
        assert!(!claim.holds(shape(&[("plane", 0)])));
        assert!(
            Claim::any()
                .on("plane", 0..=4)
                .holds(shape(&[("plane", 1)]))
        );
    }

    #[test]
    fn a_product_saturates_instead_of_wrapping() {
        let big = 1u64 << 40;
        let open = Claim::any().product(&["row", "plane"], 2..);
        assert!(open.holds(shape(&[("row", big), ("plane", big)])));
        let bounded = Claim::any().product(&["row", "plane"], ..u32::MAX);
        assert!(!bounded.holds(shape(&[("row", big), ("plane", big)])));
    }

    #[test]
    fn ranges_take_the_open_ends_of_an_extent() {
        assert_eq!(ShapeRule::new(&["a"], ..=8), ShapeRule::new(&["a"], 1..=8));
        assert_eq!(ShapeRule::new(&["a"], 5..).max, u32::MAX);
        assert_eq!(ShapeRule::new(&["a"], 2..5).max, 4);
        assert_eq!(ShapeRule::new(&["a"], 0..=256).min, 0);
        let empty = ShapeRule::new(&["a"], (Bound::Excluded(u32::MAX), Bound::Unbounded));
        assert!(empty.min > empty.max);
        let empty = ShapeRule::new(&["a"], ..0);
        assert!(empty.min > empty.max);
    }

    #[test]
    fn coverage_is_exact_over_disjoint_axis_groups() {
        let wide = Claim::any().on("plane", 5..=32);
        let narrow = Claim::any().on("plane", 5..=8);
        assert_eq!(narrow.covered_by(&[&wide]), Some(true));
        assert_eq!(wide.covered_by(&[&narrow]), Some(false));

        // Two covers that only together leave no gap.
        let low = Claim::any().on("plane", 2..=4);
        let high = Claim::any().on("plane", 5..);
        let all = Claim::any().on("plane", 2..=64);
        assert_eq!(all.covered_by(&[&low, &high]), Some(true));
        assert_eq!(all.covered_by(&[&low]), Some(false));

        // Two dimensions: the corner neither cover reaches is found.
        let target = Claim::any().on("plane", 1..=8).on("col", 1..=8);
        let left = Claim::any().on("plane", 1..=4);
        let bottom = Claim::any().on("col", 1..=4);
        assert_eq!(target.covered_by(&[&left, &bottom]), Some(false));
        let rest = Claim::any().on("plane", 5..).on("col", 5..);
        assert_eq!(target.covered_by(&[&left, &bottom, &rest]), Some(true));

        // A product group is one coordinate when nothing else splits it.
        let rows = Claim::any().product(&["row", "plane", "batch"], ..=128);
        let fewer = Claim::any().product(&["batch", "row", "plane"], ..=32);
        assert_eq!(fewer.covered_by(&[&rows]), Some(true));

        // Nothing covers the fallback's unbounded claim but another one.
        assert_eq!(Claim::any().covered_by(&[&wide]), Some(false));
        assert_eq!(Claim::any().covered_by(&[&Claim::any()]), Some(true));
        assert_eq!(wide.covered_by(&[]), Some(false));
    }

    #[test]
    fn overlapping_axis_groups_are_not_decided() {
        let rows = Claim::any().product(&["row", "plane"], ..=32);
        let plane = Claim::any().on("plane", ..=4);
        assert_eq!(plane.covered_by(&[&rows]), None);
    }

    #[test]
    fn the_json_is_the_bare_rule_list() {
        let claim = Claim::any()
            .on("plane", 2..=32)
            .product(&["row", "col"], ..=64);
        let json = serde_json::to_string(&claim).unwrap();
        assert_eq!(
            json,
            r#"[{"axes":["plane"],"min":2,"max":32},{"axes":["row","col"],"min":1,"max":64}]"#
        );
        assert_eq!(serde_json::from_str::<Claim>(&json).unwrap(), claim);
        // Catalogues written without `axes` still read.
        let rule: ShapeRule = serde_json::from_str(r#"{"min":1,"max":1}"#).unwrap();
        assert!(rule.axes.is_empty());
        assert!(serde_json::from_str::<ShapeRule>(r#"{"min":1,"max":1,"x":0}"#).is_err());
    }
}
