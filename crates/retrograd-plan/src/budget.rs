//! Budgets: what the operator asked for, what the machine reports, and what is
//! actually spendable once the baseline and the safety margin are taken out.
//!
//! Nothing here touches a device. The measurement is an input
//! ([`MemoryBaseline`]), so the same arithmetic serves the HTTP server, the CLI
//! and a test that hands it made-up numbers.

use serde::{Deserialize, Serialize};

use retrograd_core::{Error, Result};

/// Default safety margin: a share of the budget, never below a floor. It absorbs
/// fragmentation and the allocations the cost model does not represent.
const DEFAULT_MARGIN_FRACTION: f64 = 0.05;
const DEFAULT_MARGIN_FLOOR_BYTES: u64 = 256 * 1024 * 1024;

/// A budget as the operator wrote it: all of the resource, an absolute size, or
/// a fraction of the total.
///
/// Untagged on purpose - `"all"`, `"6GiB"`, `12884901888` and `0.8` are all
/// natural spellings, and forcing a tag on a configuration line this short would
/// buy nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum BudgetRequest {
    #[default]
    All,
    Bytes(u64),
    /// Share of the reported total, in `(0, 1]`.
    Fraction(f64),
}

impl BudgetRequest {
    /// Parses the three accepted spellings. A bare number is bytes when it is an
    /// integer and a fraction when it is not, which is why `0.8` and `12GiB`
    /// can share one field without ambiguity.
    pub fn parse(text: &str) -> Result<Self> {
        let trimmed = text.trim();
        if trimmed.eq_ignore_ascii_case("all") {
            return Ok(Self::All);
        }
        if let Ok(bytes) = trimmed.parse::<u64>() {
            return Self::bytes(bytes);
        }
        if let Ok(fraction) = trimmed.parse::<f64>() {
            return Self::fraction(fraction);
        }
        let (number, unit) = trimmed.split_at(
            trimmed
                .find(|character: char| character.is_ascii_alphabetic())
                .ok_or_else(|| invalid_budget(trimmed))?,
        );
        let value: f64 = number.trim().parse().map_err(|_| invalid_budget(trimmed))?;
        let scale: f64 = match unit.trim().to_ascii_lowercase().as_str() {
            "b" => 1.0,
            "k" | "kb" | "kib" => 1024.0,
            "m" | "mb" | "mib" => 1024.0 * 1024.0,
            "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
            "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
            _ => return Err(invalid_budget(trimmed)),
        };
        if !(value.is_finite() && value > 0.0) {
            return Err(invalid_budget(trimmed));
        }
        Self::bytes((value * scale) as u64)
    }

    /// A zero budget is refused rather than stored: it would resolve to "nothing
    /// fits" for every configuration, which reads as a resolver bug rather than
    /// as the misconfiguration it is.
    fn bytes(value: u64) -> Result<Self> {
        if value == 0 {
            return Err(Error::invalid("a byte budget must be greater than zero"));
        }
        Ok(Self::Bytes(value))
    }

    fn fraction(value: f64) -> Result<Self> {
        if !(value.is_finite() && value > 0.0 && value <= 1.0) {
            return Err(Error::invalid(format!(
                "a fractional budget must be in (0, 1], got {value}"
            )));
        }
        Ok(Self::Fraction(value))
    }

    /// How the request reads back to the client, so the provenance can quote it.
    pub fn describe(self) -> String {
        match self {
            Self::All => "all".to_string(),
            Self::Bytes(bytes) => bytes.to_string(),
            Self::Fraction(fraction) => fraction.to_string(),
        }
    }

    /// The absolute number of bytes this request stands for, given a reported
    /// total. `None` when the platform reports no total for the resource.
    fn against(self, total: Option<u64>) -> Option<u64> {
        let total = total?;
        Some(match self {
            Self::All => total,
            Self::Bytes(bytes) => bytes.min(total),
            Self::Fraction(fraction) => (total as f64 * fraction) as u64,
        })
    }

    /// Resolves against a reported total and the memory already in use.
    ///
    /// `total` is `None` when the platform reports none (a CPU-only build has no
    /// device budget); the effective budget is then zero, which is the honest
    /// answer: nothing can be promised about a resource that cannot be measured.
    pub fn resolve(self, total: Option<u64>, baseline: u64, margin: MarginPolicy) -> Budget {
        self.resolve_under(None, total, baseline, margin)
    }

    /// Same, with a ceiling the request may not exceed - the server-wide budget
    /// a per-run `limits` entry is capped by. A request above the ceiling
    /// is lowered and [`Budget::capped`] says so, so the provenance can explain
    /// why the run got less than it asked for.
    pub fn resolve_under(
        self,
        ceiling: Option<BudgetRequest>,
        total: Option<u64>,
        baseline: u64,
        margin: MarginPolicy,
    ) -> Budget {
        let requested = self.describe();
        let Some(mut asked) = self.against(total) else {
            return Budget {
                requested,
                total_bytes: None,
                baseline_bytes: 0,
                margin_bytes: 0,
                effective_bytes: 0,
                capped: false,
            };
        };
        let mut capped = false;
        if let Some(limit) = ceiling.and_then(|ceiling| ceiling.against(total))
            && asked > limit
        {
            asked = limit;
            capped = true;
        }
        let margin_bytes = margin.for_budget(asked);
        Budget {
            requested,
            total_bytes: total,
            baseline_bytes: baseline,
            margin_bytes,
            effective_bytes: asked.saturating_sub(baseline).saturating_sub(margin_bytes),
            capped,
        }
    }
}

fn invalid_budget(text: &str) -> Error {
    Error::invalid(format!(
        "budget '{text}' is not 'all', a byte count ('6GiB', '4096'), or a fraction ('0.8')"
    ))
}

impl<'de> Deserialize<'de> for BudgetRequest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error as _;
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Spelling {
            Integer(u64),
            Float(f64),
            Text(String),
        }
        match Spelling::deserialize(deserializer)? {
            Spelling::Integer(bytes) => Self::bytes(bytes).map_err(D::Error::custom),
            Spelling::Float(fraction) => Self::fraction(fraction).map_err(D::Error::custom),
            Spelling::Text(text) => Self::parse(&text).map_err(D::Error::custom),
        }
    }
}

impl Serialize for BudgetRequest {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.describe())
    }
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for BudgetRequest {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        utoipa::openapi::ObjectBuilder::new()
            .description(Some(
                "'all', a byte count ('6GiB', 4096), or a fraction of the total (0.8)",
            ))
            .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::ToSchema for BudgetRequest {}

/// The safety margin rule: a share of the budget with a floor.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct MarginPolicy {
    pub fraction: f64,
    pub floor_bytes: u64,
}

impl Default for MarginPolicy {
    fn default() -> Self {
        Self {
            fraction: DEFAULT_MARGIN_FRACTION,
            floor_bytes: DEFAULT_MARGIN_FLOOR_BYTES,
        }
    }
}

impl MarginPolicy {
    fn for_budget(self, budget: u64) -> u64 {
        // Never larger than the budget itself: a floor bigger than a tiny budget
        // would otherwise underflow into "nothing fits, ever".
        (((budget as f64) * self.fraction) as u64)
            .max(self.floor_bytes)
            .min(budget)
    }

    pub fn validate(&self) -> Result<()> {
        if !(self.fraction.is_finite() && (0.0..1.0).contains(&self.fraction)) {
            return Err(Error::invalid(
                "margin fraction must be finite and in [0, 1)",
            ));
        }
        Ok(())
    }
}

/// The memory totals measured once, before the process allocated anything.
///
/// Plain data: whoever can read the device fills it in (the server does it at
/// startup with `retrograd-memory`), and this crate stays free of any FFI.
///
/// Taken *before* allocating on purpose: a device reading is device-wide, so it
/// contains the compositor and every other process. Measuring it later would
/// fold our own allocations into the baseline and double-count them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryBaseline {
    pub device_total: Option<u64>,
    pub device_used: u64,
    pub host_total: Option<u64>,
    pub host_used: u64,
    /// Device and host are the same physical pool (Metal, iGPU).
    pub unified: bool,
}

/// One resolved budget.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Budget {
    /// The request as the operator wrote it: `"all"`, `"6GiB"`, or `"0.8"`.
    pub requested: String,
    /// Absent when the platform does not report a total for this budget.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes: Option<u64>,
    /// Already in use before this process allocated anything.
    pub baseline_bytes: u64,
    /// Reserve for fragmentation and unmodelled allocations.
    pub margin_bytes: u64,
    /// `min(requested, ceiling, total) − baseline − margin`, floored at zero.
    pub effective_bytes: u64,
    /// The request was above the server-wide budget and was lowered to it.
    pub capped: bool,
}

/// The pair of budgets a resolution is checked against.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Budgets {
    pub vram: Budget,
    pub ram: Budget,
    /// Device and host memory are one pool, so the two budgets above describe
    /// the *same* bytes and the check is made against the smaller of them.
    /// Treating them as independent would let the resolver promise the
    /// model twice.
    pub unified: bool,
}

impl Budgets {
    /// Resolves both budgets, each optionally narrowed by a per-run limit.
    pub fn resolve(
        baseline: MemoryBaseline,
        server: (BudgetRequest, BudgetRequest),
        limits: (Option<BudgetRequest>, Option<BudgetRequest>),
        margin: MarginPolicy,
    ) -> Self {
        let (server_vram, server_ram) = server;
        let (limit_vram, limit_ram) = limits;
        Self {
            vram: limit_vram.unwrap_or(server_vram).resolve_under(
                Some(server_vram),
                baseline.device_total,
                baseline.device_used,
                margin,
            ),
            ram: limit_ram.unwrap_or(server_ram).resolve_under(
                Some(server_ram),
                baseline.host_total,
                baseline.host_used,
                margin,
            ),
            unified: baseline.unified,
        }
    }

    /// How far a pair of footprints exceeds these budgets, in bytes, per side.
    ///
    /// On unified memory the two footprints are summed and measured against one
    /// budget; the overflow is then reported on the device side, because that is
    /// the pool the levers act on.
    pub fn overflow(&self, device_bytes: u64, host_bytes: u64) -> Overflow {
        if self.unified {
            let budget = self.vram.effective_bytes.min(self.ram.effective_bytes);
            return Overflow {
                device: device_bytes
                    .saturating_add(host_bytes)
                    .saturating_sub(budget),
                host: 0,
            };
        }
        Overflow {
            device: device_bytes.saturating_sub(self.vram.effective_bytes),
            host: host_bytes.saturating_sub(self.ram.effective_bytes),
        }
    }
}

/// Bytes over budget, zero on each side that fits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Overflow {
    pub device: u64,
    pub host: u64,
}

impl Overflow {
    pub fn fits(self) -> bool {
        self.device == 0 && self.host == 0
    }

    pub fn total(self) -> u64 {
        self.device + self.host
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_spellings_parse_and_reject() {
        assert_eq!(BudgetRequest::parse("all").unwrap(), BudgetRequest::All);
        assert_eq!(BudgetRequest::parse("ALL").unwrap(), BudgetRequest::All);
        assert_eq!(
            BudgetRequest::parse("4096").unwrap(),
            BudgetRequest::Bytes(4096)
        );
        assert_eq!(
            BudgetRequest::parse("6GiB").unwrap(),
            BudgetRequest::Bytes(6 * 1024 * 1024 * 1024)
        );
        assert_eq!(
            BudgetRequest::parse(" 1.5 gb ").unwrap(),
            BudgetRequest::Bytes(1536 * 1024 * 1024)
        );
        assert_eq!(
            BudgetRequest::parse("0.8").unwrap(),
            BudgetRequest::Fraction(0.8)
        );
        for invalid in ["", "0", "0.0", "1.2", "-4", "6ZiB", "gib"] {
            assert!(
                BudgetRequest::parse(invalid).is_err(),
                "accepted '{invalid}'"
            );
        }
    }

    #[test]
    fn budget_resolution_subtracts_the_baseline_and_the_margin() {
        let margin = MarginPolicy {
            fraction: 0.05,
            floor_bytes: 0,
        };
        let total = 10_000_u64;
        let info = BudgetRequest::All.resolve(Some(total), 1_000, margin);
        assert_eq!(info.total_bytes, Some(total));
        assert_eq!(info.margin_bytes, 500);
        assert_eq!(info.effective_bytes, 10_000 - 1_000 - 500);
        assert!(!info.capped);

        // A request above the total is capped, not honoured.
        let capped = BudgetRequest::Bytes(1_000_000).resolve(Some(total), 0, margin);
        assert_eq!(capped.effective_bytes, 9_500);

        let half = BudgetRequest::Fraction(0.5).resolve(Some(total), 0, margin);
        assert_eq!(half.effective_bytes, 5_000 - 250);

        // No reported total means nothing can be promised.
        let unknown = BudgetRequest::All.resolve(None, 0, margin);
        assert_eq!(unknown.total_bytes, None);
        assert_eq!(unknown.effective_bytes, 0);
    }

    #[test]
    fn a_margin_floor_never_exceeds_the_budget_it_protects() {
        let margin = MarginPolicy {
            fraction: 0.05,
            floor_bytes: DEFAULT_MARGIN_FLOOR_BYTES,
        };
        let info = BudgetRequest::All.resolve(Some(1024), 0, margin);
        assert_eq!(info.margin_bytes, 1024);
        assert_eq!(info.effective_bytes, 0);
    }

    #[test]
    fn a_run_limit_never_rises_above_the_server_budget() {
        let margin = MarginPolicy {
            fraction: 0.0,
            floor_bytes: 0,
        };
        let baseline = MemoryBaseline {
            device_total: Some(10_000),
            host_total: Some(10_000),
            ..Default::default()
        };
        let budgets = Budgets::resolve(
            baseline,
            (BudgetRequest::Bytes(5_000), BudgetRequest::All),
            (Some(BudgetRequest::Bytes(9_000)), None),
            margin,
        );
        assert_eq!(budgets.vram.effective_bytes, 5_000);
        assert!(budgets.vram.capped, "the run asked above the server budget");
        // Asking for less than the server allows is honoured as written.
        let modest = Budgets::resolve(
            baseline,
            (BudgetRequest::Bytes(5_000), BudgetRequest::All),
            (Some(BudgetRequest::Bytes(2_000)), None),
            margin,
        );
        assert_eq!(modest.vram.effective_bytes, 2_000);
        assert!(!modest.vram.capped);
    }

    #[test]
    fn unified_memory_checks_one_pool_instead_of_two() {
        let margin = MarginPolicy {
            fraction: 0.0,
            floor_bytes: 0,
        };
        let baseline = MemoryBaseline {
            device_total: Some(10_000),
            host_total: Some(10_000),
            unified: true,
            ..Default::default()
        };
        let budgets = Budgets::resolve(
            baseline,
            (BudgetRequest::All, BudgetRequest::All),
            (None, None),
            margin,
        );
        // Six plus six fits neither pool once they are the same pool, even
        // though each side would fit its own 10 000 on its own.
        assert!(!budgets.overflow(6_000, 6_000).fits());
        assert_eq!(budgets.overflow(6_000, 6_000).device, 2_000);
        assert!(budgets.overflow(6_000, 3_000).fits());

        let split = Budgets::resolve(
            MemoryBaseline {
                unified: false,
                ..baseline
            },
            (BudgetRequest::All, BudgetRequest::All),
            (None, None),
            margin,
        );
        assert!(split.overflow(6_000, 6_000).fits());
    }
}
