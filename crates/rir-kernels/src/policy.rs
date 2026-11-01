//! Per-variant policy overrides, read from `policy/variants.tsv`.
//!
//! The pair's policy (`IntegrationSpec::backend_policy`) is only a **ceiling**:
//! the effective policy of a variant at dispatch is `min(pair, row)`, so one
//! variant of a promoted pair can be `prefer_generated` while its siblings stay
//! `observe_generated` - measured, counted, never encoded. A variant absent
//! from the file carries the pair's policy, which is what makes the migration
//! of an existing table a no-op.
//!
//! The file is embedded with `include_str!` and parsed without a new
//! dependency: it is a table the generator reads, not a runtime input, so a
//! malformed row is a table defect and fails generation, not a runtime error.
//! `rir-gen` validates the parsed rows against the registry it is about to
//! emit (`PolicyError`), so a row naming a variant that does not exist, or a
//! policy above its pair's ceiling, is refused before any byte is written.

use rir_emit::{BackendPolicy, GgmlBackend, VariantPolicies};

/// The override table, verbatim.
pub const VARIANTS_TSV: &str = include_str!("../policy/variants.tsv");

/// One row of [`VARIANTS_TSV`]: a variant on one backend held at a policy, with
/// the measurement session that justifies it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VariantPolicyRow {
    pub variant_id: String,
    pub backend: GgmlBackend,
    pub policy: BackendPolicy,
    /// Identifier of the measurement session the row rests on; empty when the
    /// row only restates the pair's policy.
    pub session: String,
    pub note: String,
}

/// Parses the override table. Lines starting with `#` and empty lines are
/// skipped; every other line is `variant_id \t backend \t policy \t session
/// \t note`.
///
/// # Errors
///
/// A line that does not carry five fields, or a backend/policy spelling the
/// catalogue does not publish, is an error naming the line.
pub fn parse_variant_policies(text: &str) -> Result<Vec<VariantPolicyRow>, String> {
    let mut rows = Vec::new();
    let mut keys = std::collections::BTreeSet::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 5 {
            return Err(format!(
                "variants.tsv:{}: expected 5 tab-separated fields, got {}",
                lineno + 1,
                fields.len()
            ));
        }
        let backend = match fields[1] {
            "cpu" => GgmlBackend::Cpu,
            "cuda" => GgmlBackend::Cuda,
            "metal" => GgmlBackend::Metal,
            "vulkan" => GgmlBackend::Vulkan,
            other => {
                return Err(format!(
                    "variants.tsv:{}: unknown backend '{other}'",
                    lineno + 1
                ));
            }
        };
        if fields[0].is_empty() || !keys.insert((fields[0], backend)) {
            return Err(format!(
                "variants.tsv:{}: empty or duplicate variant '{}' on {}",
                lineno + 1,
                fields[0],
                backend.name()
            ));
        }
        let policy = ["native_only", "observe_generated", "prefer_generated"]
            .iter()
            .position(|name| *name == fields[2])
            .map(|i| match i {
                0 => BackendPolicy::NativeOnly,
                1 => BackendPolicy::ObserveGenerated,
                _ => BackendPolicy::PreferGenerated,
            })
            .ok_or_else(|| {
                format!(
                    "variants.tsv:{}: unknown policy '{}'",
                    lineno + 1,
                    fields[2]
                )
            })?;
        rows.push(VariantPolicyRow {
            variant_id: fields[0].to_string(),
            backend,
            policy,
            session: fields[3].to_string(),
            note: fields[4].to_string(),
        });
    }
    Ok(rows)
}

/// The parsed override table. The table is committed and read at compile time,
/// so a malformed row panics here like any other defect of a declared table
/// (`KernelId::build` does the same): the generator's own validation reports it
/// as a typed error first.
///
/// # Panics
///
/// If `policy/variants.tsv` does not parse.
pub fn variant_rows() -> Vec<VariantPolicyRow> {
    parse_variant_policies(VARIANTS_TSV).unwrap_or_else(|e| panic!("{e}"))
}

/// The parsed table as the lookup the emitters and `rir-gen promote` share:
/// a variant's override, or the pair's policy when the table does not name it.
/// Parsed once per call, so a caller resolving many rows builds it once.
pub fn variant_policies() -> VariantPolicies {
    VariantPolicies::new(
        variant_rows()
            .into_iter()
            .map(|r| (r.variant_id, r.backend, r.policy))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_committed_table_parses_and_names_known_spellings() {
        let rows = variant_rows();
        assert!(!rows.is_empty());
        for row in &rows {
            assert!(!row.variant_id.is_empty());
            // name() and the file spelling must not drift apart.
            assert_eq!(
                row.backend.name(),
                match row.backend {
                    GgmlBackend::Cpu => "cpu",
                    GgmlBackend::Cuda => "cuda",
                    GgmlBackend::Metal => "metal",
                    GgmlBackend::Vulkan => "vulkan",
                }
            );
            assert_eq!(
                row.policy.name(),
                match row.policy {
                    BackendPolicy::NativeOnly => "native_only",
                    BackendPolicy::ObserveGenerated => "observe_generated",
                    BackendPolicy::PreferGenerated => "prefer_generated",
                }
            );
        }
    }

    #[test]
    fn a_malformed_row_is_an_error_naming_its_line() {
        let err =
            parse_variant_policies("vid\tmetal\tobserve_generated\ts\tnote\textra").unwrap_err();
        assert!(err.contains("variants.tsv:1"), "{err}");
        let err = parse_variant_policies("vid\tmetal\tpromote\ts\tnote").unwrap_err();
        assert!(err.contains("unknown policy"), "{err}");
        let err = parse_variant_policies("vid\tswitch\tobserve_generated\ts\tnote").unwrap_err();
        assert!(err.contains("unknown backend"), "{err}");
    }

    #[test]
    fn an_absent_variant_carries_the_pair_policy() {
        assert_eq!(
            variant_policies().get(
                "no_such_variant",
                GgmlBackend::Metal,
                BackendPolicy::PreferGenerated
            ),
            BackendPolicy::PreferGenerated
        );
    }

    #[test]
    fn duplicate_variant_policies_cannot_silently_shadow_a_row() {
        let text = "vid\tmetal\tobserve_generated\ts\tnote\n\
                    vid\tmetal\tprefer_generated\ts\tnote\n";
        let err = parse_variant_policies(text).unwrap_err();
        assert!(
            err.contains("variants.tsv:2") && err.contains("duplicate"),
            "{err}"
        );
        assert!(parse_variant_policies("\tmetal\tobserve_generated\ts\tnote").is_err());
    }
}
