//! RIR-aware, pure execution-profile evaluation.

use std::collections::{BTreeMap, BTreeSet};

use retrograd_core::{
    BackendPolicy, ExecutionProfile, KernelPolicy, KernelVariantPolicy, PreflightReport, RirMode,
};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Confidence {
    #[default]
    Analytical,
    Declared,
    Preflighted,
    Measured,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct KernelCoverage {
    pub ggml_op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub op_variant: Option<String>,
    pub backend: String,
    pub policy: String,
    pub nodes: u64,
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub variant_id: Option<String>,
    pub native_retired: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct FallbackSummary {
    pub ggml_op: String,
    pub from_backend: String,
    pub to_backend: String,
    pub nodes: u64,
    pub bytes: u64,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Assumption {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct RejectedCandidateSummary {
    pub code: String,
    pub candidate: String,
    pub reason: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct ExecutionPlan {
    pub schema_version: u32,
    pub confidence: Confidence,
    pub profile_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub graph_fingerprint: Option<String>,
    pub expected_backend: String,
    pub rir_mode: String,
    pub kernel_coverage: Vec<KernelCoverage>,
    pub cpu_fallbacks: Vec<FallbackSummary>,
    pub assumptions: Vec<Assumption>,
    pub alternatives: Vec<RejectedCandidateSummary>,
}

impl ExecutionPlan {
    pub fn analytical(backend: &str) -> Self {
        Self {
            schema_version: 2,
            confidence: Confidence::Analytical,
            profile_fingerprint: "unavailable".to_string(),
            graph_fingerprint: None,
            expected_backend: backend.to_string(),
            rir_mode: "unknown".to_string(),
            kernel_coverage: Vec::new(),
            cpu_fallbacks: Vec::new(),
            assumptions: vec![Assumption {
                code: "execution_profile_unavailable".to_string(),
                message: "kernel and device capabilities were not supplied; execution claims are analytical only".to_string(),
            }],
            alternatives: Vec::new(),
        }
    }
}

fn rir_mode(mode: RirMode) -> &'static str {
    match mode {
        RirMode::Off => "off",
        RirMode::Observe => "observe",
        RirMode::Prefer => "prefer",
        RirMode::Require => "require",
    }
}

/// Builds the bounded execution explanation. A profile makes catalogue claims
/// `declared`; only a matching preflight is allowed to attach node/byte counts
/// or promise a graph placement.
pub fn explain_execution(
    profile: Option<&ExecutionProfile>,
    preflight: Option<&PreflightReport>,
    fallback_backend: &str,
) -> ExecutionPlan {
    let Some(profile) = profile else {
        return ExecutionPlan::analytical(fallback_backend);
    };
    let backend = profile.device.backend.clone();
    let fingerprint = profile.fingerprint();
    let matching_preflight = preflight.filter(|report| report.profile_fingerprint == fingerprint);
    let mut assumptions = Vec::new();
    if preflight.is_some() && matching_preflight.is_none() {
        assumptions.push(Assumption {
            code: "preflight_profile_mismatch".to_string(),
            message: "the supplied preflight belongs to another execution profile and was ignored"
                .to_string(),
        });
    }
    if matching_preflight.is_none() {
        assumptions.push(Assumption {
            code: "graph_not_preflighted".to_string(),
            message: "kernel coverage is declared by the catalogue; the concrete graph has not been inspected"
                .to_string(),
        });
    }

    // Keyed by the catalogue row's full identity, `op_variant` included: an op
    // published under several variants must not report each variant's counts as
    // every variant's counts.
    type CountKey = (String, Option<String>, String);
    let mut counts: BTreeMap<CountKey, (u64, u64, Option<String>)> = BTreeMap::new();
    if let Some(report) = matching_preflight {
        for row in &report.kernel_summary {
            counts.insert(
                (
                    row.ggml_op.clone(),
                    row.op_variant.clone(),
                    row.backend.clone(),
                ),
                (row.nodes, row.bytes, row.variant_id.clone()),
            );
        }
    }
    let mut kernel_coverage = profile
        .kernels
        .iter()
        // The profile's backend is the *hardware* report's spelling, which has
        // two values the catalogue has no row for (`blas`, `unknown`); comparing
        // through the published name is what keeps those from matching anything.
        .filter(|row| row.backend.name() == backend)
        .filter(|row| row.policy != BackendPolicy::NativeOnly || row.native_retired)
        .map(|row| {
            let (nodes, bytes, variant_id) = counts
                .get(&(
                    row.ggml_op.clone(),
                    row.op_variant.clone(),
                    row.backend.name().to_string(),
                ))
                .cloned()
                .unwrap_or((0, 0, None));
            KernelCoverage {
                ggml_op: row.ggml_op.clone(),
                op_variant: row.op_variant.clone(),
                backend: row.backend.name().to_string(),
                policy: row.policy.name().to_string(),
                nodes,
                bytes,
                variant_id,
                native_retired: row.native_retired,
            }
        })
        .collect::<Vec<_>>();
    kernel_coverage.sort_by(|a, b| {
        a.ggml_op
            .cmp(&b.ggml_op)
            .then(a.op_variant.cmp(&b.op_variant))
    });

    let cpu_fallbacks = matching_preflight
        .into_iter()
        .flat_map(|report| &report.placements)
        .filter(|placement| placement.backend == "cpu" && backend != "cpu")
        .map(|placement| FallbackSummary {
            ggml_op: placement.ggml_op.clone(),
            from_backend: backend.clone(),
            to_backend: "cpu".to_string(),
            nodes: placement.nodes,
            bytes: placement.bytes,
            reason: placement
                .rejection
                .clone()
                .unwrap_or_else(|| "runtime_placement".to_string()),
        })
        .collect();

    ExecutionPlan {
        schema_version: 2,
        confidence: if matching_preflight.is_some() {
            Confidence::Preflighted
        } else {
            Confidence::Declared
        },
        profile_fingerprint: fingerprint,
        graph_fingerprint: matching_preflight.map(|report| report.graph_fingerprint.clone()),
        expected_backend: backend,
        rir_mode: rir_mode(profile.runtime_policy.rir_mode).to_string(),
        kernel_coverage,
        cpu_fallbacks,
        assumptions,
        alternatives: Vec::new(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariantSelection<'a> {
    Selected(&'a KernelVariantPolicy),
    Rejected(&'static str),
}

/// Mirrors the generated registry's stable priority/shape selection for the
/// metadata the planner can know. Portable dtype/layout constraints remain a
/// preflight responsibility; shape rules, device features and workgroup limits
/// are evaluated here without backend-name guesses.
///
/// Nothing in the resolver calls this yet: `explain_execution` reports what the
/// catalogue *declares*, and per-node variant selection belongs to the preflight
/// that publishes `kernel_summary`. It is the planner-side half of that join,
/// kept beside the contract it mirrors so the two stay in step.
pub fn select_variant<'a>(
    policy: &'a KernelPolicy,
    axis_extents: &BTreeMap<String, u64>,
    device_features: &BTreeSet<String>,
    max_workgroup_size: Option<u32>,
) -> VariantSelection<'a> {
    if policy.policy == BackendPolicy::NativeOnly {
        return VariantSelection::Rejected("policy_native");
    }
    let mut saw_shape = false;
    let mut saw_feature = false;
    for variant in &policy.variants {
        let shape_matches = variant.eligible_when.iter().all(|rule| {
            let product = rule.axes.iter().try_fold(1u64, |product, axis| {
                axis_extents
                    .get(axis)
                    .and_then(|value| product.checked_mul(*value))
            });
            product.is_some_and(|value| value >= rule.min as u64 && value <= rule.max as u64)
        });
        if !shape_matches {
            saw_shape = true;
            continue;
        }
        // A requirement the profile does not list is *not* satisfied. Treating a
        // version gate (`metal>=3.1`, `vulkan>=1.3`) as met because it looks like
        // one would elect a variant the device cannot run; an unknown device
        // feature can only be a rejection.
        let features_match = variant
            .features
            .iter()
            .all(|feature| device_features.contains(feature));
        if !features_match {
            saw_feature = true;
            continue;
        }
        let workgroup = variant.workgroup.iter().copied().product::<u32>();
        if max_workgroup_size.is_some_and(|limit| workgroup > limit) {
            saw_feature = true;
            continue;
        }
        return VariantSelection::Selected(variant);
    }
    if saw_feature {
        VariantSelection::Rejected("missing_feature")
    } else if saw_shape {
        VariantSelection::Rejected("shape")
    } else {
        VariantSelection::Rejected("wrong_op")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use retrograd_core::{KernelShapeRule, KernelVariantPolicy};

    #[test]
    fn variant_selection_is_priority_ordered_and_shape_aware() {
        let variant = |id: &str, priority, min, max| KernelVariantPolicy {
            variant_id: id.into(),
            priority,
            eligible_when: vec![KernelShapeRule {
                axes: vec!["row".into()],
                min,
                max,
            }],
            workgroup: [32, 1, 1],
            ..Default::default()
        };
        let policy = KernelPolicy {
            policy: BackendPolicy::PreferGenerated,
            variants: vec![variant("narrow", 2, 1, 8), variant("wide", 1, 9, u32::MAX)],
            ..Default::default()
        };
        let mut extents = BTreeMap::from([("row".to_string(), 4)]);
        let features = BTreeSet::new();
        assert!(matches!(
            select_variant(&policy, &extents, &features, Some(64)),
            VariantSelection::Selected(v) if v.variant_id == "narrow"
        ));
        extents.insert("row".into(), 64);
        assert!(matches!(
            select_variant(&policy, &extents, &features, Some(64)),
            VariantSelection::Selected(v) if v.variant_id == "wide"
        ));
    }
}
