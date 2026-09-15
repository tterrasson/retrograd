//! Where every resolved field came from, and why.
//!
//! A flat map keyed by dotted field path - the same path grammar as `params`,
//! `GET /v1/defaults` and the `PATCH` whitelist. Flat rather than nested on
//! purpose: a client matches a provenance entry against a parameter it sent by
//! comparing two strings, not by walking two trees.
//!
//! A value the client did not supply is `derived`, and its reason names the
//! *rule* - "12 400 examples: rank 16" - rather than a profile keyword.

use std::collections::BTreeMap;

use serde::Serialize;

/// How a field got its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub enum Source {
    /// The engine's own default, untouched.
    Default,
    /// Computed by the resolver from geometry, dataset or budget.
    Derived,
    /// Computed, then confirmed or corrected by a real measurement (phase 4).
    Measured,
    /// Supplied by the caller. Never re-derived.
    Override,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct Origin {
    pub source: Source,
    /// One sentence, required for `derived` and `measured`. Absent for a value
    /// that is simply the default or simply what the caller asked for - there is
    /// nothing to explain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// The provenance map. `BTreeMap` so the serialized bytes are the same for two
/// identical resolutions (invariant 3).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[serde(transparent)]
pub struct Provenance(pub BTreeMap<String, Origin>);

impl Provenance {
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an origin for `path`.
    ///
    /// An `Override` entry is sticky: once a field is the caller's, no later
    /// derivation may claim it. That is invariant 6 expressed where it is
    /// cheapest to enforce - a resolver bug shows up as a wrong provenance in a
    /// test rather than as a silently rewritten override in production.
    pub fn record(&mut self, path: impl Into<String>, source: Source, reason: Option<String>) {
        let path = path.into();
        if self
            .0
            .get(&path)
            .is_some_and(|origin| origin.source == Source::Override)
            && source != Source::Override
        {
            return;
        }
        self.0.insert(path, Origin { source, reason });
    }

    pub fn derived(&mut self, path: impl Into<String>, reason: impl Into<String>) {
        self.record(path, Source::Derived, Some(reason.into()));
    }

    /// A value that is simply the engine's own, recorded so the client can tell
    /// "nobody chose this" from "this field was never considered".
    pub fn defaulted(&mut self, path: impl Into<String>) {
        self.record(path, Source::Default, None);
    }

    /// A derivation that a real measurement has confirmed or
    /// corrected. Only reached through the calibration pass - nothing analytic
    /// may claim to have measured anything.
    pub fn measured(&mut self, path: impl Into<String>, reason: impl Into<String>) {
        self.record(path, Source::Measured, Some(reason.into()));
    }

    /// Re-labels every field the resolver derived as measured, keeping its
    /// reason and appending `note`.
    ///
    /// The calibration pass measures a *configuration*, not a field: what it
    /// confirms is the geometry and the levers together. Promoting each derived
    /// entry one by one - rather than a subset chosen by hand - is what keeps the
    /// provenance honest when a new lever is added later.
    pub fn promote_derived_to_measured(&mut self, note: &str) {
        for origin in self.0.values_mut() {
            if origin.source != Source::Derived {
                continue;
            }
            origin.source = Source::Measured;
            origin.reason = Some(match origin.reason.take() {
                Some(reason) => format!("{reason}; {note}"),
                None => note.to_string(),
            });
        }
    }

    pub fn overridden(&mut self, path: impl Into<String>) {
        self.record(path, Source::Override, None);
    }

    pub fn get(&self, path: &str) -> Option<&Origin> {
        self.0.get(path)
    }

    /// Whether the caller owns this field, i.e. no lever may touch it.
    pub fn is_locked(&self, path: &str) -> bool {
        self.get(path)
            .is_some_and(|origin| origin.source == Source::Override)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_override_is_never_overwritten_by_a_derivation() {
        let mut provenance = Provenance::new();
        provenance.overridden("training.n_ctx");
        provenance.derived("training.n_ctx", "P99 of the dataset");
        assert_eq!(
            provenance.get("training.n_ctx").unwrap().source,
            Source::Override
        );
        assert!(provenance.is_locked("training.n_ctx"));

        // A later, explicit override still wins - a caller may restate a field.
        provenance.record("training.n_ctx", Source::Override, None);
        assert_eq!(
            provenance.get("training.n_ctx").unwrap().source,
            Source::Override
        );
    }

    #[test]
    fn a_derived_field_carries_its_reason() {
        let mut provenance = Provenance::new();
        provenance.derived(
            "training.gradient_accumulation",
            "largest batch fitting the budget",
        );
        let origin = provenance.get("training.gradient_accumulation").unwrap();
        assert_eq!(origin.source, Source::Derived);
        assert_eq!(
            origin.reason.as_deref(),
            Some("largest batch fitting the budget")
        );
    }

    #[test]
    fn a_measurement_promotes_the_derivations_and_leaves_the_rest_alone() {
        let mut provenance = Provenance::new();
        provenance.derived(
            "training.gradient_accumulation",
            "largest batch fitting the budget",
        );
        provenance.defaulted("training.max_grad_norm");
        provenance.overridden("training.ctx");
        provenance.promote_derived_to_measured("confirmed by measurement");

        let batch = provenance.get("training.gradient_accumulation").unwrap();
        assert_eq!(batch.source, Source::Measured);
        assert_eq!(
            batch.reason.as_deref(),
            Some("largest batch fitting the budget; confirmed by measurement")
        );
        // An engine default was not derived and an override is the caller's:
        // neither becomes a measurement because a run happened to be measured.
        assert_eq!(
            provenance.get("training.max_grad_norm").unwrap().source,
            Source::Default
        );
        assert_eq!(
            provenance.get("training.ctx").unwrap().source,
            Source::Override
        );
    }

    #[test]
    fn the_map_serializes_with_stable_key_order() {
        let mut provenance = Provenance::new();
        provenance.derived("training.micro_batch", "b");
        provenance.derived("lora.rank", "a");
        let json = serde_json::to_string(&provenance).unwrap();
        assert!(json.starts_with(r#"{"lora.rank""#), "{json}");
        assert_eq!(json, serde_json::to_string(&provenance.clone()).unwrap());
    }
}
