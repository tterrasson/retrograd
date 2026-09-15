//! Mechanics shared by proactive defaults and memory rescue levers.

/// The common metadata of a resolver rule.
pub(crate) trait Rule {
    fn touches(&self) -> &'static [&'static str];

    fn is_locked_by(&self, is_locked: &dyn Fn(&str) -> bool) -> bool {
        self.touches().iter().all(|path| is_locked(path))
    }
}

/// The net configuration move made by one rule.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
pub struct AppliedRule {
    pub id: &'static str,
    pub from: String,
    pub to: String,
    pub note: &'static str,
    pub cost: &'static str,
}
