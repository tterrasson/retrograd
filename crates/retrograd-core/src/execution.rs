//! Pure, versioned execution contracts shared by the runtime and planner.
//!
//! These types intentionally contain no shader, Loop IR, FFI handle, or runtime
//! object.  A producer that can inspect the engine fills them; consumers can
//! serialize, cache and reason about the resulting snapshot without loading a
//! model or touching a device.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use rir_core::catalog::{KernelCatalog, KernelPolicy};

use crate::{MemoryReport, RirMode};

pub const EXECUTION_PROFILE_VERSION: u32 = 2;
pub const PREFLIGHT_REPORT_VERSION: u32 = 2;

/// The catalogue's schema version, re-exported under the applicative side's name.
///
/// The number itself is `rir_core::catalog::SCHEMA_VERSION`: the generator that
/// writes it and the planner that refuses it read the same constant.
pub use rir_core::catalog::SCHEMA_VERSION as KERNEL_CATALOG_VERSION;

crate::wire_enum! {
    /// Which of the three versioned contracts a [`ContractError`] speaks about.
    ///
    /// The variants are the three types below, and the words beside them are
    /// the ones their messages already used - the name is carried apart from
    /// the message so a caller can tell "the catalogue is stale" from "the
    /// profile is" without reading the sentence.
    ///
    /// Nothing serializes it, so this uses the macro's label form (`enum Name
    /// { … }`, no `: serde`).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Contract {
        KernelCatalog = "kernel catalog",
        ExecutionProfile = "execution profile",
        PreflightReport = "preflight report",
    }
}

/// Kept, and not a candidate for the `thiserror` rule `AGENTS.md` states: `Contract` is not
/// an error, it is a noun interpolated into one (`#[error("unsupported
/// {contract} schema …")]`), and `thiserror` needs the `Display` to do it.
impl std::fmt::Display for Contract {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Why a snapshot cannot be trusted.
///
/// A schema this build does not know is a cache to discard and recompute; an
/// empty fingerprint is a producer bug. Both cases have distinct typed errors.
///
/// There is deliberately no `From<ContractError> for crate::Error`: the three
/// consumers each wrap it in their own typed error - `ResolveError::Invalid`,
/// `ApiError::internal` - and none flattens it into the facade. A conversion
/// nobody calls would be code, not a translation.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    /// The snapshot was written by another build of this contract. Recomputing
    /// it is the fix; nothing about the current one is readable.
    #[error("unsupported {contract} schema {found}; expected {expected}")]
    Schema {
        contract: Contract,
        found: u32,
        expected: u32,
    },
    /// An identity field is blank, so the snapshot cannot be matched to what it
    /// describes. The field is named because a profile carries two of them and
    /// "fingerprints must not be empty" never said which.
    #[error("{contract} {field} must not be empty")]
    MissingFingerprint {
        contract: Contract,
        field: &'static str,
    },
}

impl ContractError {
    fn schema(contract: Contract, found: u32, expected: u32) -> Self {
        Self::Schema {
            contract,
            found,
            expected,
        }
    }

    fn missing(contract: Contract, field: &'static str) -> Self {
        Self::MissingFingerprint { contract, field }
    }

    /// The contract the error is about, for a caller that reports on one
    /// snapshot at a time.
    pub fn contract(&self) -> Contract {
        match self {
            Self::Schema { contract, .. } | Self::MissingFingerprint { contract, .. } => *contract,
        }
    }
}

/// Validation of the catalogue, which is declared in `rir-core`.
///
/// It is an extension trait for the reason `rir_runtime::ManifestReader` is one
/// a type declared in another crate takes no inherent `impl`. The split
/// is also the right one - `rir-core` says what a catalogue *is*, and this crate
/// says what makes a cached snapshot untrustworthy, next to the other two
/// contracts that answer the same question.
pub trait CatalogContract {
    fn validate(&self) -> Result<(), ContractError>;
}

impl CatalogContract for KernelCatalog {
    fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != KERNEL_CATALOG_VERSION {
            return Err(ContractError::schema(
                Contract::KernelCatalog,
                self.schema_version,
                KERNEL_CATALOG_VERSION,
            ));
        }
        if self.fingerprint.is_empty() {
            return Err(ContractError::missing(
                Contract::KernelCatalog,
                "fingerprint",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionProfile {
    pub schema_version: u32,
    pub engine_fingerprint: String,
    pub kernel_catalog_fingerprint: String,
    pub device: DeviceProfile,
    pub runtime_policy: RuntimePolicy,
    #[serde(default)]
    pub kernels: Vec<KernelPolicy>,
    pub capabilities: ModelCapabilities,
}

impl ExecutionProfile {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != EXECUTION_PROFILE_VERSION {
            return Err(ContractError::schema(
                Contract::ExecutionProfile,
                self.schema_version,
                EXECUTION_PROFILE_VERSION,
            ));
        }
        if self.engine_fingerprint.is_empty() {
            return Err(ContractError::missing(
                Contract::ExecutionProfile,
                "engine_fingerprint",
            ));
        }
        if self.kernel_catalog_fingerprint.is_empty() {
            return Err(ContractError::missing(
                Contract::ExecutionProfile,
                "kernel_catalog_fingerprint",
            ));
        }
        Ok(())
    }

    pub fn fingerprint(&self) -> String {
        format!(
            "{}:{}:{}",
            self.engine_fingerprint, self.kernel_catalog_fingerprint, self.device.stable_id
        )
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceProfile {
    pub stable_id: String,
    pub backend: String,
    pub unified_memory: bool,
    pub device_total_bytes: Option<u64>,
    pub host_total_bytes: Option<u64>,
    #[serde(default)]
    pub features: Vec<String>,
    pub limits: DeviceLimits,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceLimits {
    pub max_workgroup_size: Option<u32>,
    pub max_workgroup_dimensions: Option<[u32; 3]>,
    pub shared_memory_bytes: Option<u64>,
    pub storage_buffer_alignment: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePolicy {
    pub rir_mode: RirMode,
    pub latched: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelCapabilities {
    pub shared_prefix_packed_training: bool,
    pub fused_sparse_cross_entropy: bool,
    pub differentiable_flash_attention: bool,
    /// Additional model/device-specific facts with stable names. A sorted map
    /// preserves byte-identical serialization.
    pub extra: BTreeMap<String, bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreflightReport {
    pub schema_version: u32,
    pub profile_fingerprint: String,
    pub graph_fingerprint: String,
    #[serde(default)]
    pub placements: Vec<OpPlacement>,
    #[serde(default)]
    pub kernel_summary: Vec<KernelSelectionSummary>,
    pub memory: MemoryReport,
    #[serde(default)]
    pub warnings: Vec<PreflightWarning>,
}

impl PreflightReport {
    pub fn validate(&self) -> Result<(), ContractError> {
        if self.schema_version != PREFLIGHT_REPORT_VERSION {
            return Err(ContractError::schema(
                Contract::PreflightReport,
                self.schema_version,
                PREFLIGHT_REPORT_VERSION,
            ));
        }
        if self.profile_fingerprint.is_empty() {
            return Err(ContractError::missing(
                Contract::PreflightReport,
                "profile_fingerprint",
            ));
        }
        if self.graph_fingerprint.is_empty() {
            return Err(ContractError::missing(
                Contract::PreflightReport,
                "graph_fingerprint",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct OpPlacement {
    pub ggml_op: String,
    pub op_variant: Option<String>,
    pub shape_class: String,
    pub nodes: u64,
    pub elements: u64,
    pub bytes: u64,
    pub backend: String,
    pub implementation: String,
    pub variant_id: Option<String>,
    pub rejection: Option<String>,
    pub native_fallback: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KernelSelectionSummary {
    pub ggml_op: String,
    /// Same discriminant as [`KernelPolicy::op_variant`]. Without it a graph
    /// summary cannot be joined to the catalogue row it belongs to, and an op
    /// published under several variants would report each one's counts as all of
    /// them.
    pub op_variant: Option<String>,
    pub backend: String,
    pub nodes: u64,
    pub bytes: u64,
    pub implementation: String,
    pub variant_id: Option<String>,
    #[serde(default)]
    pub rejected_by_reason: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct PreflightWarning {
    pub code: String,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_contracts_round_trip_strictly() {
        let profile = ExecutionProfile {
            schema_version: EXECUTION_PROFILE_VERSION,
            engine_fingerprint: "engine-a".into(),
            kernel_catalog_fingerprint: "catalog-a".into(),
            device: DeviceProfile {
                stable_id: "metal:fixture".into(),
                backend: "metal".into(),
                unified_memory: true,
                ..Default::default()
            },
            runtime_policy: RuntimePolicy {
                rir_mode: RirMode::Prefer,
                latched: true,
            },
            capabilities: ModelCapabilities {
                shared_prefix_packed_training: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let json = serde_json::to_string(&profile).unwrap();
        assert_eq!(
            serde_json::from_str::<ExecutionProfile>(&json).unwrap(),
            profile
        );
        assert!(profile.validate().is_ok());

        let mut value = serde_json::to_value(&profile).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .insert("future".into(), true.into());
        assert!(serde_json::from_value::<ExecutionProfile>(value).is_err());
    }

    #[test]
    fn validation_says_which_contract_and_which_field() {
        let stale = KernelCatalog {
            schema_version: KERNEL_CATALOG_VERSION + 1,
            fingerprint: "catalog-a".into(),
            ..Default::default()
        };
        let error = stale.validate().unwrap_err();
        assert_eq!(error.contract(), Contract::KernelCatalog);
        assert!(
            matches!(error, ContractError::Schema { found, .. } if found == KERNEL_CATALOG_VERSION + 1)
        );

        // The two fingerprints of a profile were one message; each is now named,
        // which is the whole point of the variant carrying the field.
        let mut profile = ExecutionProfile {
            schema_version: EXECUTION_PROFILE_VERSION,
            engine_fingerprint: String::new(),
            kernel_catalog_fingerprint: "catalog-a".into(),
            ..Default::default()
        };
        assert_eq!(
            profile.validate().unwrap_err(),
            ContractError::MissingFingerprint {
                contract: Contract::ExecutionProfile,
                field: "engine_fingerprint",
            }
        );
        profile.engine_fingerprint = "engine-a".into();
        profile.kernel_catalog_fingerprint = String::new();
        assert_eq!(
            profile.validate().unwrap_err().to_string(),
            "execution profile kernel_catalog_fingerprint must not be empty"
        );

        let report = PreflightReport {
            schema_version: PREFLIGHT_REPORT_VERSION,
            profile_fingerprint: "profile-a".into(),
            graph_fingerprint: String::new(),
            ..Default::default()
        };
        assert_eq!(
            report.validate().unwrap_err().contract(),
            Contract::PreflightReport
        );
    }
}
