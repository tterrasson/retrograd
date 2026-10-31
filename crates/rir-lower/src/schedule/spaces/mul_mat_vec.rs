//! The decode GEMV's geometry space and its production picks.

use rir_core::catalog::Regime;

use crate::schedule::{
    Claim, Dim, Family, Geometry, GpuBackend, MATRIX_COLS, Pick, Space, matrix_targets_for,
    partitions_subgroups,
};

/// The measurement session that justified the current rows.
const EVIDENCE: &str = "amidala-20261031T160408Z";

/// The GEMV's default geometry per backend: two rows per subgroup, two
/// subgroups where the hardware fixes the subgroup width, segments of
/// sixteen, runs of four, factored scales.
fn defaults(gpu: GpuBackend) -> Geometry {
    let subgroups = if partitions_subgroups(gpu) { 2 } else { 1 };
    Geometry::Lanes {
        lane_tile: 1,
        row_tile: 2,
        subgroups,
        segment: 16,
        reduce_run: 4,
        factor_scales: true,
        stage_b: false,
    }
}

/// The geometry space of the decode GEMV: the lane levers and the matrix tile,
/// at the values a sweep explores.
pub fn space() -> Space {
    Space {
        family: Family::MulMatVec,
        defaults,
        dims: &[
            Dim::LaneTile(&[1, 4, 8, 16]),
            Dim::RowTile(&[1, 2, 4]),
            Dim::Subgroups(&[1, 2, 4]),
            Dim::Segment(&[4, 16]),
            Dim::FactorScales,
            Dim::MatrixDepth(&[32, 64, 128]),
            Dim::MatrixPadding(&[0, 8]),
        ],
    }
}

/// The production lane picks on `gpu`. The fallback is the space's defaults,
/// implicit.
pub fn picks(gpu: GpuBackend) -> Vec<Pick> {
    let space = space();
    let defaults = (space.defaults)(gpu);
    let decode_max = Regime::DECODE_MAX_TOKENS;
    let steps = |min: u32, max: u32| Claim::any().on("plane", min..=max);
    vec![
        Pick {
            geometry: defaults.map_lanes(|l| l.lane_tile = 4),
            claim: steps(2, decode_max),
            priority: 90,
            evidence: EVIDENCE,
        },
        Pick {
            geometry: defaults.map_lanes(|l| l.lane_tile = 8),
            claim: steps(5, decode_max),
            priority: 95,
            evidence: EVIDENCE,
        },
    ]
}

/// The production matrix picks on `gpu`: empty outside the matrix targets.
pub fn matrix_picks(gpu: GpuBackend) -> Vec<Pick> {
    if !matrix_targets_for(Family::MulMatVec).any(|g| g == gpu) {
        return Vec::new();
    }
    let decode_max = Regime::DECODE_MAX_TOKENS;
    vec![Pick {
        geometry: Geometry::Matrix {
            depth: MATRIX_COLS,
            padding: 0,
        },
        claim: Claim::any().on("plane", 32..=decode_max),
        priority: 100,
        evidence: EVIDENCE,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::GpuBackend;

    /// The names the shipped artifacts carry are the derived ones: the
    /// fallback is unnamed, the column tiles are `cols4`/`cols8`, the matrix
    /// tile `mma` - on every backend the family schedules.
    #[test]
    fn the_gemv_picks_derive_the_names_the_artifacts_carry() {
        for gpu in [GpuBackend::Vulkan, GpuBackend::Metal] {
            let space = space();
            let defaults = (space.defaults)(gpu);
            assert_eq!(defaults.variant_name(defaults), None);
            let names: Vec<String> = space
                .picks(gpu)
                .iter()
                .chain(space.matrix_picks(gpu).iter())
                .map(|p| {
                    p.geometry
                        .variant_name(defaults)
                        .map(|n| n.to_string())
                        .unwrap_or_default()
                })
                .collect();
            if matrix_targets_for(Family::MulMatVec).any(|g| g == gpu) {
                assert_eq!(names, ["cols4", "cols8", "mma"]);
            } else {
                assert_eq!(names, ["cols4", "cols8"]);
            }
        }
    }
}
