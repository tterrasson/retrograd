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
            Dim::StageB,
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

    /// The decode GEMV's kernel, small enough to build in a test: a quantized
    /// weight and a dense operand read along the reduced axis.
    fn gemv_kernel() -> rir_core::ValidatedKernel {
        use rir_core::{
            Constraint, DType, Extent, KernelBuilder, Layout, QuantType, ReduceOp,
            ReductionSemantics, TensorType,
        };
        let format = QuantType::Q8_0;
        let mut k = KernelBuilder::new("gemv");
        let a = k.input(
            "a",
            TensorType {
                dtype: DType::Quant(format),
                rank: 2,
                layout: Layout::Ggml,
            },
        );
        let b = k.input("b", TensorType::f32_2d());
        let dst = k.output("dst", TensorType::f32_2d());
        let row = k.axis("row", Extent::Dim { arg: a, dim: 1 });
        let plane = k.axis("plane", Extent::Dim { arg: b, dim: 1 });
        let col = k.axis("col", Extent::Dim { arg: a, dim: 0 });
        let av = k.read(a, &[col, row]);
        let bv = k.read(b, &[col, plane]);
        let p = k.mul(av, bv);
        let s = k.reduce(ReduceOp::Sum, col, p, ReductionSemantics::Deterministic);
        k.write(dst, &[row, plane], s);
        k.constrain(Constraint::DType {
            arg: a,
            allowed: vec![DType::Quant(format)],
        });
        k.constrain(Constraint::DType {
            arg: b,
            allowed: vec![DType::F32],
        });
        k.finish().unwrap()
    }

    /// The space expands staged geometries on the partition targets and on
    /// them only: `stage_b` is refused elsewhere, so `expand` - which filters
    /// by what `lower` admits - carries it there alone.
    #[test]
    fn the_space_expands_staged_geometries_on_partition_targets_only() {
        let kernel = gemv_kernel();
        let space = space();
        for gpu in [GpuBackend::Metal, GpuBackend::Vulkan] {
            let staged = space
                .expand(gpu, &kernel)
                .into_iter()
                .filter(|g| g.lanes().is_some_and(|l| l.stage_b))
                .count();
            if crate::schedule::partitions_subgroups(gpu) {
                assert!(staged > 0, "{}: no staged geometry expanded", gpu.name());
            } else {
                assert_eq!(staged, 0, "{}: a staged geometry expanded", gpu.name());
            }
        }
    }

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
