//! The geometries a schedule's lane and matrix levers take, and the variant
//! name each one derives.
//!
//! A geometry is the **decision**; a [`Schedule`] is the lowering that carries
//! it. Keeping the two apart is what lets a table row be written as a pick -
//! geometry, claim, priority - and what lets the name be derived instead of
//! restated: the name spells the dimensions that differ from the defaults, so
//! two rows that differ cannot collide, and a row that *is* the defaults stays
//! the unnamed fallback.

use super::{GpuBackend, MATRIX_COLS, MATRIX_TILE, Schedule, VariantName};

/// One point in a family's geometry space.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum Geometry {
    /// The lane lowering: `gpu_subgroup` with the lane levers applied.
    Lanes {
        /// Columns of the second parallel axis one workgroup owns.
        lane_tile: u32,
        /// Rows of the first parallel axis one subgroup owns.
        row_tile: u32,
        /// Subgroups in one workgroup.
        subgroups: u32,
        /// Elements of the reduced axis one segment of a quantized read
        /// covers, or `0` for the format's.
        segment: u32,
        /// Consecutive elements one access reads from a dense operand.
        reduce_run: u32,
        /// Whether segmented sums factor the block scale out of their terms.
        factor_scales: bool,
        /// Stage the shared operand through workgroup memory.
        stage_b: bool,
    },
    /// The subgroup matrix product: `gpu_matrix_tile` with its padding.
    Matrix {
        /// Reduction-axis indices one staged tile covers.
        depth: u32,
        /// Extra shared words between operand rows.
        padding: u32,
    },
}

/// The lane dimensions of a [`Geometry::Lanes`], for the geometry space to
/// vary one at a time.
pub struct Lanes {
    pub lane_tile: u32,
    pub row_tile: u32,
    pub subgroups: u32,
    pub segment: u32,
    pub reduce_run: u32,
    pub factor_scales: bool,
    pub stage_b: bool,
}

impl Lanes {
    /// The lane dimensions of a plain `gpu_subgroup`: the reference a lane
    /// name is derived against when no space default applies.
    pub const NEUTRAL: Lanes = Lanes {
        lane_tile: 1,
        row_tile: 1,
        subgroups: 1,
        segment: 0,
        reduce_run: 1,
        factor_scales: false,
        stage_b: false,
    };
}

impl Geometry {
    /// The lane dimensions, when this is a lane geometry.
    pub fn lanes(&self) -> Option<Lanes> {
        match self {
            Geometry::Lanes {
                lane_tile,
                row_tile,
                subgroups,
                segment,
                reduce_run,
                factor_scales,
                stage_b,
            } => Some(Lanes {
                lane_tile: *lane_tile,
                row_tile: *row_tile,
                subgroups: *subgroups,
                segment: *segment,
                reduce_run: *reduce_run,
                factor_scales: *factor_scales,
                stage_b: *stage_b,
            }),
            Geometry::Matrix { .. } => None,
        }
    }

    pub(crate) const fn from_lanes(l: Lanes) -> Geometry {
        Geometry::Lanes {
            lane_tile: l.lane_tile,
            row_tile: l.row_tile,
            subgroups: l.subgroups,
            segment: l.segment,
            reduce_run: l.reduce_run,
            factor_scales: l.factor_scales,
            stage_b: l.stage_b,
        }
    }

    /// `self` with `f` applied to its lane dimensions; a matrix geometry is
    /// returned unchanged, since lane dimensions do not apply to it.
    pub fn map_lanes(self, f: impl FnOnce(&mut Lanes)) -> Geometry {
        match self.lanes() {
            Some(mut l) => {
                f(&mut l);
                Geometry::from_lanes(l)
            }
            None => self,
        }
    }

    /// The schedule this geometry lowers to on `gpu`, built from the `gpu_*`
    /// constructors and the `with_*` levers.
    ///
    /// The returned schedule carries whatever name the constructors gave it;
    /// naming a table row is [`Geometry::variant_name`]'s job, applied by the
    /// table on top of this.
    pub fn schedule(self, gpu: GpuBackend) -> Schedule {
        match self {
            Geometry::Lanes {
                lane_tile,
                row_tile,
                subgroups,
                segment,
                reduce_run,
                factor_scales,
                stage_b,
            } => {
                let mut s = Schedule::gpu_subgroup(gpu)
                    .with_reduce_run(reduce_run)
                    .with_segment(segment)
                    .with_row_tile(row_tile, subgroups);
                if lane_tile != 1 {
                    s = s.with_lane_tile(lane_tile);
                }
                if factor_scales {
                    s = s.with_factored_scales();
                }
                if stage_b {
                    s = s.with_staged_operand();
                }
                s
            }
            Geometry::Matrix { depth, padding } => {
                Schedule::gpu_matrix_tile(gpu, depth).with_matrix_padding(padding)
            }
        }
    }

    /// The variant name this geometry derives against `defaults`: `None` when
    /// it **is** the defaults - the fallback stays unnamed - otherwise the
    /// fragments that differ, in a fixed order: `cols{n}`, `rt{n}`, `sg{n}`,
    /// `seg{n}`, `run{n}`, `nofs`, `stb` for a lane geometry, then `mma` with
    /// `k{depth}` and `pad{n}` when they differ from the matrix variant's own
    /// defaults.
    pub fn variant_name(self, defaults: Geometry) -> Option<VariantName> {
        if self == defaults {
            return None;
        }
        let name = match self {
            Geometry::Lanes {
                lane_tile,
                row_tile,
                subgroups,
                segment,
                reduce_run,
                factor_scales,
                stage_b,
            } => {
                let d = defaults.lanes().unwrap_or(Lanes::NEUTRAL);
                let mut fragments: Vec<String> = Vec::new();
                if lane_tile != d.lane_tile {
                    fragments.push(format!("cols{lane_tile}"));
                }
                if row_tile != d.row_tile {
                    fragments.push(format!("rt{row_tile}"));
                }
                if subgroups != d.subgroups {
                    fragments.push(format!("sg{subgroups}"));
                }
                if segment != d.segment {
                    fragments.push(format!("seg{segment}"));
                }
                if reduce_run != d.reduce_run {
                    fragments.push(format!("run{reduce_run}"));
                }
                if factor_scales != d.factor_scales {
                    fragments.push("nofs".to_string());
                }
                if stage_b != d.stage_b {
                    fragments.push("stb".to_string());
                }
                if fragments.is_empty() {
                    // A lane geometry against a matrix default: the kinds
                    // differ, so the name says which kind it is.
                    fragments.push("lanes".to_string());
                }
                fragments.join("_")
            }
            Geometry::Matrix { depth, padding } => {
                let mut fragments = vec![MATRIX_TILE.to_string()];
                if depth != MATRIX_COLS {
                    fragments.push(format!("k{depth}"));
                }
                if padding != 0 {
                    fragments.push(format!("pad{padding}"));
                }
                fragments.join("_")
            }
        };
        Some(VariantName::from(name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedule::partitions_subgroups;

    /// The GEMV lane defaults: two rows per subgroup, two subgroups where the
    /// hardware fixes the subgroup width, segments of sixteen, runs of four,
    /// factored scales.
    fn gemv_defaults(gpu: GpuBackend) -> Geometry {
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

    #[test]
    fn the_defaults_stay_unnamed_and_each_lever_names_itself() {
        for gpu in [GpuBackend::Vulkan, GpuBackend::Metal] {
            let defaults = gemv_defaults(gpu);
            assert_eq!(defaults.variant_name(defaults), None);
            assert_eq!(
                defaults
                    .map_lanes(|l| l.lane_tile = 4)
                    .variant_name(defaults)
                    .map(|n| n.to_string()),
                Some("cols4".to_string())
            );
            assert_eq!(
                defaults
                    .map_lanes(|l| l.lane_tile = 8)
                    .variant_name(defaults)
                    .map(|n| n.to_string()),
                Some("cols8".to_string())
            );
            assert_eq!(
                Geometry::Matrix {
                    depth: MATRIX_COLS,
                    padding: 0
                }
                .variant_name(defaults)
                .map(|n| n.to_string()),
                Some("mma".to_string())
            );
        }
    }

    #[test]
    fn fragments_come_in_a_fixed_order_whatever_the_geometry() {
        let defaults = gemv_defaults(GpuBackend::Metal);
        let g = Geometry::Lanes {
            lane_tile: 8,
            row_tile: 4,
            subgroups: 4,
            segment: 4,
            reduce_run: 1,
            factor_scales: false,
            stage_b: true,
        };
        assert_eq!(
            g.variant_name(defaults).map(|n| n.to_string()),
            Some("cols8_rt4_sg4_seg4_run1_nofs_stb".to_string())
        );
        assert_eq!(
            Geometry::Matrix {
                depth: 64,
                padding: 8
            }
            .variant_name(defaults)
            .map(|n| n.to_string()),
            Some("mma_k64_pad8".to_string())
        );
    }

    #[test]
    fn a_stage_b_geometry_schedules_the_staged_operand() {
        let defaults = gemv_defaults(GpuBackend::Metal);
        assert!(!defaults.schedule(GpuBackend::Metal).stage_b());
        let g = defaults.map_lanes(|l| {
            l.lane_tile = 8;
            l.stage_b = true;
        });
        let s = g.schedule(GpuBackend::Metal);
        assert!(s.stage_b());
        // The constructor's name stands until the table derives the row's:
        // against the defaults the staged geometry spells `cols8_stb`.
        assert_eq!(s.variant(), Some("cols8"));
        assert_eq!(
            g.variant_name(defaults).map(|n| n.to_string()),
            Some("cols8_stb".to_string())
        );
    }

    #[test]
    fn a_geometry_schedules_through_the_existing_levers() {
        let defaults = gemv_defaults(GpuBackend::Metal);
        let s = defaults
            .map_lanes(|l| l.lane_tile = 8)
            .schedule(GpuBackend::Metal);
        assert_eq!(s.lane_tile(), 8);
        assert_eq!(s.row_tile(), 2);
        assert_eq!(s.subgroups(), 2);
        assert_eq!(s.segment(), 16);
        assert_eq!(s.reduce_run(), 4);
        assert!(s.factor_scales());
        assert_eq!(s.block(), [64, 1, 1]);
        assert_eq!(s.variant(), Some("cols8"));

        let m = Geometry::Matrix {
            depth: 64,
            padding: 8,
        }
        .schedule(GpuBackend::Metal);
        assert_eq!(m.tile_depth(), 64);
        assert_eq!(m.matrix_padding(), 8);
        assert_eq!(m.variant(), Some(MATRIX_TILE));
    }
}
