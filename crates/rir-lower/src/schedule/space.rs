//! The geometry space a family's table picks from, and the picks themselves.
//!
//! A family with a space states its defaults per backend and the dimensions a
//! sweep varies; the production table then carries **picks** - geometry,
//! claim, priority, evidence - instead of constructor calls with the numbers
//! inline, and the sweep explores the same space the table picks from, so a
//! proposal is written in the vocabulary it will be committed in.

use rir_core::ValidatedKernel;

use super::geometry::Lanes;
use super::{Claim, Family, Geometry, GpuBackend, MATRIX_COLS, Schedule};

/// The geometry space of one family.
pub struct Space {
    pub family: Family,
    /// The family's default geometry per backend: the fallback, and the
    /// reference every variant name is derived against.
    pub defaults: fn(GpuBackend) -> Geometry,
    /// The dimensions a sweep varies.
    pub dims: &'static [Dim],
}

/// One dimension of a [`Space`]: the values a sweep varies it over.
pub enum Dim {
    LaneTile(&'static [u32]),
    RowTile(&'static [u32]),
    Subgroups(&'static [u32]),
    Segment(&'static [u32]),
    FactorScales,
    StageB,
    MatrixDepth(&'static [u32]),
    MatrixPadding(&'static [u32]),
}

/// A pick: what enters the production table - a geometry, the claim that
/// arbitrates it, its priority, and the measurement that justifies the row.
#[derive(Clone, Debug)]
pub struct Pick {
    pub geometry: Geometry,
    pub claim: Claim,
    pub priority: u8,
    /// Identifier of the measurement session that justifies the row. Published
    /// nowhere: it is the evidence a reader asks for, not part of any artifact.
    pub evidence: &'static str,
}

impl Space {
    /// The pair's fallback on `gpu`: the defaults, unnamed.
    pub fn fallback(&self, gpu: GpuBackend) -> Schedule {
        (self.defaults)(gpu).schedule(gpu).unnamed()
    }

    /// A pick as a table row on `gpu`: the geometry scheduled, named by
    /// [`Geometry::variant_name`] against the defaults, claiming its rule at
    /// its priority.
    pub fn pick(&self, gpu: GpuBackend, pick: &Pick) -> Schedule {
        let defaults = (self.defaults)(gpu);
        let s = pick.geometry.schedule(gpu);
        let s = match pick.geometry.variant_name(defaults) {
            Some(name) => s.as_variant(name),
            None => s.unnamed(),
        };
        s.claiming(pick.priority, pick.claim.clone())
    }

    /// The lane picks of this space on `gpu`, in table order.
    pub fn picks(&self, gpu: GpuBackend) -> Vec<Pick> {
        super::spaces::picks(self.family, gpu)
    }

    /// The matrix picks of this space on `gpu`, in table order.
    pub fn matrix_picks(&self, gpu: GpuBackend) -> Vec<Pick> {
        super::spaces::matrix_picks(self.family, gpu)
    }

    /// Every geometry of the space on `gpu` that `lower` admits for `kernel`:
    /// the cartesian product of the lane dimensions around the defaults, plus
    /// the matrix product, each filtered by the lowering itself - a geometry
    /// the compiler refuses for this backend or format is not emitted, and
    /// never a copy of its refusal rules.
    pub fn expand(&self, gpu: GpuBackend, kernel: &ValidatedKernel) -> Vec<Geometry> {
        let defaults = (self.defaults)(gpu);
        let mut lane_geos = vec![defaults];
        let mut depths: Option<&'static [u32]> = None;
        let mut paddings: Option<&'static [u32]> = None;
        for dim in self.dims {
            match dim {
                Dim::LaneTile(vals) => lane_geos = cross(lane_geos, vals, |l, v| l.lane_tile = v),
                Dim::RowTile(vals) => lane_geos = cross(lane_geos, vals, |l, v| l.row_tile = v),
                Dim::Subgroups(vals) => lane_geos = cross(lane_geos, vals, |l, v| l.subgroups = v),
                Dim::Segment(vals) => lane_geos = cross(lane_geos, vals, |l, v| l.segment = v),
                Dim::FactorScales => {
                    lane_geos = cross(lane_geos, &[false, true], |l, v| l.factor_scales = v)
                }
                Dim::StageB => lane_geos = cross(lane_geos, &[false, true], |l, v| l.stage_b = v),
                Dim::MatrixDepth(vals) => depths = Some(vals),
                Dim::MatrixPadding(vals) => paddings = Some(vals),
            }
        }
        let mut matrix_geos = Vec::new();
        if depths.is_some() || paddings.is_some() {
            let depths = depths.unwrap_or(&[MATRIX_COLS]);
            let paddings = paddings.unwrap_or(&[0]);
            for &depth in depths {
                for &padding in paddings {
                    matrix_geos.push(Geometry::Matrix { depth, padding });
                }
            }
        }
        lane_geos
            .into_iter()
            .chain(matrix_geos)
            .filter(|g| crate::lower(kernel, g.schedule(gpu)).is_ok())
            .collect()
    }
}

/// The cartesian product of `geos` with one lane dimension varied over
/// `values`.
fn cross<T: Copy>(
    geos: Vec<Geometry>,
    values: &[T],
    set: impl Fn(&mut Lanes, T),
) -> Vec<Geometry> {
    let mut out = Vec::with_capacity(geos.len() * values.len());
    for g in geos {
        for &v in values {
            out.push(g.map_lanes(|l| set(l, v)));
        }
    }
    out
}
