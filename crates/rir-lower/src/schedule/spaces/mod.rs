//! The geometry space of each family that has one.

pub mod mul_mat_vec;

use super::{Family, GpuBackend, Pick};

/// The production lane picks of `family` on `gpu`. A family without a space
/// carries its rows inline in the table.
pub fn picks(family: Family, gpu: GpuBackend) -> Vec<Pick> {
    match family {
        Family::MulMatVec => mul_mat_vec::picks(gpu),
        _ => Vec::new(),
    }
}

/// The production matrix picks of `family` on `gpu`.
pub fn matrix_picks(family: Family, gpu: GpuBackend) -> Vec<Pick> {
    match family {
        Family::MulMatVec => mul_mat_vec::matrix_picks(gpu),
        _ => Vec::new(),
    }
}
