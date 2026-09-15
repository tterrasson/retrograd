//! Host-side storage for the PPO critic's feature matrix.
//!
//! The critic keeps one row of `hidden_dim` features per completion state for a
//! whole update: `batch_advantages` fills the matrix from the fused scoring pass,
//! reads it once for the pre-fit predictions, then reads it `value_epochs` more
//! times inside full-batch Adam. The buffer is therefore
//! `total_completion_states * hidden_dim` and it is the PPO run's largest host
//! allocation.
//!
//! The design is **store the rows narrow, compute wide**. A [`FeatureStore`] holds them in the requested
//! [`FeatureDtype`] and hands out F32 chunks on demand, so every consumer still
//! sees `&[f32]` rows and no arithmetic changed - only the stored value is
//! rounded. What it deliberately does not do is change the order of the update:
//! predictions with the pre-fit critic, GAE, then the fit. The alternatives - a
//! streamed two-pass Adam, per-epoch re-scoring - change that order or pay the
//! forwards again, and neither is implemented.

use retrograd_core::{Error, FeatureDtype, Result};

/// Rows converted per chunk. The chunk exists to bound the F32 scratch, not to
/// tile the arithmetic: 4096 rows of a 4096-wide model is 64 MiB of scratch,
/// well under the matrix it replaces, and large enough that the per-chunk
/// bookkeeping is lost in the conversion loop.
const CHUNK_ROWS: usize = 4096;

/// One update's feature matrix, row-major, `dim` features per row.
#[derive(Clone, Debug)]
pub struct FeatureStore {
    dim: usize,
    dtype: FeatureDtype,
    /// F32 rows, used verbatim when `dtype` is [`FeatureDtype::F32`] and empty
    /// otherwise. Keeping the two representations in separate fields is what
    /// makes the F32 path cost exactly what an unconverted buffer costs:
    /// no conversion, no copy, the same `Vec<f32>` handed straight to the fit.
    wide: Vec<f32>,
    /// 16-bit rows, used when `dtype` is narrow and empty otherwise.
    narrow: Vec<u16>,
    /// Decode scratch, reused across chunks and across epochs.
    scratch: Vec<f32>,
}

impl FeatureStore {
    /// An empty store for `dim`-wide rows, sized for `rows` of them.
    ///
    /// `rows` is a capacity hint, not a bound: the caller knows the exact row
    /// count up front (it is the sum of the completion lengths) and reserving it
    /// once is what keeps the fill from re-allocating mid-update.
    pub fn with_capacity(dim: usize, dtype: FeatureDtype, rows: usize) -> Result<Self> {
        let elements = rows
            .checked_mul(dim)
            .ok_or_else(|| Error::overflow("critic feature size overflows usize"))?;
        let (wide, narrow) = match dtype {
            FeatureDtype::F32 => (Vec::with_capacity(elements), Vec::new()),
            FeatureDtype::F16 | FeatureDtype::Bf16 => (Vec::new(), Vec::with_capacity(elements)),
        };
        Ok(Self {
            dim,
            dtype,
            wide,
            narrow,
            scratch: Vec::new(),
        })
    }

    pub fn dtype(&self) -> FeatureDtype {
        self.dtype
    }

    /// Number of stored rows. Zero for a zero-wide store, which is not a store
    /// with no rows but a store no row can be counted in.
    pub fn rows(&self) -> usize {
        self.len().checked_div(self.dim).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes the stored rows occupy, the figure `critic/feature_mib` reports.
    /// Capacity rather than length: the allocation is what the job pays
    /// for, and it is reserved in full before the first row lands.
    pub fn allocated_bytes(&self) -> usize {
        self.wide.capacity() * 4 + self.narrow.capacity() * 2
    }

    fn len(&self) -> usize {
        match self.dtype {
            FeatureDtype::F32 => self.wide.len(),
            FeatureDtype::F16 | FeatureDtype::Bf16 => self.narrow.len(),
        }
    }

    /// Appends `rows` produced into a caller-owned F32 buffer.
    ///
    /// The runtime writes hidden states as F32 into a `Vec<f32>` tail, so a
    /// narrow store cannot be filled in place; `fill` is the seam where that
    /// staging buffer is folded in. For an F32 store the fold is an extend of the
    /// same values, then narrows them when required. The PPO path uses
    /// [`Self::fill_from_tail`] for the staging step.
    pub fn push_rows(&mut self, rows: &[f32]) -> Result<()> {
        if self.dim == 0 || !rows.len().is_multiple_of(self.dim) {
            return Err(Error::invalid(
                "critic features are not a whole number of rows",
            ));
        }
        match self.dtype {
            FeatureDtype::F32 => self.wide.extend_from_slice(rows),
            FeatureDtype::F16 => self.narrow.extend(rows.iter().copied().map(f32_to_f16)),
            FeatureDtype::Bf16 => self.narrow.extend(rows.iter().copied().map(f32_to_bf16)),
        }
        Ok(())
    }

    /// Lends the F32 staging buffer the runtime writes into, then folds whatever
    /// was appended to it into the store and returns the rows just added, still
    /// in F32 and still exact.
    ///
    /// This keeps the fused scoring pass writing straight into a `Vec<f32>` tail,
    /// the shape its FFI signature requires - while the store decides what is
    /// retained. For an F32 store `staging` *is* the store, so the fold is a
    /// no-op and the returned slice is the store's own tail; for a narrow one the
    /// staging buffer is truncated back to empty after the conversion, so it never
    /// grows past a single rollout.
    ///
    /// The returned rows are the pre-rounding values on purpose: the pre-fit
    /// predictions that GAE consumes are taken from them, so enabling a narrow
    /// store changes what is *stored for the fit*, not what this update's
    /// advantages were computed from.
    pub fn fill_from_tail<'a, E>(
        &'a mut self,
        staging: &'a mut Vec<f32>,
        fill: impl FnOnce(&mut Vec<f32>) -> std::result::Result<(), E>,
    ) -> std::result::Result<FilledRows<'a>, E>
    where
        E: From<Error>,
    {
        match self.dtype {
            FeatureDtype::F32 => {
                let start = self.wide.len();
                fill(&mut self.wide)?;
                Ok(FilledRows {
                    rows: &self.wide[start..],
                })
            }
            FeatureDtype::F16 | FeatureDtype::Bf16 => {
                staging.clear();
                fill(staging)?;
                self.push_rows(staging).map_err(E::from)?;
                Ok(FilledRows { rows: staging })
            }
        }
    }

    /// Calls `visit` with each chunk of rows in F32, in storage order.
    ///
    /// The chunk is scratch owned by the store and is overwritten by the next
    /// call, so a visitor that needs to keep values must copy them. `visit` also
    /// receives the index of the chunk's first row, which is what lets a
    /// row-indexed target vector be walked alongside without a second cursor.
    pub fn for_each_chunk<E>(
        &mut self,
        mut visit: impl FnMut(usize, &[f32]) -> std::result::Result<(), E>,
    ) -> std::result::Result<(), E> {
        if self.dim == 0 {
            return Ok(());
        }
        match self.dtype {
            FeatureDtype::F32 => {
                // No conversion and no scratch: hand out windows of the store.
                let chunk = CHUNK_ROWS * self.dim;
                for (index, rows) in self.wide.chunks(chunk).enumerate() {
                    visit(index * CHUNK_ROWS, rows)?;
                }
                Ok(())
            }
            FeatureDtype::F16 | FeatureDtype::Bf16 => {
                let chunk = CHUNK_ROWS * self.dim;
                let decode = match self.dtype {
                    FeatureDtype::Bf16 => bf16_to_f32,
                    _ => f16_to_f32,
                };
                self.scratch.clear();
                self.scratch.reserve(chunk.min(self.narrow.len()));
                let mut first_row = 0;
                for stored in self.narrow.chunks(chunk) {
                    self.scratch.clear();
                    self.scratch.extend(stored.iter().copied().map(decode));
                    visit(first_row, &self.scratch)?;
                    first_row += stored.len() / self.dim;
                }
                Ok(())
            }
        }
    }
}

/// The rows one [`FeatureStore::fill_from_tail`] call appended, in F32.
pub struct FilledRows<'a> {
    rows: &'a [f32],
}

impl FilledRows<'_> {
    pub fn as_slice(&self) -> &[f32] {
        self.rows
    }
}

/// F32 → IEEE binary16, round-to-nearest-even.
///
/// Values above binary16's range saturate to its largest finite magnitude rather
/// than to an infinity. A saturated feature is a distorted regression input; an
/// infinite one makes the whole fit non-finite and the run fails a check written
/// for divergence, three layers away from the cause. NaN is preserved - it means
/// the *upstream* produced one, and hiding that would be worse than either.
fn f32_to_f16(value: f32) -> u16 {
    const F16_MAX: f32 = 65504.0;
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    if value.is_nan() {
        // Quiet NaN with the sign preserved; the payload does not survive and is
        // not load-bearing for anything that reads these features.
        return sign | 0x7e00;
    }
    let magnitude = value.abs();
    if magnitude >= F16_MAX {
        return sign | 0x7bff;
    }
    // Exponent of the F32 value, unbiased.
    let exponent = ((bits >> 23) & 0xff) as i32 - 127;
    if exponent < -24 {
        // Below half of the smallest subnormal: rounds to zero either way.
        return sign;
    }
    let mantissa = bits & 0x007f_ffff;
    if exponent < -14 {
        // Subnormal binary16: the implicit leading one comes back explicitly and
        // the whole significand is shifted right by how far the exponent sits
        // below the subnormal boundary.
        let significand = mantissa | 0x0080_0000;
        let shift = (-14 - exponent) as u32 + 13;
        return sign | round_shift(significand, shift) as u16;
    }
    let exponent_bits = ((exponent + 15) as u32) << 10;
    // Rounding the mantissa can carry into the exponent field; adding the two as
    // one number is what makes that carry land in the right place, and the
    // saturation above guarantees it cannot reach the infinity encoding.
    (sign as u32 | (exponent_bits + round_shift(mantissa, 13))) as u16
}

/// Shifts `value` right by `shift` bits with round-to-nearest-even.
fn round_shift(value: u32, shift: u32) -> u32 {
    let truncated = value >> shift;
    let remainder = value & ((1 << shift) - 1);
    let halfway = 1 << (shift - 1);
    if remainder > halfway || (remainder == halfway && truncated & 1 == 1) {
        truncated + 1
    } else {
        truncated
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exponent = ((bits >> 10) & 0x1f) as u32;
    let mantissa = ((bits as u32) & 0x03ff) << 13;
    match exponent {
        0 if mantissa == 0 => f32::from_bits(sign),
        0 => {
            // Subnormal binary16, normal binary32: rebuild it by multiplying the
            // significand back by 2^-24 rather than open-coding a renormalization.
            let magnitude = ((bits & 0x03ff) as f32) * 5.960_464_5e-8;
            f32::from_bits(sign) + if sign == 0 { magnitude } else { -magnitude }
        }
        0x1f => f32::from_bits(sign | 0x7f80_0000 | mantissa),
        _ => f32::from_bits(sign | ((exponent + 112) << 23) | mantissa),
    }
}

/// F32 → bfloat16: the top 16 bits, round-to-nearest-even. No range check is
/// needed - the exponent field is F32's, so nothing that fit in an F32 overflows.
fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounded = bits + 0x7fff + ((bits >> 16) & 1);
    (rounded >> 16) as u16
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_the_values_it_can_represent_exactly() {
        for value in [0.0_f32, -0.0, 1.0, -1.0, 0.5, 2048.0, -3.140_625, 65504.0] {
            let back = f16_to_f32(f32_to_f16(value));
            assert_eq!(back.to_bits(), value.to_bits(), "{value}");
        }
    }

    #[test]
    fn f16_rounds_to_nearest_even_and_reaches_subnormals() {
        // 1 + 2^-11 sits exactly halfway between 1.0 and the next binary16 value;
        // ties go to the even significand, which is 1.0.
        assert_eq!(f16_to_f32(f32_to_f16(1.0 + 2.0_f32.powi(-11))), 1.0);
        // 1 + 2^-10 is representable and must not be rounded away.
        let next = 1.0 + 2.0_f32.powi(-10);
        assert_eq!(f16_to_f32(f32_to_f16(next)), next);
        // Smallest binary16 subnormal, and half of it, which rounds to zero.
        let smallest = 2.0_f32.powi(-24);
        assert_eq!(f16_to_f32(f32_to_f16(smallest)), smallest);
        assert_eq!(f16_to_f32(f32_to_f16(smallest * 0.4)), 0.0);
    }

    #[test]
    fn f16_saturates_instead_of_producing_an_infinity() {
        for value in [1.0e30_f32, -1.0e30, f32::INFINITY, f32::NEG_INFINITY] {
            let back = f16_to_f32(f32_to_f16(value));
            assert!(back.is_finite(), "{value} -> {back}");
            assert_eq!(back.abs(), 65504.0);
            assert_eq!(back.is_sign_negative(), value.is_sign_negative());
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn bf16_keeps_the_f32_exponent_range() {
        for value in [1.0e30_f32, -1.0e-30, 1.0, 0.0] {
            let back = bf16_to_f32(f32_to_bf16(value));
            assert!(
                (back - value).abs() <= value.abs() * 0.01,
                "{value} -> {back}"
            );
        }
    }

    #[test]
    fn a_wide_store_hands_back_exactly_what_it_was_given() {
        let mut store = FeatureStore::with_capacity(2, FeatureDtype::F32, 3).unwrap();
        store.push_rows(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(store.rows(), 3);
        let mut seen = Vec::new();
        store
            .for_each_chunk(|first, rows| {
                assert_eq!(first, 0);
                seen.extend_from_slice(rows);
                Ok::<_, Error>(())
            })
            .unwrap();
        assert_eq!(seen, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn a_narrow_store_halves_the_allocation_and_stays_close() {
        let dim = 8;
        let rows = 64;
        let values: Vec<f32> = (0..rows * dim).map(|i| (i as f32) * 0.017 - 0.4).collect();
        let mut wide = FeatureStore::with_capacity(dim, FeatureDtype::F32, rows).unwrap();
        let mut narrow = FeatureStore::with_capacity(dim, FeatureDtype::F16, rows).unwrap();
        wide.push_rows(&values).unwrap();
        narrow.push_rows(&values).unwrap();
        assert_eq!(narrow.allocated_bytes() * 2, wide.allocated_bytes());
        assert_eq!(narrow.rows(), rows);
        let mut seen = Vec::new();
        narrow
            .for_each_chunk(|_, chunk| {
                seen.extend_from_slice(chunk);
                Ok::<_, Error>(())
            })
            .unwrap();
        assert_eq!(seen.len(), values.len());
        for (stored, &original) in seen.iter().zip(&values) {
            // binary16 carries 11 significand bits; anything worse than 2^-10
            // relative would mean the conversion, not the format, lost the value.
            let tolerance = original.abs() * 2.0_f32.powi(-10) + 6.0e-8;
            assert!(
                (stored - original).abs() <= tolerance,
                "{stored} vs {original}"
            );
        }
    }

    #[test]
    fn chunking_reports_the_first_row_of_every_chunk() {
        let dim = 1;
        let rows = CHUNK_ROWS * 2 + 5;
        let mut store = FeatureStore::with_capacity(dim, FeatureDtype::Bf16, rows).unwrap();
        store.push_rows(&vec![1.0; rows]).unwrap();
        let mut firsts = Vec::new();
        let mut counted = 0;
        store
            .for_each_chunk(|first, chunk| {
                firsts.push(first);
                counted += chunk.len() / dim;
                Ok::<_, Error>(())
            })
            .unwrap();
        assert_eq!(firsts, vec![0, CHUNK_ROWS, CHUNK_ROWS * 2]);
        assert_eq!(counted, rows);
    }

    #[test]
    fn a_partial_row_is_refused_rather_than_stored() {
        let mut store = FeatureStore::with_capacity(4, FeatureDtype::F16, 2).unwrap();
        assert!(store.push_rows(&[1.0, 2.0, 3.0]).is_err());
        assert!(store.is_empty());
    }

    #[test]
    fn fill_from_tail_returns_unrounded_rows_for_every_dtype() {
        for dtype in [FeatureDtype::F32, FeatureDtype::F16, FeatureDtype::Bf16] {
            let mut store = FeatureStore::with_capacity(2, dtype, 2).unwrap();
            let mut staging = Vec::new();
            let filled = store
                .fill_from_tail(&mut staging, |out| {
                    out.extend_from_slice(&[0.1, 0.2]);
                    Ok::<_, Error>(())
                })
                .unwrap();
            assert_eq!(filled.as_slice(), &[0.1_f32, 0.2], "{dtype:?}");
            let filled = store
                .fill_from_tail(&mut staging, |out| {
                    out.extend_from_slice(&[0.3, 0.4]);
                    Ok::<_, Error>(())
                })
                .unwrap();
            // The second call must see only its own rows, whichever buffer it used.
            assert_eq!(filled.as_slice(), &[0.3_f32, 0.4], "{dtype:?}");
            assert_eq!(store.rows(), 2, "{dtype:?}");
        }
    }
}
