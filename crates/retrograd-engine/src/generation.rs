//! Row splitting and forced-token refusal shared by the trainer's and the
//! engine's generation APIs. Both wrappers call the same FFI shapes and must
//! split their flat output buffers the same way, so the splitting lives here
//! once.

use super::*;

/// The generators that hand back bare tokens have nowhere to say which ones
/// the budget forced, and a caller training on them would credit the policy
/// with an end tag it never chose.
pub(crate) fn refuse_forced<'a>(forced: impl IntoIterator<Item = &'a Vec<bool>>) -> Result<()> {
    match forced.into_iter().any(|row| row.contains(&true)) {
        true => Err(Error::invalid(
            "the reasoning budget forced tokens into a generation whose caller cannot mask them; \
             use generate_sampled_continuous",
        )),
        false => Ok(()),
    }
}

/// An empty row is rejected by both wrappers, like in
/// `generate_tokens_continuous`: it would otherwise become a rollout whose
/// train mask is entirely false, surfacing much later as "rollout has no
/// trainable token".
pub(crate) fn check_lengths(lengths: &[usize], stride: usize) -> Result<()> {
    if lengths.iter().any(|&len| len == 0 || len > stride) {
        return Err(Error::runtime(
            "generation runtime returned an invalid sequence length",
        ));
    }
    Ok(())
}

/// Row-major output split into per-sequence vectors; `lengths` was validated by
/// [`check_lengths`], so every row is a non-empty slice of `stride`.
pub(crate) fn split_rows<T: Copy>(lengths: &[usize], stride: usize, flat: &[T]) -> Vec<Vec<T>> {
    lengths
        .iter()
        .enumerate()
        .map(|(row, &len)| flat[row * stride..row * stride + len].to_vec())
        .collect()
}
