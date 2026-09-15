//! Shared-state locks that outlive a poisoned critical section.
//!
//! Every lock in this crate guards a map or a small record whose mutations are
//! single insertions, removals or field writes: there is no multi-step
//! invariant a panic can leave half-applied. What a panic *does* leave is a
//! poisoned lock, and `expect` on it turns one failed request into a server
//! whose runs list, dataset index or run state cannot be read at all. Recovering
//! the guard keeps the failure where it
//! happened - one request, one error - instead of promoting it to an outage.
//!
//! This is a deliberate exception, not a general rule: a lock protecting a
//! multi-step invariant would have to reconstruct it here rather than hand the
//! state back.

use std::sync::PoisonError;

pub(crate) trait Recover<T> {
    /// The guard, whether or not a previous holder panicked.
    fn recover(self) -> T;
}

impl<T> Recover<T> for Result<T, PoisonError<T>> {
    fn recover(self) -> T {
        self.unwrap_or_else(PoisonError::into_inner)
    }
}
