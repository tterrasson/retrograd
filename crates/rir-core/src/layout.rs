//! Tensor layouts using the ggml convention.
//!
//! Invariant: **nothing in the IR assumes contiguity.** Strides
//! `nb[]` (in bytes) and shape `ne[]` are supplied at dispatch, as they are in
//! `ggml_tensor`; the IR only knows rank and dtype. A kernel that requires
//! contiguity declares `Constraint::Contiguous`, and `supports_op` rejects any
//! incompatible layout.

use crate::types::DType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Layout {
    /// Arbitrary ggml `nb[]` strides in bytes, copied unchanged from
    /// `ggml_tensor` at dispatch.
    Ggml,
    /// C-contiguous, guaranteed by a constraint checked by `supports_op`.
    Contiguous,
}

/// Dimensions a ggml tensor has: `ne[]` and `nb[]` are four wide, and so is
/// every array in `TensorDesc`. A rank above this is not a shape the dispatch
/// can describe, which is why `validate` refuses it (`ValidateError::UnsupportedRank`).
pub const MAX_TENSOR_RANK: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorType {
    pub dtype: DType,
    /// Logical rank (in ggml, dimension 0 is innermost).
    pub rank: usize,
    pub layout: Layout,
}

impl TensorType {
    pub fn f32(rank: usize) -> Self {
        Self {
            dtype: DType::F32,
            rank,
            layout: Layout::Ggml,
        }
    }

    pub fn f32_2d() -> Self {
        Self::f32(2)
    }

    /// An F16 tensor. Half the bytes, the same `nb[]`
    /// algebra: strides are in bytes, so nothing above the memory boundary
    /// changes.
    pub fn f16(rank: usize) -> Self {
        Self {
            dtype: DType::F16,
            rank,
            layout: Layout::Ggml,
        }
    }
}
