//! Host tensor views with ggml byte strides, and the argument binding the
//! interpreter runs against.

#[derive(Clone, Copy, Debug)]
pub struct TensorView<'a> {
    pub data: &'a [f32],
    /// Logical `ne[]` shape in ggml order (dimension 0 is innermost).
    pub shape: [usize; 4],
    /// `nb[]` byte strides in ggml order.
    pub nb: [usize; 4],
}

/// Byte view for quantized tensors (`nb[0]` is the block size).
#[derive(Clone, Copy, Debug)]
pub struct TensorViewBytes<'a> {
    pub data: &'a [u8],
    pub shape: [usize; 4],
    pub nb: [usize; 4],
}

#[derive(Debug)]
pub struct TensorViewMut<'a> {
    pub data: &'a mut [f32],
    pub shape: [usize; 4],
    pub nb: [usize; 4],
}

/// A **written** binding held as bytes, which is what an F16 destination is.
/// The oracle keeps no `f16` type of its own: it
/// narrows at the store, exactly where a shader does, and the caller compares
/// the bytes it gets back.
#[derive(Debug)]
pub struct TensorViewBytesMut<'a> {
    pub data: &'a mut [u8],
    pub shape: [usize; 4],
    pub nb: [usize; 4],
}

impl<'a> TensorView<'a> {
    /// Contiguous 2D view with `ne0` columns (inner dimension) and `ne1` rows.
    pub fn contiguous_2d(data: &'a [f32], ne0: usize, ne1: usize) -> Self {
        assert!(data.len() >= ne0 * ne1);
        Self {
            data,
            shape: [ne0, ne1, 1, 1],
            nb: [4, 4 * ne0, 4 * ne0 * ne1, 4 * ne0 * ne1],
        }
    }

    /// 2D view whose rows are `row_stride_elems` elements apart (with padding).
    pub fn strided_2d(data: &'a [f32], ne0: usize, ne1: usize, row_stride_elems: usize) -> Self {
        assert!(row_stride_elems >= ne0);
        assert!(data.len() >= row_stride_elems * ne1);
        let nb1 = 4 * row_stride_elems;
        Self {
            data,
            shape: [ne0, ne1, 1, 1],
            nb: [4, nb1, nb1 * ne1, nb1 * ne1],
        }
    }
}

impl<'a> TensorViewBytes<'a> {
    /// Q8_0 2D view with `ne0` logical elements per row (a multiple of 32)
    /// and rows separated by `row_stride_bytes` bytes.
    pub fn q8_0_2d(data: &'a [u8], ne0: usize, ne1: usize, row_stride_bytes: usize) -> Self {
        assert_eq!(ne0 % 32, 0, "Q8_0: ne0 is a multiple of 32");
        assert!(row_stride_bytes >= ne0 / 32 * 34);
        assert!(data.len() >= row_stride_bytes * ne1);
        Self {
            data,
            shape: [ne0, ne1, 1, 1],
            nb: [
                34,
                row_stride_bytes,
                row_stride_bytes * ne1,
                row_stride_bytes * ne1,
            ],
        }
    }
}

impl<'a> TensorViewMut<'a> {
    pub fn contiguous_2d(data: &'a mut [f32], ne0: usize, ne1: usize) -> Self {
        assert!(data.len() >= ne0 * ne1);
        Self {
            data,
            shape: [ne0, ne1, 1, 1],
            nb: [4, 4 * ne0, 4 * ne0 * ne1, 4 * ne0 * ne1],
        }
    }

    pub fn strided_2d(
        data: &'a mut [f32],
        ne0: usize,
        ne1: usize,
        row_stride_elems: usize,
    ) -> Self {
        assert!(row_stride_elems >= ne0);
        assert!(data.len() >= row_stride_elems * ne1);
        let nb1 = 4 * row_stride_elems;
        Self {
            data,
            shape: [ne0, ne1, 1, 1],
            nb: [4, nb1, nb1 * ne1, nb1 * ne1],
        }
    }

    /// Contiguous 1D view.
    pub fn contiguous_1d(data: &'a mut [f32], ne0: usize) -> Self {
        assert!(data.len() >= ne0);
        Self {
            data,
            shape: [ne0, 1, 1, 1],
            nb: [4, 4 * ne0, 4 * ne0, 4 * ne0],
        }
    }
}

pub enum BoundArg<'a> {
    In(TensorView<'a>),
    InBytes(TensorViewBytes<'a>),
    Out(TensorViewMut<'a>),
    OutBytes(TensorViewBytesMut<'a>),
}

impl BoundArg<'_> {
    pub(crate) fn shape(&self) -> [usize; 4] {
        match self {
            BoundArg::In(v) => v.shape,
            BoundArg::InBytes(v) => v.shape,
            BoundArg::Out(v) => v.shape,
            BoundArg::OutBytes(v) => v.shape,
        }
    }

    pub(crate) fn nb(&self) -> [usize; 4] {
        match self {
            BoundArg::In(v) => v.nb,
            BoundArg::InBytes(v) => v.nb,
            BoundArg::Out(v) => v.nb,
            BoundArg::OutBytes(v) => v.nb,
        }
    }
}
