//! One module per lowering strategy. Which one runs is decided by
//! `Schedule`, in `super::lower`; each module here only knows how to build
//! the nest its strategy describes.

pub mod reduction;
pub mod scan;
pub mod serial;
pub mod tiled;
