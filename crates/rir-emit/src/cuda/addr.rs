//! CUDA uses the shared address printer. The byte address is
//! `crate::printer::Printer::addr`; the constant-buffer prefix comes from
//! `Dialect::PARAMS`.
//!
//! Addresses stay byte-based here, matching `ggml_tensor.nb[]` directly. There
//! is no typed view and no element index, unlike Vulkan: a CUDA binding is a
//! `uint8_t *`, so the Loop IR's byte address is already the address, and the
//! access type only decides what it is reinterpreted as.
