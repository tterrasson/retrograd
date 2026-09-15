use super::*;

// FFI safety contract for this module: queried ids are plain values; for a
// round-trip, the source and both output allocations cover the element/block
// counts passed to the synchronous runtime call.

/// The `ggml_type` ids the training ops can decode in place, straight from
/// `GGML_RETRO_DEQUANT_TYPES` in the fork, paired with each type's name.
///
/// Tests sweep this instead of restating the list, so adding a row to the
/// table extends every parity test at once - it is the single source for
/// backend support and parity tests.
pub fn dequant_types() -> Vec<(i32, String)> {
    // SAFETY: the module contract guarantees the queried values and buffer extents passed to the runtime.
    let n = unsafe { ffi::retro_dequant_types(std::ptr::null_mut(), 0) };
    let mut ids = vec![0_i32; n];
    // SAFETY: `ids` exposes `n` writable entries for the synchronous fill call.
    let written = unsafe { ffi::retro_dequant_types(ids.as_mut_ptr(), ids.len()) };
    debug_assert_eq!(
        written, n,
        "retro_dequant_types changed its count between calls"
    );
    ids.into_iter()
        .map(|id| {
            // SAFETY: the module contract guarantees the queried values and buffer extents passed to the runtime.
            let name = unsafe { std::ffi::CStr::from_ptr(ffi::retro_ggml_type_name(id)) };
            (id, name.to_string_lossy().into_owned())
        })
        .collect()
}

/// The `ggml_type` id of a type name (`"q4_K"`), or `None` if the fork's
/// decode table does not carry it.
pub fn ggml_type_id(name: &str) -> Option<i32> {
    dequant_types()
        .into_iter()
        .find(|(_, n)| n == name)
        .map(|(id, _)| id)
}

/// `(blck_size, type_size)` of a ggml type: how many logical elements a block
/// holds and how many bytes it occupies. The RIR quantized-format table
/// restates both, and `tests/rir_quant_oracle.rs` compares them here.
pub fn quant_traits(ggml_type: i32) -> Option<(i64, usize)> {
    let (mut be, mut bytes) = (0_i64, 0_usize);
    // SAFETY: the module contract guarantees the queried values and buffer extents passed to the runtime.
    let rc = unsafe { ffi::retro_quant_traits(ggml_type, &mut be, &mut bytes) };
    (rc == 0).then_some((be, bytes))
}

/// Quantizes `src` with ggml's reference quantizer and decodes it back with
/// ggml's own `to_float`, returning `(quantized bytes, decoded values)`.
///
/// The bytes come back on purpose: a decoder is compared against ggml on
/// **identical bytes**, never against a second quantization of the same F32,
/// that would measure the quantizer.
pub fn quant_roundtrip(ggml_type: i32, src: &[f32]) -> Option<(Vec<u8>, Vec<f32>)> {
    let (be, block_bytes) = quant_traits(ggml_type)?;
    if be <= 0 || src.is_empty() || src.len() as i64 % be != 0 {
        return None;
    }
    let mut bytes = vec![0_u8; (src.len() as i64 / be) as usize * block_bytes];
    let mut decoded = vec![0.0_f32; src.len()];
    // SAFETY: the module contract guarantees the queried values and buffer extents passed to the runtime.
    let rc = unsafe {
        ffi::retro_quant_roundtrip(
            ggml_type,
            src.len() as i64,
            src.as_ptr(),
            bytes.as_mut_ptr(),
            bytes.len(),
            decoded.as_mut_ptr(),
        )
    };
    (rc == 0).then_some((bytes, decoded))
}
