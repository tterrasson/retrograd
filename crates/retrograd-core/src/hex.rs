//! Lowercase hexadecimal rendering of a digest.
//!
//! `sha2` 0.10 let a caller write `format!("{:x}", Sha256::digest(bytes))`;
//! 0.11 returns an array that implements no `LowerHex`, so every caller renders
//! the bytes itself. That was already three hand-rolled loops before the
//! versions were unified - this is the one they share.
//! `rir-gen` keeps its own copy: the RIR chain may not depend on this crate.

use std::fmt::Write;

/// Renders bytes as lowercase hexadecimal, two characters per byte.
pub fn hex_lower(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut out, byte| {
            // Writing into a String cannot fail; the result is discarded rather
            // than unwrapped so no production path panics on a formatter.
            let _ = write!(out, "{byte:02x}");
            out
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_digest_is_rendered_two_lowercase_characters_per_byte() {
        assert_eq!(hex_lower(&[0x00, 0x0f, 0xa9, 0xff]), "000fa9ff");
        assert_eq!(hex_lower(&[]), "");
    }
}
