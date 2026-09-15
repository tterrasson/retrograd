//! Byte budgeting shared by every source of observations.
//!
//! Tool results and environment observations both enter the prompt, and both
//! come from somewhere that has no idea what a token budget is. The cap lives
//! here because the providers that apply it now sit in different crates.

/// Byte cap that never cuts a character in half, and marks the cut so the
/// policy can tell an elided payload from a short one.
///
/// The result is never longer than `max_bytes`: a cap tighter than the marker
/// itself cuts the marker too rather than overshooting the budget the caller
/// asked for - the whole point of the cap is that it holds.
pub fn truncate_utf8(content: &mut String, max_bytes: usize, marker: &str) {
    if content.len() <= max_bytes {
        return;
    }
    let marker = &marker[..floor_boundary(marker, max_bytes)];
    let mut boundary = floor_boundary(content, max_bytes - marker.len());
    if boundary > content.len() {
        boundary = content.len();
    }
    content.truncate(boundary);
    content.push_str(marker);
}

/// The largest character boundary of `text` at or below `at`.
fn floor_boundary(text: &str, at: usize) -> usize {
    let mut boundary = at.min(text.len());
    while boundary > 0 && !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cap_holds_even_below_the_marker_length() {
        let marker = "\n[tool result truncated]";
        for max_bytes in 1..=marker.len() + 8 {
            let mut text = "é".repeat(40);
            truncate_utf8(&mut text, max_bytes, marker);
            assert!(
                text.len() <= max_bytes,
                "{max_bytes} byte cap produced {} bytes",
                text.len()
            );
            assert!(text.is_char_boundary(text.len()));
        }
    }
}
