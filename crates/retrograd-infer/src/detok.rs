//! Incremental detokenization: partial UTF-8 is retained, stop strings are
//! matched on a sliding suffix, and text is never emitted and then retracted.
//!
//! The backend renders one token at a time into raw piece bytes; this type
//! turns the byte stream into emittable text. A trailing partial character
//! waits for its continuation bytes, and a trailing partial *stop string*
//! waits too: if the held-back tail could still grow into a stop, it is not
//! emitted yet. When a stop completes in the text, everything
//! before it is emitted and the stop itself is not. When the stream ends
//! without a stop, the held-back tail is flushed ([`Detokenizer::finish`]),
//! because it was never a stop after all.

/// The incremental detokenizer of one sequence.
pub struct Detokenizer {
    stops: Vec<Vec<u8>>,
    max_stop: usize,
    /// Piece bytes not yet forming a complete UTF-8 character.
    pending: Vec<u8>,
    /// Complete characters not yet emitted (a possible stop prefix).
    held: String,
    /// Set once a stop matched: no further text is produced.
    stopped: bool,
}

impl Detokenizer {
    pub fn new(stops: &[String]) -> Self {
        // An empty stop would match everywhere; requests refuse them, and
        // the detokenizer ignores them regardless.
        let stops: Vec<Vec<u8>> = stops
            .iter()
            .filter(|stop| !stop.is_empty())
            .map(|stop| stop.as_bytes().to_vec())
            .collect();
        let max_stop = stops.iter().map(|stop| stop.len()).max().unwrap_or(0);
        Self {
            stops,
            max_stop,
            pending: Vec::new(),
            held: String::new(),
            stopped: false,
        }
    }

    /// Feeds one token's piece bytes. Returns the text that can be emitted
    /// now, and whether a stop string completed in the text (in
    /// which case the returned text is everything before the stop, and the
    /// stop itself is dropped).
    pub fn push_bytes(&mut self, bytes: &[u8]) -> (String, bool) {
        if self.stopped {
            return (String::new(), true);
        }
        self.pending.extend_from_slice(bytes);
        // Drain the longest complete UTF-8 prefix of `pending` into `held`.
        while !self.pending.is_empty() {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    self.held.push_str(text);
                    self.pending.clear();
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    self.held.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("valid_up_to marks a UTF-8 boundary"),
                    );
                    self.pending.drain(..valid);
                    if let Some(invalid) = error.error_len() {
                        self.held.push('\u{fffd}');
                        self.pending.drain(..invalid);
                    } else {
                        break; // Incomplete character: wait for the next piece.
                    }
                }
            }
        }
        // A piece may contain both a stop and trailing text. Choose the first
        // stop in the accumulated text and discard everything from it onward.
        let held = self.held.as_bytes();
        let stop_at = self
            .stops
            .iter()
            .filter_map(|stop| held.windows(stop.len()).position(|window| window == stop))
            .min();
        if let Some(keep) = stop_at {
            let emitted = self.held[..keep].to_string();
            self.held.clear();
            self.pending.clear();
            self.stopped = true;
            return (emitted, true);
        }
        // Hold back the longest suffix of `held` that is a proper prefix of
        // some stop: it may still grow into one.
        let mut hold = 0_usize;
        if self.max_stop > 1 {
            let held = self.held.as_bytes();
            for stop in &self.stops {
                let limit = stop.len() - 1;
                for k in (1..=limit).rev() {
                    if k > hold && held.len() >= k && held[held.len() - k..] == stop[..k] {
                        hold = k;
                        break;
                    }
                }
            }
        }
        let split = self.held.len() - hold;
        let emitted = self.held[..split].to_string();
        self.held.drain(..split);
        (emitted, false)
    }

    /// End of the stream without a stop: flush the held-back tail, which was
    /// never a stop after all.
    pub fn finish(&mut self) -> String {
        self.held.push_str(&String::from_utf8_lossy(&self.pending));
        self.pending.clear();
        std::mem::take(&mut self.held)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detok(stops: &[&str]) -> Detokenizer {
        Detokenizer::new(
            &stops
                .iter()
                .map(|stop| stop.to_string())
                .collect::<Vec<_>>(),
        )
    }

    #[test]
    fn a_multibyte_character_split_across_tokens_is_held_until_complete() {
        let mut d = detok(&[]);
        let (text, stopped) = d.push_bytes("ok ".as_bytes());
        assert_eq!((text.as_str(), stopped), ("ok ", false));
        let (text, stopped) = d.push_bytes(&"🎉".as_bytes()[..2]);
        assert_eq!((text.as_str(), stopped), ("", false));
        let (text, stopped) = d.push_bytes(&"🎉".as_bytes()[2..]);
        assert_eq!((text.as_str(), stopped), ("🎉", false));
    }

    #[test]
    fn a_stop_split_across_tokens_is_detected_and_not_emitted() {
        let mut d = detok(&["STOP"]);
        let (text, stopped) = d.push_bytes(b"abcST");
        assert_eq!((text.as_str(), stopped), ("abc", false), "ST held back");
        let (text, stopped) = d.push_bytes(b"OP");
        assert_eq!((text.as_str(), stopped), ("", true), "stop completed");
        // After the stop, nothing more is produced.
        let (text, stopped) = d.push_bytes(b"xyz");
        assert_eq!((text.as_str(), stopped), ("", true));
    }

    #[test]
    fn a_stop_in_the_middle_triggers_with_the_text_before_it() {
        let mut d = detok(&["</end>"]);
        let (text, stopped) = d.push_bytes(b"answer</end>ignored");
        assert_eq!((text.as_str(), stopped), ("answer", true));
    }

    #[test]
    fn a_held_back_tail_is_flushed_at_finish_when_no_stop_completes() {
        let mut d = detok(&["STOP"]);
        let (text, _) = d.push_bytes(b"abST");
        assert_eq!(text, "ab");
        assert_eq!(d.finish(), "ST");
    }

    #[test]
    fn emitted_text_is_always_a_prefix_of_everything_pushed() {
        let mut d = detok(&["abc"]);
        let mut emitted = String::new();
        for piece in ["x", "a", "b", "d", "e"] {
            let (text, stopped) = d.push_bytes(piece.as_bytes());
            assert!(!stopped);
            emitted.push_str(&text);
        }
        emitted.push_str(&d.finish());
        assert_eq!(emitted, "xabde");
    }

    #[test]
    fn malformed_and_incomplete_utf8_match_lossy_decoding() {
        for pieces in [
            vec![b"\xffvalid".as_slice()],
            vec![b"\xf0\x9f", b"ok"],
            vec![b"ok\xf0\x9f"],
        ] {
            let mut d = detok(&[]);
            let mut text = String::new();
            for piece in &pieces {
                text.push_str(&d.push_bytes(piece).0);
            }
            text.push_str(&d.finish());
            assert_eq!(text, String::from_utf8_lossy(&pieces.concat()));
        }
    }

    #[test]
    fn the_longest_matching_stop_prefix_is_held_back() {
        let mut d = detok(&["abcdef"]);
        let (text, stopped) = d.push_bytes(b"zzabc");
        assert_eq!((text.as_str(), stopped), ("zz", false));
        // "abc" is held; pushing "x" makes it impossible to complete the stop.
        let (text, stopped) = d.push_bytes(b"x");
        assert_eq!((text.as_str(), stopped), ("abcx", false));
    }
}
