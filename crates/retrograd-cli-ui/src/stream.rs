//! Primitives for a table emitted one row at a time, and the colour rules the
//! bench report shares with it.
//!
//! The streaming shape is the reason this is not `comfy-table`: a training row
//! is printed while the run is still producing the next one, so the frame has
//! to be written in pieces - open, row, span row, close - instead of rendered
//! once at the end.

/// Whether a higher or lower value of a metric is better, for colouring its
/// delta cell.
#[derive(Clone, Copy)]
pub enum Better {
    Higher,
    Lower,
    Neutral,
}

/// Grey, for cells that carry context rather than a result (learning rate,
/// inner epoch counter).
pub const DIM: Option<&str> = Some("38;5;244");

/// Cyan, for evaluation rows spanning the full width of the streaming table.
pub const EVAL_ROW: Option<&str> = Some("36");

/// A streaming-table column: a heading and a width, the width never narrower
/// than the heading itself.
pub struct Column {
    pub title: &'static str,
    width: usize,
}

impl Column {
    /// Widens `width` to fit `title` if needed, so a heading is never clipped.
    pub fn new(title: &'static str, width: usize) -> Self {
        Self {
            title,
            width: width.max(display_width(title)),
        }
    }
}

/// One cell of a [`StreamTable`] row: its text and an optional ANSI code to
/// paint it with.
pub struct StreamCell {
    text: String,
    code: Option<String>,
}

/// Builds a [`StreamCell`]. `code` is applied only when the table's own colour
/// is on; pass `None` for a cell that is never coloured.
pub fn cell(text: impl Into<String>, code: Option<&str>) -> StreamCell {
    StreamCell {
        text: text.into(),
        code: code.map(str::to_string),
    }
}

/// A right-aligned box-drawing table in the shape of the bench report, but
/// emitted one row at a time: training rows arrive over the course of the run,
/// so the frame has to be printed piece by piece instead of rendered at once.
pub struct StreamTable {
    color: bool,
    columns: Vec<Column>,
}

impl StreamTable {
    pub fn new(color: bool, columns: Vec<Column>) -> Self {
        Self { color, columns }
    }

    /// Top border, header row and the separator underneath it.
    pub fn open(&self) -> String {
        let header = self
            .columns
            .iter()
            .map(|column| StreamCell {
                text: column.title.to_string(),
                code: self.color.then(|| "38;5;39;1".to_string()),
            })
            .collect::<Vec<_>>();
        format!(
            "{}\n{}\n{}",
            self.rule('╭', '┬', '╮'),
            self.row(&header),
            self.rule('├', '┼', '┤'),
        )
    }

    pub fn row(&self, cells: &[StreamCell]) -> String {
        let mut line = String::from("│");
        for (column, cell) in self.columns.iter().zip(cells) {
            let padding = column.width.saturating_sub(display_width(&cell.text));
            line.push(' ');
            line.push_str(&" ".repeat(padding));
            match &cell.code {
                Some(code) if self.color => line.push_str(&paint(true, code, &cell.text)),
                _ => line.push_str(&cell.text),
            }
            line.push_str(" │");
        }
        line
    }

    /// One full-width row without column separators, for out-of-band lines
    /// (evaluation results, checkpoint notices) that must not break the frame.
    ///
    /// Anything wider than the frame is wrapped onto continuation rows rather
    /// than cut: these lines carry paths and evaluation figures, and a silent
    /// truncation loses exactly the end of them, which is where the filename
    /// and the early-stopping verdict are. The result is one row per line,
    /// separated by newlines.
    pub fn span_row(&self, text: &str, code: Option<&str>) -> String {
        // A normal row is `│` plus ` cell │` per column, so the spanned text
        // area is the total width minus the two border-and-space pairs.
        let width = self
            .columns
            .iter()
            .map(|column| column.width + 3)
            .sum::<usize>()
            .saturating_sub(3);
        wrap(text, width)
            .into_iter()
            .map(|line| {
                let padding = " ".repeat(width.saturating_sub(display_width(&line)));
                let line = match code {
                    Some(code) if self.color => paint(true, code, &line),
                    _ => line,
                };
                format!("│ {line}{padding} │")
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    pub fn close(&self) -> String {
        self.rule('╰', '┴', '╯')
    }

    fn rule(&self, left: char, middle: char, right: char) -> String {
        let mut line = String::from(left);
        for (index, column) in self.columns.iter().enumerate() {
            if index > 0 {
                line.push(middle);
            }
            line.push_str(&"─".repeat(column.width + 2));
        }
        line.push(right);
        line
    }
}

/// Sign-prefixed, except for an exact zero: a `+0.000000` reads as a tiny
/// positive move when it actually means the update did nothing.
pub fn fmt_signed(value: f64, precision: usize) -> String {
    if value == 0.0 {
        format!("{:.precision$}", 0.0)
    } else {
        format!("{value:+.precision$}")
    }
}

fn display_width(text: &str) -> usize {
    text.chars().count()
}

/// Greedy word wrap, falling back to a hard cut for a single word wider than
/// the frame - a checkpoint path has no spaces to break on. Continuation lines
/// are indented so a wrapped row reads as one entry rather than two.
fn wrap(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let indent = if width > 4 { "  " } else { "" };
    let mut lines: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && display_width(&line) + 1 + display_width(word) > width {
            lines.push(std::mem::take(&mut line));
        }
        if line.is_empty() {
            if !lines.is_empty() {
                line.push_str(indent);
            }
        } else {
            line.push(' ');
        }
        for character in word.chars() {
            if display_width(&line) == width {
                lines.push(std::mem::take(&mut line));
                line.push_str(indent);
            }
            line.push(character);
        }
    }
    if !line.is_empty() || lines.is_empty() {
        lines.push(line);
    }
    lines
}

/// Shade a value by how far it has moved from its own scale: four steps from
/// grey (noise) to a saturated green or red (a large move in the good or bad
/// direction).
pub fn magnitude_ansi(value: f64, thresholds: [f64; 3], better: Better) -> Option<&'static str> {
    let improved = match better {
        Better::Higher => value > 0.0,
        Better::Lower => value < 0.0,
        Better::Neutral => return None,
    };
    Some(shade(level(value.abs(), thresholds), improved))
}

/// Shade a change relative to the value it moved from, so a delta is read as a
/// percentage of the metric rather than in absolute units.
pub fn trend_ansi(delta: f64, base: f64, better: Better) -> Option<&'static str> {
    if !delta.is_finite() || !base.is_finite() {
        return None;
    }
    let improved = match better {
        Better::Higher => delta > 0.0,
        Better::Lower => delta < 0.0,
        Better::Neutral => return None,
    };
    let rel = delta.abs() / (base.abs() + 1.0e-9);
    Some(shade(level(rel, [0.01, 0.10, 0.50]), improved))
}

/// Green-to-red gradient for a metric that is healthy while it stays small,
/// KL divergence and clip fraction both measure how far the update has drifted
/// from the sampled policy.
pub fn budget_ansi(value: f64, thresholds: [f64; 3]) -> Option<&'static str> {
    if !value.is_finite() {
        return None;
    }
    Some(["38;5;40", "38;5;77", "38;5;214", "38;5;196"][level(value, thresholds)])
}

/// Grey out a stalled throughput: a zero means the update produced no trainable
/// tokens, which is worth spotting at a glance.
pub fn throughput_cell(tokens_per_second: f64) -> StreamCell {
    if tokens_per_second <= 0.0 {
        cell("0.0", Some("38;5;196"))
    } else {
        cell(format!("{tokens_per_second:.1}"), DIM)
    }
}

fn level(magnitude: f64, thresholds: [f64; 3]) -> usize {
    thresholds.iter().filter(|&&edge| magnitude >= edge).count()
}

fn shade(level: usize, improved: bool) -> &'static str {
    if improved {
        ["38;5;245", "38;5;150", "38;5;77", "38;5;40"][level]
    } else {
        ["38;5;245", "38;5;217", "38;5;167", "38;5;196"][level]
    }
}

pub fn paint(color: bool, code: &str, text: &str) -> String {
    if color {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_row_wraps_instead_of_dropping_the_end_of_a_line() {
        let table = StreamTable::new(false, vec![Column::new("a", 10), Column::new("b", 10)]);
        // Two 10-wide columns span 10 + 3 + 10 = 23 characters of text.
        let rows = table.span_row("eval epoch 1: loss 0.5 best 0.5 saved best.gguf", None);
        let rows: Vec<&str> = rows.split('\n').collect();
        assert!(rows.len() > 1, "{rows:?}");
        assert!(rows.iter().all(|row| row.chars().count() == 27), "{rows:?}");
        assert!(rows.last().expect("a row").contains("best.gguf"));
    }

    #[test]
    fn wrap_hard_splits_a_word_wider_than_the_frame() {
        let lines = wrap("/tmp/a-very-long-checkpoint-path.state", 12);
        assert!(
            lines.iter().all(|line| line.chars().count() <= 12),
            "{lines:?}"
        );
        assert_eq!(
            lines.concat().replace("  ", ""),
            "/tmp/a-very-long-checkpoint-path.state"
        );
    }

    #[test]
    fn fmt_signed_formats_zero_without_a_sign_and_signs_nonzero_values() {
        assert_eq!(fmt_signed(0.0, 2), "0.00");
        assert_eq!(fmt_signed(-1.5, 1), "-1.5");
        assert_eq!(fmt_signed(1.5, 1), "+1.5");
    }
}
