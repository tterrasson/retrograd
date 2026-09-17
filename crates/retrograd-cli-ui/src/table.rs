//! The `comfy-table` half: a table whose rows are all known before anything is
//! printed - a bench report, a profiler section.
//!
//! What lives here is what both binaries were writing separately. `bench.rs`
//! had `header_cell`, `styled_cell`, `group_row`, `fmt_value`, `delta_color`
//! and `delta_ansi`; `profile.rs` had `base_table`, `secs`, `bar` and its own
//! right-aligned `Cell::new` at every call site. None of it knew about the
//! other, so the two reports drifted in preset, in alignment and in the shade
//! of grey used for a neutral delta.

use std::time::Duration;

use comfy_table::presets::UTF8_FULL;
use comfy_table::{Attribute, Cell, CellAlignment, Color, ContentArrangement, Table};

use crate::stream::Better;

/// An empty table in the shape both reports use: full box drawing, columns
/// sized to their content.
pub fn base_table() -> Table {
    let mut table = Table::new();
    table
        .load_style(UTF8_FULL)
        .set_content_arrangement(ContentArrangement::Dynamic);
    table
}

/// The bench report's frame: rounded corners, and columns sized by their own
/// content rather than by the terminal - a report whose numbers are compared
/// column to column must not re-wrap when the window narrows.
pub fn rounded_table() -> Table {
    let mut table = Table::new();
    table
        .load_style(UTF8_FULL.with_rounded_corners())
        .set_content_arrangement(ContentArrangement::Disabled);
    table
}

/// A right-aligned cell - the default for a figure, and what every numeric
/// column in both reports wants.
pub fn right_cell(text: impl ToString) -> Cell {
    Cell::new(text.to_string()).set_alignment(CellAlignment::Right)
}

/// A column heading: blue and bold when the output is a terminal, plain
/// otherwise, so a piped report stays diffable.
pub fn header_cell(color: bool, text: &str) -> Cell {
    let cell = Cell::new(text);
    if color {
        cell.fg(Color::AnsiValue(39)).add_attribute(Attribute::Bold)
    } else {
        cell
    }
}

/// A full-width heading row that separates one group of metrics from the next.
pub fn group_row(table: &mut Table, color: bool, columns: usize, title: &str) {
    let title_cell = if color {
        Cell::new(title)
            .fg(Color::AnsiValue(244))
            .add_attribute(Attribute::Bold)
    } else {
        Cell::new(title)
    };
    let mut cells = vec![title_cell];
    cells.extend((1..columns).map(|_| Cell::new("")));
    table.add_row(cells);
}

/// A cell with an explicit alignment and an optional `(colour, bold)` style,
/// for a report row that [`right_cell`] and [`header_cell`] don't cover.
pub fn styled_cell(text: String, align: CellAlignment, style: Option<(Color, bool)>) -> Cell {
    let mut cell = Cell::new(text).set_alignment(align);
    if let Some((color, bold)) = style {
        cell = cell.fg(color);
        if bold {
            cell = cell.add_attribute(Attribute::Bold);
        }
    }
    cell
}

/// Formats a figure with a fixed unit suffix and, when `signed`, an explicit
/// `+`/`-` prefix.
pub fn fmt_value(value: f64, precision: usize, unit: &str, signed: bool) -> String {
    if signed {
        format!("{value:+.precision$}{unit}")
    } else {
        format!("{value:.precision$}{unit}")
    }
}

/// Pick a foreground colour for a delta cell: green shades for an improvement,
/// red shades for a regression, deeper as the relative change grows; grey when
/// the change is negligible or the metric has no better/worse direction.
///
/// The thresholds are the ones [`crate::stream::trend_ansi`] uses, because the
/// streaming table and the report describe the same movement and a metric must
/// not change colour when it moves from one to the other.
pub fn delta_color(delta: f64, base: f64, better: Better) -> (Color, bool) {
    let neutral = (Color::AnsiValue(245), false);
    if delta.abs() < 1.0e-9 {
        return neutral;
    }
    let improved = match better {
        Better::Higher => delta > 0.0,
        Better::Lower => delta < 0.0,
        Better::Neutral => return neutral,
    };
    let rel = delta.abs() / (base.abs() + 1.0e-9);
    let level = if rel < 0.01 {
        0
    } else if rel < 0.10 {
        1
    } else if rel < 0.50 {
        2
    } else {
        3
    };
    let value = if improved {
        [245u8, 150, 77, 40][level]
    } else {
        [245u8, 217, 167, 196][level]
    };
    (Color::AnsiValue(value), level >= 3)
}

/// Green above zero, red below, grey at it: for a delta whose direction is the
/// whole message and whose magnitude carries no scale.
pub fn delta_ansi(value: f64) -> &'static str {
    if value > 1.0e-9 {
        "38;5;40"
    } else if value < -1.0e-9 {
        "38;5;196"
    } else {
        "38;5;245"
    }
}

/// A 24-cell proportion bar, for a share of a total that reads better as a
/// length than as a number.
pub fn bar(pct: f32) -> String {
    let width = 24usize;
    // `pct` is a measured share, so neither bound is guaranteed: a rounding
    // artefact just above 100, or a negative reading from a counter that went
    // backwards, must widen or panic no row. The float-to-`usize` cast
    // saturates at 0 for a negative value (docs/engineering/CONVERSIONS.md), and `min`
    // closes the other end, so `filled <= width` holds and the subtraction
    // below cannot underflow.
    let filled = (((pct / 100.0) * width as f32).round() as usize).min(width);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

/// A duration in seconds with three decimals, the unit every timing column of
/// both reports uses.
pub fn secs(duration: Duration) -> String {
    format!("{:.3} s", duration.as_secs_f64())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bar_stays_the_same_width_whatever_the_share() {
        for pct in [-5.0, 0.0, 33.3, 100.0, 140.0] {
            assert_eq!(bar(pct).chars().count(), 24, "{pct}");
        }
    }

    #[test]
    fn a_negligible_delta_is_grey_in_either_direction() {
        assert_eq!(
            delta_color(0.0, 1.0, Better::Higher).0,
            Color::AnsiValue(245)
        );
        assert_eq!(
            delta_color(0.0, 1.0, Better::Lower).0,
            Color::AnsiValue(245)
        );
        assert_eq!(delta_ansi(0.0), "38;5;245");
    }
}
