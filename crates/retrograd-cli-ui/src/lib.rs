//! The terminal presentation layer of the `retrograd` and `profile` binaries.
//!
//! It is a crate rather than a module because a binary target cannot reach
//! another binary's modules.
//!
//! The split inside is by rendering model, not by caller: [`stream`] emits a
//! table one row at a time, because a training row arrives while the run is
//! still going; [`table`] renders a finished table at once through
//! `comfy-table`; [`progress`] owns stderr - spinners, bars and the labelled
//! lines that must not tear a bar's redraw.

pub mod progress;
pub mod stream;
pub mod table;

pub use progress::CliUi;
pub use stream::{
    Better, Column, DIM, EVAL_ROW, StreamCell, StreamTable, budget_ansi, cell, fmt_signed,
    magnitude_ansi, paint, throughput_cell, trend_ansi,
};
pub use table::{
    bar, base_table, delta_ansi, delta_color, fmt_value, group_row, header_cell, right_cell,
    rounded_table, secs, styled_cell,
};
