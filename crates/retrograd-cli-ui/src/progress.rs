//! Everything the two binaries write to stderr while they work.
//!
//! One type owns the decisions a terminal forces - is this a TTY, is `NO_COLOR`
//! set, is a bar currently redrawing - so no caller has to ask twice and no
//! diagnostic tears a live progress bar.

use std::env;
use std::io::IsTerminal;
use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

/// What a terminal supports, decided once at startup: whether ANSI colour may
/// be emitted and whether a live progress bar may be drawn. Every write to
/// stderr goes through this instead of checking `is_terminal()` again.
pub struct CliUi {
    /// Whether ANSI colour codes may be used - also read by callers that build
    /// their own coloured output outside `CliUi`'s own methods.
    pub color: bool,
    progress: bool,
}

impl Default for CliUi {
    fn default() -> Self {
        Self::new()
    }
}

impl CliUi {
    /// Reads the terminal once: colour unless stderr is redirected or
    /// `NO_COLOR` is set, progress bars only on a TTY.
    pub fn new() -> Self {
        Self {
            color: std::io::stderr().is_terminal() && env::var_os("NO_COLOR").is_none(),
            progress: std::io::stderr().is_terminal(),
        }
    }
    pub fn section(&self, value: &str) {
        eprintln!("{}", self.label(value, "36;1"));
    }
    pub fn info(&self, value: impl AsRef<str>) {
        eprintln!("{} {}", self.label("info", "36"), value.as_ref());
    }
    pub fn diagnostic(&self, title: &str, body: &str) {
        eprintln!("{} {title}\n{body}", self.label("diagnostic", "33;1"));
    }
    pub fn spinner(&self, value: impl Into<String>) -> ProgressBar {
        let bar = if self.progress {
            ProgressBar::new_spinner()
        } else {
            ProgressBar::hidden()
        };
        bar.set_style(ProgressStyle::with_template("{spinner:.cyan} {msg}").expect("valid style"));
        bar.set_message(value.into());
        bar.enable_steady_tick(Duration::from_millis(120));
        bar
    }
    pub fn finish_spinner(&self, bar: ProgressBar, value: impl Into<String>) {
        let value = format!("{} {}", self.label("ok", "32;1"), value.into());
        if self.progress {
            bar.finish_with_message(value)
        } else {
            eprintln!("{value}")
        }
    }
    pub fn fail_spinner(&self, bar: ProgressBar, value: impl Into<String>) {
        let value = format!("{} {}", self.label("failed", "31;1"), value.into());
        if self.progress {
            bar.abandon_with_message(value)
        } else {
            eprintln!("{value}")
        }
    }
    pub fn progress_steps(&self, len: u64, unit: &str) -> ProgressBar {
        let bar = if self.progress {
            ProgressBar::new(len)
        } else {
            ProgressBar::hidden()
        };
        // `elapsed<eta`: on a loop whose unit is an optimizer step the remaining
        // time is the figure worth reading, and it is only meaningful because
        // the bar advances between iterations rather than once per epoch.
        bar.set_style(ProgressStyle::with_template(&format!("{{spinner:.green}} [{{elapsed_precise}}<{{eta_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} {unit} {{msg}}")).expect("valid style"));
        bar
    }
    /// Print above the progress bar, unprefixed: these lines are table rows and
    /// have to stay aligned with their borders when the output is piped.
    ///
    /// A wrapped row arrives as several lines in one string; they are printed
    /// one by one so the bar redraws once per line instead of once per block.
    pub fn progress_table(&self, bar: &ProgressBar, value: impl Into<String>) {
        let value = value.into();
        for line in value.split('\n') {
            if self.progress {
                bar.println(line);
            } else {
                eprintln!("{line}");
            }
        }
    }
    /// A diagnostic block emitted while a progress bar is live. Printing it
    /// straight to stderr would tear the bar's redraw, and it is too wide and
    /// too multi-line to fit the table frame, so it goes above the bar as is.
    pub fn progress_diagnostic(&self, bar: &ProgressBar, title: &str, body: &str) {
        let block = format!("{} {title}\n{body}", self.label("diagnostic", "33;1"));
        if self.progress {
            for line in block.split('\n') {
                bar.println(line);
            }
        } else {
            eprintln!("{block}");
        }
    }
    fn label(&self, value: &str, code: &str) -> String {
        if self.color {
            format!("\x1b[{code}m{value}\x1b[0m")
        } else {
            value.into()
        }
    }
}
