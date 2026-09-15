//! The flag loop every subcommand shares.
//!
//! One place words what a mistyped command line gets back - `unknown <command>
//! flag '<flag>'`, `missing value for <flag>`, `invalid <flag> value '<value>';
//! expected <what>` and `<command> accepts <what> at most once` - so two
//! subcommands cannot say the same mistake two ways. Each subcommand keeps its
//! own `match`: which flags exist and what a value means stay next to the
//! struct they fill.
//!
//! The `profile` binary includes this file by path: a binary target cannot
//! reach another binary's modules.

use std::str::FromStr;

use retrograd::{Error, Result};

/// The arguments of one subcommand, read left to right.
pub(crate) struct Args<'a> {
    command: &'static str,
    rest: std::iter::Peekable<std::slice::Iter<'a, String>>,
}

impl<'a> Args<'a> {
    pub(crate) fn new(command: &'static str, args: &'a [String]) -> Self {
        Self {
            command,
            rest: args.iter().peekable(),
        }
    }

    /// The next argument, flag or positional.
    pub(crate) fn next_arg(&mut self) -> Option<&'a str> {
        self.rest.next().map(String::as_str)
    }

    /// The value `flag` requires.
    ///
    /// # Errors
    ///
    /// `missing value for <flag>` when the command line ends first.
    pub(crate) fn value(&mut self, flag: &str) -> Result<&'a str> {
        self.next_arg()
            .ok_or_else(|| Error::invalid(format!("missing value for {flag}")))
    }

    /// The next argument if it is a value rather than another flag, for a flag
    /// whose value may be omitted.
    pub(crate) fn optional_value(&mut self) -> Option<&'a str> {
        self.rest
            .next_if(|next| !next.starts_with('-'))
            .map(String::as_str)
    }

    /// The value `flag` requires, parsed.
    ///
    /// # Errors
    ///
    /// `missing value for <flag>`, or the message of [`parse_value`].
    pub(crate) fn parse<T: FromStr>(&mut self, flag: &str, expected: &str) -> Result<T> {
        let value = self.value(flag)?;
        parse_value(flag, value, expected)
    }

    /// The error for a flag this subcommand does not accept.
    pub(crate) fn unknown(&self, flag: &str) -> Error {
        Error::invalid(format!("unknown {} flag '{flag}'", self.command))
    }

    /// Marks `what` as given.
    ///
    /// # Errors
    ///
    /// `<command> accepts <what> at most once` on a second occurrence.
    pub(crate) fn once(&self, seen: &mut bool, what: &str) -> Result<()> {
        if std::mem::replace(seen, true) {
            return Err(Error::invalid(format!(
                "{} accepts {what} at most once",
                self.command
            )));
        }
        Ok(())
    }
}

/// `value` parsed as the type `flag` takes; `expected` ends the sentence
/// ("an integer", "a number").
///
/// # Errors
///
/// `invalid <flag> value '<value>'; expected <expected>`.
pub(crate) fn parse_value<T: FromStr>(flag: &str, value: &str, expected: &str) -> Result<T> {
    value.parse().map_err(|_| {
        Error::invalid(format!(
            "invalid {flag} value '{value}'; expected {expected}"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| value.to_string()).collect()
    }

    #[test]
    fn an_unknown_flag_names_the_subcommand() {
        let raw = strings(&["--nope"]);
        let args = Args::new("bench", &raw);
        assert!(
            args.unknown("--nope")
                .to_string()
                .ends_with("unknown bench flag '--nope'")
        );
    }

    #[test]
    fn a_flag_at_the_end_of_the_line_is_missing_its_value() {
        let raw = strings(&["--model"]);
        let mut args = Args::new("inspect", &raw);
        assert_eq!(args.next_arg(), Some("--model"));
        let error = args.value("--model").expect_err("no value follows");
        assert!(error.to_string().ends_with("missing value for --model"));
        assert!(error.is_user_error());
    }

    #[test]
    fn a_value_of_the_wrong_type_says_what_was_expected() {
        let raw = strings(&["many", "0.5"]);
        let mut args = Args::new("chat", &raw);
        let error = args
            .parse::<u32>("--ctx", "an integer")
            .expect_err("not an integer");
        assert!(
            error
                .to_string()
                .ends_with("invalid --ctx value 'many'; expected an integer")
        );
        assert_eq!(args.parse::<f32>("--temp", "a number").ok(), Some(0.5));
    }

    #[test]
    fn a_second_occurrence_is_refused_and_an_optional_value_stops_at_a_flag() {
        let raw = strings(&["--resume", "--model", "m.gguf"]);
        let mut args = Args::new("train", &raw);
        let mut seen = false;
        assert!(args.once(&mut seen, "--resume").is_ok());
        assert!(
            args.once(&mut seen, "--resume")
                .expect_err("seen twice")
                .to_string()
                .ends_with("train accepts --resume at most once")
        );
        assert_eq!(args.next_arg(), Some("--resume"));
        assert_eq!(args.optional_value(), None);
        assert_eq!(args.next_arg(), Some("--model"));
        assert_eq!(args.optional_value(), Some("m.gguf"));
    }
}
