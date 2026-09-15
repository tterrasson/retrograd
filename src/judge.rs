//! `retrograd judge eval` - how good is the judge that scores an agentic run.
//!
//! It reads the same document `train` reads and takes `[agent.judge]` out of
//! it, which is the point: a judge measured here is byte-for-byte the judge the
//! run will use. No model is loaded, on purpose - measuring a judge must not
//! require a GPU, otherwise nobody measures it.

use std::path::PathBuf;

use retrograd::config::{self, Algorithm};
use retrograd::{Error, Result};

pub(crate) const FLAGS: &[&str] = &["--fixtures"];

pub(crate) fn judge(args: Vec<String>) -> Result<()> {
    let mut args = args.into_iter();
    if args.next().as_deref() != Some("eval") {
        return Err(Error::invalid(
            "usage: retrograd judge eval CONFIG.toml --fixtures FIXTURES.jsonl",
        ));
    }
    let path = PathBuf::from(
        args.next()
            .ok_or_else(|| Error::invalid("judge eval requires CONFIG.toml"))?,
    );
    let mut fixtures = None;
    while let Some(argument) = args.next() {
        if !FLAGS.contains(&argument.as_str()) {
            return Err(Error::invalid(format!(
                "unknown judge eval argument '{argument}'"
            )));
        }
        match argument.as_str() {
            "--fixtures" => {
                fixtures =
                    Some(PathBuf::from(args.next().ok_or_else(|| {
                        Error::invalid("--fixtures requires a path")
                    })?))
            }
            other => {
                return Err(Error::invalid(format!(
                    "unknown judge eval argument '{other}'"
                )));
            }
        }
    }
    let fixtures =
        fixtures.ok_or_else(|| Error::invalid("judge eval requires --fixtures FIXTURES.jsonl"))?;
    let run_config = config::load(path)?;
    let Algorithm::AgentGrpo(agent) = &run_config.algorithm else {
        return Err(Error::invalid(
            "judge eval needs a configuration with run.algorithm = 'agent_grpo': the judge it \
             measures is the one declared in [agent.judge]",
        ));
    };
    println!("{}", retrograd::run::evaluate_judge(agent, &fixtures)?);
    Ok(())
}
