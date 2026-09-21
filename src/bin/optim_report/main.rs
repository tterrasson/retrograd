//! The optimizer evaluation campaign: what Muon and Gefen cost, and how far
//! Gefen's approximation is from the update it approximates.
//!
//! Both optimizers are *correct* - every update is checked against an
//! independent oracle on real weights and the gradient a backward actually
//! emitted - and neither is *evaluated*. Correctness says the kernel computes
//! what the algorithm says; nothing yet says the algorithm helps. This binary
//! produces the figures that decide that, and it decides nothing itself: its
//! output is a table to read, exactly like the offline kernel search.
//!
//! Three sections, each reachable on its own because they need different
//! things:
//!
//! - `quality` needs no model. It drives the two Gefen ops directly over a
//!   synthetic gradient trajectory and compares the update with AdamW's on the
//!   same inputs: cosine, relative error, the reconstruction error of the
//!   quantized first moment, and how far a per-block second moment is from the
//!   per-element one it replaces.
//! - `cost` needs a model. Persistent state bytes, measured optimizer-step
//!   wall time and the device high-water, per optimizer, on the same selection.
//! - `curves` needs a model and a corpus. The loss over a fixed number of
//!   steps, per optimizer, repeated across seeds - the only section that can
//!   say whether an optimizer helps.
//!
//!   cargo run --release --bin optim-report -- --model base.gguf --device gpu

mod cost;
mod quality;

use std::path::PathBuf;

use retrograd::Device;

struct Options {
    model: Option<PathBuf>,
    device: Device,
    sections: Vec<Section>,
    /// Optimizer steps per measured run.
    steps: u32,
    /// Independent seeds per loss curve.
    seeds: u32,
    corpus: Option<PathBuf>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Section {
    Quality,
    Cost,
    Curves,
}

const USAGE: &str = "\
optim-report - the Muon and Gefen evaluation campaign

  --model PATH        the base model the cost and curve sections train
  --device cpu|gpu    where to run (default: cpu)
  --section NAME      quality | cost | curves | all (default: all; repeatable)
  --steps N           optimizer steps per measured run (default: 8)
  --seeds N           independent seeds per loss curve (default: 3)
  --corpus PATH       plain-text corpus for the curve section
  -h, --help          this text

The quality section needs no model. The cost and curve sections are skipped
with a line saying so when no model is given.
";

fn main() {
    let options = match parse() {
        Ok(Some(options)) => options,
        Ok(None) => return,
        Err(message) => {
            eprintln!("optim-report: {message}");
            std::process::exit(2);
        }
    };

    if options.sections.contains(&Section::Quality) {
        quality::run(options.device);
    }
    if options.sections.contains(&Section::Cost) {
        match &options.model {
            Some(model) => cost::run(model, options.device, options.steps),
            None => println!("\ncost: skipped, --model is required"),
        }
    }
    if options.sections.contains(&Section::Curves) {
        match &options.model {
            Some(model) => cost::curves(
                model,
                options.device,
                options.steps,
                options.seeds,
                options.corpus.as_deref(),
            ),
            None => println!("\ncurves: skipped, --model is required"),
        }
    }
}

fn parse() -> Result<Option<Options>, String> {
    let mut options = Options {
        model: None,
        device: Device::Cpu,
        sections: Vec::new(),
        steps: 8,
        seeds: 3,
        corpus: None,
    };
    let mut arguments = std::env::args().skip(1);
    while let Some(flag) = arguments.next() {
        let mut value = || {
            arguments
                .next()
                .ok_or_else(|| format!("{flag} needs a value"))
        };
        match flag.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(None);
            }
            "--model" => options.model = Some(PathBuf::from(value()?)),
            "--corpus" => options.corpus = Some(PathBuf::from(value()?)),
            "--device" => {
                options.device = match value()?.as_str() {
                    "cpu" => Device::Cpu,
                    "gpu" | "auto" => Device::Gpu,
                    other => return Err(format!("unknown device '{other}'")),
                }
            }
            "--steps" => {
                options.steps = value()?
                    .parse()
                    .map_err(|_| "--steps needs a number".to_string())?;
            }
            "--seeds" => {
                options.seeds = value()?
                    .parse()
                    .map_err(|_| "--seeds needs a number".to_string())?;
            }
            "--section" => {
                let name = value()?;
                match name.as_str() {
                    "quality" => options.sections.push(Section::Quality),
                    "cost" => options.sections.push(Section::Cost),
                    "curves" => options.sections.push(Section::Curves),
                    "all" => {
                        options
                            .sections
                            .extend([Section::Quality, Section::Cost, Section::Curves]);
                    }
                    other => return Err(format!("unknown section '{other}'")),
                }
            }
            other => return Err(format!("unknown flag '{other}'")),
        }
    }
    if options.sections.is_empty() {
        options.sections = vec![Section::Quality, Section::Cost, Section::Curves];
    }
    Ok(Some(options))
}
