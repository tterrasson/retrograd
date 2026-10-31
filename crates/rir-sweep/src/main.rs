//! `rir-sweep` - the offline schedule search.
//!
//! ```text
//! cargo run --release -p rir-sweep -- --kernel rms_norm_back
//! cargo run --release -p rir-sweep -- --kernel unary_silu --backend vulkan --reps 9
//! cargo run --release -p rir-sweep -- --kernel rms_norm --backend metal
//! cargo run --release -p rir-sweep -- --kernel rms_norm --profile qwen3-0.6b-q8_0 --regime decode
//! cargo run --release -p rir-sweep -- --list
//! ```
//!
//! It writes nothing. The last block it prints is a **proposal** - a
//! constructor call and a shape rule, in the vocabulary of `schedules_for`,
//! which a human reads, argues with, and commits; and which then goes through
//! `scripts/test-rir.sh` like any other table change, because a sweep compares
//! RIR to RIR and promotion is decided against the native kernel.

use rir_core::catalog::Regime;
use rir_runtime::any::{AnyGpu, Backend};
use rir_sweep::candidates::{Subject, profile_shapes};
use rir_sweep::{Options, SweepError, report, sweep};

fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rir-sweep: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

fn run() -> Result<(), SweepError> {
    let mut kernel: Option<String> = None;
    let mut backend: Option<Backend> = None;
    let mut profile: Option<String> = None;
    let mut regime: Option<Regime> = None;
    let mut options = Options::default();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = |name: &str| {
            args.next().ok_or_else(|| SweepError::Unbindable {
                kernel: name.to_string(),
                why: format!("{name} takes a value"),
            })
        };
        match arg.as_str() {
            "--list" => {
                for name in Subject::names() {
                    println!("{name}");
                }
                return Ok(());
            }
            "--kernel" => kernel = Some(value("--kernel")?),
            "--profile" => profile = Some(value("--profile")?),
            "--regime" => {
                regime = Some(match value("--regime")?.as_str() {
                    "decode" => Regime::Decode,
                    "prefill" => Regime::Prefill,
                    other => {
                        return Err(SweepError::Unbindable {
                            kernel: other.to_string(),
                            why: "--regime is decode or prefill".to_string(),
                        });
                    }
                })
            }
            "--backend" => {
                let name = value("--backend")?;
                backend = Some(match name.as_str() {
                    "vulkan" => Backend::Vulkan,
                    "cuda" => Backend::Cuda,
                    "metal" => Backend::Metal,
                    other => {
                        return Err(SweepError::Unbindable {
                            kernel: other.to_string(),
                            why: "--backend is vulkan, cuda or metal".to_string(),
                        });
                    }
                });
            }
            "--candidate" => options.candidate_filters.push(value("--candidate")?),
            "--latency" => options.mode = rir_sweep::measure::Mode::Latency,
            "--reps" => options.reps = parse(&value("--reps")?)?,
            "--iters" => options.iters = parse(&value("--iters")?)?,
            "--contraction" => {
                options.contraction =
                    usize::try_from(parse(&value("--contraction")?)?).map_err(|_| {
                        SweepError::Unbindable {
                            kernel: "--contraction".into(),
                            why: "contraction exceeds usize".into(),
                        }
                    })?
            }
            "--warmup" => options.warmup = parse(&value("--warmup")?)?,
            "--shape" => options
                .shapes
                .get_or_insert_with(Vec::new)
                .push(shape(&value("--shape")?)?),
            other => {
                return Err(SweepError::Unbindable {
                    kernel: other.to_string(),
                    why: "unknown argument; --kernel, --backend, --profile, --regime, --reps, \
                          --iters, --warmup, --shape, --contraction, --candidate, --latency, --list"
                        .to_string(),
                });
            }
        }
    }

    let Some(kernel) = kernel else {
        return Err(SweepError::Unbindable {
            kernel: "-".to_string(),
            why: "--kernel is required (see --list)".to_string(),
        });
    };
    let subject = Subject::resolve(&kernel)?;

    // One sweep per regime: a profile's proposal is made for one regime, and
    // a kernel may win in one and not in the other. Without a profile there
    // is a single sweep, on `--shape` or the family's own list.
    let runs: Vec<Options> = match profile {
        None => {
            if regime.is_some() {
                return Err(SweepError::Unbindable {
                    kernel: kernel.clone(),
                    why: "--regime selects a profile's shapes; it needs --profile".to_string(),
                });
            }
            vec![options]
        }
        Some(name) => {
            if options.shapes.is_some() {
                return Err(SweepError::Unbindable {
                    kernel: kernel.clone(),
                    why: "--profile derives the shapes; it does not combine with --shape"
                        .to_string(),
                });
            }
            let p = rir_core::profiles::named(&name).ok_or_else(|| SweepError::Unbindable {
                kernel: name.clone(),
                why: format!(
                    "unknown profile; the catalogue publishes {}",
                    rir_core::profiles::reference()
                        .iter()
                        .map(|p| p.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            })?;
            let regimes = match regime {
                Some(r) => vec![r],
                None => vec![Regime::Decode, Regime::Prefill],
            };
            let mut runs = Vec::new();
            for r in regimes {
                let shapes = profile_shapes(subject.family, &p.shape, r).ok_or_else(|| {
                    SweepError::Unbindable {
                        kernel: kernel.clone(),
                        why: format!(
                            "the {} family does not run in a model's forward pass, so profile \
                             {name} has no shape for it",
                            subject.family.name()
                        ),
                    }
                })?;
                runs.push(Options {
                    shapes: Some(shapes),
                    profile: Some((name.clone(), r)),
                    ..options.clone()
                });
            }
            runs
        }
    };

    let targets: Vec<Backend> = match backend {
        Some(b) => vec![b],
        None => Backend::available(),
    };
    for target in targets {
        let gpu = match AnyGpu::open(target) {
            Ok(g) => g,
            // A backend this machine cannot open is a skip with its reason, not
            // a failure: the same discipline the device lanes carry.
            Err(e) => {
                eprintln!("{}: skipped - {e}", target.name());
                continue;
            }
        };
        for options in &runs {
            match sweep(&subject, target, &gpu, options) {
                Ok(r) => print!("{}", report::render(&r)),
                Err(e) if e.is_unavailable() => {
                    eprintln!("{}: skipped - {e}", target.name());
                    break;
                }
                Err(e) => return Err(e),
            }
        }
    }
    Ok(())
}

fn parse(v: &str) -> Result<u32, SweepError> {
    v.parse().map_err(|_| SweepError::Unbindable {
        kernel: v.to_string(),
        why: "expected a number".to_string(),
    })
}

/// `col,row,plane,batch`.
fn shape(v: &str) -> Result<[usize; 4], SweepError> {
    let bad = || SweepError::Unbindable {
        kernel: v.to_string(),
        why: "a shape is col,row,plane,batch".to_string(),
    };
    let mut ne = [1usize; 4];
    let parts: Vec<&str> = v.split(',').collect();
    if parts.len() != 4 {
        return Err(bad());
    }
    for (i, p) in parts.iter().enumerate() {
        ne[i] = p.parse().map_err(|_| bad())?;
        if ne[i] == 0 || u32::try_from(ne[i]).is_err() {
            return Err(SweepError::Unbindable {
                kernel: v.to_string(),
                why: "shape extents must be between 1 and u32::MAX".to_string(),
            });
        }
    }
    Ok(ne)
}

#[cfg(test)]
mod tests {
    use super::shape;

    #[test]
    fn shape_rejects_zero_and_values_the_manifest_would_truncate() {
        assert!(shape("0,1,1,1").is_err());
        if usize::BITS > 32 {
            assert!(shape("4294967296,1,1,1").is_err());
        }
        assert_eq!(shape("33,256,1,1").unwrap(), [33, 256, 1, 1]);
    }
}
