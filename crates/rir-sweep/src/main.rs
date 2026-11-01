//! `rir-sweep` - the offline schedule search.
//!
//! ```text
//! cargo run --release -p rir-sweep -- --kernel rms_norm_back
//! cargo run --release -p rir-sweep -- --kernel unary_silu --backend vulkan --reps 9
//! cargo run --release -p rir-sweep -- --kernel rms_norm --backend metal
//! cargo run --release -p rir-sweep -- --kernel rms_norm --profile qwen3-0.6b-q8_0 --regime decode
//! cargo run --release -p rir-sweep -- --kernel mul_mat_vec_q4_K --backend metal \
//!     --candidate cols16 --candidate mma --write-proposal /tmp/gemv-q4_K.proposal
//! cargo run --release -p rir-sweep -- --list
//! ```
//!
//! It writes no source. The last block it prints is a **proposal** - a pick
//! or a constructor call with its shape rule, in the vocabulary of the table,
//! which a human reads, argues with, and commits; and which then goes through
//! `scripts/test-rir.sh` like any other table change, because a sweep compares
//! RIR to RIR and promotion is decided against the native kernel.
//! `--write-proposal PATH` gathers the retained rows of every sweep of the run
//! into `PATH`, and refuses a path that holds anything but an earlier
//! proposal.
//!
//! The GEMV's own grid is 54 shapes and its space a few hundred geometries:
//! a full product is hours, and `--candidate` narrows it to the question at
//! hand.

use rir_core::catalog::Regime;
use rir_runtime::any::{AnyGpu, Backend};
use rir_sweep::candidates::{Subject, profile_shapes};
use rir_sweep::{Options, SweepError, SweepReport, report, sweep};

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
    let mut proposal: Option<std::path::PathBuf> = None;
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
            "--write-proposal" => proposal = Some(value("--write-proposal")?.into()),
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
                          --iters, --warmup, --shape, --contraction, --candidate, --latency, \
                          --write-proposal, --list"
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

    // Refused before any measurement: a sweep is long, and finding out at
    // its end that the path was a source file would throw it away.
    if let Some(path) = &proposal {
        check_proposal_target(path)?;
    }

    let mut reports: Vec<SweepReport> = Vec::new();
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
                Ok(r) => {
                    print!("{}", report::render(&r));
                    reports.push(r);
                }
                Err(e) if e.is_unavailable() => {
                    eprintln!("{}: skipped - {e}", target.name());
                    break;
                }
                Err(e) => return Err(e),
            }
        }
    }
    if let Some(path) = &proposal {
        std::fs::write(path, report::proposal_file(&reports))?;
        eprintln!("rir-sweep: proposal written to {}", path.display());
    }
    Ok(())
}

/// A proposal path is new, or holds an earlier proposal: anything else - the
/// table, any source file - is refused.
fn check_proposal_target(path: &std::path::Path) -> Result<(), SweepError> {
    match std::fs::read_to_string(path) {
        Ok(existing) if existing.starts_with(report::PROPOSAL_HEADER) => Ok(()),
        Ok(_) => Err(SweepError::ProposalTarget {
            path: path.to_path_buf(),
        }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err(SweepError::ProposalTarget {
            path: path.to_path_buf(),
        }),
    }
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
    use super::{check_proposal_target, shape};

    #[test]
    fn a_proposal_never_overwrites_a_source_file() {
        let dir = std::env::temp_dir().join(format!("rir-sweep-proposal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let fresh = dir.join("new.proposal");
        assert!(check_proposal_target(&fresh).is_ok());
        let earlier = dir.join("earlier.proposal");
        std::fs::write(&earlier, rir_sweep::report::proposal_file(&[])).unwrap();
        assert!(check_proposal_target(&earlier).is_ok());
        let source = dir.join("table.rs");
        std::fs::write(&source, "pub fn table() {}\n").unwrap();
        assert!(check_proposal_target(&source).is_err());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn shape_rejects_zero_and_values_the_manifest_would_truncate() {
        assert!(shape("0,1,1,1").is_err());
        if usize::BITS > 32 {
            assert!(shape("4294967296,1,1,1").is_err());
        }
        assert_eq!(shape("33,256,1,1").unwrap(), [33, 256, 1, 1]);
    }
}
