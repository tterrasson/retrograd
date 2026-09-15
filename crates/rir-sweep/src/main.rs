//! `rir-sweep` - the offline schedule search.
//!
//! ```text
//! cargo run --release -p rir-sweep -- --kernel rms_norm_back
//! cargo run --release -p rir-sweep -- --kernel unary_silu --backend vulkan --reps 9
//! cargo run --release -p rir-sweep -- --list
//! ```
//!
//! It writes nothing. The last block it prints is a **proposal** - a
//! constructor call and a shape rule, in the vocabulary of `schedules_for`,
//! which a human reads, argues with, and commits; and which then goes through
//! `scripts/test-rir.sh` like any other table change, because a sweep compares
//! RIR to RIR and promotion is decided against the native kernel.

use rir_runtime::any::{AnyGpu, Backend};
use rir_sweep::candidates::Subject;
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
            "--backend" => {
                let name = value("--backend")?;
                backend = Some(match name.as_str() {
                    "vulkan" => Backend::Vulkan,
                    "cuda" => Backend::Cuda,
                    other => {
                        return Err(SweepError::Unbindable {
                            kernel: other.to_string(),
                            why: "--backend is vulkan or cuda".to_string(),
                        });
                    }
                });
            }
            "--latency" => options.mode = rir_sweep::measure::Mode::Latency,
            "--reps" => options.reps = parse(&value("--reps")?)?,
            "--iters" => options.iters = parse(&value("--iters")?)?,
            "--warmup" => options.warmup = parse(&value("--warmup")?)?,
            "--shape" => options
                .shapes
                .get_or_insert_with(Vec::new)
                .push(shape(&value("--shape")?)?),
            other => {
                return Err(SweepError::Unbindable {
                    kernel: other.to_string(),
                    why: "unknown argument; --kernel, --backend, --reps, --iters, --warmup, \
                          --shape, --latency, --list"
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
        match sweep(&subject, target, &gpu, &options) {
            Ok(r) => print!("{}", report::render(&r)),
            Err(e) if e.is_unavailable() => eprintln!("{}: skipped - {e}", target.name()),
            Err(e) => return Err(e),
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
