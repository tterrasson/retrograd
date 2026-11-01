use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().is_some_and(|a| a == "promote") {
        return match rir_gen::promote::run(&args[1..]) {
            Ok(code) => code,
            Err(e) => {
                eprintln!("rir-gen: {e}");
                ExitCode::FAILURE
            }
        };
    }
    let mut out: Option<PathBuf> = None;
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--out" => match args.next() {
                Some(p) => out = Some(PathBuf::from(p)),
                None => {
                    eprintln!("--out expects a path");
                    return ExitCode::FAILURE;
                }
            },
            "--help" | "-h" => {
                println!("usage: rir-gen [--out <dir>]   (default: repository generated/rir)");
                println!(
                    "       rir-gen promote --ratios <file.tsv> [--tolerance 1.05] [--write] [--check]"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("unknown argument: {other}");
                return ExitCode::FAILURE;
            }
        }
    }

    let dir = out.unwrap_or_else(rir_gen::committed_dir);
    match rir_gen::write_to(&dir) {
        Ok(files) => {
            for f in &files {
                println!("wrote {f}");
            }
            println!("{} generated file(s)", files.len());
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("rir-gen: {e}");
            ExitCode::FAILURE
        }
    }
}
