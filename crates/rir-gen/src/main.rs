use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut out: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
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
