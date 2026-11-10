use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Read the actual native selection, including features enabled transitively.
    // DEP_* comes from the direct dependency's `links` metadata, not the shell.
    for backend in ["metal", "vulkan", "cuda"] {
        println!("cargo:rustc-check-cfg=cfg(retro_{backend})");
        let key = format!("DEP_RETRO_RUNTIME_{}", backend.to_ascii_uppercase());
        match env::var(&key).as_deref() {
            Ok("1") => println!("cargo:rustc-cfg=retro_{backend}"),
            Ok("0") => {}
            other => panic!("expected {key}=0 or 1 from retrograd-ffi, got {other:?}"),
        }
    }
    if env::var_os("CARGO_FEATURE_CLI").is_some() {
        record_source_commit();
    }
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    output.status.success().then_some(())?;
    Some(String::from_utf8(output.stdout).ok()?.trim().to_owned())
}

fn record_source_commit() {
    if !Path::new(".git").exists() {
        return;
    }
    let Some(commit) = git(&["rev-parse", "HEAD"]) else {
        return;
    };
    let branch = git(&["symbolic-ref", "-q", "HEAD"]);
    for name in ["HEAD", "logs/HEAD", "packed-refs"]
        .into_iter()
        .chain(branch.as_deref())
    {
        if let Some(path) = git(&["rev-parse", "--git-path", name])
            && Path::new(&path).exists()
        {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    println!("cargo:rustc-env=RETROGRAD_SOURCE_COMMIT={commit}");
}
