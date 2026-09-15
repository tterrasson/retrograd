//! Copies every generated artifact a production build must carry into the fork,
//! from the same derivation `the_fork_copies_match_the_generated_artefacts`
//! checks it with.
//!
//! A developer script, run by hand: a panic here is the error report, and the
//! workspace `unwrap_used` deny is aimed at
//! library code that has a caller to answer to. `clippy.toml` exempts tests by
//! configuration; an example needs it said here.
#![allow(clippy::unwrap_used)]
fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fork = root.join("crates/retrograd-ffi/runtime/vendor/llama.cpp/ggml/src");
    let generated = root.join("generated/rir");
    let mut n = 0;
    let mut kept: Vec<std::path::PathBuf> = Vec::new();
    for a in rir_gen::production_artifacts().unwrap() {
        let src = generated.join(&a.kernel).join(&a.source_file);
        let dst = match a.backend {
            rir_emit::GgmlBackend::Cuda => {
                fork.join(format!("ggml-cuda/rir/rir_{}.cu", a.artifact))
            }
            rir_emit::GgmlBackend::Vulkan => fork.join(format!(
                "ggml-vulkan/vulkan-shaders/rir_{}.comp",
                a.artifact
            )),
            _ => continue,
        };
        std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("{src:?} -> {dst:?}: {e}"));
        kept.push(dst);
        n += 1;
    }
    // A variant the table stopped emitting leaves a file behind, and the fork
    // compiles what it globs: a stale `rir_<artifact>.cu` names a params struct
    // the header does not declare, so it is a build error rather than dead
    // code. Removing it here is the other half of copying.
    let mut removed = 0;
    for (dir, prefix, ext) in [
        ("ggml-cuda/rir", "rir_", ".cu"),
        ("ggml-vulkan/vulkan-shaders", "rir_", ".comp"),
    ] {
        for e in std::fs::read_dir(fork.join(dir)).unwrap() {
            let path = e.unwrap().path();
            let name = path.file_name().unwrap().to_string_lossy().to_string();
            if !name.starts_with(prefix) || !name.ends_with(ext) || kept.contains(&path) {
                continue;
            }
            std::fs::remove_file(&path).unwrap();
            println!("removed stale {name}");
            removed += 1;
        }
    }
    // The generated files the fork carries that are not per-artifact: the
    // registry, the parameter header, and the three per-backend lists. Copied
    // here too, so "sync the fork" is one command and not one command plus a
    // list to remember.
    for (dir, rel, sub) in [
        ("registry", "rir_registry.h", "ggml-rir"),
        ("registry", "rir_registry.cpp", "ggml-rir"),
        ("registry", "rir_kernel_params.h", "ggml-rir"),
        ("cuda", "rir_cuda_launchers.h", "ggml-cuda/rir"),
        ("vulkan", "rir_vk_artifacts.h", "ggml-vulkan/vulkan-shaders"),
        ("metal", rir_gen::METAL_AGGREGATE, rir_gen::METAL_FORK_DIR),
        ("quant", "ggml-retro-quant.h", ""),
    ] {
        let src = generated.join(dir).join(rel);
        let dst = fork.join(sub).join(rel);
        std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("{src:?} -> {dst:?}: {e}"));
    }
    println!("{n} artifact(s) copied, {removed} stale removed");
}
