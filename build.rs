use std::env;

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
}
