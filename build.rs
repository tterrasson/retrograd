#[path = "crates/retrograd-ffi/build/backend_selection.rs"]
mod backend_selection;

use backend_selection::Backends;

fn main() {
    // Dependency build-script cfgs do not propagate to this package. Mirror
    // retrograd-ffi's selection so backend-gated integration tests are built.
    println!("cargo:rerun-if-changed=crates/retrograd-ffi/build/backend_selection.rs");
    backend_selection::declare_cargo_inputs();
    Backends::detect().emit_cfgs();
}
