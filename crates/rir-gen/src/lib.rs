//! rir-gen - the AOT generator.
//!
//! This build-time binary writes files to disk. Generated code is **committed**
//! under `generated/rir/`; the `regenerating_produces_no_diff` unit test is
//! the CI check that committed files exactly match generator output, including
//! missing and extra files.
//!
//! Every kernel is emitted for each schedule in its table:
//! `cpu.rs`, `kernel.comp` + `manifest.vulkan.json`, `kernel.metal` +
//! `manifest.metal.json`... Only a GPU lowering carries a manifest: `cpu.rs`
//! is Rust compiled into the workspace, so it has no loader to inform.
//! A (kernel, backend) pair may
//! publish several **variants**, arbitrated per shape at dispatch; each one
//! carries its name in those file names (`kernel.blocked.comp`), so two
//! lowerings of one kernel coexist instead of overwriting each other.
//!
//! ```sh
//! cargo run -p rir-gen                      # writes generated/rir/
//! cargo run -p rir-gen -- --out <dir>       # writes elsewhere
//! ```

use std::fs;
use std::io;
use std::path::Path;

use rir_lower::Backend;

#[derive(Debug, thiserror::Error)]
pub enum GenError {
    #[error("lowering: {0}")]
    Lower(#[from] rir_lower::LowerError),
    #[error("emission: {0}")]
    Emit(#[from] rir_emit::EmitError),
    #[error("schedule table: {0}")]
    Schedule(#[from] rir_lower::ScheduleError),
    /// A registry rule (kernels ↔ specs ↔ policies) not satisfied by the
    /// declared tables. It is checked before first emission by
    /// `validate_registry`, then while emitting the C++ registry for properties
    /// that depend on lowering.
    #[error("registry: {0}")]
    Registry(#[from] rir_emit::RegistryError),
    /// A backend served by no emitter. Unreachable now that the last one
    /// closed, and kept because the type makes "every backend of the table has an
    /// emitter" a checked property rather than an assumption.
    #[error("no emitter for backend {0}")]
    NoEmitter(&'static str),
}

/// Everything one registry entry produces: a subdirectory name under
/// `generated/rir/` and the files it holds.
pub struct GeneratedKernel {
    pub name: String,
    /// Relative path within the kernel directory and file contents.
    pub files: Vec<(String, String)>,
}

/// Directory name of the AOT registry pseudo-kernel under `generated/rir/`.
pub const REGISTRY_DIR: &str = "registry";

/// Versioned execution catalogue consumed by the planner and runtime probes.
pub const CATALOG_DIR: &str = "catalog";
pub const CATALOG_FILE: &str = "catalog.json";

/// Directory name of the quantized-format pseudo-kernel.
pub const QUANT_DIR: &str = "quant";

/// Directory name of the Metal aggregate pseudo-kernel.
pub const METAL_DIR: &str = "metal";

/// The single Metal file the fork carries: every production kernel body,
/// concatenated. It replaces a marker-delimited block recopied by hand into a
/// 14 000-line vendored source - 481 lines of rebase surface for code nobody
/// writes by hand.
///
/// It was an include fragment pulled into `ggml-metal.metal` until upstream
/// split that file into one translation unit per family under
/// `ggml-metal/kernels/`. The aggregate became one of them, named by their
/// convention and compiled on its own, so it now opens with the `common.h`
/// every sibling opens with; what the fork lists is a file rather than an
/// `#include`, which is the same single line of vendored patch.
pub const METAL_AGGREGATE: &str = "rir.metal";

/// Where the fork carries [`METAL_AGGREGATE`], relative to `ggml/src/`.
pub const METAL_FORK_DIR: &str = "ggml-metal/kernels";

/// Directory name of the Vulkan artifact-list pseudo-kernel.
pub const VULKAN_DIR: &str = "vulkan";

/// The single Vulkan file the fork includes: the list of generated shader
/// modules, as an X-macro. The fork includes this list through one header, so
/// adding a generated variant does not require a per-variant source edit.
pub const VULKAN_ARTIFACT_LIST: &str = "rir_vk_artifacts.h";

/// Directory name of the CUDA launcher-list pseudo-kernel.
pub const CUDA_DIR: &str = "cuda";

/// The single CUDA file the fork includes: the list of generated launch stubs,
/// as an X-macro.
///
/// CUDA's answer to the same problem Metal solved with an `#include` and Vulkan
/// with a shader list, and it exists for a reason neither of them has: a
/// `__global__` kernel is not launched by its name, so what the fork
/// needs is a **table of function pointers**, and a table written by hand is a
/// table a new kernel can be missing from.
pub const CUDA_LAUNCHER_LIST: &str = "rir_cuda_launchers.h";

/// Lines of the standalone header an emitted MSL file carries: three comment
/// lines, `#include <metal_stdlib>`, `using namespace metal;`, and a blank. The
/// aggregate drops them - the `common.h` it opens with has already opened the
/// namespace.
const MSL_HEADER_LINES: usize = 6;

pub mod catalog;
pub mod validate;

pub use catalog::{
    ProductionArtifact, cuda_launcher_list, emit_catalog, metal_aggregate, production_artifacts,
    vulkan_artifact_list,
};
pub use validate::{check_gpu_coverage, validate_registry, validate_tables};

/// Generates every registered kernel for every schedule in the table, plus
/// the C++ AOT registry under `registry/` - same data as the manifests, one
/// generator run.
///
/// # Errors
///
/// `GenError::Registry` if declared tables do not satisfy their rules - checked
/// **before** first emission - followed by schedule, lowering, and emission
/// errors from the first failing kernel.
pub fn generate_all() -> Result<Vec<GeneratedKernel>, GenError> {
    validate_registry()?;
    let entries = rir_kernels::registry();
    let specs: Vec<rir_emit::IntegrationSpec> =
        entries.iter().map(|e| e.integration.clone()).collect();
    let mut out = Vec::new();
    // (LoopKernel, spec) pairs whose backend runs the variant in production;
    // they populate the registry's variant table.
    let mut production: Vec<(rir_lower::LoopKernel, rir_emit::IntegrationSpec)> = Vec::new();
    // Kernels with a CUDA lowering that no backend dispatches; see
    // `emit_params_header`.
    let mut oracle_params: Vec<rir_lower::LoopKernel> = Vec::new();
    // MSL bodies of the production Metal variants, in generation order, for the
    // single file the fork includes.
    let mut metal_bodies: Vec<(String, String)> = Vec::new();
    // Artifact names of the production Vulkan variants, in the same order, for
    // the list the shader generator includes.
    let mut vulkan_artifacts: Vec<String> = Vec::new();
    // Artifact names of the production CUDA variants, for the launcher table the
    // fork expands.
    let mut cuda_artifacts: Vec<String> = Vec::new();
    for entry in &entries {
        let kernel = &entry.kernel;
        // The spec is part of the registration, not a row found by
        // name - the `Option` remains only because an oracle-only kernel's spec
        // declares no production backend, which the emitters read as "no
        // integration to publish".
        let spec = Some(&entry.integration);
        let mut files = Vec::new();
        let schedules = entry.schedules.clone();
        // The variant table is well-formed *before* anything is emitted: a
        // duplicate identity would overwrite an artifact, a missing fallback
        // would leave a shape selecting nothing. Both are generation errors,
        // not something a device lane should discover.
        rir_lower::check_schedule_table(kernel, &schedules)?;
        for schedule in schedules {
            let backend = schedule.backend();
            let lk = rir_lower::lower(kernel, schedule)?;
            // File names carry the variant, so a second lowering of the same
            // kernel lands beside the first instead of on top of it - the
            // failure that kept the blocked scan out of this pipeline.
            let (source_file, manifest_file) = rir_emit::artifact_files(&lk);
            let source = match backend {
                Backend::Cpu => rir_emit::emit_cpu(&lk),
                Backend::Vulkan => rir_emit::emit_vulkan(&lk),
                Backend::Metal => rir_emit::emit_metal(&lk),
                Backend::Cuda => rir_emit::emit_cuda(&lk),
            };
            let source = source?;
            let ggml_backend = rir_emit::manifest::ggml_backend(&lk);
            let in_production = spec.is_some_and(|s| s.production_on(ggml_backend));
            if in_production && ggml_backend == rir_emit::GgmlBackend::Metal {
                metal_bodies.push((rir_emit::artifact_name(&lk), source.clone()));
            }
            if in_production && ggml_backend == rir_emit::GgmlBackend::Vulkan {
                vulkan_artifacts.push(rir_emit::artifact_name(&lk));
            }
            if in_production && ggml_backend == rir_emit::GgmlBackend::Cuda {
                cuda_artifacts.push(rir_emit::artifact_name(&lk));
            }
            files.push((source_file, source));
            // No manifest for the CPU lowering: `cpu.rs` is Rust compiled into
            // the workspace, so nothing ever reads its manifest - the runtime
            // loads one only to build a Vulkan or CUDA pipeline. Emitting it
            // cost 55 files for no reader.
            if backend != Backend::Cpu {
                files.push((manifest_file, rir_emit::emit_manifest(&lk, spec)));
            }
            if let Some(spec) = spec
                && in_production
            {
                production.push((lk, spec.clone()));
            } else if ggml_backend == rir_emit::GgmlBackend::Cuda {
                // A `.cu` includes `rir_kernel_params.h` for its params
                // struct, so an oracle-only kernel with a CUDA
                // lowering needs a row there too - the parity harness compiles
                // it, and no other backend has the dependency because a `.comp`
                // and a `.metal` declare their own struct.
                oracle_params.push(lk);
            }
        }
        out.push(GeneratedKernel {
            name: kernel.name().to_string(),
            files,
        });
    }

    let refs: Vec<(&rir_lower::LoopKernel, &rir_emit::IntegrationSpec)> =
        production.iter().map(|(k, s)| (k, s)).collect();
    let (header, cpp) = rir_emit::emit_registry(&refs, &specs)?;
    out.push(GeneratedKernel {
        name: REGISTRY_DIR.to_string(),
        files: vec![
            ("rir_registry.h".to_string(), header),
            ("rir_registry.cpp".to_string(), cpp),
            (
                "rir_kernel_params.h".to_string(),
                rir_emit::emit_params_header(&refs, &oracle_params.iter().collect::<Vec<_>>())?,
            ),
        ],
    });

    out.push(GeneratedKernel {
        name: CATALOG_DIR.to_string(),
        files: vec![(CATALOG_FILE.to_string(), emit_catalog(&refs, &specs))],
    });

    // The Metal aggregate: the fork includes this one
    // file in one line instead of carrying a copy of every body between two
    // markers. It is an include fragment, not a translation unit - the
    // standalone per-variant sources keep their own header and stay what the
    // compile test compiles.
    out.push(GeneratedKernel {
        name: METAL_DIR.to_string(),
        files: vec![(METAL_AGGREGATE.to_string(), metal_aggregate(&metal_bodies))],
    });

    // The Vulkan artifact list. Vulkan has no aggregate
    // to build - each shader is compiled to its own SPIR-V module - so what the
    // fork needs from the generator is the *list*, not the code. Emitting it
    // here is what makes adding a variant stop touching a vendored file.
    out.push(GeneratedKernel {
        name: VULKAN_DIR.to_string(),
        files: vec![(
            VULKAN_ARTIFACT_LIST.to_string(),
            vulkan_artifact_list(&vulkan_artifacts),
        )],
    });

    // The CUDA launcher list. Like Vulkan's, it is a
    // list and not code: the bodies are per-artifact translation units the
    // fork's `file(GLOB "rir/*.cu")` picks up, and what a hand cannot be
    // trusted with is the *table* that maps an artifact name to its stub.
    out.push(GeneratedKernel {
        name: CUDA_DIR.to_string(),
        files: vec![(
            CUDA_LAUNCHER_LIST.to_string(),
            cuda_launcher_list(&cuda_artifacts),
        )],
    });

    // The quantized-format table. One output: the X-macro
    // header the fork, MSL and the Vulkan shader generator consume. The Rust
    // side of the same rows is expanded by `rir_core::quant_table` at compile
    // time, so no generated file sits below the generator.
    let formats = rir_core::quant_formats();
    out.push(GeneratedKernel {
        name: QUANT_DIR.to_string(),
        files: vec![(
            "ggml-retro-quant.h".to_string(),
            rir_emit::emit_ggml_header(&formats),
        )],
    });
    Ok(out)
}

/// Runs [`generate_all`] and writes its output under `dir`, one subdirectory
/// per generated kernel. Returns the paths written.
///
/// `dir` is owned entirely by the generator: a subdirectory or file left over
/// from a previous run that no longer matches a registry entry is deleted, so
/// a kernel removed from the registry cannot linger as stale committed output.
pub fn write_to(dir: &Path) -> io::Result<Vec<String>> {
    let kernels =
        generate_all().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    // The generator owns the whole output directory, not just the directories
    // it is about to write: a kernel removed from the registry must disappear
    // from `generated/rir/`, or it stays committed and compiled forever with
    // nothing regenerating it.
    let keep: std::collections::HashSet<&str> = kernels.iter().map(|k| k.name.as_str()).collect();
    if dir.exists() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name();
            if keep.contains(&*name.to_string_lossy()) {
                continue;
            }
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                fs::remove_dir_all(&path)?;
            } else {
                fs::remove_file(&path)?;
            }
        }
    }

    let mut written = Vec::new();
    for gk in kernels {
        let kdir = dir.join(&gk.name);
        // The generator owns this directory, so files from an earlier
        // configuration must not survive.
        if kdir.exists() {
            fs::remove_dir_all(&kdir)?;
        }
        fs::create_dir_all(&kdir)?;
        for (rel, content) in &gk.files {
            let path = kdir.join(rel);
            fs::write(&path, content)?;
            written.push(path.display().to_string());
        }
    }
    Ok(written)
}

/// The repository's committed output directory (`generated/rir` at the root).
pub fn committed_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../generated/rir")
}
