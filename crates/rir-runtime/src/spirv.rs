//! GLSL → SPIR-V at runtime, through the system compiler.
//!
//! SPIR-V is **not** committed: it depends on the compiler version, which would
//! break the property that regenerating produces no diff. The `.comp` remains
//! the generator output, and compilation is a runtime detail - the same one
//! already performed by the generation test.

use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::RuntimeError;

/// Discriminant for temporary files within a process.
static NEXT_OUTPUT: AtomicU64 = AtomicU64::new(0);

/// Subgroup collectives require SPIR-V 1.3, hence a Vulkan 1.1 target.
const COMPILERS: [(&str, &[&str]); 2] = [
    (
        "glslc",
        &["-fshader-stage=comp", "--target-env=vulkan1.1", "-o"],
    ),
    (
        "glslangValidator",
        &["-S", "comp", "--target-env", "vulkan1.1", "-o"],
    ),
];

/// Compiles a generated `kernel.comp` and returns the SPIR-V words.
pub fn compile_glsl(comp: &Path) -> Result<Vec<u32>, RuntimeError> {
    let Some((cmd, args)) = COMPILERS
        .iter()
        .find(|(c, _)| Command::new(c).arg("--version").output().is_ok())
    else {
        return Err(RuntimeError::NoCompiler);
    };

    // Unique output name per call: the PID separates processes and the counter
    // separates calls within one process. Without it, two concurrent
    // compilations of the *same* kernel overwrite or delete each other's output.
    let stem = comp
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "kernel".to_string());
    let seq = NEXT_OUTPUT.fetch_add(1, Ordering::Relaxed);
    let out = std::env::temp_dir().join(format!("rir-{stem}-{}-{seq}.spv", std::process::id()));

    let status = Command::new(cmd)
        .args(*args)
        .arg(&out)
        .arg(comp)
        .output()
        .map_err(|e| RuntimeError::Compile(format!("executing {cmd}: {e}")))?;
    if !status.status.success() {
        let _ = std::fs::remove_file(&out);
        return Err(RuntimeError::Compile(format!(
            "{cmd} on {}:\n{}{}",
            comp.display(),
            String::from_utf8_lossy(&status.stdout),
            String::from_utf8_lossy(&status.stderr)
        )));
    }

    let bytes = std::fs::read(&out)?;
    let _ = std::fs::remove_file(&out);
    if bytes.len() % 4 != 0 {
        return Err(RuntimeError::Compile(
            "SPIR-V size is not a multiple of 4".into(),
        ));
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}
