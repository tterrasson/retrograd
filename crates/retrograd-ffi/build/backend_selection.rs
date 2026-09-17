use std::{env, ffi::OsString};

/// Backends compiled into llama.cpp. CPU is always present; optional
/// accelerators come from Cargo features. `platform-gpu` (Metal on macOS)
/// applies only when no backend is named explicitly.
pub struct Backends {
    pub metal: bool,
    pub vulkan: bool,
    pub cuda: bool,
}

impl Backends {
    pub fn detect() -> Self {
        let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
        Self::from_env(&target_os, |name| env::var_os(name))
    }

    fn from_env(target_os: &str, var: impl Fn(&str) -> Option<OsString>) -> Self {
        let feature = |name| var(name).is_some();
        let metal = feature("CARGO_FEATURE_METAL");
        let cuda = feature("CARGO_FEATURE_CUDA");

        if metal && target_os != "macos" {
            panic!("Cargo feature `metal` is only supported on macOS targets");
        }

        if cuda && target_os == "macos" {
            panic!("Cargo feature `cuda` is not supported on macOS; use `platform-gpu` or `metal`");
        }

        let vulkan = feature("CARGO_FEATURE_VULKAN");
        // `platform-gpu` is the default, not an addition: naming any backend
        // replaces it, so `--features vulkan` needs no `--no-default-features`.
        let platform_metal = target_os == "macos"
            && feature("CARGO_FEATURE_PLATFORM_GPU")
            && !(metal || vulkan || cuda);

        Self {
            metal: metal || platform_metal,
            vulkan,
            cuda,
        }
    }

    /// Exposes the selected backends twice: as `links` metadata for direct
    /// dependents (`DEP_RETRO_RUNTIME_*`, read by the root `build.rs`), and as
    /// `cfg`s for Rust code in the current package.
    pub fn emit_cfgs(&self) {
        println!("cargo:metal={}", u8::from(self.metal));
        println!("cargo:vulkan={}", u8::from(self.vulkan));
        println!("cargo:cuda={}", u8::from(self.cuda));
        if self.metal {
            println!("cargo:rustc-cfg=retro_metal");
        }
        if self.vulkan {
            println!("cargo:rustc-cfg=retro_vulkan");
        }
        if self.cuda {
            println!("cargo:rustc-cfg=retro_cuda");
        }
    }
}

pub fn declare_cargo_inputs() {
    for backend in ["metal", "vulkan", "cuda"] {
        println!("cargo:rustc-check-cfg=cfg(retro_{backend})");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn select(target: &str, features: &[&str]) -> Backends {
        Backends::from_env(target, |name| features.contains(&name).then(|| "1".into()))
    }

    #[test]
    fn direct_ffi_defaults_to_cpu_on_every_platform() {
        for target in ["macos", "linux", "windows"] {
            let b = select(target, &[]);
            assert!(!b.metal && !b.vulkan && !b.cuda);
        }
    }

    #[test]
    fn platform_gpu_is_metal_only_on_macos() {
        for target in ["macos", "linux", "windows"] {
            let b = select(target, &["CARGO_FEATURE_PLATFORM_GPU"]);
            assert_eq!(b.metal, target == "macos");
            assert!(!b.vulkan && !b.cuda);
        }
    }

    #[test]
    fn explicit_features_are_additive() {
        let b = select(
            "linux",
            &[
                "CARGO_FEATURE_CUDA",
                "CARGO_FEATURE_VULKAN",
                "CARGO_FEATURE_PLATFORM_GPU",
            ],
        );
        assert!(!b.metal && b.vulkan && b.cuda);
        let b = select("macos", &["CARGO_FEATURE_METAL", "CARGO_FEATURE_VULKAN"]);
        assert!(b.metal && b.vulkan && !b.cuda);
    }

    #[test]
    fn an_explicit_backend_replaces_the_platform_default() {
        let b = select("macos", &["CARGO_FEATURE_PLATFORM_GPU", "CARGO_FEATURE_VULKAN"]);
        assert!(!b.metal && b.vulkan && !b.cuda);
        let b = select("macos", &["CARGO_FEATURE_PLATFORM_GPU", "CARGO_FEATURE_METAL"]);
        assert!(b.metal && !b.vulkan && !b.cuda);
    }

    #[test]
    #[should_panic(expected = "`metal` is only supported on macOS")]
    fn explicit_metal_is_rejected_on_linux() {
        select("linux", &["CARGO_FEATURE_METAL"]);
    }

    #[test]
    #[should_panic(expected = "`cuda` is not supported on macOS")]
    fn explicit_cuda_is_rejected_on_macos() {
        select("macos", &["CARGO_FEATURE_CUDA"]);
    }
}
