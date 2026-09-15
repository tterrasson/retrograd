use std::env;

/// Backends compiled into llama.cpp. CPU is always present; optional
/// accelerators come from one explicit list shared by both Cargo packages.
pub struct Backends {
    pub metal: bool,
    pub vulkan: bool,
    pub cuda: bool,
}

impl Backends {
    pub fn detect() -> Self {
        let target_os = env::var("CARGO_CFG_TARGET_OS").expect("CARGO_CFG_TARGET_OS");
        let requested = env::var("RETRO_BACKENDS").unwrap_or_else(|_| {
            if target_os == "macos" {
                "cpu,metal".to_owned()
            } else {
                "cpu".to_owned()
            }
        });

        let mut selected = Self {
            metal: false,
            vulkan: false,
            cuda: false,
        };
        for backend in requested
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
        {
            match backend.to_ascii_lowercase().as_str() {
                "cpu" => {}
                "metal" => selected.metal = true,
                "vulkan" => selected.vulkan = true,
                "cuda" => selected.cuda = true,
                other => panic!(
                    "unknown RETRO_BACKENDS backend '{other}'; use cpu, metal, vulkan, or cuda"
                ),
            }
        }

        if selected.metal && target_os != "macos" {
            panic!("RETRO_BACKENDS=metal is only supported on macOS targets");
        }
        if selected.cuda && target_os == "macos" {
            panic!(
                "RETRO_BACKENDS=cuda is not supported on macOS; NVIDIA CUDA has no macOS \
                 toolkit. Use RETRO_BACKENDS=cpu,metal on Apple hardware."
            );
        }
        selected
    }

    /// Exposes the selected backends to Rust code in the current package.
    pub fn emit_cfgs(&self) {
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
    println!("cargo:rerun-if-env-changed=RETRO_BACKENDS");
}
