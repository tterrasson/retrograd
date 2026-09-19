use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "build/backend_selection.rs"]
mod backend_selection;

use backend_selection::Backends;

/// How the vendored llama.cpp is linked into every consumer of this crate.
///
/// The two modes serve two different kinds of consumer:
///
/// - **Static** is what a binary that leaves this tree needs. It is the only
///   mode that produces a relocatable executable and the only one the Python
///   wheel can use, since a `.so` copied into site-packages finds no build tree
///   to load `libggml.dylib` from.
/// - **Shared** is what a *test* tree needs. The workspace links fifty-odd test
///   binaries against this crate, and a static one embeds the whole native side
///   in each: the 2.4 MB `__ggml_metallib` blob, and the C++ runtime with
///   llama.cpp's `common/` behind it. Measured on the root package's 36 test
///   and binary targets, one build tree: 375 MB of executables static against
///   183 MB shared plus 11 MB of shared libraries, `tests/capabilities.rs`
///   alone going from 12.3 MB to 2.9 MB. Shared makes that one copy per tree
///   rather than one per binary, and shortens every relink by the same objects.
///
/// `RETRO_GGML_LINK=static|shared` overrides the default in either direction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GgmlLink {
    Static,
    Shared,
}

impl GgmlLink {
    /// `shared` for a debug build on macOS, `static` everywhere else,
    /// `RETRO_GGML_LINK` over both.
    ///
    /// Keying on the profile rather than on the target kind is what makes this
    /// automatic: Cargo tells a build script the profile, never whether the
    /// crate above it is a test or the shipped binary. Release is the profile a
    /// build that leaves the tree uses -- `cargo build --release`, the container
    /// feature, the wheel - so it keeps the mode that survives being copied.
    ///
    /// The platform half is not a preference. A Mach-O library carries its own
    /// absolute install name, so a consumer linked against it needs nothing;
    /// the ELF equivalent, `DT_SONAME`, is a bare name the loader resolves
    /// through the search path, and the rpath that would fix it is the one
    /// thing Cargo cannot give a *dependent* crate (`cargo:rustc-link-arg`
    /// reaches this package's own targets, and rustc's `-C rpath` derives its
    /// entries from Rust crate dylibs rather than from a build script's native
    /// `-L` paths). So `shared` on Linux links, and then leaves finding the
    /// libraries to `LD_LIBRARY_PATH`; it is available there, not default.
    fn detect() -> Self {
        println!("cargo:rerun-if-env-changed=RETRO_GGML_LINK");
        match env::var("RETRO_GGML_LINK").as_deref() {
            Ok("shared") => Self::Shared,
            Ok("static") => Self::Static,
            Ok(other) => {
                panic!("RETRO_GGML_LINK: expected `static` or `shared`, got `{other}`")
            }
            Err(_)
                if cfg!(target_os = "macos") && env::var("PROFILE").as_deref() == Ok("debug") =>
            {
                Self::Shared
            }
            Err(_) => Self::Static,
        }
    }

    /// The `kind` of `cargo:rustc-link-lib=<kind>=<name>` for a llama.cpp target.
    fn kind(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Shared => "dylib",
        }
    }
}

fn main() {
    backend_selection::declare_cargo_inputs();
    let backends = Backends::detect();
    backends.emit_cfgs();

    // Vendored runtime sources are part of this build's input set.
    println!("cargo:rerun-if-changed=build/backend_selection.rs");
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let runtime_dir = manifest_dir.join("runtime");
    let llama_cpp_dir = env::var("LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| runtime_dir.join("vendor/llama.cpp"));
    let lock_path = runtime_dir.join("llama.cpp.lock");
    let llama_lock = read_llama_lock(&lock_path);
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let llama_build_dir = out_dir.join("llama-cpp-build");
    let obj_dir = out_dir.join("runtime-objects");
    fs::create_dir_all(&obj_dir).expect("create runtime object dir");

    let runtime_sources = [
        "retro_runtime.cpp",
        "retro_duty_cycle.cpp",
        "retro_backend.cpp",
        "retro_inventory.cpp",
        "retro_lora.cpp",
        "retro_training.cpp",
        "retro_preflight.cpp",
        "retro_rollout.cpp",
        "retro_probe.cpp",
        "retro_fused_ce.cpp",
        "retro_lora_train.cpp",
        "retro_checkpoint.cpp",
        "retro_trainable.cpp",
        "retro_chat_template.cpp",
        "retro_chat_parser.cpp",
    ];

    // Self-contained Jinja and chat-parser sources from llama.cpp's common/
    // directory. The selected closure is compiled because the build disables
    // llama.cpp's common target while the runtime still executes model templates.
    let llama_common_sources = [
        "unicode.cpp",
        // common_json, the JSON facade jinja speaks since the fork bumped past
        // its nlohmann::json interface: jinja/value.cpp only instantiates
        // global_from_json for this type.
        "json.cpp",
        "jinja/lexer.cpp",
        "jinja/parser.cpp",
        "jinja/runtime.cpp",
        "jinja/value.cpp",
        "jinja/string.cpp",
        // The chat parser derives a PEG grammar from the model's own template,
        // matching the renderer in retro_chat_template.cpp. A fixed
        // `<tool_call>` grammar would misread models with native tool formats
        //
        // Keep this dependency closure explicit: no arg.cpp, download.cpp, or
        // console.cpp is needed while -DLLAMA_BUILD_COMMON=OFF remains enabled.
        "chat.cpp",
        "chat-peg-parser.cpp",
        "chat-auto-parser-generator.cpp",
        "chat-auto-parser-helpers.cpp",
        "chat-diff-analyzer.cpp",
        // chat.cpp dispatches the model-specific formats to parsers/ and reads
        // tool schemas through json-schema.cpp; the list mirrors
        // common/parsers/sources.cmake.
        "json-schema.cpp",
        "parsers/parsers.cpp",
        "parsers/cohere2moe.cpp",
        "parsers/deepseek.cpp",
        "parsers/functionary-v3-2.cpp",
        "parsers/gemma4.cpp",
        "parsers/gigachat-v3.cpp",
        "parsers/gpt-oss.cpp",
        "parsers/kimi-k2.cpp",
        "parsers/kimi-k3.cpp",
        "parsers/lfm2.cpp",
        "parsers/minicpm5.cpp",
        "parsers/minimax-m3.cpp",
        "parsers/ministral3.cpp",
        "parsers/muse-glimmer.cpp",
        "parsers/qwen3-coder.cpp",
        "peg-parser.cpp",
        "json-schema-to-grammar.cpp",
        "log.cpp",
        "trie.cpp",
        "jinja/caps.cpp",
        "common.cpp",
        "sampling.cpp",
        "speculative.cpp",
        "fit.cpp",
        "ngram-cache.cpp",
        "ngram-map.cpp",
        "ngram-mod.cpp",
        "reasoning-budget.cpp",
    ];
    let ggml_link = GgmlLink::detect();
    let library = out_dir.join(match ggml_link {
        GgmlLink::Static => "libretro_lora_train.a",
        GgmlLink::Shared if cfg!(target_os = "macos") => "libretro_lora_train.dylib",
        GgmlLink::Shared => "libretro_lora_train.so",
    });

    for source in runtime_sources {
        println!(
            "cargo:rerun-if-changed={}",
            runtime_dir.join("src").join(source).display()
        );
    }
    for source in llama_common_sources {
        println!(
            "cargo:rerun-if-changed={}",
            llama_cpp_dir.join("common").join(source).display()
        );
    }
    println!(
        "cargo:rerun-if-changed={}",
        llama_cpp_dir.join("common/build-info.cpp.in").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        runtime_dir.join("src/retro_runtime.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        runtime_dir.join("include/retro_lora_train.h").display()
    );
    println!("cargo:rerun-if-changed={}", lock_path.display());
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=RETRO_NATIVE");
    println!("cargo:rerun-if-env-changed=RETRO_STRICT_LLAMA");
    println!("cargo:rerun-if-env-changed=RETRO_CUDA_ARCHITECTURES");
    println!("cargo:rerun-if-env-changed=RETRO_CUDA_GRAPHS");
    println!("cargo:rerun-if-env-changed=RETRO_CUDA_FORCE_CUBLAS");
    println!("cargo:rerun-if-env-changed=RETRO_CUDA_HOST_COMPILER");
    if !llama_cpp_dir.join("include/llama.h").exists() {
        panic!(
            "llama.cpp submodule not found at {}; run scripts/setup-llama-cpp.sh \
             (or `git submodule update --init crates/retrograd-ffi/runtime/vendor/llama.cpp`)",
            llama_cpp_dir.display()
        );
    }
    // Rebuild when vendored sources change so in-place fork edits are picked up
    // by the incremental CMake build.
    for tracked in ["CMakeLists.txt", "cmake", "include", "src", "ggml"] {
        println!(
            "cargo:rerun-if-changed={}",
            llama_cpp_dir.join(tracked).display()
        );
    }
    let fork_commit = verify_llama_cpp_checkout(&manifest_dir, &llama_cpp_dir);
    // Checkpoint manifests record the runtime commit used to build the C++ side.
    println!("cargo:rustc-env=RETRO_LLAMA_CPP_COMMIT={fork_commit}");

    if backends.metal {
        println!("cargo:warning=retrograd: building llama.cpp with Metal GPU backend enabled");
    }
    if backends.vulkan {
        println!("cargo:warning=retrograd: building llama.cpp with Vulkan GPU backend enabled");
    }
    if backends.cuda {
        println!("cargo:warning=retrograd: building llama.cpp with CUDA GPU backend enabled");
    }
    configure_llama_cpp(&llama_cpp_dir, &llama_build_dir, &backends, ggml_link);
    build_llama_cpp(&llama_build_dir, &backends);

    let cxx = env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    let mut objects = Vec::new();
    for source in runtime_sources {
        let source_path = runtime_dir.join("src").join(source);
        let object = obj_dir.join(format!(
            "{}.o",
            Path::new(source)
                .file_stem()
                .expect("runtime source has a file stem")
                .to_string_lossy()
        ));
        let mut compile = Command::new(&cxx);
        compile
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-I")
            .arg(runtime_dir.join("include"))
            .arg("-I")
            .arg(llama_cpp_dir.join("include"))
            .arg("-I")
            .arg(llama_cpp_dir.join("src"))
            .arg("-I")
            .arg(llama_cpp_dir.join("ggml/include"))
            // ggml-retro-quant.h (GGML_RETRO_DEQUANT_TYPES) lives with the ggml
            // sources; retro_dequant_types() expands it so tests can enumerate the
            // decodable types instead of restating them.
            .arg("-I")
            .arg(llama_cpp_dir.join("ggml/src"))
            .arg("-I")
            .arg(llama_cpp_dir.join("common"))
            .arg("-I")
            .arg(llama_cpp_dir.join("vendor"))
            .arg(format!("-DRETRO_LLAMA_CPP_COMMIT=\"{fork_commit}\""))
            .arg(format!(
                "-DRETRO_LLAMA_CPP_UPSTREAM_COMMIT=\"{}\"",
                llama_lock.upstream_commit
            ));
        if native_build_enabled() {
            if cfg!(target_arch = "x86") || cfg!(target_arch = "x86_64") {
                compile.arg("-march=native");
            } else if cfg!(target_arch = "aarch64") {
                compile.arg("-mcpu=native");
            }
        }
        compile.arg("-c").arg(&source_path).arg("-o").arg(&object);
        run(&mut compile, &format!("compile {}", source));
        objects.push(object);
    }

    for source in llama_common_sources {
        let source_path = llama_cpp_dir.join("common").join(source);
        let object = obj_dir.join(format!(
            "{}.o",
            Path::new(source)
                .file_stem()
                .expect("llama common source has a file stem")
                .to_string_lossy()
        ));
        let mut compile = Command::new(&cxx);
        compile
            .arg("-std=c++17")
            .arg("-O3")
            .arg("-fPIC")
            .arg("-I")
            .arg(llama_cpp_dir.join("common"))
            .arg("-I")
            .arg(llama_cpp_dir.join("vendor"))
            .arg("-I")
            .arg(llama_cpp_dir.join("include"))
            .arg("-I")
            .arg(llama_cpp_dir.join("ggml/include"));
        if native_build_enabled() {
            if cfg!(target_arch = "x86") || cfg!(target_arch = "x86_64") {
                compile.arg("-march=native");
            } else if cfg!(target_arch = "aarch64") {
                compile.arg("-mcpu=native");
            }
        }
        compile.arg("-c").arg(&source_path).arg("-o").arg(&object);
        run(&mut compile, &format!("compile common/{}", source));
        objects.push(object);
    }

    // CMake normally generates this file for the common target, which is disabled
    // here. Render the template directly so `llama_commit()` and related symbols
    // remain defined.
    objects.push(compile_build_info(
        &cxx,
        &llama_cpp_dir,
        &out_dir,
        &obj_dir,
        &fork_commit,
    ));

    if library.exists() {
        fs::remove_file(&library).expect("remove previous runtime library");
    }
    match ggml_link {
        GgmlLink::Static => {
            let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
            let mut archive = Command::new(&ar);
            archive.arg("crus").arg(&library);
            for object in &objects {
                archive.arg(object);
            }
            run(&mut archive, "archive C++ runtime");
        }
        // Shared for the same reason llama.cpp is, and for more of it: these
        // objects are the C++ runtime *plus* llama.cpp's `common/` -- the chat
        // template chain, jinja, the JSON schema converter -- and they are what
        // a test binary carries most of. An archive puts that in every one of
        // them; a library puts it in the tree once.
        GgmlLink::Shared => {
            let mut link = Command::new(&cxx);
            link.arg("-o").arg(&library);
            for object in &objects {
                link.arg(object);
            }
            if cfg!(target_os = "macos") {
                // Same absolute install name as the llama.cpp libraries, and
                // for the same reason: the consumer must need no rpath.
                link.arg("-dynamiclib").arg("-install_name").arg(&library);
            } else {
                link.arg("-shared");
            }
            // A dynamic library resolves its undefined symbols at link time,
            // where an archive deferred them to whoever pulled its objects in.
            // Everything below is what the *runtime sources* reference and the
            // final Rust link line already names for its own reasons; naming
            // them here too is what makes this library self-contained.
            link.arg(format!(
                "-L{}",
                shared_library_dir(&llama_build_dir).display()
            ));
            for name in ["llama", "ggml", "ggml-base", "ggml-cpu"] {
                link.arg(format!("-l{name}"));
            }
            if backends.metal {
                link.arg("-lggml-metal");
            }
            if cfg!(target_os = "macos") {
                // retro_rollout.cpp uses vDSP/vForce, as below on the Rust link
                // line.
                link.arg("-framework").arg("Accelerate");
            }
            run(&mut link, "link C++ runtime");
        }
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    // A static build leaves each archive beside the CMake target that produced
    // it; a shared one sends every library to `bin/`. Emitting both sets rather
    // than branching keeps the paths that do not exist harmless - a link search
    // directory that is empty costs nothing, and one that is missing is how a
    // mode switch would otherwise fail with an unresolved symbol instead of a
    // clear one.
    for directory in [
        llama_build_dir.join("src"),
        llama_build_dir.join("ggml/src"),
        shared_library_dir(&llama_build_dir),
    ] {
        println!("cargo:rustc-link-search=native={}", directory.display());
    }
    let kind = ggml_link.kind();
    println!("cargo:rustc-link-lib={kind}=retro_lora_train");
    println!("cargo:rustc-link-lib={kind}=llama");
    println!("cargo:rustc-link-lib={kind}=ggml");
    println!("cargo:rustc-link-lib={kind}=ggml-cpu");
    if backends.metal {
        // ggml-base's backend registry references ggml_backend_metal_reg(), so the
        // Metal library must be on the link line; it in turn needs the Apple GPU
        // frameworks.
        println!(
            "cargo:rustc-link-search=native={}",
            llama_build_dir.join("ggml/src/ggml-metal").display()
        );
        println!("cargo:rustc-link-lib={kind}=ggml-metal");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
        emit_compiler_rt_link(&cxx);
    }
    if backends.vulkan {
        // Vulkan shaders are embedded in ggml-vulkan at build time. At runtime
        // the backend only needs the Vulkan loader selected by CMake.
        println!(
            "cargo:rustc-link-search=native={}",
            llama_build_dir.join("ggml/src/ggml-vulkan").display()
        );
        println!("cargo:rustc-link-lib={kind}=ggml-vulkan");
        emit_vulkan_loader_link(&llama_build_dir);
    }
    if backends.cuda {
        // ggml-base's backend registry references ggml_backend_cuda_reg(), so the
        // CUDA library must be on the link line. When it is an archive the CUDA
        // runtime libraries CMake linked to it privately (cudart, cuBLAS, and -
        // unless VMM is disabled - the CUDA driver) are not propagated and must
        // be mirrored on Cargo's final link line. Mirrored in both modes: a
        // shared ggml-cuda propagates them on its own, and naming a dylib twice
        // on a link line costs nothing.
        println!(
            "cargo:rustc-link-search=native={}",
            llama_build_dir.join("ggml/src/ggml-cuda").display()
        );
        println!("cargo:rustc-link-lib={kind}=ggml-cuda");
        emit_cuda_runtime_link(&llama_build_dir);
    }
    println!("cargo:rustc-link-lib={kind}=ggml-base");
    if cfg!(target_os = "macos") {
        // retro_rollout.cpp uses vDSP/vForce for vectorized token log-softmax.
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
        // ggml-cpu is compiled with OpenMP enabled by default. Because Cargo links
        // the static ggml archives directly, CMake's transitive OpenMP::OpenMP_C
        // dependency is not propagated, so the GOMP_*/omp_* symbols would be
        // undefined at link time. Mirror the GNU OpenMP runtime on the link line.
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
}

/// Links clang's compiler-rt builtins, `libclang_rt.osx.a`, from the resource
/// directory of the compiler that built the runtime.
///
/// ggml-metal's `@available` checks call `___isPlatformVersionAtLeast`, which
/// only compiler-rt defines - no Rust sysroot library does. The clang driver
/// adds compiler-rt to what it links itself: the shared ggml-metal, and the
/// executables rustc links through it. A cdylib over the static archives is the
/// case it misses: rustc passes `-nodefaultlibs`, and the PyO3 extension is
/// linked with `-undefined dynamic_lookup`, so the reference is left to the
/// loader, the wheel links, and `import retrograd._native` fails. Naming the
/// archive resolves it at link time. Where the driver already pulled the
/// builtin in, nothing is left undefined for the archive to supply, so the
/// binaries gain no duplicate.
fn emit_compiler_rt_link(cxx: &str) {
    let resource_dir = command_stdout(
        Command::new(cxx).arg("--print-resource-dir"),
        "locate the compiler resource directory",
    );
    let darwin = Path::new(resource_dir.trim()).join("lib/darwin");
    if !darwin.join("libclang_rt.osx.a").is_file() {
        panic!(
            "libclang_rt.osx.a not found in {}; the Metal backend needs its \
             ___isPlatformVersionAtLeast (build with Apple clang, or set CXX to a clang \
             that ships compiler-rt)",
            darwin.display()
        );
    }
    println!("cargo:rustc-link-search=native={}", darwin.display());
    println!("cargo:rustc-link-lib=static=clang_rt.osx");
}

/// Renders `common/build-info.cpp.in` and compiles it. The template is the one
/// CMake uses; only the four `@…@` placeholders differ, and they carry no
/// behaviour - nothing in the chat chain reads them, they only have to be
/// defined for `common.cpp` to link.
fn compile_build_info(
    cxx: &str,
    llama_cpp_dir: &Path,
    out_dir: &Path,
    obj_dir: &Path,
    fork_commit: &str,
) -> PathBuf {
    let template_path = llama_cpp_dir.join("common/build-info.cpp.in");
    let template = fs::read_to_string(&template_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", template_path.display()));
    let target = env::var("TARGET").unwrap_or_else(|_| "unknown".to_owned());
    let rendered = template
        // The fork carries no upstream build number; 0 is the same value
        // CMake writes when the git describe fails.
        .replace("@LLAMA_BUILD_NUMBER@", "0")
        .replace("@LLAMA_BUILD_COMMIT@", fork_commit)
        .replace("@BUILD_COMPILER@", cxx)
        .replace("@BUILD_TARGET@", &target);
    let source_path = out_dir.join("build-info.cpp");
    fs::write(&source_path, rendered).expect("write generated build-info.cpp");

    let object = obj_dir.join("build-info.o");
    let mut compile = Command::new(cxx);
    compile
        .arg("-std=c++17")
        .arg("-O3")
        .arg("-fPIC")
        .arg("-I")
        .arg(llama_cpp_dir.join("common"))
        .arg("-c")
        .arg(&source_path)
        .arg("-o")
        .arg(&object);
    run(&mut compile, "compile common/build-info.cpp");
    object
}

struct LlamaLock {
    upstream_commit: String,
}

fn read_llama_lock(path: &Path) -> LlamaLock {
    let source = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
    let mut values = std::collections::BTreeMap::new();
    for line in source.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
            continue;
        }
        let (key, value) = line.split_once('=').unwrap_or_else(|| {
            panic!("invalid llama.cpp lockfile line '{line}'; expected key = value")
        });
        values.insert(key.trim(), value.trim().trim_matches('"').to_owned());
    }
    let required = |key: &str| {
        values
            .get(key)
            .cloned()
            .unwrap_or_else(|| panic!("{} is missing required key '{key}'", path.display()))
    };
    LlamaLock {
        upstream_commit: required("upstream_commit"),
    }
}

/// Reports the fork commit actually being built and enforces the submodule
/// pin. In strict mode (CI, or RETRO_STRICT_LLAMA=1) any dirty or off-pin
/// checkout aborts the build; in local development it only warns so the fork
/// can be edited in place.
///
/// A source distribution carries the fork's files without its git metadata, so
/// there is no commit to read and no pin to hold it against: the build records
/// "unknown", the value `retro_runtime.hpp` already falls back to. The test is
/// the fork's own `.git` rather than a failing `git rev-parse`, which would
/// answer with the HEAD of any repository the sources were unpacked into.
fn verify_llama_cpp_checkout(manifest_dir: &Path, source_dir: &Path) -> String {
    if !source_dir.join(".git").exists() {
        println!(
            "cargo:warning=retrograd: {} is not a git checkout (a source distribution?); \
             the llama.cpp fork commit is recorded as unknown",
            source_dir.display()
        );
        return "unknown".to_owned();
    }
    let commit = command_stdout(
        Command::new("git")
            .arg("-C")
            .arg(source_dir)
            .arg("rev-parse")
            .arg("HEAD"),
        "read llama.cpp commit",
    )
    .trim()
    .to_owned();
    let dirty = !command_stdout(
        Command::new("git")
            .arg("-C")
            .arg(source_dir)
            .arg("status")
            .arg("--porcelain")
            .arg("--untracked-files=no"),
        "read llama.cpp checkout status",
    )
    .trim()
    .is_empty();

    let strict = env::var("RETRO_STRICT_LLAMA")
        .map(|v| v != "0")
        .unwrap_or(false)
        || env::var_os("CI").is_some();
    let pinned = pinned_submodule_commit(manifest_dir);
    let off_pin = pinned.as_deref().is_some_and(|pin| pin != commit);

    if strict {
        if dirty {
            panic!(
                "llama.cpp submodule at {} has local modifications; commit them in the \
                 fork (strict mode: CI or RETRO_STRICT_LLAMA=1)",
                source_dir.display()
            );
        }
        if off_pin {
            panic!(
                "llama.cpp submodule is at {commit}, but this repository pins {}; \
                 run scripts/setup-llama-cpp.sh or commit the submodule bump",
                pinned.as_deref().unwrap_or("<unknown>")
            );
        }
    } else {
        if dirty {
            println!(
                "cargo:warning=retrograd: building a modified llama.cpp fork checkout \
                 (uncommitted changes in crates/retrograd-ffi/runtime/vendor/llama.cpp)"
            );
        }
        if off_pin {
            println!(
                "cargo:warning=retrograd: llama.cpp submodule is at {commit}, repository \
                 pins {}; commit the submodule bump when the change is intentional",
                pinned.as_deref().unwrap_or("<unknown>")
            );
        }
    }

    if dirty {
        format!("{commit}-dirty")
    } else {
        commit
    }
}

/// The submodule commit recorded in this repository's HEAD, when the default
/// vendored path is in use (a LLAMA_CPP_DIR override is deliberately unpinned).
fn pinned_submodule_commit(manifest_dir: &Path) -> Option<String> {
    if env::var_os("LLAMA_CPP_DIR").is_some() {
        return None;
    }
    let output = Command::new("git")
        .arg("-C")
        .arg(manifest_dir)
        .arg("ls-tree")
        .arg("HEAD")
        .arg("--")
        .arg("runtime/vendor/llama.cpp")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let listing = String::from_utf8(output.stdout).ok()?;
    // "160000 commit <sha>\truntime/vendor/llama.cpp"
    let mut fields = listing.split_whitespace();
    (fields.next() == Some("160000") && fields.next() == Some("commit"))
        .then(|| fields.next().map(str::to_owned))
        .flatten()
}

fn configure_llama_cpp(
    source_dir: &Path,
    build_dir: &Path,
    backends: &Backends,
    ggml_link: GgmlLink,
) {
    // Reconfigure the existing tree so changing a runtime source or rerunning
    // Cargo does not regenerate every Vulkan shader. CMake updates backend
    // options explicitly for each Cargo feature variant.
    fs::create_dir_all(build_dir).expect("create llama.cpp build dir");
    let metal_flag = if backends.metal {
        "-DGGML_METAL=ON"
    } else {
        "-DGGML_METAL=OFF"
    };
    let metal_embed = if backends.metal {
        "-DGGML_METAL_EMBED_LIBRARY=ON"
    } else {
        "-DGGML_METAL_EMBED_LIBRARY=OFF"
    };
    let vulkan_flag = if backends.vulkan {
        "-DGGML_VULKAN=ON"
    } else {
        "-DGGML_VULKAN=OFF"
    };
    let cuda_flag = if backends.cuda {
        "-DGGML_CUDA=ON"
    } else {
        "-DGGML_CUDA=OFF"
    };
    let native_flag = if native_build_enabled() {
        "-DGGML_NATIVE=ON"
    } else {
        "-DGGML_NATIVE=OFF"
    };
    // CUDA target architectures are independent of RETRO_NATIVE. `native` targets
    // GPUs present at build time; CI should pin an explicit list for reproducible
    // artifacts, for example RETRO_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real".
    let cuda_architectures = env::var("RETRO_CUDA_ARCHITECTURES")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| {
            // A release artifact built with `native` only runs on the GPU
            // generations present on the build machine; make that limitation
            // visible to packaging builds.
            if backends.cuda && env::var("PROFILE").as_deref() == Ok("release") {
                println!(
                    "cargo:warning=RETRO_CUDA_ARCHITECTURES is unset: this release build \
                     targets only the GPU architectures present on the build machine. Pin an \
                     explicit list (e.g. \"75-real;80-real;86-real;89-real\") for a \
                     distributable artifact."
                );
            }
            "native".to_owned()
        });
    // CUDA Graphs default off until graph capture is validated across batch shapes.
    let cuda_graphs = env::var("RETRO_CUDA_GRAPHS")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    let mut command = Command::new("cmake");
    command
        .arg("-S")
        .arg(source_dir)
        .arg("-B")
        .arg(build_dir)
        .arg(match ggml_link {
            GgmlLink::Static => "-DBUILD_SHARED_LIBS=OFF",
            GgmlLink::Shared => "-DBUILD_SHARED_LIBS=ON",
        })
        // The Python binding links these static archives into a cdylib
        // (.so), so every object must be position-independent. CMake does
        // not add -fPIC to static libraries by default, which otherwise
        // fails with R_X86_64_PC32 relocation errors at cdylib link time.
        .arg("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")
        .arg("-DLLAMA_BUILD_COMMON=OFF")
        .arg("-DLLAMA_BUILD_TESTS=OFF")
        .arg("-DLLAMA_BUILD_EXAMPLES=OFF")
        .arg("-DLLAMA_BUILD_TOOLS=OFF")
        .arg("-DLLAMA_BUILD_SERVER=OFF")
        .arg("-DLLAMA_BUILD_APP=OFF")
        .arg("-DLLAMA_CURL=OFF")
        .arg(metal_flag)
        .arg(metal_embed)
        .arg(vulkan_flag)
        .arg(cuda_flag)
        .arg("-DGGML_ACCELERATE=OFF")
        .arg("-DGGML_BLAS=OFF")
        .arg(native_flag)
        .arg("-DGGML_CCACHE=OFF");
    if ggml_link == GgmlLink::Shared {
        // Give every dylib an install name carrying its absolute directory,
        // rather than CMake's default `@rpath/libggml.dylib`.
        //
        // With `@rpath`, finding the library is the *consumer's* job, and Cargo
        // has no way to do it for the fifty test binaries in the crates above
        // this one: `cargo:rustc-link-arg` reaches only this package's own
        // targets, and rustc's `-C rpath` derives its entries from Rust crate
        // dylibs, not from the native `-L` paths a build script emits. An
        // absolute install name moves the job into the library, where it needs
        // no cooperation - a consumer records the path it linked against and
        // the loader follows it with no rpath, no `DYLD_LIBRARY_PATH` and no
        // copying. One directory covers all of them because llama.cpp sends
        // every shared library to `bin/`.
        //
        // Ignored on Linux, where CMake writes a `DT_SONAME` the loader always
        // resolves through the search path; that is why `shared` is not the
        // default there. See `GgmlLink::detect`.
        command
            .arg(format!(
                "-DCMAKE_INSTALL_NAME_DIR={}",
                shared_library_dir(build_dir).display()
            ))
            .arg("-DCMAKE_BUILD_WITH_INSTALL_NAME_DIR=ON");
    }

    // Diagnostic knob for the CUDA training-forward NaN (`docs/engineering/cuda/STATUS.md`):
    // compute-sanitizer traces an uninitialized __global__ read in the Q8_0
    // matmul of the training forward. Forcing cuBLAS suppresses the integer MMQ
    // path (one confirmed offender) but does not by itself make quantized
    // training finite, so it stays OFF by default (no silent perf regression)
    // until the compute-buffer initialization is fixed. Set
    // RETRO_CUDA_FORCE_CUBLAS=1 to take the cuBLAS path while investigating.
    let cuda_force_cublas = env::var("RETRO_CUDA_FORCE_CUBLAS")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false);
    if backends.cuda {
        command
            .arg(format!("-DCMAKE_CUDA_ARCHITECTURES={cuda_architectures}"))
            .arg(if cuda_force_cublas {
                "-DGGML_CUDA_FORCE_CUBLAS=ON"
            } else {
                "-DGGML_CUDA_FORCE_CUBLAS=OFF"
            })
            .arg(if cuda_graphs {
                "-DGGML_CUDA_GRAPHS=ON"
            } else {
                "-DGGML_CUDA_GRAPHS=OFF"
            })
            // Keep the supported CUDA scope single-GPU until peer-copy and
            // placement tests cover multi-GPU collectives.
            .arg("-DGGML_CUDA_NCCL=OFF");
        let host_compiler = cuda_host_compiler();
        if let Some(host_compiler) = &host_compiler {
            command.arg(format!("-DCMAKE_CUDA_HOST_COMPILER={host_compiler}"));
        }
        reset_stale_cuda_compiler_probe(build_dir, host_compiler.as_deref());
    }
    run(&mut command, "configure llama.cpp");
}

/// CMake caches its nvcc probe in `CMakeFiles/<ver>/CMakeCUDACompiler.cmake` and
/// reuses it across reconfigures, so a later `-DCMAKE_CUDA_HOST_COMPILER` never
/// reaches the `-ccbin` on the compile lines - and neither does a system
/// compiler upgrade under nvcc's default `cc`. Drop the probe when the host
/// compiler it recorded no longer matches the one we intend to use, which makes
/// the next configure redetect and regenerate the flags.
fn reset_stale_cuda_compiler_probe(build_dir: &Path, host_compiler: Option<&str>) {
    let expected_major = gnu_major(host_compiler.unwrap_or("g++"));
    let Ok(entries) = fs::read_dir(build_dir.join("CMakeFiles")) else {
        return;
    };
    for entry in entries.flatten() {
        let probe = entry.path().join("CMakeCUDACompiler.cmake");
        let Ok(contents) = fs::read_to_string(&probe) else {
            continue;
        };
        let recorded = |key: &str| -> Option<String> {
            let prefix = format!("set(CMAKE_CUDA_{key} \"");
            contents.lines().find_map(|line| {
                let value = line.trim().strip_prefix(&prefix)?;
                Some(value.strip_suffix("\")")?.to_owned())
            })
        };
        // Either signal going stale means the recorded compile lines are wrong:
        // a different `-ccbin` than the one we now pass (typically none at all),
        // or the same driver name upgraded underneath nvcc's default `cc`.
        let recorded_path = recorded("HOST_COMPILER").unwrap_or_default();
        let path_matches = match host_compiler {
            Some(expected) => {
                Path::new(&recorded_path).file_name() == Path::new(expected).file_name()
            }
            None => recorded_path.is_empty(),
        };
        let recorded_major = recorded("HOST_COMPILER_VERSION")
            .and_then(|version| version.split('.').next()?.parse::<u32>().ok());
        if !path_matches || (expected_major.is_some() && recorded_major != expected_major) {
            fs::remove_file(&probe)
                .unwrap_or_else(|err| panic!("failed to remove {}: {err}", probe.display()));
        }
    }
}

/// nvcc refuses any host compiler newer than the version its `crt/host_config.h`
/// whitelists (`#error -- unsupported GNU version!`), and distributions move the
/// default `g++` ahead of the installed toolkit. Return the newest `g++-N` nvcc
/// still accepts so the CUDA backend keeps building, or `None` to leave the
/// choice to CMake (default compiler already accepted, or no better candidate).
/// `RETRO_CUDA_HOST_COMPILER` overrides the probe; set it empty to disable it.
fn cuda_host_compiler() -> Option<String> {
    if let Ok(explicit) = env::var("RETRO_CUDA_HOST_COMPILER") {
        let explicit = explicit.trim().to_owned();
        return (!explicit.is_empty()).then_some(explicit);
    }
    let supported_major = nvcc_max_gnu_major()?;
    let default_major = gnu_major("g++")?;
    if default_major <= supported_major {
        return None;
    }
    // Walk down from the newest accepted version: `g++-N` may exist but report a
    // different major (Debian keeps the suffix in step, other distros need not).
    (5..=supported_major).rev().find_map(|major| {
        let candidate = format!("g++-{major}");
        (gnu_major(&candidate) == Some(major)).then_some(candidate)
    })
}

/// Highest GCC major version nvcc accepts, read from the guard in the toolkit's
/// `crt/host_config.h` (`#if __GNUC__ > 15`). `None` when the header cannot be
/// located or parsed, in which case the host compiler is left untouched.
fn nvcc_max_gnu_major() -> Option<u32> {
    let mut roots: Vec<PathBuf> = Vec::new();
    for key in ["CUDAToolkit_ROOT", "CUDA_PATH", "CUDA_HOME"] {
        if let Ok(root) = env::var(key) {
            roots.push(PathBuf::from(root));
        }
    }
    // `nvcc` on PATH (or CUDACXX) sits in <root>/bin.
    let nvcc = env::var("CUDACXX").unwrap_or_else(|_| "nvcc".to_owned());
    if let Some(path) = which(&nvcc)
        && let Some(root) = path.parent().and_then(Path::parent)
    {
        roots.push(root.to_path_buf());
    }
    roots.push(PathBuf::from("/usr/local/cuda"));

    let header = roots.iter().find_map(|root| {
        [
            root.join("include/crt/host_config.h"),
            root.join("targets/x86_64-linux/include/crt/host_config.h"),
        ]
        .into_iter()
        .find_map(|path| fs::read_to_string(path).ok())
    })?;
    header.lines().find_map(|line| {
        line.trim()
            .strip_prefix("#if __GNUC__ >")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}

/// Major version reported by a GCC-compatible driver, or `None` if it is absent.
fn gnu_major(compiler: &str) -> Option<u32> {
    let output = Command::new(compiler).arg("-dumpversion").output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .split('.')
        .next()?
        .parse()
        .ok()
}

fn which(program: &str) -> Option<PathBuf> {
    let path = PathBuf::from(program);
    if path.components().count() > 1 {
        return path.is_file().then_some(path);
    }
    env::split_paths(&env::var_os("PATH")?).find_map(|dir| {
        let candidate = dir.join(program);
        candidate.is_file().then_some(candidate)
    })
}

fn native_build_enabled() -> bool {
    env::var("RETRO_NATIVE")
        .map(|value| value != "0" && !value.eq_ignore_ascii_case("false"))
        .unwrap_or(false)
}

/// Where a shared llama.cpp build puts its libraries: one directory for all of
/// them, unlike the static build, which leaves each archive beside the CMake
/// target that produced it.
fn shared_library_dir(build_dir: &Path) -> PathBuf {
    build_dir.join("bin")
}

fn build_llama_cpp(build_dir: &Path, backends: &Backends) {
    let jobs = env::var("NUM_JOBS").unwrap_or_else(|_| "4".to_string());
    let mut command = Command::new("cmake");
    command
        .arg("--build")
        .arg(build_dir)
        .arg("--target")
        .arg("llama");
    if backends.metal {
        command.arg("ggml-metal");
    }
    if backends.vulkan {
        command.arg("ggml-vulkan");
    }
    if backends.cuda {
        command.arg("ggml-cuda");
    }
    command.arg("-j").arg(jobs);
    run(&mut command, "build llama.cpp");
}

/// Mirrors the Vulkan loader selected by CMake on Cargo's final link line.
/// A static `ggml-vulkan` does not propagate its private `Vulkan::Vulkan`
/// dependency, so Cargo has to name the loader itself. Mirrored in the shared
/// mode too, where a dylib named twice on a link line costs nothing.
fn emit_vulkan_loader_link(build_dir: &Path) {
    let cache_path = build_dir.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", cache_path.display()));
    let library = cache.lines().find_map(|line| {
        let (key_and_type, value) = line.split_once('=')?;
        let (key, _) = key_and_type.split_once(':')?;
        (key == "Vulkan_LIBRARY" || key == "Vulkan_LIBRARY_RELEASE").then(|| PathBuf::from(value))
    });
    let library = library.unwrap_or_else(|| {
        panic!(
            "CMake enabled Vulkan but did not record Vulkan_LIBRARY in {}",
            cache_path.display()
        )
    });
    if let Some(parent) = library.parent() {
        println!("cargo:rustc-link-search=native={}", parent.display());
    }
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let loader = if target_os == "windows" {
        "vulkan-1"
    } else {
        "vulkan"
    };
    println!("cargo:rustc-link-lib=dylib={loader}");
}

/// Mirrors the CUDA runtime libraries CMake linked privately into the static
/// `ggml-cuda` archive. Because Cargo links that archive itself, CMake's private
/// `CUDA::cudart` / `CUDA::cublas` / `CUDA::cuda_driver` dependencies are not
/// propagated and their symbols would be undefined at the final link. CMake's
/// `CMakeCache.txt` is used as the source of truth for the toolkit library
/// directory rather than guessing an SDK layout.
fn emit_cuda_runtime_link(build_dir: &Path) {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        panic!("CUDA linking was requested on macOS, which has no CUDA toolkit");
    }
    let cache_path = build_dir.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", cache_path.display()));
    let cache_value = |wanted: &str| -> Option<String> {
        cache.lines().find_map(|line| {
            let (key_and_type, value) = line.split_once('=')?;
            let (key, _) = key_and_type.split_once(':')?;
            (key == wanted && !value.is_empty()).then(|| value.to_owned())
        })
    };

    // Candidate library directories, in order of preference. FindCUDAToolkit
    // records the directory directly on modern CMake; otherwise it is derived
    // from the toolkit root or the nvcc bin directory it did record.
    let mut lib_dirs: Vec<PathBuf> = Vec::new();
    let mut push_dir = |dir: PathBuf| {
        if dir.is_dir() && !lib_dirs.contains(&dir) {
            lib_dirs.push(dir);
        }
    };
    if let Some(dir) = cache_value("CUDAToolkit_LIBRARY_DIR") {
        push_dir(PathBuf::from(dir));
    }
    for root_key in [
        "CUDAToolkit_ROOT_DIR",
        "CUDAToolkit_TARGET_DIR",
        "CUDA_TOOLKIT_ROOT_DIR",
    ] {
        if let Some(root) = cache_value(root_key) {
            let root = PathBuf::from(root);
            push_dir(root.join("lib64"));
            push_dir(root.join("lib/x64")); // Windows layout
            push_dir(root.join("lib"));
        }
    }
    if let Some(bin) = cache_value("CUDAToolkit_BIN_DIR")
        && let Some(root) = PathBuf::from(bin).parent()
    {
        push_dir(root.join("lib64"));
        push_dir(root.join("lib"));
    }
    if lib_dirs.is_empty() {
        panic!(
            "CMake enabled CUDA but CMakeCache.txt in {} recorded no toolkit library \
             directory; cannot relay cudart/cublas to the final link line",
            cache_path.display()
        );
    }
    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
        // libcuda.so (the driver) ships as a linker stub under lib64/stubs when the
        // driver itself is not on the default search path (e.g. build containers).
        let stubs = dir.join("stubs");
        if stubs.is_dir() {
            println!("cargo:rustc-link-search=native={}", stubs.display());
        }
    }

    // Dynamic CUDA runtime (GGML_STATIC is not set): ggml-cuda's objects reference
    // cudart, cuBLAS, cuBLASLt and, since VMM is left enabled, the CUDA driver.
    for lib in ["cudart", "cublas", "cublasLt", "cuda"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    if target_os != "windows" {
        for lib in ["dl", "rt", "pthread"] {
            println!("cargo:rustc-link-lib=dylib={lib}");
        }
    }
}

fn run(command: &mut Command, label: &str) {
    let output = command.output().unwrap_or_else(|err| {
        panic!("failed to {label}: {err}");
    });
    if !output.status.success() {
        panic!(
            "failed to {label}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

fn command_stdout(command: &mut Command, label: &str) -> String {
    let output = command.output().unwrap_or_else(|err| {
        panic!("failed to {label}: {err}");
    });
    if !output.status.success() {
        panic!(
            "failed to {label}\nstatus: {}\nstdout:\n{}\nstderr:\n{}",
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout)
        .unwrap_or_else(|err| panic!("failed to decode stdout for {label}: {err}"))
}
