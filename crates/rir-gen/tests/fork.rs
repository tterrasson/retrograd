//! The fork side of generation: the copies under
//! `crates/retrograd-ffi/runtime/` match the generated artifacts, adding a
//! kernel or a variant reaches no vendored file, the declared ggml ops exist,
//! and the caps the fork compiles with hold for the widest pair. These are the
//! tests whose subject **is** the text.

use std::path::Path;

use rir_gen::*;

/// The llama.cpp fork carries committed copies of RIR artefacts (the
/// Vulkan `.comp`, the generated MSL file `ggml-metal.metal` includes, the
/// AOT registry). The fork must stay buildable standalone, so the copies are
/// real files there - and this test is what keeps them from drifting:
/// byte-for-byte identity against the generator output.
#[test]
fn the_fork_copies_match_the_generated_artefacts() {
    let fork = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../retrograd-ffi/runtime/vendor/llama.cpp/ggml/src");
    if !fork.exists() {
        eprintln!("llama.cpp submodule absent - test skipped");
        return;
    }
    let kernels = generate_all().unwrap();
    let file = |kernel: &str, rel: &str| -> String {
        kernels
            .iter()
            .find(|k| k.name == kernel)
            .and_then(|k| k.files.iter().find(|(r, _)| r == rel))
            .map(|(_, c)| c.clone())
            .unwrap_or_else(|| panic!("{kernel}/{rel} absent from generation"))
    };

    // Which artefacts the fork must carry is derived from the schedule and
    // integration tables, not listed here: promoting a second op - or a
    // second *variant* of one op - already means editing those tables, and
    // a copy this test forgets to check is a copy that can drift silently.
    let artifacts = production_artifacts().unwrap();
    let production = |backend: rir_emit::GgmlBackend| -> Vec<&ProductionArtifact> {
        artifacts.iter().filter(|a| a.backend == backend).collect()
    };

    // Vulkan: full-file copy under vulkan-shaders/rir_<artifact>.comp, one
    // per variant - the fallback keeps the bare kernel name.
    let vulkan = production(rir_emit::GgmlBackend::Vulkan);
    assert!(!vulkan.is_empty(), "no production Vulkan kernel to check");
    for a in &vulkan {
        let rel = format!("ggml-vulkan/vulkan-shaders/rir_{}.comp", a.artifact);
        let comp = std::fs::read_to_string(fork.join(&rel))
            .unwrap_or_else(|e| panic!("{rel} copy absent from fork: {e}"));
        assert_eq!(
            comp,
            file(&a.kernel, &a.source_file),
            "{rel} drifted - copy generated/rir/{}/{}",
            a.kernel,
            a.source_file
        );
    }

    // Metal: one generated file, copied whole among the fork's Metal
    // translation units - the same full-file copy as Vulkan's and the
    // registry's, so this check is an equality rather than a substring
    // search.
    assert!(
        !production(rir_emit::GgmlBackend::Metal).is_empty(),
        "no production Metal kernel to check"
    );
    let aggregate = std::fs::read_to_string(fork.join(METAL_FORK_DIR).join(METAL_AGGREGATE))
        .unwrap_or_else(|e| panic!("{METAL_FORK_DIR}/{METAL_AGGREGATE} absent from fork: {e}"));
    assert_eq!(
        aggregate,
        file(METAL_DIR, METAL_AGGREGATE),
        "{METAL_AGGREGATE} drifted - copy generated/rir/{METAL_DIR}/{METAL_AGGREGATE}"
    );
    // And the fork must actually build it: upstream compiles one translation
    // unit per file listed in ggml-metal's CMakeLists, so a line dropped by a
    // rebase would leave the copy above green and the metallib without a
    // single RIR kernel.
    let cmake = std::fs::read_to_string(fork.join("ggml-metal/CMakeLists.txt")).unwrap();
    assert!(
        cmake.contains(&format!("kernels/{METAL_AGGREGATE}")),
        "ggml-metal/CMakeLists.txt no longer builds kernels/{METAL_AGGREGATE}"
    );

    // Vulkan: the artifact list the shader generator expands.
    // Same kind of full-file copy as the Metal
    // aggregate, and for the same reason - a list nobody checks is a third
    // place to keep by hand, which is what this phase exists to remove.
    let list_rel = format!("ggml-vulkan/vulkan-shaders/{VULKAN_ARTIFACT_LIST}");
    let list = std::fs::read_to_string(fork.join(&list_rel))
        .unwrap_or_else(|e| panic!("{VULKAN_ARTIFACT_LIST} copy absent from fork: {e}"));
    assert_eq!(
        list,
        file(VULKAN_DIR, VULKAN_ARTIFACT_LIST),
        "{VULKAN_ARTIFACT_LIST} drifted - copy generated/rir/{VULKAN_DIR}/{VULKAN_ARTIFACT_LIST}"
    );

    // CUDA: one translation unit per production artifact under
    // `ggml-cuda/rir/rir_<artifact>.cu`, plus the launcher list the single
    // hand-written unit expands.
    //
    // A CUDA artifact requires a policy above `native_only`. Derive the copied
    // units from that policy so new admitted pairs are covered automatically.
    for a in production(rir_emit::GgmlBackend::Cuda) {
        let rel = format!("ggml-cuda/rir/rir_{}.cu", a.artifact);
        let cu = std::fs::read_to_string(fork.join(&rel))
            .unwrap_or_else(|e| panic!("{rel} copy absent from fork: {e}"));
        assert_eq!(
            cu,
            file(&a.kernel, &a.source_file),
            "{rel} drifted - copy generated/rir/{}/{}",
            a.kernel,
            a.source_file
        );
    }
    let launchers_rel = format!("ggml-cuda/rir/{CUDA_LAUNCHER_LIST}");
    let launchers = std::fs::read_to_string(fork.join(&launchers_rel))
        .unwrap_or_else(|e| panic!("{CUDA_LAUNCHER_LIST} copy absent from fork: {e}"));
    assert_eq!(
        launchers,
        file(CUDA_DIR, CUDA_LAUNCHER_LIST),
        "{CUDA_LAUNCHER_LIST} drifted - copy generated/rir/{CUDA_DIR}/{CUDA_LAUNCHER_LIST}"
    );

    // The quantized-format header. It sits directly
    // under ggml/src/ because ggml, ggml-metal.metal and the Vulkan shader
    // generator all include it by that path.
    let quant_h = std::fs::read_to_string(fork.join("ggml-retro-quant.h"))
        .expect("ggml-retro-quant.h copy absent from fork");
    assert_eq!(
        quant_h,
        file(QUANT_DIR, "ggml-retro-quant.h"),
        "ggml-retro-quant.h drifted - copy generated/rir/quant/ggml-retro-quant.h"
    );

    // Registry: full-file copies under ggml-rir/.
    for rel in ["rir_registry.h", "rir_registry.cpp", "rir_kernel_params.h"] {
        let committed = std::fs::read_to_string(fork.join("ggml-rir").join(rel))
            .unwrap_or_else(|e| panic!("{rel} copy absent from fork: {e}"));
        assert_eq!(
            committed,
            file(REGISTRY_DIR, rel),
            "{rel} drifted - copy generated/rir/registry/{rel}"
        );
    }
}

/// Adding a RIR kernel must stay invisible to the two vendored units that
/// have no reason to see it: `ggml-vulkan.cpp` and the vendored Metal sources
/// are kept independent of generated RIR code.
///
/// The property is stated as an absence, which is exactly why it needs a
/// test: nothing fails when a rebase - or a hurried integration - puts one
/// `#include` of a generated header back. The build simply becomes slow
/// again, and the rebase diff large again, silently.
#[test]
fn adding_a_kernel_does_not_reach_the_vendored_units() {
    let fork = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../retrograd-ffi/runtime/vendor/llama.cpp/ggml/src");
    if !fork.exists() {
        eprintln!("llama.cpp submodule absent - test skipped");
        return;
    }
    let vk = std::fs::read_to_string(fork.join("ggml-vulkan/ggml-vulkan.cpp")).unwrap();
    // The two generated headers a new kernel edits. The RIR shader header
    // belongs to `ggml-vulkan-rir.cpp` alone; the params header to
    // `ggml-rir.cpp` alone.
    let includes: Vec<&str> = vk
        .lines()
        .map(str::trim_start)
        .filter(|l| l.starts_with("#include"))
        .collect();
    for header in ["rir_kernel_params.h", "ggml-vulkan-rir-shaders.hpp"] {
        assert!(
            !includes.iter().any(|l| l.contains(header)),
            "ggml-vulkan.cpp includes {header}: adding a RIR kernel recompiles 20,000 lines"
        );
    }
    // And no per-artifact symbol either - that is what the map keyed on
    // `rir_variant_desc.artifact` replaced.
    for a in production_artifacts().unwrap() {
        if a.backend != rir_emit::GgmlBackend::Vulkan {
            continue;
        }
        assert!(
            !vk.contains(&format!("rir_{}_data", a.artifact)),
            "ggml-vulkan.cpp names blob rir_{}_data: the per-artifact table is bypassed",
            a.artifact
        );
    }
}

/// The integration rule, stated where it can
/// fail: **adding a variant touches no vendored file**.
///
/// Metal holds it with one `#include`, Vulkan through the generated list: one
/// hand-written `string_to_spv("rir_…")` per variant would grow the patch
/// with every generated kernel. The property is again stated as an
/// absence, and again that is why it needs a test: putting one line back
/// would work, and nothing else would object.
#[test]
fn adding_a_variant_touches_no_vendored_file() {
    let fork = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../retrograd-ffi/runtime/vendor/llama.cpp/ggml/src");
    if !fork.exists() {
        eprintln!("llama.cpp submodule absent - test skipped");
        return;
    }
    let gen_cpp = fork.join("ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp");
    let text = std::fs::read_to_string(&gen_cpp).unwrap();
    assert!(
        text.contains(&format!("#include \"{VULKAN_ARTIFACT_LIST}\"")),
        "vulkan-shaders-gen.cpp no longer includes {VULKAN_ARTIFACT_LIST}: \
         no RIR shader would be compiled"
    );
    assert!(
        text.contains("RIR_VK_ARTIFACTS("),
        "vulkan-shaders-gen.cpp no longer expands the generated list"
    );
    // The expansion itself writes `string_to_spv("rir_" #name, …)`, so what
    // must not appear is a *name* after the prefix - a literal spelled out
    // instead of stringized.
    for line in text.lines() {
        let hand_written = line
            .split("string_to_spv(\"rir_")
            .skip(1)
            .any(|rest| !rest.starts_with('"'));
        assert!(
            !hand_written,
            "vulkan-shaders-gen.cpp names a RIR artifact by hand: the generated \
             list is bypassed and the vendored patch grows again by one line per \
             variant\n{line}"
        );
    }
    // CUDA's half, which is the same property with a sharper edge:
    // a `__global__` is reachable only through its
    // generated stub, so the one unit that names them must name them **through
    // the generated list** and never one by one. A hand-written
    // `rir_launch_add_vec4` here would compile, work, and put the vendored
    // patch back on a growth of one line per variant.
    let launch = std::fs::read_to_string(fork.join("ggml-cuda/rir/rir-cuda-launch.cu")).unwrap();
    assert!(
        launch.contains(&format!("#include \"{CUDA_LAUNCHER_LIST}\"")),
        "rir-cuda-launch.cu no longer includes {CUDA_LAUNCHER_LIST}: no RIR \
         kernel would be launchable"
    );
    assert!(
        launch.contains("RIR_CUDA_LAUNCHERS("),
        "rir-cuda-launch.cu no longer expands the generated list"
    );
    for line in launch.lines() {
        // Code only: the prose above the table legitimately mentions
        // `ggml_cuda_rir_launch_fn`, and a substring search that read comments
        // would be testing the comments.
        let line = line.split("//").next().unwrap_or("");
        // The expansions write `rir_launch_##name`; what must not appear is a
        // name after the prefix, spelled out instead of pasted. The prefix is
        // matched as a whole token, so a longer identifier ending in it is not
        // a hit.
        let hand_written = line
            .match_indices("rir_launch_")
            .filter(|(at, _)| {
                line[..*at]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !c.is_alphanumeric() && c != '_')
            })
            .any(|(at, m)| !line[at + m.len()..].starts_with("##"));
        assert!(
            !hand_written,
            "rir-cuda-launch.cu names a RIR stub by hand: the generated list is \
             bypassed\n{line}"
        );
    }

    // Metal's half of the same property: the
    // aggregate is the one unit that defines RIR entrypoints, and every
    // sibling under kernels/ is vendored source a variant must not reach.
    // Upstream split ggml-metal.metal into that directory, so the check
    // reads the complete directory rather than a single source file.
    for entry in std::fs::read_dir(fork.join(METAL_FORK_DIR)).unwrap() {
        let path = entry.unwrap().path();
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        if name == METAL_AGGREGATE || path.extension().is_none_or(|e| e != "metal") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(
            !text.contains("kernel void rir_"),
            "{name} defines a RIR entrypoint: the generated aggregate is \
             bypassed and a copy has returned to vendored source"
        );
    }
}

/// A spec's `ggml_op` must exist in the vendored ggml fork. This is the
/// test form of the rule: no production kernel may carry an implicit
/// or nonexistent op.
#[test]
fn declared_ggml_ops_exist_in_the_fork() {
    let header = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../retrograd-ffi/runtime/vendor/llama.cpp/ggml/include/ggml.h");
    let Ok(text) = std::fs::read_to_string(&header) else {
        eprintln!("llama.cpp submodule absent - test skipped");
        return;
    };
    for spec in rir_kernels::integrations() {
        if let Some(op) = spec.ggml_op {
            assert!(
                text.contains(&format!("{op},")),
                "{}: {op} absent from ggml.h",
                spec.kernel
            );
        }
    }
}

/// The two caps of the fork are constants, so they do not grow by one per
/// variant the way a list does - they jump, and they have jumped three
/// times (8 → 16 → 24 → 32). Both behaved correctly each time: a
/// `GGML_ABORT` on the first node, never a silent degradation.
///
/// A device abort is still the most expensive place to learn it. The same
/// fact is derivable here - the widest (op, backend) pair of the table - so
/// the generator says it before a shader is compiled.
#[test]
fn the_forks_variant_caps_hold_for_the_widest_pair() {
    use std::collections::BTreeMap;
    let specs = rir_kernels::integrations();
    let mut per_pair: BTreeMap<(&str, String), usize> = BTreeMap::new();
    for a in production_artifacts().unwrap() {
        let spec = specs
            .iter()
            .find(|s| s.kernel == a.kernel)
            .unwrap_or_else(|| panic!("{}: no spec", a.kernel));
        let Some(op) = spec.ggml_op else { continue };
        *per_pair
            .entry((op, format!("{:?}", a.backend)))
            .or_default() += 1;
    }
    let ((op, backend), widest) = per_pair
        .iter()
        .max_by_key(|(_, n)| **n)
        .map(|((op, b), n)| ((*op, b.clone()), *n))
        .expect("no production variant");

    let header = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../retrograd-ffi/runtime/vendor/llama.cpp/ggml/src/ggml-rir");
    if !header.exists() {
        eprintln!("llama.cpp submodule absent - test skipped");
        return;
    }
    let read_cap = |file: &str, needle: &str| -> usize {
        let text = std::fs::read_to_string(header.join(file)).unwrap();
        let line = text
            .lines()
            .find(|l| l.contains(needle))
            .unwrap_or_else(|| panic!("{needle} absent from {file}"));
        line.split_whitespace()
            .last()
            .unwrap()
            .trim_end_matches(';')
            .parse()
            .unwrap_or_else(|_| panic!("unreadable value for {needle}: {line:?}"))
    };
    for (file, needle) in [
        ("ggml-rir.h", "#define GGML_RIR_MAX_SITE_VARIANTS"),
        ("ggml-rir.cpp", "constexpr uint32_t RIR_MAX_PAIR_VARIANTS"),
    ] {
        let cap = read_cap(file, needle);
        assert!(
            cap >= widest,
            "{needle} is {cap} in the fork, while {op}/{backend} publishes \
             {widest}: the first node would abort"
        );
    }
}

/// `DomainRestriction`'s discriminants are the `ggml_rir_reject` enum, and
/// that enum's order is ABI: the mask this side builds is compared against
/// counters the other side fills in. A reordering
/// there would keep both sides compiling and silently excuse the wrong
/// restriction - the one failure mode the whole mechanism exists to remove.
///
/// So the two are checked against each other rather than trusted to agree:
/// each restriction's name must sit at its own value in the C enum.
#[test]
fn the_domain_restrictions_are_the_forks_reject_taxonomy() {
    let header = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../retrograd-ffi/runtime/vendor/llama.cpp/ggml/src/ggml-rir/ggml-rir.h");
    let Ok(text) = std::fs::read_to_string(&header) else {
        eprintln!("llama.cpp submodule absent - test skipped");
        return;
    };
    use rir_emit::DomainRestriction::*;
    for r in [
        DType,
        Rank,
        Shape,
        Stride,
        QuantBlock,
        IntegerRange,
        OpVariant,
    ] {
        // `GGML_RIR_REJECT_DTYPE         = 2,` - the spelling is the name
        // uppercased, which is also what `ggml_rir_reject_name` prints
        // lowercased, so the two spellings cannot drift apart either.
        let needle = format!("GGML_RIR_REJECT_{}", r.name().to_uppercase());
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(&needle))
            .unwrap_or_else(|| panic!("{needle} absent from ggml-rir.h"));
        let value: u32 = line
            .split('=')
            .nth(1)
            .and_then(|v| v.trim().trim_end_matches(',').parse().ok())
            .unwrap_or_else(|| panic!("unreadable value for {needle}: {line:?}"));
        assert_eq!(
            r.bit(),
            1u32 << value,
            "{needle} is {value} in the fork, {r:?} publishes bit {:#x}",
            r.bit()
        );
    }
}
