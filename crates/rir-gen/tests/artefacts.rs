//! The emitted artifacts: regeneration is a no-op, generation is
//! deterministic, the registry and the manifests agree, the shaders compile,
//! and each scan variant is the artifact its schedule describes.

#![allow(clippy::unwrap_used)]
// `allow-unwrap-in-tests` covers `#[test]` bodies, not the closures this file
// spawns into a thread scope. Same reasoning, said where the configuration
// cannot reach.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use rir_lower::Backend;

use rir_gen::*;

/// Runs one independent compiler invocation per job, across the machine's
/// cores, and reports **every** failure rather than the first.
///
/// The three toolchain checks below spawn one compiler *process* per generated
/// source - eighty-one for GLSL, as many for MSL - and a process spawn is
/// wall-clock time the machine has cores to absorb. Run in sequence they were
/// the two slowest tests of the fast lane by an order of magnitude while every
/// other core sat idle. The jobs are parallel by construction and not by luck:
/// each one writes its own file, named after the kernel, under a directory
/// already unique to this process.
///
/// Reporting all the failures rather than the first is not a bonus of the
/// rewrite, it is what keeps the parallel version usable: a scheduling order
/// nobody chose must not decide *which* broken shader the message names.
fn compile_jobs<J: Sync>(jobs: &[J], run: impl Fn(&J) -> Result<(), String> + Sync) {
    if jobs.is_empty() {
        return;
    }
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .min(jobs.len());
    let next = AtomicUsize::new(0);
    let failures = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                while let Some(job) = jobs.get(next.fetch_add(1, Ordering::Relaxed)) {
                    if let Err(why) = run(job) {
                        failures.lock().unwrap().push(why);
                    }
                }
            });
        }
    });
    let mut failures = failures.into_inner().unwrap();
    failures.sort();
    assert!(
        failures.is_empty(),
        "{} of {} sources did not compile:\n{}",
        failures.len(),
        jobs.len(),
        failures.join("\n")
    );
}

/// Local equivalent of the CI rule: regeneration produces no diff from
/// committed files, including both content and the exact file set.
#[test]
fn regenerating_produces_no_diff() {
    let root = committed_dir();
    let kernels = generate_all().expect("generation");

    // The root itself, not only each kernel directory: a directory left
    // behind by a kernel removed from the registry is exactly the kind of
    // stale generated code this test exists to catch.
    let expected_dirs: BTreeSet<String> = kernels.iter().map(|k| k.name.clone()).collect();
    let on_disk_dirs: BTreeSet<String> = std::fs::read_dir(&root)
        .unwrap_or_else(|e| panic!("{} unreadable: {e}", root.display()))
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(
        on_disk_dirs,
        expected_dirs,
        "extra or missing directories under {} - run `cargo run -p rir-gen`",
        root.display()
    );

    for gk in kernels {
        let kdir = root.join(&gk.name);
        let mut expected = BTreeSet::new();
        for (rel, content) in &gk.files {
            expected.insert(rel.clone());
            let path = kdir.join(rel);
            let committed = std::fs::read_to_string(&path).unwrap_or_else(|e| {
                panic!(
                    "missing generated file ({e}): {} - run `cargo run -p rir-gen`",
                    path.display()
                )
            });
            assert_eq!(
                &committed,
                content,
                "diff in {} - run `cargo run -p rir-gen` and inspect the diff",
                path.display()
            );
        }
        let on_disk: BTreeSet<String> = std::fs::read_dir(&kdir)
            .unwrap_or_else(|e| panic!("{} unreadable: {e}", kdir.display()))
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            on_disk,
            expected,
            "extra or missing files under {}",
            kdir.display()
        );
    }
}

/// Generation itself is deterministic across two independent passes.
#[test]
fn generation_is_deterministic() {
    let a = generate_all().unwrap();
    let b = generate_all().unwrap();
    assert_eq!(a.len(), b.len());
    for (ka, kb) in a.iter().zip(b.iter()) {
        assert_eq!(ka.name, kb.name);
        for ((ra, ca), (rb, cb)) in ka.files.iter().zip(kb.files.iter()) {
            assert_eq!(ra, rb);
            assert_eq!(ca, cb);
        }
    }
}

/// The C++ registry and the declared tables are two views of one datum: every
/// production (kernel, backend, variant) has its registry row, with the
/// entrypoint and the op it declares, and the registry has no row without one.
///
/// **The expected side is typed.** The expected values come from `registry()`
/// rather than from parsing generated JSON, so this test compares the C++
/// output with the typed source data.
/// The pairs below come from `registry()` - the same values the emitters are
/// given - so the only text still searched is the C++ table, which is the object
/// of the test.
#[test]
fn the_registry_matches_the_production_manifests() {
    let kernels = generate_all().unwrap();
    let registry_files = kernels
        .iter()
        .find(|k| k.name == REGISTRY_DIR)
        .expect("pseudo-kernel registry absent");
    let cpp = &registry_files
        .files
        .iter()
        .find(|(r, _)| r == "rir_registry.cpp")
        .unwrap()
        .1;

    // One row per production GPU lowering, derived the way the generator derives
    // it: the schedules of the registration, minus the backends with no emitter,
    // minus the pairs the spec keeps native.
    let mut expected: Vec<(String, String, String)> = Vec::new();
    for entry in rir_kernels::registry() {
        for schedule in entry.schedules {
            // Every GPU backend, CUDA included: what
            // filters a lowering out of the registry is the spec's policy, and
            // naming a backend here would make that filter say two things.
            if schedule.backend() == Backend::Cpu {
                continue;
            }
            let lk = rir_lower::lower(&entry.kernel, schedule).unwrap();
            let backend = rir_emit::manifest::ggml_backend(&lk);
            if !entry.integration.production_on(backend) {
                continue;
            }
            expected.push((
                entry.kernel.name().to_string(),
                rir_emit::entrypoint(&lk),
                entry.integration.ggml_op.unwrap().to_string(),
            ));
        }
    }
    assert!(
        !expected.is_empty(),
        "no production lowering at all: the derivation is wrong, not the table"
    );

    for (kernel, entry, op) in &expected {
        assert!(
            cpp.contains(&format!("\"{entry}\"")),
            "{kernel}: entrypoint {entry} absent from the registry"
        );
        assert!(
            cpp.contains(&format!("\"{op}\"")),
            "{kernel}: op {op} absent from the registry"
        );
    }

    let count: usize = cpp
        .split_once("rir_variant_count = ")
        .and_then(|(_, r)| r.split_once(';'))
        .and_then(|(n, _)| n.trim().parse().ok())
        .unwrap();
    assert_eq!(
        count,
        expected.len(),
        "registry count != production lowerings of the registry"
    );

    // The manifests and registry must describe the same set.
    let manifests = kernels
        .iter()
        .flat_map(|gk| {
            gk.files
                .iter()
                .filter(|(rel, content)| {
                    rel.starts_with("manifest.") && content.contains("\"production\": true")
                })
                .map(move |_| gk.name.clone())
        })
        .count();
    assert_eq!(
        manifests,
        expected.len(),
        "production manifests != production lowerings"
    );
}

/// If `glslangValidator` or `glslc` is available, every generated shader
/// must compile. Otherwise the test reports the skip and passes; validation
/// becomes effective on machines with the Vulkan SDK, including GPU lanes.
#[test]
fn the_generated_shaders_compile_if_a_glsl_compiler_is_available() {
    // Subgroup collectives require SPIR-V 1.3, hence a Vulkan 1.1 target.
    // Neither invocation may default its output: `glslangValidator` writes
    // `comp.spv` in the *current* directory when no `-o` is given - the
    // working directory of the test, i.e. the crate root, where two parallel
    // runs would also race on the same file. Both compilers take an explicit
    // `-o` under the temporary directory below.
    let compilers: [(&str, &[&str]); 2] = [
        (
            "glslangValidator",
            &["-S", "comp", "--target-env", "vulkan1.1"],
        ),
        ("glslc", &["-fshader-stage=comp", "--target-env=vulkan1.1"]),
    ];
    let Some((cmd, args)) = compilers.iter().find(|(c, _)| {
        std::process::Command::new(c)
            .arg("--version")
            .output()
            .is_ok()
    }) else {
        eprintln!("no GLSL compiler - shader validation skipped");
        return;
    };
    // Per process: two lanes running this test at once must not write each
    // other's sources or SPIR-V.
    let tmp = std::env::temp_dir().join(format!("rir-glsl-check-{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let jobs: Vec<(String, String)> = generate_all()
        .unwrap()
        .into_iter()
        .flat_map(|gk| {
            gk.files
                .into_iter()
                .filter(|(rel, _)| rel.ends_with(".comp"))
                .map(move |(rel, content)| (format!("{}-{rel}", gk.name), content))
        })
        .collect();
    compile_jobs(&jobs, |(name, content)| {
        let path = tmp.join(name);
        std::fs::write(&path, content).unwrap();
        let out = std::process::Command::new(cmd)
            .args(*args)
            .arg("-o")
            .arg(path.with_extension("spv"))
            .arg(&path)
            .output()
            .unwrap_or_else(|e| panic!("executing {cmd}: {e}"));
        if out.status.success() {
            return Ok(());
        }
        Err(format!(
            "invalid GLSL for {name} ({cmd}):\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    });
}

/// On macOS with the full Xcode toolchain, compile every generated Metal
/// source to AIR. Command Line Tools alone do not ship `metal`, so absence
/// remains an explicit skip just like the optional GLSL compiler above.
#[test]
fn the_generated_shaders_compile_if_a_metal_compiler_is_available() {
    // `xcrun -f metal` may find an Xcode shim even when the separately
    // downloadable Metal Toolchain component is absent. Execute the shim
    // far enough to prove that the compiler itself is usable.
    let available = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "--version"])
        .output()
        .is_ok_and(|out| out.status.success());
    if !available {
        eprintln!("no Metal compiler - shader validation skipped");
        return;
    }

    let tmp = std::env::temp_dir().join("rir-metal-check");
    std::fs::create_dir_all(&tmp).unwrap();
    // The pairing of a source with the language version *its own* manifest
    // claims is resolved here, before anything is spawned: it reads the whole
    // file list of each kernel so every emitted source is covered.
    let mut jobs: Vec<(String, String, String)> = Vec::new();
    for gk in generate_all().unwrap() {
        for (rel, content) in &gk.files {
            // `kernel<.variant>.metal` only: the aggregate the fork includes
            // is a fragment with neither header nor namespace, so compiling
            // it alone would fail for a reason that says nothing about the
            // kernels it carries.
            if !rel.starts_with("kernel") || !rel.ends_with(".metal") {
                continue;
            }
            // Compile at the language version the manifest *claims*, not at a
            // version hardcoded here: that is what makes this test check the
            // published `metal>=x.y` feature rather than merely the syntax.
            // `kernel<.variant>.metal` pairs with `manifest<.variant>.metal.json`:
            // one variant's MSL must be compiled at the version *its own*
            // manifest claims, not at another variant's.
            let want = format!(
                "manifest{}.metal.json",
                rel.trim_start_matches("kernel").trim_end_matches(".metal")
            );
            let manifest = gk
                .files
                .iter()
                .find(|(r, _)| *r == want)
                .map(|(_, c)| c.as_str())
                .unwrap_or_else(|| panic!("{}: {rel} without {want}", gk.name));
            let version = manifest
                .split_once("\"metal>=")
                .and_then(|(_, rest)| rest.split_once('"'))
                .map(|(v, _)| v.to_string())
                .unwrap_or_else(|| panic!("{}: manifest without metal>= feature", gk.name));
            jobs.push((format!("{}-{rel}", gk.name), version, content.clone()));
        }
    }
    compile_jobs(&jobs, |(name, version, content)| {
        let source = tmp.join(name);
        let air = source.with_extension("air");
        std::fs::write(&source, content).unwrap();
        let out = std::process::Command::new("xcrun")
            // The driver spells the macOS dialect `macos-metal<x.y>`;
            // a bare `metal<x.y>` only exists from Metal 3.0 on.
            .args([
                "-sdk",
                "macosx",
                "metal",
                &format!("-std=macos-metal{version}"),
                "-c",
            ])
            .arg(&source)
            .arg("-o")
            .arg(&air)
            .output()
            .unwrap_or_else(|e| panic!("executing xcrun metal: {e}"));
        if out.status.success() {
            return Ok(());
        }
        Err(format!(
            "invalid MSL for {name}:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    });
}

/// If a CUDA toolkit is available, every generated `.cu` must compile - with
/// the flags the fork compiles `ggml-cuda` with, `-use_fast_math` included.
///
/// That last point is a decision made into a test: a
/// promoted kernel must be compiled exactly as it is shipped, or the lane's
/// measurement would be about a binary that does not exist. Compiling the
/// generated sources under stricter flags here would hide the substitution of
/// `__expf` and `__fsqrt_rn` that the fork actually performs.
///
/// The layout under the temporary directory is the fork's, not a convenience:
/// a generated unit includes `"../../ggml-rir/rir_kernel_params.h"`, which is
/// where the header sits relative to `ggml/src/ggml-cuda/rir/`. Compiling it
/// anywhere else would be compiling a different file.
#[test]
fn the_generated_kernels_compile_if_a_cuda_toolkit_is_available() {
    let nvcc = std::env::var_os("CUDACXX")
        .filter(|compiler| !compiler.is_empty())
        .unwrap_or_else(|| "nvcc".into());
    if std::process::Command::new(&nvcc)
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("no CUDA toolkit - .cu validation skipped");
        return;
    }
    // Per process: two lanes running this test at once must not write each
    // other's sources.
    let root = std::env::temp_dir().join(format!("rir-cuda-check-{}", std::process::id()));
    let rir = root.join("ggml-cuda/rir");
    let params = root.join("ggml-rir");
    std::fs::create_dir_all(&rir).unwrap();
    std::fs::create_dir_all(&params).unwrap();

    let kernels = generate_all().unwrap();
    let header = kernels
        .iter()
        .find(|k| k.name == REGISTRY_DIR)
        .and_then(|k| k.files.iter().find(|(r, _)| r == "rir_kernel_params.h"))
        .map(|(_, c)| c.clone())
        .expect("rir_kernel_params.h absent from generation");
    std::fs::write(params.join("rir_kernel_params.h"), header).unwrap();

    // The flags the fork uses (`ggml-cuda/CMakeLists.txt`), plus whatever this
    // machine needs to make `nvcc` accept its host compiler. The second part is
    // probed rather than assumed: a toolkit refusing the system `gcc` is a
    // property of the machine, and hardcoding `-ccbin g++-15` here would make
    // this test pass on one box and skip on every other.
    let mut flags: Vec<String> = vec!["-use_fast_math".into()];
    if let Some(arch) = std::env::var_os("RIR_CUDA_ARCH") {
        flags.push(format!("-arch={}", arch.to_string_lossy()));
    }
    let probe = root.join("probe.cu");
    std::fs::write(&probe, "__global__ void rir_probe() {}\n").unwrap();
    let host: Vec<String> = std::env::var("RIR_NVCC_CCBIN")
        .ok()
        .into_iter()
        .map(|c| vec![c])
        .chain([vec![], vec!["g++-15".into()], vec!["g++-14".into()]])
        .find_map(|ccbin| {
            let mut cmd = std::process::Command::new(&nvcc);
            cmd.args(&flags);
            for c in &ccbin {
                cmd.args(["-ccbin", c]);
            }
            let ok = cmd
                .arg("-c")
                .arg(&probe)
                .arg("-o")
                .arg(root.join("probe.o"))
                .output()
                .is_ok_and(|o| o.status.success());
            ok.then(|| {
                ccbin
                    .iter()
                    .flat_map(|c| ["-ccbin".to_string(), c.clone()])
                    .collect()
            })
        })
        .unwrap_or_else(|| {
            panic!(
                "nvcc is present but compiles nothing: no usable host compiler \
                 (set RIR_NVCC_CCBIN)"
            )
        });

    let jobs: Vec<(String, String)> = kernels
        .iter()
        .flat_map(|gk| {
            gk.files
                .iter()
                .filter(|(rel, _)| rel.ends_with(".cu"))
                .map(|(rel, content)| {
                    // The name the fork gives it, so a diagnostic names the
                    // artifact rather than a temporary file: `rir_<artifact>.cu`.
                    let stem = rel.trim_start_matches("kernel").trim_end_matches(".cu");
                    (
                        format!("rir_{}{}.cu", gk.name, stem.replace('.', "_")),
                        content.clone(),
                    )
                })
        })
        .collect();
    assert!(
        !jobs.is_empty(),
        "no generated .cu at all: the CUDA half of the schedule table is empty"
    );
    compile_jobs(&jobs, |(name, content)| {
        let source = rir.join(name);
        std::fs::write(&source, content).unwrap();
        let out = std::process::Command::new(&nvcc)
            .args(&flags)
            .args(&host)
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(source.with_extension("o"))
            .output()
            .unwrap_or_else(|e| panic!("executing nvcc: {e}"));
        if out.status.success() {
            return Ok(());
        }
        Err(format!(
            "invalid CUDA for {name}:\n{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ))
    });
    let _ = std::fs::remove_dir_all(&root);
}

/// The blocked scan, now that it is a table entry rather than a compiler
/// capability: both GPU emitters print it, its manifest publishes the
/// semantics the strategy assumes, and - the property the AOT pipeline
/// gained - its artifacts are **distinct** from the fallback's.
///
/// The last assertion is the one that matters here. Emitting two variants
/// under one set of names is what kept the blocked scan out of production:
/// the second overwrote the first, so `schedules_for` could only publish
/// one.
#[test]
fn the_blocked_scan_is_a_distinct_artifact_from_the_fallback() {
    let kernel = rir_kernels::cumsum::build().unwrap();
    let vk = rir_lower::lower(&kernel, rir_lower::Schedule::vulkan_blocked_scan()).unwrap();
    let mtl = rir_lower::lower(&kernel, rir_lower::Schedule::metal_blocked_scan()).unwrap();
    assert_eq!(
        format!("{:?}", vk.body),
        format!("{:?}", mtl.body),
        "the two blocked-scan lowerings diverge"
    );

    let glsl = rir_emit::emit_vulkan(&vk).expect("blocked-scan GLSL");
    assert!(
        glsl.contains("subgroupExclusiveAdd"),
        "collective prefix missing from GLSL"
    );
    let msl = rir_emit::emit_metal(&mtl).expect("blocked-scan MSL");
    assert!(
        msl.contains("simd_prefix_exclusive_sum"),
        "collective prefix missing from MSL"
    );

    // The manifest must publish semantics assumed by the strategy; otherwise
    // a caller would believe exact order was preserved.
    assert_eq!(
        vk.reduction_semantics,
        vec![rir_core::ReductionSemantics::Deterministic],
        "blocked scan must declare itself Deterministic"
    );

    // No emitted name may collide with fallback names: files, variant
    // identity, or entrypoint. This lets both coexist in `generated/` and
    // in the fork.
    let base_vk = rir_lower::lower(&kernel, rir_lower::Schedule::vulkan_grid([64, 1, 1])).unwrap();
    let base_mtl = rir_lower::lower(&kernel, rir_lower::Schedule::metal_grid([64, 1, 1])).unwrap();
    for (base, variant) in [(&base_vk, &vk), (&base_mtl, &mtl)] {
        assert_ne!(
            rir_emit::artifact_name(base),
            rir_emit::artifact_name(variant)
        );
        assert_ne!(
            rir_emit::artifact_files(base),
            rir_emit::artifact_files(variant)
        );
        assert_ne!(rir_emit::variant_id(base), rir_emit::variant_id(variant));
    }
    // On Metal both entrypoints share one ggml source, so a collision would
    // become a compilation error there.
    assert_ne!(rir_emit::entrypoint(&base_mtl), rir_emit::entrypoint(&mtl));
}

/// `SharedTree` is a workgroup collective, not a wider subgroup in
/// disguise: both emitters must allocate shared storage, synchronize the
/// full workgroup, and avoid publishing a subgroup requirement.
#[test]
fn the_shared_scan_is_workgroup_wide_and_subgroup_independent() {
    let kernel = rir_kernels::cumsum::build().unwrap();
    let vk = rir_lower::lower(&kernel, rir_lower::schedule::bench::vulkan_shared_scan()).unwrap();
    let mtl = rir_lower::lower(&kernel, rir_lower::schedule::bench::metal_shared_scan()).unwrap();
    assert_eq!(
        format!("{:?}", vk.body),
        format!("{:?}", mtl.body),
        "the two SharedTree lowerings diverge"
    );
    assert_eq!(vk.schedule.block(), [256, 1, 1]);
    assert!(!vk.uses_subgroup());
    assert!(vk.uses_shared());

    let glsl = rir_emit::emit_vulkan(&vk).expect("GLSL SharedTree");
    assert!(glsl.contains("shared float rir_shared_"));
    assert!(glsl.contains("barrier();"));
    assert!(!glsl.contains("subgroupExclusiveAdd"));

    let msl = rir_emit::emit_metal(&mtl).expect("MSL SharedTree");
    assert!(msl.contains("threadgroup float rir_shared_"));
    assert!(msl.contains("threadgroup_barrier(mem_flags::mem_threadgroup)"));
    assert!(msl.contains("thread_index_in_threadgroup"));
    assert!(!msl.contains("thread_index_in_simdgroup"));

    // The published strategy name is ABI - a consumer reads that spelling out of
    // the JSON - so it stays a substring. What the shader *needs* is not:
    // "declares no subgroup capability" was a `!contains("subgroup_arithmetic")`
    // that a renamed feature would have satisfied by accident. It is a property
    // of the nest, and the nest can be asked.
    for lk in [&vk, &mtl] {
        assert!(
            rir_emit::emit_manifest(lk, None).contains("\"reduction\": \"shared_tree\""),
            "the published reduction name is part of the manifest ABI"
        );
        assert!(
            !lk.uses_subgroup(),
            "the shared scan must not require subgroup arithmetic"
        );
        assert!(
            lk.uses_shared(),
            "the shared scan works through shared memory"
        );
    }
}

/// What the tiled scan claims over the blocked one is an **access plan**,
/// and an access plan is not visible in a result: a version that staged its
/// tile lane-contiguously would compute the same prefixes, pass every
/// parity test, and buy nothing. So the two
/// properties that *are* the strategy are asserted here.
///
/// They are asserted on the **loop nest**, where lowering materializes the
/// plan before either backend prints it. The two multiplications that carry the
/// plan are compared as values rather than as emitted-text substrings.
#[test]
fn the_tiled_scan_stages_interleaved_and_pads_its_runs() {
    use rir_lower::{Inst, LExpr, Stmt};

    let kernel = rir_kernels::cumsum::build().unwrap();
    let (lanes, items) = (256u32, 16u32);
    let vk = rir_lower::lower(
        &kernel,
        rir_lower::Schedule::vulkan_tiled_scan(lanes, items),
    )
    .unwrap();
    let mtl =
        rir_lower::lower(&kernel, rir_lower::Schedule::metal_tiled_scan(lanes, items)).unwrap();
    assert_eq!(
        format!("{:?}", vk.body),
        format!("{:?}", mtl.body),
        "the two tiled lowerings diverge"
    );

    // Every constant multiplier the nest computes on the index bank.
    fn factors(stmts: &[Stmt], out: &mut Vec<u32>) {
        for s in stmts {
            if let Stmt::Compute(Inst {
                expr: LExpr::IMulC(_, c),
                ..
            }) = s
            {
                out.push(*c);
            }
            for child in rir_lower::child_blocks_of(s) {
                factors(child, out);
            }
        }
    }
    let mut muls = Vec::new();
    factors(&vk.body, &mut muls);
    // Interleaving: a lane's slot is `i · lanes + lane`, so adjacent lanes
    // read adjacent addresses. `lane · items` would be the blocked scan's
    // plan under another name.
    assert!(
        muls.contains(&lanes),
        "tiling does not stage interleaved data: {muls:?}"
    );
    // Skew: one lane's run has a stride of `items + 1`, which is what keeps
    // the middle phase off a single memory bank.
    assert!(
        muls.contains(&(items + 1)),
        "runs are no longer skewed: {muls:?}"
    );
    // No negative assertion on `items` itself: the tree's own level
    // constants are the powers of two up to `lanes`, and one of them is
    // `items`. What separates the skewed plan from the flat one is that the
    // *run* stride is `items + 1`, asserted above.
    // And the storage that plan needs, declared by the kernel rather than
    // derived from a statement.
    assert!(
        vk.shared
            .iter()
            .any(|(_, len)| *len == lanes * items + lanes),
        "shared storage is no longer sized by the skewed tile: {:?}",
        vk.shared
    );

    // The two backends still *print* it, and printing it is now all they
    // do: no emitter function expands a scan.
    for src in [
        rir_emit::emit_vulkan(&vk).expect("tiled GLSL"),
        rir_emit::emit_metal(&mtl).expect("tiled MSL"),
    ] {
        assert!(
            src.contains(&format!("[{}]", lanes * items + lanes)),
            "{src}"
        );
    }
}

/// The property this buys, stated where it can fail:
/// **every barrier a scan shader prints is a `Stmt::Barrier` of its loop
/// nest**.
///
/// Two witnesses that do not look at the same object cannot disagree
/// usefully: an oracle executing `WorkgroupScan` as one operation while the
/// backends print a series of levels would count 0 barriers on one side and
/// 18 on the other. The barriers are in the IR, `interp` runs them, and this
/// counts them on both sides.
///
/// The kernel is the tiled scan because it is the one that carries both
/// collectives; a kernel with a `WorkgroupReduce` or a `StageTiles` would
/// legitimately print more, those two statements being the ones kept
/// as fixed expansions (a group of accumulators under one series of
/// barriers, a cooperative staging round).
#[test]
fn every_barrier_printed_by_a_scan_is_a_barrier_in_its_loop_nest() {
    use rir_lower::Stmt;

    let kernel = rir_kernels::cumsum::build().unwrap();
    fn barriers(stmts: &[Stmt]) -> usize {
        stmts
            .iter()
            .map(|s| {
                usize::from(matches!(s, Stmt::Barrier))
                    + rir_lower::child_blocks_of(s)
                        .into_iter()
                        .map(|b| barriers(b))
                        .sum::<usize>()
            })
            .sum()
    }
    for (schedule, backend) in [
        (rir_lower::Schedule::vulkan_tiled_scan(256, 16), "vulkan"),
        (rir_lower::Schedule::metal_tiled_scan(256, 16), "metal"),
    ] {
        let lk = rir_lower::lower(&kernel, schedule).unwrap();
        let declared = barriers(&lk.body);
        assert!(declared > 0, "{backend}: a tiled scan with no barrier");
        let (src, needle) = match backend {
            "vulkan" => (rir_emit::emit_vulkan(&lk).unwrap(), "barrier();"),
            _ => (
                rir_emit::emit_metal(&lk).unwrap(),
                "threadgroup_barrier(mem_flags::mem_threadgroup);",
            ),
        };
        let printed = src.matches(needle).count();
        assert_eq!(
            printed, declared,
            "{backend}: {printed} barriers printed for {declared} in the nest - an emitter \
             is expanding a collective again"
        );
    }
}

/// The Metal schedules mirror their Vulkan twins field for field, and
/// `lower` never reads `schedule.backend`. Asserting the two loop nests
/// are identical is what carries the oracle's Vulkan parity over to Metal:
/// without it, a divergence in the schedule table would only surface on
/// a device nothing in this workspace drives.
#[test]
fn a_metal_schedule_lowers_like_its_vulkan_twin() {
    for entry in rir_kernels::registry() {
        let (kernel, schedules) = (&entry.kernel, &entry.schedules);
        // Paired **by variant**, not "the first Vulkan schedule against the
        // first Metal one": with several variants per pair that would only
        // ever compare the fallbacks, and a Metal specialization could
        // diverge from its Vulkan twin unnoticed.
        let variants: Vec<Option<&'static str>> = {
            let mut v: Vec<Option<&'static str>> = schedules
                .iter()
                .filter(|s| s.backend() != Backend::Cpu)
                .map(|s| s.variant())
                .collect();
            v.dedup();
            v
        };
        for want in variants {
            let pick = |b: Backend| {
                schedules
                    .iter()
                    .find(|s| s.backend() == b && s.variant() == want)
                    .cloned()
            };
            let name = want.unwrap_or("fallback");
            match (pick(Backend::Vulkan), pick(Backend::Metal)) {
                (Some(vk), Some(mtl)) => {
                    assert_eq!(vk.block(), mtl.block(), "{}/{name}: block", kernel.name());
                    assert_eq!(
                        vk.priority(),
                        mtl.priority(),
                        "{}/{name}: priority",
                        kernel.name()
                    );
                    assert_eq!(
                        vk.eligible_when(),
                        mtl.eligible_when(),
                        "{}/{name}: divergent shape rules - the same shape would go to \
                         different variants depending on backend",
                        kernel.name()
                    );
                    let lvk = rir_lower::lower(kernel, vk).unwrap();
                    let lmtl = rir_lower::lower(kernel, mtl).unwrap();
                    assert_eq!(
                        format!("{:?}", lvk.body),
                        format!("{:?}", lmtl.body),
                        "{}/{name}: the two GPU lowerings diverge",
                        kernel.name()
                    );
                }
                (None, None) => {}
                _ => panic!(
                    "{}/{name}: only one of the two GPU backends is scheduled",
                    kernel.name()
                ),
            }
        }
    }
}

/// Every generated manifest parses back into the schema that wrote it, and
/// re-serializes byte for byte.
///
/// This guard checks the shared schema: a field added to
/// `rir_core::manifest::Manifest` without regenerating, a reader that renames
/// one. Both show up here as a text that differs from what the file carries.
#[test]
fn every_manifest_round_trips_through_the_schema() {
    let mut checked = 0;
    for gk in generate_all().unwrap() {
        for (rel, content) in &gk.files {
            if !rel.starts_with("manifest") || !rel.ends_with(".json") {
                continue;
            }
            let parsed: rir_core::manifest::Manifest =
                serde_json::from_str(content).unwrap_or_else(|e| panic!("{}/{rel}: {e}", gk.name));
            let mut again =
                serde_json::to_string_pretty(&parsed).expect("a manifest is plain data");
            again.push('\n');
            assert_eq!(&again, content, "{}/{rel}: emit → parse → emit", gk.name);
            checked += 1;
        }
    }
    // A silent zero would make the assertion above vacuous.
    assert!(checked > 100, "only {checked} manifests seen");
}

/// The committed catalogue parses back into the schema that wrote it,
/// re-serializes byte for byte, and says the same thing typed as it did as
/// text.
///
/// The same shape as the manifest's guard above, with the catalogue's own
/// subject. What it
/// watches is the seam the move created: the generator writes
/// `rir_core::catalog`, the planner reads it, and nothing else is left to keep
/// equal by hand - so a field renamed on either side has to surface as a file
/// that does not round-trip. The typed assertions also verify that backend and
/// policy values are not re-spelled into parallel representations.
#[test]
fn the_catalogue_round_trips_through_the_schema() {
    use rir_core::catalog::{BackendPolicy, GgmlBackend, KernelCatalog};

    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../generated/rir/catalog/catalog.json"
    );
    let content = std::fs::read_to_string(path).expect("the committed catalogue");
    let parsed: KernelCatalog = serde_json::from_str(&content).expect("catalog.json");
    let mut again = serde_json::to_string_pretty(&parsed).expect("a catalogue is plain data");
    again.push('\n');
    assert_eq!(again, content, "emit → parse → emit");

    assert_eq!(parsed.schema_version, rir_core::catalog::SCHEMA_VERSION);
    assert!(!parsed.fingerprint.is_empty());
    // Every backend has a row for every production op, so all four appear; and
    // at least one row is a promotion, or the catalogue would be describing a
    // build that dispatches nothing.
    for backend in [
        GgmlBackend::Cpu,
        GgmlBackend::Cuda,
        GgmlBackend::Metal,
        GgmlBackend::Vulkan,
    ] {
        assert!(
            parsed.kernels.iter().any(|row| row.backend == backend),
            "no row for {}",
            backend.name()
        );
    }
    assert!(
        parsed
            .kernels
            .iter()
            .any(|row| row.policy.dispatches() && row.backend != GgmlBackend::Cpu),
        "no row prefers a generated kernel"
    );
    assert!(
        parsed
            .kernels
            .iter()
            .filter(|row| row.backend == GgmlBackend::Cpu)
            .all(|row| row.policy == BackendPolicy::NativeOnly),
        "the CPU backend runs a generated kernel"
    );
}
