//! The declared tables: registry validation, GPU coverage, the README's
//! canonical family table, and the variant tables' shape rules.

use std::path::Path;

use rir_lower::Backend;

use rir_gen::*;

/// Table rules - one spec per kernel, no orphan spec, matching arity and
/// parameter order, production with explicit op, native CUDA and CPU - are
/// enforced by `validate_registry`, which `generate_all` calls before first
/// emission.
#[test]
fn the_declared_tables_pass_registry_validation() {
    validate_registry().expect("registry tables");
}

/// The family table in `docs/engineering/rir/KERNELS.md` is the canonical
/// description of what the compiler generates. It is compared with the
/// registry rather than merely reviewed so the documentation cannot drift.
///
/// Format follows the file: between the two
/// `<!-- rir-gen:families:{start,end} -->` markers, one table row per family.
/// The first cell lists names in backticks - an exact name or a
/// `prefix_<…>` pattern naming all kernels with that prefix - and the second
/// gives the declared cardinality.
#[test]
fn the_readme_kernel_table_matches_the_registry() {
    let readme =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../docs/engineering/rir/KERNELS.md");
    let text = std::fs::read_to_string(&readme).expect("docs/engineering/rir/KERNELS.md");
    let table = text
        .split_once("<!-- rir-gen:families:start -->")
        .and_then(|(_, r)| r.split_once("<!-- rir-gen:families:end -->"))
        .map(|(t, _)| t)
        .expect("family table markers absent");

    let names: Vec<String> = rir_kernels::all()
        .iter()
        .map(|k| k.name().to_string())
        .collect();
    let mut covered: Vec<&str> = Vec::new();
    let mut rows = 0usize;
    for line in table.lines() {
        let line = line.trim();
        // The header and separator are not families.
        if !line.starts_with('|') || line.contains("---") || line.contains("| Family |") {
            continue;
        }
        let cells: Vec<&str> = line.trim_matches('|').split('|').collect();
        let declared: usize = cells[1]
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("unreadable count: {line}"));
        let patterns: Vec<&str> = cells[0]
            .split('`')
            .skip(1)
            .step_by(2)
            .map(str::trim)
            .collect();
        assert!(!patterns.is_empty(), "unnamed family: {line}");
        let mut matched = 0usize;
        for pattern in patterns {
            let hits: Vec<&str> = match pattern.split_once('<') {
                Some((prefix, _)) => names
                    .iter()
                    .filter(|n| n.starts_with(prefix))
                    .map(String::as_str)
                    .collect(),
                None => names
                    .iter()
                    .filter(|n| *n == pattern)
                    .map(String::as_str)
                    .collect(),
            };
            assert!(!hits.is_empty(), "{pattern}: no kernel with this name");
            for hit in hits {
                assert!(
                    !covered.contains(&hit),
                    "{hit}: counted by two README families"
                );
                covered.push(hit);
                matched += 1;
            }
        }
        assert_eq!(
            matched,
            declared,
            "{}: README declares {declared} members, registry has {matched}",
            cells[0].trim()
        );
        rows += 1;
    }
    assert!(rows > 0, "empty family table");
    let mut missing: Vec<&str> = names
        .iter()
        .map(String::as_str)
        .filter(|n| !covered.contains(n))
        .collect();
    missing.sort_unstable();
    assert!(
        missing.is_empty(),
        "kernels absent from the README table: {missing:?}"
    );
}

/// And the counterpart: what the phase **rejects**. Without it, validation that
/// validated nothing would also pass.
#[test]
fn registry_validation_rejects_a_malformed_table() {
    let kernels = rir_kernels::all();
    let specs = rir_kernels::integrations();

    // An orphan spec.
    let mut orphan = specs.clone();
    orphan.push(rir_emit::IntegrationSpec {
        kernel: "kernel_qui_n_existe_pas".to_string(),
        ..specs[0].clone()
    });
    let e = validate_tables(&kernels, &orphan).expect_err("orphan spec accepted");
    assert!(matches!(e, GenError::Registry(_)), "{e}");

    // A kernel without a spec must not pass as an implicit oracle.
    let dropped: Vec<_> = specs
        .iter()
        .filter(|s| s.kernel != kernels[0].name())
        .cloned()
        .collect();
    let e = validate_tables(&kernels, &dropped).expect_err("kernel without a spec accepted");
    assert!(matches!(e, GenError::Registry(_)), "{e}");

    // A production variant promoted on CUDA.
    let promoted: Vec<_> = specs
        .iter()
        .cloned()
        .map(|mut s| {
            if s.production {
                s.backend_policy = s
                    .backend_policy
                    .into_iter()
                    .map(|(b, p)| match b {
                        rir_emit::GgmlBackend::Cuda => {
                            (b, rir_emit::BackendPolicy::PreferGenerated)
                        }
                        _ => (b, p),
                    })
                    .collect();
            }
            s
        })
        .collect();
    let e = validate_tables(&kernels, &promoted).expect_err("promoted CUDA accepted");
    assert!(matches!(e, GenError::Registry(_)), "{e}");
}

/// The rejection this guards against: a production
/// policy on a backend the schedule table never reaches.
///
/// The test builds a **registration** by hand so it can exercise the reachable
/// half of the rule:
///
/// a family arm that schedules the CPU alone while a spec
/// promotes the pair on a GPU backend. The registry row would name an
/// same defect: a family arm that schedules the CPU alone while a spec
/// promotes the pair on a GPU backend. The registry row would name an
/// artifact the table never emits - a policy of `prefer_generated` with
/// nothing to prefer, which no other test fails because
/// `check_schedule_table` sees a well-formed one-entry CPU table.
#[test]
fn a_production_policy_without_a_schedule_is_a_generation_error() {
    use rir_core::{Extent, KernelBuilder, TensorType};

    let mut kb = KernelBuilder::new("add");
    let x = kb.input("x", TensorType::f32_2d());
    let y = kb.output("y", TensorType::f32_2d());
    let row = kb.axis("row", Extent::Dim { arg: x, dim: 1 });
    let col = kb.axis("col", Extent::Dim { arg: x, dim: 0 });
    let v = kb.read(x, &[col, row]);
    kb.write(y, &[col, row], v);
    let kernel = kb.finish().expect("test kernel");

    let spec = rir_emit::IntegrationSpec {
        kernel: "add".to_string(),
        ggml_op: Some("GGML_OP_ADD"),
        ggml_op_variant: None,
        args: vec![rir_emit::ArgSource::Src(0), rir_emit::ArgSource::Dst],
        params: vec![],
        production: true,
        backend_policy: vec![
            (
                rir_emit::GgmlBackend::Vulkan,
                rir_emit::BackendPolicy::PreferGenerated,
            ),
            (
                rir_emit::GgmlBackend::Metal,
                rir_emit::BackendPolicy::PreferGenerated,
            ),
        ],
        assumed_domain: vec![],
        retired_native: vec![],
        native_exception: None,
    };
    // The state a future family arm could produce: promoted on both GPU
    // backends, scheduled on neither.
    let cpu_only = rir_kernels::KernelRegistration {
        id: rir_kernels::KernelId::Cumsum,
        kernel: kernel.clone(),
        family: rir_lower::Family::Elementwise,
        schedules: vec![rir_lower::Schedule::cpu_serial()],
        integration: spec.clone(),
    };
    let e = check_gpu_coverage(std::slice::from_ref(&cpu_only))
        .expect_err("a production policy with no schedule was accepted");
    let GenError::Registry(r) = &e else {
        panic!("{e}");
    };
    assert!(
        r.detail.contains("no schedule for that backend"),
        "{}",
        r.detail
    );

    // And the same kernel declared oracle-only is accepted: the check is
    // about the *pairing*, not about being absent from the table.
    let oracle = rir_kernels::KernelRegistration {
        integration: rir_emit::IntegrationSpec {
            ggml_op: None,
            production: false,
            backend_policy: vec![],
            ..spec
        },
        ..cpu_only
    };
    assert!(check_gpu_coverage(&[oracle]).is_ok());
}

/// Half of the same rejection, in the cheap direction: a schedule
/// on a backend a refusal names.
///
/// The refusal is checked before emission, alongside the family refusal, so
/// generation cannot partially write an invalid table.
///
/// Both refusals are exercised, and the family one
/// is the live half: `GPU_REFUSED` is empty - the entry it carried was CUDA's,
/// for want of an emitter - while `FAMILY_REFUSED` holds the five families
/// CUDA does not serve. The table-wide arm is therefore written to hold when
/// the list is empty rather than to index into it, because "no backend is
/// refused outright" is the state this list is *meant* to reach.
#[test]
fn a_schedule_on_a_refused_backend_is_a_generation_error() {
    if let Some((refused, _)) = rir_lower::GPU_REFUSED.first() {
        let mut entry = rir_kernels::registry().remove(0);
        entry
            .schedules
            .push(rir_lower::Schedule::gpu_subgroup(*refused));
        let e = check_gpu_coverage(&[entry]).expect_err("a refused backend was scheduled");
        let GenError::Registry(r) = &e else {
            panic!("{e}");
        };
        assert!(r.detail.contains("which the table refuses"), "{}", r.detail);
    }

    // The family's refusal, which is a refusal and not a preference: an arm
    // that named the backend anyway would reach an emitter the family's
    // lowering is not ready for - on CUDA today, a collective the v1 emitter
    // states it does not print.
    let (family, gpu, _) = rir_lower::FAMILY_REFUSED
        .first()
        .expect("no family refusal to check");
    let mut entry = rir_kernels::registry()
        .into_iter()
        .find(|e| e.family == *family)
        .unwrap_or_else(|| panic!("no kernel of the {} family", family.name()));
    entry
        .schedules
        .push(rir_lower::Schedule::gpu_subgroup(*gpu));
    let e = check_gpu_coverage(&[entry]).expect_err("a family-refused backend was scheduled");
    let GenError::Registry(r) = &e else {
        panic!("{e}");
    };
    assert!(
        r.detail
            .contains(&format!("which the {} family refuses", family.name())),
        "{}",
        r.detail
    );
}

/// The quantized kernels stay oracle-only. `GGML_OP_SUM_ROWS` keeps its
/// input dtype, so a kernel that sums a quantized tensor into F32 has no
/// exact op - and a kernel with no op must never claim production, or the
/// registry would carry a variant no dispatcher can select.
///
/// This is also where CUDA's guarantee holds for phase E: no quantized
/// variant is production anywhere, so `cuda_policy_is_locked_to_native`
/// above has nothing to relax. The CUDA loaders keep decoding, and their
/// parity is checked against the same bytes by `tests/rir_quant_oracle.rs`
/// through ggml's CPU decoder.
#[test]
fn no_quantized_kernel_claims_production() {
    for format in rir_kernels::sum_rows_quant::variants() {
        let name = rir_kernels::sum_rows_quant::kernel_name(format);
        let spec = rir_kernels::integration_for(&name).unwrap_or_else(|| panic!("{name}: no spec"));
        assert!(!spec.production, "{name}: production without ggml_op");
        assert_eq!(spec.ggml_op, None, "{name}: implicit op");
    }
}

/// The variant table of every kernel is well-formed: exactly one fallback
/// per (kernel, backend), distinct identities, every specialization
/// selectable. `generate_all` enforces this, and this test states it as a
/// property of the shipped table rather than of one generation run.
#[test]
fn every_schedule_table_is_a_well_formed_variant_table() {
    for entry in rir_kernels::registry() {
        rir_lower::check_schedule_table(&entry.kernel, &entry.schedules)
            .unwrap_or_else(|e| panic!("{}: {e}", entry.kernel.name()));
    }
}

/// The shape rule that arbitrates the two `cumsum` lowerings, pinned to
/// the measurement it came from.
///
/// Neither threshold is a taste. The row bound comes from the level-1
/// crossover sweep (`rir-runtime/tests/device_timing.rs`), which has the
/// blocked scan ahead at every row length up to 128 rows and behind from 256
/// on. The tiled scan's `col ≥ 4096` is one tile, and the lane is what put
/// it there: at `col = 1024` it ties the blocked scan, three quarters of its
/// lanes being past the end of the row.
///
/// A silent edit of either number would send a whole shape class to a
/// lowering measured slower there, and nothing else in the pipeline would
/// object - the registry would still be well-formed.
#[test]
fn the_cumsum_variants_split_the_shape_space_where_the_bench_says() {
    let schedules = rir_lower::schedules_for(rir_lower::Family::Cumsum);
    for backend in [Backend::Vulkan, Backend::Metal] {
        let of = |v: Option<&str>| {
            schedules
                .iter()
                .find(|s| s.backend() == backend && s.variant() == v)
                .unwrap_or_else(|| panic!("{backend:?}: variant {v:?} absent"))
        };
        let fallback = of(None);
        let blocked = of(Some(rir_lower::BLOCKED_SCAN));
        let tiled = of(Some(rir_lower::TILED_SCAN));

        assert!(
            fallback.eligible_when().is_empty(),
            "{backend:?}: fallback must accept every shape"
        );
        assert_eq!(
            blocked.eligible_when(),
            vec![rir_lower::ShapeRule {
                axes: &["row", "plane", "batch"],
                min: 1,
                max: 128
            }],
            "{backend:?}: blocked-scan rule no longer describes what the benchmark measured"
        );
        assert!(
            blocked.priority() > fallback.priority(),
            "{backend:?}: blocked scan would never be selected"
        );
        assert_eq!(
            tiled.eligible_when(),
            vec![
                rir_lower::ShapeRule {
                    axes: &["row", "plane", "batch"],
                    min: 1,
                    max: 128,
                },
                rir_lower::ShapeRule {
                    axes: &["col"],
                    min: 4096,
                    max: u32::MAX,
                },
            ],
            "{backend:?}: tiled scan no longer covers very long low-occupancy shapes"
        );
        assert!(
            tiled.priority() > blocked.priority(),
            "{backend:?}: tiled scan would lose the intersection against the 32-lane scan"
        );
    }
}

/// The shape rule that arbitrates the two `rms_norm_back` lowerings, pinned
/// to the lane measurement it came from.
///
/// The two bounds are not symmetric in what they protect against, and both
/// matter. The row ceiling is the measurement itself: the lane has the
/// 32-lane fallback at 0.45 on 256 rows, 0.82 on 64 and 1.97 on 16, so
/// raising it sends shapes the fallback already wins to a lowering with
/// barriers - which is exactly what a ceiling of 64 did to `[256,4,16,1]`
/// (1.09 on Metal, the shape that kept the pair in `observe`).
/// The column floor protects the other direction - under 256 columns a
/// 256-lane workgroup has lanes owning nothing, so the barriers would be
/// paid for no work at all.
#[test]
fn the_rms_norm_back_variants_split_the_shape_space_where_the_lane_says() {
    let kernel = rir_kernels::rms_norm_back::build().unwrap();
    let schedules = rir_lower::schedules_for(rir_lower::Family::RmsNorm);
    for backend in [Backend::Vulkan, Backend::Metal] {
        let of = |v: Option<&str>| {
            schedules
                .iter()
                .find(|s| s.backend() == backend && s.variant() == v)
                .unwrap_or_else(|| panic!("{backend:?}: variant {v:?} absent"))
        };
        let fallback = of(None);
        let shared = of(Some(rir_lower::SHARED_REDUCE));

        assert!(
            fallback.eligible_when().is_empty(),
            "{backend:?}: fallback must accept every shape"
        );
        assert_eq!(fallback.block(), [32, 1, 1]);
        assert_eq!(shared.block(), [256, 1, 1]);
        assert_eq!(
            shared.eligible_when(),
            vec![
                rir_lower::ShapeRule {
                    axes: &["row", "plane", "batch"],
                    min: 1,
                    max: 32,
                },
                rir_lower::ShapeRule {
                    axes: &["col"],
                    min: 256,
                    max: u32::MAX,
                },
            ],
            "{backend:?}: rule no longer describes what the lane measured"
        );
        assert!(
            shared.priority() > fallback.priority(),
            "{backend:?}: shared variant would never be selected"
        );

        // The collective is the whole point of the variant: a `SharedTree`
        // that lowered to a subgroup reduction would be the fallback with a
        // wider block, and 224 of its 256 lanes would go unreduced.
        let lk = rir_lower::lower(&kernel, shared.clone()).unwrap();
        assert!(lk.uses_shared(), "{backend:?}: no shared collective");
        assert!(!lk.uses_subgroup(), "{backend:?}: subgroup collective");
    }
}

/// The promotion rule, as a property of the
/// shipped table rather than of one generation run: **no promoted pair is
/// retirable, unretired and silent**.
///
/// The three legitimate states are a published restriction, a retired
/// native, and a chosen native exception with its reason. The fourth,
/// nothing left to close, native still in the vendored file, no line saying
/// why - is the forbidden one. `emit_registry` refuses it too; stating it here is what makes the
/// rule readable without reading the emitter.
#[test]
fn every_promoted_pair_says_why_its_native_is_still_there() {
    use rir_emit::{BackendPolicy, GgmlBackend};
    let specs = rir_kernels::integrations();
    let mut seen: Vec<&str> = Vec::new();
    for spec in &specs {
        let Some(op) = spec.ggml_op else { continue };
        if !spec.production || seen.contains(&op) {
            continue;
        }
        seen.push(op);
        let serving: Vec<&rir_emit::IntegrationSpec> = specs
            .iter()
            .filter(|s| s.production && s.ggml_op == Some(op))
            .collect();
        let domain = serving
            .iter()
            .fold(u32::MAX, |m, s| m & s.assumed_domain_mask());
        for backend in [GgmlBackend::Vulkan, GgmlBackend::Metal] {
            if spec.policy_for(backend) != BackendPolicy::PreferGenerated {
                continue;
            }
            let retired = serving.iter().any(|s| s.native_retired_on(backend));
            let excepted = serving.iter().any(|s| s.native_exception.is_some());
            assert!(
                domain != 0 || retired || excepted,
                "{op}/{}: promoted, removable, not retired, without a line to \
                 explain it",
                backend.name()
            );
            assert!(
                !(retired && excepted),
                "{op}/{}: native kernel retired and native exception declared",
                backend.name()
            );
        }
    }
}

/// A kernel with no ggml consumer still emits the whole artifact set, and that
/// is a decision rather than an omission.
///
/// The twelve quantized `sum_rows` variants, `l2_norm_fwd` and its derived
/// gradient have `ggml_op: None`: nothing dispatches them, and they carry CPU,
/// Vulkan and Metal lowerings with a manifest for each of the two GPU ones,
/// the CPU lowering has none, because nothing reads it. Dropping the GPU half
/// would save about a third of `generated/rir/`, and it would remove the only
/// thing that makes them worth keeping: **they are the independent witness
/// that a newly added quantized format decodes on a device**.
/// `sum_rows_<format>` is the smallest kernel that reads a format end to end,
/// so its Vulkan artifact is what the device lane runs when a format is added,
/// the CPU artifact alone would prove the table, not the hardware decode.
///
/// So the decision is: keep them, and make the two things that could turn the
/// decision wrong visible. The size, so growth is a number in a diff rather than
/// a surprise; and the artifact set itself, so "oracle-only" never quietly
/// becomes "CPU-only".
#[test]
fn an_oracle_only_kernel_still_emits_every_artifact() {
    let kernels = generate_all().unwrap();
    let oracle_only: Vec<String> = rir_kernels::registry()
        .into_iter()
        .filter(|r| r.integration.ggml_op.is_none())
        .map(|r| r.kernel.name().to_string())
        .collect();
    assert!(
        oracle_only.len() >= 14,
        "the oracle-only set shrank to {}: {oracle_only:?}",
        oracle_only.len()
    );

    for name in &oracle_only {
        let gk = kernels
            .iter()
            .find(|k| &k.name == name)
            .unwrap_or_else(|| panic!("{name}: no generated artifacts at all"));
        let files: Vec<&str> = gk.files.iter().map(|(r, _)| r.as_str()).collect();
        for required in [
            "cpu.rs",
            "kernel.comp",
            "kernel.metal",
            "manifest.vulkan.json",
            "manifest.metal.json",
        ] {
            assert!(
                files.contains(&required),
                "{name}: {required} is missing - an oracle-only kernel keeps the \
                 full set on purpose, see this test's rationale"
            );
        }
    }

    // The size budget. 2.7 MiB today across 442 files; the ceiling is loose
    // because the point is not to be tight, it is that doubling it fails here
    // instead of being noticed by a clone.
    let total: usize = kernels
        .iter()
        .flat_map(|k| k.files.iter().map(|(_, c)| c.len()))
        .sum();
    assert!(
        total < 6 * 1024 * 1024,
        "generated artifacts total {total} bytes: past the budget this test \
         carries, decide again whether the oracle-only half still pays"
    );
}
