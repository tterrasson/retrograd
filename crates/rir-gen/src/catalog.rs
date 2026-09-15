//! The catalog, the Metal aggregate, the Vulkan artifact list, and the list of
//! artifacts a production policy actually dispatches.

use rir_lower::Backend;
use sha2::{Digest, Sha256};

use crate::{GenError, MSL_HEADER_LINES};

/// Serializes the same lowered variants and integration table as the C++
/// registry into a strict, versioned planner catalogue. No registry rule is
/// reconstructed here: policies and domain declarations come directly from
/// `IntegrationSpec`, while variant metadata is read from the manifest emitter
/// fed by the same `LoopKernel`.
pub fn emit_catalog(
    variants: &[(&rir_lower::LoopKernel, &rir_emit::IntegrationSpec)],
    specs: &[rir_emit::IntegrationSpec],
) -> String {
    use std::collections::BTreeMap;

    use rir_core::catalog::{
        KernelCatalog, KernelDomainAssumption, KernelPolicy, KernelShapeRule, KernelVariantPolicy,
        SCHEMA_VERSION,
    };
    use rir_emit::GgmlBackend;

    type Key = (String, Option<String>, GgmlBackend);
    let mut rows: BTreeMap<Key, KernelPolicy> = BTreeMap::new();
    let backends = [
        GgmlBackend::Cpu,
        GgmlBackend::Cuda,
        GgmlBackend::Metal,
        GgmlBackend::Vulkan,
    ];

    for spec in specs {
        let Some(op) = spec.ggml_op else { continue };
        // The same filter as `rir_op_policies`: a spec that does not claim
        // production has no row in the table it is supposed to mirror.
        if !spec.production {
            continue;
        }
        let op_variant = spec.ggml_op_variant.map(str::to_string);
        // Since this rule was introduced one op may be served by several kernels - one per `src0`
        // dtype - all sharing this row's key.
        let serving = specs
            .iter()
            .filter(|other| {
                other.production
                    && other.ggml_op == Some(op)
                    && other.ggml_op_variant == spec.ggml_op_variant
            })
            .collect::<Vec<_>>();
        // The declared domain is the **intersection** of what the serving
        // kernels declare, exactly as the C++ registry publishes it
        // (`rir-emit/src/registry/`, ADR-4): each says what
        // *it* does not claim, so the op leaves to the native kernel only what
        // none of them claims. Unioning them instead publishes a row that
        // rejects every input it is meant to serve - `GGML_OP_OUT_PROD` would
        // declare both "F32 src0 not claimed" and "quantized src0 not claimed".
        let domain = serving
            .iter()
            .fold(u32::MAX, |mask, other| mask & other.assumed_domain_mask());
        let mut declared_domain: Vec<KernelDomainAssumption> = Vec::new();
        for other in &serving {
            for assumption in &other.assumed_domain {
                if assumption.restriction.bit() & domain == 0 {
                    continue; // covered by a sibling kernel, so not the op's
                }
                let assumption = KernelDomainAssumption {
                    reject: assumption.restriction.name().to_string(),
                    kernel: other.kernel.to_string(),
                    why: assumption
                        .why
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" "),
                };
                // Kernels of one op usually share a restriction verbatim (the
                // broadcast one is the same sentence thirteen times). The reason
                // is published once, attributed to the first kernel that stated
                // it - same rule as the registry's comment block.
                let said = declared_domain
                    .iter()
                    .any(|other| other.reject == assumption.reject && other.why == assumption.why);
                if !said {
                    declared_domain.push(assumption);
                }
            }
        }
        for backend in backends {
            let key = (op.to_string(), op_variant.clone(), backend);
            // `emit_registry` already refused two policies, two retirement
            // claims or two exceptions for one op, so the serving kernels agree
            // and the first of them speaks for the whole row. Merging with
            // `max`/`|=` here would quietly repair a disagreement the generator
            // is supposed to reject.
            rows.entry(key).or_insert_with(|| KernelPolicy {
                ggml_op: op.to_string(),
                op_variant: op_variant.clone(),
                backend,
                policy: spec.policy_for(backend),
                variants: Vec::new(),
                declared_domain: declared_domain.clone(),
                native_retired: spec.native_retired_on(backend),
                native_exception: serving
                    .iter()
                    .find_map(|other| other.native_exception)
                    .map(str::to_string),
            });
        }
    }

    for (kernel, spec) in variants {
        let backend = rir_emit::manifest::ggml_backend(kernel);
        let Some(op) = spec.ggml_op else { continue };
        let key = (
            op.to_string(),
            spec.ggml_op_variant.map(str::to_string),
            backend,
        );
        let manifest: serde_json::Value =
            serde_json::from_str(&rir_emit::emit_manifest(kernel, Some(spec)))
                .expect("the manifest emitter must emit valid JSON");
        let strings = |field: &str| {
            manifest[field]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect::<Vec<_>>()
        };
        let eligible_when = manifest["eligible_when"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|rule| KernelShapeRule {
                axes: rule["axes"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|axis| axis.as_str().map(str::to_string))
                    .collect(),
                min: rule["min"].as_u64().unwrap_or(0) as u32,
                max: rule["max"].as_u64().unwrap_or(u32::MAX as u64) as u32,
            })
            .collect();
        let workgroup_values = manifest["workgroup"].as_array().expect("workgroup");
        let workgroup = std::array::from_fn(|i| {
            workgroup_values
                .get(i)
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1) as u32
        });
        rows.get_mut(&key)
            .expect("every production variant has a policy row")
            .variants
            .push(KernelVariantPolicy {
                variant_id: manifest["variant_id"]
                    .as_str()
                    .unwrap_or_default()
                    .to_string(),
                kernel: kernel.name.clone(),
                priority: manifest["priority"].as_u64().unwrap_or(0) as u8,
                constraints: strings("constraints"),
                eligible_when,
                features: strings("features"),
                workgroup,
                vector_width: manifest["vector_width"].as_u64().unwrap_or(1) as u32,
                reduction: manifest["reduction"]
                    .as_str()
                    .unwrap_or("unknown")
                    .to_string(),
                scan: manifest["scan"].as_str().unwrap_or("unknown").to_string(),
            });
    }

    let mut kernels: Vec<KernelPolicy> = rows.into_values().collect();
    for row in &mut kernels {
        row.declared_domain.sort_by(|a, b| {
            a.reject
                .cmp(&b.reject)
                .then(a.kernel.cmp(&b.kernel))
                .then(a.why.cmp(&b.why))
        });
        row.variants.sort_by(|a, b| {
            b.priority
                .cmp(&a.priority)
                .then(a.variant_id.cmp(&b.variant_id))
        });
    }
    let content = serde_json::to_vec(&(SCHEMA_VERSION, &kernels)).expect("catalog content");
    let digest = Sha256::digest(content);
    let fingerprint = format!(
        "sha256:{}",
        digest
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    );
    let catalog = KernelCatalog {
        schema_version: SCHEMA_VERSION,
        fingerprint,
        kernels,
    };
    format!(
        "{}\n",
        serde_json::to_string_pretty(&catalog).expect("catalog JSON")
    )
}

/// Concatenates the production MSL bodies into the file `ggml-metal.metal`
/// includes. Each body is its standalone source minus the header lines: the
/// including source has already included `<metal_stdlib>` and opened the
/// namespace, and repeating either would be a redefinition.
pub fn metal_aggregate(bodies: &[(String, String)]) -> String {
    let mut out = String::from(
        "#include \"common.h\" // retro delta: fork-owned RIR kernels\n\
         // Generated by rir-gen - DO NOT EDIT MANUALLY.\n\
         // Every production RIR kernel body for Metal, concatenated.\n\
         // The fork lists this file among its Metal translation units in one\n\
         // line; there is no marker-delimited copy to maintain.\n\
         //\n\
         // It is a translation unit of the fork's `ggml-metal/kernels/` layout,\n\
         // so it opens with the `common.h` its siblings open with and nothing\n\
         // else. The standalone per-variant sources under\n\
         // generated/rir/<kernel>/ remain what the compile test compiles.\n",
    );
    for (artifact, source) in bodies {
        out.push_str(&format!("\n// ---- {artifact} ----\n"));
        out.push_str(
            &source
                .lines()
                .skip(MSL_HEADER_LINES)
                .collect::<Vec<_>>()
                .join("\n"),
        );
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    out
}

/// Writes the X-macro the Vulkan shader generator expands into one
/// `string_to_spv` call per generated module (ADR-4 section 8).
///
/// The list is the whole content: a RIR shader takes no `#define`, its file
/// name is its artifact name, and its module name is that name prefixed,
/// three facts the expansion in `vulkan-shaders-gen.cpp` carries once instead
/// of once per line. That is why one X-macro replaces sixty-six lines and why
/// the sixty-seventh costs nothing.
pub fn vulkan_artifact_list(artifacts: &[String]) -> String {
    let mut out = String::from(
        "// Generated by rir-gen - DO NOT EDIT MANUALLY.\n\
         // Every production RIR shader module for Vulkan, as an X-macro.\n\
         // `vulkan-shaders-gen.cpp` expands this list in\n\
         // one place; adding a generated variant does not edit a vendored file.\n\
         //\n\
         // The name is the artifact name, without its `rir_` prefix: the module,\n\
         // the source file and the registry's `artifact` field are all derived\n\
         // from it, so a row here is a name and nothing else.\n\
         #pragma once\n\n\
         #define RIR_VK_ARTIFACTS(X) \\\n",
    );
    if artifacts.is_empty() {
        // An empty X-macro must still expand to something a compiler accepts,
        // and it must not silently look like a list.
        out.push_str("    /* no RIR shader in this build */\n");
        return out;
    }
    for (i, a) in artifacts.iter().enumerate() {
        out.push_str(&format!(
            "    X({a}){}\n",
            if i + 1 < artifacts.len() { " \\" } else { "" }
        ));
    }
    out
}

/// Writes the X-macro the fork expands into one `extern "C"` declaration and
/// one table row per generated launch stub.
///
/// The list is the whole content, for the reason Vulkan's is: a RIR stub's name
/// is `rir_launch_<artifact>`, its file is `rir_<artifact>.cu`, and its
/// signature is fixed - three facts the expansion in `rir-cuda-launch.cu`
/// carries once instead of once per line.
///
/// What CUDA adds is that the table is the *only* way in. Vulkan can look a blob
/// up by name because a SPIR-V module is data; a `__global__` is a symbol, and a
/// symbol absent from this list is a kernel the fork cannot launch at all.
pub fn cuda_launcher_list(artifacts: &[String]) -> String {
    let mut out = String::from(
        "// Generated by rir-gen - DO NOT EDIT MANUALLY.\n\
         // Every production RIR launch stub for CUDA, as an X-macro.\n\
         // `rir-cuda-launch.cu` expands this list twice: once to declare the\n\
         // stubs, once to build the artifact → pointer table.\n\
         //\n\
         // The name is the artifact name. The stub it declares is\n\
         // `rir_launch_<name>`, defined by `rir/rir_<name>.cu`, which the\n\
         // backend's `file(GLOB \"rir/*.cu\")` compiles - so adding a variant\n\
         // edits no vendored file.\n\
         #pragma once\n\n\
         #define RIR_CUDA_LAUNCHERS(X) \\\n",
    );
    if artifacts.is_empty() {
        // An empty X-macro must still expand to something a compiler accepts,
        // and it must not silently look like a list.
        out.push_str("    /* no RIR kernel in this build */\n");
        return out;
    }
    for (i, a) in artifacts.iter().enumerate() {
        out.push_str(&format!(
            "    X({a}){}\n",
            if i + 1 < artifacts.len() { " \\" } else { "" }
        ));
    }
    out
}

/// One generated GPU artifact a production build must carry: a `(kernel,
/// backend, variant)` triple, resolved to the file that holds it here and the
/// name the fork knows it by.
#[derive(Clone, Debug)]
pub struct ProductionArtifact {
    /// Directory under `generated/rir/` - the kernel name.
    pub kernel: String,
    /// `rir_<artifact>` in the fork: the shader file, the SPIR-V symbol, the
    /// MSL entrypoint. Equal to `kernel` for a pair's fallback variant.
    pub artifact: String,
    pub backend: rir_emit::GgmlBackend,
    /// Variant name, `None` for the fallback.
    pub variant: Option<String>,
    /// File name inside the kernel directory.
    pub source_file: String,
}

/// Every GPU artifact the fork must carry, derived from the schedule table and
/// the integration table rather than listed anywhere.
///
/// This is the enumeration a second variant made necessary: "one shader per
/// production kernel" stopped being true the moment a (kernel, backend) pair
/// could publish two lowerings, and every consumer that assumed it - the
/// drift test, the shader-generator list, the pipeline table - has to walk
/// variants instead.
pub fn production_artifacts() -> Result<Vec<ProductionArtifact>, GenError> {
    let mut out = Vec::new();
    for entry in rir_kernels::registry() {
        let (kernel, spec) = (&entry.kernel, &entry.integration);
        for schedule in entry.schedules {
            if schedule.backend() == Backend::Cpu {
                continue;
            }
            let variant = schedule.variant().map(str::to_string);
            let lk = rir_lower::lower(kernel, schedule)?;
            let backend = rir_emit::manifest::ggml_backend(&lk);
            if !spec.production_on(backend) {
                continue;
            }
            out.push(ProductionArtifact {
                kernel: kernel.name().to_string(),
                artifact: rir_emit::artifact_name(&lk),
                backend,
                variant,
                source_file: rir_emit::artifact_files(&lk).0,
            });
        }
    }
    Ok(out)
}
