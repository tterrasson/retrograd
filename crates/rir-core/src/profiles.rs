//! The model profiles the catalogue publishes: plain data, no decision.
//!
//! A profile names an exact model shape on one backend, so that
//! `rir-sweep --profile` can measure the kernels on that model's own shapes.
//! It selects nothing: the dispatch stays per-claim, and a profile is a
//! geometry a measurement grid is derived from, never a runtime choice.
//!
//! The shapes are the reference models' published hyperparameters. The weight
//! formats are the two quantizations a reference GGUF is usually taken in:
//! `Q8_0` (every matrix `q8_0`) and `Q4_K_M` (a `q4_K`/`q6_K` mix).
//!
//! Beside the Qwen3 sizes, one 8B-class Llama shape: its FFN is 14 336 wide,
//! the depth at which the decode GEMV measured furthest from the native
//! kernel, and no Qwen3 reference size reaches it.

use crate::catalog::{GgmlBackend, ModelProfile, ModelShape};

/// `(name, n_embd, n_head, n_head_kv, head_dim, n_ff)` of the dense Qwen3
/// reference sizes.
const QWEN3: [(&str, u32, u32, u32, u32, u32); 3] = [
    ("qwen3-0.6b", 1024, 16, 8, 128, 3072),
    ("qwen3-1.7b", 2048, 16, 8, 128, 6144),
    ("qwen3-4b", 2560, 32, 8, 128, 9728),
];

/// `(name, architecture, n_embd, n_head, n_head_kv, head_dim, n_ff)` of the
/// reference sizes that are not Qwen3.
const OTHERS: [(&str, &str, u32, u32, u32, u32, u32); 1] =
    [("llama3-8b", "llama", 4096, 32, 8, 128, 14336)];

/// `(suffix, weight formats)` of the reference quantizations.
const FORMATS: [(&str, &[&str]); 2] = [("q8_0", &["q8_0"]), ("q4_k_m", &["q4_K", "q6_K"])];

/// Every profile the catalogue publishes, in a stable order.
pub fn reference() -> Vec<ModelProfile> {
    let sizes = QWEN3
        .iter()
        .map(|&(name, n_embd, n_head, n_head_kv, head_dim, n_ff)| {
            (name, "qwen3", n_embd, n_head, n_head_kv, head_dim, n_ff)
        })
        .chain(OTHERS);
    let mut profiles = Vec::new();
    for (name, architecture, n_embd, n_head, n_head_kv, head_dim, n_ff) in sizes {
        for (suffix, formats) in FORMATS {
            profiles.push(ModelProfile {
                name: format!("{name}-{suffix}"),
                backend: GgmlBackend::Metal,
                shape: ModelShape {
                    architecture: architecture.into(),
                    n_embd,
                    n_head,
                    n_head_kv,
                    head_dim,
                    n_ff,
                    weight_formats: formats.iter().map(|f| f.to_string()).collect(),
                },
            });
        }
    }
    profiles
}

/// The published profile of that name.
pub fn named(name: &str) -> Option<ModelProfile> {
    reference().into_iter().find(|p| p.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_profiles_have_unique_names_and_consistent_heads() {
        let profiles = reference();
        let mut names: Vec<&str> = profiles.iter().map(|p| p.name.as_str()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), profiles.len());
        for p in &profiles {
            // Grouped-query attention: the query heads are a whole number of
            // KV groups.
            assert_eq!(p.shape.n_head % p.shape.n_head_kv, 0, "{}", p.name);
            assert!(named(&p.name).is_some());
            // The head products fit, so `matmul_shapes` never saturates on a
            // published profile.
            assert!(
                p.shape
                    .n_head
                    .checked_mul(p.shape.head_dim)
                    .is_some_and(|q| q < u32::MAX),
                "{}",
                p.name
            );
        }
    }

    /// The weight products follow the hyperparameters: Q, K, V, output, gate,
    /// up, down, each `(m, k)`.
    #[test]
    fn the_llama_profile_derives_its_projections() {
        let p = named("llama3-8b-q4_k_m").unwrap();
        assert_eq!(p.shape.architecture, "llama");
        assert_eq!(p.shape.weight_formats, ["q4_K", "q6_K"]);
        assert_eq!(
            p.shape.matmul_shapes(),
            [
                (4096, 4096),
                (1024, 4096),
                (1024, 4096),
                (4096, 4096),
                (14336, 4096),
                (14336, 4096),
                (4096, 14336),
            ]
        );
        assert!(named("llama3-8b-q8_0").is_some());
    }
}
