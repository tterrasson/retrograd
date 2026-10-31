//! The model profiles the catalogue publishes: plain data, no decision.
//!
//! A profile names an exact model shape on one backend. It is declared before
//! anything is chosen for it, so that `rir-sweep --profile` can measure the
//! kernels on that model's own shapes; a selection is added here, by hand,
//! once a sweep and the promotion lane say a variant wins in a regime. Until
//! then a profile changes nothing at dispatch.
//!
//! The shapes are the reference models' published hyperparameters. The weight
//! formats are the two quantizations a reference GGUF is usually taken in:
//! `Q8_0` (every matrix `q8_0`) and `Q4_K_M` (a `q4_K`/`q6_K` mix).

use crate::catalog::{GgmlBackend, ModelProfile, ModelShape};

/// `(name, n_embd, n_head, n_head_kv, head_dim, n_ff)` of the dense Qwen3
/// reference sizes.
const QWEN3: [(&str, u32, u32, u32, u32, u32); 3] = [
    ("qwen3-0.6b", 1024, 16, 8, 128, 3072),
    ("qwen3-1.7b", 2048, 16, 8, 128, 6144),
    ("qwen3-4b", 2560, 32, 8, 128, 9728),
];

/// `(suffix, weight formats)` of the reference quantizations.
const FORMATS: [(&str, &[&str]); 2] = [("q8_0", &["q8_0"]), ("q4_k_m", &["q4_K", "q6_K"])];

/// Every profile the catalogue publishes, in a stable order.
pub fn reference() -> Vec<ModelProfile> {
    let mut profiles = Vec::new();
    for (name, n_embd, n_head, n_head_kv, head_dim, n_ff) in QWEN3 {
        for (suffix, formats) in FORMATS {
            profiles.push(ModelProfile {
                name: format!("{name}-{suffix}"),
                backend: GgmlBackend::Metal,
                shape: ModelShape {
                    architecture: "qwen3".into(),
                    n_embd,
                    n_head,
                    n_head_kv,
                    head_dim,
                    n_ff,
                    weight_formats: formats.iter().map(|f| f.to_string()).collect(),
                },
                selections: Vec::new(),
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
        }
    }
}
