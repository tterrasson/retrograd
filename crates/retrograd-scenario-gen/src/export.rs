use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use retrograd_agent_core::{Error, Result, Scenario};
use retrograd_core::hex_lower;
use sha2::{Digest, Sha256};

use crate::{GeneratedCorpus, GenerationManifest};

/// The manifest path a corpus at `path` is checked against: `path` with
/// `.manifest.json` appended.
pub fn manifest_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.manifest.json", path.display()))
}

fn corpus_bytes(scenarios: &[Scenario]) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    for scenario in scenarios {
        serde_json::to_writer(&mut bytes, scenario)
            .map_err(|error| Error::invalid(format!("serialize generated scenario: {error}")))?;
        bytes.push(b'\n');
    }
    Ok(bytes)
}

pub(crate) fn corpus_hash(scenarios: &[Scenario]) -> Result<String> {
    Ok(hex_lower(&Sha256::digest(corpus_bytes(scenarios)?)))
}

/// Writes the corpus and its manifest atomically, refusing to clobber an
/// existing pair unless `overwrite` is set.
///
/// Both files are written to a temporary name and synced, then renamed into
/// place - and the existence check is repeated right before the rename - so a
/// reader never observes a manifest without its corpus, a corpus without its
/// manifest, or a corpus this call silently replaced.
pub fn write_corpus(corpus: &GeneratedCorpus, path: &Path, overwrite: bool) -> Result<()> {
    let manifest = manifest_path(path);
    if !overwrite && (path.exists() || manifest.exists()) {
        return Err(Error::invalid(format!(
            "refusing to overwrite {} or its manifest without --force",
            path.display()
        )));
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent).map_err(io_error)?;
    let corpus_data = corpus_bytes(&corpus.scenarios)?;
    let actual_hash = hex_lower(&Sha256::digest(&corpus_data));
    if actual_hash != corpus.manifest.corpus_sha256 {
        return Err(Error::invalid(
            "generated corpus changed after its manifest was built",
        ));
    }
    let manifest_data = serde_json::to_vec_pretty(&corpus.manifest)
        .map_err(|error| Error::invalid(format!("serialize generation manifest: {error}")))?;
    let nonce = format!(
        "{}.{}",
        std::process::id(),
        actual_hash.get(..8).unwrap_or("tmp")
    );
    let tmp_manifest = parent.join(format!(".retrograd-manifest-{nonce}.tmp"));
    let tmp_corpus = parent.join(format!(".retrograd-corpus-{nonce}.tmp"));
    write_synced(&tmp_manifest, &manifest_data)?;
    write_synced(&tmp_corpus, &corpus_data)?;
    if !overwrite && (path.exists() || manifest.exists()) {
        let _ = fs::remove_file(&tmp_manifest);
        let _ = fs::remove_file(&tmp_corpus);
        return Err(Error::invalid(
            "scenario output appeared concurrently; refusing to overwrite it",
        ));
    }
    fs::rename(&tmp_manifest, &manifest).map_err(io_error)?;
    fs::rename(&tmp_corpus, path).map_err(io_error)?;
    Ok(())
}

fn write_synced(path: &Path, data: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(io_error)?;
    file.write_all(data).map_err(io_error)?;
    file.sync_all().map_err(io_error)?;
    Ok(())
}

/// Reads the manifest beside `path`, if there is one, and checks the corpus's
/// hash against `corpus_sha256`.
///
/// `Ok(None)` when no manifest exists - a corpus written by hand, or by
/// something other than [`write_corpus`], is not an error. `Err` when a
/// manifest exists but the corpus no longer matches it, which is what a
/// hand-edit or a partial write looks like.
pub fn verify_manifest(path: &Path) -> Result<Option<GenerationManifest>> {
    let manifest_path = manifest_path(path);
    if !manifest_path.exists() {
        return Ok(None);
    }
    let manifest: GenerationManifest =
        serde_json::from_slice(&fs::read(&manifest_path).map_err(io_error)?).map_err(|error| {
            Error::invalid(format!(
                "{}: invalid generation manifest: {error}",
                manifest_path.display()
            ))
        })?;
    let actual = hex_lower(&Sha256::digest(fs::read(path).map_err(io_error)?));
    if actual != manifest.corpus_sha256 {
        return Err(Error::invalid(format!(
            "{} does not match corpus_sha256 in {}",
            path.display(),
            manifest_path.display()
        )));
    }
    Ok(Some(manifest))
}

fn io_error(error: std::io::Error) -> Error {
    Error::Tool(format!("scenario corpus I/O: {error}"))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::types::{CatalogSummary, RejectionCounts};

    /// A directory of its own under the system temp dir, removed on drop.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "retrograd-scenario-gen-{}-{name}",
                std::process::id()
            ));
            let _ = fs::remove_dir_all(&dir);
            fs::create_dir_all(&dir).expect("create the scratch directory");
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn corpus() -> GeneratedCorpus {
        let scenarios = vec![Scenario {
            id: "s-1".into(),
            system: None,
            user: "List the files in the workspace.".into(),
            metadata: Default::default(),
        }];
        let manifest = GenerationManifest {
            schema_version: 1,
            generator_model: "model".into(),
            generator_base_url: "http://localhost".into(),
            prompt_version: crate::PROMPT_VERSION.into(),
            prompt_sha256: String::new(),
            catalog_sha256: String::new(),
            catalog: CatalogSummary {
                tools: 1,
                session_tools: 0,
                mcp_servers: Vec::new(),
                stateless_shared: Vec::new(),
                resources: 0,
                warnings: Vec::new(),
            },
            config_sha256: String::new(),
            corpus_sha256: corpus_hash(&scenarios).expect("hash the scenarios"),
            requested: 1,
            accepted: 1,
            rejected: RejectionCounts::default(),
            difficulty_distribution: BTreeMap::new(),
            tool_coverage: BTreeMap::new(),
            near_duplicate_threshold: 0.9,
        };
        GeneratedCorpus {
            scenarios,
            manifest,
        }
    }

    #[test]
    fn a_written_corpus_verifies_against_its_manifest() {
        let scratch = Scratch::new("round-trip");
        let path = scratch.0.join("nested/scenarios.jsonl");
        let corpus = corpus();
        write_corpus(&corpus, &path, false).expect("write the corpus");
        let manifest = verify_manifest(&path)
            .expect("verify the corpus")
            .expect("a manifest beside it");
        assert_eq!(manifest.corpus_sha256, corpus.manifest.corpus_sha256);
    }

    #[test]
    fn an_existing_corpus_is_replaced_only_when_asked() {
        let scratch = Scratch::new("overwrite");
        let path = scratch.0.join("scenarios.jsonl");
        let corpus = corpus();
        write_corpus(&corpus, &path, false).expect("first write");
        let error = write_corpus(&corpus, &path, false).expect_err("second write");
        assert!(
            error.to_string().contains("refusing to overwrite"),
            "{error}"
        );
        write_corpus(&corpus, &path, true).expect("forced write");
    }

    #[test]
    fn a_corpus_edited_after_its_manifest_is_refused() {
        let scratch = Scratch::new("tampered");
        let path = scratch.0.join("scenarios.jsonl");
        write_corpus(&corpus(), &path, false).expect("write the corpus");
        let mut file = OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open the corpus");
        file.write_all(b"{}\n").expect("append a row");
        let error = verify_manifest(&path).expect_err("hash mismatch");
        assert!(
            error.to_string().contains("does not match corpus_sha256"),
            "{error}"
        );
    }

    #[test]
    fn a_corpus_without_a_manifest_has_nothing_to_verify() {
        let scratch = Scratch::new("no-manifest");
        let path = scratch.0.join("scenarios.jsonl");
        fs::write(&path, b"{}\n").expect("write a bare corpus");
        let manifest = verify_manifest(&path).expect("a missing manifest is not an error");
        assert!(manifest.is_none());
    }

    #[test]
    fn a_manifest_built_for_other_scenarios_is_not_written() {
        let scratch = Scratch::new("stale-manifest");
        let path = scratch.0.join("scenarios.jsonl");
        let mut corpus = corpus();
        corpus.scenarios[0].user.push_str(" Then delete them.");
        let error = write_corpus(&corpus, &path, false).expect_err("stale manifest");
        assert!(
            error
                .to_string()
                .contains("changed after its manifest was built"),
            "{error}"
        );
        assert!(!path.exists() && !manifest_path(&path).exists());
    }
}
