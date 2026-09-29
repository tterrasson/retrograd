//! `GET /v1/model-files`: the GGUF files a recipe may name.
//!
//! A bounded walk of the path roots, by name only - no header is read here. The
//! geometry of the file a client picks comes from `POST /v1/preflight`, which is
//! already rate-limited by the probe semaphore; listing a directory of forty
//! models must not open forty of them.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use crate::dto;

/// How deep under a root the walk goes.
pub const MAX_DEPTH: usize = 4;
/// How many files a listing reports before it says it stopped.
pub const MAX_FILES: usize = 2_000;
/// How many directory entries the walk reads in all, files or not. A root that
/// is someone's home directory must not cost a minute.
const MAX_ENTRIES: usize = 50_000;
/// How long a listing is reused before the disk is read again.
pub const CACHE_TTL: Duration = Duration::from_secs(30);

/// A listing and when it was taken.
#[derive(Clone, Debug)]
pub struct Cached {
    pub taken: Instant,
    pub listing: dto::ModelFileListing,
}

impl Cached {
    pub fn fresh(&self) -> bool {
        self.taken.elapsed() < CACHE_TTL
    }
}

/// Walks `roots` for `*.gguf` files.
///
/// Hidden entries are skipped. A symbolic link is followed only when what it
/// points at is still inside the root it was found in: a link is how a models
/// directory is usually assembled, and a link out of the root is how a listing
/// would describe files the server refuses to open anyway.
pub fn scan(roots: &[PathBuf]) -> dto::ModelFileListing {
    let mut files = Vec::new();
    let mut truncated = false;
    let mut budget = MAX_ENTRIES;
    let mut listed_roots = Vec::new();
    for root in roots {
        let Ok(root) = root.canonicalize() else {
            continue;
        };
        listed_roots.push(root.display().to_string());
        walk(&root, &root, 0, &mut files, &mut truncated, &mut budget);
        if truncated {
            break;
        }
    }
    files.sort_by(|left: &dto::ModelFile, right| left.path.cmp(&right.path));
    dto::ModelFileListing {
        roots: listed_roots,
        files,
        truncated,
    }
}

fn walk(
    root: &Path,
    directory: &Path,
    depth: usize,
    files: &mut Vec<dto::ModelFile>,
    truncated: &mut bool,
    budget: &mut usize,
) {
    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        if *truncated {
            return;
        }
        if *budget == 0 {
            *truncated = true;
            return;
        }
        *budget -= 1;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.starts_with('.') {
            continue;
        }
        let Ok(kind) = entry.file_type() else {
            continue;
        };
        let path = if kind.is_symlink() {
            match entry.path().canonicalize() {
                Ok(target) if target.starts_with(root) => target,
                _ => continue,
            }
        } else {
            entry.path()
        };
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if metadata.is_dir() {
            if depth + 1 < MAX_DEPTH {
                walk(root, &path, depth + 1, files, truncated, budget);
            }
            continue;
        }
        let is_gguf = Path::new(&name)
            .extension()
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"));
        if !metadata.is_file() || !is_gguf {
            continue;
        }
        if files.len() == MAX_FILES {
            *truncated = true;
            return;
        }
        // Reported as found, under the root it was found in: the link's own
        // path, which is what a client sees in its directory, not the target.
        let listed = entry.path();
        files.push(dto::ModelFile {
            path: listed.display().to_string(),
            root: root.display().to_string(),
            relative: listed
                .strip_prefix(root)
                .unwrap_or(&listed)
                .display()
                .to_string(),
            bytes: metadata.len(),
            modified_at: metadata
                .modified()
                .ok()
                .and_then(|time| time.duration_since(SystemTime::UNIX_EPOCH).ok())
                .map(|elapsed| elapsed.as_secs()),
            role: role_of(&name),
        });
    }
}

/// What a file is for, judged by its name alone. A vision projector and a
/// vocabulary-only file are GGUFs a recipe cannot train, and are marked rather
/// than hidden: a client that expected one should see why it cannot pick it.
fn role_of(name: &str) -> dto::ModelFileRole {
    let lower = name.to_ascii_lowercase();
    if lower.starts_with("mmproj") {
        dto::ModelFileRole::Projector
    } else if lower.contains("vocab") {
        dto::ModelFileRole::Vocab
    } else {
        dto::ModelFileRole::Model
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "retrograd-model-files-{label}-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp root");
        dir
    }

    #[test]
    fn files_are_found_by_extension_and_marked_by_name() {
        let root = temp_root("roles");
        std::fs::create_dir_all(root.join("qwen")).expect("dir");
        std::fs::write(root.join("qwen/model-q8_0.gguf"), b"x").expect("file");
        std::fs::write(root.join("qwen/mmproj-f16.gguf"), b"x").expect("file");
        std::fs::write(root.join("ggml-vocab-llama.gguf"), b"x").expect("file");
        std::fs::write(root.join("notes.txt"), b"x").expect("file");
        std::fs::write(root.join(".hidden.gguf"), b"x").expect("file");
        let listing = scan(std::slice::from_ref(&root));
        let roles: Vec<(&str, dto::ModelFileRole)> = listing
            .files
            .iter()
            .map(|file| (file.relative.as_str(), file.role))
            .collect();
        assert_eq!(
            roles,
            vec![
                ("ggml-vocab-llama.gguf", dto::ModelFileRole::Vocab),
                ("qwen/mmproj-f16.gguf", dto::ModelFileRole::Projector),
                ("qwen/model-q8_0.gguf", dto::ModelFileRole::Model),
            ]
        );
        assert!(!listing.truncated);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn the_walk_stops_at_its_depth() {
        let root = temp_root("depth");
        let deep = root.join("a/b/c/d");
        std::fs::create_dir_all(&deep).expect("dir");
        std::fs::write(root.join("a/b/c/shallow.gguf"), b"x").expect("file");
        std::fs::write(deep.join("deep.gguf"), b"x").expect("file");
        let listing = scan(std::slice::from_ref(&root));
        let names: Vec<&str> = listing
            .files
            .iter()
            .map(|file| file.relative.as_str())
            .collect();
        assert_eq!(names, vec!["a/b/c/shallow.gguf"]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_listing_past_its_bound_says_it_stopped() {
        let root = temp_root("bound");
        for index in 0..=MAX_FILES {
            std::fs::write(root.join(format!("m{index:05}.gguf")), b"").expect("file");
        }
        let listing = scan(std::slice::from_ref(&root));
        assert_eq!(listing.files.len(), MAX_FILES);
        assert!(listing.truncated);
        let _ = std::fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn a_link_out_of_the_root_is_not_followed() {
        let root = temp_root("links");
        let outside = temp_root("links-outside");
        std::fs::write(outside.join("secret.gguf"), b"x").expect("file");
        std::fs::create_dir_all(root.join("real")).expect("dir");
        std::fs::write(root.join("real/inside.gguf"), b"x").expect("file");
        std::os::unix::fs::symlink(outside.join("secret.gguf"), root.join("escape.gguf"))
            .expect("symlink");
        std::os::unix::fs::symlink(root.join("real/inside.gguf"), root.join("alias.gguf"))
            .expect("symlink");
        let listing = scan(std::slice::from_ref(&root));
        let names: Vec<&str> = listing
            .files
            .iter()
            .map(|file| file.relative.as_str())
            .collect();
        assert_eq!(names, vec!["alias.gguf", "real/inside.gguf"]);
        let _ = std::fs::remove_dir_all(root);
        let _ = std::fs::remove_dir_all(outside);
    }
}
