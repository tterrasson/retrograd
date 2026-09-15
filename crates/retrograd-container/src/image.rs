//! Getting an image, and pinning what was got.

use bollard::query_parameters::CreateImageOptionsBuilder;
use futures::StreamExt;
use retrograd_agent_core::{Error, Result};

use crate::client::{DockerClient, docker_error, is_not_found};

/// Makes sure `reference` is present locally and returns the digest it resolved
/// to.
///
/// Two properties, both about a run being reproducible rather than about speed:
///
/// - **it never builds.** Building at rollout time makes the environment a
///   function of the machine and of the day; `ensure_image` pulls a published
///   image or fails;
/// - **it resolves to a digest.** `python:3.12` is a moving tag. A run resumed
///   three weeks later would train against a different environment and nothing
///   would say so. The digest goes into the run metadata, the same discipline as
///   pinning llama.cpp in this repository.
///
/// The returned reference is `name@sha256:…` when the daemon knows a repo
/// digest. An image built locally and never pushed has none, and then the
/// image's own id (`sha256:…`) is used instead: it is not portable to another
/// machine, but it is immutable, which the tag it was built under is not. Only
/// an image with neither - which should not happen - falls back to the input.
pub async fn ensure_image(client: &DockerClient, reference: &str) -> Result<String> {
    if reference.trim().is_empty() {
        return Err(Error::invalid("image reference is empty"));
    }
    match client.docker().inspect_image(reference).await {
        Ok(image) => Ok(pinned(
            reference,
            image.repo_digests.unwrap_or_default(),
            image.id.as_deref(),
        )),
        Err(error) if is_not_found(&error) => {
            pull(client, reference).await?;
            let image = client
                .docker()
                .inspect_image(reference)
                .await
                .map_err(|error| {
                    docker_error(&format!("inspect '{reference}' after pull"), error)
                })?;
            Ok(pinned(
                reference,
                image.repo_digests.unwrap_or_default(),
                image.id.as_deref(),
            ))
        }
        Err(error) => Err(docker_error(&format!("inspect image '{reference}'"), error)),
    }
}

async fn pull(client: &DockerClient, reference: &str) -> Result<()> {
    let (image, tag) = split_reference(reference);
    tracing::info!(image = %reference, "pulling image");
    let options = CreateImageOptionsBuilder::default()
        .from_image(image)
        .tag(tag)
        .build();
    let mut stream = client.docker().create_image(Some(options), None, None);
    let mut last = String::new();
    while let Some(item) = stream.next().await {
        let info = item.map_err(|error| docker_error(&format!("pull '{reference}'"), error))?;
        if let Some(status) = info.status {
            // One line per distinct status rather than one per layer chunk:
            // a pull emits thousands of progress events.
            if status != last {
                tracing::info!(image = %reference, status = %status, "pull progress");
                last = status;
            }
        }
        if let Some(detail) = info.error_detail {
            return Err(Error::Tool(format!(
                "pull '{reference}': {}",
                detail.message.unwrap_or_default()
            )));
        }
    }
    Ok(())
}

/// Splits `name:tag` / `name@sha256:…` the way the pull endpoint wants it, being
/// careful that a registry host may itself carry a port (`host:5000/img`).
fn split_reference(reference: &str) -> (&str, &str) {
    if let Some((name, digest)) = reference.split_once('@') {
        return (name, digest);
    }
    match reference.rsplit_once(':') {
        Some((name, tag)) if !tag.contains('/') => (name, tag),
        _ => (reference, "latest"),
    }
}

fn pinned(reference: &str, repo_digests: Vec<String>, id: Option<&str>) -> String {
    if reference.contains('@') {
        return reference.to_string();
    }
    let (name, _) = split_reference(reference);
    if let Some(digest) = repo_digests.iter().find(|digest| digest.starts_with(name)) {
        tracing::info!(image = %reference, digest = %digest, "pinned image");
        return digest.clone();
    }
    // No repo digest: an image built here and never pushed. Its id is still
    // immutable, and a container can be created from it, so it is a better
    // record of what ran than the tag someone will rebuild tomorrow. What it is
    // not is portable - hence the warning rather than silence.
    match id.filter(|id| id.starts_with("sha256:")) {
        Some(id) => {
            tracing::warn!(
                image = %reference, id = %id,
                "image has no repository digest (built locally?); pinning to its local image id, \
                 which no other machine can resolve"
            );
            id.to_string()
        }
        None => {
            tracing::warn!(
                image = %reference,
                "image has neither a repository digest nor an id; the run is not reproducible \
                 from this reference alone"
            );
            reference.to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reference_splits_the_way_the_registry_expects() {
        assert_eq!(split_reference("python:3.12-slim"), ("python", "3.12-slim"));
        assert_eq!(split_reference("python"), ("python", "latest"));
        assert_eq!(
            split_reference("ghcr.io/me/tasks@sha256:ab"),
            ("ghcr.io/me/tasks", "sha256:ab")
        );
        // A registry port is not a tag.
        assert_eq!(
            split_reference("registry:5000/me/tasks"),
            ("registry:5000/me/tasks", "latest")
        );
    }

    #[test]
    fn a_tag_is_replaced_by_the_digest_it_resolved_to() {
        assert_eq!(
            pinned("python:3.12", vec!["python@sha256:beef".into()], None),
            "python@sha256:beef"
        );
        // Already pinned: left alone, digests of other repositories ignored.
        assert_eq!(
            pinned(
                "python@sha256:beef",
                vec!["other@sha256:dead".into()],
                Some("sha256:cafe")
            ),
            "python@sha256:beef"
        );
        // A locally built image has no repo digest, but it does have an
        // immutable id - which beats a tag that will be rebuilt tomorrow.
        assert_eq!(
            pinned("my-local:dev", Vec::new(), Some("sha256:cafe")),
            "sha256:cafe"
        );
        // Neither: a warning and the tag, never a failure.
        assert_eq!(pinned("my-local:dev", Vec::new(), None), "my-local:dev");
    }
}
