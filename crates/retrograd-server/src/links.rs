//! Signed download links.
//!
//! A browser cannot put an `Authorization` header on a navigation, and a file of
//! several hundred megabytes cannot go through `fetch` into memory first. So an
//! authenticated caller asks for a link (`POST …/artifacts/{name}/link`), and the
//! link itself - for that one artefact, for a minute - stands in for the token.
//!
//! The key is drawn at startup and never written anywhere: a restart kills every
//! outstanding link, which is the intended lifetime of something that lasts a
//! minute anyway.

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

/// How long a link stays valid.
pub const LINK_TTL_SECONDS: u64 = 60;

/// The signing key of this process.
#[derive(Clone)]
pub struct LinkKey([u8; 32]);

impl std::fmt::Debug for LinkKey {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("LinkKey(..)")
    }
}

impl LinkKey {
    /// A fresh key from the operating system's generator.
    pub fn generate() -> Self {
        let mut key = [0u8; 32];
        // The OS generator failing is a machine that cannot do TLS either;
        // there is no degraded mode worth offering for it.
        getrandom::fill(&mut key).expect("the operating system's random generator is available");
        Self(key)
    }

    #[cfg(test)]
    pub fn fixed(byte: u8) -> Self {
        Self([byte; 32])
    }

    /// The signature of a link to `name` of run `run`, valid until `expires`.
    pub fn sign(&self, run: &str, name: &str, expires: u64) -> String {
        hex(&self.mac(run, name, expires).finalize().into_bytes())
    }

    /// Whether `signature` is this key's signature of that link and the link
    /// has not expired at `now`. The comparison is the MAC's own, in constant
    /// time.
    pub fn verify(&self, run: &str, name: &str, expires: u64, signature: &str, now: u64) -> bool {
        if now > expires {
            return false;
        }
        let Some(bytes) = unhex(signature) else {
            return false;
        };
        self.mac(run, name, expires).verify_slice(&bytes).is_ok()
    }

    fn mac(&self, run: &str, name: &str, expires: u64) -> Hmac<Sha256> {
        let mut mac = <Hmac<Sha256> as KeyInit>::new_from_slice(&self.0)
            .expect("HMAC accepts any key length");
        mac.update(format!("{run}\n{name}\n{expires}").as_bytes());
        mac
    }
}

/// The run and artefact a path names, when it is exactly
/// `/v1/runs/{id}/artifacts/{name}` - the one route a signature opens.
pub fn artifact_route(path: &str) -> Option<(&str, &str)> {
    let rest = path.strip_prefix("/v1/runs/")?;
    let (run, rest) = rest.split_once('/')?;
    let name = rest.strip_prefix("artifacts/")?;
    (!run.is_empty() && !name.is_empty() && !name.contains('/')).then_some((run, name))
}

/// `sig` and `exp` out of a query string.
pub fn signature_of(query: &str) -> Option<(&str, u64)> {
    let mut signature = None;
    let mut expires = None;
    for pair in query.split('&') {
        match pair.split_once('=') {
            Some(("sig", value)) => signature = Some(value),
            Some(("exp", value)) => expires = value.parse().ok(),
            _ => {}
        }
    }
    Some((signature?, expires?))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn unhex(text: &str) -> Option<Vec<u8>> {
    if !text.len().is_multiple_of(2) {
        return None;
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(text.get(index..index + 2)?, 16).ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_signature_opens_its_own_link_until_it_expires() {
        let key = LinkKey::fixed(7);
        let signature = key.sign("run", "adapter", 100);
        assert!(key.verify("run", "adapter", 100, &signature, 100));
        assert!(!key.verify("run", "adapter", 100, &signature, 101));
        assert!(!key.verify("run", "checkpoints", 100, &signature, 50));
        assert!(!key.verify("other", "adapter", 100, &signature, 50));
        assert!(!key.verify("run", "adapter", 200, &signature, 50));
        assert!(!LinkKey::fixed(8).verify("run", "adapter", 100, &signature, 50));
        assert!(!key.verify("run", "adapter", 100, "zz", 50));
    }

    #[test]
    fn only_the_download_route_is_recognised() {
        assert_eq!(
            artifact_route("/v1/runs/abc/artifacts/adapter"),
            Some(("abc", "adapter"))
        );
        assert_eq!(artifact_route("/v1/runs/abc/artifacts/adapter/link"), None);
        assert_eq!(artifact_route("/v1/runs/abc/artifacts"), None);
        assert_eq!(artifact_route("/v1/runs/abc"), None);
        assert_eq!(artifact_route("/v1/datasets/abc/artifacts/x"), None);
    }

    #[test]
    fn the_query_must_carry_both_halves() {
        assert_eq!(signature_of("sig=ab&exp=12"), Some(("ab", 12)));
        assert_eq!(signature_of("exp=12&x=1&sig=ab"), Some(("ab", 12)));
        assert_eq!(signature_of("sig=ab"), None);
        assert_eq!(signature_of("sig=ab&exp=soon"), None);
    }
}
