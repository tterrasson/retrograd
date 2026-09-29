//! The web interface's static files, and how they are served.
//!
//! Everything about the assets lives here - where they come from, their media
//! types, their cache and security headers, and the fallback that lets the
//! single-page application own its URLs - so that the server sees one function,
//! `router`, and nothing else.
//!
//! The resolution of a path, in order:
//!
//! 1. an exact file of the build answers, pre-compressed when the build holds a
//!    `.br` or `.gz` sibling the client accepts;
//! 2. a path with no extension is a route of the application, and gets
//!    `index.html`, whose router reads the URL;
//! 3. anything else is a plain 404 - a missing script must not receive
//!    `index.html`, which the browser would then refuse with a misleading
//!    media-type error.
//!
//! The serving is written against [`Assets`] rather than the embedded build
//! itself, so it is tested on a handful of files in memory without a UI build.

use std::borrow::Cow;
use std::sync::Arc;

use axum::Router;
use axum::body::Body;
use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use http::{HeaderMap, HeaderValue, Method, StatusCode, header};

/// The content security policy of every response.
///
/// `style-src 'unsafe-inline'` is the one concession: the component library
/// writes its theme as CSS variables at run time. Scripts stay `'self'` only,
/// which is what matters against text a model or a dataset produced.
pub const CONTENT_SECURITY_POLICY: &str = "default-src 'self'; script-src 'self'; \
     style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; \
     frame-ancestors 'none'; base-uri 'none'; form-action 'self'";

/// The file the application starts from, and the answer to any of its routes.
const INDEX: &str = "index.html";

/// One file of the build.
#[derive(Clone, Debug)]
pub struct Asset {
    pub bytes: Cow<'static, [u8]>,
    /// SHA-256 of `bytes`, the entity tag.
    pub sha256: [u8; 32],
}

/// Where the files come from: the embedded build, or a test's handful.
pub trait Assets: Send + Sync + 'static {
    /// The file at `path`, relative to the root of the build, without a
    /// leading `/`.
    fn get(&self, path: &str) -> Option<Asset>;
}

/// The router serving `assets`: every method is answered, `GET` and `HEAD`
/// with files and anything else with `405`.
pub fn router_with(assets: impl Assets) -> Router {
    let assets: Arc<dyn Assets> = Arc::new(assets);
    Router::new().fallback(serve).with_state(assets)
}

/// The router serving the build embedded in this binary.
#[cfg(feature = "embed")]
pub fn router() -> Router {
    router_with(embedded::Embedded)
}

#[cfg(feature = "embed")]
mod embedded {
    use super::{Asset, Assets};

    #[derive(rust_embed::Embed)]
    #[folder = "../../web/dist"]
    struct Dist;

    pub(super) struct Embedded;

    impl Assets for Embedded {
        fn get(&self, path: &str) -> Option<Asset> {
            let file = Dist::get(path)?;
            Some(Asset {
                sha256: file.metadata.sha256_hash(),
                bytes: file.data,
            })
        }
    }
}

async fn serve(State(assets): State<Arc<dyn Assets>>, request: Request) -> Response {
    let method = request.method().clone();
    if method != Method::GET && method != Method::HEAD {
        return finish(
            plain(StatusCode::METHOD_NOT_ALLOWED, "method not allowed"),
            None,
        );
    }
    let Some(path) = normalize(request.uri().path()) else {
        return finish(plain(StatusCode::NOT_FOUND, "not found"), None);
    };
    let (path, asset) = match assets.get(&path) {
        Some(asset) => (path, asset),
        None if !has_extension(&path) => match assets.get(INDEX) {
            Some(asset) => (INDEX.to_string(), asset),
            None => return finish(plain(StatusCode::NOT_FOUND, "not found"), None),
        },
        None => return finish(plain(StatusCode::NOT_FOUND, "not found"), None),
    };
    let response = respond(assets.as_ref(), &path, asset, request.headers(), &method);
    finish(response, Some(&path))
}

/// The build-relative path a request names, or `None` for one that tries to
/// leave the build.
fn normalize(path: &str) -> Option<String> {
    let trimmed = path.trim_start_matches('/');
    if trimmed
        .split('/')
        .any(|segment| segment == ".." || segment == "." || segment.contains('\\'))
    {
        return None;
    }
    Some(if trimmed.is_empty() || trimmed.ends_with('/') {
        format!("{trimmed}{INDEX}")
    } else {
        trimmed.to_string()
    })
}

fn has_extension(path: &str) -> bool {
    path.rsplit('/')
        .next()
        .is_some_and(|name| name.contains('.'))
}

fn respond(
    assets: &dyn Assets,
    path: &str,
    asset: Asset,
    headers: &HeaderMap,
    method: &Method,
) -> Response {
    let etag = entity_tag(&asset.sha256);
    if headers
        .get(header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| {
            value
                .split(',')
                .any(|tag| tag.trim() == etag || tag.trim() == "*")
        })
    {
        let mut response = StatusCode::NOT_MODIFIED.into_response();
        insert(&mut response, header::ETAG, &etag);
        return response;
    }

    // A pre-compressed sibling carries the same entity: the tag stays the one
    // of the uncompressed file, and `Vary` tells a cache the two differ.
    let accepted = accepted_encodings(headers);
    let (bytes, encoding) = ["br", "gzip"]
        .into_iter()
        .filter(|encoding| accepted.contains(encoding))
        .find_map(|encoding| {
            let suffix = if encoding == "br" { "br" } else { "gz" };
            assets
                .get(&format!("{path}.{suffix}"))
                .map(|compressed| (compressed.bytes, Some(encoding)))
        })
        .unwrap_or((asset.bytes, None));

    let length = bytes.len();
    let body = if method == Method::HEAD {
        Body::empty()
    } else {
        Body::from(bytes)
    };
    let mut response = Response::new(body);
    insert(&mut response, header::CONTENT_TYPE, media_type(path));
    insert(&mut response, header::CONTENT_LENGTH, &length.to_string());
    insert(&mut response, header::ETAG, &etag);
    insert(&mut response, header::VARY, "Accept-Encoding");
    if let Some(encoding) = encoding {
        insert(&mut response, header::CONTENT_ENCODING, encoding);
    }
    response
}

/// The encodings the client accepts, without the ones it refuses with `q=0`.
fn accepted_encodings(headers: &HeaderMap) -> Vec<&'static str> {
    let Some(value) = headers
        .get(header::ACCEPT_ENCODING)
        .and_then(|value| value.to_str().ok())
    else {
        return Vec::new();
    };
    value
        .split(',')
        .filter_map(|entry| {
            let mut parts = entry.split(';').map(str::trim);
            let name = parts.next()?;
            let refused = parts.any(|parameter| {
                parameter
                    .strip_prefix("q=")
                    .and_then(|q| q.parse::<f32>().ok())
                    .is_some_and(|q| q <= 0.0)
            });
            if refused {
                return None;
            }
            match name {
                "br" => Some("br"),
                "gzip" => Some("gzip"),
                _ => None,
            }
        })
        .collect()
}

/// Headers every response of this router carries, and the cache policy of the
/// file it served, if it served one.
///
/// Files under `assets/` are named by their content hash, so they never change
/// and are cached for a year. Everything else - `index.html` above all - is
/// revalidated on every use: it is what names the current hashes.
fn finish(mut response: Response, path: Option<&str>) -> Response {
    insert(&mut response, header::X_CONTENT_TYPE_OPTIONS, "nosniff");
    insert(&mut response, header::REFERRER_POLICY, "no-referrer");
    insert(
        &mut response,
        header::CONTENT_SECURITY_POLICY,
        CONTENT_SECURITY_POLICY,
    );
    let cache = match path {
        Some(path) if path.starts_with("assets/") => "public, max-age=31536000, immutable",
        Some(_) => "no-cache",
        None => "no-store",
    };
    insert(&mut response, header::CACHE_CONTROL, cache);
    response
}

fn plain(status: StatusCode, message: &'static str) -> Response {
    (
        status,
        [(header::CONTENT_TYPE, "text/plain; charset=utf-8")],
        message,
    )
        .into_response()
}

fn insert(response: &mut Response, name: header::HeaderName, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        response.headers_mut().insert(name, value);
    }
}

fn entity_tag(sha256: &[u8; 32]) -> String {
    let mut tag = String::with_capacity(66);
    tag.push('"');
    for byte in sha256 {
        tag.push_str(&format!("{byte:02x}"));
    }
    tag.push('"');
    tag
}

/// The media type of a file of the build, by extension. The build produces a
/// short, known list; anything else is served as bytes, and `nosniff` keeps the
/// browser from guessing better.
fn media_type(path: &str) -> &'static str {
    let extension = path.rsplit('.').next().unwrap_or_default();
    match extension {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" | "map" => "application/json",
        "webmanifest" => "application/manifest+json",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "woff" => "font/woff",
        "ttf" => "font/ttf",
        "txt" => "text/plain; charset=utf-8",
        "wasm" => "application/wasm",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;

    struct Memory(BTreeMap<&'static str, &'static [u8]>);

    impl Assets for Memory {
        fn get(&self, path: &str) -> Option<Asset> {
            let bytes = self.0.get(path)?;
            // Distinct per file, which is all a test tag needs to be.
            let mut sha256 = [0u8; 32];
            sha256[0] = u8::try_from(bytes.len()).unwrap_or(u8::MAX);
            sha256[1] = u8::try_from(path.len()).unwrap_or(u8::MAX);
            Some(Asset {
                bytes: Cow::Borrowed(bytes),
                sha256,
            })
        }
    }

    fn app() -> Router {
        router_with(Memory(BTreeMap::from([
            ("index.html", b"<html>app</html>".as_slice()),
            ("assets/app-1a2b.js", b"console.log(1)".as_slice()),
            ("assets/app-1a2b.js.br", b"BR".as_slice()),
            ("assets/app-1a2b.js.gz", b"GZ".as_slice()),
            ("favicon.svg", b"<svg/>".as_slice()),
        ])))
    }

    async fn get(path: &str, headers: &[(&str, &str)]) -> (StatusCode, HeaderMap, Vec<u8>) {
        let mut request = Request::builder().uri(path);
        for (name, value) in headers {
            request = request.header(*name, *value);
        }
        let response = app()
            .oneshot(request.body(Body::empty()).expect("request"))
            .await
            .expect("infallible");
        let status = response.status();
        let headers = response.headers().clone();
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes()
            .to_vec();
        (status, headers, body)
    }

    #[tokio::test]
    async fn the_root_and_every_application_route_answer_the_index() {
        for path in ["/", "/runs", "/runs/abc/trajectories"] {
            let (status, headers, body) = get(path, &[]).await;
            assert_eq!(status, StatusCode::OK, "{path}");
            assert_eq!(body, b"<html>app</html>", "{path}");
            assert_eq!(headers[header::CACHE_CONTROL], "no-cache", "{path}");
            assert!(
                headers[header::CONTENT_TYPE]
                    .to_str()
                    .expect("ascii")
                    .starts_with("text/html")
            );
        }
    }

    #[tokio::test]
    async fn a_missing_file_is_a_plain_404_and_never_the_index() {
        let (status, headers, body) = get("/assets/missing.js", &[]).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert_ne!(body, b"<html>app</html>");
        assert!(
            headers[header::CONTENT_TYPE]
                .to_str()
                .expect("ascii")
                .starts_with("text/plain")
        );
    }

    #[tokio::test]
    async fn hashed_assets_are_immutable_and_served_precompressed_when_accepted() {
        let (status, headers, body) = get("/assets/app-1a2b.js", &[]).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, b"console.log(1)");
        assert_eq!(
            headers[header::CACHE_CONTROL],
            "public, max-age=31536000, immutable"
        );
        assert!(headers.get(header::CONTENT_ENCODING).is_none());

        let (_, headers, body) = get(
            "/assets/app-1a2b.js",
            &[("accept-encoding", "gzip, deflate, br")],
        )
        .await;
        assert_eq!(body, b"BR");
        assert_eq!(headers[header::CONTENT_ENCODING], "br");
        assert_eq!(headers[header::VARY], "Accept-Encoding");

        let (_, headers, body) = get(
            "/assets/app-1a2b.js",
            &[("accept-encoding", "gzip, br;q=0")],
        )
        .await;
        assert_eq!(body, b"GZ");
        assert_eq!(headers[header::CONTENT_ENCODING], "gzip");
    }

    #[tokio::test]
    async fn a_matching_entity_tag_is_answered_not_modified() {
        let (_, headers, _) = get("/", &[]).await;
        let etag = headers[header::ETAG].to_str().expect("ascii").to_string();
        let (status, _, body) = get("/", &[("if-none-match", &etag)]).await;
        assert_eq!(status, StatusCode::NOT_MODIFIED);
        assert!(body.is_empty());
    }

    #[tokio::test]
    async fn every_response_carries_the_security_headers() {
        for path in ["/", "/favicon.svg", "/assets/missing.js"] {
            let (_, headers, _) = get(path, &[]).await;
            assert_eq!(
                headers[header::CONTENT_SECURITY_POLICY],
                CONTENT_SECURITY_POLICY
            );
            assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
            assert_eq!(headers[header::REFERRER_POLICY], "no-referrer");
        }
    }

    #[tokio::test]
    async fn a_path_that_leaves_the_build_is_refused() {
        let (status, _, _) = get("/assets/../index.html", &[]).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn only_reads_are_served() {
        let response = app()
            .oneshot(
                Request::builder()
                    .method(Method::POST)
                    .uri("/")
                    .body(Body::empty())
                    .expect("request"),
            )
            .await
            .expect("infallible");
        assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    }
}
