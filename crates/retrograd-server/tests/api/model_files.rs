//! `GET /v1/model-files`: the GGUF files a recipe may name.

use http::StatusCode;

use crate::support::*;

#[tokio::test]
async fn the_files_under_the_roots_are_listed_with_the_path_a_recipe_takes() {
    let fixture = Fixture::new("model-files");
    std::fs::create_dir_all(fixture.dir.join("vision")).expect("dir");
    std::fs::write(fixture.dir.join("vision/mmproj-f16.gguf"), b"x").expect("file");
    let root = fixture.dir.canonicalize().expect("canonical root");
    let router = build_router(state_with(
        &fixture,
        FakeEngine::succeeding(),
        &format!(
            "calibrate_runs = false\npath_roots = [\"{}\"]\n",
            root.display()
        ),
    ));

    let (status, body) = get(&router, "/v1/model-files").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["roots"][0], root.display().to_string());
    assert_eq!(body["truncated"], false);
    let files = body["files"].as_array().expect("files");
    let model = files
        .iter()
        .find(|file| file["relative"] == "model.gguf")
        .expect("the fixture's model");
    assert_eq!(model["role"], "model");
    assert_eq!(model["path"], root.join("model.gguf").display().to_string());
    assert!(model["bytes"].as_u64().expect("bytes") > 0);
    let projector = files
        .iter()
        .find(|file| file["relative"] == "vision/mmproj-f16.gguf")
        .expect("the projector");
    assert_eq!(projector["role"], "projector");

    // Cached: a file added now is not seen until the listing is refreshed.
    std::fs::write(fixture.dir.join("late.gguf"), b"x").expect("file");
    let (_, body) = get(&router, "/v1/model-files").await;
    assert!(!body.to_string().contains("late.gguf"), "{body}");
    let (_, body) = get(&router, "/v1/model-files?refresh=true").await;
    assert!(body.to_string().contains("late.gguf"), "{body}");

    let (status, _) = get(&router, "/v1/model-files?deep=true").await;
    assert!(status.is_client_error(), "an unknown parameter is refused");
}
