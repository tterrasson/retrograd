//! `GET /v1/config-schema`, and the configuration document in the OpenAPI.

use http::StatusCode;

use crate::support::*;

#[cfg(feature = "openapi")]
#[tokio::test]
async fn the_schema_is_served_standalone_and_marked_by_this_server() {
    let fixture = Fixture::new("config-schema");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, schema) = get(&router, "/v1/config-schema").await;
    assert_eq!(status, StatusCode::OK, "{schema}");
    assert_eq!(
        schema["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert!(
        schema["$id"]
            .as_str()
            .expect("$id")
            .contains(env!("CARGO_PKG_VERSION"))
    );
    assert_eq!(schema["additionalProperties"], false);
    let definitions = &schema["$defs"];
    assert_eq!(
        definitions["PpoToml"]["properties"]["reward_command"]["x-retrograd-server-declared"],
        true
    );
    assert_eq!(
        definitions["TrainingToml"]["properties"]["lr"]["x-retrograd-patchable"],
        true
    );
    assert_eq!(
        schema["properties"]["sft"]["x-retrograd-applies-to"],
        serde_json::json!(["sft"])
    );
    let text = schema.to_string();
    assert!(!text.contains("#/components/"), "every reference is local");
}

#[cfg(feature = "openapi")]
#[tokio::test]
async fn the_three_documents_are_typed_by_the_configuration_component() {
    let fixture = Fixture::new("config-schema-openapi");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, document) = get(&router, "/v1/openapi.json").await;
    assert_eq!(status, StatusCode::OK);
    let schemas = &document["components"]["schemas"];
    assert!(schemas["ConfigDocument"].is_object());
    for (component, field) in [
        ("PlanRequest", "config"),
        ("PlanResponse", "effective_config"),
        ("RunView", "effective_config"),
    ] {
        let property = schemas[component]["properties"][field].to_string();
        assert!(
            property.contains("#/components/schemas/ConfigDocument"),
            "{component}.{field} is not typed: {property}"
        );
    }
    assert!(schemas["Problem"].is_object());
    assert!(schemas["FieldError"].is_object());
    assert_eq!(
        document["paths"]["/v1/runs"]["get"]["responses"]["default"]["content"]["application/problem+json"]
            ["schema"]["$ref"],
        "#/components/schemas/Problem"
    );
}

#[cfg(not(feature = "openapi"))]
#[tokio::test]
async fn without_the_derived_schemas_the_route_says_it_is_not_built() {
    let fixture = Fixture::new("config-schema-absent");
    let router = router_for(&fixture, FakeEngine::succeeding());
    let (status, body) = get(&router, "/v1/config-schema").await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["type"]
            .as_str()
            .expect("type")
            .ends_with("not-implemented")
    );
}
