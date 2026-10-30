//! Where a stored response lives: the [`ResponseStore`] trait, with a bounded
//! in-memory LRU as the default and a directory of one JSON per response as
//! the on-disk option. `store: false` writes nothing, and a
//! `previous_response_id` that points at an unknown or evicted response is a
//! 404 in the OpenAI envelope whose message says it may have been evicted.
//!
//! Empty until the stateful surface is served; the trait is what the routes
//! will be written against.

/// A stored response and the input items a continuation replays.
pub trait ResponseStore: Send + Sync {
    /// Stores one response and returns its id.
    fn put(&self, response: StoredResponse) -> Result<ResponseId, std::io::Error>;
    /// The response `id` names, or `None` when it is unknown or was evicted.
    fn get(&self, id: &ResponseId) -> Option<StoredResponse>;
    /// Removes one response, releasing any pin its descendants held.
    fn delete(&self, id: &ResponseId) -> bool;
    /// The input items of one response, for `GET /v1/responses/{id}/input_items`.
    fn input_items(&self, id: &ResponseId) -> Option<Vec<serde_json::Value>>;
}

/// The `resp_…` id of one stored response.
pub struct ResponseId(pub String);

/// One stored response: the object the API returns, plus the input items a
/// `previous_response_id` continuation replays.
pub struct StoredResponse {
    pub response: serde_json::Value,
    pub input_items: Vec<serde_json::Value>,
}
