//! The Responses API surface: `POST /v1/responses` and, once a response can be
//! stored, its `GET`/`DELETE` and `input_items` routes.
//!
//! Nothing is served from here yet - the routes are mounted when the surface
//! lands. What is already decided: it translates its `input` items into the
//! same [`GenerationRequest`](crate::request::GenerationRequest) the chat
//! surface produces ([`convert`]), and it branches on the same backend - the
//! [`Session`](crate::session::Session) for weights on disk, a live run's own
//! thread for a run that is training. Its wire documents are [`wire`], its
//! streamed answer is a typed sequence of [`events`], and a stored response
//! lives in a [`store`].

pub mod convert;
pub mod events;
pub mod store;
pub mod wire;
