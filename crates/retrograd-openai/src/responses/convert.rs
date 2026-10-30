//! From the Responses `input` items to the common
//! [`GenerationRequest`](crate::request::GenerationRequest): `message` items
//! (system/developer/user/assistant, `input_text` and `output_text` parts),
//! re-rendered `reasoning` items, and `function_call` /
//! `function_call_output` items reconciled by their `call_id`.
//!
//! Empty until the surface is served. Anything but text - an image, a file -
//! is refused with the field named, never dropped, exactly as the chat
//! conversion does.
