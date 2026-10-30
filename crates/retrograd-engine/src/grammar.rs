//! Grammar handles for constrained decoding: the runtime's JSON Schema ->
//! GBNF converter (already compiled into the C++ runtime) exposed as free
//! functions returning the grammar text, so a caller can hold the GBNF
//! (cheap to clone, e.g. in [`retrograd_core::SamplingParamsV2`]) and let the
//! runtime compile it once per request and choice.

use retrograd_ffi as ffi;

use crate::{Error, Result};

/// Converts a JSON Schema (as a JSON document string) to GBNF. The schema is
/// the caller's responsibility to have validated against the supported
/// subset; anything the converter rejects is an `Err` naming the problem.
pub fn grammar_from_json_schema(schema_json: &str) -> Result<String> {
    ffi::Grammar::from_json_schema(schema_json)
        .map(|grammar| grammar.gbnf())
        .map_err(Error::invalid)
}

/// The built-in any-JSON grammar (what `response_format: json_object` asks
/// for), as GBNF text.
pub fn json_object_grammar() -> Result<String> {
    Ok(ffi::Grammar::json_object().gbnf())
}

/// Owns a nonempty GBNF document. The runtime compiles it against the model
/// vocabulary before decoding the first sampling row.
pub fn grammar_from_gbnf(gbnf: &str) -> Result<String> {
    ffi::Grammar::from_gbnf(gbnf)
        .map(|grammar| grammar.gbnf())
        .map_err(Error::invalid)
}
