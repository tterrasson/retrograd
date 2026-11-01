//! The `/v1/systemone` answer, as llama-server writes it: `model`, `answers`
//! keyed in the order of the questions, `usage`. The request body has no type
//! here: it is read as an ordered [`Json`](retrograd_decision::Json) - the
//! key order of its `state` is part of the prompt - and validated by
//! `retrograd-decision` against the readout of the loaded model.

use indexmap::IndexMap;
use retrograd_decision::answer::Answer;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemOneResponse {
    /// The model id the request named, or the only one served.
    pub model: String,
    pub answers: IndexMap<String, Answer>,
    pub usage: SystemOneUsage,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct SystemOneUsage {
    /// Tokens evaluated, every variant of every question counted.
    pub input_tokens: u64,
    /// Always 0: a decision model generates nothing.
    pub output_tokens: u64,
}
