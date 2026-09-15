//! What a configuration file says, without anything that acts on it.
//!
//! One crate for the *declarations* the whole agentic stack shares - the
//! environment, the judge, the tool plan - so that reading a configuration and
//! running one are two different dependencies. `retrograd-config` and
//! `retrograd-plan` link this and stop there; `retrograd-env`, `retrograd-judge`
//! and `retrograd-tools` link it too, and each adds what only it can do.
//!
//! The rule for what belongs here: a type whose meaning is entirely in the file
//! it was read from. A `build()` that opens a socket, a `resolve()` that reads
//! the disk and a `tools()` that consults a registry all stay with the executant
//! - as an extension trait, so the call sites do not move either.

pub mod env;
pub mod judge;
pub mod tools;
