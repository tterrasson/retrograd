# Contributing to Retrograd

This page collects the rules that matter when changing the project. The
reference documentation describes the configuration options and the available
commands; here, the goal is to pick the right scope and the right validation.

## Before you change anything

- Keep changes focused. A behavior change must update its test and its
  reference documentation.
- Use domain types and errors. A user error must stay identifiable by the
  frontends (CLI, HTTP and Python) rather than collapse into a generic string.
- Avoid adding a runtime dependency on a data structure or a configuration
  file. Declarations stay readable without starting a container, a server or a
  network client.
- Do not duplicate the formats exchanged between crates: a serialized schema
  has a single typed definition, with a single version.

## Architecture rules

These rules hold across the workspace, and most of them are checked by a test
that fails when they stop holding.

### A shared schema is written once

When two crates exchange a structured document, the schema lives in a single
crate, with real types (`enum`, not strings parsed back on the fly) and `serde`
derives. The producer builds a typed value and then serializes it; JSON is never
assembled by string concatenation. A schema version is one constant, not a
literal copied at every emission site. A type declared in another crate that
needs a reader-side method gets a trait, since Rust forbids an inherent `impl`
on a foreign type.

### Declarations do not depend on execution

Reading a document never requires being able to run it. Declarative types - what
a document says - live in a leaf crate (`retrograd-spec`) with no transport. The
code that executes a declaration (connect, launch a container, open a socket) is
an extension trait implemented by the crate that owns the transport, never the
other way around. An optional transport stays behind a Cargo feature, and
`scripts/test-fast-rust.sh` inspects the dependency graph so a transport cannot
silently creep back in.

### Two chains, two error façades

The RIR crates (`rir-*`) and the applicative crates share no dependency: no
`rir-*` crate depends on `retrograd-core`, which
`crates/rir-gen/tests/chain_boundary.rs` checks. A schema both chains need, such
as the kernel catalogue, lives in `rir-core`, which `retrograd-core` may read.
Each chain has one façade: `rir_gen::GenError` and `retrograd_core::Error`.

`retrograd_core::Error` has one variant per class of message a user reads:
`Config`, `Overflow`, `Tokenize`, `Dataset { path, line }`, `Checkpoint`,
`Runtime`, `InvalidArgument` and `Io`. Choosing a variant is choosing the
sentence the caller reads, and `is_user_error()` is what frontends branch on
(422 against 500 over HTTP, `ValueError` against the module's exception in
Python), so a misplaced variant is a wrong status code.

New error types use `thiserror` and carry their message in `#[error("…")]`. A
`From` into `retrograd_core::Error` is written in the crate that owns the source
error, never in `retrograd-core`. A `.map_err(|error| Error::invalid(error.to_string()))`
at a boundary means that `From` is missing. A validation returns a typed error
the caller can match on, not `Result<(), String>`.

### One vocabulary, one spelling

An enum whose variants are also a wire form (a `serde` rename, a catalogue
string, a URL segment) is declared once and derives its text forms (`ALL`,
`as_str()`) rather than being copied into parallel lists kept in step by a test.
Versions and dependencies shared by several members are declared once, in
`[workspace.package]` and `[workspace.dependencies]`; each crate keeps the
features it actually requests.

## Numbers

For numeric conversions, the intent must be explicit: widen before a
multiplication; clamp a value before narrowing it; document a bit
reinterpretation; return a typed error when an unbounded size comes from user
input. A silent truncation is never a capacity check. The full convention is in
[numeric conversions](./CONVERSIONS).

## Comments

1. A comment says **why**: an invariant, a contract, a measured choice. It does
   not paraphrase the code.
2. **No history.** "Before this table…", "used to", "replaces the old…" belong
   in the commit message; the comment keeps the rule in force.
3. **No internal plan identifiers** (`C3.7`, `D6.3`, `O8`, `§R0`, a plan name).
   If the invariant matters, write the sentence; if it has a public page, link
   it.
4. `retro delta:` is reserved for hunks of the llama.cpp fork, where it marks
   every fork change kept in an upstream file. It has no meaning in the
   workspace's Rust.
5. Rustdoc: one summary sentence, an empty `///` line, then the details.
   `# Errors`, `# Panics` and `# Safety` sections where they are true.
6. A measurement that justifies a default fits in one sentence and an order of
   magnitude. The table of measurements goes in the documentation.
7. Keep `SAFETY:` comments, the proofs beside narrowing conversions, and the
   justification of every `allow`.

## RIR and GPU backends

A generated RIR kernel replaces a native kernel only when both conditions hold
on the same hardware and the targeted shapes:

1. the result is correct;
2. it is no slower than the native kernel.

Day-to-day work uses RIR parity: it compares the generated shader against the
Loop IR oracle without loading a model. A geometry search can propose a
candidate, but only the promotion lane decides a policy change. See [the kernel
families](./rir/KERNELS), [the support matrix](./SUPPORT) and [the
tests](./tests/notice) before changing a schedule, an emitter or a registry.

## Where to document a change

- An option, a default or a command: the [Reference](/reference/configuration)
  or [CLI](/reference/cli) documentation.
- A training behavior: the guide of that algorithm under [Training](/training/sft).
- An HTTP contract: [server errors](./server/ERRORS).
- A hardware compatibility state: [support](./SUPPORT) or
  [CUDA](./cuda/STATUS).
- A validation or integration procedure: [tests](./tests/notice) or [llama.cpp
  fork](./LLAMA_CPP_FORK_WORKFLOW).
