# Agent notes

## Tests

Do not run raw `cargo test` / `pytest` invocations from scratch - use the
lane scripts under `scripts/test-*.sh`. Use the fast lanes on every change and
the model/GPU lanes before a PR when their component is affected.

The PyO3 package in `python/native` is named `retrograd-python` (its library is
`_native`): exclude it from a workspace command with
`--exclude retrograd-python`.

Touching a RIR kernel, schedule or emitter: `scripts/test-rir-parity.sh` is the
one to run while working - the generated shaders against the Loop IR oracle, in
release, no model, about a minute warm. `scripts/test-fast-rust.sh` compiles
those two binaries but no longer runs them.

Before changing a policy in the registry, run `scripts/test-rir.sh`. It is the
lane that decides whether a generated kernel may replace a hand-written one -
correctness *and* timing against the native kernel, on the same shapes, without
a model.

While shaping a kernel or a schedule, the short loop is
`RIR_TIME=1 cargo test --release -p rir-runtime --test device_timing --
--nocapture`: the generated shader timed alone, in seconds, with the schedule
written in the test. It compares RIR to RIR - it never decides a promotion.

When the question is *which* geometry rather than what one costs, the offline
search is `cargo run --release -p rir-sweep -- --kernel <name>`: the same
emit-and-time loop over a small product of candidates, refusing a gain inside
the measured noise and a shape rule that would claim a form the candidate loses
Its last block is a proposal to read and commit by hand; it decides no more
than the loop above does.

## Errors

**`thiserror`, always.** A new error type derives `thiserror::Error` and carries
its message in `#[error("…")]` - never a hand-written `impl Display` plus an
empty `impl std::error::Error`. That pairing is how fifteen types came to spell
the same thing fifteen ways.

Two façades, one per chain, because the two chains share no dependency and must
not start:

- **RIR** - `rir_gen::GenError`. It sits above `LowerError`, `EmitError`,
  `ScheduleError` and `RegistryError`, and each of them reaches it through
  `#[from]`. A `rir-*` crate never depends on `retrograd-core` - checked, not
  just written: `crates/rir-gen/tests/chain_boundary.rs` reads `cargo tree` for
  the six crates of the chain. A schema both chains need goes in `rir-core`
  (`manifest`, `catalog`), which `retrograd-core` may read.
- **applicative** - `retrograd_core::Error`, with one `From` per crate.

A `From` **into** `retrograd_core::Error` is written in the crate that owns the
source error, never in `retrograd-core`: that crate sits at the bottom of the
graph and knows nothing of the agent, the dataset or the plan. The orphan rule
allows it (`impl From<LocalError> for ForeignError`), and it keeps the
translation next to the type being translated - see
`retrograd-agent-core/src/error.rs`. Corollary: never write
`.map_err(|error| Error::invalid(error.to_string()))` at a boundary. If there is
no `From`, the `From` is what is missing.

Picking a variant of `retrograd_core::Error` is picking the sentence the user
reads: `Config` for a document, `Overflow` for arithmetic, `Tokenize`,
`Dataset { path, line }`, `Checkpoint`, and `InvalidArgument` for something a
caller actually passed. `is_user_error()` is what frontends branch on (422 vs
500, `ValueError` vs the module's exception), so a misplaced variant is a wrong
status code and not only a wrong word.

A wrapping variant gets `#[from]` **only when the conversion carries nothing**.
Where it does carry something - `From<MergeError> for ResolveError` turns a JSON
Pointer into the dotted path the API publishes - the hand-written `impl From`
*is* the design and stays. Deleting it would be deleting the translation, not
the boilerplate.

## Numeric conversions

Before adding or "cleaning up" an `as`, classify it. Widening is fine and is
most of them; a narrowing needs either a proof written beside it or a
treatment chosen by what the value feeds - typed error for a contract,
saturation for a budget. A bit pattern is not a conversion.
