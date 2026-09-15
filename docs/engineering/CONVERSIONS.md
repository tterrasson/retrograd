# Numeric conversion convention

Rule that applies to the whole repository and replaces the question "should
this `as` be converted?" with a classification into four cases.

## The finding the rule is based on

A count of casts (`as`) in production measures a **density**, not a risk. A
sort of the densest files surfaces only a handful of genuinely dangerous
cases. The reason is that the repository already applies, without having
written it down, the two patterns that make a conversion safe:

- **widen before multiplying** - converting each dimension to `u64` *before*
  the byte product is the defense against overflow;
- **bound before narrowing** - `value.clamp(1, u32::MAX as u64) as u32`,
  where the cast can no longer lose anything.

A "replace every `as`" pass would therefore convert hundreds of safe lines
while finding nothing. What this document replaces is that pass.

## The four cases

### 1. Widening - do nothing

`u32 → u64`, `u32 → usize`, integer → `f32`/`f64`. Total on 64-bit targets,
so `as` is the most readable form here. Converting it to `try_into()` adds a
`?` on an unreachable error path.

This is **almost all** of the casts in the repository: registry indices,
cost or metric widenings.

Corollary not to lose sight of: if a product must fit in a wide type, the
widening must come **before** the operator, never after. `(a * b) as u64`
and `a as u64 * b as u64` do not overflow at the same place.

### 2. Narrowing whose proof is in scope - comment it, don't convert it

An `as u32` preceded by a `clamp`, or a guard that already rejects the
too-large value. Converting it to `try_from` would produce a dead error
path, and worse: a `try_from` guard placed *before* the business guard would
reject the input while naming the wrong reason.

The rule is therefore to write the proof next to the cast, naming **the
line** that carries it. Example:

```rust
// `group_size <= training.n_batch` is checked ten lines above and
// `n_batch` is a u32: the cast cannot lose bits.
training.n_seq_max = group_size as u32;
```

Prefer, when possible, the form that makes the proof unnecessary -
`clamp(1, u32::MAX as u64) as u32` reads correctly without going back up
into the function.

### 3. Bit pattern - never convert, document

A quantized byte split into nibbles, a `wrapping_add` seed, an IEEE-754
field reassembled by hand. Here `as i8` / `as u32` **is** the operation: it
reinterprets bits, it measures nothing. A checked conversion here would be a
non sequitur, and an overflow here is the expected behavior.

Rule: *convert what carries a size, document what carries a bit pattern.*

### 4. Narrowing without proof - the only case to handle

A size, or a product of sizes, coming from user input and placed in a
32-bit field without an upper bound.

The treatment depends on what the value feeds, **not** on the type it
takes:

| what the value feeds | treatment | example in place |
|---|---|---|
| a geometry or thread contract | typed error, at the boundary, before any read of the field | `ScheduleError::Malformed` (`schedule/mod.rs::check_shape`) |
| the emission of an identifier | `IrId::at` - documented panic, `try_at` to test it | `rir-core/src/ids.rs` |
| a cost or capacity estimate | **deliberate** saturation, in the direction that refuses | `retrograd_core::saturating_dim` |

The third row is the one whose interest lies in the direction: a truncation
turns an impossible job into a small job, and the downstream budget accepts
it. The helper keeps the value in the cost's `u64` arithmetic and only
saturates at `u64::MAX`; it does not cap prematurely at the width of the
original field. The budget thus rejects the real value, or the largest
value it can represent, without threading a `Result` through the cost
model.

## Where the helpers live

The repository's two chains are disjoint and stay that way:

- RIR chain - `rir_core::IrId::at` / `try_at` for identifiers,
  `ScheduleError::Malformed` for geometry.
- applicative chain - `retrograd_core::{saturating_dim, saturating_dim_product}`
  (`crates/retrograd-core/src/dims.rs`).

No shared crate for two functions: creating one would create a dependency
arc between two chains that have none, which costs more than duplicating
one line.

## What the rule refuses

- A blanket pass over all `as`. Case 1 is the majority, and converting it
  would make reviewing cases 2 through 4 impossible.
- A `#![deny(clippy::cast_possible_truncation)]` lint. It fires on case 3
  (where the narrowing is the operation) as much as on case 4, so it would
  end up as a rain of `allow`s - that is, the disappearance of the signal
  it was meant to provide.
- A case-4 cast left untreated on the grounds that "no real input reaches
  it". That's true of all of them, and it's why they survive.
