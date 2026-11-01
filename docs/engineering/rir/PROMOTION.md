# RIR kernel promotion

A generated RIR kernel replaces a hand-written one for an `(op, backend)` pair
only when it is **correct** and **at least as fast** as the native kernel, on
the shapes that matter and on the same hardware. Both halves are proven without
loading a model or launching a training run: the real training graph is a
coverage check, not the work loop.

This page describes the measurements that decide a promotion. The commands and
their flags are listed in [the test lanes](../tests/lanes.md).

## Two measurement levels

The two levels are deliberately distinct in cost, and they answer different
questions. Neither substitutes for the other.

### Level 1 - the shader alone

```sh
RIR_TIME=1 cargo test --release -p rir-runtime --test device_timing -- --nocapture
```

`rir-runtime`'s `tests/device_timing.rs` emits the kernel **in process** from a
`Schedule` written in the test, and times the generated shader on a device
without ggml and without rebuilding the fork. Changing a schedule is a one-line
edit, which makes this the loop to work in while a kernel is being shaped.

Two properties make the number usable:

- `Pipeline::prepare` returns a `Session` that separates upload, dispatch and
  readback, so a timed run measures the shader and its submission, not host
  copies.
- `Session::time` records N runs in a **single** command buffer separated by a
  memory barrier. One submission per iteration hits a fixed submit and
  fence-wait quantum on macOS (about 8 ms) that swamps any shorter kernel.
  The barrier cannot be dropped: the runs share their output buffers, and even
  idempotent writes need write-after-write ordering. `Session::time_stream`
  therefore takes the same barriers, and both measure a serialized latency with
  the submission cost amortized - never an overlapping throughput. Under
  MoltenVK, where the barrier costs more than most kernels, that latency can
  hide the difference between two schedules.

The harness drives Vulkan, CUDA (`--features cuda`) and, on an Apple target,
Metal. The Metal arm compiles the generated `kernel.metal` at run time, at the
language version its manifest claims, and binds it the way the fork does:
tensors at buffer indices `0..n` in manifest order, the params struct right
after them. It records the N runs in one serial compute encoder, which orders
them the way the Vulkan barrier does, so the number means the same thing. On a
Mac this measures MSL natively instead of through MoltenVK.

Level 1 compares RIR to RIR. It never decides a promotion: it has no native
kernel to compare against, and it does not take the ggml path.

### Offline schedule search

```sh
cargo run --release -p rir-sweep -- --kernel rms_norm_back --backend vulkan --reps 9
cargo run --release -p rir-sweep -- --kernel rms_norm --backend metal
cargo run --release -p rir-sweep -- --kernel rms_norm --backend metal --profile qwen3-0.6b-q8_0 --regime decode
cargo run --release -p rir-sweep -- --kernel mul_mat_vec_q4_K --backend metal --candidate mma --write-proposal /tmp/q4_K.proposal
```

`rir-sweep` runs the level-1 loop over a small product of valid candidates and
prints a proposal in `schedules_for`'s vocabulary. It refuses a candidate:

- whose gain is inside the measured noise - each candidate is compared with its
  own relative spread `(max - min) / median` over `reps` repetitions, added to
  the reference's spread;
- whose derived `ShapeRule` does not exclude every shape it loses;
- whose output bytes differ from the variant it would replace beyond
  `sqrt(n) * 8 * eps`, checked before its time is read: a kernel that is wrong
  and fast is never a proposal.

Nothing is written into the source. The last block of a sweep is a proposal
to review and commit by hand, and a table change then goes through the
promotion lane like any other. `--write-proposal PATH` gathers the retained
rows of every sweep of the run - a family with a geometry space proposes
`Pick`s - into `PATH`, and refuses a path that holds anything but an earlier
proposal.

A family with a geometry space (`rir_lower::schedule::spaces`) draws its
candidates from it, so a proposal is written in the vocabulary of the picks it
would join. The GEMV's own grid crosses 1, 2, 4, 8, 16 and 32 tokens with
depths and row counts of 1024, 4096 and 14336; against a space of a few
hundred geometries, `--candidate` narrows a run to the question at hand.

The loop end to end: `rir-sweep` (RIR against RIR) proposes picks; a human
pastes them into the table and regenerates with `rir-gen`;
`scripts/test-rir.sh --emit-ratios` measures them against the native kernel;
`rir-gen promote` proposes the per-variant policies; a human commits.

`--profile` derives the shapes from a reference model's dimensions in
`rir_core::profiles`: hidden and head norms, FFN activations, residuals and
weight projections (`ModelShape::matmul_shapes`: Q, K, V, output, gate, up,
down). Decode covers 1, 4, 8, 16 and 32 tokens; prefill covers
512. Without `--regime`, the two are reported separately. The catalogue
publishes the profiles as measurement geometries. They carry no selections;
dispatch choices remain per-variant claims and policies.

### Level 2 - the promotion lane

```sh
scripts/test-rir.sh
```

The fork's `test-backend-ops` is the only harness that can build a *view* into
a packed tensor, which is the shape a real graph sends. It already knows how to
check correctness and to time, and the RIR mode is an environment variable, so
the same matrix run twice gives the native and RIR comparison on exactly the
same shapes. The lane:

1. **discovers the pairs** from the generated registry: every non-`NATIVE_ONLY`
   Metal or Vulkan pair. Adding an op to the registry adds it to the lane;
2. **builds the fork** if needed;
3. **runs the matrix twice** per pair, native then RIR, on the same binary. A
   pair still in `observe_generated` is promoted for the process by
   `RETRO_RIR_TEST_PREFER`, without which nothing would encode it - and so is a
   pair whose variants do not all share one policy, so every variant of it can
   be measured (the elevation applies to the rows as well as to the pair);
4. **checks the per-site counters**: `rir > 0`, and every native fallback must
   be explained by a portable-contract rejection the registry declares for that
   pair. A device rejection, or a fallback nobody declared, fails the lane;
5. **times both paths** `--repeat` times, alternating native and RIR on every
   pass, and rules on the **median** of the ratios, with `min-max` published
   beside it.

The ratio is a verdict for a `prefer_generated` pair and an open work item for
an `observe_generated` one - except for a pair whose variants do not all share
one policy, whose verdict is made per variant at the end of the lane (see
below). A shape whose spread covers the whole distance to the native kernel
prints `parity`: neither a gain to claim nor a regression to fail on.

Three properties the script enforces on itself: MoltenVK is configured by
detection, a requested device that is missing fails the lane instead of
skipping it, and every run is bounded by a watchdog, because a candidate kernel
can be a hundred times slower and "too slow to measure" is a verdict.

### Pairs scoped to one regime

A dispatch site may offer RIR only part of an op. `MUL_MAT` is the case: the
Metal and Vulkan sites offer a node only when `src1` has at most
`RIR_DECODE_MAX_TOKENS` columns (the decode regime, published in the generated
registry header from `rir_core::catalog::Regime`) and when the backend fuses
nothing into it. A prefill or fused node never reaches the chain, so it is
neither counted nor declared.

The lane follows the same scope when it times the pair. Upstream's perf matrix
times `MUL_MAT` over every format and up to 512 columns; the lane passes
`test-backend-ops` a `-p` filter that keeps the decode steps in the formats the
registry's variants claim, so no row compares the native kernel with itself. The
decode cases themselves are the projections of a profiled model layer at the
step sizes decode is profiled at. Every timed shape carries its regime, and a
second table gives the worst median ratio per pair and regime: a promotion
in one regime rests on that number and says nothing about the other.

The lane runs `repeat x 2` matrices per pair. Aim a high `--repeat` at the pair
being promoted (`--op`), not at the whole lane.

## Why the verdict is a median

One run is a draw, not a measurement. Replayed four times in a row without a
rebuild, a single-sample lane gave `RMS_NORM_BACK` ratios of 1.12, 0.99, 1.08
and 1.01 on the same pair: a 12 % spread against a 5 % tolerance, so a pair at
parity would be promoted or refused at random. On the median of five passes the
verdict settles on one reproducible shape.

For the same reason, a ratio never leaves its session. Machine drift of tens of
percent has been measured between two days on the same native kernel and shape.
To compare two RIR versions, run the lane twice in one session and check that
the native column is stable across both.

## Acceptance criterion

An `(op, backend)` pair moves to `prefer_generated` when, and only when:

- the ggml semantics are exact, aliasing and parameters included;
- `scripts/test-rir.sh` is green for the pair: native matrix OK, RIR matrix OK,
  `rir > 0`, and every native fallback explained by a declared restriction;
- the ratio table shows no shape where RIR is slower than native beyond the
  tolerance;
- the generated shader's device parity against the Loop IR oracle is green
  (`scripts/test-rir-parity.sh`);
- `require` fails cleanly on a shape outside the contract.

## Promotion is per variant, under the pair's ceiling

A pair's policy in `rir_op_policies` is only a **ceiling**. Every variant row
also carries its own `policy` (`rir_variant_desc.policy`), read from
`crates/rir-kernels/policy/variants.tsv` at generation time: the effective policy
at dispatch is `min(pair, variant)`, so a variant held at `observe_generated` on
a `prefer_generated` pair is registered, measured and counted but never encodes
- the node passes to the next candidate, then to the native kernel. A variant
absent from the file carries the pair's policy, which is what makes adding the
file a no-op for the pairs it does not name. `rir-gen` refuses a row that names
an unknown variant, sits above its pair's ceiling, or belongs to an op with no
attribution adapter.

A variant moves to `prefer_generated` under the same criterion as a pair, but
applied to the bucket: **at least one measured shape attributed to it, and a
median ratio at or under the tolerance on every one of them**.

- The lane writes the per-shape ratios with `--emit-ratios PATH`, committed
  under `generated/rir/perf/<session>.tsv` when a promotion rests on them:
  `session  pair  regime  label  nat_us  rir_us  median  min  max  passes
  verdict`. `scripts/rir-domain-baseline.tsv` keeps its own role (coverage).
- `cargo run -p rir-gen -- promote --ratios <file>` attributes each measured
  shape to the variant that served it - the same selection the dispatcher ran,
  replayed in Rust: same `src0` dtype claim, same shape claim, same priority
  rule - and prints the proposed `variants.tsv` diff (`prefer_generated` when
  the criterion holds, `observe_generated` otherwise, never above the pair's
  ceiling). `--write` applies it to the file; the commit stays human.
- `promote --check` is called at the end of `test-rir.sh` for every pair whose
  variants do not all share one policy, and fails the lane when a variant at
  `prefer_generated` has an attributed shape beyond the tolerance. For those
  pairs it replaces the per-pair verdict.

## Retiring a native kernel

Keeping both paths has a cost: a native kernel that never runs is not a safety
net, it is code that diverges. Retiring one additionally requires that
`supports_op` rests on the RIR contract for the pair, and that a real training
graph has run with no fallback on it - which is what `scripts/test-rir-graph.sh`
checks. The CPU oracle and the device parity lane stay: once a native kernel is
gone, they are the only independent reference.

## Choosing the next kernel

The lanes above only see ops RIR already covers. The census sees the others:

```sh
RETRO_RIR_CENSUS=1 RETRO_RIR_MODE=prefer \
RETRO_RIR_TEST_MODEL=/path/to/model.gguf \
  cargo test --release --test rir_graph_coverage -- --nocapture the_backward_graph_census
```

It walks every node of every graph a real training step computes and reports,
per `(ggml_op, backend)`, the node count, the bytes read and written, and up to
six destination shapes. It reports **work, not time**: timing a node would need
a synchronization per node, whose fixed cost exceeds most nodes and would
flatten the ranking. The shapes it names go to `test-rir.sh`, which times them
in isolation.

The top of the ranking is not automatically a candidate: `MUL_MAT` and the
fused sparse cross-entropy weigh the most, and are also the most optimized
native kernels.
