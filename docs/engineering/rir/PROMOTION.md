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
  Under MoltenVK the barrier itself costs more than most kernels, so
  `Session::time_stream` repeats the run without one; replaying a run is
  idempotent, which makes the throughput figure legitimate.

Level 1 compares RIR to RIR. It never decides a promotion: it has no native
kernel to compare against, and it does not take the ggml path.

### Offline schedule search

```sh
cargo run --release -p rir-sweep -- --kernel rms_norm_back --backend vulkan --reps 9
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

Nothing is written automatically. The last block of a sweep is a proposal to
review and commit by hand, and a table change then goes through the promotion
lane like any other.

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
   `RETRO_RIR_TEST_PREFER`, without which nothing would encode it;
4. **checks the per-site counters**: `rir > 0`, and every native fallback must
   be explained by a portable-contract rejection the registry declares for that
   pair. A device rejection, or a fallback nobody declared, fails the lane;
5. **times both paths** `--repeat` times, alternating native and RIR on every
   pass, and rules on the **median** of the ratios, with `min-max` published
   beside it.

The ratio is a verdict for a `prefer_generated` pair and an open work item for
an `observe_generated` one. A shape whose spread covers the whole distance to
the native kernel prints `parity`: neither a gain to claim nor a regression to
fail on.

Three properties the script enforces on itself: MoltenVK is configured by
detection, a requested device that is missing fails the lane instead of
skipping it, and every run is bounded by a watchdog, because a candidate kernel
can be a hundred times slower and "too slow to measure" is a verdict.

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
