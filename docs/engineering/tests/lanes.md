# Test lanes in detail

What each lane runs, what it costs, and the pitfalls that change what you would
naively run. To pick the lane for a change, start with
[Tests and validation](./notice).

## TL;DR

```sh
scripts/test-fast-rust.sh       # ~5 min cold, ~27 s warm - every commit
scripts/test-fast-python.sh     # ~1 min cold (build) - every commit
scripts/test-server.sh          # ~30 s - the HTTP control plane on its own
scripts/test-abi.sh             # ~1 min - before a PR touching the runtime
scripts/test-cpu-integration.sh # ~2 min warm - before a PR touching training
scripts/test-cpu-integration.sh grpo_runtime # one model test binary while iterating
scripts/test-container.sh       # ~2 min - needs a container daemon; before a PR
                                #   touching retrograd-container
scripts/test-rir-parity.sh      # ~30 s warm - needs a GPU; before a PR
                                #   touching a kernel, an emitter or a schedule
cargo run -p rir-sweep -- --list  # not a lane: the offline schedule search,
                                #   which proposes a table row and commits nothing
scripts/test-rir.sh             # needs a GPU - before promoting a RIR kernel
scripts/test-rir-graph.sh       # needs a GPU + the CPU fixture - before
                                #   retiring a native kernel
```

`test-server.sh` is a *subset* of `test-fast-rust.sh`, not an extra lane to
remember: the fast lane calls it. Run it directly while iterating on a
handler, so a one-line change does not recompile the workspace.

### Each lane keeps the build tree it needs warm

Backend features create separate native variants in the shared Cargo tree.
The CPU fast lane and the default macOS Metal build reuse their warmed variants
without reconfiguring the same native output at every switch. Common Rust
dependencies are shared; feature unification can still produce several Rust
variants. Keep other native environment inputs constant when measuring.

| lane | target dir | why |
| --- | --- | --- |
| `fast-rust`, `server`, `abi`, `cpu-integration` | `target` | Cargo backend features distinguish CPU and GPU variants |
| `rir-parity` | `target` | release artifacts; no dependency on `retrograd-ffi` |
| `fast-python` | `python/target` | the extension requires static native linking |
| docs CI | `target/lanes/docs` | separate cache, outside the local alternation |

`CARGO_TARGET_DIR` still overrides Cargo's tree; `RIR_PARITY_TARGET_DIR` selects
the parity tree. GPU trees configured externally should only be merged after
auditing native variables and `RUSTFLAGS`. The old `target/lanes/fast` cache can
be removed after comparing measurements; it is no longer used by the lane.
On macOS debug the native libraries remain shared, each binary referencing its
own feature variant's paths (see [build variants](../../reference/builds)).

Within a lane's log, `step=<name> duration=<n>s` lines (via
`scripts/lib-step-timing.sh`) split compile time from run time per test binary.

## Lanes

### `fast-rust` - Rust unit tests

Three CPU-only package selections compile and run library and binary tests:

```sh
cargo test --workspace --exclude retrograd --exclude retrograd-python --lib --bins
cargo test -p retrograd --no-default-features --features agent --lib --bins
cargo test -p retrograd-python --no-default-features --lib --bins
```

Before compilation, each selection's resolved graph (including dev-dependencies)
is checked with `cargo tree`: FFI must be present and none of `platform-gpu`,
`metal`, `vulkan`, `cuda` may be enabled. Workspace defaults remain on for the
judge's HTTP coverage. No GPU and no model are required.

Some integration binaries are included anyway, for the same reason: they need
neither a GGUF nor a device, and what they cover is the kind of thing that
regresses silently.

- `retrograd-server`'s ten `tests/api_*.rs` binaries, run through
  `scripts/test-server.sh` (see below).
- `retrograd-plan`'s `tests/resolve_snapshots.rs`,
  `tests/resolve_properties.rs` and `tests/derivation_table.rs` run the resolver
  over synthetic model geometries. The snapshots live in
  `crates/retrograd-plan/tests/snapshots/`; regenerate them with
  `RETRO_UPDATE_SNAPSHOTS=1 cargo test -p retrograd-plan` and **read the diff** -
  that file is what turns a silent change in the memory cost model into a visible
  one. `derivation_table.rs` is the one that fails when `GET /v1/defaults` stops
  describing what a resolution actually does.

- `retrograd-tools`' `tests/mcp_stdio.rs` drives the stdio MCP transport end to
  end - namespacing, a real call, the per-call timeout - by re-executing the test
  binary itself as the server (`RETROGRAD_MCP_ECHO_HELPER`). No daemon, no
  network, ~2 s.

- `rir-gen`'s `tests/{tables,artefacts,fork}.rs` are the generation lane:
  regeneration produces no diff, generation is deterministic, the declared
  tables validate, the README's family table matches the registry, and the fork
  copies match the generated artefacts (`cargo test -p rir-gen --tests`, ~3 s).

  Three of `artefacts.rs`'s cases spawn one compiler *process* per generated
  source - `glslc` over the `.comp`, `xcrun metal` over the MSL, `nvcc` over the
  `.cu` where a toolkit exists. They stay in the fast lane because they need no
  device, and they run across the machine's cores (~11.5 s sequential, ~2.4 s
  parallel). Every failure is reported, not just the first, so a scheduling
  order nobody chose cannot decide which broken shader the message names.

- `rir-kernels`' `tests/f16_oracles.rs` compares the **three** F16 conversions
  RIR carries on purpose - the canonical decoder, the interpreter's, and the one
  printed into `generated/rir/*/cpu.rs` - over all 65 536 halves. It costs
  milliseconds, and it is the test that found two of the three decoding
  subnormals one binade too small while agreeing with each other, which no
  parity test could see.

- Public-contract tests with neither a model nor a device:
  `retrograd-agent`'s `train_sequences`, `retrograd-engine`'s
  `chat_parser_roundtrip` (on chat-template fixtures), `retrograd-judge`'s
  `reward_batch` and `retrograd-training`'s `value_head`.

`rir-runtime`'s `tests/device_parity.rs` and `tests/family_parity.rs` are
**compiled but not run** here: they run in `rir-parity` below. The fast lane
still builds them, so a parity test that stops compiling fails on the commit
that broke it rather than on the day someone runs the GPU lane.

No other `tests/*.rs` binary runs in this lane.

The lane also carries **graph** checks rather than tests, because each
optionality property is one distracted `[dependencies]` line away from being
lost:

- `bollard` must not be reachable from `retrograd-agent` in the default
  configuration nor in the MCP-only one (`--no-default-features --features
  mcp`), nor from the `retrograd` binary unless `--features container` asks
  for it;
- `rustls` must not be reachable from `retrograd-judge` without its `http`
  feature, and `cargo check -p retrograd-judge --no-default-features` proves the
  crate still compiles that way;
- `retrograd-scenario-gen` must not reach `retrograd-engine`, `retrograd-ffi`
  or `rmcp`.

### `server` - the HTTP control plane, no model and no socket

`scripts/test-server.sh`: `retrograd-server`'s unit tests plus its ten
`tests/api_*.rs` binaries, which drive `build_router` in memory
(`tower::ServiceExt::oneshot`) against a fake `ModelProbe` and a fake
`RunEngine`. Nothing here loads a GGUF, touches a device or opens a port, and
the HTTP wire contract is the last thing that should only be checked before a
PR. All ten share `tests/support/mod.rs`.

| Binary | What it pins |
|---|---|
| `api_discovery` | health, capabilities, presets, the operator catalogue, preflight |
| `api_plan` | the three body forms, the guard, redaction, the measured pass (a fake probe reporting a fixed multiple of the cost model) |
| `api_runs` | the whole runtime: state machine, device queue, journal on disk, idempotency, listing and paging |
| `api_control` | pause/resume/cancel, the on-demand checkpoint, the `PATCH` whitelist - against a fake engine that really *polls* the control channel twice per iteration |
| `api_events` | SSE replay, the reconnection property (`Last-Event-ID`, no gap and no repeat), the metrics pull, the aggregate stream |
| `api_inference` | `evaluate` and `generate`: the round trip through the control channel, the reply's `global_step`, a paused run answering, the 504 |
| `api_artifacts` | the checkpoint listing, the closed artefact inventory, download by name, `DELETE` |
| `api_fork` | `fork_from` in both shapes, the trajectory refusal, an incomplete checkpoint |
| `api_hardening` | the bearer token, the loopback refusals, the body limit, every middleware failure as a problem document, path redaction, path roots, OpenAPI |
| `api_datasets` | content-addressed upload and idempotence, collected line-error validation, preview, listing, delete, the per-dataset body limit |

The one server test **not** in this lane is `tests/e2e_cpu.rs`, which needs the
CPU fixture - see `cpu-integration`.

### `fast-python` - Python API tests

Runs `ruff check` and `ruff format --check` over `python/`, `examples/` and
`scripts/` (configuration in the root `ruff.toml`), builds the native PyO3
extension with defaults disabled via a common
`uv run --config-settings-package 'retrograd:build-args=--no-default-features'`
command for lint, format, reinstall and tests. After forced reinstall, a fresh
process calls `retrograd.list_backends()` and requires CPU present / GPU absent.
Then it runs the full `pytest` suite. **The CPU selection is required**: the
suite includes a real smoke test against the compiled extension, and building it
with the macOS default (`cpu,metal`) instead can segfault the whole `pytest`
process on a Metal buffer allocation when no usable Metal queue exists.

The CPU-only extension stays installed after the lane: a plain `uv run` does
not rebuild it, because uv does not treat a change of build settings as a
reason to reinstall. To get the platform default (Metal on macOS) back, run
`uv run --reinstall-package retrograd python -c "import retrograd"` from
`python/`.

Extra `pytest` args can be appended, e.g.
`scripts/test-fast-python.sh tests/test_trainer.py -k grpo`.

### `abi` - runtime and kernel contracts, no model

`retrograd-ffi`'s unit tests, `backend_devices`, `metal_ops`, `weighted_ce`,
`gated_delta_net_chunked`, `flash_attn_back`, `out_prod_quant`, and the one
`engine_contracts` case that needs no model. Nothing here loads a model, so it
never needs the CPU fixture. This is also where the C ABI is pinned:
`contract_tests::c_struct_layouts_match_the_published_header` asserts the size,
alignment and field offsets of every `#[repr(C)]` struct in `retrograd-ffi`, and
the error paths of the introspection entry points (`retro_read_model_info`) are
checked here because they need no model either. `RETRO_ABI_FEATURES` defaults to
`platform-gpu` (Metal on macOS, CPU elsewhere). Set `RETRO_ABI_FEATURES=` for
CPU alone, or a comma-separated list of `platform-gpu`, `metal`, `vulkan`,
`cuda`. The lane validates names before Cargo and applies the same backend
selection to both FFI and root tests, including every RIR mode pass. A named
list is passed as `--features <list>` with the root's defaults kept, as a user
build would spell it, so the lane reuses that build's native variant.

`gated_delta_net_chunked` compares two *CPU* implementations of
`GGML_OP_GATED_DELTA_NET_BACK` - the per-token reverse scan and a chunkwise
form - against each other, on shapes and gate regimes no GPU test covers
(single tokens, chunk-straddling token counts, gates that force the chunk
guard to halve or bail out). It belongs to a no-GPU lane on purpose: when a
GPU parity test fails, this one says whether the derivation or the kernel is
at fault.

`flash_attn_back` exercises the native CPU implementation against the
independent analytic streaming oracle. It covers F16/F32 KV, a KV gradient
window, attention sinks, causal masking and softcap, then pins bit-identical
repeated execution for the fixed-order dK/dV reduction.

`out_prod_quant` is the model-free CPU contract: every type in the complete
24-entry CPU/CUDA/Vulkan `OUT_PROD` table executes, outputs remain
finite and non-vacuous, and the CUDA scratch budget has no effect on the native
CPU result.

`rir_quant_oracle` checks the other half of that table - the RIR side. It
asserts that the block geometry the canonical format table declares is the
one ggml actually uses, that the portable RIR decoders reproduce ggml's
`to_float` **bit for bit on identical bytes**, and that a `NativeIntrinsic`
format refuses to decode rather than guessing. Same bytes on both sides is
the point: comparing two quantizations of the same F32 would measure the
quantizer, not the decoder.

### `cpu-integration` - model-dependent smoke tests

Fetches and checksum-verifies the CPU GGUF fixture, then runs CLI,
capabilities, PPO/GRPO, checkpoint/resume, `engine_contracts`,
`kv_projection_gradients`, the CPU-only LoRA/generation/parity tests, and
`retrograd-server`'s `tests/e2e_cpu.rs`. The root test binaries are built with
`--no-default-features --features agent` (the `-p retrograd-server` ones need
nothing: that package is CPU-only by default), and every run sets
`RETRO_REQUIRE_CPU_FIXTURE=1` (missing fixture is a hard failure, not a silent
skip). Model-loading tests share process-global llama.cpp runtime state, so
this lane runs with `--test-threads=1` and a cross-process file lock
(`RETRO_RUNTIME_LOCK_PATH`) - expect about two minutes warm; this is a pre-PR
lane, not a per-commit one.

The fixture is `tests/fixtures/LFM2.5-230M-Q4_K_M.gguf` (Unsloth LFM2.5 230M
Q4_K_M); its URL, size and SHA-256 live in `tests/fixtures/CPU_FIXTURE.toml`.
`RETRO_CPU_FIXTURE=/path/to/model.gguf` reuses a verified local copy. The fetch
script writes a sidecar `.verified` stamp containing the checksum, size and
mtime, so repeated lane runs avoid hashing the full file unless it changed.

The lane defaults to four llama.cpp CPU workers. Override with
`RETRO_TEST_CPU_THREADS=<n>`. Set `RETRO_PROFILE_TESTS=1` only when diagnosing compilation: it restores the
extra `cargo test --no-run` passes, omitted in normal runs because Cargo already
builds before executing. `RETRO_TEST_TIMING=1` prints model-load, lock-wait and
training-operation timings.

For iteration on one expensive binary, pass it as an argument:
`scripts/test-cpu-integration.sh <test-binary> [filter]`. It keeps the fixture,
backend, runtime lock, thread count and hard-failure guarantees of the full
lane.

`e2e_cpu.rs` is the server's one model-dependent test, and it is here because a
fake engine cannot answer whether the pieces *join*: it plans with
`?calibrate=true` (so the resolver's measured pass is checked against
`Trainer::memory_report` rather than against a fake), creates a run, pauses it,
evaluates and generates against the paused model, asks for a checkpoint, cancels
at a boundary, lists what landed, replays the event stream, and forks. It skips
itself without `RETRO_CPU_FIXTURE` and **fails** without it when
`RETRO_REQUIRE_CPU_FIXTURE` is set, which is what this lane sets.

`tests/recurrent_families.rs` covers one family of recurrent state per row:
`shortconv` runs on the default fixture, `conv_ssm` and `gated_delta_net` skip
unless `RETRO_FALCON_H1_TEST_MODEL` and `RETRO_QWEN3NEXT_TEST_MODEL` point at a
GGUF.

### `container` - sandboxes against a real daemon

`scripts/test-container.sh`: `retrograd-container`'s `tests/daemon.rs`, which
creates, execs into, writes to and destroys real containers.

This lane depends on a service rather than on the machine, which changes one
rule: when no daemon answers it prints `SKIPPED` and exits **non-zero** (unless
`RETRO_CONTAINER_OPTIONAL=1`). A lane that silently passes because its
prerequisite was missing is worse than no lane - the properties here are
precisely the ones whose absence is invisible:

- **isolation between two leases.** An episode writes a marker and leaves a
  background process; the next episode on the *same* recycled container must see
  neither. If this regresses, a group's members stop being independent samples
  and the GRPO relative baseline measures the leftovers instead of the policy.
  No unit test can prove it - only the runtime can.
- **limits actually applied by the runtime**, not merely requested: the timeout
  kills, `max_output_bytes` cuts on a UTF-8 boundary, a greedy allocation fails
  as a readable *observation* rather than as a broken sandbox.
- **no network by default** - a name lookup fails deterministically.
- **byte-identical observations** across two different containers for the same
  command, which is the reason a container beats `LocalSandbox` for training.
- **the reaper**, and a final count asserting nothing labelled `retrograd.pool`
  survived the suite - including after a test that failed.

The daemon may be Docker Desktop, colima or Podman in Docker-compatible mode
(`DOCKER_HOST`, or the `/var/run/docker.sock` symlink Podman installs); all
three are exercised the same way. The image needs GNU `find` and coreutils
`timeout`, which any Debian-based one has - override with
`RETRO_CONTAINER_TEST_IMAGE`. The lane runs `--test-threads=1`, because the tests
share one daemon and assert on how many containers are labelled as theirs.

Everything decidable *without* a daemon stays in `fast-rust` as unit tests, and
must stay there: the pool's whole policy (backpressure, retirement, "a doubtful
cleanup destroys rather than recycles", prewarming) runs against a fake
`SandboxSource`, the spec's refusals and the reaper's rule are pure functions.
`retrograd-container`'s unit tests create no container and have no side effects.

### `rir-parity` - the generated shaders against the oracle, no model

```sh
scripts/test-rir-parity.sh              # every parity binary
scripts/test-rir-parity.sh out_prod     # a filter, while iterating
```

`rir-runtime`'s device-parity binaries, in **release**, in
`target/lanes/rir-parity`. Nothing here loads a model or reads the fixture: what
it proves is that the committed artifacts under `generated/rir/` compute on a
real device what the Loop IR interpreter computes on the host.

- `device_parity.rs` - one hand-written case per kernel, on geometries a derived
  harness cannot produce: a view with a gap between rows, a specific tiled or
  blocked schedule, extents that are no multiple of a workgroup. It also carries
  the negative cases (a binding that is not the one prepared, an unmeetable CUDA
  feature) and, where `glslc` is installed, compiles every shader to SPIR-V.
- `family_parity.rs` - the same property **derived** from
  `rir_kernels::registry()`, so a kernel is covered the day it is registered,
  on packed strides, with the accounting that every registered lowering is
  covered, skipped with a reason, or refused in writing by its family. The two
  are complementary, not redundant - one checks the shapes nobody would derive,
  the other checks the kernels nobody wrote a test for.
- `dispatch_plan.rs` - the same property for a **capability** rather than a
  kernel: one kernel executed as several dispatches with a scratch buffer
  between them. Three dispatches return the scan one dispatch returns, which is
  what makes the plan's strides, barriers, grid expressions and scratch
  lifetimes checkable at all. Its timing half - the plan against the three
  single-dispatch scans, row length by row length - stays behind `RIR_TIME=1`,
  like every other timing in this crate.

All three skip themselves, printing why, without `libvulkan`, a GPU or a GLSL
compiler, so the lane is a no-op on a bare CI runner.

**Why this is a lane and not part of `fast-rust`.** The skip rules make these
binaries cost ~0.2 s on a bare runner. They do not skip on a developer machine
with MoltenVK and glslc installed: there they run the whole registry through an
interpreted oracle, which in the fast lane's debug build took **over fifteen
minutes**. Two things keep it bearable, and both are needed:

- the profile - the expensive half is the oracle, not the device (`out_prod` and
  the tiled scan are interpreted loop nests, statement by statement), so the
  lane builds `--release`;
- the parallelism - `every_registered_kernel_agrees_with_the_oracle_on_the_device`
  is *one* `#[test]` walking the whole registry, so libtest has nothing to
  spread. It plans the work without touching a device, then runs it across the
  machine's cores with one `AnyGpu` per worker (149 s sequential, 26 s
  parallel). `RIR_PARITY_THREADS` overrides the worker count.

### GPU lanes (`metal`, `vulkan`, `cuda`)

**Set `RETRO_REQUIRE_GPU_RESIDENT=1` for every GPU-lane invocation below.** It
turns each run's `require_gpu_resident` on, so a training-graph node that falls
back to the CPU fails the preflight instead of running quietly and slowly. This
is a lane property, not a per-run one, which is why it is an environment
variable: `chunked_cross_entropy` is on by default, and its two fused nodes are
exactly the kind of thing that silently lands on the CPU tail for one head
geometry on one backend. A run that sets the field itself keeps the guard
regardless of the environment.

These lanes are not wrapped in a script: they need a runner with the matching
driver and, for `vulkan`/`cuda`, a GGUF model outside the versioned CPU fixture
(see `RETRO_VULKAN_TEST_MODEL`, `RETRO_CUDA_TEST_MODEL`,
`RETRO_FALCON_H1_TEST_MODEL` in `tests/common/mod.rs`). Run the `cargo test`
invocations below directly on such a machine.

| Binary | What it pins |
| --- | --- |
| `backend_devices` | the build links the GPU backend and registers a device (no model) |
| `metal_ops` | every hand-written Metal training kernel against the CPU reference (no model) |
| `weighted_ce` | the fork's weighted cross-entropy against an analytic reference, CPU and Metal (no model) |
| `lora_metal` | trainable LoRA tensors allocated on the Metal buffer, and a short step updating them |
| `model_offload` | `--device` actually offloads model tensors to the GPU |
| `train_parity` | a full CPU against Metal epoch (train, save, reload) with matching losses |
| `device_memory` | the optimizer path's measured device budget, for an adapter run and for a base one |
| `base_training` | its two `device_resident` cases: a base run's model export and trainable bundle, read off the device |
| `vulkan_backend` | Vulkan registration, isolated ops, model offload, LoRA placement, a minimal training step |
| `cuda_backend` | CUDA registration, CPU against CUDA op parity, model offload, LoRA placement, the training preflight |

On Apple hardware the Metal lane is `metal_ops`, `lora_metal`, **`fused_ce`
and `scoring_logprobs`** - the last two are not Metal-specific by name, but
they run on whatever GPU is registered: `fused_ce` is the parity coverage for
the Metal `FUSED_SPARSE_CE` kernels, and `scoring_logprobs` is the parity
coverage for the on-device gather of behavior log-probabilities against the
host oracle ([sampling path](../optims/SAMPLING#device-side-target-logprob-extraction)).
Both skip themselves without a GPU device, which is why neither belongs to the
CPU lanes:

```sh
RETRO_REQUIRE_GPU_RESIDENT=1 cargo test --features metal --release \
  --test metal_ops --test lora_metal --test fused_ce --test scoring_logprobs \
  -- --test-threads=1
```

`device_memory` and the base-training export cases need the CPU fixtures rather
than a device-specific model, so they take the lane's fixture variables and run
beside the list above. Every GPU case in both binaries skips itself without a
device. Two invocations rather than one: a libtest filter applies to every
selected binary, and `base_training` needs one to keep the rest of that binary
- which belongs to the CPU lane - out of the GPU run.

```sh
export RETRO_REQUIRE_GPU_RESIDENT=1
export RETRO_CPU_FIXTURE=tests/fixtures/LFM2.5-230M-Q4_K_M.gguf
export RETRO_TINY_FIXTURE=tests/fixtures/retrograd-tiny-qwen2-f32.gguf
cargo test --features metal --test device_memory -- --test-threads=1
cargo test --features metal --test base_training -- --test-threads=1 device_resident
```

Vulkan and CUDA, whole binaries. Model-dependent Vulkan cases want
`RETRO_VULKAN_TEST_MODEL`; CUDA cases default to the in-repo CPU fixture, and
`RETRO_CUDA_TEST_MODEL` / `RETRO_CUDA_FALCON_H1_TEST_MODEL` select another
architecture and the recurrent regression model. The CUDA reference sequence,
fused cross-entropy and export/resume included, is in
[CUDA status](../cuda/STATUS).

```sh
RETRO_REQUIRE_GPU_RESIDENT=1 \
  cargo test --features vulkan --test vulkan_backend -- --test-threads=1
RETRO_REQUIRE_GPU_RESIDENT=1 \
  cargo test --features cuda --test cuda_backend -- --test-threads=1
```

The model-free Vulkan GDN chunking coverage can be run on its own; it compares
the chunked default, its local adverse-gate fallback and the retained sequential
pipeline against the CPU oracle:

```sh
RETRO_REQUIRE_GPU_RESIDENT=1 cargo test --features vulkan --test vulkan_backend \
  gated_delta_net_back_vulkan -- --test-threads=1
```

The backward Flash Attention F2 is covered without a model on CUDA and Vulkan.
The cases compare the matrix path against the CPU oracle, the forced scalar
fallback and a repetition, by tolerance (Q/dO are rounded to F16 before MMA):

```sh
RETRO_REQUIRE_GPU_RESIDENT=1 cargo test --features cuda --test cuda_backend \
  flash_attn_back_cuda_ -- --test-threads=1
RETRO_REQUIRE_GPU_RESIDENT=1 cargo test --features vulkan --test vulkan_backend \
  flash_attn_back_vulkan_ -- --test-threads=1
```

The fallbacks can also be forced outside the test with
`GGML_CUDA_FA_BACK_MMA=0` and `GGML_VK_FA_BACK_MMA=0`.

### `test-rir-graph` - RIR coverage on a real training graph

```sh
scripts/test-rir-graph.sh                  # every GPU backend present
scripts/test-rir-graph.sh --backend metal  # one of them
```

`test-rir.sh` judges a kernel in isolation, on the shapes a conformance bench
enumerates. This lane asks the other question: on the graph a model actually
trains, did every node that went native do so for a reason the registry
*declares* - and does every pair whose native has been retired still serve
all of its nodes, since nothing else can.

It loads a model, which is what separates it from `test-rir.sh` and what makes
it slow. The model is the versioned CPU fixture by default, fetched and
checksum-verified like the `cpu-integration` lane's, so the numbers are
reproducible; `RETRO_RIR_TEST_MODEL` overrides it. One process per backend,
because the RIR mode and the backend list are both read once, before the first
context exists.

Each backend is built with `--features <backend>`.
The first build of a variant is expensive; later runs reuse it. CUDA's native
architecture setting and caller `RUSTFLAGS` remain inputs outside the feature
hash; their values must be held constant when comparing timings. The script
adds no CUDA cfg: the root gets it from FFI metadata. CUDA validation requires
a real CUDA machine and the fixture, with counters and no skipped test.

**A lane that skips is a lane that lies**, so `RETRO_REQUIRE_RIR_GRAPH=1` (which
the script sets) turns every reason the test has to step aside - no GPU, no
model, a graph with no registered op - into a failure. Run by hand, the same
test still skips loudly:

```sh
  RETRO_RIR_TEST_MODEL=tests/fixtures/LFM2.5-230M-Q4_K_M.gguf \
  cargo test --features metal --release --test rir_graph_coverage -- --nocapture
```

It publishes the per-site coverage - the number to read before touching a
kernel - and, for a pair whose native is already gone, says so:

```text
rir graph coverage: GGML_OP_ADD           metal  1062/1062 nodes (100 %) rejects=[]           domain=[dtype|shape]
rir graph coverage: GGML_OP_MUL           metal  640/920   nodes (69 %)  rejects=[shape=280]  domain=[dtype|shape]
rir graph coverage: GGML_OP_OUT_PROD      metal  680/736   nodes (92 %)  rejects=[dtype=56]   domain=[dtype|shape]
rir graph coverage: GGML_OP_RMS_NORM_BACK metal  280/280   nodes (100 %) rejects=[]           domain=[none]
rir graph coverage: GGML_OP_RMS_NORM_BACK metal  native retired - the generated variant is the only implementation
```

The `site-retired` row behind that last line comes from
`rir_op_policy.native_retired`, not from `native=0`: a lucky graph produces
`native=0` on a pair whose native kernel is very much still there. On a retired
pair the test then asserts the exit criterion - no declared domain, no native
dispatch, every node served.

### `test-rir` - the RIR promotion lane

```sh
scripts/test-rir.sh                  # every (op, backend) pair, every GPU present
scripts/test-rir.sh --op CUMSUM      # one op, while iterating on its schedule
scripts/test-rir.sh --no-perf        # correctness only, ~10x faster
scripts/test-rir.sh --repeat 5       # more timing passes; the verdict is the median
scripts/test-rir.sh --escalate 0     # do not re-measure an undecided refusal
```

This is the lane that decides whether a generated kernel may replace a
hand-written one ([RIR kernel promotion](../rir/PROMOTION)). It loads no model
and trains nothing. For every pair the generated registry declares dispatchable
on Metal or Vulkan, it builds the fork if needed, runs the op matrix **twice on
the same shapes** - once native, once RIR - asserts the per-site counters, then
times both and prints `native | RIR | ratio`.

**Which GPU it builds for.** The fork is configured with Metal on macOS and
Vulkan everywhere; Metal is *off* elsewhere on purpose, because
`find_library(Foundation)` is a configure **error** on Linux and not a skipped
backend - `-DGGML_METAL=ON` there stops the lane before it compiles anything.
On the Linux GPU box the lane therefore covers Vulkan, and a Metal verdict needs
an Apple machine.

Running both modes is not redundancy. `RETRO_RIR_MODE=off` exercises the native
kernel and nothing else - it is what caught a native Vulkan kernel failing on a
packed-QKV shape while the RIR variant passed every case. A green RIR run says
nothing about the kernel it is supposed to be equivalent to - and, symmetrically,
the counter assertion is what stops a matrix that RIR never encoded from
reading as a green RIR run.

**Where there still is a native kernel.** A pair whose native has been retired
has no second column, and the lane says so rather than keeping the shape of
its output:

```text
== L2_NORM_BACK / metal (policy: prefer_generated, native retired)
    native matrix: - (native retired, no witness left)
    RIR matrix: OK
    -- no differential timing: native is retired
```

The run is skipped rather than attempted, and the reason is worth knowing:
under `off`, `ggml_rir_supports_op` answers false for such a pair, the backend
declines every case, and `test-backend-ops` prints `0/0 tests passed` followed
by a green `OK`. The lane would publish "native matrix: OK" for a matrix that
ran nothing, so it refuses that reading for *any* pair - a matrix green on zero
cases is a failure. What is lost with the differential is the ratio, not the
correctness: the RIR matrix still compares the generated kernel against the
CPU reference, case by case.

The ratio is a **verdict** for a pair the registry has already promoted
(`prefer_generated`): a regression there fails the lane. For a pair still in
`observe_generated` it is printed as `work item` - an open performance item,
which is exactly what that policy means.

The timing pass runs **three times** by default (`--repeat N`) and the verdict
is the **median** of the ratios; the column beside it publishes min-max. One run
is a draw, not a measurement: between 5 and 40 µs per dispatch, two runs of the
same binary differ by up to 12 %, more than twice the 5 % tolerance, so a pair
at parity would be promoted or refused at random
([why the verdict is a median](../rir/PROMOTION#why-the-verdict-is-a-median)).
Each pass re-times both paths back to back, so a machine-wide slowdown moves the
two columns together and cancels.

**A refusal the passes disagree about is not a verdict**: when a promoted
pair's median exceeds the tolerance while at least one pass came in under it,
the lane runs `--escalate N` more passes (six by default) before calling it a
regression. A median whose *best* pass is still over the tolerance is a
regression on every draw and is not escalated. **And an `ok` the passes do not
support is not a win**: a shape prints `parity` when its worst pass crossed the
tolerance, or when its min-max interval contains 1.00 - the shape's own spread
then covers the whole distance to the native kernel, which means the lane
measured its launch floor and not a difference. Neither is a win to claim nor
a regression to fail on, and naming them is the point. The `passes` column
says how many draws produced the row.

**A ratio never leaves its session, and the lane says which one.** The run
opens with a stamp - machine, devices, repo and fork revisions (with `+dirty`
when the tree is dirty), `--repeat`/`--escalate`/`--tolerance` - and the
performance table repeats its identifier. Machine drift of tens of percent has
been measured between two days on the same native kernel and the same shape, so
a ratio quoted without its stamp is not evidence - and neither is a before/after
built from two runs on two days. **To compare two RIR versions, run the lane
twice in the same session and check the native column is stable across the two
passes**; that stability is what makes the two RIR columns comparable.

The matrix itself lives in the fork's own `test-backend-ops`, not in a Rust
lane, because it is the only harness that can build a *view* into a packed
tensor - the shape the probe cannot express and the real graph always sends.
The script drives it; the equivalent by hand is:

```sh
RETRO_RIR_MODE=prefer RETRO_RIR_STATS=1 ./build-rir/bin/test-backend-ops -o L2_NORM_BACK -b MTL0
```

On macOS the Vulkan backend runs through MoltenVK, which the script configures
by detection; by hand it is:

```sh
DYLD_LIBRARY_PATH=/opt/homebrew/lib \
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
  ./build-rir/bin/test-backend-ops -o L2_NORM_BACK -b Vulkan0
```

A pair in `observe_generated` never encodes on its own, so nothing could measure
it; `RETRO_RIR_TEST_PREFER=<OP>` promotes it for one process. It only raises
observe to prefer - a `NATIVE_ONLY` pair has no pipeline and stays unreachable.

A pair may publish **several variants**, arbitrated per shape by the registry's
shape rules, so `rir=N` does not say *what* ran. The lane prints the breakdown,
and it is what to read first when a ratio surprises you:

```text
counters: seen=15 rir=15 native=0 [f32_4d_subgroup_tree_blocked]=14 [f32_4d_serial]=1
```

`native` is not required to be zero. A RIR variant may claim less than its ggml
op - `OUT_PROD` declines a quantized `src0` and a broadcast `src0`, both
legitimate ggml - and the lane checks *why* each fallback happened rather than
that none did. A device reason (`missing_feature`, `pipeline`, `device_grid`,
`device_alignment`) or `wrong_op` / `policy_native` fails the lane, and so does
any fallback the reject reasons do not account for.

A portable-contract reason (`dtype`, `rank`, `shape`, `stride`, `quant_block`,
`integer_range`) is a published domain **only if the registry declared it**.
Each integration lists in `rir_kernels::integrations` the parts of its ggml op
it does not claim, with the reason for each; that list becomes
`rir_op_policy.assumed_domain` and prints on the site line. A fallback outside
it is a node the kernel claimed and did not serve - a regression, and the lane
says so:

```text
counters: seen=193 rir=20 native=173 [f32_4d_serial]=20
coverage: 20/193 nodes claimed (10 %), 173 out of contract (dtype=122 shape=51)
!! fallback outside declared domain (dtype): shape=19
```

That declaration names a *category*, never a set of nodes, so it cannot see a
narrowing **inside** a category it already lists - `dtype` declared for F16 also
excuses a kernel that stops serving an F32. The claimed-node count does see it:
the matrix is a fixed case list, so `rir` is a constant per pair, recorded in
`scripts/rir-domain-baseline.tsv` and compared on every run.

```text
!! domain narrowing: 34 → 32 nodes claimed out of 86
```

The two checks are complementary - the mask says *why* a fallback is legitimate,
the baseline notices *how many* stopped being served. Re-record after a
deliberate domain change:

```bash
scripts/test-rir.sh --no-perf --update-domain-baseline
```

A pair with no recorded row is a note, not a failure (a newly integrated op has
to be able to enter the lane), and a changed `seen` is reported without being
judged - that is the matrix moving under a llama.cpp update, not the kernel.

The rates are collected into a table at the end of the run, next to the
performance one. They answer different questions and neither substitutes for the
other: a pair can be green on every shape and still be unfit for the removal of
its native kernel, because the native remains the only path for what the kernel
never claimed.

```text
== claimed domain (matrix nodes; "declared" = restrictions published by the registry)
pair                  claimed    rate   out of contract     declared
RMS_NORM_BACK/metal   30/30       100 %  -                   none
OUT_PROD/metal        20/193      10 %   dtype=122 shape=51  dtype|shape
```

The matrix enumerates dtypes and broadcasts because it is a conformance bench;
a model samples them. So this table says what the kernel *would* refuse, and
`rir_graph_coverage` above says what it actually refuses on a real graph - the
two numbers can differ widely and both be true; only the second is the
condition on removing a native kernel.

The same counters line carries `ggml-rir: selftest selection=0x0` - the
selection rule replayed on synthetic tables inside the process under test
(priority, policy veto, tie order, shape rules). The lane fails on anything but
`0x0`.

#### The short loop underneath it

```sh
RIR_TIME=1 cargo test --release -p rir-runtime --test device_timing -- --nocapture
```

`rir-runtime`'s `tests/device_timing.rs` times the **generated shader alone**,
on a device, without ggml and without rebuilding the fork - seconds instead of
minutes ([level 1](../rir/PROMOTION#level-1-the-shader-alone)). It lowers and
emits the kernel in-process from a `Schedule` written in the test, so trying
another schedule is a one-line edit; it is the loop to work in while a kernel is
still being shaped. It is opt-in (`RIR_TIME`) because a timing number has no
business failing a correctness lane.

It measures RIR against RIR and nothing else. **No promotion is decided here** -
that needs the native kernel on the same shapes, which only `test-rir.sh` runs.

#### The op census - which kernel to write next

```sh
RETRO_RIR_CENSUS=1 RETRO_RIR_MODE=prefer \
RETRO_RIR_TEST_MODEL=/path/to/model.gguf \
  cargo test --features metal --release --test rir_graph_coverage -- --nocapture the_backward_graph_census
```

The two lanes above are about the ops RIR **already covers**. The census is
about the ones it does not: it walks every node of every graph a real training
step computes and reports, per `(ggml_op, backend)`, the node count, the bytes
read and written, and up to six destination shapes
([choosing the next kernel](../rir/PROMOTION#choosing-the-next-kernel)). That
is what ranks the candidates for the next kernel; the site counters structurally
cannot, because an unregistered op never reaches a site.

`RETRO_RIR_CENSUS` is latched at the first graph, so it has to be in the
environment before the process starts - a test cannot turn it on for itself. The
same rows also print at process exit, next to the `RETRO_RIR_STATS` lines.

It reports **work, not time**. Timing a node would need a synchronization per
node, whose fixed submission cost on macOS exceeds most nodes and would flatten
the ranking it exists to produce. The census hands the shapes to `test-rir.sh`,
which times them in isolation.

## Continuous integration

`.github/workflows/ci.yml` runs on demand (`workflow_dispatch`), CPU-only, on
`ubuntu-latest`: no GPU, no model fixture. It runs `fast-rust` and
`fast-python`, plus `cargo clippy --all-targets -- -D warnings` and
`cargo doc --no-deps` (with `RUSTDOCFLAGS=-D warnings`) over the same three
CPU package selections as `fast-rust`, so they reuse its `retrograd-ffi`
variant instead of configuring llama.cpp again for the root's `platform-gpu`
default, and
`ruff check` / `ruff format --check`. It builds the llama.cpp submodule from a
clean checkout, which is what makes `RETRO_STRICT_LLAMA`'s "in CI" half of its
own description true (`crates/retrograd-ffi/build.rs` reads the `CI` env var
GitHub Actions sets, no extra flag needed). The GPU lanes (Metal, Vulkan, CUDA)
are **not** wrapped - they need real hardware - and are run by hand before a PR,
per the table in [Tests and validation](./notice).

## Everything at once - audit only, not a lane

```sh
cargo test --workspace --all-targets -- --list
```

Forces compilation of every binary and backend combination (~2 min cold).
Use to audit what exists, not as a routine check.

## Coverage

```sh
cargo llvm-cov --workspace --lib --summary-only
```

Rust library code only (~60% lines) - no `tests/*.rs` integration binaries, no
C++, no PyO3 binding. Treat it as one lane's report, not overall project
coverage.

## Known gaps

- GPU-lane scripts (`metal`, `vulkan`, `cuda`) are not wrapped yet.
- `retrograd-engine`, `retrograd-agent::policy`, and the PyO3 binding have
  little to no dedicated unit coverage.
- Two checkpoint continuity tests remain `#[ignore]`d pending an agreed
  numerical tolerance.
