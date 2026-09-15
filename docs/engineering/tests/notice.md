# Tests and validation

Use the lane scripts: they select the right backends, fixtures and build
directories. Do not run a raw `cargo test` or `pytest` to validate a change in
the repository. What each lane runs, what it costs and its pitfalls are in
[the test lanes in detail](./lanes).

## Choosing a lane

| Change | Command to run |
| --- | --- |
| Everyday Rust | `scripts/test-fast-rust.sh` |
| Python API | `scripts/test-fast-python.sh` |
| HTTP handler | `scripts/test-server.sh` while iterating; `test-fast-rust.sh` before shipping |
| Runtime, ABI or FFI | `scripts/test-abi.sh` |
| Training or CPU integration | `scripts/test-cpu-integration.sh` |
| Recurrent or hybrid architecture | `scripts/test-cpu-integration.sh`, with a model per family |
| Container integration | `scripts/test-container.sh` |
| RIR kernel, emitter or schedule | `scripts/test-rir-parity.sh` |
| Promoting an RIR kernel | `scripts/test-rir.sh` |
| Removing a native RIR kernel | `scripts/test-rir-graph.sh` |

`tests/recurrent_families.rs` runs inside the CPU integration lane and covers
one family of recurrent state per row, not one model: `shortconv` is the
default fixture and always runs, `conv_ssm` and `gated_delta_net` skip unless
`RETRO_FALCON_H1_TEST_MODEL` and `RETRO_QWEN3NEXT_TEST_MODEL` point at a GGUF.
Touching the packed-training capability, the micro-batch finiteness check or
the recurrent rollback derivation means running that lane with all three set -
a lane that only ever loads the default fixture proves nothing about the other
two families.

`test-fast-rust.sh` and `test-fast-python.sh` are the minimal checks for a
change that touches both layers. Lanes that need a GPU, a model or a daemon say
so and must be run before a PR when their scope is affected.

## RIR: iterate, then decide

To adjust a shader or a geometry, start with:

```sh
scripts/test-rir-parity.sh
```

This lane checks the generated shader against the Loop IR oracle, without a
model. To quickly measure a given schedule, use the RIR timing test:

```sh
RIR_TIME=1 cargo test --release -p rir-runtime --test device_timing -- --nocapture
```

An offline search can propose geometries:

```sh
cargo run --release -p rir-sweep -- --kernel <kernel-name>
```

Neither of those two commands promotes anything. Only `scripts/test-rir.sh`
compares the generated kernel and the native kernel, on the same shapes and the
same hardware; it is therefore what decides a registry policy (see
[RIR kernel promotion](../rir/PROMOTION)).

## Build trees

The build directories are deliberately separated by backend and profile:
sharing them can trigger needless llama.cpp recompilations. Each script's
variables let you override them if needed.

## Before a PR

Run the fast lane matching your layer and any lane listed in the table for the
components you changed. Add the GPU lane for a backend change; add the ABI lane
for the runtime or the FFI. A lane failure is the starting point for diagnosis:
keep its log rather than retrying with a different raw command.
