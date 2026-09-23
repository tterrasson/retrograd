# llama.cpp fork source notes

Retrograd consumes a dedicated fork vendored as a git submodule rather than
applying a patch series at build time.

```text
fork:       https://github.com/tterrasson/llama.cpp-retrograd.git
submodule:  crates/retrograd-ffi/runtime/vendor/llama.cpp
metadata:   crates/retrograd-ffi/runtime/llama.cpp.lock (upstream base + patch contribution status)
upstream:   https://github.com/ggml-org/llama.cpp.git
branch:     retrograd/main
```

The pinned fork commit is the submodule gitlink recorded in this repository's
history; `crates/retrograd-ffi/runtime/llama.cpp.lock` tracks the upstream commit the fork descends
from. `build.rs` performs an out-of-source CMake build of the checkout. In CI
(or with `RETRO_STRICT_LLAMA=1`) the build requires a clean checkout at the
pinned commit; locally it only warns, so the fork can be edited in place.

## Fork policy

- Keep one Git commit per patch family, named `retro(<family>): <what>`, with
  its tests and an upstream status recorded in the lockfile; a change to a family
  is a `fixup!` of its commit until the next sync folds it in.
- Keep the Rust-facing ABI in `crates/retrograd-ffi/runtime/include/retro_lora_train.h`; do not leak
  upstream C++ objects through it.
- Put integration and model-profile code in `runtime/src/retro_*`. Changes to
  upstream `ggml`, model graph construction, or a hardware backend belong in
  the fork, never in a file overlay.
- When upstream accepts a delta, drop its fork commit during the next rebase.

## Editing the fork

1. Run `scripts/setup-llama-cpp.sh`; it initializes the submodule on the first
   run, checks out the `retrograd/main` branch, and adds the `upstream` remote.
   A later run leaves an existing checkout where it is and only reports a
   divergence from the pinned commit; `--pin` is what resets it.
2. Edit and build: `cargo build` picks up in-place changes to the vendored
   sources and prints a warning while the checkout is dirty.
3. Commit in `crates/retrograd-ffi/runtime/vendor/llama.cpp` (a new family, or a
   `fixup!` of an existing one), then
   run `scripts/push-llama-cpp-fork.sh` - it pushes the fork branch and
   commits the submodule pointer bump in retrograd.

## Upstream update workflow

1. Run `scripts/update-llama-cpp.sh upstream/master` and resolve rebase
   conflicts in `crates/retrograd-ffi/runtime/vendor/llama.cpp`.
2. Run `cargo test` and applicable CPU/Metal model checks. Audit every rebased
   file under `ggml/src/ggml-cuda`, then run the serialized CUDA and fused-CE
   lanes on the NVIDIA runner.
3. Update `upstream_commit` and contribution statuses in
   `crates/retrograd-ffi/runtime/llama.cpp.lock`.
4. Publish with `scripts/push-llama-cpp-fork.sh --force-with-lease` (a rebase
   rewrites history; ordinary fast-forward publishes do not need the flag).
   `scripts/check-llama-cpp-integration.sh` verifies the final state.

## Current fork commits

The fork is one commit per patch family on top of the upstream base:
`git -C crates/retrograd-ffi/runtime/vendor/llama.cpp log --oneline upstream/master..HEAD`.

The authoritative list is the `[upstream_status]` section of
`crates/retrograd-ffi/runtime/llama.cpp.lock`. Every family there declares a
disposition - `upstream` with its PR, `proposable` with its odds of acceptance,
or `retrograd` for what is specific to this product and stays patch - plus a
`probe`, the symbol whose appearance upstream would make the family redundant.
`scripts/check-llama-cpp-integration.sh` enforces that every family declares
one, and `--upstream-status` runs the probes.
