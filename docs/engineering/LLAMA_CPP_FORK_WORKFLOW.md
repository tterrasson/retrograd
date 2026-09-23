# Managing the llama.cpp fork day to day

The fork [`tterrasson/llama.cpp-retrograd`](https://github.com/tterrasson/llama.cpp-retrograd)
is a **git submodule** mounted at `crates/retrograd-ffi/runtime/vendor/llama.cpp`, on branch
`retrograd/main` (= `upstream/master` + the Retrograd patch stack). The pinned
commit is the gitlink recorded in retrograd's history;
`crates/retrograd-ffi/runtime/llama.cpp.lock` only keeps the upstream base and each patch's
contribution status.

## Setup and sync

```sh
scripts/setup-llama-cpp.sh
```

On first run, the script fetches and initializes the submodule at the pinned
commit. On subsequent runs, it syncs the checkout to the pinned commit,
switches to branch `retrograd/main`, and adds the `upstream` remote
(`ggml-org/llama.cpp`).

## Daily dev loop

Edit sources directly under `crates/retrograd-ffi/runtime/vendor/llama.cpp` (Metal kernels,
ggml ops, …), then:

```sh
cargo build          # detects the changes and rebuilds incrementally
cargo test           # + the relevant CPU/Metal smoke checks
```

While the checkout is modified, the build shows a warning
(`building a modified llama.cpp fork checkout`) and the define
`RETRO_LLAMA_CPP_COMMIT` gets a `-dirty` suffix. That's the normal
development mode. In CI - or locally with `RETRO_STRICT_LLAMA=1` - the build
instead requires a clean checkout at the pinned commit.

## Committing and publishing a fork change

The fork is a linear series with **one commit per patch family**, named
`retro(<family>): <what>` after its `[upstream_status]` key in
`crates/retrograd-ffi/runtime/llama.cpp.lock` (see `RETRO_FORK.md` at the root
of the fork). A change to an existing family is a fixup of its commit, folded
in at the next upstream sync; a new family is a new commit and a new lockfile
entry.

```sh
# 1a. change an existing family: a fixup, folded at the next sync
git -C crates/retrograd-ffi/runtime/vendor/llama.cpp add -p
git -C crates/retrograd-ffi/runtime/vendor/llama.cpp commit \
    --fixup="$(git -C crates/retrograd-ffi/runtime/vendor/llama.cpp log -1 --format=%H \
               --grep '^retro(<family>)' upstream/master..HEAD)"

# 1b. or a new family: a new commit ...
git -C crates/retrograd-ffi/runtime/vendor/llama.cpp commit -m "retro(<family>): ..."
#     ... and its [upstream_status] entry in crates/retrograd-ffi/runtime/llama.cpp.lock

# 2. publish the fork + bump the pointer in one command
scripts/push-llama-cpp-fork.sh
```

The script pushes `retrograd/main` to the fork (fast-forward only), then
commits and pushes the submodule bump and the lockfile in retrograd.
`--dry-run` shows what would be done, `--yes` skips confirmation.

To commit the bump by hand instead of via the script:

```sh
git add .gitmodules crates/retrograd-ffi/runtime/vendor/llama.cpp crates/retrograd-ffi/runtime/llama.cpp.lock
git commit -m "chore: pin llama.cpp fork to $(git -C crates/retrograd-ffi/runtime/vendor/llama.cpp rev-parse HEAD)"
```

## Rebasing / syncing against upstream

```sh
scripts/update-llama-cpp.sh                  # rebase onto upstream/master
scripts/update-llama-cpp.sh upstream/b1234   # or a specific revision
```

The rebase folds pending `fixup!` commits into their family (`--autosquash`),
so the series comes out with one commit per family again.

In case of conflicts: resolve them in `crates/retrograd-ffi/runtime/vendor/llama.cpp`, then
`git add` + `git rebase --continue` (or `git rebase --abort` to back out).
Inspect the rebased stack:

```sh
git -C crates/retrograd-ffi/runtime/vendor/llama.cpp log --oneline upstream/master..HEAD
git -C crates/retrograd-ffi/runtime/vendor/llama.cpp diff upstream/master...HEAD
```

Then:

1. `cargo test` + the relevant CPU/Metal smoke checks; explicitly audit the
   deltas under `ggml/src/ggml-cuda`, then run on the NVIDIA runner:
   `cargo test --features cuda --test cuda_backend -- --test-threads=1`
   and `cargo test --test fused_ce -- --test-threads=1`;
2. update `upstream_commit` in `crates/retrograd-ffi/runtime/llama.cpp.lock`
   (`git -C crates/retrograd-ffi/runtime/vendor/llama.cpp rev-parse upstream/master`) and remove
   from `[upstream_status]` the patches absorbed by upstream;
3. publish - a rebase rewrites history, so:

```sh
scripts/push-llama-cpp-fork.sh --force-with-lease
```

## Checking / repairing state

```sh
scripts/check-llama-cpp-integration.sh    # pin, cleanliness, origin, upstream ancestry
git submodule status                      # + before the SHA = uncommitted pointer
scripts/setup-llama-cpp.sh --pin          # revert to the pinned commit (refuses if modified)
```

Common cases:

- **`git status` shows `modified: crates/retrograd-ffi/runtime/vendor/llama.cpp (new commits)`** -
  the fork has advanced locally; that's the "bump" step to commit (or
  `--pin` to revert).
- **Detached checkout after a `git submodule update`** - rerun
  `scripts/setup-llama-cpp.sh` to get back on branch `retrograd/main`.
- **After a `git pull` of retrograd that moves the pin** -
  `git submodule update` (or `scripts/setup-llama-cpp.sh --pin`) to align
  the checkout.
