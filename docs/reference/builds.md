# Build variants

The backend set is selected when Cargo builds the project. Setting
`RETRO_BACKENDS` only when launching an existing binary cannot add a backend
that was not compiled into it. CPU is always compiled; list any GPU backend
alongside it in `RETRO_BACKENDS`.

| Backend | Supported targets | Build selection |
| --- | --- | --- |
| CPU | all | always available |
| CUDA | Linux and Windows with NVIDIA CUDA | `RETRO_BACKENDS=cpu,cuda` |
| Vulkan | platforms with a usable Vulkan SDK and driver | `RETRO_BACKENDS=cpu,vulkan` |
| Metal | macOS only | `RETRO_BACKENDS=cpu,metal` (the macOS default) |

At runtime, use `--device auto` to prefer an available compiled GPU and fall
back to CPU, `--device cpu` to force CPU execution, or `--device gpu` to fail
when no compiled GPU backend can be used. Backend names are not runtime device
selectors: a binary with more than one GPU backend uses the first GPU device
the runtime enumerates.

## Default build

```bash
cargo build --release
```

CPU is always available. On macOS, the default build enables Metal. On other
systems, the default build is CPU-only.

Check the resulting binary and its backend capabilities before training:

```bash
./target/release/retrograd inspect --model /models/base.gguf --device auto
./target/release/retrograd preflight --model /models/base.gguf --device auto
```

## CUDA

CUDA requires Linux or Windows, an NVIDIA GPU, a compatible driver, and the
CUDA Toolkit with `nvcc`. CUDA is not supported on macOS.

For a local build, target the compute capability of the installed GPU:

```bash
RETRO_BACKENDS=cpu,cuda \
RETRO_CUDA_ARCHITECTURES=89-real \
  cargo build --release
```

For a distributable build, list every target that must load the binary instead
of using `native`:

```bash
RETRO_BACKENDS=cpu,cuda \
RETRO_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real" \
  cargo build --release
```

`RETRO_CUDA_ARCHITECTURES` defaults to `native`, which is convenient locally
but produces a release binary only for the GPU architectures visible during
the build. Pin explicit architectures for an artifact that must run elsewhere.

CUDA Graphs are opt-in:

```bash
RETRO_BACKENDS=cpu,cuda \
RETRO_CUDA_ARCHITECTURES=89-real \
RETRO_CUDA_GRAPHS=1 \
  cargo build --release
```

The current CUDA implementation is single-device. What runs on CUDA, kernel by
kernel, is in [CUDA status](../engineering/cuda/STATUS). Before a long run,
verify that the compiled binary can register the GPU:

```bash
./target/release/retrograd preflight \
  --model /models/base.gguf --device gpu
```

## Vulkan

Vulkan requires a Vulkan SDK with the loader, headers, `glslc`, and SPIR-V
headers. Build the binary with CPU and Vulkan support:

```bash
RETRO_BACKENDS=cpu,vulkan cargo build --release
```

On macOS, MoltenVK supplies the Vulkan implementation. If the loader does not
find the ICD, expose it before running the binary:

```bash
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
```

Verify the selected device:

```bash
./target/release/retrograd preflight \
  --model /models/base.gguf --device gpu
```

If the command fails, the binary either lacks the Vulkan backend or the loader
cannot register a Vulkan device. Rebuild after changing `RETRO_BACKENDS`.

Adapter export and reload are validated on Vulkan; a comparison of the logits
before saving and after reloading is not implemented yet.

## Metal

Metal is supported on macOS only and uses the Metal toolchain supplied by Xcode
or the Xcode Command Line Tools. It is enabled by the default build on macOS:

```bash
cargo build --release
```

Set the backend explicitly when making a reproducible build command or when
switching back from another backend:

```bash
RETRO_BACKENDS=cpu,metal cargo build --release
```

Verify that a Metal device is available before training:

```bash
./target/release/retrograd preflight \
  --model /models/base.gguf --device gpu
```

The build rejects `metal` on non-macOS targets. To use Vulkan on a Mac instead,
select `RETRO_BACKENDS=cpu,vulkan` and configure MoltenVK as described above.

Upstream Metal has no training backward ops, so the fork carries hand-written
kernels for `SILU_BACK`, `RMS_NORM_BACK`, `L2_NORM_BACK`, `OUT_PROD` (F32 and
the common quantized `src0` types), `SSM_CONV_BACK`, `SSM_SCAN_BACK`,
`GATED_DELTA_NET_BACK`, `CONV_RS_GATHER`, `SOFT_MAX_BACK`,
`CROSS_ENTROPY_LOSS(_BACK)` and `GET_ROWS_BACK`, validated against the CPU
reference by `tests/metal_ops.rs`. The backward pass is GPU-resident; the only
CPU residue is the forward `GET_ROWS` on the CPU-resident `token_embd.weight`,
a weight-placement matter rather than a missing kernel.

## Linking the native runtime

llama.cpp and the C++ runtime beside it are linked into every Rust binary that
reaches them - the CLI, but also each of the workspace's test binaries. How they
are linked is chosen by profile and platform:

| Build | Mode | Why |
| --- | --- | --- |
| `--release`, any platform | static | the binary is relocatable: copy it anywhere and it runs |
| debug on macOS | shared | a test tree links some fifty binaries, and each static one carries the whole native side again |
| debug elsewhere | static | see below |

Set `RETRO_GGML_LINK=static` or `RETRO_GGML_LINK=shared` to override.

The mode is worth gigabytes on a machine that runs the test lanes. Measured on
the root package's 36 test and binary targets, one build tree: 375 MB of
executables static against 183 MB shared plus 11 MB of shared libraries, and
the rest of the workspace links the same runtime again. What a static build
repeats in every binary is the 2.4 MB `__ggml_metallib` blob and the C++
runtime with llama.cpp's `common/` behind it.

Shared is the default only on macOS because a Mach-O library carries its own
absolute install name: a binary linked against one finds it with no rpath and no
`DYLD_LIBRARY_PATH`. The ELF equivalent is a bare `DT_SONAME` resolved through
the loader's search path, so `RETRO_GGML_LINK=shared` on Linux links but leaves
locating the libraries to `LD_LIBRARY_PATH`.

A **release** binary is static in both cases, so nothing you ship or copy
depends on a build tree. The Python wheel is static for the same reason,
whatever the profile - `python/.cargo/config.toml` pins it.

With Metal, the link also takes `libclang_rt.osx.a` from the resource directory
of the C++ compiler (`CXX`, default `c++`). ggml-metal's `@available` checks
call `___isPlatformVersionAtLeast`, which only compiler-rt defines. A binary
gets it from the clang driver, but the Python extension is linked with
`-undefined dynamic_lookup`: without the archive it links, then fails at
`import retrograd._native`. The build stops with an explicit error when the
compiler ships no compiler-rt; Apple clang does.

## Container-enabled builds

The root package defines these relevant Cargo features:

| Feature | Purpose |
| --- | --- |
| default | Enables the agentic loop. |
| `mcp` | Adds MCP transports. |
| `container` | Enables the Docker-compatible container pool and the agentic loop. |

Build with container support using:

```bash
cargo build --release --features container
```

The container feature is required only when an `agent_grpo` configuration uses
`[agent.environment] type = "container"`. A single-turn `[grpo]` run that
calls an external reward command does not need this feature.

The runtime connects to a local Docker-compatible daemon. Docker Desktop,
Colima, and Podman in Docker-compatible mode are supported; set `DOCKER_HOST`
when the daemon is not on the default socket.
