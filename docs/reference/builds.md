# Build variants

GPU backends are chosen at build time with Cargo features. The CPU backend is
always included.

| Backend | Platform | Build command |
| --- | --- | --- |
| Metal | macOS | `cargo build --release` (default on macOS) |
| CPU only | all | `cargo build --release --no-default-features --features cli,agent` |
| CUDA | Linux, Windows | `cargo build --release --features cuda` |
| Vulkan | any with a Vulkan SDK | `cargo build --release --features vulkan` |

Naming a backend replaces the default one: `--features vulkan` on a Mac builds
Vulkan without Metal. Combine them with a comma (`--features metal,vulkan`).
Outside macOS, the default build is CPU-only.

At run time, `--device auto` uses a GPU when the binary has one and the machine
provides it, and falls back to the CPU. Check what a binary can do on a given
model before a long run:

```bash
./target/release/retrograd preflight --model /models/base.gguf --device gpu
```

What each backend supports is in the
[support matrix](../engineering/SUPPORT).

## Optional features

| Feature | Adds |
| --- | --- |
| `mcp` | MCP servers as tools for [agentic GRPO](../training/agent). |
| `container` | Docker/Podman sandboxes for agentic GRPO. |

`cli` (the command-line tools) and `agent` (agentic GRPO) are enabled by
default; keep them when you pass `--no-default-features`.

## CUDA

Requires an NVIDIA GPU, its driver and the CUDA Toolkit (`nvcc`). Not
available on macOS. Single GPU only.

By default the build targets the GPUs present on the build machine. To build a
binary for other machines, list the compute capabilities:

```bash
RETRO_CUDA_ARCHITECTURES="75-real;80-real;86-real;89-real" \
  cargo build --release --features cuda
```

`RETRO_CUDA_GRAPHS=1` at build time enables CUDA Graphs. Details are in
[CUDA status](../engineering/cuda/STATUS).

## Vulkan

Requires a Vulkan SDK with the loader, headers, `glslc` and SPIR-V headers.

On macOS, Vulkan runs through MoltenVK. If no device is found, point the
loader at it:

```bash
export VK_ICD_FILENAMES="$(brew --prefix molten-vk)/etc/vulkan/icd.d/MoltenVK_icd.json"
```

## Metal

Requires Xcode or the Xcode Command Line Tools. Enabled by default on macOS;
`--features metal` names it explicitly.

## Other build variables

| Variable | Effect |
| --- | --- |
| `RETRO_NATIVE=1` | Optimize CPU kernels for the build machine. The binary may not run elsewhere. |
| `RETRO_GGML_LINK=static\|shared` | How the native runtime is linked. Release builds are static, so the binary can be copied anywhere. |

Changing one of these variables rebuilds the native runtime in the same
`target/` directory. Keep them constant, or use a separate `CARGO_TARGET_DIR`
per variant, to avoid rebuilding back and forth.
