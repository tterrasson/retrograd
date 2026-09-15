# Changelog

All notable changes to Retrograd are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/): while the version is 0.x, a minor
release may break the configuration format or the APIs.

## [0.1.0] - Unreleased

First public release.

### Added

- LoRA training directly on GGUF base models through a `llama.cpp`/`ggml` fork,
  with the base tensors frozen and the adapter exported as a standalone GGUF
  that `llama_adapter_lora_init()` reloads.
- Training algorithms: SFT, PPO, GRPO, distillation from a teacher model, and
  multi-turn agentic GRPO.
- Backends: CPU, and on GPU Metal, Vulkan and CUDA, with the backward pass on
  the device.
- CLI: `inspect`, `preflight`, `train` (with `--resume`), `bench` and `chat`.
- HTTP control plane for training runs (`retrograd-server`).
- Python package wrapping the Rust trainer through a PyO3 extension.
- RIR, the kernel IR and generator that produces the Metal, Vulkan and CUDA
  kernels vendored into the fork.

[0.1.0]: https://github.com/tterrasson/retrograd/releases/tag/v0.1.0
