# CUDA implementation status

Reference environment: Linux x86_64, RTX 4090 (compute capability 8.9),
CUDA Toolkit 13.3 (`nvcc`), driver 610.43, a single GPU detected - this
delivery is therefore explicitly **single-GPU** (`CUDA0`, no sharding/NCCL/peer-copy).

## What is delivered and verified

- **Build.** `Backends { cuda }` is recognized in `RETRO_BACKENDS`, macOS is
  rejected with a clear message. `cargo:rustc-check-cfg=cfg(retro_cuda)` is
  only emitted when the backend is actually built and linked. Options:
  `RETRO_CUDA_ARCHITECTURES` (default `native`), `RETRO_CUDA_GRAPHS` (default
  off), `-DGGML_CUDA_NCCL=OFF`. The root package and the FFI share a single
  implementation of `RETRO_BACKENDS` selection
  (`crates/retrograd-ffi/build/backend_selection.rs`), so both `build.rs`
  files emit the same `cfg`s with the same defaults.
- **Backend-name-independent runtime contract.** Capabilities
  (`supports_flash_attn_back`, `supports_device_sampling`, …) rely on
  `ggml_backend_dev_supports_op`, never on a text comparison against `Vulkan`
  or `CUDA`. `backend_report` publishes `gpu_device`, `cap_flash_attn_back`,
  `cap_device_sampling`.
- **Probe suite** (`tests/cuda_backend.rs`, mirroring `vulkan_backend.rs`):
  CPU↔CUDA passing for `SILU_BACK`, `RMS_NORM_BACK`, `SOFT_MAX_BACK`,
  `GET_ROWS_BACK`, `OUT_PROD` (F32 and quantized), `CROSS_ENTROPY_LOSS_BACK`,
  `FLASH_ATTN_BACK`, `SSM_CONV_BACK`, `SSM_SCAN_BACK`, `FUSED_SPARSE_CE[_BACK]`,
  `CONV_RS_GATHER`, AdamW F16.

## Backward kernels - status by operator

| Kernel | Coverage | Notes |
|---|---|---|
| `OUT_PROD` (quantized) | 24 types (23 quantizations + F16) | direct decode into shared memory, no full F32 scratch; the tiled path and its `GGML_CUDA_DEQUANT_BUDGET_MB` budget only remain as a fallback for a type outside the table |
| `FUSED_SPARSE_CE[_BACK]` | F32, F16, Q4_0/Q4_1, Q5_0/Q5_1, Q8_0, Q2_K…Q6_K | cuBLAS GEMM per vocabulary tile (≤1024 rows), no dense `[n_vocab × n_tokens]` tensor, only `grad_h` is produced; dequantization scratch bounded to one tile |
| `FLASH_ATTN_BACK` | heads 64/128/256, K/V F16 or F32, GQA, softcap | templated by `head_dim` bucket (warp-split); the dK/dV gradient window is bounded to `n_ubatch`, not the whole cache span (`ggml_flash_attn_ext_set_grad_window`) - cost proportional to the step, not the context |
| `SSM_CONV_BACK` / `SSM_SCAN_BACK` | contiguous F32 (+ I32 ids) | one block per channel / per (group, sequence, head) |
| AdamW F16 | F16 parameter, F32 moments/gradient | stochastic rounding bit-exact with the CPU (`__float2half_rn` + the same bit manipulation as `ggml_sr_uniform`) |
| `CONV_RS_GATHER` | CPU, CUDA, Vulkan, Metal | replaces the host `view`+`cpy` loop of the recurrent-state rollback; reachable from retrograd since `n_rs_seq` became a derived value, but only with `RETRO_RECURRENT_ROLLBACK=auto` - the derivation is reported on every load, applying it is opt-in until the scorer counters justify it (`docs/engineering/SUPPORT.md`) |

An untested `head_dim` bucket (beyond 256) must **never** be advertised by
`supports_op` without a parity test at that exact dimension - a silently
wrong gradient is worse than an explicit CPU fallback.

### Invariant discovered along the way: device buffers are not guaranteed to be zeroed

CUDA training forward was producing NaNs in `CROSS_ENTROPY_LOSS` from
otherwise finite logits. Cause: the labels input (a dense tensor) was
partially rewritten at each step by an "incremental clear" that assumes a
reused buffer stays at zero between two allocations - true for a host
`malloc` never touched, false for a reused CUDA pool region. The fix is
generic: fully zero the dense labels tensor (`ggml_set_zero`, backend-aware)
rather than relying on the buffer's residual state. The same trap exists for
any device scratch assumed to be pre-zeroed - it was found and fixed a
second time in the fused CE head.

### Reparallelization of the backward kernels

The first implementations were deliberately "correctness-first" - one unit
of work per thread - then reparallelized onto one block or one warp per
unit once correctness was established, with the internal work spread over
the touched tensor's contiguous axis (`flash-attn-back.cu`,
`gated-delta-net-back.cu`, `ssm-back.cu`). The probe tolerances cover the F32
reassociation introduced by reductions and cross-unit `atomicAdd`s.

`gated-delta-net-back.cu` later received a second implementation
("chunkwise", six matrix products + one block-triangular solve per chunk of
`C` tokens instead of a token-by-token chain), selectable via an operator
parameter; the sequential kernel remains the fallback when the gates undo the
`khat = k/A` normalization. The sequential fallback remains part of the
correctness contract when that normalization is not applicable.

### Vulkan / Metal parity

All three GPU backends target the same functional parity with designs
specific to each API (subgroups on Vulkan, dynamically-sized
`threadgroup memory` on Metal). None of the three is uniformly ahead:
depending on the kernel, sometimes CUDA, sometimes Vulkan has the more
efficient design (e.g. `flash_attn_back_kv.comp` accumulates in registers
with no `atomicAdd`, whereas CUDA still does one per (Q row, key)). Vulkan
has no equivalent of CUDA/Metal's dynamic shared memory: `GDN_S_V_MAX` caps
`S_v` at compile time, which is enough for every GDN architecture in service.
Metal has no `FLASH_ATTN_BACK` path at all (falls back to the materialized
F32 backward graph).

## Validation and known limitations

The product-level view of these limitations - which backend claims what, and
which rows are manual - is [`../SUPPORT.md`](../SUPPORT.md). This section is the
CUDA detail behind it.

- **Pre-existing regression, unrelated to this work**:
  `checkpoint_resume::restoring_the_optimizer_state_tracks_the_uninterrupted_run_more_closely`
  fails on CPU with or without the CUDA changes.
- **Device VRAM instrumentation.** The memory report used to add up
  `ggml_backend_buffer` buffers and ignore each backend's own scratch (CUDA
  pools, Vulkan `prealloc_*`). `ggml_backend_scratch_bytes()` and
  `llama_opt_get_memory()` now cover this term; `device_*` remains a
  machine-wide budget (only deltas are meaningful), `scratch_*` is
  process-specific and directly assertable.
- **`checkpoint_dtype`** - checkpoint value reads may go through an F16/BF16
  round trip (`cast(cast(c, T), F32)`) instead of F32; the gradient chain
  stays intact. Measured cost: ~1e-3 relative in 16 bits, four orders of
  magnitude above an F32 round trip (bit-exact) - a last-resort lever for a
  run that would not otherwise fit, not a free gain.
- **Tiled K/V Flash Attention backward** - written and compiled, never run on
  real hardware in this iteration (no driver available on the machine that
  produced it). Verified by host simulation (bit-exact over 243 shape
  combinations).
- **CI.** The repo has no CI for the CUDA lane; it is run by hand, reference
  sequence below. The `flash-attn-back.cu` / `out-prod.cu` kernels therefore
  have no automatic coverage - track manually before any PR that touches
  them.
- **Out of scope.** Multi-GPU (no second GPU on the reference platform).

## Commands

```sh
# Build + device
RETRO_BACKENDS=cpu,cuda RETRO_CUDA_ARCHITECTURES=89-real cargo test --test backend_devices
# Probes + placement (serialized)
RETRO_BACKENDS=cpu,cuda RETRO_CUDA_ARCHITECTURES=89-real cargo test --test cuda_backend -- --test-threads=1
# Same lane with CUDA Graphs capture
RETRO_BACKENDS=cpu,cuda RETRO_CUDA_ARCHITECTURES=89-real RETRO_CUDA_GRAPHS=1 cargo test --test cuda_backend -- --test-threads=1
# CPU non-regression
RETRO_BACKENDS=cpu cargo test
```
