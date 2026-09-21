# CUDA status

CUDA is supported for training and inference on Linux, on a single GPU.
It is not available on macOS.

Tested on: Linux x86_64, RTX 4090 (compute capability 8.9), CUDA Toolkit 13.3,
driver 610.43.

For the comparison with CPU, Metal and Vulkan, see [SUPPORT.md](../SUPPORT.md).

## Building

Enable the Cargo feature `cuda`:

```sh
cargo build --features cuda
```

| Variable | Default | Effect |
|---|---|---|
| `RETRO_CUDA_ARCHITECTURES` | `native` | GPU architectures to compile for, for example `89-real` |
| `RETRO_CUDA_GRAPHS` | off | Enable CUDA Graphs capture |
| `GGML_CUDA_DEQUANT_BUDGET_MB` | 64 | F32 scratch ceiling of the sliced `OUT_PROD` dequantization; 0 or less means no slicing |

NCCL is always disabled.

## What works on the GPU

The backward kernels below run on CUDA. Anything not listed falls back to the
CPU.

| Kernel | Supported |
|---|---|
| `OUT_PROD` | F32, F16, BF16 and 23 quantized types |
| `FUSED_SPARSE_CE` and its backward | F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K |
| `FLASH_ATTN_BACK` | `head_dim` up to 512; K/V in F16 or F32; GQA; softcap; no attention sinks |
| `SSM_CONV_BACK`, `SSM_SCAN_BACK` | contiguous F32 |
| `SILU_BACK`, `RMS_NORM_BACK`, `SOFT_MAX_BACK`, `GET_ROWS_BACK`, `CROSS_ENTROPY_LOSS_BACK` | yes |
| `CONV_RS_GATHER` | yes, used when `RETRO_RECURRENT_ROLLBACK=auto` |
| AdamW, SGD | F16 and BF16 parameters, with stochastic rounding identical to the CPU |

The kernel picks the smallest register bucket covering the head dimension -
128, 256 or 512 (`FA_BACK_MAX_D`); a wider head, or a model with attention
sinks, takes the materialized F32 backward graph instead.

The runtime detects these capabilities from the device itself.
`backend_report` shows them in `gpu_device`, `cap_flash_attn_back` and
`cap_device_sampling`.

## Base weights in F16 and BF16

A GGUF whose matrices are stored as F16 or BF16 trains on CUDA in place, under
AdamW or SGD, with no conversion of the file and no F32 master copy. The
gradient and both AdamW moments stay F32; only the parameter is half, and the
update is rounded stochastically from a stream seeded by the optimizer's
iteration counter, so a resume lands on the same weights bit for bit. The
activation gradient of such a weight is `out_prod(W, dy)`, decoded through the
type-generic `to_fp32` path.

Measured by `tests/f16_base_training.rs` on each generated fixture against its
own F32 twin, on the hardware named at the top of this page. Under AdamW:

| | CPU / F16 | CUDA / F16 | CPU / BF16 | CUDA / BF16 |
|---|---|---|---|---|
| worst relative gradient gap, one step over 24576 elements | 5.5e-4 | 1.7e-3 | 5.5e-3 | 4.9e-3 |
| elements whose update is more than one grid point from the F32 trajectory | 1.2e-3 | 2.1e-3 | 2.3e-3 | 2.2e-3 |
| relative loss gap after 2000 steps | 2.1e-4 | 7.9e-4 | 5.9e-3 | 1.5e-3 |

And under SGD, whose step is the gradient itself rather than a normalized one:

| | CPU / F16 | CUDA / F16 | CPU / BF16 | CUDA / BF16 |
|---|---|---|---|---|
| worst relative gradient gap, one step over 24576 elements | 5.5e-4 | 1.7e-3 | 5.5e-3 | 4.9e-3 |
| elements whose update is more than one grid point from the F32 trajectory | 5.7e-4 | 3.3e-3 | 8.5e-4 | 6.5e-4 |
| relative loss gap after 2000 steps | 2.2e-3 | 2.3e-5 | 6.3e-3 | 5.0e-3 |

The gradient row is the same under both optimizers: it comes out of the
forward, which does not know which step will read it. In F16, CUDA's gaps are
about three times the CPU's because the forward feeds them differently;
in BF16 the two backends land together, since the storage costs an order of
magnitude more than that difference does.

Each kernel's rounding is compared to the CPU's element for element, through
the op probe with no model at all, by
`{f16,bf16}_adamw_cuda_kernel_matches_cpu` and
`half_precision_sgd_cuda_kernel_matches_cpu` in `tests/cuda_backend.rs`:
equality, not a tolerance. All four kernels share the same helpers
(`ggml/src/ggml-cuda/retro-stochastic-round.cuh`), so the rounding is written
once for the backend.

## Limitations

- **One GPU only.** The runtime uses `CUDA0`. No multi-GPU, sharding or
  peer-copy.
- **No CI.** CUDA tests are run by hand with the commands below. Run them
  before a PR that touches `flash-attn-back.cu` or `out-prod.cu`.
- **`checkpoint_dtype` F16/BF16** saves memory but costs about 1e-3 relative
  error. Use it only when a run does not fit otherwise.
- **VRAM numbers.** In the memory report, `scratch_*` is the memory used by
  this process. `device_*` is for the whole GPU, so only compare it between
  two points in time.
- **A quantized `OUT_PROD` does not always decode in place.** Wide
  projections use a dequantize+SGEMM path whose F32 scratch is bounded by
  `GGML_CUDA_DEQUANT_BUDGET_MB`; small and strided outputs keep the
  scratch-free in-place decoder. So the budget moves a step's scratch peak by
  design, and it bounds the scratch without changing any gradient.

## Running the tests

```sh
# Device detection
RETRO_CUDA_ARCHITECTURES=89-real cargo test --features cuda --test backend_devices

# Kernel parity against the CPU
RETRO_CUDA_ARCHITECTURES=89-real cargo test --features cuda --test cuda_backend -- --test-threads=1

# Same, with CUDA Graphs
RETRO_CUDA_ARCHITECTURES=89-real RETRO_CUDA_GRAPHS=1 cargo test --features cuda --test cuda_backend -- --test-threads=1

# CPU non-regression
cargo test --no-default-features --features agent

# Half-precision base weights, CPU column and CUDA column in one run
RETRO_TINY_FIXTURE=tests/fixtures/retrograd-tiny-qwen2-f32.gguf \
RETRO_TINY_F16_FIXTURE=tests/fixtures/retrograd-tiny-qwen2-f16.gguf \
RETRO_TINY_BF16_FIXTURE=tests/fixtures/retrograd-tiny-qwen2-bf16.gguf \
RETRO_TINY_BF16_CONTROL_FIXTURE=tests/fixtures/retrograd-tiny-qwen2-bf16ctl-f32.gguf \
  cargo test --features cuda --test f16_base_training -- --test-threads=1 --nocapture

# The device-memory guard (discrete GPU only): on unified memory a "device"
# allocation is a host allocation.
RETRO_REQUIRE_GPU_RESIDENT=1 \
RETRO_CPU_FIXTURE=tests/fixtures/LFM2.5-230M-Q4_K_M.gguf \
RETRO_TINY_FIXTURE=tests/fixtures/retrograd-tiny-qwen2-f32.gguf \
  cargo test --release --features cuda --test device_memory --test gefen_ops -- --test-threads=1
```

Replace `89-real` with the compute capability of your GPU.
