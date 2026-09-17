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

NCCL is always disabled.

## What works on the GPU

The backward kernels below run on CUDA. Anything not listed falls back to the
CPU.

| Kernel | Supported |
|---|---|
| `OUT_PROD` | F32, F16 and 23 quantized types |
| `FUSED_SPARSE_CE` and its backward | F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K to Q6_K |
| `FLASH_ATTN_BACK` | `head_dim` 64, 128 or 256; K/V in F16 or F32; GQA; softcap |
| `SSM_CONV_BACK`, `SSM_SCAN_BACK` | contiguous F32 |
| `SILU_BACK`, `RMS_NORM_BACK`, `SOFT_MAX_BACK`, `GET_ROWS_BACK`, `CROSS_ENTROPY_LOSS_BACK` | yes |
| `CONV_RS_GATHER` | yes, used when `RETRO_RECURRENT_ROLLBACK=auto` |
| AdamW | F16 parameters, with stochastic rounding identical to the CPU |

A `head_dim` other than 64, 128 or 256 uses the CPU path for flash attention
backward.

The runtime detects these capabilities from the device itself.
`backend_report` shows them in `gpu_device`, `cap_flash_attn_back` and
`cap_device_sampling`.

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
- **Known failing test, not CUDA related.**
  `checkpoint_resume::restoring_the_optimizer_state_tracks_the_uninterrupted_run_more_closely`
  also fails on CPU.

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
```

Replace `89-real` with the compute capability of your GPU.
