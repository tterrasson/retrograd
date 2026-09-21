# Support matrix

What Retrograd claims per backend, and what it does not. A capability is
listed where it is verified, not where a code path merely exists. A row
marked *manual* has no automatic coverage: it is run by hand on release, per
[Release criteria](#release-criteria) below.

Backends are chosen at build time with Cargo features (`metal`, `vulkan`,
`cuda`; CPU is always included, and `platform-gpu` defaults to Metal on macOS
for the root and Python packages, CPU elsewhere). `metal` is macOS-only and
`cuda` is refused there.

The authoritative, per-build answer is `GET /v1/capabilities`: it reports
what this binary actually registered, on the device it is actually running
on. The tables below are the general shape; that endpoint is the specific
one.

## Training algorithms

| Capability | CPU | Metal | Vulkan | CUDA |
|---|---|---|---|---|
| LoRA training (SFT) | ✅ | ✅ | ✅ | ✅ |
| Base-weight training (full / partial / hybrid) | ✅ | ✅ | ✅ | ✅ |
| PPO / GRPO / agentic GRPO | ✅ | ✅ | ✅ | ✅ |
| Recurrent and hybrid model families | ✅ | ✅ | ✅ | ✅ |

## Optimizers

| Optimizer | CPU | Metal | Vulkan | CUDA |
|---|---|---|---|---|
| AdamW | ✅ | ✅ | ✅ | ✅ |
| SGD | ✅ | ✅ | ✅ | ✅ |
| Muon | ✅ | ✅ | ✅ | ✅ |
| Gefen | ✅ | ✅ | 🟡 [a] | ✅ |

Gefen's two update phases are implemented on every backend. A run still asks
the live device whether it has them (`cap_opt_step_device` in the capability
report) and is refused at preflight where it does not, rather than silently
falling back to a copy that would leave the real optimizer state stale.

- **[a]** Vulkan carries both phases, with one restriction under
  `variant = "quantized_m"`: the block size must be a multiple of four.
  `shared_v` has no such restriction. The default (1024) satisfies it, and a
  run that asks for a non-multiple is refused at preflight.

Both optimizers are correct on every backend and neither is yet recommended
over AdamW: see [Optimizer cost and quality](optims/OPTIMIZERS.md) for what
they were measured to cost and to approximate.

## Kernels and execution

| Capability | CPU | Metal | Vulkan | CUDA |
|---|---|---|---|---|
| Differentiable flash attention | ➖ [1] | 🟡 [2] | ✅ [3] | 🟡 [4] |
| Chunked cross-entropy | ✅ | ✅ [5] | ✅ [5] | ✅ [5] |
| Device-side sampling | ➖ | 🟡 [6] | 🟡 [6] | 🟡 [6] |
| Generated RIR kernels | ❌ [7] | 🟡 [8] | 🟡 [8] | 🟡 [9] |
| Multi-GPU | ➖ | ❌ | ❌ | ❌ [10] |
| Automatic CI lane | ✅ [11] | ⚠️ | ⚠️ | ⚠️ |

- **[1]** GPU only: a CPU run builds the materialized F32 backward graph, and
  so does any GPU shape the device declines. What follows is probed per model
  at load time (`supports_flash_attn_back`), never read off a backend name.
- **[2]** Metal stops at `head_dim` 256 (`GGML_METAL_FA_BACK_MAX_D`), K/V in
  F16 or F32, no attention sinks.
- **[3]** Vulkan reaches `head_dim` 512 and is the only backend accepting
  attention sinks - the only one with a probe exercising them. It needs a
  32-wide subgroup.
- **[4]** CUDA reaches `head_dim` 512 (`FA_BACK_MAX_D`), K/V in F16 or F32, no
  attention sinks.
- **[5]** On by default everywhere, and every backend ships both fused nodes -
  but they are probed against the model's head geometry and weight type
  (`cap_fused_sparse_ce`). A device that declines either sends the whole loss
  tail to the CPU, which the backend report names
  (`chunked_cross_entropy_status: cpu_fallback`) and
  `RETRO_REQUIRE_GPU_RESIDENT=1` turns into a preflight failure.
- **[6]** Advertised per `ggml_backend_dev_supports_op`, never by a backend-name
  comparison.
- **[7]** CPU is always native and serves as the Loop IR oracle; it never runs a
  generated kernel.
- **[8]** Per-kernel policy in the RIR registry
  (`crates/rir-kernels/src/integration.rs`): a generated kernel may replace a
  native one, and two families (`L2_NORM_BACK`, `RMS_NORM_BACK`) have had
  theirs retired from the fork outright.
- **[9]** Same registry, one gate more: a CUDA policy above native-only has to
  be a line in `CUDA_ADMITTED` (`crates/rir-gen/src/validate.rs`), with its
  measurement. Two ops are promoted there today, `L2_NORM_BACK` and `SCALE`;
  the rest stay native or in observe.
- **[10]** Single device (`CUDA0`); no sharding, NCCL or peer-copy.
- **[11]** CPU-only features, fast Rust and Python lanes. A *manual* row is run
  by hand on the reference hardware (§Release criteria).

Recurrent and hybrid model families train correctly everywhere; what varies
per family is whether they can pack several sequences into one micro-batch
row, which is a throughput property, not a correctness one, and is reported
per model at load time rather than declared here.

## Release criteria

An artifact is releasable when, for the backends it claims:

1. the fast lanes and the lane of every touched subsystem pass
   (`docs/engineering/tests/notice.md`);
2. a claimed CUDA build pins `RETRO_CUDA_ARCHITECTURES` to an explicit list -
   the default `native` produces a binary that only loads on the build
   machine's GPU generation, and the build warns about it in release;
3. every *manual* row above has been run by hand on the reference hardware -
   GitHub Actions runs the CPU fast lanes only - and the result is recorded
   (`docs/engineering/cuda/STATUS.md` for the CUDA lane).
