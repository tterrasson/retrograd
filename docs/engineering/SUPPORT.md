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
| Gefen | ✅ | ✅ | ❌ | ❌ |

Gefen's two update phases are only implemented on CPU and Metal. A run asks
the live device whether it has them (`cap_opt_step_device` in the capability
report) and is refused at preflight on Vulkan or CUDA, rather than silently
falling back to a copy that would leave the real optimizer state stale.

## Kernels and execution

| Capability | CPU | Metal | Vulkan | CUDA |
|---|---|---|---|---|
| Differentiable flash attention | ➖ | ❌ [1] | ✅ | 🟡 [2] |
| Chunked cross-entropy | ✅ | 🟡 [3] | ✅ | ✅ |
| Device-side sampling | ➖ | 🟡 [4] | 🟡 [4] | 🟡 [4] |
| Generated RIR kernels | ❌ [5] | 🟡 [6] | 🟡 [6] | 🟡 [7] |
| Multi-GPU | ➖ | ❌ | ❌ | ❌ [8] |
| Automatic CI lane | ✅ [9] | ⚠️ | ⚠️ | ⚠️ |

- **[1]** Metal has no differentiable flash-attention path; it runs a
  materialized F32 backward graph instead.
- **[2]** CUDA covers `head_dim` 64, 128 and 256 only.
- **[3]** Chunked cross-entropy is off by default on Metal; two nodes fall back
  to the CPU.
- **[4]** Advertised per `ggml_backend_dev_supports_op`, never by a backend-name
  comparison.
- **[5]** CPU is always native and serves as the Loop IR oracle; it never runs a
  generated kernel.
- **[6]** Per-kernel policy in the RIR registry
  (`crates/rir-kernels/src/integration.rs`): a generated kernel may replace a
  native one.
- **[7]** Native by default; a kernel is promoted only after parity and timing
  validation, per a spec.
- **[8]** Single device (`CUDA0`); no sharding, NCCL or peer-copy.
- **[9]** CPU-only features, fast Rust and Python lanes. A *manual* row is run
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
