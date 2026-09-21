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

## Base-weight storage precision

Base training reads the weights at the precision the GGUF stores them at and
writes them back there. Nothing is converted, at load or at checkpoint. Below
F32 the update rounds stochastically instead of to nearest, which is what
keeps a step smaller than half a grid point from being discarded; there is no
F32 master copy, because one costs more per parameter than training in F32
would.

| Stored as | CPU | Metal | Vulkan | CUDA |
|---|---|---|---|---|
| F32 | ✅ | ✅ | ✅ | ✅ |
| F16, under AdamW or SGD | ✅ | ❌ [c] | ❌ [c] | ✅ |
| BF16, under AdamW or SGD | ✅ | ❌ [c] [e] | ❌ [c] [e] | ✅ |
| F16 or BF16, under Muon or Gefen | ❌ [d] | ❌ [d] | ❌ [d] | ❌ [d] |

A cell is one row of `BASE_DTYPE_TABLE`
(`crates/retrograd-core/src/base_dtype.rs`), indexed by (dtype, optimizer,
backend): a row is a measurement, run by `tests/f16_base_training.rs` with the
tolerances read from the row. A combination with no row is refused before the
graph is built, by dtype, optimizer and backend. `cap_opt_step` in the backend
report is the other half of the answer, which storages the live device runs
each step on; the table declares and the device is asked, and either can
refuse.

- **[c]** The update kernel exists on both and an exact CPU-against-device
  equality test holds it to the CPU's rounding, but no lane has run a model on
either, so no row admits them. The Metal test has not been run at all, for
  want of a machine.
- **[d]** Both steps write F32 parameters only: Muon's ends in a
  Newton-Schulz orthogonalization, Gefen's already approximates the first
  moment, and a rounded store would stack a second approximation on the
  first. The refusal says that, not an empty list of backends.
- **[e]** Neither backend decodes a BF16 weight in `OUT_PROD`, which an
  activation gradient needs; the CPU and CUDA do. A row needs that too.

BF16 keeps 8 significand bits against F16's 11, so its grid is eight times
coarser and the measured gaps follow: ten times the F16 ones on the CPU,
three times on CUDA, where BF16's own grid then dominates the reduction-order
difference. Stochastic rounding keeps the difference a rounding rather than a
bias.

The optimizer matters too. An AdamW step is wider than the grid, so it mostly
lands on the same grid point whichever backend computed the gradient; an SGD
step is the gradient itself, about one grid point wide, so a small difference
between backends decides which side of a point each element lands on. That is
why the SGD rows' outlier fractions move more between CPU and CUDA than
AdamW's do.

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
