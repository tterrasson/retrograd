# Support matrix

What Retrograd claims per backend, and what it does not. The point of this page
is that a capability is named where it is verified, rather than implied by the
existence of a code path elsewhere. A row that says *manual* is a row with no
automatic coverage: it is run by hand, and it is a release criterion (§Release
criteria) rather than a guarantee.

Backends are chosen at build time with `RETRO_BACKENDS` (`cpu`, `metal`,
`vulkan`, `cuda`; default `cpu,metal` on macOS, `cpu` elsewhere). `metal` is
macOS-only and `cuda` is refused there.

## Backends

| Capability | CPU | Metal | Vulkan | CUDA |
|---|---|---|---|---|
| LoRA training (SFT) | ✅ | ✅ | ✅ | ✅ |
| PPO / GRPO / agentic GRPO | ✅ | ✅ | ✅ | ✅ |
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
- **[9]** `RETRO_BACKENDS=cpu`, fast Rust and Python lanes. A *manual* row is run
  by hand on the reference hardware (§Release criteria).

The RIR registry states a policy per kernel *and* per backend
(`crates/rir-kernels/src/integration.rs`): Metal and Vulkan are where a
generated kernel may replace a native one, CPU stays native and is the oracle,
and CUDA is `NativeOnly` except where a spec says otherwise. "RIR support" is
therefore not a single yes/no for the product - read the registry, or
`GET /v1/capabilities`, which publishes what this build actually registered.

## Known limitations

- **CPU resume regression.** The `checkpoint_resume` case comparing a restored
  optimizer state against an uninterrupted run fails on CPU. Pre-existing and
  unrelated to the CUDA work (`docs/engineering/cuda/STATUS.md`).
- **No CUDA CI.** `flash-attn-back.cu` and `out-prod.cu` have no automatic
  coverage; the reference command sequence is in `docs/engineering/cuda/STATUS.md`.
- **Recurrent and hybrid models.** No decision is taken on an architecture
  name any more, and the limitation is now stated per property rather than per
  model. Three properties, each resolved at load and published in the
  capability report:

  - *Packed multi-sequence training.* The graph declares it -
    `llama_model_supports_packed_seq()`, read in the fork at the entries to the
    packed micro-batch split - and the device is then measured
    against that declaration (a run decoded alone, then beside a neighbour, on
    the packed path; the declaration may only be downgraded). True for every
    attention-only model; among recurrent and hybrid graphs, true today for
    LFM2/LFM2-MoE only, whose ShortConv window is gathered by index. The other
    families train one sequence per row, which is correct and slower, not
    unsupported.
  - *Micro-batch finiteness.* A GPU run decodes the requested micro-batch once
    at load and checks that its logits are finite; a failure escalates to the
    full logical batch and, if that fails too, refuses the run naming both
    widths and both counts. This replaces the Falcon-H1/MoltenVK rule that
    branched on architecture name, macOS, Vulkan and `n_batch >= 16` - the
    escalation and the batch floor are now consequences of a measurement on
    the (model, driver, micro-batch) triple at hand.
  - *Recurrent-state rollback.* `n_rs_seq` is derived, not hardwired: zero
    where nothing needs it (no recurrent state, or two sequence slots, where
    the scorer drops a whole sequence), otherwise the shared-prefix scorer's
    rollback window bounded by the micro-batch (`split_equal` requires
    `n_ubatch > n_rs_seq + 1`) and by a state budget. The derivation is
    reported on every load; applying it is opt-in
    (`RETRO_RECURRENT_ROLLBACK=auto`) until the scorer counters
    (`prefix_reprefills`, `branch_evictions_refused`) have been read on a real
    run: an optimization is enabled by default only once a measurement
    justifies it.

  What remains genuinely narrow: only one recurrent family is packable, only
  the `shortconv` family is covered by a lane without an extra model
  (`tests/recurrent_families.rs` names the environment variable for each of
  the three), and the Falcon-H1 non-finite behaviour on MoltenVK is still
  present upstream - it is now detected and worked around by measurement
  rather than declared per name.
- **LoRA `auto` targets.** Capability detection covers separate `attn_q`/`attn_v`
  matrices and fused `attn_qkv`. Any other tensor layout needs explicit patterns
  (`docs/engineering/CONFIG.md`).
- **Device selection.** A run picks `auto`, `cpu` or `gpu`. There is no way to
  ask for a *family*: on a machine with several GPU backends compiled in, the
  runtime takes the first GPU device it enumerates. `metal`/`vulkan`/`cuda` are
  not accepted as device values precisely because they would not be honoured.
- **Container images.** A custom image must provide `sh`, GNU-compatible
  `find -printf`, `grep`, `ls`, `cat`, `kill`, `mkdir`, and `timeout` when the
  wrapper is on. This is checked at the first container creation and refused
  there, not at declaration.
- **Environment isolation.** `local` runs generated code with the privileges and
  network of the host process and requires `allow_unsandboxed = true`. Container
  networking, when enabled, is ordinary bridge egress with no destination
  allow-list: an egress policy has to be imposed outside Retrograd.

## Release criteria

An artifact is releasable when, for the backends it claims:

1. the fast lanes and the lane of every touched subsystem pass
   (`docs/engineering/tests/notice.md`);
2. a claimed CUDA build pins `RETRO_CUDA_ARCHITECTURES` to an explicit list -
   the default `native` produces a binary that only loads on the build
   machine's GPU generation, and the build warns about it in release;
3. every *manual* row above has been run by hand on the reference hardware -
   GitHub Actions runs the CPU fast lanes only - and the result is recorded
   (`docs/engineering/cuda/STATUS.md` for the CUDA lane);