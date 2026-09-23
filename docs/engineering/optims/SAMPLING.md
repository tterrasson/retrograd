# The GRPO sampling path (generation + behavior scoring)

Status: shipped and measured. On the reference bench (Qwen3.5-0.8B-Q6_K, Apple M1,
`updates=1`, `prompts/upd=3`, `group=4` → 12 sequences, `ctx=1024`,
`batch=ubatch=16`, `max_new_tokens=96`, longest prompt ≈ 800 tokens), generation
time dropped from 72.4 s to 11.9 s and behavior scoring from 21.9 s to 8.2 s -
total sampling-phase time 116.4 → 42.3 s, now of the same order as the
optimizer (20.4 s), which becomes the dominant cost again.

This document describes the current architecture of the sampling path and the
invariants that justify it. For the user-facing memory controls, see
[Performance and memory](/operations/performance); for test lanes, see
[`../tests/notice.md`](../tests/notice.md).

## Original finding: prefill dominates, not decoding

Dividing the generation budget by 12 (`--max-new-tokens 8` vs. 96) only saved
15% of generation time. The dominant cost was therefore not the number of
tokens sampled, but the **number of `llama_decode` launches**: each launch
rereads the model's entire quantized weights, and an 800-position prefill cost
800 full sweeps for 9,600 actually useful tokens.

## Current architecture

### Generation (continuous, grouped rollout)

The GRPO path (`generate_continuous_batch_impl` /
`generate_continuous_batch_device`, in the
`crates/retrograd-ffi/runtime/src/retro_rollout.cpp` implementation)
differs from the single-prompt path used by PPO/evaluation
(`generate_batch_impl`), which already correctly prefilled a single prompt
then copied the KV cache. The continuous path now applies the same
principles:

- **Per-sequence prefill, chunked by `n_batch`**, rather than a position wave
  that decodes one token per sequence and per launch. The physical order of
  launches is derived from **content** (prompt length, tokens, sampling
  parameters), never from the caller's submission rank: otherwise the amount
  of KV already resident when a prompt is decoded - and hence the attention
  tiling it sees - would depend on an arbitrary order, and two calls that
  differ only by a permutation of their rows would produce different results.
- **Logits requested only on the terminal row** of each prompt during
  prefill, never on intermediate positions.
- **Deduplication of identical prompts.** Members of a GRPO group share the
  same prompt by construction; only one is prefilled, the others receive a
  cache copy (`llama_memory_seq_cp`) rather than an independent prefill. The
  canonical order above already groups identical prompts next to each other,
  so detection costs only a comparison with the neighbor.
- **Generation geometry decoupled from training.** The generation context no
  longer inherits `n_batch`/`n_ubatch` from the training context (sized for
  the backward pass's activation budget). A derived default targets the
  largest batch whose logits buffer fits within a fixed memory budget (capped
  and bounded by `n_ctx`) - a fixed `n_batch` independent of vocabulary size
  would be either too small on a large vocabulary or needlessly memory-costly
  on a small one.
- **Device sampler active by default** as soon as the capability is present
  and no generation logprob is requested (`RETRO_DEVICE_SAMPLING=0` to fall
  back to the CPU oracle). The contract is only **distribution parity**, not
  token-for-token equality - even though both draws share the same
  pseudo-random generator and in practice produce the same sequence.

### Behavior scoring (rollout-policy logprobs)

A group's shared prefix is decoded **once**, in a working sequence copied
from the retained prefix (`seq_cp` + purge pattern), rather than rolled back
via `seq_rm` after each member. Rollback fails on a hybrid recurrent model
whenever the cache holds no snapshot deep enough - which is the default, since
`n_rs_seq` is derived to zero unless `RETRO_RECURRENT_ROLLBACK=auto` asks for
the snapshots and the geometry can pay for them
(`docs/engineering/SUPPORT.md`) - in which case the old path
re-prefilled the entire prefix per member - a `group_size` factor of
redundant work, measured and confirmed on Qwen3.5/GDN (4 prefix prefills per
group before the fix, 1 after).

A single-sequence context (SFT/PPO geometry) keeps the rollback path: it's
the only configuration where it costs nothing.

This approach extends to the three teacher-forced passes of an update:
behavior logprobs, frozen reference (`kl_coefficient ≠ 0`), and re-scoring
under the current policy at each GRPO epoch - the third is the most expensive
of the three (one rollout and one epoch, not once per update).

### Device-side target logprob extraction

A target token's generation logprob is extracted directly on the device
(`llama_set_target_logprobs` / `llama_get_target_logprob_ith`) instead of
bringing an `n_vocab`-float row back to the host to do the log-softmax
reduction there. The gather is built from ordinary decode-graph operations
(`ggml_soft_max` → `ggml_reshape` → `ggml_get_rows`), not a dedicated kernel:
this avoids porting a new kernel to all three backends, and the scheduler
automatically falls back to the host via `cap_device_logprobs` if a backend
rejects the node.

**Numeric safeguard.** A probability that underflows F32 (`p < ~1e-38`) would
give `-inf` in a GRPO ratio. The runtime detects the non-finite value and
**replays the entire call on the host oracle**, whose reduction is done in
double precision - the case is pathological by construction (the token comes
from the policy itself), but the fallback must never be silent.

On a unified-memory machine (Apple Silicon), the measured gain comes mainly
from the eliminated host log-softmax reduction, not from the transfer: a row
fetch there is a `memcpy`, not a bus crossing. On a discrete GPU (CUDA/Vulkan),
where the same row crosses PCIe, most of the expected gain is elsewhere and
remains to be measured on those targets before publishing a number.

## Closed decision: no prefix cache across updates

A KV cache cached from one update to the next would be **wrong**, not just
stale: the cache depends on the LoRA adapter, which changes at every update.
The prefix-sharing gain therefore only exists within a single update, where
generation deduplication and shared-prefix scoring already capture it.

## What's still expected, and what isn't

Changing the prefill geometry changes the GEMM tiling `ggml` picks, hence the
F32 reductions, hence the tokens drawn. This is **expected**: non-regression
tests check internal consistency (determinism, independence from a row's
physical position in the batch), never absolute-value equality of the loss
from one version to another.

## Measurement protocol

```sh
cargo run --features metal --bin profile --release -- examples/smoke_tiny_grpo.toml
```

Two discriminating measurements to verify prefill stays fixed:

- `--max-new-tokens 8` isolates prefill from decoding - the ratio
  `generation(96) / generation(8)` must stay clearly above 1.
- `batch = ubatch = 512` must **no longer** change generation time once the
  geometry is decoupled (the generation context no longer depends on
  `training.micro_batch`); it still, however, affects scoring, which runs on
  the training context by construction (GRPO ratios depend on it).

Non-regression lanes (see [`../tests/notice.md`](../tests/notice.md)):
`scripts/test-fast-rust.sh` and `scripts/test-abi.sh` on every change,
`scripts/test-cpu-integration.sh` before a PR (covers the canonical-order
invariant and the batch/scalar binary equality of scoring), and the GPU lanes
(`cuda`, `vulkan`) before any PR that touches decode geometry - the logprob
gather adds nodes to the graph that must be re-verified on discrete memory.
