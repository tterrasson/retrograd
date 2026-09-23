# Engineering documentation

This section is for people who modify Retrograd. To train a model or tune a
run, start with the [Getting started](/getting-started/quickstart),
[Training](/training/sft) and [Reference](/reference/configuration) guides
instead.

## Contributing

- [Contribution principles](./contributing): architecture, errors, numbers and
  comments.
- [Numeric conversions](./CONVERSIONS): when an `as` needs a proof, a typed
  error or a saturation.
- [Tests and validation](./tests/notice): which lane to run for a given
  change.
- [Test lanes in detail](./tests/lanes): what each lane runs, its cost and its
  pitfalls.
- [llama.cpp fork](./LLAMA_CPP_FORK_WORKFLOW): syncing and publishing runtime
  changes.

## Backends and RIR

- [RIR kernel families](./rir/KERNELS)
- [RIR kernel promotion](./rir/PROMOTION): what decides that a generated kernel
  replaces a native one.
- [Support matrix](./SUPPORT)
- [CUDA status](./cuda/STATUS)

## Performance notes

- [The GRPO sampling path](./optims/SAMPLING): generation and behavior scoring.
- [Optimizer cost and quality](./optims/OPTIMIZERS): what Muon and Gefen were
  measured to cost, and how far Gefen's approximation is from the update it
  approximates.

## Diagnostic environment variables

These switch off a runtime check or optimization, to compare against it or to
work around it on a given device.

| Variable | Effect |
| --- | --- |
| `RETRO_UBATCH_FINITE_CHECK=0` | Skip the load-time check that the micro-batch produces finite logits on the GPU. |
| `RETRO_PACKED_SEQ_PROBE=0` | Trust the model's packed multi-sequence declaration instead of probing it. |
| `RETRO_RECURRENT_ROLLBACK=auto` | Apply the derived recurrent-state rollback depth (off by default). |
| `RETRO_RECURRENT_ROLLBACK_BUDGET_MB` | Memory budget for that depth (default 64). |
| `RETRO_GENERATION_PREFIX_CACHE=0` | Re-decode the whole prompt at every agentic turn instead of reusing the KV cache. |

## Interface contracts

- [Server errors](./server/ERRORS)
