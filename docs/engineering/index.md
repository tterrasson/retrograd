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

## Implementation notes

The user guides say how to run an algorithm, and the
[configuration reference](/reference/configuration) lists every key and its
default. These pages say how the loops are built, why each key is refused when
it is, and what the tests pin.

- [The configuration document](./CONFIG): every key's rationale and refusal
  rules, including the agentic sections.
- [PPO](./PPO) and [GRPO](./GRPO): the update loop, the objective and the
  validation map.
- [Agentic GRPO](./AGENTIC_GRPO): trajectories, tools, environments and judges.
- [The GRPO sampling path](./optims/SAMPLING): generation and behavior scoring.
- [Optimizer cost and quality](./optims/OPTIMIZERS): what Muon and Gefen were
  measured to cost, and how far Gefen's approximation is from the update it
  approximates.

## Interface contracts

- [Server errors](./server/ERRORS)
