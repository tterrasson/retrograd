# Performance and memory

## When a run does not fit

Try these in order:

1. Lower `training.micro_batch`. Raise `gradient_accumulation` (SFT) to keep
   the same step size.
2. Lower `training.generation_concurrency` (GRPO, agentic GRPO,
   distillation): fewer answers are generated at once.
3. Enable `training.gradient_checkpointing`, optionally with
   `checkpoint_dtype = "bf16"`.
4. Keep `fast_sampling_context`, `kv_dtype = "f16"` and
   `chunked_cross_entropy` at their defaults, which already save memory.

`ctx`, `group_size` and `prompts_per_update` also change memory use, but they
change what is being trained: do not use them only to save memory.

With a GPU, run `retrograd preflight --model base.gguf` to check that no
training operation falls back to the CPU; each fallback slows every step.
`training.require_gpu_resident = true` makes such a fallback an error.

## Sharing a GPU

`training.max_gpu_duty_cycle` limits the share of time the run keeps the GPU
busy, leaving regular gaps for another workload:

```toml
[training]
max_gpu_duty_cycle = 0.5
```

It frees **compute time, not memory**: everything the run allocated stays
allocated. If another process runs out of memory, reduce this run's memory use
instead (see above). Enabling the limit has a fixed cost, so it is only worth
it at `0.75` and below. It has no effect on the CPU; use `training.threads`
there.

## Profiling

The `profile` binary runs a short GRPO or agentic GRPO workload and reports
time per phase (generation, reward, scoring, optimizer), host and device
memory, and backend allocations:

```bash
cargo run --release --bin profile -- examples/smoke_tiny_grpo.toml \
  --model /models/base.gguf --device gpu
```

By default it shortens the workload (`--updates 1 --group-size 4 --epochs 1
--prompts 3`); pass those flags to change it, or `--full` to run the
configuration as written. Add `--features container` for a container-based
agentic configuration. Run it from the repository root so that relative reward
commands resolve.

When `max_gpu_duty_cycle` is set, the report adds a `duty cycle` line. If its
`wall share` is much lower than the setting, the run is spending most of its
time off the GPU (data loading, reward, judge), and the limit frees little.
