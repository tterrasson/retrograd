# Profiling a GRPO run

The `profile` binary measures the full GRPO execution path instead of only one
kernel or one optimizer call. It reports initialization, rollout, reward,
reference scoring, optimizer timing, host memory, device memory, and backend
buffer allocations.

## Build and run

For a regular single-turn GRPO configuration:

```bash
cargo build --release
cargo run --release --bin profile -- examples/smoke_tiny_grpo.toml \
  --model /models/base.gguf --device gpu
```

For an agentic GRPO configuration with a container environment, build with the
container feature and run the same binary:

```bash
cargo run --release --features container --bin profile -- \
  path/to/agent_grpo.toml \
  --model /models/base.gguf --device gpu --full
```

The container daemon must be running before the profile starts. The first
profile invocation may also pull the configured image.

## Workload overrides

The profiler accepts these workload controls:

```text
--updates N
--group-size N
--epochs N
--prompts N
--micro-batch N
--max-new-tokens N
--full
```

The first five numeric options are intended to shorten a profile. They can
change the measured workload, so use `--full` when comparing a configuration
as written. `--model` and `--device` override the TOML values for this process.

The profiler currently accepts `grpo` and `agent_grpo` configurations only. A
configuration for `sft` or `ppo` is rejected before the model is loaded.

## Reading the report

Use the phase table to separate rollout generation, reward execution, fixed-base
reference scoring, and optimizer time. Use the memory tables to distinguish
host RSS, device usage, and backend-owned scratch. The report also shows the
selected backend and the effective workload after overrides.

## Compute headroom is not VRAM headroom

`training.max_gpu_duty_cycle` and the memory tables answer different questions,
and a run that needs one will not be helped by the other.

The duty cycle releases *time*: the trainer stops submitting for a share of the
wall clock so another process can run its own kernels in between. Everything the
run allocated - weights, KV caches, retained activations, optimizer state -
stays allocated for the whole run, including while it sleeps. Halving the duty
cycle frees no bytes at all.

Freeing *bytes* is a geometry change: a smaller `micro_batch`, a lower
`generation_concurrency`, a shorter `ctx`, gradient checkpointing. Those change
what the run computes per step, which the duty cycle deliberately does not.

So: a neighbour that fails to allocate needs the memory tables and a smaller
geometry. A neighbour that allocates fine but crawls needs the duty cycle.

When a duty cycle is set, the report prints a `duty cycle` line after the
phase table, with the requested fraction, the two shares and the cumulative
compute and idle seconds. (The backend report carries only the static
`gpu_duty_cycle_requested` / `_active` / `_reason` lines; the seconds move at
every GPU boundary and that report is cached.) A run showing `observed=0.50`
against a much lower `wall share` is already spending most of its wall clock
off the device - in data loading, judging, tokenization or checkpoint I/O - so
the compute an operator hoped to release is not the trainer's to give. On a CPU
backend the line instead warns that the setting was accepted and is not being
honoured.
