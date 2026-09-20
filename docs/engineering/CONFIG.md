# The configuration document

One TOML document describes a whole run, whichever frontend reads it - the CLI,
the server, the Python binding. The schema is strict: unknown keys are refused
rather than ignored, because a misspelled key that trains something other than
what its author read is the failure mode this file exists to prevent.

Nine sections are shared by every algorithm: `[run]`, `[model]`, `[lora]`,
`[output]`, `[trainable]`, `[training]`, `[metrics]`, `[evaluation]`,
`[checkpoint]`, plus `[observe]` for the rollout algorithms. `[lora]` and
`[trainable]` are each required or refused depending on `training.trainable`
(see [`[lora]`](#lora) and [`[trainable]`](#trainable) below); `[output]` is
always present. `[run].algorithm` selects exactly one of `[sft]`,
`[ppo]`, `[grpo]`, `[distill]` or `[agent]` as the run's own section. Declaring a second one is an error, not a silently ignored
leftover.

The [configuration reference](/reference/configuration) is the summary table of
every key. This page gives, for each key, what it means, what it defaults to when
absent, and what makes it refused, and it covers the agentic sections the
reference does not detail. `[distill]` is documented in the reference and in
the [distillation guide](/training/distill). The schema itself lives in
`crates/retrograd-config/src/lib.rs` (and `agent.rs` for `[agent]`), the
validation of the agentic loop in `retrograd-agent-core`, and the judge and
environment declarations in `retrograd-spec`.

- [`[run]`](#run)
- [`[model]`](#model)
- [`[lora]`](#lora)
- [`[output]`](#output)
- [`[trainable]`](#trainable)
- [`[training]`](#training)
- [`[metrics]`](#metrics)
- [`[evaluation]`](#evaluation)
- [`[checkpoint]`](#checkpoint)
- [`[observe]`](#observe)
- [`[sft]`](#sft)
- [`[ppo]`](#ppo)
- [`[grpo]`](#grpo)
- [`[agent]`](#agent)
- [`[agent.judge]` / `[grpo.judge]`](#agentjudge--grpojudge)
- [`[agent.environment]`](#agentenvironment)

---

## `[run]`

`algorithm` - **required**, one of `"sft"`, `"ppo"`, `"grpo"`, `"distill"`,
`"agent_grpo"`. It picks the loop *and* the section that must be present:
`agent_grpo` reads `[agent]`, the others read the section of the same name.

`verbose` - default `false`. Turns on the runtime's own logging underneath the
progress table.

## `[model]`

The section is optional as a *document* section, because a frontend may supply
the model itself - the CLI's `--model` and `--device`, a server request field.
What is not optional is the resolved model: a document that names no path and
gets no override is refused.

`path` - the base model GGUF. Relative paths resolve against the document's
directory; a path passed on the command line resolves against the caller's
working directory.

`device` - `"auto"` (default), `"cpu"` or `"gpu"`. `auto` uses a compiled GPU
backend when one is available and falls back to CPU; `gpu` fails at trainer
creation when no backend is present, which is what you want in CI.

## `[lora]`

Required when `training.trainable` is `"lora"` (the default) or `"hybrid"`.
Refused for `"full"` and `"partial"`, which train base tensors and create no
adapter.

`rank` - default `8`. `alpha` - default `16.0`. The scaling applied to the
adapter is `alpha / rank`.

`seed` - default `42`. Seeds the adapter initialization, and (for SFT) the
per-epoch row permutation.

`targets` - default `["q", "k", "v", "o", "ffn_up", "ffn_down", "ffn_gate"]`.
The seven aliases expand to `blk.*.attn_q.weight` and friends. A value
containing `*` or `.weight` is taken as a literal tensor pattern. `["auto"]`
alone defers the choice to the architecture-specific defaults in the runtime,
and cannot be combined with other entries.

`auto` is capability detection, not a promise that every GGUF architecture has
a profile. It currently recognizes separate `blk.*.attn_q.weight` +
`blk.*.attn_v.weight` matrices and fused `blk.*.attn_qkv.weight`. Other tensor
layouts must use explicit patterns; a failure lists the candidate 2-D weight
families found in that model. The explicit seven-alias default likewise assumes
the conventional `blk.*.attn_*` / `blk.*.ffn_*` naming scheme.

`dtype` - `"f16"` (default) or `"f32"`. Storage precision of the A/B matrices
only: gradients and AdamW moments stay F32 either way, which is why F16 costs
nothing measurable in training quality.

`init_adapter` - resume from an existing adapter GGUF instead of creating a
fresh one. Rank, alpha, seed, dtype and targets then come *from the file*, so
combining it with any of those five keys is refused rather than silently
overridden. Mutually exclusive with `checkpoint.resume_from`, which owns the
adapter it restores.

## `[output]`

Always present. Where the run writes its result, and which kind of result
that is; a document naming no `[output].path` is refused.

`path` - **required**. Destination file.

`kind` - defaults to `"adapter"` for `training.trainable = "lora"`, otherwise
`"trainable"`. `"adapter"` is a portable LoRA GGUF; refused for a run that
trains no adapter (`full`/`partial`). `"trainable"` is a Retrograd bundle of
the trained base tensors by absolute value, with the adapter beside it for a
`hybrid` run; refused for a LoRA-only run, which has no base tensor to carry.
`"model"` exports a standalone GGUF that needs neither the source model nor
this loader: the file the model was loaded from, with the trained weights
folded in. It is only valid for `full`/`partial`; `lora`/`hybrid` are
refused because folding an adapter into the weights it multiplies is a merge
with no parity coverage here. `"model"` is checked before the first step: the
architecture must be one this build has written and loaded back, and the
filesystem under `path` must have room for a file the model's size.

## `[trainable]`

Which base tensors a `partial` or `hybrid` run trains. **Required** by those
two policies, **refused** beside `training.trainable = "lora"` (a selector
the run would ignore) and beside `"full"`, which derives every supported
eligible tensor on its own and takes no narrowing selector.

`layers` - default `"all"`. `"all"`, `"last:<count>"`, or an inclusive
`"<first>..<last>"`. Bounds-checked against the model.

`modules` - default none. Module aliases (`attn`, `ffn`), individual stems
(`attn_q`), or explicit tensor patterns. Norms are not modules: they follow
`norms`.

`norms` - default `false`. Every norm: the in-range block norms, and every
norm carrying no block index.

`biases` - default `false`. The `.bias` tensors of the selected modules.

`output_head` - default `false`. The output projection and its bias,
independently of the layer range. A head sharing the input-embedding storage
stays frozen, and asking for it is refused.

At least one of `modules`, `norms`, `biases` or `output_head` must select
something. `hybrid` initially permits only `norms` and `biases` beside the
adapter. Quantized tensors are never selected (a quantized model may still
carry trainable F32 norms), and selection validates the *selected* tensors
rather than the model's dominant dtype. The resolved exclusions (input
embedding, rotary constants wherever they sit, a tied head, unsupported
dtypes, duplicate storage) are reported at the start of the run.

Selecting the output head changes the loss graph: the fused cross-entropy
folds the projection into the loss and differentiates only its input, so a run
that trains the head takes the dense path instead. `training.chunked_cross_entropy`
is honoured for every other selection and resolved off for this one; the
backend report and the training preflight both name the result as `loss_path`.

## `[training]`

Every key is optional. The geometry keys are the ones worth understanding
first, because two divisibility rules connect them.

### Geometry

`ctx` - default `128`. The trained window, in tokens.

`micro_batch` - default `32`. The *physical* forward/backward width - llama.cpp's
`n_ubatch`, TRL's `per_device_train_batch_size`. This is the activation-memory
lever: lower it to fit, it changes no statistics.

`gradient_accumulation` - micro-batches accumulated before one AdamW step, TRL's
`gradient_accumulation_steps`. `micro_batch * gradient_accumulation` is the token
window one optimizer step trains, and it must divide `ctx`.

On a rollout algorithm (`ppo`, `grpo`, `agent_grpo`) there is exactly one
admissible value - one optimizer step per rollout means the step spans the whole
context - so the key is best omitted: it is pinned to `ctx / micro_batch`. A
value that contradicts the pin is reported with the arithmetic rather than
overridden. Outside a rollout it defaults to `1`.

`shared_prefix_fanout` - `"auto"` (default), `"off"`, `"max"`, or an integer
`>= 2`. Physical completion fanout for differentiable shared-prefix training: the
members of a group share their prompt's forward instead of each paying for it. An
explicit integer above `group_size` is refused.
`auto` adapts fanout per physical pass and reports why it falls back to rows;
`max` and integer fanouts fail when their requested geometry cannot run.
See [packing geometry and compatibility](../training/grpo.md#shared-prefix-packing-and-fallback)
for the width formula, device restrictions, and padding costs.

`threads` - default `0`, meaning "select performance cores automatically". The
`RETRO_THREADS` environment variable takes precedence at runtime.

### Trainable policy

`trainable` - `"lora"` (default), `"full"`, `"partial"`, or `"hybrid"`. `lora`
trains only the adapter and requires `[lora]`, refusing `[trainable]`. `full`
trains every supported eligible base tensor, requires no `[trainable]`
selector, and refuses `[lora]`. `partial` trains a named subset of base
tensors via `[trainable]` and likewise refuses `[lora]`. `hybrid` trains an
adapter (`[lora]`) alongside a `[trainable]`-selected subset of base tensors,
initially limited to norms and biases beside the adapter. See
[`[trainable]`](#trainable) for the selector and [`[output]`](#output) for
where the result lands.

A run that trains base weights and carries an enabled KL term
(`kl_coefficient > 0` in `[grpo]`/`[agent]`, or on-policy `[distill]`) needs a
`[reference]` section: without it the anchor would be "this model with its
adapter disabled," which is the original policy only while the base weights
stay frozen.

### Optimizer

`optimizer` - `"adamw"` (default), `"sgd"`, `"muon"`, or `"gefen"`. Each of the
last three reads its own hyperparameter table under `[optimizer.<name>]`
(refused unless selected). Only AdamW's update kernel writes an F16
parameter, so every other name is refused beside the default F16 adapter
dtype; set `lora.dtype = "f32"` or train base weights instead. Gefen's
update phases are written for the CPU and for Metal; the run probes the live
device for its own two nodes and is refused at preflight where they are
missing, because its state mutations must not be answered on a fallback
backend.

`sgd` keeps no persistent state (`0` bytes/param) but still has a step
counter and a schedule: "no slot" and "no optimizer" are different states.
`muon` orthogonalizes the momentum of eligible **hidden base matrices**:
tensors with exactly two non-trivial logical dimensions, excluding
embeddings, the output head, norms and biases by role. Everything Muon
declines, including LoRA factors, is updated by AdamW at its own
`fallback_learning_rate`, which is not a ratio of `training.lr`, since an
orthogonalized update and an AdamW one are not in the same units. `gefen`
keeps fixed-block state (`min_numel` below which a parameter falls back to
AdamW) under a `variant`: `shared_v` (default, `4N + 4K`) or `quantized_m`
(`N + 8K`, an 8-bit first moment against a shared 256-entry codebook). The two
variants are different slot layouts, so a checkpoint written under one is not
readable as the other. Every non-AdamW optimizer's ineligible or unsupported
tensors fall back to AdamW, and the planner's state-bytes and memory-plan
figures always price that fallback rather than assume the whole model is
eligible.

#### `[optimizer.muon]`

| Key | Default | Description |
| --- | ---: | --- |
| `momentum` | `0.95` | EMA coefficient of the first moment, in `[0, 1]`. |
| `nesterov` | `true` | Use the updated momentum in the update direction. |
| `ns_steps` | `5` | Newton-Schulz iterations, at least `1`. Structural: decides the update graph's size. |
| `ns_epsilon` | `1e-7` | Added to the Frobenius norm before normalizing. |
| `fallback_learning_rate` | `0.001` | AdamW's rate for the parameters Muon declines. |

Muon keeps one F32 momentum per eligible parameter (`4N`) plus AdamW's pair
for the rest. F32 weights only.

#### `[optimizer.gefen]`

Experimental.

| Key | Default | Description |
| --- | ---: | --- |
| `variant` | `"shared_v"` | `"shared_v"` (F32 first moment + one F32 second moment per block) or `"quantized_m"` (one-byte first moment against a shared codebook, F32 scale + second moment per block). |
| `block_size` | `1024` | Elements per block; a positive power of two. A partial trailing block still costs a row. |
| `min_numel` | `4096` | Below this element count a selected parameter falls back to AdamW; its fallback state is part of the reported total. |
| `codebook` | `"uniform"` | Only value that exists. |
| `codebook_levels` | `256` | `quantized_m` only, and must equal `256`: the index is one unsigned byte. |
| `partition` | `"fixed"` | Only value that exists. |
| `beta1` | `0.9` | First-moment coefficient. |
| `beta2` | `0.999` | Per-block second-moment coefficient. |
| `eps` | `1e-8` | Epsilon in the update denominator. |

At `block_size = 1`, `shared_v` keeps AdamW's own second moment, the cheapest
available correctness anchor. F32 weights only; both update phases are CPU
only.

`epochs` - default `1`. Passes over an SFT dataset. A rollout algorithm counts in
updates instead and spells its own passes `grpo_epochs` / `ppo_epochs` /
`epochs_per_update`.

`lr` - default `1e-4`. Must be finite and positive. Base learning rate of the
chosen optimizer; Muon and Gefen read their own rates as described above.

`weight_decay` - default `0.0`. `max_grad_norm` - default `1.0`, the global L2
norm all trainable gradients are clipped to together, preserving their relative
direction.

`lr_scheduler` - `"constant"` (default), `"linear"` or `"cosine"`.
`warmup_steps` - default `0`.

### Sampling context (rollout algorithms)

`generation_concurrency` - how many rollout sequences decode at once. GRPO-only:
setting it on an SFT run is refused. It defaults to
`min(prompts_per_update * group_size, micro_batch * gradient_accumulation, 256)`,
i.e. "the whole update in one wave" whenever that fits, and it is bounded by all
three of those quantities.

It is independent from the group size - a group may be sampled over several waves
without changing its statistical meaning, so this is purely a throughput knob.
Its cost is the dedicated generation KV cache, which grows **linearly**:
`ctx * concurrency` positions, at 2 bytes an element under
`fast_sampling_context`. The line the CLI prints at startup (`gen KV …`) divided
by the configured concurrency is the per-sequence price; four times the
concurrency is four times that number and nothing else - `generation_batch` is
already capped at 512, and the host-side rollout buffers are sized by
`scenarios_per_update * group_size`, not by this key. See
`crates/retrograd-plan/src/cost.rs`.

`generation_batch` - default `0`, meaning "derive it": the largest value keeping
the reserved output-logits buffer inside 64 MiB, capped by `ctx` and 512, then
raised to `generation_concurrency` so one decode wave fits a single launch.
Sampling has no backward graph, so it has no reason to inherit the optimizer's
`micro_batch` - a prompt prefilled in 16-token chunks pays one full sweep of the
quantized weights per chunk. See [`optims/SAMPLING.md`](optims/SAMPLING.md).

`fast_sampling_context` - default `true`. Builds the generation context with an
F16 KV cache and flash attention instead of the optimizer context's exact
F32 / no-FA settings. Halves the sampling KV footprint and speeds up decoding;
the sampling distribution differs from the trained policy by ulp-level rounding.
Set it to `false` for bit-exact sampling.

### Memory levers

`kv_dtype` - `"f16"` (default) or `"f32"`, for the *differentiable optimizer*
context. F16 halves the term that grows with the context, and it is never applied
blind: the runtime probes the device for differentiable flash attention at this
model's head geometry and rebuilds on F32 when the probe declines, reporting
which of the two it got.

`chunked_cross_entropy` - default `true`. Streams the vocabulary in tiles so the
full `[n_vocab, n_tokens]` logits tensor is never materialized. Not a
memory-for-speed trade: never building the largest tensor of the step removes
both its write and its read. Packed (rollout) path only.

`chunked_ce_tiles` - default `8`. Vocabulary tiles `C`; the peak logits footprint
is about `n_vocab / C`, paid for in recompute. Ignored without
`chunked_cross_entropy`.

`chunked_ce_seq_chunk` - default `512`. Bounds the tiled intermediate to this many
flattened `(batch × seq)` tokens, capping the peak independently of the sequence
length. `0` processes all tokens at once. Vocabulary tiling alone leaves the
intermediate growing along the token axis, which is why the default is not `0`.

`gradient_checkpointing` - default `false`. Recomputes transformer-layer
activations during the packed backward, trading forward compute for a lower
activation peak.

`checkpoint_every_n_layers` - default `4`. Keeps one activation checkpoint every
N layers. Not `1`: a stride of one stores *every* boundary - the largest retained
term checkpointing can produce - and still recomputes each layer's internals. The
peak is proportional to `n_layer / stride + stride`, which bottoms out around
`sqrt(n_layer)`.

`checkpoint_dtype` - `"f32"` (default), `"f16"` or `"bf16"`, the precision the
retained checkpoints are held in. Refused unless `gradient_checkpointing` is on:
without it the field has no effect, and "ignored" is indistinguishable from
"applied" in every artifact the run produces. `bf16` keeps F32's exponent range,
so it cannot overflow on a wide residual stream the way F16 can.

`require_gpu_resident` - default `false`. Fails the training preflight instead of
letting the scheduler send a training-graph op back to the CPU. A fallback is
correct - it only costs a scheduler split and a device↔host round trip per node -
so this is for when that cost is what is being measured or guarded against.

`max_gpu_duty_cycle` - default `1.0`, meaning no limit. An upper bound on the
fraction of wall time this trainer spends with GPU work submitted and in flight,
so a neighbouring workload gets regular compute windows. At `0.5`, a 200 ms
compute burst earns roughly 200 ms of idle time. Finite, in `(0, 1]`; zero is
refused, because the run-control pause is already the safe way to stop a live
run.

It is deliberately a *duty cycle* and not GPU utilization. Backend telemetry,
kernel occupancy, memory bandwidth and other processes can all make
`nvidia-smi` or Activity Monitor show something else, and nothing here
coordinates with another process: two trainers each set to `0.5` may idle in the
same windows and leave the device half empty, or overlap and saturate it.
Releasing time is unilateral and unsynchronized, which is exactly why it needs
no permissions, no NVML, no MPS and no MIG.

It releases **compute time, not device memory**. Weights, KV caches, retained
activations and optimizer state stay allocated while the trainer sleeps; a
neighbour that needs VRAM still needs a smaller `micro_batch`, a lower
`generation_concurrency`, or its own memory policy.

The limit costs something to turn on. Generation normally overlaps host-side
sampling with the previous decode still in flight, and enforcing a duty cycle
means synchronizing each accounted decode, which removes that overlap - once,
before any sleep. So `0.99` is not approximately `1.0`: crossing from disabled
to enabled has a price no fraction avoids. The honest advice is that the setting
is worth its overhead at `0.75` and below, and that `0.95` buys nothing.

The contract is a maximum, not a target: under contention the measured share is
below the request, because the host time this trainer spends waiting for another
process counts as its own. Accepted but inactive when the device resolves to
CPU - `auto` can resolve there too - and the backend report says
`gpu_duty_cycle_active: false` with `reason: cpu_backend` rather than promising
throttling that is not happening. `threads` remains the CPU control.

The backend report carries the three static lines above and no more. The
seconds move at every GPU boundary, and the report is cached, so the cumulative
accounting is read separately: the `profile` binary prints a `duty cycle`
line, and the Rust API exposes it as `Trainer::duty_cycle_stats()`.

It carries two ratios. `observed` is compute over the accounted windows and
mostly echoes the setting back, because every accounted window is by
construction followed by its own repayment. `wall_share` divides by the whole
trainer wall clock instead, so the unaccounted phases - data loading, judging,
tokenization, checkpoint I/O - are the gap between the two. A run showing
`observed: 0.50` and `wall_share: 0.20` is not misconfigured; it is CPU-bound,
and no duty cycle will free the compute you were hoping to release.

Not part of the trajectory signature: sleeping cannot alter the training problem
or the optimizer state, so a checkpoint may be resumed under a different duty
cycle.

## `[metrics]`

`tensorboard_dir` - where TensorBoard event files are written.
`wandb_export_dir` - where a W&B-importable export is written. Both optional;
omit the section entirely to log neither.

## `[evaluation]`

An "iteration" is one SFT epoch or one PPO/GRPO/agent update.

`data` - **required**. The held-out set.

`every_iterations` - default `1`.

`max_examples` - default: the whole file. A sample changes composition between
passes, so two passes at equal policy quality can differ by a whole difficulty
tier; prefer the full set when it is small enough.

`patience` - default: none, i.e. never stop early. Consecutive evaluations
without an improvement of at least `min_delta` before the run stops.

`min_delta` - default `0.0`. Set it to the resolution of the metric, not below:
a `min_delta` inside the noise makes `patience` a coin flip.

With `run.algorithm = "agent_grpo"`, `[evaluation]` requires an
`[agent.environment]`: an agentic evaluation measures the reward the environment
puts on a trajectory, and a judge cannot stand in - every RULER strategy scores
the members of a group against each other, so its scores are renormalized at
every update and a mean over them is not comparable across the run.

## `[reference]`

The frozen model a fixed-reference term is scored against. Its own section
shared by GRPO, agentic GRPO and on-policy distillation. PPO uses the rollout
policy for its KL term and does not consume this section.

`model` - **required**. The GGUF the anchor is loaded from, resolved relative
to the document.

`ctx` - the anchor's own context width, defaulting to `training.ctx`. A value
below `training.ctx` is refused: the anchor scores the sequences the run
produces, so it cannot hold fewer tokens than they can carry.

There is no precision key. The anchor's dtype is the one in its file, because a
setting that converted it on load would make the penalty depend on a number the
document chose rather than on the model the path names.

Two refusals, one per direction. A run that trains base weights and carries an
enabled KL term without this section is refused: the penalty would be taken
against "the model with its adapter disabled", which is the original policy
only while the base weights are frozen. And a document that declares this
section with no enabled KL term is refused too - the model would be loaded,
budgeted and never read.

The anchor is loaded at model-load time, not at the first scoring pass, and is
checked before it is used: vocabulary size and a fixed set of witness sentences
must tokenize identically to the trained model's, and one teacher-forced pass
must return finite, non-positive values. Its file's content fingerprint goes
into every checkpoint, so a resume refuses an anchor that is not the one the
penalty was measured against before it.

The anchor is forward-only - no adapter, therefore no backward graph, no
gradients and no optimizer state - so it costs its weights plus its KV, which
is the term the memory estimate carries beside the trained model's.

## `[checkpoint]`

`directory` - **required**. `mode` - **required**, one of `"steps"`,
`"best_eval"`, `"steps_and_best_eval"`.

`every_steps` - required and positive when the mode includes `steps`, refused
when it does not.

`resume_from` - a checkpoint to restore. Mutually exclusive with
`lora.init_adapter`.

A mode including `best_eval` without an `[evaluation]` section is refused.

## `[observe]`

Live export of the rollouts, with a static viewer; see
[Observing rollouts](/training/observe). Accepted for `ppo`, `grpo` and
`agent_grpo`, refused for `sft` and `distill`.

`directory` - **required**, resolved relative to the configuration file and created if missing. A directory that cannot be
created prevents the run from starting, as does failure to start the writer
thread. If locking is unavailable or another writer holds the lock, export is disabled
with a warning. Disk write failures also disable export without stopping training.

`every` - default `1`, positive. The texts of updates `every`, `2 × every`, …
are exported; update summaries always are.

`max_text_chars` - default `0` (no truncation). Limits each exported text to this
many Unicode characters, then appends a marker showing how many were removed.
Identifiers and tool names are preserved; this does not limit total batch size.

## `[sft]`

`data` - **required**.

`data_format` - `"text"` (alias `"txt"`) or `"jsonl"` (aliases `"chat"`,
`"chat-jsonl"`). Optional: inferred from the extension, and by sniffing the
content when the extension is unknown.

`shuffle` - default `true`. Permutes the training rows at the start of every
epoch, seeded from `lora.seed`. An SFT file is almost always ordered - by source,
by length, by whatever the generator emitted last - and replaying that order
every epoch both correlates the rows inside a micro-batch and makes the
per-epoch metrics an artefact of the file. Only the training part is permuted;
the evaluation rows of a split dataset keep their order. Epoch `N`'s permutation
is a pure function of `(seed, N)`, so a resumed run draws what an uninterrupted
one would have drawn.

## `[ppo]`

See [`PPO.md`](PPO.md) for the algorithm.

`prompts` - **required**. `reward_command` - **required**, argv of the process
that scores completions; its first element must be a non-empty executable.

`reward_mode` - `"persistent"` (default) or `"oneshot"`. Persistent keeps one
worker alive for the whole loop behind a version handshake, which matters because
a rollout algorithm calls its reward once per sampling wave: a one-shot command
pays its interpreter, its imports and whatever model it loads several hundred
times a run. The command must write each response line *and flush it* - nothing
closes its stdin between batches. Use `"oneshot"` for a command that reads its
stdin to the end before answering.

`reward_timeout_seconds` - default `300`. Deadline of one reward *batch*, which
also covers the worker's startup on the first batch in persistent mode.

`updates`, `rollout_batch_size`, `ppo_epochs` - **required**, all positive.

`clip_range` - **required**, strictly between 0 and 1.

`kl_coefficient` - **required**, non-negative. The KL penalty against the
policy that generated the rollout; it does not use `[reference]`.

### `[ppo.critic]`

Optional table; omitted, the defaults below apply.

`enabled` - default `true`. `gamma` - default `1.0`, in `(0, 1]`; `1.0` is
undiscounted. `gae_lambda` - default `0.95`, in `[0, 1]`; `1.0` is Monte-Carlo
returns. `value_lr` - default `1e-2`. `value_epochs` - default `8`, full-batch
Adam epochs per PPO update.

`feature_dtype` - `"f32"` (default), `"f16"` or `"bf16"`. Storage precision of the
host-side feature matrix (`total_completion_states * hidden_dim`, usually the
run's largest host allocation). The fit stays F32 whatever this says; a 16-bit
setting halves the buffer and rounds the rows it regresses on. Refused when the
critic is disabled.

### `[ppo.sampling]`

`temperature`, `top_p`, `max_new_tokens`, `seed` - all **required**. `top_p` must
be in `(0, 1]`.

## `[grpo]`

Single-turn Dr. GRPO with the DAPO corrections. See [`GRPO.md`](GRPO.md).

`prompts`, `reward_command`, `reward_mode`, `reward_timeout_seconds` - as in
`[ppo]`, same contract and same defaults.

`updates`, `prompts_per_update`, `grpo_epochs` - **required**, all positive.

`group_size` - **required**, at least 2 and at most 256. It must also fit the
optimizer window (`micro_batch * gradient_accumulation`).

`clip_range_low` / `clip_range_high` - **required**, both in `(0, 1)`, with
`high >= low`. DAPO Clip-Higher: the surrogate clip band is decoupled, `low`
bounding the ratio from below (`1 − low`) and `high` from above (`1 + high`),
typically looser - 0.2 / 0.28. A symmetric band caps how much a low-probability
token with positive advantage can grow, which drives entropy collapse.

`kl_coefficient` - **required**, non-negative. With verifiable rewards, `0.0` is a
good default: the anchor mostly slows learning there.

`mask_truncated` - default `false`. DAPO overlong filtering: excludes completions
that used their whole generation budget from both the group baseline and the
optimizer epochs, since their reward judges an incomplete response. A group left
with fewer than two unmasked members is dropped whole.

`baseline` - `"mean"` (default) or `"leave_one_out"` (alias `"rloo"`).

`prompt_order` - `"sequential"` (default) or `"shuffled"`.

`max_stalled_updates` - default `25`. Consecutive zero-signal updates tolerated
before the run stops; `0` never stops.

### `[grpo.sampling]`

`temperature`, `top_p`, `max_new_tokens`, `seed` - all **required**. Dr. GRPO is
strictly on-policy, so `temperature = 1` and `top_p = 1` are *enforced* at
configuration time rather than merely recommended.

### `[grpo.overlong_penalty]`

Optional. A soft alternative to `mask_truncated`: instead of dropping the
overruns, penalize length inside a buffer before the budget, so the policy learns
to conclude rather than to ignore the limit.

`buffer_tokens` - **required**, in `1..max_new_tokens`. `max_penalty` -
**required**, positive.

### `[grpo.kl_schedule]`

Optional, and refused unless `kl_coefficient > 0`.

`warmup_updates` - default `0`. Linear ramp of the effective coefficient from 0 to
`kl_coefficient` over the first N updates.

`target` - default: none, i.e. the coefficient stays at its warmed-up value. Set,
it is a PPO-style adaptive controller: after each update the effective
coefficient is multiplied up or down to chase this measured KL.

### `[grpo.dynamic_sampling]`

Optional. DAPO dynamic sampling: after zero-signal groups are dropped, keep
drawing replacement prompts until the update carries `prompts_per_update`
informative groups. Keeps the effective batch full as the policy converges.

`max_resample_factor` - **required**, at least 2 (1 would allow no resampling).
Caps the candidate groups per update at `prompts_per_update * factor`.

### `[grpo.judge]` and its three keys

`[grpo.judge]` takes the same shape as [`[agent.judge]`](#agentjudge--grpojudge).
Its verdict never replaces the reward command: it is *added* to it, weighted, so
the verifiable part keeps deciding what it can decide and the judge only
separates the candidates it left tied.

`judge_weight` - **required as soon as a judge is declared**. `reward + weight *
verdict`, with `verdict` in `[0, 1]`. A judge contributing an unstated amount to
the gradient is the one thing this section must not allow.

`judge_failure` - `"drop_group"` (default) or `"fail"`.

`max_judge_dropped_fraction` - default `0.5`, in `[0, 1]`.

All three are refused without a `[grpo.judge]` section rather than ignored: a
`judge_weight` written next to no judge is someone expecting a verdict in their
reward.

## `[agent]`

Multi-turn GRPO: a trajectory in place of a completion. See
[`AGENTIC_GRPO.md`](AGENTIC_GRPO.md).

Unlike `[ppo]` and `[grpo]`, every key here has a default - except the one the
section cannot invent.

`scenarios` - **required**. The JSONL scenario file.

A run needs *something* that grades: a document with neither `[agent.judge]` nor
`[agent.environment]` is refused at load time, because finding that out at the
first update is an hour of rollouts too late.

### The loop

`updates` - default `1`. `scenarios_per_update` - default `1`. Groups per
gradient, and the statistical width of an update: GRPO's baseline is intra-group,
so the noise on a step comes from how many groups it averages, not from
`group_size`.

`group_size` - default `8`, at least 2. Must fit the optimizer window.

`epochs_per_update` - default `4`. Optimizer passes over one update's rollouts.
Spelled in full because `training.epochs` is a different quantity.

`clip_range_low` / `clip_range_high` - defaults `0.2` / `0.28`, same rule as
`[grpo]`.

`kl_coefficient` - default `0.0`. A reference pass over every trainable
sequence is the most expensive thing a KL coefficient buys; on a
task whose reward is an exit-code-grade fact there is no reward hacking for it to
leash. Raise it to 0.01–0.05, not to 5e-4, if the policy starts leaving its
language behind.

`seed` - default `42`.

### Rollout limits

**These set the scale of the gradient.** The loss is divided by the trajectory's
token *budget* - `min(max_turns * max_new_tokens_per_turn, max_trajectory_tokens)`
- not by its realized length, which is Dr-GRPO's normalization and what keeps long
trajectories from being systematically demoted. The consequence to hold on to:
halving `max_turns` doubles the gradient without touching `lr`. Re-tune `lr` when
the limits move.

`max_turns` - default `6`. One tool call is one turn, so this has to clear the
task's own action budget with a few turns to spare.

`max_new_tokens_per_turn` - default `512`. Generation budget for one assistant
turn; `[grpo.sampling].max_new_tokens` is the single-turn equivalent.

`max_trajectory_tokens` - default: the model's whole context. Set above it, it is
an error rather than a silent clamp - a budget the model cannot hold is a
configuration someone has to fix, and the loss denominator is derived from it.

`max_rollout_secs` - default `300`. Wall-clock budget for a whole rollout, checked
at turn boundaries, so a single turn may overshoot it. Overrunning *truncates* the
trajectory - the same policy as overrunning the token budget, not an error. `0`
disables the deadline.

`end_on_no_tool_call` - default `true`. What a turn the parser read as content,
with no call and no malformed call in it, means. `true` reads it as the policy's
answer and ends the trajectory. `false` reads it as a wasted turn: one error
observation, no environment step, and the episode runs on until the environment
says `done` or the turn budget is spent. Set it for a world that owns its own
terminal state - otherwise the member that answered in prose banks no step
rewards, beats every sibling that paid for its moves, and the intra-group
baseline trains the policy out of calling tools. Ignored by a run that declared
no tools. See AGENTIC_GRPO.md, "A turn that calls no tool".

`max_failed_turns` - default `0`, never. Consecutive turns without a valid tool
call - prose, or a call the parser refused - after which the trajectory is cut.
The cut is a truncation, exactly like running out of turns, so `truncation`
prices it and `min_reward` leaves it at the bottom of its group, where a member
that never engaged the environment already sits; what it saves is the decode
of every turn after the N-th. One valid call resets the count. Meaningful under
`end_on_no_tool_call = false`, since under the default the first such turn
already ends the trajectory.

### Selection and failure

`truncation` - `"drop"` (default) or `"min_reward"`. `drop` deletes exactly the
trajectories that wandered and selects the update on the members that happened to
finish, so wandering costs nothing; `min_reward` keeps them at the bottom of their
own group, counting the step rewards they already banked. Truncated members count
against `max_dropped_fraction` under either policy.

`max_dropped_fraction` - default `0.5`, in `[0, 1]`. Truncated, failed and
unscored members counted together, once, over the rollouts the update asked for.

`judge_failure` - `"drop_group"` (default) or `"fail"`. Judge failures never
become neutral rewards.

`drop_degenerate_groups` - default `false`. Discards groups whose members all
received the same score. Such a group contributes no gradient, but dropping it
changes the number of groups an update trains on, so it is a decision to state
rather than one to inherit - watch `judge/degenerate_group_fraction` first.

`skip_empty_updates` - default `false`. Lets an update left with fewer than two
trainable trajectories pass without an optimizer step instead of ending the run.
Off by default: an update with nothing to train on normally means the environment
or the judge stopped working. Every skip is logged with the same accounting the
failure would have reported.

### Tools

`mcp_servers` - an array of tables, `[[agent.mcp_servers]]`. Per server:

- `name` - **required**.
- Exactly one transport: `command = [...]` (stdio, with an optional `env` table),
  `url = "..."` (streamable HTTP, with an optional `headers` table), or a nested
  `transport = { type = "stdio" | "streamable_http", … }`.
- `tool_timeout_secs` - default `30`.
- `allowed_tools` / `denied_tools` - default: everything the server advertises,
  nothing hidden. `*` is a wildcard, so `read_*` keeps a family without listing
  it; a pattern without `*` is an exact name. Denials apply after allowances,
  which is how a mostly-useful server gets adopted without exposing the one tool
  that writes to production.
- `max_tool_result_bytes` - default `65536`. An unbounded tool result silently
  eats the trajectory's token budget.
- `required` - default `true`. A required server that fails to connect fails the
  run; an optional one is skipped with a warning, and the tool it provided is
  then simply absent - which changes what the policy can learn.
- `stateless` - default `false`. **With an environment declared, every MCP server
  must set it to `true`**, checked at parse time: a `ToolProvider` is shared by
  every trajectory of every group, so a stateful one lets group members
  contaminate each other, which is exactly what the relative baseline assumes
  cannot happen.
- `cwd`, `env_passthrough` - optional.

`mcp_config` - a path or a list of paths to MCP configuration files. Declared,
never merged at load time: resolving them is I/O against the running machine, and
a planner or a server reads documents for machines that are not theirs.

### `[agent.scenario_generation]`

Optional. Generates the scenario file with an external LLM before the run.

`model`, `base_url`, `api_key_env` - **required** (non-empty) as soon as the table
is present.

`count` - default `24`. `batch_size` - default `12`, at most `count`.
`timeout_secs` - default `120`. `max_retries` - default `2`.
`seed` - defaults to `agent.seed`. `shuffle` - default `true`.
`min_difficulty` / `max_difficulty` - defaults `1` / `5`, within `1..=5`.
`max_catalog_bytes` - default `262144`, the budget for the tool catalogue shown
to the generator. `custom_instructions` - default empty.

## `[agent.judge]` / `[grpo.judge]`

One `type` field selects the backend. The RULER fields are written inline next to
`type` in TOML (a nested table would need its own header); the Python binding's
nested `config = {…}` spelling deserializes to the same value, and mixing the two
is refused.

### `type = "command"`

`command` - **required**, argv. An empty command is refused with the suggestion to
drop the table entirely and grade with the environment alone.
`timeout_secs` - default `30`.

### `type = "ruler"`

A relative judge: it ranks the members of a group against each other, which is
exactly the unit GRPO centres its advantage on - and the reason its scores are not
comparable across updates.

`base_url`, `model` - **required**, non-empty. `api_key_env` - default
`"OPENAI_API_KEY"`.

`rubric` - default: the built-in one. `pairwise_rubric` - default: a
pairwise-specific built-in, *not* `rubric`, whose wording asks for per-trajectory
scores and confuses a two-way verdict.

`temperature` - default: the endpoint's own. Must be finite and non-negative.

`max_concurrency` - default `4`. `timeout_secs` - default `120`. `max_retries` -
default `2`.

`cache_path` - optional; the one path inside the judge resolved relative to the
document.

#### `[agent.judge.strategy]`

`mode` - `"auto"` (default), `"listwise"`, `"chunked"` or `"pairwise"`.

- `auto` - one listwise request per group when it fits the context budget, split
  into anchored chunks when it does not. Correct on small groups, and it degrades
  into chunking instead of failing on large ones.
- `listwise` - always one request. Cheapest, and the only mode whose scores come
  from a single comparison, at the cost of failing outright when the group
  overruns the window.
- `chunked` - always split. `anchor` (default `true`) repeats the group's first
  trajectory in every chunk so per-chunk scores share a scale.
- `pairwise` - two trajectories per request, aggregated. The most reliable signal
  per judgement and the most robust to long trajectories, at `max_pairs` requests
  per group. `max_pairs` - default: every pair; must be positive when set, and a
  value below the group size is raised to it, since fewer comparisons than
  members would leave a trajectory unrewarded.
  `both_orders` (default `true`) judges every pair in both presentation orders and
  keeps only the agreements - **this doubles the request count**, and what it buys
  is `judge/position_disagreement`, the number that says whether the chosen judge
  deserves to be listened to at all. `aggregation` - `"win_rate"` (default) or
  `"bradley_terry"`; Bradley-Terry earns its cost exactly when the schedule is
  truncated, since beating the group's best is then not the same evidence as
  beating its worst.

#### `[agent.judge.context]`

Budgets applied when rendering a group into a request. Each must be positive, and
they must nest: `max_message_chars <= max_trajectory_chars <= max_request_chars`.

`max_request_chars` - default `60000`. Groups whose rendering exceeds it are split
into several requests. `max_trajectory_chars` - default `8000`; beyond it,
messages are elided from the middle. `max_message_chars` - default `2000`; long
tool observations are the usual offender.

`head_ratio` - default `0.4`, in `[0, 1]`. Share of an elided message kept from
its head, the rest from its tail: tool output carries its conclusion at the end,
assistant reasoning its intent at the start.

`include_env_state` - default `true`. Shows each member's terminal environment
state - the diff, for a code task - next to its transcript. It costs nothing when
the environment reports none, and judging the dialogue rather than the result is
the main source of reward noise on any task whose outcome is a file.

#### `[agent.judge.compaction]`

Optional, off by default. LLM compaction of the middle of each transcript.

`trigger_chars` - **required**. A group is compacted when its longest member
exceeds this; below it nothing is sent and nothing is spent. `target_chars` -
**required**, strictly below `trigger_chars` (otherwise compacting cannot shorten
anything); advisory, the model is asked to respect it and the result is elided if
it does not. `keep_last` - **required**, positive: closing messages kept verbatim,
because the result of a trajectory is the part a judge is least able to
reconstruct from a summary.

## `[agent.environment]`

One stateful world per trajectory, created per rollout, `reset` before the first
turn and closed at the end - including on the failure and deadline paths. That is
the property MCP tools cannot have, and the reason a task with per-episode state
serves an environment rather than an MCP façade.

`type` - **required**, one of `"http"`, `"container"`, `"local"`.

Reading a document never requires being able to run it: only the *declaration* is
validated at load time. Whether this build has a container backend, and whether
the tools named exist in its registry, are the runner's sentences to say.

### `type = "http"`

The world lives behind `reset` / `step` / `state` / `close` over HTTP - also how an
existing OpenEnv server plugs in.

`base_url` - **required**, an `http://` or `https://` URL.

`request_timeout_secs` - default `60`. Bounds **one call**, not the trajectory;
`agent.max_rollout_secs` does that, and one hung `step` must not consume it.

`connect_timeout_secs` - default `10`.

`pool_size` - default `16`. Concurrent connections kept alive. A group of N runs N
environments at once, so a pool below the group size serializes the turn that
lockstep batching just made parallel.

`max_result_bytes` - default `65536`. Cap on an observation, applied before it
enters the prompt.

`headers` - optional table added to every request: an authorization token, a
tenant id.

### `type = "container"`

A pool of containers on the local Docker daemon. Requires the `container` feature;
without it the table still parses, so the error is "not compiled into this binary"
rather than "unknown field".

`profile` - `"python"` (default), `"typescript"` or `"custom"` (nothing
preselected; the configuration lists the tools itself).

`tools` - default: the profile's set. `deny_tools` - default empty. Extra tools
beyond the profile's are not configurable - that is the programmatic extension
point - but removing one is.

`image` - default: the profile's. Prefer `name@sha256:…`: a run resumed three
weeks later on a moving tag is not the same environment. What the run records is
the repository digest when the daemon knows one, otherwise the local image id
(immutable, but resolvable on no other machine), otherwise the reference exactly
as written - a mutable tag, logged as a warning, and the one case where the
recorded environment is not reproducible.

At the first container creation, a custom image is checked for the utilities
the sandbox itself needs (`sh`, GNU-compatible `find -printf`, `grep`, `ls`,
`cat`, `kill`, `mkdir`, and `timeout` when the wrapper is enabled). Missing
capabilities fail before the container enters the pool. A profile may require
additional tools such as `bash`, Python or Node.

`allow_network` - default `false`. A task that needs the network says so in the
operator's configuration, not in a scenario. Enabling it grants ordinary bridge
egress; Retrograd does not provide a destination allow-list or block metadata
and private-network endpoints. Apply that policy outside the container runtime
before enabling network for untrusted scenarios.

`cache_volume` - default: the profile's package cache, mounted read-only. One
`pip install` per run instead of one per episode is the real payoff of container
reuse.

`setup_timeout_secs` - default `300`, must be positive. `verify_timeout_secs` -
default: none.

`run_id` - default: derived from the process id. Identifies this run's containers
so a crash leaves something the reaper can recognize.

#### `[agent.environment.limits]`

`cpus` - default `1.0`. `memory_mb` - default `1024`. `pids` - default `256`.
`exec_timeout_secs` - default `30`. `max_output_bytes` - default `65536`.

#### `[agent.environment.pool]`

`max_live` - default `8`, must be positive. `min_idle` - default `0`, and it
cannot exceed `max_live`: the pool would warm containers it is not allowed to
hold. `max_leases_per_container` - default `32`, must be positive; a recycled
container accumulates whatever escaped the workspace, and retiring it after N
episodes bounds that.

`reuse` - `"never"` (default) or `"workspace"`. `never` destroys the container at
the end of its episode, and it is the only policy where isolation is a property of
the runtime rather than of our cleanup code being right. `workspace` recycles it
with the workspace wiped and leftover processes killed; what survives, and what
one accepts by choosing it, is the image layers, the read-only caches, and any
damage a previous episode did outside the workspace.

### `type = "local"`

The host, with no isolation.

`allow_unsandboxed` - default `false`, and the declaration is **refused** while it
is. Model-generated code runs on the training machine with the training process's
privileges and network, so someone has to type this.

`profile`, `tools`, `deny_tools`, `setup_timeout_secs` (default `300`),
`verify_timeout_secs` - as in the container environment.
