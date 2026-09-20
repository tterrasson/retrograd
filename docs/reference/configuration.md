# Configuration reference

Retrograd uses one strict TOML document per run. The `[run].algorithm` value
selects exactly one algorithm section: `[sft]`, `[ppo]`, `[grpo]`, `[distill]`,
or `[agent]` when the value is `agent_grpo`. Unknown keys and unrelated
algorithm sections are rejected.

## Shared sections

### `[run]`

| Key | Default | Description |
| --- | ---: | --- |
| `algorithm` | required | `sft`, `ppo`, `grpo`, `distill`, or `agent_grpo`. |
| `verbose` | `false` | Enable additional runtime logging. |

### `[model]`

| Key | Default | Description |
| --- | ---: | --- |
| `path` | required after overrides | Base GGUF model. Relative paths use the TOML directory. |
| `device` | `auto` | `auto`, `cpu`, or `gpu`. `gpu` fails if no compiled GPU backend is available. |

The CLI `--model` and `--device` flags override these values. A model path must
come from either the document or the CLI.

### `[lora]`

Required when the run trains an adapter (`training.trainable = "lora"`, the
default, or `"hybrid"`). Refused for `"full"` and `"partial"`, which train base
tensors and create no adapter.

| Key | Default | Description |
| --- | ---: | --- |
| `rank` | `8` | LoRA rank. Higher values add trainable parameters. |
| `alpha` | `16.0` | LoRA scaling; the effective scale is `alpha / rank`. |
| `seed` | `42` | Adapter initialization and SFT shuffle seed. |
| `targets` | seven common targets | Aliases: `q`, `k`, `v`, `o`, `ffn_up`, `ffn_down`, `ffn_gate`. Patterns containing `*` or `.weight` are used as literal tensor patterns. `auto` asks the runtime to select architecture-specific targets. |
| `dtype` | `f16` | Adapter matrix storage: `f16` or `f32`. Optimizer moments remain F32. |
| `init_adapter` | none | Load an existing adapter GGUF as a cold adapter start. Cannot be combined with rank, alpha, seed, dtype, targets, or `checkpoint.resume_from`. |

### `[output]`

Where the run writes its result, and which kind of result it is. A run that
does not name `[output].path` is rejected.

| Key | Default | Description |
| --- | ---: | --- |
| `path` | required | Destination file. |
| `kind` | `adapter` for `lora`, `trainable` otherwise | `adapter` is a portable LoRA GGUF. `trainable` is a Retrograd bundle: the trained base tensors by absolute value, plus the adapter beside it for a hybrid run; it requires the matching base model and this loader. `model` is a standalone GGUF that needs neither the source model nor this loader: the file the model was loaded from, with the trained weights in it. |

A kind the policy does not produce is rejected rather than defaulted: `adapter`
for a run that trains no adapter would be an empty file, `adapter` for a hybrid
run would drop its trained base tensors, and `trainable` for a LoRA run has no
base tensor to carry.

`model` belongs to `full` and `partial`, the two policies whose whole result is
in the weights. `lora` and `hybrid` are rejected: folding an adapter into the
weights it multiplies is a merge with no parity coverage here, and a model
written around it would load and not be the run. Two further checks run before
the first step rather than after the last one - the architecture must be one
this build has written and loaded back, and the filesystem under
`[output].path` must have room for a file the size of the model.

### `[trainable]`

Which base tensors a `partial` or `hybrid` run trains. Required by those two,
rejected beside `training.trainable = "lora"` (a selector the policy ignores is
a selector you believe is in effect) and beside `"full"`, which derives every
supported eligible tensor and takes no narrowing selector.

| Key | Default | Description |
| --- | ---: | --- |
| `layers` | all blocks | `all`, `last:<count>`, or an inclusive `<first>..<last>`. Bounds-checked against the model. |
| `modules` | none | Module aliases (`attn`, `ffn`), individual stems (`attn_q`), or explicit tensor patterns. Norms are not modules: they follow `norms`. |
| `norms` | `false` | Every norm: the in-range block norms, and every norm carrying no block index. |
| `biases` | `false` | The `.bias` tensors of the selected modules. |
| `output_head` | `false` | The output projection **and its bias**, independently of the layer range. A head sharing the input-embedding storage stays frozen and asking for it is an error. |

At least one of `modules`, `norms`, `biases` or `output_head` must select
something. `hybrid` initially permits only `norms` and `biases` beside the
adapter. Quantized tensors are never selected; a quantized model may still
carry trainable F32 norms, and selection validates the *selected* tensors
rather than the model's dominant dtype. The resolved exclusions - input
embedding, rotary constants (wherever they sit, including inside a block), a
tied head, unsupported dtypes, duplicate storage - are reported at the start of
the run.

Selecting the output head changes the loss graph. The fused cross-entropy
folds the projection into the loss and differentiates only its input, so a run
that trains the head takes the dense path instead: `training.chunked_cross_entropy`
is honoured for every other selection and resolved off for this one. The
backend report and the training preflight both name the result as `loss_path`,
and the planner budgets the vocabulary buffer the dense path allocates.

### `[training]`

| Key | Default | Description |
| --- | ---: | --- |
| `trainable` | `lora` | Which family of parameters this run trains: `lora`, `full`, `partial`, or `hybrid`. |
| `optimizer` | `adamw` | `adamw` or `sgd`. `muon` and `gefen` parse and are rejected: this build has no update step for them, and accepting the name would record an optimizer the run never used. `sgd`'s update kernel is F32-only, so it is rejected beside the default F16 adapter. |
| `ctx` | `128` | Trained context window in tokens. |
| `micro_batch` | `32` | Physical forward/backward width and primary activation-memory control. |
| `gradient_accumulation` | `1` for SFT; derived for rollout | Micro-batches per optimizer step. Its product with `micro_batch` must divide `ctx`. Rollout algorithms default to `ctx / micro_batch`. |
| `shared_prefix_fanout` | `auto` | GRPO physical fanout for completions sharing a prompt: `auto`, `off`, `max`, or an integer of at least `2`. |
| `threads` | `0` | CPU worker threads; `0` selects automatically. `RETRO_THREADS` overrides it. |
| `epochs` | `1` | SFT passes. PPO and GRPO use their own epoch counters. |
| `lr` | `0.0001` | AdamW learning rate. |
| `weight_decay` | `0.0` | AdamW weight decay. |
| `max_grad_norm` | `1.0` | Global L2 gradient clipping threshold. |
| `lr_scheduler` | `constant` | `constant`, `linear`, or `cosine`. |
| `warmup_steps` | `0` | Learning-rate warmup steps. |
| `fast_sampling_context` | `true` | Use F16 KV storage and flash attention in the dedicated generation context. Set `false` for bit-exact sampling context behavior. |
| `kv_dtype` | `f16` | Optimizer-context KV storage: `f16` or `f32`. The runtime may fall back to F32 when the device cannot use the requested path. |
| `generation_concurrency` | derived for GRPO | Live rollout sequences. Supported for GRPO and agent GRPO; bounded by the optimizer window, rollout count, and `256`. |
| `generation_batch` | derived | Generation batch size. The runtime derives a value within its output-logits memory budget. |
| `chunked_cross_entropy` | `true` | Stream vocabulary tiles for packed GRPO/agent training. Resolved off for a run that trains the output head, which needs the dense loss path. |
| `chunked_ce_tiles` | `8` | Number of vocabulary tiles when chunked cross entropy is enabled. |
| `chunked_ce_seq_chunk` | `512` | Flattened token chunk size for the tiled intermediate; `0` processes all tokens at once. |
| `gradient_checkpointing` | `false` | Recompute transformer activations to reduce the activation peak. |
| `checkpoint_every_n_layers` | `4` | Retained activation checkpoint stride when checkpointing is enabled. |
| `checkpoint_dtype` | `f32` | Retained activation precision: `f32`, `f16`, or `bf16`. Non-F32 values require gradient checkpointing. |
| `require_gpu_resident` | `false` | Fail preflight if a training-graph operation would fall back to CPU. |
| `max_gpu_duty_cycle` | `1.0` | Upper bound on the fraction of wall time the trainer waits on GPU work it submitted, so another workload gets regular compute windows. Finite, in `(0, 1]`. Releases compute, not VRAM. Accepted but inactive on a CPU device. |

The optimizer window is `micro_batch × gradient_accumulation`. Lower
`micro_batch` when memory is constrained. It is a geometry setting, not a
replacement for the rollout group size.

`constant` holds the learning rate. `linear` and `cosine` warm up over
`warmup_steps` updates, then decay to zero over the rest of the run. For SFT the
horizon is the real row count; for PPO and GRPO it is re-sized after each update
on the steps actually taken, so the decay still ends at zero.

### `[metrics]`

| Key | Default | Description |
| --- | ---: | --- |
| `tensorboard_dir` | none | Directory for TensorBoard event files. |
| `wandb_export_dir` | none | Directory for W&B-importable exports. |

### `[evaluation]`

| Key | Default | Description |
| --- | ---: | --- |
| `data` | required | Held-out text or chat JSONL file. |
| `every_iterations` | `1` | Evaluate every SFT epoch or rollout update. |
| `patience` | none | Stop after this many evaluations without improvement. |
| `min_delta` | `0.0` | Minimum improvement counted by patience. |
| `max_examples` | all | Maximum evaluation examples for rollout evaluation. |

### `[observe]`

PPO, GRPO and agentic GRPO only. See [Observing rollouts](../training/observe).

| Key | Default | Description |
| --- | ---: | --- |
| `directory` | required | Export directory: `observe.jsonl`, the viewer and its feed. |
| `every` | `1` | Positive interval: export rollouts for updates N, 2N, …; summaries for every update. |
| `max_text_chars` | `0` | Keep this many characters per text, plus a truncation marker; `0` keeps full texts. |

### `[reference]`

The frozen model used by the KL penalty in GRPO, agentic GRPO and on-policy
distillation. PPO takes its KL against the rollout policy and does not use
this section.

Without it, the reference policy is this model with its adapter disabled -
which is the original policy only while the base weights are frozen. A run that
trains base weights and carries a KL term needs this section; a run that
carries no KL term may not declare it.

| Key | Default | Description |
| --- | ---: | --- |
| `model` | required | GGUF the anchor is loaded from. |
| `ctx` | `training.ctx` | Context width of the anchor. Never narrower than `training.ctx`. |

The anchor's precision is the one in its file. There is no conversion knob: a
setting that re-quantized it on load would make the penalty depend on a number
the document chose rather than on the model the path names.

The anchor's tokenizer is compared with the trained model's before it is used,
and its file's content fingerprint is recorded in every checkpoint, so a resume
refuses an anchor that is not the one the first half of the run measured
against.

### `[checkpoint]`

| Key | Default | Description |
| --- | ---: | --- |
| `directory` | required | Checkpoint directory. |
| `mode` | required | `steps`, `best_eval`, or `steps_and_best_eval`. |
| `every_steps` | required for step modes | Positive optimizer-step interval. Do not set it for `best_eval` alone. |
| `resume_from` | none | A complete `.state` checkpoint, or an adapter path whose sibling state directory resolves unambiguously. Mutually exclusive with `lora.init_adapter`. |

## SFT section

| Key | Default | Description |
| --- | ---: | --- |
| `sft.data` | required | Training file. |
| `sft.data_format` | inferred | `text`/`txt` or `jsonl`/`chat`/`chat-jsonl`. |
| `sft.shuffle` | `true` | Shuffle training rows between epochs using `lora.seed`. |

See [SFT training](../training/sft) for the data contract and a complete file.

## PPO sections

### `[ppo]`

| Key | Default | Description |
| --- | ---: | --- |
| `prompts` | required | Chat JSONL prompts ending in a user message. |
| `reward_command` | required | Executable argv array. |
| `reward_mode` | `persistent` | `persistent` or `oneshot`. |
| `reward_timeout_seconds` | `300` | Deadline for one reward batch. |
| `updates` | required | Fresh rollout batches. |
| `rollout_batch_size` | required | Rollouts per update. |
| `ppo_epochs` | required | Policy passes over one rollout batch. |
| `clip_range` | required | Strictly between `0` and `1`. |
| `kl_coefficient` | required | Non-negative KL penalty against the policy that generated the rollout. |

### `[ppo.critic]`

| Key | Default | Description |
| --- | ---: | --- |
| `enabled` | `true` | Enable the value head and GAE. |
| `gamma` | `1.0` | Per-token discount in `(0, 1]`. |
| `gae_lambda` | `0.95` | GAE setting in `[0, 1]`. |
| `value_lr` | `0.01` | Value-head Adam learning rate. |
| `value_epochs` | `8` | Full-batch value-head passes per update. |
| `feature_dtype` | `f32` | Host feature storage: `f32`, `f16`, or `bf16`. Requires the critic. |

### `[ppo.sampling]`

| Key | Default | Description |
| --- | ---: | --- |
| `temperature` | required | Sampling temperature. |
| `top_p` | required | Nucleus threshold in `(0, 1]`. |
| `max_new_tokens` | required | Completion token budget. |
| `seed` | required | Sampling seed. |

## GRPO sections

### `[grpo]`

| Key | Default | Description |
| --- | ---: | --- |
| `prompts` | required | Chat JSONL prompts ending in a user message. |
| `reward_command` | required | Executable argv array. |
| `reward_mode` | `persistent` | `persistent` or `oneshot`. |
| `reward_timeout_seconds` | `300` | Deadline for one reward batch. |
| `updates` | required | Grouped rollout updates. |
| `prompts_per_update` | required | Distinct prompts per update. |
| `group_size` | required | Completions per prompt; `2..256`, and no larger than the optimizer window. |
| `grpo_epochs` | required | Optimizer passes over one grouped batch. |
| `clip_range_low` | required | Lower ratio clip in `(0, 1)`. |
| `clip_range_high` | required | Upper ratio clip in `(0, 1)` and at least the lower range. |
| `kl_coefficient` | required | Non-negative fixed-base KL coefficient. |
| `mask_truncated` | `false` | Exclude completions that consume their generation budget from the group baseline and optimizer. |
| `baseline` | `mean` | `mean` or `leave_one_out`/`rloo`. |
| `prompt_order` | `sequential` | `sequential` or `shuffled`. |
| `max_stalled_updates` | `25` | Consecutive zero-signal updates before stopping; `0` disables the stop. |
| `sampling` | required | See the sampling table below; GRPO requires temperature and top-p of `1.0`. |

### Optional GRPO tables

| Section | Keys | Description |
| --- | --- | --- |
| `[grpo.overlong_penalty]` | `buffer_tokens`, `max_penalty` | Soft penalty near the end of the generation budget. `buffer_tokens` must be below `max_new_tokens`. |
| `[grpo.kl_schedule]` | `warmup_updates`, `target` | KL warmup and optional adaptive target; requires `kl_coefficient > 0`. |
| `[grpo.dynamic_sampling]` | `max_resample_factor` | Replacement groups after zero-signal filtering; must be at least `2`. |
| `[grpo.judge]` | judge config plus `judge_weight`, `judge_failure`, `max_judge_dropped_fraction` | Adds a group-relative judge verdict to the reward command score. |

### `[grpo.sampling]`

| Key | Default | Description |
| --- | ---: | --- |
| `temperature` | required, must be `1.0` | On-policy sampling requirement. |
| `top_p` | required, must be `1.0` | On-policy sampling requirement. |
| `max_new_tokens` | required | Completion budget and constant loss denominator. |
| `seed` | required | Sampling seed. |

### `[distill]`

Distillation against a frozen teacher, in one of two modes. See
[the distillation guide](../training/distill.md) for what the objectives are;
this table is the schema.

| Key | Default | Description |
| --- | ---: | --- |
| `mode` | `on_policy` | `on_policy` (the student samples, the teacher scores its tokens) or `topk_offline` (the teacher's precomputed truncated distribution over a fixed corpus). |
| `teacher_path` | required | Teacher GGUF. Must share the student's tokenizer; the run refuses the pair otherwise. Read by the run itself in `on_policy`, and by `retrograd distill-teacher` in `topk_offline`. |

**`mode = "topk_offline"` only.** The three keys below are required in that mode
and refused in the other, rather than ignored: a document that names a sidecar
expects it to be read.

| Key | Default | Description |
| --- | ---: | --- |
| `data` | required | Chat JSONL, the same shape `[sft].data` reads. |
| `sidecar` | required | The `.topk` file `retrograd distill-teacher` produced for `data`. Its header carries the fingerprints of the corpus and of the tokenizer, and a mismatched pair is refused before the first step. |
| `offline_epochs` | `1` | Passes over the corpus, and the resume unit. |

**`mode = "on_policy"` only.** Required there and optional in the schema, so an
offline document does not have to write keys nothing will read.

| Key | Default | Description |
| --- | ---: | --- |
| `prompts` | required | Chat JSONL prompts ending in a user message - the same format `[grpo]` reads. |
| `updates` | required | Rollout updates. |
| `prompts_per_update` | required | Distinct prompts per update. |
| `samples_per_prompt` | `1` | Completions per prompt; `1..256`. Unlike `grpo.group_size`, one is admissible: the advantage is per-token, so a group of one carries signal. |
| `distill_epochs` | `1` | Optimizer passes over one batch. `1` is strictly on-policy - the ratio is exactly `1` and the token weight is the advantage itself. |
| `clip_range_low` | `0.2` | Lower ratio clip in `(0, 1)`; read only when `distill_epochs > 1`. |
| `clip_range_high` | `0.28` | Upper ratio clip in `(0, 1)` and at least the lower range. |
| `weight_clip` | `5.0` | Bound on `\|A_t\|`, in nats. Finite and above zero. |
| `kl_coefficient` | `0.0` | Fixed-base KL coefficient. Zero skips the reference pass entirely: the teacher is already the anchor. |
| `mask_truncated` | `true` | Exclude completions that consume their generation budget from the optimizer. |
| `prompt_order` | `sequential` | `sequential` or `shuffled`. |
| `sampling` | required | Same table as `[grpo.sampling]`, with the same temperature and top-p requirement of `1.0`. |

An **on-policy** `distill` run holds **two models**: the student with its adapter and optimizer
state, and the teacher, which never gets an adapter and therefore costs its
weights plus its KV cache and nothing else. The planner's memory estimate only
carries the teacher when it was given the teacher's geometry, and says so
otherwise (`teacher_absent_from_the_memory_budget`).

An **offline** `distill` run holds one model. It never opens the teacher - the
sidecar is what the teacher left behind - so it is sized as an SFT run is, and
neither the co-residency term nor that warning applies to it.

## Agentic GRPO

`run.algorithm = "agent_grpo"` selects `[agent]`. It uses multi-turn
trajectories rather than one completion and requires either an environment or a
judge to provide a score. The required field is:

```toml
[agent]
scenarios = "scenarios.jsonl"
```

Agentic runs also use the shared model, LoRA, training, evaluation, checkpoint,
and metrics sections. Their additional fields cover rollout limits, tools,
environments, judges, and optional scenario generation. Treat them as a
separate integration surface; a simple SFT, PPO, or GRPO run does not need
`[agent]`.

Optional prompt settings apply to both training and evaluation:

```toml
[agent]
scenarios = "scenarios.jsonl"
system_suffix = "Answer with one tool call and nothing else."
template_variables = { enable_thinking = false }
```

- `system_suffix` appends an instruction to each scenario's system message,
  creating one if absent. Default: `""`.
- `template_variables` passes values to the model's chat template. Supported
  names depend on the template; `enable_thinking` is an example. Default: `{}`.

Changing either option prevents resuming a checkpoint from the previous configuration.
