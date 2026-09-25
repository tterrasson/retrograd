# Configuration reference

A run is described by one TOML file. `[run].algorithm` selects the algorithm
and its section: `[sft]`, `[ppo]`, `[grpo]`, `[distill]`, or `[agent]` for
`agent_grpo`. Unknown keys, and sections that do not apply to the run, are
errors. Relative paths resolve against the directory of the TOML file.

## Common sections

### `[run]`

| Key | Default | Description |
| --- | ---: | --- |
| `algorithm` | required | `sft`, `ppo`, `grpo`, `distill` or `agent_grpo`. |
| `verbose` | `false` | Extra runtime logging. |

### `[model]`

| Key | Default | Description |
| --- | ---: | --- |
| `path` | required | Base GGUF model. Can be given with `--model` instead. |
| `device` | `auto` | `auto` (GPU if available, else CPU), `cpu`, or `gpu` (fail without a GPU). |

### `[output]`

| Key | Default | Description |
| --- | ---: | --- |
| `path` | required | Where the result is written. |
| `kind` | see below | `adapter`, `trainable` or `model`. |

| `kind` | Contents | Allowed with |
| --- | --- | --- |
| `adapter` | A standard LoRA GGUF, loadable by llama.cpp. Default for `lora`. | `lora` |
| `trainable` | The trained base tensors (and the adapter for `hybrid`). Needs the base model and Retrograd to load. Default otherwise. | `full`, `partial`, `hybrid` |
| `model` | A complete standalone GGUF with the trained weights. | `full`, `partial` |

### `[lora]`

Required when `training.trainable` is `lora` (the default) or `hybrid`;
refused otherwise.

| Key | Default | Description |
| --- | ---: | --- |
| `rank` | `8` | Adapter rank. |
| `alpha` | `16.0` | Scale; the effective scale is `alpha / rank`. |
| `seed` | `42` | Initialization seed, also used to shuffle SFT data. |
| `targets` | `q, k, v, o, ffn_up, ffn_down, ffn_gate` | Aliases, tensor patterns (`blk.*.attn_q.weight`), or `auto` for architecture-specific targets. `retrograd inspect` lists the candidates. |
| `dtype` | `f16` | Adapter storage: `f16` or `f32`. |
| `init_adapter` | none | Start from an existing adapter GGUF (weights only, no optimizer state). Excludes the other keys and `--resume`. |

### `[trainable]` {#trainable}

Selects base tensors for `training.trainable = "partial"` or `"hybrid"`.
Refused for `lora` and `full`.

| Key | Default | Description |
| --- | ---: | --- |
| `layers` | all | `all`, `last:<n>`, or `<first>..<last>` (inclusive). |
| `modules` | none | `attn`, `ffn`, a single projection (`attn_q`), or a tensor pattern. |
| `norms` | `false` | Train normalization weights. |
| `biases` | `false` | Train the biases of the selected modules. |
| `output_head` | `false` | Train the output projection. Not possible when it shares storage with the input embedding. |

At least one selector must match something. `hybrid` allows only `norms` and
`biases` next to the adapter. Quantized tensors, the input embedding and
rotary constants are never trained; the run lists what it excluded at start-up.

### `[training]`

| Key | Default | Description |
| --- | ---: | --- |
| `trainable` | `lora` | `lora`, `full`, `partial` or `hybrid`. See [SFT](../training/sft#training-base-weights). |
| `optimizer` | `adamw` | `adamw`, `sgd`, `muon` or `gefen`. |
| `lr` | `0.0001` | Learning rate. |
| `lr_scheduler` | `constant` | `constant`, `linear` or `cosine`. `linear` and `cosine` decay to zero by the end of the run. |
| `warmup_steps` | `0` | Warm-up steps. |
| `weight_decay` | `0.0` | Decoupled weight decay. |
| `max_grad_norm` | `1.0` | Global gradient-norm clipping. |
| `epochs` | `1` | Passes over the SFT dataset. Rollout algorithms use their own counters. |
| `ctx` | `128` | Context length in tokens. |
| `micro_batch` | `32` | Tokens per forward/backward pass. The main memory knob. |
| `gradient_accumulation` | `1` | Passes per optimizer step; `micro_batch × gradient_accumulation` must divide `ctx`. Fixed to `ctx / micro_batch` for rollout algorithms: leave it out. |
| `threads` | automatic | CPU threads. Leave it out for automatic; `0` is refused. |
| `generation_concurrency` | derived | Answers generated at once (GRPO, agentic GRPO, distillation). Lower to save memory. |
| `generation_batch` | derived | Generation batch size. |
| `shared_prefix_fanout` | `auto` | GRPO packing of answers that share a prompt: `auto`, `off`, `max`, or an integer ≥ 2. |
| `fast_sampling_context` | `true` | F16 KV cache and flash attention for generation. `false` for bit-exact sampling. |
| `kv_dtype` | `f16` | KV cache type for training: `f16` or `f32`. Falls back to F32 when the device requires it. |
| `gradient_checkpointing` | `false` | Recompute activations to save memory. |
| `checkpoint_every_n_layers` | `4` | Checkpoint stride when gradient checkpointing is on. |
| `checkpoint_dtype` | `f32` | Stored activation precision: `f32`, `f16` or `bf16`. Needs gradient checkpointing. |
| `chunked_cross_entropy` | `true` | Compute the loss in vocabulary tiles to save memory (rollout algorithms). |
| `chunked_ce_tiles` | `8` | Number of vocabulary tiles. |
| `chunked_ce_seq_chunk` | `512` | Tokens per tile chunk; `0` for all at once. |
| `master_weights` | `auto` | F32 master copy for F16/BF16 base weights: `auto`, `f32` or `off`. Costs 4 bytes per trained parameter. |
| `require_gpu_resident` | `false` | Fail if any training operation would run on the CPU. |
| `max_gpu_duty_cycle` | `1.0` | Cap on the share of time spent on the GPU, in `(0, 1]`. See [Performance](../operations/performance#sharing-a-gpu). |

**Optimizers.** AdamW and SGD can update F16 and BF16 weights. Muon and Gefen
update F32 weights only, so they need `lora.dtype = "f32"` for an adapter. The
last two read their settings from `[optimizer.<name>]`; parameters they do not
handle are updated with AdamW.

### `[optimizer.muon]`

Muon applies to 2-D hidden weight matrices; everything else, including LoRA
factors, uses AdamW at `fallback_learning_rate`.

| Key | Default | Description |
| --- | ---: | --- |
| `momentum` | `0.95` | Momentum coefficient, in `[0, 1]`. |
| `nesterov` | `true` | Nesterov momentum. |
| `ns_steps` | `5` | Newton-Schulz iterations. |
| `ns_epsilon` | `1e-7` | Normalization epsilon. |
| `fallback_learning_rate` | `0.001` | AdamW learning rate for parameters Muon does not handle. |

### `[optimizer.gefen]`

Experimental. Block-wise second moments to reduce optimizer memory.

| Key | Default | Description |
| --- | ---: | --- |
| `variant` | `shared_v` | `shared_v`, or `quantized_m` (8-bit first moment). Checkpoints are not interchangeable between variants. |
| `block_size` | `1024` | Elements per block; a power of two. |
| `min_numel` | `4096` | Smaller parameters use AdamW. |
| `beta1` | `0.9` | First-moment coefficient. |
| `beta2` | `0.999` | Second-moment coefficient. |
| `eps` | `1e-8` | Update epsilon. |

`codebook = "uniform"`, `codebook_levels = 256` and `partition = "fixed"` are
the only accepted values of the remaining keys.

### `[evaluation]`

| Key | Default | Description |
| --- | ---: | --- |
| `data` | required | Held-out file, in the training data format. |
| `every_iterations` | `1` | Evaluate every N epochs (SFT) or updates (rollout algorithms). |
| `max_examples` | all | Rollout algorithms: cap on evaluated prompts, spread evenly across the file. |
| `patience` | none | Stop after N evaluations without improvement. |
| `min_delta` | `0.0` | Smallest change counted as an improvement. |

### `[checkpoint]`

| Key | Default | Description |
| --- | ---: | --- |
| `directory` | required | Checkpoint directory. |
| `mode` | required | `steps`, `best_eval` or `steps_and_best_eval`. The last two need `[evaluation]`. |
| `every_steps` | required for step modes | Save every N optimizer steps. |
| `resume_from` | none | Checkpoint to resume from; same as `train --resume`. |

See [Checkpoints](../operations/checkpoints).

### `[metrics]`

| Key | Default | Description |
| --- | ---: | --- |
| `tensorboard_dir` | none | TensorBoard log directory. |
| `wandb_export_dir` | none | Offline export for Weights & Biases. |

### `[observe]`

PPO, GRPO and agentic GRPO. See [Observing rollouts](../training/observe).

| Key | Default | Description |
| --- | ---: | --- |
| `directory` | required | Output directory. |
| `every` | `1` | Record rollouts every N updates. |
| `max_text_chars` | `0` | Truncate texts to N characters; `0` keeps them whole. |

### `[reference]`

The frozen model used by the KL penalty (`kl_coefficient > 0`) in GRPO,
agentic GRPO and on-policy distillation. Without it, the reference is the base
model without its adapter, which only works while base weights are frozen: a
run that trains base weights with a KL penalty requires this section.

| Key | Default | Description |
| --- | ---: | --- |
| `model` | required | Reference GGUF. Must share the model's tokenizer. |
| `ctx` | `training.ctx` | Context length; at least `training.ctx`. |

## `[sft]`

| Key | Default | Description |
| --- | ---: | --- |
| `data` | required | Training file. |
| `data_format` | inferred | `text` or `jsonl`. See [Datasets](../getting-started/datasets#format-detection). |
| `shuffle` | `true` | Shuffle rows at each epoch. |

## `[ppo]` {#ppo}

| Key | Default | Description |
| --- | ---: | --- |
| `prompts` | required | Chat JSONL prompts. |
| `reward_command` | required | Reward program, as an argument list. |
| `reward_mode` | `persistent` | `persistent` or `oneshot`. See [reward program](../training/ppo#reward-program). |
| `reward_timeout_seconds` | `300` | Deadline for one batch of rewards. |
| `updates` | required | Number of rollout batches. |
| `rollout_batch_size` | required | Answers per update. |
| `ppo_epochs` | required | Optimizer passes per batch. |
| `clip_range` | required | In `(0, 1)`, typically `0.2`. |
| `kl_coefficient` | required | Penalty toward the policy that generated the batch; `≥ 0`. |

`[ppo.critic]`:

| Key | Default | Description |
| --- | ---: | --- |
| `enabled` | `true` | Use a value head and GAE advantages. |
| `gamma` | `1.0` | Discount, in `(0, 1]`. |
| `gae_lambda` | `0.95` | GAE λ, in `[0, 1]`. |
| `value_lr` | `0.01` | Value head learning rate. |
| `value_epochs` | `8` | Value head passes per update. |
| `feature_dtype` | `f32` | Stored feature precision: `f32`, `f16` or `bf16`. |

`[ppo.sampling]` (all required): `temperature`, `top_p` (in `(0, 1]`),
`max_new_tokens`, `seed`.

## `[grpo]` {#grpo}

| Key | Default | Description |
| --- | ---: | --- |
| `prompts` | required | Chat JSONL prompts. |
| `reward_command` | required | Reward program, as an argument list. |
| `reward_mode` | `persistent` | `persistent` or `oneshot`. |
| `reward_timeout_seconds` | `300` | Deadline for one batch of rewards. |
| `updates` | required | Number of updates. |
| `prompts_per_update` | required | Prompts per update. |
| `group_size` | required | Answers per prompt, `2` to `256`. |
| `grpo_epochs` | required | Optimizer passes per update. |
| `clip_range_low` | required | Lower clip, in `(0, 1)`, e.g. `0.2`. |
| `clip_range_high` | required | Upper clip, at least the lower one, e.g. `0.28`. |
| `kl_coefficient` | required | KL penalty toward the reference; `0` disables it. |
| `mask_truncated` | `false` | Ignore answers cut at `max_new_tokens`. |
| `baseline` | `mean` | `mean` or `leave_one_out`. |
| `prompt_order` | `sequential` | `sequential` or `shuffled`. |
| `max_stalled_updates` | `25` | Stop after N updates in a row without signal; `0` never stops. |
| `judge_weight` | required with a judge | Weight of the judge's verdict. |
| `judge_failure` | `drop_group` | `drop_group` or `fail`. |
| `max_judge_dropped_fraction` | `0.5` | Largest share of groups an update may lose to judge failures. |

`[grpo.sampling]` (all required): `temperature = 1.0`, `top_p = 1.0` (both
enforced), `max_new_tokens`, `seed`.

Optional tables:

| Table | Keys | Description |
| --- | --- | --- |
| `[grpo.overlong_penalty]` | `buffer_tokens`, `max_penalty` | Penalty over the last `buffer_tokens` of the budget. |
| `[grpo.kl_schedule]` | `warmup_updates`, `target` | KL warm-up and adaptive target; needs `kl_coefficient > 0`. |
| `[grpo.dynamic_sampling]` | `max_resample_factor` | Replace zero-signal groups, up to this multiple of `prompts_per_update` (≥ 2). |
| `[grpo.judge]` | see [Judge](#judge) | LLM or command judge. |

## `[distill]` {#distill}

| Key | Default | Description |
| --- | ---: | --- |
| `mode` | `on_policy` | `on_policy` or `topk_offline`. |
| `teacher_path` | required | Teacher GGUF, with the student's tokenizer. |

On-policy only:

| Key | Default | Description |
| --- | ---: | --- |
| `prompts` | required | Chat JSONL prompts. |
| `updates` | required | Number of updates. |
| `prompts_per_update` | required | Prompts per update. |
| `samples_per_prompt` | `1` | Answers per prompt, `1` to `256`. |
| `distill_epochs` | `1` | Optimizer passes per update. |
| `clip_range_low` / `clip_range_high` | `0.2` / `0.28` | Used only when `distill_epochs > 1`. |
| `weight_clip` | `5.0` | Cap on a single token's weight, in nats. |
| `kl_coefficient` | `0.0` | Extra KL penalty toward the base model. |
| `mask_truncated` | `false` | Ignore answers cut at `max_new_tokens`. |
| `prompt_order` | `sequential` | `sequential` or `shuffled`. |
| `[distill.sampling]` | required | As `[grpo.sampling]`. |

Offline only:

| Key | Default | Description |
| --- | ---: | --- |
| `data` | required | Chat JSONL corpus. |
| `sidecar` | required | The `.topk` file written by `retrograd distill-teacher`. |
| `offline_epochs` | `1` | Passes over the corpus. |

Keys of one mode are refused in the other.

## Agentic GRPO {#agentic-grpo}

`run.algorithm = "agent_grpo"` reads `[agent]`. See the
[agentic GRPO guide](../training/agent). A run needs `[agent.judge]`,
`[agent.environment]`, or both.

### `[agent]`

| Key | Default | Description |
| --- | ---: | --- |
| `scenarios` | required | Scenario JSONL file. |
| `updates` | `1` | Number of updates. |
| `scenarios_per_update` | `1` | Scenarios per update. |
| `group_size` | `8` | Trajectories per scenario (≥ 2). |
| `epochs_per_update` | `4` | Optimizer passes per update. |
| `max_turns` | `6` | Assistant turns per trajectory. |
| `max_new_tokens_per_turn` | `512` | Tokens per turn. |
| `max_trajectory_tokens` | model context | Token budget for a whole trajectory. |
| `max_rollout_secs` | `300` | Time budget per trajectory; `0` disables it. |
| `end_on_no_tool_call` | `true` | End the trajectory on a turn without a tool call. |
| `max_failed_turns` | `0` | Cut a trajectory after N turns in a row without a valid call; `0` never cuts. |
| `truncation` | `drop` | `drop` or `min_reward` for trajectories that hit a limit. |
| `max_dropped_fraction` | `0.5` | Stop when more than this share of an update is lost. |
| `skip_empty_updates` | `false` | Skip, rather than fail, an update with fewer than two usable trajectories. |
| `judge_failure` | `drop_group` | `drop_group` or `fail`. |
| `drop_degenerate_groups` | `false` | Drop groups the judge scored identically. |
| `clip_range_low` / `clip_range_high` | `0.2` / `0.28` | Clip range. |
| `kl_coefficient` | `0.0` | KL penalty toward the reference. |
| `system_suffix` | `""` | Text appended to every system message. |
| `template_variables` | `{}` | Chat template variables, e.g. `{ enable_thinking = false }`. |
| `seed` | `42` | Seed. |
| `mcp_config` | none | Path, or list of paths, to `mcp.json` files. |

### `[[agent.mcp_servers]]`

| Key | Default | Description |
| --- | ---: | --- |
| `name` | required | Server name. |
| `command` / `url` | one required | Local server command (with optional `env`), or remote URL (with optional `headers`). |
| `allowed_tools` / `denied_tools` | all / none | Tool name filters; `*` is a wildcard and deny wins. |
| `required` | `true` | Fail the run if the server cannot connect. |
| `stateless` | `false` | Must be `true` when an environment is declared. |
| `tool_timeout_secs` | `30` | Per-call timeout. |
| `max_tool_result_bytes` | `65536` | Cap on a tool result. |
| `cwd` | inherited | Working directory of a local server. |
| `env_passthrough` | all | Names of the environment variables a local server inherits; `PATH` and `HOME` always are. |

### `[agent.environment]`

`type` is `container`, `http` or `local`.

| Key | Default | Description |
| --- | ---: | --- |
| **container** | | Requires the `container` build feature. |
| `profile` | `python` | Default image and package cache: `python`, `typescript` or `custom`. |
| `image` | profile's | Container image; prefer a digest. |
| `tools` | required | Toolsets, see below. |
| `allow_network` | `false` | Allow outbound network. |
| `cache_volume` | none | Volume mounted read-only on the profile's package cache, so `reuse = "workspace"` installs once per run. |
| `setup_timeout_secs` | `300` | Timeout for a scenario's `setup`. |
| `verify_timeout_secs` | none | Timeout for a scenario's `verify`. |
| `limits` | | `cpus = 1.0`, `memory_mb = 1024`, `pids = 256`, `exec_timeout_secs = 30`, `max_output_bytes = 65536`. |
| `pool` | | `max_live = 8`, `min_idle = 0`, `reuse = "never"` (or `"workspace"`), `max_leases_per_container = 32`. |
| **http** | | |
| `base_url` | required | Environment server URL. |
| `request_timeout_secs` | `60` | Timeout per request. |
| `connect_timeout_secs` | `10` | Connection timeout. |
| `pool_size` | `16` | Concurrent connections; keep at least `group_size`. |
| `max_result_bytes` | `65536` | Cap on an observation. |
| `headers` | none | Headers added to every request. |
| **local** | | Runs tools on the host, without isolation. |
| `allow_unsandboxed` | `false` | Must be `true`. |
| `tools`, `setup_timeout_secs`, `verify_timeout_secs` | | As for containers. |

### `[agent.environment.tools]`

For `container` and `local`. See [Tools and toolsets](../training/tools).

| Key | Default | Description |
| --- | ---: | --- |
| `default` | required | Toolset of a scenario that names none (`base`, `python`, `typescript`, or one you define). |
| `scenario_toolsets` | none | Other toolsets a scenario may select with `metadata.env.toolset`. |
| `files` | none | Definition files (`[[tool]]`, `[toolset.NAME]`), relative to this config. |
| `[[...tools.tool]]` | none | Inline tool definitions: `id`, `version`, `name`, `description`, `input_schema`, and `builtin` or `exec`. |
| `[...tools.toolset.NAME]` | none | Inline toolsets: `include`, `tools` (`id` or `id@version`), `deny`. |

### `[agent.scenario_generation]`

Used by `retrograd scenarios generate`. `model`, `base_url` and `api_key_env`
are required; `count` (`24`), `batch_size` (`12`), `min_difficulty` /
`max_difficulty` (`1` / `5`, between 1 and 5), `custom_instructions`, `seed`,
`shuffle` (`true`), `timeout_secs` (`120`), `max_retries` (`2`) and
`max_catalog_bytes` (`262144`) are optional.

## Judge {#judge}

`[grpo.judge]` and `[agent.judge]` share the same format.

`type = "command"`: `command` (required, argument list), `timeout_secs`
(`30`).

`type = "ruler"`, an OpenAI-compatible LLM judge:

| Key | Default | Description |
| --- | ---: | --- |
| `base_url`, `model` | required | Endpoint and model. |
| `api_key_env` | `OPENAI_API_KEY` | Environment variable holding the API key. |
| `rubric` | built-in | Grading criteria. A prompt or scenario can set its own. |
| `pairwise_rubric` | built-in | Criteria for pairwise comparisons. |
| `temperature` | endpoint default | `0.0` for the most repeatable verdicts. |
| `max_concurrency` | `4` | Parallel requests. |
| `timeout_secs` | `120` | Request timeout. |
| `max_retries` | `2` | Retries on transient errors. |
| `cache_path` | none | Verdict cache file. |

`[<section>.judge.strategy]`:

| Key | Default | Description |
| --- | ---: | --- |
| `mode` | `auto` | `auto` (one request per group, split into chunks if too long), `listwise` (always one request), `chunked` (always split), `pairwise` (two answers per request, most reliable, most requests). |
| `anchor` | `true` | Chunked: repeat one answer in every chunk to keep scores on one scale. |
| `max_pairs` | all pairs | Pairwise: comparisons per group. |
| `both_orders` | `true` | Pairwise: judge each pair in both orders to cancel position bias (doubles requests). |
| `aggregation` | `win_rate` | Pairwise: `win_rate` or `bradley_terry` (better when `max_pairs` limits comparisons). |

`[<section>.judge.context]`, budgets in characters: `max_request_chars`
(`60000`), `max_trajectory_chars` (`8000`), `max_message_chars` (`2000`),
`head_ratio` (`0.4`, share of an elided message kept from its start),
`include_env_state` (`true`, show the environment's final state to the judge).

`[<section>.judge.compaction]`, optional: summarize the middle of long
transcripts with the judge model. `trigger_chars`, `target_chars` and
`keep_last` are required.
