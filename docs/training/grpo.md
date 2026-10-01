# GRPO

GRPO samples a group of answers for each prompt and rewards each answer by how
much better it scored than the rest of its group. It needs no critic, and the
reward only has to rank answers to the same prompt, not be on an absolute scale.

Retrograd implements Dr. GRPO with the DAPO refinements: no reward
normalization by the group's standard deviation, a constant loss divisor, an
asymmetric clip range, and optional truncation masking. GSPO's sequence-level
importance ratio is available as an option.

## Configuration

```toml
[run]
algorithm = "grpo"

[model]
path = "base.gguf"

[output]
path = "grpo-adapter.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 512
lr = 0.00001

[grpo]
prompts = "prompts.jsonl"
reward_command = ["python3", "reward.py"]
updates = 50
prompts_per_update = 4
group_size = 8
grpo_epochs = 2
clip_range_low = 0.2
clip_range_high = 0.28
kl_coefficient = 0.0
mask_truncated = true

[grpo.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 128
seed = 42
```

```bash
retrograd train grpo.toml
```

Prompts are chat JSONL ending with a user message, and the reward program uses
the same protocol as [PPO](./ppo#reward-program). `examples/smoke_tiny_grpo.toml`
is a runnable example, and `examples/register_machine` a complete one with
held-out evaluation.

## How an update works

1. Take `prompts_per_update` prompts and sample `group_size` answers for each.
2. Score every answer with the reward program.
3. Compute each answer's advantage: its reward minus the group's mean
   (`baseline = "leave_one_out"` excludes the answer itself from the mean).
4. Drop groups where every answer got the same reward: they carry no signal.
   With `mask_truncated = true`, answers cut off at `max_new_tokens` are left
   out first.
5. Make `grpo_epochs` optimizer passes over the batch.

Sampling must be on-policy: `temperature` and `top_p` must both be `1.0`.

If every group has zero signal for `max_stalled_updates` updates in a row
(default `25`), the run stops. This usually means the reward is too easy or
too hard for the model; check the reward spread first.

## Optional controls

```toml
[grpo.overlong_penalty]    # softly penalize answers that approach the budget
buffer_tokens = 16
max_penalty = 1.0

[grpo.kl_schedule]         # needs kl_coefficient > 0
warmup_updates = 10
target = 0.05

[grpo.dynamic_sampling]    # replace zero-signal groups with fresh prompts
max_resample_factor = 3
```

`overlong_penalty` and `mask_truncated` are two ways of handling long answers:
the first teaches the model to finish in time, the second ignores unfinished
answers. Pick one based on whether your reward can judge an incomplete answer.

## Sequence-level importance sampling (GSPO)

By default the clipped ratio between the current and the sampling policy is
taken token by token. With `importance_sampling_level = "sequence"`, each answer
gets one ratio, the geometric mean of its token ratios, and is clipped as a
whole. The advantage is already one value for the whole answer; a per-token
ratio adds noise that grows with the answer's length, and with Mixture-of-Experts
models whose routing changes between updates. This is GSPO (Zheng et al.,
"Group Sequence Policy Optimization", 2025).

```toml
[grpo]
importance_sampling_level = "sequence"
clip_range_low = 0.0003
clip_range_high = 0.0004
```

- A geometric mean stays very close to 1, so the clip ranges must be much
  narrower than per token: the values above are the paper's. With `0.2`, the
  clip almost never binds, and the run logs a warning at startup.
- Those ranges assume long answers, where the mean smooths the token ratios.
  On answers of a few tokens it smooths almost nothing: after a few optimizer
  steps every answer can be outside the band, and the run stops on the
  trust-region guard. Use fewer `grpo_epochs` there, or wider ranges.
- `policy/clip_fraction` is much higher than in token mode: around 10-20 % of
  the tokens is normal, since an answer is clipped all at once. The trust-region
  guard, which stops a run above 90 %, is unchanged.
- The KL penalty toward the reference stays per token.
- This is the `importance_sampling_level = "sequence"` setting of TRL and
  Unsloth. Their `loss_type = "dr_grpo"` has nothing to match here: retrograd
  always uses the Dr. GRPO loss.

## Adding an LLM judge

When correctness is partly qualitative (style, helpfulness, a rubric), a judge
can rank the answers of each group in addition to the reward program. The final
reward is:

```text
reward + judge_weight × verdict     (verdict in [0, 1])
```

```toml
[grpo]
judge_weight = 0.35

[grpo.judge]
type = "ruler"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"   # the key itself never goes in the file
rubric = "Prefer correct, direct answers. Penalize unsupported claims."
cache_path = "judge-cache.jsonl"
```

- A prompt line can carry its own `rubric`, which replaces the run-wide one.
- The reward program can override the weight for one answer by replying
  `{"reward": 0.8, "judge_weight": 0.0}`, for instance when a checker already
  settled it.
- If a judge request fails, the group is dropped (`judge_failure =
  "drop_group"`) or the run stops (`"fail"`). `max_judge_dropped_fraction`
  (default `0.5`) caps how many groups an update may lose.
- `[grpo.judge.strategy] mode = "pairwise"` compares two answers per request.
  It costs more requests but gives the most reliable ranking; see
  [judge strategies](../reference/configuration#judge).
- `type = "command"` uses your own program as the judge instead of an
  OpenAI-compatible endpoint.

The judge only affects training; held-out evaluation uses the reward program.

## Speed and memory

Two settings change how the work is done, never the results:

- `training.generation_concurrency` is the number of answers generated at
  once. Lower it to save memory, raise it for speed.
- `training.shared_prefix_fanout` (default `"auto"`) packs the answers to one
  prompt into a single pass, so the prompt is processed once. When it cannot
  pack, `"auto"` falls back to one answer per pass and logs why; `"off"`
  disables packing.

Packing needs `micro_batch` to hold the prompt plus all packed answers: for a
250-token prompt and eight 15-token answers, `micro_batch = 512` fits and `256`
does not. See [Performance and memory](../operations/performance).

## Watching a run

`[observe]` saves each group's answers, rewards and advantages, shown in the
Trajectories tab of the web interface. See [Observing rollouts](./observe).
