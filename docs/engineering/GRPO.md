# GRPO training

Retrograd trains a LoRA adapter with Dr. GRPO (GRPO Done Right, the
bias-corrected variant of DeepSeekMath's Group Relative Policy Optimization)
end to end through the llama.cpp fork. GRPO is PPO without the learned critic:
every prompt is sampled `group_size` times and a completion's advantage is its
reward *centered within its own group* (`r − mean_group`). Unlike original
GRPO, Dr. GRPO removes the two normalizations that bias the objective: the
advantage is **not** divided by the group's reward std (which over-weights
low-variance prompts), and the loss is divided by the constant generation
budget `max_new_tokens` instead of each completion's own length (which
rewards rambling on negative advantages). It shares PPO's exact
clipped-surrogate gradient, but regularizes KL toward a fixed base-model
reference ([PPO.md](PPO.md)) - no Python, no second framework, no value head.

On top of Dr. GRPO, three corrections from DAPO apply:

- **Clip-Higher.** The surrogate clip band is decoupled: `clip_range_low`
  bounds the ratio from below (`1 − low`), `clip_range_high` from above
  (`1 + high`, typically looser, e.g. 0.2 / 0.28). A symmetric band caps how
  much a low-probability token with positive advantage can grow at `1 + ε`
  of almost nothing, which drives entropy collapse; the looser upper range
  relaxes exactly that direction.
- **Zero-signal group filtering.** A group whose rewards are all identical
  carries no learning signal; instead of taking KL-only steps on its tokens,
  the whole group is dropped from the optimizer epochs (and from the
  fixed-reference scoring pass). Always on.
- **Truncation masking** (`mask_truncated`, optional, off by default). A
  completion that used its entire generation budget was truncated, so its
  reward judges an incomplete response. When enabled, such completions are
  excluded from their group's baseline and from the epochs; a group left
  with fewer than two unmasked members is dropped entirely.

Dr. GRPO is strictly on-policy: sampling must use the unmodified model
distribution, so `temperature = 1` and `top_p = 1` are enforced at
configuration time.

## Running it

A complete document is in the [GRPO guide](/training/grpo#configuration), and
every key with its default is in the
[configuration reference](/reference/configuration#grpo-sections). This page
describes how the loop is implemented.

Shared-prefix fanout changes only physical execution, not GRPO statistics: a
reward group remains one logical AdamW update even when it is split into
several packed graphs. Each graph contains its own prompt copy plus at most the
selected fanout of completions, and their gradients accumulate before the
single optimizer step. An explicitly selected fanout (`max` or an integer)
fails if the model or `micro_batch` cannot represent it; `auto` may reduce the
fanout or use the row path. Runs expose the selected fanout, number of passes,
shared-prefix fraction, physical-token count, and padding fraction under the
`packing/` metric namespace.

As with PPO, each prompt uses the chat `messages` JSONL envelope, may contain
system instructions and earlier turns, and must end in a non-empty `user`
message. The full conversation is formatted through the GGUF chat template
with an assistant-generation prefix before sampling.

## GRPO counters: `epochs`, `updates`, and `grpo_epochs`

| Setting | Applies to | Meaning |
| --- | --- | --- |
| `training.epochs` | SFT only | Complete passes over the fixed supervised dataset. It does **not** control GRPO and is ignored by the GRPO loop. |
| `grpo.updates` | GRPO | Number of fresh grouped rollout batches: sample completions, obtain rewards, and compute group-relative advantages. |
| `grpo.grpo_epochs` | GRPO | Number of optimizer passes over the *same* grouped rollout batch before sampling new groups. |

For example, `updates = 2`, `grpo_epochs = 3`, `prompts_per_update = 4`, and
`group_size = 8` samples 32 completions twice, then makes three policy passes
over each of those two sampled batches: six GRPO optimizer epochs total.

### The two geometry knobs

`[training]` describes the optimizer step with the same two quantities TRL
does, and nothing else:

| Retrograd | TRL | What it is |
| --- | --- | --- |
| `micro_batch` | `per_device_train_batch_size` | Physical forward/backward width, in tokens (llama.cpp's `n_ubatch`). **The activation-memory lever.** |
| `gradient_accumulation` | `gradient_accumulation_steps` | Micro-batches accumulated before one AdamW step. |
| `ctx` | `max_seq_length` | The trained window: prompt + `max_new_tokens` must fit here. |

Their product - `micro_batch * gradient_accumulation` - is the token window one
optimizer step trains, and it must divide `ctx`. That window is *not* a memory
knob: it only sizes llama.cpp's host-side output buffer
(`window * n_vocab * 4`), which the training path reserves and never reads.
Lower `micro_batch` to bound memory.

On a rollout algorithm `gradient_accumulation` is not a free parameter and is
best left out of the file: Retrograd pins it to `ctx / micro_batch`, so one
optimizer step spans the whole context. A packed row is one rollout, and a
smaller window would turn a single completion into several AdamW steps - most
of them over prompt positions carrying no label - leaving the trust region the
clipped surrogate assumes inside the first update. A value that contradicts the
pinned one is reported, not silently corrected.

That geometry bounds `global_step` from above, it does not pin it: it guarantees
no rollout is ever split across several optimizer steps, but the reverse
grouping still happens. Gradient accumulation is carried across row boundaries
and a row is only evaluated up to its last supervised label, so an epoch packs
consecutive rollouts into one step while their combined micro-batches fit the
window. Short completions therefore share a step and `global_step` advances
*at most* once per rollout per epoch - the exact count depends on the sampled
completion lengths and is reported, not predictable from the configuration
alone.

Each update samples `prompts_per_update * group_size` rollouts, so the cost
per update is that of a PPO update with the same product as its
`rollout_batch_size`.

### Generation concurrency and VRAM

`training.generation_concurrency` controls how many rollout sequences share the
dedicated generation context. Generation KV-cache memory grows linearly with
this value. It does not change the number of rollouts collected, the GRPO group
size, or the optimizer batch: groups that do not fit in one physical wave are
reassembled before reward normalization and training.

The valid range starts at `1`. The value must not exceed
`prompts_per_update * group_size`, the optimizer window
(`micro_batch * gradient_accumulation`), or the backend limit of 256
sequences. If the setting is omitted, Retrograd retains the previous
maximum-throughput behavior up to those limits.

For `prompts_per_update = 4` and `group_size = 8`, useful settings are:

| Setting | Behavior |
| ---: | --- |
| `1` | Minimum persistent generation KV memory; completions are decoded one at a time. |
| `8` | One complete GRPO group per wave; balanced VRAM and throughput. |
| `16` | Two groups per wave. |
| `32` | All rollouts in one continuous batch; maximum throughput and KV memory. |

The exact memory per sequence depends on the model and `training.ctx`. For a
fixed model it is linear in both `generation_concurrency` and `ctx`.
`fast_sampling_context = true` normally halves generation KV memory by using
F16 instead of F32 storage. Consequently, the lowest-VRAM configuration is:

```toml
[training]
generation_concurrency = 1
fast_sampling_context = true
```

Increase `generation_concurrency` until throughput stops improving or the
desired VRAM headroom is reached. `training.micro_batch` is the separate
primary knob for optimizer workspace memory; changing `ctx`, `group_size`, or
`prompts_per_update` also changes the training problem and should not be used
only as memory tuning controls.

### Fast sampling context (default)

Sampling runs on a dedicated generation context, separate from the optimizer
context. By default, `training.fast_sampling_context = true` builds that context with an
F16 KV cache and flash-attention instead of mirroring the optimizer's exact
F32 / no-flash-attention settings:

- KV cache memory halves, which is what caps `group_size * n_ctx` on long
  prompts (the agentic setup in [AGENTIC_GRPO.md](AGENTIC_GRPO.md) first).
- Decoding gets faster, increasingly so as the prompt grows.

The optimizer context is never touched: `llama_opt` differentiates through the
non-flash-attention graph and needs the F32 cache, and `old_logprobs` are still
re-scored on it, so **all ratios remain 1 before the first optimizer step**.
(On a GPU that gathers the behavior log-probabilities inside the decode graph -
`cap_device_logprobs`, [`optims/SAMPLING.md`](optims/SAMPLING.md) S6 - that reduction is F32 rather
than the host oracle's double, which puts ~1e-5 of F32 noise on each
log-probability. `RETRO_DEVICE_LOGPROBS=0` restores the exact oracle.) What the flag does change is that the tokens were drawn from a policy
that differs from the trained one by rounding - an ulp-level off-policy bias
with no importance correction. Set `fast_sampling_context = false` for a
bit-exact reference run.

### Reward protocol

Identical to PPO's. The reward command is executed directly (never through a
shell), receives one `{"prompt": "...", "completion": "..."}` JSON line per
rollout on stdin (`prompt` is the final user message, without chat-template
markers), and must print one `{"reward": <finite number>}` line per input, in
order. A non-zero exit, malformed line, or count mismatch aborts the run.

`reward_mode` is PPO's too, and it matters more here: GRPO calls its reward
once per *sampling wave*, so with `[grpo.dynamic_sampling]` an update scores up
to `max_resample_factor` batches. The default `"persistent"` answers all of
them from one worker, which has to complete the `retrograd-reward/1` handshake
and flush each response line; `"oneshot"` spawns one process per batch, as
every reward command did before. Both are described in full under
[PPO.md](PPO.md), "One worker, not one process per batch".

A response line may also carry `"judge_weight": <finite number ≥ 0>`, which
only `[grpo.judge]` below honours. Without a judge to weigh, the field is an
error rather than something quietly dropped: a reward process that asks for one
was written against a loop that would have used it.

### A judge next to the reward command

`[grpo.judge]` adds an LLM verdict to what the reward command already scored.
It is the same section as `[agent.judge]` - same backend, same transport, same
on-disk verdict cache, same `judge/*` metrics - because it grades the same
shape: a *group*, i.e. the `group_size` completions of one prompt, compared
against each other. That is exactly the unit GRPO centres its advantage on.

```toml
[grpo]
reward_command = ["python3", "reward.py"]
# Weight of a verdict for a completion whose reward line states none. Required
# as soon as a judge is declared.
judge_weight = 0.30
judge_failure = "drop_group"        # or "fail"
max_judge_dropped_fraction = 0.5

[grpo.judge]
type = "ruler"                      # or "command"
base_url = "https://…/v1/"
model = "…"
api_key_env = "OPENAI_API_KEY"
cache_path = "/tmp/judge.cache"
temperature = 0.0
```

The final reward of a completion is `reward + judge_weight * verdict`, with
`verdict` in [0, 1]. Weighing per line rather than per run is what lets the
verifiable part keep the answers it can settle on its own: a completion it
matched exactly, or rejected on format, is already ranked, and paying for a
verdict on it only adds noise. The reward process states that with a
`judge_weight` of zero, and the document's `judge_weight` covers the lines that
say nothing. A group whose members *all* weigh the verdict at zero is not sent
at all: no request, and no exposure to a judge outage that could not have
changed one of its rewards.

Two consequences of a RULER verdict being *relative inside a group* - it says
which of these completions is better, not what any of them is worth in the
absolute:

- a group is judged whole or not at all. A group the judge failed on keeps its
  verifiable reward and is excluded from the baseline and the optimizer epochs,
  exactly like a group without spread; losing more than
  `max_judge_dropped_fraction` of an update's groups stops the run, because a
  judge that answers nothing must be seen rather than trained through. The
  threshold is counted over the whole update - dynamic sampling judges one batch
  per resampling wave, and a bad wave that the next one made up for is not a
  broken judge. Under `judge_failure = "fail"` the first failed group stops it
  outright, wherever it is.
- `[evaluation]` never calls the judge. A held-out pass generates one
  completion per prompt, so there is no group to rank, and a score renormalized
  inside a group of one would make two passes incomparable - which is the one
  thing an evaluation is for. Held-out reward is the reward command alone, and
  `checkpoint.mode = "best_eval"` selects on that.

A prompt dataset line may carry a `rubric` field next to `messages`, holding
the criteria for that prompt alone. It reaches the judge as the group's own
rubric (replacing the run-wide `rubric` in listwise mode, appended to it in
pairwise), and it is the only way to show a judge what an expert would have
answered without that answer becoming a training target - a GRPO prompt stops
at the user message. Nothing else reads the field.

## How an update works

For each of `updates` iterations:

1. **Group rollout.** `prompts_per_update` prompts are taken round-robin from
   the prompts file. Each prompt is tokenized once - a prompt whose tokens
   plus the full `max_new_tokens` budget do not fit the trained window is
   rejected up front - and sampled `group_size` times with distinct derived
   seeds (model distribution, deterministic per seed) up to `max_new_tokens`
   or an end-of-generation token. As in PPO, the rollout-time logprobs
   (`old_logprobs`) are captured by re-scoring each sequence through the same
   teacher-forced path used later. All ratios are exactly 1 before the first
   optimizer step; subsequent size-one SGD steps legitimately move the policy
   before later rollouts.
2. **Reward.** All (prompt, completion) pairs of the wave go to
   `reward_command` in one batch - one exchange with the persistent worker, or
   one process under `reward_mode = "oneshot"`. With `[grpo.judge]`, each group then goes to the judge in one
   request, and the verdicts are blended into the returned scores before
   anything else reads them - including dynamic sampling, which resamples on
   the reward an update is actually trained on.
3. **Group-relative advantage and filtering.** With `mask_truncated`
   enabled, completions that used the entire budget are first masked out.
   Within each group, the surviving rewards are centered against their own
   mean: `A_i = r_i − mean_group`, constant over the tokens of completion
   `i`. There is deliberately no division by the group's reward std. A group
   without signal - all live rewards equal, or fewer than two live members -
   is marked dead: its rollouts are excluded from the reference scoring and
   the optimizer epochs rather than taking KL-only steps.
4. **Fixed reference score.** Each live sequence is scored once with the
   LoRA temporarily disabled. This produces `log π_ref` from the immutable
   base model; the same reference is used for every update and optimizer
   epoch. Scored group by group through the same shared-prefix batch as the
   behavior pass, so a group costs one prefix decode rather than `group_size`
   full prompt prefills.
5. **GRPO epochs.** Each epoch visits every live rollout once, in a
   deterministic seeded shuffle (re-shuffled per update and epoch), and
   re-scores the sequence under the *current* policy. The clipped surrogate
   ratio uses `π/π_old` in the Clip-Higher band `[1 − clip_range_low,
   1 + clip_range_high]`, while the k3 KL term uses `π_ref/π`. One
   stochastic minibatch-size-one weighted optimizer step then runs through
   the runtime. Weights are recomputed immediately before every step, so
   ratios are not stale.

Progress events carry metrics grouped by type so each TensorBoard tag group
stays small: under `reward/` - `mean`, `min`, `max`, `std` (across the whole
batch), `group_std` (mean within-group std over live members, the
collapse diagnostic), and `verifiable_mean` / `judge_mean`, which split `mean`
into the reward command's own figure and what the judge added on top
(`verdict * judge_weight`, zero without a judge, and the two always sum back to
`mean`). That split is what makes a training reward comparable with
`eval/mean_reward`: an evaluation generates one completion per prompt, so there
is no group to rank and it drops the judge term deliberately - without the
split, a rising `reward/mean` against a falling `eval/mean_reward` cannot be
told apart from the two simply measuring different quantities. Under
`completions/` - `length_mean` / `length_min` /
`length_max` and `truncation_fraction` (fraction of completions that used the
entire `max_new_tokens` budget - one that emits its end-of-generation token
exactly on the last budgeted position counts too); under `batch/` -
`zero_std_group_fraction` (fraction of zero-signal groups: all live rewards
identical, or fewer than two live members), `advantage_abs_mean` (over
trained rollouts), and `trained_fraction` (fraction of rollouts that survived
zero-signal filtering and truncation masking and were actually stepped);
under `policy/` - `surrogate_loss`, `kl`, `clip_fraction`, and `total_loss`
(`surrogate_loss + kl_coefficient * kl`, also reported as the train loss);
plus `optimizer/learning_rate`. With a judge, its own series are exported under
`judge/` - request count, latency, cache hit rate, dropped and degenerate group
fractions, score mean and std - under the names the agentic loop uses, so one
dashboard reads both, and `timing/judge_seconds` says what the verdicts cost
next to `timing/reward_seconds`. They flow to the TensorBoard / W&B sinks like SFT
and PPO metrics. The CLI prints one line per GRPO optimizer epoch, labelled
with both `update=i/N` and `grpo_epoch=j/M`, including the trained
completion-token throughput of the epoch; GRPO has no held-out validation
loss.

## The objective, and what is shared with PPO

The differentiable machinery shares PPO's detached-coefficient weighted
cross-entropy reduction, row packing (prompt and padding masked, completion
token `c` supervised at position `n_prompt + c − 1`), and clip quadrants. It
lives in the `crates/retrograd-training/src/rollout/` implementation, used by
both `ppo.rs` and `grpo.rs`; what GRPO adds there is its
own fixed-reference k3 coefficient (`grpo_token_weights`). Everything
documented in [PPO.md](PPO.md) under "The differentiable objective" applies
unchanged, including:

- the base model is fingerprint-verified frozen on every step;
- a fully clipped (or zero-advantage) batch is a legitimate zero-gradient
  no-op, not an error;
- `training.ctx` is the *trained window*: prompt + `max_new_tokens` must fit
  in it even when llama.cpp rounds the runtime context up. GRPO checks this
  before sampling each prompt, since the constant loss denominator assumes
  every completion could have used its full budget.

The runtime normally averages only non-zero weighted labels inside each
physical ubatch. Before training, Retrograd rescales those coefficients per
ubatch so the resulting gradient is divided by a caller-chosen constant,
including clipped zero-gradient tokens. PPO divides by each completion's own
length (the `1/|o_i|` reduction of the GRPO paper); Dr. GRPO instead divides
every completion by the configured generation budget `max_new_tokens`. The
constant denominator removes the length bias of `1/|o_i|`, under which a
negative-advantage completion could dilute its per-token penalty by running
long.

Compared with PPO, GRPO changes the advantage, the loss reduction, and the KL
anchor:

| | PPO | Dr. GRPO |
|---|---|---|
| Baseline | linear-probe value head + GAE (or batch whitening) | group mean reward (live members) |
| Advantage | per-token | per-sequence, constant over tokens |
| Loss reduction | `1 / completion_len` | `1 / max_new_tokens`, constant |
| Surrogate clip | symmetric `clip_range` | Clip-Higher: `[1 − low, 1 + high]` |
| KL anchor | rollout policy | fixed base model (LoRA disabled) |
| Extra passes | hidden-state extraction + head regression | fixed-reference scoring (live rollouts) |
| Sampling | free `temperature` / `top_p` | enforced `temperature = top_p = 1` |
| Config | `rollout_batch_size`, `[ppo.critic]` | `prompts_per_update`, `group_size` |

### KL anchor

The GRPO paper regularizes toward a fixed initial policy. Retrograd obtains it
by temporarily disabling the LoRA during a teacher-forced scoring pass. Since
the base tensors are fingerprint-verified frozen, this reference cannot drift
as training proceeds. The clipped surrogate still uses the rollout policy as
`π_old`; only the KL anchor is fixed.

With rule-based or otherwise verifiable rewards, `kl_coefficient = 0.0` is a
good starting point (DAPO and the Dr. GRPO paper both train without a KL
term): the reward cannot be hacked the way a learned reward model can, and
the anchor mostly slows learning. Keep a positive coefficient when the reward
is learned or exploitable.

## Choosing the group

The group baseline lives or dies on reward spread *within* a group:

- `group_size` is the variance knob. 4 is a floor for a usable baseline
  (a 2-member group centers its two rewards to `±(r_1 − r_2) / 2`, so the
  whole signal rides on one pairwise comparison); 8–16 is typical. Larger
  groups cost linearly more rollouts per update.
- **Watch `reward/group_std`, `batch/zero_std_group_fraction`, and
  `batch/trained_fraction`.** If the std trends to zero (equivalently, the
  zero-std fraction trends to one and the trained fraction to zero), groups
  have collapsed to identical rewards - because the policy has converged, or
  because the reward is too coarse (e.g. a binary reward the model always
  passes or always fails). Sampling temperature is pinned at 1, so it is not
  a knob here. Zero-signal groups are skipped, so a collapsing batch trains
  on ever fewer rollouts rather than taking KL-only steps.
- An update where *nothing* is trainable prints a warning naming how many it
  has been in a row. It is not by itself an error: a model that has not yet
  earned any reward scores every completion identically, which is a normal
  early state. Only `max_stalled_updates` consecutive ones stop the run
  (default 25); `max_stalled_updates = 0` never stops and lets the schedule
  run out.
- Reward *scale* now matters: without the std normalization, advantages are
  in reward units, so the effective step size is `reward scale × lr`. Keep
  rewards in a sensible fixed range (a binary 0/1 reward works well *as long
  as groups are mixed* - the group baseline then reduces to pass-rate
  centering) rather than letting their magnitude drift. Ranking alone is no
  longer the whole story: within a group, larger margins get proportionally
  larger gradients.
- `prompts_per_update` trades gradient variance against prompt coverage per
  update; unlike PPO's batch whitening, prompts never share a baseline, so
  even `prompts_per_update = 1` is statistically sound.

## Possible improvements

Roughly in order of value for the effort:

1. **Leave-one-out baseline (RLOO).** Using the mean of the *other*
   `group_size − 1` rewards as each member's baseline removes the small bias
   the member introduces into its own baseline; a one-line change to
   `group_advantages`, most valuable for small groups. (With the group-mean
   baseline this is a uniform `(G − 1)/G` rescale of the advantage, so the
   gradient direction is already identical.)
2. **Duplicate-completion dedup.** Identical completions within a group
   (likely on short generation budgets or near-deterministic prompts) could
   be collapsed with a multiplicity weight instead of being scored and
   stepped separately.
3. **Dynamic sampling (full DAPO).** *Done* - `dynamic_sampling =
   { max_resample_factor = N }`. Instead of merely skipping zero-signal
   groups, the batch-assembly loop resamples replacement prompts (continuing
   the round-robin) until the update carries `prompts_per_update` informative
   groups, or `max_resample_factor * prompts_per_update` candidates are
   exhausted; a shortfall is padded with dead groups so the scheduler horizon
   stays fixed. Keeps the effective batch size constant as the policy
   converges (`batch/groups_sampled_fraction` reports the resampling
   pressure).

## Validation map

GRPO reuses the runtime building blocks PPO already pins (see
[PPO.md](PPO.md)'s validation map for generation, scoring, and the weighted
step). What GRPO adds is tested on top:

| Layer | Test | What it pins |
|---|---|---|
| Group baseline | unit tests in `crates/retrograd-training/src/grpo.rs` | per-group centering without std normalization, zero-signal group filtering, truncation-masked members excluded from the baseline, groups without two live members dropped, mean within-group std and zero-std fraction, zero group means, rejection of non-representable centered rewards, deterministic seeded epoch shuffle |
| Step math | unit tests in `crates/retrograd-training/src/rollout/` | clip quadrants, decoupled Clip-Higher ranges, fixed-reference KL sign, row packing, constant-budget normalization, exact match against the numeric derivative of the objective |
| Fixed reference | `tests/ppo_runtime.rs::fixed_reference_ignores_lora_updates_and_restores_the_policy` | LoRA is disabled only for reference scoring, the base scores do not drift, and the trained policy is restored |
| Full loop | `tests/grpo_runtime.rs::grpo_runs_end_to_end` | prompts → groups of rollouts → external rewards → group-relative updates, metrics finite, base frozen |
| Dynamic sampling | `group_signal_*` unit tests in `crates/retrograd-training/src/grpo.rs`, e2e `tests/grpo_runtime.rs::dynamic_sampling_resamples_zero_signal_groups` | the signal test mirrors the `group_advantages` filter (spread, truncation mask, penalized rewards); zero-signal groups are resampled up to the cap, then padded - one live group out of two, `groups_sampled_fraction = 2` |
| Config | `[grpo]` validation in `crates/retrograd-config/src/lib.rs` | group_size >= 2, clip ranges in (0, 1) with high >= low, finite non-negative KL coefficient, max_new_tokens > 0, on-policy sampling (`temperature = top_p = 1`), `dynamic_sampling.max_resample_factor >= 2` |

Model-dependent tests skip when no GGUF is present (`RETRO_TEST_MODEL`
overrides the default path).
