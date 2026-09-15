# PPO training

Retrograd trains a LoRA adapter with PPO end to end through the llama.cpp
fork: rollouts are sampled from the current policy, scored by an external
reward command, and updated with an exact clipped-surrogate gradient - no
Python, no second framework.

## Running it

A complete document is in the [PPO guide](/training/ppo#configuration), and every
key with its default is in the
[configuration reference](/reference/configuration#ppo-sections). This page
describes how the loop is implemented. `gradient_accumulation` is best left out
of a PPO document: a rollout algorithm pins it to `ctx / micro_batch`, so one
optimizer step never spans less than a rollout (see [GRPO](./GRPO), "The two
geometry knobs").

The prompts file uses the same chat JSONL records as SFT. A prompt may include
a `system` message and earlier user/assistant turns, but it must end in a
non-empty `user` message. Retrograd formats the whole conversation with the
GGUF chat template and adds the assistant-generation prefix before sampling.

## PPO counters: `epochs`, `updates`, and `ppo_epochs`

These names apply to different loops and should not be used interchangeably:

| Setting | Applies to | Meaning |
| --- | --- | --- |
| `training.epochs` | SFT only | Complete passes over the fixed supervised dataset. It does **not** control PPO and is ignored by the PPO loop. |
| `ppo.updates` | PPO | Number of fresh rollout batches: sample completions, obtain rewards, and compute advantages. |
| `ppo.ppo_epochs` | PPO | Number of optimizer passes over the *same* rollout batch before sampling a new one. |
| `ppo.critic.value_epochs` | PPO critic only | Full-batch Adam passes used to fit the value head for one PPO update. It does not update the policy LoRA. |

So `updates = 5` and `ppo_epochs = 1` means five newly sampled/rewarded
batches and five policy optimizer epochs in total. With `ppo_epochs = 3`, the
same configuration still samples five batches but makes fifteen policy passes.
Pinning the optimizer window to `ctx` keeps a rollout from being split across
several optimizer steps, but accumulation is carried across row boundaries: a
pass groups consecutive rollouts into one step while their combined
micro-batches fit that window, so each pass performs *at most* one optimizer
step per rollout. The exact `global_step` depends on the sampled completion
lengths as well as on `rollout_batch_size` and the trained window.

### Reward protocol

The reward command is executed directly (never through a shell). It receives
one JSON object per rollout on stdin:

```json
{"prompt": "...", "completion": "..."}
```

`prompt` is the content of the conversation's final user message, not the
model-specific formatted chat string. This keeps reward programs independent
of the selected GGUF template.

and must print one `{"reward": <finite number>}` line per input, in order.
A non-zero exit, malformed line, or count mismatch aborts the run.

#### One worker, not one process per batch

`reward_mode` decides the lifetime of the process speaking that protocol, and
`reward_timeout_seconds` bounds one exchange with it:

```toml
[ppo]                                # identical under [grpo]
reward_command = ["python3", "reward.py"]
reward_mode = "persistent"           # the default; "oneshot" for the old shape
reward_timeout_seconds = 300         # deadline of one batch
```

The batches are many and the startup is identical every time: PPO scores one
per update, GRPO one per *sampling wave* - up to `max_resample_factor` per
update with dynamic sampling - plus one per held-out evaluation pass. A
`updates = 300` GRPO run therefore spawns several hundred reward processes in
`"oneshot"`, each re-importing the same modules and reloading the same
reference data. `"persistent"` spawns one and keeps it.

What a persistent command must do, and it is the whole difference: answer a
version handshake at startup, then **flush every response line**. Nothing
closes its stdin between batches, so an answer sitting in a buffer is an answer
the trainer never receives.

```
trainer  → command : {"protocol":"retrograd-reward/1"}
command  → trainer : {"protocol":"retrograd-reward/1"}     (flushed)
then, per batch:
trainer  → command : N request lines carrying _retrograd_batch/_retrograd_index
command  → trainer : N response lines echoing those fields (flushed)
trainer  → command : {"_retrograd_batch_end": B}
command  → trainer : {"_retrograd_batch_end": B}           (flushed)
```

```python
import json, sys

PROTOCOL = "retrograd-reward/1"
handshake = json.loads(sys.stdin.readline())
if handshake.get("protocol") != PROTOCOL:
    raise SystemExit(f"unsupported reward protocol: {handshake}")
print(json.dumps({"protocol": PROTOCOL}), flush=True)

references = load_references()          # paid once, not once per batch

for line in sys.stdin:
    request = json.loads(line)
    if "_retrograd_batch_end" in request:
        print(json.dumps(request), flush=True)
        continue
    batch = request.pop("_retrograd_batch")
    index = request.pop("_retrograd_index")
    print(json.dumps({
        "reward": score(request, references),
        "_retrograd_batch": batch,
        "_retrograd_index": index,
    }), flush=True)
```

The two reserved fields correlate every response with the request that caused
it; the echoed end marker closes the batch. Together they make an extra or late
line a protocol error instead of allowing it to become the first reward of the
next batch. A persistent stdout line is limited to 8 MiB.

Set `reward_mode = "oneshot"` for a command that cannot do that - one that
reads its stdin to the end before answering, or that must start from a clean
state every batch. A one-shot command asked to be persistent fails on the
handshake with that sentence in the message, rather than hanging until the
deadline.

Three consequences of a worker that outlives a batch:

- **the first batch pays the startup.** `reward_timeout_seconds` covers the
  handshake and the batch together on that first call, so a command loading a
  model needs a timeout that covers loading it;
- **a failed batch kills the worker.** Nothing can resynchronize a stream whose
  reader gave up, so the process is closed and the next call starts a fresh
  one. The failed batch itself is *not* retried - it aborts the run, exactly as
  a failing one-shot command does;
- **the reward should stay a pure function.** State kept between batches is
  state two runs of the same configuration do not share.

At the end of the run the worker's stdin is closed - its cue to flush a report
or close a file - and its process group is killed shortly after if it has not
exited.

## How an update works

For each of `updates` iterations:

1. **Rollout.** `rollout_batch_size` prompts are taken round-robin from the
   prompts file. Each is tokenized and a completion is sampled
   (temperature/top-p, deterministic per seed) up to `max_new_tokens`, an
   end-of-generation token, or the trained window. The rollout-time policy
   logprobs (`old_logprobs`) are captured by re-scoring the full sequence
   through the same teacher-forced path used later. All ratios are exactly 1
   before the first optimizer step; subsequent size-one SGD steps legitimately
   move the policy before later rollouts.
2. **Reward.** All (prompt, completion) pairs go to `reward_command`.
3. **Advantage.** With the critic enabled (default), per-token advantages come
   from GAE over the value head's predictions, then are whitened across all
   completion tokens of the batch; the head is fitted toward this batch's
   TD(lambda) returns afterwards (see "The critic" below). With the critic
   disabled, the advantage is the whitened per-sequence reward, constant over
   the tokens of a rollout.
4. **PPO epochs.** For each epoch and each rollout, the sequence is re-scored
   under the *current* policy, per-token weights are derived (below), and one
   weighted optimizer step runs through the runtime. Because weights are
   recomputed from a fresh forward before every step, each step is an exact
   PPO step (no stale ratios).

Progress events carry `reward/mean`, `policy/surrogate_loss`, `policy/kl`,
`policy/clip_fraction`, `optimizer/learning_rate`, and - when the critic is
enabled - `policy/value_loss` and `critic/feature_mib`; they flow to the
TensorBoard / W&B sinks like SFT metrics.
The CLI prints one line per PPO optimizer epoch, labelled with both
`update=i/N` and `ppo_epoch=j/M`. When `[evaluation]` is configured, the held-out
prompt dataset is generated and scored after the selected updates; the sinks
receive `eval/mean_reward`, `eval/reward_min`, and `eval/reward_max`.
`evaluation.max_examples` caps one evaluation at that many prompts, taken
evenly spaced across the held-out set, since each evaluated prompt costs a
full generation.

## The differentiable objective

The runtime does not build a bespoke PPO graph. The clipped surrogate with
*detached* per-token coefficients reduces exactly to a weighted cross-entropy:
for sampled token `a_t`, the gradient of `w_t · (−log π(a_t))` w.r.t. the
logits is `w_t · (softmax − one_hot)`, which equals the PPO policy gradient
when

```
w_t = A · r_t · 1[not clipped]  +  k · (1 − r_t)
r_t = exp(logπ(a_t) − logπ_old(a_t))
1[not clipped] = !(A > 0 ∧ r > 1+ε) ∧ !(A < 0 ∧ r < 1−ε)
```

The `k · (1 − r_t)` term is the exact gradient of Schulman's k3 estimator of
`KL(π_old ‖ π)`, scaled by `kl_coefficient`: it pulls the policy back toward
the rollout policy symmetrically (negative when `r > 1`, positive when
`r < 1`, zero at `r = 1`).

The weights ride through `llama_opt_epoch_weighted()` (fork API): each label's
one-hot value is scaled by its weight, and the fork's generalized
cross-entropy backward `(Σlabels·softmax − labels) · d / n_active` makes the
gradient exact for non-unit label mass. Zero weight masks a position exactly
like a `-1` label; prompt and padding tokens are always masked. Before entering
the runtime, coefficients are rescaled per physical ubatch so its
active-position mean is algebraically the full completion-token mean,
including clipped zero-gradient positions.

Consequences worth knowing:

- The base model is fingerprint-verified frozen on every step; only LoRA
  tensors may change.
- A fully clipped batch has a genuinely zero gradient. Unlike SFT, an
  unchanged LoRA after a weighted step is not an error.
- `training.ctx` is the *trained window*. llama.cpp may round the runtime
  context up (e.g. 64 → 256), but only the first `training.ctx` positions of
  a packed row receive gradient, so prompt + completion must fit in
  `training.ctx`.

## The critic

PPO's advantage is `A_t = Q_t − V(s_t)`. The baseline `V` only reduces the
variance of the policy gradient - a poor critic makes PPO noisier, never
wrong - which is why a deliberately modest critic is a sound design.

Retrograd's critic is a **linear probe**: a single linear layer
`V(s) = w · h(s) + b` over the model's *frozen* final-layer hidden states,
implemented in pure Rust
(the `crates/retrograd-training/src/value.rs` implementation).
The runtime exposes the features through `retro_trainer_hidden_states()`
(teacher-forced forward, one hidden-state row per position); no ggml graph or
fork change is involved, and the head's closed-form regression makes it
exactly testable.

Per update:

1. For every rollout, the hidden state of each *pre-token* state is extracted
   (the state before emitting completion token `c` is the prefix ending at
   sequence index `n_prompt + c − 1`). Features are captured once per update;
   they move slowly as the LoRA trains.
2. The head predicts `V(s_t)` and **GAE** turns the terminal reward into
   per-token advantages: `δ_t = r_t + γ·V(s_{t+1}) − V(s_t)` (terminal value
   0, only the last step carries the reward), `A_t = Σ (γλ)^k δ_{t+k}`.
3. Advantages are whitened across all completion tokens of the batch.
4. The head is fitted (full-batch Adam, MSE) toward the TD(lambda) returns
   `A_t + V(s_t)` - *after* the advantages are taken from its pre-fit
   predictions, the standard PPO ordering. `policy/value_loss` reports the
   post-fit regression error; it should trend down across updates.

The head is zero-initialized, so on the very first update (and whenever
`gamma = gae_lambda = 1`) the advantages reduce exactly to the whitened
returns - the same signal as the critic-less path. What the critic adds as it
learns is **per-token credit**: positions where the predicted value climbs
absorb the reward locally instead of every token sharing one scalar.

### The critic's memory, and `feature_dtype`

The features of step 1 are kept for the whole update - predictions, GAE and every
`value_epochs` pass read the same matrix - so the critic holds
`total_completion_states × hidden_dim` floats in **host** RAM. It is the largest
host allocation of a PPO run, it grows with the rollout batch and the completion
length, and `critic/feature_mib` publishes it on every update.

`feature_dtype` decides what that matrix is stored in: `f32` (the default, the
hidden states exactly as the runtime produced them), `f16` or `bf16`, both of
which halve it. The fit itself stays in F32 whatever this says - rows are
converted back one chunk at a time - so nothing about the arithmetic changes;
what changes is that the value head regresses on rounded features. Between the
two 16-bit forms, `f16` is the more precise while the residual stream stays
inside its range, and `bf16` cannot leave the range at all, which matters on a
larger model than a fixture.

Reach for it when the host figure is a real constraint, not by default: the
rounding is small and it only affects the baseline (which reduces the variance of
the policy gradient and never biases the update), but small is not zero. The
advantages of an update are unaffected either way - they are taken from the
unrounded rows, before anything is stored.

Practical implications:

- `rollout_batch_size = 1` still degenerates without a critic (the batch mean
  baseline vanishes). With the critic it remains usable, but 4–8 rollouts per
  update stabilize both the whitening and the head's regression.
- Reward scale is neutralized by the whitening; only the ranking within a
  batch matters for the surrogate. The value head, however, regresses raw
  returns - rewards in a sane range (say, [−10, 10]) keep `value_lr`
  defaults reasonable.
- If `policy/value_loss` diverges (error mentioning `ppo.value_lr`), lower the
  learning rate or reduce `value_epochs`.

### Possible improvements

Roughly in order of value for the effort:

1. **Reward-to-go normalization for the head.** Whitening return targets
   (running mean/std) before the regression would decouple `value_lr` from
   the reward scale entirely.
2. **RLOO / group baselines.** Sampling `k` completions per prompt and using
   the leave-one-out mean as a per-prompt baseline composes with the critic
   (it debiases prompt difficulty, which a state-value probe also captures
   but more slowly). GRPO ([GRPO.md](GRPO.md)) implements the pure
   group-baseline variant; RLOO would combine it with this critic.
3. **A deeper probe.** The linear head can become a small MLP (one hidden
   layer) in pure Rust at negligible cost; worth it only if `policy/value_loss`
   plateaus high while rewards are clearly predictable from the text.
4. **Per-token rewards.** The reward protocol currently returns one scalar
   per completion. Accepting an optional per-token vector (e.g. from a
   process reward model) would slot directly into the GAE `r_t` terms.
5. **A critic trained through the network.** A value head whose gradients
   flow into the transformer (or a dedicated "value LoRA") is the classic
   PPO architecture and the strongest baseline, but it requires adding
   trainable tensors and an MSE loss to the fork's optimizer graph - a
   chunk of work comparable to the weighted-loss objective itself, with the
   added complexity of two losses sharing one graph. Do this only if the
   probe demonstrably limits training quality.
6. **Feature caching.** Hidden states are recomputed once per update; if
   `value_epochs` grows or rollout batches get large, features could be
   cached and the head refit several times between updates (e.g. minibatch
   Adam) without touching the runtime.

## Validation map

Each building block is tested in isolation:

| Layer | Test | What it pins |
|---|---|---|
| ggml op (fork) | `tests/weighted_ce.rs` | weighted CE forward/backward vs analytic reference (one-hot, weighted, masked, negative rows); Metal vs CPU parity |
| Rollout | `tests/ppo_runtime.rs::generation_is_seed_deterministic_and_bounded` | seeded determinism, bounds, finite logprobs |
| Scoring | `tests/ppo_runtime.rs::scoring_matches_generation_logprobs` | deterministic teacher-forced path, agreement with rollout logprobs |
| Features | `tests/ppo_runtime.rs::hidden_states_have_the_declared_shape_and_are_deterministic` | shape, determinism, no state leaking into the scoring path |
| Value head | unit tests in `crates/retrograd-training/src/value.rs` | recovers a known linear function, loss decreases, shape/lr validation |
| GAE | unit tests in `crates/retrograd-training/src/ppo.rs` | hand-computed episode, zero-value reduction to plain returns, perfect-critic zero advantage |
| Weighted step | `tests/ppo_runtime.rs::unit_weights_reproduce_the_sft_loss` | weights = 1 is bit-for-bit the SFT objective |
| Differentiability | `tests/ppo_runtime.rs::weight_sign_steers_the_policy` | positive/negative weights move logprobs up/down, base frozen |
| Degenerate batch | `tests/ppo_runtime.rs::zero_weights_are_a_legitimate_noop_step` | all-clipped batches are a no-op, not an error |
| Full loop | `tests/ppo_runtime.rs::ppo_runs_end_to_end` | prompts → rollouts → external rewards → updates |
| Loop math | unit tests in `crates/retrograd-training/src/ppo.rs` and `crates/retrograd-training/src/rollout/` | advantage whitening and GAE (ppo.rs); clip quadrants, KL sign, row packing (rollout.rs, shared with GRPO) |

Model-dependent tests skip when no GGUF is present (`RETRO_TEST_MODEL`
overrides the default path).
