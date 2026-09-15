# Distillation

Distillation trains a small student to reproduce a larger teacher's policy.
Retrograd offers two ways to do it, selected by `distill.mode`:

| `mode` | What the teacher provides | When |
| --- | --- | --- |
| `on_policy` (default) | log-probabilities of the tokens the student just sampled | the student's own mistakes are what you want corrected |
| `topk_offline` | a truncated distribution over a fixed corpus, precomputed once | the corpus is fixed, or the teacher is too large to keep resident during training |

They compose rather than compete: an offline pass is cheap per token because it
generates nothing, an on-policy pass covers the states the student actually
reaches. Doing the first and then the second is a reasonable schedule; this page
describes on-policy first, and the offline path in
[its own section](#offline-top-k-distillation).

## On-policy distillation

The student samples, the teacher scores the very same tokens, and the gap between
their log-probabilities becomes the learning signal.

It is the choice when you have a model that already behaves the way you want but
is too large to serve, and no reward command that could rank two answers. Where
GRPO needs a way to *score* a completion, distillation needs only a second model
and a set of prompts.

## What the objective is

The student `pi_S` samples `y` from a prompt. The teacher `p_T` is asked what it
would have assigned to those same tokens. For each trained token:

```text
A_t = clamp( log p_T(y_t | y_<t) - log pi_S(y_t | y_<t), ±weight_clip )
```

`A_t` is detached and becomes a per-token advantage in the same weighted
objective GRPO uses - no new kernel, no second loss. A token the teacher liked
more than the student did gets a positive weight and is reinforced; one the
student was overconfident about gets a negative weight and is pushed down.

Averaged over sampled tokens, `-A_t` estimates `KL(pi_S ‖ p_T)` per token, and
that is the quantity a run reports and minimizes. It is measured in nats, so a
distillation run - unlike a GRPO one - has an objective whose value means the
same thing on any two runs.

Two consequences worth knowing before you start:

- **The teacher must share the student's tokenizer.** Scoring the student's ids
  under a different vocabulary returns the log-probabilities of *other tokens*:
  the numbers are finite, the run trains, and the objective is nonsense. This is
  checked at the start of every run, on the vocabulary size and on three witness
  sentences, and an unmatched pair is refused before a single token is scored.
  A teacher that is instruct-tuned and a student that is a base checkpoint are
  fine - only the vocabulary has to match, not the chat template.
- **Both models are resident for the whole run.** The teacher never gets an
  adapter, so it has no backward graph, no gradients and no optimizer moments:
  it costs its weights plus its KV cache and nothing else. That is still a whole
  model beside a student that *does* carry AdamW state, and it is the constraint
  that decides which teacher you can use. Quantize the teacher harder than the
  student.

## Configuration

```toml
[run]
algorithm = "distill"

[model]
path = "student.gguf"
device = "auto"

[lora]
output = "distill-adapter.gguf"
rank = 8
alpha = 16.0
seed = 42

[training]
ctx = 1024
micro_batch = 32
generation_concurrency = 8
lr = 0.00001
max_grad_norm = 1.0

[distill]
teacher_path = "teacher-q4_k_m.gguf"
prompts = "prompts.jsonl"
updates = 200
prompts_per_update = 8
samples_per_prompt = 1
distill_epochs = 1
weight_clip = 5.0
kl_coefficient = 0.0
mask_truncated = true

[distill.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 512
seed = 42
```

Run it with:

```bash
retrograd train distill.toml
```

The prompt file is the same prompt-only chat JSONL GRPO reads: one conversation
per line, ending on a `user` turn, with no reference answer and no reward.

```json
{"messages":[{"role":"user","content":"..."}]}
```

The only preparation that matters is **coverage**. The student will match the
teacher where the prompts take it and nowhere else. A few thousand varied
prompts beat a hundred thousand of one shape.

## Update sequence

One update is:

1. Draw `prompts_per_update` prompts, in file order or shuffled.
2. Sample `samples_per_prompt` completions from the student for each, at
   temperature 1.
3. Re-score those completions under the student, teacher-forced. These are the
   behaviour log-probabilities the advantage subtracts.
4. Score the same tokens with the teacher, one shared-prefix pass per group.
5. Turn each pair into `A_t`, clamped.
6. Take `distill_epochs` weighted optimizer passes over the batch.

Steps 3 and 4 go through the *same* batched call on both sides, deliberately:
`llama_decode` is not invariant by batch, so scoring one model in a group and
the other alone would fold a difference between two tile arrangements into the
difference between two models.

`samples_per_prompt = 1` is admissible and is the default, unlike GRPO's
`group_size`, which must be at least 2. GRPO needs a group to build a baseline;
here the signal is already per-token, so one sample carries it. Raising it buys
prefix reuse across a group, not a baseline.

`distill_epochs = 1` is strictly on-policy: the policy ratio is exactly 1 and
the token weight *is* the advantage. Above 1, the clipped surrogate of
`clip_range_low` / `clip_range_high` applies, exactly as in GRPO.

## Why `weight_clip` is not cosmetic

On a token the student was confident about and the teacher was not, the gap runs
to -20 nats. Gradient-norm clipping then rescales every other token to nearly
nothing, and one token owns the update. `weight_clip` bounds that before it
happens.

The series to watch is `distill/advantage_clipped_fraction`. If it does not fall
as the run progresses, either the bound is too tight or the student is
diverging - and until it falls, `distill/teacher_kl_mean` is reporting a clipped
quantity rather than the divergence.

## What a run reports

| Series | Meaning |
| --- | --- |
| `distill/teacher_kl_mean` | Mean `-A_t` over trained tokens, in nats. The objective. |
| `distill/teacher_kl_p95` | The tail of the same quantity. A mean that falls while this does not means the easy tokens aligned and the disagreements did not. |
| `distill/advantage_clipped_fraction` | Share of tokens that hit `weight_clip`. |
| `policy/entropy` | `-mean(log pi)`; the entropy-collapse symptom. |
| `completions/truncation_fraction` | Share of completions cut at `max_new_tokens`. |
| `timing/teacher_scoring_seconds` | The one phase this loop has and GRPO does not. |
| `eval/mean_neg_teacher_kl` | Scheduled evaluation: the held-out divergence, negated so that larger is better. |

The evaluation series is negated because everything downstream - the best
checkpoint rule, early stopping - reads a scalar that is larger when the run is
better, and a divergence is not.

## Held-out evaluation

An `[evaluation]` section works as it does for the other rollout algorithms:
prompts are generated at fixed seeds and the student's divergence from the
teacher is reported. Two differences from the training loop, both deliberate:

- The advantage clamp is **not** applied. A measurement that inherited it would
  report the bound rather than the divergence, and would stop moving exactly
  when the run got worse.
- Truncation masking is **not** applied. A held-out set is not the run's to
  filter, and dropping truncated completions would make two evaluations under
  different budgets incomparable.

An evaluation loads no second teacher: it uses the one the run already holds.

## Measuring against a baseline

`bench` reports three figures for a `distill` document:

```bash
retrograd bench distill.toml --data held-out.jsonl --adapter distill-adapter.gguf
```

- **KL to the teacher**, per token, with its distribution across prompts.
- **Top-1 agreement**: the share of positions where the student and the teacher
  would greedily emit the same token. This is exact - it reads both models'
  truncated distributions rather than one sample - and it is the figure a
  divergence cannot supply. Two models can assign near-identical probability to
  the tokens that were drawn while disagreeing about every token they would
  actually emit.
- **Teacher-forced perplexity** of the reference answers, when the held-out
  lines carry a final `assistant` turn. This is the number that makes a
  distillation bench comparable to an SFT bench on the same corpus, which is the
  comparison worth making: on-policy distillation is only interesting if it
  beats supervised fine-tuning on the teacher's own outputs.

Run `bench` once without `--adapter` before training to get the baseline column.

## Distillation parameters

| Key | Default | Notes |
| --- | ---: | --- |
| `teacher_path` | required | Checked for existence and tokenizer agreement when the run opens, not when the document is parsed. |
| `prompts` | required | Prompt-only chat JSONL. |
| `updates` | required | |
| `prompts_per_update` | required | |
| `samples_per_prompt` | `1` | `1..256`; must not exceed the trainer's sequence capacity. |
| `distill_epochs` | `1` | |
| `clip_range_low` | `0.2` | Read only when `distill_epochs > 1`. |
| `clip_range_high` | `0.28` | |
| `weight_clip` | `5.0` | Finite, above zero, in nats. |
| `kl_coefficient` | `0.0` | Zero skips the base-model reference pass entirely - the teacher is the anchor. |
| `mask_truncated` | `true` | |
| `prompt_order` | `sequential` | `sequential` or `shuffled`. |

The full schema is in the
[configuration reference](../reference/configuration.md#distill).

## Planning a distillation run

The planner's memory estimate carries the teacher only when it was given the
teacher's geometry. When it was not, it emits
`teacher_absent_from_the_memory_budget` and the device figure it reports is a
lower bound - short by a whole model. A plan that fits by a margin thinner than
the teacher is not a plan.

## Offline top-k distillation

`mode = "topk_offline"` replaces the sampler and the per-token advantage by a
distribution the teacher wrote down once. For each position of a fixed corpus
the teacher's top `k` tokens and their probabilities are stored in a *sidecar*
next to the JSONL, and training minimizes

```text
loss_t = - sum_j p_j(t) * log softmax(z_t)[v_j(t)]
```

against them. There is no estimator here and no bias: this is the exact gradient
of the cross-entropy against the teacher's truncated distribution, computed by
the same fused operator the one-hot objective uses, with `k` targets per position
instead of one.

What it buys, and what it costs:

- **No generation.** An epoch is a supervised pass over the corpus, so it costs
  what SFT costs; the teacher's share of the work was paid once, offline.
- **No second model resident.** The training run never opens the teacher - the
  sidecar is what it left behind - so the memory constraint that decides which
  teacher an on-policy run can use does not apply. A teacher too large to sit
  beside the student is usable here.
- **Fixed states.** The student is fitted to the teacher's opinion about the
  corpus, not about the sequences the student itself produces. That is the
  difference between the two modes, and it is why the on-policy one exists.
- **Truncation.** The mass the teacher put outside its top `k` is renormalized
  away, so the target is the conditional distribution "given the teacher's top
  `k`". `k = 16` is the default; the sidecar records its own.

### Producing the sidecar

```bash
retrograd distill-teacher run.toml
```

It reads `[distill].teacher_path`, `[distill].data` and `[distill].sidecar` from
the document, tokenizes the corpus through the same path the trainer will use,
and writes the file. `--k`, `--data` and `--out` override the defaults.

The corpus goes through the repository's own dataset preparation on purpose. A
producer that retokenized the corpus with its own idea of the chat template
would misalign the sidecar by a token or two, and that failure is *silent*: the
run trains, the loss falls, and the student is fitted to the teacher's opinion
about the wrong positions. Here the alignment is a property of the code path.
The sidecar's header carries the corpus's fingerprint and the tokenizer's, and
a run whose pair does not match is refused before its first step.

The command loads one model at a time - the student to prepare the corpus, then
the teacher to score it - so it runs on a machine that could not hold both.

### Configuration

```toml
[run]
algorithm = "distill"

[model]
path = "student.gguf"

[lora]
output = "kd-adapter.gguf"
rank = 8
alpha = 16.0

[training]
ctx = 2048
micro_batch = 512
epochs = 1          # unused by this mode; `offline_epochs` is what it reads

[distill]
mode = "topk_offline"
teacher_path = "teacher-q4_k_m.gguf"
data = "corpus.jsonl"
sidecar = "corpus.topk"
offline_epochs = 3
```

Neither `prompts`, `updates`, `prompts_per_update` nor `[distill.sampling]` is
read in this mode, and none of them is required. The reverse also holds: `data`,
`sidecar` and `offline_epochs` next to `mode = "on_policy"` are refused rather
than ignored, because a document that names a sidecar expects it to be read.

| Key | Default | Notes |
| --- | ---: | --- |
| `mode` | `on_policy` | `on_policy` or `topk_offline`. |
| `data` | required | Chat JSONL, the same shape an SFT run reads. |
| `sidecar` | required | The `.topk` file `distill-teacher` wrote for `data`. |
| `offline_epochs` | `1` | Passes over the corpus. The epoch is also the resume unit. |
| `teacher_path` | required | Read by `distill-teacher`; the training run itself never opens it. |
| `weight_clip` | `5.0` | Validated but unused by this mode. |

### What a run reports

An offline run publishes the SFT series - `train/loss`, `train/epoch_loss`,
learning rate, throughput - plus `distill/topk_entries`, the `k` its sidecar
declared. It publishes no `distill/teacher_kl_*`: those come from scoring the
student against a live teacher, which this path deliberately does not hold.
`bench` on the same document is what reads the divergence, and it opens the
teacher to do so.

There is no scheduled evaluation on this path. An SFT evaluation measures the
loss against one-hot targets, which is not the quantity being optimized here,
and reporting it as `eval_loss` would label the wrong number.

## Python

```python
from retrograd import DistillConfig, Trainer

metrics = trainer.fit_distill(
    DistillConfig(
        teacher_path="teacher-q4_k_m.gguf",
        prompts="prompts.jsonl",
        updates=200,
        prompts_per_update=8,
    )
)
```

`TrainingConfig.max_sequences` must be at least `samples_per_prompt`: the
teacher scores a group as that many branched sequences.

`fit_distill` is the on-policy binding. The offline mode reads two files and
runs no sampler, so it is driven through a `[distill] mode = "topk_offline"`
document and `retrograd distill-teacher`, not through a call that hands over a
prompt list and a sampling seed.
