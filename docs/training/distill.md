# Distillation

Distillation trains a small *student* model to behave like a larger *teacher*.
It needs no reward and no reference answers, only the teacher and a set of
prompts or texts. Two modes are available through `distill.mode`:

| Mode | How it works | Choose it when |
| --- | --- | --- |
| `on_policy` (default) | The student generates, the teacher scores the student's tokens. | You want the student corrected on its own mistakes. |
| `topk_offline` | The teacher's top-k predictions over a fixed corpus are computed once, then trained on like SFT. | The teacher is too large to keep in memory next to the student, or generation is too slow. |

The two combine well: an offline pass first (cheap, no generation), then an
on-policy pass.

**The teacher must use the same tokenizer as the student.** This is checked
before training starts. The chat template may differ.

## On-policy

```toml
[run]
algorithm = "distill"

[model]
path = "student.gguf"

[output]
path = "distill-adapter.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 1024
lr = 0.00001

[distill]
teacher_path = "teacher-q4_k_m.gguf"
prompts = "prompts.jsonl"
updates = 200
prompts_per_update = 8
mask_truncated = true

[distill.sampling]
temperature = 1.0
top_p = 1.0
max_new_tokens = 512
seed = 42
```

```bash
retrograd train distill.toml
```

Prompts use the same chat JSONL as GRPO, ending with a user message. What
matters most is coverage: the student only learns to match the teacher on the
kind of prompts you give it. A few thousand varied prompts beat many more of a
single kind.

Both models stay loaded for the whole run. The teacher is used for inference
only, so it costs its weights and KV cache; a more heavily quantized teacher
leaves more room for the student.

Each update samples `samples_per_prompt` answers (default `1`) per prompt from
the student, scores the same tokens with both models, and pushes the student
toward the teacher token by token. `weight_clip` (default `5.0`) bounds how
much a single token can weigh: keep an eye on
`distill/advantage_clipped_fraction`, which should fall as the run progresses.

### What to watch

| Metric | Meaning |
| --- | --- |
| `distill/teacher_kl_mean` | Divergence from the teacher, in nats. The quantity being minimized. |
| `distill/teacher_kl_p95` | Its tail. If the mean falls but this does not, only the easy tokens are aligning. |
| `distill/advantage_clipped_fraction` | Share of tokens capped by `weight_clip`. |
| `completions/truncation_fraction` | Answers cut at `max_new_tokens`. |

An `[evaluation]` section reports the same divergence on held-out prompts,
using the teacher already loaded.

## Offline top-k

First write the teacher's predictions to a *sidecar* file, once:

```bash
retrograd distill-teacher distill.toml
```

This reads `teacher_path`, `data` and `sidecar` from `[distill]` (`--k`,
`--data` and `--out` override them; `k` defaults to 16). It loads one model at
a time, so it works on a machine that cannot hold both.

Then train:

```toml
[run]
algorithm = "distill"

[model]
path = "student.gguf"

[output]
path = "kd-adapter.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 2048
micro_batch = 512

[distill]
mode = "topk_offline"
teacher_path = "teacher-q4_k_m.gguf"
data = "corpus.jsonl"
sidecar = "corpus.topk"
offline_epochs = 3
```

`data` is chat JSONL, as for SFT. Training never opens the teacher, so it uses
as much memory as SFT. The sidecar records which corpus and tokenizer it was
made from, and a mismatch is refused. The run reports the SFT metrics
(`train/loss`, …); it has no scheduled evaluation, so measure it with `bench`.

## Measuring the result

```bash
retrograd bench distill.toml --data held-out.jsonl                              # baseline
retrograd bench distill.toml --data held-out.jsonl --adapter distill-adapter.gguf
```

`bench` loads the teacher and reports, for the student with and without the
adapter:

- **KL to the teacher**, per token.
- **Top-1 agreement**: how often the student and the teacher would pick the
  same next token.
- **Perplexity** of the reference answers, when held-out lines end with an
  assistant message. This makes the result comparable with an SFT run on the
  teacher's own outputs.

All `[distill]` keys are in the
[configuration reference](../reference/configuration#distill).

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

`fit_distill` runs on-policy distillation. The offline mode is driven through a
TOML file and `retrograd distill-teacher`.
