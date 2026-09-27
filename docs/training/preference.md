# Preference optimization

Preference optimization trains on pairs of answers to the same prompt, one
preferred over the other. Use it when you can say which of two answers is
better but cannot write, or score, the best one. It runs offline: nothing is
generated and no reward command is called.

## Configuration

```toml
[run]
algorithm = "preference"

[model]
path = "base.gguf"

[output]
path = "dpo-adapter.gguf"

[lora]
init_adapter = "sft-adapter.gguf"   # optional: start from an SFT adapter

[training]
ctx = 1024
micro_batch = 256
epochs = 1
lr = 5e-6

[preference]
data = "pairs.jsonl"
loss = "dpo"
beta = 0.1
```

```bash
retrograd train dpo.toml
```

## Data

One pair per line: the `prompt` both answers respond to, the `chosen` answer and
the `rejected` one, each a list of chat messages.

```jsonl
{"prompt":[{"role":"user","content":"Name a primary color."}],"chosen":[{"role":"assistant","content":"Blue."}],"rejected":[{"role":"assistant","content":"Green."}]}
```

- `prompt` ends on a user turn (or on the `tool` observations that answer a
  call); a system message may only open it. Assistant turns inside the prompt
  are context and are not trained.
- `chosen` and `rejected` start with an assistant turn and must differ. A
  response may span several turns, tool calls and observations included; only
  what the model writes is trained.
- A `tools` catalog at the top level is offered to both answers, as in a
  [tool conversation](../getting-started/datasets#tool-conversations).
  `metadata` is kept and never read.

Both conversations are rendered with the model's chat template and must share
the prompt token for token; a template that rewrites the prompt once an answer
follows it is refused. A pair longer than `training.ctx` is refused rather than
truncated, because cutting an answer changes which one is better.

## Losses

| `loss` | Reference | Default `beta` | Extra keys |
| --- | --- | ---: | --- |
| `dpo` | yes | `0.1` | `label_smoothing` (conservative DPO), in `[0, 0.5)` |
| `ipo` | yes | `0.1` | |
| `simpo` | no | `2.0` | `gamma_beta_ratio` (target margin), default `0.5` |
| `orpo` | no | `0.1` | |

DPO and IPO measure each answer against a reference model; SimPO and ORPO use
the average log-probability of each answer and need none. ORPO adds the chosen
answer's negative log-likelihood to its loss, so it does not need an SFT stage
first; its `beta` weights the odds-ratio term, and the NLL is averaged per pair.
A key that the chosen loss does not read is an error.

## Reference

`dpo` and `ipo` compare the policy with a reference:

| Source | How | Use it when |
| --- | --- | --- |
| The initial policy | default (`reference = "initial"`) | Almost always. It is the model as it is before the first step, including an adapter loaded by `lora.init_adapter`: this is DPO after SFT. Works when base weights are trained. |
| The base model | `reference = "base"` | You train an adapter and want the base without it. Refused when base weights are trained. |
| Another model | a `[reference]` section | You have a separate reference GGUF. Omit `preference.reference`. |

The reference is scored once, before the first step. With a `[checkpoint]`
section its scores are kept in `preference-reference.bin` in the checkpoint
directory: the initial policy no longer exists once training has started, so a
run whose file is missing cannot be resumed and has to start again.

## Steps

A pair is two sequences that share the prompt. One optimizer step trains as
many whole pairs as fit in `training.ctx` tokens (`pairs_per_step` caps the
count), so `training.gradient_accumulation` is pinned to
`ctx / micro_batch`; lower `micro_batch` to save memory. When
`training.shared_prefix_fanout` allows it and the model supports it, both
sequences of a pair run in one pass over their shared prompt. Choose `ctx` so
that a whole pair fits: a pair larger than the window is still trained, over
several steps, and counted in `preference/split_pairs`.

Pairs are shuffled at each epoch (`shuffle = false` keeps file order), and a
run can resume from any step.

## Metrics

| Metric | Meaning |
| --- | --- |
| `preference/loss` | The loss being minimized. |
| `preference/accuracy` | Share of pairs whose chosen answer has the larger implicit reward. |
| `preference/margin` | Mean chosen reward minus rejected reward. |
| `preference/chosen_reward`, `preference/rejected_reward` | The implicit rewards: `beta` times the log-ratio to the reference, or times the mean log-probability. |
| `preference/chosen_logps`, `preference/rejected_logps` | Mean per-token log-probability of each answer. |
| `preference/nll`, `preference/log_odds` | ORPO only. |

Watch `chosen_logps`. A margin that grows because both answers become less
likely is the usual way DPO goes wrong; when `chosen_logps` has fallen by more
than `logps_drop_warn` nats (default `2.0`) together with `rejected_logps`, the
run says so once. It does not stop.

## Evaluation

```toml
[evaluation]
data = "eval_pairs.jsonl"
patience = 2
```

A preference file of held-out pairs, evaluated after every epoch:
`eval/preference_loss`, `eval/accuracy` and `eval/margin`. Early stopping reads
the loss. `max_examples` keeps that many pairs, evenly spaced.

## Pairs from agentic rollouts

`retrograd collect --pairs` rolls out the scenarios of an `agent_grpo`
configuration and writes, for each scenario, its best trace against its worst:

```bash
retrograd collect agent.toml --out pairs.jsonl --k 8 --pairs --min-gap 0.5
```

`chosen` is the highest-reward trace that passes every filter; `rejected` is
the lowest-reward complete one, whatever filter it failed. A trace with a
malformed call cannot be written as a record yet, so the next worst stands in
for it. A scenario whose two rewards differ by less than `--min-gap`
gives no pair, and neither does a group that all failed or all succeeded alike.
The prompt is what both traces open with, the catalog comes along as `tools`,
and the tool observations inside a response are context, never trained. See
[`collect`](../reference/cli#collect).

## From SFT to DPO

1. Train an SFT adapter on demonstrations ([SFT](sft)).
2. Build pairs: two answers per prompt, the better one as `chosen`.
3. Run `algorithm = "preference"` with `lora.init_adapter` set to the SFT
   adapter and the default `reference = "initial"`: the reference is the SFT
   policy, which is what DPO assumes.
4. Keep `lr` low - DPO usually wants a rate ten times below SFT's - and one
   or two epochs.

From Python, `Trainer.fit_preference(PreferenceConfig("pairs.jsonl"))` runs the
same loop, without checkpoints.
