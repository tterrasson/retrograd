# PPO

PPO samples one answer per prompt, scores it with your reward program, and
reinforces answers that scored well. An optional critic lowers the variance of
the updates.

Use PPO when a program can give each answer an absolute score. If scores are
only meaningful when comparing answers to the same prompt, use
[GRPO](./grpo).

## Configuration

```toml
[run]
algorithm = "ppo"

[model]
path = "base.gguf"

[output]
path = "ppo-adapter.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 512
lr = 0.00001

[ppo]
prompts = "prompts.jsonl"
reward_command = ["python3", "reward.py"]
updates = 50
rollout_batch_size = 8
ppo_epochs = 2
clip_range = 0.2
kl_coefficient = 0.05

[ppo.sampling]
temperature = 0.8
top_p = 0.9
max_new_tokens = 128
seed = 42
```

```bash
retrograd train ppo.toml
```

Prompts are chat JSONL ending with a user message (see
[Datasets](../getting-started/datasets)). Each update samples
`rollout_batch_size` answers, scores them, then makes `ppo_epochs` optimizer
passes over that batch. The run does `updates` such rounds; `training.epochs`
is not used.

## Reward program

`reward_command` is run directly (not through a shell). It receives one JSON
line per answer and replies with one JSON line per answer, in the same order:

```json
{"prompt": "What is 2 + 2?", "completion": "4"}
```

```json
{"reward": 1.0}
```

`prompt` is the content of the prompt's last user message. The reward must be
a finite number.

By default (`reward_mode = "persistent"`) one process stays alive for the
whole run. It must answer a handshake first, echo the `_retrograd_batch` and
`_retrograd_index` fields of each request, echo the `_retrograd_batch_end`
marker, and flush after every line:

```python
import json, sys

PROTOCOL = "retrograd-reward/1"
hello = json.loads(sys.stdin.readline())
assert hello.get("protocol") == PROTOCOL
print(json.dumps({"protocol": PROTOCOL}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if "_retrograd_batch_end" in request:
        print(line.strip(), flush=True)
        continue
    score = 1.0 if request["completion"].strip() else 0.0
    print(json.dumps({
        "reward": score,
        "_retrograd_batch": request["_retrograd_batch"],
        "_retrograd_index": request["_retrograd_index"],
    }), flush=True)
```

`examples/smoke_rl_reward.py` is a complete example. Use
`reward_mode = "oneshot"` for a program that reads all of stdin before
answering: it is started once per batch and needs none of the extra fields.

Each batch must be answered within `reward_timeout_seconds` (default `300`,
including start-up for the first batch). Raise it if the program loads a model.

## Critic

The critic is enabled by default. It learns a per-token value estimate used
to compute advantages (GAE). Disable it with `[ppo.critic] enabled = false`
to use the whitened sequence reward instead. Its settings are in the
[configuration reference](../reference/configuration#ppo).

## Watching a run

`[observe]` saves every prompt, answer, reward and advantage, with a viewer.
See [Observing rollouts](./observe).
