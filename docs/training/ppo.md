# PPO training

PPO samples one completion for each prompt, sends the prompt and completion to
an external reward command, and updates the LoRA adapter with the returned
scalar rewards. The optional critic estimates per-token values and reduces
policy-gradient variance.

## Configuration

```toml
[run]
algorithm = "ppo"

[model]
path = "base.gguf"
device = "auto"

[lora]
output = "ppo-adapter.gguf"
rank = 8
alpha = 16.0
seed = 42

[training]
ctx = 512
micro_batch = 32
lr = 0.00001
max_grad_norm = 1.0

[ppo]
prompts = "prompts.jsonl"
reward_command = ["python3", "reward.py"]
reward_mode = "persistent"
reward_timeout_seconds = 300
updates = 50
rollout_batch_size = 8
ppo_epochs = 2
clip_range = 0.2
kl_coefficient = 0.05

[ppo.critic]
enabled = true
gamma = 1.0
gae_lambda = 0.95
value_lr = 0.01
value_epochs = 8
feature_dtype = "f32"

[ppo.sampling]
temperature = 0.8
top_p = 0.9
max_new_tokens = 128
seed = 42
```

Run it with:

```bash
retrograd train ppo.toml
```

The prompts file uses the chat JSONL envelope. A prompt may include system and
previous assistant turns, but it must end with a non-empty user message.

## Update sequence

Each PPO update performs these operations:

1. Sample `rollout_batch_size` completions from the current adapter.
2. Send each prompt and completion to `reward_command`.
3. Compute advantages with GAE when the critic is enabled. With the critic
   disabled, the scalar sequence reward is whitened and applied to its tokens.
4. Re-score the rollouts under the current policy and make `ppo_epochs` clipped
   optimizer passes over the same rollout batch.

`ppo.updates` controls how many new rollout batches are collected.
`ppo.ppo_epochs` controls how many policy passes use each batch. They are
independent from `training.epochs`, which is an SFT-only setting.

## Reward command protocol

The command is executed directly; `reward_command = ["./reward.sh"]` does not
run through a shell. For each request, the command receives a JSON object on
standard input and returns one JSON object on standard output:

```json
{"prompt":"What is 2 + 2?","completion":"4"}
```

```json
{"reward":1.0}
```

The reward must be finite. The number and order of response lines must match
the request batch.

`persistent` keeps one worker alive for the run. It begins with a handshake and
requires every response to be flushed:

```json
{"protocol":"retrograd-reward/1"}
{"protocol":"retrograd-reward/1"}
```

The worker then receives request lines with `_retrograd_batch` and
`_retrograd_index` fields and must echo those fields in each response. It must
also echo the `_retrograd_batch_end` marker. Use `oneshot` for a command that
reads stdin to end before producing output or cannot keep a worker alive.

Example persistent reward worker:

```python
import json
import sys

protocol = "retrograd-reward/1"
hello = json.loads(sys.stdin.readline())
if hello.get("protocol") != protocol:
    raise SystemExit("unsupported protocol")
print(json.dumps({"protocol": protocol}), flush=True)

for line in sys.stdin:
    request = json.loads(line)
    if "_retrograd_batch_end" in request:
        print(json.dumps(request), flush=True)
        continue
    batch = request.pop("_retrograd_batch")
    index = request.pop("_retrograd_index")
    score = 1.0 if request.get("completion", "").strip() else 0.0
    print(json.dumps({
        "reward": score,
        "_retrograd_batch": batch,
        "_retrograd_index": index,
    }), flush=True)
```

The default batch timeout is 300 seconds and includes startup time for the
first persistent batch. Increase it when the reward program loads a model.

## Critic settings

| Parameter | Default | Description |
| --- | ---: | --- |
| `ppo.critic.enabled` | `true` | Fit a value head and use GAE advantages. |
| `ppo.critic.gamma` | `1.0` | Per-token discount; `1.0` is undiscounted. |
| `ppo.critic.gae_lambda` | `0.95` | GAE bias/variance setting; `1.0` is Monte Carlo. |
| `ppo.critic.value_lr` | `0.01` | Value-head Adam learning rate. |
| `ppo.critic.value_epochs` | `8` | Full-batch value-head passes per update. |
| `ppo.critic.feature_dtype` | `f32` | Host feature storage: `f32`, `f16`, or `bf16`. |

## Sharing the GPU

`training.max_gpu_duty_cycle` bounds the fraction of wall time the trainer
spends waiting on GPU work it submitted, leaving the rest to another workload:

```toml
[training]
max_gpu_duty_cycle = 0.5
```

Rollout generation, teacher-forced scoring and the optimizer are all covered,
so an update spends its idle time where it spends its compute rather than only
at the optimizer step.

It frees **compute time, not device memory**: everything the run has allocated
stays allocated while it sleeps. Enabling it also costs the decode pipelining
once, before any sleep, so it is worth its overhead at `0.75` and below. See
[Configuration reference](../reference/configuration) for the full contract.

## PPO parameters

| Parameter | Description |
| --- | --- |
| `ppo.prompts` | Required chat JSONL prompt file. |
| `ppo.reward_command` | Required executable argv array. |
| `ppo.reward_mode` | `persistent` or `oneshot`. Default: `persistent`. |
| `ppo.reward_timeout_seconds` | Batch deadline. Default: `300`. |
| `ppo.updates` | Required number of rollout batches. |
| `ppo.rollout_batch_size` | Required prompts sampled per update. |
| `ppo.ppo_epochs` | Required policy passes over each rollout batch. |
| `ppo.clip_range` | Required value strictly between `0` and `1`; default examples use `0.2`. |
| `ppo.kl_coefficient` | Required non-negative anchor toward the frozen base policy. |
| `ppo.sampling.temperature` | Required rollout temperature. |
| `ppo.sampling.top_p` | Required nucleus threshold in `(0, 1]`. |
| `ppo.sampling.max_new_tokens` | Required completion token budget. |
| `ppo.sampling.seed` | Required sampling seed. |
