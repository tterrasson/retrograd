# Agentic GRPO

Agentic GRPO trains a model on multi-turn tasks where it calls tools: run a
shell command, edit a file, query an API. Each rollout is a whole conversation
(a *trajectory*), and trajectories are compared within their group as in
[GRPO](./grpo). Only the tokens the model wrote are trained; prompts and tool
results are context.

## Build

`agent_grpo` is part of the default build. Two optional pieces need a Cargo
feature:

| You use | Build with |
| --- | --- |
| MCP servers as tools | `--features mcp` |
| Docker/Podman sandboxes | `--features container` |

```bash
cargo build --release --features mcp,container
```

A configuration that needs a missing feature is refused at load time with the
feature to add.

## Scenarios

The task list is a JSONL file, one scenario per line:

```json
{"id": "math-1", "system": "Use tools when useful.", "user": "What is 37 + 5?"}
```

`id` and `user` are required. `metadata` is optional and carries a per-scenario
`rubric` for the judge, or a task for a sandbox environment (see below).

## Configuration

A run needs something that produces a reward: an environment that grades the
task, an LLM judge, or both.

```toml
[run]
algorithm = "agent_grpo"

[model]
path = "model.gguf"

[output]
path = "agent.gguf"

[lora]
rank = 8
alpha = 16.0

[training]
ctx = 4096
lr = 0.00001

[agent]
scenarios = "scenarios.jsonl"
updates = 100
scenarios_per_update = 2
group_size = 8
max_turns = 6
max_new_tokens_per_turn = 512

[agent.judge]
type = "ruler"
base_url = "https://api.openai.com/v1"
model = "gpt-5-mini"
api_key_env = "OPENAI_API_KEY"

[[agent.mcp_servers]]
name = "calc"
command = ["python3", "calculator_server.py"]
```

Prefer a reward the environment can verify (tests pass, puzzle solved) over a
judge's opinion. When a group is graded by its environment, the judge is not
called for it.

## Tools

A sandbox environment hands each trajectory a **toolset**: built-in tools,
your own tools in any language (run inside the sandbox), versioned so they can
be compared. See [Tools and toolsets](./tools).

**MCP servers** (`[[agent.mcp_servers]]`) are shared by every trajectory. Use
`command = [...]` for a local server or `url = "..."` for a remote one.
`allowed_tools` / `denied_tools` filter what the model sees (`*` wildcard, deny
wins), and `required = false` skips a server that cannot connect. An existing
`mcp.json` can be loaded with `mcp_config = "mcp.json"`.

List the tools a configuration exposes, without loading a model:

```bash
retrograd tools list agent.toml
```

## Environments

An environment gives each trajectory its own isolated world, so members of a
group cannot interfere with each other. Declare at most one in
`[agent.environment]`. When an environment is present, every MCP server must
set `stateless = true`.

### Containers

Each trajectory runs in its own container. `profile` picks the default image
and package cache; `tools.default` picks the toolset (`python` is `bash`,
`python`, `run_tests`, `read_file`, `write_file`, `edit_file`, `list_dir`,
`grep` and `submit`).

```toml
[agent.environment]
type = "container"
profile = "python"         # or "typescript", or "custom" with your own image
allow_network = false

[agent.environment.tools]
default = "python"

[agent.environment.limits]
cpus = 1.0
memory_mb = 1024
exec_timeout_secs = 30

[agent.environment.pool]
max_live = 8               # containers alive at once
reuse = "never"            # "workspace" recycles a cleaned container
```

A scenario describes its task in `metadata.env`. `verify` turns the result into
a reward; `summary` is what the judge sees instead of the transcript; `toolset`
selects one of the environment's `scenario_toolsets`:

```json
{"id": "fix-parser", "user": "test_parse_empty fails. Fix it.",
 "metadata": {"env": {
   "files": {"src/parser.py": "...", "tests/test_parser.py": "..."},
   "setup": ["pip install -e ."],
   "verify": {"command": ["pytest", "-q"], "reward_on_success": 1.0},
   "summary": ["git", "diff"]}}}
```

The container daemon must be running (Docker Desktop, Colima, or Podman in
Docker mode; set `DOCKER_HOST` if needed). Containers run without network,
as non-root, with a read-only root filesystem. Pin images by digest
(`image = "name@sha256:..."`) for reproducible runs. Docker is not a security
boundary against hostile code.

`type = "local"` runs the same tools directly on the host, without isolation.
It is refused unless `allow_unsandboxed = true`.

### HTTP

`type = "http"` connects to your own environment server (OpenEnv servers are
compatible):

```toml
[agent.environment]
type = "http"
base_url = "http://127.0.0.1:8099"
```

| Route | Request | Response |
| --- | --- | --- |
| `GET /tools` | | `{"tools": [...]}` |
| `POST /reset` | `{"scenario": ..., "seed": n}` | `{"env_id": "...", "observation": null}` |
| `POST /step` | `{"env_id": ..., "call": ...}` | `{"result": ..., "reward": null, "done": false}` |
| `GET /state/{id}` | | final state for the judge, or `404` |
| `POST /close` | `{"env_id": ...}` | any 2xx |

A `reward` returned by `/step` is a step reward; `done: true` ends the episode.

## Rollout settings

| Key | Default | Meaning |
| --- | ---: | --- |
| `max_turns` | `6` | Assistant turns per trajectory. |
| `max_new_tokens_per_turn` | `512` | Tokens per assistant turn. |
| `max_rollout_secs` | `300` | Time budget per trajectory; `0` disables it. |
| `end_on_no_tool_call` | `true` | A turn without a tool call ends the trajectory. Set `false` when the environment decides when the task is over. |
| `truncation` | `"drop"` | What to do with a trajectory that ran out of budget: drop it, or keep it with the group's lowest reward (`"min_reward"`). |
| `max_dropped_fraction` | `0.5` | Stop if more than this share of an update's trajectories is lost. |

The loss is divided by the token budget (`max_turns × max_new_tokens_per_turn`,
capped at `max_trajectory_tokens`), so changing these limits scales the gradient: re-tune `lr` when you change
them. Every key is listed in the
[configuration reference](../reference/configuration#agentic-grpo).

## Evaluation and checkpoints

`[evaluation]` and `[checkpoint]` work as for the other algorithms, with a
scenario file as held-out data. Evaluation reports the environment's reward,
not the judge's: judge scores are relative to a group and cannot be compared
across updates. Held-out scenarios must therefore be verifiable (a `verify`
command, or an HTTP environment that returns rewards).

## Tooling

```bash
retrograd tools list agent.toml --json          # the exact tool catalog
retrograd scenarios generate agent.toml         # write scenarios with an LLM
retrograd judge eval agent.toml --fixtures f.jsonl   # measure a judge against labels
```

`scenarios generate` reads `[agent.scenario_generation]` (`model`, `base_url`,
`api_key_env`, `count`) and is never run implicitly by `train`.

## What to watch

| Metric | Meaning |
| --- | --- |
| `agent/truncated_fraction` | Trajectories that hit a limit. Above ~20%, raise the limits or use `truncation = "min_reward"`. |
| `agent/failed_fraction` | Trajectories that crashed (tool or runtime). |
| `agent/tool_calls_per_turn` | Falls first when the model stops using tools. |
| `judge/degenerate_group_fraction` | Groups where the judge gave everyone the same score. |
| `env/acquire_ms_mean` | Time waiting for a container. |

[`[observe]`](./observe) shows full conversations with their tool calls,
rewards and judge explanations.

## Python

```python
from retrograd import AgenticGRPOConfig, LoraConfig, McpServer, RulerJudge, Scenario, Trainer

config = AgenticGRPOConfig(
    scenarios=(Scenario("math-1", "Use the calculator: 37 + 5"),),
    judge=RulerJudge(base_url="https://api.openai.com/v1", model="gpt-5-mini"),
    mcp_servers=(McpServer(name="calc", command=("python3", "calculator_server.py")),),
    updates=10,
    group_size=8,
)

with Trainer("model.gguf", lora=LoraConfig()) as trainer:
    trainer.fit_agentic_grpo(config, callback=print)
    trainer.save_adapter("agent.gguf")
```

A sandbox environment names its toolset explicitly:
`environment=ContainerEnvironment(Tools("python"))`, or
`Tools("rust", files=("tools.toml",))` for your own definitions.
