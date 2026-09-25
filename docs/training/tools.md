# Tools and toolsets

A sandbox environment (`container` or `local`) gives each trajectory a
**toolset**: a named selection of **tools**. A tool is identified by
`id@version`. The model sees its name, description and schema, and those are
trained on. Changing any of them creates a new version instead of editing the
existing one, so two versions can be compared without retraining on the wrong
one.

```toml
[agent.environment]
type = "container"
profile = "python"          # image and package cache only

[agent.environment.tools]
default = "python"          # required
```

## Built-in toolsets

| Toolset | Tools |
| --- | --- |
| `base` | `bash`, `read_file`, `write_file`, `edit_file`, `list_dir`, `grep`, `submit` |
| `python` | `base` + `python`, `run_tests` (pytest) |
| `typescript` | `base` + `node`, `run_tests` (npm test) |

Every built-in tool is at version 1. Built-in toolsets pin exact versions,
so adding a new version of a tool does not change any existing run. Run
`retrograd tools list agent.toml` to see the resolved catalog, with each
tool's `id@version`, every toolset, and the catalog hash.

## Defining tools and toolsets

Put definitions inline under `[agent.environment.tools]`, or in TOML files
listed in `files`. Both use the same format:

```toml
# tools.toml
[[tool]]
id = "check_answer"
version = 1
description = "Check the final answer. Ends the episode."
input_schema = { type = "object", properties = { answer = { type = "string" } }, required = ["answer"] }
exec = { argv = ["python3", "-c"], script = "check_answer.py", protocol = "json" }

[[tool]]
id = "cargo_test"
version = 1
name = "run_tests"                                   # the name the model calls
builtin = { factory = "run_tests", params = { argv = ["cargo", "test", "-q"] } }

[toolset.rust]
include = ["base"]
tools = ["cargo_test@1", "check_answer"]             # no version = latest
deny = ["write_file"]                                # exposed-name patterns, `*` wildcard
```

```toml
[agent.environment.tools]
default = "rust"
files = ["tools.toml"]
```

A toolset lists its `include`d toolsets first, then its own `tools`. A tool
in `tools` replaces an included tool that has the same id or the same exposed
name. `deny` is applied last. Defining an `id@version` or a toolset name that
already exists is an error.

### Builtin tools

Set `builtin = { factory = "...", params = { ... } }`. You can change `name`
and `description`, but not `input_schema`: the implementation reads its own
arguments.

| Factory | Params |
| --- | --- |
| `shell` | `program` (default `bash`) |
| `interpreter` | `language`, `argv` (the code is appended) |
| `run_tests` | `argv` (the model's `args` are appended) |
| `read_file`, `write_file`, `edit_file`, `list_dir`, `grep`, `submit` | none |

A Rust embedder adds its own implementations with
`ToolRegistry::register_factory` and names them the same way.

### Exec tools (any language)

An exec tool is a program that runs **inside the trajectory's sandbox**, so it
is isolated per trajectory the same way the built-in tools are. The model's
arguments arrive on stdin as one JSON object.

| Key | Meaning |
| --- | --- |
| `argv` | Command to run. It must exist in the image, unless `script` provides it. |
| `script` | Host file whose contents are appended to `argv` as its last element (e.g. `argv = ["python3", "-c"]`). The path is relative to the file that declares it. Keep scripts under ~100 KB, which is the argument size limit. |
| `protocol` | `"text"` (default): the exit code, stdout and stderr are shown to the model. `"json"`: stdout must be a reply object (below). |
| `timeout_secs` | Defaults to the sandbox's exec timeout. |

With `protocol = "json"`, the program prints:

```json
{"content": "correct", "is_error": false, "reward": 1.0, "done": true}
```

Only `content` is required. `reward` is a step reward. `done` ends the episode
and runs the scenario's `verify` command, if it has one. If the command fails
or times out, the output is shown to the model like any other failed command.
If the command exits with status 0 but prints an invalid reply, the tool is
considered broken and the trajectory is dropped.

::: warning
The model can call a tool that returns a reward as many times as it likes. For
the final grade, use a `done` tool or the scenario's `verify` command.
:::

## Per-scenario toolsets

List the other toolsets that scenarios may select, then pick one per scenario
with `metadata.env.toolset`:

```toml
[agent.environment.tools]
default = "python"
scenario_toolsets = ["python-v2"]
```

```json
{"id": "t1", "user": "...", "metadata": {"env": {"toolset": "python-v2"}}}
```

All trajectories in a group share the same scenario, so they get the same
tools. If a scenario asks for a toolset that is not selectable, the run is
refused before the first rollout. Use this to train across several tool
vocabularies, or to A/B two versions of a tool in the same run.

## Reproducibility

The catalog hash covers every tool's spec and implementation, including the
exec argv, script contents, protocol and timeout, as well as the toolsets. It is
part of the run signature, so a resume fails if a definition file or a
script has changed since the run started.

## MCP servers

MCP servers (`[[agent.mcp_servers]]`) run on the host and are shared by every
trajectory. They are separate from toolsets. When the run also has an
environment, each server must set `stateless = true`. Use MCP for read-only
services such as search or documentation lookups, and use exec tools for
anything that acts on the trajectory's workspace.
