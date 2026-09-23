from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

PathLike: TypeAlias = str | os.PathLike[str]


@dataclass(frozen=True, slots=True)
class Scenario:
    id: str
    user: str
    system: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("scenario id must not be empty")
        if not self.user:
            raise ValueError("scenario user message must not be empty")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class McpServer:
    name: str
    command: tuple[str, ...] | None = None
    url: str | None = None
    env: Mapping[str, str] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    tool_timeout: int = 30
    #: Tool-name patterns to expose; ``None`` exposes everything the server
    #: advertises. ``*`` is a wildcard, so ``read_*`` keeps a family of tools.
    allowed_tools: tuple[str, ...] | None = None
    #: Patterns to hide, applied after ``allowed_tools`` and winning over it.
    denied_tools: tuple[str, ...] | None = None
    max_tool_result_bytes: int = 64 * 1024
    #: Whether an unreachable server fails the run instead of being skipped.
    required: bool = True
    #: Assert that sharing this server across group members has no stateful effect.
    stateless: bool = False

    def __post_init__(self) -> None:
        if not self.name or "__" in self.name:
            raise ValueError("MCP server name must be non-empty and must not contain '__'")
        if (self.command is None) == (self.url is None):
            raise ValueError("exactly one of command or url must be configured")
        if self.command is not None:
            object.__setattr__(self, "command", tuple(self.command))
            if not self.command:
                raise ValueError("MCP command must not be empty")
        if self.tool_timeout <= 0 or self.max_tool_result_bytes <= 0:
            raise ValueError("MCP timeout and result limit must be greater than zero")
        object.__setattr__(self, "env", dict(self.env))
        object.__setattr__(self, "headers", dict(self.headers))
        for attribute in ("allowed_tools", "denied_tools"):
            patterns = getattr(self, attribute)
            if patterns is not None:
                object.__setattr__(self, attribute, tuple(patterns))

    def native_dict(self) -> dict[str, object]:
        transport: dict[str, object]
        if self.command is not None:
            transport = {"type": "stdio", "command": self.command, "env": self.env}
        else:
            transport = {
                "type": "streamable_http",
                "url": self.url,
                "headers": self.headers,
            }
        return {
            "name": self.name,
            "transport": transport,
            "tool_timeout_secs": self.tool_timeout,
            "allowed_tools": self.allowed_tools,
            "denied_tools": self.denied_tools,
            "max_tool_result_bytes": self.max_tool_result_bytes,
            "required": self.required,
            "stateless": self.stateless,
        }


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """What one episode may consume. The defaults are the container crate's."""

    cpus: float = 1.0
    memory_mb: int = 1024
    pids: int = 256
    #: Per-command budget. Enforced *inside* the container, so a runaway process
    #: is killed rather than merely abandoned.
    exec_timeout: int = 30
    #: Cap on a tool result. An unbounded one eats the trajectory's whole token
    #: budget on a single call.
    max_output_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        if min(self.cpus, self.memory_mb, self.pids, self.exec_timeout) <= 0:
            raise ValueError("sandbox limits must all be greater than zero")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be greater than zero")

    def native_dict(self) -> dict[str, object]:
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "pids": self.pids,
            "exec_timeout_secs": self.exec_timeout,
            "max_output_bytes": self.max_output_bytes,
        }


@dataclass(frozen=True, slots=True)
class SandboxPool:
    """How containers are bounded, warmed and recycled."""

    #: Hard bound on live containers, whatever a group asks for. A group larger
    #: than this serializes instead of flattening the machine.
    max_live: int = 8
    #: Kept warm ahead of the next update, created while the optimizer runs.
    min_idle: int = 0
    #: ``"never"`` destroys a container per episode; ``"workspace"`` recycles it
    #: after wiping the workspace and killing leftovers.
    reuse: Literal["never", "workspace"] = "never"
    max_leases_per_container: int = 32

    def __post_init__(self) -> None:
        if self.max_live <= 0 or self.max_leases_per_container <= 0:
            raise ValueError("max_live and max_leases_per_container must be greater than zero")
        if self.min_idle < 0:
            raise ValueError("min_idle must not be negative")
        if self.min_idle > self.max_live:
            raise ValueError("min_idle must not exceed max_live")
        if self.reuse not in ("never", "workspace"):
            raise ValueError("reuse must be never or workspace")

    def native_dict(self) -> dict[str, object]:
        return asdict(self)


EnvironmentProfile: TypeAlias = Literal["python", "typescript", "custom"]


@dataclass(frozen=True, slots=True)
class ContainerEnvironment:
    """Container-backed environment with one pool per trajectory.

    Requires a native build with the ``container`` feature.
    """

    profile: EnvironmentProfile = "python"
    #: Overrides the profile's image. Prefer ``name@sha256:…`` - a run resumed
    #: three weeks later on a moving tag is not the same environment.
    image: str | None = None
    allow_network: bool = False
    limits: SandboxLimits = field(default_factory=SandboxLimits)
    pool: SandboxPool = field(default_factory=SandboxPool)
    #: Tool names to hide, ``*`` allowed. Adding a tool is done in Rust; taking
    #: one away is configuration.
    deny_tools: tuple[str, ...] = ()
    #: Positive registry selection. ``None`` keeps the profile preset.
    tools: tuple[str, ...] | None = None
    #: Volume mounted read-only on the profile's package cache. This is the real
    #: payoff of ``reuse="workspace"``: one install per run, not one per episode.
    cache_volume: str | None = None
    setup_timeout: int = 300
    verify_timeout: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "deny_tools", tuple(self.deny_tools))
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(self.tools))
        if self.profile not in ("python", "typescript", "custom"):
            raise ValueError(f"unknown environment profile {self.profile!r}")
        if self.profile == "custom" and not self.image:
            raise ValueError("profile 'custom' has no default image: set image")
        if self.setup_timeout <= 0:
            raise ValueError("setup_timeout must be greater than zero")
        if self.verify_timeout is not None and self.verify_timeout <= 0:
            raise ValueError("verify_timeout must be greater than zero")

    def native_dict(self) -> dict[str, object]:
        config: dict[str, object] = {
            "type": "container",
            "profile": self.profile,
            "image": self.image,
            "allow_network": self.allow_network,
            "limits": self.limits.native_dict(),
            "pool": self.pool.native_dict(),
            "deny_tools": list(self.deny_tools),
            "cache_volume": self.cache_volume,
            "setup_timeout_secs": self.setup_timeout,
            "verify_timeout_secs": self.verify_timeout,
            "run_id": None,
        }
        if self.tools is not None:
            config["tools"] = list(self.tools)
        return config


@dataclass(frozen=True, slots=True)
class LocalEnvironment:
    """Tools running on this machine, unconfined.

    Model-generated code runs with the training process's privileges and
    network. ``allow_unsandboxed`` must be typed: there is no default that turns
    it on.
    """

    profile: EnvironmentProfile = "python"
    allow_unsandboxed: bool = False
    deny_tools: tuple[str, ...] = ()
    tools: tuple[str, ...] | None = None
    setup_timeout: int = 300
    verify_timeout: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "deny_tools", tuple(self.deny_tools))
        if self.tools is not None:
            object.__setattr__(self, "tools", tuple(self.tools))
        if self.profile not in ("python", "typescript", "custom"):
            raise ValueError(f"unknown environment profile {self.profile!r}")
        if not self.allow_unsandboxed:
            raise ValueError(
                "a local environment runs model-generated code on this machine with no "
                "isolation; pass allow_unsandboxed=True to accept that"
            )

    def native_dict(self) -> dict[str, object]:
        config: dict[str, object] = {
            "type": "local",
            "profile": self.profile,
            "allow_unsandboxed": self.allow_unsandboxed,
            "deny_tools": list(self.deny_tools),
            "setup_timeout_secs": self.setup_timeout,
            "verify_timeout_secs": self.verify_timeout,
        }
        if self.tools is not None:
            config["tools"] = list(self.tools)
        return config


@dataclass(frozen=True, slots=True)
class HttpEnvironment:
    """An environment server speaking reset/step/state/close - the OpenEnv shape."""

    base_url: str
    request_timeout: int = 60
    #: Deadline of the connection itself, distinct from the request deadline: a
    #: server that is not listening should fail the rollout in seconds rather
    #: than hold it for the full request budget.
    connect_timeout: int = 10
    #: Sent with every call - an authorization header, a tenant id. The value is
    #: an operator secret: it belongs in the configuration, never in a scenario.
    headers: Mapping[str, str] = field(default_factory=dict)
    #: Connections kept alive to the server; keep it at least ``group_size``.
    pool_size: int = 16
    max_result_bytes: int = 64 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(self, "headers", dict(self.headers))
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http:// or https:// URL")
        if (
            min(
                self.request_timeout,
                self.connect_timeout,
                self.pool_size,
                self.max_result_bytes,
            )
            <= 0
        ):
            raise ValueError("HTTP environment timeouts and limits must be greater than zero")

    def native_dict(self) -> dict[str, object]:
        return {
            "type": "http",
            "base_url": self.base_url,
            "request_timeout_secs": self.request_timeout,
            "connect_timeout_secs": self.connect_timeout,
            "headers": dict(self.headers),
            "pool_size": self.pool_size,
            "max_result_bytes": self.max_result_bytes,
        }


Environment: TypeAlias = ContainerEnvironment | LocalEnvironment | HttpEnvironment


@dataclass(frozen=True, slots=True)
class JudgeContext:
    """Character budgets bounding a judge prompt.

    Trajectories are elided to fit rather than overrunning the judge's context
    window, which would surface as an API error and a dropped group.
    """

    max_request_chars: int = 60_000
    max_trajectory_chars: int = 8_000
    max_message_chars: int = 2_000
    head_ratio: float = 0.4

    def __post_init__(self) -> None:
        if min(self.max_request_chars, self.max_trajectory_chars, self.max_message_chars) <= 0:
            raise ValueError("judge context budgets must be greater than zero")
        if self.max_message_chars > self.max_trajectory_chars:
            raise ValueError("max_message_chars must not exceed max_trajectory_chars")
        if self.max_trajectory_chars > self.max_request_chars:
            raise ValueError("max_trajectory_chars must not exceed max_request_chars")
        if not 0.0 <= self.head_ratio <= 1.0:
            raise ValueError("head_ratio must be in [0, 1]")

    def native_dict(self) -> dict[str, object]:
        return asdict(self)


JudgeMode: TypeAlias = Literal["auto", "listwise", "chunked", "pairwise"]


@dataclass(frozen=True, slots=True)
class RulerJudge:
    base_url: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    rubric: str | None = None
    #: Rubric used for two-way comparisons; only read when ``mode="pairwise"``.
    pairwise_rubric: str | None = None
    temperature: float | None = None
    max_concurrency: int = 4
    timeout: int = 120
    max_retries: int = 2
    cache_path: PathLike | None = None
    #: ``auto`` sends one listwise request per group and falls back to anchored
    #: chunks when the group exceeds ``context.max_request_chars``. ``pairwise``
    #: compares two trajectories at a time and aggregates win rates: the most
    #: reliable signal, at more requests per group.
    mode: JudgeMode = "auto"
    #: Pairwise only. ``None`` compares every pair; otherwise keep at least one
    #: comparison per trajectory.
    max_pairs: int | None = None
    #: ``chunked`` only. Repeats the group's first trajectory in every chunk so
    #: per-chunk scores share a scale.
    chunk_anchor: bool = True
    context: JudgeContext = field(default_factory=JudgeContext)

    def __post_init__(self) -> None:
        if not self.base_url or not self.model or not self.api_key_env:
            raise ValueError("RULER base_url, model, and api_key_env must not be empty")
        if self.max_concurrency <= 0 or self.timeout <= 0 or self.max_retries < 0:
            raise ValueError("RULER concurrency/timeout must be positive and retries non-negative")
        if self.temperature is not None and self.temperature < 0:
            raise ValueError("RULER temperature must not be negative")
        if self.mode not in ("auto", "listwise", "chunked", "pairwise"):
            raise ValueError(f"unknown judge mode {self.mode!r}")
        if self.max_pairs is not None and self.max_pairs <= 0:
            raise ValueError("max_pairs must be greater than zero")

    def _strategy_dict(self) -> dict[str, object]:
        strategy: dict[str, object] = {"mode": self.mode}
        if self.mode == "pairwise" and self.max_pairs is not None:
            strategy["max_pairs"] = self.max_pairs
        if self.mode == "chunked":
            strategy["anchor"] = self.chunk_anchor
        return strategy

    def native_dict(self) -> dict[str, object]:
        config = asdict(self)
        config["timeout_secs"] = config.pop("timeout")
        config["cache_path"] = os.fspath(self.cache_path) if self.cache_path is not None else None
        # The native side takes one tagged `strategy` value; the flat fields
        # exist so callers do not have to build it by hand.
        for key in ("mode", "max_pairs", "chunk_anchor"):
            config.pop(key)
        config["strategy"] = self._strategy_dict()
        config["context"] = self.context.native_dict()
        return {"type": "ruler", "config": config}


@dataclass(frozen=True, slots=True)
class CommandJudge:
    command: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", tuple(self.command))
        if not self.command:
            raise ValueError("judge command must not be empty")

    def native_dict(self) -> dict[str, object]:
        return {"type": "command", "command": self.command}


ScenarioInput: TypeAlias = Scenario | Mapping[str, object]
Judge: TypeAlias = RulerJudge | CommandJudge


@dataclass(frozen=True, slots=True)
class AgenticGRPOConfig:
    scenarios: PathLike | Sequence[ScenarioInput]
    #: Optional judge. When absent, the environment must provide the reward.
    judge: Judge | None = None
    mcp_servers: tuple[McpServer, ...] = ()
    #: One stateful environment per trajectory. MCP servers may coexist only
    #: when each is explicitly declared stateless.
    environment: Environment | None = None
    updates: int = 1
    scenarios_per_update: int = 1
    group_size: int = 8
    epochs: int = 4
    max_turns: int = 6
    max_new_tokens: int = 512
    max_trajectory_tokens: int | None = None
    #: Wall-clock budget for one rollout. Overrunning truncates the trajectory,
    #: like overrunning the token budget. ``0`` disables the deadline.
    max_rollout_secs: int = 300
    #: What a turn that named no tool means. ``True`` - the default - reads it
    #: as the policy's final answer and ends the trajectory. ``False`` turns it
    #: into an error observation and keeps the episode going, which is what a
    #: world that decides its own terminal state needs: otherwise "answer in
    #: prose and stop" is the cheapest trajectory of its group, and GRPO trains
    #: the policy out of calling tools.
    end_on_no_tool_call: bool = True
    #: Consecutive turns without a valid tool call after which the trajectory
    #: is cut as a truncation, priced by ``truncation``. ``0`` never cuts. The
    #: early exit for a run under ``end_on_no_tool_call = False`` whose policy
    #: keeps missing the call format: it lands at the bottom of its group
    #: either way, this just stops paying to decode the rest of the episode.
    max_failed_turns: int = 0
    clip_range_low: float = 0.2
    clip_range_high: float = 0.28
    kl_coefficient: float = 0.0
    judge_failure: Literal["drop_group", "fail"] = "drop_group"
    max_dropped_fraction: float = 0.5
    #: Drop groups with identical judge scores; they have zero advantage and
    #: train nothing.
    drop_degenerate_groups: bool = False
    #: Skip updates with fewer than two trainable trajectories instead of raising.
    skip_empty_updates: bool = False
    #: Policy for trajectories that exceed their budget: drop them or assign the
    #: group's minimum reward.
    truncation: Literal["drop", "min_reward"] = "drop"
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "mcp_servers", tuple(self.mcp_servers))
        if self.judge is None and self.environment is None:
            raise ValueError(
                "neither a judge nor an environment: nothing would give a trajectory a reward"
            )
        if self.environment is not None:
            stateful = [server.name for server in self.mcp_servers if not server.stateless]
            if stateful:
                raise ValueError(
                    "MCP servers shared with an environment must be declared stateless: "
                    + ", ".join(stateful)
                )
        if (
            min(
                self.updates,
                self.scenarios_per_update,
                self.group_size,
                self.epochs,
                self.max_turns,
                self.max_new_tokens,
            )
            <= 0
        ):
            raise ValueError("agent counts and token limits must be greater than zero")
        if self.group_size < 2:
            raise ValueError("group_size must be at least 2")
        if self.max_trajectory_tokens is not None and self.max_trajectory_tokens < 2:
            raise ValueError("max_trajectory_tokens must be at least 2")
        if not 0 < self.clip_range_low < 1 or not 0 < self.clip_range_high < 1:
            raise ValueError("clip ranges must be in (0, 1)")
        if self.kl_coefficient < 0:
            raise ValueError("kl_coefficient must not be negative")
        if self.judge_failure not in ("drop_group", "fail"):
            raise ValueError("judge_failure must be drop_group or fail")
        if not 0 <= self.max_dropped_fraction <= 1:
            raise ValueError("max_dropped_fraction must be in [0, 1]")
        if self.truncation not in ("drop", "min_reward"):
            raise ValueError("truncation must be drop or min_reward")
        if self.max_rollout_secs < 0:
            raise ValueError("max_rollout_secs must not be negative")

    def scenarios_json(self) -> str:
        if isinstance(self.scenarios, (str, os.PathLike)):
            records = []
            with Path(self.scenarios).open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        raise ValueError(f"scenario line {line_number} is empty")
                    value = json.loads(line)
                    try:
                        records.append(asdict(Scenario(**value)))
                    except TypeError as error:
                        raise ValueError(
                            f"scenario line {line_number} has invalid fields: {error}"
                        ) from error
        else:
            records = [
                asdict(value if isinstance(value, Scenario) else Scenario(**value))
                for value in self.scenarios
            ]
        if not records:
            raise ValueError("scenarios must not be empty")
        return json.dumps(records)

    def judge_json(self) -> str | None:
        if self.judge is None:
            return None
        return json.dumps(self.judge.native_dict())

    def mcp_servers_json(self) -> str:
        return json.dumps([server.native_dict() for server in self.mcp_servers])

    def environment_json(self) -> str | None:
        if self.environment is None:
            return None
        return json.dumps(self.environment.native_dict())

    def native_kwargs(self, *, max_trajectory_tokens: int) -> dict[str, object]:
        """Return keyword arguments accepted by ``_Trainer.fit_agentic_grpo``."""

        return {
            "environment_json": self.environment_json(),
            "updates": self.updates,
            "scenarios_per_update": self.scenarios_per_update,
            "group_size": self.group_size,
            "epochs": self.epochs,
            "max_turns": self.max_turns,
            "max_new_tokens": self.max_new_tokens,
            "max_trajectory_tokens": max_trajectory_tokens,
            "max_rollout_secs": self.max_rollout_secs,
            "end_on_no_tool_call": self.end_on_no_tool_call,
            "max_failed_turns": self.max_failed_turns,
            "clip_range_low": self.clip_range_low,
            "clip_range_high": self.clip_range_high,
            "kl_coefficient": self.kl_coefficient,
            "judge_failure": self.judge_failure,
            "max_dropped_fraction": self.max_dropped_fraction,
            "drop_degenerate_groups": self.drop_degenerate_groups,
            "skip_empty_updates": self.skip_empty_updates,
            "truncation": self.truncation,
            "seed": self.seed,
        }
