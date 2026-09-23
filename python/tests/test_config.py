import json
from pathlib import Path

import pytest

from retrograd import (
    AgenticGRPOConfig,
    CommandJudge,
    ContainerEnvironment,
    CriticConfig,
    DistillConfig,
    GefenOptions,
    GRPOConfig,
    HttpEnvironment,
    JudgeContext,
    LocalEnvironment,
    LoraConfig,
    McpServer,
    MuonOptions,
    PPOConfig,
    RulerJudge,
    SamplingConfig,
    SandboxLimits,
    SandboxPool,
    Scenario,
    TrainableConfig,
    TrainingConfig,
    TrainSequence,
    WeightedBatch,
)


def test_lora_aliases_are_expanded_for_native_runtime() -> None:
    assert LoraConfig(targets="auto").native_targets() is None
    assert LoraConfig(targets="qv").native_targets() == [
        "blk.*.attn_q.weight",
        "blk.*.attn_v.weight",
    ]
    assert LoraConfig(targets=("q", "blk.0.custom.weight")).native_targets() == [
        "blk.*.attn_q.weight",
        "blk.0.custom.weight",
    ]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (lambda: TrainingConfig(context_size=0), "context_size"),
        (lambda: TrainingConfig(context_size=64), "context_size must be divisible"),
        (
            lambda: TrainingConfig(micro_batch=48),
            "context_size must be divisible",
        ),
        (
            lambda: TrainingConfig(max_sequences=129),
            "must not exceed micro_batch \\* gradient_accumulation",
        ),
        (lambda: TrainingConfig(max_gpu_duty_cycle=0.0), "max_gpu_duty_cycle"),
        (lambda: TrainingConfig(max_gpu_duty_cycle=-0.5), "max_gpu_duty_cycle"),
        (lambda: TrainingConfig(max_gpu_duty_cycle=1.5), "max_gpu_duty_cycle"),
        (lambda: TrainingConfig(max_gpu_duty_cycle=float("nan")), "max_gpu_duty_cycle"),
        (lambda: SamplingConfig(top_p=1.1), "top_p"),
        (lambda: LoraConfig(rank=0), "rank"),
        (lambda: LoraConfig(dtype="bf16"), "dtype"),
        (lambda: TrainableConfig(policy="lora"), "absence of TrainableConfig"),
        (lambda: TrainableConfig(policy="everything"), "policy must be"),
        (lambda: TrainableConfig(norms=True), "takes no selectors"),
        (lambda: TrainableConfig(layers="last:2"), "takes no selectors"),
        (lambda: TrainableConfig(policy="partial"), "selects nothing"),
        (lambda: TrainableConfig(policy="partial", layers="last:0"), "greater than zero"),
        (lambda: TrainableConfig(policy="partial", norms=True, layers="4..2"), "inclusive"),
        (lambda: TrainableConfig(policy="partial", norms=True, layers="most"), "layers must be"),
        (lambda: TrainableConfig(policy="partial", norms=True, layers="1..abc"), "not a number"),
        (
            lambda: TrainableConfig(policy="partial", norms=True, layers="1..4294967296"),
            "larger than a block index",
        ),
        (
            lambda: TrainableConfig(policy="partial", norms=True, layers="last:9" + "9" * 20),
            "larger than a block index",
        ),
        # These pass `str.isdigit` but not the runtime's `u32` parser.
        (lambda: TrainableConfig(policy="partial", norms=True, layers="٣..4"), "not a number"),
        (lambda: TrainableConfig(policy="partial", norms=True, layers="²..4"), "not a number"),
        (
            lambda: TrainableConfig(policy="hybrid", modules=("attn",)),
            "norms and biases beside the adapter",
        ),
        (
            lambda: TrainableConfig(policy="hybrid", norms=True, output_head=True),
            "norms and biases beside the adapter",
        ),
    ],
)
def test_configs_fail_early(config, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        config()


def test_weighted_batch_validates_flat_shapes() -> None:
    with pytest.raises(ValueError, match="each contain 4"):
        WeightedBatch(tokens=(1,), labels=(1,), weights=(1.0,), rows=1, context_size=4)


def test_rollout_configs_validate_algorithm_invariants() -> None:
    with pytest.raises(ValueError, match="reward_command"):
        PPOConfig("prompts.jsonl", ())
    with pytest.raises(ValueError, match="group_size"):
        GRPOConfig("prompts.jsonl", ("reward",), group_size=1)
    with pytest.raises(ValueError, match="group_size"):
        GRPOConfig("prompts.jsonl", ("reward",), group_size=257)


def test_configs_coerce_sequences_paths_and_preserve_native_options(tmp_path: Path) -> None:
    training = TrainingConfig(
        context_size=256,
        micro_batch=64,
        gradient_accumulation=2,
        max_sequences=2,
        generation_concurrency=2,
        fast_generation_context=True,
        kv_dtype="f16",
        scheduler="cosine",
        chunked_cross_entropy=True,
        gradient_checkpointing=True,
        device="cpu",
    )
    assert training.native_kwargs() == {
        "n_ctx": 256,
        "n_batch": 128,
        "n_ubatch": 64,
        "n_seq_max": 2,
        "generation_concurrency": 2,
        "generation_batch": 0,
        "fast_generation_context": True,
        "kv_dtype": "f16",
        "threads": 0,
        "epochs": 1,
        "learning_rate": 1.0e-4,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "scheduler": "cosine",
        "warmup_steps": 0,
        "optimizer": "adamw",
        "trainable_policy": "lora",
        "trainable_layers": "all",
        "trainable_modules": [],
        "trainable_norms": False,
        "trainable_biases": False,
        "trainable_output_head": False,
        "chunked_cross_entropy": True,
        "chunked_ce_tiles": 8,
        "chunked_ce_seq_chunk": 512,
        "gradient_checkpointing": True,
        "checkpoint_every_n_layers": 4,
        "master_weights": "auto",
        "shuffle": True,
        "shuffle_seed": 42,
        "max_gpu_duty_cycle": 1.0,
        "device": "cpu",
        "verbose": False,
    }
    # `None` and an explicit 1.0 both reach the ABI as 1.0: C has no need for the
    # distinction the dataclass keeps between omitted and "no limit".
    assert TrainingConfig(max_gpu_duty_cycle=1.0).native_kwargs()["max_gpu_duty_cycle"] == 1.0
    assert TrainingConfig(max_gpu_duty_cycle=0.5).native_kwargs()["max_gpu_duty_cycle"] == 0.5
    assert LoraConfig(targets=["q", "blk.0.custom.weight"], dtype="f16").targets == (
        "q",
        "blk.0.custom.weight",
    )
    assert PPOConfig(tmp_path / "prompts.jsonl", ["python", "reward.py"]).reward_command == (
        "python",
        "reward.py",
    )


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: TrainingConfig(threads=-1), "threads"),
        (lambda: TrainingConfig(generation_concurrency=129), "generation_concurrency"),
        (lambda: TrainingConfig(scheduler="exponential"), "scheduler"),
        (lambda: TrainingConfig(device="cuda"), "device"),
        (lambda: TrainingConfig(kv_dtype="bf16"), "kv_dtype"),
        (lambda: LoraConfig(dropout=1), "dropout"),
        (lambda: SamplingConfig(temperature=0), "temperature"),
        (lambda: CriticConfig(gamma=2), "gamma"),
        (lambda: PPOConfig("p", ("r",), clip_range=1), "clip_range"),
        (lambda: PPOConfig("p", ("r",), reward_mode="pipe"), "reward_mode"),
        (lambda: GRPOConfig("p", ("r",), reward_timeout_seconds=0), "reward_timeout_seconds"),
        (lambda: GRPOConfig("p", ("r",), clip_range_low=0.3, clip_range_high=0.2), "clip ranges"),
        (lambda: WeightedBatch((1,), (1,), (float("nan"),), 0, 1), "rows"),
        (lambda: TrainSequence((1, 2), (float("nan"),), (False, True), 1, 0), "finite"),
    ],
)
def test_all_public_data_validations_fail_before_native(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_agentic_serialization_is_strict_and_normalizes_mcp_values(tmp_path: Path) -> None:
    scenario_file = tmp_path / "scenarios.jsonl"
    scenario_file.write_text('{"id": "one", "user": "solve"}\n', encoding="utf-8")
    mcp = McpServer(name="tools", command=["tool-server"], allowed_tools=["sum"])
    config = AgenticGRPOConfig(
        scenarios=scenario_file,
        judge=RulerJudge("https://judge.invalid", "judge", cache_path=tmp_path / "cache"),
        mcp_servers=[mcp],
    )
    assert json.loads(config.scenarios_json()) == [
        {"id": "one", "user": "solve", "system": None, "metadata": {}}
    ]
    assert json.loads(config.mcp_servers_json())[0]["transport"]["command"] == ["tool-server"]
    optional = json.loads(
        AgenticGRPOConfig(
            scenarios=scenario_file,
            judge=CommandJudge(("judge",)),
            mcp_servers=[
                McpServer(
                    name="search",
                    url="https://tools.invalid/mcp",
                    denied_tools=["write_*"],
                    required=False,
                )
            ],
        ).mcp_servers_json()
    )[0]
    assert optional["required"] is False
    assert optional["denied_tools"] == ["write_*"]
    assert json.loads(config.judge_json())["config"]["cache_path"] == str(tmp_path / "cache")
    assert CommandJudge(["python", "judge.py"]).command == ("python", "judge.py")

    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"id": "one", "user": "solve", "extra": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fields"):
        AgenticGRPOConfig(bad_file, CommandJudge(("judge",))).scenarios_json()
    with pytest.raises(ValueError, match="exactly one"):
        McpServer(name="invalid", command=("tool",), url="https://tools.invalid")


def test_environments_serialize_into_the_native_tagged_shape() -> None:
    scenario = ({"id": "id", "user": "solve"},)
    judge = CommandJudge(("judge",))

    container = json.loads(
        AgenticGRPOConfig(
            scenario,
            judge,
            environment=ContainerEnvironment(
                profile="typescript",
                deny_tools=["bash"],
                pool=SandboxPool(max_live=4, min_idle=2, reuse="workspace"),
                limits=SandboxLimits(memory_mb=2048),
            ),
        ).environment_json()
    )
    assert container["type"] == "container"
    assert container["pool"] == {
        "max_live": 4,
        "min_idle": 2,
        "reuse": "workspace",
        "max_leases_per_container": 32,
    }
    # The native side reads durations in seconds under their own names.
    assert container["limits"]["exec_timeout_secs"] == 30
    assert container["limits"]["memory_mb"] == 2048
    assert container["setup_timeout_secs"] == 300

    local = json.loads(
        AgenticGRPOConfig(
            scenario, judge, environment=LocalEnvironment(allow_unsandboxed=True)
        ).environment_json()
    )
    assert local == {
        "type": "local",
        "profile": "python",
        "allow_unsandboxed": True,
        "deny_tools": [],
        "setup_timeout_secs": 300,
        "verify_timeout_secs": None,
    }
    selected = json.loads(
        AgenticGRPOConfig(
            scenario,
            judge,
            environment=LocalEnvironment(allow_unsandboxed=True, tools=("shell", "read_file")),
        ).environment_json()
    )
    assert selected["tools"] == ["shell", "read_file"]

    http = json.loads(
        AgenticGRPOConfig(
            scenario, judge, environment=HttpEnvironment("http://127.0.0.1:8099")
        ).environment_json()
    )
    assert http["type"] == "http" and http["request_timeout_secs"] == 120

    # No environment is the tool-only case, and it stays absent rather than
    # becoming an empty object the native side would have to interpret.
    assert AgenticGRPOConfig(scenario, judge).environment_json() is None


def test_a_judge_is_optional_but_something_has_to_grade_the_rollout() -> None:
    scenario = ({"id": "id", "user": "solve"},)
    # An environment that grades its own steps supplies the reward, so no judge
    # is required.
    graded = AgenticGRPOConfig(scenario, environment=HttpEnvironment("http://127.0.0.1:8099"))
    assert graded.judge_json() is None

    with pytest.raises(ValueError, match="neither a judge nor an environment"):
        AgenticGRPOConfig(scenario)


def test_an_environment_refuses_what_no_run_should_get_by_omission() -> None:
    scenario = ({"id": "id", "user": "solve"},)
    judge = CommandJudge(("judge",))
    # Unconfined execution is typed, never defaulted.
    with pytest.raises(ValueError, match="allow_unsandboxed"):
        LocalEnvironment()
    # A pool that must keep more warm than it may hold never converges.
    with pytest.raises(ValueError, match="min_idle"):
        SandboxPool(max_live=2, min_idle=8)
    with pytest.raises(ValueError, match="custom"):
        ContainerEnvironment(profile="custom")
    with pytest.raises(ValueError, match="base_url"):
        HttpEnvironment("127.0.0.1:8099")
    # Shared MCP tools need an explicit stateless assertion before they may be
    # composed with one isolated environment per trajectory.
    with pytest.raises(ValueError, match="declared stateless"):
        AgenticGRPOConfig(
            scenario,
            judge,
            mcp_servers=[McpServer(name="tools", command=("tool-server",))],
            environment=LocalEnvironment(allow_unsandboxed=True),
        )
    composed = AgenticGRPOConfig(
        scenario,
        judge,
        mcp_servers=[McpServer(name="tools", command=("tool-server",), stateless=True)],
        environment=LocalEnvironment(allow_unsandboxed=True),
    )
    assert json.loads(composed.mcp_servers_json())[0]["stateless"] is True


def test_judge_strategy_and_context_serialize_into_the_native_shape() -> None:
    # The default judge stays listwise-with-fallback on a default budget.
    default = json.loads(
        AgenticGRPOConfig(
            ({"id": "id", "user": "solve"},),
            RulerJudge("https://judge.invalid", "judge"),
        ).judge_json()
    )["config"]
    assert default["strategy"] == {"mode": "auto"}
    assert default["context"]["max_request_chars"] == 60_000
    # The flat helper fields never reach the native side.
    assert "mode" not in default and "max_pairs" not in default

    pairwise = json.loads(
        AgenticGRPOConfig(
            ({"id": "id", "user": "solve"},),
            RulerJudge(
                "https://judge.invalid",
                "judge",
                mode="pairwise",
                max_pairs=12,
                pairwise_rubric="pick the better one",
                context=JudgeContext(max_request_chars=20_000, max_trajectory_chars=4_000),
            ),
        ).judge_json()
    )["config"]
    assert pairwise["strategy"] == {"mode": "pairwise", "max_pairs": 12}
    assert pairwise["pairwise_rubric"] == "pick the better one"
    assert pairwise["context"]["max_trajectory_chars"] == 4_000

    chunked = json.loads(
        AgenticGRPOConfig(
            ({"id": "id", "user": "solve"},),
            RulerJudge("https://judge.invalid", "judge", mode="chunked", chunk_anchor=False),
        ).judge_json()
    )["config"]
    assert chunked["strategy"] == {"mode": "chunked", "anchor": False}


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: Scenario(" ", "solve"), "scenario id"),
        (lambda: Scenario("id", ""), "user message"),
        (lambda: McpServer("bad__name", command=("tool",)), "MCP server name"),
        (lambda: McpServer("tools", command=()), "MCP command"),
        (lambda: RulerJudge("", "judge"), "RULER base_url"),
        (lambda: RulerJudge("url", "judge", temperature=-1), "temperature"),
        (lambda: RulerJudge("url", "judge", mode="vibes"), "judge mode"),
        (lambda: RulerJudge("url", "judge", mode="pairwise", max_pairs=0), "max_pairs"),
        (
            lambda: JudgeContext(max_message_chars=9_000, max_trajectory_chars=8_000),
            "max_message_chars",
        ),
        (lambda: JudgeContext(head_ratio=1.5), "head_ratio"),
        (lambda: CommandJudge(()), "judge command"),
        (
            lambda: AgenticGRPOConfig(
                ({"id": "id", "user": "solve"},), CommandJudge(("j",)), group_size=1
            ),
            "group_size",
        ),
        (
            lambda: AgenticGRPOConfig(
                ({"id": "id", "user": "solve"},), CommandJudge(("j",)), judge_failure="skip"
            ),
            "judge_failure",
        ),
    ],
)
def test_agentic_values_fail_before_native_execution(factory, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()


def test_distill_refuses_a_group_outside_its_bounds() -> None:
    with pytest.raises(ValueError):
        DistillConfig("teacher.gguf", "prompts.jsonl", samples_per_prompt=0)
    with pytest.raises(ValueError):
        DistillConfig("teacher.gguf", "prompts.jsonl", samples_per_prompt=257)
    # One sample per prompt is the on-policy default, and unlike GRPO it is
    # admissible: the advantage is dense, so a group of one still carries it.
    assert DistillConfig("teacher.gguf", "prompts.jsonl").samples_per_prompt == 1


def test_distill_refuses_an_unusable_weight_clip() -> None:
    for clip in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            DistillConfig("teacher.gguf", "prompts.jsonl", weight_clip=clip)


def test_distill_keeps_truncated_completions_by_default_as_the_toml_does() -> None:
    assert DistillConfig("teacher.gguf", "prompts.jsonl").mask_truncated is False


def test_a_trainable_policy_reaches_the_binding_as_its_own_keywords() -> None:
    selection = TrainableConfig(policy="partial", layers="last:2", modules=["attn", "ffn_up"])
    sent = TrainingConfig(trainable=selection).native_kwargs()
    assert sent["trainable_policy"] == "partial"
    assert sent["trainable_layers"] == "last:2"
    assert sent["trainable_modules"] == ["attn", "ffn_up"]
    assert sent["trainable_norms"] is False
    assert TrainingConfig().native_kwargs()["trainable_policy"] == "lora"


@pytest.mark.parametrize("layers", ["all", "last:2", "last:+2", "00..02", "0..4294967295"])
def test_the_layer_grammar_is_the_one_the_runtime_parses(layers: str) -> None:
    # Both sides must accept the same strings for this check to mean anything.
    assert TrainableConfig(policy="partial", norms=True, layers=layers).layers == layers


def test_the_lora_defaults_are_not_a_shared_mutable() -> None:
    first = TrainingConfig().native_kwargs()
    first["trainable_modules"].append("attn")
    assert TrainingConfig().native_kwargs()["trainable_modules"] == []


def test_an_optimizer_section_only_reaches_the_binding_for_its_own_optimizer() -> None:
    sent = TrainingConfig(
        optimizer="muon", optimizer_options=MuonOptions(ns_steps=3, nesterov=False)
    ).native_kwargs()
    assert sent["optimizer"] == "muon"
    # Only the keys the caller set; an explicit `None` is dropped too.
    assert sent["muon"] == {"ns_steps": 3, "nesterov": False}
    assert "gefen" not in sent

    sent = TrainingConfig(
        optimizer="gefen", optimizer_options=GefenOptions(variant="quantized_m")
    ).native_kwargs()
    assert sent["gefen"] == {"variant": "quantized_m"}
    assert "muon" not in sent

    # No options at all crosses no section.
    assert "muon" not in TrainingConfig(optimizer="muon").native_kwargs()


def test_an_optimizer_section_for_another_optimizer_is_refused_in_python() -> None:
    """The same refusal the TOML frontend performs, made before the model is
    opened.
    """

    with pytest.raises(ValueError, match="does not use"):
        TrainingConfig(optimizer="adamw", optimizer_options=MuonOptions(momentum=0.9))
    with pytest.raises(ValueError, match="does not use"):
        TrainingConfig(optimizer="muon", optimizer_options=GefenOptions(block_size=512))
    with pytest.raises(ValueError, match="MuonOptions or a GefenOptions"):
        TrainingConfig(optimizer="muon", optimizer_options="muon")  # type: ignore[arg-type]
