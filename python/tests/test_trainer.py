from __future__ import annotations

from pathlib import Path

import pytest

from retrograd import (
    AgenticGRPOConfig,
    CommandJudge,
    CriticConfig,
    GRPOConfig,
    LoraConfig,
    PPOConfig,
    RetrogradError,
    SamplingConfig,
    Scenario,
    TrainableConfig,
    Trainer,
    TrainingConfig,
    TrainSequence,
    WeightedBatch,
)

METRICS = (1, True, 3, 0.25, float("nan"), 123.0, 1.0e-4)


class FakeNativeError(RuntimeError):
    pass


class FakeDataset:
    context_size = 8
    examples = 2
    supervised_tokens = 7


class FakeTrainer:
    def __init__(self, model_path: str, **config) -> None:
        self.model_path = model_path
        self.config = config
        self.closed = False
        self.context_size = 8
        self.eos_token = 2
        self.hidden_size = 2
        self.created_lora = None
        self.saved_to = None
        self.saved_bundle = None
        self.saved_model = None
        self.reference_path = None
        self.trainable_set = (
            ("full", [("blk.0.attn_q.weight", "base", "F32", [4, 4, 1, 1], 16, 64)], [])
            if config["trainable_policy"] != "lora"
            else None
        )

    def save_trainable(self, path: str) -> None:
        self.saved_bundle = path

    def load_trainable(self, path: str) -> None:
        self.loaded_bundle = path

    def save_model(self, path: str) -> None:
        self.saved_model = path

    def attach_reference(self, path: str, n_ctx: int | None) -> None:
        self.reference_path = path
        self.reference_ctx = n_ctx

    def create_lora(self, **config) -> None:
        self.created_lora = config

    def load_lora(self, path: str) -> None:
        self.loaded_from = path

    def save_lora(self, path: str) -> None:
        self.saved_to = path

    def prepare_dataset(self, path: str, format: str, context_size: int) -> FakeDataset:
        self.prepared = (path, format, context_size)
        return FakeDataset()

    def fit_sft(self, train, eval, callback):
        if callback:
            callback(METRICS)
        return METRICS

    def tokenize(self, text: str) -> list[int]:
        return [ord(character) for character in text]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)

    def format_chat(self, messages, add_assistant: bool) -> str:
        suffix = "assistant:" if add_assistant else ""
        return "".join(f"{role}:{content};" for role, content in messages) + suffix

    def generate(self, prompt, **sampling):
        self.generation_call = (prompt, sampling)
        return [79, 75], [-0.1, -0.2]

    def score(self, tokens):
        return [-0.5] * (len(tokens) - 1)

    def score_reference(self, tokens):
        return [-1.0] * (len(tokens) - 1)

    def hidden_states(self, tokens):
        return [float(value) for token in tokens for value in (token, token + 1)]

    def train_tokens(self, tokens):
        return METRICS

    def train_weighted(self, *args):
        self.weighted_call = args
        return METRICS

    def fit_ppo(self, prompts, reward_command, **config):
        self.ppo_call = (prompts, reward_command, config)
        if config["callback"]:
            config["callback"]((METRICS, [("reward/mean", 0.75)]))
        return METRICS

    def fit_grpo(self, prompts, reward_command, **config):
        self.grpo_call = (prompts, reward_command, config)
        if config["callback"]:
            config["callback"]((METRICS, [("reward/mean", 0.5)]))
        return METRICS

    def train_grpo_batch(self, *args, **config):
        self.grpo_batch_call = (args, config)
        if config["callback"]:
            config["callback"]((METRICS, [("batch/trained_fraction", 1.0)]))
        return METRICS

    def fit_agentic_grpo(self, *args, **config):
        self.agentic_call = (args, config)
        if config["callback"]:
            config["callback"]((METRICS, [("agent/turns_per_traj_mean", 2.0)]))
        return METRICS

    def describe_lora(self):
        return "lora"

    def backend_report(self):
        return "backend"

    def capability_report(self):
        return "capabilities"

    def preflight(self):
        return "preflight"

    def close(self) -> None:
        self.closed = True


class FakeModule:
    _Trainer = FakeTrainer
    RetrogradNativeError = FakeNativeError


def trainer(**kwargs) -> Trainer:
    return Trainer("model.gguf", _native_module=FakeModule, **kwargs)


def test_constructor_can_create_a_high_level_lora() -> None:
    model = trainer(lora=LoraConfig(rank=4, targets=("q", "v")))
    assert model._native.created_lora["rank"] == 4
    assert model._native.created_lora["targets"] == [
        "blk.*.attn_q.weight",
        "blk.*.attn_v.weight",
    ]


def test_unset_shuffle_seed_follows_the_lora_seed() -> None:
    # An unset shuffle seed inherits the LoRA seed; otherwise it uses its own value.
    assert trainer(lora=LoraConfig(seed=7))._native.config["shuffle_seed"] == 7
    assert trainer(lora=LoraConfig())._native.config["shuffle_seed"] == 42
    assert trainer()._native.config["shuffle_seed"] == 42

    explicit = trainer(
        training=TrainingConfig(shuffle=False, shuffle_seed=99),
        lora=LoraConfig(seed=7),
    )
    assert explicit._native.config["shuffle_seed"] == 99
    assert explicit._native.config["shuffle"] is False


def test_the_duty_cycle_reaches_the_native_trainer() -> None:
    # `None` and an explicit 1.0 are the same request - no limit - and both
    # reach the ABI as 1.0, which is how C spells it.
    assert trainer()._native.config["max_gpu_duty_cycle"] == 1.0
    throttled = trainer(training=TrainingConfig(max_gpu_duty_cycle=0.5))
    assert throttled._native.config["max_gpu_duty_cycle"] == 0.5


def test_prepare_and_fit_paths_with_python_metrics(tmp_path: Path) -> None:
    model = trainer(lora=LoraConfig())
    progress = []
    metrics = model.fit(tmp_path / "train.jsonl", callback=progress.append)

    assert model._native.prepared == (str(tmp_path / "train.jsonl"), "chat_jsonl", 8)
    assert metrics.global_step == 3
    assert len(progress) == 1
    assert progress[0].global_step == metrics.global_step
    assert progress[0].train_loss == metrics.train_loss


def test_generate_and_chat_return_text_and_policy_data() -> None:
    model = trainer()
    generation = model.chat(
        [{"role": "user", "content": "hello"}],
        sampling=SamplingConfig(max_new_tokens=2, seed=7),
    )

    assert generation.text == "OK"
    assert generation.tokens == (79, 75)
    assert generation.logprobs == (-0.1, -0.2)
    assert model._native.generation_call[1]["seed"] == 7


def test_score_and_hidden_states_are_shaped_at_the_python_boundary() -> None:
    model = trainer()
    assert model.score("abc").total_logprob == pytest.approx(-1.0)
    assert model.score("abc", reference=True).logprobs == (-1.0, -1.0)
    assert model.hidden_states([1, 2]) == ((1.0, 2.0), (2.0, 3.0))


def test_direct_training_and_adapter_reports_delegate_to_native(tmp_path: Path) -> None:
    model = trainer(lora=LoraConfig())
    assert model.train_tokens([1, 2]).global_step == 3
    weighted = model.train_weighted(
        WeightedBatch((1, 2), (1, 2), (1.0, 1.0), 1, 2), scheduler_total_steps=5
    )
    assert weighted.global_step == 3
    assert model._native.weighted_call == ([1, 2], [1, 2], [1.0, 1.0], 1, 2, 5)
    assert model.describe_adapter() == "lora"
    assert model.backend_report() == "backend"
    assert model.capability_report() == "capabilities"
    assert model.load_adapter(tmp_path / "adapter.gguf") is model
    assert model._native.loaded_from == str(tmp_path / "adapter.gguf")


def test_rollout_algorithms_expose_named_progress_metrics() -> None:
    model = trainer(
        training=TrainingConfig(max_sequences=4),
        lora=LoraConfig(),
    )
    ppo_progress = []
    grpo_progress = []

    ppo = model.fit_ppo(
        PPOConfig("prompts.jsonl", ("python", "reward.py")), callback=ppo_progress.append
    )
    grpo = model.fit_grpo(
        GRPOConfig("prompts.jsonl", ("python", "reward.py")), callback=grpo_progress.append
    )

    assert ppo.global_step == grpo.global_step == 3
    assert ppo_progress[0].values["reward/mean"] == 0.75
    assert grpo_progress[0].values["reward/mean"] == 0.5


def test_rollout_configurations_translate_every_native_keyword(tmp_path: Path) -> None:
    model = trainer(training=TrainingConfig(max_sequences=8))
    model.fit_ppo(
        PPOConfig(
            tmp_path / "ppo.jsonl",
            ["judge"],
            reward_mode="oneshot",
            reward_timeout_seconds=45,
            updates=2,
            rollout_batch_size=3,
            epochs=5,
            clip_range=0.15,
            kl_coefficient=0.2,
            critic=CriticConfig(False, 0.8, 0.7, 0.03, 6, feature_dtype="f16"),
            sampling=SamplingConfig(0.6, 0.9, 11, 9),
        )
    )
    prompts, reward_command, ppo = model._native.ppo_call
    assert prompts == str(tmp_path / "ppo.jsonl")
    assert reward_command == ["judge"]
    assert ppo == {
        "reward_mode": "oneshot",
        "reward_timeout_seconds": 45,
        "updates": 2,
        "rollout_batch_size": 3,
        "ppo_epochs": 5,
        "clip_range": 0.15,
        "kl_coefficient": 0.2,
        "critic_enabled": False,
        "gamma": 0.8,
        "gae_lambda": 0.7,
        "value_learning_rate": 0.03,
        "value_epochs": 6,
        "feature_dtype": "f16",
        "temperature": 0.6,
        "top_p": 0.9,
        "max_new_tokens": 11,
        "seed": 9,
        "callback": None,
    }
    model.fit_grpo(
        GRPOConfig(
            tmp_path / "grpo.jsonl",
            ["judge"],
            "persistent",
            120,
            2,
            3,
            4,
            5,
            0.1,
            0.3,
            0.2,
            True,
            12,
            7,
        )
    )
    prompts, reward_command, grpo = model._native.grpo_call
    assert prompts == str(tmp_path / "grpo.jsonl")
    assert reward_command == ["judge"]
    assert grpo == {
        "reward_mode": "persistent",
        "reward_timeout_seconds": 120,
        "updates": 2,
        "prompts_per_update": 3,
        "group_size": 4,
        "grpo_epochs": 5,
        "clip_range_low": 0.1,
        "clip_range_high": 0.3,
        "kl_coefficient": 0.2,
        "mask_truncated": True,
        "max_new_tokens": 12,
        "seed": 7,
        "callback": None,
    }


def test_grpo_requires_a_context_sized_for_its_group() -> None:
    model = trainer()
    with pytest.raises(ValueError, match="max_sequences"):
        model.fit_grpo(GRPOConfig("prompts.jsonl", ("reward",), group_size=4))


def test_pre_generated_grpo_batch_maps_masks_groups_and_progress() -> None:
    model = trainer(lora=LoraConfig())
    progress = []
    rows = (
        TrainSequence((1, 2, 3), (-0.5,), (False, False, True), 0.0, 9),
        TrainSequence((1, 2, 4), (-0.7,), (False, False, True), 1.0, 9),
    )

    metrics = model.train_grpo_batch(rows, loss_denominator=8, callback=progress.append)

    args, kwargs = model._native.grpo_batch_call
    assert args[0] == [[1, 2, 3], [1, 2, 4]]
    assert args[2] == [[False, False, True], [False, False, True]]
    assert args[4] == [9, 9]
    assert kwargs["loss_denominator"] == 8
    assert metrics.global_step == 3
    assert progress[0].values["batch/trained_fraction"] == 1.0


def test_train_sequence_rejects_misaligned_policy_scores() -> None:
    with pytest.raises(ValueError, match="align"):
        TrainSequence((1, 2, 3), (), (False, False, True), 1.0, 1)


def test_agentic_grpo_serializes_strict_scenarios_and_judge() -> None:
    model = trainer(lora=LoraConfig())
    progress = []
    config = AgenticGRPOConfig(
        scenarios=(Scenario("s1", "solve"),),
        judge=CommandJudge(("python", "judge.py")),
        group_size=2,
        max_new_tokens=2,
        max_trajectory_tokens=8,
    )

    metrics = model.fit_agentic_grpo(config, callback=progress.append)

    args, kwargs = model._native.agentic_call
    assert '"id": "s1"' in args[0]
    assert '"type": "command"' in args[1]
    assert kwargs["group_size"] == 2
    assert kwargs["max_trajectory_tokens"] == 8
    assert metrics.global_step == 3
    assert progress[0].values["agent/turns_per_traj_mean"] == 2.0


def test_dataset_cannot_be_used_with_another_model(tmp_path: Path) -> None:
    first = trainer()
    second = trainer()
    dataset = first.prepare_dataset(tmp_path / "train.txt")
    with pytest.raises(ValueError, match="another Trainer"):
        second.fit(dataset)


def test_context_manager_closes_native_trainer() -> None:
    with trainer() as model:
        native = model._native
        assert not model.closed
    assert native.closed


def test_pathlikes_messages_and_callback_errors_stay_at_python_boundary(tmp_path: Path) -> None:
    model = trainer()
    assert model.format_chat(({"role": "user", "content": "hello"},)) == "user:hello;"
    with pytest.raises(ValueError, match="require role and content"):
        model.format_chat(({"role": "user"},))
    with pytest.raises(ValueError, match="role, content"):
        model.format_chat((("user", "hello", "extra"),))
    assert model.save_adapter(tmp_path / "adapter.gguf") == tmp_path / "adapter.gguf"

    def fail(_: object) -> None:
        raise LookupError("callback failed")

    with pytest.raises(LookupError, match="callback failed"):
        model.fit(tmp_path / "train.txt", callback=fail)


def test_native_runtime_errors_are_exposed_as_public_error() -> None:
    model = trainer()

    def fail():
        raise FakeNativeError("boom")

    model._native.preflight = fail
    with pytest.raises(RetrogradError, match="boom"):
        model.preflight()


def test_a_base_policy_reaches_the_binding_and_reports_its_resolved_set() -> None:
    model = trainer(training=TrainingConfig(trainable=TrainableConfig(policy="full")))

    assert model._native.config["trainable_policy"] == "full"
    selection = model.trainable_set
    assert selection is not None
    assert selection.policy == "full"
    assert selection.tensors[0].name == "blk.0.attn_q.weight"
    assert selection.tensors[0].shape == (4, 4, 1, 1)
    assert selection.n_parameters == 16
    assert selection.n_bytes == 64
    assert trainer(lora=LoraConfig()).trainable_set is None


def test_the_bundle_and_the_model_export_have_their_own_paths(tmp_path: Path) -> None:
    model = trainer(training=TrainingConfig(trainable=TrainableConfig(policy="full")))

    assert model.save_trainable(tmp_path / "bundle.gguf") == tmp_path / "bundle.gguf"
    assert model._native.saved_bundle == str(tmp_path / "bundle.gguf")
    model.load_trainable(tmp_path / "bundle.gguf")
    assert model._native.loaded_bundle == str(tmp_path / "bundle.gguf")
    assert model.save_model(tmp_path / "model.gguf") == tmp_path / "model.gguf"
    assert model._native.saved_model == str(tmp_path / "model.gguf")


def test_an_anchor_is_attached_with_its_own_context_width(tmp_path: Path) -> None:
    model = trainer(training=TrainingConfig(trainable=TrainableConfig(policy="full")))
    model.attach_reference(tmp_path / "anchor.gguf", context_size=512)

    assert model._native.reference_path == str(tmp_path / "anchor.gguf")
    assert model._native.reference_ctx == 512
    assert model.reference_path == str(tmp_path / "anchor.gguf")
    model.attach_reference(tmp_path / "anchor.gguf")
    assert model._native.reference_ctx is None


@pytest.mark.parametrize(
    ("policy", "kwargs", "message"),
    [
        ("full", {"lora": LoraConfig()}, "trains base tensors and no adapter"),
        ("partial", {"adapter": "a.gguf"}, "trains base tensors and no adapter"),
        ("hybrid", {}, "trains an adapter beside the base tensors"),
    ],
)
def test_a_policy_and_an_adapter_have_to_agree(policy, kwargs, message: str) -> None:
    selection = (
        TrainableConfig(policy=policy)
        if policy == "full"
        else TrainableConfig(policy=policy, norms=True)
    )
    with pytest.raises(ValueError, match=message):
        trainer(training=TrainingConfig(trainable=selection), **kwargs)
