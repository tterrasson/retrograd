"""Direct contracts for the compiled PyO3 extension.

The fixture is deliberately optional for the fast Python lane: the test is a
real binding smoke test when both the extension and the CPU GGUF are present.
The CPU integration lane supplies the fixture and therefore must execute it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _fixture() -> Path:
    configured = os.environ.get("RETRO_CPU_FIXTURE")
    fallback = Path(__file__).parents[2] / "tests" / "fixtures" / "LFM2.5-230M-Q4_K_M.gguf"
    fixture = Path(configured) if configured else fallback
    if not fixture.is_file():
        pytest.skip("CPU GGUF fixture is unavailable")
    return fixture


def _tiny_fixture() -> Path:
    configured = os.environ.get("RETRO_TINY_FIXTURE")
    fallback = Path(__file__).parents[2] / "tests" / "fixtures" / "retrograd-tiny-qwen2-f32.gguf"
    fixture = Path(configured) if configured else fallback
    if not fixture.is_file():
        pytest.skip("generated tiny GGUF fixture is unavailable")
    return fixture


def test_pyo3_validates_arguments_before_loading_a_model() -> None:
    native = pytest.importorskip("retrograd._native")
    with pytest.raises(ValueError, match="scheduler"):
        native._Trainer("missing.gguf", scheduler="invalid")
    with pytest.raises(ValueError, match="device"):
        native._Trainer("missing.gguf", device="invalid")
    # Refused before the model is opened.
    with pytest.raises(ValueError, match="takes no selectors"):
        native._Trainer("missing.gguf", trainable_policy="full", trainable_norms=True)
    with pytest.raises(ValueError, match="selects nothing"):
        native._Trainer("missing.gguf", trainable_policy="partial")
    with pytest.raises(ValueError, match="trains no base tensor"):
        native._Trainer("missing.gguf", trainable_norms=True)


def test_pyo3_real_fixture_tokenizes_and_closes() -> None:
    native = pytest.importorskip("retrograd._native")
    trainer = native._Trainer(str(_fixture()), n_ctx=128, n_batch=128, n_ubatch=32)
    try:
        tokens = trainer.tokenize("hello")
        assert tokens
        assert isinstance(trainer.detokenize(tokens), str)
    finally:
        trainer.close()
    assert trainer.closed
    with pytest.raises(RuntimeError, match="closed"):
        trainer.tokenize("again")


def test_pyo3_resolves_a_base_set_and_round_trips_its_bundle(tmp_path: Path) -> None:
    native = pytest.importorskip("retrograd._native")
    trainer = native._Trainer(
        str(_tiny_fixture()),
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        trainable_policy="partial",
        trainable_layers="last:2",
        trainable_modules=["attn_q"],
    )
    try:
        policy, entries, _ = trainer.trainable_set
        assert policy == "partial"
        assert [name for name, *_ in entries] == [
            "blk.2.attn_q.weight",
            "blk.3.attn_q.weight",
        ]
        assert all(dtype == "F32" for _, _, dtype, *_ in entries)
        assert all(shape[:2] == [64, 64] for *_, shape, _, _ in entries)

        bundle = tmp_path / "bundle.gguf"
        trainer.save_trainable(str(bundle))
        assert bundle.is_file()
        trainer.load_trainable(str(bundle))
        model = tmp_path / "model.gguf"
        trainer.save_model(str(model))
        assert model.is_file()
    finally:
        trainer.close()


#: Long enough that the byte-fallback tokenizer produces several rows.
_CONTEXT = 256

_CORPUS = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump! "
    "Sphinx of black quartz, judge my vow. "
) * 8


def _base_trainer(native, model: Path, **overrides: object):
    """A base-weight run over two `attn_q` matrices."""

    keywords: dict[str, object] = {
        "n_ctx": _CONTEXT,
        "n_batch": _CONTEXT,
        "n_ubatch": 64,
        "trainable_policy": "partial",
        "trainable_layers": "last:2",
        "trainable_modules": ["attn_q"],
        "shuffle": False,
    }
    keywords.update(overrides)
    return native._Trainer(str(model), **keywords)


def test_pyo3_checkpoints_a_base_run_and_resumes_it_exactly(tmp_path: Path) -> None:
    """Checkpoint a base-weight run and resume it in a second trainer.

    The assertion is equality, not a tolerance: a checkpoint that dropped the
    moments would still score close, but not exactly.
    """

    native = pytest.importorskip("retrograd._native")
    model = _tiny_fixture()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    state = tmp_path / "state"

    trainer = _base_trainer(native, model)
    try:
        dataset = trainer.prepare_dataset(str(corpus), "text", _CONTEXT)
        metrics = trainer.fit_sft(dataset, None, None)
        global_step = metrics[2]
        # The optimizer graph exists by now, so its state is part of the cost.
        adapter_bytes, trainable_bytes, optimizer_bytes = trainer.checkpoint_footprint()
        assert adapter_bytes == 0, "a partial run publishes no adapter"
        assert trainable_bytes > 0
        assert optimizer_bytes > 0, "AdamW keeps two moments per parameter"

        trainer.save_checkpoint(
            str(state),
            dataset,
            checkpoint_id=f"step-{global_step:012d}",
            algorithm="sft",
            global_step=global_step,
            epoch=1,
            seeds={"shuffle": 42},
        )
        expected = trainer.score(trainer.tokenize("hello world"))
        expected_step = global_step
    finally:
        trainer.close()

    resumed = _base_trainer(native, model)
    try:
        dataset = resumed.prepare_dataset(str(corpus), "text", _CONTEXT)
        # Cold, before the restore: must score differently, or the equality
        # below would hold for a checkpoint that restored nothing.
        cold = resumed.score(resumed.tokenize("hello world"))
        info = resumed.load_checkpoint(str(state), dataset, algorithm="sft")
        (
            global_step,
            epoch,
            _cursor,
            seeds,
            adapter,
            trainable,
            had_graph,
            slots,
        ) = info
        assert (global_step, epoch) == (expected_step, 1)
        assert dict(seeds) == {"shuffle": 42}
        assert adapter is None, "a partial run has no adapter to restore"
        assert trainable is not None
        assert had_graph and slots > 0

        assert resumed.score(resumed.tokenize("hello world")) == expected
        assert cold != expected
    finally:
        resumed.close()


def test_pyo3_refuses_a_checkpoint_taken_against_another_trajectory(tmp_path: Path) -> None:
    """A resume compares the run, not only the model."""

    native = pytest.importorskip("retrograd._native")
    model = _tiny_fixture()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    state = tmp_path / "state"

    trainer = _base_trainer(native, model)
    try:
        dataset = trainer.prepare_dataset(str(corpus), "text", _CONTEXT)
        metrics = trainer.fit_sft(dataset, None, None)
        global_step = metrics[2]
        trainer.save_checkpoint(
            str(state),
            dataset,
            checkpoint_id=f"step-{global_step:012d}",
            algorithm="sft",
            global_step=global_step,
            epoch=1,
        )
    finally:
        trainer.close()

    other = _base_trainer(native, model, learning_rate=5.0e-4)
    try:
        dataset = other.prepare_dataset(str(corpus), "text", _CONTEXT)
        with pytest.raises((ValueError, RuntimeError)):
            other.load_checkpoint(str(state), dataset, algorithm="sft")
    finally:
        other.close()

    # The algorithm's own settings ride in `trajectory_extra`.
    same = _base_trainer(native, model)
    try:
        dataset = same.prepare_dataset(str(corpus), "text", _CONTEXT)
        with pytest.raises((ValueError, RuntimeError)):
            same.load_checkpoint(
                str(state), dataset, algorithm="sft", trajectory_extra="group_size=8"
            )
    finally:
        same.close()


def test_pyo3_carries_an_optimizer_section_into_the_run(tmp_path: Path) -> None:
    """A Python caller can choose the gefen variant, and the checkpoint pins
    it: a `quantized_m` state does not restore into a `shared_v` run.
    """

    native = pytest.importorskip("retrograd._native")
    model = _tiny_fixture()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(_CORPUS, encoding="utf-8")
    state = tmp_path / "state"
    section = {"variant": "quantized_m", "block_size": 512, "min_numel": 1}

    trainer = _base_trainer(native, model, optimizer="gefen", gefen=section)
    try:
        dataset = trainer.prepare_dataset(str(corpus), "text", _CONTEXT)
        metrics = trainer.fit_sft(dataset, None, None)
        global_step = metrics[2]
        trainer.save_checkpoint(
            str(state),
            dataset,
            checkpoint_id=f"step-{global_step:012d}",
            algorithm="sft",
            global_step=global_step,
            epoch=1,
        )
    finally:
        trainer.close()

    other = _base_trainer(native, model, optimizer="gefen")
    try:
        dataset = other.prepare_dataset(str(corpus), "text", _CONTEXT)
        with pytest.raises((ValueError, RuntimeError)):
            other.load_checkpoint(str(state), dataset, algorithm="sft")
    finally:
        other.close()

    same = _base_trainer(native, model, optimizer="gefen", gefen=section)
    try:
        dataset = same.prepare_dataset(str(corpus), "text", _CONTEXT)
        info = same.load_checkpoint(str(state), dataset, algorithm="sft")
        assert info[6] and info[7] > 0
    finally:
        same.close()


def test_pyo3_refuses_a_section_for_an_optimizer_the_run_did_not_choose() -> None:
    native = pytest.importorskip("retrograd._native")
    with pytest.raises(ValueError, match="does not use"):
        native._Trainer("missing.gguf", optimizer="adamw", muon={"momentum": 0.9})
    with pytest.raises(ValueError, match="power of two"):
        native._Trainer("missing.gguf", optimizer="gefen", gefen={"block_size": 3})
    with pytest.raises(ValueError, match="must be 256"):
        native._Trainer("missing.gguf", optimizer="gefen", gefen={"codebook_levels": 64})
    with pytest.raises(ValueError, match="must be 'uniform'"):
        native._Trainer("missing.gguf", optimizer="gefen", gefen={"codebook": "learned"})
    with pytest.raises(ValueError, match=r"gefen\.block_size must be"):
        native._Trainer("missing.gguf", optimizer="gefen", gefen={"block_size": "wide"})


def test_pyo3_refuses_a_base_run_reference_it_was_never_given() -> None:
    native = pytest.importorskip("retrograd._native")
    trainer = native._Trainer(
        str(_tiny_fixture()),
        n_ctx=32,
        n_batch=32,
        n_ubatch=32,
        trainable_policy="full",
    )
    try:
        assert trainer.reference_path is None
        with pytest.raises(ValueError, match="separate frozen model"):
            trainer.score_reference(trainer.tokenize("hello"))
        trainer.attach_reference(str(_tiny_fixture()))
        assert trainer.reference_path == str(_tiny_fixture())
        tokens = trainer.tokenize("hello")
        assert len(trainer.score_reference(tokens)) == len(tokens) - 1
    finally:
        trainer.close()


#: One table, fed to both layer-range parsers. Validating ahead of the runtime
#: is only worth it while both accept the same strings: one that refuses here
#: and passes there makes the Python check decoration, and one that passes here
#: and fails there reports the wrong reason for the failure.
LAYER_RANGES = [
    ("all", True),
    ("ALL", True),
    (" all ", True),
    ("last:2", True),
    ("last:+2", True),
    ("00..02", True),
    ("0..4294967295", True),
    ("12 .. 15", True),
    ("", False),
    ("most", False),
    ("LAST:4", False),
    ("last:", False),
    ("last:0", False),
    ("4..2", False),
    ("1..abc", False),
    ("1..4294967296", False),
    ("last:99999999999999999999", False),
    # `str.isdigit` is true of both and `u32::from_str` accepts neither.
    ("\u0663..4", False),
    ("\u00b2..4", False),
    ("-1..2", False),
    ("1..2..3", False),
]


@pytest.mark.parametrize(("layers", "accepted"), LAYER_RANGES)
def test_both_layer_range_parsers_read_the_same_language(layers: str, accepted: bool) -> None:
    from retrograd.config import _validate_layer_range

    native = pytest.importorskip("retrograd._native")

    python_accepted = True
    try:
        _validate_layer_range(layers)
    except ValueError:
        python_accepted = False
    assert python_accepted is accepted

    # The native parser runs before the model is opened, so a missing file
    # separates the two outcomes without a fixture: a range it refuses raises
    # ValueError from the parser, and one it accepts gets as far as the load
    # and raises RetrogradNativeError there.
    with pytest.raises(Exception) as raised:
        native._Trainer(
            "missing.gguf",
            trainable_policy="partial",
            trainable_modules=["attn"],
            trainable_layers=layers,
        )
    native_accepted = isinstance(raised.value, native.RetrogradNativeError)
    assert native_accepted is accepted, str(raised.value)
