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
