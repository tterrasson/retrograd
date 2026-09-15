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


def test_pyo3_validates_arguments_before_loading_a_model() -> None:
    native = pytest.importorskip("retrograd._native")
    with pytest.raises(ValueError, match="scheduler"):
        native._Trainer("missing.gguf", scheduler="invalid")
    with pytest.raises(ValueError, match="device"):
        native._Trainer("missing.gguf", device="invalid")


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
