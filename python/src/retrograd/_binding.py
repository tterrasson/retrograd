from __future__ import annotations

from types import ModuleType


def load_native() -> ModuleType:
    try:
        from . import _native
    except ImportError as error:  # pragma: no cover - depends on installation state
        raise RuntimeError(
            "the Retrograd native extension is not installed; run `uv sync` in the "
            "python/ directory (or `uv run maturin develop`)"
        ) from error
    return _native
