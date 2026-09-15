from __future__ import annotations

import sys

import pytest

import retrograd
from retrograd import RetrogradError


class NativeError(RuntimeError):
    pass


class Native:
    RetrogradNativeError = NativeError

    @staticmethod
    def list_backends() -> str:
        return "cpu\tCPU\tportable\ngpu\tMetal\taccelerated\n"


def test_list_backends_parses_the_native_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retrograd, "load_native", lambda: Native)
    assert retrograd.list_backends() == (
        retrograd.Backend("cpu", "CPU", "portable"),
        retrograd.Backend("gpu", "Metal", "accelerated"),
    )


def test_list_backends_translates_native_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingNative(Native):
        @staticmethod
        def list_backends() -> str:
            raise NativeError("backend unavailable")

    monkeypatch.setattr(retrograd, "load_native", lambda: FailingNative)
    with pytest.raises(RetrogradError, match="backend unavailable"):
        retrograd.list_backends()


def test_missing_native_extension_has_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from retrograd import _binding

    monkeypatch.delattr(retrograd, "_native", raising=False)
    monkeypatch.setitem(sys.modules, "retrograd._native", None)
    with pytest.raises(RuntimeError, match="native extension is not installed"):
        _binding.load_native()
