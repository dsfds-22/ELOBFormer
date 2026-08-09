"""Compatibility support for MMSeg models that do not use MMCV custom ops."""

from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec


def install_mmcv_lite_ext_stub() -> None:
    """Allow MMSeg registry discovery when only mmcv-lite is installed."""

    try:
        import mmcv._ext  # type: ignore[import-not-found]  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class ExtStub(types.ModuleType):
        def __getattr__(self, name: str):
            def unavailable(*args, **kwargs):
                raise RuntimeError(
                    f"MMCV compiled operator '{name}' is unavailable in "
                    "mmcv-lite."
                )

            return unavailable

    stub = ExtStub("mmcv._ext")
    stub.__file__ = os.path.abspath("mmcv_ext_stub.py")
    stub.__path__ = []
    stub.__spec__ = ModuleSpec("mmcv._ext", loader=None)
    sys.modules["mmcv._ext"] = stub
