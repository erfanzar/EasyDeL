# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference-mode helpers for Python and JAX-traced code.

The active mode is stored in a ``ContextVar`` so nested ``with`` blocks behave
correctly in regular Python execution. When available, the same state is also
mirrored into JAX user context so switching modes invalidates JIT cache keys
and traced code can safely branch on ``is_inference_mode()``.
"""

from __future__ import annotations

import contextlib
import contextvars
import typing as tp
from collections.abc import Iterator

import jax
from jax._src.lib import xla_client

config_ext = xla_client._xla.config

_inference_mode_var: contextvars.ContextVar[bool] = contextvars.ContextVar("easydeL_inference_mode", default=False)

_JAX_INFERENCE_MODE: tp.Any = None
try:
    _JAX_INFERENCE_MODE = jax.make_user_context(default_value=False)
except AttributeError:  # pragma: no cover - older JAX fallback
    pass

_INFERENCE_MODE_STATE: tp.Any = getattr(_JAX_INFERENCE_MODE, "_obj", None) if _JAX_INFERENCE_MODE is not None else None
if _INFERENCE_MODE_STATE is None:
    _INFERENCE_MODE_STATE = config_ext.Config("easydeL_inference_mode_state", False, include_in_jit_key=True)


def is_inference_mode() -> bool:
    """Return ``True`` when execution is inside ``set_inference_mode()``."""

    return _inference_mode_var.get()


@contextlib.contextmanager
def set_inference_mode(enabled: bool = True) -> Iterator[bool]:
    """Temporarily mark execution as inference mode.

    Args:
        enabled: Whether inference mode should be active inside the block.

    Yields:
        The ``enabled`` value for convenience.
    """

    if not isinstance(enabled, bool):
        raise TypeError("set_inference_mode expects a boolean `enabled` argument.")

    token = _inference_mode_var.set(enabled)
    previous_state = _INFERENCE_MODE_STATE.swap_local(enabled) if _INFERENCE_MODE_STATE else None
    try:
        yield enabled
    finally:
        _inference_mode_var.reset(token)
        if _INFERENCE_MODE_STATE:
            _INFERENCE_MODE_STATE.set_local(previous_state)


__all__ = ("is_inference_mode", "set_inference_mode")
