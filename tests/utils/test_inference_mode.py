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

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from easydel import is_inference_mode, set_inference_mode


def test_inference_mode_roundtrip():
    assert not is_inference_mode()

    with set_inference_mode() as enabled:
        assert enabled is True
        assert is_inference_mode()

        with set_inference_mode(False) as disabled:
            assert disabled is False
            assert not is_inference_mode()

        assert is_inference_mode()

    assert not is_inference_mode()


@jax.jit
def _mode_sensitive_add(x):
    return x + (1 if is_inference_mode() else 0)


def test_jax_jit_reads_inference_mode():
    assert int(_mode_sensitive_add(jnp.array(0))) == 0

    with set_inference_mode():
        assert int(_mode_sensitive_add(jnp.array(0))) == 1

    assert int(_mode_sensitive_add(jnp.array(0))) == 0


def test_non_bool_inference_mode_rejected():
    with pytest.raises(TypeError):
        with set_inference_mode("yes"):  # pyright: ignore[reportArgumentType]
            pass
