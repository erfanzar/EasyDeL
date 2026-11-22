# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""``AttentionModule._softmax_aux`` resolves attention-sink logits.

Layers spell the attribute either ``sinks`` or ``softmax_aux``, and it may be a
bare array or a parameter wrapper exposing ``.value``. The four-way probe used
to be copy-pasted at twelve call sites; these tests pin the behaviour of the
single shared implementation against every shape the old idiom handled.
"""

import pytest
from easydel.layers.attention._flexible import AttentionModule
from jax import numpy as jnp


class _Wrapper:
    """Stands in for a parameter object exposing ``.value``."""

    def __init__(self, value):
        self.value = value


class _Bare(AttentionModule):
    """Attention layer that declares no sinks."""

    def __init__(self):
        pass


class _WithSinks(_Bare):
    def __init__(self, sinks):
        self.sinks = sinks


class _WithSoftmaxAux(_Bare):
    def __init__(self, softmax_aux):
        self.softmax_aux = softmax_aux


def test_returns_none_when_layer_has_no_sinks():
    assert _Bare()._softmax_aux() is None


def test_reads_bare_sinks_array():
    arr = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(_WithSinks(arr)._softmax_aux(), arr)


def test_unwraps_parameter_wrapper_on_sinks():
    arr = jnp.arange(4, dtype=jnp.float32)
    assert jnp.array_equal(_WithSinks(_Wrapper(arr))._softmax_aux(), arr)


def test_falls_back_to_softmax_aux_attribute():
    arr = jnp.arange(3, dtype=jnp.float32)
    assert jnp.array_equal(_WithSoftmaxAux(arr)._softmax_aux(), arr)


def test_unwraps_parameter_wrapper_on_softmax_aux():
    arr = jnp.arange(3, dtype=jnp.float32)
    assert jnp.array_equal(_WithSoftmaxAux(_Wrapper(arr))._softmax_aux(), arr)


def test_sinks_wins_over_softmax_aux():
    """``sinks`` is probed first, matching the original nested-getattr order."""
    layer = _WithSinks(jnp.ones((2,), jnp.float32))
    layer.softmax_aux = jnp.zeros((2,), jnp.float32)
    assert jnp.array_equal(layer._softmax_aux(), jnp.ones((2,), jnp.float32))


@pytest.mark.parametrize("value", [None, jnp.zeros((2,), jnp.float32)])
def test_matches_the_replaced_idiom(value):
    """Equivalence against the exact expression the twelve call sites used."""
    layer = _WithSinks(_Wrapper(value)) if value is not None else _Bare()

    legacy = getattr(layer, "sinks", getattr(layer, "softmax_aux", None))
    legacy = getattr(legacy, "value", legacy)

    got = layer._softmax_aux()
    if legacy is None:
        assert got is None
    else:
        assert jnp.array_equal(got, legacy)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
