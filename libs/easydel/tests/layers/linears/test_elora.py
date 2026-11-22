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

"""Regression tests for the ``eLoRA`` EasyDeL calling-convention wrapper.

``eLoRA`` re-implements spectrax ``nn.LoRA.forward`` to forward extra
``*args`` / ``**kwargs`` (tied-projection ``w=...``) and to expose
``native_forward``. The re-implementation must preserve the base class's
numerical contract: the delta is ``alpha / rank`` scaled when ``alpha`` is
set, and a base-module-free adapter returns the delta alone.
"""

import numpy as np
import spectrax as spx
from easydel.layers import ColumnParallelLinear, eLoRA
from jax import numpy as jnp
from spectrax import nn


def _x():
    return jnp.asarray(np.random.default_rng(0).standard_normal((3, 8)), jnp.float32)


def test_alpha_scaling_matches_spectrax_lora():
    """With ``alpha`` set, ``eLoRA`` must scale the delta by ``alpha / rank``.

    A nonzero ``b_init`` makes the delta nonzero so the scaling is observable;
    same seeds keep both wrappers' factors identical.
    """
    kwargs = dict(rank=2, alpha=8.0, b_init=spx.init.ones, rngs=spx.Rngs(0))
    wrapped = eLoRA(8, d_out=4, **kwargs)
    reference = nn.LoRA(8, 2, 4, alpha=8.0, b_init=spx.init.ones, rngs=spx.Rngs(0))

    x = _x()
    np.testing.assert_allclose(np.asarray(wrapped(x)), np.asarray(reference(x)), rtol=1e-6)

    # And the explicit closed form: (x @ A @ B) * (alpha / rank).
    expected = (x @ wrapped.lora_a[...] @ wrapped.lora_b[...]) * (8.0 / 2)
    np.testing.assert_allclose(np.asarray(wrapped(x)), np.asarray(expected), rtol=1e-6)


def test_native_forward_applies_alpha_scaling():
    """``native_forward`` (chunked LM-head path) must match ``forward``."""
    wrapped = eLoRA(8, 2, 4, alpha=16.0, b_init=spx.init.ones, rngs=spx.Rngs(0))
    x = _x()
    np.testing.assert_allclose(np.asarray(wrapped.native_forward(x)), np.asarray(wrapped(x)), rtol=1e-6)


def test_no_base_module_returns_delta_without_error():
    """A base-module-free adapter is a supported shape (additive adapter)."""
    wrapped = eLoRA(8, 2, 4, b_init=spx.init.ones, rngs=spx.Rngs(0))
    x = _x()
    out = wrapped(x)
    expected = x @ wrapped.lora_a[...] @ wrapped.lora_b[...]
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=1e-6)


def test_base_module_output_added():
    """With a base linear, output is ``base(x) + delta`` and kwargs forward."""
    base = ColumnParallelLinear(8, 4, use_bias=False, rngs=spx.Rngs(1))
    wrapped = eLoRA(8, 2, 4, base_module=base, b_init=spx.init.ones, rngs=spx.Rngs(0))
    x = _x()
    expected = base(x) + (x @ wrapped.lora_a[...] @ wrapped.lora_b[...])
    np.testing.assert_allclose(np.asarray(wrapped(x)), np.asarray(expected), rtol=1e-6)
