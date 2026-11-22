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

"""Regression tests for LoRA weight merging in ``unwrap_lora_to_layers``.

The unwrapped (merged) base weight must reproduce the wrapped module's
forward pass exactly. Since ``eLoRA``/``nn.LoRA`` scale the delta by
``alpha / rank`` whenever ``alpha`` is provided, the merge must apply the
same factor: ``W_merged = W + (alpha / rank) * A @ B``.
"""

import jax
import numpy as np
import spectrax as spx
from easydel.infra.utils import unwrap_lora_to_layers
from easydel.layers.linears._lora import eLoRA
from jax import numpy as jnp
from spectrax import nn

_ALPHA = 6.0
_RANK = 2


class _AdapterHolder(spx.Module):
    """Minimal container exposing one LoRA-wrapped linear as ``.adapter``."""

    def __init__(self, alpha: float | None) -> None:
        super().__init__()
        self.adapter = eLoRA(
            8,
            _RANK,
            4,
            base_module=nn.Linear(8, 4, use_bias=False, rngs=spx.Rngs(1)),
            alpha=alpha,
            b_init=spx.init.ones,
            rngs=spx.Rngs(0),
        )


def _x():
    return jnp.asarray(np.random.default_rng(0).standard_normal((4, 8)), jnp.float32)


def test_unwrap_with_alpha_matches_wrapped_forward():
    """After unwrapping, the bare base layer must equal the alpha-scaled wrapper."""
    model = _AdapterHolder(alpha=_ALPHA)
    x = _x()
    # Compare at fp32 matmul precision: TPU DEFAULT precision is bf16, and the wrapped
    # (W x + s * x A B) and merged ((W + s A B) x) forms round differently there.
    with jax.default_matmul_precision("highest"):
        expected = np.asarray(model.adapter(x))

    w0 = np.asarray(model.adapter.base_module.weight.value)
    a0 = np.asarray(model.adapter.lora_a.value)
    b0 = np.asarray(model.adapter.lora_b.value)

    unwrapped = unwrap_lora_to_layers(model, verbose=False)

    assert isinstance(unwrapped.adapter, nn.Linear)
    merged = np.asarray(unwrapped.adapter.weight.value)
    np.testing.assert_allclose(merged, w0 + (_ALPHA / _RANK) * (a0 @ b0), rtol=1e-5)

    with jax.default_matmul_precision("highest"):
        got = np.asarray(unwrapped.adapter(x))
    np.testing.assert_allclose(got, expected, rtol=1e-5)


def test_unwrap_without_alpha_is_unscaled():
    """Adapters created without ``alpha`` keep the historical ``W + A @ B`` merge."""
    model = _AdapterHolder(alpha=None)
    w0 = np.asarray(model.adapter.base_module.weight.value)
    a0 = np.asarray(model.adapter.lora_a.value)
    b0 = np.asarray(model.adapter.lora_b.value)

    unwrapped = unwrap_lora_to_layers(model, verbose=False)

    merged = np.asarray(unwrapped.adapter.weight.value)
    np.testing.assert_allclose(merged, w0 + a0 @ b0, rtol=1e-5)
