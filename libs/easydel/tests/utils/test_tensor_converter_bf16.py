# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression test: the NumPy bridge of ``TensorConverter.jax_to_pytorch`` must
handle bfloat16.

``torch.from_numpy`` cannot ingest ``ml_dtypes.bfloat16`` arrays (NumPy has no
native bfloat16), so the EASY_SAFE_TRANSFER path raised
``TypeError: can't convert np.ndarray of type ml_dtypes.bfloat16`` for any
bf16 model export (``module.to_torch`` / ``save_pretrained(to_torch=True)``).
The fix reinterprets the bits as uint16 and views them back as torch bfloat16,
which must be bit-exact.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.utils.parameters_transformation import TensorConverter

torch = pytest.importorskip("torch")


def _via_numpy_bridge(monkeypatch, array):
    monkeypatch.setenv("EASY_SAFE_TRANSFER", "1")
    return TensorConverter.jax_to_pytorch(array)


def test_bf16_numpy_bridge_converts(monkeypatch):
    x = jnp.linspace(-4.0, 4.0, 256, dtype=jnp.float32).reshape(16, 16).astype(jnp.bfloat16)
    t = _via_numpy_bridge(monkeypatch, x)
    assert t.dtype == torch.bfloat16
    assert tuple(t.shape) == (16, 16)


def test_bf16_numpy_bridge_is_bit_exact(monkeypatch):
    x = (jnp.arange(1024, dtype=jnp.float32) * 0.37 - 200.0).astype(jnp.bfloat16)
    t = _via_numpy_bridge(monkeypatch, x)
    # Compare raw bit patterns: the uint16 payloads must match exactly.
    jax_bits = np.asarray(x).view(np.uint16)
    torch_bits = t.view(torch.uint16).numpy()
    assert np.array_equal(jax_bits, torch_bits)


def test_non_bf16_dtypes_unaffected(monkeypatch):
    for dtype, torch_dtype in [(jnp.float32, torch.float32), (jnp.int32, torch.int32)]:
        x = jnp.ones((4, 4), dtype=dtype)
        t = _via_numpy_bridge(monkeypatch, x)
        assert t.dtype == torch_dtype
