# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""TileLang parity tests for ragged_decode_attention."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ejkernel.ops import FwdParams

from ._helpers import (
    _SEED,
    _max_abs,
    _randn,
    _tl,
    _xla,
)


def test_ragged_decode_attention_ranges_gqa():
    B, L, H, HKV, D = 3, 17, 6, 2, 32
    key = jax.random.PRNGKey(_SEED)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    k = _randn(k2, (B, L, HKV, D))
    v = _randn(k3, (B, L, HKV, D))
    starts = jnp.array([2, 0, 5], dtype=jnp.int32)
    ends = jnp.array([13, 9, 17], dtype=jnp.int32)
    kwargs = {"fwd_params": FwdParams(kv_blocksize=8, num_stages=2, num_warps=4)}
    out_tl = _tl("ragged_decode_attention")(q, k, v, starts, ends, **kwargs)
    out_xla = _xla("ragged_decode_attention")(q, k, v, starts, ends, **kwargs)
    assert _max_abs(out_tl, out_xla) < 2e-2


def test_ragged_decode_attention_window_softcap_sinks():
    B, L, H, HKV, D = 2, 21, 6, 2, 32
    key = jax.random.PRNGKey(_SEED + 1)
    k1, k2, k3 = jax.random.split(key, 3)
    q = _randn(k1, (B, H, D))
    k = _randn(k2, (B, L, HKV, D))
    v = _randn(k3, (B, L, HKV, D))
    starts = jnp.array([3, 1], dtype=jnp.int32)
    ends = jnp.array([19, 14], dtype=jnp.int32)
    softmax_aux = jnp.linspace(-0.2, 0.3, H * 3, dtype=jnp.float32).reshape(H, 3)
    kwargs = {
        "softmax_scale": 1.0 / math.sqrt(D),
        "fwd_params": FwdParams(kv_blocksize=8),
        "sliding_window": (6, 0),
        "logits_soft_cap": 3.0,
        "softmax_aux": softmax_aux,
    }
    out_tl = _tl("ragged_decode_attention")(q, k, v, starts, ends, **kwargs)
    out_xla = _xla("ragged_decode_attention")(q, k, v, starts, ends, **kwargs)
    assert _max_abs(out_tl, out_xla) < 5e-2
