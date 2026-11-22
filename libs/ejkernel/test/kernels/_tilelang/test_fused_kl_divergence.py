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

"""TileLang parity tests for fused_kl_divergence."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._helpers import _FP16_BWD_TOL, _FP16_FWD_TOL, _SEED, _max_abs, _randn, _tl, _xla


def test_fused_kl_divergence_forward_fwd_bwd():
    rows, vocab = 4, 64
    key = jax.random.PRNGKey(_SEED + 42)
    k_student, k_teacher = jax.random.split(key)
    student = _randn(k_student, (rows, vocab), scale=0.2)
    teacher = _randn(k_teacher, (rows, vocab), scale=0.2)
    weights = jnp.array([1.0, 1.0, 0.0, 1.0], dtype=jnp.float32)
    kwargs = {"reduction": "mean", "direction": "forward", "block_v": 64, "block_m": 1}
    tl, xla = _tl("fused_kl_divergence"), _xla("fused_kl_divergence")

    loss_tl = tl(student, teacher, weights, **kwargs)
    loss_xla = xla(student, teacher, weights, **kwargs)
    assert _max_abs(loss_tl, loss_xla) < _FP16_FWD_TOL

    g_tl = jax.grad(lambda x: tl(x, teacher, weights, **kwargs))(student)
    g_xla = jax.grad(lambda x: xla(x, teacher, weights, **kwargs))(student)
    assert _max_abs(g_tl, g_xla) < _FP16_BWD_TOL
