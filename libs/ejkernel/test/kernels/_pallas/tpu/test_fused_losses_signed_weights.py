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

"""TPU Pallas CE/KL: forward and backward agree on signed weights; fp32 teachers stay fp32.

The Pallas forwards used ``abs(weight)`` while their analytic backwards (and the XLA path the
autotuner picks between) use the signed weight, so a negative-weight row reported a positive loss
but received the negative-weight gradient. The Pallas KL entry also rounded an fp32 teacher to the
student dtype before the kernel (the XLA entry did the same; both now keep the teacher dtype).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.fused_cross_entropy import fused_cross_entropy as pallas_ce
from ejkernel.kernels._pallas.tpu.fused_kl_divergence import fused_kl_divergence as pallas_kl
from ejkernel.kernels._xla.fused_cross_entropy import fused_cross_entropy as xla_ce
from ejkernel.kernels._xla.fused_kl_divergence import fused_kl_divergence as xla_kl

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU Pallas kernels")


def _signed_weights(rows, seed=0):
    """Mix of positive, negative and zero row weights."""
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=rows), jnp.float32)


def test_pallas_ce_signed_weights_match_xla_value_and_grad():
    rows, vocab = 256, 4096
    logits = (jax.random.normal(jax.random.PRNGKey(0), (rows, vocab)) * 2.0).astype(jnp.float32)
    targets = jax.random.randint(jax.random.PRNGKey(1), (rows,), 0, vocab)
    weights = _signed_weights(rows)

    def loss(fn, x):
        return fn(x, targets, weights, reduction="sum")[0]

    p_val, p_grad = jax.value_and_grad(lambda x: loss(pallas_ce, x))(logits)
    x_val, x_grad = jax.value_and_grad(lambda x: loss(xla_ce, x))(logits)
    np.testing.assert_allclose(float(p_val), float(x_val), rtol=1e-4)
    np.testing.assert_allclose(np.asarray(p_grad), np.asarray(x_grad), atol=1e-5)


def test_pallas_kl_signed_weights_and_fp32_teacher_match_xla():
    rows, vocab = 256, 4096
    student = (jax.random.normal(jax.random.PRNGKey(2), (rows, vocab)) * 2.0).astype(jnp.bfloat16)
    # fp32 teacher whose values are NOT representable in bf16: rounding it to the student dtype
    # changes the target distribution.
    teacher = jax.random.normal(jax.random.PRNGKey(3), (rows, vocab), jnp.float32) * 2.0 + 1e-3
    weights = _signed_weights(rows, seed=1)

    def loss(fn, s):
        return fn(s, teacher, weights, reduction="sum", direction="forward")

    p_val, p_grad = jax.value_and_grad(lambda s: loss(pallas_kl, s))(student)
    x_val, x_grad = jax.value_and_grad(lambda s: loss(xla_kl, s))(student)
    np.testing.assert_allclose(float(p_val), float(x_val), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(np.asarray(p_grad, np.float32), np.asarray(x_grad, np.float32), atol=2e-3)
