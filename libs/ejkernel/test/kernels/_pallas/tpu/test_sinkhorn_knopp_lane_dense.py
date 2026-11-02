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

"""Lane-dense Sinkhorn-Knopp (Pallas TPU): bit parity with the tiled kernel, independent references, gradients.

Small fp32 matrices take a lane-dense kernel that stores entry ``(i, j)`` of every matrix as one ``[rows, 128]``
slab, so its row and column sums are elementwise adds in an explicit order. Training shares the op, so its forward
must stay bit-identical to the tiled kernel it replaced (the tiled kernel still serves larger and non-fp32 matrices)
and the backward must stay today's rule. The independent anchors are a float64 NumPy loop, the column normalisation
the trailing step guarantees, row convergence no worse than the exact recurrence, and the gradient of a test-local
XLA loop.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ejkernel.kernels._pallas.tpu.sinkhorn_knopp import _pallas_impl_fwd as impl

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU kernels need a TPU")

ITERS = 20
EPS = 1e-6

# Decode rows per forward (one block, one slab row), the prefill bucket (two 8-row blocks), a training-like batch
# (four blocks), padded tails inside one block and across blocks, ragged leading dims, and matrix aspects up to 16
# entries whose summation trees differ (odd, non-power-of-two and single-row/column widths).
LANE_DENSE_SHAPES = [
    (1, 128, 4, 4),
    (1, 2048, 4, 4),
    (8, 512, 4, 4),
    (1, 1023, 4, 4),
    (1, 1100, 4, 4),
    (7, 5, 4, 4),
    (2, 3, 3, 3),
    (1, 300, 2, 8),
    (1, 300, 8, 2),
    (1, 300, 3, 5),
    (1, 300, 16, 1),
    (1, 300, 1, 16),
]


def _comb(shape, seed):
    """Model-like input: a softmax over the last axis plus ``eps``, as the mHC mixer feeds Sinkhorn."""
    logits = jax.random.normal(jax.random.key(seed), shape, jnp.float32) * 2.0
    return jax.nn.softmax(logits, axis=-1) + EPS


def _float64_sinkhorn(matrix, n_iters=ITERS, eps=EPS):
    """Independent float64 NumPy loop: column normalisation, then ``n_iters - 1`` (row, column) pairs."""
    m = np.asarray(matrix, np.float64)
    m = m / (m.sum(axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        m = m / (m.sum(axis=-1, keepdims=True) + eps)
        m = m / (m.sum(axis=-2, keepdims=True) + eps)
    return m


def _xla_sinkhorn(matrix, n_iters=ITERS, eps=EPS):
    """Test-local XLA loop, differentiated for the gradient reference."""
    matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        matrix = matrix / (jnp.sum(matrix, axis=-1, keepdims=True) + eps)
        matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    return matrix


@functools.partial(jax.custom_vjp, nondiff_argnums=(1, 2))
def _tiled_rule(matrix, n_iters, eps):
    """The op as it was before the lane-dense kernel: tiled Pallas forward, VJP of the XLA loop as the backward."""
    return impl._sinkhorn_pallas_call(matrix, n_iters, eps)


def _tiled_rule_fwd(matrix, n_iters, eps):
    return impl._sinkhorn_pallas_call(matrix, n_iters, eps), (matrix,)


def _tiled_rule_bwd(n_iters, eps, res, cotangent):
    (matrix,) = res
    return jax.vjp(lambda m: _xla_sinkhorn(m, n_iters, eps), matrix)[1](cotangent)


_tiled_rule.defvjp(_tiled_rule_fwd, _tiled_rule_bwd)


def _bits(x):
    return np.asarray(x, np.float32).view(np.int32)


@pytest.mark.parametrize("shape", LANE_DENSE_SHAPES)
def test_lane_dense_forward_is_bit_identical_to_tiled_kernel(shape):
    """Training shares this op: the lane-dense kernel and the registered entry reproduce the tiled kernel exactly."""
    m = _comb(shape, seed=sum(shape))
    tiled = jax.jit(lambda x: impl._sinkhorn_pallas_call(x, ITERS, EPS))(m)
    dense = jax.jit(lambda x: impl._sinkhorn_slab_call(x, ITERS, EPS))(m)
    entry = impl.sinkhorn_knopp_tpu(m, n_iters=ITERS, eps=EPS)
    for out in (dense, entry):
        assert out.shape == m.shape
        assert out.dtype == m.dtype
        np.testing.assert_array_equal(_bits(out), _bits(tiled))


@pytest.mark.parametrize("shape", [(1, 128, 4, 4), (1, 2048, 4, 4), (1, 1100, 4, 4), (2, 3, 3, 3), (1, 300, 2, 8)])
def test_forward_matches_float64_reference_and_is_doubly_stochastic(shape):
    """Independent anchors: a float64 loop and, for square mixers, the doubly-stochastic property itself."""
    m = _comb(shape, seed=7 + sum(shape))
    got = np.asarray(impl.sinkhorn_knopp_tpu(m, n_iters=ITERS, eps=EPS), np.float64)
    want = _float64_sinkhorn(np.asarray(m))
    assert np.all(np.isfinite(got))
    delta = float(np.max(np.abs(got - want)))
    assert delta < 1e-5, f"{shape}: max|delta| vs float64 = {delta:.3e}"
    if shape[-1] == shape[-2]:
        # The trailing step normalises columns, so they sum to one up to eps. Rows converge rather than being exact
        # and must converge as far as the exact recurrence does.
        np.testing.assert_allclose(got.sum(axis=-2), 1.0, atol=1e-5)
        row_err = float(np.max(np.abs(got.sum(axis=-1) - 1.0)))
        assert row_err <= float(np.max(np.abs(want.sum(axis=-1) - 1.0))) + 1e-5


@pytest.mark.parametrize("shape", [(1, 2048, 4, 4), (8, 512, 4, 4), (2, 3, 3, 3)])
def test_gradient_is_todays_rule_and_matches_xla_reference(shape):
    """The backward is unchanged: bit-identical to the tiled rule, and within fp32 rounding of the XLA loop's grad."""
    m = _comb(shape, seed=3)
    w = jnp.linspace(-0.7, 1.3, m.size, dtype=jnp.float32).reshape(shape)

    def grad_of(fn):
        return np.asarray(jax.jit(jax.grad(lambda x: jnp.sum(fn(x) * w)))(m))

    got = grad_of(lambda x: impl.sinkhorn_knopp_tpu(x, n_iters=ITERS, eps=EPS))
    today = grad_of(lambda x: _tiled_rule(x, ITERS, EPS))
    xla = grad_of(_xla_sinkhorn)
    assert np.all(np.isfinite(got))
    np.testing.assert_array_equal(_bits(got), _bits(today))
    scale = max(1.0, float(np.max(np.abs(xla))))
    delta = float(np.max(np.abs(got.astype(np.float64) - xla.astype(np.float64))))
    assert delta < 1e-4 * scale, f"{shape}: gradient max|delta| vs XLA = {delta:.3e} (scale {scale:.3e})"
