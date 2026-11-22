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

"""Sinkhorn-Knopp: every backend must reproduce the XLA reference.

The matrices are tiny -- DeepSeek-V4's hyper-connections normalise a 4x4 -- so
the arithmetic is free and the whole cost is dispatch: unrolled, 20 iterations
become ~117 device ops, and every reduction is a fusion boundary XLA cannot
cross. Measured on v5p-8, that projection cost 6.45 ms of a 35.35 ms decode step
(18%), and the fused kernel returned +19.1% end to end.

Three things are pinned here, each of which broke at least once while the kernel
was being built:

* forward parity with the reference, across decode AND prefill-length shapes --
  Mosaic pads the two trailing dims to a full (8, 128) tile, so a 4x4 occupies
  4 KB and long inputs blew VMEM until the kernel gained a grid;
* GRADIENT parity -- a Mosaic call carries no VJP, so the fused path needs a
  custom rule or anything differentiating through it fails at linearisation;
* the doubly-stochastic property itself, so the tests cannot all pass by two
  implementations being wrong together.
"""

import jax
import numpy as np
import pytest
from ejkernel.modules import sinkhorn_knopp
from jax import numpy as jnp

ITERS = 20
EPS = 1e-6


def _reference(matrix, n_iters=ITERS, eps=EPS):
    """The unrolled loop the kernel replaces."""
    matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    for _ in range(n_iters - 1):
        matrix = matrix / (jnp.sum(matrix, axis=-1, keepdims=True) + eps)
        matrix = matrix / (jnp.sum(matrix, axis=-2, keepdims=True) + eps)
    return matrix


def _matrix(shape, seed=0):
    # Strictly positive, matching the real input (a softmax output plus eps).
    return jax.random.uniform(jax.random.key(seed), shape, jnp.float32, 0.05, 1.0)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 1, 4, 4),  # smallest
        (32, 1, 4, 4),  # decode at the served concurrency
        (128, 1, 4, 4),
        (1, 2048, 4, 4),  # prefill length -- OOM'd before the kernel was blocked
        (4, 512, 4, 4),
        (1, 1023, 4, 4),  # not a multiple of the block
        (7, 5, 4, 4),  # ragged leading dims
        (2, 3, 8, 8),  # larger matrices
    ],
)
def test_matches_reference(shape):
    """Forward parity is the contract every backend must meet."""
    m = _matrix(shape, seed=sum(shape))
    got = np.asarray(sinkhorn_knopp(m), np.float64)
    want = np.asarray(_reference(m), np.float64)
    assert np.all(np.isfinite(got))
    delta = float(np.max(np.abs(got - want)))
    assert delta < 1e-5, f"{shape}: diverged from reference, max|delta|={delta:.3e}"


def test_result_is_doubly_stochastic():
    """Anchors both implementations against the property, not each other."""
    out = np.asarray(sinkhorn_knopp(_matrix((4, 1, 4, 4))), np.float64)
    # The trailing step normalises columns, so those are exact; rows converge.
    np.testing.assert_allclose(out.sum(axis=-2), 1.0, atol=1e-5)
    np.testing.assert_allclose(out.sum(axis=-1), 1.0, atol=1e-3)


def test_gradient_matches_reference():
    """A Mosaic call has no VJP; without a custom rule this fails outright.

    Caught for real: routing the kernel through the executor raised
    "Linearization failed to produce known values for all output primals"
    until the fused path got a custom_vjp that differentiates the reference.
    """
    m = _matrix((8, 1, 4, 4), seed=3)
    g_fused = jax.grad(lambda x: jnp.sum(sinkhorn_knopp(x) ** 2))(m)
    g_ref = jax.grad(lambda x: jnp.sum(_reference(x) ** 2))(m)
    assert np.all(np.isfinite(g_fused))
    delta = float(np.max(np.abs(np.asarray(g_fused, np.float64) - np.asarray(g_ref, np.float64))))
    assert delta < 1e-4, f"gradient diverged from reference, max|delta|={delta:.3e}"


def test_explicit_platforms_agree():
    """The XLA fallback and the fused path must not drift apart."""
    m = _matrix((16, 1, 4, 4), seed=5)
    xla = np.asarray(sinkhorn_knopp(m, platform="xla"), np.float64)
    auto = np.asarray(sinkhorn_knopp(m), np.float64)
    assert float(np.max(np.abs(xla - auto))) < 1e-5
