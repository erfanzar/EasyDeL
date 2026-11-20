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

"""The fused MoE combine backward (``_kernel_unsort_combine_bwd``) permutes with key-value sorts.

It moves the routing weights and their gradients by the dispatch permutation and its inverse. A sort keyed on the
other permutation puts every value in the slot a 1-D gather would, so the results must equal the gather form bit for
bit, NaN payloads and signed zeros included.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.layers.moe import _communication_utils as cu

pytestmark = pytest.mark.skipif(jax.default_backend() != "tpu", reason="Pallas TPU row-gather kernels")

BF16, F32 = jnp.bfloat16, jnp.float32
T, K, H = 1024, 8, 2048


def _bits(x) -> np.ndarray:
    return np.asarray(x).view(np.uint32)


def test_permute_by_key_equals_the_gather_by_the_inverse_permutation():
    rng = np.random.default_rng(11)
    rows = T * K
    sort_indices = jnp.asarray(rng.permutation(rows), jnp.int32)
    inverse_indices = cu._inverse_permutation(sort_indices)
    values = jnp.asarray(rng.standard_normal(rows), F32).at[:3].set(jnp.asarray([jnp.nan, -0.0, jnp.inf], F32))
    by_sort = jax.jit(cu._permute_by_key)(inverse_indices, values)
    by_gather = jax.jit(lambda v, i: v[i])(values, sort_indices)
    np.testing.assert_array_equal(_bits(by_sort), _bits(by_gather))


def test_combine_backward_equals_the_gather_form():
    rng = np.random.default_rng(12)
    rows = T * K
    sort_indices = jnp.asarray(rng.permutation(rows), jnp.int32)
    inverse_indices = cu._inverse_permutation(sort_indices)
    intermediate = jnp.asarray(rng.standard_normal((rows, H)), F32).astype(BF16)
    weights = jnp.asarray(rng.random((T, K)), F32)
    d_out = jnp.asarray(rng.standard_normal((T, H)), F32).astype(BF16)
    residuals = (intermediate, weights, sort_indices, inverse_indices)

    def gathers(res, ct):
        inter, w, si, ii = res
        gather, _ = cu._moe_row_gather_kernels()
        rows_of = jax.lax.div(si, jnp.asarray(w.shape[1], si.dtype))
        d_inter, d_sorted = gather(ct, rows_of, w.reshape(-1).astype(F32)[si], inter)
        return d_inter.astype(inter.dtype), d_sorted[ii].reshape(w.shape).astype(w.dtype)

    got = jax.jit(lambda res, ct: cu._kernel_unsort_combine_bwd(BF16, res, ct)[:2])(residuals, d_out)
    want = jax.jit(gathers)(residuals, d_out)
    np.testing.assert_array_equal(np.asarray(got[0], np.float32), np.asarray(want[0], np.float32))
    np.testing.assert_array_equal(_bits(got[1]), _bits(want[1]))
