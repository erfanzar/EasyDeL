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

"""Public indexer selection primitives behave as documented (accelerator test)."""

import numpy as np
import pytest
from easydel.layers.indexer import is_vacuous_selection, ste_score_proxy, topk_select
from jax import numpy as jnp


def test_topk_select_pads_invalid_picks_with_minus_one():
    scores = jnp.asarray([[[0.1, 0.9, 0.5, 0.3]]], jnp.float32)
    valid = jnp.asarray([[[True, True, False, True]]])
    np.testing.assert_array_equal(np.asarray(topk_select(scores, 2, valid=valid)), [[[1, -1]]])
    np.testing.assert_array_equal(np.asarray(topk_select(scores, 8)), [[[1, 2, 3, 0]]])


@pytest.mark.cpu_ok
def test_is_vacuous_selection_threshold():
    assert is_vacuous_selection(4, 4)
    assert not is_vacuous_selection(3, 4)


def test_ste_score_proxy_is_zero_primal_identity_tangent():
    import jax

    proxy = jnp.asarray([0.5, 2.0, -1.0], jnp.float32)
    selected = jnp.asarray([True, False, True])
    np.testing.assert_array_equal(np.asarray(ste_score_proxy(proxy, selected)), [0.0, 0.0, 0.0])
    grad = np.asarray(jax.grad(lambda p: ste_score_proxy(p, selected).sum())(proxy))
    np.testing.assert_array_equal(grad, [1.0, 0.0, 1.0])
