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

"""XLA top-k filter dtype and exact integer ordering contracts."""

import numpy as np
import pytest
from ejkernel.kernels._xla.topk._interface import topk
from jax import numpy as jnp


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([False, True, False], [False, True, False]),
        ([2**24, 2**24 + 1, -1], [-(2**31), 2**24 + 1, -(2**31)]),
        ([0, 2**24 + 1, 2**24], [-(2**31), 2**24 + 1, -(2**31)]),
    ],
)
def test_filter_default_preserves_exact_integer_order(values, expected):
    """Distinct int32 values beyond float32's exact range must not tie."""
    dtype = jnp.bool_ if isinstance(values[0], bool) else jnp.int32
    result = topk(jnp.asarray([values], dtype=dtype), 1, mode="filter")
    reference = np.asarray([expected], dtype=np.bool_ if dtype == jnp.bool_ else np.int32)
    np.testing.assert_array_equal(np.asarray(result), reference)


@pytest.mark.parametrize("mode", ["mask", "filter"])
def test_float64_values_are_not_narrowed_into_ties(mode):
    """1.0 and 1.0 + 2**-30 are equal in float32; the float64 key must still separate them (r10 S04-4)."""
    import jax

    with jax.enable_x64(True):
        operand = jnp.asarray([[1.0, 1.0 + 2.0**-30, -3.0]], dtype=jnp.float64)
        result = np.asarray(topk(operand, 1, mode=mode))
    if mode == "mask":
        np.testing.assert_array_equal(result, [[False, True, False]])
    else:
        assert result[0, 1] == 1.0 + 2.0**-30
        # The fill is finfo(float64).min; TPU emulates float64 with a narrower exponent range, so it
        # can surface as -inf there. Only require that non-kept entries were filled, not kept.
        assert result[0, 0] < -1e30
        assert result[0, 2] < -1e30


@pytest.mark.parametrize("values", [[2**24, 2**24 + 1, 0], [2**32 - 2, 2**32 - 1, 0]])
def test_filter_unsigned_integer_uses_dtype_minimum(values):
    """Unsigned filtering must preserve order, even at the maximum key."""
    result = topk(jnp.asarray([values], dtype=jnp.uint32), 1, mode="filter")
    np.testing.assert_array_equal(np.asarray(result), np.asarray([[0, values[1], 0]], dtype=np.uint32))
