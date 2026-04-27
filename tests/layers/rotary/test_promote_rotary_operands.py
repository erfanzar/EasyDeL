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

import jax.numpy as jnp

from easydel.layers.rotary._modules import _promote_rotary_operands


def test_promote_same_dtype_noop():
    a = jnp.ones((2, 4), dtype=jnp.bfloat16)
    b = jnp.ones((2, 4), dtype=jnp.bfloat16)
    results = _promote_rotary_operands(a, b)
    assert all(r.dtype == jnp.bfloat16 for r in results)


def test_promote_mixed_standard_dtypes():
    a = jnp.ones((2,), dtype=jnp.float16)
    b = jnp.ones((2,), dtype=jnp.float32)
    results = _promote_rotary_operands(a, b)
    assert all(r.dtype == jnp.float32 for r in results)


def test_promote_lowfloat_goes_to_float32():
    a = jnp.ones((2,), dtype=jnp.float8_e4m3fn)
    b = jnp.ones((2,), dtype=jnp.bfloat16)
    results = _promote_rotary_operands(a, b)
    assert all(r.dtype == jnp.float32 for r in results)


def test_promote_all_lowfloat():
    a = jnp.ones((2,), dtype=jnp.float8_e4m3fn)
    b = jnp.ones((2,), dtype=jnp.float8_e5m2)
    results = _promote_rotary_operands(a, b)
    assert all(r.dtype == jnp.float32 for r in results)
