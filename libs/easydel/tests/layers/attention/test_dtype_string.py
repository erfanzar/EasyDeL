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

"""String-serialized attention dtypes resolve to real dtypes, including bfloat16 / float16."""

import pytest
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("<class 'jax.numpy.bfloat16'>", jnp.bfloat16),
        ("<class 'jax.numpy.float16'>", jnp.float16),
        ("<class 'jax.numpy.float32'>", jnp.float32),
        ("bfloat16", jnp.bfloat16),
        ("float16", jnp.float16),
        ("float32", jnp.float32),
    ],
)
def test_known_dtype_strings_resolve(text, expected):
    config = EasyDeLBaseConfig(
        attn_mechanism="vanilla", attn_dtype=text, attn_softmax_dtype="float32", sharding_axis_dims=(1, 1, -1, 1, 1, 1)
    )
    module = FlexibleAttentionModule(config, softmax_scale=1.0, requires_cache=False)
    assert jnp.dtype(module.metadata.runtime_dtype) == jnp.dtype(expected)
    assert jnp.dtype(module.metadata.runtime_softmax_dtype) == jnp.dtype(jnp.float32)


@pytest.mark.parametrize("field", ["attn_dtype", "attn_softmax_dtype"])
def test_unknown_dtype_is_rejected_at_public_constructor(field):
    config = EasyDeLBaseConfig(
        attn_mechanism="vanilla", sharding_axis_dims=(1, 1, -1, 1, 1, 1), **{field: "not-a-dtype"}
    )
    with pytest.raises(TypeError, match="not-a-dtype"):
        FlexibleAttentionModule(config, softmax_scale=1.0, requires_cache=False)
