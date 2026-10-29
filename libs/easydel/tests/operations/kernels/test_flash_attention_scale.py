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

"""Explicit softmax scales must be honored by the FlashAttention adapter."""

import jax
import numpy as np
from easydel.infra import EasyDeLBaseConfig
from easydel.operations import OperationMetadata
from easydel.operations.kernels.flash_attention import FlashAttn
from jax import numpy as jnp


def test_flash_attention_explicit_scale_matches_dense_reference():
    metadata = OperationMetadata(
        runtime_dtype=jnp.bfloat16,
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, 1, 1)),
    )
    op = FlashAttn(metadata)
    q = jnp.arange(1 * 4 * 2 * 64, dtype=jnp.float32).reshape(1, 4, 2, 64).astype(jnp.bfloat16) / 128
    k = jnp.flip(q, axis=1)
    v = jnp.sin(q.astype(jnp.float32)).astype(jnp.bfloat16)
    qf = q.astype(jnp.float32).transpose(0, 2, 1, 3)
    kf = k.astype(jnp.float32).transpose(0, 2, 1, 3)
    vf = v.astype(jnp.float32).transpose(0, 2, 1, 3)

    with metadata.mesh:
        for scale in (0.01, 2.0):
            got = op(query=q, key=k, value=v, softmax_scale=scale, causal=False).attention_outputs
            weights = jax.nn.softmax(jnp.einsum("bhtd,bhsd->bhts", qf, kf) * scale, axis=-1)
            expected = jnp.einsum("bhts,bhsd->bhtd", weights, vf).transpose(0, 2, 1, 3)
            np.testing.assert_allclose(np.asarray(got, dtype=np.float32), np.asarray(expected), atol=0.04, rtol=0.04)
