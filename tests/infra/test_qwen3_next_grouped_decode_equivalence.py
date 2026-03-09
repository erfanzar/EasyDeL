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

import jax
import jax.numpy as jnp
import numpy as np
from eformer.escale import PartitionAxis, PartitionManager
from ejkernel.kernels._xla.gated_delta_rule._xla_impl_fwd import _single_step_gdr_fwd
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from easydel.modules.qwen3_next.modeling_qwen3_next import _preserve_array_sharding, apply_grouped_single_step_gdr


def _make_decode_inputs(dtype=jnp.bfloat16):
    rng = jax.random.key(0)
    query = jax.random.normal(rng, (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 1, 6, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    recurrent_state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 6, 4, 5), dtype=jnp.float32).astype(dtype)
    return query, key, value, beta, decay, recurrent_state


def _legacy_single_step(query, key, value, beta, decay, recurrent_state):
    expand_ratio = value.shape[2] // query.shape[2]
    legacy_output, legacy_state = _single_step_gdr_fwd(
        query=jnp.repeat(query, expand_ratio, axis=2).transpose(0, 2, 1, 3),
        key=jnp.repeat(key, expand_ratio, axis=2).transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=None if decay is None else decay.transpose(0, 2, 1),
        recurrent_state=recurrent_state,
    )
    return legacy_output.transpose(0, 2, 1, 3), legacy_state


def test_grouped_single_step_gdr_matches_repeated_heads_with_decay():
    query, key, value, beta, decay, recurrent_state = _make_decode_inputs()

    legacy_output, legacy_state = _legacy_single_step(query, key, value, beta, decay, recurrent_state)
    grouped_output, grouped_state = apply_grouped_single_step_gdr(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=recurrent_state,
    )

    # Grouped path computes in float32 then casts output back; legacy stays in
    # bfloat16 throughout, so minor rounding differences are expected.
    assert jnp.allclose(grouped_output.astype(jnp.float32), legacy_output.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(grouped_state.astype(jnp.float32), legacy_state.astype(jnp.float32), rtol=0.02, atol=0.05)


def test_grouped_single_step_gdr_matches_repeated_heads_without_decay():
    query, key, value, beta, _, recurrent_state = _make_decode_inputs()

    legacy_output, legacy_state = _legacy_single_step(query, key, value, beta, None, recurrent_state)
    grouped_output, grouped_state = apply_grouped_single_step_gdr(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=None,
        recurrent_state=recurrent_state,
    )

    assert jnp.allclose(grouped_output.astype(jnp.float32), legacy_output.astype(jnp.float32), rtol=0.02, atol=0.05)
    assert jnp.allclose(grouped_state.astype(jnp.float32), legacy_state.astype(jnp.float32), rtol=0.02, atol=0.05)


def test_preserve_array_sharding_matches_reference_array():
    mesh = Mesh(np.array(jax.devices()[:1]), ("data",))
    partition_axis = PartitionAxis(batch_axis="data", head_axis=None)
    partition_manager = PartitionManager(partition_axis)
    sharding = NamedSharding(mesh, PartitionSpec("data", None, None, None))

    with mesh:
        preserved = _preserve_array_sharding(
            jnp.zeros((2, 3, 4, 5), dtype=jnp.float32),
            partition_manager=partition_manager,
            partition_axis=partition_axis,
        )

    assert preserved.sharding == sharding
