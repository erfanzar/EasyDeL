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

from easydel.operations.kernels.kda import _single_step_kda_fwd, _single_step_kda_fwd_bthd


def _make_decode_inputs(dtype=jnp.bfloat16):
    rng = jax.random.key(0)
    query = jax.random.normal(rng, (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 1, 3, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 1, 3), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 1, 3), dtype=jnp.float32).astype(dtype)
    recurrent_state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 3, 4, 5), dtype=jnp.float32).astype(dtype)
    return query, key, value, beta, decay, recurrent_state


def test_single_step_kda_bthd_matches_legacy_layout_with_decay():
    query, key, value, beta, decay, recurrent_state = _make_decode_inputs()

    legacy_output, legacy_state = _single_step_kda_fwd(
        query=query.transpose(0, 2, 1, 3),
        key=key.transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=decay.transpose(0, 2, 1),
        recurrent_state=recurrent_state,
    )
    fast_output, fast_state = _single_step_kda_fwd_bthd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=recurrent_state,
    )

    assert jnp.array_equal(fast_output, legacy_output.transpose(0, 2, 1, 3))
    assert jnp.array_equal(fast_state, legacy_state)


def test_single_step_kda_bthd_matches_legacy_layout_without_decay():
    query, key, value, beta, _, recurrent_state = _make_decode_inputs()

    legacy_output, legacy_state = _single_step_kda_fwd(
        query=query.transpose(0, 2, 1, 3),
        key=key.transpose(0, 2, 1, 3),
        value=value.transpose(0, 2, 1, 3),
        beta=beta.transpose(0, 2, 1),
        decay=None,
        recurrent_state=recurrent_state,
    )
    fast_output, fast_state = _single_step_kda_fwd_bthd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=None,
        recurrent_state=recurrent_state,
    )

    assert jnp.array_equal(fast_output, legacy_output.transpose(0, 2, 1, 3))
    assert jnp.array_equal(fast_state, legacy_state)
