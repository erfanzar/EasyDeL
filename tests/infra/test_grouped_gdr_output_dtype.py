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

"""Tests that apply_grouped_single_step_gdr preserves input dtype for outputs."""

import jax
import jax.numpy as jnp
import pytest

from easydel.modules.qwen3_next.modeling_qwen3_next import apply_grouped_single_step_gdr


def _make_inputs(dtype):
    rng = jax.random.key(42)
    query = jax.random.normal(rng, (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 1, 6, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 1, 6), dtype=jnp.float32).astype(dtype)
    recurrent_state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 6, 4, 5), dtype=jnp.float32).astype(dtype)
    return query, key, value, beta, decay, recurrent_state


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_grouped_gdr_output_dtype_matches_input(dtype):
    query, key, value, beta, decay, recurrent_state = _make_inputs(dtype)
    output, new_state = apply_grouped_single_step_gdr(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=recurrent_state,
    )
    assert output.dtype == dtype, f"Expected output dtype {dtype}, got {output.dtype}"
    # Recurrent state stays in float32 for precision
    assert new_state.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_grouped_gdr_output_dtype_no_decay(dtype):
    query, key, value, beta, _, recurrent_state = _make_inputs(dtype)
    output, _new_state = apply_grouped_single_step_gdr(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=None,
        recurrent_state=recurrent_state,
    )
    assert output.dtype == dtype


def test_grouped_gdr_float32_output_numerically_matches_bfloat16_upcast():
    """Verify that float32 computation produces same results regardless of input dtype casting."""
    query_f32, key_f32, value_f32, beta_f32, decay_f32, state_f32 = _make_inputs(jnp.float32)

    out_f32, _ = apply_grouped_single_step_gdr(
        query=query_f32,
        key=key_f32,
        value=value_f32,
        beta=beta_f32,
        decay=decay_f32,
        recurrent_state=state_f32,
    )

    out_bf16, _ = apply_grouped_single_step_gdr(
        query=query_f32.astype(jnp.bfloat16),
        key=key_f32.astype(jnp.bfloat16),
        value=value_f32.astype(jnp.bfloat16),
        beta=beta_f32.astype(jnp.bfloat16),
        decay=decay_f32.astype(jnp.bfloat16),
        recurrent_state=state_f32.astype(jnp.bfloat16),
    )

    # bf16 output should be close to f32 output (both compute internally in f32,
    # but bf16 inputs lose precision before the compute begins)
    assert jnp.allclose(out_f32, out_bf16.astype(jnp.float32), rtol=0.05, atol=0.1)
