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

"""KDA single-step decode dtype contract.

Pins the surviving decode entry points — the BTHD shim
``_single_step_kda_fwd_bthd`` and the shared ``_single_step_kda_core`` —
to preserve the input dtype on the output while the recurrent state stays
float32 for numerical precision. (Ported from the former
``test_single_step_dtype.py`` when the redundant BHTD shim
``_single_step_kda_fwd`` was removed; the assertions here cover both
shims' shared contract.)
"""

import jax
import jax.numpy as jnp
import pytest
from easydel.operations.kernels.kda import _single_step_kda_core, _single_step_kda_fwd_bthd


def _make_core_inputs(dtype):
    rng = jax.random.key(7)
    query = jax.random.normal(rng, (2, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 3, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 3), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 3), dtype=jnp.float32).astype(dtype)
    state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 3, 4, 5), dtype=jnp.float32).astype(dtype)
    return query, key, value, beta, decay, state


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_kda_core_output_dtype(dtype):
    """The core update returns output in the input dtype and state in f32."""
    query, key, value, beta, decay, state = _make_core_inputs(dtype)
    output, new_state = _single_step_kda_core(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=state,
    )
    assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"
    # State stays float32 for numerical precision
    assert new_state.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_kda_core_output_dtype_no_decay(dtype):
    """The dtype contract holds with decay disabled (None)."""
    query, key, value, beta, _, state = _make_core_inputs(dtype)
    output, _new_state = _single_step_kda_core(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=None,
        recurrent_state=state,
    )
    assert output.dtype == dtype
    assert _new_state.dtype == jnp.float32


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
def test_kda_bthd_output_dtype(dtype):
    """The BTHD decode shim preserves the input dtype through the squeeze."""
    rng = jax.random.key(7)
    query = jax.random.normal(rng, (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    key = jax.random.normal(jax.random.fold_in(rng, 1), (2, 1, 3, 4), dtype=jnp.float32).astype(dtype)
    value = jax.random.normal(jax.random.fold_in(rng, 2), (2, 1, 3, 5), dtype=jnp.float32).astype(dtype)
    beta = jax.random.normal(jax.random.fold_in(rng, 3), (2, 1, 3), dtype=jnp.float32).astype(dtype)
    decay = jax.random.normal(jax.random.fold_in(rng, 4), (2, 1, 3), dtype=jnp.float32).astype(dtype)
    state = jax.random.normal(jax.random.fold_in(rng, 5), (2, 3, 4, 5), dtype=jnp.float32).astype(dtype)

    output, new_state = _single_step_kda_fwd_bthd(
        query=query,
        key=key,
        value=value,
        beta=beta,
        decay=decay,
        recurrent_state=state,
    )
    assert output.dtype == dtype, f"Expected {dtype}, got {output.dtype}"
    assert output.shape == (2, 1, 3, 5)
    # State stays float32 for numerical precision
    assert new_state.dtype == jnp.float32
