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

"""Regression tests for eSurge fused/SPMD LM-head logit post-processing.

The fused SPMD and split-SPMD LM-head executables in ``ModelStepExecutor``
must apply the *same* final logit transform as the model's own
``compute_lm_logits`` (e.g. Gemma-2 ``cap * tanh(logits / cap)`` soft-cap,
Cohere ``logit_scale``), not the raw ``apply_lm_head`` vocab projection.
Otherwise those families sample from the wrong distribution at
``temperature > 0`` (greedy is unaffected because the transforms here are
monotonic). This exercises the real ``_build_lm_head_fn`` executable — the
SPMD split LM-head projection site.
"""

import easydel as ed
import jax
import spectrax as spx
from easydel.inference.esurge.runners.executors.model_executor import ModelStepExecutor
from easydel.infra.sharding import replicated_named_sharding
from jax import numpy as jnp


def _build_gemma2(final_logit_softcapping):
    """Tiny Gemma2 CausalLM with a configurable final logit soft-cap."""
    config = ed.Gemma2Config(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=64,
        head_dim=8,
        final_logit_softcapping=final_logit_softcapping,
    )
    return ed.Gemma2ForCausalLM(config=config, rngs=spx.Rngs(0))


def _make_spmd_lm_head_fn(model):
    """Build the real SPMD ``_lm_head_step`` executable around ``model``.

    Constructs a ``ModelStepExecutor`` shell wired with only the state the
    SPMD ``_build_lm_head_fn`` path reads, mirroring how
    ``ExecutionManager`` splits the model and derives the replicated
    sharding. Returns ``(lm_head_fn, graphdef, graphstate_arg,
    graphother_arg)`` ready to invoke on pre-gathered hidden rows.
    """
    graphdef, graphstate, graphother = model.split_module()
    assert not model.mesh.is_mpmd, "test targets the SPMD LM-head path"

    executor = object.__new__(ModelStepExecutor)
    executor.model = model
    executor.mesh = model.mesh
    executor._empty_sharding = replicated_named_sharding(model.mesh)
    executor._flat_state_args = not model.mesh.is_mpmd
    executor._graphstate_treedef = jax.tree_util.tree_structure(graphstate)
    executor._graphother_treedef = jax.tree_util.tree_structure(graphother)

    lm_head_fn = executor._build_lm_head_fn(
        graphstate_template=graphstate,
        graphother_template=graphother,
    )
    graphstate_arg = tuple(jax.tree_util.tree_leaves(graphstate))
    graphother_arg = tuple(jax.tree_util.tree_leaves(graphother))
    return lm_head_fn, graphdef, graphstate_arg, graphother_arg


def test_spmd_lm_head_applies_final_softcapping():
    """eSurge SPMD LM-head must match compute_lm_logits (soft-cap applied)."""
    model = _build_gemma2(final_logit_softcapping=5.0)
    lm_head_fn, graphdef, gs_arg, go_arg = _make_spmd_lm_head_fn(model)

    # Scale hidden rows so raw logits comfortably exceed the ±cap band,
    # making the soft-cap transform (and thus the bug) plainly observable.
    padded_num_reqs = 4
    hidden = 25.0 * jax.random.normal(jax.random.key(1), (padded_num_reqs, 16), dtype=jnp.float32)

    logits = lm_head_fn(graphdef, gs_arg, go_arg, hidden)

    expected = model.compute_lm_logits(hidden)  # soft-capped
    raw = model.apply_lm_head(hidden)  # raw projection (buggy output)

    # The executable must reproduce the model's post-processed logits ...
    assert jnp.allclose(logits, expected, atol=1e-4), (
        f"max abs diff vs compute_lm_logits = {float(jnp.max(jnp.abs(logits - expected)))}"
    )
    # ... and must NOT be the raw projection: the soft-cap has to bite here,
    # which is exactly what the fix restores (fails on the pre-fix code that
    # called apply_lm_head directly).
    assert float(jnp.max(jnp.abs(expected - raw))) > 0.5, (
        "soft-cap did not change logits; test is not exercising the bug"
    )
    assert not jnp.allclose(logits, raw, atol=0.5)


def test_spmd_lm_head_is_noop_without_postprocess():
    """For families with identity compute_lm_logits, output == raw projection."""
    model = _build_gemma2(final_logit_softcapping=None)
    lm_head_fn, graphdef, gs_arg, go_arg = _make_spmd_lm_head_fn(model)

    padded_num_reqs = 4
    hidden = 25.0 * jax.random.normal(jax.random.key(2), (padded_num_reqs, 16), dtype=jnp.float32)

    logits = lm_head_fn(graphdef, gs_arg, go_arg, hidden)
    raw = model.apply_lm_head(hidden)

    # No soft-cap / scale configured -> compute_lm_logits is identity over the
    # raw projection, so the executable must be byte-for-byte the projection.
    assert jnp.allclose(logits, raw, atol=1e-5)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
