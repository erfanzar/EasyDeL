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

"""Regression test for the inline-MTP drafter cross-request cache leak (SP2).

The ``Qwen3_5MTPDrafter`` keeps a single rolling MTP KV cache keyed only on
batch size. Without a per-request / per-verify-window reset boundary it is
reused across requests, so a previous request's K/V pollutes the next drafter
call and acceptance collapses. ``reset_request()`` clears the cache; the eSurge
runner calls it at the start of each verify window.

This test is red-on-revert: if ``reset_request`` is turned back into a no-op the
"after reset" draft distribution no longer matches the fresh-drafter reference,
and ``test_mtp_reset_request_prevents_cross_request_leak`` fails.
"""

from __future__ import annotations

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import spectrax as spx
from easydel.inference.speculative import Qwen3_5MTPDrafter
from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

_HIDDEN = 32


def _tiny_mtp_model() -> Qwen3_5ForCausalLM:
    cfg = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=_HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        layer_types=["full_attention", "full_attention"],
        mtp_num_hidden_layers=1,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )
    return Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)


def _hidden(seed: int) -> jnp.ndarray:
    return jax.random.normal(jax.random.fold_in(jax.random.PRNGKey(7), seed), (1, 1, _HIDDEN), dtype=jnp.float32)


def _draft_b(drafter: Qwen3_5MTPDrafter) -> jnp.ndarray:
    """Draft the fixed 'request B' step and return its full log-probs."""
    step = drafter.draft(
        input_ids=jnp.array([[11]], dtype=jnp.int32),
        target_hidden_states=_hidden(99),
        position_ids=jnp.array([[3]], dtype=jnp.int32),
        return_full_log_probs=True,
    )
    return np.asarray(step.full_log_probs)


def _fill_request_a(drafter: Qwen3_5MTPDrafter) -> None:
    """Populate the MTP cache with three 'request A' decode steps at positions 0..2."""
    for pos in range(3):
        drafter.draft(
            input_ids=jnp.array([[5]], dtype=jnp.int32),
            target_hidden_states=_hidden(pos),
            position_ids=jnp.array([[pos]], dtype=jnp.int32),
        )


def test_mtp_reset_request_prevents_cross_request_leak():
    """After ``reset_request`` a new request drafts as if the MTP cache were fresh."""
    model = _tiny_mtp_model()

    reference = Qwen3_5MTPDrafter(model)
    reference.reset(1)
    clean_b = _draft_b(reference)

    tested = Qwen3_5MTPDrafter(model)
    tested.reset(1)
    _fill_request_a(tested)
    tested.reset_request()
    after_reset_b = _draft_b(tested)

    # With the boundary in place, request B is unpolluted by request A.
    assert np.allclose(after_reset_b, clean_b, atol=1e-5), (
        f"reset_request did not clear cross-request MTP cache state "
        f"(max abs diff {float(np.max(np.abs(after_reset_b - clean_b)))})"
    )


def test_mtp_cache_leaks_without_reset_boundary():
    """Sanity anchor: without the reset boundary, request A pollutes request B.

    This locks in that the regression above is real — the two B distributions
    genuinely diverge when request A's K/V is left in the cache — so the
    ``reset_request`` assertion is meaningful rather than tautological.
    """
    model = _tiny_mtp_model()

    reference = Qwen3_5MTPDrafter(model)
    reference.reset(1)
    clean_b = _draft_b(reference)

    polluted = Qwen3_5MTPDrafter(model)
    polluted.reset(1)
    _fill_request_a(polluted)
    polluted_b = _draft_b(polluted)  # deliberately no reset_request

    assert not np.allclose(polluted_b, clean_b, atol=1e-4), (
        "expected request A's stale MTP K/V to change request B's distribution"
    )


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
