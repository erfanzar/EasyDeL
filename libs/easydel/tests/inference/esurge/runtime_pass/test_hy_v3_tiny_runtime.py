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

"""eSurge CPU runtime smoke + decode parity for the hy_v3 family.

Drives a tiny randomly initialized ``HyV3ForCausalLM`` (Qwen3-style QK-norm
GQA + sigmoid-router MoE) through the full eSurge engine on CPU: prefill,
page-crossing greedy decode, a batch of two concurrent requests, and greedy
determinism. The critical check teacher-forces the engine's continuation
through the plain uncached forward and requires exact argmax agreement.
"""

from __future__ import annotations

import jax
import pytest
import spectrax as spx
from jax import numpy as jnp

import easydel as ed

try:
    from tests.inference.esurge.runtime_pass._tiny_runtime_common import (
        MAX_TOKENS,
        assert_cached_decode_matches_uncached_forward,
        build_engine,
        load_test_tokenizer,
        request_token_ids,
        run_greedy_batch,
        run_greedy_single_twice,
    )
except ImportError:  # pragma: no cover - direct invocation
    from _tiny_runtime_common import (  # pyright: ignore[reportImplicitRelativeImport]
        MAX_TOKENS,
        assert_cached_decode_matches_uncached_forward,
        build_engine,
        load_test_tokenizer,
        request_token_ids,
        run_greedy_batch,
        run_greedy_single_twice,
    )

PROMPTS = [
    "The quick brown fox jumps over the lazy dog while the cat sleeps.",
    "Numbers: one two three four five six seven eight nine ten eleven.",
]


def _build_tiny_hy_v3(tokenizer):
    """Build a tiny float32 hy_v3 model (one dense + one sparse MoE layer)."""
    config = ed.HyV3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        intermediate_size=128,
        max_position_embeddings=512,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        num_shared_experts=1,
        moe_intermediate_size=64,
        router_scaling_factor=2.826,
        enable_moe_fp32_combine=True,
        mlp_layer_types=["dense", "sparse"],
        kvdtype=jnp.float32,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # Pure tensor parallelism: keeps the expert mesh at ep=1 (XLA:CPU
        # cannot lower the fused-MoE ragged-all-to-all) while still sharding
        # every projection across the 8-fake-device CPU mesh.
        sharding_axis_dims=(1, 1, 1, 1, jax.device_count(), 1),
    )
    config.moe_force_xla_gmm = True
    config.fsdp_is_ep_bound = False
    with config.mesh:
        return ed.HyV3ForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


@pytest.fixture(scope="module")
def hy_v3_runtime():
    """Run the tiny hy_v3 engines once and share model + outputs across tests.

    The default (async-scheduled, overlapped) engine provides the batched
    smoke + parity outputs; a sync-configured engine provides the bit-exact
    greedy determinism evidence.
    """
    tokenizer = load_test_tokenizer()
    model = _build_tiny_hy_v3(tokenizer)

    engine = build_engine(model, tokenizer)
    try:
        batch = run_greedy_batch(engine, PROMPTS)
    finally:
        engine.terminate()

    sync_engine = build_engine(model, tokenizer, sync=True)
    try:
        single_a, single_b = run_greedy_single_twice(sync_engine, PROMPTS[0])
    finally:
        sync_engine.terminate()

    yield {"model": model, "batch": batch, "single_a": single_a, "single_b": single_b}


def test_hy_v3_batched_greedy_generates_full_budget(hy_v3_runtime):
    """Both concurrent requests must decode the full greedy token budget."""
    batch = hy_v3_runtime["batch"]
    assert set(batch) == set(PROMPTS)
    vocab_size = hy_v3_runtime["model"].config.vocab_size
    for prompt in PROMPTS:
        prompt_ids, generated = request_token_ids(batch[prompt])
        assert len(prompt_ids) > 1
        assert len(generated) == MAX_TOKENS
        assert all(0 <= tok < vocab_size for tok in generated)
        # Non-degenerate decode: the continuation is not a single repeated id.
        assert len(set(generated)) > 4
    gen_a = request_token_ids(batch[PROMPTS[0]])[1]
    gen_b = request_token_ids(batch[PROMPTS[1]])[1]
    assert gen_a != gen_b, "distinct prompts produced identical continuations"


def test_hy_v3_greedy_decode_is_deterministic(hy_v3_runtime):
    """Two identical single-request greedy runs must match token-for-token."""
    first_ids = request_token_ids(hy_v3_runtime["single_a"])[1]
    second_ids = request_token_ids(hy_v3_runtime["single_b"])[1]
    assert len(first_ids) == MAX_TOKENS
    assert first_ids == second_ids


def test_hy_v3_cached_decode_matches_uncached_forward(hy_v3_runtime):
    """Paged-cache decode must reproduce the uncached teacher-forced argmax."""
    model = hy_v3_runtime["model"]
    for prompt in PROMPTS:
        prompt_ids, generated = request_token_ids(hy_v3_runtime["batch"][prompt])
        assert_cached_decode_matches_uncached_forward(model, prompt_ids, generated)
    prompt_ids, generated = request_token_ids(hy_v3_runtime["single_a"])
    assert_cached_decode_matches_uncached_forward(model, prompt_ids, generated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
