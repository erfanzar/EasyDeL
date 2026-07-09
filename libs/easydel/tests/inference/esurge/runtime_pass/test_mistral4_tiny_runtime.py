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

"""eSurge CPU runtime smoke + decode parity for the mistral4 family.

Drives a tiny randomly initialized ``Mistral4ForCausalLM`` (DeepSeek-V3-shaped
MLA + softmax grouped-top-k MoE + Llama-4 attention temperature) through the
full eSurge engine on CPU. Like deepseek_v3, the engine must serve this family
through the weight-absorbed MLA ragged-pages path
(``multi_latent_ragged_page_attention_v2``), which is asserted explicitly.

The Llama-4 temperature ``1 + beta * log1p(floor(pos / original_max))``
depends on the *absolute* position, so the config uses a small
``original_max_position_embeddings=32`` and prompts long enough that the scale
steps both inside the prefill (position 32) and mid-decode (position 64). The
teacher-forced parity check against the plain uncached forward therefore fails
if the absorbed decode path drops the cache position offset in either RoPE or
the temperature.
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

ORIGINAL_MAX_POSITION_EMBEDDINGS = 32

PROMPTS = [
    "the quick brown fox jumps over that lazy sleeping dog again and again " * 4,
    "one two three four five six seven eight nine ten eleven twelve " * 4,
]


def _build_tiny_mistral4(tokenizer):
    """Build a tiny float32 mistral4 model (one dense + one MoE layer)."""
    config = ed.Mistral4Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.0,
        kv_lora_rank=16,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        norm_topk_prob=True,
        max_position_embeddings=512,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 4.0,
            "original_max_position_embeddings": ORIGINAL_MAX_POSITION_EMBEDDINGS,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
        rope_interleave=True,
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
    with config.mesh:
        return ed.Mistral4ForCausalLM(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )


@pytest.fixture(scope="module")
def mistral4_runtime():
    """Run the tiny mistral4 engines once and share model + outputs across tests.

    The default (async-scheduled, overlapped) engine provides the batched
    smoke + parity outputs; a sync-configured engine provides the bit-exact
    greedy determinism evidence.
    """
    tokenizer = load_test_tokenizer()
    model = _build_tiny_mistral4(tokenizer)

    engine = build_engine(model, tokenizer)
    try:
        runner_attn = str(engine.runner.model.config.get_text_config().attn_mechanism)
        batch = run_greedy_batch(engine, PROMPTS)
    finally:
        engine.terminate()

    sync_engine = build_engine(model, tokenizer, sync=True)
    try:
        single_a, single_b = run_greedy_single_twice(sync_engine, PROMPTS[0])
    finally:
        sync_engine.terminate()

    yield {
        "model": model,
        "batch": batch,
        "single_a": single_a,
        "single_b": single_b,
        "runner_attn": runner_attn,
    }


def test_mistral4_serves_through_mla_ragged_path(mistral4_runtime):
    """The engine must select the absorbed MLA ragged path (deepseek_v3 parity)."""
    assert mistral4_runtime["runner_attn"] == "multi_latent_ragged_page_attention_v2"


def test_mistral4_batched_greedy_generates_full_budget(mistral4_runtime):
    """Both concurrent requests must decode the full greedy token budget."""
    batch = mistral4_runtime["batch"]
    assert set(batch) == set(PROMPTS)
    vocab_size = mistral4_runtime["model"].config.vocab_size
    for prompt in PROMPTS:
        prompt_ids, generated = request_token_ids(batch[prompt])
        # The prompt must cross the first Llama-4 temperature boundary during
        # prefill and the second one mid-decode, otherwise the parity test
        # would not witness the position-dependent attention scale.
        assert len(prompt_ids) > ORIGINAL_MAX_POSITION_EMBEDDINGS
        assert len(prompt_ids) + MAX_TOKENS > 2 * ORIGINAL_MAX_POSITION_EMBEDDINGS
        assert len(generated) == MAX_TOKENS
        assert all(0 <= tok < vocab_size for tok in generated)
        # Non-degenerate decode: the continuation is not a single repeated id.
        assert len(set(generated)) > 4
    gen_a = request_token_ids(batch[PROMPTS[0]])[1]
    gen_b = request_token_ids(batch[PROMPTS[1]])[1]
    assert gen_a != gen_b, "distinct prompts produced identical continuations"


def test_mistral4_greedy_decode_is_deterministic(mistral4_runtime):
    """Two identical single-request greedy runs must match token-for-token."""
    first_ids = request_token_ids(mistral4_runtime["single_a"])[1]
    second_ids = request_token_ids(mistral4_runtime["single_b"])[1]
    assert len(first_ids) == MAX_TOKENS
    assert first_ids == second_ids


def test_mistral4_cached_decode_matches_uncached_forward(mistral4_runtime):
    """Absorbed MLA decode must reproduce the uncached teacher-forced argmax.

    The reference forward runs the non-absorbed (decompressed) MLA branch with
    vanilla attention, so this asserts absorbed-vs-standard equivalence of the
    whole serving stack, including RoPE offsets and the Llama-4 temperature at
    absolute positions on both sides of the ``original_max`` boundaries.
    """
    model = mistral4_runtime["model"]
    for prompt in PROMPTS:
        prompt_ids, generated = request_token_ids(mistral4_runtime["batch"][prompt])
        assert_cached_decode_matches_uncached_forward(model, prompt_ids, generated)
    prompt_ids, generated = request_token_ids(mistral4_runtime["single_a"])
    assert_cached_decode_matches_uncached_forward(model, prompt_ids, generated)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
