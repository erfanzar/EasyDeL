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

"""MTP heads own their cache and must not alias decoder layers in operation-cache discovery (issue #288).

The Qwen3.5 MTP block lives at ``mtp.layers.0``; before the fix its attention was discovered as decoder layer 0,
turning that linear-attention layer into a ``ParallelHybridCacheView`` and breaking cached generation after the
first token.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from easydel.caching import RecurrentCacheView, TransformerCacheView
from transformers import GenerationConfig

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def _qwen3_5(mtp_layers: int):
    import spectrax as spx
    from easydel.modules.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=len(LAYER_TYPES),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        max_position_embeddings=256,
        layer_types=LAYER_TYPES,
        mtp_num_hidden_layers=mtp_layers,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        initializer_range=0.2,
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        scan_layers=False,
    )
    return Qwen3_5ForCausalLM(
        config=config,
        rngs=spx.Rngs(0),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
    )


@pytest.mark.parametrize("mtp_layers", [0, 1, 2])
def test_mtp_head_does_not_change_decoder_cache_views(mtp_layers):
    views = _qwen3_5(mtp_layers).get_operations_cache_view()
    assert views == {
        0: RecurrentCacheView,
        1: RecurrentCacheView,
        2: RecurrentCacheView,
        3: TransformerCacheView,
    }


def test_cached_generate_with_mtp_head_matches_uncached_greedy():
    model = _qwen3_5(1)
    prompt = [1, 50, 300, 7, 99, 120, 42, 5, 17, 80, 3, 64]
    steps = 12

    reference, cur = [], list(prompt)
    with model.mesh:
        for _ in range(steps):
            logits = np.asarray(model(input_ids=jnp.asarray([cur])).logits)
            nxt = int(logits[0, -1].argmax())
            reference.append(nxt)
            cur.append(nxt)

        generation_config = GenerationConfig(max_new_tokens=steps, do_sample=False, pad_token_id=0, eos_token_id=None)
        # Built directly (not via from_pretrained), so the model carries no default generation config.
        model.generation_config = generation_config
        out = model.generate(
            input_ids=jnp.asarray([prompt]),
            attention_mask=jnp.ones((1, len(prompt)), jnp.int32),
            generation_config=generation_config,
        )
    sequences = out.sequences[0] if hasattr(out, "sequences") else out[0]
    generated = np.asarray(sequences)[len(prompt) :].tolist()
    assert generated == reference
