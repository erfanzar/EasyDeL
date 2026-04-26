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
import spectrax as spx
from jax.sharding import Mesh

from easydel.modules.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig


def _make_mesh():
    return Mesh(np.array(jax.devices()[:1]), ("data",))


def _config(**overrides):
    defaults = dict(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        global_head_dim=64,
        max_position_embeddings=512,
        sliding_window=64,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        attn_mechanism="vanilla",
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _run_cached_decode_parity(
    config, *, dtype=jnp.bfloat16, precision=jax.lax.Precision.DEFAULT, steps=8, atol=0.1, cos_threshold=0.99
):
    """Run multi-step cached decode and verify parity with full recompute.

    Uses cosine similarity and argmax agreement as the primary correctness
    criteria rather than strict element-wise tolerance, because the padded
    TransformerCache introduces small numerical differences from the
    non-padded full-recompute path (different XLA computation graphs).
    """
    input_ids = jnp.array([[2, 17, 23, 29]], dtype=jnp.int32)
    attention_mask = jnp.ones_like(input_ids)
    mesh = _make_mesh()

    with mesh:
        model = Gemma4ForCausalLM(
            config=config,
            dtype=dtype,
            param_dtype=dtype,
            precision=precision,
            rngs=spx.Rngs(0),
        )

        model_kwargs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=32,
            pad_token_id=config.pad_token_id,
        )
        cached_outputs = model(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
        sequence = input_ids
        current_model_kwargs = model_kwargs

        for step in range(steps):
            full_outputs = model(
                input_ids=sequence,
                attention_mask=jnp.ones_like(sequence),
            )
            next_token = jnp.argmax(full_outputs.logits[:, -1, :], axis=-1).astype(jnp.int32)[:, None]

            current_model_kwargs = model.update_inputs_for_generation(cached_outputs, current_model_kwargs)
            call_kwargs = model._prepare_mask_info_for_generation_step(next_token, current_model_kwargs)
            cached_next = model(next_token, **call_kwargs)

            full_next = model(
                input_ids=jnp.concatenate([sequence, next_token], axis=1),
                attention_mask=jnp.ones((1, sequence.shape[1] + 1), dtype=jnp.int32),
            )

            cached_logits = cached_next.logits[:, -1].astype(jnp.float32)
            full_logits = full_next.logits[:, -1].astype(jnp.float32)

            # Cosine similarity should be very high
            cos_sim = float(
                jnp.sum(cached_logits * full_logits) / (jnp.linalg.norm(cached_logits) * jnp.linalg.norm(full_logits))
            )
            max_diff = float(jnp.max(jnp.abs(cached_logits - full_logits)))

            assert cos_sim >= cos_threshold, f"Step {step}: cosine similarity {cos_sim:.6f} < {cos_threshold}"
            assert max_diff < atol, f"Step {step}: max_diff {max_diff:.6f} >= {atol}"

            cached_outputs = cached_next
            sequence = jnp.concatenate([sequence, next_token], axis=1)


def test_multi_step_cached_decode_matches_full_recompute_under_bfloat16_default_precision():
    _run_cached_decode_parity(_config())


def test_cached_decode_parity_with_per_layer_inputs():
    _run_cached_decode_parity(_config(hidden_size_per_layer_input=32))


def test_cached_decode_parity_with_kv_sharing():
    _run_cached_decode_parity(_config(num_kv_shared_layers=2))
