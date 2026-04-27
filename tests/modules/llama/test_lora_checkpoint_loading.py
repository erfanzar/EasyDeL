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

from __future__ import annotations

import jax.numpy as jnp
import spectrax as spx
from spectrax import nn

import easydel  # noqa: F401
from easydel.modules.llama.llama_configuration import LlamaConfig
from easydel.modules.llama.modeling_llama import LlamaForCausalLM


def _make_model(*, tie_word_embeddings: bool) -> LlamaForCausalLM:
    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=16,
        tie_word_embeddings=tie_word_embeddings,
    )
    cfg.add_basic_configurations(
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        use_sharding_constraint=False,
    )
    with cfg.mesh:
        model = LlamaForCausalLM(
            config=cfg,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            rngs=spx.Rngs(0),
        )
        return model.apply_lora_to_layers(
            lora_rank=2,
            lora_pattern=r"lm_head",
            rngs=spx.Rngs(1),
        )


def test_lora_checkpoint_round_trip_supports_from_pretrained(tmp_path):
    model = _make_model(tie_word_embeddings=False)

    with model.mesh:
        model.save_pretrained(tmp_path)
        loaded = LlamaForCausalLM.from_pretrained(
            tmp_path,
            auto_shard_model=False,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
        )
        outputs = loaded(input_ids=jnp.ones((2, 4), dtype=jnp.int32))

    assert loaded.lora_is_enabled
    assert isinstance(loaded.get_lm_head(), nn.LoRA)
    assert outputs.logits is not None
    assert outputs.logits.shape == (2, 4, loaded.config.vocab_size)


def test_make_lm_head_fn_supports_lora_wrapped_tied_lm_head():
    model = _make_model(tie_word_embeddings=True)
    hidden_states = jnp.ones((2, 4, model.config.hidden_size), dtype=jnp.float32)

    with model.mesh:
        lm_head_fn = model.make_lm_head_fn()
        logits_via_fn = lm_head_fn(hidden_states)
        logits_via_compute = model.compute_lm_logits(hidden_states)

    assert jnp.allclose(logits_via_fn, logits_via_compute, atol=1e-5)
