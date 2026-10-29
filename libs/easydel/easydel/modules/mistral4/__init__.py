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

"""Mistral4 model implementation for EasyDeL.

Mistral4 is Mistral AI's DeepSeek-V3-shaped architecture: Multi-head Latent
Attention (low-rank Q/KV projections, split nope/rope head dims, compressed
KV latents) combined with a DeepseekV3-style mixture-of-experts block
(stacked routed experts plus one shared expert, grouped top-k routing).

Key architectural features:
    - Multi-head Latent Attention (MLA): compresses key-value cache into
      low-rank latents (``kv_lora_rank``) with a shared single-head RoPE
      key, dramatically shrinking the KV cache footprint.
    - Softmax MoE router: routing scores are a plain softmax over the
      router logits (no sigmoid scoring, no ``e_score_correction_bias``),
      restricted to the ``topk_group`` best of ``n_group`` expert groups.
    - Llama-4 attention temperature: queries are scaled by
      ``1 + beta * log(1 + floor(position / original_max))`` for stable
      long-context attention entropy.
    - YaRN RoPE with interleaved (GPT-J) channel pairing on the rope slice
      (``rope_interleave=True``) and a very long default context
      (1,048,576 positions extended from an 8,192-token window).

Model variants:
    - Mistral4Model: Base decoder-only transformer with MLA and MoE.
    - Mistral4ForCausalLM: Language modeling head for next-token prediction.
    - Mistral4ForSequenceClassification: Sequence classification head.

Usage example:
    ```python
    from easydel import AutoEasyDeLModelForCausalLM
    import jax.numpy as jnp

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        "mistralai/Mistral-Small-4-119B-2603",
        dtype=jnp.bfloat16,
        param_dtype=jnp.bfloat16,
    )
    ```
"""

from .mistral4_configuration import Mistral4Config
from .modeling_mistral4 import Mistral4ForCausalLM, Mistral4ForSequenceClassification, Mistral4Model

__all__ = ("Mistral4Config", "Mistral4ForCausalLM", "Mistral4ForSequenceClassification", "Mistral4Model")
