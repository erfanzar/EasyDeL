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

"""Shared fixtures for the eSurge spec-decode / MTP tests.

``make_tiny_qwen35`` builds the tiny Qwen3.5 + MTP model that every
spec-decode / MTP test used to define its own byte-identical copy of. Import it
with the standard sibling-helper fallback::

    try:
        from ._common import make_tiny_qwen35
    except ImportError:  # standalone `python test_x.py`
        from _common import make_tiny_qwen35
"""

from __future__ import annotations

import jax.numpy as jnp
import spectrax as spx
from easydel.modules.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

VOCAB = 256
HIDDEN = 128


def make_tiny_qwen35(mtp_layers: int = 1) -> Qwen3_5ForCausalLM:
    """Build the tiny Qwen3.5 + MTP model used across the spec-decode tests.

    Args:
        mtp_layers: Number of MTP head layers (``mtp_num_hidden_layers``).

    Returns:
        A float32 ``Qwen3_5ForCausalLM`` with a 4-layer (3 linear + 1 full)
        backbone and an MTP head.
    """
    cfg = Qwen3_5TextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        mtp_num_hidden_layers=mtp_layers,
        mtp_loss_coef=0.0,
        attn_output_gate=True,
        rms_norm_eps=1e-6,
        partial_rotary_factor=0.25,
    )
    return Qwen3_5ForCausalLM(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
