"""GLM-5-Next's default W8A8 pattern covers MLP and DSA projections and leaves KDA, indexer and kv_b in bf16."""

from __future__ import annotations

import re

import pytest
from easydel.modules.glm5_next.glm5_next_configuration import Glm5NextTextConfig

pytestmark = pytest.mark.cpu_ok


def test_pattern_selects_mlp_and_dsa_projections_only():
    config = Glm5NextTextConfig(
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
    )
    pattern = re.compile(config.default_quantization_pattern("w8a8"))
    quantized = [
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.2.mlp.experts.down_proj",
        "model.layers.2.mlp.shared_experts.up_proj",
        "model.layers.3.self_attn.q_a_proj",
        "model.layers.3.self_attn.kv_a_proj_with_mqa",
        "model.layers.3.self_attn.o_proj",
    ]
    kept = [
        "model.layers.0.self_attn.q_a_proj",  # KDA layer
        "model.layers.3.self_attn.kv_b_proj",
        "model.layers.3.self_attn.indexer.wq_b",
        "model.layers.13.self_attn.o_proj",  # index 13 must not match layer 3 by prefix
        "lm_head",
        "model.embed_tokens",
    ]
    assert all(pattern.search(path) for path in quantized)
    assert not any(pattern.search(path) for path in kept)
