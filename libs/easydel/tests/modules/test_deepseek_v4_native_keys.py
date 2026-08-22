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

"""DeepSeek-V4 native-checkpoint key mapping.

DeepSeek publishes V4 only in its own inference naming (``layers.N.attn.wq_a``),
never in transformers naming -- the repo's ``inference/convert.py`` runs
HF -> native, so the transformers form is that script's input and was not
released. These cases pin the inverse rewrite against keys taken verbatim from
the published ``model.safetensors.index.json`` of
``deepseek-ai/DeepSeek-V4-Flash-0731``.
"""

import pytest
from easydel.modules.deepseek_v4._native_checkpoint import native_key_to_easydel

# (native key, expected EasyDeL parameter) -- all left-hand sides are real
# patterns from the published index.
CASES = [
    # model-level tensors
    ("embed.weight", "model.embed_tokens.weight"),
    ("head.weight", "lm_head.weight"),
    ("norm.weight", "model.norm.weight"),
    ("hc_head_base", "model.hc_head.hc_base"),
    # MLA projections + norms
    ("layers.3.attn.wq_a.weight", "model.layers.3.self_attn.q_a_proj.weight"),
    ("layers.3.attn.wq_b.weight", "model.layers.3.self_attn.q_b_proj.weight"),
    ("layers.3.attn.wkv.weight", "model.layers.3.self_attn.kv_proj.weight"),
    ("layers.3.attn.wo_a.weight", "model.layers.3.self_attn.o_a_proj.weight"),
    ("layers.3.attn.wo_b.weight", "model.layers.3.self_attn.o_b_proj.weight"),
    ("layers.3.attn.q_norm.weight", "model.layers.3.self_attn.q_a_norm.weight"),
    ("layers.3.attn.kv_norm.weight", "model.layers.3.self_attn.kv_norm.weight"),
    ("layers.3.attn.attn_sink", "model.layers.3.self_attn.sinks"),
    ("layers.3.attn_norm.weight", "model.layers.3.input_layernorm.weight"),
    ("layers.3.ffn_norm.weight", "model.layers.3.post_attention_layernorm.weight"),
    # per-layer heavily-compressed tensors move into submodules
    ("layers.7.hc_attn_base", "model.layers.7.attn_hc.base"),
    ("layers.7.hc_ffn_scale", "model.layers.7.ffn_hc.scale"),
    # the main compressor
    ("layers.5.attn.compressor.wkv.weight", "model.layers.5.self_attn.compressor.kv_proj.weight"),
    ("layers.5.attn.compressor.wgate.weight", "model.layers.5.self_attn.compressor.gate_proj.weight"),
    ("layers.5.attn.compressor.norm.weight", "model.layers.5.self_attn.compressor.kv_norm.weight"),
    ("layers.5.attn.compressor.ape", "model.layers.5.self_attn.compressor.position_bias"),
    # MoE router
    ("layers.9.ffn.gate.weight", "model.layers.9.mlp.gate.weight"),
    ("layers.9.ffn.gate.bias", "model.layers.9.mlp.gate.e_score_correction_bias"),
    ("layers.9.ffn.gate.tid2eid", "model.layers.9.mlp.gate.tid2eid"),
    # experts: w1/w3/w2 -> gate/up/down (structural stacking is reform_param's job)
    ("layers.9.ffn.experts.17.w1.weight", "model.layers.9.mlp.experts.17.gate_proj.weight"),
    ("layers.9.ffn.experts.17.w3.weight", "model.layers.9.mlp.experts.17.up_proj.weight"),
    ("layers.9.ffn.experts.17.w2.weight", "model.layers.9.mlp.experts.17.down_proj.weight"),
    ("layers.9.ffn.shared_experts.w1.weight", "model.layers.9.mlp.shared_experts.gate_proj.weight"),
    # quant scales ride alongside their weight and keep the suffix
    ("layers.9.ffn.experts.17.w1.scale", "model.layers.9.mlp.experts.17.gate_proj.scale"),
    ("layers.3.attn.wq_a.scale", "model.layers.3.self_attn.q_a_proj.scale"),
]


@pytest.mark.parametrize(("native", "expected"), CASES)
def test_native_key_maps_to_easydel_name(native, expected):
    assert native_key_to_easydel(native) == expected


# The nesting is INVERTED between the two layouts: the checkpoint puts the
# compressor inside the indexer, EasyDeL puts the indexer inside the
# compressor, and the scoring projection gains a `scorer` level. Getting this
# backwards silently lands indexer weights on the main compressor.
INDEXER_CASES = [
    (
        "layers.5.attn.indexer.compressor.wkv.weight",
        "model.layers.5.self_attn.compressor.indexer.kv_proj.weight",
    ),
    (
        "layers.5.attn.indexer.compressor.norm.weight",
        "model.layers.5.self_attn.compressor.indexer.kv_norm.weight",
    ),
    (
        "layers.5.attn.indexer.compressor.ape",
        "model.layers.5.self_attn.compressor.indexer.position_bias",
    ),
    (
        "layers.5.attn.indexer.wq_b.weight",
        "model.layers.5.self_attn.compressor.indexer.q_b_proj.weight",
    ),
    (
        "layers.5.attn.indexer.weights_proj.weight",
        "model.layers.5.self_attn.compressor.indexer.scorer.weights_proj.weight",
    ),
]


@pytest.mark.parametrize(("native", "expected"), INDEXER_CASES)
def test_indexer_nesting_is_inverted(native, expected):
    assert native_key_to_easydel(native) == expected


def test_indexer_does_not_collide_with_the_main_compressor():
    """The two compressors must not map onto the same parameter."""
    main = native_key_to_easydel("layers.5.attn.compressor.wkv.weight")
    indexer = native_key_to_easydel("layers.5.attn.indexer.compressor.wkv.weight")
    assert main != indexer
    assert "indexer" not in main
    assert indexer.endswith("compressor.indexer.kv_proj.weight")


@pytest.mark.parametrize(
    "native",
    [
        "mtp.0.attn.wq_a.weight",
        "mtp.0.ffn.experts.3.w1.weight",
        "mtp.0.markov_head.markov_w1.weight",
    ],
)
def test_mtp_stack_is_reported_unowned(native):
    """The multi-token-prediction stack is a separate optional head.

    Returning ``None`` (rather than raising, or inventing a name) keeps the
    loader's unused-key reporting truthful for the 45 ``mtp.*`` patterns.
    """
    assert native_key_to_easydel(native) is None


def test_native_per_expert_tensors_consolidate_through_from_pretrained():
    """The published per-expert tensors must reach the runtime's stacked params.

    DeepSeek ships one tensor per expert (``ffn.experts.<i>.w1``); the runtime
    wants one ``[experts, hidden, intermediate]`` parameter. Once the key
    normalizer renames them into EasyDeL's expert naming, the loader's existing
    MoE consolidation stacks them -- no DeepSeek-specific rule needed. This
    pins that the whole chain works and, critically, that expert ORDER is
    preserved: a silent permutation here routes every token to the wrong
    expert while every shape still checks out.
    """
    import easydel as ed
    import jax
    import numpy as np
    import spectrax as spx
    import torch
    from jax import numpy as jnp

    hidden, inter, experts = 32, 16, 4
    cfg = ed.DeepseekV4Config(
        num_hidden_layers=1,
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=experts,
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        vocab_size=64,
        index_n_heads=2,
        index_head_dim=8,
        o_lora_rank=8,
        o_groups=2,
        layer_types=["sliding_attention"],
        mlp_layer_types=["moe"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
    )
    cfg.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    cfg.attach_custom_arguments()
    with cfg.mesh:
        model = ed.DeepseekV4ForCausalLM(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))

    state_dict = {}
    for e in range(experts):
        # a per-expert constant offset makes a permutation detectable
        state_dict[f"layers.0.ffn.experts.{e}.w1.weight"] = (
            torch.arange(inter * hidden, dtype=torch.float32).reshape(inter, hidden) + e * 1000
        )
        state_dict[f"layers.0.ffn.experts.{e}.w3.weight"] = torch.randn(inter, hidden)
        state_dict[f"layers.0.ffn.experts.{e}.w2.weight"] = torch.randn(hidden, inter)

    converted = model.pure_transform_fn(state_dict=state_dict)
    flat = {
        ".".join(str(getattr(p, "key", p)) for p in path): v
        for path, v in jax.tree_util.tree_leaves_with_path(converted)
    }

    gate = flat["model.layers.0.mlp.experts.gate_proj.weight"]
    up = flat["model.layers.0.mlp.experts.up_proj.weight"]
    down = flat["model.layers.0.mlp.experts.down_proj.weight"]
    assert np.shape(gate) == (experts, hidden, inter)
    assert np.shape(up) == (experts, hidden, inter)
    assert np.shape(down) == (experts, inter, hidden)

    gate = np.asarray(gate)
    for e in range(experts):
        source = np.asarray(state_dict[f"layers.0.ffn.experts.{e}.w1.weight"]).T
        assert np.allclose(gate[e], source), f"expert {e} landed in the wrong slot"


def test_hash_moe_layers_do_not_declare_a_score_correction_bias():
    """A parameter a layer never reads is still one the checkpoint must supply.

    V4's first ``num_hash_layers`` MLP layers route through the frozen
    ``tid2eid`` table and never consult ``e_score_correction_bias``; DeepSeek
    ships no value for them. Declaring it anyway makes those leaves REQUIRED,
    and the load fails at the end with "abstract trainable parameter" -- after
    materializing the whole model.
    """
    import easydel as ed
    import spectrax as spx
    from easydel.utils.traversals import flatten_dict
    from jax import numpy as jnp

    cfg = ed.DeepseekV4Config(
        num_hidden_layers=2,
        hidden_size=32,
        moe_intermediate_size=16,
        n_routed_experts=4,
        num_attention_heads=4,
        head_dim=16,
        q_lora_rank=16,
        vocab_size=64,
        index_n_heads=2,
        index_head_dim=8,
        o_lora_rank=8,
        o_groups=2,
        layer_types=["sliding_attention", "sliding_attention"],
        mlp_layer_types=["hash_moe", "moe"],
        compress_rates={"compressed_sparse_attention": 2, "heavily_compressed_attention": 4},
    )
    cfg.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    cfg.attach_custom_arguments()
    with cfg.mesh:
        model = ed.DeepseekV4ForCausalLM.lazy_init(
            config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
        )

    declared = set()
    for collection, tree in spx.export(model)[1].raw().items():
        del collection
        for path in flatten_dict(tree):
            declared.add(".".join(str(p) for p in path))

    hash_bias = "model.layers.0.mlp.gate.e_score_correction_bias"
    learned_bias = "model.layers.1.mlp.gate.e_score_correction_bias"
    assert hash_bias not in declared, "hash_moe layer must not require a correction bias DeepSeek never ships"
    assert learned_bias in declared, "learned moe layer still selects with the correction bias"
    # The hash layer's own routing table is still required.
    assert "model.layers.0.mlp.gate.tid2eid" in declared
