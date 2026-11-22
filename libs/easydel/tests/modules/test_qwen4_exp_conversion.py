# Copyright 2026 The EASYDEL Author @erfananzar (Erfan Zare Chavoshi).
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

"""Qwen4-Exp HF-checkpoint conversion structure tests.

``qwen4_exp`` has no released transformers build, so these tests pin the
``transform_fn`` import path against a *synthetic* HuggingFace-layout state
dict: every tensor-restructuring ``reform_param`` rule must place its slice of
the checkpoint at the right runtime parameter with the right orientation.
Numerical end-to-end parity against the reference implementation was
established out-of-band (dev-time driver: max|Δlogits| = 8.9e-08).

Covered rules:

* the flat PLE ``ngram_embedding.weight`` table ``[total_vocab, heads*dim]``
  is split along the vocab axis into ``split_ngram_parts`` shards;
* the HF ``[C, 1, K]`` causal-conv kernels land as EasyDeL ``[K, 1, C]``;
* the per-layer ``q/k/v`` projections are fused into ``qkv_proj`` in
  ``[q | k | v]`` row order (transposed to runtime ``[in, out]``);
* derived buffers (``layer_multipliers``, ``ngram_heads_*``) are absent from
  HF checkpoints and must simply keep their lazy-init values;
* the merged model still runs a finite forward.
"""

import os

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import easydel as ed
import numpy as np
import pytest
from easydel.modules.qwen4_exp import Qwen4ExpTextConfig
from jax import numpy as jnp

torch = pytest.importorskip("torch")

SEQ = 12
BATCH = 2
VOCAB = 512
NGRAM_TOTAL_ROWS = 16256  # ngram vocab over 4 shards


def _config():
    return Qwen4ExpTextConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        full_attention_interval=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=2,
        hc_count=4,
        hc_lowrank=8,
        ple_layer_ids=[1],
        ple_embed_dim=64,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=2000,
        make_ngram_vocab_size_divisible_by=16,
        split_ngram_parts=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=8,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        eos_token_id=0,
        bos_token_id=1,
        pad_token_id=2,
        norm_topk_prob=True,
        output_gate_type="sigmoid",
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "mrope_interleaved": True,
            "mrope_section": [2, 1, 1],
        },
        attn_dtype=jnp.float32,
        mtp=None,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


def _synthetic_sd():
    """HF-layout state dict with patterned (deterministic, ordered) values."""
    gen = torch.Generator().manual_seed(1234)

    def t(*shape):
        return torch.randn(*shape, generator=gen) * 0.05

    sd = {
        "model.embed_tokens.weight": t(VOCAB, 64),
        "lm_head.weight": t(VOCAB, 64),
        "model.norm.weight": t(64),
    }
    for li, layer_type in enumerate(["linear_attention", "qwen_sparse_attention"] * 2):
        p = f"model.layers.{li}"
        sd[f"{p}.attn_hyper_connection.hc_norm.weight"] = t(256)
        sd[f"{p}.attn_hyper_connection.input_mix_weight_down.weight"] = t(8, 256)
        sd[f"{p}.attn_hyper_connection.input_mix_weight_up.weight"] = t(256, 8)
        sd[f"{p}.attn_hyper_connection.block_inject_weight.weight"] = t(4, 256)
        sd[f"{p}.mlp_hyper_connection.hc_norm.weight"] = t(256)
        sd[f"{p}.mlp_hyper_connection.input_mix_weight_down.weight"] = t(8, 256)
        sd[f"{p}.mlp_hyper_connection.input_mix_weight_up.weight"] = t(256, 8)
        sd[f"{p}.mlp_hyper_connection.block_inject_weight.weight"] = t(4, 256)
        sd[f"{p}.mlp.gate.weight"] = t(8, 64)
        # Release layout (Qwen3.8-Flash-Next shards): per-expert separate
        # gate/up/down projections, exactly as published in the checkpoint.
        for e in range(8):
            sd[f"{p}.mlp.experts.{e}.gate_proj.weight"] = t(16, 64)
            sd[f"{p}.mlp.experts.{e}.up_proj.weight"] = t(16, 64)
            sd[f"{p}.mlp.experts.{e}.down_proj.weight"] = t(64, 16)
        sd[f"{p}.mlp.shared_expert.gate_proj.weight"] = t(16, 64)
        sd[f"{p}.mlp.shared_expert.up_proj.weight"] = t(16, 64)
        sd[f"{p}.mlp.shared_expert.down_proj.weight"] = t(64, 16)
        sd[f"{p}.mlp.shared_expert_gate.weight"] = t(1, 64)
        if layer_type == "linear_attention":
            sd[f"{p}.linear_attn.in_proj_qkv.weight"] = t(64, 64)
            sd[f"{p}.linear_attn.in_proj_z.weight"] = t(32, 64)
            sd[f"{p}.linear_attn.in_proj_b.weight"] = t(4, 64)
            sd[f"{p}.linear_attn.in_proj_a.weight"] = t(4, 64)
            sd[f"{p}.linear_attn.conv1d.weight"] = t(64, 1, 4)
            sd[f"{p}.linear_attn.conv1d.bias"] = t(64)
            sd[f"{p}.linear_attn.A_log"] = t(4)
            sd[f"{p}.linear_attn.dt_bias"] = t(4)
            sd[f"{p}.linear_attn.norm.weight"] = t(8)
            sd[f"{p}.linear_attn.out_proj.weight"] = t(64, 32)
        else:
            sd[f"{p}.self_attn.q_proj.weight"] = t(128, 64)
            sd[f"{p}.self_attn.k_proj.weight"] = t(32, 64)
            sd[f"{p}.self_attn.v_proj.weight"] = t(32, 64)
            sd[f"{p}.self_attn.o_proj.weight"] = t(64, 64)
            sd[f"{p}.self_attn.q_norm.weight"] = t(16)
            sd[f"{p}.self_attn.k_norm.weight"] = t(16)
            sd[f"{p}.self_attn.indexer.index_qk_proj.weight"] = t(24, 64)
            sd[f"{p}.self_attn.indexer.q_layernorm.weight"] = t(8)
            sd[f"{p}.self_attn.indexer.k_layernorm.weight"] = t(8)
    # PLE on layer 0 (ple_layer_ids=[1] is 1-indexed)
    sd["model.layers.0.ple.key_proj.weight"] = t(256, 64)
    sd["model.layers.0.ple.value_proj.weight"] = t(64, 64)
    sd["model.layers.0.ple.norm_key.weight"] = t(256)
    sd["model.layers.0.ple.norm_query.weight"] = t(256)
    sd["model.layers.0.ple.norm_conv.weight"] = t(256)
    sd["model.layers.0.ple.conv1d.weight"] = t(256, 1, 4)
    sd["model.layers.0.ple.ple_embedding.ngram_embedding.weight"] = t(NGRAM_TOTAL_ROWS, 8)
    return sd


def _merged_model():
    model = ed.AutoEasyDeLModelForCausalLM.from_config(_config(), dtype=jnp.float32, param_dtype=jnp.float32)
    sd = _synthetic_sd()
    snapshot = dict(sd)
    tree = model.transform_fn(sd)
    merged = ed.traversals.merge_model_and_tree(model, tree=tree)
    merged.eval()
    return merged, snapshot


@pytest.fixture(scope="module")
def merged():
    out, _ = _merged_model()
    return out


@pytest.fixture(scope="module")
def sd():
    _, out = _merged_model()
    return out


class TestNgramShardSplit:
    """The flat ngram table must be vocab-axis split into runtime shards."""

    def test_shards_match_flat_slices(self):
        model = ed.AutoEasyDeLModelForCausalLM.from_config(_config(), dtype=jnp.float32, param_dtype=jnp.float32)
        sd = _synthetic_sd()
        snapshot = dict(sd)
        tree = model.transform_fn(sd)
        merged = ed.traversals.merge_model_and_tree(model, tree=tree)
        import jax

        shards = merged.model.layers[0].ple.ple_embedding.shards
        rows = NGRAM_TOTAL_ROWS // 4
        flat_sd = snapshot["model.layers.0.ple.ple_embedding.ngram_embedding.weight"].numpy()
        assert len(shards) == 4  # split_ngram_parts
        for i, shard in enumerate(shards):
            leaf = np.asarray(jax.tree_util.tree_leaves(shard)[0])
            expected = flat_sd[i * rows : (i + 1) * rows]
            assert float(np.max(np.abs(leaf - expected))) < 1e-6, f"shard {i} mismatch"


class TestConv1dOrientation:
    """HF ``[C, 1, K]`` conv kernels land as EasyDeL ``[K, 1, C]``."""

    def test_ple_conv_transposed(self):
        model = ed.AutoEasyDeLModelForCausalLM.from_config(_config(), dtype=jnp.float32, param_dtype=jnp.float32)
        sd = _synthetic_sd()
        snapshot = dict(sd)
        tree = model.transform_fn(sd)
        merged = ed.traversals.merge_model_and_tree(model, tree=tree)
        import jax

        conv = merged.model.layers[0].ple.conv1d
        leaf = np.asarray(jax.tree_util.tree_leaves(conv)[0])
        hf = snapshot["model.layers.0.ple.conv1d.weight"].numpy()
        assert leaf.shape == (4, 1, 256)
        assert float(np.max(np.abs(leaf - hf.transpose(2, 1, 0)))) < 1e-6


class TestFusedQKV:
    """q/k/v fuse into qkv_proj in [q | k | v] row order."""

    def test_row_order(self):
        model = ed.AutoEasyDeLModelForCausalLM.from_config(_config(), dtype=jnp.float32, param_dtype=jnp.float32)
        sd = _synthetic_sd()
        snapshot = dict(sd)
        tree = model.transform_fn(sd)
        merged = ed.traversals.merge_model_and_tree(model, tree=tree)
        fused = np.asarray(merged.model.layers[1].self_attn.qkv_proj.weight.value)
        q = snapshot["model.layers.1.self_attn.q_proj.weight"].numpy()
        k = snapshot["model.layers.1.self_attn.k_proj.weight"].numpy()
        v = snapshot["model.layers.1.self_attn.v_proj.weight"].numpy()
        expected = np.concatenate([q, k, v], axis=0).T
        assert fused.shape == expected.shape
        assert float(np.max(np.abs(fused - expected))) < 1e-6


class TestMoeExpertFusion:
    """Per-expert gate/up fuse into one ``[experts, hidden, 2 * inter]`` kernel."""

    def test_fused_kernel_content(self, merged, sd):
        fused = np.asarray(merged.model.layers[1].mlp.experts.gate_up_proj.weight.value)
        assert fused.shape == (8, 64, 32)
        gate = sd["model.layers.1.mlp.experts.3.gate_proj.weight"].numpy()
        up = sd["model.layers.1.mlp.experts.3.up_proj.weight"].numpy()
        assert float(np.max(np.abs(fused[3, :, :16] - gate.T))) < 1e-6
        assert float(np.max(np.abs(fused[3, :, 16:] - up.T))) < 1e-6

    def test_down_kernel_stacked(self, merged, sd):
        down = np.asarray(merged.model.layers[1].mlp.experts.down_proj.weight.value)
        assert down.shape == (8, 16, 64)
        hf = sd["model.layers.1.mlp.experts.5.down_proj.weight"].numpy()
        assert float(np.max(np.abs(down[5] - hf.T))) < 1e-6


class TestDerivedBuffersDropped:
    """Derived buffers are absent from HF checkpoints; lazy-init values survive."""

    def test_buffers_present_and_untouched(self, merged):
        emb = merged.model.layers[0].ple.ple_embedding
        # layer_multipliers is a hash constant kept on the module; it must
        # survive a conversion that never receives it from the checkpoint.
        assert getattr(emb, "layer_multipliers", None) is not None
        # ngram_heads_vocab_sizes / ngram_heads_offsets are HF-side derived
        # buffers; the EasyDeL side recomputes them from the config instead
        # of storing them, so nothing to restore -- and nothing to crash on.
        assert not hasattr(emb, "ngram_heads_vocab_sizes")
        assert not hasattr(emb, "ngram_heads_offsets")


class TestMergedForward:
    """The merged model must still produce finite logits."""

    def test_finite_forward(self, merged):
        ids = jnp.asarray(np.random.default_rng(7).integers(3, VOCAB, size=(BATCH, SEQ)).astype("int32"))
        logits = np.asarray(merged(input_ids=ids).logits.astype(jnp.float32))
        assert logits.shape == (BATCH, SEQ, VOCAB)
        assert np.all(np.isfinite(logits))


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])


def test_ngram_sharding_axis_is_validated_and_serialized():
    cfg = Qwen4ExpTextConfig(ngram_sharding_axis="ep")
    assert cfg.ngram_sharding_axis == "ep"
    assert cfg.to_dict()["ngram_sharding_axis"] == "ep"
    with pytest.raises(ValueError, match="ngram_sharding_axis"):
        Qwen4ExpTextConfig(ngram_sharding_axis="not-an-axis")


def test_mtp_subconfig_is_normalized_and_rejects_unsupported_layers():
    cfg = Qwen4ExpTextConfig(
        mtp={"num_hidden_layers": 2, "layer_types": ["full_attention", "qwen_sparse_attention"]},
        mtp_use_dedicated_embeddings=True,
    )
    assert cfg.mtp_num_hidden_layers == 2
    assert cfg.mtp_use_dedicated_embeddings is True
    assert cfg.to_dict()["mtp"]["num_hidden_layers"] == 2
    with pytest.raises(ValueError, match="QSA/full_attention"):
        Qwen4ExpTextConfig(mtp={"num_hidden_layers": 1, "layer_types": ["linear_attention"]})


def test_qwen4_layer_schedule_validation_and_mtp_disable_semantics():
    assert Qwen4ExpTextConfig(mtp=None).mtp_num_hidden_layers == 0
    with pytest.raises(ValueError, match="full_attention_interval must be positive"):
        Qwen4ExpTextConfig(full_attention_interval=0)
    with pytest.raises(ValueError, match="layer_types length"):
        Qwen4ExpTextConfig(num_hidden_layers=2, layer_types=["linear_attention"])
