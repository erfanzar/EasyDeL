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

"""Tests for the GLM-5-Next (glm5_next_text) model family.

The installed ``transformers`` has no ``glm5_next`` port, so HF parity is not
available; these tests run the EasyDeL-only forward / generation paths plus
checkpoint-conversion checks for the family's fused-layout ``reform_param``
rules (HF fused ``conv1d`` and fused expert ``gate_up_proj``).
"""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
import torch
from easydel.infra.sequence_packing import fold_sequence_packing_segments

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


def _tiny_kwargs(small_model_config):
    """Tiny GLM-5-Next kwargs (2 KDA layers + 2 DSA layers, mHC=2, MoE)."""
    return dict(
        vocab_size=small_model_config["vocab_size"],
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=8,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,  # NoPE
        qk_nope_head_dim=8,
        v_head_dim=8,
        n_group=1,
        topk_group=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        hidden_act=small_model_config["hidden_act"],
        max_position_embeddings=small_model_config["max_position_embeddings"],
        initializer_range=small_model_config["initializer_range"],
        rms_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=small_model_config["pad_token_id"],
        bos_token_id=small_model_config["bos_token_id"],
        eos_token_id=small_model_config["eos_token_id"],
        tie_word_embeddings=False,
        mlp_layer_types=["dense", "sparse", "dense", "sparse"],
        attention_bias=False,
        attention_dropout=0.0,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
        index_kpool_always_select_tail=True,
        layer_types=[
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        swiglu_limit=10.0,
        linear_head_dim=8,
        linear_num_heads=2,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        hc_mult=2,
        hc_eps=1e-6,
        hc_sinkhorn_iters=5,
    )


@pytest.fixture
def glm5_small_config(small_model_config):
    """Small-model dict with expert parallelism disabled.

    XLA:CPU cannot lower ``ragged-all-to-all`` (the MoE expert-parallel
    collective), so the default ``(1, 1, 1, -1, tp, 1)`` mesh fails on the
    CPU test host. This recipe shards the token batch over fsdp
    (``fsdp_is_ep_bound=False``, dp=1) and leaves ep unused — a multi-device
    mesh without expert-parallel collectives.
    """
    import jax

    device_count = jax.device_count()
    fsdp_dim = max(1, device_count)
    return {
        **small_model_config,
        "sharding_axis_dims": (1, 1, fsdp_dim, 1, 1, 1),
        "batch_size": max(small_model_config["batch_size"], fsdp_dim),
    }


@pytest.fixture
def glm5_config(glm5_small_config):
    """Create the GLM-5-Next tiny config."""
    config = ed.Glm5NextTextConfig(**_tiny_kwargs(glm5_small_config))
    config.moe_force_xla_gmm = True
    config.fsdp_is_ep_bound = False
    return config


class TestGlm5Next:
    """Test suite for the GLM-5-Next text model."""

    def test_causal_lm(self, glm5_config, glm5_small_config):
        """Test Glm5NextForCausalLM forward (logits + loss present, finite)."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="glm5_next_text",
            hf_class=None,  # transformers has no glm5_next port
            task=ed.TaskType.CAUSAL_LM,
            config=glm5_config,
            small_model_config=glm5_small_config,
        )
        assert result.success, f"GLM-5-Next CAUSAL_LM failed: {result.error_message}"

    def test_generation(self, glm5_config, glm5_small_config):
        """Test GLM-5-Next autoregressive generation through HybridCache."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="glm5_next_text",
            hf_class=None,
            config=glm5_config,
            small_model_config=glm5_small_config,
            max_new_tokens=4,
        )
        assert result.success, f"GLM-5-Next generation failed: {result.error_message}"

    def test_gradients_finite(self, glm5_config, glm5_small_config):
        """Gradients flow through KDA, DSA, mHC and MoE and stay finite."""
        config = glm5_config
        config.sharding_axis_dims = glm5_small_config["sharding_axis_dims"]
        with config.mesh:
            model = ed.Glm5NextForCausalLM(
                config=config,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                rngs=spx.Rngs(0),
            )
            graphdef, params = spx.export(model)
            ids = jnp.ones((glm5_small_config["batch_size"], 8), dtype="i4")

            def loss_fn(p):
                module = spx.bind(graphdef, p)
                return module(input_ids=ids).logits.astype(jnp.float32).sum()

            grads = jax.grad(loss_fn, allow_int=True)(params)
        flat = [g for g in jax.tree.leaves(grads) if jnp.issubdtype(g.dtype, jnp.floating)]
        assert flat, "no floating gradients produced"
        assert all(bool(jnp.isfinite(g).all()) for g in flat), "non-finite gradients"

    def test_packed_kda_segment_isolation(self, glm5_config, glm5_small_config):
        """Packed segment threading must isolate documents inside the KDA layers.

        Two documents are packed into one row and routed through the model via
        ``fold_sequence_packing_segments`` (the trainer → model contract). The
        KDA conv + delta-rule state must reset at the document boundary, so
        changing only document 1 must leave document 2's first-token output
        exactly unchanged. Both runs keep the same sequence shape and segment
        boundaries: an isolated run has a different shape and unsegmented
        conv/recurrence paths, so it need not be bit-identical even at HIGHEST
        matmul precision. Without segment ids, the same counterfactual must
        change document 2's first token. The standalone-document comparison
        remains as a relative-error check over all of document 2.
        """
        config = glm5_config
        config.sharding_axis_dims = glm5_small_config["sharding_axis_dims"]
        batch_size = glm5_small_config["batch_size"]
        len_doc1, len_doc2 = 4, 5

        with config.mesh:
            model = ed.Glm5NextForCausalLM(
                config=config,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                rngs=spx.Rngs(0),
            )
            graphdef, params = spx.export(model)
            module = spx.bind(graphdef, params)

            generator = np.random.default_rng(7)
            doc1 = generator.integers(10, 1000, size=(batch_size, len_doc1))
            doc2 = generator.integers(10, 1000, size=(batch_size, len_doc2))
            packed_ids = jnp.asarray(np.concatenate([doc1, doc2], axis=1), dtype="i4")
            # Change every doc1 token while staying in the original token range.
            # Doc2 and its absolute positions/pool boundaries are unchanged.
            counterfactual_doc1 = 10 + (doc1 - 10 + 495) % 990
            counterfactual_ids = jnp.asarray(np.concatenate([counterfactual_doc1, doc2], axis=1), dtype="i4")
            doc2_ids = jnp.asarray(doc2, dtype="i4")
            segment_ids = jnp.concatenate(
                [
                    jnp.zeros((batch_size, len_doc1), dtype="i4"),
                    jnp.ones((batch_size, len_doc2), dtype="i4"),
                ],
                axis=1,
            )

            packed_kwargs = fold_sequence_packing_segments({"input_ids": packed_ids, "segment_ids": segment_ids})
            assert "mask_info" in packed_kwargs, "segment ids must fold into mask_info"
            assert "segment_ids" not in packed_kwargs

            counterfactual_kwargs = fold_sequence_packing_segments(
                {"input_ids": counterfactual_ids, "segment_ids": segment_ids}
            )
            logits_packed = np.asarray(module(**packed_kwargs).logits)
            logits_counterfactual = np.asarray(module(**counterfactual_kwargs).logits)
            logits_alone = np.asarray(module(input_ids=doc2_ids).logits)
            logits_unthreaded = np.asarray(module(input_ids=packed_ids).logits)
            logits_unthreaded_counterfactual = np.asarray(module(input_ids=counterfactual_ids).logits)

        doc2_packed = logits_packed[:, len_doc1:]
        first_token = slice(len_doc1, len_doc1 + 1)
        first_token_diff = float(np.abs(logits_packed[:, first_token] - logits_counterfactual[:, first_token]).max())
        assert first_token_diff == 0.0, (
            f"document 2's first token must be invariant to document 1, got diff {first_token_diff}"
        )

        # Same-shape negative control: this exact prefix perturbation must be
        # observable at the boundary when segment threading is omitted.
        counterfactual_leak = float(
            np.abs(logits_unthreaded[:, first_token] - logits_unthreaded_counterfactual[:, first_token]).max()
        )
        assert counterfactual_leak > 0.0, "unthreaded first token is insensitive to the prefix perturbation"

        # Control: without threading, document 1 leaks into document 2.
        leak = float(np.abs(logits_unthreaded[:, len_doc1:] - logits_alone).max())
        assert leak > 0.0, "control run shows no leak — the probe cannot detect a regression"

        # Threaded doc2 stays far closer to the isolated run than the leaky one.
        # (Later doc2 tokens differ slightly even when threaded: the DSA k-pool
        # indexer pools over the full context, so pool membership depends on
        # packing. That is indexer context sensitivity, not state leakage.)
        threaded = float(np.abs(doc2_packed - logits_alone).max())
        assert threaded < 0.5 * leak, f"threaded diff {threaded} not clearly below leak {leak}"


def _conversion_config(glm5_config):
    """Expose the converter-facing config (dtype + reform_param)."""
    return glm5_config


class TestGlm5NextConversion:
    """Checkpoint-conversion checks for the fused-layout reform rules."""

    def _reform_param(self, glm5_config):
        """Build the model and collect its reform rules."""
        with glm5_config.mesh:
            model = ed.Glm5NextForCausalLM(
                config=glm5_config,
                dtype=jnp.float32,
                param_dtype=jnp.float32,
                precision=jax.lax.Precision.HIGHEST,
                rngs=spx.Rngs(0),
            )
            return model._get_reform_param()

    @staticmethod
    def _process(key, tensor, reform_param):
        config = {
            "embedding_layer_names": ["embed_tokens"],
            "layernorm_names": ["norm", "layernorm", "k_norm"],
            "moe_block_names": set(),
            "moe_names": set(),
            "lm_head_name": None,
            "uses_tie_word_embedding": False,
            "dtype": jnp.float32,
            "consolidated_moe_keys": set(),
            "reform_param": reform_param,
        }
        return ed.StateDictConverter.process_tensor(key, tensor, config)

    @staticmethod
    def _tuple_key(path: str):
        """Convert a dotted parameter path into the converter's tuple key."""
        return tuple(int(p) if p.isdigit() else p for p in path.split("."))

    def test_reform_param_collected(self, glm5_config):
        """The model tree declares conv / expert / shared-MLP reform rules."""
        reform_param = self._reform_param(glm5_config)
        joined = "\n".join(reform_param)
        assert "conv1d.weight$" in joined
        assert "experts.gate_up_proj$" in joined
        assert "experts.down_proj$" in joined
        # gate_up rules: 2 dense layers + 2 shared experts + 2 expert stacks.
        assert joined.count("gate_up_proj") >= 6

    def test_fused_conv_splits_per_stream(self, glm5_config):
        """HF ``conv1d.weight`` (3qkv, 1, d) splits into q/k/v flax kernels."""
        reform_param = self._reform_param(glm5_config)
        rule_key = next(k for k in reform_param if k.endswith("self_attn.conv1d.weight$"))
        hf_key = rule_key.rstrip("$")  # the incoming checkpoint key
        qkv, d_conv = 16, 4  # linear_num_heads * linear_head_dim, conv kernel
        rng = np.random.default_rng(0)
        torch_weight = torch.from_numpy(rng.standard_normal((3 * qkv, 1, d_conv)).astype(np.float32))

        results = self._process(hf_key, torch_weight, reform_param)
        by_key = {k: v for k, v in results}

        prefix = hf_key[: -len("conv1d.weight")]
        q_key = self._tuple_key(prefix + "q_conv1d.weight")
        k_key = self._tuple_key(prefix + "k_conv1d.weight")
        v_key = self._tuple_key(prefix + "v_conv1d.weight")
        assert {q_key, k_key, v_key} <= set(by_key), (set(by_key), {q_key, k_key, v_key})

        qkv_split = qkv
        np.testing.assert_allclose(np.asarray(by_key[q_key]), torch_weight[:qkv_split].permute(2, 1, 0), atol=1e-6)
        np.testing.assert_allclose(
            np.asarray(by_key[k_key]), torch_weight[qkv_split : 2 * qkv_split].permute(2, 1, 0), atol=1e-6
        )
        np.testing.assert_allclose(np.asarray(by_key[v_key]), torch_weight[2 * qkv_split :].permute(2, 1, 0), atol=1e-6)

    def test_fused_expert_gate_up_splits(self, glm5_config):
        """HF experts ``gate_up_proj`` [E, 2M, H] splits into gate/up kernels."""
        reform_param = self._reform_param(glm5_config)
        rule_key = next(k for k in reform_param if k.endswith("experts.gate_up_proj$"))
        hf_key = rule_key.rstrip("$")
        experts, moe_inter, hidden = 8, 16, 32
        rng = np.random.default_rng(1)
        fused = torch.from_numpy(rng.standard_normal((experts, 2 * moe_inter, hidden)).astype(np.float32))

        results = self._process(hf_key, fused, reform_param)
        by_key = {k: v for k, v in results}

        prefix = hf_key[: -len("gate_up_proj")]
        gate_key = self._tuple_key(prefix + "gate_proj.weight")
        up_key = self._tuple_key(prefix + "up_proj.weight")
        assert {gate_key, up_key} <= set(by_key)
        np.testing.assert_allclose(np.asarray(by_key[gate_key]), fused[:, :moe_inter, :].permute(0, 2, 1), atol=1e-6)
        np.testing.assert_allclose(np.asarray(by_key[up_key]), fused[:, moe_inter:, :].permute(0, 2, 1), atol=1e-6)

    def test_fused_expert_down_transposes(self, glm5_config):
        """HF experts ``down_proj`` [E, H, M] lands as [E, M, H] kernels."""
        reform_param = self._reform_param(glm5_config)
        rule_key = next(k for k in reform_param if k.endswith("experts.down_proj$"))
        hf_key = rule_key.rstrip("$")
        experts, moe_inter, hidden = 8, 16, 32
        rng = np.random.default_rng(2)
        down = torch.from_numpy(rng.standard_normal((experts, hidden, moe_inter)).astype(np.float32))

        results = self._process(hf_key, down, reform_param)
        by_key = {k: v for k, v in results}
        down_key = self._tuple_key(hf_key[: -len("down_proj")] + "down_proj.weight")
        assert down_key in set(by_key)
        np.testing.assert_allclose(np.asarray(by_key[down_key]), down.permute(0, 2, 1), atol=1e-6)
