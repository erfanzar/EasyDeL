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

"""Tests for DBRX MoE model."""

import easydel as ed
import pytest
import spectrax as spx
import transformers
from easydel.layers.moe import BaseMoeModule
from easydel.modules.dbrx.modeling_dbrx import DbrxFFN
from packaging import version

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


class TestDBRX:
    """Test suite for DBRX MoE model."""

    @pytest.fixture
    def dbrx_config(self, small_model_config):
        """Create DBRX-specific config."""
        # transformers < 5.17 applies DBRX's expert matrices transposed, which only
        # type-checks when ffn_hidden_size == d_model; newer releases (and EasyDeL)
        # use the checkpoint orientation, so they get the real, wider FFN.
        ffn_hidden_size = small_model_config["intermediate_size"]
        if version.parse(transformers.__version__) < version.parse("5.17"):
            ffn_hidden_size = small_model_config["hidden_size"]
        return ed.DbrxConfig(
            d_model=small_model_config["hidden_size"],
            n_heads=small_model_config["num_attention_heads"],
            n_layers=small_model_config["num_hidden_layers"],
            ffn_config=ed.DbrxFFNConfig(
                ffn_hidden_size=ffn_hidden_size,
                moe_top_k=small_model_config.get("num_experts_per_tok", 4),
                moe_num_experts=small_model_config.get("num_local_experts", 16),
            ),
            attn_config=ed.DbrxAttentionConfig(),
            max_seq_len=small_model_config["max_position_embeddings"],
        )

    def test_sparse_moe_uses_grouped_dispatch(self, dbrx_config):
        block = DbrxFFN(config=dbrx_config, rngs=spx.Rngs(0))
        assert isinstance(block, BaseMoeModule)
        assert block.experts.mlp.w1.value.shape[0] == (
            dbrx_config.ffn_config.moe_num_experts * dbrx_config.ffn_config.ffn_hidden_size
        )
        assert "moe_call" in DbrxFFN.forward.__code__.co_names

    def test_standard_moe_matches_fused_moe(self, dbrx_config):
        """``MoEMethods.STANDARD_MOE`` (direct expert-layer call) equals the fused path on the same weights.

        Covers the grouped expert forward of ``DbrxExperts`` and the per-call routing hooks (softmax + weight
        renormalization) reaching the standard path.
        """
        import jax
        import jax.numpy as jnp
        import numpy as np

        dbrx_config.moe_force_xla_gmm = True
        x = jnp.asarray(np.random.default_rng(0).standard_normal((4, 8, dbrx_config.d_model)).astype(np.float32))
        results = {}
        states = {}
        with dbrx_config.mesh:
            for method in ("fused_moe", "standard_moe"):
                dbrx_config.moe_method = method
                block = DbrxFFN(
                    config=dbrx_config,
                    dtype=jnp.float32,
                    param_dtype=jnp.float32,
                    precision=jax.lax.Precision.HIGHEST,
                    rngs=spx.Rngs(0),
                )
                states[method] = jax.tree_util.tree_leaves(spx.export(block)[1])
                out, logits = block(x)
                results[method] = (np.asarray(out), np.asarray(logits))
        for a, b in zip(states["fused_moe"], states["standard_moe"], strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        np.testing.assert_allclose(results["standard_moe"][1], results["fused_moe"][1], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(results["standard_moe"][0], results["fused_moe"][0], rtol=1e-4, atol=1e-4)

    @pytest.mark.skipif(
        version.parse(transformers.__version__) < version.parse("5.17"),
        reason="transformers<5.17 applies DBRX expert matrices transposed; EasyDeL follows the checkpoint layout",
    )
    def test_causal_lm(self, dbrx_config, small_model_config):
        """Test DbrxForCausalLM."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        dbrx_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="dbrx",
            hf_class=transformers.DbrxForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=dbrx_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"DBRX CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, dbrx_config, small_model_config):
        """Test DBRX text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        dbrx_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="dbrx",
            hf_class=transformers.DbrxForCausalLM,
            config=dbrx_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"DBRX generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
