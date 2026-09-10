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

"""Tests for the HYV3 (Tencent Hunyuan V3) model.

transformers 5.5.0 does not ship ``HYV3ForCausalLM`` (hy_v3 lives on
transformers main), so HF parity is gated on
``hasattr(transformers, "HYV3ForCausalLM")`` — the tests run EasyDeL-only
(``hf_class=None``) today and automatically upgrade to full logits parity
once transformers ships the class.
"""

import easydel as ed
import pytest
import transformers

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


# hy_v3 is not in transformers 5.5.0; parity activates when it lands.
HF_HYV3_CLASS = getattr(transformers, "HYV3ForCausalLM", None)


class TestHyV3:
    """Test suite for the HYV3 model."""

    @pytest.fixture
    def hy_v3_small_config(self, small_model_config):
        """Return a test-config dict with the ``ep`` mesh axis folded into ``fsdp``.

        XLA:CPU cannot lower the ``ragged-all-to-all`` collective the fused MoE
        path emits when the expert mesh has ``ep > 1`` (the same pre-existing
        limit that affects the other MoE model tests on CPU). Together with
        ``fsdp_is_ep_bound = False`` on the model config this keeps the expert
        mesh at ``ep=1`` (no all-to-all) while still exercising TP sharding and
        the fsdp axis on the 8-fake-device CPU mesh; ep > 1 routing itself is
        hardware-validated.
        """
        local_cfg = dict(small_model_config)
        dims = local_cfg["sharding_axis_dims"]
        local_cfg["sharding_axis_dims"] = (dims[0], dims[1], dims[3], 1, dims[4], dims[5])
        return local_cfg

    @pytest.fixture
    def hy_v3_config(self, small_model_config):
        """Create a HYV3-specific config covering one dense and one sparse layer."""
        num_hidden_layers = small_model_config["num_hidden_layers"]
        return ed.HyV3Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=small_model_config["num_attention_heads"],
            num_key_value_heads=small_model_config["num_key_value_heads"],
            intermediate_size=small_model_config["intermediate_size"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            head_dim=small_model_config["hidden_size"] // small_model_config["num_attention_heads"],
            num_experts=small_model_config.get("num_experts", 8),
            num_experts_per_tok=2,
            num_shared_experts=1,
            moe_intermediate_size=64,
            router_scaling_factor=2.826,
            enable_moe_fp32_combine=True,
            # First layer dense, rest sparse — the HF default schedule.
            mlp_layer_types=["dense"] + ["sparse"] * (num_hidden_layers - 1),
        )

    def test_causal_lm(self, hy_v3_config, hy_v3_small_config):
        """Test HyV3ForCausalLM (EasyDeL-only until transformers ships hy_v3)."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        hy_v3_config.moe_force_xla_gmm = True
        # Keep the expert mesh at ep=1 on CPU (see hy_v3_small_config).
        hy_v3_config.fsdp_is_ep_bound = False
        tester = CausalLMTester()
        result = tester.run(
            module_name="hy_v3",
            hf_class=HF_HYV3_CLASS,
            task=ed.TaskType.CAUSAL_LM,
            config=hy_v3_config,
            small_model_config=hy_v3_small_config,
        )
        assert result.success, f"HyV3 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, hy_v3_config, hy_v3_small_config):
        """Test HYV3 text generation."""
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        hy_v3_config.moe_force_xla_gmm = True
        # Keep the expert mesh at ep=1 on CPU (see hy_v3_small_config).
        hy_v3_config.fsdp_is_ep_bound = False
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="hy_v3",
            hf_class=HF_HYV3_CLASS,
            config=hy_v3_config,
            small_model_config=hy_v3_small_config,
            max_new_tokens=16,
        )
        assert result.success, f"HyV3 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
