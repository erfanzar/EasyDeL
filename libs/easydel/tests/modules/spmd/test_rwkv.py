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

"""Tests for RWKV model."""

import easydel as ed
import pytest
import spectrax as spx
import transformers
from easydel.modules.rwkv import RwkvConfig

try:
    from tests.modules.test_utils import CausalLMTester, setup_config
    from tests.modules.test_utils.model_factory import cleanup_models, create_hf_model
except ImportError:
    from tests.modules.test_utils import CausalLMTester, setup_config  # pyright: ignore[reportImplicitRelativeImport]
    from tests.modules.test_utils.model_factory import (  # pyright: ignore[reportImplicitRelativeImport]
        cleanup_models,
        create_hf_model,
    )


class TestRWKV:
    """Test suite for RWKV model."""

    @pytest.fixture
    def rwkv_config(self, small_model_config):
        """Create RWKV-specific config."""
        return RwkvConfig(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            num_hidden_layers=small_model_config["num_hidden_layers"],
            attention_hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
        )

    def test_causal_lm(self, rwkv_config, small_model_config):
        """Test RwkvForCausalLM."""
        tester = CausalLMTester()
        result = tester.run(
            module_name="rwkv",
            hf_class=transformers.RwkvForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=rwkv_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"RWKV CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, rwkv_config, small_model_config):
        """Test RWKV text generation."""
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="rwkv",
            hf_class=transformers.RwkvForCausalLM,
            task=ed.TaskType.CAUSAL_LM,
            config=rwkv_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"RWKV generation failed: {result.error_message}"

    def test_ln_out_uses_torch_default_eps(self, rwkv_config, small_model_config):
        """``ln_out`` mirrors HF's bare ``nn.LayerNorm(hidden_size)`` (eps=1e-5), not ``layer_norm_epsilon``."""
        rwkv_config.layer_norm_epsilon = 1e-3
        config = setup_config(rwkv_config, small_model_config)
        hf_model = create_hf_model(transformers.RwkvModel, config)
        assert hf_model.ln_out.eps == 1e-5

        _, module_class = ed.get_modules_by_type("rwkv", ed.TaskType.BASE_MODULE)
        with config.mesh:
            ed_model = module_class.lazy_init(
                config=config,
                dtype=small_model_config["dtype"],
                param_dtype=small_model_config["dtype"],
                precision=small_model_config["precision"],
                rngs=spx.Rngs(0),
            )
        assert ed_model.ln_out.epsilon == hf_model.ln_out.eps

        cleanup_models(hf_model)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
