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

"""Tests for the Mistral4 model (DeepSeek-V3-shaped MLA + MoE from Mistral AI)."""

import easydel as ed
import jax
import pytest
import transformers

try:
    from tests.modules.test_utils import CausalLMTester
except ImportError:
    from tests.modules.test_utils import CausalLMTester  # pyright: ignore[reportImplicitRelativeImport]


def _resolve_hf_class(top_level_name: str, module_path: str, class_name: str):
    cls = getattr(transformers, top_level_name, None)
    if cls is not None:
        return cls
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name, None)
    except Exception:
        return None


class TestMistral4:
    """Test suite for the Mistral4 MoE model."""

    @pytest.fixture
    def small_model_config(self, small_model_config):
        """Override the shared mesh with a pure tensor-parallel layout.

        The default ``(1, 1, 1, -1, tp, 1)`` layout resolves the expert axis
        (fsdp x ep) to > 1 on multi-device hosts, and the fused-MoE dispatch
        then lowers ``ragged-all-to-all`` which XLA:CPU cannot compile (the
        known CPU limitation that also fails the other MoE families' parity
        tests). Putting every device on TP keeps the expert axis at 1, works
        for the batch-1 generation path (dp=1), and exercises the
        TP-interleaved fused expert layout at the full device count.
        """
        return {
            **small_model_config,
            "sharding_axis_dims": (1, 1, 1, 1, jax.device_count(), 1),
        }

    @pytest.fixture
    def hf_mistral4_class(self):
        """Resolve the HF Mistral4ForCausalLM class (ships with transformers 5.5)."""
        return _resolve_hf_class(
            top_level_name="Mistral4ForCausalLM",
            module_path="transformers.models.mistral4.modeling_mistral4",
            class_name="Mistral4ForCausalLM",
        )

    @pytest.fixture
    def mistral4_config(self, small_model_config):
        """Create a Mistral4-specific config.

        Uses default ``rope_interleave=True`` and a small YaRN window
        (``original_max_position_embeddings=64`` < ``sequence_length=128``)
        so both the interleaved RoPE pairing and the llama-4 attention
        temperature (non-unit for positions >= 64) are exercised by the
        parity check. ``first_k_dense_replace=1`` covers the dense/MoE
        layer split. Eight heads keep every projection divisible by the
        pure-TP mesh (tp = 8 on the standard 8-device CPU trio).
        """
        return ed.Mistral4Config(
            vocab_size=small_model_config["vocab_size"],
            hidden_size=small_model_config["hidden_size"],
            intermediate_size=small_model_config["intermediate_size"],
            moe_intermediate_size=64,
            num_hidden_layers=small_model_config["num_hidden_layers"],
            num_attention_heads=8,
            num_key_value_heads=8,
            n_shared_experts=1,
            n_routed_experts=8,
            routed_scaling_factor=1.0,
            kv_lora_rank=16,
            q_lora_rank=32,
            qk_rope_head_dim=16,
            qk_nope_head_dim=16,
            v_head_dim=32,
            n_group=2,
            topk_group=1,
            num_experts_per_tok=2,
            first_k_dense_replace=1,
            norm_topk_prob=True,
            hidden_act=small_model_config["hidden_act"],
            max_position_embeddings=small_model_config["max_position_embeddings"],
            initializer_range=small_model_config["initializer_range"],
            rms_norm_eps=small_model_config["rms_norm_eps"],
            use_cache=small_model_config["use_cache"],
            pad_token_id=small_model_config["pad_token_id"],
            bos_token_id=small_model_config["bos_token_id"],
            eos_token_id=small_model_config["eos_token_id"],
            pretraining_tp=1,
            tie_word_embeddings=small_model_config["tie_word_embeddings"],
            rope_parameters={
                "rope_type": "yarn",
                "rope_theta": 10000.0,
                "factor": 2.0,
                "original_max_position_embeddings": 64,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "llama_4_scaling_beta": 0.1,
            },
            rope_interleave=True,
            attention_bias=small_model_config["attention_bias"],
            attention_dropout=small_model_config["attention_dropout"],
        )

    def test_causal_lm(self, mistral4_config, small_model_config, hf_mistral4_class):
        """Test Mistral4ForCausalLM logits/loss parity against the HF reference."""
        if hf_mistral4_class is None:
            pytest.skip("HF Mistral4ForCausalLM not available")
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        mistral4_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.run(
            module_name="mistral4",
            hf_class=hf_mistral4_class,
            task=ed.TaskType.CAUSAL_LM,
            config=mistral4_config,
            small_model_config=small_model_config,
        )
        assert result.success, f"Mistral4 CAUSAL_LM failed: {result.error_message or result.comparison.details}"

    def test_generation(self, mistral4_config, small_model_config, hf_mistral4_class):
        """Test Mistral4 text generation."""
        if hf_mistral4_class is None:
            pytest.skip("HF Mistral4ForCausalLM not available")
        # Use XLA GMM to avoid Pallas TPU grouped_matmul kernel constraints
        mistral4_config.moe_force_xla_gmm = True
        tester = CausalLMTester()
        result = tester.test_generation(
            module_name="mistral4",
            hf_class=hf_mistral4_class,
            config=mistral4_config,
            small_model_config=small_model_config,
            max_new_tokens=16,
        )
        assert result.success, f"Mistral4 generation failed: {result.error_message}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-s"])
