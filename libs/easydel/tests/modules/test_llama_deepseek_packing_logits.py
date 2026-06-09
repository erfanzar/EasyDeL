from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_platform_name", "cpu")

ATOL = 1e-3


def _packed_equivalence(model, *, vocab_size: int):
    doc_a = np.array([[5, 9, 2, 7, 1]], dtype=np.int32)
    doc_b = np.array([[3, 8, 4]], dtype=np.int32)
    packed = np.concatenate([doc_a, doc_b], axis=1)
    segment_ids = np.array([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=np.int32)

    logits_a = np.asarray(model(input_ids=jnp.asarray(doc_a)).logits.astype(jnp.float32))[0]
    logits_b = np.asarray(model(input_ids=jnp.asarray(doc_b)).logits.astype(jnp.float32))[0]
    logits_packed = np.asarray(
        model(input_ids=jnp.asarray(packed), segment_ids=jnp.asarray(segment_ids)).logits.astype(jnp.float32)
    )[0]

    assert logits_packed.shape == (8, vocab_size)
    assert np.all(np.isfinite(logits_packed))

    delta_a = float(np.max(np.abs(logits_packed[: doc_a.shape[1]] - logits_a)))
    delta_b = float(np.max(np.abs(logits_packed[doc_a.shape[1] :] - logits_b)))
    assert delta_a < ATOL, f"packed docA logits diverged from independent logits, max|delta|={delta_a:.2e}"
    assert delta_b < ATOL, f"packed docB logits diverged from independent logits, max|delta|={delta_b:.2e}"


def _llama_model():
    import spectrax as spx

    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        scan_layers=False,
    )
    return LlamaForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32), 128


def _deepseek_v2_model():
    import spectrax as spx

    from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
    from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2ForCausalLM

    config = DeepseekV2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=None,
        n_routed_experts=None,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        first_k_dense_replace=2,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        use_scan_mlp=False,
        scan_layers=False,
        layer_types=["full_attention", "full_attention"],
    )
    return DeepseekV2ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32), 128


def _deepseek_v3_model():
    import spectrax as spx

    from easydel.modules.deepseek_v3.deepseek_configuration import DeepseekV3Config
    from easydel.modules.deepseek_v3.modeling_deepseek import DeepseekV3ForCausalLM

    config = DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=None,
        n_routed_experts=None,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=16,
        first_k_dense_replace=2,
        num_nextn_predict_layers=0,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        scan_layers=False,
        layer_types=["full_attention", "full_attention"],
    )
    return DeepseekV3ForCausalLM(config=config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32), 128


@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(_llama_model, id="llama"),
        pytest.param(_deepseek_v2_model, id="deepseek_v2"),
        pytest.param(_deepseek_v3_model, id="deepseek_v3"),
    ],
)
def test_packed_logits_match_independent_logits(model_factory):
    model, vocab_size = model_factory()
    _packed_equivalence(model, vocab_size=vocab_size)
