"""Accelerator-only checks for pairwise generation masks and shared KV history."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.modules.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp


@pytest.mark.parametrize("target_length", [2, 3, 5])
def test_generation_preserves_pairwise_prompt_visibility_positions_and_axes(target_length):
    prompt_mask = np.array([[[[True, False, False], [True, True, False], [False, True, True]]]])
    positions = jnp.array([[7, 8, 9]], dtype=jnp.int32)
    original = MaskInfo.from_attention_mask(
        jnp.asarray(prompt_mask),
        q_positions=positions,
        kv_positions=positions,
        batch_axis_name="custom_batch",
        sequence_axis_name=None,
    )

    class MaskPreparationModel(EasyGenerationMixin):
        def init_cache(self, **kwargs):
            return None

        def prepare_inputs_for_call(self, **kwargs):
            return kwargs

    prepared = MaskPreparationModel().prepare_inputs_for_generation(
        jnp.array([[2, 7, 11]], dtype=jnp.int32),
        max_length=target_length,
        pad_token_id=0,
        starts=0,
        mask_info=original,
    )
    result = prepared["mask_info"]
    actual = np.asarray(result.attention_mask)
    retained = min(3, target_length)
    np.testing.assert_array_equal(actual[..., :retained, :retained], prompt_mask[..., :retained, :retained])
    expected_positions = np.arange(7, 7 + target_length)[None, :]
    np.testing.assert_array_equal(np.asarray(result.q_positions), expected_positions)
    np.testing.assert_array_equal(np.asarray(result.kv_positions), expected_positions)
    assert result.batch_axis_name == "custom_batch"
    assert result.sequence_axis_name is None
    assert result.causal_mask_baked_in
    if target_length > 3:
        # New decode rows attend the valid past; original blocked prompt edges
        # remain blocked rather than being reconstructed from padding flags.
        np.testing.assert_array_equal(actual[0, 0, -1], np.ones(target_length, dtype=bool))
        assert not actual[0, 0, 2, 0]
    np.testing.assert_array_equal(np.asarray(original.attention_mask), prompt_mask)
    np.testing.assert_array_equal(np.asarray(original.q_positions), np.array([[7, 8, 9]]))


# Isolate cache/history semantics from shape-dependent projection rounding.
# Explicit model precision under low ambient precision is a separate contract,
# covered without numerical execution elsewhere.
@jax.default_matmul_precision("highest")
def test_homogeneous_gemma4_shared_cache_matches_full_history():
    config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        global_head_dim=64,
        max_position_embeddings=32,
        layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=1,
        attn_mechanism="vanilla",
        # Model dtype/precision do not override config-owned attention and KV
        # storage dtypes. Keep both paths fp32 for strict cache-history parity.
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
    )
    model = Gemma4ForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    # Batch four is divisible by the real four-chip test mesh.
    tokens = jnp.broadcast_to(jnp.array([[2, 7, 11]], dtype=jnp.int32), (4, 3))
    kwargs = model.prepare_inputs_for_generation(tokens, max_length=8, pad_token_id=0)
    assert kwargs["past_key_values"].views[0].key.dtype == jnp.float32
    assert kwargs["past_key_values"].views[0].value.dtype == jnp.float32
    assert kwargs["past_key_values"].views[1] is None
    cached = model._call_generation_model_step(model, tokens, kwargs)
    history = tokens
    for token_id in (13, 17):
        kwargs = model.update_inputs_for_generation(cached, kwargs)
        next_token = jnp.full((4, 1), token_id, dtype=jnp.int32)
        cached = model._call_generation_model_step(model, next_token, kwargs)
        history = jnp.concatenate([history, next_token], axis=1)
        # Gemma4 always returns CausalLMOutput; its native forward does not
        # accept HF's return_dict/use_cache flags. Omitting past_key_values
        # recomputes the full history without the incremental cache buffers.
        reference = model(input_ids=history, past_key_values=None)
        assert all(view is None for view in reference.past_key_values.views)
        # Check the decoder before the tied head as well, so a failure separates
        # cache/history semantics from shape-dependent output-projection numerics.
        np.testing.assert_allclose(
            np.asarray(cached.last_hidden_state[:, -1]),
            np.asarray(reference.last_hidden_state[:, -1]),
            rtol=1e-4,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            np.asarray(cached.logits[:, -1]), np.asarray(reference.logits[:, -1]), rtol=1e-4, atol=1e-4
        )
        assert cached.past_key_values.views[1] is None
