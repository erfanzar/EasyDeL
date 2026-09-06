"""Actual quantized Llama optimizer step with codes explicitly frozen."""

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.layers.quantization import QuantizationConfig
from easydel.layers.quantization._quants import EasyQuantizer


@pytest.mark.parametrize("mode", ["w4a16", "w8a16", "w4a4", "w8a8"])
def test_quantized_lm_step_updates_scales_and_preserves_codes(mode):
    cfg = ed.LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=64,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    with cfg.mesh:
        model = ed.LlamaForCausalLM(
            cfg, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(14)
        )
        model = EasyQuantizer(quantization_config=QuantizationConfig.for_matmul(mode)).apply_quantization(
            model, verbose=False
        )
        state = EasyDeLState.create(
            model=model, trainable_selector=spx.path_endswith("quant_scales"), tx=optax.sgd(1e-3), init_opt_state=True
        )
        ids = jnp.array([[1, 7, 9, 21, 3, 11, 17, 2]], jnp.int32)

        def loss(gs):
            logits = state.merge(gs)(input_ids=ids).logits.astype(jnp.float32)
            return optax.softmax_cross_entropy_with_integer_labels(logits[:, :-1], ids[:, 1:]).mean()

        value, grads = jax.jit(jax.value_and_grad(loss))(state.graphstate)
        assert jnp.isfinite(value)
        leaves = jax.tree.leaves(grads)
        assert leaves and all(np.isfinite(g).all() for g in leaves)
        assert any(np.any(np.asarray(g) != 0) for g in leaves)
        updated = state.apply_gradients(grads=grads)
        before = jax.tree.leaves(state.graphstate)
        after = jax.tree.leaves(updated.graphstate)
        assert any(np.any(np.asarray(a) != np.asarray(b)) for a, b in zip(before, after, strict=True))
        for a, b in zip(jax.tree.leaves(state.graphother), jax.tree.leaves(updated.graphother), strict=True):
            np.testing.assert_array_equal(a, b)
