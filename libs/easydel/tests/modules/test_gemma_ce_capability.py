"""Gemma trainer loss/update parity against an independent CE objective (accelerator only)."""

import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.modules.gemma import GemmaConfig, GemmaForCausalLM
from easydel.trainers.trainer._fn import training_step
from jax import numpy as jnp


@pytest.mark.parametrize("empty_microbatch", [False, True])
def test_gemma_trainer_token_weighted_ce_loss_and_update(empty_microbatch):
    """Unequal/zero-label microbatches must reproduce full-batch token-mean SGD.

    Exercise the public training step with the real Gemma forward and loss. The
    reference differentiates a direct log-softmax/gather objective, not the
    trainer's capability detector, weighting helper, or compute_loss method.
    """
    config = GemmaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    with jax.default_matmul_precision("highest"), config.mesh:
        model = GemmaForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        state = EasyDeLState.create(model=model, tx=optax.sgd(0.1), init_opt_state=True)
        input_ids = jnp.asarray([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], dtype=jnp.int32)
        labels = input_ids.at[1, : (5 if empty_microbatch else 4)].set(-100)
        batch = {"input_ids": input_ids, "labels": labels, "attention_mask": jnp.ones_like(input_ids)}

        def reference_loss(tree):
            logits = state.merge(tree)(input_ids=input_ids, attention_mask=batch["attention_mask"]).logits
            shifted_labels = labels[:, 1:]
            valid = shifted_labels != -100
            safe_labels = jnp.where(valid, shifted_labels, 0)
            log_probs = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
            token_losses = -jnp.take_along_axis(log_probs, safe_labels[..., None], axis=-1)[..., 0]
            return jnp.sum(jnp.where(valid, token_losses, 0.0)) / jnp.sum(valid)

        expected_loss, expected_gradients = jax.value_and_grad(reference_loss)(state.graphstate)
        expected_state = state.apply_gradients(grads=expected_gradients)
        updated, metrics = training_step(
            state,
            batch,
            gradient_accumulation_steps=2,
            learning_rate_fn=optax.constant_schedule(0.1),
        )

    np.testing.assert_allclose(metrics.loss, expected_loss, rtol=1e-4, atol=1e-5)
    assert int(updated.step) == int(state.step) + 1
    initial_leaves = jax.tree_util.tree_leaves(state.graphstate)
    updated_leaves = jax.tree_util.tree_leaves(updated.graphstate)
    expected_leaves = jax.tree_util.tree_leaves(expected_state.graphstate)
    assert any(not np.array_equal(a, b) for a, b in zip(initial_leaves, updated_leaves, strict=True))
    for actual, expected in zip(updated_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
