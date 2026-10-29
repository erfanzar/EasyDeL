"""Accelerator-only DPO and distillation numerical regressions."""

from collections import namedtuple
from types import SimpleNamespace

import jax
import numpy as np
import optax
import pytest
from easydel.trainers import fused_optimizers as fused
from easydel.trainers.direct_preference_optimization_trainer.dpo_trainer import DPOTrainer
from easydel.trainers.distillation_trainer import _fn as kd
from jax import numpy as jnp


@pytest.mark.parametrize("shape,mask_shape", [((2, 3, 4), (2, 3)), ((2, 4, 3, 3), (2, 1, 3, 3))])
def test_masked_feature_mse_matches_numpy_mean(shape, mask_shape):
    values = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) / 8
    targets = jnp.zeros_like(values)
    mask = np.ones(mask_shape, dtype=np.float32)
    unmasked = kd._masked_mse(values, targets, None)
    np.testing.assert_allclose(kd._masked_mse(values, targets, jnp.asarray(mask)), unmasked, rtol=1e-6)
    mask[..., -1] = 0
    full_mask = np.broadcast_to(mask.reshape((*mask.shape, *((1,) * (len(shape) - mask.ndim)))), shape)
    expected = np.sum(np.square(np.asarray(values)) * full_mask) / full_mask.sum()
    actual = kd._masked_mse(values.astype(jnp.bfloat16), targets.astype(jnp.bfloat16), jnp.asarray(mask))
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(kd._masked_mse(values, targets, jnp.zeros(mask_shape)), 0.0)


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 4])
@pytest.mark.parametrize("shift_tokens", [False, True])
def test_maskless_chunked_loss_and_gradient_match_dense(chunk_size, shift_tokens):
    student = jnp.asarray([[[0.0, 0.5], [1.0, -0.5], [0.0, -0.5], [0.25, 1.0]]])
    teacher = jnp.asarray([[[2.0, 0.0], [0.0, 1.0], [2.0, -1.0], [1.5, 0.5]]])

    def dense(hidden):
        return kd.distillation_loss(hidden, teacher, alpha=1.0, temperature=1.0, shift_tokens=shift_tokens)[0]

    def chunked(hidden):
        return kd.chunked_distillation_loss(
            hidden,
            teacher,
            lambda x: x,
            lambda x: x,
            alpha=1.0,
            temperature=1.0,
            chunk_size=chunk_size,
            shift_tokens=shift_tokens,
        )[0]

    # Independent double-precision NumPy reference; default shift leaves three
    # true positions, so chunk sizes two/four require synthetic padding.
    def log_softmax(x):
        x = np.asarray(x, dtype=np.float64)
        x -= x.max(axis=-1, keepdims=True)
        return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))

    s = log_softmax(student[:, :-1] if shift_tokens else student)
    t = log_softmax(teacher[:, :-1] if shift_tokens else teacher)
    expected = np.mean(np.sum(np.exp(t) * (t - s), axis=-1))
    dense_loss, dense_gradient = jax.value_and_grad(dense)(student)
    loss, gradient = jax.value_and_grad(chunked)(student)
    np.testing.assert_allclose(dense_loss, expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(loss, expected, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(gradient, dense_gradient, rtol=2e-5, atol=1e-6)


class UniformModel:
    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        return SimpleNamespace(logits=jnp.zeros((*input_ids.shape, 8), dtype=jnp.float32))


@pytest.mark.parametrize("ld_alpha", [0.0, 0.4])
def test_precomputed_ld_reference_scores_match_unequal_length_numpy(ld_alpha):
    trainer = object.__new__(DPOTrainer)
    trainer.reference_state = SimpleNamespace(model=UniformModel())
    trainer.padding_value = 0
    trainer.arguments = SimpleNamespace(
        is_encoder_decoder=False,
        label_pad_token_id=-100,
        max_length=4,
        truncation_mode="keep_end",
        aux_loss_enabled=False,
        loss_type="sigmoid",
        logprob_vocab_chunk_size=None,
        use_weighting=False,
        ld_alpha=ld_alpha,
    )
    batch = {
        "prompt_input_ids": jnp.asarray([[2, 3]]),
        "prompt_attention_mask": jnp.asarray([[1, 1]]),
        "chosen_input_ids": jnp.asarray([[4, 0]]),
        "chosen_attention_mask": jnp.asarray([[1, 0]]),
        "rejected_input_ids": jnp.asarray([[4, 5]]),
        "rejected_attention_mask": jnp.asarray([[1, 1]]),
    }
    chosen, rejected = trainer.compute_reference_log_probs(batch)
    np.testing.assert_allclose(chosen, [-np.log(8)], rtol=1e-6)
    np.testing.assert_allclose(rejected, [-np.log(8) * (1 + ld_alpha)], rtol=1e-6)


Pair = namedtuple("Pair", ["left", "right"])


@pytest.mark.parametrize(
    "optimizer,kwargs",
    [
        # The fused core defaults to zero decay because the trainer factory
        # adds it separately; Optax's standalone AdamW defaults to 1e-4.
        pytest.param("adamw", {"weight_decay": 0.0}, id="adamw"),
        pytest.param("adamw", {"weight_decay": 0.02}, id="adamw-decay"),
        pytest.param("lion", {}, id="lion"),
        pytest.param("rmsprop", {}, id="rmsprop"),
    ],
)
@pytest.mark.parametrize("container", ["tuple", "namedtuple", "nested"])
def test_fused_optimizers_preserve_tuple_parameter_trees(optimizer, kwargs, container):
    a, b = jnp.asarray([1.0, -2.0]), jnp.asarray([0.5, 3.0])
    params = (a, b) if container == "tuple" else Pair(a, b)
    if container == "nested":
        params = {"stack": (Pair(a, b), {"weights": (b, a)})}
    factory = {"adamw": fused.fused_adamw, "lion": fused.fused_lion, "rmsprop": fused.fused_rmsprop}[optimizer]
    fused_tx = factory(learning_rate=0.01, **kwargs)
    reference_tx = getattr(optax, optimizer)(learning_rate=0.01, **kwargs)
    state, reference_state = fused_tx.init(params), reference_tx.init(params)
    reference_params = params
    for step in range(3):
        grads = jax.tree_util.tree_map(lambda p, step=step: p * 0.1 + step * 0.05, params)
        updates, state = jax.jit(fused_tx.update)(grads, state, params)
        expected, reference_state = reference_tx.update(grads, reference_state, reference_params)
        assert jax.tree_util.tree_structure(updates) == jax.tree_util.tree_structure(params)
        for actual_leaf, expected_leaf in zip(
            jax.tree_util.tree_leaves(updates), jax.tree_util.tree_leaves(expected), strict=True
        ):
            np.testing.assert_allclose(actual_leaf, expected_leaf, rtol=2e-5, atol=1e-7)
        params = optax.apply_updates(params, updates)
        reference_params = optax.apply_updates(reference_params, expected)
