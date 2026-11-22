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

"""Base trainer step: gradient accumulation reproduces the full-batch token-mean update.

The microbatches below carry very different numbers of supervised tokens. The causal-LM loss
is a mean over valid tokens, so a uniform average of per-microbatch means is a different
estimator from the full-batch mean; the step must weight microbatches by valid-token count.
"""

from __future__ import annotations

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.trainers.trainer._fn import training_step


def _tiny_llama():
    cfg = ed.LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        mlp_bias=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    return cfg


def _batch():
    rng = np.random.default_rng(0)
    batch_size, seq_len = 4, 16
    input_ids = jnp.asarray(rng.integers(1, 120, (batch_size, seq_len)), dtype="i4")
    labels = np.asarray(input_ids).copy()
    # Microbatch 0 (rows 0,1): fully supervised. Microbatch 1 (rows 2,3): 2 supervised tokens each.
    labels[2:, :-2] = -100
    return {
        "input_ids": input_ids,
        "attention_mask": jnp.ones((batch_size, seq_len), dtype="i4"),
        "labels": jnp.asarray(labels, dtype="i4"),
    }


def _assert_grad_accum_matches_full_batch(cfg, model_cls):
    with jax.default_matmul_precision("highest"), cfg.mesh:
        model = model_cls(config=cfg, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        batch = _batch()
        lr_fn = optax.constant_schedule(1.0)

        @ed.ejit(static_argnums=(2, 3, 4, 5, 6, 7))  # pyright: ignore[reportUntypedFunctionDecorator]
        def step(st, b, loss_config, lr, partition_spec, grad_accum, ste, loss_type):
            return training_step(st, b, loss_config, lr, partition_spec, grad_accum, ste, loss_type)

        full_state, full_metrics = step(state, batch, None, lr_fn, None, 1, None, "nll")
        accum_state, accum_metrics = step(state, batch, None, lr_fn, None, 2, None, "nll")

        # Independent reference for the old (uniform) estimator, built from two separate sub-batch
        # steps. With plain SGD at lr=1 each sub-step gives ``init - g_i``, so the uniform update is
        # the mean of the two sub-step parameter trees and its loss is the mean of the two losses.
        def _sub_step(rows):
            sub = {k: v[rows] for k, v in batch.items()}
            sub_state, metrics = step(state, sub, None, lr_fn, None, 1, None, "nll")
            return sub_state, float(metrics.loss)

        sub0_state, sub0_loss = _sub_step(slice(0, 2))
        sub1_state, sub1_loss = _sub_step(slice(2, 4))
        uniform_loss = 0.5 * (sub0_loss + sub1_loss)

        full_loss = float(full_metrics.loss)
        accum_loss = float(accum_metrics.loss)
        init_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(state.graphstate)]
        full_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(full_state.graphstate)]
        accum_leaves = [np.asarray(x) for x in jax.tree_util.tree_leaves(accum_state.graphstate)]
        uniform_leaves = [
            0.5 * (np.asarray(a) + np.asarray(b))
            for a, b in zip(
                jax.tree_util.tree_leaves(sub0_state.graphstate),
                jax.tree_util.tree_leaves(sub1_state.graphstate),
                strict=True,
            )
        ]

    loss_rtol, loss_atol = 1e-4, 1e-6
    assert np.isfinite(full_loss) and np.isfinite(accum_loss)
    # The batch is constructed so the old estimator's loss is far outside the parity tolerance.
    assert abs(uniform_loss - full_loss) > 10 * (loss_rtol * abs(full_loss) + loss_atol)
    np.testing.assert_allclose(accum_loss, full_loss, rtol=loss_rtol, atol=loss_atol)

    def _update_distance(leaves_a, leaves_b):
        """L2 distance between two parameter updates (both relative to init)."""
        return float(
            np.sqrt(
                sum(np.sum(((a - i) - (b - i)) ** 2) for a, b, i in zip(leaves_a, leaves_b, init_leaves, strict=True))
            )
        )

    assert any(not np.array_equal(a, b) for a, b in zip(init_leaves, full_leaves, strict=True))
    accum_vs_full = _update_distance(accum_leaves, full_leaves)
    uniform_vs_full = _update_distance(uniform_leaves, full_leaves)
    assert uniform_vs_full > 0.0
    # The accumulated update reproduces the full-batch update, not the old uniform estimator.
    assert accum_vs_full < 0.05 * uniform_vs_full, (accum_vs_full, uniform_vs_full)
    for full, accum in zip(full_leaves, accum_leaves, strict=True):
        np.testing.assert_allclose(accum, full, rtol=1e-3, atol=1e-5)


def test_base_training_step_grad_accum_matches_full_batch_token_mean():
    _assert_grad_accum_matches_full_batch(_tiny_llama(), ed.LlamaForCausalLM)


def test_base_training_step_grad_accum_gemma_matches_full_batch_token_mean():
    """A reviewed forward override must weight unequal valid-token microbatches."""
    cfg = ed.GemmaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    _assert_grad_accum_matches_full_batch(cfg, ed.GemmaForCausalLM)
