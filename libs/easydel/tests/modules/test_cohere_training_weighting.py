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

"""Public Cohere training-step loss/update regressions; accelerator computation only.

Unlike the Python-only forward-contract test, these exercise real model kernels,
compute_loss, gradient accumulation and SGD. They deliberately have no cpu_ok marker.
"""

import easydel as ed
import jax
import numpy as np
import optax
import pytest
import spectrax as spx
from easydel.infra.base_state import EasyDeLState
from easydel.infra.modeling_outputs import MoeCausalLMOutput
from easydel.modules.cohere.cohere_configuration import CohereConfig
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM
from easydel.trainers.trainer._fn import training_step
from jax import numpy as jnp


class CohereWithAuxiliaryObjective(CohereForCausalLM):
    """Custom test head with a row-mean, not token-mean, auxiliary objective."""

    # A test-only head must not replace Cohere's public factory registration
    # when pytest imports this module during collection.
    _auto_register = False

    def forward(self, input_ids=None, apply_lm_head=True, **kwargs):
        outputs = super().forward(input_ids=input_ids, apply_lm_head=apply_lm_head, **kwargs)
        # Each row touches a different embedding entry. Dropping or token-weighting
        # the zero-label rows therefore changes an observable parameter update.
        aux_loss = jnp.mean(input_ids[:, 0] + self.get_embedding().weight.value[input_ids[:, 0], 0])
        return MoeCausalLMOutput(
            logits=outputs.logits,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
            aux_loss=aux_loss,
        )


def _numpy_token_mean(logits, labels):
    """Independent shifted CE reference, including ignored-label microbatches."""
    shifted = np.asarray(logits, dtype=np.float64)[:, :-1]
    targets = labels[:, 1:]
    valid = targets != -100
    if not np.any(valid):
        return 0.0
    shifted = shifted - shifted.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    selected = np.take_along_axis(log_probs, np.where(valid, targets, 0)[..., None], axis=-1)[..., 0]
    return float(-selected[valid].mean())


@pytest.mark.parametrize("model_class", [CohereForCausalLM, CohereWithAuxiliaryObjective])
@pytest.mark.parametrize("sparse_tokens", [0, 2])
def test_cohere_training_step_preserves_token_ce_and_row_mean_aux(model_class, sparse_tokens):
    """Accumulation must match a full-batch update, not either incorrect estimator."""
    config = CohereConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        logit_scale=1.0,
        attention_dropout=0.0,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    has_aux = model_class is CohereWithAuxiliaryObjective
    token_ids = np.arange(1, 33, dtype=np.int32).reshape(4, 8)
    labels = token_ids.copy()
    labels[2:] = -100
    if sparse_tokens:
        labels[2:, -sparse_tokens:] = token_ids[2:, -sparse_tokens:]
    batch = {
        "input_ids": jnp.asarray(token_ids),
        "attention_mask": jnp.ones(token_ids.shape, dtype=jnp.int32),
        "labels": jnp.asarray(labels),
    }
    counts = [int(np.count_nonzero(labels[rows, 1:] != -100)) for rows in (slice(0, 2), slice(2, 4))]
    shares = np.asarray(counts, dtype=np.float64) / sum(counts)

    with jax.default_matmul_precision("highest"), config.mesh:
        model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        state = EasyDeLState.create(model=model, tx=optax.sgd(1.0), init_opt_state=True)
        logits = np.asarray(model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits)
        embeddings = np.asarray(model.get_embedding().weight.value)
        ce = [_numpy_token_mean(logits[rows], labels[rows]) for rows in (slice(0, 2), slice(2, 4))]
        aux = [
            float((token_ids[rows, 0] + embeddings[token_ids[rows, 0], 0]).mean()) if has_aux else 0.0
            for rows in (slice(0, 2), slice(2, 4))
        ]
        expected_loss = float(np.dot(shares, ce) + np.mean(aux))
        wrong_loss = float(np.dot(shares, np.add(ce, aux)) if has_aux else np.mean(ce))
        # Ensure the fixture would detect dropping zero-label aux or uniformly
        # averaging unequal token counts, rather than passing a vacuous parity check.
        assert abs(expected_loss - wrong_loss) > 1e-4

        @ed.ejit(static_argnums=(2,))
        def step(current_state, current_batch, accumulation):
            return training_step(
                current_state,
                current_batch,
                learning_rate_fn=optax.constant_schedule(1.0),
                gradient_accumulation_steps=accumulation,
            )

        full_state, full_metrics = step(state, batch, 1)
        accumulated_state, accumulated_metrics = step(state, batch, 2)
        sub_states = [
            step(state, {key: value[rows] for key, value in batch.items()}, 1)[0] for rows in (slice(0, 2), slice(2, 4))
        ]

    np.testing.assert_allclose(float(full_metrics.loss), expected_loss, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(float(accumulated_metrics.loss), expected_loss, rtol=1e-4, atol=1e-6)
    full_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(full_state.graphstate)]
    accumulated_leaves = [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(accumulated_state.graphstate)]
    sub_leaves = [[np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(sub.graphstate)] for sub in sub_states]
    # Plain SGD makes averages of independent sub-step parameter trees an
    # independent reference for the incorrect whole-objective weighting.
    wrong_shares = shares if has_aux else (0.5, 0.5)
    wrong_leaves = [wrong_shares[0] * left + wrong_shares[1] * right for left, right in zip(*sub_leaves, strict=True)]
    correct_error = sum(
        np.sum((actual - full) ** 2) for actual, full in zip(accumulated_leaves, full_leaves, strict=True)
    )
    wrong_error = sum(np.sum((wrong - full) ** 2) for wrong, full in zip(wrong_leaves, full_leaves, strict=True))
    assert wrong_error > 1e-8
    assert correct_error < 0.05 * wrong_error
    for actual, expected in zip(accumulated_leaves, full_leaves, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
