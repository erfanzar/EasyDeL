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

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from easydel.trainers.direct_preference_optimization_trainer._fn import (
    _ld_logp_weights,
    compute_dpo_losses,
    get_loss_function,
)
from easydel.trainers.direct_preference_optimization_trainer.dpo_config import DPOConfig
from easydel.trainers.direct_preference_optimization_trainer.dpo_trainer import DPOTrainer
from easydel.trainers.trainer.trainer import Trainer


class _State:
    def __init__(self, graphstate):
        self.graphstate = graphstate

    def replace(self, **kwargs):
        return type(self)(kwargs.get("graphstate", self.graphstate))


def test_dpo_bco_pair_matches_trl_uncentered_rewards():
    chosen_logps = jnp.array([-0.15, -1.10, -0.35], dtype=jnp.float32)
    rejected_logps = jnp.array([-1.30, -0.95, -1.40], dtype=jnp.float32)
    ref_chosen_logps = jnp.array([-0.45, -1.00, -0.60], dtype=jnp.float32)
    ref_rejected_logps = jnp.array([-1.10, -1.20, -1.55], dtype=jnp.float32)
    beta = 0.3

    loss_fn = get_loss_function("bco_pair", beta=beta, label_smoothing=0.0)
    actual = loss_fn(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta,
        0.0,
    )

    chosen_rewards = beta * (chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (rejected_logps - ref_rejected_logps)
    expected = -jax_logsigmoid(chosen_rewards) - jax_logsigmoid(-rejected_rewards)

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_dpo_exo_pair_rejects_zero_label_smoothing():
    with pytest.raises(ValueError, match=r"Label smoothing must be greater than 0\.0"):
        DPOConfig(loss_type="exo_pair")


def test_dpo_robust_rejects_degenerate_label_smoothing():
    with pytest.raises(ValueError, match=r"should lie in \[0\.0, 0\.5\)"):
        DPOConfig(loss_type="robust", label_smoothing=0.5)


def test_dpo_exo_pair_loss_accepts_jax_label_smoothing():
    chosen_logps = jnp.array([-0.15, -1.10], dtype=jnp.float32)
    rejected_logps = jnp.array([-1.30, -0.95], dtype=jnp.float32)
    ref_chosen_logps = jnp.array([-0.45, -1.00], dtype=jnp.float32)
    ref_rejected_logps = jnp.array([-1.10, -1.20], dtype=jnp.float32)

    loss_fn = get_loss_function("exo_pair", beta=0.3, label_smoothing=0.1)
    losses = loss_fn(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        0.3,
        0.1,
    )

    assert losses.shape == chosen_logps.shape
    assert np.isfinite(np.asarray(losses)).all()


def test_dpo_multi_loss_and_weights_match_weighted_sum():
    chosen_logps = jnp.array([-0.15, -1.10], dtype=jnp.float32)
    rejected_logps = jnp.array([-1.30, -0.95], dtype=jnp.float32)
    ref_chosen_logps = jnp.array([-0.45, -1.00], dtype=jnp.float32)
    ref_rejected_logps = jnp.array([-1.10, -1.20], dtype=jnp.float32)

    combined = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type=("sigmoid", "hinge"),
        loss_weights=(0.25, 0.75),
    )
    sigmoid_loss = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type="sigmoid",
    )
    hinge_loss = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type="hinge",
    )

    np.testing.assert_allclose(np.asarray(combined), np.asarray(0.25 * sigmoid_loss + 0.75 * hinge_loss))


def test_dpo_f_divergence_forward_kl_matches_trl_score_transform():
    chosen_logps = jnp.array([-0.15, -1.10], dtype=jnp.float32)
    rejected_logps = jnp.array([-1.30, -0.95], dtype=jnp.float32)
    ref_chosen_logps = jnp.array([-0.45, -1.00], dtype=jnp.float32)
    ref_rejected_logps = jnp.array([-1.10, -1.20], dtype=jnp.float32)
    beta = 0.3

    actual = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
        label_smoothing=0.0,
        loss_type="sigmoid",
        f_divergence_type="forward_kl",
    )
    chosen_scores = -jnp.exp(-(chosen_logps - ref_chosen_logps))
    rejected_scores = -jnp.exp(-(rejected_logps - ref_rejected_logps))
    expected = -jax_logsigmoid(beta * (chosen_scores - rejected_scores))

    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_dpo_wpo_weights_scale_per_sequence_losses():
    chosen_logps = jnp.array([-0.15, -1.10], dtype=jnp.float32)
    rejected_logps = jnp.array([-1.30, -0.95], dtype=jnp.float32)
    ref_chosen_logps = jnp.array([-0.45, -1.00], dtype=jnp.float32)
    ref_rejected_logps = jnp.array([-1.10, -1.20], dtype=jnp.float32)
    weights = jnp.array([0.5, 2.0], dtype=jnp.float32)

    unweighted = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type="sigmoid",
    )
    weighted = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type="sigmoid",
        wpo_weights=weights,
    )

    np.testing.assert_allclose(np.asarray(weighted), np.asarray(unweighted * weights), rtol=1e-6, atol=1e-6)


def test_dpo_rpo_alpha_adds_chosen_nll_term():
    chosen_logps = jnp.array([-2.0, -4.0], dtype=jnp.float32)
    rejected_logps = jnp.array([-3.0, -5.0], dtype=jnp.float32)
    ref_chosen_logps = jnp.array([-2.5, -4.5], dtype=jnp.float32)
    ref_rejected_logps = jnp.array([-3.5, -5.5], dtype=jnp.float32)
    chosen_lengths = jnp.array([2.0, 4.0], dtype=jnp.float32)

    base = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type="sigmoid",
        chosen_lengths=chosen_lengths,
    )
    rpo = compute_dpo_losses(
        chosen_logps,
        rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=0.3,
        label_smoothing=0.0,
        loss_type="sigmoid",
        chosen_lengths=chosen_lengths,
        rpo_alpha=0.2,
    )

    expected = base + 0.2 * (-chosen_logps / chosen_lengths)
    np.testing.assert_allclose(np.asarray(rpo), np.asarray(expected), rtol=1e-6, atol=1e-6)


def test_dpo_config_normalizes_loss_type_and_validates_weights():
    cfg = DPOConfig(loss_type=["sigmoid", "hinge"], loss_weights=[0.2, 0.8])

    assert cfg.loss_type == ("sigmoid", "hinge")
    assert cfg.loss_weights == (0.2, 0.8)
    assert DPOConfig(precompute_ref_batch_size=7).precompute_ref_batch_size == 7
    assert DPOConfig(pad_to_multiple_of=8).pad_to_multiple_of == 8
    assert DPOConfig(pad_token="<pad>").pad_token == "<pad>"
    assert DPOConfig(padding_free=True).padding_free is True
    with pytest.raises(ValueError, match="loss_weights"):
        DPOConfig(loss_type=["sigmoid", "hinge"], loss_weights=[1.0])
    with pytest.raises(ValueError, match="precompute_ref_batch_size"):
        DPOConfig(precompute_ref_batch_size=0)
    with pytest.raises(ValueError, match="pad_to_multiple_of"):
        DPOConfig(pad_to_multiple_of=0)
    assert DPOConfig(activation_offloading=True).activation_offloading is True
    assert DPOConfig(rpo_alpha=0.1).rpo_alpha == 0.1
    with pytest.raises(ValueError, match="rpo_alpha"):
        DPOConfig(rpo_alpha=-0.1)
    with pytest.raises(ValueError, match="ref_model_mixup_alpha"):
        DPOConfig(ref_model_mixup_alpha=1.5)
    with pytest.raises(ValueError, match="ref_model_sync_steps"):
        DPOConfig(ref_model_sync_steps=0)


def test_dpo_pad_token_override_accepts_processor_wrappers():
    class Tokenizer:
        pad_token = None

    class Processor:
        tokenizer = Tokenizer()

    processor = Processor()
    DPOTrainer._apply_pad_token_override(processor, "<pad>")

    assert processor.tokenizer.pad_token == "<pad>"


def test_dpo_padding_free_falls_back_to_padded_batches():
    cfg = DPOConfig(padding_free=True)

    assert DPOTrainer._resolve_padding_free(cfg) is False
    assert cfg.padding_free is False


def test_dpo_disable_dropout_puts_policy_and_reference_in_eval_mode():
    class Model:
        def __init__(self):
            self.eval_calls = 0

        def eval(self):
            self.eval_calls += 1

    policy = SimpleNamespace(model=Model())
    reference = SimpleNamespace(model=Model())

    DPOTrainer._disable_state_dropout(policy, reference)

    assert policy.model.eval_calls == 1
    assert reference.model.eval_calls == 1


def test_dpo_reference_sync_uses_mixup_alpha():
    trainer = object.__new__(DPOTrainer)
    trainer.arguments = DPOConfig(sync_ref_model=True, ref_model_mixup_alpha=0.25, ref_model_sync_steps=2)
    trainer.reference_state = _State({"w": jnp.asarray([1.0, 5.0], dtype=jnp.float32)})
    policy_state = _State({"w": jnp.asarray([9.0, 1.0], dtype=jnp.float32)})

    returned_state, returned_metrics = DPOTrainer.on_step_end(trainer, policy_state, {"loss": 1.0}, step=2)

    assert returned_state is policy_state
    assert returned_metrics == {"loss": 1.0}
    np.testing.assert_allclose(
        np.asarray(trainer.reference_state.graphstate["w"]),
        np.asarray([3.0, 4.0], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_dpo_force_use_ref_model_overrides_reference_free():
    trainer = object.__new__(DPOTrainer)
    trainer.arguments = DPOConfig(reference_free=True, force_use_ref_model=False)

    assert trainer._effective_reference_free() is True

    trainer.arguments = DPOConfig(reference_free=True, force_use_ref_model=True)

    assert trainer._effective_reference_free() is False


def test_dpo_precompute_ref_batch_size_is_used_for_train_and_eval(monkeypatch):
    trainer = object.__new__(DPOTrainer)
    trainer.arguments = DPOConfig(precompute_ref_log_probs=True, precompute_ref_batch_size=7)
    trainer.dataset_train = object()
    trainer.dataset_eval = object()
    trainer._train_source = object()
    trainer._eval_source = object()
    trainer._precomputed_train_ref_log_probs = False
    trainer._precomputed_eval_ref_log_probs = False
    calls = []

    monkeypatch.setattr(DPOTrainer, "training_batch_size", property(lambda self: 3))
    monkeypatch.setattr(DPOTrainer, "evaluation_batch_size", property(lambda self: 5))
    monkeypatch.setattr(DPOTrainer, "_source_has_reference_logps", staticmethod(lambda source: False))
    monkeypatch.setattr(Trainer, "configure_dataloaders", lambda self: "configured")

    def fake_precompute(self, *, dataset_attr, source_attr, batch_size, is_train, desc):
        del self, source_attr, desc
        calls.append((dataset_attr, batch_size, is_train))
        return True

    monkeypatch.setattr(DPOTrainer, "_precompute_reference_log_probs_for_split", fake_precompute)

    assert DPOTrainer.configure_dataloaders(trainer) == "configured"
    assert calls == [
        ("dataset_train", 7, True),
        ("dataset_eval", 7, False),
    ]


def test_dpo_ld_alpha_weights_tail_tokens_after_shared_length():
    loss_mask = jnp.array(
        [
            [1, 1, 1, 0],  # chosen pair 0 length 3, shared with rejected length 1
            [1, 1, 0, 0],  # chosen pair 1 length 2, shared with rejected length 4
            [1, 0, 0, 0],  # rejected pair 0
            [1, 1, 1, 1],  # rejected pair 1
        ],
        dtype=bool,
    )

    weights = _ld_logp_weights(loss_mask, num_examples=2, ld_alpha=0.25)

    expected = jnp.array(
        [
            [1.0, 0.25, 0.25, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.25, 0.25],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(weights), np.asarray(expected), rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="ld_alpha"):
        DPOConfig(ld_alpha=1.5)


def jax_logsigmoid(x):
    return -jnp.logaddexp(0.0, -x)
