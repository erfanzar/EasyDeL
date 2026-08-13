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

"""Scheduled-MPMD microbatch loss weighting: parity with the production CE loss.

The scheduled (pipeline-parallel) trainer path reduces per-microbatch losses
into the step loss. For the token-mean causal-LM loss this must be the
valid-token-weighted mean — ``sum_mb(w_mb * loss_mb) / sum(w)`` with
``w_mb`` equal to the microbatch's loss denominator — otherwise pipeline
training silently optimizes a different objective than the SPMD trainer.

These tests pin the base adapter's weight callable against the production
loss (``ForCausalLMLoss``): the weight must equal the loss's own
``weight_sum`` denominator, and the weighted recombination of microbatch
losses must reproduce the full-batch loss exactly.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from easydel.infra.loss_utils import ForCausalLMLoss, LossConfig
from easydel.trainers.trainer._fn import (
    _base_scheduled_weight_semantics,
    _make_base_scheduled_weight,
)

_B, _T, _V = 4, 12, 32
_M = 2  # microbatches


class _FakeLossFn:
    """Stand-in for a model loss function with a controllable ``__name__``."""

    def __init__(self, name: str):
        self.__name__ = name


class _FakeModel:
    def __init__(self, loss_name: str):
        self.loss_function = _FakeLossFn(loss_name)


class _FakeState:
    def __init__(self, loss_name: str):
        self.model = _FakeModel(loss_name)


class _FakeCall:
    """Minimal ScheduledStepCall stand-in exposing ``get`` and ``state``.

    Defaults to a causal-LM model: the weight callable is only valid for
    ``ForCausalLMLoss`` and returns ``None`` (uniform weighting) for any
    other head loss.
    """

    def __init__(self, loss_name: str = "ForCausalLMLoss", **bound):
        self._bound = bound
        self.state = _FakeState(loss_name)

    def get(self, name, default=None):
        return self._bound.get(name, default)


def _make_batch(seed: int = 0):
    """Synthetic causal-LM batch with uneven valid-token counts per row."""
    key = jax.random.PRNGKey(seed)
    k_logits, k_labels = jax.random.split(key)
    logits = jax.random.normal(k_logits, (_B, _T, _V), dtype=jnp.float32)
    labels = jax.random.randint(k_labels, (_B, _T), 0, _V)
    # Mask a different prompt/pad prefix length per row (completion-only SFT shape).
    prefix = jnp.asarray([2, 5, 9, 11])
    positions = jnp.arange(_T)[None, :]
    labels = jnp.where(positions < prefix[:, None], -100, labels)
    attention_mask = jnp.ones((_B, _T), dtype=jnp.int32)
    return logits, labels, attention_mask


def test_weight_fn_matches_loss_weight_sum_without_attention_mask():
    loss_config = LossConfig()
    weight_fn = _make_base_scheduled_weight(_FakeCall(loss_config=loss_config))
    assert weight_fn is not None
    logits, labels, _ = _make_batch()
    metrics = ForCausalLMLoss(logits, labels, attention_mask=None, config=loss_config)
    got = weight_fn(None, {"labels": labels})
    assert jnp.allclose(got, metrics.weight_sum), (got, metrics.weight_sum)


def test_weight_fn_matches_loss_weight_sum_with_attention_mask():
    loss_config = LossConfig()
    weight_fn = _make_base_scheduled_weight(_FakeCall(loss_config=loss_config))
    assert weight_fn is not None
    logits, labels, attention_mask = _make_batch(seed=1)
    # attention_mask is 1 over -100 prompt positions (completion-only SFT);
    # the loss ANDs it with validity — the weight must match.
    metrics = ForCausalLMLoss(logits, labels, attention_mask=attention_mask, config=loss_config)
    got = weight_fn(None, {"labels": labels, "attention_mask": attention_mask})
    assert jnp.allclose(got, metrics.weight_sum), (got, metrics.weight_sum)


def test_weighted_microbatch_recombination_reproduces_full_batch_loss():
    """sum(w_mb * loss_mb) / sum(w) == full-batch loss for the production CE."""
    loss_config = LossConfig()
    weight_fn = _make_base_scheduled_weight(_FakeCall(loss_config=loss_config))
    assert weight_fn is not None
    logits, labels, attention_mask = _make_batch(seed=2)

    full = ForCausalLMLoss(logits, labels, attention_mask=attention_mask, config=loss_config)

    mb_size = _B // _M
    weighted_sum = 0.0
    weight_total = 0.0
    uniform_sum = 0.0
    for mb in range(_M):
        sl = slice(mb * mb_size, (mb + 1) * mb_size)
        mb_metrics = ForCausalLMLoss(
            logits[sl],
            labels[sl],
            attention_mask=attention_mask[sl],
            config=loss_config,
        )
        w = weight_fn(None, {"labels": labels[sl], "attention_mask": attention_mask[sl]})
        weighted_sum = weighted_sum + w * mb_metrics.loss
        weight_total = weight_total + w
        uniform_sum = uniform_sum + mb_metrics.loss

    weighted = weighted_sum / weight_total
    uniform = uniform_sum / _M
    assert jnp.allclose(weighted, full.loss, atol=1e-6, rtol=1e-6), (weighted, full.loss)
    # The synthetic batch must actually separate the two estimators, or this
    # test could not falsify a mean-of-means regression.
    assert not jnp.allclose(uniform, full.loss, atol=1e-4)


def test_weight_semantics_gating():
    """Non-token-mean losses must keep the historical uniform weighting."""
    assert _base_scheduled_weight_semantics(LossConfig(), "nll") == "masked"
    assert _base_scheduled_weight_semantics(None, "nll") == "masked"
    assert _base_scheduled_weight_semantics(LossConfig(), "dft") == "valid_only"
    assert _base_scheduled_weight_semantics(LossConfig(reduction="mean"), "nll") == "masked"
    assert _base_scheduled_weight_semantics(LossConfig(reduction="sum"), "nll") is None
    assert _base_scheduled_weight_semantics(LossConfig(divide_weight_sum=True), "nll") is None
    assert _base_scheduled_weight_semantics(LossConfig(loss_normalizing_factor=128.0), "nll") is None
    assert _make_base_scheduled_weight(_FakeCall(loss_config=LossConfig(reduction="sum"))) is None


def test_weight_fn_is_none_for_non_causal_lm_heads():
    """The CLM-shaped weight must not apply to other head losses.

    QA / token-classification heads route ``compute_loss`` elsewhere; the
    ``input_ids`` fallback would produce a silently wrong denominator for
    them, so the adapter must return ``None`` (uniform weighting).
    """
    for loss_name in ("ForQuestionAnsweringLoss", "ForTokenClassification", "ForSequenceClassificationLoss"):
        assert _make_base_scheduled_weight(_FakeCall(loss_name=loss_name, loss_config=LossConfig())) is None
    # A call with no resolvable state fails closed too.
    call = _FakeCall(loss_config=LossConfig())
    call.state = None
    assert _make_base_scheduled_weight(call) is None
    # dft computes its own causal-LM metrics and keeps the weight regardless
    # of the head's declared loss function.
    assert _make_base_scheduled_weight(_FakeCall(loss_name="ForQuestionAnsweringLoss", loss_type="dft")) is not None
