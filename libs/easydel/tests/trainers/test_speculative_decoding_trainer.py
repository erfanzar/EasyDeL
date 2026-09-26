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

"""Tests for the speculative-decoding trainer and loss functions.

Validates ``speculative_draft_loss``, ``_deepspec_block_loss``, and
``SpeculativeDecodingConfig`` behavior including perfect-match loss,
multi-step chain logits, MTP-style offsets, and configuration validation.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("ENABLE_DISTRIBUTED_INIT", "0")

import easydel.trainers  # noqa: F401
import jax
import jax.numpy as jnp
import pytest
from easydel.trainers import SpeculativeDecodingConfig, SpeculativeDecodingTrainer
from easydel.trainers.speculative_decoding_trainer._fn import (
    _deepspec_block_loss,
    speculative_draft_loss,
)
from easydel.utils import Registry


def _logits(batch: int = 2, sequence: int = 8, vocab: int = 16) -> jax.Array:
    """Create a random logits tensor for testing.

    Args:
        batch: Batch size. Defaults to 2.
        sequence: Sequence length. Defaults to 8.
        vocab: Vocabulary size. Defaults to 16.

    Returns:
        jax.Array: Random normal logits of shape (batch, sequence, vocab).
    """
    return jax.random.normal(jax.random.PRNGKey(0), (batch, sequence, vocab), dtype=jnp.float32)


def test_speculative_loss_perfect_next_token_match_is_zero() -> None:
    """Assert that perfect next-token match yields zero speculative loss.

    When draft logits exactly match target logits, the speculative draft loss
    and KL divergence should be near zero, and the draft acceptance rate should
    be 1.0.
    """
    target = _logits()
    input_ids = jnp.arange(target.shape[0] * target.shape[1], dtype=jnp.int32).reshape(target.shape[:2])
    loss, metrics = speculative_draft_loss(
        target,
        target,
        input_ids=input_ids,
        attention_mask=jnp.ones(target.shape[:2], dtype=jnp.int32),
        temperature=1.0,
        alpha=1.0,
    )
    assert abs(float(loss)) < 1e-6
    other_metrics = metrics.other_metrics
    assert other_metrics is not None
    assert abs(float(other_metrics["spec_kl_loss"])) < 1e-6
    assert float(other_metrics["draft_accept_rate"]) == pytest.approx(1.0)
    assert float(other_metrics["draft_step1_accept_rate"]) == pytest.approx(1.0)
    assert float(other_metrics["draft_acceptance_match"]) == pytest.approx(1.0)


def test_speculative_loss_supports_mtp_style_offset() -> None:
    """Assert speculative draft loss supports MTP-style offset.

    Shifts the draft logits by one position relative to the target and uses
    ``draft_target_offset=1``. The loss and step-1 KL divergence should be near
    zero when the shifted draft matches the target.
    """
    target = _logits(sequence=10)
    draft = jnp.concatenate([target[:, 1:], target[:, -1:]], axis=1)
    loss, metrics = speculative_draft_loss(
        draft,
        target,
        input_ids=jnp.ones(target.shape[:2], dtype=jnp.int32),
        attention_mask=jnp.ones(target.shape[:2], dtype=jnp.int32),
        temperature=1.0,
        alpha=1.0,
        draft_target_offset=1,
    )
    assert abs(float(loss)) < 1e-6
    other_metrics = metrics.other_metrics
    assert other_metrics is not None
    assert abs(float(other_metrics["draft_step1_kl_loss"])) < 1e-6


def test_speculative_loss_supports_multi_step_chain_logits() -> None:
    """Assert speculative draft loss supports multi-step chain logits.

    Constructs a chain of two draft steps and passes ``num_draft_tokens=2``.
    Verifies that the loss is near zero, the reported number of draft steps is
    2, and per-step acceptance rates and KL divergences are correct.
    """
    target = _logits(sequence=12)
    b, _, v = target.shape
    step1 = jnp.concatenate([target[:, 1:], target[:, -1:]], axis=1)
    step2 = jnp.concatenate([target[:, 2:], jnp.broadcast_to(target[:, -1:], (b, 2, v))], axis=1)
    chain = jnp.stack([step1, step2], axis=0)

    loss, metrics = speculative_draft_loss(
        chain,
        target,
        input_ids=jnp.ones(target.shape[:2], dtype=jnp.int32),
        attention_mask=jnp.ones(target.shape[:2], dtype=jnp.int32),
        temperature=1.0,
        alpha=1.0,
        num_draft_tokens=2,
        draft_target_offset=1,
    )
    assert abs(float(loss)) < 1e-6
    other_metrics = metrics.other_metrics
    assert other_metrics is not None
    assert int(other_metrics["draft_steps"]) == 2
    assert float(other_metrics["draft_accept_rate"]) == pytest.approx(1.0)
    assert float(other_metrics["draft_step2_accept_rate"]) == pytest.approx(1.0)
    assert abs(float(other_metrics["draft_step2_kl_loss"])) < 1e-6


def test_deepspec_block_loss_reports_accept_rate() -> None:
    """Assert _deepspec_block_loss computes correct acceptance metrics.

    Uses hand-crafted block logits and target IDs that perfectly match to
    verify that the block loss is near zero and the reported acceptance rates
    and ``tau_probabilistic`` match expected values.
    """
    block_logits = jnp.array(
        [
            [
                [[18.0, -18.0, -18.0], [-18.0, 18.0, -18.0]],
                [[-18.0, -18.0, 18.0], [18.0, -18.0, -18.0]],
            ]
        ],
        dtype=jnp.float32,
    )
    outputs = SimpleNamespace(
        block_logits=block_logits,
        target_ids=jnp.array([[[0, 1], [2, 0]]], dtype=jnp.int32),
        eval_mask=jnp.ones((1, 2, 2), dtype=jnp.bool_),
        block_keep_mask=jnp.ones((1, 2), dtype=jnp.bool_),
        aligned_target_logits=block_logits,
    )
    config = SimpleNamespace(
        ce_loss_alpha=1.0,
        l1_loss_alpha=0.5,
        confidence_head_alpha=0.0,
        loss_decay_gamma=None,
    )

    loss, metrics = _deepspec_block_loss(outputs, config)
    assert float(loss) < 1e-5
    other_metrics = metrics.other_metrics
    assert other_metrics is not None
    assert float(other_metrics["draft_accept_rate"]) == pytest.approx(1.0)
    assert float(other_metrics["draft_step1_accept_rate"]) == pytest.approx(1.0)
    assert float(other_metrics["tau_probabilistic"]) == pytest.approx(3.0)


def test_speculative_config_validation_and_registry(tmp_path) -> None:
    """Assert SpeculativeDecodingConfig validation and registry entries.

    Creates a valid configuration and checks that ``assistant_only_loss`` and
    ``completion_only_loss`` aliases are synchronized, the rich progress-bar
    metrics list is correct, and the trainer and argument classes are
    registered. Also verifies that invalid ``alpha``, ``temperature``, and
    ``num_draft_tokens`` values raise ``ValueError``.

    Args:
        tmp_path: Pytest temporary path fixture.
    """
    cfg = SpeculativeDecodingConfig(
        save_directory=str(tmp_path / "spec"),
        alpha=0.5,
        temperature=2.0,
        num_draft_tokens=2,
        draft_target_offset=1,
        completion_only_loss=True,
    )
    assert cfg.assistant_only_loss is True
    assert cfg.completion_only_loss is True
    assert cfg.metrics_to_show_in_rich_pbar == [
        "loss",
        "draft_accept_rate",
        "tau_probabilistic",
        "deepspec_block_ce_loss",
        "deepspec_block_l1_loss",
    ]
    assert Registry.get("trainer", "spec") is SpeculativeDecodingTrainer
    assert Registry.get("trainer-arguments", "spec_decoding") is SpeculativeDecodingConfig

    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(save_directory=str(tmp_path / "bad_alpha"), alpha=1.5)
    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(save_directory=str(tmp_path / "bad_temp"), temperature=0.0)
    with pytest.raises(ValueError):
        SpeculativeDecodingConfig(save_directory=str(tmp_path / "bad_tokens"), num_draft_tokens=0)
