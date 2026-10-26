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
"""Distillation loss-group weighting: the weighted step optimises sum_g w_g * mean_g. Runs on TPU."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.trainers.distillation_trainer import _fn as distill_fn
from easydel.trainers.distillation_trainer.distillation_config import DistillationConfig
from easydel.trainers.distillation_trainer.distillation_trainer import DistillationTrainer
from easydel.trainers.distillation_trainer.loss_groups import (
    LOSS_GROUP_IDS_KEY,
    LOSS_TOKEN_WEIGHTS_KEY,
    loss_group_token_weights,
    loss_supervision_mask,
)
from easydel.trainers.training_utils import zero_gradient_accumulator
from jax import numpy as jnp

VOCAB, SEQ, BATCH = 64, 16, 4
STEP_KW = dict(temperature=2.0, alpha=0.8, logits_chunk_size=8)


def _tiny_state(seed: int):
    from easydel.modules.llama.llama_configuration import LlamaConfig
    from easydel.modules.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=VOCAB,
        max_position_embeddings=SEQ,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    config.attn_dtype = jnp.float32
    model = LlamaForCausalLM(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(seed),
    )
    return model.to_state(trainable_selector="parameters")


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(0)
    input_ids = rng.integers(1, VOCAB, size=(BATCH, SEQ)).astype(np.int32)
    completion = (rng.random((BATCH, SEQ)) > 0.3).astype(np.int32)
    completion[:, :2] = 0
    group_ids = np.zeros((BATCH, SEQ), np.int32)
    group_ids[2:] = 1
    group_ids[3, SEQ // 2 :] = 0
    batch = {
        "input_ids": input_ids,
        "attention_mask": np.ones((BATCH, SEQ), np.int32),
        "completion_mask": completion,
        "labels": input_ids.copy(),
    }
    return _tiny_state(1), _tiny_state(2), batch, group_ids


def _grad_step(student, teacher, batch, num_groups=0):
    acc = zero_gradient_accumulator(student.graphstate)
    with student.model.mesh, jax.default_matmul_precision("highest"):
        acc, metrics, weight = distill_fn.distillation_step(
            student_state=student,
            batch={k: jnp.asarray(v) for k, v in batch.items()},
            teacher_state=teacher,
            is_training=True,
            num_loss_groups=num_groups,
            accumulate_into=acc,
            **STEP_KW,
        )
    grads = jax.tree_util.tree_map(lambda g: np.asarray(g / weight), acc["grads"])
    return float(metrics.loss), grads, distill_fn.finalize_loss_group_metrics(metrics, num_groups)


def _weighted_batch(batch, group_ids, weights):
    token_weights, stats = loss_group_token_weights(group_ids, loss_supervision_mask(batch, shift_tokens=True), weights)
    return {**batch, LOSS_GROUP_IDS_KEY: group_ids, LOSS_TOKEN_WEIGHTS_KEY: token_weights}, stats


def _single_group_batch(batch, group_ids, g):
    return {**batch, "completion_mask": batch["completion_mask"] * (group_ids == g)}


def _leaves(tree):
    return jax.tree_util.tree_leaves(tree)


def test_token_share_weights_reproduce_the_unweighted_step(setup):
    teacher, student, batch, group_ids = setup
    supervised = loss_supervision_mask(batch, shift_tokens=True)
    shares = [float(supervised[group_ids == g].sum()) / float(supervised.sum()) for g in (0, 1)]
    weighted, _ = _weighted_batch(batch, group_ids, shares)
    sup = supervised != 0
    np.testing.assert_array_equal(weighted[LOSS_TOKEN_WEIGHTS_KEY][sup], 1.0)

    plain_loss, plain_grads, _ = _grad_step(student, teacher, batch)
    w_loss, w_grads, _ = _grad_step(student, teacher, weighted, num_groups=2)
    np.testing.assert_allclose(w_loss, plain_loss, rtol=1e-6)
    for a, b in zip(_leaves(w_grads), _leaves(plain_grads), strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-8)


def test_weighted_loss_and_grads_are_the_weighted_sum_of_single_group_steps(setup):
    teacher, student, batch, group_ids = setup
    weights = (0.7, 0.3)
    weighted, stats = _weighted_batch(batch, group_ids, weights)
    w_loss, w_grads, w_metrics = _grad_step(student, teacher, weighted, num_groups=2)

    singles = [_grad_step(student, teacher, _single_group_batch(batch, group_ids, g)) for g in (0, 1)]
    ref_loss = sum(w * s[0] for w, s in zip(weights, singles, strict=True))
    np.testing.assert_allclose(w_loss, ref_loss, rtol=1e-5)
    for leaf_w, leaf_0, leaf_1 in zip(_leaves(w_grads), _leaves(singles[0][1]), _leaves(singles[1][1]), strict=True):
        ref = weights[0] * leaf_0 + weights[1] * leaf_1
        np.testing.assert_allclose(leaf_w, ref, rtol=1e-4, atol=1e-6 * max(float(np.abs(ref).max()), 1.0))

    np.testing.assert_allclose(stats["weight"], weights, rtol=1e-12)
    for g in (0, 1):
        group_kl = float(w_metrics.other_metrics[f"loss_group/{g}/kl"])
        # per-token (xent - entropy) vs sum(xent) - sum(entropy): fp32 cancellation of O(1) terms
        np.testing.assert_allclose(group_kl, float(singles[g][2].other_metrics["kl_loss"]), rtol=0, atol=1e-5)
    assert "group_kl_ratio" not in w_metrics.other_metrics


def test_absent_group_renormalises_and_cap_binds():
    group_ids = np.array([[0, 0, 0, 1, 1, 1, 1, 1]], np.int32)
    supervised = np.ones((1, 8))
    token_weights, stats = loss_group_token_weights(group_ids, supervised, [0.5, 0.25, 0.25])
    np.testing.assert_allclose(stats["weight"], [2 / 3, 1 / 3, 0.0], rtol=1e-12)
    np.testing.assert_allclose(stats["tokens"], [3, 5, 0])
    ratio = token_weights[0, 0] / token_weights[0, 3]
    np.testing.assert_allclose(ratio, (2 / 3 / 3) / (1 / 3 / 5), rtol=1e-6)
    assert token_weights.dtype == np.float32 and token_weights.min() == pytest.approx(1.0)

    capped, capped_stats = loss_group_token_weights(group_ids, supervised, [0.9, 0.1], max_multiplier=1.5)
    assert capped_stats["weight"][0] < 0.9
    assert capped[0, 0] / capped[0, 3] <= 1.5 / (0.1 * 8 / 5) + 1e-6


def test_orphan_supervised_token_is_refused():
    group_ids = np.array([[0, 0, -1, 1]], np.int32)
    with pytest.raises(ValueError, match="no loss group"):
        loss_group_token_weights(group_ids, np.ones((1, 4)), [0.5, 0.5])
    loss_group_token_weights(group_ids, np.array([[1, 1, 0, 1]]), [0.5, 0.5])


def _host_trainer(**overrides):
    trainer = object.__new__(DistillationTrainer)
    arguments = dict(
        loss_group_weights=[0.25, 0.75],
        loss_group_dataset_map=None,
        loss_group_max_multiplier=None,
        loss_config=None,
    )
    arguments.update(overrides)
    trainer.arguments = SimpleNamespace(**arguments)
    return trainer


def test_trainer_attaches_groups_from_token_and_row_attribution():
    batch = {
        "input_ids": np.ones((2, 4), np.int32),
        "completion_mask": np.array([[0, 1, 1, 1], [0, 1, 1, 0]], np.int32),
    }
    infos: dict = {}
    trainer = _host_trainer(loss_group_dataset_map=[1, 0, -1])
    out = trainer._attach_loss_groups(dict(batch), infos, np.array([[1, 1, 0, 0], [0, 0, 0, 2]]), None, is_train=True)
    np.testing.assert_array_equal(out[LOSS_GROUP_IDS_KEY], [[0, 0, 1, 1], [1, 1, 1, -1]])
    assert out[LOSS_TOKEN_WEIGHTS_KEY].dtype == np.float32
    assert infos["loss_group/0/tokens"] == 1.0 and infos["loss_group/1/tokens"] == 4.0
    np.testing.assert_array_equal(out["completion_mask"], batch["completion_mask"])

    rows = trainer._attach_loss_groups(dict(batch), {}, None, np.array([0, 1]), is_train=False)
    np.testing.assert_array_equal(rows[LOSS_GROUP_IDS_KEY], [[1] * 4, [0] * 4])
    assert LOSS_TOKEN_WEIGHTS_KEY not in rows
    with pytest.raises(ValueError, match="dataset attribution"):
        trainer._attach_loss_groups(dict(batch), {}, None, None, is_train=True)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(loss_group_weights=[-0.1, 1.0]), "non-negative"),
        (dict(loss_group_weights=[0.0, 0.0]), "positive sum"),
        (dict(loss_group_weights=[0.5, 0.5], loss_group_dataset_map=[0, 2]), "loss_group_dataset_map"),
        (dict(loss_group_dataset_map=[0, 1]), "need `loss_group_weights`"),
        (dict(loss_group_weights=[1.0], loss_group_max_multiplier=0.0), "max_multiplier"),
    ],
)
def test_config_refusals(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        DistillationConfig(save_directory=str(tmp_path), use_wandb=False, **kwargs)


def test_config_normalises_weights_and_roundtrips(tmp_path):
    cfg = DistillationConfig(
        save_directory=str(tmp_path),
        use_wandb=False,
        loss_group_weights=[3.0, 1.0],
        loss_group_dataset_map=[1, -1, 0],
        loss_group_max_multiplier=4.0,
    )
    assert cfg.loss_group_weights == [0.75, 0.25]
    rebuilt = DistillationConfig.from_dict(cfg.to_dict())
    assert rebuilt.loss_group_weights == [0.75, 0.25]
    assert rebuilt.loss_group_dataset_map == [1, -1, 0] and rebuilt.loss_group_max_multiplier == 4.0


def test_jax_backend_is_an_accelerator():
    assert jax.default_backend() in ("tpu", "gpu")
