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

from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.metrics import JSONProgressBar, MetricsTracker, StepMetrics


def test_step_metrics_reports_fractional_epoch_progress(monkeypatch):
    step_metrics = StepMetrics(arguments=SimpleNamespace(performance_mode=True))
    step_metrics.start_time = 0.0
    step_metrics.step_start_time = 10.0
    monkeypatch.setattr("easydel.trainers.metrics.time.time", lambda: 12.0)

    metrics = LossMetrics(
        loss=1.0,
        accuracy=0.5,
        execution_time=2.0,
    )

    results = step_metrics.calculate(
        metrics=metrics,
        current_step=25,
        epoch=0,
        epoch_progress=0.25,
        flops_per_token=1.0,
        extra_flops_per_token=0.0,
        batch_size=2,
        seq_length=4,
        learning_rate=1e-4,
        mode="train",
    )

    assert results["train/epoch"] == 0.25
    assert results["train/epoch_index"] == 0
    assert results["performance/execution_time"] == 2.0
    assert results["performance/preprocessing_time"] == 0.0
    assert results["performance/train_step_time"] == 2.0


def test_step_metrics_offsets_fractional_progress_by_epoch_index(monkeypatch):
    step_metrics = StepMetrics(arguments=SimpleNamespace(performance_mode=True))
    step_metrics.start_time = 0.0
    step_metrics.step_start_time = 10.0
    monkeypatch.setattr("easydel.trainers.metrics.time.time", lambda: 12.0)

    metrics = LossMetrics(
        loss=1.0,
        accuracy=None,
        execution_time=2.0,
    )

    results = step_metrics.calculate(
        metrics=metrics,
        current_step=10,
        epoch=1,
        epoch_progress=0.1,
        flops_per_token=1.0,
        extra_flops_per_token=0.0,
        batch_size=2,
        seq_length=4,
        learning_rate=1e-4,
        mode="train",
    )

    assert results["train/epoch"] == 1.1
    assert results["train/epoch_index"] == 1


def test_step_metrics_host_converts_jax_scalars_without_jax_exp(monkeypatch):
    def fail_jax_exp(_):
        raise AssertionError("host metrics must not call jnp.exp")

    monkeypatch.setattr("easydel.trainers.metrics.jnp.exp", fail_jax_exp)
    step_metrics = StepMetrics(arguments=SimpleNamespace(performance_mode=True))
    step_metrics.start_time = 0.0
    step_metrics.step_start_time = 10.0
    monkeypatch.setattr("easydel.trainers.metrics.time.time", lambda: 12.0)

    metrics = LossMetrics(
        loss=jnp.asarray(0.5, dtype=jnp.float32),
        z_loss=jnp.asarray(0.25, dtype=jnp.float32),
        accuracy=jnp.asarray(0.75, dtype=jnp.float32),
        chosen_rewards=jnp.asarray([1.0, 3.0], dtype=jnp.float32),
        rejected_rewards=jnp.asarray([0.0, 2.0], dtype=jnp.float32),
        other_metrics={"kl_loss": jnp.asarray(0.125, dtype=jnp.float32)},
        execution_time=2.0,
    )

    results = step_metrics.calculate(
        metrics=metrics,
        current_step=3,
        epoch=0,
        epoch_progress=None,
        flops_per_token=1.0,
        extra_flops_per_token=0.0,
        batch_size=2,
        seq_length=4,
        learning_rate=jnp.asarray(1e-4, dtype=jnp.float32),
        mode="train",
        mean_loss=jnp.asarray(0.5, dtype=jnp.float32),
    )

    assert results["train/loss"] == pytest.approx(0.5)
    assert results["train/perplexity"] == pytest.approx(float(np.exp(np.asarray(0.5, dtype=np.float32))))
    assert results["train/z_loss"] == pytest.approx(0.25)
    assert results["train/accuracy"] == pytest.approx(0.75)
    assert results["train/chosen_rewards"] == pytest.approx(2.0)
    assert results["train/rejected_rewards"] == pytest.approx(1.0)
    assert results["train/kl_loss"] == pytest.approx(0.125)
    assert results["train/mean_loss"] == pytest.approx(0.5)


def test_metrics_tracker_counts_updates_and_ignores_missing_accuracy():
    tracker = MetricsTracker()

    mean_loss, mean_accuracy = tracker.update(loss=2.0, accuracy=None, step=50)

    assert mean_loss == 2.0
    assert mean_accuracy is None

    mean_loss, mean_accuracy = tracker.update(loss=4.0, accuracy=0.5, step=51)

    assert mean_loss == 3.0
    assert mean_accuracy == 0.5


def test_metrics_tracker_update_accepts_jax_scalars():
    tracker = MetricsTracker()

    mean_loss, mean_accuracy = tracker.update(
        loss=jnp.asarray(1.5, dtype=jnp.float32),
        accuracy=jnp.asarray(0.25, dtype=jnp.float32),
        step=1,
    )

    assert mean_loss == pytest.approx(1.5)
    assert mean_accuracy == pytest.approx(0.25)


def test_metrics_tracker_ignores_non_finite_accuracy():
    tracker = MetricsTracker()

    tracker.update(loss=1.0, accuracy=float("inf"), step=1)
    mean_loss, mean_accuracy = tracker.update(loss=3.0, accuracy=0.75, step=2)

    assert mean_loss == 2.0
    assert mean_accuracy == 0.75


def test_json_progress_bar_nests_parental_metric_keys(monkeypatch):
    logged: list[str] = []
    monkeypatch.setattr("easydel.trainers.metrics.logger.info", logged.append)

    JSONProgressBar().set_postfix(
        loss=1.0,
        **{"performance/train_step_time": 2.0, "performance/data_collection_time": 0.25},
    )

    assert logged == [
        "{'loss': 1.0, \x1b[96m'performance': {'train_step_time': 2.0, 'data_collection_time': 0.25}\x1b[0m}"
    ]
