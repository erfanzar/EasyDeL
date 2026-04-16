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

from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.metrics import MetricsTracker, StepMetrics


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


def test_metrics_tracker_counts_updates_and_ignores_missing_accuracy():
    tracker = MetricsTracker()

    mean_loss, mean_accuracy = tracker.update(loss=2.0, accuracy=None, step=50)

    assert mean_loss == 2.0
    assert mean_accuracy is None

    mean_loss, mean_accuracy = tracker.update(loss=4.0, accuracy=0.5, step=51)

    assert mean_loss == 3.0
    assert mean_accuracy == 0.5


def test_metrics_tracker_ignores_non_finite_accuracy():
    tracker = MetricsTracker()

    tracker.update(loss=1.0, accuracy=float("inf"), step=1)
    mean_loss, mean_accuracy = tracker.update(loss=3.0, accuracy=0.75, step=2)

    assert mean_loss == 2.0
    assert mean_accuracy == 0.75
