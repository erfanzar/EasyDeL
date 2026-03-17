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

from easydel.infra.loss_utils import LossMetrics
from easydel.trainers.metrics import StepMetrics


def test_step_metrics_reports_fractional_epoch_progress(monkeypatch):
    step_metrics = StepMetrics(arguments=SimpleNamespace(performance_mode=True))
    step_metrics.start_time = 0.0
    step_metrics.step_start_time = 10.0
    monkeypatch.setattr("easydel.trainers.metrics.time.time", lambda: 12.0)

    metrics = LossMetrics(
        loss=jnp.asarray(1.0),
        accuracy=jnp.asarray(0.5),
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
