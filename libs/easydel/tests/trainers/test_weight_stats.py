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

import jax
import jax.numpy as jnp
import spectrax as spx

from easydel.trainers import training_configurations as training_configurations_mod
from easydel.trainers.metrics import MetricsHistogram, compute_weight_stats
from easydel.trainers.training_configurations import TrainingArguments


def test_compute_weight_stats_accepts_plain_dict():
    params = {"layer": {"weight": jnp.arange(8, dtype=jnp.float32)}}

    stats = jax.device_get(compute_weight_stats(params, r".*weight"))

    histogram = stats["layer/weight/histogram"]
    assert isinstance(histogram, MetricsHistogram)
    assert histogram.size == 8


def test_compute_weight_stats_accepts_spectrax_state():
    params = spx.State({"parameters": {"layer.weight": jnp.arange(8, dtype=jnp.float32)}})

    stats = jax.device_get(compute_weight_stats(params, r".*weight"))

    histogram = stats["parameters/layer/weight/histogram"]
    assert isinstance(histogram, MetricsHistogram)
    assert histogram.size == 8


def test_compute_weight_stats_accepts_leaves_on_different_devices():
    devices = jax.devices()
    if len(devices) < 3:
        return

    params = {
        "parameters": {
            "lm_head": {"weight": jax.device_put(jnp.arange(8, dtype=jnp.float32), devices[0])},
            "layer": {"weight": jax.device_put(jnp.arange(8, 16, dtype=jnp.float32), devices[2])},
        }
    }

    stats = jax.device_get(compute_weight_stats(params, r".*weight"))

    assert set(stats) == {
        "parameters/layer/weight/histogram",
        "parameters/lm_head/weight/histogram",
    }
    assert all(isinstance(histogram, MetricsHistogram) for histogram in stats.values())


def test_training_arguments_weight_distribution_accepts_spectrax_state(monkeypatch):
    class _State:
        graphstate = spx.State({"parameters": {"layer.weight": jnp.arange(8, dtype=jnp.float32)}})

    args = TrainingArguments(weight_distribution_log_steps=1, report_metrics=False)
    warnings = []

    monkeypatch.setattr(training_configurations_mod.logger, "warning", lambda message: warnings.append(message))
    args.log_weight_distribution(_State(), step=1)

    assert not any("Failed to log weight distribution" in str(message) for message in warnings)
