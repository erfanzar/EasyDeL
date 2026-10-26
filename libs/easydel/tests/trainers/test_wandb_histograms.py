"""Public metrics logging retains array distributions without any W&B connection."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.trainers import training_configurations as config

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("dtype", [np.float32, config.jnp.bfloat16])
@pytest.mark.parametrize("array_source", ["numpy", "jax"])
def test_public_logging_retains_array_histograms(monkeypatch, dtype, array_source):
    class HostJaxArray:
        """Host-only stand-in for JAX array transport; never initializes a device."""

        def __init__(self, values):
            self.values = values
            self.size = values.size

        def __array__(self, dtype=None, copy=None):
            return np.asarray(self.values, dtype=dtype)

    class Histogram:
        def __init__(self, value):
            self.values = np.asarray(value)

    sink = []
    monkeypatch.setattr(config.jax, "Array", HostJaxArray)
    monkeypatch.setattr(config.jax, "device_get", lambda value: value)
    monkeypatch.setattr(
        config, "wandb", SimpleNamespace(Histogram=Histogram, log=lambda metrics, step: sink.append((metrics, step)))
    )
    monkeypatch.setattr(config.TrainingArguments, "get_tensorboard", lambda self: None)
    args = object.__new__(config.TrainingArguments)
    args.can_log_metrics = True
    args.use_wandb = True
    values = np.asarray([-2.5, 0.0, 2.0, 70016.0], dtype=dtype)
    array = HostJaxArray(values) if array_source == "jax" else values
    args.log_metrics({"distribution": array, "loss": 0.25, "absent": None}, step=7)
    assert len(sink) == 1
    metrics, step = sink[0]
    assert step == 7
    assert metrics["loss"] == 0.25
    assert "absent" not in metrics
    histogram = metrics["distribution"]
    assert isinstance(histogram, Histogram)
    assert histogram.values.dtype == np.float32
    np.testing.assert_array_equal(histogram.values, values.astype(np.float32))
    assert np.isfinite(histogram.values).all()
