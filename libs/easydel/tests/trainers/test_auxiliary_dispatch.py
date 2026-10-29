"""Exercise auxiliary executor selection using pure-Python runtime stand-ins."""

import pytest
from easydel.trainers import training_utils

pytestmark = pytest.mark.cpu_ok


class Rows:
    """An opaque batch leaf, with no NumPy or JAX array storage."""

    def __init__(self, values):
        self.values = tuple(values)
        self.shape = (len(self.values),)


def test_explicit_worker_selection_is_independent_for_cached_auxiliary_calls(monkeypatch):
    class Executor:
        def __init__(self, *, use_workers):
            self.use_workers = use_workers

        def dispatch_many(self, compiled, batches):
            for args in batches:
                yield {**compiled(*args), "workers": self.use_workers}

    monkeypatch.setattr(training_utils.spx, "MpmdPipelineExecutor", Executor)
    monkeypatch.setattr(training_utils.spx, "jit", lambda fn, **kwargs: fn)
    monkeypatch.setattr(training_utils, "_SCHEDULED_AUXILIARY_CACHE", {})
    monkeypatch.setattr(training_utils, "_SCHEDULED_AUX_PIPELINE_EXECUTOR", {})
    monkeypatch.setattr(training_utils, "_SCHED_SECTION_LOG_BUDGET", [0])
    monkeypatch.setattr(
        training_utils.lax, "slice_in_dim", lambda leaf, start, stop, axis: Rows(leaf.values[start:stop])
    )
    monkeypatch.setattr(
        training_utils.jnp, "concatenate", lambda leaves, axis: Rows(value for leaf in leaves for value in leaf.values)
    )

    def forward(batch):
        return {"rows": batch}

    mesh = object()
    batch = Rows(["a", "b", "c", "d"])
    for use_workers in (False, True, False):
        auxiliary = training_utils.cached_scheduled_auxiliary(
            forward, mesh, microbatches=2, batch_argnums=0, use_workers=use_workers
        )
        result = auxiliary(batch)
        assert result["rows"].values == batch.values
        assert result["workers"] is use_workers
