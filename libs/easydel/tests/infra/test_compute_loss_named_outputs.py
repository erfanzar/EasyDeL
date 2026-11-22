"""Loss orchestration uses the always-named forward output without numerical array execution.

EasyDeL forwards always return their named output type; ``compute_loss`` must
consume it directly and never inject the deprecated ``return_dict`` switch.
"""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.loss_utils import LossMetrics


@dataclass
class _Output:
    logits: str
    loss: float | None = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


class _Strategy:
    def plan_forward(self, **kwargs):
        return SimpleNamespace(forward_kwargs={})

    def compute(self, *, outputs, labels, **kwargs):
        assert outputs.logits == "named-logits"
        assert labels == "labels"
        return LossMetrics(loss=3.5)


class _Model:
    compute_loss = EasyDeLBaseModule.compute_loss
    loss_function = staticmethod(lambda: None)

    def __init__(self):
        self.config = SimpleNamespace(mesh=None, partition_axis=None)
        self.mesh = SimpleNamespace(is_mpmd=False)
        self.loss_strategy = _Strategy()
        self.received = None

    def __call__(self, **kwargs):
        return self.forward(**kwargs)


class _KwargsModel(_Model):
    def forward(self, input_ids, **kwargs):
        self.received = dict(kwargs)
        return _Output("named-logits")


class _ExplicitModel(_Model):
    def forward(self, input_ids):
        return _Output("named-logits")


@pytest.mark.cpu_ok
def test_compute_loss_consumes_named_output_without_injecting_return_dict():
    model = _KwargsModel()
    output, metrics = model.compute_loss(input_ids="tokens", labels="labels")
    assert output == _Output("named-logits", loss=3.5)
    assert metrics.loss == 3.5
    assert "return_dict" not in model.received


@pytest.mark.cpu_ok
def test_compute_loss_drops_keywords_an_explicit_forward_does_not_accept():
    model = _ExplicitModel()
    output, metrics = model.compute_loss(input_ids="tokens", labels="labels", return_dict=False, ignored="metadata")
    assert output == _Output("named-logits", loss=3.5)
    assert metrics.loss == 3.5
