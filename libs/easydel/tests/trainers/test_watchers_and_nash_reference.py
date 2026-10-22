"""Host-only watcher and pre-compilation reference initialization regressions."""

from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import spectrax as spx
from easydel.trainers.group_relative_policy_optimization import grpo_trainer
from easydel.trainers.metrics import LogWatcher, run_watchers
from easydel.trainers.nash_md_trainer import nash_md_trainer

pytestmark = pytest.mark.cpu_ok


def test_watchers_accept_real_spectrax_state_with_python_leaves():
    params = spx.State({"layer": {"weight": 3.0, "bias": 2.0}})
    calls = []

    def watch(value):
        calls.append(value)
        return {"double": value * 2}

    watchers = [LogWatcher("probe", watch, interval=2, pattern=r"layer\.weight$")]
    assert run_watchers(watchers, params, step=1) == {}
    assert calls == []
    assert run_watchers(watchers, params, step=2) == {"probe/layer/weight/double": 6.0}
    assert calls == [3.0]


@dataclass(frozen=True)
class HostState:
    model: object
    graphstate: str = "policy-parameters"
    graphother: str = "buffers"
    step: int = 0
    graphdef: str = "definition"
    tx: object = None
    opt_state: object = None

    def replace(self, **kwargs):
        return replace(self, **kwargs)


@pytest.mark.parametrize("explicit_reference", [False, True])
def test_nash_zero_beta_installs_reference_before_parent_initialization(monkeypatch, explicit_reference):
    monkeypatch.setattr(grpo_trainer, "EasyDeLState", HostState)
    monkeypatch.setattr(nash_md_trainer, "EasyDeLState", HostState)
    monkeypatch.setattr(grpo_trainer, "jnp", SimpleNamespace(asarray=lambda values, dtype: values))
    monkeypatch.setattr(grpo_trainer, "replicated_named_sharding", lambda mesh: None)
    # HostState carries only Python leaves; the real frozen-policy copier can
    # safely exercise immutable replacement without allocating any device data.
    policy = HostState(SimpleNamespace(mesh=None))
    reference = replace(policy, graphstate="reference-parameters") if explicit_reference else None
    seen = []

    def parent_init(self, **kwargs):
        seen.append(self.ref_state)
        assert self.ref_state is not None
        assert self.ref_state.graphstate == ("reference-parameters" if explicit_reference else "policy-parameters")

    monkeypatch.setattr(grpo_trainer.Trainer, "__init__", parent_init)
    args = nash_md_trainer.NashMDConfig(beta=0.0, disable_dropout=False, log_completions=False)
    trainer = nash_md_trainer.NashMDTrainer(
        args,
        policy,
        reward_funcs=lambda **kwargs: [0.5],
        reference_model=reference,
        processing_class=SimpleNamespace(pad_token_id=0),
    )
    assert seen == [trainer.ref_state]
    assert trainer.ref_state is not policy
    assert policy.graphstate == "policy-parameters"
