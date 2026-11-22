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

"""Static SPMD CE-only forward eligibility; no model or JAX arrays constructed."""

from types import SimpleNamespace

import pytest
from easydel.infra.loss_utils import ForCausalLMLoss, LossConfig
from easydel.modules._base.causal_lm_module import BaseCausalLMModule
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM
from easydel.modules.gemma.modeling_gemma import GemmaForCausalLM
from easydel.modules.gemma2.modeling_gemma2 import Gemma2ForCausalLM
from easydel.modules.mixtral.modeling_mixtral import MixtralForCausalLM
from easydel.trainers.trainer import _fn
from easydel.trainers.trainer._fn import _has_reviewed_ce_only_forward

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("model_type", [GemmaForCausalLM, Gemma2ForCausalLM, CohereForCausalLM])
def test_reviewed_ce_only_forwards_are_eligible(model_type):
    assert model_type.forward is not BaseCausalLMModule.forward
    assert _has_reviewed_ce_only_forward(model_type)


def test_moe_and_unreviewed_subclasses_fail_closed():
    class UnreviewedGemma(GemmaForCausalLM):
        pass

    assert not _has_reviewed_ce_only_forward(MixtralForCausalLM)
    assert not _has_reviewed_ce_only_forward(UnreviewedGemma)


class _UnreviewedForward(BaseCausalLMModule):
    """CLM head with an overridden forward that is not on the reviewed CE-only list."""

    loss_function = property(lambda self: ForCausalLMLoss)
    _router_aux_loss_feature = None

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _OverridesComputeLoss(_UnreviewedForward):
    def compute_loss(self, **kwargs):
        raise NotImplementedError


def _call(model, batch=None):
    return {
        "state": SimpleNamespace(model=model),
        "batch": {"input_ids": object()} if batch is None else batch,
        "loss_config": LossConfig(),
        "loss_type": "nll",
    }


@pytest.fixture
def warnings_seen(monkeypatch):
    seen = []
    monkeypatch.setattr(_fn.logger, "warn_once", lambda message, *a, **k: seen.append(message))
    return seen


def test_non_clm_model_and_compute_loss_override_fall_back_with_a_named_reason(warnings_seen):
    assert _fn._make_base_spmd_weight(_call(object())) is None
    assert "not a BaseCausalLMModule" in warnings_seen[-1]

    model = object.__new__(_OverridesComputeLoss)
    assert _fn._make_base_spmd_weight(_call(model)) is None
    assert "_OverridesComputeLoss" in warnings_seen[-1]
    assert "overrides compute_loss" in warnings_seen[-1]


@pytest.mark.parametrize(
    ("probe", "fragment"),
    [
        (lambda *a: True, "adds an unnormalized auxiliary loss"),
        (lambda *a: (_ for _ in ()).throw(ValueError("trace boom")), "tracing compute_loss"),
    ],
)
def test_unreviewed_forward_with_aux_or_untraceable_falls_back_uniform(monkeypatch, warnings_seen, probe, fragment):
    monkeypatch.setattr(_fn, "_compute_loss_emits_aux", probe)
    model = object.__new__(_UnreviewedForward)
    assert _fn._make_base_spmd_weight(_call(model)) is None
    assert fragment in warnings_seen[-1]


def test_unreviewed_forward_proven_aux_free_is_token_weighted(monkeypatch, warnings_seen):
    """A non-whitelisted forward whose traced compute_loss has no aux term now gets token weighting."""
    monkeypatch.setattr(_fn, "_compute_loss_emits_aux", lambda *a: False)
    model = object.__new__(_UnreviewedForward)
    assert _fn._make_base_spmd_weight(_call(model)) is not None
    assert warnings_seen == []


def test_ce_aux_split_applies_only_to_standard_clm_nll():
    """The CE/aux split is built for the standard causal-LM compute_loss, never for DFT or other heads."""
    assert _fn._ce_aux_split_count(_call(object())) is None
    assert _fn._ce_aux_split_count(_call(object.__new__(_OverridesComputeLoss))) is None
    model = object.__new__(_UnreviewedForward)
    assert _fn._ce_aux_split_count(_call(model)) is not None
    assert _fn._ce_aux_split_count({**_call(model), "loss_type": "dft"}) is None


def test_unexpected_exception_type_in_the_probe_is_not_swallowed(monkeypatch):
    """Only the expected exception types fall back; anything else must surface."""

    def boom(*a):
        raise KeyboardInterrupt

    monkeypatch.setattr(_fn, "_compute_loss_emits_aux", boom)
    with pytest.raises(KeyboardInterrupt):
        _fn._make_base_spmd_weight(_call(object.__new__(_UnreviewedForward)))
