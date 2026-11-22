"""Host-only snapshot and mask-routing regressions; no JAX dispatch."""

import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.modules._base import BaseCausalLMModule
from easydel.modules._base._features import RouterAuxLossFeature
from easydel.modules.cohere.modeling_cohere import CohereForCausalLM

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("override_kind", ["helper", "closure", "super"])
def test_composed_family_cap_snapshots_settings_and_releases_backbone(override_kind):
    class Backbone:
        pass

    class HostTask(BaseCausalLMModule):
        _auto_register = False
        get_lm_logit_postprocess = CohereForCausalLM.get_lm_logit_postprocess

        def __init__(self):
            self._logit_cap_feature = None
            self.config = SimpleNamespace(offset=0.5)
            self.logit_scale = 2.0
            self.backbone = Backbone()

        def transform(self, logits):
            return logits + self.config.offset

        def apply_logit_cap(self, logits):
            if override_kind == "super":
                logits = super().apply_logit_cap(logits)
            return self.transform(logits)

    def make_cap(owner):
        return lambda logits: owner.transform(logits)

    model = HostTask()
    if override_kind == "closure":
        model.apply_logit_cap = make_cap(model)
    callback = model.get_lm_logit_postprocess()
    model_ref, backbone_ref = weakref.ref(model), weakref.ref(model.backbone)
    model.config.offset = 100.0
    model.logit_scale = 99.0
    del model
    gc.collect()
    assert model_ref() is None
    assert backbone_ref() is None
    np.testing.assert_array_equal(callback(np.array([-1.0, 0.0, 1.0])), np.array([-1.0, 1.0, 3.0]))


@pytest.mark.parametrize("setting_access", ["property", "getattr", "computed_getattr"])
def test_scalar_cap_snapshots_support_descriptors_and_dynamic_attribute_names(setting_access):
    class Backbone:
        pass

    class HostTask(BaseCausalLMModule):
        _auto_register = False

        def __init__(self):
            self._logit_cap_feature = None
            self._offset = 0.5
            self.offset_name = "_offset"
            self.backbone = Backbone()

        @property
        def offset(self):
            return self._offset

        def apply_logit_cap(self, logits):
            if setting_access == "property":
                return logits + self.offset
            if setting_access == "getattr":
                return logits + getattr(self, "_offset", 0.0)
            return logits + getattr(self, self.offset_name, 0.0)

    model = HostTask()
    callback = model.get_base_lm_logit_postprocess()
    expected = np.array([0.5, 1.5])
    np.testing.assert_array_equal(callback(np.array([0.0, 1.0])), expected)
    model_ref, backbone_ref = weakref.ref(model), weakref.ref(model.backbone)
    model._offset = 99.0
    del model
    gc.collect()
    assert model_ref() is None
    assert backbone_ref() is None
    np.testing.assert_array_equal(callback(np.array([0.0, 1.0])), expected)


@pytest.mark.parametrize("entrypoint", ["forward", "forward_moe"])
def test_public_moe_aux_loss_excludes_padding(entrypoint, monkeypatch):
    from easydel.modules._base import _base_task_module

    # Substitute only the numerical primitive with a NumPy independent
    # reference. The public forward and raw-logit fallback remain real; this
    # checks their observable mask routing without dispatching JAX operations.
    def host_load_balance(*, gate_logits, num_experts, top_k, attention_mask):
        logits = np.concatenate(gate_logits, axis=0)
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        selected = np.argsort(logits, axis=-1)[:, -top_k:]
        assignments = np.eye(num_experts)[selected].mean(axis=1)
        mask = np.ones(logits.shape[0]) if attention_mask is None else np.asarray(attention_mask).reshape(-1)
        load = (assignments * mask[:, None]).sum(axis=0) / mask.sum()
        probability = (probabilities * mask[:, None]).sum(axis=0) / mask.sum()
        return num_experts * (load * probability).sum()

    monkeypatch.setattr(_base_task_module, "auxiliary_load_balancing_loss_func", host_load_balance)

    class HostTask(BaseCausalLMModule):
        _auto_register = False

        def __init__(self, logits):
            self.config = SimpleNamespace(num_experts_per_tok=1, output_router_logits=True)
            self._router_aux_loss_feature = RouterAuxLossFeature(0.25)
            self.logits = logits
            self._base_model_name = "backbone"
            self.backbone = self.run_backbone

        def run_backbone(self, **kwargs):
            return SimpleNamespace(
                last_hidden_state=None,
                hidden_states=None,
                attentions=None,
                past_key_values=None,
                router_logits=(self.logits,),
            )

    real_logits = np.array([[3.0, 0.0], [2.0, 0.0]])
    padded_logits = np.concatenate((real_logits, [[-100.0, 100.0], [-100.0, 100.0]]))
    unpadded, padded = HostTask(real_logits), HostTask(padded_logits)
    expected = getattr(unpadded, entrypoint)(attention_mask=np.array([[1, 1]]), apply_lm_head=False).aux_loss
    actual = getattr(padded, entrypoint)(attention_mask=np.array([[1, 1, 0, 0]]), apply_lm_head=False).aux_loss
    unmasked = getattr(padded, entrypoint)(attention_mask=None, apply_lm_head=False).aux_loss
    np.testing.assert_allclose(actual, expected)
    assert not np.isclose(unmasked, expected)
