"""Qwen2-VL output integration without JAX numerics or model initialization.

EasyDeL forwards always return their named output type and the updated cache;
training, generation and serving consume it directly.
"""

from types import SimpleNamespace

import pytest
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.loss_utils import LossMetrics
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.infra.modeling_outputs import BaseModelOutput
from easydel.modules.qwen2_vl import modeling_qwen2_vl

pytestmark = pytest.mark.cpu_ok


class _Backbone:
    """Real Qwen2-VL backbone forward with a non-numerical text decoder."""

    forward = modeling_qwen2_vl.Qwen2VLModel.forward

    def __init__(self, model):
        self.config = model.config
        self.model = model

    def __call__(self, **kwargs):
        return self.forward(**kwargs)

    def compute_embedding(self, *args, **kwargs):
        return object()

    def get_rope_index(self, **kwargs):
        return SimpleNamespace(ndim=3), None

    def language_model(self, **kwargs):
        return BaseModelOutput(last_hidden_state=self.model.hidden, past_key_values=self.model.cache)


class _Model:
    """Real Qwen2-VL forwards with non-numerical text-decoder/head boundaries."""

    forward = modeling_qwen2_vl.Qwen2VLForConditionalGeneration.forward
    compute_loss = EasyDeLBaseModule.compute_loss
    _call_generation_model_step = EasyGenerationMixin._call_generation_model_step

    def __init__(self):
        self.hidden = object()
        self.logits = object()
        self.cache = object()
        self.config = SimpleNamespace(
            output_attentions=False,
            output_hidden_states=False,
            runtime_sharding_resolver=None,
            mesh=None,
            partition_axis=None,
            sharding_axis_names=(),
            sharding_axis_dims=(),
            text_config=SimpleNamespace(use_cache=True),
        )
        self.base_model = _Backbone(self)
        self.mesh = SimpleNamespace(is_mpmd=False)
        self.loss_strategy = SimpleNamespace(
            plan_forward=lambda **kwargs: SimpleNamespace(forward_kwargs={}),
            compute=self._loss,
        )

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def compute_lm_logits(self, hidden):
        return self.logits

    def apply_logit_cap(self, logits):
        return logits

    def _prepare_mask_info_for_generation_step(self, running_token, kwargs):
        return kwargs

    def loss_function(self):
        """Only its name is read by the base loss dispatcher."""

    def _loss(self, **kwargs):
        assert kwargs["outputs"].logits is self.logits
        return LossMetrics(loss=7.0)


@pytest.fixture
def model(monkeypatch):
    monkeypatch.setattr(modeling_qwen2_vl, "apply_logical_sharding", lambda value, **kwargs: value)
    # Position inference is a numerical boundary too: embeddings-only forwards
    # infer it from MaskInfo instead of the token-based RoPE helper.
    monkeypatch.setattr(
        modeling_qwen2_vl,
        "MaskInfo",
        SimpleNamespace(dynamic_init=lambda **kwargs: SimpleNamespace(q_position_ids=SimpleNamespace(ndim=3))),
    )
    return _Model()


def test_task_forward_returns_named_outputs(model):
    output = model()
    assert output.logits is model.logits
    assert output.last_hidden_state is model.hidden
    assert output.past_key_values is model.cache


@pytest.mark.parametrize("use_embeddings", [False, True])
def test_generation_step_returns_named_outputs(model, use_embeddings):
    kwargs = {"inputs_embeds": object()} if use_embeddings else {}
    output = model._call_generation_model_step(model, object(), kwargs)
    assert output.logits is model.logits
    assert output.past_key_values is model.cache


def test_compute_loss_consumes_named_outputs(model):
    output, metrics = model.compute_loss(labels=object())
    assert output.logits is model.logits
    assert output.loss == metrics.loss == 7.0


@pytest.mark.parametrize("config_use_cache", [False, True])
@pytest.mark.parametrize("use_cache", [None, False, True])
def test_task_forward_always_returns_updated_cache(model, config_use_cache, use_cache):
    model.config.text_config.use_cache = config_use_cache
    kwargs = {} if use_cache is None else {"use_cache": use_cache}
    output = model(past_key_values=object(), **kwargs)
    assert output.past_key_values is model.cache


@pytest.mark.parametrize("use_embeddings", [False, True])
def test_generation_step_keeps_updated_cache_when_checkpoint_disables_cache(model, use_embeddings):
    model.config.text_config.use_cache = False
    kwargs = {"past_key_values": object()}
    if use_embeddings:
        kwargs["inputs_embeds"] = object()
    output = model._call_generation_model_step(model, object(), kwargs)
    assert output.past_key_values is model.cache


@pytest.mark.parametrize("apply_lm_head", [False, True])
def test_serving_style_call_keeps_updated_cache_when_checkpoint_disables_cache(model, apply_lm_head):
    # Mirrors the eSurge executor call: the model is invoked directly.
    model.config.text_config.use_cache = False
    output = model(past_key_values=object(), cache_metadata=None, apply_lm_head=apply_lm_head)
    assert output.past_key_values is model.cache
    assert output.last_hidden_state is model.hidden
    assert output.logits is (model.logits if apply_lm_head else None)
