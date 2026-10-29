"""Metadata and symbolic forward regressions; no JAX array computation."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.infra.modeling_outputs import DecoderLayerOutput
from easydel.modules.auto.auto_configuration import AutoEasyDeLConfig, infer_task_from_config
from easydel.modules.aya_vision.aya_vision_configuration import AyaVisionConfig
from easydel.modules.clip.clip_configuration import CLIPConfig, CLIPTextConfig, CLIPVisionConfig
from easydel.modules.clip.modeling_clip import CLIPForImageClassification
from easydel.modules.cohere import modeling_cohere
from easydel.modules.cohere2 import modeling_cohere2
from easydel.modules.dbrx.modeling_dbrx import DbrxForCausalLM
from transformers import CLIPConfig as HFCLIPConfig
from transformers import SiglipConfig as HFSiglipConfig

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "config_class,architecture",
    [(HFCLIPConfig, "CLIPForImageClassification"), (HFSiglipConfig, "SiglipForImageClassification")],
)
def test_explicit_image_classifier_architecture_overrides_family(config_class, architecture):
    config = config_class(architectures=[architecture])
    assert infer_task_from_config(config) is TaskType.IMAGE_CLASSIFICATION
    config.architectures = [architecture.replace("ForImageClassification", "Model")]
    assert infer_task_from_config(config) is TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION
    config.architectures = None
    assert infer_task_from_config(config) is TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION


@pytest.mark.parametrize(
    "architecture,task", [("CLIPTextModel", TaskType.BASE_MODULE), ("CLIPTextModelWithProjection", TaskType.EMBEDDING)]
)
def test_clip_text_default_auto_bind_roundtrip(tmp_path, architecture, task):
    original = CLIPTextConfig(architectures=[architecture], projection_dim=17)
    original.save_pretrained(tmp_path)
    loaded = AutoEasyDeLConfig.from_pretrained(str(tmp_path), local_files_only=True)
    assert isinstance(loaded, CLIPTextConfig)
    assert loaded.projection_dim == 17
    assert infer_task_from_config(loaded) is task
    assert registry.get_module_registration(task, loaded.model_type).module.__name__ == architecture


@pytest.mark.parametrize("composite", [False, True])
def test_clip_classifier_config_auto_bind_and_serialized_registry(tmp_path, composite):
    vision = CLIPVisionConfig(
        hidden_size=32, num_attention_heads=2, architectures=["CLIPForImageClassification"], num_labels=3
    )
    original = (
        CLIPConfig(vision_config=vision.to_dict(), architectures=["CLIPForImageClassification"], num_labels=3)
        if composite
        else vision
    )
    original.save_pretrained(tmp_path)
    loaded = AutoEasyDeLConfig.from_pretrained(str(tmp_path), local_files_only=True)
    assert loaded.num_labels == 3
    registration = registry.get_module_registration(TaskType.IMAGE_CLASSIFICATION, loaded.model_type)
    assert registration.module is CLIPForImageClassification
    assert (loaded.vision_config if composite else loaded).hidden_size == 32
    loaded.save_pretrained(tmp_path / "again")
    reloaded = AutoEasyDeLConfig.from_pretrained(
        str(tmp_path / "again"), model_task=TaskType.IMAGE_CLASSIFICATION, local_files_only=True
    )
    assert type(reloaded) is type(loaded)
    assert reloaded.num_labels == 3


@pytest.mark.parametrize("from_torch", [False, True])
def test_auto_config_transport_options_do_not_become_serializable_attributes(tmp_path, from_torch):
    from easydel.modules.cohere.cohere_configuration import CohereConfig

    CohereConfig(architectures=["CohereForCausalLM"]).save_pretrained(tmp_path)
    loaded = AutoEasyDeLConfig.from_pretrained(
        str(tmp_path),
        from_torch=from_torch,
        local_files_only=True,
        token="review-secret",
        proxies={"https": "https://user:password@example.invalid"},
        hidden_size=32,
    )
    assert loaded.hidden_size == 32
    for key in ("token", "proxies", "local_files_only"):
        assert key not in vars(loaded)
        assert key not in loaded.to_dict()
    assert "review-secret" not in loaded.to_json_string()
    assert "user:password" not in loaded.to_json_string()


def test_aya_default_vision_heads_are_valid_and_roundtrip():
    config = AyaVisionConfig()
    assert config.vision_config.hidden_size == 1152
    assert config.vision_config.num_attention_heads == 16
    assert config.vision_config.hidden_size % config.vision_config.num_attention_heads == 0
    restored = AyaVisionConfig.from_dict(config.to_dict())
    assert restored.vision_config.num_attention_heads == 16


@pytest.mark.parametrize("heads", [0, 7])
def test_aya_rejects_invalid_explicit_vision_head_dimensions(heads):
    with pytest.raises(ValueError, match="divisible"):
        AyaVisionConfig(
            vision_config={"model_type": "clip_vision_model", "hidden_size": 32, "num_attention_heads": heads}
        )


class _Hidden:
    shape = (4, 8, 64)


class _Layers(list):
    def scan(self, body, carry, **kwargs):
        for layer in self:
            carry = body(layer, carry)
        return carry


@pytest.mark.parametrize("module,class_name", [(modeling_cohere, "CohereModel"), (modeling_cohere2, "Cohere2Model")])
@pytest.mark.parametrize("override", [None, False, True])
def test_cohere_backbone_output_flags_follow_config_and_explicit_override(monkeypatch, module, class_name, override):
    hidden, attention = _Hidden(), object()
    monkeypatch.setattr(module, "apply_logical_sharding", lambda value, **kwargs: value)
    monkeypatch.setattr(
        module, "MaskInfo", SimpleNamespace(dynamic_init=lambda **kwargs: SimpleNamespace(q_position_ids=None))
    )

    def layer(**kwargs):
        return DecoderLayerOutput(hidden_states=hidden, attention_weight=attention, cache_view=None)

    probe = SimpleNamespace(
        config=SimpleNamespace(
            output_attentions=True, output_hidden_states=True, max_position_embeddings=32, runtime_sharding_resolver=None
        ),
        layers=_Layers([layer]),
        norm=lambda x: x,
        frequencies=None,
        _layer_scan_trace=lambda *args, **kwargs: True,
        _layer_scan_kwargs=lambda *args, **kwargs: {},
        _layer_stage_context=lambda *args, **kwargs: nullcontext(),
        _layer_cache_view_at=lambda *args, **kwargs: None,
        _layer_cache_view_update=lambda *args, **kwargs: None,
        _mark_layer_stage_boundary=lambda x, *args, **kwargs: x,
    )
    output = getattr(module, class_name).forward(
        probe,
        inputs_embeds=hidden,
        mode="train",
        past_key_values=SimpleNamespace(views=[None]),
        output_attentions=override,
        output_hidden_states=override,
    )
    if override is False:
        assert output.hidden_states is None
        assert output.attentions is None
    else:
        assert output.hidden_states == (hidden, hidden)
        assert output.attentions == (attention,)


def test_cohere_causal_lm_always_returns_named_outputs():
    hidden, cache = object(), object()
    backbone = SimpleNamespace(last_hidden_state=hidden, hidden_states=None, attentions=None, past_key_values=cache)
    probe = SimpleNamespace(config=SimpleNamespace(), model=lambda **kwargs: backbone)
    output = modeling_cohere.CohereForCausalLM.forward(probe, apply_lm_head=False)
    assert output.last_hidden_state is hidden
    assert output.past_key_values is cache
    assert output.logits is None


def test_dbrx_causal_lm_accepts_embeddings_only_before_backbone_dispatch():
    embeddings = object()
    probe = SimpleNamespace(forward_moe=lambda **kwargs: kwargs, _compute_aux_loss=None)
    output = DbrxForCausalLM.forward(probe, inputs_embeds=embeddings)
    assert output["input_ids"] is None
    assert output["inputs_embeds"] is embeddings


def test_legacy_composite_clip_classifier_preserves_vision_and_label_metadata(monkeypatch):
    from easydel.modules._base import BaseImageClassificationModule

    def initialize(self, **kwargs):
        self.config = kwargs["config"]

    monkeypatch.setattr(BaseImageClassificationModule, "__init__", initialize)
    config = CLIPConfig(vision_config={"hidden_size": 32, "num_attention_heads": 2}, num_labels=7)
    model = CLIPForImageClassification(config, rngs=None)
    assert isinstance(model.config, CLIPVisionConfig)
    assert model.config.hidden_size == 32
    assert model.config.num_labels == 7
    assert model.config.sharding_axis_dims == config.sharding_axis_dims


@pytest.mark.parametrize("shape", [(4, 16, 16), (4, 16, 16, 4), (4, 3, 16, 3)])
def test_clip_public_vision_rejects_invalid_or_ambiguous_layouts(shape):
    from functools import partial

    from easydel.modules.clip.modeling_clip import CLIPVisionModel, CLIPVisionTransformer

    # Invalid inputs must be rejected before embedding/convolution dispatch;
    # shape-only objects intentionally cannot execute any array computation.
    tower = SimpleNamespace(config=CLIPVisionConfig())
    probe = SimpleNamespace(vision_model=partial(CLIPVisionTransformer.forward, tower))
    pixels = SimpleNamespace(ndim=len(shape), shape=shape)
    with pytest.raises(ValueError, match=r"rank-four|unambiguous channel axis"):
        CLIPVisionModel.forward(probe, pixel_values=pixels)
