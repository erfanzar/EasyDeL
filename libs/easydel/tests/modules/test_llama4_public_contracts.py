"""Non-computational regression coverage for model registry and output contracts."""

import pickle
from dataclasses import fields
from types import SimpleNamespace

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.infra.modeling_outputs import VLMCausalLMOutput
from easydel.modules.llama import VisionLlamaConfig
from easydel.modules.llama4 import Llama4VisionConfig, Llama4VisionModel
from easydel.modules.llama4.modeling_llama4 import Llama4CausalLMOutputWithPast, Llama4ForConditionalGeneration


@pytest.mark.cpu_ok
@pytest.mark.parametrize("task", [TaskType.BASE_VISION, TaskType.BASE_MODULE])
def test_llama4_vision_config_resolves_registered_model(task):
    registration = registry.get_module_registration(task, Llama4VisionConfig.model_type)
    assert registration.module is Llama4VisionModel
    assert registration.config is Llama4VisionConfig
    assert registry.get_module_registration(task, "llama4_vision").module is Llama4VisionModel


@pytest.mark.cpu_ok
def test_llama4_forward_returns_documented_output_and_preserves_vlm_fields():
    logits, cache, hidden, attentions = object(), object(), object(), object()
    language_output = SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=hidden, attentions=attentions)
    model = SimpleNamespace(
        config=SimpleNamespace(output_attentions=True, output_hidden_states=True),
        language_model=lambda **kwargs: language_output,
    )
    output = Llama4ForConditionalGeneration.forward(model, inputs_embeds=object())
    assert isinstance(output, Llama4CausalLMOutputWithPast)
    assert isinstance(output, VLMCausalLMOutput)
    assert output.logits is logits
    assert output.past_key_values is cache
    assert output.hidden_states is hidden
    assert output.attentions is attentions
    assert output.image_hidden_states is None
    assert output.video_hidden_states is None


@pytest.mark.cpu_ok
def test_llama4_output_preserves_original_positional_field_mapping():
    values = tuple(object() for _ in range(6))
    output = Llama4CausalLMOutputWithPast(*values)
    names = ("loss", "logits", "past_key_values", "hidden_states", "attentions", "image_hidden_states")
    assert tuple(getattr(output, name) for name in names) == values
    assert tuple(output.keys()) == names
    assert output.to_tuple() == values
    assert output[0] is values[0]
    assert output[1] is values[1]


@pytest.mark.cpu_ok
def test_llama4_output_keeps_loss_first_with_optional_vlm_fields():
    output = Llama4CausalLMOutputWithPast(loss=1, logits=2, video_hidden_states=3, last_hidden_state=4)
    assert tuple(output.keys()) == ("loss", "logits", "last_hidden_state", "video_hidden_states")
    assert output.to_tuple() == (1, 2, 4, 3)
    assert output[0] == 1
    assert output[1] == 2
    restored = pickle.loads(pickle.dumps(output))
    assert isinstance(restored, Llama4CausalLMOutputWithPast)
    assert restored.to_tuple() == output.to_tuple()
    assert restored.loss == 1
    assert restored.logits == 2
    assert restored.last_hidden_state == 4
    assert restored.video_hidden_states == 3


@pytest.mark.cpu_ok
@pytest.mark.parametrize("constructor", [dict, list, iter])
def test_llama4_output_preserves_mapping_and_pair_iterable_constructors(constructor):
    pairs = [("loss", 1), ("logits", 2), ("image_hidden_states", 3), ("video_hidden_states", 4)]
    output = Llama4CausalLMOutputWithPast(constructor(pairs))
    assert output.loss == 1
    assert output.logits == 2
    assert output.image_hidden_states == 3
    assert output.video_hidden_states == 4
    assert tuple(output.items()) == tuple(pairs)
    assert output.to_tuple() == (1, 2, 3, 4)
    restored = pickle.loads(pickle.dumps(output))
    assert tuple(restored.items()) == tuple(pairs)


@pytest.mark.cpu_ok
def test_llama4_output_dataclass_order_matches_public_constructor():
    names = ("loss", "logits", "past_key_values", "hidden_states", "attentions", "image_hidden_states")
    assert tuple(field.name for field in fields(Llama4CausalLMOutputWithPast))[:6] == names


@pytest.mark.cpu_ok
def test_llama4_output_scalar_pytree_roundtrip_preserves_field_mapping():
    from jax.tree_util import tree_flatten, tree_unflatten

    output = Llama4CausalLMOutputWithPast(1, 2, 3, 4, 5, 6, video_hidden_states=7)
    leaves, structure = tree_flatten(output)
    restored = tree_unflatten(structure, leaves)
    assert isinstance(restored, VLMCausalLMOutput)
    assert tuple(restored.items()) == tuple(output.items())
    assert restored.loss == 1
    assert restored.logits == 2
    assert restored.image_hidden_states == 6
    assert restored.video_hidden_states == 7


@pytest.mark.cpu_ok
def test_llama4_output_mapping_without_loss_keeps_model_output_semantics():
    values = {"logits": 2, "hidden_states": 3}
    output = Llama4CausalLMOutputWithPast(values)
    assert tuple(output.items()) == tuple(values.items())
    # ModelOutput keeps the constructor's first field unless the mapping supplies it.
    assert output.loss is values
    assert output.logits == 2
    assert output.hidden_states == 3


@pytest.mark.cpu_ok
def test_llama4_output_rejects_malformed_pair_iterable():
    with pytest.raises(ValueError, match="Cannot set key/value"):
        Llama4CausalLMOutputWithPast([("loss", 1), ("logits",)])


@pytest.mark.cpu_ok
def test_public_vision_llama_config_preserves_vision_settings_on_roundtrip():
    config = VisionLlamaConfig(vision_vocab_size=32, tie_vision_embeddings=True, sample_mode="random")
    restored = VisionLlamaConfig.from_dict(config.to_dict())
    assert restored.vision_vocab_size == 32
    assert restored.tie_vision_embeddings is True
    assert restored.sample_mode == "random"
