"""Non-computation serialization and cached-feature input-validation regressions."""

import copy
import pickle
from dataclasses import fields
from types import SimpleNamespace

import pytest
from easydel.infra.modeling_outputs import EmbeddingInfo
from easydel.modules.qwen3_vl import Qwen3VLCausalLMOutputWithPast
from easydel.modules.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLModel
from easydel.modules.qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast
from easydel.modules.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration, Qwen3VLMoeModel

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("output_cls", [Qwen3VLCausalLMOutputWithPast, Qwen3VLMoeCausalLMOutputWithPast])
@pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
@pytest.mark.parametrize("sparse", [False, True])
def test_family_output_pickle_and_deepcopy_preserve_all_fields_and_mapping(output_cls, protocol, sparse):
    values = {field.name: (field.name, index) for index, field in enumerate(fields(output_cls))}
    if sparse:
        values.update(loss=None, past_key_values=None, hidden_states=None, video_hidden_states=None)
    output = output_cls(**values)
    output.review_metadata = {"unchanged": [1, 2]}
    for restored in (pickle.loads(pickle.dumps(output, protocol=protocol)), copy.deepcopy(output)):
        assert type(restored) is output_cls
        assert {field.name: getattr(restored, field.name) for field in fields(restored)} == values
        assert list(restored.items()) == list(output.items())
        assert restored.to_tuple() == output.to_tuple()
        assert restored[:] == output[:]
        assert restored.review_metadata == output.review_metadata
        assert restored.review_metadata is not output.review_metadata
        replaced = restored.replace(logits="replacement")
        assert replaced.logits == "replacement"
        assert replaced.rope_deltas == values["rope_deltas"]


@pytest.mark.parametrize("cls", [Qwen3VLModel, Qwen3VLMoeModel])
@pytest.mark.parametrize("modality", ["image", "video"])
@pytest.mark.parametrize("features", [None, SimpleNamespace(shape=(2, 3))])
def test_cached_deepstack_requires_corresponding_feature_shapes(cls, modality, features):
    model = object.__new__(cls)
    kwargs = {
        f"{modality}_embeds": features,
        f"deepstack_{modality}_embeds": [SimpleNamespace(shape=(1, 3))],
    }
    with pytest.raises(ValueError, match=f"Cached {modality} DeepStack residuals require matching"):
        model.compute_embedding_with_info(object(), **kwargs)


@pytest.mark.parametrize("cls", [Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration])
def test_wrapper_surfaces_cached_deepstack_and_positions_from_one_embedding_pass(cls):
    wrapper = object.__new__(cls)
    images, videos = [object()], [object()]
    positions, deltas, mask, embeddings = (object() for _ in range(4))
    calls = []

    class Encoder:
        def compute_embedding_with_info(self, input_ids, **kwargs):
            calls.append(input_ids)
            return embeddings, EmbeddingInfo(
                deepstack_image_embeds=kwargs["deepstack_image_embeds"],
                deepstack_video_embeds=kwargs["deepstack_video_embeds"],
                deepstack_visual_embeds=images + videos,
                visual_pos_masks=mask,
                position_ids=positions,
                rope_deltas=deltas,
            )

    object.__setattr__(wrapper, "model", Encoder())
    ids = object()
    result, info = wrapper.compute_embedding_with_info(ids, deepstack_image_embeds=images, deepstack_video_embeds=videos)
    assert result is embeddings
    assert info.deepstack_image_embeds is images
    assert info.deepstack_video_embeds is videos
    assert info.deepstack_visual_embeds == images + videos
    assert info.visual_pos_masks is mask
    assert info.position_ids is positions
    assert info.rope_deltas is deltas
    assert calls == [ids]
