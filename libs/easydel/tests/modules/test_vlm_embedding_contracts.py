"""Host-only shared model contracts; no device arrays or model initialization."""

from types import SimpleNamespace

import pytest
from easydel.modules._base import BaseCausalLMModule, QwenVLMultimodalMixin
from easydel.trainers.trainer._fn import _has_reviewed_ce_only_forward

pytestmark = pytest.mark.cpu_ok


class _TextEmbeddings:
    def compute_embedding(self, input_ids):
        return self.text_embeddings


class _Model(QwenVLMultimodalMixin, _TextEmbeddings):
    config = SimpleNamespace(image_token_id=10, video_token_id=20)

    def __init__(self):
        self.text_embeddings = object()

    def get_rope_index(self, **kwargs):
        return "positions", "deltas"


def test_precomputed_text_and_deepstack_are_preserved_without_pixels():
    model = _Model()
    text, mask, residual = object(), object(), object()
    result, info = model.prepare_multimodal_embeddings(
        None, inputs_embeds=text, visual_pos_masks=mask, deepstack_visual_embeds=[residual]
    )
    assert result is text
    assert info.visual_pos_masks is mask
    assert info.deepstack_visual_embeds == [residual]
    assert info.deepstack_image_embeds is None
    assert info.deepstack_video_embeds is None


def test_text_only_embedding_and_info_use_existing_base_interface():
    model = _Model()
    assert model.compute_embedding("token ids") is model.text_embeddings
    embeddings, info = model.compute_embedding_with_info("token ids")
    assert embeddings is model.text_embeddings
    assert info.position_ids == "positions"
    assert info.rope_deltas == "deltas"
    assert info.visual_pos_masks is None
    assert info.deepstack_visual_embeds is None


def test_embedding_boundaries_reject_missing_inputs_and_deepstack_mask():
    model = _Model()
    with pytest.raises(ValueError, match="input_ids"):
        model.compute_embedding(None)
    with pytest.raises(ValueError, match="compute_embedding_with_info"):
        model.compute_embedding_with_info(None, inputs_embeds=object())
    with pytest.raises(ValueError, match="visual_pos_masks"):
        model.prepare_multimodal_embeddings(None, inputs_embeds=object(), deepstack_visual_embeds=[object()])


@pytest.mark.parametrize("grid", [None, [], [[1, 2]], [[0, 2, 2]], [[1, -1, 2]]])
def test_invalid_host_grid_is_rejected_before_vision_execution(grid):
    with pytest.raises(ValueError, match="image_grid_thw"):
        _Model().get_image_features(object(), image_grid_thw=grid)


def test_single_modality_deepstack_metadata_is_not_repacked():
    mask, residuals = object(), [object()]
    assert _Model.merge_deepstack_features(mask, None, residuals, None) == (mask, residuals)
    assert _Model.merge_deepstack_features(None, mask, None, residuals) == (mask, residuals)
    assert _Model.merge_deepstack_features(None, None, None, None) == (None, None)


class _HostMask:
    def __init__(self, count):
        self.count = count

    def sum(self):
        return self.count


class _HostIDs:
    def __eq__(self, token_id):
        return _HostMask(2 if token_id == 10 else 1)


def test_placeholder_validation_uses_supplied_feature_sizes():
    model = _Model()
    text = SimpleNamespace(shape=(1, 5, 4))
    images = SimpleNamespace(size=8, shape=(2, 4))
    videos = SimpleNamespace(size=4, shape=(1, 4))
    image_mask, video_mask = model.get_placeholder_mask(_HostIDs(), text, images, videos)
    assert (image_mask.sum(), video_mask.sum()) == (2, 1)
    with pytest.raises(ValueError, match="Image features and image tokens do not match"):
        model.get_placeholder_mask(_HostIDs(), text, videos, None)
    with pytest.raises(ValueError, match="Video features and video tokens do not match"):
        model.get_placeholder_mask(_HostIDs(), text, None, images)


def test_ce_only_capability_accepts_base_forward_but_not_unreviewed_custom_subclasses():
    class InheritedBase(BaseCausalLMModule):
        pass

    class CustomCEOnly(BaseCausalLMModule):
        _ce_only_forward = True

        def forward(self):
            raise NotImplementedError

    class UnreviewedSubclass(CustomCEOnly):
        pass

    class ReviewedSubclass(CustomCEOnly):
        _ce_only_forward = True

    assert BaseCausalLMModule._ce_only_forward is False
    assert _has_reviewed_ce_only_forward(InheritedBase)
    assert _has_reviewed_ce_only_forward(CustomCEOnly)
    assert not _has_reviewed_ce_only_forward(UnreviewedSubclass)
    assert _has_reviewed_ce_only_forward(ReviewedSubclass)
