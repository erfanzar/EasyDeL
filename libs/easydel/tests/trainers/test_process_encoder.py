# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for on-the-fly vision-encoder processing (``process_encoder``).

The load-bearing claim is that lifting the vision tower out of the train step changes
*nothing* about what the decoder sees. The parity tests assert that directly against a
real VLM forward; the rest pin the packing/bucketing arithmetic and the discovery logic
that decides whether a family can be driven this way at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import easydel as ed
import jax
import numpy as np
import pytest
from easydel.trainers.process_encoder import (
    EncoderProcessor,
    PixelLayout,
    ProcessEncoderConfig,
    bucket_row_count,
    find_vision_tower_attr,
    flatten_features,
    pack_vision_inputs,
    pad_features_to_constant,
    resolve_encoder_binding,
    resolve_pixel_layout,
    select_vision_rows,
    validate_process_encoder,
)
from jax import numpy as jnp


def _tiny_paligemma():
    """A minimal PaliGemma: real vision tower, real merge path, cheap on CPU."""
    text_config = ed.GemmaConfig(
        vocab_size=512,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=256,
        max_position_embeddings=512,
        head_dim=32,
        tie_word_embeddings=True,
    )
    config = ed.PaliGemmaConfig(
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "image_size": 56,
            "patch_size": 14,
            "vision_use_head": False,
        },
        text_config=text_config,
        # Must match the text hidden size: the merge writes projected patch
        # embeddings straight into the text embedding stream.
        projection_dim=128,
        image_token_index=511,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_mechanism = "vanilla"
    # attn_dtype defaults to bfloat16 independently of the model dtype. Pin it so
    # the model really is float32: bf16 attention rounding amplifies last-bit
    # feature noise far past the tolerance, which the shape-numerics comparison
    # is not about.
    for sub in (config, config.get_text_config(), config.vision_config):
        sub.attn_dtype = jnp.float32
    model = ed.AutoEasyDeLModelForImageTextToText.from_config(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=ed.Rngs(0),
    )
    return model, config


def _vision_batch(config, batch_size: int = 2, seq_len: int = 32):
    """A batch whose placeholder count matches the tower's patch output exactly."""
    patches = (56 // 14) ** 2
    rng = np.random.default_rng(0)
    input_ids = np.asarray(rng.integers(1, 400, (batch_size, seq_len)), dtype=np.int32)
    # Placeholders occupy the first `patches` positions of every row, so total
    # placeholders == batch_size * patches == the flattened feature row count.
    input_ids[:, :patches] = config.image_token_index
    pixel_values = jnp.asarray(rng.normal(size=(batch_size, 3, 56, 56)), dtype=jnp.float32)
    return {
        "input_ids": jnp.asarray(input_ids),
        "attention_mask": jnp.ones((batch_size, seq_len), dtype=jnp.int32),
        "pixel_values": pixel_values,
    }


@pytest.fixture(scope="module")
def paligemma():
    return _tiny_paligemma()


def test_binding_discovery_finds_tower_and_feature_keyword(paligemma):
    model, _ = paligemma
    binding = resolve_encoder_binding(model)

    assert binding is not None, "PaliGemma should support out-of-band vision"
    assert binding.feature_kwarg in ("image_features", "image_embeds")
    assert find_vision_tower_attr(model) is not None
    assert "pixel_values" in binding.accepted_kwargs
    assert binding.feature_valid_length_kwarg == "image_features_valid_length"


def test_text_only_model_reports_no_binding():
    """A decoder-only model must degrade to the in-model path, not crash."""
    config = ed.LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = ed.AutoEasyDeLModelForCausalLM.from_config(
        config=config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=ed.Rngs(0)
    )
    assert resolve_encoder_binding(model) is None
    assert find_vision_tower_attr(model) is None

    with pytest.raises(ValueError, match="does not expose an out-of-band vision path"):
        validate_process_encoder(ProcessEncoderConfig(enabled=True), model)

    # require_support=False downgrades the same situation to a warning + fallback.
    assert validate_process_encoder(ProcessEncoderConfig(enabled=True, require_support=False), model) is None


def test_precomputed_features_match_the_in_model_vision_path(paligemma):
    """The whole feature rests on this: same logits with or without the tower in-step."""
    model, config = paligemma
    batch = _vision_batch(config)

    reference = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
    ).logits

    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=1))
    assert processor.supported
    processed = processor.process(batch)

    assert "pixel_values" not in processed, "pixels must be dropped so the tower does not re-run"
    feature_kwarg = processor.binding.feature_kwarg
    assert feature_kwarg in processed

    candidate = model(
        input_ids=processed["input_ids"],
        attention_mask=processed["attention_mask"],
        **{
            feature_kwarg: processed[feature_kwarg],
            processor.binding.feature_valid_length_kwarg: processed[processor.binding.feature_valid_length_kwarg],
        },
    ).logits

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(reference), rtol=1e-6, atol=1e-6)


def test_trailing_features_are_inert_in_the_merge():
    """The property the whole bucketing scheme rests on, asserted exactly.

    Explicit validity permits a capacity buffer; even a nonzero tail must not affect
    the merge. Without validity metadata, excess features remain an error.
    """
    from easydel.modules._base.vision_language_module import BaseVisionLanguageModule

    hidden, seq_len, placeholder = 8, 6, 99
    input_ids = jnp.asarray([[placeholder, placeholder, 1, 2, placeholder, 3]], dtype=jnp.int32)
    inputs_embeds = jnp.asarray(np.random.default_rng(0).normal(size=(1, seq_len, hidden)), dtype=jnp.float32)
    exact = jnp.asarray(np.random.default_rng(1).normal(size=(3, hidden)), dtype=jnp.float32)

    merged_exact = BaseVisionLanguageModule.merge_multimodal_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=exact,
        placeholder_token_id=placeholder,
    )
    merged_padded = BaseVisionLanguageModule.merge_multimodal_embeddings(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        multimodal_embeddings=jnp.concatenate([exact, jnp.full((29, hidden), 123.0)], axis=0),
        placeholder_token_id=placeholder,
        multimodal_embeddings_valid_length=jnp.asarray(3, dtype=jnp.int32),
    )

    np.testing.assert_array_equal(np.asarray(merged_padded), np.asarray(merged_exact))


def test_bucketed_padding_matches_the_in_model_path_within_shape_numerics(paligemma):
    """A coarse bucket must not change the result beyond float32 shape noise.

    Not bitwise: encoding a bucketed 8-row tower batch instead of 2 rows changes XLA's
    reduction order, which perturbs the genuine features in the last bits. The exact
    inertness claim is covered by ``test_trailing_features_are_inert_in_the_merge``; this
    test guards against a real misalignment, which would move logits by O(1).
    """
    model, config = paligemma
    batch = _vision_batch(config)

    reference = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        pixel_values=batch["pixel_values"],
    ).logits

    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=8))
    processed = processor.process(batch)
    feature_kwarg = processor.binding.feature_kwarg
    assert processed[feature_kwarg].shape == (128, 128)
    valid_length = processed[processor.binding.feature_valid_length_kwarg]
    assert valid_length.shape == ()
    assert valid_length.dtype == jnp.int32
    assert int(valid_length) == 32
    assert np.any(np.asarray(processed[feature_kwarg][32:]) != 0), "repeated-image tail should be nonzero"

    candidate = model(
        input_ids=processed["input_ids"],
        attention_mask=processed["attention_mask"],
        **{
            feature_kwarg: processed[feature_kwarg],
            processor.binding.feature_valid_length_kwarg: processed[processor.binding.feature_valid_length_kwarg],
        },
    ).logits

    np.testing.assert_allclose(np.asarray(candidate), np.asarray(reference), rtol=0, atol=1e-4)


def test_features_keep_a_constant_shape_across_varying_image_counts(paligemma):
    """The step's input signature must not move when the image count does."""
    model, config = paligemma
    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=4))

    shapes = set()
    for batch_size in (2, 3, 4):
        processed = processor.process(_vision_batch(config, batch_size=batch_size))
        shapes.add(processed[processor.binding.feature_kwarg].shape)

    assert len(shapes) == 1, f"feature shape moved across image counts: {shapes}"


def test_disabled_and_excluded_bucket_are_exact_no_ops(paligemma):
    model, config = paligemma
    batch = _vision_batch(config)

    disabled = EncoderProcessor(model, ProcessEncoderConfig(enabled=False))
    assert disabled.process(batch) is batch

    restricted = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, only_buckets=[1]))
    assert restricted.process(batch, bucket_index=0) is batch
    assert restricted.process(batch, bucket_index=None) is batch
    assert restricted.process(batch, bucket_index=1) is not batch


def test_batch_without_vision_is_untouched(paligemma):
    model, _ = paligemma
    text_only = {
        "input_ids": jnp.ones((2, 8), dtype=jnp.int32),
        "attention_mask": jnp.ones((2, 8), dtype=jnp.int32),
    }
    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True))
    assert processor.process(text_only) is text_only


def test_max_rows_per_step_falls_back_instead_of_dropping_images(paligemma):
    """Exceeding the cap must retain pixel_values, never silently discard an image."""
    model, config = paligemma
    batch = _vision_batch(config, batch_size=4)
    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, max_rows_per_step=2))

    processed = processor.process(batch)
    assert processed is batch
    assert "pixel_values" in processed


@pytest.mark.parametrize(
    ("count", "multiple", "ceiling", "expected"),
    [(0, 8, 512, 0), (1, 8, 512, 8), (300, 8, 512, 304), (509, 8, 512, 512), (512, 8, 512, 512), (7, 1, 512, 7)],
)
def test_bucket_row_count_rounds_up_without_exceeding_the_batch(count, multiple, ceiling, expected):
    assert bucket_row_count(count, multiple, ceiling) == expected


def test_image_major_batch_is_detected_without_a_presence_probe():
    """Dense pixels (what every collator emits today) must classify as image-major."""
    batch = {
        "input_ids": jnp.ones((4, 8), dtype=jnp.int32),
        "pixel_values": jnp.ones((6, 3, 8, 8), dtype=jnp.float32),
    }
    assert resolve_pixel_layout(batch) is PixelLayout.IMAGE_MAJOR

    selection = select_vision_rows(batch, ProcessEncoderConfig(enabled=True, row_bucket_multiple=4))
    assert selection is not None
    assert selection.selected_rows == 6
    assert selection.layout is PixelLayout.IMAGE_MAJOR


def test_row_major_batch_packs_away_text_only_rows():
    """With a presence signal, text-only rows are dropped and order is preserved."""
    pixels = jnp.asarray(np.stack([np.full((3, 4, 4), float(i)) for i in range(4)]), dtype=jnp.float32)
    batch = {
        "input_ids": jnp.ones((4, 8), dtype=jnp.int32),
        "pixel_values": pixels,
        "has_image": jnp.asarray([True, False, True, False]),
    }
    config = ProcessEncoderConfig(enabled=True, presence_key="has_image", row_bucket_multiple=1)

    assert resolve_pixel_layout(batch, "has_image") is PixelLayout.ROW_MAJOR
    selection = select_vision_rows(batch, config)
    assert selection is not None
    assert selection.selected_rows == 2
    np.testing.assert_array_equal(selection.indices, [0, 2])
    assert selection.saves_work

    packed = pack_vision_inputs(batch, selection)
    # Rows 0 and 2 survive, in that order (the merge is positional).
    np.testing.assert_allclose(np.asarray(packed["pixel_values"][0]), np.asarray(pixels[0]))
    np.testing.assert_allclose(np.asarray(packed["pixel_values"][1]), np.asarray(pixels[2]))


def test_zeroed_rows_are_detected_as_row_major():
    pixels = np.ones((3, 3, 4, 4), dtype=np.float32)
    pixels[1] = 0.0
    batch = {"input_ids": jnp.ones((3, 8), dtype=jnp.int32), "pixel_values": jnp.asarray(pixels)}

    assert resolve_pixel_layout(batch) is PixelLayout.ROW_MAJOR
    selection = select_vision_rows(batch, ProcessEncoderConfig(enabled=True, row_bucket_multiple=1))
    np.testing.assert_array_equal(selection.indices, [0, 2])


def test_pack_pads_by_repetition_to_the_bucket():
    """Padding repeats the last real row: grid metadata stays self-consistent."""
    pixels = jnp.asarray(np.stack([np.full((3, 2, 2), float(i)) for i in range(3)]), dtype=jnp.float32)
    batch = {"input_ids": jnp.ones((3, 4), dtype=jnp.int32), "pixel_values": pixels}
    config = ProcessEncoderConfig(enabled=True, row_bucket_multiple=1)
    selection = select_vision_rows(batch, config)
    selection.padded_rows = 3

    packed = pack_vision_inputs(batch, selection)
    assert packed["pixel_values"].shape[0] == 3


def test_pad_features_to_constant_appends_zeros_and_never_truncates():
    features = jnp.ones((5, 8), dtype=jnp.float32)

    padded = pad_features_to_constant(features, 8)
    assert padded.shape == (8, 8)
    np.testing.assert_allclose(np.asarray(padded[:5]), 1.0)
    np.testing.assert_allclose(np.asarray(padded[5:]), 0.0)

    assert pad_features_to_constant(features, 3).shape == (5, 8)


def test_flatten_features_handles_arrays_and_per_image_sequences():
    stacked = jnp.ones((2, 4, 16), dtype=jnp.float32)
    assert flatten_features(stacked).shape == (8, 16)

    per_image = [jnp.ones((4, 16), dtype=jnp.float32), jnp.ones((6, 16), dtype=jnp.float32)]
    assert flatten_features(per_image).shape == (10, 16)


def test_training_arguments_roundtrip_preserves_the_config():
    args = ed.TrainingArguments(
        model_name="pe",
        save_directory="/tmp/process-encoder-test",
        process_encoder={"enabled": True, "row_bucket_multiple": 16, "feature_dtype": "bfloat16"},
    )
    assert isinstance(args.process_encoder, ProcessEncoderConfig)
    assert args.process_encoder.row_bucket_multiple == 16

    restored = ed.TrainingArguments.from_dict(args.to_dict())
    assert isinstance(restored.process_encoder, ProcessEncoderConfig)
    assert restored.process_encoder.enabled
    assert restored.process_encoder.feature_dtype == "bfloat16"


@pytest.mark.parametrize("bad", [{"row_bucket_multiple": 0}, {"max_rows_per_step": 0}, {"only_buckets": []}])
def test_config_rejects_invalid_settings(bad):
    with pytest.raises(ValueError):
        ProcessEncoderConfig(**bad)


def _tiny_qwen2_vl():
    """A minimal Qwen2-VL: the ``image_embeds`` camp, with grid-enumerated patch rows."""
    hidden_size, num_heads = 64, 4
    head_dim = hidden_size // num_heads
    half = head_dim // 2
    mrope_section = [half // 4, (half - half // 4) // 2, half - half // 4 - (half - half // 4) // 2]
    vocab = 256
    config = ed.Qwen2VLConfig(
        text_config=dict(
            vocab_size=vocab,
            hidden_size=hidden_size,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=num_heads,
            num_key_value_heads=2,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            rope_theta=1000000.0,
            rope_scaling={"rope_type": "default", "mrope_section": mrope_section},
            tie_word_embeddings=True,
        ),
        vision_config=dict(
            depth=2,
            embed_dim=hidden_size,
            hidden_size=hidden_size,
            num_heads=4,
            in_channels=3,
            patch_size=4,
            spatial_merge_size=2,
            temporal_patch_size=1,
        ),
        image_token_id=vocab - 1,
        video_token_id=vocab - 4,
        vision_start_token_id=vocab - 3,
        vision_end_token_id=vocab - 2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.attn_mechanism = "vanilla"
    model = ed.AutoEasyDeLModelForImageTextToText.from_config(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=ed.Rngs(0),
    )
    return model, config


def test_grid_family_reports_the_image_embeds_binding():
    """Discovery for the ``image_embeds`` camp.

    Numerical parity for this camp is NOT covered here: constructing a synthetic Qwen2-VL
    vision tower whose mRoPE geometry accepts a hand-built patch grid proved fiddly, and a
    wrong geometry fails inside the tower on the reference path too, testing nothing. The
    threading edit in `modeling_qwen2_vl.py` is therefore verified structurally only --
    paligemma carries the end-to-end numerical proof of the mechanism.
    """
    model, _ = _tiny_qwen2_vl()
    binding = resolve_encoder_binding(model)

    assert binding is not None
    assert binding.feature_kwarg == "image_embeds"
    assert "image_grid_thw" in binding.accepted_kwargs


def test_grid_batches_are_not_row_padded():
    """Row padding would describe an image the grid does not contain."""
    batch = {
        "input_ids": jnp.ones((2, 8), dtype=jnp.int32),
        "pixel_values": jnp.ones((18, 48), dtype=jnp.float32),
        "image_grid_thw": jnp.asarray([[1, 3, 3], [1, 3, 3]], dtype=jnp.int32),
    }
    selection = select_vision_rows(batch, ProcessEncoderConfig(enabled=True, row_bucket_multiple=8))
    assert selection is not None
    assert selection.padded_rows == 18, "grid-described rows must not be bucketed"


class _ExactFeatureModel:
    """Small deterministic encoder with an exact-count consumer and real row identity."""

    vision_tower = object()
    _multimodal_merge_feature = SimpleNamespace(image_token_id=99)

    def __init__(self, output="rows", mesh=None):
        self.output = output
        self.mesh = mesh

    def forward(self, input_ids, image_features=None, **kwargs):
        return image_features

    def get_image_features(self, pixel_values, image_grid_thw=None):
        if self.output == "list":
            # Variable per-image lengths: flattening and dividing by bucket size
            # would claim 5.5 tokens per real pair instead of exactly 2 + 3.
            return [jnp.full((int(row[0]), 3), row[0]) for row in np.asarray(pixel_values)]
        if self.output == "flat":
            return jnp.broadcast_to(pixel_values[:, :1], (pixel_values.shape[0], 3))
        # Two intervening axes make the valid prefix rows * 2 * 2, not rows * 2.
        return jnp.broadcast_to(pixel_values[:, :1, None, None], (pixel_values.shape[0], 2, 2, 3))


class _PrefixFeatureModel(_ExactFeatureModel):
    def forward(self, input_ids, image_features=None, image_features_valid_length=None, **kwargs):
        return image_features


def _feature_batch(values, token_count):
    return {
        "input_ids": jnp.full((1, token_count), 99, dtype=jnp.int32),
        "pixel_values": jnp.asarray(values, dtype=jnp.float32).reshape(-1, 1),
    }


@pytest.mark.parametrize("image_count", [1, 3])
def test_missing_or_excess_images_rejected_despite_sufficient_bucket_capacity(paligemma, image_count):
    model, config = paligemma
    batch = _vision_batch(config)  # 32 placeholders, requiring exactly two images.
    batch["pixel_values"] = _vision_batch(config, batch_size=image_count)["pixel_values"]
    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=8))
    with pytest.raises(ValueError, match="genuine image features"):
        processor.process(batch)


def test_only_explicit_valid_length_parameter_advertises_capacity_support():
    exact = resolve_encoder_binding(_ExactFeatureModel())
    prefix = resolve_encoder_binding(_PrefixFeatureModel())
    assert exact.feature_valid_length_kwarg is None  # **kwargs alone is not a promise.
    assert prefix.feature_valid_length_kwarg == "image_features_valid_length"


def test_processor_preserves_selected_image_and_patch_order_with_nonzero_padding():
    model = _PrefixFeatureModel()
    processor = EncoderProcessor(
        model, ProcessEncoderConfig(enabled=True, presence_key="has_image", row_bucket_multiple=4)
    )
    batch = _feature_batch([1, 2, 3, 4], 8)
    batch["has_image"] = jnp.asarray([False, True, False, True])
    result = processor.process(batch)
    assert int(result["image_features_valid_length"]) == 8
    np.testing.assert_array_equal(np.asarray(result["image_features"][:, 0]), [2] * 4 + [4] * 12)
    assert result["image_features"].shape == (16, 3)
    assert "pixel_values" not in result


def test_nonmetadata_consumer_gets_exact_prefix_without_high_water_padding():
    processor = EncoderProcessor(_ExactFeatureModel(), ProcessEncoderConfig(enabled=True, row_bucket_multiple=4))
    large = processor.process(_feature_batch([1, 2, 3, 4, 5], 20))
    small = processor.process(_feature_batch([7, 8], 8))
    assert large["image_features"].shape == (20, 3)
    assert small["image_features"].shape == (8, 3)
    assert "image_features_valid_length" not in small
    np.testing.assert_array_equal(np.asarray(small["image_features"][:, 0]), [7] * 4 + [8] * 4)


def test_metadata_consumer_retains_capacity_but_updates_valid_prefix_and_dtype():
    processor = EncoderProcessor(
        _PrefixFeatureModel(),
        ProcessEncoderConfig(enabled=True, row_bucket_multiple=4, feature_dtype="bfloat16"),
    )
    large = processor.process(_feature_batch([1, 2, 3, 4, 5], 20))
    small = processor.process(_feature_batch([7, 8], 8))
    assert large["image_features"].shape == small["image_features"].shape == (32, 3)
    assert int(large["image_features_valid_length"]) == 20
    assert int(small["image_features_valid_length"]) == 8
    assert small["image_features"].dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(small["image_features"][16:]), 0)


def test_grid_large_then_small_does_not_leak_output_capacity():
    processor = EncoderProcessor(_ExactFeatureModel(output="flat"), ProcessEncoderConfig(enabled=True))
    for values in ([1, 2, 3, 4, 5], [6, 7]):
        batch = _feature_batch(values, len(values))
        batch["image_grid_thw"] = jnp.asarray([[1, 1, len(values)]], dtype=jnp.int32)
        result = processor.process(batch)
        assert result["image_features"].shape == (len(values), 3)
        np.testing.assert_array_equal(np.asarray(result["image_features"][:, 0]), values)
        np.testing.assert_array_equal(np.asarray(result["image_grid_thw"]), np.asarray(batch["image_grid_thw"]))


@pytest.mark.parametrize("metadata", [False, True])
def test_variable_per_image_list_counts_genuine_prefix_before_flattening(metadata):
    model = (_PrefixFeatureModel if metadata else _ExactFeatureModel)(output="list")
    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=4))
    result = processor.process(_feature_batch([2, 3], 5))
    expected = [2, 2, 3, 3, 3]
    if metadata:
        expected += [3] * 6
        assert int(result["image_features_valid_length"]) == 5
    np.testing.assert_array_equal(np.asarray(result["image_features"][:, 0]), expected)


@pytest.mark.parametrize("metadata", [False, True])
def test_padded_already_flat_output_falls_back_without_guessing_image_boundaries(metadata):
    model = (_PrefixFeatureModel if metadata else _ExactFeatureModel)(output="flat")
    processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=4))
    batch = _feature_batch([2, 3], 2)
    assert processor.process(batch) is batch
    # Without padding, the entire output is genuine, so a flat array is safe.
    exact_processor = EncoderProcessor(model, ProcessEncoderConfig(enabled=True, row_bucket_multiple=1))
    result = exact_processor.process(batch)
    np.testing.assert_array_equal(np.asarray(result["image_features"][:, 0]), [2, 3])
    if metadata:
        assert int(result["image_features_valid_length"]) == 2


def test_setup_rejects_gradient_accumulation_only_for_enabled_supported_processing():
    config = ProcessEncoderConfig(enabled=True)
    with pytest.raises(ValueError, match="gradient_accumulation_steps=1"):
        validate_process_encoder(config, _PrefixFeatureModel(), gradient_accumulation_steps=2)
    assert (
        validate_process_encoder(
            ProcessEncoderConfig(enabled=False), _PrefixFeatureModel(), gradient_accumulation_steps=2
        )
        is None
    )
    assert (
        validate_process_encoder(
            ProcessEncoderConfig(enabled=True, require_support=False), object(), gradient_accumulation_steps=2
        )
        is None
    )


@pytest.mark.parametrize("mesh", [SimpleNamespace(is_mpmd=True), SimpleNamespace(mpmd_dim=2)])
def test_setup_rejects_active_scheduled_mpmd_microbatching(mesh):
    from spectrax.runtime.schedules import Std1F1B

    with pytest.raises(ValueError, match="scheduled MPMD microbatching"):
        validate_process_encoder(
            ProcessEncoderConfig(enabled=True),
            _PrefixFeatureModel(mesh=mesh),
            mpmd_scheduler=Std1F1B(microbatches=2),
        )


def test_inert_scheduler_and_single_microbatch_do_not_disable_processing():
    from spectrax.runtime.schedules import Std1F1B

    for mesh, microbatches in ((SimpleNamespace(is_mpmd=False), 4), (SimpleNamespace(is_mpmd=True), 1)):
        model = _PrefixFeatureModel(mesh=mesh)
        config = ProcessEncoderConfig(enabled=True)
        binding = validate_process_encoder(config, model, mpmd_scheduler=Std1F1B(microbatches=microbatches))
        result = EncoderProcessor(model, config, binding=binding).process(_feature_batch([1, 2], 8))
        assert int(result["image_features_valid_length"]) == 8
        np.testing.assert_array_equal(np.asarray(result["image_features"][:8, 0]), [1] * 4 + [2] * 4)


def test_disabled_and_unsupported_processing_ignore_active_scheduler():
    from spectrax.runtime.schedules import Std1F1B

    schedule = Std1F1B(microbatches=2)
    model = _PrefixFeatureModel(mesh=SimpleNamespace(is_mpmd=True))
    assert validate_process_encoder(ProcessEncoderConfig(enabled=False), model, mpmd_scheduler=schedule) is None
    assert (
        validate_process_encoder(
            ProcessEncoderConfig(enabled=True, require_support=False), object(), mpmd_scheduler=schedule
        )
        is None
    )


@pytest.mark.parametrize(
    ("global_steps", "overrides", "only_buckets", "bad_bucket"),
    [(1, [None, 2], None, 1), (1, [None, 2], [1], 1), (2, [1, None], [1], 1)],
)
def test_applicable_bucket_accumulation_overrides_cannot_bypass_setup_guard(
    global_steps, overrides, only_buckets, bad_bucket
):
    with pytest.raises(ValueError, match=f"gradient_accumulation_steps=1 for bucket {bad_bucket}"):
        validate_process_encoder(
            ProcessEncoderConfig(enabled=True, only_buckets=only_buckets),
            _PrefixFeatureModel(),
            gradient_accumulation_steps=global_steps,
            bucket_gradient_accumulation_steps=overrides,
        )


@pytest.mark.parametrize(("global_steps", "overrides"), [(1, [None, 2]), (2, [1, None])])
def test_excluded_bucket_accumulation_does_not_reject_safe_processing(global_steps, overrides):
    model = _PrefixFeatureModel()
    config = ProcessEncoderConfig(enabled=True, only_buckets=[0])
    binding = validate_process_encoder(
        config,
        model,
        gradient_accumulation_steps=global_steps,
        bucket_gradient_accumulation_steps=overrides,
    )
    processor = EncoderProcessor(model, config, binding=binding)
    batch = _feature_batch([1, 2], 8)
    processed = processor.process(batch, bucket_index=0)
    assert int(processed["image_features_valid_length"]) == 8
    np.testing.assert_array_equal(np.asarray(processed["image_features"][:8, 0]), [1] * 4 + [2] * 4)
    assert processor.process(batch, bucket_index=1) is batch


def test_disabled_or_unsupported_processing_ignores_bucket_accumulation_overrides():
    assert (
        validate_process_encoder(
            ProcessEncoderConfig(enabled=False),
            _PrefixFeatureModel(),
            bucket_gradient_accumulation_steps=[2],
        )
        is None
    )
    assert (
        validate_process_encoder(
            ProcessEncoderConfig(enabled=True, require_support=False),
            object(),
            bucket_gradient_accumulation_steps=[2],
        )
        is None
    )
