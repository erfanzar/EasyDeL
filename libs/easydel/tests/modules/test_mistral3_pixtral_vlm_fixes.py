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

"""Regression tests for Pixtral / Mistral3 / LLaVA vision-path parity with HF.

Covers:
    * Mistral3 patch merger feature order (HF ``F.unfold`` is channel-major).
    * Pixtral block-diagonal mask reaching the attention as a real mask
      (previously an additive float mask was turned into all-False by
      ``MaskInfo.dynamic_init``) and non-causal vision attention.
    * Pixtral cropping padded images to ``image_sizes`` (variable-size batches).
    * LLaVA ``vision_feature_layer`` given as a list (HF concatenates layers).
"""

import copy

import easydel as ed
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

try:
    from tests.modules.test_utils import create_model_pair
except ImportError:
    from tests.modules.test_utils import create_model_pair  # pyright: ignore[reportImplicitRelativeImport]


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    """Compare against fp32 torch at full precision (TPU's default f32 matmul is a single bf16 pass)."""
    with jax.default_matmul_precision("highest"):
        yield


def test_mistral3_patch_merger_matches_hf_unfold_order():
    """EasyDeL merger must build ``(c, ki, kj)`` features like HF ``F.unfold``."""
    from easydel.modules.mistral3.modeling_mistral3 import Mistral3PatchMerger
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config as HFMistral3Config
    from transformers.models.mistral3.modeling_mistral3 import Mistral3PatchMerger as HFMistral3PatchMerger

    hidden, patch, merge = 8, 4, 2
    hf_config = HFMistral3Config(
        vision_config={"model_type": "pixtral", "hidden_size": hidden, "patch_size": patch},
        spatial_merge_size=merge,
    )
    torch.manual_seed(0)
    hf_merger = HFMistral3PatchMerger(hf_config).eval()

    ed_config = ed.Mistral3Config(
        vision_config=ed.PixtralVisionConfig(hidden_size=hidden, patch_size=patch),
        spatial_merge_size=merge,
    )
    ed_merger = Mistral3PatchMerger(ed_config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ed_merger.merging_layer.weight.value = jnp.asarray(hf_merger.merging_layer.weight.detach().numpy().T)

    # Non-square grids: (4 x 6) and (2 x 4) patches.
    image_sizes = [(16, 24), (8, 16)]
    num_tokens = sum((h // patch) * (w // patch) for h, w in image_sizes)
    features = np.random.default_rng(0).standard_normal((num_tokens, hidden)).astype(np.float32)

    with torch.no_grad():
        expected = hf_merger(torch.from_numpy(features), torch.tensor(image_sizes)).numpy()
    got = np.asarray(ed_merger(jnp.asarray(features), image_sizes))

    assert got.shape == expected.shape == (4 * 6 // 4 + 2 * 4 // 4, hidden)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_pixtral_block_segment_ids_match_hf_block_mask():
    """Segment IDs must reproduce HF's block-diagonal mask through ``MaskInfo``."""
    from easydel.modules.pixtral.modeling_pixtral import generate_block_segment_ids
    from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
    from transformers.models.pixtral.modeling_pixtral import generate_block_attention_mask as hf_block_mask

    lengths = [6, 3, 5]
    seq_len = sum(lengths)
    hf_mask = hf_block_mask(lengths, torch.zeros(1, seq_len, 4, dtype=torch.float32)).numpy()
    hf_allowed = hf_mask == 0.0  # (1, 1, S, S)

    segment_ids = generate_block_segment_ids(lengths, jnp.zeros((1, seq_len, 4), dtype=jnp.float32))
    allowed = np.asarray(MaskInfo.from_segments(segment_ids).attention_mask).astype(bool)

    assert allowed.shape[-2:] == (seq_len, seq_len)
    np.testing.assert_array_equal(np.broadcast_to(allowed, hf_allowed.shape), hf_allowed)


def _pixtral_config(small_model_config):
    cfg = small_model_config.copy()
    cfg["attn_dtype"] = jnp.float32
    cfg["attn_softmax_dtype"] = jnp.float32
    config = ed.PixtralVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=64,
        patch_size=16,
    )
    return config, cfg


def test_pixtral_variable_image_sizes_match_hf(small_model_config):
    """Padded batch of differently sized images: non-causal attention + per-image crop."""
    config, cfg = _pixtral_config(small_model_config)
    ed_model, hf_model, config = create_model_pair(
        module_name="pixtral",
        hf_class=transformers.PixtralVisionModel,
        task=ed.TaskType.BASE_VISION,
        config=config,
        small_model_config=cfg,
    )
    image_sizes = np.array([[64, 48], [32, 64]], dtype=np.int64)
    pixel_values = np.random.default_rng(0).standard_normal((2, 3, 64, 64)).astype(np.float32)

    with torch.no_grad():
        hf_out = hf_model(
            torch.from_numpy(pixel_values),
            image_sizes=torch.from_numpy(image_sizes),
        ).last_hidden_state.numpy()

    with config.mesh:
        ed_out = ed_model(
            pixel_values=jnp.asarray(pixel_values),
            image_sizes=image_sizes,
        ).last_hidden_state

    expected_tokens = (64 // 16) * (48 // 16) + (32 // 16) * (64 // 16)
    assert hf_out.shape == (1, expected_tokens, 64)
    assert ed_out.shape == hf_out.shape
    np.testing.assert_allclose(np.asarray(ed_out), hf_out, rtol=2e-4, atol=2e-4)


def test_llava_list_vision_feature_layer_matches_hf(small_model_config):
    """``vision_feature_layer`` as a list concatenates the selected layers like HF."""
    cfg = small_model_config.copy()
    cfg["attn_dtype"] = jnp.float32
    cfg["attn_softmax_dtype"] = jnp.float32
    vision_config = ed.CLIPVisionConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        image_size=28,
        patch_size=14,
    )
    text_config = ed.LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=1,
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        intermediate_size=cfg["intermediate_size"],
        max_position_embeddings=cfg["max_position_embeddings"],
    )
    config = ed.LlavaConfig(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=cfg["vocab_size"] - 1,
        vision_feature_layer=[-3, -1],
        vision_feature_select_strategy="default",
    )
    ed_model, hf_model, config = create_model_pair(
        module_name="llava",
        hf_class=transformers.LlavaForConditionalGeneration,
        task=ed.TaskType.IMAGE_TEXT_TO_TEXT,
        config=copy.deepcopy(config),
        small_model_config=cfg,
    )
    pixel_values = np.random.default_rng(0).standard_normal((2, 3, 28, 28)).astype(np.float32)

    with torch.no_grad():
        hf_features = hf_model.model.get_image_features(
            pixel_values=torch.from_numpy(pixel_values),
            vision_feature_layer=[-3, -1],
            vision_feature_select_strategy="default",
        )
    hf_features = getattr(hf_features, "pooler_output", hf_features)
    hf_features = torch.stack(list(hf_features)).numpy()

    with config.mesh:
        ed_features = ed_model.get_image_features(jnp.asarray(pixel_values))

    assert ed_features.shape == hf_features.shape
    np.testing.assert_allclose(np.asarray(ed_features), hf_features, rtol=2e-4, atol=2e-4)
