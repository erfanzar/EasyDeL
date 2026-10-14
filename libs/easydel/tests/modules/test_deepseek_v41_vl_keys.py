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

"""Vision-language key mapping of DeepSeek-V4.1 against the REAL released key index.

The expected parameter names are listed explicitly from the module attribute structure of
``modeling_deepseek_v41_vision.py`` (``DeepseekV41VisionModel`` / ``DeepseekV41Aligner``) and of
``DeepseekV41VLModel`` (``image_start`` / ``image_end`` / ``image_newline`` parameters, ``gate.bias_vl`` on every MoE
router), never derived from the mapper. Pure Python: no array computation.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest
from easydel.modules.deepseek_v41 import DeepseekV41Config
from easydel.modules.deepseek_v41.modeling_deepseek_v41 import DEEPSEEK_V41_KEY_RULES, DEEPSEEK_V41_VL_KEY_RULES

pytestmark = pytest.mark.cpu_ok

# tests/modules/<this file>: parents[0]=modules, [1]=tests, [2]=libs/easydel, [3]=libs, [4]=repository root.
_INDEX_PATH = Path(__file__).resolve().parents[4] / ".xerxes/projects/deepseek-v41/ref/model.safetensors.index.json"
_VISION_ROOTS = ("vision.", "aligner.")
_IMAGE_ROOTS = ("image_start", "image_end", "image_newline")


@pytest.fixture(scope="module")
def keys() -> list[str]:
    if not _INDEX_PATH.is_file():
        pytest.skip(f"released DeepSeek-V4.1 key index not found at {_INDEX_PATH}")
    with _INDEX_PATH.open() as f:
        return list(json.load(f)["weight_map"])


@pytest.fixture(scope="module")
def config() -> DeepseekV41Config:
    return DeepseekV41Config(vision_config={"model_type": "deepseek_v41_vision", "num_hidden_layers": 32})


def _is_vl_key(key: str) -> bool:
    """Independent statement of the keys only the vision-language model owns."""
    return (
        key.startswith(_VISION_ROOTS)
        or key in _IMAGE_ROOTS
        or (key.startswith("layers.") and key.endswith(".ffn.gate.bias_vl"))
    )


def _expected_vl_names(config: DeepseekV41Config) -> set[str]:
    """Every vision / aligner / image-delimiter / bias_vl parameter of ``DeepseekV41ForConditionalGeneration``."""
    names = {
        # DeepseekV41VisionModel: patch_embed.proj (ColumnParallelLinear with bias), final norm
        "model.vision.patch_embed.proj.weight",
        "model.vision.patch_embed.proj.bias",
        "model.vision.norm.weight",
        # DeepseekV41Aligner: w1 / w2 with bias
        "model.aligner.w1.weight",
        "model.aligner.w1.bias",
        "model.aligner.w2.weight",
        "model.aligner.w2.bias",
        # DeepseekV41VLModel image-span delimiter vectors
        "model.image_start",
        "model.image_end",
        "model.image_newline",
    }
    for i in range(config.vision_params()["num_hidden_layers"]):
        block = f"model.vision.blocks.{i}."
        names |= {
            block + "norm1.weight",
            block + "norm2.weight",
            block + "attn.wqkv.weight",
            block + "attn.wqkv.bias",
            block + "attn.wo.weight",
            block + "attn.wo.bias",
            block + "mlp.w1.weight",  # bias-free
            block + "mlp.w2.weight",
        }
    for i in range(config.num_hidden_layers):
        names.add(f"model.layers.{i}.mlp.gate.bias_vl")
    return names


def test_vision_enabled_follows_the_layer_count():
    assert DeepseekV41Config().vision_enabled is False
    assert DeepseekV41Config(vision_config={}).vision_enabled is False
    assert DeepseekV41Config(vision_config={"num_hidden_layers": 0}).vision_enabled is False
    assert DeepseekV41Config(vision_config={"num_hidden_layers": 2}).vision_enabled is True


def test_vision_params_default_to_the_released_values():
    params = DeepseekV41Config(vision_config={"num_hidden_layers": 32}).vision_params()
    assert params == {
        "num_hidden_layers": 32,
        "hidden_size": 1024,
        "num_attention_heads": 16,
        "intermediate_size": 2816,
        "patch_size": 14,
        "rope_theta": 10000.0,
        "downsample_ratio": 3,
        "max_image_tokens": 1024,
        "min_pixels": 295936,
        "max_wh_ratio": None,
        "rms_norm_eps": 1e-6,
    }


def test_vision_config_survives_a_json_roundtrip():
    cfg = DeepseekV41Config(vision_config={"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4})
    again = DeepseekV41Config.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.vision_enabled
    assert again.vision_params() == cfg.vision_params()


def test_vl_normalizer_maps_every_vl_key_without_collisions(keys: list[str], config: DeepseekV41Config):
    vl_keys = [k for k in keys if _is_vl_key(k)]
    # 1 + 1 patch_embed (weight, bias), 32 * 8 block tensors, final norm, 4 aligner tensors, 3 delimiters, 40 biases
    assert len(vl_keys) == 2 + 32 * 8 + 1 + 4 + 3 + 40 == 306
    mapped = {k: DEEPSEEK_V41_VL_KEY_RULES(k) for k in vl_keys}
    assert all(isinstance(v, str) and v for v in mapped.values()), [k for k, v in mapped.items() if not v][:10]
    owners: dict[str, list[str]] = collections.defaultdict(list)
    for key, name in mapped.items():
        owners[name].append(key)
    assert not {n: s for n, s in owners.items() if len(s) > 1}, "two native keys map to one EasyDeL name"
    assert set(owners) == _expected_vl_names(config)


def test_text_normalizer_still_leaves_vl_keys_unowned(keys: list[str]):
    for key in (k for k in keys if _is_vl_key(k)):
        assert DEEPSEEK_V41_KEY_RULES(key) is None, key


def test_vl_normalizer_agrees_with_the_text_normalizer_elsewhere(keys: list[str]):
    for key in (k for k in keys if not _is_vl_key(k) and not k.startswith("mtp.")):
        assert DEEPSEEK_V41_VL_KEY_RULES(key) == DEEPSEEK_V41_KEY_RULES(key), key


def test_mtp_keys_stay_unmapped(keys: list[str]):
    mtp = [k for k in keys if k.startswith("mtp.")]
    assert mtp, "the release carries an MTP stack"
    assert any(k.endswith(".ffn.gate.bias_vl") for k in mtp)  # mtp.N.ffn.gate.bias_vl must not leak into layers
    assert all(DEEPSEEK_V41_VL_KEY_RULES(k) is None for k in mtp)
