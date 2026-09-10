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

import pytest
from easydel.infra.mixins.bridge import _normalize_quantization_config
from easydel.layers.quantization import QuantizationConfig, QuantizationType

#: The exact ``quantization_config`` block DeepSeek-V4-Flash ships in its HF
#: ``config.json``, which EasyDeL-converted checkpoints inherit verbatim.
HF_FP8_DESCRIPTOR = {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "scale_fmt": "ue8m0",
    "weight_block_size": [128, 128],
}


def test_coerce_none_and_instance_passthrough():
    assert QuantizationConfig.coerce(None) is None
    config = QuantizationConfig(dtype=QuantizationType.NF4, group_size=64)
    assert QuantizationConfig.coerce(config) is config


def test_coerce_easydel_style_dict_unchanged():
    config = QuantizationConfig.coerce({"dtype": "int8", "group_size": "128"})
    assert isinstance(config, QuantizationConfig)
    assert config.dtype is QuantizationType.INT8
    assert config.group_size == 128


def test_checkpoint_inherited_hf_descriptor_is_dropped_not_raised():
    """Regression: ``from_pretrained`` coerces the checkpoint's inherited
    ``quantization_config``; an HF descriptor block must degrade to ``None``
    (no load-time quantization requested), not raise ``TypeError`` on
    unknown keys like ``activation_scheme``."""
    assert QuantizationConfig.coerce(HF_FP8_DESCRIPTOR, strict=False) is None
    assert _normalize_quantization_config(HF_FP8_DESCRIPTOR, strict=False) is None


def test_user_supplied_hf_descriptor_fails_loudly():
    """An explicitly passed HF block cannot silently skip the quantization
    the caller asked for; strict mode rejects it with guidance."""
    with pytest.raises(ValueError, match="quant_method"):
        QuantizationConfig.coerce(HF_FP8_DESCRIPTOR)


def test_user_supplied_unknown_keys_strict_raises_with_valid_fields():
    with pytest.raises(ValueError, match="not_a_real_field"):
        QuantizationConfig.coerce({"dtype": "int8", "not_a_real_field": 1})


def test_lenient_unknown_keys_keep_known_fields():
    config = QuantizationConfig.coerce({"dtype": "int8", "not_a_real_field": 1}, strict=False)
    assert isinstance(config, QuantizationConfig)
    assert config.dtype is QuantizationType.INT8
