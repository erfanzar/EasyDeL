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

"""DeepSeek-V4.1 model family (text model and vision-language model; serving of images and MTP pending)."""

from ._dspark_loader import load_deepseek_v41_dspark
from .deepseek_v41_configuration import DeepseekV41Config
from .modeling_deepseek_v41 import (
    DeepseekV41ForCausalLM,
    DeepseekV41ForConditionalGeneration,
    DeepseekV41Model,
    DeepseekV41VLModel,
)
from .modeling_deepseek_v41_dspark import DeepseekV41DSpark, DeepseekV41DSparkState


def _register_with_transformers() -> None:
    """Teach ``transformers.AutoConfig`` about ``model_type="deepseek_v41"``.

    The released `deepseek-ai/DeepSeek-V4.1-Flash` checkpoint declares ``model_type: "deepseek_v41"``, which
    transformers 5.18 does not ship; the sequential checkpoint converter resolves the config through
    ``AutoConfig.from_pretrained`` and fails without this. Same pattern as ``modules/glm5_next``: on transformers
    builds that already ship the family (or on double import) the call fails and is swallowed.
    """
    try:
        from transformers import AutoConfig

        AutoConfig.register("deepseek_v41", DeepseekV41Config)
    except Exception:
        pass


_register_with_transformers()

__all__ = [
    "DeepseekV41Config",
    "DeepseekV41DSpark",
    "DeepseekV41DSparkState",
    "DeepseekV41ForCausalLM",
    "DeepseekV41ForConditionalGeneration",
    "DeepseekV41Model",
    "DeepseekV41VLModel",
    "load_deepseek_v41_dspark",
]
