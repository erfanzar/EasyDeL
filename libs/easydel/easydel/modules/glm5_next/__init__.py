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

"""GLM-5-Next (GLM-5.3-Flash) model family — hybrid KDA + DSA/MLA MoE with mHC."""

from .glm5_next_configuration import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from .modeling_glm5_next import (
    Glm5NextCompositeCausalLM,
    Glm5NextDecoderLayer,
    Glm5NextForCausalLM,
    Glm5NextHyperConnection,
    Glm5NextIndexer,
    Glm5NextLinearAttention,
    Glm5NextTextModel,
)


def _register_with_transformers() -> None:
    """Teach ``transformers.AutoConfig`` about the glm5_next config classes.

    The published `zai-org/GLM-5.3-Flash` checkpoint declares
    ``model_type: "glm5_next"`` (composite) / ``"glm5_next_text"`` /
    ``"glm5_next_vision"``. Hosts running transformers builds that predate the
    upstream port (e.g. 5.13) cannot resolve those identifiers, which the
    sequential checkpoint converter needs at ``AutoConfig.from_pretrained``.
    Registration is idempotent-safe: on transformers builds that already ship
    the family (or on double import) the call is a no-op failure that is
    swallowed.
    """
    try:
        from transformers import AutoConfig

        AutoConfig.register("glm5_next_text", Glm5NextTextConfig)
        AutoConfig.register("glm5_next_vision", Glm5NextVisionConfig)
        AutoConfig.register("glm5_next", Glm5NextConfig)
    except Exception:
        pass


_register_with_transformers()

__all__ = (
    "Glm5NextCompositeCausalLM",
    "Glm5NextConfig",
    "Glm5NextDecoderLayer",
    "Glm5NextForCausalLM",
    "Glm5NextHyperConnection",
    "Glm5NextIndexer",
    "Glm5NextLinearAttention",
    "Glm5NextTextConfig",
    "Glm5NextTextModel",
    "Glm5NextVisionConfig",
)
