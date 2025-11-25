# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Generic base modules for task-specific model wrappers.

This package provides a modular, type-safe system for creating task-specific
model wrappers (ForCausalLM, ForSequenceClassification, etc.) with minimal
boilerplate and maximum flexibility.

Features:
- Generic typing with Protocol-based interfaces
- Automatic model registration
- Modular features (logit cap, tie embeddings, router aux loss, etc.)
- Factory functions for dynamic class generation

Example Usage:

Explicit Subclassing (Recommended):
    ```python
    from easydel.layers.base_modules import BaseCausalLMModule
    from easydel.modules.arctic import ArcticModel, ArcticConfig

    class ArcticForCausalLM(BaseCausalLMModule[ArcticModel, ArcticConfig]):
        _task_type = TaskType.CAUSAL_LM
        _model_type = "arctic"
        _config_class = ArcticConfig

        def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
            super().__init__(
                config=config,
                base_model_class=ArcticModel,
                base_model_name="model",
                dtype=dtype,
                rngs=rngs,
                router_aux_loss_coef=0.001,
            )
    ```

Factory Function Approach:
    ```python
    from easydel.layers.base_modules import create_causal_lm_class
    from easydel.modules.arctic import ArcticModel, ArcticConfig

    ArcticForCausalLM = create_causal_lm_class(
        "Arctic",
        ArcticModel,
        ArcticConfig,
        model_type="arctic",
        router_aux_loss_coef=0.001,
    )
    ```
"""

# Protocol definitions
# Factory functions
from ._auto_mapper import (
    AUTO_MODEL_FACTORY_REGISTRY,
    create_causal_lm_class,
    create_conditional_generation_class,
    create_image_classification_class,
    create_question_answering_class,
    create_sequence_classification_class,
    create_task_model_class,
    create_token_classification_class,
)

# Base task module
from ._base_task_module import BaseTaskModule

# Feature implementations
from ._features import (
    GradientCheckpointingFeature,
    LogitCapFeature,
    RouterAuxLossFeature,
    SequenceLengthPoolingFeature,
    TieEmbeddingsFeature,
)
from ._protocols import (
    BaseModelProtocol,
    EncoderDecoderProtocol,
    VisionLanguageProtocol,
    VisionModelProtocol,
)

# VLM feature implementations
from ._vlm_features import (
    MultiDimensionalRoPEFeature,
    MultiModalMergeFeature,
    VideoProcessingFeature,
    VisionEncoderFeature,
)

# Task-specific base modules
from .causal_lm_module import BaseCausalLMModule
from .conditional_generation_module import BaseConditionalGenerationModule
from .image_classification_module import BaseImageClassificationModule
from .question_answering_module import BaseQuestionAnsweringModule
from .sequence_classification_module import BaseSequenceClassificationModule
from .token_classification_module import BaseTokenClassificationModule
from .vision_language_module import BaseVisionLanguageModule

__all__ = [
    "AUTO_MODEL_FACTORY_REGISTRY",
    "BaseCausalLMModule",
    "BaseConditionalGenerationModule",
    "BaseImageClassificationModule",
    "BaseModelProtocol",
    "BaseQuestionAnsweringModule",
    "BaseSequenceClassificationModule",
    "BaseTaskModule",
    "BaseTokenClassificationModule",
    "BaseVisionLanguageModule",
    "EncoderDecoderProtocol",
    "GradientCheckpointingFeature",
    "LogitCapFeature",
    "MultiDimensionalRoPEFeature",
    "MultiModalMergeFeature",
    "RouterAuxLossFeature",
    "SequenceLengthPoolingFeature",
    "TieEmbeddingsFeature",
    "VideoProcessingFeature",
    "VisionEncoderFeature",
    "VisionLanguageProtocol",
    "VisionModelProtocol",
    "create_causal_lm_class",
    "create_conditional_generation_class",
    "create_image_classification_class",
    "create_question_answering_class",
    "create_sequence_classification_class",
    "create_task_model_class",
    "create_token_classification_class",
]
