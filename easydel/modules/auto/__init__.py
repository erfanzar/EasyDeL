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

"""Auto model classes for dynamic model loading and configuration in EasyDeL.

This module provides automatic model selection and instantiation utilities that enable
seamless model loading without explicitly specifying the model class. Similar to
HuggingFace's AutoModel classes, these utilities automatically determine the correct
model architecture based on configuration or model type.

Key Components:
    - AutoEasyDeLConfig: Auto-detect and load model configurations
    - AutoEasyDeLModel: Auto-instantiate base models
    - Task-Specific Auto Classes: Auto models for specific tasks (CausalLM, Seq2Seq, etc.)
    - AutoState: Load model weights from checkpoints
    - AutoShardAndGatherFunctions: Utilities for model parallelism

Usage Examples:
    ```python
    # Load configuration automatically
    from easydel.modules.auto import AutoEasyDeLConfig
    config = AutoEasyDeLConfig.from_pretrained("meta-llama/Llama-3.1-8B")

    # Load model for causal LM
    from easydel.modules.auto import AutoEasyDeLModelForCausalLM
    from flax import nnx as nn
    import jax

    model = AutoEasyDeLModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Load vision-language model
    from easydel.modules.auto import AutoEasyDeLModelForImageTextToText
    vlm_model = AutoEasyDeLModelForImageTextToText.from_pretrained(
        "CohereForAI/aya-vision-8b",
        dtype=jax.numpy.bfloat16,
        param_dtype=jax.numpy.bfloat16,
        rngs=nn.Rngs(0),
    )

    # Query available models by type
    from easydel.modules.auto import get_modules_by_type
    from easydel.infra.factory import TaskType

    config_class, model_class = get_modules_by_type(
        model_type="llama",
        task_type=TaskType.CAUSAL_LM,
    )
    ```

Supported Tasks:
    - CAUSAL_LM: Autoregressive language modeling (GPT, LLaMA, etc.)
    - SEQUENCE_TO_SEQUENCE: Encoder-decoder models (T5, BART, etc.)
    - SPEECH_SEQUENCE_TO_SEQUENCE: Speech-to-text models (Whisper, etc.)
    - IMAGE_TEXT_TO_TEXT: Vision-language models (LLaVA, Aya Vision, etc.)
    - DIFFUSION_LM: Diffusion-based language models (GIDD, etc.)
    - ZERO_SHOT_IMAGE_CLASSIFICATION: CLIP-style models
    - SEQUENCE_CLASSIFICATION: Text classification models

Auto Classes:
    - AutoEasyDeLConfig: Automatic configuration loading
    - AutoEasyDeLModel: Base model auto-loading
    - AutoEasyDeLModelForCausalLM: Causal language modeling
    - AutoEasyDeLModelForSeq2SeqLM: Sequence-to-sequence
    - AutoEasyDeLModelForSpeechSeq2Seq: Speech-to-text
    - AutoEasyDeLModelForImageTextToText: Vision-language
    - AutoEasyDeLModelForDiffusionLM: Diffusion language modeling
    - AutoEasyDeLModelForSequenceClassification: Text classification
    - AutoEasyDeLVisionModel: Vision encoders
    - AutoState: Load model weights/checkpoints
"""

from .auto_configuration import AutoEasyDeLConfig, AutoShardAndGatherFunctions, get_modules_by_type
from .auto_modeling import (
    AutoEasyDeLAnyToAnyModel,
    AutoEasyDeLModel,
    AutoEasyDeLModelForCausalLM,
    AutoEasyDeLModelForDiffusionLM,
    AutoEasyDeLModelForImageTextToText,
    AutoEasyDeLModelForSeq2SeqLM,
    AutoEasyDeLModelForSequenceClassification,
    AutoEasyDeLModelForSpeechSeq2Seq,
    AutoEasyDeLModelForZeroShotImageClassification,
    AutoEasyDeLVisionModel,
    AutoState,
    AutoStateAnyToAnyModel,
    AutoStateForCausalLM,
    AutoStateForDiffusionLM,
    AutoStateForImageSequenceClassification,
    AutoStateForImageTextToText,
    AutoStateForSeq2SeqLM,
    AutoStateForSpeechSeq2Seq,
    AutoStateForZeroShotImageClassification,
    AutoStateVisionModel,
)

__all__ = (
    "AutoEasyDeLAnyToAnyModel",
    "AutoEasyDeLConfig",
    "AutoEasyDeLModel",
    "AutoEasyDeLModelForCausalLM",
    "AutoEasyDeLModelForDiffusionLM",
    "AutoEasyDeLModelForImageTextToText",
    "AutoEasyDeLModelForSeq2SeqLM",
    "AutoEasyDeLModelForSequenceClassification",
    "AutoEasyDeLModelForSpeechSeq2Seq",
    "AutoEasyDeLModelForZeroShotImageClassification",
    "AutoEasyDeLVisionModel",
    "AutoShardAndGatherFunctions",
    "AutoState",
    "AutoStateAnyToAnyModel",
    "AutoStateForCausalLM",
    "AutoStateForDiffusionLM",
    "AutoStateForImageSequenceClassification",
    "AutoStateForImageTextToText",
    "AutoStateForSeq2SeqLM",
    "AutoStateForSpeechSeq2Seq",
    "AutoStateForZeroShotImageClassification",
    "AutoStateVisionModel",
    "get_modules_by_type",
)
