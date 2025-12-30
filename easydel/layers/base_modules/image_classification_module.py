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

"""Generic base class for Image Classification tasks.

This module provides BaseImageClassificationModule for vision models
like ViT, CLIP, etc.
"""

from collections.abc import Callable

import jax
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array, Float

from easydel.infra.modeling_outputs import ImageClassifierOutput
from easydel.infra.utils import auto_remat
from easydel.layers.linear import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseImageClassificationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Image Classification.

    This class provides image classification with support for vision models.

    Example:
        ```python
        class CLIPForImageClassification(
            BaseImageClassificationModule[CLIPVisionModel, CLIPVisionConfig]
        ):
            _task_type = TaskType.IMAGE_CLASSIFICATION
            _model_type = "clip"
            _config_class = CLIPVisionConfig
        ```

    Type Parameters:
        ModelT: The base vision model type
        ConfigT: The configuration type

    Attributes:
        classifier: The image classification head
    """

    def __init__(
        self,
        config: ConfigT,
        base_model: ModelT | None = None,
        base_model_class: type[ModelT] | None = None,
        base_model_name: str = "vision_model",
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        pooling_strategy: str = "first",  # For ViT, use first token ([CLS])
        classifier_bias: bool = True,
        classifier_kernel_init: Callable | None = None,
    ):
        """Initialize the Image Classification module.

        Args:
            config: Model configuration (must have num_labels attribute)
            base_model: Pre-instantiated base model (optional)
            base_model_class: Base model class to instantiate (optional)
            base_model_name: Attribute name for the base model
            dtype: Data type for computations
            param_dtype: Data type for parameters
            precision: Precision setting for JAX operations
            rngs: Random number generators
            pooling_strategy: How to pool vision features ("first", "mean", "max")
            classifier_bias: Whether to use bias in classifier
            classifier_kernel_init: Custom kernel initializer
        """
        assert hasattr(config, "num_labels"), "config must have num_labels attribute"

        vision_config = getattr(config, "vision_config", config)
        hidden_size = getattr(vision_config, "hidden_size", None) or getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise AttributeError(
                "Unable to infer vision hidden size from config (expected `hidden_size` or `vision_config.hidden_size`)."
            )
        num_labels = config.num_labels
        super().__init__(
            config=config,
            base_model=base_model,
            base_model_class=base_model_class,
            base_model_name=base_model_name,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            pooling_strategy=pooling_strategy,
            head_bias=classifier_bias,
            head_kernel_init=classifier_kernel_init,
        )

        # Create classifier head
        classifier_block = ColumnParallelLinear
        if self._gradient_checkpointing_feature.should_checkpoint():
            classifier_block = auto_remat(
                classifier_block,
                **self._gradient_checkpointing_feature.get_config(),
            )

        self.classifier = None
        if num_labels > 0:
            self.classifier = classifier_block(
                hidden_size,
                num_labels,
                dtype=dtype,
                param_dtype=param_dtype,
                use_bias=self._head_bias,
                kernel_init=self._head_kernel_init,
                precision=precision,
                rngs=rngs,
            )

    def __call__(
        self,
        pixel_values: Float[Array, "batch channels height width"],
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> ImageClassifierOutput:
        """Forward pass for image classification.

        Args:
            pixel_values: Input image pixel values
            output_attentions: Whether to output attention weights
            output_hidden_states: Whether to output hidden states

        Returns:
            ImageClassifierOutput with classification logits
        """
        outputs = self.base_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        # Pool the visual features
        # For ViT-like models, typically use the [CLS] token (first position)
        hidden_states = outputs.last_hidden_state
        pooled_output = self.pool_sequence(hidden_states, input_ids=None)

        # Apply classifier (or return pooled features if no classifier head)
        logits = pooled_output if self.classifier is None else self.classifier(pooled_output)

        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_task_head(self):
        """Returns the classification head."""
        return self.classifier

    def get_lm_head(self):
        """Raises NotImplementedError."""
        raise NotImplementedError("Image classification models don't have an lm_head.")

    def get_encoder(self):
        """For image classification, the base model is the encoder."""
        return self.base_model

    def get_decoder(self):
        """Raises NotImplementedError."""
        raise NotImplementedError("Image classification models don't have a decoder.")
