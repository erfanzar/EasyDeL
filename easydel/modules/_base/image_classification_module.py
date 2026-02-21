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

This module provides BaseImageClassificationModule for vision models like
ViT, CLIP, DINOv2, etc. that perform image-level classification.

Image classification takes an image as input and produces a probability
distribution over a fixed set of classes. The model processes the image
through a vision transformer and pools the output to a single vector,
which is then projected to class logits.

Key Features:
    - Support for various vision architectures (ViT, CLIP, etc.)
    - Configurable pooling strategy (CLS token, mean pooling, etc.)
    - Optional gradient checkpointing for memory efficiency
    - Automatic registration with EasyDeL factory system

Example:
    Creating an image classification model:

    ```python
    class ViTForImageClassification(
        BaseImageClassificationModule[ViTModel, ViTConfig]
    ):
        _task_type = TaskType.IMAGE_CLASSIFICATION
        _model_type = "vit"
        _config_class = ViTConfig

        def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
            # Ensure config has num_labels
            config.num_labels = 1000  # ImageNet classes
            super().__init__(
                config=config,
                base_model_class=ViTModel,
                base_model_name="vision_model",
                dtype=dtype,
                rngs=rngs,
                pooling_strategy="first",  # Use CLS token
            )
    ```

See Also:
    - BaseTaskModule: Parent class with common functionality
    - easydel.infra.modeling_outputs.ImageClassifierOutput: Output dataclass
"""

from collections.abc import Callable

import jax
from flax import nnx as nn
from jax import numpy as jnp
from jaxtyping import Array, Float

from easydel.infra.modeling_outputs import ImageClassifierOutput
from easydel.infra.utils import auto_remat
from easydel.layers.components import ColumnParallelLinear

from ._base_task_module import BaseTaskModule, ConfigT, ModelT


class BaseImageClassificationModule(BaseTaskModule[ModelT, ConfigT]):
    """Generic base class for Image Classification.

    This class provides image classification with support for various vision
    model architectures. It handles the common pattern of:
    1. Processing images through a vision encoder
    2. Pooling the output to a single vector
    3. Projecting to class logits

    The class supports different pooling strategies for extracting a single
    vector from the sequence of patch embeddings:
        - "first": Use the first token (CLS token for ViT)
        - "mean": Average all tokens
        - "max": Max pooling over tokens

    Example:
        Explicit subclassing:

        ```python
        class CLIPForImageClassification(
            BaseImageClassificationModule[CLIPVisionModel, CLIPVisionConfig]
        ):
            _task_type = TaskType.IMAGE_CLASSIFICATION
            _model_type = "clip"
            _config_class = CLIPVisionConfig

            def __init__(self, config, dtype=jnp.bfloat16, *, rngs):
                config.num_labels = 1000  # Required attribute
                super().__init__(
                    config=config,
                    base_model_class=CLIPVisionModel,
                    base_model_name="vision_model",
                    dtype=dtype,
                    rngs=rngs,
                )
        ```

    Type Parameters:
        ModelT: The base vision model type. Must accept pixel_values and return
            an object with last_hidden_state attribute.
        ConfigT: The configuration type. Must have num_labels and hidden_size
            (or vision_config.hidden_size) attributes.

    Attributes:
        classifier (ColumnParallelLinear | None): The classification head that
            projects pooled features to num_labels logits. None if num_labels <= 0.
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

        Sets up the vision encoder and classification head for image-level
        classification tasks.

        Args:
            config: Model configuration. Must have the following attributes:
                - num_labels (int): Number of classification classes
                - hidden_size (int): Hidden dimension (or vision_config.hidden_size)
                Can also have optional attributes like gradient_checkpointing.
            base_model: Pre-instantiated base vision model. If provided,
                base_model_class is ignored.
            base_model_class: Vision model class to instantiate. Required if
                base_model is not provided.
            base_model_name: Attribute name for storing the vision model.
                Defaults to "vision_model".
            dtype: Data type for computations. Defaults to jnp.bfloat16.
            param_dtype: Data type for parameters. Defaults to jnp.bfloat16.
            precision: JAX precision setting for matrix operations.
            rngs: Flax NNX random number generators.
            pooling_strategy: How to pool vision features to a single vector:
                - "first": Use first token (CLS token). Default for ViT.
                - "mean": Average all tokens.
                - "max": Max pooling over tokens.
                Defaults to "first".
            classifier_bias: Whether to use bias in the classification head.
                Defaults to True.
            classifier_kernel_init: Custom kernel initializer for the classifier.
                If None, uses normal initialization with stddev from config.

        Raises:
            AssertionError: If config does not have num_labels attribute.
            AttributeError: If unable to infer hidden_size from config.

        Example:
            ```python
            config = ViTConfig(
                num_labels=1000,
                hidden_size=768,
                # ... other ViT config
            )

            model = BaseImageClassificationModule(
                config=config,
                base_model_class=ViTModel,
                dtype=jnp.float32,
                rngs=nn.Rngs(0),
                pooling_strategy="first",  # Use CLS token
            )
            ```
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

        Processes input images through the vision encoder, pools the output,
        and produces classification logits.

        Args:
            pixel_values: Input images of shape (batch_size, channels, height, width).
                Typically normalized to model-specific range (e.g., [-1, 1] or
                ImageNet normalization).
            output_attentions: Whether to return attention weights from all
                vision transformer layers. Defaults to config setting.
            output_hidden_states: Whether to return hidden states from all
                layers. Defaults to config setting.

        Returns:
            ImageClassifierOutput containing:
                - logits: Classification logits of shape (batch_size, num_labels)
                  if classifier exists, otherwise pooled features of shape
                  (batch_size, hidden_size).
                - hidden_states: Tuple of hidden states from each layer if
                  output_hidden_states=True, else None.
                - attentions: Tuple of attention weights from each layer if
                  output_attentions=True, else None.

        Example:
            ```python
            # Single image classification
            pixel_values = preprocess(image)  # Shape: (1, 3, 224, 224)
            outputs = model(pixel_values)
            predicted_class = jnp.argmax(outputs.logits, axis=-1)

            # Batch classification with hidden states
            outputs = model(
                pixel_values=batch_images,
                output_hidden_states=True,
            )
            # outputs.hidden_states is tuple of (batch, patches, hidden)
            ```
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
        """Return the classification head.

        Returns:
            The classifier module (ColumnParallelLinear) that projects
            pooled features to class logits. May be None if num_labels <= 0.
        """
        return self.classifier

    def get_lm_head(self):
        """Raise NotImplementedError for image classification models.

        Image classification models use a classifier head, not a language
        modeling head.

        Raises:
            NotImplementedError: Always raised for image classification models.
        """
        raise NotImplementedError("Image classification models don't have an lm_head.")

    def get_encoder(self):
        """Return the vision encoder.

        For image classification, the entire base model serves as the encoder
        that transforms images into feature representations.

        Returns:
            The base vision model (e.g., ViTModel, CLIPVisionModel).
        """
        return self.base_model

    def get_decoder(self):
        """Raise NotImplementedError for image classification models.

        Image classification models don't have a decoder component; they
        directly classify from encoder outputs.

        Raises:
            NotImplementedError: Always raised for image classification models.
        """
        raise NotImplementedError("Image classification models don't have a decoder.")
