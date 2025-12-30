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

"""Value head model wrapper for PPO training.

This module provides the CausalLMWithValueHead class that wraps a causal language
model with a scalar value head, which is essential for PPO-style RLHF training.
The value head predicts the expected return from each state, enabling advantage
estimation and variance reduction during policy optimization.
"""

from __future__ import annotations

import typing as tp

from flax import nnx as nn

from easydel.infra.base_module import EasyDeLBaseModule


def _infer_hidden_size(config) -> int:
    """Infer the hidden size from a model configuration.

    Searches for common hidden size attribute names across different model
    architectures (GPT, T5, BERT, etc.) to determine the embedding dimension.

    Args:
        config: Model configuration object with hidden size attributes.

    Returns:
        The hidden size (embedding dimension) of the model.

    Raises:
        ValueError: If no recognized hidden size attribute is found.
    """
    for name in ("hidden_size", "n_embd", "d_model", "model_dim", "dim", "embed_dim"):
        val = getattr(config, name, None)
        if isinstance(val, int) and val > 0:
            return val
    raise ValueError(
        "Unable to infer hidden size from config. Expected one of: "
        "`hidden_size`, `n_embd`, `d_model`, `model_dim`, `dim`, `embed_dim`."
    )


class CausalLMWithValueHead(EasyDeLBaseModule):
    """A lightweight wrapper that adds a scalar value head to a causal LM.

    This wrapper is essential for PPO-style RLHF training where a value function
    is needed to estimate expected returns and compute advantages. The wrapper
    delegates forward/generation calls to the underlying model and provides
    trainable `value_head` parameters for PPO optimization.

    The value head is a single linear layer that projects the hidden states
    to scalar values, initialized with zeros for stable training startup.

    Attributes:
        config_class: Configuration class from the base model.
        base_model_prefix: Prefix used for model parameter naming.
        model: The underlying causal language model.
        value_head: Linear layer projecting hidden states to scalar values.

    Example:
        >>> base_model = AutoModelForCausalLM.from_pretrained("gpt2")
        >>> model_with_value = CausalLMWithValueHead(base_model)
        >>> outputs = model_with_value(input_ids, attention_mask)
        >>> values = model_with_value.value_head(outputs.hidden_states[-1])
    """

    config_class: tp.Any = None
    base_model_prefix: str = "model"

    def __init__(
        self,
        base_model: EasyDeLBaseModule,
        *,
        rngs: nn.Rngs | None = None,
    ):
        """Initialize the CausalLMWithValueHead wrapper.

        Args:
            base_model: The underlying causal language model to wrap.
            rngs: Optional random number generators for initialization.
        """
        self.config_class = getattr(base_model, "config_class", None)
        super().__init__(
            config=base_model.config,
            dtype=base_model.dtype,
            param_dtype=base_model.param_dtype,
            precision=base_model.precision,
            rngs=rngs or base_model.rngs,
        )

        self.model = base_model
        hidden_size = _infer_hidden_size(base_model.config)
        self.value_head = nn.Linear(
            hidden_size,
            1,
            use_bias=False,
            dtype=base_model.dtype,
            param_dtype=base_model.param_dtype,
            precision=base_model.precision,
            kernel_init=nn.initializers.zeros,
            rngs=self.rngs,
        )

    def __call__(self, *args, **kwargs):
        """Forward pass delegated to the underlying model."""
        return self.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        """Generate sequences using the underlying model."""
        return self.model.generate(*args, **kwargs)

    def flops_per_token(self, *args, **kwargs):
        """Calculate FLOPs per token for the underlying model."""
        return self.model.flops_per_token(*args, **kwargs)

    # eSurge is an inference-only path that relies on instantiating
    # `self.__class__(config=...)` when swapping attention mechanisms.
    # This wrapper is not constructible from `config`, so we delegate all eSurge
    # APIs to the underlying base model.

    @property
    def esurge_graphdef(self):  # pragma: no cover
        """Get the eSurge graph definition from the underlying model."""
        return self.model.esurge_graphdef

    @property
    def esurge_compatible_model(self):  # pragma: no cover
        """Get the eSurge compatible model from the underlying model."""
        return self.model.esurge_compatible_model

    def get_esurge(self, *args, **kwargs):  # pragma: no cover
        """Get eSurge engine from the underlying model."""
        return self.model.get_esurge(*args, **kwargs)

    def esurge_generate(self, *args, **kwargs):  # pragma: no cover
        """Generate using eSurge from the underlying model."""
        return self.model.esurge_generate(*args, **kwargs)

    def pause_esurge(self, *args, **kwargs):  # pragma: no cover
        """Pause eSurge engine in the underlying model."""
        return self.model.pause_esurge(*args, **kwargs)

    def resume_esurge(self, *args, **kwargs):  # pragma: no cover
        """Resume eSurge engine in the underlying model."""
        return self.model.resume_esurge(*args, **kwargs)

    def list_esurge_engines(self, *args, **kwargs):  # pragma: no cover
        """List eSurge engines from the underlying model."""
        return self.model.list_esurge_engines(*args, **kwargs)
