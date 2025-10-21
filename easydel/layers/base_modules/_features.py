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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied,
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modular features for task-specific modules.

This module provides reusable feature implementations that can be mixed and matched
in task-specific model classes. Features include logit capping, embedding tying,
router auxiliary loss, and more.
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx as nn
from jaxtyping import Array, Float


class LogitCapFeature:
    """Applies logit capping to prevent extreme values.

    Logit capping is used to stabilize training by preventing logits from becoming
    too large or too small. This is particularly useful for models that use
    temperature scaling or models prone to numerical instability.

    Attributes:
        cap_value: Maximum absolute value for logits
    """

    def __init__(self, cap_value: float):
        """Initialize logit capping feature.

        Args:
            cap_value: Maximum absolute value for logits. Logits will be clipped
                to the range [-cap_value, cap_value].
        """
        if cap_value <= 0:
            raise ValueError(f"cap_value must be positive, got {cap_value}")
        self.cap_value = cap_value

    def apply(self, logits: Float[Array, "batch seq_len vocab"]) -> Float[Array, "batch seq_len vocab"]:
        """Apply logit capping to the given logits.

        Args:
            logits: Input logits tensor

        Returns:
            Clipped logits tensor
        """
        return jnp.clip(logits, -self.cap_value, self.cap_value)

    def __repr__(self) -> str:
        return f"LogitCapFeature(cap_value={self.cap_value})"


class TieEmbeddingsFeature:
    """Ties input embeddings with output head weights.

    Weight tying is a technique where the input embedding matrix is shared with
    the output projection matrix (LM head). This reduces the number of parameters
    and can improve performance, especially for smaller models.

    Note:
        This is typically configured during model initialization and doesn't
        require runtime application.
    """

    def __init__(self, tie: bool = True):
        """Initialize embedding tying feature.

        Args:
            tie: Whether to tie embeddings
        """
        self.tie = tie

    def setup(self, embedding_module: nn.Module, lm_head_module: nn.Module) -> None:
        """Setup weight tying between embedding and LM head.

        Args:
            embedding_module: The embedding layer
            lm_head_module: The LM head linear layer
        """
        if not self.tie:
            return

        # In Flax NNX, weight tying is handled differently
        # This is a placeholder for the actual implementation
        # which would depend on how the embedding and lm_head are structured
        # Typically: lm_head.kernel = embedding.embedding
        pass

    def __repr__(self) -> str:
        return f"TieEmbeddingsFeature(tie={self.tie})"


class RouterAuxLossFeature:
    """Computes auxiliary loss for MoE router load balancing.

    Mixture-of-Experts (MoE) models use routers to distribute inputs across
    different expert networks. To prevent all inputs from being routed to a
    small number of experts, an auxiliary loss encourages balanced load distribution.

    Attributes:
        coef: Coefficient for the auxiliary loss
    """

    def __init__(self, coef: float):
        """Initialize router auxiliary loss feature.

        Args:
            coef: Coefficient to multiply the auxiliary loss by. Common values
                are in the range [0.001, 0.1].
        """
        if coef < 0:
            raise ValueError(f"coef must be non-negative, got {coef}")
        self.coef = coef

    def compute_loss(self, router_losses: list[Array] | tuple[Array, ...] | None) -> Array | None:
        """Compute the weighted auxiliary loss from router logits.

        Args:
            router_losses: List/tuple of router loss values from each MoE layer,
                or None if the model doesn't have routers

        Returns:
            Weighted sum of router losses, or None if no router losses
        """
        if router_losses is None or len(router_losses) == 0:
            return None

        total_loss = sum(router_losses)
        return total_loss * self.coef

    def __repr__(self) -> str:
        return f"RouterAuxLossFeature(coef={self.coef})"


class GradientCheckpointingFeature:
    """Configures gradient checkpointing for model components.

    Gradient checkpointing trades compute for memory by recomputing intermediate
    activations during the backward pass instead of storing them. This is
    particularly useful for large models.

    Attributes:
        policy: Checkpointing policy (e.g., "nothing_saveable", "checkpoint_dots")
        save_names: Names of operations to save during checkpointing
        exclude_names: Names of operations to exclude from checkpointing
    """

    def __init__(
        self,
        policy: str | None = None,
        save_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ):
        """Initialize gradient checkpointing feature.

        Args:
            policy: Checkpointing policy from config
            save_names: List of operation names to always save
            exclude_names: List of operation names to exclude from checkpointing
        """
        self.policy = policy
        self.save_names = save_names or []
        self.exclude_names = exclude_names or []

    def should_checkpoint(self) -> bool:
        """Check if gradient checkpointing should be applied.

        Returns:
            True if checkpointing is enabled, False otherwise
        """
        return self.policy is not None and self.policy != "nothing_saveable"

    def get_config(self) -> dict[str, Any]:
        """Get checkpointing configuration.

        Returns:
            Dictionary with policy, save_names, and exclude_names
        """
        return {
            "policy": self.policy,
            "save_names": self.save_names,
            "exclude_names": self.exclude_names,
        }

    def __repr__(self) -> str:
        return f"GradientCheckpointingFeature(policy={self.policy})"


class SequenceLengthPoolingFeature:
    """Pools sequence representations for classification tasks.

    For sequence classification, we need to reduce the sequence of hidden states
    to a single vector. This feature provides different pooling strategies.

    Attributes:
        strategy: Pooling strategy ("last", "first", "mean", "max")
        pad_token_id: Token ID used for padding (for "last" strategy)
    """

    def __init__(self, strategy: str = "last", pad_token_id: int | None = None):
        """Initialize sequence pooling feature.

        Args:
            strategy: Pooling strategy - one of:
                - "last": Use the last non-padding token (requires pad_token_id)
                - "first": Use the first token ([CLS] token)
                - "mean": Mean pooling over sequence
                - "max": Max pooling over sequence
            pad_token_id: Padding token ID (required for "last" strategy)
        """
        valid_strategies = {"last", "first", "mean", "max"}
        if strategy not in valid_strategies:
            raise ValueError(f"strategy must be one of {valid_strategies}, got {strategy}")

        if strategy == "last" and pad_token_id is None:
            raise ValueError("pad_token_id is required for 'last' pooling strategy")

        self.strategy = strategy
        self.pad_token_id = pad_token_id

    def pool(
        self,
        hidden_states: Float[Array, "batch seq_len hidden"],
        input_ids: Array | None = None,
    ) -> Float[Array, "batch hidden"]:
        """Pool hidden states to get sequence representation.

        Args:
            hidden_states: Sequence of hidden states
            input_ids: Input token IDs (required for "last" strategy)

        Returns:
            Pooled representation of shape (batch_size, hidden_dim)
        """
        batch_size = hidden_states.shape[0]

        if self.strategy == "first":
            # Use first token (e.g., [CLS])
            return hidden_states[:, 0]

        elif self.strategy == "last":
            # Use last non-padding token
            if input_ids is None:
                raise ValueError("input_ids required for 'last' pooling strategy")

            if self.pad_token_id is None:
                # If no padding token, use actual last token
                sequence_lengths = jnp.full(batch_size, hidden_states.shape[1] - 1)
            else:
                # Find last non-padding position
                sequence_lengths = jnp.argmax(jnp.equal(input_ids, self.pad_token_id).astype("i4"), -1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]

            return hidden_states[jnp.arange(batch_size), sequence_lengths]

        elif self.strategy == "mean":
            # Mean pooling over sequence
            return jnp.mean(hidden_states, axis=1)

        elif self.strategy == "max":
            # Max pooling over sequence
            return jnp.max(hidden_states, axis=1)

        else:
            raise ValueError(f"Unknown pooling strategy: {self.strategy}")

    def __repr__(self) -> str:
        return f"SequenceLengthPoolingFeature(strategy={self.strategy}, pad_token_id={self.pad_token_id})"
