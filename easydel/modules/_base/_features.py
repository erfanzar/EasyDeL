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

"""Modular features for task-specific modules.

This module provides reusable feature implementations that can be mixed and matched
in task-specific model classes. Features encapsulate common functionality to avoid
code duplication and enable consistent behavior across different model architectures.

Available Features:
    LogitCapFeature: Clips logits to prevent extreme values
    TieEmbeddingsFeature: Shares weights between input embeddings and output head
    RouterAuxLossFeature: Computes auxiliary loss for MoE router load balancing
    GradientCheckpointingFeature: Configures activation checkpointing for memory efficiency
    SequenceLengthPoolingFeature: Pools sequences for classification tasks

Design Philosophy:
    Each feature class follows the single responsibility principle, encapsulating
    one specific functionality. Features are instantiated in BaseTaskModule and
    used by task-specific subclasses as needed.

Example:
    Using features in a task module:

    ```python
    class MyForCausalLM(BaseTaskModule):
        def __init__(self, config, logit_cap=30.0, ...):
            # Features are created in BaseTaskModule
            self._logit_cap_feature = LogitCapFeature(logit_cap)

        def __call__(self, input_ids, ...):
            logits = self.lm_head(hidden_states)
            # Apply logit capping
            logits = self._logit_cap_feature.apply(logits)
            return logits
    ```

See Also:
    - BaseTaskModule: Uses these features internally
    - _vlm_features: Vision-Language specific features
"""

from typing import Any

import jax.numpy as jnp
from flax import nnx as nn
from jaxtyping import Array, Float


class LogitCapFeature:
    """Apply logit capping to prevent extreme values.

    Logit capping is used to stabilize training by preventing logits from becoming
    too large or too small. This is particularly useful for:
        - Models that use temperature scaling
        - Models prone to numerical instability
        - Preventing overconfident predictions
        - Stabilizing gradients during training

    The capping operation clips logits to the range [-cap_value, cap_value],
    which bounds the maximum probability ratio between any two tokens.

    Attributes:
        cap_value (float): Maximum absolute value for logits. Must be positive.

    Example:
        ```python
        cap_feature = LogitCapFeature(cap_value=30.0)

        # Raw logits might have extreme values
        raw_logits = lm_head(hidden_states)  # Could be [-100, 200, ...]

        # Capped logits are bounded
        capped_logits = cap_feature.apply(raw_logits)  # Now in [-30, 30]
        ```

    Note:
        A cap_value of 30.0 means the probability ratio between the most and
        least likely tokens is bounded by exp(60) which is still astronomically
        large but prevents numerical overflow.
    """

    def __init__(self, cap_value: float):
        """Initialize logit capping feature.

        Args:
            cap_value: Maximum absolute value for logits. Logits will be clipped
                to the range [-cap_value, cap_value]. Must be a positive number.

        Raises:
            ValueError: If cap_value is not positive.

        Example:
            ```python
            # Cap logits at +/- 30
            feature = LogitCapFeature(cap_value=30.0)

            # This would raise ValueError
            feature = LogitCapFeature(cap_value=-1.0)
            ```
        """
        if cap_value <= 0:
            raise ValueError(f"cap_value must be positive, got {cap_value}")
        self.cap_value = cap_value

    def apply(self, logits: Float[Array, "batch seq_len vocab"]) -> Float[Array, "batch seq_len vocab"]:
        """Apply logit capping to the given logits.

        Clips all values in the logits tensor to be within [-cap_value, cap_value].
        This operation is differentiable with zero gradient for clipped values.

        Args:
            logits: Input logits tensor of shape (batch_size, sequence_length, vocab_size)
                or any other shape. Values outside the cap range will be clipped.

        Returns:
            Clipped logits tensor of the same shape as input, with all values
            in the range [-cap_value, cap_value].

        Example:
            ```python
            feature = LogitCapFeature(30.0)

            # Extreme logits get clipped
            logits = jnp.array([[-50.0, 0.0, 100.0]])
            capped = feature.apply(logits)
            # capped = [[-30.0, 0.0, 30.0]]
            ```
        """
        return jnp.clip(logits, -self.cap_value, self.cap_value)

    def __repr__(self) -> str:
        """Return string representation of the feature.

        Returns:
            String showing the feature class name and cap_value parameter.
        """
        return f"LogitCapFeature(cap_value={self.cap_value})"


class TieEmbeddingsFeature:
    """Tie input embeddings with output head weights.

    Weight tying is a technique where the input embedding matrix is shared with
    the output projection matrix (LM head). This reduces the number of parameters
    and can improve performance, especially for smaller models.

    Benefits of weight tying:
        - Reduces parameters by vocab_size * hidden_size
        - Ensures consistent token representations
        - Can improve generalization, especially for rare tokens
        - Commonly used in models like GPT-2, T5, ALBERT

    Attributes:
        tie (bool): Whether weight tying is enabled.

    Example:
        ```python
        tie_feature = TieEmbeddingsFeature(tie=True)

        # During initialization
        if tie_feature.tie:
            # lm_head will use embedding weights transposed
            lm_head.kernel = embedding.embedding.T
        ```

    Note:
        This feature is typically configured during model initialization and
        doesn't require runtime application. The actual weight sharing is
        handled by passing the embedding weights to the LM head forward method.
    """

    def __init__(self, tie: bool = True):
        """Initialize embedding tying feature.

        Args:
            tie: Whether to tie input embeddings with output head weights.
                When True, the LM head uses the transpose of the embedding
                matrix instead of separate parameters. Defaults to True.

        Example:
            ```python
            # Enable weight tying (default)
            feature = TieEmbeddingsFeature(tie=True)

            # Disable weight tying
            feature = TieEmbeddingsFeature(tie=False)
            ```
        """
        self.tie = tie

    def setup(self, embedding_module: nn.Module, lm_head_module: nn.Module) -> None:
        """Set up weight tying between embedding and LM head.

        This method configures the relationship between the input embedding
        layer and the output projection (LM head). In Flax NNX, this typically
        means the LM head's forward method will accept the embedding weights
        as a parameter rather than using its own kernel.

        Args:
            embedding_module: The input embedding layer, typically an nn.Embed
                with an `embedding` attribute containing the embedding matrix.
            lm_head_module: The LM head linear layer. When tying is enabled,
                this module should accept an optional weight parameter in
                its forward method.

        Note:
            In Flax NNX, actual weight tying is handled differently than in
            PyTorch. Instead of sharing parameters, the embedding weights are
            passed to the LM head during forward pass:

            ```python
            # In forward pass
            if tie_embeddings:
                logits = lm_head(hidden, w=embedding.embedding.T)
            else:
                logits = lm_head(hidden)
            ```
        """
        if not self.tie:
            return

        # In Flax NNX, weight tying is handled differently
        # This is a placeholder for the actual implementation
        # which would depend on how the embedding and lm_head are structured
        # Typically: lm_head.kernel = embedding.embedding
        pass

    def __repr__(self) -> str:
        """Return string representation of the feature.

        Returns:
            String showing the feature class name and tie parameter.
        """
        return f"TieEmbeddingsFeature(tie={self.tie})"


class RouterAuxLossFeature:
    """Compute auxiliary loss for MoE router load balancing.

    Mixture-of-Experts (MoE) models use routers to distribute inputs across
    different expert networks. Without regularization, the router may learn
    to always select the same few experts, leading to:
        - Underutilization of model capacity
        - Experts that never receive gradients
        - Effective reduction in model size

    The auxiliary loss encourages balanced load distribution by penalizing
    uneven expert utilization. Common formulations include:
        - Load balancing loss (encourages uniform expert selection)
        - Router z-loss (penalizes large router logits)

    Attributes:
        coef (float): Coefficient multiplied with the auxiliary loss.

    Example:
        ```python
        aux_feature = RouterAuxLossFeature(coef=0.01)

        # In forward pass
        outputs = model(input_ids)
        if outputs.all_router_losses:
            aux_loss = aux_feature.compute_loss(outputs.all_router_losses)
            total_loss = lm_loss + aux_loss
        ```

    Note:
        The coefficient typically ranges from 0.001 to 0.1. Higher values
        encourage more balanced routing but may hurt model quality.
    """

    def __init__(self, coef: float):
        """Initialize router auxiliary loss feature.

        Args:
            coef: Coefficient to multiply the auxiliary loss by. This controls
                the trade-off between the main task loss and load balancing.
                Common values are in the range [0.001, 0.1].
                - Lower values: Less regularization, potentially unbalanced routing
                - Higher values: More balanced routing, may hurt task performance

        Raises:
            ValueError: If coef is negative.

        Example:
            ```python
            # Moderate regularization
            feature = RouterAuxLossFeature(coef=0.01)

            # Strong regularization for balanced routing
            feature = RouterAuxLossFeature(coef=0.1)
            ```
        """
        if coef < 0:
            raise ValueError(f"coef must be non-negative, got {coef}")
        self.coef = coef

    def compute_loss(self, router_losses: list[Array] | tuple[Array, ...] | None) -> Array | None:
        """Compute the weighted auxiliary loss from router losses.

        Aggregates router losses from all MoE layers and applies the
        coefficient weighting.

        Args:
            router_losses: List or tuple of router loss values from each MoE layer.
                Each element is a scalar loss value from one layer's router.
                Can be None if the model doesn't have routers or router losses
                weren't computed.

        Returns:
            Weighted sum of router losses (sum(losses) * coef), or None if
            router_losses is None or empty.

        Example:
            ```python
            feature = RouterAuxLossFeature(coef=0.01)

            # Router losses from 4 MoE layers
            router_losses = [0.5, 0.3, 0.4, 0.6]  # Sum = 1.8
            aux_loss = feature.compute_loss(router_losses)
            # aux_loss = 1.8 * 0.01 = 0.018

            # No router losses
            aux_loss = feature.compute_loss(None)
            # aux_loss = None
            ```
        """
        if router_losses is None or len(router_losses) == 0:
            return None

        total_loss = sum(router_losses)
        return total_loss * self.coef

    def __repr__(self) -> str:
        """Return string representation of the feature.

        Returns:
            String showing the feature class name and coef parameter.
        """
        return f"RouterAuxLossFeature(coef={self.coef})"


class GradientCheckpointingFeature:
    """Configure gradient checkpointing for model components.

    Gradient checkpointing (also called activation checkpointing or rematerialization)
    trades compute for memory by recomputing intermediate activations during the
    backward pass instead of storing them. This is particularly useful for:
        - Training large models that don't fit in memory
        - Increasing batch size for better throughput
        - Training with longer sequences

    The trade-off is approximately:
        - Memory: O(sqrt(n)) instead of O(n) for n layers
        - Compute: ~30-40% increase due to recomputation

    Attributes:
        policy (str | None): Checkpointing policy controlling what to save/recompute.
        save_names (list[str]): Operation names to always save (not recompute).
        exclude_names (list[str]): Operation names to exclude from checkpointing.

    Example:
        ```python
        feature = GradientCheckpointingFeature(
            policy="checkpoint_dots",  # Recompute matmuls
            save_names=["attention"],   # But save attention outputs
        )

        if feature.should_checkpoint():
            layer = auto_remat(layer, **feature.get_config())
        ```

    Note:
        Common policies include:
            - "nothing_saveable": Save nothing (maximum memory savings)
            - "checkpoint_dots": Recompute matmuls (good balance)
            - "checkpoint_dots_with_no_batch_dims": More aggressive checkpointing
    """

    def __init__(
        self,
        policy: str | None = None,
        save_names: list[str] | None = None,
        exclude_names: list[str] | None = None,
    ):
        """Initialize gradient checkpointing feature.

        Args:
            policy: Checkpointing policy string. Determines which operations
                are checkpointed. Common values:
                    - None or "nothing_saveable": No checkpointing
                    - "checkpoint_dots": Recompute matrix multiplications
                    - "checkpoint_dots_with_no_batch_dims": More aggressive
                Defaults to None (no checkpointing).
            save_names: List of operation names to always save during forward
                pass. These operations won't be recomputed in the backward pass.
                Useful for expensive operations that shouldn't be repeated.
            exclude_names: List of operation names to exclude from checkpointing
                entirely. These operations will always be saved.

        Example:
            ```python
            # Enable checkpointing with custom configuration
            feature = GradientCheckpointingFeature(
                policy="checkpoint_dots",
                save_names=["rope", "rmsnorm"],
                exclude_names=["final_layer"],
            )

            # Disable checkpointing
            feature = GradientCheckpointingFeature(policy=None)
            ```
        """
        self.policy = policy
        self.save_names = save_names or []
        self.exclude_names = exclude_names or []

    def should_checkpoint(self) -> bool:
        """Check if gradient checkpointing should be applied.

        Determines whether checkpointing is enabled based on the policy setting.
        A policy of None or "nothing_saveable" is considered as disabled.

        Returns:
            True if checkpointing should be applied (policy is set and not
            "nothing_saveable"), False otherwise.

        Example:
            ```python
            feature = GradientCheckpointingFeature(policy="checkpoint_dots")
            if feature.should_checkpoint():
                model = apply_checkpointing(model)
            ```
        """
        return self.policy is not None and self.policy != "nothing_saveable"

    def get_config(self) -> dict[str, Any]:
        """Get checkpointing configuration as a dictionary.

        Returns a dictionary suitable for passing to checkpointing utilities
        like auto_remat.

        Returns:
            Dictionary containing:
                - policy: The checkpointing policy string
                - save_names: List of operation names to save
                - exclude_names: List of operation names to exclude

        Example:
            ```python
            feature = GradientCheckpointingFeature(
                policy="checkpoint_dots",
                save_names=["attention"],
            )
            config = feature.get_config()
            # config = {
            #     "policy": "checkpoint_dots",
            #     "save_names": ["attention"],
            #     "exclude_names": [],
            # }

            checkpointed_layer = auto_remat(layer, **config)
            ```
        """
        return {
            "policy": self.policy,
            "save_names": self.save_names,
            "exclude_names": self.exclude_names,
        }

    def __repr__(self) -> str:
        """Return string representation of the feature.

        Returns:
            String showing the feature class name and policy parameter.
        """
        return f"GradientCheckpointingFeature(policy={self.policy})"


class SequenceLengthPoolingFeature:
    """Pool sequence representations for classification tasks.

    For sequence classification tasks, we need to reduce the sequence of hidden
    states to a single vector representation. This feature provides different
    pooling strategies to accomplish this:
        - "last": Use the last non-padding token (decoder-style)
        - "first": Use the first token, typically [CLS] (encoder-style)
        - "mean": Average all token representations
        - "max": Max pooling over the sequence dimension

    Attributes:
        strategy (str): Pooling strategy to use.
        pad_token_id (int | None): Token ID for padding (needed for "last" strategy).

    Example:
        ```python
        # For BERT-style models (use [CLS] token)
        pool = SequenceLengthPoolingFeature(strategy="first")

        # For GPT-style models (use last token)
        pool = SequenceLengthPoolingFeature(
            strategy="last",
            pad_token_id=tokenizer.pad_token_id
        )

        # Pool hidden states
        pooled = pool.pool(hidden_states, input_ids)
        logits = classifier(pooled)
        ```

    Note:
        The "last" strategy requires either pad_token_id to find the sequence
        end, or an attention_mask to determine valid token positions.
    """

    def __init__(self, strategy: str = "last", pad_token_id: int | None = None):
        """Initialize sequence pooling feature.

        Args:
            strategy: Pooling strategy to use. Must be one of:
                - "last": Use the last non-padding token. Best for decoder-only
                  models where the last token summarizes the sequence.
                - "first": Use the first token ([CLS] token). Best for
                  encoder models like BERT where [CLS] is trained for this.
                - "mean": Average all tokens in the sequence. Robust general
                  choice that considers all context.
                - "max": Max pooling over sequence. Captures the most
                  salient features.
                Defaults to "last".
            pad_token_id: Padding token ID. Required when using "last" strategy
                without attention_mask, as it's used to find the actual
                sequence end. Can be set to -1 for models without padding.

        Raises:
            ValueError: If strategy is not one of the valid options.
            ValueError: If strategy is "last" and pad_token_id is None.

        Example:
            ```python
            # BERT-style (first token)
            pool = SequenceLengthPoolingFeature(strategy="first")

            # GPT-style (last token)
            pool = SequenceLengthPoolingFeature(
                strategy="last",
                pad_token_id=50256  # GPT-2 pad token
            )

            # Mean pooling (no pad_token needed)
            pool = SequenceLengthPoolingFeature(strategy="mean")
            ```
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
        attention_mask: Array | None = None,
    ) -> Float[Array, "batch hidden"]:
        """Pool hidden states to get sequence-level representation.

        Reduces a sequence of token representations to a single vector
        per sequence using the configured pooling strategy.

        Args:
            hidden_states: Sequence of hidden states of shape
                (batch_size, sequence_length, hidden_dim).
            input_ids: Input token IDs of shape (batch_size, sequence_length).
                Required for "last" strategy when attention_mask is not provided,
                used to find non-padding positions.
            attention_mask: Optional mask of shape (batch_size, sequence_length)
                where 1 indicates valid tokens and 0 indicates padding.
                Preferred over input_ids for finding sequence lengths.

        Returns:
            Pooled representation of shape (batch_size, hidden_dim).
            Each sequence in the batch is represented by a single vector.

        Raises:
            ValueError: If using "last" strategy without input_ids or attention_mask.
            ValueError: If an unknown pooling strategy is configured.

        Example:
            ```python
            pool = SequenceLengthPoolingFeature(strategy="last", pad_token_id=0)

            # hidden_states: (2, 10, 768) - 2 sequences of 10 tokens
            # First sequence: 5 real tokens + 5 padding
            # Second sequence: 8 real tokens + 2 padding

            pooled = pool.pool(hidden_states, input_ids=input_ids)
            # pooled: (2, 768) - one vector per sequence
            # pooled[0] = hidden_states[0, 4]  # 5th token (last real)
            # pooled[1] = hidden_states[1, 7]  # 8th token (last real)
            ```
        """
        batch_size = hidden_states.shape[0]

        if self.strategy == "first":
            # Use first token (e.g., [CLS])
            return hidden_states[:, 0]

        elif self.strategy == "last":
            # Use last non-padding token
            if attention_mask is not None:
                lengths = jnp.sum(attention_mask.astype("i4"), axis=-1) - 1
                lengths = jnp.maximum(lengths, 0)
                return hidden_states[jnp.arange(batch_size), lengths]

            if input_ids is None:
                raise ValueError("input_ids required for 'last' pooling strategy")

            # Find last non-padding position using the configured pad_token_id.
            # If pad_token_id is a sentinel that doesn't appear in input_ids (e.g. -1),
            # argmax will return 0 and we fall back to the actual last token via modulo.
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
        """Return string representation of the feature.

        Returns:
            String showing the feature class name, strategy, and pad_token_id.
        """
        return f"SequenceLengthPoolingFeature(strategy={self.strategy}, pad_token_id={self.pad_token_id})"
