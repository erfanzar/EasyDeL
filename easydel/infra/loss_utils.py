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

"""Loss computation utilities for EasyDeL models.

This module provides a comprehensive suite of loss functions optimized for large-scale
language model training and inference. It includes memory-efficient implementations
using chunking strategies, custom VJP for gradient computation, and support for
various normalization and regularization techniques.

Classes:
    SpecialLossNormalizingFactor: Enum for dynamic loss normalization strategies.
    LossConfig: Configuration class for customizing loss computation behavior.
    LossMetrics: Container for loss metrics and auxiliary training information.

Loss Functions:
    ForCausalLMLoss: Causal language modeling with token shifting.
    ForSequenceClassificationLoss: Single/multi-label classification and regression.
    ForQuestionAnsweringLoss: Span-based question answering (SQuAD-style).
    ForTokenClassification: Token-level classification (NER, POS tagging).

Utility Functions:
    cross_entropy_blockwise_logits: Memory-efficient CE for large vocabularies.
    sparse_cross_entropy_chunked_vocab: Chunked vocabulary processing.
    sparse_cross_entropy_chunked_tokens: Chunked token processing.
    compute_weighted_cross_entropy: Standard weighted CE with z-loss.
    auxiliary_load_balancing_loss_func: MoE load balancing loss.

Key Features:
    - Memory-efficient chunking for large vocabulary/sequence lengths.
    - Flexible loss normalization (per-token, per-sequence, weighted).
    - Label smoothing and z-loss regularization.
    - Custom VJP for efficient gradient computation.
    - Support for packed sequences and attention masking.
    - Mixed precision computation support.
    - MoE auxiliary loss for expert load balancing.

Example:
    >>> from easydel.infra import LossConfig, ForCausalLMLoss
    >>>
    >>> # Configure loss with label smoothing and z-loss
    >>> config = LossConfig(
    ...     label_smoothing=0.1,
    ...     z_loss=1e-4,
    ...     loss_normalizing_factor="NUM_REAL_TARGET_TOKENS",
    ...     chunk_vocab_size=8192  # Enable vocabulary chunking
    ... )
    >>>
    >>> # Compute loss for language modeling
    >>> metrics = ForCausalLMLoss(
    ...     logits=model_output,  # [batch, seq_len, vocab_size]
    ...     labels=targets,        # [batch, seq_len]
    ...     attention_mask=mask,   # [batch, seq_len]
    ...     config=config
    ... )
    >>> print(f"Loss: {metrics.loss}, Accuracy: {metrics.accuracy}")
"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import inspect
import typing as tp
from dataclasses import fields
from functools import reduce
from operator import mul

import flax
import flax.struct
import jax
import jax.numpy as jnp
from eformer.escale import PartitionAxis
from eformer.escale.partition.constraints import with_sharding_constraint
from eformer.pytree import auto_pytree
from jax import lax
from jax.sharding import PartitionSpec
from jaxtyping import Array

from easydel.utils.compiling_utils import hash_fn


@dataclasses.dataclass(frozen=True)
class LossForwardPlan:
    """Forward-pass requirements declared by a loss strategy.

    A frozen dataclass that communicates how the forward pass should be
    configured before the loss is computed. Strategies inspect the model,
    labels, and config in ``plan_forward`` and return a plan whose
    ``forward_kwargs`` are merged into the model's ``__call__`` invocation.

    For example, ``CausalLMLossStrategy`` returns
    ``LossForwardPlan(forward_kwargs={"apply_lm_head": False})`` so that
    the model skips the dense LM-head projection when the strategy will
    handle it in memory-efficient token chunks instead.

    Attributes:
        forward_kwargs: Extra keyword arguments injected into the model
            forward call. An empty dict (the default) means "run the
            model normally with no overrides".
    """

    forward_kwargs: dict[str, tp.Any] = dataclasses.field(default_factory=dict)


def _resolve_module_vocab_size(module: tp.Any) -> int:
    """Resolve vocab size for text and multimodal configs.

    Some configs expose ``vocab_size`` on the root config, while others keep it
    on ``text_config`` or behind ``get_text_config()``. Models may also cache it
    directly on ``self.vocab_size``.
    """

    candidates: list[tp.Any] = [
        getattr(getattr(module, "config", None), "vocab_size", None),
        getattr(module, "vocab_size", None),
    ]

    config = getattr(module, "config", None)
    if config is not None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_config = get_text_config()
            except TypeError:
                text_config = get_text_config(True)
            except Exception:
                text_config = None
            candidates.append(getattr(text_config, "vocab_size", None))

        text_config = getattr(config, "text_config", None)
        candidates.append(getattr(text_config, "vocab_size", None))

    for candidate in candidates:
        if candidate is None:
            continue
        try:
            return int(candidate)
        except (TypeError, ValueError):
            continue

    raise AttributeError(
        f"Could not resolve vocabulary size from module {type(module).__name__}; "
        "expected one of `config.vocab_size`, `config.text_config.vocab_size`, "
        "`config.get_text_config().vocab_size`, or `module.vocab_size`."
    )


class BaseLossStrategy:
    """Two-stage loss interface: plan the forward pass, then consume outputs.

    Subclasses implement a two-phase protocol used by the training loop:

    1. **plan_forward** — inspect the model, labels, and loss config *before*
       the forward pass and return a ``LossForwardPlan`` whose
       ``forward_kwargs`` are merged into the model call. This lets the
       strategy disable expensive operations (e.g. the LM-head projection)
       when it will handle them more efficiently during loss computation.
    2. **compute** — receive the model outputs (possibly modified by the plan)
       and produce a ``LossMetrics`` scalar.

    The base class returns a no-op plan and raises ``NotImplementedError``
    from ``compute`` so that every concrete strategy is forced to provide
    its own loss logic.
    """

    __name__ = "BaseLossStrategy"

    def plan_forward(
        self,
        *,
        module: tp.Any,
        labels: jax.Array | None,
        loss_config: LossConfig | None,
        batch: collections.abc.Mapping[str, Array],
        loss_kwargs: dict[str, tp.Any],
    ) -> LossForwardPlan:
        """Return forward-pass overrides for the upcoming model call.

        Args:
            module: The EasyDeL model instance.
            labels: Target token IDs (may be ``None`` at inference time).
            loss_config: Loss configuration, or ``None`` for defaults.
            batch: Full training batch mapping.
            loss_kwargs: Extra keyword arguments destined for the loss.

        Returns:
            A ``LossForwardPlan``. The default implementation returns an
            empty plan (no overrides).
        """
        del module, labels, loss_config, batch, loss_kwargs
        return LossForwardPlan()

    def compute(
        self,
        *,
        module: tp.Any,
        outputs: tp.Any,
        labels: jax.Array | None,
        loss_config: LossConfig | None,
        batch: collections.abc.Mapping[str, Array],
        loss_kwargs: dict[str, tp.Any],
        paxis: PartitionAxis | None,
        forward_plan: LossForwardPlan,
    ) -> LossMetrics:
        """Compute the loss from model outputs.

        Args:
            module: The EasyDeL model instance.
            outputs: Model forward-pass outputs (a namespace with ``logits``,
                ``last_hidden_state``, etc.).
            labels: Target token IDs.
            loss_config: Loss configuration.
            batch: Full training batch mapping.
            loss_kwargs: Extra keyword arguments for loss computation.
            paxis: Partition axis spec for distributed sharding, or ``None``.
            forward_plan: The plan returned by ``plan_forward`` for this step.

        Returns:
            A ``LossMetrics`` dataclass with ``loss``, ``z_loss``,
            ``weight_sum``, and ``accuracy``.

        Raises:
            NotImplementedError: Always, in the base class.
        """
        del module, outputs, labels, loss_config, batch, loss_kwargs, paxis, forward_plan
        raise NotImplementedError


class FunctionalLossStrategy(BaseLossStrategy):
    """Compatibility adapter that wraps a plain loss callable as a strategy.

    Many existing loss functions (e.g. ``ForCausalLMLoss``,
    ``ForSequenceClassificationLoss``) follow a simple post-forward
    signature::

        loss_fn(labels=..., config=..., paxis=..., **outputs, **batch)

    This adapter makes any such callable usable through the two-stage
    ``BaseLossStrategy`` protocol by delegating ``compute`` directly to
    the wrapped function. ``plan_forward`` is inherited unchanged (no-op),
    so the model runs its full default forward pass.

    Args:
        loss_fn: A callable that accepts keyword arguments and returns
            ``LossMetrics``.
    """

    def __init__(self, loss_fn: tp.Callable[..., LossMetrics]):
        """Wrap a plain loss callable as a ``BaseLossStrategy``.

        Args:
            loss_fn: A callable that accepts keyword arguments (``labels``,
                ``config``, ``paxis``, plus model outputs and batch entries)
                and returns a ``LossMetrics`` instance.
        """
        self.loss_fn = loss_fn
        self.__name__ = getattr(loss_fn, "__name__", type(loss_fn).__name__)

    def compute(
        self,
        *,
        module: tp.Any,
        outputs: tp.Any,
        labels: jax.Array | None,
        loss_config: LossConfig | None,
        batch: collections.abc.Mapping[str, Array],
        loss_kwargs: dict[str, tp.Any],
        paxis: PartitionAxis | None,
        forward_plan: LossForwardPlan,
    ) -> LossMetrics:
        """Delegate to the wrapped loss callable, ignoring the forward plan.

        The wrapped function receives ``labels``, ``config``, ``paxis``,
        plus all entries from ``loss_kwargs``, ``outputs``, and ``batch``
        unpacked as keyword arguments.
        """
        del module, forward_plan
        return self.loss_fn(
            labels=labels,
            config=loss_config,
            paxis=paxis,
            **loss_kwargs,
            **outputs,
            **batch,
        )


@enum.unique
class SpecialLossNormalizingFactor(enum.Enum):
    """Enumeration for dynamic loss normalization strategies.

    This enum specifies how the loss should be normalized based on properties
    of the input batch, rather than using a fixed constant. These strategies
    are particularly useful for handling variable-length sequences and packed
    batches where padding tokens should not contribute to the loss.

    Attributes:
        NO_WEIGHT_NUM_REAL_TARGET_TOKENS: Divides the loss by the number of
            non-padding target tokens, ignoring any provided loss weights.
            Useful when you want to normalize purely by token count without
            considering importance weights.
        NUM_REAL_TARGET_TOKENS: Divides the loss by the number of non-padding
            target tokens, considering provided loss weights. This is the
            default and most commonly used normalization strategy.
        NUM_TOTAL_TARGET_TOKENS: Divides the loss by the total number of target
            tokens, including padding. Use this when you want consistent
            normalization regardless of padding.
        AVERAGE_PER_SEQUENCE: Computes the average loss per sequence in the
            batch. This normalizes each sequence independently before averaging,
            which can be useful for variable-length sequences to prevent longer
            sequences from dominating the gradient.

    Example:
        >>> from easydel.infra.loss_utils import SpecialLossNormalizingFactor as SLNF
        >>> config = LossConfig(loss_normalizing_factor=SLNF.NUM_REAL_TARGET_TOKENS)
        >>> # Or using string representation
        >>> config = LossConfig(loss_normalizing_factor="NUM_REAL_TARGET_TOKENS")
    """

    NO_WEIGHT_NUM_REAL_TARGET_TOKENS = 0
    NUM_REAL_TARGET_TOKENS = 1
    NUM_TOTAL_TARGET_TOKENS = 2
    AVERAGE_PER_SEQUENCE = 3


SLNF = SpecialLossNormalizingFactor

# Type alias for loss normalizing factor that can be a float, int, string, or enum.
FACTOR_TYPE = float | int | str | SLNF | None


@auto_pytree
class LossConfig:
    """Configuration class for customizing loss computation behavior.

    This class encapsulates all configurable parameters for loss computation,
    including regularization techniques (label smoothing, z-loss), normalization
    strategies, and memory optimization options (chunking).

    Attributes:
        ignore_index: Target value that is ignored and does not contribute to
            the loss or gradient. Commonly set to -100 for padding tokens.
        label_smoothing: Smoothing factor in [0, 1). When > 0, replaces hard
            one-hot targets with soft targets where the true class has
            probability (1 - label_smoothing) and other classes share the
            remaining probability. Helps prevent overconfidence.
        z_loss: Coefficient for z-loss regularization term, which penalizes
            large logits and encourages the log-partition function (logsumexp)
            to remain small. Helps stabilize training for large models.
        loss_normalizing_factor: Strategy for normalizing the loss. Can be:
            - A float/int constant
            - A string matching a SpecialLossNormalizingFactor name
            - A SpecialLossNormalizingFactor enum value
        num_labels: Number of labels for classification tasks. Required for
            sequence classification.
        problem_type: Type of classification problem. One of "regression",
            "single_label_classification", or "multi_label_classification".
        divide_weight_sum: If True, additionally divides loss by sum of weights
            after applying loss_normalizing_factor.
        shift_tokens: If True (default for causal LM), shifts logits and labels
            so the model predicts the next token. Set to False for non-autoregressive
            tasks.
        break_on_nan: If True, raises EasyDeLBreakRequest when NaN is encountered.
        reduction: Reduction method for the loss. One of "none", "mean", or "sum".
            If None, uses the default for the specific loss function.
        num_classification_labels: Alias for num_labels for sequence classification.
        classification_problem_type: Alias for problem_type.
        chunk_vocab_size: If set, enables vocabulary-dimension chunking with this
            chunk size. Reduces memory for large vocabularies.
        chunk_token_size: If set, enables token-dimension chunking with this
            chunk size. Reduces memory for long sequences.
        chunk_block_size: If set, enables blockwise processing with this block
            size. Alternative memory optimization strategy (default=4096).
        compute_dtype: Data type for computation. One of "fp32" or "bf16".
            If None, uses the input dtype.

    Example:
        >>> # Standard configuration with label smoothing
        >>> config = LossConfig(
        ...     label_smoothing=0.1,
        ...     z_loss=1e-4,
        ...     loss_normalizing_factor="NUM_REAL_TARGET_TOKENS"
        ... )
        >>>
        >>> # Memory-efficient configuration for large vocabulary
        >>> config = LossConfig(
        ...     chunk_vocab_size=8192,
        ...     compute_dtype="bf16"
        ... )
    """

    ignore_index: int = -100
    label_smoothing: float = 0.0
    z_loss: float = 0.0
    loss_normalizing_factor: FACTOR_TYPE = "NUM_REAL_TARGET_TOKENS"
    num_labels: int | None = None
    problem_type: str | None = None
    divide_weight_sum: bool = False
    shift_tokens: bool = True
    break_on_nan: bool = True
    reduction: tp.Literal["none", "mean", "sum"] | None = None
    num_classification_labels: int | None = None
    classification_problem_type: (
        tp.Literal["regression", "single_label_classification", "multi_label_classification"] | None
    ) = None
    chunk_vocab_size: int | None = None
    chunk_token_size: int | None = None
    chunk_block_size: int | None = 4096
    compute_dtype: tp.Literal["fp32", "bf16"] | None = None

    def __repr__(self):
        """Return a detailed string representation of the configuration.

        Returns:
            str: Multi-line string showing all configuration fields and values.
        """
        cls_name = self.__class__.__name__
        field_lines = [f"    {f.name}: {getattr(self, f.name)!r}".replace("\n", "\n    ") for f in fields(self)]
        return f"{cls_name}(\n" + "\n".join(field_lines) + "\n)"

    __str__ = __repr__
    __hash__ = hash_fn


@auto_pytree
class LossMetrics:
    """Container for loss metrics and auxiliary training information.

    This class aggregates various metrics computed during loss calculation,
    providing a unified interface for accessing loss values, regularization
    terms, and diagnostic information useful for monitoring training.

    Attributes:
        loss: The primary computed loss value. This is the value that should
            be used for backpropagation.
        z_loss: The computed z-loss regularization term. Included in the total
            loss but tracked separately for monitoring.
        weight_sum: Sum of weights used in loss calculation. Useful for
            verifying normalization and debugging.
        accuracy: Computed accuracy as fraction of correct predictions,
            weighted by the loss weights if applicable.
        learning_rate: Learning rate used for the current step. Populated
            during training for logging purposes.
        max_grad_norm: Maximum gradient norm observed across all parameters.
            Useful for detecting exploding gradients.
        mean_grad_norm: Mean gradient norm across all parameters.
        grad_norms: PyTree containing gradient norms for each parameter.
            Useful for detailed gradient analysis.
        chosen_rewards: Rewards for chosen sequences in preference-based
            training (e.g., DPO, RLHF).
        rejected_rewards: Rewards for rejected sequences in preference-based
            training.
        other_metrics: Dictionary for storing additional custom metrics
            specific to certain training scenarios.
        execution_time: Wall-clock time taken for the computation step
            in seconds.

    Example:
        >>> metrics = ForCausalLMLoss(logits, labels, config=config)
        >>> print(f"Loss: {metrics.loss:.4f}")
        >>> print(f"Accuracy: {metrics.accuracy:.2%}")
        >>> if metrics.z_loss is not None:
        ...     print(f"Z-Loss: {metrics.z_loss:.6f}")
    """

    loss: float | Array | None = None
    z_loss: float | Array | None = None
    weight_sum: float | Array | None = None
    accuracy: float | Array | None = None
    learning_rate: float | Array | None = None
    max_grad_norm: flax.struct.PyTreeNode | None = None
    mean_grad_norm: flax.struct.PyTreeNode | None = None
    grad_norms: flax.struct.PyTreeNode | None = None
    chosen_rewards: jax.Array | None = None
    rejected_rewards: jax.Array | None = None
    other_metrics: collections.abc.Mapping[str, jax.Array] | None = None
    execution_time: float | None = None


def _logsumexp_chunked(x: jnp.ndarray, chunk_size: int) -> jnp.ndarray:
    """Compute logsumexp over the last dimension in chunks for memory efficiency.

    This function computes log(sum(exp(x))) over the last dimension using a
    two-pass chunked approach to reduce peak memory usage. The first pass
    computes the maximum value across chunks, and the second pass computes
    the sum of shifted exponentials.

    This is particularly useful for large vocabulary sizes where materializing
    the full softmax distribution would exceed memory limits.

    Args:
        x: Input array with shape [..., V] where V is the vocabulary size or
            the dimension over which to compute logsumexp.
        chunk_size: Size of chunks to process at a time. Smaller values reduce
            memory usage but may increase computation time.

    Returns:
        Array with shape [...] containing the logsumexp values for each position
        in the leading dimensions.

    Note:
        The chunked computation is mathematically equivalent to the standard
        logsumexp but uses O(chunk_size) memory instead of O(V) for the
        intermediate computations.

    Example:
        >>> logits = jnp.randn(32, 1024, 50000)  # Large vocabulary
        >>> lse = _logsumexp_chunked(logits, chunk_size=8192)
        >>> assert lse.shape == (32, 1024)
    """
    # x: [..., V]
    V: int = x.shape[-1]  # static python int
    n_full = V // chunk_size
    tail = V - n_full * chunk_size  # static python int

    # Pass 1: max
    def max_body(i, m):
        """Compute running maximum across chunks."""
        start = i * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, chunk_size, axis=-1)
        return jnp.maximum(m, jnp.max(chunk, axis=-1))

    m = jnp.full(x.shape[:-1], -jnp.inf, dtype=x.dtype)
    m = lax.fori_loop(0, n_full, max_body, m)
    if tail:
        start = n_full * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, tail, axis=-1)
        m = jnp.maximum(m, jnp.max(chunk, axis=-1))

    # Pass 2: sum of exp(x - m)
    def sum_body(i, s):
        """Compute running sum of shifted exponentials."""
        start = i * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, chunk_size, axis=-1)
        return s + jnp.sum(jnp.exp(chunk - m[..., None]), axis=-1)

    s = jnp.zeros_like(m)
    s = lax.fori_loop(0, n_full, sum_body, s)
    if tail:
        start = n_full * chunk_size
        chunk = lax.dynamic_slice_in_dim(x, start, tail, axis=-1)
        s = s + jnp.sum(jnp.exp(chunk - m[..., None]), axis=-1)

    return jnp.log(s) + m


def _label_smoothing_params(
    vocab_size: int,
    label_smoothing: float,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute label-smoothing coefficients equivalent to the dense one-hot formulation.

    Given a smoothing factor ``eps``, the smoothed target distribution is::

        q(k) = (1 - eps) * delta(k, y) + eps / V

    This function returns the three scalars needed to evaluate the
    cross-entropy against that distribution *without* materializing the
    full ``[..., V]`` one-hot tensor:

    * ``confidence`` = 1 - eps   (mass on the true class)
    * ``low_confidence`` = eps / (V - 1)   (mass on each non-true class)
    * ``normalizing_constant`` = H(q)   (entropy of the smoothed target,
      subtracted so that a perfect model achieves zero loss)

    Args:
        vocab_size: Number of classes (V).
        label_smoothing: Smoothing factor in (0, 1).
        dtype: Computation dtype for the returned arrays.

    Returns:
        Tuple of ``(confidence, low_confidence, normalizing_constant)``,
        each a scalar ``jax.Array`` of the requested dtype.
    """
    confidence = jnp.asarray(1.0 - label_smoothing, dtype=dtype)
    low_confidence = (jnp.asarray(1.0, dtype=dtype) - confidence) / jnp.asarray(vocab_size - 1, dtype=dtype)
    normalizing_constant = -(
        confidence * jnp.log(confidence)
        + jnp.asarray(vocab_size - 1, dtype=dtype) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    return confidence, low_confidence, normalizing_constant


def _apply_sparse_label_smoothing(
    log_z: jax.Array,
    target_logit: jax.Array,
    sum_logits: jax.Array,
    *,
    vocab_size: int,
    label_smoothing: float,
    dtype: jnp.dtype,
) -> jax.Array:
    """Compute label-smoothed cross-entropy in the sparse (index-based) setting.

    This is mathematically equivalent to::

        (1 - eps) * NLL  +  eps * (log_z - mean_logits)

    but uses the dense-parity decomposition from ``_label_smoothing_params``
    to avoid the ``eps * mean`` approximation and to match the dense
    one-hot loss path exactly. The result is::

        log_z - [(confidence - low_confidence) * target_logit
                 + low_confidence * sum_logits]
        - normalizing_constant

    where ``confidence``, ``low_confidence``, and ``normalizing_constant``
    come from ``_label_smoothing_params``.

    Args:
        log_z: Log-partition function ``logsumexp(logits, axis=-1)``,
            shape ``[...]``.
        target_logit: Logit at the true-class index for each token,
            shape ``[...]``.
        sum_logits: Sum of all logits along the vocab axis, shape ``[...]``.
        vocab_size: Number of classes (V).
        label_smoothing: Smoothing factor in (0, 1).
        dtype: Computation dtype.

    Returns:
        Smoothed per-token cross-entropy, same shape as ``log_z``.
    """
    confidence, low_confidence, normalizing_constant = _label_smoothing_params(
        vocab_size=vocab_size,
        label_smoothing=label_smoothing,
        dtype=dtype,
    )
    target_mass_logits = (confidence - low_confidence) * target_logit + low_confidence * sum_logits
    return log_z - target_mass_logits - normalizing_constant


def cross_entropy_blockwise_logits(
    logits: jax.Array,  # [B, T, V] or [N, V]
    targets: jax.Array,  # [B, T] or [N]
    weights: jax.Array | None = None,  # [B, T] or [N]
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    block_size: int = 8192,
    dtype: jnp.dtype | None = jnp.float32,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute blockwise sparse cross-entropy without materializing full softmax.

    This function implements a memory-efficient cross-entropy computation that
    processes the vocabulary dimension in blocks. It avoids materializing the
    full softmax distribution or one-hot encoded targets, making it suitable
    for large vocabulary sizes.

    The implementation uses a streaming approach to compute:
    1. Running logsumexp for normalization
    2. Target logits via sparse indexing
    3. Sum of logits for label smoothing
    4. Running argmax for accuracy computation

    Args:
        logits: Model output logits with shape [B, T, V] (batch, sequence, vocab)
            or [N, V] (flattened tokens, vocab).
        targets: Target token indices with shape [B, T] or [N]. Values should be
            in [0, V) or equal to ignore_index for masked positions.
        weights: Optional per-token weights with shape [B, T] or [N]. If None,
            uses binary weights based on ignore_index masking.
        ignore_index: Index value to ignore in loss computation. Typically -100
            for padding tokens.
        label_smoothing: Label smoothing factor in [0, 1). When > 0, distributes
            some probability mass to non-target classes.
        z_loss: Coefficient for z-loss regularization. Adds z_loss * logsumexp^2
            to the loss to encourage smaller logits.
        block_size: Size of vocabulary blocks to process at a time. Must be > 0.
            Smaller values reduce memory but may increase computation.
        dtype: Data type for computation. If None, uses input dtype. Float32
            recommended for numerical stability.

    Returns:
        Tuple of four arrays:
            - total_loss: Sum of weighted per-token losses (scalar).
            - total_z_loss: Sum of weighted z-loss terms (scalar).
            - weight_sum: Sum of weights for normalization (scalar).
            - accuracy: Weighted accuracy as fraction in [0, 1] (scalar).

    Raises:
        ValueError: If logits has invalid shape (not 2D or 3D) or block_size <= 0.

    Example:
        >>> logits = jnp.randn(4, 512, 50000)  # Large vocabulary
        >>> targets = jnp.randint(0, 50000, (4, 512))
        >>> loss, z_loss, w_sum, acc = cross_entropy_blockwise_logits(
        ...     logits, targets, block_size=8192
        ... )
        >>> normalized_loss = loss / w_sum
    """
    compute_dtype = jnp.dtype(dtype or logits.dtype)

    # Flatten tokens
    if logits.ndim == 3:
        B, T, V = logits.shape
        L = B * T
        logits2d = logits.reshape(L, V)
        y = targets.reshape(L)
        w = None if weights is None else weights.reshape(L).astype(compute_dtype)
    elif logits.ndim == 2:
        L, V = logits.shape  # pyright: ignore[reportConstantRedefinition]
        logits2d = logits
        y = targets
        w = None if weights is None else weights.astype(compute_dtype)
    else:
        raise ValueError(f"logits must be [B, T, V] or [N, V], got {logits.shape}")

    if block_size <= 0:
        raise ValueError(f"block_size must be > 0, got {block_size}")

    # Upcast for numerical stability
    logits2d = logits2d.astype(compute_dtype)

    # Valid/weights
    valid = y != ignore_index
    y_safe = jnp.where(valid, y, 0)
    w = valid.astype(compute_dtype) if w is None else valid.astype(compute_dtype) * w

    # Accumulators
    neg_inf = jnp.array(-jnp.inf, dtype=compute_dtype)
    m = jnp.full((L,), neg_inf)
    log_z = jnp.full((L,), neg_inf)
    o = jnp.zeros((L,), dtype=compute_dtype)  # sum of target logits
    sum_logits = jnp.zeros((L,), dtype=compute_dtype)  # for smoothing
    best_logit = jnp.full((L,), neg_inf)
    best_id = jnp.zeros((L,), dtype=jnp.int32)

    n_full = V // block_size
    tail = V - n_full * block_size

    def process_block(start: int, size: int, m, log_z, o, sum_logits, best_logit, best_id):
        """Process a single vocabulary block and update accumulators."""
        # Static slice sizes: size is either block_size (in a fori_loop) or tail (once)
        chunk = lax.dynamic_slice_in_dim(logits2d, start, size, axis=1)  # [L, size]

        # Running logsumexp via updated max
        chunk_max = jnp.max(chunk, axis=1)
        new_m = jnp.maximum(m, chunk_max)
        log_z = new_m + jnp.log(jnp.exp(log_z - new_m) + jnp.sum(jnp.exp(chunk - new_m[:, None]), axis=1))
        m = new_m

        # Accumulate target logit (sparse)
        in_block = (y_safe >= start) & (y_safe < start + size)
        idx = jnp.where(in_block, (y_safe - start).astype(jnp.int32), 0)
        logit_y_b = jnp.take_along_axis(chunk, idx[:, None], axis=1)[:, 0]
        o = o + jnp.where(in_block, logit_y_b, 0.0)

        # Sum logits for smoothing
        sum_logits = sum_logits + jnp.sum(chunk, axis=1)

        # Streamed argmax for accuracy
        block_best = jnp.argmax(chunk, axis=1)
        block_best_id = start + block_best.astype(jnp.int32)
        update = chunk_max > best_logit
        best_logit = jnp.where(update, chunk_max, best_logit)
        best_id = jnp.where(update, block_best_id, best_id)

        return m, log_z, o, sum_logits, best_logit, best_id

    # The blockwise CE loop can otherwise save every [tokens, block_size] chunk
    # for backward, which explodes memory at long sequence lengths.
    process_block = jax.checkpoint(process_block, prevent_cse=False, static_argnums=(1,))

    def full_body(i, carry):
        """Process a full block in the fori_loop."""
        start = i * block_size
        return process_block(start, block_size, *carry)

    carry = (m, log_z, o, sum_logits, best_logit, best_id)
    if n_full > 0:
        carry = lax.fori_loop(0, n_full, full_body, carry)
    if tail:
        start = n_full * block_size
        carry = process_block(start, tail, *carry)

    m, log_z, o, sum_logits, best_logit, best_id = carry

    # Base CE
    nll = log_z - o  # [L]

    if label_smoothing and label_smoothing != 0.0:
        nll = _apply_sparse_label_smoothing(
            log_z,
            o,
            sum_logits,
            vocab_size=V,
            label_smoothing=label_smoothing,
            dtype=compute_dtype,
        )

    # z-loss term
    zterm = (z_loss * (log_z**2)) if (z_loss and z_loss != 0.0) else 0.0
    per_tok = nll + (zterm if (z_loss and z_loss != 0.0) else 0.0)

    total_loss = jnp.sum(per_tok * w)
    total_z_loss = jnp.sum((zterm if (z_loss and z_loss != 0.0) else 0.0) * w)
    weight_sum = jnp.sum(w)

    # Weighted accuracy
    acc = jnp.sum((best_id == y_safe).astype(compute_dtype) * w) / jnp.maximum(
        weight_sum,
        jnp.asarray(1e-8, dtype=compute_dtype),
    )

    return total_loss, total_z_loss, weight_sum, acc


def sparse_cross_entropy_chunked_vocab(
    logits: jnp.ndarray,  # [..., V]
    targets: jnp.ndarray,  # [...]
    weights: jnp.ndarray | None = None,  # [...]
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "mean",
    chunk_size: int = 8192,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute sparse cross-entropy loss with vocabulary chunking.

    This function chunks along the vocabulary dimension to reduce memory usage
    when computing cross-entropy for large vocabularies. It uses the chunked
    logsumexp implementation for numerical stability.

    Unlike `cross_entropy_blockwise_logits`, this function uses a simpler
    approach that first computes the full logsumexp, then gathers the target
    logit. This is more efficient when the vocabulary fits in memory after
    chunking but the full softmax distribution would not.

    Args:
        logits: Model logits with shape [..., V] where V is vocabulary size.
            Can be any number of leading dimensions.
        targets: Target token indices with shape [...] matching logits' leading
            dimensions.
        weights: Optional per-token weights with shape [...]. If None, uses
            binary weights based on ignore_index.
        ignore_index: Index value to ignore in loss computation.
        label_smoothing: Label smoothing factor in [0, 1).
        z_loss: Coefficient for z-loss regularization.
        reduction: Reduction type. "mean" divides by weight sum, "sum" returns
            raw sum.
        chunk_size: Vocabulary chunk size for memory efficiency.
        compute_dtype: Data type for computation.

    Returns:
        Tuple of four arrays:
            - total_loss: Reduced loss value.
            - total_z_loss: Sum of weighted z-loss terms.
            - weight_sum: Sum of weights.
            - accuracy: Weighted accuracy.

    Example:
        >>> logits = jnp.randn(8, 256, 100000)  # Very large vocabulary
        >>> targets = jnp.randint(0, 100000, (8, 256))
        >>> loss, z_loss, w_sum, acc = sparse_cross_entropy_chunked_vocab(
        ...     logits, targets, chunk_size=16384
        ... )
    """
    logits = logits.astype(compute_dtype)
    valid = targets != ignore_index
    safe_targets = jnp.where(valid, targets, 0)

    lse = _logsumexp_chunked(logits, chunk_size)  # [...,]
    logit_y = jnp.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    nll = lse - logit_y

    if label_smoothing > 0.0:
        nll = _apply_sparse_label_smoothing(
            lse,
            logit_y,
            jnp.sum(logits, axis=-1),
            vocab_size=logits.shape[-1],
            label_smoothing=label_smoothing,
            dtype=compute_dtype,
        )

    z_term = (z_loss * jnp.square(lse)) if z_loss > 0.0 else 0.0
    nll = nll + (z_term if z_loss > 0.0 else 0.0)

    w = valid.astype(compute_dtype) if weights is None else valid.astype(compute_dtype) * weights.astype(compute_dtype)

    total_loss = jnp.sum(nll * w)
    total_z_loss = jnp.sum((z_term if z_loss > 0.0 else 0.0) * w)
    weight_sum = jnp.sum(w)

    if reduction == "mean":
        total_loss = total_loss / jnp.maximum(weight_sum, 1e-8)

    # Weighted accuracy
    correct = (jnp.argmax(logits, axis=-1) == targets).astype(compute_dtype) * w
    accuracy = jnp.sum(correct) / jnp.maximum(weight_sum, 1e-8)

    return total_loss, total_z_loss, weight_sum, accuracy


def sparse_cross_entropy_chunked_tokens(
    logits: jnp.ndarray,  # [B, T, V] or [N, V]
    targets: jnp.ndarray,  # [B, T] or [N]
    weights: jnp.ndarray | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    reduction: str = "sum",  # sum here; normalize outside for consistency
    token_chunk_size: int = 8192,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute sparse cross-entropy loss with token sequence chunking.

    This function chunks along the token/batch dimension to reduce memory usage
    when computing cross-entropy for long sequences or large batches. This is
    complementary to vocabulary chunking and can be used when the sequence
    length is the memory bottleneck.

    The function processes tokens in chunks using a fori_loop for the full
    chunks and handles any remaining tokens separately.

    Args:
        logits: Model logits with shape [B, T, V] (batch, sequence, vocab)
            or [N, V] (flattened tokens, vocab).
        targets: Target token indices with shape [B, T] or [N].
        weights: Optional per-token weights with shape matching targets.
        ignore_index: Index value to ignore in loss computation.
        label_smoothing: Label smoothing factor in [0, 1).
        z_loss: Coefficient for z-loss regularization.
        reduction: Reduction type. "mean" divides by weight sum, "sum" returns
            raw sum. Default is "sum" for external normalization.
        token_chunk_size: Number of tokens to process at a time.
        compute_dtype: Data type for computation.

    Returns:
        Tuple of four arrays:
            - total_loss: Reduced loss value.
            - total_z_loss: Sum of weighted z-loss terms.
            - weight_sum: Sum of weights.
            - accuracy: Weighted accuracy.

    Example:
        >>> # Long sequence with moderate vocabulary
        >>> logits = jnp.randn(2, 32768, 32000)
        >>> targets = jnp.randint(0, 32000, (2, 32768))
        >>> loss, z_loss, w_sum, acc = sparse_cross_entropy_chunked_tokens(
        ...     logits, targets, token_chunk_size=4096
        ... )
    """
    logits = logits.astype(compute_dtype)
    V = logits.shape[-1]
    logits2d = logits.reshape(-1, V)
    targets1d = targets.reshape(-1)
    weights1d = None if weights is None else weights.reshape(-1).astype(compute_dtype)
    N: int = logits2d.shape[0]
    n_full = N // token_chunk_size
    tail = N - n_full * token_chunk_size

    def body(i, carry):
        """Process a chunk of tokens and accumulate results."""
        tot, wsum, acc_sum, zsum = carry
        start = i * token_chunk_size
        chunk_logits = lax.dynamic_slice_in_dim(logits2d, start, token_chunk_size, axis=0)
        chunk_targets = lax.dynamic_slice_in_dim(targets1d, start, token_chunk_size, axis=0)
        chunk_weights = (
            None if weights1d is None else lax.dynamic_slice_in_dim(weights1d, start, token_chunk_size, axis=0)
        )

        lse = jax.scipy.special.logsumexp(chunk_logits, axis=-1)
        valid = chunk_targets != ignore_index
        safe_chunk_targets = jnp.where(valid, chunk_targets, 0)
        logit_y = jnp.take_along_axis(chunk_logits, safe_chunk_targets[:, None], axis=-1)[:, 0]
        nll = lse - logit_y

        if label_smoothing > 0.0:
            nll = _apply_sparse_label_smoothing(
                lse,
                logit_y,
                jnp.sum(chunk_logits, axis=-1),
                vocab_size=V,
                label_smoothing=label_smoothing,
                dtype=compute_dtype,
            )

        zterm = (z_loss * jnp.square(lse)) if z_loss > 0.0 else 0.0
        nll = nll + (zterm if z_loss > 0.0 else 0.0)

        w = valid.astype(compute_dtype) if chunk_weights is None else valid.astype(compute_dtype) * chunk_weights
        loss_sum = jnp.sum(nll * w)
        w_sum = jnp.sum(w)
        z_sum = jnp.sum((zterm if z_loss > 0.0 else 0.0) * w)
        acc = jnp.sum((jnp.argmax(chunk_logits, axis=-1) == chunk_targets).astype(compute_dtype) * w)

        return (tot + loss_sum, wsum + w_sum, acc_sum + acc, zsum + z_sum)

    # Full chunks
    init = (
        jnp.array(0.0, compute_dtype),
        jnp.array(0.0, compute_dtype),
        jnp.array(0.0, compute_dtype),
        jnp.array(0.0, compute_dtype),
    )
    carry = init
    carry = lax.fori_loop(0, n_full, body, carry)

    # Tail
    if tail:
        start = n_full * token_chunk_size
        chunk_logits = lax.dynamic_slice_in_dim(logits2d, start, tail, axis=0)
        chunk_targets = lax.dynamic_slice_in_dim(targets1d, start, tail, axis=0)
        chunk_weights = None if weights1d is None else lax.dynamic_slice_in_dim(weights1d, start, tail, axis=0)

        lse = jax.scipy.special.logsumexp(chunk_logits, axis=-1)
        valid = chunk_targets != ignore_index
        safe_chunk_targets = jnp.where(valid, chunk_targets, 0)
        logit_y = jnp.take_along_axis(chunk_logits, safe_chunk_targets[:, None], axis=-1)[:, 0]
        nll = lse - logit_y

        if label_smoothing > 0.0:
            nll = _apply_sparse_label_smoothing(
                lse,
                logit_y,
                jnp.sum(chunk_logits, axis=-1),
                vocab_size=V,
                label_smoothing=label_smoothing,
                dtype=compute_dtype,
            )

        zterm = (z_loss * jnp.square(lse)) if z_loss > 0.0 else 0.0
        nll = nll + (zterm if z_loss > 0.0 else 0.0)

        w = valid.astype(compute_dtype) if chunk_weights is None else valid.astype(compute_dtype) * chunk_weights
        loss_sum = jnp.sum(nll * w)
        w_sum = jnp.sum(w)
        z_sum = jnp.sum((zterm if z_loss > 0.0 else 0.0) * w)
        acc = jnp.sum((jnp.argmax(chunk_logits, axis=-1) == chunk_targets).astype(compute_dtype) * w)

        carry = (carry[0] + loss_sum, carry[1] + w_sum, carry[2] + acc, carry[3] + z_sum)

    total_loss, total_wsum, acc_sum, total_z_loss = carry

    if reduction == "mean":
        total_loss = total_loss / jnp.maximum(total_wsum, 1e-8)

    accuracy = acc_sum / jnp.maximum(total_wsum, 1e-8)
    return total_loss, total_z_loss, total_wsum, accuracy


def dynamic_cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weight: jnp.ndarray | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    label_smoothing: float = 0.0,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute cross-entropy loss with flexible reduction and label smoothing.

    This function provides a PyTorch-like interface for cross-entropy loss with
    support for ignore index, label smoothing, and various reduction modes. It
    handles the masking of ignored positions and computes the appropriate
    normalization factors.

    Args:
        logits: Model logits with shape (batch_size, ..., num_classes). The last
            dimension should be the class dimension.
        targets: Target class indices with shape (batch_size, ...). Values should
            be integers in [0, num_classes) or equal to ignore_index.
        weight: Optional per-element weights with shape (batch_size, ...). If
            provided, the loss for each position is multiplied by the
            corresponding weight.
        ignore_index: Index value to ignore in loss computation. Positions with
            this target value contribute neither to the loss nor to normalization.
        reduction: Specifies the reduction to apply to the output:
            - "mean": Returns the weighted mean of the loss.
            - "sum": Returns the sum of the weighted loss.
            - "none": Returns the per-element loss without reduction.
        label_smoothing: Label smoothing factor in [0, 1). When > 0, the target
            distribution becomes (1 - eps) * one_hot + eps * uniform.
        compute_dtype: Data type for internal computation.

    Returns:
        Tuple of two arrays:
            - loss: The computed loss. Scalar if reduction is "mean" or "sum",
              array with same shape as targets if reduction is "none".
            - norm: Normalization factor. Sum of weights for "mean"/"sum"
              reduction, per-element weights for "none" reduction.

    Raises:
        ValueError: If reduction is not one of "mean", "sum", or "none".

    Example:
        >>> logits = jnp.randn(4, 10, 1000)  # 4 samples, 10 positions, 1000 classes
        >>> targets = jnp.randint(0, 1000, (4, 10))
        >>> loss, norm = dynamic_cross_entropy_loss(logits, targets, reduction="mean")
    """
    logits = logits.astype(compute_dtype)
    valid = targets != ignore_index
    safe_targets = jnp.where(valid, targets, 0)
    lse = jax.scipy.special.logsumexp(logits, axis=-1)
    logit_y = jnp.take_along_axis(logits, safe_targets[..., None], axis=-1)[..., 0]
    nll = lse - logit_y
    if label_smoothing > 0.0:
        nll = _apply_sparse_label_smoothing(
            lse,
            logit_y,
            jnp.sum(logits, axis=-1),
            vocab_size=logits.shape[-1],
            label_smoothing=label_smoothing,
            dtype=compute_dtype,
        )
    w = valid.astype(compute_dtype) if weight is None else valid.astype(compute_dtype) * weight.astype(compute_dtype)
    loss = nll * w
    norm = jnp.maximum(jnp.sum(w), 1e-8)

    if reduction == "mean":
        return jnp.sum(loss) / norm, norm
    elif reduction == "sum":
        return jnp.sum(loss), norm
    elif reduction == "none":
        return loss, w
    else:
        raise ValueError(f"Invalid reduction: {reduction}. Must be 'mean', 'sum', or 'none'.")


def sigmoid_cross_entropy_with_logits(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    weights: jnp.ndarray | None = None,
    label_smoothing: float = 0.0,
    axis: int | tuple | None = None,
) -> jnp.ndarray:
    """Compute sigmoid cross-entropy loss for multi-label classification.

    This function computes the binary cross-entropy loss using sigmoid activation,
    suitable for multi-label classification where each class is independent and
    not mutually exclusive. Each output is treated as a separate binary
    classification problem.

    The loss is computed as: -labels * log(sigmoid(logits)) - (1-labels) * log(1-sigmoid(logits))

    This is numerically stable and avoids computing the sigmoid explicitly.

    Args:
        logits: Model logits with arbitrary shape. Each element is treated as
            an independent binary classification.
        labels: Target labels with same shape as logits. Values should be in
            [0, 1] for binary labels, or soft labels for label smoothing.
        weights: Optional weights with same shape as logits. If provided,
            the loss for each position is multiplied by the corresponding weight.
        label_smoothing: Label smoothing factor in [0, 1). When > 0, labels are
            smoothed toward 0.5: labels * (1 - eps) + 0.5 * eps.
        axis: Axis or axes along which to compute the mean. If None, computes
            the mean over all elements.

    Returns:
        Scalar or array containing the mean loss. Shape depends on axis parameter.

    Example:
        >>> # Multi-label classification with 5 classes
        >>> logits = jnp.randn(32, 5)
        >>> labels = jnp.array([[1, 0, 1, 0, 1], ...])  # Multi-hot encoding
        >>> loss = sigmoid_cross_entropy_with_logits(logits, labels)
    """
    if label_smoothing > 0.0:
        labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    loss = -labels * log_p - (1.0 - labels) * log_not_p

    if weights is not None:
        loss *= weights

    if axis is None:
        return jnp.mean(loss)
    else:
        return jnp.mean(loss, axis=axis)


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
    """Create one-hot encoded representations of integer labels.

    This function converts integer class labels to one-hot vectors, with support
    for custom on/off values to enable label smoothing or other soft target
    distributions.

    Args:
        labels: Array of integer labels with arbitrary shape. Values should be
            in [0, num_classes).
        num_classes: Total number of classes. Determines the size of the last
            dimension in the output.
        on_value: Value to use for the "on" position (the true class).
            Defaults to 1.0.
        off_value: Value to use for all "off" positions (non-true classes).
            Defaults to 0.0.

    Returns:
        One-hot encoded array with shape labels.shape + (num_classes,). The
        position corresponding to each label value contains on_value, and all
        other positions contain off_value.

    Example:
        >>> labels = jnp.array([0, 2, 1])
        >>> onehot(labels, num_classes=3)
        Array([[1., 0., 0.],
               [0., 0., 1.],
               [0., 1., 0.]], dtype=float32)
        >>>
        >>> # With label smoothing (confidence 0.9, smoothing 0.1)
        >>> onehot(labels, 3, on_value=0.9, off_value=0.05)
    """
    x = lax.eq(labels[..., None], jnp.arange(num_classes)[(None,) * labels.ndim])
    y = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
    return y


@jax.custom_vjp
def cross_entropy_with_logits(
    logits: Array,
    targets: Array,
    z_loss: float,
) -> tuple[Array, Array]:
    """Compute cross-entropy loss with optional z-loss regularization.

    This function calculates the standard cross-entropy loss between logits and
    soft targets (typically one-hot encoded). It includes an optional z-loss
    term that penalizes large log-partition function values, encouraging the
    model to produce smaller logits.

    A custom VJP (vector-Jacobian product) is defined for efficient gradient
    computation, avoiding redundant softmax calculations in the backward pass.

    Args:
        logits: Model logits with shape (batch_size, ..., num_classes). The last
            dimension is the class dimension.
        targets: Target distribution with same shape as logits. Typically one-hot
            encoded but can be soft targets for label smoothing.
        z_loss: Coefficient for z-loss regularization. The z-loss term is
            z_loss * logsumexp(logits)^2. Set to 0 to disable.

    Returns:
        Tuple of two arrays:
            - loss: Cross-entropy loss plus z-loss for each sample, shape
              (batch_size, ...).
            - z_loss_value: The z-loss contribution for each sample, shape
              (batch_size, ...). Returns zeros if z_loss coefficient is 0.

    Note:
        This function expects soft targets (probability distributions). For
        integer labels, first convert using `onehot()` or use
        `compute_weighted_cross_entropy()` which handles this internally.

    Example:
        >>> logits = jnp.randn(4, 1000)
        >>> targets = jax.nn.one_hot(jnp.array([5, 10, 3, 7]), 1000)
        >>> loss, z_loss_val = cross_entropy_with_logits(logits, targets, z_loss=1e-4)
    """
    logsumexp = jax.scipy.special.logsumexp(logits, axis=-1, keepdims=False)
    log_softmax = logits - logsumexp[..., None]
    ce = -jnp.sum(targets * log_softmax, axis=-1)
    z = z_loss * jnp.square(logsumexp)
    return ce + z, z


def _cross_entropy_with_logits_fwd(
    logits: Array,
    targets: Array,
    z_loss: float = 0.0,
) -> tuple[
    tuple[
        Array,
        Array,
    ],
    tuple[
        Array,
        float,
        Array,
        Array,
        Array,
        Array,
    ],
]:
    """Forward pass for cross_entropy_with_logits custom VJP.

    Computes cross-entropy loss with z-loss regularization and saves intermediate
    values needed for efficient gradient computation in the backward pass. This
    implementation uses the numerically stable max-subtraction trick.

    Args:
        logits: Model predictions with shape (batch_size, ..., num_classes).
        targets: One-hot encoded targets with same shape as logits.
        z_loss: Coefficient for z-loss regularization.

    Returns:
        Tuple containing:
            - (loss, z_loss_value): The computed losses, both with shape
              (batch_size, ...).
            - Residuals tuple for backward pass containing:
              (targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z)
              where exp_shifted and sum_exp are intermediate values from the
              softmax computation.
    """
    max_logit = logits.max(axis=-1, keepdims=True)
    shifted = logits - max_logit
    exp_shifted = jnp.exp(shifted)
    sum_exp = jnp.sum(exp_shifted, axis=-1, keepdims=True)
    log_softmax = shifted - jnp.log(sum_exp)
    ce = -jnp.sum(targets * log_softmax, axis=-1)
    log_z = jnp.squeeze(jnp.log(sum_exp) + max_logit, axis=-1)
    z = z_loss * jax.lax.square(log_z)
    ce_plus_z = ce + z
    y = (ce_plus_z, z)
    res = (targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z)
    return y, res


def _cross_entropy_with_logits_bwd(
    res: tuple[
        Array,
        float,
        Array,
        Array,
        Array,
        Array,
    ],
    g: tuple[Array, Array],
) -> tuple[Array, Array, Array]:
    """Backward pass for cross_entropy_with_logits custom VJP.

    Computes gradients with respect to logits and targets using saved intermediate
    values from the forward pass. This avoids recomputing the softmax, making the
    backward pass more efficient.

    The gradient with respect to logits is:
        d_loss/d_logits = softmax - targets + 2 * z_loss * log_z * softmax

    The gradient with respect to targets is:
        d_loss/d_targets = -log_softmax

    Args:
        res: Residuals from forward pass containing:
            (targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z)
        g: Gradient tuple (g_loss, g_z_loss) where g_loss is the gradient of the
            final loss with respect to the cross-entropy output.

    Returns:
        Tuple of gradients (g_logits, g_targets, g_z_loss) with respect to the
        three inputs of the forward function.
    """
    g0 = g[0]
    targets, z_loss, exp_shifted, sum_exp, log_softmax, log_z = res
    softmax = exp_shifted / sum_exp
    deriv = softmax - targets + jnp.expand_dims(2 * z_loss * log_z, -1) * softmax
    g_logits = jnp.expand_dims(g0, -1) * deriv
    g_targets = -jnp.expand_dims(g0, -1) * log_softmax
    return (
        jnp.asarray(g_logits, dtype=log_softmax.dtype),
        jnp.asarray(g_targets, dtype=targets.dtype),
        jnp.array(0.0, dtype=log_softmax.dtype),
    )


cross_entropy_with_logits.defvjp(_cross_entropy_with_logits_fwd, _cross_entropy_with_logits_bwd)


def compute_weighted_cross_entropy(
    logits: Array,
    targets: Array,
    weights: Array | None = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: float | None = None,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, Array, Array]:
    """Compute weighted cross-entropy loss with label smoothing and z-loss.

    This function provides a complete cross-entropy computation pipeline including:
    - Label smoothing for regularization
    - Z-loss for numerical stability
    - Optional weighting for masked or importance-weighted training
    - Optional normalization factor

    The label smoothing modifies the target distribution from a one-hot encoding
    to a mixture of the one-hot with a uniform distribution.

    Args:
        logits: Model logits with shape (..., num_classes).
        targets: Integer class labels with shape (...). Must have one fewer
            dimension than logits.
        weights: Optional weights with shape (...) matching targets. If provided,
            the loss is multiplied by these weights element-wise.
        label_smoothing: Smoothing factor in [0, 1). The target distribution
            becomes (1 - eps) * one_hot + eps / (num_classes - 1) * (1 - one_hot).
        z_loss: Coefficient for z-loss regularization.
        loss_normalizing_factor: If provided, divides the total loss and z-loss
            by this factor.
        compute_dtype: Data type for internal computation.

    Returns:
        Tuple of three arrays:
            - total_loss: Sum of weighted cross-entropy losses (scalar).
            - total_z_loss: Sum of weighted z-loss terms (scalar).
            - weight_sum: Sum of weights or total number of elements (scalar).

    Raises:
        TypeError: If logits, targets, or weights are not JAX arrays (or None
            for weights).
        ValueError: If label_smoothing is not in [0, 1), z_loss is negative,
            or shapes are incompatible.

    Example:
        >>> logits = jnp.randn(4, 10, 1000)
        >>> targets = jnp.randint(0, 1000, (4, 10))
        >>> mask = jnp.ones((4, 10))
        >>> loss, z_loss, w_sum = compute_weighted_cross_entropy(
        ...     logits, targets, weights=mask, label_smoothing=0.1
        ... )
        >>> normalized_loss = loss / w_sum
    """
    if not isinstance(logits, jax.Array):
        raise TypeError(f"logits must be a JAX array, got {type(logits)}")
    if not isinstance(targets, jax.Array):
        raise TypeError(f"targets must be a JAX array, got {type(targets)}")
    if weights is not None and not isinstance(weights, jax.Array):
        raise TypeError(f"weights must be a JAX array or None, got {type(weights)}")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in range 0~1, got {label_smoothing}")
    if z_loss < 0.0:
        raise ValueError(f"z_loss must be non-negative, got {z_loss}")
    if logits.ndim != targets.ndim + 1:
        raise ValueError(f"Incorrect shapes. Got shape {logits.shape} logits and {targets.shape} targets")
    vocab_size = logits.shape[-1]
    confidence = 1.0 - label_smoothing
    low_confidence = (1.0 - confidence) / (vocab_size - 1)
    normalizing_constant = -(
        confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
    )
    soft_targets = onehot(targets, vocab_size, on_value=confidence, off_value=low_confidence).astype(compute_dtype)
    total_loss, total_z_loss = cross_entropy_with_logits(logits.astype(compute_dtype), soft_targets, z_loss=z_loss)
    total_loss = total_loss - normalizing_constant

    weight_sum = jnp.array(reduce(mul, targets.shape, 1), dtype=compute_dtype)
    if weights is not None:
        total_loss = total_loss * weights
        total_z_loss = total_z_loss * weights
        weight_sum = jnp.sum(weights.astype(compute_dtype))

    if loss_normalizing_factor is not None:
        total_loss /= loss_normalizing_factor
        total_z_loss /= loss_normalizing_factor

    return jnp.sum(total_loss), jnp.sum(total_z_loss), weight_sum


def compute_weighted_cross_entropy_and_accuracy(
    logits: Array,
    targets: Array,
    weights: Array | None = None,
    label_smoothing: float = 0.0,
    z_loss: float = 0.0,
    loss_normalizing_factor: float | None = None,
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[Array, Array, Array, Array]:
    """Compute weighted cross-entropy loss, z-loss, weight sum, and accuracy.

    This is an extension of `compute_weighted_cross_entropy` that also computes
    the prediction accuracy, useful for monitoring training progress.

    Args:
        logits: Model logits with shape (..., num_classes).
        targets: Integer class labels with shape (...).
        weights: Optional weights with shape (...) matching targets.
        label_smoothing: Smoothing factor in [0, 1).
        z_loss: Coefficient for z-loss regularization.
        loss_normalizing_factor: Optional factor to divide the loss.
        compute_dtype: Data type for internal computation.

    Returns:
        Tuple of four arrays:
            - total_loss: Sum of weighted cross-entropy losses (scalar).
            - total_z_loss: Sum of weighted z-loss terms (scalar).
            - weight_sum: Sum of weights (scalar).
            - accuracy: Weighted accuracy as fraction of correct predictions (scalar).

    Example:
        >>> logits = jnp.randn(4, 10, 1000)
        >>> targets = jnp.randint(0, 1000, (4, 10))
        >>> loss, z_loss, w_sum, acc = compute_weighted_cross_entropy_and_accuracy(
        ...     logits, targets, label_smoothing=0.1
        ... )
        >>> print(f"Loss: {loss/w_sum:.4f}, Accuracy: {acc:.2%}")
    """
    total_loss, total_z_loss, weight_sum = compute_weighted_cross_entropy(
        logits=logits,
        targets=targets,
        weights=weights,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        loss_normalizing_factor=loss_normalizing_factor,
        compute_dtype=compute_dtype,
    )

    predictions = jnp.argmax(logits, axis=-1)
    correct = (predictions == targets).astype(compute_dtype)
    if weights is None:
        accuracy = jnp.mean(correct)
    else:
        w = weights.astype(compute_dtype)
        denom = jnp.maximum(jnp.sum(w), 1e-8)
        accuracy = jnp.sum(correct * w) / denom
    return total_loss, total_z_loss, weight_sum, accuracy


def cross_entropy_loss_and_accuracy(
    source,
    target,
    valid=None,
    compute_dtype: jnp.dtype = jnp.float32,
):
    """Compute cross-entropy loss and accuracy with optional masking.

    This is a simple and efficient implementation for computing both loss and
    accuracy in a single pass through the data. Unlike other functions in this
    module, it uses a straightforward approach without label smoothing or z-loss,
    making it suitable for simple use cases or validation.

    Args:
        source: Model logits with shape [..., num_classes].
        target: Integer labels with shape [...].
        valid: Optional boolean mask with shape [...] indicating valid positions.
            If None, all positions are considered valid.
        compute_dtype: Data type for computation.

    Returns:
        Tuple of (loss, accuracy) as scalar values. Loss is the mean negative
        log-likelihood over valid positions. Accuracy is the fraction of valid
        positions where the predicted class matches the target.

    Example:
        >>> logits = jnp.randn(8, 128, 32000)
        >>> targets = jnp.randint(0, 32000, (8, 128))
        >>> mask = targets != -100  # Mask padding
        >>> loss, acc = cross_entropy_loss_and_accuracy(logits, targets, valid=mask)
    """
    source = source.astype(compute_dtype)
    if valid is None:
        valid = jnp.ones_like(target, dtype=compute_dtype)
    else:
        valid = valid.astype(compute_dtype)

    lse = jax.scipy.special.logsumexp(source, axis=-1)
    logit_y = jnp.take_along_axis(source, target[..., None], axis=-1)[..., 0]
    nll = (lse - logit_y) * valid
    weight_sum = jnp.maximum(jnp.sum(valid), 1e-8)
    loss = jnp.sum(nll) / weight_sum

    preds = jnp.argmax(source, axis=-1)
    correct = (preds == target).astype(compute_dtype) * valid
    accuracy = jnp.sum(correct) / weight_sum
    return loss, accuracy


def convert_special_loss_normalizing_factor_to_enum(x: str) -> SLNF:
    """Convert a string to SpecialLossNormalizingFactor enum.

    This utility function converts string representations of loss normalizing
    factors to their corresponding enum values, enabling configuration from
    strings (e.g., from config files or command line arguments).

    Args:
        x: String representation of the enum value. Case-insensitive. Valid
            values are: "NUM_REAL_TARGET_TOKENS", "NUM_TOTAL_TARGET_TOKENS",
            "AVERAGE_PER_SEQUENCE", "NO_WEIGHT_NUM_REAL_TARGET_TOKENS".

    Returns:
        The corresponding SpecialLossNormalizingFactor enum value.

    Raises:
        ValueError: If the string does not match any valid enum value.

    Example:
        >>> factor = convert_special_loss_normalizing_factor_to_enum("NUM_REAL_TARGET_TOKENS")
        >>> factor == SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
        True
    """
    x = x.upper()
    if x == "NUM_REAL_TARGET_TOKENS":
        return SLNF.NUM_REAL_TARGET_TOKENS
    if x == "NUM_TOTAL_TARGET_TOKENS":
        return SLNF.NUM_TOTAL_TARGET_TOKENS
    if x == "AVERAGE_PER_SEQUENCE":
        return SLNF.AVERAGE_PER_SEQUENCE
    if x == "NO_WEIGHT_NUM_REAL_TARGET_TOKENS":
        return SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS
    raise ValueError(f'Could not convert string "{x}" to SpecialLossNormalizingFactor')


@jax.vmap
def _sum_weights_per_segment(
    positions: Array,
    segment_ids: Array,
    weights: Array,
) -> Array:
    """Sum weights per segment for packed sequence normalization.

    This function is used for handling packed sequences where multiple sequences
    are concatenated together. It computes the sum of weights for each segment
    to enable per-sequence normalization when using AVERAGE_PER_SEQUENCE
    loss normalization.

    The function operates by:
    1. Identifying segment boundaries from position resets
    2. Computing cumulative weights within each segment
    3. Propagating the final cumulative weight back to all positions

    Args:
        positions: Position indices within each segment with shape (seq_len,).
            A value of 0 indicates the start of a new segment.
        segment_ids: Segment identifiers with shape (seq_len,). Non-zero values
            indicate valid positions; zero indicates padding.
        weights: Weights to sum per segment with shape (seq_len,).

    Returns:
        Array with shape (seq_len,) containing the total weight for the segment
        that each position belongs to. This can be used to normalize each
        position's contribution by its segment's total weight.

    Note:
        This function is vmapped over the batch dimension, so it operates on
        individual sequences.
    """

    def _repeat_last_nonnegative(xs, reverse=False):
        """Propagate the last non-zero value through zeros in the array.

        This helper function replaces zeros with the most recent non-zero value,
        either scanning forward (default) or backward (reverse=True).

        Args:
            xs: Input array with shape (seq_len,).
            reverse: If True, scan from right to left.

        Returns:
            Array with zeros replaced by propagated non-zero values.
        """

        def fn(prev, x):
            """Scan step that propagates the previous non-zero value forward.

            Args:
                prev: Carry value from the previous scan step.
                x: Current element of the scanned array.

            Returns:
                Tuple of (new_carry, output) where zeros in ``x`` are
                replaced by ``prev``.
            """
            y = jnp.where(x == 0, prev, x)
            return y, y

        return jax.lax.scan(fn, jnp.zeros_like(xs[0]), xs, reverse=reverse)[1]

    start_positions = positions == 0
    final_positions = jnp.concatenate([start_positions[1:], jnp.ones(1)])
    final_positions *= segment_ids != 0
    final_cumulative_weights = final_positions * jnp.cumsum(weights)
    final_total_weights = jnp.concatenate(
        [
            final_cumulative_weights[0:1],
            jnp.diff(_repeat_last_nonnegative(final_cumulative_weights)),
        ]
    )
    normalizer = _repeat_last_nonnegative(final_total_weights, reverse=True)
    return normalizer


def get_factor_and_weight(
    loss_normalizing_factor: FACTOR_TYPE,
    batch: collections.abc.Mapping[str, Array],
    compute_dtype: jnp.dtype = jnp.float32,
) -> tuple[float | Array | None, Array | None]:
    """Get loss normalizing factor and weights from batch data.

    This function resolves dynamic loss normalization factors based on batch
    properties. It handles both constant factors and special enum-based factors
    that depend on the actual batch content (e.g., number of non-padding tokens).

    Args:
        loss_normalizing_factor: The normalization strategy to use. Can be:
            - None: Return (None, existing weights or None)
            - A float/int constant: Return (constant, existing weights)
            - A string or SLNF enum: Compute dynamic factor from batch
        batch: Dictionary containing batch data. Expected keys:
            - "decoder_target_tokens": Target tokens for computing masks
            - "decoder_loss_weights" (optional): Pre-computed loss weights
            - "decoder_segment_ids" (optional): Segment IDs for packed sequences
            - "decoder_positions" (optional): Position indices for packed sequences
        compute_dtype: Data type for computed weights.

    Returns:
        Tuple of (normalizing_factor, loss_weights):
            - normalizing_factor: Float value to divide the loss by, or None.
            - loss_weights: Array of per-token weights, or None.

    Raises:
        ValueError: If an unsupported loss_normalizing_factor value is provided.

    Example:
        >>> batch = {
        ...     "decoder_target_tokens": jnp.array([[1, 2, 0, 0], [1, 2, 3, 0]]),
        ...     "decoder_loss_weights": jnp.array([[1, 1, 0, 0], [1, 1, 1, 0]]),
        ... }
        >>> factor, weights = get_factor_and_weight("NUM_REAL_TARGET_TOKENS", batch)
        >>> # factor is the sum of non-zero weights
    """
    loss_weights = batch.get("decoder_loss_weights", None)
    if loss_normalizing_factor is None or not isinstance(loss_normalizing_factor, (str, SLNF)):
        return loss_normalizing_factor, loss_weights

    if isinstance(loss_normalizing_factor, str):
        loss_normalizing_factor = convert_special_loss_normalizing_factor_to_enum(loss_normalizing_factor)

    if loss_weights is None:
        loss_weights = jnp.asarray(batch["decoder_target_tokens"] > 0, compute_dtype)

    output_normalizing_factor = None
    if loss_normalizing_factor == SLNF.NUM_REAL_TARGET_TOKENS:
        output_normalizing_factor = jnp.sum(loss_weights)
    elif loss_normalizing_factor == SLNF.NUM_TOTAL_TARGET_TOKENS:
        output_normalizing_factor = jnp.prod(batch["decoder_target_tokens"].shape)
    elif loss_normalizing_factor == SLNF.AVERAGE_PER_SEQUENCE:
        if "decoder_segment_ids" in batch:
            norm_vec = _sum_weights_per_segment(
                batch["decoder_positions"],
                batch["decoder_segment_ids"],
                loss_weights,
            )
        else:
            norm_vec = jnp.sum(loss_weights, axis=-1, keepdims=True)
        loss_weights = jnp.nan_to_num(loss_weights / norm_vec, nan=0, posinf=0, neginf=0)
        output_normalizing_factor = jnp.sum(loss_weights)
    else:
        raise ValueError(f"Unsupported value of loss_normalizing_factor: {loss_normalizing_factor}")

    return output_normalizing_factor, loss_weights


def auxiliary_load_balancing_loss_func(
    gate_logits: Array | tuple[Array, ...] | None,
    num_experts: int | None,
    top_k: int,
    attention_mask: Array | None = None,
    compute_dtype: jnp.dtype = jnp.float32,
) -> jax.Array | int:
    """Compute auxiliary load balancing loss for Mixture-of-Experts models.

    This function implements the load balancing loss from the Switch Transformer
    paper (https://arxiv.org/abs/2101.03961), equations (4)-(6). The loss
    penalizes unbalanced routing between experts, encouraging the model to
    utilize all experts equally.

    The loss is computed as:
        loss = num_experts * sum_i(f_i * P_i)

    where:
        - f_i is the fraction of tokens routed to expert i
        - P_i is the fraction of router probability assigned to expert i

    This loss is minimized when routing is perfectly balanced (f_i = P_i = 1/num_experts).

    Args:
        gate_logits: Router logits from the MoE layers. Can be:
            - A tuple/list of arrays, one per layer, each with shape
              [batch_size * sequence_length, num_experts]
            - A single stacked array with shape
              [num_layers * batch_size * sequence_length, num_experts]
            - None, in which case 0 is returned
        num_experts: Total number of experts in the MoE layer. Required if
            gate_logits is not None.
        top_k: Number of experts selected per token. This affects how the
            expert mask is computed.
        attention_mask: Optional attention mask with shape [batch_size, sequence_length].
            If provided, masked positions are excluded from the loss computation.
        compute_dtype: Data type for intermediate computations.

    Returns:
        The auxiliary load balancing loss as a scalar JAX array, or 0 if
        gate_logits is None.

    Raises:
        ValueError: If num_experts is None when gate_logits is provided, or if
            attention_mask has invalid shape.
        TypeError: If gate_logits is not a valid type.

    Example:
        >>> # Single layer gate logits
        >>> gate_logits = jnp.randn(32 * 128, 8)  # batch*seq, num_experts
        >>> loss = auxiliary_load_balancing_loss_func(gate_logits, num_experts=8, top_k=2)
        >>>
        >>> # Multiple layers
        >>> gate_logits = [jnp.randn(32 * 128, 8) for _ in range(24)]
        >>> loss = auxiliary_load_balancing_loss_func(gate_logits, num_experts=8, top_k=2)
    """
    if gate_logits is None:
        return 0
    if num_experts is None:
        raise ValueError("num_experts must be specified if gate_logits is provided.")

    # If gate_logits is a tuple or list, concatenate them.
    # Assumes individual layer logits are already on the correct device.
    if isinstance(gate_logits, (tuple, list)):
        # Ensure all logits are JAX arrays before concatenation
        gate_logits_list = [jnp.asarray(layer_gate.reshape(-1, num_experts)) for layer_gate in gate_logits]
        concatenated_gate_logits = jnp.concatenate(gate_logits_list, axis=0)
    elif isinstance(gate_logits, jnp.ndarray):
        concatenated_gate_logits = gate_logits
    else:
        raise TypeError(f"gate_logits must be a JAX array, tuple/list of JAX arrays, or None. Got {type(gate_logits)}")

    routing_weights = jax.nn.softmax(concatenated_gate_logits, axis=-1)
    _, selected_experts = jax.lax.top_k(routing_weights, k=top_k)
    expert_mask = jax.nn.one_hot(selected_experts, num_classes=num_experts, dtype=compute_dtype)

    if attention_mask is None:
        tokens_per_expert = jnp.mean(expert_mask, axis=0)
        router_prob_per_expert = jnp.mean(routing_weights, axis=0)
    else:
        attention_mask = jnp.asarray(attention_mask)
        if attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [batch_size, sequence_length], got {attention_mask.shape}")

        batch_size, sequence_length = attention_mask.shape
        total_tokens_per_layer = batch_size * sequence_length
        num_effective_tokens = concatenated_gate_logits.shape[0]

        if num_effective_tokens % total_tokens_per_layer != 0:
            raise ValueError(
                f"Total tokens in gate_logits ({num_effective_tokens}) is not divisible by "
                f"batch_size*sequence_length ({total_tokens_per_layer}). Ensure gate_logits are correctly concatenated."
            )

        num_hidden_layers = num_effective_tokens // total_tokens_per_layer
        mask_expanded = jnp.expand_dims(attention_mask, axis=(0, 3, 4))
        target_mask_shape = (
            num_hidden_layers,
            batch_size,
            sequence_length,
            top_k,
            num_experts,
        )
        expert_attention_mask_broadcast = jnp.broadcast_to(mask_expanded, target_mask_shape)
        expert_attention_mask = jnp.reshape(expert_attention_mask_broadcast, (-1, top_k, num_experts))
        masked_expert_contributions = expert_mask * expert_attention_mask
        tokens_per_expert_numerator = jnp.sum(masked_expert_contributions, axis=0)
        tokens_per_expert_denominator = jnp.sum(expert_attention_mask, axis=0)
        tokens_per_expert_denominator = jnp.where(tokens_per_expert_denominator == 0, 1.0, tokens_per_expert_denominator)
        tokens_per_expert = tokens_per_expert_numerator / tokens_per_expert_denominator
        mask_expanded_router = jnp.expand_dims(attention_mask, axis=(0, 3))

        target_router_mask_shape = (
            num_hidden_layers,
            batch_size,
            sequence_length,
            num_experts,
        )
        router_attention_mask_broadcast = jnp.broadcast_to(mask_expanded_router, target_router_mask_shape)
        router_per_expert_attention_mask = jnp.reshape(router_attention_mask_broadcast, (-1, num_experts))

        masked_routing_weights = routing_weights * router_per_expert_attention_mask

        router_prob_numerator = jnp.sum(masked_routing_weights, axis=0)
        router_prob_denominator = jnp.sum(router_per_expert_attention_mask, axis=0)
        router_prob_denominator = jnp.where(router_prob_denominator == 0, 1.0, router_prob_denominator)
        router_prob_per_expert = router_prob_numerator / router_prob_denominator
    router_prob_per_expert_expanded = jnp.expand_dims(router_prob_per_expert, axis=0)
    per_expert_loss_terms = tokens_per_expert * router_prob_per_expert_expanded
    overall_loss = jnp.sum(per_expert_loss_terms)
    final_loss = overall_loss * num_experts

    return jnp.asarray(final_loss, dtype=jnp.float32)


def fixed_cross_entropy(
    source: jax.Array | None,
    target: jax.Array | None,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    num_items_in_batch: int | None = None,
    batch: collections.abc.Mapping[str, Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """Compute cross-entropy loss with comprehensive configuration options.

    This is the main entry point for cross-entropy loss computation in EasyDeL.
    It supports multiple computation strategies (chunked, blockwise, standard),
    various normalization modes, and additional regularization terms.

    The function automatically selects the appropriate computation strategy
    based on the LossConfig settings, optimizing for memory efficiency when
    chunking is enabled.

    Args:
        source: Model logits with shape (batch_size, num_classes) or
            (batch_size * seq_len, num_classes) or (batch_size, seq_len, num_classes).
        target: Target labels with shape matching source's leading dimensions.
            Must be integer indices for sparse cross-entropy.
        attention_mask: Optional boolean mask indicating valid positions. If None,
            positions where target != ignore_index are considered valid.
        config: LossConfig object specifying loss computation parameters. If None,
            uses default configuration.
        num_items_in_batch: Optional number of items for batch-level normalization.
            If provided, divides the final loss by this value.
        batch: Optional batch dictionary for dynamic loss normalization. Used when
            loss_normalizing_factor is a SpecialLossNormalizingFactor.
        **kwargs: Additional keyword arguments (unused, for API compatibility).

    Returns:
        LossMetrics object containing:
            - loss: The computed loss value
            - z_loss: Z-loss regularization term (if z_loss > 0 in config)
            - weight_sum: Sum of loss weights
            - accuracy: Prediction accuracy

    Raises:
        ValueError: If source or target is None.

    Example:
        >>> config = LossConfig(
        ...     label_smoothing=0.1,
        ...     z_loss=1e-4,
        ...     loss_normalizing_factor="NUM_REAL_TARGET_TOKENS"
        ... )
        >>> metrics = fixed_cross_entropy(logits, targets, config=config)
        >>> print(f"Loss: {metrics.loss}, Accuracy: {metrics.accuracy}")
    """
    if config is None:
        config = LossConfig()
    if source is None or target is None:
        raise ValueError("Logits and labels cannot be None")
    compute_dtype = (
        jnp.float32
        if config.compute_dtype == "fp32"
        else (jnp.bfloat16 if config.compute_dtype == "bf16" else source.dtype)
    )
    mask = attention_mask if attention_mask is not None else (target != config.ignore_index)
    loss_factor = config.loss_normalizing_factor

    if config.reduction is not None:
        loss, norm = dynamic_cross_entropy_loss(
            logits=source,
            targets=target,
            weight=mask.astype(compute_dtype),
            ignore_index=config.ignore_index,
            reduction=config.reduction,
            label_smoothing=config.label_smoothing,
            compute_dtype=compute_dtype,
        )
        total_z_loss = jnp.array(0.0, compute_dtype)
        weight_sum = norm
        preds = jnp.argmax(source, axis=-1)
        correct = (preds == target).astype(compute_dtype) * mask.astype(compute_dtype)
        accuracy = jnp.sum(correct) / jnp.maximum(norm, 1e-8)

    elif (
        loss_factor == SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS
        or loss_factor == SLNF.NO_WEIGHT_NUM_REAL_TARGET_TOKENS.value
    ):
        loss, accuracy = cross_entropy_loss_and_accuracy(
            source.astype(compute_dtype), target, mask.astype(compute_dtype)
        )
        total_z_loss = jnp.array(0.0, compute_dtype)
        weight_sum = jnp.sum(mask.astype(compute_dtype))

    else:
        if batch is None:
            lf = config.loss_normalizing_factor
            if isinstance(lf, str):
                lf = convert_special_loss_normalizing_factor_to_enum(lf)
            batch = (
                {"decoder_target_tokens": target, "decoder_loss_weights": mask.astype(compute_dtype)}
                if lf == SLNF.NUM_REAL_TARGET_TOKENS
                else {}
            )

        loss_normalizing_factor, loss_weights = get_factor_and_weight(config.loss_normalizing_factor, batch)

        use_chunk_vocab = config.chunk_vocab_size is not None
        use_chunk_tokens = config.chunk_token_size is not None
        use_block_size = config.chunk_block_size is not None

        if use_chunk_vocab:
            total_loss, total_z_loss, weight_sum, accuracy = sparse_cross_entropy_chunked_vocab(
                source,
                target,
                weights=loss_weights,  # <- use loss_weights, not raw mask
                ignore_index=config.ignore_index,
                label_smoothing=config.label_smoothing,
                z_loss=config.z_loss,
                reduction="sum",
                chunk_size=config.chunk_vocab_size,
                compute_dtype=compute_dtype,
            )
        elif use_chunk_tokens:
            total_loss, total_z_loss, weight_sum, accuracy = sparse_cross_entropy_chunked_tokens(
                source,
                target,
                weights=loss_weights,
                ignore_index=config.ignore_index,
                label_smoothing=config.label_smoothing,
                z_loss=config.z_loss,
                reduction="sum",
                token_chunk_size=config.chunk_token_size,
                compute_dtype=compute_dtype,
            )
        elif use_block_size:
            total_loss, total_z_loss, weight_sum, accuracy = cross_entropy_blockwise_logits(
                logits=source,
                targets=target,
                weights=loss_weights,
                ignore_index=config.ignore_index,
                label_smoothing=config.label_smoothing,
                z_loss=config.z_loss,
                block_size=int(config.chunk_block_size),
                dtype=compute_dtype,
            )
            if loss_normalizing_factor is not None:
                total_loss = total_loss / loss_normalizing_factor
                total_z_loss = total_z_loss / loss_normalizing_factor
        else:
            total_loss, total_z_loss, weight_sum, accuracy = compute_weighted_cross_entropy_and_accuracy(
                logits=source,
                targets=target,
                weights=loss_weights,
                label_smoothing=config.label_smoothing,
                z_loss=config.z_loss,
                loss_normalizing_factor=loss_normalizing_factor,
                compute_dtype=compute_dtype,
            )

        # Apply loss_normalizing_factor in chunked paths (dense path already applied it)
        if (use_chunk_vocab or use_chunk_tokens) and (loss_normalizing_factor is not None):
            total_loss = total_loss / loss_normalizing_factor
            total_z_loss = total_z_loss / loss_normalizing_factor

        # Optional post-normalization
        if num_items_in_batch is not None:
            loss = total_loss / num_items_in_batch
        elif config.divide_weight_sum:
            loss = total_loss / jnp.maximum(weight_sum, 1e-8)
        else:
            loss = total_loss

    return LossMetrics(loss=loss, z_loss=total_z_loss, weight_sum=weight_sum, accuracy=accuracy)


def resolve_causal_lm_chunk_token_size(
    hidden_states: jax.Array,
    vocab_size: int,
    config: LossConfig | None = None,
) -> int:
    """Choose an optimal token-dimension chunk size for chunked LM-head loss.

    When computing causal-LM loss without materializing the full
    ``[B, T, V]`` logits tensor, the token dimension is split into chunks
    of this size. Each chunk independently projects hidden states through
    the LM head and computes cross-entropy, keeping peak memory bounded
    to roughly ``B * chunk * V * dtype_bytes``.

    The heuristic targets a ~16 GiB temporary logits buffer, rounds down
    to a power of two, and clamps to ``[1, 4096]`` without exceeding
    the computed budget. If
    ``config.chunk_token_size`` is already set, that value is used
    directly (clamped to the actual sequence length).

    Args:
        hidden_states: Model hidden states with shape ``[B, T, H]``.
        vocab_size: Vocabulary size (V).
        config: Optional ``LossConfig``. If its ``chunk_token_size`` field
            is set, that value takes priority.

    Returns:
        Token chunk size as a positive integer, never exceeding ``T``.
    """
    if config is not None and config.chunk_token_size is not None:
        return max(1, min(int(config.chunk_token_size), int(hidden_states.shape[1])))

    batch_size = max(1, int(hidden_states.shape[0]))
    seq_len = max(1, int(hidden_states.shape[1]))
    vocab_size = max(1, int(vocab_size))

    # The outer chunked LM-head path materializes [B, chunk, V] logits before
    # fixed_cross_entropy consumes them, so the chunk must not exceed the target
    # memory budget. Use the widest dtype implied by the hidden states and the
    # requested loss compute dtype to avoid underestimating mixed-precision runs.
    projection_dtype = (
        jnp.float32
        if config is not None and config.compute_dtype == "fp32"
        else (jnp.bfloat16 if config is not None and config.compute_dtype == "bf16" else hidden_states.dtype)
    )
    dtype_bytes = max(1, hidden_states.dtype.itemsize, jnp.dtype(projection_dtype).itemsize)
    target_bytes = 16 * 1024 * 1024 * 1024
    raw_chunk = max(1, target_bytes // max(1, batch_size * vocab_size * dtype_bytes))
    chunk = 1 << max(0, raw_chunk.bit_length() - 1)
    chunk = min(4096, chunk)
    return min(chunk, seq_len)


def causal_lm_loss_chunked_lm_head(
    hidden_states: jax.Array | None,
    labels: jax.Array | None,
    lm_head_fn: tp.Callable[[jax.Array], jax.Array],
    *,
    vocab_size: int,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    num_items_in_batch: int | None = None,
    batch: collections.abc.Mapping[str, Array] | None = None,
    logit_cap_fn: tp.Callable[[jax.Array], jax.Array] | None = None,
    token_chunk_size: int | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """Compute causal-LM cross-entropy by projecting hidden states through the
    LM head in token-dimension chunks, avoiding the full ``[B, T, V]`` logit
    materialization.

    The token sequence is split into equal-sized chunks (padded with
    ``ignore_index`` labels if necessary). For each chunk a ``jax.lax.scan``
    iteration:

    1. Projects the chunk's hidden states to logits via ``lm_head_fn``.
    2. Optionally caps logits with ``logit_cap_fn``.
    3. Computes ``fixed_cross_entropy`` on the chunk.
    4. Accumulates loss, z-loss, weight sum, and correct-token counts.

    Each chunk body is wrapped with ``jax.checkpoint`` so that backward
    recomputes logits per chunk instead of storing all of them.

    Args:
        hidden_states: Model hidden states, shape ``[B, T, H]``.
        labels: Target token IDs, shape ``[B, T]``.
        lm_head_fn: Callable that maps ``[B, chunk, H] -> [B, chunk, V]``
            (the model's LM-head projection).
        vocab_size: Vocabulary size, used for automatic chunk sizing.
        attention_mask: Optional ``[B, T]`` mask (1 = valid, 0 = padding).
        config: ``LossConfig`` controlling shifting, smoothing, z-loss,
            normalizing factor, etc. Defaults to ``LossConfig()``.
        num_items_in_batch: If set, the total loss is divided by this value
            after the global normalizing factor is applied.
        batch: Full training batch mapping. ``decoder_loss_weights`` and
            ``decoder_target_tokens`` are extracted if present.
        logit_cap_fn: Optional callable applied to logits before loss
            (e.g. ``tanh`` capping for Gemma-2 style models).
        token_chunk_size: Explicit chunk size. If ``None``, chosen
            automatically via ``resolve_causal_lm_chunk_token_size``.
        **kwargs: Forwarded to ``fixed_cross_entropy`` for each chunk.

    Returns:
        ``LossMetrics`` with aggregated ``loss``, ``z_loss``,
        ``weight_sum``, and ``accuracy`` across all chunks.

    Raises:
        ValueError: If ``hidden_states`` or ``labels`` is ``None``, or if
            their shapes are not ``[B, T, H]`` and ``[B, T]`` respectively.
    """
    if hidden_states is None or labels is None:
        raise ValueError("hidden_states and labels cannot be None")
    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "causal_lm_loss_chunked_lm_head expects hidden_states [B, T, H] and labels [B, T], "
            f"got {hidden_states.shape} and {labels.shape}."
        )

    if config is None:
        config = LossConfig()

    shift_attn_m = attention_mask
    if config.shift_tokens:
        shift_hidden_states = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        if attention_mask is not None:
            shift_attn_m = attention_mask[:, 1:]
    else:
        shift_hidden_states = hidden_states
        shift_labels = labels

    if shift_hidden_states.shape[1] == 0:
        zero = jnp.array(0.0, dtype=shift_hidden_states.dtype)
        return LossMetrics(loss=zero, z_loss=zero, weight_sum=zero, accuracy=zero)

    compute_dtype = (
        jnp.float32
        if config.compute_dtype == "fp32"
        else (jnp.bfloat16 if config.compute_dtype == "bf16" else shift_hidden_states.dtype)
    )
    global_loss_batch = dict(batch or {})
    if "decoder_target_tokens" not in global_loss_batch:
        global_loss_batch["decoder_target_tokens"] = shift_labels
    else:
        global_loss_batch["decoder_target_tokens"] = jnp.asarray(global_loss_batch["decoder_target_tokens"])
    if "decoder_loss_weights" in global_loss_batch:
        shift_loss_weights = jnp.asarray(global_loss_batch["decoder_loss_weights"], compute_dtype)
        if config.shift_tokens and shift_loss_weights.shape == labels.shape:
            shift_loss_weights = shift_loss_weights[:, 1:]
        elif shift_loss_weights.shape != shift_labels.shape:
            raise ValueError(
                "decoder_loss_weights must match labels or shifted labels, "
                f"got {shift_loss_weights.shape}, labels {labels.shape}, shifted {shift_labels.shape}."
            )
        global_loss_batch["decoder_loss_weights"] = shift_loss_weights
    else:
        if shift_attn_m is not None:
            global_loss_batch["decoder_loss_weights"] = shift_attn_m.astype(compute_dtype)
        else:
            global_loss_batch["decoder_loss_weights"] = (shift_labels != config.ignore_index).astype(compute_dtype)

    global_loss_factor, _ = get_factor_and_weight(
        config.loss_normalizing_factor,
        global_loss_batch,
        compute_dtype=compute_dtype,
    )
    # The outer LM-head chunking path already materializes [B, chunk, V] logits
    # per scan step. Inner vocab/block CE chunking cannot reduce that dominant
    # allocation, but it does add a second chunking loop and can significantly
    # slow training. Keep token chunking at the outer projection boundary and
    # run the per-chunk CE densely.
    chunk_loss_config = dataclasses.replace(
        config,
        loss_normalizing_factor=None,
        chunk_vocab_size=None,
        chunk_token_size=None,
        chunk_block_size=None,
    )

    chunk_size = token_chunk_size or resolve_causal_lm_chunk_token_size(
        hidden_states=shift_hidden_states,
        vocab_size=vocab_size,
        config=config,
    )
    # Clamp against the auto-resolver's budget even when the user provides
    # an explicit chunk size — a manually-specified 4096 with a 248k vocab
    # produces ~30 GB logit tensors that OOM during backward recomputation.
    safe_chunk = resolve_causal_lm_chunk_token_size(
        hidden_states=shift_hidden_states,
        vocab_size=vocab_size,
        config=config,
    )
    chunk_size = min(chunk_size, safe_chunk)

    _batch_size, seq_len, _hidden_dim = shift_hidden_states.shape
    pad_len = (-seq_len) % chunk_size

    if pad_len:
        shift_hidden_states = jnp.pad(shift_hidden_states, ((0, 0), (0, pad_len), (0, 0)))
        shift_labels = jnp.pad(shift_labels, ((0, 0), (0, pad_len)), constant_values=config.ignore_index)
        if shift_attn_m is not None:
            shift_attn_m = jnp.pad(shift_attn_m, ((0, 0), (0, pad_len)))

    padded_seq_len = shift_hidden_states.shape[1]
    num_chunks = padded_seq_len // chunk_size
    loss_weights = global_loss_batch["decoder_loss_weights"]
    if pad_len:
        loss_weights = jnp.pad(loss_weights, ((0, 0), (0, pad_len)))

    def _chunk_loss(chunk_hidden_states, chunk_labels, chunk_attention_mask, chunk_loss_weights):
        """Compute cross-entropy loss for a single chunk of the sequence.

        Projects hidden states through the LM head, optionally caps logits,
        and computes the fixed cross-entropy loss with accuracy tracking.

        Args:
            chunk_hidden_states: Hidden states for this chunk, shape
                ``(batch, chunk_size, hidden_dim)``.
            chunk_labels: Target token IDs, shape ``(batch, chunk_size)``.
            chunk_attention_mask: Attention mask or None, shape
                ``(batch, chunk_size)``.
            chunk_loss_weights: Per-token loss weights, shape
                ``(batch, chunk_size)``.

        Returns:
            Tuple of (loss, z_loss, weight_sum, correct_count).
        """
        logits = lm_head_fn(chunk_hidden_states)
        if logit_cap_fn is not None:
            logits = logit_cap_fn(logits)
        chunk_loss_batch = {
            "decoder_target_tokens": chunk_labels,
            "decoder_loss_weights": chunk_loss_weights.astype(compute_dtype),
        }
        chunk_metrics = fixed_cross_entropy(
            source=logits,
            target=chunk_labels,
            attention_mask=chunk_attention_mask,
            config=chunk_loss_config,
            num_items_in_batch=None,
            batch=chunk_loss_batch,
            **kwargs,
        )
        correct = chunk_metrics.accuracy.astype(chunk_metrics.weight_sum.dtype) * chunk_metrics.weight_sum
        return chunk_metrics.loss, chunk_metrics.z_loss, chunk_metrics.weight_sum, correct

    _chunk_loss = jax.checkpoint(_chunk_loss, prevent_cse=False)

    def _accumulate_chunk(i: int, carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array]):
        """Accumulate loss metrics from the i-th chunk into running totals.

        Slices the i-th chunk from the padded sequence, computes its loss
        via ``_chunk_loss``, and adds the results to the carry accumulators.

        Args:
            i: Zero-based chunk index.
            carry: Running totals of (loss, z_loss, weight_sum, correct_count).

        Returns:
            Updated carry tuple with this chunk's contributions added.
        """
        start = i * chunk_size
        chunk_hidden_states = lax.dynamic_slice_in_dim(shift_hidden_states, start, chunk_size, axis=1)
        chunk_labels = lax.dynamic_slice_in_dim(shift_labels, start, chunk_size, axis=1)
        chunk_attention_mask = (
            lax.dynamic_slice_in_dim(shift_attn_m, start, chunk_size, axis=1) if shift_attn_m is not None else None
        )
        chunk_loss_weights = lax.dynamic_slice_in_dim(loss_weights, start, chunk_size, axis=1)
        chunk_loss, chunk_z_loss, chunk_weight_sum, chunk_correct = _chunk_loss(
            chunk_hidden_states,
            chunk_labels,
            chunk_attention_mask,
            chunk_loss_weights,
        )
        return (
            carry[0] + chunk_loss,
            carry[1] + chunk_z_loss,
            carry[2] + chunk_weight_sum,
            carry[3] + chunk_correct,
        )

    zero = jnp.array(0.0, dtype=compute_dtype)
    total_loss, total_z_loss, weight_sum, correct_sum = jax.lax.fori_loop(
        0,
        num_chunks,
        _accumulate_chunk,
        (zero, zero, zero, zero),
    )

    if global_loss_factor is not None:
        total_loss = total_loss / global_loss_factor
        total_z_loss = total_z_loss / global_loss_factor

    if num_items_in_batch is not None:
        total_loss = total_loss / num_items_in_batch

    accuracy = correct_sum / jnp.maximum(weight_sum, jnp.asarray(1e-8, dtype=weight_sum.dtype))
    return LossMetrics(loss=total_loss, z_loss=total_z_loss, weight_sum=weight_sum, accuracy=accuracy)


def _should_chunk_causal_lm_loss(
    *,
    module: tp.Any,
    labels: jax.Array | None,
    loss_config: LossConfig | None,
    token_chunk_size: int | None = None,
) -> bool:
    """Decide whether the chunked LM-head loss path should be activated.

    Returns ``True`` when all of the following hold:

    * ``labels`` is not ``None`` and has at least 2 dimensions.
    * The module supports the chunked protocol (has ``compute_lm_logits``
      and an ``apply_lm_head`` parameter in ``__call__``).
    * The loss config does not use features incompatible with the chunked
      path (custom ``reduction``, ``divide_weight_sum``, or per-sequence
      normalizing factors).
    * Either an explicit ``token_chunk_size`` / ``config.chunk_token_size``
      is set, *or* the projected logit tensor would exceed 2^28 elements
      (~1 GiB in fp32), making the unchunked path memory-prohibitive.

    Args:
        module: The EasyDeL model instance.
        labels: Target token IDs, or ``None``.
        loss_config: Loss configuration, or ``None`` for defaults.
        token_chunk_size: Explicit chunk size override, if any.

    Returns:
        ``True`` if the chunked path should be used.
    """
    if labels is None:
        return False
    if not _supports_chunked_causal_lm_forward(module):
        return False

    config = loss_config or LossConfig()
    if config.reduction is not None or config.divide_weight_sum:
        return False

    loss_factor = config.loss_normalizing_factor
    if hasattr(loss_factor, "name"):
        loss_factor = loss_factor.name
    if isinstance(loss_factor, str):
        loss_factor = loss_factor.upper()
    if loss_factor in {"NO_WEIGHT_NUM_REAL_TARGET_TOKENS", "AVERAGE_PER_SEQUENCE"}:
        return False

    model_chunk_size = module.config.lmhead_chunksize
    explicit_chunk = token_chunk_size is not None or config.chunk_token_size is not None or model_chunk_size is not None
    memory_tuned = explicit_chunk or config.chunk_block_size is not None or config.chunk_vocab_size is not None
    if not memory_tuned or labels.ndim < 2:
        return False

    effective_seq_len = max(1, int(labels.shape[1]) - (1 if config.shift_tokens else 0))
    projected_logit_elements = int(labels.shape[0]) * effective_seq_len * _resolve_module_vocab_size(module)
    return explicit_chunk or projected_logit_elements >= (1 << 28)


def _supports_chunked_causal_lm_forward(module: tp.Any) -> bool:
    """Check whether *module* exposes the API needed for chunked LM-head loss.

    The chunked path requires two things from the model:

    1. A ``compute_lm_logits`` method that projects hidden states to vocab
       logits (used as the per-chunk LM-head callable).
    2. An ``apply_lm_head`` parameter in ``__call__`` so the forward pass
       can be told to skip the built-in LM-head projection.

    Args:
        module: The model instance to inspect.

    Returns:
        ``True`` if both requirements are satisfied.
    """
    if not hasattr(module, "compute_lm_logits"):
        return False

    try:
        call_signature = inspect.signature(module.__call__)
    except (TypeError, ValueError):
        return False
    return "apply_lm_head" in call_signature.parameters


class CausalLMLossStrategy(BaseLossStrategy):
    """Planning-aware loss strategy for causal language modelling.

    This strategy has two operating modes chosen automatically in
    ``plan_forward``:

    **Dense mode** (default fallback) — the model runs its full forward
    pass including the LM-head projection. Loss is computed by the
    existing ``ForCausalLMLoss`` function on the materialised logits.

    **Chunked mode** — when the projected logit tensor would be very
    large (or the user explicitly sets ``chunk_token_size``), the plan
    tells the model to skip the LM-head (``apply_lm_head=False``).
    ``compute`` then calls ``causal_lm_loss_chunked_lm_head`` which
    projects hidden states through the LM head in small token-dimension
    chunks, keeping peak memory bounded.

    The strategy name is set to ``"ForCausalLMLoss"`` so that
    ``resolve_loss_strategy`` can match it by name.
    """

    __name__ = "ForCausalLMLoss"

    def plan_forward(
        self,
        *,
        module: tp.Any,
        labels: jax.Array | None,
        loss_config: LossConfig | None,
        batch: collections.abc.Mapping[str, Array],
        loss_kwargs: dict[str, tp.Any],
    ) -> LossForwardPlan:
        """Decide whether to request the chunked (headless) forward path.

        If ``_should_chunk_causal_lm_loss`` returns ``True``, the plan
        sets ``apply_lm_head=False`` so that the model returns raw hidden
        states instead of logits. Otherwise an empty plan is returned and
        the model runs normally.
        """
        del batch
        token_chunk_size = loss_kwargs.get("token_chunk_size")
        if token_chunk_size is None:
            token_chunk_size = module.config.lmhead_chunksize
        if _should_chunk_causal_lm_loss(
            module=module,
            labels=labels,
            loss_config=loss_config,
            token_chunk_size=token_chunk_size,
        ):
            return LossForwardPlan(forward_kwargs={"apply_lm_head": False})
        return LossForwardPlan()

    def compute(
        self,
        *,
        module: tp.Any,
        outputs: tp.Any,
        labels: jax.Array | None,
        loss_config: LossConfig | None,
        batch: collections.abc.Mapping[str, Array],
        loss_kwargs: dict[str, tp.Any],
        paxis: PartitionAxis | None,
        forward_plan: LossForwardPlan,
    ) -> LossMetrics:
        """Compute causal-LM loss, dispatching to dense or chunked path.

        If the forward plan left ``apply_lm_head`` as ``True`` (or
        absent), this delegates to the existing ``ForCausalLMLoss``
        function which expects pre-computed logits in *outputs*.

        Otherwise it extracts ``last_hidden_state`` from *outputs*,
        optionally runs ``module.prepare_lm_head_inputs``, and calls
        ``causal_lm_loss_chunked_lm_head`` to project + compute loss in
        token chunks.

        Raises:
            TypeError: If the chunked path was selected but the model did
                not return ``last_hidden_state``.
        """
        del paxis
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if forward_plan.forward_kwargs.get("apply_lm_head", True):
            return ForCausalLMLoss(
                labels=labels,
                config=loss_config,
                paxis=module.config.partition_axis,
                **loss_kwargs,
                **outputs,
                **batch,
            )
        if last_hidden_state is None:
            raise TypeError(
                f"{type(module).__name__} opted into chunked causal-LM loss but did not return "
                "`last_hidden_state` when `apply_lm_head=False`."
            )

        chunked_loss_kwargs = dict(loss_kwargs)
        if "num_items_in_batch" not in chunked_loss_kwargs and "num_items_in_batch" in batch:
            chunked_loss_kwargs["num_items_in_batch"] = batch["num_items_in_batch"]
        if "token_chunk_size" not in chunked_loss_kwargs or chunked_loss_kwargs["token_chunk_size"] is None:
            model_chunk_size = module.config.lmhead_chunksize
            if model_chunk_size is not None:
                chunked_loss_kwargs["token_chunk_size"] = model_chunk_size

        lm_head_inputs = (
            module.prepare_lm_head_inputs(last_hidden_state)
            if hasattr(module, "prepare_lm_head_inputs")
            else last_hidden_state
        )

        # Use the trace-safe projection bypass.  module.compute_lm_logits
        # goes through lm_head.__call__ which carries nn.remat — calling it
        # from inside fori_loop / jax.checkpoint triggers TraceContextError.
        _lm_head_fn = (
            module.make_lm_head_fn()
            if hasattr(module, "make_lm_head_fn")
            else module.compute_lm_logits
        )

        return causal_lm_loss_chunked_lm_head(
            hidden_states=lm_head_inputs,
            labels=labels,
            lm_head_fn=_lm_head_fn,
            vocab_size=_resolve_module_vocab_size(module),
            attention_mask=batch.get("attention_mask"),
            config=loss_config,
            batch=batch,
            **chunked_loss_kwargs,
        )


def resolve_loss_strategy(loss_fn: tp.Callable[..., LossMetrics]) -> BaseLossStrategy:
    """Wrap a loss callable in the appropriate ``BaseLossStrategy``.

    If *loss_fn* is the standard ``ForCausalLMLoss`` (including common
    ``functools.wraps`` wrappers), returns a ``CausalLMLossStrategy``
    that can optionally skip the dense LM-head projection for memory
    savings. For all other callables, returns a
    ``FunctionalLossStrategy`` that simply delegates ``compute`` to the
    original function.

    Args:
        loss_fn: Any callable that produces ``LossMetrics``.

    Returns:
        A ``BaseLossStrategy`` instance wrapping *loss_fn*.
    """
    if isinstance(loss_fn, BaseLossStrategy):
        return loss_fn
    try:
        unwrapped_loss_fn = inspect.unwrap(loss_fn)
    except (AttributeError, TypeError, ValueError):
        unwrapped_loss_fn = loss_fn
    if unwrapped_loss_fn is ForCausalLMLoss:
        return CausalLMLossStrategy()
    return FunctionalLossStrategy(loss_fn)


def ForCausalLMLoss(
    logits: jax.Array | None,
    labels: jax.Array | None,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    num_items_in_batch: int | None = None,
    batch: collections.abc.Mapping[str, Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """Compute loss for causal language modeling (next-token prediction).

    This function implements the standard causal language modeling loss where
    the model predicts the next token at each position. It handles the necessary
    shifting of logits and labels so that position i predicts token i+1.

    The function supports distributed training through optional sharding
    constraints on the logits and labels.

    Args:
        logits: Model output logits with shape (batch_size, seq_len, vocab_size).
        labels: Target token IDs with shape (batch_size, seq_len). Positions with
            value equal to ignore_index (default -100) are not included in loss.
        attention_mask: Optional mask with shape (batch_size, seq_len) indicating
            valid positions (1) vs padding (0).
        config: LossConfig specifying loss parameters. If None, uses defaults.
            Note: config.shift_tokens controls whether to shift logits/labels
            (default True for causal LM).
        paxis: Optional PartitionAxis for distributed training. If provided,
            applies sharding constraints to logits and labels.
        num_items_in_batch: Optional batch-level normalization factor.
        batch: Optional batch dictionary for dynamic normalization.
        **kwargs: Additional arguments passed to fixed_cross_entropy.

    Returns:
        LossMetrics containing loss, z_loss, weight_sum, and accuracy.

    Raises:
        ValueError: If logits or labels is None.

    Example:
        >>> # Standard causal LM loss
        >>> logits = model(input_ids)  # (batch, seq_len, vocab_size)
        >>> labels = input_ids.copy()
        >>> labels = labels.at[:, :-1].set(input_ids[:, 1:])
        >>> metrics = ForCausalLMLoss(logits, labels)
        >>>
        >>> # With custom config
        >>> config = LossConfig(label_smoothing=0.1, z_loss=1e-4)
        >>> metrics = ForCausalLMLoss(logits, labels, config=config)
    """
    if logits is None or labels is None:
        raise ValueError("Logits and labels cannot be None")
    if paxis is not None:
        logits = with_sharding_constraint(
            logits,
            PartitionSpec(
                paxis.batch_axis,
                paxis.sequence_axis,
                paxis.hidden_state_axis,
            ),
        )
        labels = with_sharding_constraint(
            labels,
            PartitionSpec(
                paxis.batch_axis,
                paxis.sequence_axis,
            ),
        )
    shift_attn_m = attention_mask
    if config is None:
        config = LossConfig()
    assert logits is not None and labels is not None
    if config.shift_tokens:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        if attention_mask is not None:
            shift_attn_m = attention_mask[:, 1:]
    else:
        shift_logits = logits
        shift_labels = labels

        if attention_mask is not None:
            shift_attn_m = attention_mask

    loss = fixed_cross_entropy(
        source=shift_logits,
        target=shift_labels,
        attention_mask=shift_attn_m,
        config=config,
        num_items_in_batch=num_items_in_batch,
        batch=batch,
        **kwargs,
    )
    return loss


def ForSequenceClassificationLoss(
    logits: jax.Array | None,
    labels: jax.Array | None,
    attention_mask: jax.Array | None = None,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    batch: collections.abc.Mapping[str, Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """Compute loss for sequence classification tasks.

    This function supports three types of sequence classification:
    - Regression: Mean squared error for continuous targets
    - Single-label classification: Cross-entropy for mutually exclusive classes
    - Multi-label classification: Sigmoid cross-entropy for independent labels

    The problem type is automatically inferred from num_labels and label dtype
    if not explicitly specified in the config.

    Args:
        logits: Model output logits with shape:
            - (batch_size, num_labels) for classification
            - (batch_size, 1) or (batch_size,) for regression
        labels: Target values with shape:
            - (batch_size,) integer labels for single-label classification
            - (batch_size, num_labels) multi-hot for multi-label classification
            - (batch_size,) or (batch_size, 1) floats for regression
        attention_mask: Optional mask. Note: For sequence classification, this
            is typically not used as the loss is per-sequence, not per-token.
        config: LossConfig with required num_labels field. problem_type is
            inferred if not set.
        paxis: Optional PartitionAxis for distributed training.
        batch: Optional batch dictionary for dynamic normalization.
        **kwargs: Additional arguments passed to fixed_cross_entropy.

    Returns:
        LossMetrics containing the computed loss. For regression and multi-label
        classification, only loss is populated. For single-label classification,
        includes accuracy and other metrics.

    Raises:
        ValueError: If logits or labels is None, num_labels is not set, or
            problem_type is invalid.

    Example:
        >>> # Single-label classification
        >>> config = LossConfig(num_labels=3)
        >>> logits = jnp.randn(8, 3)
        >>> labels = jnp.array([0, 1, 2, 0, 1, 2, 0, 1])
        >>> metrics = ForSequenceClassificationLoss(logits, labels, config=config)
        >>>
        >>> # Multi-label classification
        >>> config = LossConfig(num_labels=5, problem_type="multi_label_classification")
        >>> logits = jnp.randn(8, 5)
        >>> labels = jnp.array([[1, 0, 1, 0, 0], ...])  # Multi-hot
        >>> metrics = ForSequenceClassificationLoss(logits, labels, config=config)
    """

    if logits is None or labels is None:
        raise ValueError("Logits and labels cannot be None")
    if config is None:
        config = LossConfig()

    num_labels = config.num_labels if config.num_labels is not None else config.num_classification_labels
    if isinstance(num_labels, str):
        try:
            num_labels = int(num_labels)
        except ValueError as exc:
            raise ValueError(f"num_labels must be an int, got {num_labels!r}") from exc
    if num_labels is None:
        raise ValueError("num_labels must be set for sequence classification loss")
    problem_type = config.problem_type
    if problem_type is None and config.classification_problem_type is not None:
        problem_type = config.classification_problem_type
    if problem_type is None:
        if num_labels == 1:
            problem_type = "regression"
        elif num_labels > 1 and (labels.dtype == jnp.int32 or labels.dtype == jnp.int64):
            problem_type = "single_label_classification"
        else:
            problem_type = "multi_label_classification"

    if problem_type == "regression":
        loss = jnp.mean((logits.squeeze() - labels.squeeze()) ** 2)
    elif problem_type == "single_label_classification":
        # `attention_mask` is typically token-level (batch, seq_len) for encoder models.
        # Sequence classification loss is computed per-sequence, so masking should not be applied here.
        if attention_mask is not None and attention_mask.ndim != 1:
            attention_mask = None
        return fixed_cross_entropy(
            source=logits.reshape(-1, num_labels),
            target=labels.reshape(-1),
            attention_mask=attention_mask,
            config=config,
            batch=batch,
            **kwargs,
        )
    elif problem_type == "multi_label_classification":
        loss = jnp.mean(
            sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=labels,
                label_smoothing=config.label_smoothing,
            )
        )
    else:
        raise ValueError(f"Invalid problem_type: {problem_type}")
    return LossMetrics(loss=loss)


def ForQuestionAnsweringLoss(
    start_logits: jax.Array | None,
    end_logits: jax.Array | None,
    start_positions: jax.Array | None,
    end_positions: jax.Array | None,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    batch: collections.abc.Mapping[str, Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """Compute loss for extractive question answering (SQuAD-style).

    This function implements the standard extractive QA loss where the model
    predicts the start and end positions of the answer span within the context.
    The total loss is the average of the start and end position cross-entropy losses.

    Positions that exceed the sequence length are clipped and treated as the
    ignore index, handling cases where the answer is not in the context.

    Args:
        start_logits: Logits for start position prediction with shape
            (batch_size, seq_len).
        end_logits: Logits for end position prediction with shape
            (batch_size, seq_len).
        start_positions: Ground truth start positions with shape (batch_size,).
            Values should be in [0, seq_len) or a special value for "no answer".
        end_positions: Ground truth end positions with shape (batch_size,).
        config: LossConfig specifying loss parameters. The ignore_index is
            automatically set to seq_len for out-of-range positions.
        paxis: Optional PartitionAxis for distributed training.
        batch: Optional batch dictionary for dynamic normalization.
        **kwargs: Additional arguments passed to fixed_cross_entropy.

    Returns:
        LossMetrics containing averaged loss, z_loss, weight_sum, and accuracy
        from both start and end position predictions.

    Raises:
        ValueError: If any required input is None.

    Example:
        >>> # Batch of 4 samples, sequence length 384
        >>> start_logits = jnp.randn(4, 384)
        >>> end_logits = jnp.randn(4, 384)
        >>> start_positions = jnp.array([10, 50, 100, 200])
        >>> end_positions = jnp.array([15, 55, 110, 210])
        >>> metrics = ForQuestionAnsweringLoss(
        ...     start_logits, end_logits, start_positions, end_positions
        ... )
    """
    if start_logits is None or end_logits is None or start_positions is None or end_positions is None:
        raise ValueError("Logits and labels cannot be None")

    ignored_index = start_logits.shape[1]
    start_positions = jnp.clip(start_positions, 0, ignored_index)
    end_positions = jnp.clip(end_positions, 0, ignored_index)

    cfg = dataclasses.replace(config or LossConfig(), ignore_index=ignored_index)

    start_loss = fixed_cross_entropy(source=start_logits, target=start_positions, config=cfg, batch=batch, **kwargs)
    end_loss = fixed_cross_entropy(source=end_logits, target=end_positions, config=cfg, batch=batch, **kwargs)
    loss = (start_loss.loss + end_loss.loss) / 2
    accuracy = (start_loss.accuracy + end_loss.accuracy) / 2
    z_loss = (start_loss.z_loss + end_loss.z_loss) / 2
    weight_sum = (start_loss.weight_sum + end_loss.weight_sum) / 2
    return LossMetrics(
        loss=loss,
        accuracy=accuracy,
        z_loss=z_loss,
        weight_sum=weight_sum,
    )


def ForTokenClassification(
    logits: jax.Array | None,
    labels: jax.Array | None,
    config: LossConfig | None = None,
    paxis: PartitionAxis | None = None,
    batch: collections.abc.Mapping[str, Array] | None = None,
    **kwargs: tp.Any,
) -> LossMetrics:
    """Compute loss for token classification tasks.

    This function implements loss computation for token-level classification
    tasks such as Named Entity Recognition (NER), Part-of-Speech (POS) tagging,
    or any task where each token receives an independent label.

    Unlike sequence classification, the loss is computed for each token position
    independently, with support for masking special tokens (e.g., [CLS], [SEP],
    padding) via the ignore_index mechanism.

    Args:
        logits: Model output logits with shape (batch_size, seq_len, num_labels).
        labels: Target labels with shape (batch_size, seq_len). Positions with
            value equal to ignore_index are not included in the loss.
        config: LossConfig specifying loss parameters including num_labels and
            ignore_index.
        paxis: Optional PartitionAxis for distributed training.
        batch: Optional batch dictionary for dynamic normalization.
        **kwargs: Additional arguments passed to fixed_cross_entropy.

    Returns:
        LossMetrics containing loss, z_loss, weight_sum, and accuracy.

    Raises:
        ValueError: If logits or labels is None.

    Example:
        >>> # NER with BIO tagging (9 labels: O, B-PER, I-PER, B-LOC, etc.)
        >>> config = LossConfig(num_labels=9, ignore_index=-100)
        >>> logits = jnp.randn(8, 128, 9)  # batch, seq_len, num_labels
        >>> labels = jnp.randint(-100, 9, (8, 128))  # -100 for special tokens
        >>> metrics = ForTokenClassification(logits, labels, config=config)
    """
    if logits is None or labels is None:
        raise ValueError("Logits and labels cannot be None")

    loss = fixed_cross_entropy(
        source=logits,
        target=labels,
        config=config,
        batch=batch,
        **kwargs,
    )
    return loss


# Mapping from task name suffix to loss function for automatic selection.
LOSS_MAPPING = {
    "ForCausalLM": ForCausalLMLoss,
    "ForQuestionAnswering": ForQuestionAnsweringLoss,
    "ForSequenceClassification": ForSequenceClassificationLoss,
    "ForTokenClassification": ForTokenClassification,
}
