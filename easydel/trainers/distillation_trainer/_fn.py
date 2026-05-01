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

"""Internal functions for knowledge distillation training.

This module contains the core computational functions used by the distillation trainer,
including loss functions and training/evaluation step implementations. These functions
implement knowledge distillation as described by Hinton et al., where a student model
learns to mimic a teacher model's output distributions.

The distillation process uses temperature scaling to soften probability distributions,
allowing the student to learn from the teacher's confidence across all classes rather
than just the hard labels. The loss combines KL divergence between teacher and student
distributions with optional supervised learning loss.

All functions are designed for JAX/spectrax models and support distributed training.
"""

import collections.abc
import functools
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from jax import Array as JaxArray
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    cached_scheduled_auxiliary,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    stop_gradient_tree,
    sync_module_schedule_config,
    update_metrics,
    update_state_respectfully,
)


def _per_token_xent(
    teacher_logits: Array,
    student_logits: Array,
    temperature: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Compute per-token distillation cross-entropy and teacher entropy.

    Teacher logits are processed first so their scaled intermediates can be
    freed before student intermediates are materialised — peak vocab-sized
    float32 tensors drops from 3x to 2x ``[..., V]``.

    Returns ``(per_token_distill_xent, per_token_teacher_entropy)``.
    """
    teacher_scaled = jax.lax.stop_gradient(teacher_logits.astype(jnp.float32) / temperature)
    teacher_logsumexp = jax.nn.logsumexp(teacher_scaled, axis=-1, keepdims=True)
    teacher_log_probs = teacher_scaled - teacher_logsumexp
    teacher_probs = jnp.exp(teacher_log_probs)
    per_token_teacher_entropy = -jnp.sum(teacher_probs * teacher_log_probs, axis=-1).astype(dtype)

    student_scaled = student_logits.astype(jnp.float32) / temperature
    student_logsumexp = jax.nn.logsumexp(student_scaled, axis=-1)
    per_token_distill_xent = (student_logsumexp - jnp.sum(teacher_probs * student_scaled, axis=-1)).astype(dtype)
    return per_token_distill_xent, per_token_teacher_entropy


def _compute_kl_and_ce(
    student_logits: Array,
    teacher_logits: Array,
    mask: Array,
    safe_labels: Array,
    use_hard_labels: bool,
    temperature: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array]:
    """Per-token distillation sums for one chunk of logits.

    Returns ``(distill_xent_sum, teacher_entropy_sum, ce_sum, mask_sum)``,
    where all values are scalar accumulators.
    """
    per_token_distill_xent, per_token_teacher_entropy = _per_token_xent(
        teacher_logits,
        student_logits,
        temperature,
        dtype,
    )

    distill_xent_sum = jnp.sum(per_token_distill_xent * mask)
    teacher_entropy_sum = jnp.sum(per_token_teacher_entropy * mask)
    mask_sum = jnp.sum(mask)

    ce_sum = jnp.zeros((), dtype=dtype)
    if use_hard_labels:
        per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
            student_logits.astype(jnp.float32),
            safe_labels,
        ).astype(dtype)
        ce_sum = jnp.sum(per_token_ce * mask)

    return distill_xent_sum, teacher_entropy_sum, ce_sum, mask_sum


def _finalize_distillation_metrics(
    distill_xent_sum: Array,
    teacher_entropy_sum: Array,
    ce_sum: Array,
    mask_sum: Array,
    temperature: float,
    alpha: float,
    use_hard_labels: bool,
    dtype: jnp.dtype,
) -> tuple[Array, dict[str, Array]]:
    """Normalise accumulated distillation/CE sums into final scalar metrics/loss."""
    alpha_s = jnp.array(alpha, dtype=dtype)
    temp_sq = jnp.array(temperature * temperature, dtype=dtype)
    normalizer = jnp.maximum(mask_sum, jnp.ones((), dtype=dtype))

    distill_xent_loss = (distill_xent_sum / normalizer) * temp_sq
    teacher_entropy_loss = (teacher_entropy_sum / normalizer) * temp_sq
    kl_loss = distill_xent_loss - teacher_entropy_loss
    total_loss = alpha_s * kl_loss

    ce_loss = jnp.zeros((), dtype=dtype)
    if use_hard_labels:
        ce_loss = ce_sum / normalizer
        total_loss = total_loss + (jnp.ones((), dtype=dtype) - alpha_s) * ce_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=dtype),
        "distill_xent_loss": jnp.asarray(distill_xent_loss, dtype=dtype),
        "teacher_entropy_loss": jnp.asarray(teacher_entropy_loss, dtype=dtype),
        "ce_loss": jnp.asarray(ce_loss, dtype=dtype),
    }
    return total_loss, metrics


def _build_mask_and_labels(
    attention_mask: Array | None,
    loss_mask: Array | None,
    labels: Array | None,
    dtype: jnp.dtype,
    seq_len: int,
    batch_size: int,
) -> tuple[Array, Array, bool]:
    """Build a combined per-token mask and safe labels array.

    Returns (mask, safe_labels, has_labels).
    """
    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = jnp.ones((batch_size, seq_len), dtype=dtype)

    has_labels = labels is not None
    if has_labels:
        valid_label_mask = (labels != -100).astype(dtype)
        mask = mask * valid_label_mask
        safe_labels = jnp.where(labels == -100, 0, labels)
    else:
        safe_labels = jnp.zeros((batch_size, seq_len), dtype=jnp.int32)

    return mask, safe_labels, has_labels


def distillation_loss(
    student_logits: Array,
    teacher_logits: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
) -> tuple[Array, dict[str, Array]]:
    """Compute knowledge distillation loss between student and teacher models.

    This function implements the distillation loss as described in Hinton et al.'s
    "Distilling the Knowledge in a Neural Network". It combines KL divergence loss
    between temperature-scaled teacher and student distributions with optional
    supervised learning loss on hard labels.

    Args:
        student_logits (Array): Raw logits from the student model.
            Shape: [batch_size, sequence_length, vocab_size]
        teacher_logits (Array): Raw logits from the teacher model.
            Shape: [batch_size, sequence_length, vocab_size]
        attention_mask (Array | None): Mask indicating valid tokens.
            1 for valid tokens, 0 for padding. Shape: [batch_size, sequence_length]
        loss_mask (Array | None): Optional task-specific token mask used for loss
            computation. When provided, this takes priority over attention_mask.
            Useful for assistant-only objectives where prompt tokens are masked out.
        labels (Array | None): Ground truth labels for supervised loss.
            Shape: [batch_size, sequence_length]
        use_hard_labels (bool): Whether to include supervised loss with hard labels.
            If True, combines distillation loss with cross-entropy loss.
        temperature (float): Temperature for softening probability distributions.
            Higher values create softer distributions. Default: 4.0
        alpha (float): Weight for distillation loss vs supervised loss.
            1.0 means pure distillation, 0.0 means pure supervised. Default: 0.9

    Returns:
        tuple[Array, dict[str, Array]]: Scalar loss value combining distillation
        and optional supervised loss together with the individual components.

    Note:
        The distillation metrics are:
        ``distill_xent_loss = E_t[-log p_s] * T^2``,
        ``teacher_entropy_loss = E_t[-log p_t] * T^2``,
        ``kl_loss = distill_xent_loss - teacher_entropy_loss``.
        Masking semantics follow ``loss_mask`` > ``attention_mask`` and combine
        with ``labels != -100`` when labels are provided.
    """
    dtype = student_logits.dtype
    alpha_s = jnp.array(alpha, dtype=dtype)
    temp_sq = jnp.array(temperature * temperature, dtype=dtype)

    per_token_distill_xent, per_token_teacher_entropy = _per_token_xent(
        teacher_logits,
        student_logits,
        temperature,
        dtype,
    )

    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = None

    if labels is not None:
        valid_label_mask = (labels != -100).astype(dtype)
        mask = valid_label_mask if mask is None else mask * valid_label_mask

    if mask is not None:
        normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
        distill_xent_loss = jnp.sum(per_token_distill_xent * mask) / normalizer
        teacher_entropy_loss = jnp.sum(per_token_teacher_entropy * mask) / normalizer
    else:
        distill_xent_loss = jnp.mean(per_token_distill_xent)
        teacher_entropy_loss = jnp.mean(per_token_teacher_entropy)
    distill_xent_loss = distill_xent_loss * temp_sq
    teacher_entropy_loss = teacher_entropy_loss * temp_sq
    kl_loss = distill_xent_loss - teacher_entropy_loss
    total_loss = alpha_s * kl_loss
    ce_loss = jnp.array(0.0, dtype=dtype)
    if use_hard_labels and labels is not None:
        safe_labels = jnp.where(labels == -100, 0, labels)
        per_token_ce = optax.softmax_cross_entropy_with_integer_labels(
            student_logits.astype(jnp.float32),
            safe_labels,
        ).astype(dtype)

        if mask is not None:
            ce_loss = per_token_ce * mask
            normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
            ce_loss = jnp.sum(ce_loss) / normalizer
        else:
            ce_loss = jnp.mean(per_token_ce)

        total_loss = total_loss + (jnp.array(1.0, dtype=dtype) - alpha_s) * ce_loss

    metrics = {
        "kl_loss": jnp.asarray(kl_loss, dtype=dtype),
        "distill_xent_loss": jnp.asarray(distill_xent_loss, dtype=dtype),
        "teacher_entropy_loss": jnp.asarray(teacher_entropy_loss, dtype=dtype),
        "ce_loss": jnp.asarray(ce_loss, dtype=dtype),
    }
    return total_loss, metrics


def chunked_distillation_loss(
    student_hidden: Array,
    teacher_hidden: Array,
    student_lm_head_fn: tp.Callable[[Array], Array],
    teacher_lm_head_fn: tp.Callable[[Array], Array],
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
    chunk_size: int = 128,
) -> tuple[Array, dict[str, Array]]:
    """Memory-efficient distillation loss that avoids materialising full logits.

    Instead of receiving pre-computed ``[B, L, V]`` logits, this function takes
    the last hidden states from both models and their lm_head projection
    functions.  It processes the sequence in chunks of *chunk_size* tokens,
    projecting each chunk to vocab logits on-the-fly and immediately reducing
    to scalar KL / CE contributions.  Peak logit memory drops from
    ``O(B * L * V)`` to ``O(B * chunk_size * V)``.

    The scan body is wrapped in ``jax.checkpoint`` so that during the backward
    pass each chunk's logits are *recomputed* from the hidden states rather
    than stored, keeping memory constant regardless of sequence length.

    Distillation metrics follow:
    ``distill_xent_loss = E_t[-log p_s] * T^2``,
    ``teacher_entropy_loss = E_t[-log p_t] * T^2``,
    ``kl_loss = distill_xent_loss - teacher_entropy_loss``.
    """
    dtype = student_hidden.dtype
    B, L = student_hidden.shape[:2]

    # Pad sequence length to a multiple of chunk_size.
    pad_len = (-L) % chunk_size
    if pad_len:
        student_hidden = jnp.pad(student_hidden, ((0, 0), (0, pad_len), (0, 0)))
        teacher_hidden = jnp.pad(teacher_hidden, ((0, 0), (0, pad_len), (0, 0)))
        if attention_mask is not None:
            attention_mask = jnp.pad(attention_mask, ((0, 0), (0, pad_len)))
        if loss_mask is not None:
            loss_mask = jnp.pad(loss_mask, ((0, 0), (0, pad_len)))
        if labels is not None:
            labels = jnp.pad(labels, ((0, 0), (0, pad_len)), constant_values=-100)

    L_padded = L + pad_len
    n_chunks = L_padded // chunk_size

    mask, safe_labels, has_labels = _build_mask_and_labels(
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        dtype=dtype,
        seq_len=L_padded,
        batch_size=B,
    )

    # Reshape to [n_chunks, B, chunk_size, ...] for scanning.
    s_chunks = student_hidden.reshape(B, n_chunks, chunk_size, -1).transpose(1, 0, 2, 3)
    t_chunks = teacher_hidden.reshape(B, n_chunks, chunk_size, -1).transpose(1, 0, 2, 3)
    m_chunks = mask.reshape(B, n_chunks, chunk_size).transpose(1, 0, 2)
    l_chunks = safe_labels.reshape(B, n_chunks, chunk_size).transpose(1, 0, 2)

    _use_hard = use_hard_labels and has_labels

    @jax.checkpoint
    def _chunk_kl_ce(s_h, t_h, m, sl):
        """Project a token-chunk of hidden states and compute KL/CE contributions.

        Args:
            s_h: Student hidden-state slice ``[batch, chunk, hidden_dim]``.
            t_h: Teacher hidden-state slice with the same shape.
            m: Loss-mask slice ``[batch, chunk]``.
            sl: Safe-label slice (with ``-100`` replaced by 0) ``[batch, chunk]``.

        Returns:
            ``(distill_xent, teacher_entropy, ce, mask_sum)`` scalars
            for this chunk.
        """
        s_logits = student_lm_head_fn(s_h)
        t_logits = teacher_lm_head_fn(t_h)
        return _compute_kl_and_ce(
            student_logits=s_logits,
            teacher_logits=t_logits,
            mask=m,
            safe_labels=sl,
            use_hard_labels=_use_hard,
            temperature=temperature,
            dtype=dtype,
        )

    def _scan_body(carry, xs):
        """Add one chunk's KL/CE contributions into the scan accumulators.

        Args:
            carry: ``(distill_xent_sum, teacher_entropy_sum, ce_sum,
                mask_sum)`` accumulator.
            xs: ``(s_h, t_h, m, sl)`` batched chunk inputs.

        Returns:
            ``(new_carry, None)`` per the ``jax.lax.scan`` contract.
        """
        s_h, t_h, m, sl = xs
        distill_xent, teacher_entropy, ce, ms = _chunk_kl_ce(s_h, t_h, m, sl)
        return (carry[0] + distill_xent, carry[1] + teacher_entropy, carry[2] + ce, carry[3] + ms), None

    _zero = jnp.zeros((), dtype=dtype)
    (distill_xent_sum, teacher_entropy_sum, ce_sum, mask_sum), _ = jax.lax.scan(
        _scan_body,
        (_zero, _zero, _zero, _zero),
        (s_chunks, t_chunks, m_chunks, l_chunks),
    )

    return _finalize_distillation_metrics(
        distill_xent_sum=distill_xent_sum,
        teacher_entropy_sum=teacher_entropy_sum,
        ce_sum=ce_sum,
        mask_sum=mask_sum,
        temperature=temperature,
        alpha=alpha,
        use_hard_labels=_use_hard,
        dtype=dtype,
    )


def _resolve_indices(
    collection_length: int,
    indices: tuple[int, ...] | None,
    *,
    default_all: bool,
) -> tuple[int, ...]:
    """Resolve user-supplied (possibly negative) layer indices to positive ones.

    Args:
        collection_length: Total number of layers/attentions available.
        indices: Optional layer indices (negative values count from the
            end).  ``None`` falls back to either all layers or just the
            last layer based on ``default_all``.
        default_all: When ``True`` and ``indices`` is empty, return all
            layer indices; otherwise return only the last layer.

    Returns:
        A tuple of strictly positive indices in
        ``[0, collection_length)``.

    Raises:
        ValueError: If ``collection_length`` is zero.
        IndexError: If any resolved index is out of range.
    """
    if collection_length == 0:
        raise ValueError("Cannot select layers from an empty collection.")
    if not indices:
        if default_all:
            return tuple(range(collection_length))
        return (collection_length - 1,)
    resolved: list[int] = []
    for idx in indices:
        resolved_idx = idx if idx >= 0 else collection_length + idx
        if resolved_idx < 0 or resolved_idx >= collection_length:
            raise IndexError(f"Layer index {idx} is out of range for collection of length {collection_length}.")
        resolved.append(int(resolved_idx))
    return tuple(resolved)


def _masked_mse(values: jax.Array, targets: jax.Array, mask: jax.Array | None) -> jax.Array:
    """Compute mean squared error optionally restricted to a mask.

    Args:
        values: Predicted tensor.
        targets: Reference tensor with the same shape as ``values``.
        mask: Optional broadcastable mask; positions where ``mask`` is
            zero are excluded from both numerator and denominator.

    Returns:
        A scalar MSE.

    Raises:
        ValueError: If ``values`` and ``targets`` have mismatched shapes.
    """
    if values.shape != targets.shape:
        raise ValueError(f"Mismatched tensor shapes for distillation: {values.shape} vs {targets.shape}.")
    diff = values - targets
    if mask is not None:
        mask = mask.astype(diff.dtype)
        while mask.ndim < diff.ndim:
            mask = mask[..., None]
        diff = diff * mask
        denom = jnp.maximum(mask.sum(), jnp.array(1.0, dtype=diff.dtype))
    else:
        denom = jnp.array(diff.size, dtype=diff.dtype)
    return jnp.sum(diff * diff) / denom


def _build_attention_mask(attention_mask: jax.Array | None, *, dtype: jnp.dtype) -> jax.Array | None:
    """Expand a 2-D padding mask into a 4-D attention-matrix mask.

    Args:
        attention_mask: ``[batch, seq_len]`` 0/1 mask; ``None`` short-
            circuits to ``None``.
        dtype: Output dtype.

    Returns:
        A ``[batch, 1, seq_len, seq_len]`` mask suitable for masking
        attention probability matrices.
    """
    if attention_mask is None:
        return None
    mask = attention_mask.astype(dtype)
    return mask[:, None, :, None] * mask[:, None, None, :]


def _normalize_attention(tensor: jax.Array) -> jax.Array:
    """Row-normalise an attention probability tensor along the last axis.

    Args:
        tensor: ``[..., q, k]`` attention scores.

    Returns:
        A tensor of the same shape whose last-axis sums are 1
        (row-stochastic), with a tiny denominator floor for numerical
        stability.
    """
    denom = jnp.sum(tensor, axis=-1, keepdims=True)
    denom = jnp.maximum(denom, jnp.finfo(tensor.dtype).tiny)
    return tensor / denom


def _stop_gradient_tree(tree):
    """Apply :func:`jax.lax.stop_gradient` to every JAX array leaf.

    Args:
        tree: Pytree of JAX-array and Python-leaf mixed values.

    Returns:
        A pytree of the same shape with all JAX arrays detached from
        the autograd graph.
    """
    return jax.tree_util.tree_map(lambda x: jax.lax.stop_gradient(x) if isinstance(x, JaxArray) else x, tree)


def _distillation_forward_outputs(
    model,
    batch: collections.abc.Mapping[str, jax.Array],
    *,
    use_chunked: bool,
    request_hidden_states: bool,
    request_attentions: bool,
) -> dict[str, tp.Any]:
    """Run the model forward and collect the outputs needed by the distillation loss.

    Args:
        model: Student or teacher model module.
        batch: Input batch.
        use_chunked: If ``True``, the LM head is *not* applied so the
            chunked path can stream logits later; the last hidden state
            is returned under ``hidden_for_kl``.
        request_hidden_states: Whether to also collect the per-layer
            hidden states.
        request_attentions: Whether to also collect the per-layer
            attention probabilities.

    Returns:
        A dict with at minimum ``logits`` (or ``hidden_for_kl`` in the
        chunked path) plus optional ``hidden_states`` and ``attentions``
        keys.

    Raises:
        TypeError: If the model does not return logits in the
            non-chunked path.
    """
    call_kwargs = dict(batch)
    call_kwargs.pop("labels", None)
    call_kwargs.pop("completion_mask", None)
    call_kwargs.pop("assistant_masks", None)
    for key in ("teacher_logits", "teacher_hidden_for_kl", "teacher_hidden_states", "teacher_attentions"):
        call_kwargs.pop(key, None)
    if use_chunked:
        call_kwargs["apply_lm_head"] = False
    if request_hidden_states:
        call_kwargs["output_hidden_states"] = True
    if request_attentions:
        call_kwargs["output_attentions"] = True
    call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
    call_kwargs = sanitize_model_call_kwargs(call_kwargs)
    outputs = model(**call_kwargs)

    result: dict[str, tp.Any] = {}
    if use_chunked:
        result["hidden_for_kl"] = outputs.last_hidden_state
    else:
        logits = getattr(outputs, "logits", None)
        if logits is None:
            raise TypeError(f"{type(model).__name__} did not return logits for distillation.")
        result["logits"] = logits
    if request_hidden_states:
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is not None:
            result["hidden_states"] = tuple(hidden_states)
    if request_attentions:
        attentions = getattr(outputs, "attentions", None)
        if attentions is not None:
            result["attentions"] = tuple(attentions)
    return result


@functools.lru_cache(maxsize=16)
def _make_distillation_aux_forward(
    use_chunked: bool,
    request_hidden_states: bool,
    request_attentions: bool,
):
    """Build a cached auxiliary forward closure for the scheduled-VJP path.

    Args:
        use_chunked: Whether the chunked logits path is active.
        request_hidden_states: Whether to surface per-layer hidden
            states.
        request_attentions: Whether to surface per-layer attentions.

    Returns:
        A two-argument callable ``forward(model, batch)`` returning the
        distillation forward outputs.
    """

    def forward(model, batch):
        """Run :func:`_distillation_forward_outputs` with the captured options.

        Args:
            model: Student or teacher model module.
            batch: Input batch.

        Returns:
            The distillation forward outputs dict.
        """
        return _distillation_forward_outputs(
            model,
            batch,
            use_chunked=use_chunked,
            request_hidden_states=request_hidden_states,
            request_attentions=request_attentions,
        )

    return forward


def distillation_step(
    student_state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    teacher_state: EasyDeLState,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    temperature: float = 4.0,
    alpha: float = 0.9,
    hidden_state_weight: float = 0.0,
    hidden_state_layers: tuple[int, ...] | None = None,
    hidden_state_loss: tp.Literal["mse"] = "mse",
    attention_weight: float = 0.0,
    attention_layers: tuple[int, ...] | None = None,
    attention_normalize: bool = False,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    logits_chunk_size: int | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Perform a single knowledge-distillation training or evaluation step.

    Runs the teacher model on the batch (with gradients stopped), then
    computes the distillation loss between student and teacher outputs.
    Optionally includes hidden-state MSE and attention-matrix MSE losses
    for deeper distillation. When ``logits_chunk_size`` is set, uses a
    memory-efficient chunked strategy that avoids materialising the full
    ``[B, L, V]`` logit tensor.

    During training the function also computes student gradients via
    minibatch accumulation and updates the student state.

    Args:
        student_state: Current state of the student model.
        batch: Input batch mapping. Must contain at minimum ``input_ids``
            and ``attention_mask``. May also include ``labels`` and
            ``completion_mask``.
        teacher_state: Frozen state of the teacher model.
        loss_config: Optional loss configuration for gradient clipping etc.
        learning_rate_fn: Learning rate schedule function.
        partition_spec: Sharding specification for the batch tensors.
        gradient_accumulation_steps: Number of minibatch accumulation steps.
        is_training: If True, compute gradients and update the student.
            If False, only compute evaluation metrics.
        temperature: Temperature for softening probability distributions
            in the KL-divergence computation.
        alpha: Weight balancing distillation loss vs supervised CE loss.
            1.0 means pure distillation, 0.0 means pure supervised.
        hidden_state_weight: Coefficient for hidden-state MSE loss.
            Set to 0.0 to disable.
        hidden_state_layers: Which transformer layers to distill hidden
            states from. ``None`` defaults to the final layer.
        hidden_state_loss: Distance metric for hidden-state distillation.
            Currently only ``"mse"`` is supported.
        attention_weight: Coefficient for attention-matrix MSE loss.
            Set to 0.0 to disable.
        attention_layers: Which attention layers to distill. ``None``
            defaults to all layers.
        attention_normalize: Whether to L1-normalize attention matrices
            before computing the distillation loss.
        straight_through_emulator: Optional function for quantization-aware
            straight-through gradient estimation.
        logits_chunk_size: When set, compute the KL loss in chunks of this
            many tokens to save memory. ``None`` uses the standard full-logits path.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: When ``is_training``
            is True, returns the updated student state and loss metrics.
            When False, returns only the loss metrics.
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=student_state.model.mesh, ignore_mpmd=True)

    if hidden_state_loss != "mse":
        raise ValueError(f"Unsupported hidden state loss '{hidden_state_loss}'. Only 'mse' is available.")

    request_hidden_states = hidden_state_weight != 0.0
    request_attentions = attention_weight != 0.0
    use_chunked = logits_chunk_size is not None and logits_chunk_size > 0

    def teacher_forward(minibatch: collections.abc.Mapping[str, jax.Array]) -> dict[str, tp.Any]:
        """Run the teacher in stop-gradient mode for one minibatch.

        Args:
            minibatch: Input minibatch dictionary.

        Returns:
            A dict with teacher logits / hidden states / attentions
            ready to be consumed by the distillation loss.

        Raises:
            TypeError: If the teacher does not return logits in the
                non-chunked code path.
        """
        teacher_call_kwargs = dict(minibatch)
        teacher_call_kwargs.pop("labels", None)
        teacher_call_kwargs.pop("completion_mask", None)
        teacher_call_kwargs.pop("assistant_masks", None)
        if use_chunked:
            teacher_call_kwargs["apply_lm_head"] = False
        if request_hidden_states:
            teacher_call_kwargs["output_hidden_states"] = True
        if request_attentions:
            teacher_call_kwargs["output_attentions"] = True
        teacher_call_kwargs = filter_kwargs_for_callable(
            getattr(teacher_state.model, "forward", teacher_state.model), teacher_call_kwargs
        )
        teacher_call_kwargs = sanitize_model_call_kwargs(teacher_call_kwargs)
        teacher_static_kwargs = {
            key: teacher_call_kwargs.pop(key)
            for key in list(teacher_call_kwargs)
            if not hasattr(teacher_call_kwargs[key], "shape")
        }

        @functools.partial(
            jax.checkpoint,
            prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        def _teacher_fwd(kw, t_graphstate):
            """Re-materializable teacher forward used inside ``teacher_forward``.

            Args:
                kw: Dynamic kwargs forwarded to the teacher module.
                t_graphstate: Stop-gradient teacher graphstate.

            Returns:
                A dict with stop-gradient teacher outputs (logits or
                hidden states plus optional layer activations).
            """
            teacher_module = teacher_state.merge(t_graphstate)
            teacher_outputs = teacher_module(**kw, **teacher_static_kwargs)
            result: dict[str, tp.Any] = {}
            if use_chunked:
                result["hidden_for_kl"] = jax.lax.stop_gradient(teacher_outputs.last_hidden_state)
            else:
                result["logits"] = jax.lax.stop_gradient(teacher_outputs.logits)
            if request_hidden_states:
                teacher_hidden = getattr(teacher_outputs, "hidden_states", None)
                if teacher_hidden is not None:
                    result["hidden_states"] = _stop_gradient_tree(tuple(teacher_hidden))
            if request_attentions:
                teacher_attns = getattr(teacher_outputs, "attentions", None)
                if teacher_attns is not None:
                    result["attentions"] = _stop_gradient_tree(tuple(teacher_attns))
            return result

        return _teacher_fwd(
            teacher_call_kwargs,
            jax.lax.stop_gradient(teacher_state.graphstate),
        )

    def loss_fn(tree, minibatch):
        """Compute the distillation loss for one minibatch.

        Runs the student forward (with quantization-aware STE in
        training), pulls the teacher outputs from
        :func:`teacher_forward`, evaluates the KL/CE term and adds the
        optional hidden-state and attention MSE terms.

        Args:
            tree: Student graphstate to differentiate against.
            minibatch: One minibatch slice.

        Returns:
            ``(loss, metrics)`` where ``metrics`` is a populated
            :class:`LossMetrics`.
        """
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = student_state.merge(tree)
        teacher_outputs = teacher_forward(minibatch)
        call_kwargs = dict(minibatch)
        call_kwargs.pop("labels", None)
        call_kwargs.pop("completion_mask", None)
        call_kwargs.pop("assistant_masks", None)
        if use_chunked:
            teacher_hidden_for_kl = teacher_outputs["hidden_for_kl"]
            call_kwargs["apply_lm_head"] = False
        else:
            teacher_logits = teacher_outputs["logits"]
        teacher_hiddens = teacher_outputs.get("hidden_states")
        teacher_attns = teacher_outputs.get("attentions")
        if request_hidden_states:
            call_kwargs["output_hidden_states"] = True
        if request_attentions:
            call_kwargs["output_attentions"] = True
        call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        student_outputs = module(**call_kwargs)
        labels = minibatch.get("labels", None)
        attention_mask = minibatch.get("attention_mask", None)
        completion_mask = minibatch.get("completion_mask", None)

        if use_chunked:
            total_loss, _loss_components = chunked_distillation_loss(
                student_hidden=student_outputs.last_hidden_state,
                teacher_hidden=teacher_hidden_for_kl,
                student_lm_head_fn=module.make_lm_head_fn(),
                teacher_lm_head_fn=teacher_state.model.make_lm_head_fn(),
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
                chunk_size=int(logits_chunk_size),
            )
        else:
            total_loss, loss_components = distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_logits,
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
            )
        metrics_map: dict[str, jax.Array] = dict(loss_components)

        if request_hidden_states:
            student_hidden = getattr(student_outputs, "hidden_states", None)
            if student_hidden is None or teacher_hiddens is None:
                raise ValueError(
                    "Hidden-state distillation requested but models did not return hidden states. "
                    "Please ensure `output_hidden_states` is supported."
                )
            student_indices = _resolve_indices(len(student_hidden), hidden_state_layers, default_all=False)
            teacher_indices = _resolve_indices(len(teacher_hiddens), hidden_state_layers, default_all=False)
            if len(student_indices) != len(teacher_indices):
                raise ValueError(
                    "Hidden-state layer selections for student and teacher have different lengths. "
                    "Please align the requested layers across both models."
                )
            hidden_losses = []
            for s_idx, t_idx in zip(student_indices, teacher_indices, strict=True):
                hidden_losses.append(_masked_mse(student_hidden[s_idx], teacher_hiddens[t_idx], attention_mask))
            hidden_loss_value = jnp.mean(jnp.stack(hidden_losses))
            hidden_loss_value = hidden_loss_value.astype(total_loss.dtype)
            total_loss = total_loss + jnp.asarray(hidden_state_weight, dtype=total_loss.dtype) * hidden_loss_value
            metrics_map["hidden_state_loss"] = hidden_loss_value

        if request_attentions:
            student_attentions = getattr(student_outputs, "attentions", None)
            if student_attentions is None or teacher_attns is None:
                raise ValueError(
                    "Attention distillation requested but models did not return attention probabilities. "
                    "Please ensure `output_attentions` is supported."
                )
            student_indices = _resolve_indices(len(student_attentions), attention_layers, default_all=True)
            teacher_indices = _resolve_indices(len(teacher_attns), attention_layers, default_all=True)
            if len(student_indices) != len(teacher_indices):
                raise ValueError(
                    "Attention layer selections for student and teacher have different lengths. "
                    "Please align the requested layers across both models."
                )
            attn_mask = _build_attention_mask(attention_mask, dtype=total_loss.dtype)
            attention_losses = []
            for s_idx, t_idx in zip(student_indices, teacher_indices, strict=True):
                s_attn = student_attentions[s_idx]
                t_attn = teacher_attns[t_idx]
                if attention_normalize:
                    s_attn = _normalize_attention(s_attn)
                    t_attn = _normalize_attention(t_attn)
                attention_losses.append(_masked_mse(s_attn, t_attn, attn_mask))
            attention_loss_value = jnp.mean(jnp.stack(attention_losses))
            attention_loss_value = attention_loss_value.astype(total_loss.dtype)
            total_loss = total_loss + jnp.asarray(attention_weight, dtype=total_loss.dtype) * attention_loss_value
            metrics_map["attention_loss"] = attention_loss_value

        metrics = LossMetrics(
            loss=total_loss,
            other_metrics={key: jnp.asarray(value) for key, value in metrics_map.items()},
        )
        return total_loss, metrics

    # Compute gradients and metrics across minibatches.
    if is_training:
        gradients, metrics = minibatch_call(
            state=student_state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
        student_state = update_state_respectfully(
            state=student_state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=update_metrics(
                metrics=metrics,
                learning_rate_fn=learning_rate_fn,
                step=student_state.step,
                gradients=gradients,
            ),
        )
        return student_state, metrics
    else:
        _, metrics = loss_fn(tree=student_state.graphstate, minibatch=batch)
        return metrics


def _prepare_distillation_scheduled_batch(call) -> dict[str, tp.Any]:
    """Inject precomputed teacher outputs into ``call.batch``.

    Runs the teacher model under :func:`cached_scheduled_auxiliary` and
    stashes either ``teacher_logits`` (full path) or
    ``teacher_hidden_for_kl`` (chunked path), plus optional teacher
    hidden states and attentions when the trainer requested them.

    Args:
        call: The :class:`ScheduledStepCall` describing the current
            step.

    Returns:
        A copy of ``call.batch`` with the appropriate teacher outputs
        attached.

    Raises:
        RuntimeError: If no teacher state is available.
    """
    batch = dict(call.batch)
    use_chunked = call.get("logits_chunk_size") is not None and call.get("logits_chunk_size") > 0
    request_hidden_states = call.get("hidden_state_weight", 0.0) != 0.0
    request_attentions = call.get("attention_weight", 0.0) != 0.0
    required_key = "teacher_hidden_for_kl" if use_chunked else "teacher_logits"
    if required_key in batch:
        return batch

    teacher_state = call.get("teacher_state")
    if teacher_state is None:
        raise RuntimeError("Distillation scheduled MPMD training requires teacher_state.")

    teacher_model = teacher_state.model
    teacher_model.eval()
    sync_module_schedule_config(teacher_model, call.schedule)
    constrained_batch = constrain_scheduled_batch(teacher_model, batch, call.get("partition_spec"))
    forward_fn = _make_distillation_aux_forward(use_chunked, request_hidden_states, request_attentions)
    teacher_forward = cached_scheduled_auxiliary(forward_fn, teacher_model.mesh)
    teacher_outputs = stop_gradient_tree(teacher_forward(teacher_model, constrained_batch))
    if use_chunked:
        batch["teacher_hidden_for_kl"] = teacher_outputs["hidden_for_kl"]
    else:
        batch["teacher_logits"] = teacher_outputs["logits"]
    if request_hidden_states and "hidden_states" in teacher_outputs:
        batch["teacher_hidden_states"] = teacher_outputs["hidden_states"]
    if request_attentions and "attentions" in teacher_outputs:
        batch["teacher_attentions"] = teacher_outputs["attentions"]
    return batch


def _distillation_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the distillation scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering all distillation knobs that influence
        compilation (temperature, alpha, hidden-state / attention
        weights and layer indices, logits chunk size, partition spec,
        plus the teacher state and quantizer identities).
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "partition_spec",
            "temperature",
            "alpha",
            "hidden_state_weight",
            "hidden_state_layers",
            "hidden_state_loss",
            "attention_weight",
            "attention_layers",
            "attention_normalize",
            "logits_chunk_size",
        ),
        object_fields=("straight_through_emulator", "teacher_state"),
    )


def _make_distillation_scheduled_loss(call):
    """Build a SpectraX-scheduled distillation scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` carrying the trainer's
            current configuration.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready to feed to
        :func:`spx.sxvalue_and_grad`.

    Raises:
        ValueError: If an unsupported hidden-state loss is configured.
    """
    partition_spec = call.get("partition_spec")
    temperature = call.get("temperature", 4.0)
    alpha = call.get("alpha", 0.9)
    hidden_state_weight = call.get("hidden_state_weight", 0.0)
    hidden_state_layers = call.get("hidden_state_layers")
    hidden_state_loss = call.get("hidden_state_loss", "mse")
    attention_weight = call.get("attention_weight", 0.0)
    attention_layers = call.get("attention_layers")
    attention_normalize = call.get("attention_normalize", False)
    logits_chunk_size = call.get("logits_chunk_size")
    use_chunked = logits_chunk_size is not None and logits_chunk_size > 0
    request_hidden_states = hidden_state_weight != 0.0
    request_attentions = attention_weight != 0.0
    teacher_state = call.get("teacher_state")

    if hidden_state_loss != "mse":
        raise ValueError(f"Unsupported hidden state loss '{hidden_state_loss}'. Only 'mse' is available.")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar distillation loss inside the SpectraX scheduled VJP.

        Combines the KL/CE term, the optional hidden-state MSE term,
        and the optional attention MSE term using the captured weights.

        Args:
            tree: Student graphstate to differentiate against.
            batch: Minibatch dict with precomputed teacher outputs.

        Returns:
            The combined scalar distillation loss.

        Raises:
            ValueError: If hidden-state / attention distillation is
                requested but the relevant outputs are missing.
            RuntimeError: If the chunked path is requested without a
                teacher state.
        """
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)
        student_outputs = _distillation_forward_outputs(
            module,
            call_batch,
            use_chunked=use_chunked,
            request_hidden_states=request_hidden_states,
            request_attentions=request_attentions,
        )
        labels = call_batch.get("labels", None)
        attention_mask = call_batch.get("attention_mask", None)
        completion_mask = call_batch.get("completion_mask", None)

        if use_chunked:
            if teacher_state is None:
                raise RuntimeError("Chunked distillation scheduled MPMD training requires teacher_state.")
            total_loss, _loss_components = chunked_distillation_loss(
                student_hidden=student_outputs["hidden_for_kl"],
                teacher_hidden=jax.lax.stop_gradient(call_batch["teacher_hidden_for_kl"]),
                student_lm_head_fn=module.make_lm_head_fn(),
                teacher_lm_head_fn=teacher_state.model.make_lm_head_fn(),
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
                chunk_size=int(logits_chunk_size),
            )
        else:
            total_loss, _loss_components = distillation_loss(
                student_logits=student_outputs["logits"],
                teacher_logits=jax.lax.stop_gradient(call_batch["teacher_logits"]),
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
            )

        if request_hidden_states:
            student_hidden = student_outputs.get("hidden_states")
            teacher_hiddens = call_batch.get("teacher_hidden_states")
            if student_hidden is None or teacher_hiddens is None:
                raise ValueError(
                    "Hidden-state distillation requested but models did not return hidden states. "
                    "Please ensure `output_hidden_states` is supported."
                )
            student_indices = _resolve_indices(len(student_hidden), hidden_state_layers, default_all=False)
            teacher_indices = _resolve_indices(len(teacher_hiddens), hidden_state_layers, default_all=False)
            hidden_losses = []
            for s_idx, t_idx in zip(student_indices, teacher_indices, strict=True):
                hidden_losses.append(
                    _masked_mse(student_hidden[s_idx], jax.lax.stop_gradient(teacher_hiddens[t_idx]), attention_mask)
                )
            hidden_loss_value = jnp.mean(jnp.stack(hidden_losses)).astype(total_loss.dtype)
            total_loss = total_loss + jnp.asarray(hidden_state_weight, dtype=total_loss.dtype) * hidden_loss_value

        if request_attentions:
            student_attentions = student_outputs.get("attentions")
            teacher_attns = call_batch.get("teacher_attentions")
            if student_attentions is None or teacher_attns is None:
                raise ValueError(
                    "Attention distillation requested but models did not return attention probabilities. "
                    "Please ensure `output_attentions` is supported."
                )
            student_indices = _resolve_indices(len(student_attentions), attention_layers, default_all=True)
            teacher_indices = _resolve_indices(len(teacher_attns), attention_layers, default_all=True)
            attn_mask = _build_attention_mask(attention_mask, dtype=total_loss.dtype)
            attention_losses = []
            for s_idx, t_idx in zip(student_indices, teacher_indices, strict=True):
                s_attn = student_attentions[s_idx]
                t_attn = jax.lax.stop_gradient(teacher_attns[t_idx])
                if attention_normalize:
                    s_attn = _normalize_attention(s_attn)
                    t_attn = _normalize_attention(t_attn)
                attention_losses.append(_masked_mse(s_attn, t_attn, attn_mask))
            attention_loss_value = jnp.mean(jnp.stack(attention_losses)).astype(total_loss.dtype)
            total_loss = total_loss + jnp.asarray(attention_weight, dtype=total_loss.dtype) * attention_loss_value

        return total_loss

    return scheduled_loss


register_scheduled_loss_adapter(
    distillation_step,
    ScheduledLossAdapter(
        name="distillation",
        make_loss=_make_distillation_scheduled_loss,
        make_cache_key=_distillation_scheduled_loss_cache_key,
        prepare_batch=_prepare_distillation_scheduled_batch,
    ),
)
