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
from ejkernel.modules.operations import fused_cross_entropy as _fused_ce
from ejkernel.modules.operations import fused_kl_divergence as _fused_kl
from jax import Array as JaxArray
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint
from spectrax.common_types import BATCH, LENGTH, MODE_TRAIN, VOCAB
from spectrax.sharding import named_sharding_for_shape, reshape_with_named_shardings, transpose_with_named_shardings

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    _scheduled_terminal_stage_rank,
    bind_scheduled_module,
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


def _constrain_distillation_input_batch(
    batch: collections.abc.Mapping[str, tp.Any],
    partition_spec: PartitionSpec | None,
    *,
    mesh: tp.Any,
) -> dict[str, tp.Any]:
    """Apply a sharding constraint to every leaf of a distillation batch.

    Wraps :func:`spectrax.with_sharding_constraint` with
    ``ignore_mpmd=True`` so the constraint is a no-op when the trainer
    is running under an MPMD scheduler that owns its own placement.

    Args:
        batch: Distillation input mapping.
        partition_spec: Target partition spec; ``None`` is a no-op.
        mesh: Active JAX/spectrax mesh.

    Returns:
        A new ``dict`` whose leaves are constrained to
        ``partition_spec`` on ``mesh``.
    """
    return tp.cast(
        dict[str, tp.Any],
        dict(with_sharding_constraint(batch, partition_spec, mesh=mesh, ignore_mpmd=True)),
    )


def _chunk_sequence_for_scan(value: Array, chunk_size: int, *, context: str) -> Array:
    """Reshape ``(B, L, ...)`` to ``(chunks, B, chunk, ...)`` while preserving sharding.

    Used by :func:`chunked_distillation_loss` to feed
    sequence-chunked tensors into :func:`jax.lax.scan`. The function
    propagates the input's :class:`~jax.sharding.NamedSharding`
    through the reshape and the leading-axis transpose so that the
    chunked tensor stays consistent with the rest of the mesh layout
    (otherwise jax inserts implicit reshards that defeat the chunking
    memory win).

    Args:
        value: Tensor shaped ``(batch, seq_len, ...)``. ``seq_len``
            must be a multiple of ``chunk_size``.
        chunk_size: Number of sequence positions per chunk.
        context: Debug label used when constructing the propagated
            ``NamedSharding`` (helpful for spectrax error messages).

    Returns:
        A tensor shaped
        ``(n_chunks, batch, chunk_size, ...)`` ready to be scanned
        over the leading axis.
    """
    bsz, seq_len = value.shape[:2]
    n_chunks = seq_len // chunk_size
    source_sharding = getattr(value, "sharding", None)
    source_parts: list[object] | None = None
    if isinstance(source_sharding, jax.sharding.NamedSharding):
        source_parts = list(tuple(source_sharding.spec))
        while len(source_parts) < len(value.shape):
            source_parts.append(None)

    reshape_axes = (bsz, n_chunks, chunk_size, *value.shape[2:])
    reshape_spec = None
    reshape_sharding = None
    if source_parts is not None:
        reshape_spec = PartitionSpec(source_parts[0], None, source_parts[1], *source_parts[2 : len(value.shape)])
        reshape_sharding = named_sharding_for_shape(
            source_sharding,
            tuple(int(dim) for dim in reshape_axes),
            reshape_spec,
            context=f"{context}:reshape",
        )

    if isinstance(source_sharding, jax.sharding.NamedSharding) and reshape_sharding is not None:
        reshaped = reshape_with_named_shardings(
            value,
            reshape_axes,
            in_sharding=source_sharding,
            out_sharding=reshape_sharding,
        )
    else:
        reshaped = value.reshape(reshape_axes)

    permutation = (1, 0, 2, *range(3, len(reshape_axes)))
    scan_sharding = None
    if source_parts is not None:
        scan_spec = PartitionSpec(None, source_parts[0], source_parts[1], *source_parts[2 : len(value.shape)])
        scan_shape = tuple(reshape_axes[axis] for axis in permutation)
        scan_sharding = named_sharding_for_shape(
            source_sharding,
            tuple(int(dim) for dim in scan_shape),
            scan_spec,
            context=f"{context}:transpose",
        )

    if isinstance(reshape_sharding, jax.sharding.NamedSharding) and isinstance(
        scan_sharding, jax.sharding.NamedSharding
    ):
        chunked = transpose_with_named_shardings(
            reshaped,
            permutation,
            in_sharding=reshape_sharding,
            out_sharding=scan_sharding,
        )
    else:
        chunked = reshaped.transpose(permutation)
    return chunked


def _per_token_xent(
    teacher_logits: Array,
    student_logits: Array,
    temperature: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Compute per-token distillation cross-entropy and teacher entropy.

    Computes
    ``H(p_t, p_s) = -sum_v p_t(v) * log p_s(v)`` and
    ``H(p_t) = -sum_v p_t(v) * log p_t(v)``
    over the vocabulary axis for every token position, where
    ``p_t`` / ``p_s`` are the temperature-softened teacher /
    student distributions. The KL contribution at token ``i`` is
    then ``H(p_t, p_s)_i - H(p_t)_i``.

    Teacher logits are processed first so their scaled intermediates can be
    freed before student intermediates are materialised -- peak vocab-sized
    float32 tensors drops from 3x to 2x ``[..., V]``.

    Args:
        teacher_logits: Teacher logits ``[..., vocab]``. Stop-gradient
            is applied internally.
        student_logits: Student logits ``[..., vocab]``.
        temperature: Softmax temperature.
        dtype: Output dtype for the per-token tensors.

    Returns:
        ``(per_token_distill_xent, per_token_teacher_entropy)``: two
        tensors of shape ``[...]`` (i.e. without the vocab axis).
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


def _kl_div_from_log_probs(log_target: Array, log_input: Array) -> Array:
    """Return per-token ``KL(target || input)`` from log-probabilities."""
    target_probs = jnp.exp(log_target)
    return jnp.sum(target_probs * (log_target - log_input), axis=-1)


def _per_token_generalized_jsd(
    teacher_logits: Array,
    student_logits: Array,
    temperature: float,
    beta: float,
    dtype: jnp.dtype,
) -> Array:
    """Compute per-token generalized Jensen-Shannon distillation loss.

    Uses the same convention as EasyDeL GKD:
    ``m = (1 - beta) * p_s + beta * p_t`` and
    ``beta * KL(p_t || m) + (1 - beta) * KL(p_s || m)``.
    """
    student_log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32) / temperature, axis=-1)
    teacher_log_probs = jax.lax.stop_gradient(
        jax.nn.log_softmax(teacher_logits.astype(jnp.float32) / temperature, axis=-1)
    )
    if beta <= 0.0:
        return _kl_div_from_log_probs(student_log_probs, teacher_log_probs).astype(dtype)
    if beta >= 1.0:
        return _kl_div_from_log_probs(teacher_log_probs, student_log_probs).astype(dtype)
    beta_val = jnp.asarray(beta, dtype=jnp.float32)
    mixture_log_probs = jax.scipy.special.logsumexp(
        jnp.stack(
            [
                student_log_probs + jnp.log1p(-beta_val),
                teacher_log_probs + jnp.log(beta_val),
            ]
        ),
        axis=0,
    )
    kl_teacher = _kl_div_from_log_probs(teacher_log_probs, mixture_log_probs)
    kl_student = _kl_div_from_log_probs(student_log_probs, mixture_log_probs)
    per_token = beta_val * kl_teacher + (jnp.asarray(1.0, dtype=jnp.float32) - beta_val) * kl_student
    return per_token.astype(dtype)


def _per_token_topk_xent(
    teacher_logits: Array,
    student_logits: Array,
    temperature: float,
    top_k: int,
    add_tail: bool,
    dtype: jnp.dtype,
) -> tuple[Array, Array]:
    """Compute teacher-top-k distillation cross-entropy and entropy.

    When ``add_tail`` is enabled, all non-top-k mass is represented by one
    extra bucket. Otherwise both teacher and student distributions are
    re-normalized over the selected support, matching TRL's top-k KD surface.
    """
    vocab_size = int(teacher_logits.shape[-1])
    k = min(max(int(top_k), 1), vocab_size)
    teacher_log_probs = jax.lax.stop_gradient(
        jax.nn.log_softmax(teacher_logits.astype(jnp.float32) / temperature, axis=-1)
    )
    student_log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32) / temperature, axis=-1)
    top_teacher_log_probs, top_indices = jax.lax.top_k(teacher_log_probs, k)
    top_student_log_probs = jnp.take_along_axis(student_log_probs, top_indices, axis=-1)

    if add_tail and k < vocab_size:
        top_teacher_probs = jnp.exp(top_teacher_log_probs)
        top_student_probs = jnp.exp(top_student_log_probs)
        eps = jnp.asarray(1e-7, dtype=jnp.float32)
        teacher_tail_prob = jnp.clip(1.0 - jnp.sum(top_teacher_probs, axis=-1), eps, 1.0)
        student_tail_prob = jnp.clip(1.0 - jnp.sum(top_student_probs, axis=-1), eps, 1.0)
        teacher_tail_log_prob = jnp.log(teacher_tail_prob)
        student_tail_log_prob = jnp.log(student_tail_prob)
        xent = -(jnp.sum(top_teacher_probs * top_student_log_probs, axis=-1) + teacher_tail_prob * student_tail_log_prob)
        entropy = -(
            jnp.sum(top_teacher_probs * top_teacher_log_probs, axis=-1) + teacher_tail_prob * teacher_tail_log_prob
        )
        return xent.astype(dtype), entropy.astype(dtype)

    teacher_support_log_probs = jax.nn.log_softmax(top_teacher_log_probs, axis=-1)
    student_support_log_probs = jax.nn.log_softmax(top_student_log_probs, axis=-1)
    teacher_support_probs = jnp.exp(teacher_support_log_probs)
    xent = -jnp.sum(teacher_support_probs * student_support_log_probs, axis=-1)
    entropy = -jnp.sum(teacher_support_probs * teacher_support_log_probs, axis=-1)
    return xent.astype(dtype), entropy.astype(dtype)


def _compute_kl_and_ce(
    student_logits: Array,
    teacher_logits: Array,
    mask: Array,
    safe_labels: Array,
    use_hard_labels: bool,
    temperature: float,
    dtype: jnp.dtype,
) -> tuple[Array, Array, Array, Array]:
    """Reduce per-token KL/CE contributions over one chunk of logits.

    Wraps :func:`_per_token_xent` with mask-weighted summation and the
    optional hard-label cross-entropy on the student logits. Returns
    scalars so the caller (``jax.lax.scan`` in
    :func:`chunked_distillation_loss`) can accumulate cheaply.

    Args:
        student_logits: Student logits ``[..., vocab]`` for the chunk.
        teacher_logits: Teacher logits with the same shape (already
            stop-gradient'd upstream).
        mask: Per-token loss mask matching the leading axes of the
            logits.
        safe_labels: Integer labels with ``-100`` replaced by ``0``
            (so the CE path is safe to gather even on masked
            positions).
        use_hard_labels: Whether to also accumulate the supervised
            CE term against ``safe_labels``.
        temperature: Softmax temperature forwarded to
            :func:`_per_token_xent`.
        dtype: Output dtype.

    Returns:
        ``(distill_xent_sum, teacher_entropy_sum, ce_sum, mask_sum)``;
        ``ce_sum`` is zero when ``use_hard_labels`` is ``False``.
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
    """Normalise accumulated distillation/CE sums into the final loss and metrics.

    Divides the scan-accumulated sums by the total mask weight,
    multiplies the distillation half by ``temperature**2`` (the
    canonical Hinton rescaling), and mixes the soft-target term with
    the supervised CE via ``alpha``.

    Args:
        distill_xent_sum: Accumulated masked sum of per-token
            distillation cross-entropy.
        teacher_entropy_sum: Accumulated masked sum of per-token
            teacher entropy.
        ce_sum: Accumulated masked sum of supervised CE
            (``0`` when ``use_hard_labels`` is ``False``).
        mask_sum: Total mask weight across all processed tokens.
        temperature: Softmax temperature (used to rescale the KL term
            by ``T^2``).
        alpha: KL / CE mixing weight; ``1.0`` is pure distillation.
        use_hard_labels: Whether the CE term should be folded into the
            total loss.
        dtype: Output dtype.

    Returns:
        ``(total_loss, metrics)`` where ``metrics`` is a dict carrying
        ``kl_loss``, ``distill_xent_loss``, ``teacher_entropy_loss``,
        and ``ce_loss`` as scalar arrays.
    """
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
    """Compose the per-token loss mask and produce a safe-label tensor.

    Priority for the base mask: ``loss_mask`` > ``attention_mask`` >
    all-ones. When ``labels`` are provided, positions with
    ``label == -100`` are zeroed out of the mask and the labels are
    cloned with ``-100`` replaced by ``0`` so they remain safe to
    gather even on masked positions.

    Args:
        attention_mask: Optional ``[batch, seq_len]`` padding mask.
        loss_mask: Optional task-specific token mask (takes precedence
            over ``attention_mask``).
        labels: Optional integer label tensor.
        dtype: Output dtype for the mask.
        seq_len: Sequence length (used for the all-ones fallback).
        batch_size: Batch size (used for the all-ones fallback).

    Returns:
        ``(mask, safe_labels, has_labels)``: combined per-token mask,
        a label tensor safe for the CE gather (zeros where
        ``has_labels`` is ``False``), and a Python bool indicating
        whether labels were supplied.
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


def _resolve_vocab_parallel_mesh(partition_manager: tp.Any) -> tp.Any:
    """Stage-local ``jax.Mesh`` for vocab-parallel reduction, or ``None``.

    Resolves the in-context mesh (``shard_map`` needs a raw ``jax.Mesh``; under MPMD/pipeline we want
    the per-stage submesh, not the global mpmd mesh). Returns ``None`` when no ``PartitionManager`` is
    supplied or the mesh has no real (>1) ``tp`` axis -- callers then fall back to the dense
    (non-sharded) kernel path, for which vocab-parallel would be a pure-overhead no-op.
    """
    if partition_manager is None:
        return None
    mesh = spx.get_current_stage_mesh(spx.get_incontext_mesh(raise_error=False), raise_error=False)
    # Engage vocab-parallel shard_map only for a real (>1) ``tp`` axis. At ``tp == 1`` there is no vocab
    # dimension to shard: the ``psum`` collapses to an identity (zero benefit) while still forcing every
    # leading [batch, seq] dim to be divisible by the meshed axes -- a divisibility constraint the dense
    # path never imposes. Gating on ``tp > 1`` keeps the path on for genuine tensor parallelism only.
    if mesh is not None and "tp" in mesh.axis_names and int(mesh.shape["tp"]) > 1:
        return mesh
    return None


def _vp_leading_pads(leading_shape: tuple[int, ...], token_spec: tp.Any, mesh: tp.Any) -> list[int]:
    """Per-dim padding so every meshed ``[batch, seq]`` dim divides evenly by the axes sharding it.

    ``jax.shard_map`` requires each meshed dim be evenly divisible by the product of its mesh axes; a
    non-divisible micro-batch (sharded over ``fsdp*dp``) or a sequence length not divisible by ``sp`` would
    otherwise raise. Returns the pad amount per leading dim (0 when already divisible; all Python ints,
    static at trace time).
    """

    def _axis_prod(entry: tp.Any) -> int:
        names = entry if isinstance(entry, tuple) else ((entry,) if entry is not None else ())
        prod = 1
        for nm in names:
            prod *= int(mesh.shape[nm])
        return prod

    return [
        (-int(d)) % max(_axis_prod(token_spec[i]) if i < len(token_spec) else 1, 1) for i, d in enumerate(leading_shape)
    ]


def _pad_leading(arr: Array, pads: list[int], fill: float = 0.0) -> Array:
    """Right-pad the leading ``len(pads)`` dims of ``arr`` (trailing vocab dim, if any, untouched)."""
    spec = [(0, int(p)) for p in pads] + [(0, 0)] * (arr.ndim - len(pads))
    return jnp.pad(arr, spec, constant_values=fill)


def distillation_loss(
    student_logits: Array,
    teacher_logits: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    labels: Array | None = None,
    use_hard_labels: bool = False,
    temperature: float = 4.0,
    alpha: float = 0.9,
    beta: float | None = None,
    loss_top_k: int = 0,
    loss_add_tail: bool = False,
    partition_manager: tp.Any = None,
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
    # The teacher is frozen in distillation: detach it so no gradient leaks into the
    # teacher branch (the fused-KL forward path only stop-grads in some temperature
    # regimes). No-op for real training where teacher params aren't trainable.
    teacher_logits = jax.lax.stop_gradient(teacher_logits)
    alpha_s = jnp.array(alpha, dtype=dtype)
    temp_sq = jnp.array(temperature * temperature, dtype=dtype)

    # Combined per-token loss mask: loss_mask > attention_mask, AND labels != -100.
    if loss_mask is not None:
        mask = loss_mask.astype(dtype)
    elif attention_mask is not None:
        mask = attention_mask.astype(dtype)
    else:
        mask = None
    if labels is not None:
        valid_label_mask = (labels != -100).astype(dtype)
        mask = valid_label_mask if mask is None else mask * valid_label_mask

    # Vocab-parallel reduction. When a ``PartitionManager`` is supplied (and the in-context mesh has a
    # real >1 ``tp`` axis), resolve the semantic ``[BATCH, LENGTH, VOCAB]`` / ``[BATCH, LENGTH]`` specs
    # and run the KL/CE inside a ``jax.shard_map`` so the vocabulary stays TP-sharded -- the full
    # ``[B, S, V]`` logits are never all-reduced (the large-vocab distillation OOM). The op infers the
    # vocab axis from the logit spec and forces ``check_vma=True`` for an exact gradient.
    _vp_mesh = _resolve_vocab_parallel_mesh(partition_manager)
    _vp_logit_spec = None
    _vp_token_spec = None
    if _vp_mesh is not None:
        _vp_logit_spec = partition_manager.resolve([BATCH, LENGTH, VOCAB], MODE_TRAIN)
        _vp_token_spec = partition_manager.resolve([BATCH, LENGTH], MODE_TRAIN)

    if loss_top_k > 0:
        # Teacher-top-k KD has no fused-kernel equivalent -> compute natively.
        per_token_distill_xent, per_token_teacher_entropy = _per_token_topk_xent(
            teacher_logits=teacher_logits,
            student_logits=student_logits,
            temperature=temperature,
            top_k=loss_top_k,
            add_tail=loss_add_tail,
            dtype=dtype,
        )
        per_token_divergence = per_token_distill_xent - per_token_teacher_entropy
        if mask is not None:
            normalizer = jnp.maximum(jnp.sum(mask), jnp.array(1.0, dtype=dtype))
            distill_xent_loss = jnp.sum(per_token_distill_xent * mask) / normalizer
            teacher_entropy_loss = jnp.sum(per_token_teacher_entropy * mask) / normalizer
            divergence_loss = jnp.sum(per_token_divergence * mask) / normalizer
        else:
            distill_xent_loss = jnp.mean(per_token_distill_xent)
            teacher_entropy_loss = jnp.mean(per_token_teacher_entropy)
            divergence_loss = jnp.mean(per_token_divergence)
        distill_xent_loss = distill_xent_loss * temp_sq
        teacher_entropy_loss = teacher_entropy_loss * temp_sq
        kl_loss = divergence_loss * temp_sq
    else:
        # ejkernel fused KL: forward KL by default (beta is None), or the generalized
        # JSD when beta is set. The kernel streams both softmaxes over the vocabulary
        # without materializing a [..., V] log-softmax and supplies the analytic student
        # gradient (teacher detached). The loss is already masked-mean * T^2.
        if beta is None:
            direction, want_entropy = "forward", True  # KL(p_t || p_s), report entropy split
        elif beta <= 0.0:
            direction, want_entropy = "reverse", False  # KL(p_s || p_t)
        elif beta >= 1.0:
            direction, want_entropy = "forward", False  # KL(p_t || p_s)
        else:
            direction, want_entropy = "jsd", False

        # Vocab-parallel KL only applies to forward KL (the kernel's vocab-parallel direction); the
        # all-reduce it removes -- two ~60GB ``[B, S, V]`` tensors for a 248K vocab -- is the large-vocab
        # distillation OOM cause.
        use_vocab_parallel = _vp_mesh is not None and direction == "forward"
        if use_vocab_parallel:
            # Vocab-parallel KL via the module op: passing mesh/in_specs/out_specs makes the op wrap the
            # XLA kernel in ``jax.shard_map`` and reduce the softmax normalizer with a ``psum`` over the TP
            # axis, so the full unsharded ``[B, S, V]`` logits are NEVER all-reduced. The op infers the vocab
            # axis from ``in_specs[0]`` (the resolved ``[BATCH, LENGTH, VOCAB]`` spec) and forces
            # ``check_vma=True``, which the vocab-parallel custom backward needs for an exact gradient
            # (verified value+grad parity vs dense for T=1/2/4). ``return_teacher_entropy`` is honored even
            # here: the op computes H(p_t) on the vocab-sharded ``teacher_logits`` outside shard_map, where
            # ``log_softmax`` keeps the tensor sharded and only the cheap ``[B, S]`` normalizer all-reduces
            # -- so the metric split (``distill_xent_loss``/``teacher_entropy_loss``) matches the dense path
            # without re-gathering the full ``[B, S, V]``. ``mask`` is passed straight through (``None`` when
            # there is none -- the op then needs no weights operand, and a non-None mask drives the kernel's
            # per-row sparse early-exit so masked rows are skipped).
            # Pad the leading [batch, seq] dims up to the meshed-axis product so shard_map never hits a
            # non-divisible micro-batch / sp-indivisible sequence; padded rows carry zero weight so they
            # drop out of the masked mean (and the teacher-entropy reduction).
            _kl_pads = _vp_leading_pads(student_logits.shape[:-1], _vp_token_spec, _vp_mesh)
            _kl_s, _kl_t, _kl_w = student_logits, teacher_logits, mask
            if any(_kl_pads):
                if _kl_w is None:
                    _kl_w = jnp.ones(student_logits.shape[:-1], dtype=jnp.float32)
                _kl_s = _pad_leading(student_logits, _kl_pads)
                _kl_t = _pad_leading(teacher_logits, _kl_pads)
                _kl_w = _pad_leading(_kl_w, _kl_pads, fill=0.0)
            out = _fused_kl(
                _kl_s,
                _kl_t,
                _kl_w,
                reduction="mean",
                direction="forward",
                temperature=temperature,
                return_teacher_entropy=want_entropy,
                platform="xla",
                mesh=_vp_mesh,
                in_specs=(_vp_logit_spec, _vp_logit_spec, _vp_token_spec),
                out_specs=PartitionSpec(),
            )
            kl_loss = out.loss.astype(dtype)
            if want_entropy:
                teacher_entropy_loss = out.teacher_entropy.astype(dtype)
                distill_xent_loss = kl_loss + teacher_entropy_loss
            else:
                teacher_entropy_loss = jnp.zeros((), dtype=dtype)
                distill_xent_loss = kl_loss
        else:
            out = _fused_kl(
                student_logits,
                teacher_logits,
                mask,
                reduction="mean",
                direction=direction,
                temperature=temperature,
                beta=(0.5 if beta is None else float(beta)),
                return_teacher_entropy=want_entropy,
                platform="xla",  # XLA is the bandwidth-optimal KL backend on TPU (Pallas loses)
            )
            kl_loss = out.loss.astype(dtype)
            if want_entropy:
                teacher_entropy_loss = out.teacher_entropy.astype(dtype)
                distill_xent_loss = kl_loss + teacher_entropy_loss
            else:
                teacher_entropy_loss = jnp.zeros((), dtype=dtype)
                distill_xent_loss = kl_loss

    total_loss = alpha_s * kl_loss
    ce_loss = jnp.array(0.0, dtype=dtype)
    if use_hard_labels and labels is not None:
        safe_labels = jnp.where(labels == -100, 0, labels)
        if _vp_mesh is not None:
            # Match the KL: with a tensor-parallel-sharded vocabulary, compute the supervised CE
            # vocab-parallel too via the module op (softmax normalizer + target-logit gather via psum
            # over the TP axis) so the CE never all-reduces the full [B, S, V] logits either. ``reduction
            # ="none"`` returns the per-token [B, S] loss (kept TP/data-sharded via ``out_specs``); the
            # masking/normalization below is shared with the dense path for exact parity. The op infers the
            # vocab axis from ``in_specs[0]`` and forces ``check_vma=True`` so the vocab-parallel custom
            # backward gives an exact gradient (verified value+grad vs dense). Pass the loss ``mask`` as the
            # weights (``None`` when there is none) so masked rows hit the kernel's per-row sparse early-exit
            # instead of paying the full softmax; the per-row mask multiply below is then idempotent.
            # Pad leading [batch, seq] up to the meshed-axis product (shard_map divisibility); padded labels
            # are ``-100`` (ignored) and the per-token loss is sliced back to [B, S] before the mask multiply.
            _ce_pads = _vp_leading_pads(student_logits.shape[:-1], _vp_token_spec, _vp_mesh)
            _ce_s = student_logits
            _ce_l = safe_labels.astype(jnp.int32)
            _ce_w = mask
            if any(_ce_pads):
                _ce_s = _pad_leading(student_logits, _ce_pads)
                _ce_l = _pad_leading(safe_labels.astype(jnp.int32), _ce_pads, fill=-100)
                _ce_w = _pad_leading(mask, _ce_pads, fill=0.0) if mask is not None else None
            ce_out = _fused_ce(
                _ce_s,
                _ce_l,
                _ce_w,
                reduction="none",
                ignore_index=-100,
                platform="xla",
                mesh=_vp_mesh,
                in_specs=(_vp_logit_spec, _vp_token_spec, _vp_token_spec),
                out_specs=_vp_token_spec,
            )
            per_token_ce = ce_out.loss.astype(dtype)
            if any(_ce_pads):
                per_token_ce = per_token_ce[tuple(slice(0, d) for d in student_logits.shape[:-1])]
        else:
            # Pass the loss ``mask`` as weights (``None`` when there is none) for the kernel's per-row
            # sparse early-exit; the per-row mask multiply below stays exact (idempotent for a 0/1 mask).
            ce_out = _fused_ce(
                student_logits,
                safe_labels.astype(jnp.int32),
                mask,
                reduction="none",
                ignore_index=-100,
                platform="xla",
            )
            per_token_ce = ce_out.loss.astype(dtype)

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


def mtp_distillation_loss(
    student_mtp_logits: Array,
    teacher_logits: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    temperature: float = 4.0,
    beta: float | None = None,
    partition_manager: tp.Any = None,
) -> Array:
    """Soft KD for a Multi-Token-Prediction head.

    The student's MTP head at position ``t`` predicts token ``t + 2``, i.e. the
    distribution ``P_student(x_{t+2} | x_{<=t+1})``. The *teacher's own ordinary
    next-token head* at position ``t + 1`` is exactly the same conditional,
    ``P_teacher(x_{t+2} | x_{<=t+1})`` — so the teacher needs no MTP head: we just
    shift its logits left by one and use them as the soft target. Alignment and
    masking mirror ``Qwen3_5ForCausalLM.compute_mtp_loss`` (shift-by-2 / ignore the
    trailing two positions, which have no valid ``t + 2`` target).

    Args:
        student_mtp_logits: ``(B, S, V)`` student MTP logits (predicting ``t + 2``).
        teacher_logits: ``(B, S, V)`` teacher next-token logits (predicting ``t + 1``).
        attention_mask: ``(B, S)`` padding mask (fallback when ``loss_mask`` is None).
        loss_mask: ``(B, S)`` completion/assistant mask (takes priority).
        temperature: Softmax temperature, shared with the main KD term.
        beta: Same divergence selector as :func:`distillation_loss` (``None`` ->
            forward KL, ``<=0`` -> reverse, ``>=1`` -> forward, else generalized JSD).

    Returns:
        Scalar masked-mean divergence (already scaled by ``T**2`` by the kernel).
    """
    dtype = student_mtp_logits.dtype
    b = student_mtp_logits.shape[0]
    # teacher target for position t = teacher_logits[:, t + 1]; pad the last column
    # (masked out) so shapes match (B, S, V).
    teacher_target = jnp.concatenate([teacher_logits[:, 1:], teacher_logits[:, -1:]], axis=1)
    base = loss_mask if loss_mask is not None else attention_mask
    if base is not None:
        base = base.astype(dtype)
        # mask[:, t] valid iff x_{t+2} is a real token -> base shifted by 2, trailing 2 zeroed.
        mtp_mask = jnp.concatenate([base[:, 2:], jnp.zeros((b, 2), dtype=dtype)], axis=1)
    else:
        mtp_mask = None

    if beta is None or beta >= 1.0:
        direction = "forward"
    elif beta <= 0.0:
        direction = "reverse"
    else:
        direction = "jsd"

    # Vocab-parallel forward KL via shard_map (same mechanism as ``distillation_loss``): keeps the
    # vocab TP-sharded so the full [B, S, V] MTP logits are never all-reduced. Forward direction only
    # (the kernel's vocab-parallel direction); reverse/JSD fall through to the dense path below.
    vp_mesh = _resolve_vocab_parallel_mesh(partition_manager) if direction == "forward" else None
    if vp_mesh is not None:
        kl_weights = mtp_mask if mtp_mask is not None else jnp.ones(student_mtp_logits.shape[:-1], dtype=jnp.float32)
        logit_spec = partition_manager.resolve([BATCH, LENGTH, VOCAB], MODE_TRAIN)
        token_spec = partition_manager.resolve([BATCH, LENGTH], MODE_TRAIN)
        # Pad leading [batch, seq] for shard_map divisibility; padded rows carry zero weight (masked mean).
        _pads = _vp_leading_pads(student_mtp_logits.shape[:-1], token_spec, vp_mesh)
        _s = student_mtp_logits
        _t = jax.lax.stop_gradient(teacher_target)
        if any(_pads):
            _s = _pad_leading(student_mtp_logits, _pads)
            _t = _pad_leading(_t, _pads)
            kl_weights = _pad_leading(kl_weights, _pads, fill=0.0)
        out = _fused_kl(
            _s,
            _t,
            kl_weights,
            reduction="mean",
            direction="forward",
            temperature=temperature,
            platform="xla",
            mesh=vp_mesh,
            in_specs=(logit_spec, logit_spec, token_spec),
            out_specs=PartitionSpec(),
        )
        return out.loss.astype(dtype)

    out = _fused_kl(
        student_mtp_logits,
        jax.lax.stop_gradient(teacher_target),
        mtp_mask,
        reduction="mean",
        direction=direction,
        temperature=temperature,
        beta=(0.5 if beta is None else float(beta)),
        return_teacher_entropy=False,
        platform="xla",
    )
    return out.loss.astype(dtype)


def mtp_chain_distillation_loss(
    chain_logits: Array,
    teacher_logits: Array,
    attention_mask: Array | None = None,
    loss_mask: Array | None = None,
    temperature: float = 4.0,
    beta: float | None = None,
    partition_manager: tp.Any = None,
) -> tuple[Array, list[Array]]:
    """Multi-step soft KD for a recursively-applied MTP head (FastMTP-style).

    Given ``chain_logits`` of shape ``(K, B, S, V)`` from
    ``Qwen3_5ForCausalLM.compute_mtp_chain`` — where step ``k`` (1-indexed) at
    position ``t`` predicts ``x_{t+k+1}`` — each step is distilled against the
    teacher's own next-token distribution at the matching offset: the teacher head
    at position ``t+k`` is exactly ``P(x_{t+k+1} | x_{<=t+k})``, so step ``k`` uses
    ``teacher_logits`` shifted left by ``k`` (with the trailing ``k+1`` positions
    masked — no valid target there). This trains the head to draft ``K`` tokens
    ahead, matching how the inference drafter recursively re-applies it.

    Args:
        chain_logits: ``(K, B, S, V)`` recursive MTP logits.
        teacher_logits: ``(B, S, V)`` teacher next-token logits.
        attention_mask / loss_mask: ``(B, S)`` masks (loss_mask takes priority).
        temperature: Shared softmax temperature.
        beta: Divergence selector (see :func:`distillation_loss`).

    Returns:
        ``(mean_kd_over_steps, [per_step_kd, ...])``.
    """
    dtype = chain_logits.dtype
    n_steps, b, s, _ = chain_logits.shape
    base = loss_mask if loss_mask is not None else attention_mask
    base = base.astype(dtype) if base is not None else None
    if beta is None or beta >= 1.0:
        direction = "forward"
    elif beta <= 0.0:
        direction = "reverse"
    else:
        direction = "jsd"

    # Vocab-parallel forward KL via shard_map (forward direction only -- the kernel's vocab-parallel
    # direction); keeps each step's [B, S, V] logits TP-sharded with no all-reduce. Same mesh/specs for
    # every step. reverse/JSD fall through to the dense path.
    vp_mesh = _resolve_vocab_parallel_mesh(partition_manager) if direction == "forward" else None
    vp_logit_spec = partition_manager.resolve([BATCH, LENGTH, VOCAB], MODE_TRAIN) if vp_mesh is not None else None
    vp_token_spec = partition_manager.resolve([BATCH, LENGTH], MODE_TRAIN) if vp_mesh is not None else None

    per_step: list[Array] = []
    total = jnp.zeros((), dtype)
    for j in range(int(n_steps)):
        k = j + 1
        # teacher target for step k = teacher_logits shifted left by k (pad tail, masked).
        tail = jnp.broadcast_to(teacher_logits[:, -1:], (b, k, teacher_logits.shape[-1]))
        teacher_target = jnp.concatenate([teacher_logits[:, k:], tail], axis=1)[:, :s]
        if base is not None:
            mask = jnp.concatenate([base[:, k + 1 :], jnp.zeros((b, k + 1), dtype=dtype)], axis=1)[:, :s]
        else:
            mask = None
        if vp_mesh is not None:
            kl_weights = mask if mask is not None else jnp.ones(chain_logits[j].shape[:-1], dtype=jnp.float32)
            # Pad leading [batch, seq] for shard_map divisibility; padded rows carry zero weight (masked mean).
            _pads = _vp_leading_pads(chain_logits[j].shape[:-1], vp_token_spec, vp_mesh)
            _cs = chain_logits[j]
            _ct = jax.lax.stop_gradient(teacher_target)
            if any(_pads):
                _cs = _pad_leading(chain_logits[j], _pads)
                _ct = _pad_leading(_ct, _pads)
                kl_weights = _pad_leading(kl_weights, _pads, fill=0.0)
            out = _fused_kl(
                _cs,
                _ct,
                kl_weights,
                reduction="mean",
                direction="forward",
                temperature=temperature,
                platform="xla",
                mesh=vp_mesh,
                in_specs=(vp_logit_spec, vp_logit_spec, vp_token_spec),
                out_specs=PartitionSpec(),
            )
        else:
            out = _fused_kl(
                chain_logits[j],
                jax.lax.stop_gradient(teacher_target),
                mask,
                reduction="mean",
                direction=direction,
                temperature=temperature,
                beta=(0.5 if beta is None else float(beta)),
                return_teacher_entropy=False,
                platform="xla",
            )
        step_loss = out.loss.astype(dtype)
        per_step.append(step_loss)
        total = total + step_loss
    return total / jnp.asarray(n_steps, dtype), per_step


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
    checkpoint_chunks: bool = True,
    hidden_partition_spec: PartitionSpec | None = None,
    hidden_shard_stage: int | None = None,
) -> tuple[Array, dict[str, Array]]:
    """Memory-efficient distillation loss that avoids materialising full logits.

    Instead of receiving pre-computed ``[B, L, V]`` logits, this function takes
    the last hidden states from both models and their lm_head projection
    functions. It processes the sequence in chunks of ``chunk_size`` tokens,
    projecting each chunk to vocab logits on-the-fly and immediately reducing
    to scalar KL / CE contributions. Peak logit memory drops from
    ``O(B * L * V)`` to ``O(B * chunk_size * V)``.

    When ``checkpoint_chunks`` is ``True`` (default) the per-chunk body is
    wrapped in ``jax.checkpoint`` so that during the backward pass each chunk's
    logits are *recomputed* from the hidden states rather than stored, keeping
    memory constant regardless of sequence length. Set it ``False`` to skip the
    recompute (faster backward) when every chunk's logits fit in memory at once.

    Distillation metrics follow:
    ``distill_xent_loss = E_t[-log p_s] * T^2``,
    ``teacher_entropy_loss = E_t[-log p_t] * T^2``,
    ``kl_loss = distill_xent_loss - teacher_entropy_loss``.

    Args:
        student_hidden: Student last-layer hidden states
            ``[B, L, hidden_dim]``.
        teacher_hidden: Teacher last-layer hidden states with the
            same shape (already stop-gradient'd by the caller).
        student_lm_head_fn: Callable projecting student hidden states
            to vocab logits.
        teacher_lm_head_fn: Callable projecting teacher hidden states
            to vocab logits.
        attention_mask: Optional ``[B, L]`` padding mask.
        loss_mask: Optional ``[B, L]`` task-specific token mask
            (takes precedence over ``attention_mask`` for masking).
        labels: Optional ``[B, L]`` integer labels for the supervised
            CE term.
        use_hard_labels: Whether to fold the supervised CE term into
            the total loss (gated additionally by ``labels`` being
            non-``None``).
        temperature: Softmax temperature.
        alpha: KL / CE mixing weight.
        chunk_size: Number of sequence positions per scan iteration.
        checkpoint_chunks: When ``True``, ``jax.checkpoint`` the
            per-chunk body so vocab-sized logits are recomputed in
            the backward pass.
        hidden_partition_spec: Optional sharding spec applied to the
            hidden-state slice before the LM-head projection (used to
            pin the slice to the LM-head's mesh).
        hidden_shard_stage: Optional MPMD stage rank used with the
            hidden-state sharding constraint.

    Returns:
        ``(total_loss, metrics)`` where ``metrics`` is a dict with
        ``kl_loss``, ``distill_xent_loss``, ``teacher_entropy_loss``,
        and ``ce_loss``.
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

    mask, safe_labels, has_labels = _build_mask_and_labels(
        attention_mask=attention_mask,
        loss_mask=loss_mask,
        labels=labels,
        dtype=dtype,
        seq_len=L_padded,
        batch_size=B,
    )

    # Reshape to [n_chunks, B, chunk_size, ...] for scanning while carrying
    # the incoming layout through the split/transpose.
    s_chunks = _chunk_sequence_for_scan(student_hidden, chunk_size, context="student_hidden")
    t_chunks = _chunk_sequence_for_scan(teacher_hidden, chunk_size, context="teacher_hidden")
    m_chunks = _chunk_sequence_for_scan(mask, chunk_size, context="distillation_mask")
    l_chunks = _chunk_sequence_for_scan(safe_labels, chunk_size, context="distillation_labels")

    _use_hard = use_hard_labels and has_labels

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
        if hidden_partition_spec is not None:
            s_h = with_sharding_constraint(s_h, hidden_partition_spec, stage=hidden_shard_stage)
            t_h = with_sharding_constraint(t_h, hidden_partition_spec, stage=hidden_shard_stage)
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

    if checkpoint_chunks:
        _chunk_kl_ce = jax.checkpoint(_chunk_kl_ce)

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

    Used to detach the teacher forward outputs from the autograd
    graph before they enter the student loss. Python leaves are left
    untouched so model output objects with non-array members survive.

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
    for key in (
        "teacher_logits",
        "teacher_hidden_for_kl",
        "teacher_hidden_states",
        "teacher_attentions",
        "_teacher_logits",
        "_teacher_hidden_for_kl",
        "_teacher_hidden_states",
        "_teacher_attentions",
    ):
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
    checkpoint_kl_loss: bool = True,
    beta: float | None = None,
    loss_top_k: int = 0,
    loss_add_tail: bool = False,
    mtp_distillation: bool = False,
    mtp_kd_weight: float = 0.3,
    mtp_draft_tokens: int = 1,
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
        checkpoint_kl_loss: When ``True`` (default) and the chunked path is active, wrap each
            chunk's KL/CE body in ``jax.checkpoint`` so its vocab-sized logits are recomputed in
            the backward pass instead of stored. Set ``False`` to skip the recompute.

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
    batch = with_sharding_constraint(batch, partition_spec)

    if hidden_state_loss != "mse":
        raise ValueError(f"Unsupported hidden state loss '{hidden_state_loss}'. Only 'mse' is available.")

    request_hidden_states = hidden_state_weight != 0.0
    request_attentions = attention_weight != 0.0
    use_advanced_vocab_loss = beta is not None or loss_top_k > 0 or loss_add_tail
    use_chunked = logits_chunk_size is not None and logits_chunk_size > 0 and not use_advanced_vocab_loss

    # SPMD tensor-parallel vocab path. With a >1 ``tp`` axis the model's LM head is row-parallel and
    # emits FULL ``[B, S, V]`` logits via a giant all-reduce (the large-vocab distillation OOM). Route
    # the projection through the hidden-state path with a COLUMN-PARALLEL (vocab-sharded) LM head
    # (``vocab_shard_stage=0`` -> constraint ``P(None, ("fsdp","sp","tp"))``) so the full vocab is never
    # materialized -- and do it as a single full-sequence "chunk" so there is no sequence chunking and
    # ``checkpoint_kl_loss`` (jax.checkpoint) recomputes the vocab-sized logits in the backward. Users
    # never need to set ``logits_chunk_size``; an explicit value still narrows the per-chunk peak.
    _stage_mesh = spx.get_current_stage_mesh(
        getattr(getattr(student_state, "model", None), "mesh", None), raise_error=False
    )
    _tp_parallel = _stage_mesh is not None and "tp" in _stage_mesh.axis_names and _stage_mesh.shape["tp"] > 1
    if _tp_parallel and not use_advanced_vocab_loss and not use_chunked and not mtp_distillation:
        # Single full-sequence pass through the column-parallel projection (below) -- no sequence
        # chunking; the vocab-sharded logits + jax.checkpoint keep the distillation KL bounded.
        # Skipped under ``mtp_distillation``: the MTP aux loss only runs on the non-chunked path
        # (``if mtp_distillation and not use_chunked``), so force-flipping here would silently drop it.
        logits_chunk_size = int(batch["input_ids"].shape[1])
        use_chunked = True

    def teacher_forward(input_batch: collections.abc.Mapping[str, jax.Array]) -> dict[str, tp.Any]:
        """Run the teacher in stop-gradient mode for one minibatch.

        The teacher is intentionally called from inside ``loss_fn`` so
        the compiled distillation step receives ``teacher_state``
        directly, matching the main-branch path.

        Args:
            input_batch: Full input batch dictionary.

        Returns:
            A dict with teacher logits / hidden states / attentions
            ready to be consumed by the distillation loss.

        Raises:
            TypeError: If the teacher does not return logits in the
                non-chunked code path.
        """
        result: dict[str, tp.Any] = {}
        if use_chunked:
            teacher_hidden_for_kl = input_batch.get("_teacher_hidden_for_kl", input_batch.get("teacher_hidden_for_kl"))
            if teacher_hidden_for_kl is not None:
                result["hidden_for_kl"] = jax.lax.stop_gradient(teacher_hidden_for_kl)
        else:
            teacher_logits = input_batch.get("_teacher_logits", input_batch.get("teacher_logits"))
            if teacher_logits is not None:
                result["logits"] = jax.lax.stop_gradient(teacher_logits)

        if result:
            teacher_hiddens = input_batch.get("_teacher_hidden_states", input_batch.get("teacher_hidden_states"))
            if request_hidden_states and teacher_hiddens is not None:
                result["hidden_states"] = _stop_gradient_tree(tuple(teacher_hiddens))
            teacher_attns = input_batch.get("_teacher_attentions", input_batch.get("teacher_attentions"))
            if request_attentions and teacher_attns is not None:
                result["attentions"] = _stop_gradient_tree(tuple(teacher_attns))
            return result

        teacher_call_kwargs = dict(input_batch)
        teacher_call_kwargs.pop("labels", None)
        teacher_call_kwargs.pop("completion_mask", None)
        teacher_call_kwargs.pop("assistant_masks", None)
        for key in (
            "teacher_logits",
            "teacher_hidden_for_kl",
            "teacher_hidden_states",
            "teacher_attentions",
            "_teacher_logits",
            "_teacher_hidden_for_kl",
            "_teacher_hidden_states",
            "_teacher_attentions",
        ):
            teacher_call_kwargs.pop(key, None)
        if use_chunked:
            teacher_call_kwargs["apply_lm_head"] = False
        if request_hidden_states:
            teacher_call_kwargs["output_hidden_states"] = True
        if request_attentions:
            teacher_call_kwargs["output_attentions"] = True
        teacher_call_kwargs = filter_kwargs_for_callable(teacher_state.model.__call__, teacher_call_kwargs)
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
            """Run the frozen teacher forward pass and stop-gradient outputs."""
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

    batch = dict(batch)

    def loss_fn(tree, minibatch):
        """Compute the distillation loss for one minibatch.

        Runs the student forward (with quantization-aware STE in
        training), runs the frozen teacher from ``teacher_state`` inside
        the same compiled distillation step, evaluates the KL/CE term,
        and adds the optional hidden-state and attention MSE terms.

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
        call_kwargs = filter_kwargs_for_callable(module.__call__, call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        student_outputs = module(**call_kwargs)
        labels = minibatch.get("labels", None)
        attention_mask = minibatch.get("attention_mask", None)
        completion_mask = minibatch.get("completion_mask", None)

        if use_chunked:
            # On a TP mesh, project the LM head column-parallel (vocab-sharded) so the full
            # ``[B, S, V]`` logits are never all-reduced; SPMD has a single stage so the shard stage
            # is 0. Off-TP keeps the legacy (row-parallel) projection.
            _vocab_stage = 0 if _tp_parallel else None
            total_loss, loss_components = chunked_distillation_loss(
                student_hidden=student_outputs.last_hidden_state,
                teacher_hidden=teacher_hidden_for_kl,
                student_lm_head_fn=module.make_lm_head_fn(vocab_shard_stage=_vocab_stage),
                teacher_lm_head_fn=teacher_state.model.make_lm_head_fn(vocab_shard_stage=_vocab_stage),
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
                chunk_size=int(logits_chunk_size),
                checkpoint_chunks=checkpoint_kl_loss,
                hidden_partition_spec=_LMHEAD_HIDDEN_PSPEC,
                hidden_shard_stage=_vocab_stage,
            )
        else:
            # If the vocabulary is tensor-parallel-sharded, route the KL/CE through the vocab-parallel
            # path so the full [B, S, V] logits are never all-reduced onto each device (the OOM cause for
            # large vocabularies). Pass the student's PartitionManager so distillation_loss can resolve the
            # [BATCH, LENGTH, VOCAB] / [BATCH, LENGTH] specs semantically and run the loss inside a
            # shard_map; it re-resolves the stage-local mesh from the in-context mesh. Gate on a real
            # (>1) TP axis -- on a non-TP mesh ``None`` keeps the simpler dense path.
            partition_manager = None
            stage_mesh = spx.get_current_stage_mesh(getattr(module, "mesh", None), raise_error=False)
            if stage_mesh is not None and "tp" in stage_mesh.axis_names and stage_mesh.shape["tp"] > 1:
                partition_manager = spx.PartitionManager(paxis=module.config.partition_axis)
            total_loss, loss_components = distillation_loss(
                student_logits=student_outputs.logits,
                teacher_logits=teacher_logits,
                attention_mask=attention_mask,
                loss_mask=completion_mask,
                labels=labels,
                use_hard_labels=(labels is not None),
                temperature=temperature,
                alpha=alpha,
                beta=beta,
                loss_top_k=loss_top_k,
                loss_add_tail=loss_add_tail,
                partition_manager=partition_manager,
            )
        metrics_map: dict[str, jax.Array] = dict(loss_components)

        # Include the student's auxiliary loss (Qwen3.5 self-supervised MTP CE folded in via
        # `mtp_loss_coef`, and/or MoE router losses) which the distillation loss otherwise drops.
        aux_loss = getattr(student_outputs, "aux_loss", None)
        if aux_loss is not None:
            aux_loss = aux_loss.astype(total_loss.dtype)
            total_loss = total_loss + aux_loss
            metrics_map["aux_loss"] = aux_loss

        # Soft MTP knowledge distillation: the teacher's next-token distribution at t+1
        # supervises the student's MTP head (predicting t+2). The model exposes the MTP
        # logits on the output (already computed for the aux loss), so no extra projection.
        if mtp_distillation and not use_chunked:
            compute_chain = getattr(module, "compute_mtp_chain", None)
            if mtp_draft_tokens > 1 and compute_chain is not None:
                # draft K tokens ahead: recursively apply the MTP head and distill
                # each step against the teacher's next-token dist at the matching offset.
                chain_logits = compute_chain(
                    student_outputs,
                    minibatch.get("input_ids"),
                    int(mtp_draft_tokens),
                    attention_mask=attention_mask,
                )
                if chain_logits is not None:
                    mtp_kd_value, mtp_per_step = mtp_chain_distillation_loss(
                        chain_logits=chain_logits,
                        teacher_logits=teacher_logits,
                        attention_mask=attention_mask,
                        loss_mask=completion_mask,
                        temperature=temperature,
                        beta=beta,
                        partition_manager=spx.PartitionManager(paxis=module.config.partition_axis),
                    )
                    mtp_kd_value = mtp_kd_value.astype(total_loss.dtype)
                    total_loss = total_loss + jnp.asarray(mtp_kd_weight, dtype=total_loss.dtype) * mtp_kd_value
                    metrics_map["mtp_kd_loss"] = mtp_kd_value
                    for _i, _ps in enumerate(mtp_per_step):
                        metrics_map[f"mtp_kd_step{_i + 1}"] = _ps.astype(total_loss.dtype)
            else:
                student_mtp_logits = getattr(student_outputs, "mtp_logits", None)
                if student_mtp_logits is not None:
                    mtp_kd_value = mtp_distillation_loss(
                        student_mtp_logits=student_mtp_logits,
                        teacher_logits=teacher_logits,
                        attention_mask=attention_mask,
                        loss_mask=completion_mask,
                        temperature=temperature,
                        beta=beta,
                        partition_manager=spx.PartitionManager(paxis=module.config.partition_axis),
                    )
                    mtp_kd_value = mtp_kd_value.astype(total_loss.dtype)
                    total_loss = total_loss + jnp.asarray(mtp_kd_weight, dtype=total_loss.dtype) * mtp_kd_value
                    metrics_map["mtp_kd_loss"] = mtp_kd_value

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


_LMHEAD_HIDDEN_PSPEC = PartitionSpec(("dp", "fsdp"), "sp", None)


def _prepare_distillation_scheduled_batch(call) -> dict[str, tp.Any]:
    """Return the scheduled batch unchanged.

    Distillation follows the main trainer design: ``teacher_state`` is passed
    into the compiled step and the frozen teacher forward is executed inside
    that step. Preparing teacher logits here would compile/run an extra
    auxiliary JAX program before the student step and changes the contract.

    Args:
        call: The :class:`ScheduledStepCall` describing the current
            step.

    Returns:
        A copy of ``call.batch``.
    """
    return dict(call.batch)


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
            "checkpoint_kl_loss",
            "beta",
            "loss_top_k",
            "loss_add_tail",
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
    checkpoint_kl_loss = bool(call.get("checkpoint_kl_loss", True))
    beta = call.get("beta")
    loss_top_k = int(call.get("loss_top_k", 0) or 0)
    loss_add_tail = bool(call.get("loss_add_tail", False))
    use_advanced_vocab_loss = beta is not None or loss_top_k > 0 or loss_add_tail
    use_chunked = logits_chunk_size is not None and logits_chunk_size > 0 and not use_advanced_vocab_loss
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
        with jax.named_scope("easydel/trainer/distillation/scheduled_loss/bind_module"):
            module = bind_scheduled_module(call, tree)
            call_batch = _constrain_distillation_input_batch(batch, partition_spec, mesh=module.mesh)
        with jax.named_scope("easydel/trainer/distillation/scheduled_loss/student_forward"):
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

        with jax.named_scope("easydel/trainer/distillation/scheduled_loss/teacher_forward"):
            if teacher_state is None:
                raise RuntimeError("Distillation scheduled MPMD training requires teacher_state.")
            teacher_module = teacher_state.merge(teacher_state.graphstate)
            sync_module_schedule_config(teacher_module, call.schedule)
            teacher_outputs = stop_gradient_tree(
                _distillation_forward_outputs(
                    teacher_module,
                    call_batch,
                    use_chunked=use_chunked,
                    request_hidden_states=request_hidden_states,
                    request_attentions=request_attentions,
                )
            )
            teacher_lm_head_module = teacher_module

        with jax.named_scope("easydel/trainer/distillation/scheduled_loss/distillation_loss"):
            if use_chunked:
                if teacher_lm_head_module is None:
                    raise RuntimeError("Chunked distillation scheduled MPMD training requires teacher_state.")
                _terminal_rank = _scheduled_terminal_stage_rank(module, call.schedule)
                total_loss, _loss_components = chunked_distillation_loss(
                    student_hidden=student_outputs["hidden_for_kl"],
                    teacher_hidden=teacher_outputs["hidden_for_kl"],
                    student_lm_head_fn=module.make_lm_head_fn(vocab_shard_stage=_terminal_rank),
                    teacher_lm_head_fn=teacher_lm_head_module.make_lm_head_fn(vocab_shard_stage=_terminal_rank),
                    attention_mask=attention_mask,
                    loss_mask=completion_mask,
                    labels=labels,
                    use_hard_labels=(labels is not None),
                    temperature=temperature,
                    alpha=alpha,
                    chunk_size=int(logits_chunk_size),
                    checkpoint_chunks=checkpoint_kl_loss,
                    hidden_partition_spec=(_LMHEAD_HIDDEN_PSPEC if _terminal_rank is not None else None),
                    hidden_shard_stage=_terminal_rank,
                )
            else:
                total_loss, _loss_components = distillation_loss(
                    student_logits=student_outputs["logits"],
                    teacher_logits=teacher_outputs["logits"],
                    attention_mask=attention_mask,
                    loss_mask=completion_mask,
                    labels=labels,
                    use_hard_labels=(labels is not None),
                    temperature=temperature,
                    alpha=alpha,
                    beta=beta,
                    loss_top_k=loss_top_k,
                    loss_add_tail=loss_add_tail,
                )

        if request_hidden_states:
            with jax.named_scope("easydel/trainer/distillation/scheduled_loss/hidden_state_loss"):
                student_hidden = student_outputs.get("hidden_states")
                teacher_hiddens = teacher_outputs.get("hidden_states")
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
            with jax.named_scope("easydel/trainer/distillation/scheduled_loss/attention_loss"):
                student_attentions = student_outputs.get("attentions")
                teacher_attns = teacher_outputs.get("attentions")
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
    step_fn=distillation_step,
    adapter=ScheduledLossAdapter(
        name="distillation",
        make_loss=_make_distillation_scheduled_loss,
        make_cache_key=_distillation_scheduled_loss_cache_key,
        prepare_batch=_prepare_distillation_scheduled_batch,
    ),
)
