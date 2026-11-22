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

"""Internal functions for Self-Distillation Policy Optimization (SDPO).

The key idea is to use the *current* policy in two roles simultaneously:

- **Student** - the policy prompted only with the original question ``x``.
  Receives gradient updates.
- **Self-teacher** - the *same* policy prompted with ``(x, feedback, y)``,
  i.e., the question plus environment feedback plus the student's original
  attempt.  Evaluated under ``stop_gradient`` so no updates flow through it.

The distillation loss minimises the divergence between the student's next-token
distribution and the self-teacher's feedback-conditioned distribution.

Because computing full-vocabulary KL/JSD every step is expensive, this module
uses a sampled-token surrogate objective with a detached distillation weight:

    w_t = stop_gradient(student_logp_t - target_logp_t)
    L_t = w_t * student_logp_t

where ``target_logp_t`` is the teacher log-prob (KL) or the log-mixture term
for JSD. This preserves the correct update direction (increase student log-prob
when teacher is higher, decrease when teacher is lower) while remaining cheap.
"""

import collections.abc
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..group_relative_policy_optimization._fn import _global_token_normalizer, get_per_token_logps
from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)


def _apply_completion_loss_token_skip(completion_mask: jax.Array, num_loss_tokens_to_skip: int) -> jax.Array:
    """Mask the first completion tokens out of the SDPO/SDFT loss."""
    if num_loss_tokens_to_skip <= 0:
        return completion_mask
    token_positions = jnp.arange(completion_mask.shape[1])
    keep_mask = token_positions[None, :] >= int(num_loss_tokens_to_skip)
    return completion_mask * keep_mask.astype(completion_mask.dtype)


def _completion_logits(model, input_ids: jax.Array, attention_mask: jax.Array, prompt_length: int) -> jax.Array:
    """Return next-token logits aligned with generated completion tokens."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    if logits is None:
        raise TypeError(f"{type(model).__name__} did not return logits for full-logit SDPO.")
    return logits[:, prompt_length - 1 : -1, :]


def _gather_token_logps(log_probs: jax.Array, token_ids: jax.Array) -> jax.Array:
    """Gather log-probabilities for the sampled completion tokens."""
    return jnp.take_along_axis(log_probs, token_ids[..., None], axis=-1).squeeze(-1)


def _topk_vocab_divergence(
    student_log_probs: jax.Array,
    teacher_log_probs: jax.Array,
    *,
    top_k: int,
    add_tail: bool,
    distillation_type: str = "kl",
) -> jax.Array:
    """Compute top-k KL(teacher || student) or symmetric JSD with an optional tail bucket."""
    if distillation_type not in ("kl", "jsd"):
        raise ValueError(f"Unknown distillation_type '{distillation_type}'. Must be 'kl' or 'jsd'.")
    vocab_size = int(teacher_log_probs.shape[-1])
    k = min(max(int(top_k), 1), vocab_size)
    top_teacher_log_probs, top_indices = jax.lax.top_k(teacher_log_probs, k)
    top_student_log_probs = jnp.take_along_axis(student_log_probs, top_indices, axis=-1)

    if add_tail and k < vocab_size:
        top_teacher_probs = jnp.exp(top_teacher_log_probs)
        top_student_probs = jnp.exp(top_student_log_probs)
        eps = jnp.asarray(1e-7, dtype=student_log_probs.dtype)
        teacher_tail_prob = jnp.clip(1.0 - jnp.sum(top_teacher_probs, axis=-1), eps, 1.0)
        student_tail_prob = jnp.clip(1.0 - jnp.sum(top_student_probs, axis=-1), eps, 1.0)
        teacher_tail_logp = jnp.log(teacher_tail_prob)
        student_tail_logp = jnp.log(student_tail_prob)
        if distillation_type == "kl":
            return jnp.sum(
                top_teacher_probs * (top_teacher_log_probs - top_student_log_probs), axis=-1
            ) + teacher_tail_prob * (teacher_tail_logp - student_tail_logp)
        teacher_bucket_logp = jnp.concatenate([top_teacher_log_probs, teacher_tail_logp[..., None]], axis=-1)
        student_bucket_logp = jnp.concatenate([top_student_log_probs, student_tail_logp[..., None]], axis=-1)
        teacher_bucket_probs = jnp.concatenate([top_teacher_probs, teacher_tail_prob[..., None]], axis=-1)
        student_bucket_probs = jnp.concatenate([top_student_probs, student_tail_prob[..., None]], axis=-1)
        mixture_bucket_logp = jnp.logaddexp(student_bucket_logp, teacher_bucket_logp) - jnp.log(2.0)
        teacher_kl = jnp.sum(teacher_bucket_probs * (teacher_bucket_logp - mixture_bucket_logp), axis=-1)
        student_kl = jnp.sum(student_bucket_probs * (student_bucket_logp - mixture_bucket_logp), axis=-1)
        return 0.5 * (teacher_kl + student_kl)

    teacher_support_log_probs = jax.nn.log_softmax(top_teacher_log_probs, axis=-1)
    student_support_log_probs = jax.nn.log_softmax(top_student_log_probs, axis=-1)
    teacher_support_probs = jnp.exp(teacher_support_log_probs)
    if distillation_type == "kl":
        return jnp.sum(teacher_support_probs * (teacher_support_log_probs - student_support_log_probs), axis=-1)
    student_support_probs = jnp.exp(student_support_log_probs)
    mixture_support_log_probs = jnp.logaddexp(student_support_log_probs, teacher_support_log_probs) - jnp.log(2.0)
    teacher_kl = jnp.sum(teacher_support_probs * (teacher_support_log_probs - mixture_support_log_probs), axis=-1)
    student_kl = jnp.sum(student_support_probs * (student_support_log_probs - mixture_support_log_probs), axis=-1)
    return 0.5 * (teacher_kl + student_kl)


def _full_vocab_sdpo_loss(
    *,
    student_logits: jax.Array,
    teacher_logits: jax.Array,
    completion_ids: jax.Array,
    distillation_type: str,
    distillation_topk: int | None,
    distillation_add_tail: bool,
    distillation_clip: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute full-vocabulary or top-k SDPO divergence per completion token.

    Note on KL direction: full-vocab ``"kl"`` computes forward
    ``KL(teacher || student)`` (standard knowledge distillation), whereas the
    sampled-token surrogate computes the on-policy reverse
    ``KL(student || teacher)`` REINFORCE estimator. ``"jsd"`` is symmetric in
    both modes with matching ``0.5`` gradient scale.
    """
    student_log_probs = jax.nn.log_softmax(student_logits.astype(jnp.float32), axis=-1)
    teacher_log_probs = jax.lax.stop_gradient(jax.nn.log_softmax(teacher_logits.astype(jnp.float32), axis=-1))
    if distillation_topk is not None:
        per_token = _topk_vocab_divergence(
            student_log_probs,
            teacher_log_probs,
            top_k=distillation_topk,
            add_tail=distillation_add_tail,
            distillation_type=distillation_type,
        )
    elif distillation_type == "kl":
        teacher_probs = jnp.exp(teacher_log_probs)
        per_token = jnp.sum(teacher_probs * (teacher_log_probs - student_log_probs), axis=-1)
    elif distillation_type == "jsd":
        mixture_log_probs = jnp.logaddexp(student_log_probs, teacher_log_probs) - jnp.log(2.0)
        teacher_probs = jnp.exp(teacher_log_probs)
        student_probs = jnp.exp(student_log_probs)
        teacher_kl = jnp.sum(teacher_probs * (teacher_log_probs - mixture_log_probs), axis=-1)
        student_kl = jnp.sum(student_probs * (student_log_probs - mixture_log_probs), axis=-1)
        per_token = 0.5 * (teacher_kl + student_kl)
    else:
        raise ValueError(f"Unknown distillation_type '{distillation_type}'. Must be 'kl' or 'jsd'.")

    if distillation_clip is not None:
        per_token = jnp.minimum(per_token, jnp.asarray(distillation_clip, dtype=per_token.dtype))
    return (
        per_token,
        _gather_token_logps(student_log_probs, completion_ids),
        _gather_token_logps(teacher_log_probs, completion_ids),
    )


def _sampled_token_sdpo_loss(
    *,
    student_logps: jax.Array,
    teacher_logps: jax.Array,
    distillation_type: str,
    distillation_weight: float,
    distillation_is_clip: float | None,
) -> jax.Array:
    """Compute sampled-token SDPO surrogate loss.

    For ``distillation_type="jsd"``, the exact policy-gradient of symmetric
    ``JSD(S, T) = 0.5 * KL(T || M) + 0.5 * KL(S || M)`` is
    ``0.5 * E_{y~S}[(log S(y) - log M(y)) grad log S(y)]``, so the ``0.5``
    factor keeps sampled-token JSD gradient scale aligned with full-vocab JSD.
    """
    if distillation_type == "kl":
        target_logps = teacher_logps
        scale = 1.0
    elif distillation_type == "jsd":
        target_logps = jnp.logaddexp(student_logps, teacher_logps) - jnp.log(2.0)
        scale = 0.5
    else:
        raise ValueError(f"Unknown distillation_type '{distillation_type}'. Must be 'kl' or 'jsd'.")
    distill_weight = jax.lax.stop_gradient(scale * (student_logps - target_logps))
    if distillation_is_clip is not None:
        clip_value = jnp.asarray(distillation_is_clip, dtype=distill_weight.dtype)
        distill_weight = jnp.clip(distill_weight, -clip_value, clip_value)
    return distillation_weight * distill_weight * student_logps


def _safe_masked_k3_kl(
    ref_per_token_logps: jax.Array,
    student_logps: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Compute overflow-safe k3 KL ``exp(ref - student) - (ref - student) - 1``."""
    kl_delta = ref_per_token_logps.astype(jnp.float32) - student_logps.astype(jnp.float32)
    safe_delta = jnp.where(mask > 0, jnp.clip(kl_delta, -80.0, 80.0), 0.0)
    return jnp.exp(safe_delta) - safe_delta - 1.0


def sdpo_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    num_generations: int,
    teacher_prompt_length: int,
    beta: float,
    distillation_type: str,
    distillation_weight: float,
    logprob_vocab_chunk_size: int | None,
    max_loss_completion_tokens: int | None,
    completion_chunk_size: int | None,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    num_loss_tokens_to_skip: int = 0,
    full_logit_distillation: bool = False,
    distillation_topk: int | None = None,
    distillation_is_clip: float | None = None,
    distillation_add_tail: bool = False,
    teacher_state: EasyDeLState | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Single SDPO training / evaluation step.

    The batch must contain the following arrays (produced by
    :meth:`SDPOTrainer._preprocess_batch_input`):

    - ``prompt_ids``         : ``[B, prompt_len]``
    - ``prompt_mask``        : ``[B, prompt_len]``
    - ``completion_ids``     : ``[B*G, comp_len]``
    - ``completion_mask``    : ``[B*G, comp_len]``
    - ``teacher_ids``        : ``[B*G, teacher_len]`` -
        full teacher context = left_padded(prompt || feedback) || completion
    - ``teacher_mask``       : ``[B*G, teacher_len]``
    - ``num_items_in_batch`` : scalar (total completion tokens, for loss normalisation)

    Optionally (when ``beta > 0``):

    - ``ref_per_token_logps``: ``[B*G, comp_len]`` - frozen-reference log-probs

    Args:
        state: Current EasyDeL model/optimiser state.
        batch: Pre-processed batch from :meth:`SDPOTrainer._preprocess_batch_input`.
        num_generations: Number of completions sampled per prompt (``G``). **STATIC**.
        teacher_prompt_length: Nominal length of the teacher prefix
            (``max_prompt_length + effective_feedback_length``). The actual
            prefix length is derived dynamically from
            ``teacher_ids.shape[1] - completion_ids.shape[1]`` so runtime
            prompt-length variation never shifts teacher scoring. **STATIC**.
        beta: Weight of KL penalty toward the frozen reference model.
            Set to 0 to disable (default for SDPO). **STATIC**.
        distillation_type: ``'kl'`` or ``'jsd'``. **STATIC**.
        distillation_weight: Weight applied to the sampled-token
            self-distillation loss. **STATIC**.
        max_loss_completion_tokens: Optional cap on completion tokens used by
            the SDPO loss. When set, both the student and teacher scoring
            paths are truncated to the first ``max_loss_completion_tokens``
            completion tokens so the compiled graph does not scale with the
            full sampled completion length. **STATIC**.
        completion_chunk_size: Optional cap on the number of completions scored
            in a single inner SDPO loss pass. When set, the student/teacher
            scoring work is split across several smaller completion batches to
            reduce peak activation and temporary-buffer memory (supported for
            both sampled-token and full-logit/top-k distillation). **STATIC**.
        loss_config: Optional loss / gradient-clipping configuration.
        learning_rate_fn: Learning-rate schedule used for metric logging.
        partition_spec: Partition spec for sharding the batch.
        gradient_accumulation_steps: Number of minibatch accumulation steps.
        is_training: Whether to compute and apply gradients.
        straight_through_emulator: Optional STE for quantisation-aware training.
        teacher_state: Optional separate EMA teacher state (used when
            ``teacher_regularization == "ema"``). When ``None``, the current
            policy parameters are evaluated under ``stop_gradient``.

    Returns:
        ``(updated_state, metrics)`` when ``is_training=True``, or just
        ``metrics`` when ``is_training=False``.
    """
    scope_root = "easydel/trainer/sdpo/" + ("train_step" if is_training else "eval_step")
    effective_ga_steps = gradient_accumulation_steps if is_training else 1
    with jax.named_scope(scope_root + "/prepare_batch"):
        # Align every example-dependent leaf before minibatch_call slices the
        # completion axis. Keep accepting the public [B, P] prompt layout as well
        # as already-expanded [B*G, P] batches.
        batch = dict(batch)
        prompt_rows = batch["prompt_ids"].shape[0]
        completion_rows = batch["completion_ids"].shape[0]
        if prompt_rows != completion_rows:
            if num_generations <= 0 or prompt_rows * num_generations != completion_rows:
                raise ValueError("SDPO prompt rows must match completion rows or expand by num_generations.")
            batch["prompt_ids"] = batch["prompt_ids"].repeat(num_generations, axis=0)
            batch["prompt_mask"] = batch["prompt_mask"].repeat(num_generations, axis=0)
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=effective_ga_steps,
            batch_partition_spec=partition_spec,
        )
        state_model = getattr(state, "model", None)
        state_mesh = getattr(state_model, "mesh", None)
        batch = with_sharding_constraint(batch, partition_spec, mesh=state_mesh, ignore_mpmd=True)

    def loss_fn(tree, minibatch):
        """Compute the SDPO loss for a single minibatch.

        The student (current policy parameters under gradient) and the
        self-teacher (either ``teacher_state`` when EMA regularization is
        active, or the student parameters under ``stop_gradient``) are scored
        either in one shot or in completion-axis chunks (when
        ``completion_chunk_size`` is set).
        """
        if is_training and straight_through_emulator is not None:
            with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                tree = straight_through_emulator(tree)
        with jax.named_scope(scope_root + "/loss_fn/merge_state"):
            module = state.merge(tree)
            if teacher_state is not None:
                teacher_graphstate = jax.tree_util.tree_map(
                    lambda x: jax.lax.stop_gradient(x) if hasattr(x, "shape") else x,
                    teacher_state.graphstate,
                )
                teacher_module = teacher_state.merge(teacher_graphstate)
            else:
                teacher_module = module

        prompt_ids = minibatch["prompt_ids"]
        prompt_mask = minibatch["prompt_mask"]
        completion_ids = minibatch["completion_ids"]
        completion_mask = minibatch["completion_mask"]
        teacher_ids = minibatch["teacher_ids"]
        teacher_mask = minibatch["teacher_mask"]

        raw_completion_len = int(completion_ids.shape[1])
        derived_teacher_prompt_length = int(teacher_ids.shape[1]) - raw_completion_len
        if derived_teacher_prompt_length < 0:
            raise ValueError(
                f"SDPO teacher_ids length ({teacher_ids.shape[1]}) must be >= completion_ids length "
                f"({raw_completion_len})."
            )
        actual_teacher_prompt_length = (
            derived_teacher_prompt_length if derived_teacher_prompt_length > 0 else int(teacher_prompt_length)
        )

        completion_was_truncated = False
        if max_loss_completion_tokens is not None and completion_ids.shape[1] > max_loss_completion_tokens:
            completion_ids = completion_ids[:, :max_loss_completion_tokens]
            completion_mask = completion_mask[:, :max_loss_completion_tokens]
            teacher_ids = teacher_ids[:, : actual_teacher_prompt_length + max_loss_completion_tokens]
            teacher_mask = teacher_mask[:, : actual_teacher_prompt_length + max_loss_completion_tokens]
            completion_was_truncated = True
        # The skip only removes tokens from the loss; the student must still attend to them, so keep the
        # unskipped mask for attention and use the skipped copy for loss weighting / token counts.
        completion_attn_mask = completion_mask
        completion_mask = _apply_completion_loss_token_skip(completion_mask, num_loss_tokens_to_skip)

        prompt_len = prompt_ids.shape[-1]
        completion_token_count = jnp.sum(completion_mask)
        # `num_items_in_batch` is the full-batch token count (a 0-dim scalar that
        # minibatch_call does not slice). Divide it by gradient_accumulation_steps
        # only when training so the K per-microbatch losses recombine correctly
        # after minibatch_call's 1/K averaging, without inflating eval losses by Kx.
        num_items = _global_token_normalizer(
            minibatch.get("num_items_in_batch"),
            completion_token_count,
            completion_was_truncated,
            gradient_accumulation_steps,
            is_training=is_training,
        )

        use_full_vocab_loss = full_logit_distillation or distillation_topk is not None
        use_chunked_completion_loss = (
            completion_chunk_size is not None and completion_ids.shape[0] > completion_chunk_size
        )
        if use_chunked_completion_loss:
            expanded_prompt_ids = prompt_ids
            expanded_prompt_mask = prompt_mask
            completion_batch_size = int(completion_ids.shape[0])
            loss_numerator = jnp.array(0.0, dtype=jnp.float32)
            advantage_num = jnp.array(0.0, dtype=jnp.float32)
            advantage_pos_num = jnp.array(0.0, dtype=jnp.float32)
            student_logps_num = jnp.array(0.0, dtype=jnp.float32)
            teacher_logps_num = jnp.array(0.0, dtype=jnp.float32)
            per_token_loss_num = jnp.array(0.0, dtype=jnp.float32)
            mean_kl_num = jnp.array(0.0, dtype=jnp.float32)
            ref_logps_num = jnp.array(0.0, dtype=jnp.float32)

            for start in range(0, completion_batch_size, completion_chunk_size):
                end = min(start + completion_chunk_size, completion_batch_size)
                chunk_completion_ids = completion_ids[start:end]
                chunk_completion_mask = completion_mask[start:end]
                chunk_student_input_ids = jnp.concatenate([expanded_prompt_ids[start:end], chunk_completion_ids], axis=1)
                chunk_student_attn_mask = jnp.concatenate(
                    [expanded_prompt_mask[start:end], completion_attn_mask[start:end]], axis=1
                )
                if use_full_vocab_loss:
                    with jax.named_scope(scope_root + "/loss_fn/chunked/full_vocab_logits"):
                        chunk_student_logits = _completion_logits(
                            module,
                            chunk_student_input_ids,
                            chunk_student_attn_mask,
                            prompt_len,
                        )
                        chunk_teacher_logits = jax.lax.stop_gradient(
                            _completion_logits(
                                teacher_module,
                                teacher_ids[start:end],
                                teacher_mask[start:end],
                                actual_teacher_prompt_length,
                            )
                        )
                    with jax.named_scope(scope_root + "/loss_fn/chunked/full_vocab_sdpo_loss"):
                        chunk_per_token_loss, chunk_student_logps, chunk_teacher_logps = _full_vocab_sdpo_loss(
                            student_logits=chunk_student_logits,
                            teacher_logits=chunk_teacher_logits,
                            completion_ids=chunk_completion_ids,
                            distillation_type=distillation_type,
                            distillation_topk=distillation_topk,
                            distillation_add_tail=distillation_add_tail,
                            distillation_clip=distillation_is_clip,
                        )
                        chunk_per_token_loss = distillation_weight * chunk_per_token_loss
                else:
                    with jax.named_scope(scope_root + "/loss_fn/chunked/student_logps"):
                        chunk_student_logps = get_per_token_logps(
                            module,
                            chunk_student_input_ids,
                            chunk_student_attn_mask,
                            prompt_len,
                            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                        )
                    with jax.named_scope(scope_root + "/loss_fn/chunked/teacher_logps"):
                        chunk_teacher_logps = jax.lax.stop_gradient(
                            get_per_token_logps(
                                teacher_module,
                                teacher_ids[start:end],
                                teacher_mask[start:end],
                                actual_teacher_prompt_length,
                                logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                            )
                        )
                    chunk_per_token_loss = _sampled_token_sdpo_loss(
                        student_logps=chunk_student_logps,
                        teacher_logps=chunk_teacher_logps,
                        distillation_type=distillation_type,
                        distillation_weight=distillation_weight,
                        distillation_is_clip=distillation_is_clip,
                    )

                if beta != 0.0:
                    ref_per_token_logps = minibatch["ref_per_token_logps"][start:end]
                    if ref_per_token_logps.shape[1] != chunk_completion_ids.shape[1]:
                        ref_per_token_logps = ref_per_token_logps[:, : chunk_completion_ids.shape[1]]
                    chunk_per_token_kl = _safe_masked_k3_kl(
                        ref_per_token_logps,
                        chunk_student_logps,
                        chunk_completion_mask,
                    )
                    chunk_per_token_loss = chunk_per_token_loss + beta * chunk_per_token_kl
                    mean_kl_num = mean_kl_num + jnp.sum(chunk_per_token_kl * chunk_completion_mask)
                    ref_logps_num = ref_logps_num + jnp.sum(ref_per_token_logps * chunk_completion_mask)

                chunk_advantage = chunk_teacher_logps - chunk_student_logps
                loss_numerator = loss_numerator + jnp.sum(chunk_per_token_loss * chunk_completion_mask)
                advantage_num = advantage_num + jnp.sum(chunk_advantage * chunk_completion_mask)
                advantage_pos_num = advantage_pos_num + jnp.sum(
                    (chunk_advantage > 0).astype(jnp.float32) * chunk_completion_mask
                )
                student_logps_num = student_logps_num + jnp.sum(chunk_student_logps * chunk_completion_mask)
                teacher_logps_num = teacher_logps_num + jnp.sum(chunk_teacher_logps * chunk_completion_mask)
                per_token_loss_num = per_token_loss_num + jnp.sum(chunk_per_token_loss * chunk_completion_mask)

            loss = loss_numerator / num_items
            token_denom = jnp.maximum(completion_token_count, 1.0)
            other_metrics: dict[str, jax.Array] = {
                "sdpo/advantage_mean": advantage_num / token_denom,
                "sdpo/advantage_pos_frac": advantage_pos_num / token_denom,
                "sdpo/student_logps": student_logps_num / token_denom,
                "sdpo/teacher_logps": teacher_logps_num / token_denom,
                "sdpo/per_token_loss": per_token_loss_num / token_denom,
            }
            if beta != 0.0:
                other_metrics["mean_kl"] = mean_kl_num / token_denom
                other_metrics["ref_per_token_logps"] = ref_logps_num / token_denom
        else:
            student_input_ids = jnp.concatenate([prompt_ids, completion_ids], axis=1)
            student_attn_mask = jnp.concatenate([prompt_mask, completion_attn_mask], axis=1)
            if use_full_vocab_loss:
                with jax.named_scope(scope_root + "/loss_fn/full_vocab_logits"):
                    student_logits = _completion_logits(
                        module,
                        student_input_ids,
                        student_attn_mask,
                        prompt_len,
                    )
                    teacher_logits = jax.lax.stop_gradient(
                        _completion_logits(
                            teacher_module,
                            teacher_ids,
                            teacher_mask,
                            actual_teacher_prompt_length,
                        )
                    )
                with jax.named_scope(scope_root + "/loss_fn/full_vocab_sdpo_loss"):
                    per_token_loss, student_logps, teacher_logps = _full_vocab_sdpo_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        completion_ids=completion_ids,
                        distillation_type=distillation_type,
                        distillation_topk=distillation_topk,
                        distillation_add_tail=distillation_add_tail,
                        distillation_clip=distillation_is_clip,
                    )
                    per_token_loss = distillation_weight * per_token_loss
            else:
                with jax.named_scope(scope_root + "/loss_fn/student_logps"):
                    student_logps = get_per_token_logps(
                        module,
                        student_input_ids,
                        student_attn_mask,
                        prompt_len,
                        logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                    )

                with jax.named_scope(scope_root + "/loss_fn/teacher_logps"):
                    teacher_logps = jax.lax.stop_gradient(
                        get_per_token_logps(
                            teacher_module,
                            teacher_ids,
                            teacher_mask,
                            actual_teacher_prompt_length,
                            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                        )
                    )

                with jax.named_scope(scope_root + "/loss_fn/compute_sdpo_loss"):
                    per_token_loss = _sampled_token_sdpo_loss(
                        student_logps=student_logps,
                        teacher_logps=teacher_logps,
                        distillation_type=distillation_type,
                        distillation_weight=distillation_weight,
                        distillation_is_clip=distillation_is_clip,
                    )

            ref_per_token_logps = None
            if beta != 0.0:
                ref_per_token_logps = minibatch["ref_per_token_logps"]
                if ref_per_token_logps.shape[1] != completion_ids.shape[1]:
                    ref_per_token_logps = ref_per_token_logps[:, : completion_ids.shape[1]]
                per_token_kl = _safe_masked_k3_kl(
                    ref_per_token_logps,
                    student_logps,
                    completion_mask,
                )
                per_token_loss = per_token_loss + beta * per_token_kl
            else:
                per_token_kl = None

            loss = jnp.sum(per_token_loss * completion_mask) / num_items

            def masked_mean(x):
                """Compute the mean of ``x`` over completion-mask positions."""
                return jnp.sum(x * completion_mask) / jnp.maximum(completion_token_count, 1.0)

            per_token_advantage = teacher_logps - student_logps

            other_metrics = {
                "sdpo/advantage_mean": masked_mean(per_token_advantage),
                "sdpo/advantage_pos_frac": masked_mean((per_token_advantage > 0).astype(jnp.float32)),
                "sdpo/student_logps": masked_mean(student_logps),
                "sdpo/teacher_logps": masked_mean(teacher_logps),
                "sdpo/per_token_loss": masked_mean(per_token_loss),
            }

            if beta != 0.0 and per_token_kl is not None:
                other_metrics["mean_kl"] = masked_mean(per_token_kl)
                if ref_per_token_logps is not None:
                    other_metrics["ref_per_token_logps"] = masked_mean(ref_per_token_logps)

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics=other_metrics,
        )

    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
            )
        with jax.named_scope(scope_root + "/update_state"):
            state = update_state_respectfully(
                state=state,
                gradients=gradients,
                loss_config=loss_config,
                metrics=update_metrics(
                    metrics=metrics,
                    learning_rate_fn=learning_rate_fn,
                    step=state.step,
                    gradients=gradients,
                ),
            )
        return state, metrics
    else:
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics


def _sdpo_step_with_teacher_state(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    teacher_state: EasyDeLState,
    num_generations: int,
    teacher_prompt_length: int,
    beta: float,
    distillation_type: str,
    distillation_weight: float,
    logprob_vocab_chunk_size: int | None,
    max_loss_completion_tokens: int | None,
    completion_chunk_size: int | None,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    num_loss_tokens_to_skip: int = 0,
    full_logit_distillation: bool = False,
    distillation_topk: int | None = None,
    distillation_is_clip: float | None = None,
    distillation_add_tail: bool = False,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Positional wrapper routing ``_train_shared_fn_extra_args`` ``teacher_state`` into :func:`sdpo_step`."""
    return sdpo_step(
        state=state,
        batch=batch,
        num_generations=num_generations,
        teacher_prompt_length=teacher_prompt_length,
        beta=beta,
        distillation_type=distillation_type,
        distillation_weight=distillation_weight,
        logprob_vocab_chunk_size=logprob_vocab_chunk_size,
        max_loss_completion_tokens=max_loss_completion_tokens,
        completion_chunk_size=completion_chunk_size,
        loss_config=loss_config,
        learning_rate_fn=learning_rate_fn,
        partition_spec=partition_spec,
        gradient_accumulation_steps=gradient_accumulation_steps,
        is_training=is_training,
        straight_through_emulator=straight_through_emulator,
        num_loss_tokens_to_skip=num_loss_tokens_to_skip,
        full_logit_distillation=full_logit_distillation,
        distillation_topk=distillation_topk,
        distillation_is_clip=distillation_is_clip,
        distillation_add_tail=distillation_add_tail,
        teacher_state=teacher_state,
    )
