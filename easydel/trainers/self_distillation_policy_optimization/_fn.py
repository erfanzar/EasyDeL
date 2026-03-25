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
from eformer.escale import with_sharding_constraint
from jax import numpy as jnp
from jax.sharding import PartitionSpec

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..group_relative_policy_optimization._fn import get_per_token_logps
from ..training_utils import (
    make_assertions_and_get_sizes,
    minibatch_call,
    update_metrics,
    update_state_respectfully,
)


def sdpo_step(
    state: EasyDeLState,
    batch: collections.abc.Mapping[str, jax.Array],
    num_generations: int,
    teacher_prompt_length: int,
    beta: float,
    distillation_type: str,
    logprob_vocab_chunk_size: int | None,
    max_loss_completion_tokens: int | None,
    completion_chunk_size: int | None,
    loss_config: LossConfig | None = None,
    learning_rate_fn: optax.Schedule = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    is_training: bool = True,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Single SDPO training / evaluation step.

    The batch must contain the following arrays (produced by
    :meth:`SDPOTrainer._preprocess_batch_input`):

    - ``prompt_ids``         : ``[B, prompt_len]``
    - ``prompt_mask``        : ``[B, prompt_len]``
    - ``completion_ids``     : ``[B*G, comp_len]``
    - ``completion_mask``    : ``[B*G, comp_len]``
    - ``teacher_ids``        : ``[B*G, teacher_len]`` -
        full teacher context = prompt || feedback_pad || completion
    - ``teacher_mask``       : ``[B*G, teacher_len]``
    - ``num_items_in_batch`` : scalar (total completion tokens, for loss normalisation)

    Optionally (when ``beta > 0``):

    - ``ref_per_token_logps``: ``[B*G, comp_len]`` - frozen-reference log-probs

    Args:
        state: Current EasyDeL model/optimiser state.
        batch: Pre-processed batch from :meth:`SDPOTrainer._preprocess_batch_input`.
        num_generations: Number of completions sampled per prompt (``G``). **STATIC**.
        teacher_prompt_length: Number of tokens in the teacher prefix
            (prompt + feedback separator), i.e. where the completion starts
            inside ``teacher_ids``. **STATIC**.
        beta: Weight of KL penalty toward the frozen reference model.
            Set to 0 to disable (default for SDPO). **STATIC**.
        distillation_type: ``'kl'`` or ``'jsd'``. **STATIC**.
        max_loss_completion_tokens: Optional cap on completion tokens used by
            the SDPO loss. When set, both the student and teacher scoring
            paths are truncated to the first ``max_loss_completion_tokens``
            completion tokens so the compiled graph does not scale with the
            full sampled completion length. **STATIC**.
        completion_chunk_size: Optional cap on the number of completions scored
            in a single inner SDPO loss pass. When set, the student/teacher
            scoring work is split across several smaller completion batches to
            reduce peak activation and temporary-buffer memory. **STATIC**.
        loss_config: Optional loss / gradient-clipping configuration.
        learning_rate_fn: Learning-rate schedule used for metric logging.
        partition_spec: Partition spec for sharding the batch.
        gradient_accumulation_steps: Number of minibatch accumulation steps.
        is_training: Whether to compute and apply gradients.
        straight_through_emulator: Optional STE for quantisation-aware training.

    Returns:
        ``(updated_state, metrics)`` when ``is_training=True``, or just
        ``metrics`` when ``is_training=False``.
    """
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(arr=batch, sharding=partition_spec)

    def loss_fn(tree, minibatch):
        if is_training and straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree)

        prompt_ids = minibatch["prompt_ids"]
        prompt_mask = minibatch["prompt_mask"]
        completion_ids = minibatch["completion_ids"]
        completion_mask = minibatch["completion_mask"]
        teacher_ids = minibatch["teacher_ids"]
        teacher_mask = minibatch["teacher_mask"]

        completion_was_truncated = False
        if max_loss_completion_tokens is not None and completion_ids.shape[1] > max_loss_completion_tokens:
            completion_ids = completion_ids[:, :max_loss_completion_tokens]
            completion_mask = completion_mask[:, :max_loss_completion_tokens]
            teacher_ids = teacher_ids[:, : teacher_prompt_length + max_loss_completion_tokens]
            teacher_mask = teacher_mask[:, : teacher_prompt_length + max_loss_completion_tokens]
            completion_was_truncated = True

        prompt_len = prompt_ids.shape[-1]
        completion_token_count = jnp.sum(completion_mask)
        num_items = (
            completion_token_count
            if completion_was_truncated
            else minibatch.get("num_items_in_batch", completion_token_count)
        )

        use_chunked_completion_loss = (
            completion_chunk_size is not None and completion_ids.shape[0] > completion_chunk_size
        )
        if use_chunked_completion_loss:
            expanded_prompt_ids = prompt_ids.repeat(num_generations, 0)
            expanded_prompt_mask = prompt_mask.repeat(num_generations, 0)
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
                    [expanded_prompt_mask[start:end], chunk_completion_mask], axis=1
                )
                chunk_student_logps = get_per_token_logps(
                    module,
                    chunk_student_input_ids,
                    chunk_student_attn_mask,
                    prompt_len,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
                chunk_teacher_logps = jax.lax.stop_gradient(
                    get_per_token_logps(
                        module,
                        teacher_ids[start:end],
                        teacher_mask[start:end],
                        teacher_prompt_length,
                        logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                    )
                )

                if distillation_type == "kl":
                    chunk_target_logps = chunk_teacher_logps
                elif distillation_type == "jsd":
                    chunk_target_logps = jnp.logaddexp(chunk_student_logps, chunk_teacher_logps) - jnp.log(2.0)
                else:
                    raise ValueError(f"Unknown distillation_type '{distillation_type}'. Must be 'kl' or 'jsd'.")

                chunk_distill_weight = jax.lax.stop_gradient(chunk_student_logps - chunk_target_logps)
                chunk_per_token_loss = chunk_distill_weight * chunk_student_logps

                if beta != 0.0:
                    ref_per_token_logps = minibatch["ref_per_token_logps"][start:end]
                    if ref_per_token_logps.shape[1] != chunk_completion_ids.shape[1]:
                        ref_per_token_logps = ref_per_token_logps[:, : chunk_completion_ids.shape[1]]
                    chunk_per_token_kl = (
                        jnp.exp(ref_per_token_logps - chunk_student_logps)
                        - (ref_per_token_logps - chunk_student_logps)
                        - 1
                    )
                    chunk_per_token_loss = chunk_per_token_loss + beta * chunk_per_token_kl
                    mean_kl_num = mean_kl_num + jnp.sum(chunk_per_token_kl * chunk_completion_mask)
                    ref_logps_num = ref_logps_num + jnp.sum(ref_per_token_logps * chunk_completion_mask)
                else:
                    chunk_per_token_kl = None

                chunk_advantage = chunk_teacher_logps - chunk_student_logps
                loss_numerator = loss_numerator + jnp.sum(chunk_per_token_loss * chunk_completion_mask)
                advantage_num = advantage_num + jnp.sum(chunk_advantage * chunk_completion_mask)
                advantage_pos_num = advantage_pos_num + jnp.sum(
                    (chunk_advantage > 0).astype(jnp.float32) * chunk_completion_mask
                )
                student_logps_num = student_logps_num + jnp.sum(chunk_student_logps * chunk_completion_mask)
                teacher_logps_num = teacher_logps_num + jnp.sum(chunk_teacher_logps * chunk_completion_mask)
                per_token_loss_num = per_token_loss_num + jnp.sum(chunk_per_token_loss * chunk_completion_mask)

            loss = loss_numerator / jnp.maximum(num_items, 1.0)
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
            student_input_ids = jnp.concatenate([prompt_ids.repeat(num_generations, 0), completion_ids], axis=1)
            student_attn_mask = jnp.concatenate([prompt_mask.repeat(num_generations, 0), completion_mask], axis=1)
            student_logps = get_per_token_logps(
                module,
                student_input_ids,
                student_attn_mask,
                prompt_len,
                logprob_vocab_chunk_size=logprob_vocab_chunk_size,
            )

            teacher_logps = jax.lax.stop_gradient(
                get_per_token_logps(
                    module,
                    teacher_ids,
                    teacher_mask,
                    teacher_prompt_length,
                    logprob_vocab_chunk_size=logprob_vocab_chunk_size,
                )
            )

            if distillation_type == "kl":
                target_logps = teacher_logps
            elif distillation_type == "jsd":
                target_logps = jnp.logaddexp(student_logps, teacher_logps) - jnp.log(2.0)
            else:
                raise ValueError(f"Unknown distillation_type '{distillation_type}'. Must be 'kl' or 'jsd'.")
            distill_weight = jax.lax.stop_gradient(student_logps - target_logps)
            per_token_loss = distill_weight * student_logps

            if beta != 0.0:
                ref_per_token_logps = minibatch["ref_per_token_logps"]
                if ref_per_token_logps.shape[1] != completion_ids.shape[1]:
                    ref_per_token_logps = ref_per_token_logps[:, : completion_ids.shape[1]]
                per_token_kl = jnp.exp(ref_per_token_logps - student_logps) - (ref_per_token_logps - student_logps) - 1
                per_token_loss = per_token_loss + beta * per_token_kl
            else:
                per_token_kl = None

            loss = jnp.sum(per_token_loss * completion_mask) / jnp.maximum(num_items, 1.0)

            def masked_mean(x):
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
                other_metrics["ref_per_token_logps"] = jnp.mean(ref_per_token_logps)

        return loss, LossMetrics(
            loss=loss,
            accuracy=1,
            other_metrics=other_metrics,
        )

    if is_training:
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
        )
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
        _, metrics = loss_fn(tree=state.graphstate, minibatch=batch)
        return metrics
