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
"""Triple preference optimization trainer and loss functions."""

from __future__ import annotations

import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.nn import log_sigmoid as logsigmoid
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from .._logprob_utils import (
    compute_sequence_scores_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)
from .._shared import apply_paired_truncation, gather_multimodal_kwargs
from ..training_utils import (
    ScheduledLossAdapter,
    _scheduled_terminal_stage_rank,
    bind_scheduled_module,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    make_scheduled_module_binder,
    minibatch_call,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)


def compute_tpo_losses(
    chosen_logps: Array,
    rejected_logps: Array,
    *,
    beta: float,
    label_smoothing: float,
    loss_type: str,
    tpo_l_gamma: float,
) -> Array:
    """Compute TRL TPO pairwise losses without reference-policy subtraction.

    TPO is reference-free: chosen and rejected sequence log-probabilities are
    compared directly, with optional length-normalized variants handled by the
    forward pass. The returned array is per-example and is averaged by the
    training or evaluation step.
    """
    delta_score = chosen_logps - rejected_logps
    if loss_type == "sigmoid":
        return -(
            logsigmoid(beta * delta_score) * (1 - label_smoothing) + logsigmoid(-beta * delta_score) * label_smoothing
        )
    if loss_type == "hinge":
        return jnp.maximum(1 - beta * delta_score, 0)
    if loss_type == "ipo":
        return (delta_score - 1 / (2 * beta)) ** 2
    if loss_type == "tpo-l":
        shifted_delta = delta_score - float(tpo_l_gamma) / float(beta)
        return -(
            logsigmoid(beta * shifted_delta) * (1 - label_smoothing)
            + logsigmoid(-beta * shifted_delta) * label_smoothing
        )
    raise ValueError(f"Unknown TPO loss type: {loss_type!r}. Expected one of 'sigmoid', 'hinge', 'ipo', or 'tpo-l'.")


def _tpo_concatenated_inputs(batch: dict[str, Array], padding_value: int, include_reference: bool) -> dict[str, Array]:
    """Stack chosen, rejected, and optional reference completions for one forward.

    Prompt tensors are repeated for each completion branch. Completion tensors
    are right-padded to a shared length so a single model call can score all TPO
    branches and then split results back into chosen/rejected/reference pieces.
    """
    prompt_parts = [batch["prompt_input_ids"], batch["prompt_input_ids"]]
    prompt_mask_parts = [batch["prompt_attention_mask"], batch["prompt_attention_mask"]]
    completion_parts = [batch["chosen_input_ids"], batch["rejected_input_ids"]]
    completion_mask_parts = [batch["chosen_attention_mask"], batch["rejected_attention_mask"]]
    if include_reference:
        prompt_parts.append(batch["prompt_input_ids"])
        prompt_mask_parts.append(batch["prompt_attention_mask"])
        completion_parts.append(batch["reference_input_ids"])
        completion_mask_parts.append(batch["reference_attention_mask"])

    max_completion_length = max(int(part.shape[1]) for part in completion_parts)
    num_branches = 3 if include_reference else 2
    output: dict[str, Array] = {
        "prompt_input_ids": jnp.concatenate(prompt_parts, axis=0),
        "prompt_attention_mask": jnp.concatenate(prompt_mask_parts, axis=0),
        "completion_input_ids": jnp.concatenate(
            [
                jnp.pad(part, ((0, 0), (0, max_completion_length - part.shape[1])), constant_values=padding_value)
                for part in completion_parts
            ],
            axis=0,
        ),
        "completion_attention_mask": jnp.concatenate(
            [
                jnp.pad(part, ((0, 0), (0, max_completion_length - part.shape[1])), constant_values=0)
                for part in completion_mask_parts
            ],
            axis=0,
        ),
    }
    for key in ("pixel_values", "pixel_attention_mask", "image_sizes"):
        if key in batch:
            output[key] = jnp.concatenate([batch[key]] * num_branches, axis=0)
    return output


def tpo_concatenated_forward(
    model: tp.Any,
    batch: dict[str, Array],
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: str = "keep_start",
    aux_loss_enabled: bool = False,
    loss_type: str = "sigmoid",
    tpo_alpha: float = 1.0,
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
) -> dict[str, Array]:
    """Run a TPO chosen/rejected/(gold reference) forward pass.

    The function concatenates all branches, applies model-compatible truncation
    and multimodal kwargs, computes per-token log-probs (using headless chunked
    LM-head scoring when ``resolve_lmhead_chunksize(model)`` is enabled), then
    returns sequence log-probs and lengths needed by TPO losses. When a gold
    reference branch is included, it also returns its NLL contribution.
    """
    include_reference = bool(tpo_alpha != 0.0 and "reference_input_ids" in batch)
    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = _tpo_concatenated_inputs(batch, padding_value, include_reference)
    model_kwargs = gather_multimodal_kwargs(concatenated_batch, aux_loss_enabled=aux_loss_enabled)

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]
    lmhead_chunksize = None

    if is_encoder_decoder:
        labels = jnp.where(completion_attention_mask == 0, label_pad_token_id, completion_input_ids)
        call_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "labels": labels,
            **model_kwargs,
        }
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        outputs = model(**sanitize_model_call_kwargs(call_kwargs))
        logits = outputs.logits
        loss_mask = completion_attention_mask.astype(bool)
    else:
        input_ids = jnp.concatenate([prompt_input_ids, completion_input_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_attention_mask, completion_attention_mask], axis=1)
        loss_mask = jnp.concatenate([jnp.zeros_like(prompt_attention_mask), completion_attention_mask], axis=1)
        input_ids, attention_mask, loss_mask = apply_paired_truncation(
            input_ids,
            attention_mask,
            loss_mask,
            max_length=max_length,
            truncation_mode=truncation_mode,
        )
        lmhead_chunksize = resolve_lmhead_chunksize(model)
        call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, **model_kwargs}
        if lmhead_chunksize is not None:
            call_kwargs["apply_lm_head"] = False
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        outputs = model(**sanitize_model_call_kwargs(call_kwargs))
        logits = outputs.logits
        labels = jnp.roll(input_ids, shift=-1, axis=1)
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(bool).at[:, -1].set(False)

    labels = labels.astype(jnp.int32)
    labels = jnp.where(loss_mask, labels, 0)

    if not is_encoder_decoder and logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        if hidden_states.shape[:2] != labels.shape[:2]:
            hidden_states = hidden_states[:, -labels.shape[1] :, :]
        sum_logps, _, token_counts = compute_sequence_scores_from_hidden_states(
            model=model,
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
            vocab_shard_stage=vocab_shard_stage,
        )
    else:
        if logits is None:
            raise TypeError(f"{type(model).__name__} did not return logits.")
        if logits.shape[:2] != labels.shape[:2]:
            logits = logits[:, -labels.shape[1] :]
        per_token_logps = jnp.where(
            loss_mask,
            compute_token_logps_and_entropies_chunked(
                logits, labels, return_entropy=False, chunk_size=logprob_vocab_chunk_size
            )[0],
            0.0,
        )
        sum_logps = per_token_logps.sum(-1)
        token_counts = loss_mask.sum(-1).astype(jnp.float32)

    lengths = jnp.maximum(token_counts, 1.0)
    all_logps = sum_logps / lengths if loss_type in {"ipo", "tpo-l"} else sum_logps

    output = {
        "chosen_logps": all_logps[:num_examples],
        "rejected_logps": all_logps[num_examples : 2 * num_examples],
        "chosen_lengths": lengths[:num_examples],
        "rejected_lengths": lengths[num_examples : 2 * num_examples],
    }
    if include_reference:
        reference_logps_sum = sum_logps[2 * num_examples :]
        reference_token_counts = token_counts[2 * num_examples :]
        output["reference_nll"] = -jnp.sum(reference_logps_sum) / jnp.maximum(jnp.sum(reference_token_counts), 1.0)
    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss_enabled and aux_loss is not None:
        output["aux_loss"] = aux_loss
    return output


def tpo_training_step(
    state: EasyDeLState,
    batch: dict[str, Array],
    learning_rate_fn: tp.Callable,
    concatenated_forward: tp.Callable,
    beta: float = 0.01,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    tpo_alpha: float = 1.0,
    tpo_l_gamma: float = 0.5,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Run one TPO optimization step over a triple-preference batch.

    The batch is sharding-constrained, split into minibatches according to
    gradient accumulation, differentiated through the TPO loss closure (with
    optional QAT straight-through emulation), and applied with the shared
    EasyDeL optimizer update utilities.
    """
    _, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: spx.State, call_batch: dict[str, Array]) -> tuple[Array, LossMetrics]:
        """Compute TPO loss and reward metrics for one differentiated minibatch.

        The closure optionally applies ``straight_through_emulator``, merges
        the candidate graphstate, runs the concatenated forward, applies the
        configured TPO loss, adds optional reference NLL and auxiliary loss
        terms, and returns scalar metrics for gradient accumulation.
        """
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)
        model_output = concatenated_forward(module, call_batch)
        losses = compute_tpo_losses(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            tpo_l_gamma=tpo_l_gamma,
        )
        if "reference_nll" in model_output:
            losses = losses + float(tpo_alpha) * model_output["reference_nll"]
        aux_loss = model_output.get("aux_loss")
        if aux_loss is not None:
            losses = losses + aux_loss
        metrics = LossMetrics(
            loss=losses.mean(),
            chosen_rewards=beta * jax.lax.stop_gradient(model_output["chosen_logps"]),
            rejected_rewards=beta * jax.lax.stop_gradient(model_output["rejected_logps"]),
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
    )
    metrics = update_metrics(metrics=metrics, learning_rate_fn=learning_rate_fn, step=state.step, gradients=gradients)
    state = update_state_respectfully(state=state, gradients=gradients, loss_config=loss_config, metrics=metrics)
    return state, metrics


def _tpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the TPO scheduled-loss compilation."""
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "beta",
            "label_smoothing",
            "loss_type",
            "tpo_alpha",
            "tpo_l_gamma",
            "partition_spec",
        ),
        object_fields=("concatenated_forward", "straight_through_emulator"),
    )


def _make_tpo_scheduled_loss(call):
    """Build a SpectraX-scheduled TPO scalar-loss closure for ``call``."""
    concatenated_forward_fn = call.get("concatenated_forward")
    beta = call.get("beta", 0.01)
    label_smoothing = call.get("label_smoothing", 0.0)
    loss_type = call.get("loss_type", "sigmoid")
    tpo_alpha = call.get("tpo_alpha", 1.0)
    tpo_l_gamma = call.get("tpo_l_gamma", 0.5)
    partition_spec = call.get("partition_spec")
    bind_module = make_scheduled_module_binder(call, binder_fn=bind_scheduled_module)
    schedule = call.schedule

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar TPO loss inside the SpectraX scheduled VJP."""
        with jax.named_scope("easydel/trainer/tpo/scheduled_loss"):
            with jax.named_scope("easydel/trainer/tpo/scheduled_loss/bind_module"):
                module = bind_module(tree)
                call_batch = constrain_scheduled_batch(module, batch, partition_spec)
                _terminal_rank = _scheduled_terminal_stage_rank(module, schedule)
            with jax.named_scope("easydel/trainer/tpo/scheduled_loss/policy_forward"):
                forward_kwargs = {"vocab_shard_stage": _terminal_rank} if _terminal_rank is not None else {}
                if forward_kwargs:
                    forward_kwargs = filter_kwargs_for_callable(concatenated_forward_fn, forward_kwargs)
                model_output = concatenated_forward_fn(module, call_batch, **forward_kwargs)
            with jax.named_scope("easydel/trainer/tpo/scheduled_loss/compute_tpo_loss"):
                losses = compute_tpo_losses(
                    model_output["chosen_logps"],
                    model_output["rejected_logps"],
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    tpo_l_gamma=tpo_l_gamma,
                )
                if "reference_nll" in model_output:
                    losses = losses + float(tpo_alpha) * model_output["reference_nll"]
                aux_loss = model_output.get("aux_loss")
                if aux_loss is not None:
                    losses = losses + aux_loss
            return losses.mean()

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=tpo_training_step,
    adapter=ScheduledLossAdapter(
        name="tpo",
        make_loss=_make_tpo_scheduled_loss,
        make_cache_key=_tpo_scheduled_loss_cache_key,
    ),
)


def tpo_evaluation_step(
    state: EasyDeLState,
    batch: dict[str, Array],
    concatenated_forward: tp.Callable,
    beta: float = 0.01,
    label_smoothing: float = 0.0,
    loss_type: str = "sigmoid",
    tpo_alpha: float = 1.0,
    tpo_l_gamma: float = 0.5,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Evaluate TPO loss metrics without updating model state.

    Evaluation mirrors the training loss computation but merges the current
    graphstate directly and skips gradient/optimizer work. The output is a
    :class:`LossMetrics` object ready for trainer logging.
    """
    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)
    model_output = concatenated_forward(state.merge(state.graphstate), batch)
    losses = compute_tpo_losses(
        model_output["chosen_logps"],
        model_output["rejected_logps"],
        beta=beta,
        label_smoothing=label_smoothing,
        loss_type=loss_type,
        tpo_l_gamma=tpo_l_gamma,
    )
    if "reference_nll" in model_output:
        losses = losses + float(tpo_alpha) * model_output["reference_nll"]
    return LossMetrics(
        loss=losses.mean(),
        chosen_rewards=beta * model_output["chosen_logps"],
        rejected_rewards=beta * model_output["rejected_logps"],
    )
