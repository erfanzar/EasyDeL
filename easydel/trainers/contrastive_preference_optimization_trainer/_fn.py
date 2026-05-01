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
"""Loss and step implementations for the CPO trainer.

Hosts the concatenated forward (chosen / rejected sequences in one
batch), the loss-type dispatch (sigmoid / hinge / IPO / SimPO), the
combined CPO loss with auxiliary supervised log-likelihood, and the
scheduled-VJP loss adapter used under MPMD pipelining.
"""

from __future__ import annotations

import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.nn import log_sigmoid, relu
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics
from easydel.trainers._logprob_utils import (
    compute_sequence_scores_from_hidden_states,
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
)
from easydel.trainers._shared import apply_paired_truncation, gather_multimodal_kwargs
from easydel.trainers.direct_preference_optimization_trainer._fn import concatenated_inputs

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)

LOSS_TYPES = tp.Literal["sigmoid", "hinge", "ipo", "simpo"]


def concatenated_forward(
    model: EasyDeLBaseModule,
    batch: dict[str, tp.Any],
    *,
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    aux_loss_enabled: bool = False,
    loss_type: LOSS_TYPES = "sigmoid",
    logprob_vocab_chunk_size: int | None = None,
) -> dict[str, jax.Array]:
    """Runs the policy model on concatenated chosen/rejected sequences.

    This mirrors the behaviour of TRL's CPO forward helper while leveraging the
    JAX-specific utilities already used by the DPO trainer. We concatenate the
    chosen and rejected completions to share a single forward pass, compute
    per-token log-probabilities and expose additional statistics required by the
    CPO objective (raw log-prob sums and token lengths).
    """

    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = concatenated_inputs(batch=batch, padding_value=padding_value)

    model_kwargs: dict[str, jax.Array] = gather_multimodal_kwargs(concatenated_batch, aux_loss_enabled=aux_loss_enabled)

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]
    lmhead_chunksize = None

    if is_encoder_decoder:
        labels = jnp.where(
            completion_attention_mask == 0,
            label_pad_token_id,
            completion_input_ids,
        )
        call_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "labels": labels,
            **model_kwargs,
        }
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits
        loss_mask = completion_attention_mask.astype(bool)
    else:
        input_ids = jnp.concatenate([prompt_input_ids, completion_input_ids], axis=1)
        attention_mask = jnp.concatenate([prompt_attention_mask, completion_attention_mask], axis=1)
        loss_mask = jnp.concatenate(
            [jnp.zeros_like(prompt_attention_mask), completion_attention_mask],
            axis=1,
        )
        input_ids, attention_mask, loss_mask = apply_paired_truncation(
            input_ids,
            attention_mask,
            loss_mask,
            max_length=max_length,
            truncation_mode=truncation_mode,
        )
        call_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **model_kwargs,
        }
        lmhead_chunksize = resolve_lmhead_chunksize(model)
        if lmhead_chunksize is not None:
            call_kwargs["apply_lm_head"] = False
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits
        labels = jnp.roll(input_ids, shift=-1, axis=1)
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(bool)

    if logits is not None and logits.shape[:2] != loss_mask.shape:
        seq_len = loss_mask.shape[1]
        logits = logits[:, -seq_len:]

    labels = jnp.where(loss_mask, labels, 0)

    if not is_encoder_decoder and logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        if hidden_states.shape[:2] != labels.shape:
            hidden_states = hidden_states[:, -labels.shape[1] :, :]
        sum_logps, token_logit_sums, token_counts = compute_sequence_scores_from_hidden_states(
            model=model,
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            token_chunk_size=lmhead_chunksize,
            vocab_chunk_size=logprob_vocab_chunk_size,
        )
    else:
        gathered_logps, _ = compute_token_logps_and_entropies_chunked(
            logits,
            labels,
            return_entropy=False,
            chunk_size=logprob_vocab_chunk_size,
        )
        per_token_logps = jnp.where(loss_mask, gathered_logps, 0.0)
        if not is_encoder_decoder:
            per_token_logps = jnp.roll(per_token_logps, shift=1, axis=1)
        sum_logps = per_token_logps.sum(axis=1)
        token_counts = jnp.sum(loss_mask.astype(jnp.float32), axis=1)
        token_logit_sums = jnp.sum(
            jnp.where(loss_mask, logits.astype(jnp.float32).sum(axis=-1), 0.0),
            axis=1,
        )

    token_counts = jnp.maximum(token_counts, 1.0)
    if loss_type in ("ipo", "simpo"):
        scaled_logps = jnp.where(token_counts > 0, sum_logps / token_counts, 0.0)
    else:
        scaled_logps = sum_logps

    chosen_logps = scaled_logps[:num_examples]
    rejected_logps = scaled_logps[num_examples:]
    chosen_logps_raw = sum_logps[:num_examples]
    rejected_logps_raw = sum_logps[num_examples:]
    chosen_lengths = token_counts[:num_examples]
    rejected_lengths = token_counts[num_examples:]

    chosen_logits_sum = token_logit_sums[:num_examples].sum()
    rejected_logits_sum = token_logit_sums[num_examples:].sum()
    chosen_denom = jnp.maximum(chosen_lengths.sum(), 1.0)
    rejected_denom = jnp.maximum(rejected_lengths.sum(), 1.0)
    mean_chosen_logits = chosen_logits_sum / chosen_denom
    mean_rejected_logits = rejected_logits_sum / rejected_denom

    outputs_dict: dict[str, jax.Array] = {
        "chosen_logps": chosen_logps,
        "rejected_logps": rejected_logps,
        "chosen_logps_raw": chosen_logps_raw,
        "rejected_logps_raw": rejected_logps_raw,
        "chosen_lengths": chosen_lengths,
        "rejected_lengths": rejected_lengths,
        "mean_chosen_logits": mean_chosen_logits,
        "mean_rejected_logits": mean_rejected_logits,
    }
    if aux_loss_enabled and hasattr(outputs, "aux_loss"):
        outputs_dict["aux_loss"] = outputs.aux_loss
    return outputs_dict


def cpo_loss(
    policy_chosen_logps: jax.Array,
    policy_rejected_logps: jax.Array,
    *,
    beta: float,
    label_smoothing: float,
    loss_type: LOSS_TYPES,
    simpo_gamma: float,
    alpha: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Compute the CPO/SimPO/AlphaPO contrastive loss and per-example rewards.

    Reference-free preference loss: rather than penalising
    ``logp_policy - logp_reference`` like DPO, CPO penalises the
    log-ratio between chosen and rejected completions directly,
    optionally shaped by an AlphaPO probability transform or a SimPO
    margin.

    The path is:

    1. **Reward construction.** When ``alpha == 0`` the rewards are the
       summed log-probs themselves (``chosen_rewards = policy_chosen_logps``).
       When ``alpha != 0`` (AlphaPO), ``rewards = (1 - p**(-alpha)) / alpha``
       where ``p = exp(logp)``; ``logits`` is the chosen-minus-rejected
       differential of those rewards.
    2. **Loss shape.** Selected by ``loss_type``:
       - ``"sigmoid"`` -- ``-(1 - eps) * logsigmoid(beta * logits)
         - eps * logsigmoid(-beta * logits)`` with ``eps = label_smoothing``.
       - ``"simpo"`` -- same shape as sigmoid but on
         ``logits - simpo_gamma/beta`` (target margin ``simpo_gamma``).
       - ``"hinge"`` -- ``relu(1 - beta * logits)``.
       - ``"ipo"`` -- squared error ``(logits - 1/(2*beta))**2``.
    3. **Reward rescaling.** The returned ``chosen_rewards`` /
       ``rejected_rewards`` are always multiplied by ``beta`` so that
       downstream metrics have a consistent unit regardless of the
       ``alpha`` branch.

    Args:
        policy_chosen_logps: ``[batch]`` summed completion logps for
            chosen completions (length-normalized when running SimPO).
        policy_rejected_logps: Same for rejected completions.
        beta: CPO inverse-temperature on the reward differential.
        label_smoothing: cDPO-style smoothing factor; only consulted
            by the sigmoid and SimPO branches.
        loss_type: One of ``"sigmoid"``, ``"hinge"``, ``"ipo"``,
            ``"simpo"`` (``"alphapo"`` is rewritten upstream).
        simpo_gamma: Target margin used by the SimPO branch.
        alpha: AlphaPO reward shaping coefficient. ``0.0`` keeps the
            rewards equal to the raw logps.

    Returns:
        ``(losses, chosen_rewards, rejected_rewards)`` where
        ``losses`` is ``[batch]`` per-example loss and the reward
        arrays are ``beta``-scaled per-example diagnostics.

    Raises:
        ValueError: If ``loss_type`` is not one of the supported
            variants.
    """

    if alpha != 0.0:
        chosen_probs = jnp.exp(policy_chosen_logps)
        rejected_probs = jnp.exp(policy_rejected_logps)
        chosen_rewards = (1.0 - jnp.power(chosen_probs, -alpha)) / alpha
        rejected_rewards = (1.0 - jnp.power(rejected_probs, -alpha)) / alpha
        logits = chosen_rewards - rejected_rewards
    else:
        chosen_rewards = policy_chosen_logps
        rejected_rewards = policy_rejected_logps
        logits = policy_chosen_logps - policy_rejected_logps

    if loss_type == "simpo":
        gamma_logratios = simpo_gamma / beta
        logits = logits - gamma_logratios
        losses = -log_sigmoid(beta * logits) * (1.0 - label_smoothing) - log_sigmoid(-beta * logits) * label_smoothing
    elif loss_type == "sigmoid":
        losses = -log_sigmoid(beta * logits) * (1.0 - label_smoothing) - log_sigmoid(-beta * logits) * label_smoothing
    elif loss_type == "hinge":
        losses = relu(1.0 - beta * logits)
    elif loss_type == "ipo":
        losses = jnp.square(logits - (1.0 / (2.0 * beta)))
    else:
        raise ValueError(f"Unknown loss type '{loss_type}'. Expected one of ['sigmoid', 'hinge', 'ipo', 'simpo'].")

    if alpha != 0.0:
        chosen_rewards = beta * jnp.asarray(chosen_rewards)
        rejected_rewards = beta * jnp.asarray(rejected_rewards)
    else:
        chosen_rewards = beta * policy_chosen_logps
        rejected_rewards = beta * policy_rejected_logps

    return losses, chosen_rewards, rejected_rewards


def _policy_nll_loss(
    chosen_logps_raw: jax.Array,
    chosen_lengths: jax.Array,
) -> jax.Array:
    """Compute the supervised behaviour-cloning NLL term used by CPO.

    CPO mixes a contrastive preference loss with a token-averaged
    negative log-likelihood on the chosen completion (weighted by
    ``cpo_alpha``). This helper returns ``-sum(logp) / total_tokens``
    so the regulariser is independent of batch composition.

    Args:
        chosen_logps_raw: Per-token (already masked) log-probs of the
            chosen completion summed per example, ``[batch]``.
        chosen_lengths: Per-example loss-mask token count, ``[batch]``.

    Returns:
        Scalar token-averaged NLL.
    """
    total_tokens = jnp.maximum(jnp.sum(chosen_lengths), 1)
    total_logprob = jnp.sum(chosen_logps_raw)
    return -total_logprob / total_tokens


def training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    learning_rate_fn: tp.Callable[[jax.Array], jax.Array] | None,
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
    label_smoothing: float,
    loss_type: LOSS_TYPES,
    cpo_alpha: float,
    simpo_gamma: float,
    alpha: float,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Run one CPO training step (forward, loss, backward, optimizer update).

    Combines two losses on each minibatch:

    1. The CPO/SimPO/AlphaPO contrastive loss between chosen and
       rejected completions (see :func:`compute_cpo_loss`).
    2. An auxiliary supervised NLL on the chosen completion, scaled by
       ``cpo_alpha`` (see :func:`_policy_nll_loss`). Set
       ``cpo_alpha == 0.0`` to recover the pure contrastive
       (AlphaPO/SimPO) objective.

    Unlike DPO this step does *not* run a reference model: the
    contrastive term consumes only policy logps. The
    ``concatenated_forward_fn`` is expected to return per-example
    summed logps for both halves of the chosen/rejected concatenation
    plus raw / length tensors for the supervised regulariser.

    Args:
        state: Policy ``EasyDeLState`` being differentiated.
        batch: CPO minibatch with paired chosen/rejected sequences.
        learning_rate_fn: Schedule mapping step to learning rate.
        concatenated_forward_fn: Captured forward closure that returns
            ``{"chosen_logps", "rejected_logps", "chosen_logps_raw",
            "chosen_lengths", ...}``.
        beta: CPO inverse-temperature.
        label_smoothing: cDPO-style smoothing for the contrastive loss.
        loss_type: One of ``"sigmoid"``, ``"hinge"``, ``"ipo"``,
            ``"simpo"``.
        cpo_alpha: Weight on the supervised NLL term.
        simpo_gamma: Target reward margin for the SimPO branch.
        alpha: AlphaPO reward shaping coefficient.
        loss_config: ``LossConfig`` controlling NaN handling.
        partition_spec: Batch sharding spec under the model's mesh.
        gradient_accumulation_steps: Gradient-accumulation factor; the
            batch must be evenly divisible.
        straight_through_emulator: Optional STE callable applied to
            the graphstate before the forward pass (QAT path).

    Returns:
        ``(new_state, metrics)`` where ``metrics`` is a ``LossMetrics``
        with the mean total loss and the per-example
        ``chosen_rewards`` / ``rejected_rewards`` arrays for logging.
    """

    _, minibatch_size, batch_partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, batch_partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: spx.State, call_batch: dict[str, jax.Array]):
        """Compute the CPO loss and diagnostic metrics for one minibatch.

        Runs the concatenated forward, evaluates the chosen
        ``loss_type``-specific contrastive loss, adds the auxiliary
        chosen-NLL term scaled by ``cpo_alpha``, and folds in any
        router/aux-loss reported by the model.

        Args:
            tree: Policy graphstate to differentiate against.
            call_batch: Minibatch of preference triples.

        Returns:
            ``(loss, metrics)`` with reward margin/accuracy and chosen
            / rejected logit means recorded in ``metrics.other_metrics``.
        """
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        policy_model = state.merge(tree=tree)
        model_outputs = concatenated_forward_fn(policy_model, call_batch)

        losses, chosen_rewards, rejected_rewards = cpo_loss(
            model_outputs["chosen_logps"],
            model_outputs["rejected_logps"],
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            simpo_gamma=simpo_gamma,
            alpha=alpha,
        )

        chosen_rewards = jax.lax.stop_gradient(chosen_rewards)
        rejected_rewards = jax.lax.stop_gradient(rejected_rewards)
        policy_nll_loss = _policy_nll_loss(
            model_outputs["chosen_logps_raw"],
            model_outputs["chosen_lengths"],
        )

        loss = losses.mean() + cpo_alpha * policy_nll_loss
        aux_loss = model_outputs.get("aux_loss")
        if aux_loss is not None:
            loss = loss + aux_loss

        reward_margin = jnp.mean(chosen_rewards - rejected_rewards)
        reward_accuracy = jnp.mean((chosen_rewards > rejected_rewards).astype(jnp.float32))

        other_metrics = {
            "policy_nll_loss": policy_nll_loss,
            "reward_margin": reward_margin,
            "reward_accuracy": reward_accuracy,
            "mean_chosen_logits": model_outputs["mean_chosen_logits"],
            "mean_rejected_logits": model_outputs["mean_rejected_logits"],
        }
        metrics = LossMetrics(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            other_metrics=other_metrics,
        )
        return loss, metrics

    grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=grad_fn,
    )

    metrics = update_metrics(
        metrics=metrics,
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=gradients,
    )
    new_state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=loss_config,
        metrics=metrics,
    )
    return new_state, metrics


def _cpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the CPO scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering all CPO knobs that influence compilation
        (loss type, beta, smoothing, simpo gamma, alpha, partition spec,
        forward fn identity, and the quantization emulator identity).
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "beta",
            "label_smoothing",
            "loss_type",
            "cpo_alpha",
            "simpo_gamma",
            "alpha",
            "partition_spec",
        ),
        object_fields=("concatenated_forward_fn", "straight_through_emulator"),
    )


def _make_cpo_scheduled_loss(call):
    """Build a SpectraX-scheduled CPO loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` exposing the trainer's
            current configuration.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready to feed to
        :func:`spx.sxvalue_and_grad`.
    """
    concatenated_forward_fn = call.get("concatenated_forward_fn")
    beta = call.get("beta")
    label_smoothing = call.get("label_smoothing")
    loss_type = call.get("loss_type")
    cpo_alpha = call.get("cpo_alpha")
    simpo_gamma = call.get("simpo_gamma")
    alpha = call.get("alpha")
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar CPO loss inside the SpectraX scheduled VJP.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Minibatch of preference triples.

        Returns:
            The combined CPO + chosen-NLL scalar loss (with optional
            aux-loss term added).
        """
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)
        model_outputs = concatenated_forward_fn(module, call_batch)

        losses, _, _ = cpo_loss(
            model_outputs["chosen_logps"],
            model_outputs["rejected_logps"],
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_type,
            simpo_gamma=simpo_gamma,
            alpha=alpha,
        )
        policy_nll_loss = _policy_nll_loss(
            model_outputs["chosen_logps_raw"],
            model_outputs["chosen_lengths"],
        )
        loss = losses.mean() + cpo_alpha * policy_nll_loss
        aux_loss = model_outputs.get("aux_loss")
        if aux_loss is not None:
            loss = loss + aux_loss
        return loss

    return scheduled_loss


register_scheduled_loss_adapter(
    training_step,
    ScheduledLossAdapter(
        name="cpo",
        make_loss=_make_cpo_scheduled_loss,
        make_cache_key=_cpo_scheduled_loss_cache_key,
    ),
)


def evaluation_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
    label_smoothing: float,
    loss_type: LOSS_TYPES,
    cpo_alpha: float,
    simpo_gamma: float,
    alpha: float,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Run one CPO evaluation step (forward only, no parameter update).

    Computes the same combined contrastive + supervised loss as
    :func:`training_step` and the same reward / accuracy diagnostics,
    but without minibatch gradient accumulation or optimizer
    interaction. ``partition_spec`` is accepted for API symmetry and
    discarded; eval batches are run as-is on the model's mesh.

    Args:
        state: Policy state to evaluate.
        batch: CPO eval minibatch.
        concatenated_forward_fn: Forward closure with tokenization
            knobs baked in.
        beta: CPO inverse-temperature.
        label_smoothing: cDPO smoothing factor.
        loss_type: Variant key (``"sigmoid"``, ``"hinge"``, ``"ipo"``,
            ``"simpo"``).
        cpo_alpha: Weight on the supervised NLL.
        simpo_gamma: SimPO target margin.
        alpha: AlphaPO reward shaping coefficient.
        partition_spec: Unused at eval time (accepted for symmetry).

    Returns:
        ``LossMetrics`` with mean loss, per-example chosen/rejected
        rewards, and the standard CPO diagnostics (policy NLL, reward
        margin, reward accuracy, mean logits).
    """
    del partition_spec

    model_outputs = concatenated_forward_fn(state.model, batch)
    losses, chosen_rewards, rejected_rewards = cpo_loss(
        model_outputs["chosen_logps"],
        model_outputs["rejected_logps"],
        beta=beta,
        label_smoothing=label_smoothing,
        loss_type=loss_type,
        simpo_gamma=simpo_gamma,
        alpha=alpha,
    )

    chosen_rewards = jax.lax.stop_gradient(chosen_rewards)
    rejected_rewards = jax.lax.stop_gradient(rejected_rewards)
    policy_nll_loss = _policy_nll_loss(
        model_outputs["chosen_logps_raw"],
        model_outputs["chosen_lengths"],
    )

    loss = losses.mean() + cpo_alpha * policy_nll_loss
    aux_loss = model_outputs.get("aux_loss")
    if aux_loss is not None:
        loss = loss + aux_loss

    reward_margin = jnp.mean(chosen_rewards - rejected_rewards)
    reward_accuracy = jnp.mean((chosen_rewards > rejected_rewards).astype(jnp.float32))

    other_metrics = {
        "policy_nll_loss": policy_nll_loss,
        "reward_margin": reward_margin,
        "reward_accuracy": reward_accuracy,
        "mean_chosen_logits": model_outputs["mean_chosen_logits"],
        "mean_rejected_logits": model_outputs["mean_rejected_logits"],
    }

    return LossMetrics(
        loss=loss,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        other_metrics=other_metrics,
    )
