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
"""Loss and step implementations for the BCO trainer.

Binary Classifier Optimization minimises a logistic-regression-style
loss on per-example log-probability ratios between a policy and a
reference, optionally reweighted by an estimated density ratio (UDM).
This module hosts the concatenated forward, the calculate-loss path
used inside the JIT step, the running-moments helper for the BCO delta
parameter, and the scheduled-VJP loss adapter used under MPMD pipelining.
"""

from __future__ import annotations

import collections.abc
import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.nn import log_sigmoid
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

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    prepare_scheduled_reference_outputs,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)


class RunningMoments:
    """Simple running mean/variance tracker for BCO delta parameter.

    Tracks running statistics compatible with host-side updates for maintaining
    the BCO delta parameter across training steps.
    """

    def __init__(self):
        """Initialize the running stats with zero mean / unit variance.

        ``count`` is seeded with a tiny positive epsilon so that early
        ``count``-weighted updates remain numerically well-defined.
        """
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-24

    def update(self, values: collections.abc.Sequence[float] | jnp.ndarray | None):
        """Update running statistics with new values.

        Args:
            values: New values to incorporate into running statistics.
        """
        if values is None:
            return
        arr = jnp.asarray(values).reshape(-1)
        if arr.size == 0:
            return
        batch_mean = float(arr.mean())
        batch_var = float(arr.var())
        batch_count = float(arr.size)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_sum = batch_var * batch_count
        old_sum = self.var * self.count + delta * delta * self.count * batch_count / tot_count
        tot_sum = old_sum + new_sum

        self.mean += delta * batch_count / tot_count
        self.var = tot_sum / tot_count
        self.count = tot_count

    def as_dict(self) -> dict[str, float]:
        """Export statistics as dictionary.

        Returns:
            Dictionary with mean, var, and count.
        """
        return {"mean": float(self.mean), "var": float(self.var), "count": float(self.count)}

    def load_dict(self, data: dict[str, float]):
        """Load statistics from dictionary.

        Args:
            data: Dictionary with mean, var, and count.
        """
        self.mean = float(data.get("mean", 0.0))
        self.var = float(data.get("var", 1.0))
        self.count = float(data.get("count", 1e-24))


def concatenated_forward(
    model: EasyDeLBaseModule,
    batch: dict[str, jax.Array],
    *,
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: tp.Literal["keep_end", "keep_start"] = "keep_end",
    aux_loss_enabled: bool = False,
    logprob_vocab_chunk_size: int | None = None,
) -> dict[str, jax.Array]:
    """Run a BCO model forward pass and accumulate per-example completion logps.

    Unlike DPO this is a *single-stream* forward (no chosen/rejected
    pairing) -- BCO labels are unary, so each row already represents
    one (prompt, completion, label) triple. The forward path branches
    on architecture:

    - **Encoder-decoder**: prompt is fed to the encoder, completion as
      decoder labels; the LM head is applied directly inside the
      model and per-token logps are summed under the
      ``completion_attention_mask``.
    - **Causal LM**: prompt + completion are concatenated, optionally
      truncated via :func:`apply_paired_truncation`, and either
      (a) processed end-to-end and gathered with
      :func:`compute_token_logps_and_entropies_chunked`, or (b)
      processed up to the last hidden state with
      ``apply_lm_head=False`` and scored chunk-by-chunk through
      :func:`compute_sequence_scores_from_hidden_states`. The chunked
      headless path activates whenever the model exposes a positive
      ``lmhead_chunksize`` to keep peak memory bounded for large
      vocabularies.

    Args:
        model: Policy or reference model exposing the EasyDeL forward
            interface.
        batch: BCO batch with ``prompt_input_ids``,
            ``prompt_attention_mask``, ``completion_input_ids``,
            ``completion_attention_mask``, ``completion_labels`` and
            optional multimodal keys.
        is_encoder_decoder: Selects the encoder-decoder branch when
            ``True``.
        label_pad_token_id: Sentinel token id marking positions that
            should be excluded from the loss mask in the causal LM
            branch.
        padding_value: Token id used when padding completions
            (forwarded by the caller; this function does not pad
            itself).
        max_length: Maximum total sequence length used by the causal
            LM branch's truncation.
        truncation_mode: ``"keep_end"`` or ``"keep_start"`` truncation
            policy.
        aux_loss_enabled: When ``True``, the model's auxiliary load
            balance loss (if any) is forwarded under the ``aux_loss``
            key.
        logprob_vocab_chunk_size: Vocab chunk size for
            :func:`compute_token_logps_and_entropies_chunked`.

    Returns:
        Dict with ``completion_logps`` (``[batch]`` summed log-probs),
        ``completion_logits`` (raw logits or ``None`` under headless
        scoring), ``mean_completion_logits`` (scalar mean for
        diagnostics), and optionally ``aux_loss``.
    """

    prompt_input_ids = batch["prompt_input_ids"]
    prompt_attention_mask = batch["prompt_attention_mask"]
    completion_input_ids = batch["completion_input_ids"]
    completion_attention_mask = batch["completion_attention_mask"]
    completion_labels = batch["completion_labels"]

    model_kwargs: dict[str, jax.Array] = gather_multimodal_kwargs(batch, aux_loss_enabled=aux_loss_enabled)

    lmhead_chunksize = None

    if is_encoder_decoder:
        call_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "decoder_input_ids": batch.get("completion_decoder_input_ids"),
            "labels": completion_labels,
            **model_kwargs,
        }
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits
        loss_mask = completion_attention_mask.astype(bool)
        labels_safe = jnp.where(loss_mask, completion_labels, 0)
        gathered_logps, _ = compute_token_logps_and_entropies_chunked(
            logits,
            labels_safe,
            return_entropy=False,
            chunk_size=logprob_vocab_chunk_size,
        )
        completion_logps = jnp.where(loss_mask, gathered_logps, 0.0).sum(axis=1)
        token_logit_sums = jnp.sum(
            jnp.where(loss_mask, logits.astype(jnp.float32).sum(axis=-1), 0.0),
            axis=1,
        )
        token_counts = jnp.sum(loss_mask.astype(jnp.float32), axis=1)
        mean_logits = token_logit_sums.sum() / jnp.maximum(token_counts.sum(), 1.0)
    else:
        input_ids, attention_mask, completion_labels = apply_paired_truncation(
            completion_input_ids,
            completion_attention_mask,
            completion_labels,
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

        labels_shifted = completion_labels[:, 1:]
        loss_mask = labels_shifted != label_pad_token_id
        labels_safe = jnp.where(loss_mask, labels_shifted, 0)
        if logits is None and lmhead_chunksize is not None:
            hidden_states = outputs.last_hidden_state
            if hidden_states is None:
                raise TypeError(
                    f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
                )
            hidden_states = hidden_states[:, :-1, :]
            completion_logps, token_logit_sums, token_counts = compute_sequence_scores_from_hidden_states(
                model=model,
                hidden_states=hidden_states,
                labels=labels_safe,
                loss_mask=loss_mask,
                token_chunk_size=lmhead_chunksize,
                vocab_chunk_size=logprob_vocab_chunk_size,
            )
        else:
            logits_shifted = logits[:, :-1, :]
            gathered_logps, _ = compute_token_logps_and_entropies_chunked(
                logits_shifted,
                labels_safe,
                return_entropy=False,
                chunk_size=logprob_vocab_chunk_size,
            )
            completion_logps = jnp.where(loss_mask, gathered_logps, 0.0).sum(axis=1)
            token_logit_sums = jnp.sum(
                jnp.where(loss_mask, logits_shifted.astype(jnp.float32).sum(axis=-1), 0.0),
                axis=1,
            )
            token_counts = jnp.sum(loss_mask.astype(jnp.float32), axis=1)

        mean_logits = token_logit_sums.sum() / jnp.maximum(token_counts.sum(), 1.0)

    output = {
        "completion_logps": completion_logps,
        "completion_logits": logits,
        "mean_completion_logits": mean_logits,
    }

    if aux_loss_enabled and hasattr(outputs, "aux_loss"):
        output["aux_loss"] = outputs.aux_loss
    return output


def compute_bco_loss(
    policy_logps: jax.Array,
    reference_logps: jax.Array,
    chosen_mask: jax.Array,
    rejected_mask: jax.Array,
    *,
    beta: float,
    delta: float,
    udm_weights: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the BCO logistic loss and per-example implicit rewards.

    The BCO objective treats the per-example reward
    ``r = beta * (log pi(y|x) - log pi_ref(y|x))`` as the score of an
    implicit binary classifier whose decision threshold is the running
    reference reward ``delta``. Desirable examples are pushed *above*
    ``delta`` and undesirable ones *below* it, giving:

    - ``chosen_loss   = -log_sigmoid(r - delta)``    (label=1)
    - ``rejected_loss = -log_sigmoid(-(r - delta))`` (label=0)

    Optionally, when UDM (underlying density model) reweighting is
    enabled, undesirable examples are scaled by their density-ratio
    weights to correct for prompt distribution mismatch between the
    desirable and undesirable streams. The final scalar loss is the
    weighted average.

    Args:
        policy_logps: ``[batch]`` summed completion log-probs under the
            policy.
        reference_logps: ``[batch]`` summed completion log-probs under
            the frozen reference (typically the SFT model).
        chosen_mask: Boolean ``[batch]`` mask selecting desirable
            (positive-label) examples.
        rejected_mask: Boolean ``[batch]`` mask selecting undesirable
            examples. ``chosen_mask`` and ``rejected_mask`` should be
            disjoint.
        beta: Inverse-temperature on the implicit reward.
        delta: Running estimate of the mean implicit reward used as
            the classifier threshold (maintained outside this function
            via :class:`RunningMoments`).
        udm_weights: Optional ``[batch]`` density-ratio weights applied
            to the undesirable slice. ``None`` disables UDM
            reweighting.

    Returns:
        A 7-tuple ``(loss, chosen_rewards, rejected_rewards,
        chosen_losses, rejected_losses, chosen_mask_f, rejected_mask_f)``
        where ``loss`` is the scalar (weighted) mean BCO loss, the
        ``*_rewards`` arrays carry the unmasked per-example implicit
        rewards (for logging/delta updates), and the ``*_losses`` /
        ``*_mask_f`` arrays carry the per-example masked losses and
        float-cast masks.
    """

    chosen_mask_f = chosen_mask.astype(policy_logps.dtype)
    rejected_mask_f = rejected_mask.astype(policy_logps.dtype)

    reward_delta = beta * (policy_logps - reference_logps)
    chosen_rewards = reward_delta * chosen_mask_f
    rejected_rewards = reward_delta * rejected_mask_f

    chosen_losses = jnp.where(chosen_mask_f > 0, -log_sigmoid(chosen_rewards - delta), 0.0)
    rejected_losses = jnp.where(rejected_mask_f > 0, -log_sigmoid(-(rejected_rewards - delta)), 0.0)

    chosen_count = jnp.maximum(chosen_mask_f.sum(), 0.0)
    rejected_count = jnp.maximum(rejected_mask_f.sum(), 0.0)

    if udm_weights is not None:
        rejected_weights = udm_weights.astype(policy_logps.dtype) * rejected_mask_f
        total_weight = jnp.maximum(chosen_count + rejected_weights.sum(), 1.0)
        loss = (chosen_losses.sum() + (rejected_losses * rejected_weights).sum()) / total_weight
    else:
        total_weight = jnp.maximum(chosen_count + rejected_count, 1.0)
        loss = (chosen_losses.sum() + rejected_losses.sum()) / total_weight

    return (
        loss,
        chosen_rewards,
        rejected_rewards,
        chosen_losses,
        rejected_losses,
        chosen_mask_f,
        rejected_mask_f,
    )


def training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState | None,
    learning_rate_fn: tp.Callable | None,
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
    loss_config: LossConfig | None,
    partition_spec: PartitionSpec | None,
    gradient_accumulation_steps: int,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Run one BCO training step (forward, loss, backward, optimizer update).

    Drives the binary-classifier objective: each example carries a
    boolean ``label`` (desirable vs. undesirable) and the loss tries to
    place the implicit reward ``beta * (logp - logp_ref)`` on the
    correct side of the running ``delta`` threshold (see
    :func:`compute_bco_loss`). The threshold itself is tracked across
    steps via :class:`RunningMoments`; the trainer passes the current
    estimate in ``batch["running_mean"]``.

    Pipeline inside the step:

    1. Pop ``running_mean`` (the BCO ``delta``) out of the batch so it
       is treated as a scalar parameter rather than a per-row feature.
    2. Resolve the gradient-accumulation minibatch size and shard the
       batch under the model's mesh.
    3. ``minibatch_call`` computes value-and-grad of the inner
       :func:`calculate_loss` closure, which:

       - Optionally rewrites ``tree`` through ``straight_through_emulator``.
       - Forwards the policy through ``concatenated_forward_fn`` to get
         summed completion logps.
       - Forwards the *reference* model (or reads precomputed
         ``reference_logps`` from the batch) and stop-gradient's the
         result.
       - Optionally weights undesirable rows with UDM
         ``udm_weights`` from the batch.
       - Calls :func:`compute_bco_loss` and adds any model
         ``aux_loss``.
    4. ``update_state_respectfully`` applies the gradients with NaN
       guards from ``loss_config``.

    Args:
        state: Policy state being differentiated.
        batch: BCO minibatch with ``prompt_*``/``completion_*``,
            boolean ``label``, optional ``reference_logps``, optional
            ``udm_weights``, and the scalar ``running_mean`` (= BCO
            ``delta``).
        reference_state: Frozen reference state. Falls back to the
            policy itself when ``None`` *and* the batch lacks
            ``reference_logps``.
        learning_rate_fn: Schedule mapping step to learning rate.
        concatenated_forward_fn: Captured forward closure with
            tokenization knobs (max length, truncation mode, ...) baked
            in.
        beta: BCO inverse-temperature.
        loss_config: ``LossConfig`` controlling NaN handling.
        partition_spec: Sharding spec applied to the input batch.
        gradient_accumulation_steps: Number of accumulation
            sub-steps; the batch must be evenly divisible.
        straight_through_emulator: Optional STE callable applied to
            the graphstate before the forward (QAT path).

    Returns:
        ``(new_state, metrics)`` where ``metrics`` is a ``LossMetrics``
        instance with per-example chosen/rejected rewards and the
        scalar BCO loss.
    """

    running_delta = batch.get("running_mean")
    if running_delta is None:
        running_delta = jnp.array(0.0, dtype=jnp.float32)
    else:
        running_delta = jnp.asarray(running_delta).reshape(())

    step_batch = dict(batch)
    step_batch.pop("running_mean", None)

    _, minibatch_size, batch_partition_spec = make_assertions_and_get_sizes(
        batch=step_batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    step_batch = with_sharding_constraint(step_batch, batch_partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: jax.ArrayTree, call_batch: dict[str, jax.Array]):
        """Compute the BCO loss for one minibatch.

        Runs the concatenated forward (and the reference forward when no
        precomputed ``reference_logps`` is provided), assembles the
        chosen/rejected masks from binary labels, optionally weights the
        rejected slice by UDM density estimates, and returns the
        combined logistic loss alongside diagnostic metrics.

        Args:
            tree: The current policy graphstate (differentiated against).
            call_batch: One minibatch slice of the BCO batch.

        Returns:
            ``(loss, metrics)`` where ``loss`` is the scalar BCO loss
            and ``metrics`` is a populated :class:`LossMetrics`.
        """
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        model = state.merge(tree=tree)
        policy_outputs = concatenated_forward_fn(model, call_batch)
        completion_logps = policy_outputs["completion_logps"]

        labels = call_batch["label"].astype(bool)
        chosen_mask = labels
        rejected_mask = jnp.logical_not(labels)

        if "reference_logps" in call_batch:
            reference_completion_logps = call_batch["reference_logps"]
        else:
            if reference_state is None:
                reference_model = model
            else:
                reference_model = reference_state.model
            ref_outputs = concatenated_forward_fn(reference_model, call_batch)
            reference_completion_logps = jax.lax.stop_gradient(ref_outputs["completion_logps"])

        udm_weights = call_batch.get("udm_weights", None)
        if udm_weights is not None:
            rejected_weights = jnp.where(rejected_mask, udm_weights.astype(completion_logps.dtype), 0.0)
        else:
            rejected_weights = None

        (
            losses,
            chosen_rewards_masked,
            rejected_rewards_masked,
            chosen_losses,
            rejected_losses,
            chosen_valid,
            rejected_valid,
        ) = compute_bco_loss(
            completion_logps,
            reference_completion_logps,
            chosen_mask,
            rejected_mask,
            beta=beta,
            delta=running_delta,
            udm_weights=rejected_weights,
        )

        if policy_outputs.get("aux_loss") is not None:
            losses = losses + policy_outputs["aux_loss"]

        metrics = LossMetrics(
            loss=losses,
            chosen_rewards=chosen_rewards_masked,
            rejected_rewards=rejected_rewards_masked,
            other_metrics={
                "delta": running_delta,
                "logps/chosen": (completion_logps * chosen_mask.astype(completion_logps.dtype)).sum()
                / jnp.maximum(chosen_valid.sum(), 1.0),
                "logps/rejected": (completion_logps * rejected_mask.astype(completion_logps.dtype)).sum()
                / jnp.maximum(rejected_valid.sum(), 1.0),
                "logits/mean": policy_outputs["mean_completion_logits"],
                "count/chosen": chosen_valid.sum(),
                "count/rejected": rejected_valid.sum(),
                "loss/chosen": chosen_losses.sum() / jnp.maximum(chosen_valid.sum(), 1.0),
                "loss/rejected": rejected_losses.sum() / jnp.maximum(rejected_valid.sum(), 1.0),
            },
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=step_batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
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


def _prepare_bco_scheduled_batch(call) -> dict[str, tp.Any]:
    """Inject precomputed reference log-probabilities into ``call.batch``.

    When the batch already carries ``reference_logps`` it is returned
    unchanged.  Otherwise the helper invokes the reference state to
    compute completion log-probabilities and stashes them under
    ``reference_logps`` so the scheduled loss can run without re-running
    the reference forward inside the VJP.

    Args:
        call: The :class:`ScheduledStepCall` describing the current
            step.

    Returns:
        A copy of ``call.batch`` with ``reference_logps`` populated.
    """
    batch = dict(call.batch)
    if "reference_logps" in batch:
        return batch

    return prepare_scheduled_reference_outputs(
        call,
        reference_state_field="reference_state",
        forward_field="concatenated_forward_fn",
        output_to_batch={"completion_logps": "reference_logps"},
        missing_error="BCO scheduled MPMD training requires reference_state and concatenated_forward_fn.",
    )


def _bco_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the BCO scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple suitable for keying a per-trainer cache of compiled
        scheduled losses (covers ``beta`` / partition spec / forward fn
        / quantization emulator).
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "partition_spec"),
        object_fields=("concatenated_forward_fn", "straight_through_emulator"),
    )


def _make_bco_scheduled_loss(call):
    """Build a SpectraX-scheduled BCO loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` providing forward fn,
            beta, and partition spec.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready to be wrapped
        with :func:`spx.sxvalue_and_grad`.
    """
    concatenated_forward_fn = call.get("concatenated_forward_fn")
    beta = call.get("beta")
    partition_spec = call.get("partition_spec")
    call.get("straight_through_emulator")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the BCO scalar loss for the scheduled-VJP path.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Minibatch dict carrying ``label``, ``reference_logps``
                and (optionally) ``udm_weights``.

        Returns:
            The scalar BCO loss (with optional aux-loss term added).

        Raises:
            RuntimeError: If ``reference_logps`` is missing from the
                batch.
        """
        running_delta = batch.get("running_mean")
        if running_delta is None:
            running_delta = jnp.array(0.0, dtype=jnp.float32)
        else:
            running_delta = jnp.asarray(running_delta).reshape(())

        call_batch = dict(batch)
        call_batch.pop("running_mean", None)
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, call_batch, partition_spec)

        policy_outputs = concatenated_forward_fn(module, call_batch)
        completion_logps = policy_outputs["completion_logps"]

        reference_completion_logps = call_batch.get("reference_logps")
        if reference_completion_logps is None:
            raise RuntimeError("BCO scheduled MPMD loss requires precomputed reference_logps in the batch.")
        reference_completion_logps = jax.lax.stop_gradient(reference_completion_logps)

        labels = call_batch["label"].astype(bool)
        chosen_mask = labels
        rejected_mask = jnp.logical_not(labels)
        udm_weights = call_batch.get("udm_weights", None)
        rejected_weights = (
            jnp.where(rejected_mask, udm_weights.astype(completion_logps.dtype), 0.0)
            if udm_weights is not None
            else None
        )

        loss, *_ = compute_bco_loss(
            completion_logps,
            reference_completion_logps,
            chosen_mask,
            rejected_mask,
            beta=beta,
            delta=running_delta,
            udm_weights=rejected_weights,
        )
        if policy_outputs.get("aux_loss") is not None:
            loss = loss + policy_outputs["aux_loss"]
        return loss

    return scheduled_loss


register_scheduled_loss_adapter(
    training_step,
    ScheduledLossAdapter(
        name="bco",
        make_loss=_make_bco_scheduled_loss,
        make_cache_key=_bco_scheduled_loss_cache_key,
        prepare_batch=_prepare_bco_scheduled_batch,
    ),
)


def evaluation_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState | None,
    concatenated_forward_fn: tp.Callable[..., dict[str, jax.Array]],
    beta: float,
) -> LossMetrics:
    """Run one BCO evaluation step (forward only, no parameter update).

    Computes the BCO logistic loss on the eval batch using the same
    desirable/undesirable masking and (optional) UDM reweighting as
    :func:`training_step`, but skips gradient accumulation and
    optimizer interaction. The threshold ``delta`` is pulled from
    ``batch["running_mean"]`` if present (defaults to ``0.0``); UDM
    weights are read from ``batch["udm_weights"]`` when supplied.

    Args:
        state: Policy state to evaluate.
        batch: BCO evaluation minibatch (same structure as the
            training batch).
        reference_state: Frozen reference state; when ``None`` and the
            batch lacks ``reference_logps``, the policy stands in as
            its own reference (purely diagnostic).
        concatenated_forward_fn: Forward closure with tokenization
            knobs baked in.
        beta: BCO inverse-temperature.

    Returns:
        ``LossMetrics`` populated with ``loss`` and the per-example
        ``chosen_rewards`` / ``rejected_rewards`` arrays.
    """

    running_delta = batch.get("running_mean")
    if running_delta is None:
        running_delta = jnp.array(0.0, dtype=jnp.float32)
    else:
        running_delta = jnp.asarray(running_delta).reshape(())

    policy_outputs = concatenated_forward_fn(state.model, batch)
    completion_logps = policy_outputs["completion_logps"]

    labels = batch["label"].astype(bool)
    chosen_mask = labels
    rejected_mask = jnp.logical_not(labels)

    if "reference_logps" in batch:
        reference_completion_logps = batch["reference_logps"]
    else:
        if reference_state is None:
            reference_model = state.model
        else:
            reference_model = reference_state.model
        ref_outputs = concatenated_forward_fn(reference_model, batch)
        reference_completion_logps = ref_outputs["completion_logps"]

    udm_weights = batch.get("udm_weights", None)
    if udm_weights is not None:
        rejected_weights = jnp.where(rejected_mask, udm_weights.astype(completion_logps.dtype), 0.0)
    else:
        rejected_weights = None

    (
        losses,
        chosen_rewards,
        rejected_rewards,
        chosen_losses,
        rejected_losses,
        chosen_valid,
        rejected_valid,
    ) = compute_bco_loss(
        completion_logps,
        reference_completion_logps,
        chosen_mask,
        rejected_mask,
        beta=beta,
        delta=running_delta,
        udm_weights=rejected_weights,
    )

    metrics = LossMetrics(
        loss=losses,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        other_metrics={
            "delta": running_delta,
            "logps/chosen": (completion_logps * chosen_mask.astype(completion_logps.dtype)).sum()
            / jnp.maximum(chosen_valid.sum(), 1.0),
            "logps/rejected": (completion_logps * rejected_mask.astype(completion_logps.dtype)).sum()
            / jnp.maximum(rejected_valid.sum(), 1.0),
            "logits/mean": policy_outputs["mean_completion_logits"],
            "count/chosen": chosen_valid.sum(),
            "count/rejected": rejected_valid.sum(),
            "loss/chosen": chosen_losses.sum() / jnp.maximum(chosen_valid.sum(), 1.0),
            "loss/rejected": rejected_losses.sum() / jnp.maximum(rejected_valid.sum(), 1.0),
            "mask/chosen": chosen_valid,
            "mask/rejected": rejected_valid,
        },
    )
    return metrics


def detach_metrics(metrics: LossMetrics) -> LossMetrics:
    """Convert metrics to host scalars for serialization.

    Args:
        metrics: Loss metrics to detach.

    Returns:
        Detached metrics with scalar values.
    """

    if metrics.other_metrics is not None:
        filtered_other = {}
        for key, value in metrics.other_metrics.items():
            if key.startswith("mask/"):
                continue
            filtered_other[key] = float(value)
    else:
        filtered_other = None

    return metrics.replace(
        loss=float(metrics.loss),
        chosen_rewards=None,
        rejected_rewards=None,
        other_metrics=filtered_other,
    )


__all__ = [
    "RunningMoments",
    "compute_bco_loss",
    "concatenated_forward",
    "detach_metrics",
    "evaluation_step",
    "training_step",
]
