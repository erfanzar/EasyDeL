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
"""Loss and step implementations for the KTO trainer.

Implements the per-example KTO loss (a prospect-theory-inspired
sigmoid-of-margin function), the running KL estimator used to
"normalise" desirable-vs-undesirable completion rewards, and the
scheduled-VJP variants for MPMD pipeline parallelism.
"""

from __future__ import annotations

import typing as tp

import jax
import spectrax as spx
from jax import numpy as jnp
from jax.nn import sigmoid
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    cached_scheduled_auxiliary,
    constrain_scheduled_batch,
    make_assertions_and_get_sizes,
    minibatch_call,
    prepare_scheduled_reference_outputs,
    register_scheduled_loss_adapter,
    scheduled_loss_cache_key,
    stop_gradient_tree,
    sync_module_schedule_config,
    update_metrics,
    update_state_respectfully,
)

KTO_LOSS_TYPES = ("kto", "apo_zero_unpaired")


def _build_kl_batch(batch: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Create mismatched prompt/completion batch for KL estimation.

    Rolls completion sequences by one position to create mismatched pairs,
    enabling estimation of KL divergence between policy and reference.

    Args:
        batch: Original batch with prompts and completions.

    Returns:
        Batch with rolled completions for KL computation.
    """

    kl_batch: dict[str, jax.Array] = {
        "prompt_input_ids": batch["prompt_input_ids"],
        "prompt_attention_mask": batch["prompt_attention_mask"],
    }

    for key in ("pixel_values", "pixel_attention_mask", "image_sizes"):
        if key in batch:
            kl_batch[key] = batch[key]

    def _rolled(name: str):
        """Insert a 1-position rolled copy of ``batch[name]`` into ``kl_batch``.

        Used to build the KL-estimation batch by pairing each prompt
        with the *next* example's completion.

        Args:
            name: Batch column name to roll into the KL batch.
        """
        if name in batch:
            kl_batch[name] = jnp.roll(batch[name], shift=1, axis=0)

    for field in (
        "completion_input_ids",
        "completion_attention_mask",
        "completion_labels",
        "completion_decoder_input_ids",
    ):
        _rolled(field)

    return kl_batch


def kto_objective(
    policy_logps: jax.Array,
    reference_logps: jax.Array,
    labels: jax.Array,
    *,
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_type: str,
    policy_kl_logps: jax.Array | None = None,
    reference_kl_logps: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Compute the KTO (or APO-zero-unpaired) loss and per-example implicit rewards.

    Implements the unpaired prospect-theoretic objective from
    Ethayarajh et al. 2024. Given the policy/reference summed
    completion logps and a binary ``labels`` array, the loss is built
    around the implicit reward
    ``r = beta * (logp - logp_ref)`` and a non-negative running KL
    estimate ``z = ReLU(mean(policy_kl_logps - reference_kl_logps))``
    obtained from a *mismatched* (prompt_i, completion_{i+1}) batch
    (see :func:`_build_kl_batch`).

    For ``loss_type == "kto"`` the loss is

    - ``L_chosen   = 1 - sigmoid(beta * (r - z))``  on desirable rows
    - ``L_rejected = 1 - sigmoid(beta * (z - r))``  on undesirable rows

    so each side is shaped relative to the running KL anchor ``z``.

    For ``loss_type == "apo_zero_unpaired"`` the rewards are anchored
    at zero instead of ``z`` and the rejected branch is symmetric:

    - ``L_chosen   = 1 - sigmoid(beta * r)``       (push reward up)
    - ``L_rejected =     sigmoid(beta * r)``       (push reward down)

    The two halves are then weighted by ``desirable_weight`` /
    ``undesirable_weight`` and averaged over the *count* of examples
    (not separately per side) so the optimisation is robust to
    label imbalance. ``z`` is always stop-gradient'd.

    Args:
        policy_logps: ``[batch]`` summed completion logps under the
            policy.
        reference_logps: ``[batch]`` summed completion logps under
            the frozen reference.
        labels: Boolean ``[batch]`` labels (``True`` = desirable).
        beta: KTO inverse-temperature.
        desirable_weight: Loss multiplier for desirable rows.
        undesirable_weight: Loss multiplier for undesirable rows.
        loss_type: ``"kto"`` or ``"apo_zero_unpaired"``.
        policy_kl_logps: Optional ``[batch]`` policy logps on the
            mismatched-prompt KL batch. ``None`` disables the KL
            anchor (``z = 0``).
        reference_kl_logps: Optional reference logps on the same
            mismatched-prompt batch.

    Returns:
        ``(loss, chosen_rewards, rejected_rewards, kl)`` where
        ``loss`` is the scalar weighted KTO loss, the reward arrays
        are masked per-example implicit rewards (for logging), and
        ``kl`` is the stop-gradient'd scalar KL anchor.

    Raises:
        ValueError: If ``loss_type`` is unrecognised.
    """

    if loss_type not in KTO_LOSS_TYPES:
        raise ValueError(f"Unsupported KTO loss type: {loss_type}")

    dtype = policy_logps.dtype
    labels_bool = labels.astype(bool)
    chosen_mask = labels_bool.astype(dtype)
    rejected_mask = (~labels_bool).astype(dtype)

    logratios = policy_logps - reference_logps

    if policy_kl_logps is not None and reference_kl_logps is not None:
        kl = jnp.maximum(jnp.mean(policy_kl_logps - reference_kl_logps), 0.0)
    else:
        kl = jnp.zeros((), dtype=dtype)
    kl = jax.lax.stop_gradient(kl)

    def _safe_sigmoid(x):
        """Numerically stable sigmoid (inputs clipped to ``[-30, 30]``).

        Args:
            x: Pre-activation values.

        Returns:
            ``sigmoid(clip(x, -30, 30))``.
        """
        return sigmoid(jnp.clip(x, -30.0, 30.0))

    if loss_type == "kto":
        chosen_term = beta * (logratios - kl)
        rejected_term = beta * (kl - logratios)
        chosen_losses = chosen_mask * (1.0 - _safe_sigmoid(chosen_term))
        rejected_losses = rejected_mask * (1.0 - _safe_sigmoid(rejected_term))
    else:  # apo_zero_unpaired
        chosen_term = beta * logratios
        rejected_term = beta * logratios
        chosen_losses = chosen_mask * (1.0 - _safe_sigmoid(chosen_term))
        rejected_losses = rejected_mask * _safe_sigmoid(rejected_term)

    chosen_rewards = beta * logratios * chosen_mask
    rejected_rewards = beta * logratios * rejected_mask

    total_examples = jnp.maximum(chosen_mask.sum() + rejected_mask.sum(), 1.0)
    weighted_chosen = desirable_weight * chosen_losses.sum()
    weighted_rejected = undesirable_weight * rejected_losses.sum()
    loss = (weighted_chosen + weighted_rejected) / total_examples

    return loss, chosen_rewards, rejected_rewards, kl


def training_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState,
    learning_rate_fn: tp.Callable[[jax.Array], jax.Array],
    forward_fn: tp.Callable[[EasyDeLState | EasyDeLState.model, dict[str, jax.Array]], dict[str, jax.Array]],
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_type: str,
    calculate_kl: bool,
    aux_loss_coef: float,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Run one KTO training step (forward, loss, backward, optimizer update).

    Drives the unpaired KTO objective: each row carries a binary
    ``label`` and the loss anchors the implicit reward against the
    running KL estimate ``z`` (see :func:`kto_objective`). When
    ``calculate_kl`` is ``True`` the trainer also forwards a
    *mismatched-completion* KL batch (built by
    :func:`_build_kl_batch`) through both policy and reference so
    ``z`` reflects the current divergence between the two
    distributions.

    Pipeline inside the step:

    1. Resolve the gradient-accumulation minibatch size and shard the
       batch under the model's mesh.
    2. ``minibatch_call`` computes value-and-grad of the inner
       :func:`calculate_loss` closure, which:

       - Optionally rewrites ``tree`` through ``straight_through_emulator``.
       - Forwards the policy and (when not precomputed) the
         reference through ``forward_fn`` on both the standard batch
         and the mismatched KL batch.
       - Calls :func:`kto_objective` and adds an aux loss scaled by
         ``aux_loss_coef`` if the model exposes one.
    3. ``update_state_respectfully`` applies the gradients with NaN
       guards from ``loss_config``.

    Args:
        state: Policy ``EasyDeLState`` being differentiated.
        batch: KTO minibatch with ``prompt_*`` / ``completion_*``,
            boolean ``label``, and (optionally) precomputed
            ``reference_logps`` and ``reference_kl_logps``.
        reference_state: Frozen reference state used when reference
            logps are not already cached on the batch.
        learning_rate_fn: Schedule mapping step to learning rate.
        forward_fn: Forward closure that returns
            ``{"completion_logps": ..., "aux_loss": ...}`` (and is
            also driven on the mismatched KL batch).
        beta: KTO inverse-temperature.
        desirable_weight: Loss multiplier for desirable rows.
        undesirable_weight: Loss multiplier for undesirable rows.
        loss_type: ``"kto"`` or ``"apo_zero_unpaired"``.
        calculate_kl: When ``True``, the mismatched KL batch is built
            and run to maintain a non-zero ``z`` anchor.
        aux_loss_coef: Multiplier on any auxiliary load-balance loss
            returned by the model forward.
        loss_config: ``LossConfig`` controlling NaN handling.
        partition_spec: Sharding spec applied to the input batch.
        gradient_accumulation_steps: Gradient-accumulation factor.
        straight_through_emulator: Optional STE callable applied to
            the graphstate before the forward (QAT path).

    Returns:
        ``(new_state, metrics)`` where ``metrics`` carries the scalar
        KTO loss, per-example chosen/rejected rewards, the running
        ``kl`` anchor, and any auxiliary metrics in ``other_metrics``.
    """

    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    batch = dict(batch)
    if "reference_logps" not in batch:
        ref_out = forward_fn(reference_state.model, batch)
        batch["reference_logps"] = jax.lax.stop_gradient(ref_out["completion_logps"])

    if calculate_kl:
        kl_batch = _build_kl_batch(batch)
        ref_kl_out = forward_fn(reference_state.model, kl_batch)
        batch["_reference_kl_logps"] = jax.lax.stop_gradient(ref_kl_out["completion_logps"])

    def _loss_fn(tree: spx.State, minibatch: dict[str, jax.Array]):
        """Compute the KTO loss for one minibatch.

        Runs the policy concatenated forward, optionally builds the KL
        batch (rolled completions) for both policy and reference,
        evaluates :func:`kto_objective`, and folds in any auxiliary
        loss the policy reports.

        Args:
            tree: Policy graphstate to differentiate against.
            minibatch: One minibatch slice with ``label`` and the
                precomputed reference (and optional KL) log-probs.

        Returns:
            ``(loss, metrics)`` with chosen / rejected rewards and the
            KL diagnostic recorded under ``other_metrics["kl"]``.
        """
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)
        policy_out = forward_fn(module, minibatch)
        policy_logps = policy_out["completion_logps"]

        reference_logps = jax.lax.stop_gradient(minibatch["reference_logps"])

        if calculate_kl:
            kl_batch = _build_kl_batch(minibatch)
            policy_kl_logps = jax.lax.stop_gradient(forward_fn(module, kl_batch)["completion_logps"])
            reference_kl_logps = jax.lax.stop_gradient(minibatch["_reference_kl_logps"])
        else:
            policy_kl_logps = reference_kl_logps = None

        loss, chosen_rewards, rejected_rewards, kl = kto_objective(
            policy_logps,
            reference_logps,
            minibatch["label"],
            beta=beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            loss_type=loss_type,
            policy_kl_logps=policy_kl_logps,
            reference_kl_logps=reference_kl_logps,
        )

        if aux_loss_coef > 0.0 and "aux_loss" in policy_out:
            loss = loss + aux_loss_coef * policy_out["aux_loss"]

        metrics = LossMetrics(
            loss=loss,
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            other_metrics={"kl": kl},
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(_loss_fn, has_aux=True),
    )

    metrics = update_metrics(
        metrics=metrics,
        learning_rate_fn=learning_rate_fn,
        step=state.step,
        gradients=gradients,
    )
    state = update_state_respectfully(
        state=state,
        gradients=gradients,
        loss_config=loss_config,
        metrics=metrics,
    )
    return state, metrics


def _prepare_kto_scheduled_batch(call) -> dict[str, tp.Any]:
    """Inject precomputed reference (and KL) log-probabilities into ``call.batch``.

    Always populates ``reference_logps``; when ``calculate_kl`` is
    enabled also runs the rolled-completion KL batch through both the
    policy and reference models and stashes their stop-gradient
    completion log-probs.

    Args:
        call: The :class:`ScheduledStepCall` describing the current
            step.

    Returns:
        A copy of ``call.batch`` ready for the scheduled loss closure.

    Raises:
        RuntimeError: If reference state / forward fn are missing while
            KL estimation is requested.
    """
    batch = dict(call.batch)
    if "reference_logps" not in batch:
        batch = prepare_scheduled_reference_outputs(
            call,
            reference_state_field="reference_state",
            forward_field="forward_fn",
            output_to_batch={"completion_logps": "reference_logps"},
            missing_error="KTO scheduled MPMD training requires reference_state and forward_fn.",
        )

    if not bool(call.get("calculate_kl", False)):
        return batch

    if "_reference_kl_logps" in batch and "_policy_kl_logps" in batch:
        return batch

    reference_state = call.get("reference_state")
    forward_fn = call.get("forward_fn")
    if reference_state is None or forward_fn is None:
        raise RuntimeError("KTO scheduled MPMD KL requires reference_state and forward_fn.")

    kl_batch = _build_kl_batch(batch)
    partition_spec = call.get("partition_spec")

    ref_model = reference_state.model
    ref_model.eval()
    sync_module_schedule_config(ref_model, call.schedule)
    ref_kl_batch = constrain_scheduled_batch(ref_model, kl_batch, partition_spec)
    ref_forward = cached_scheduled_auxiliary(forward_fn, ref_model.mesh)
    ref_out = stop_gradient_tree(ref_forward(ref_model, ref_kl_batch))
    batch["_reference_kl_logps"] = ref_out["completion_logps"]

    policy_model = call.state.model
    sync_module_schedule_config(policy_model, call.schedule)
    policy_kl_batch = constrain_scheduled_batch(policy_model, kl_batch, partition_spec)
    policy_forward = cached_scheduled_auxiliary(forward_fn, policy_model.mesh)
    policy_out = stop_gradient_tree(policy_forward(policy_model, policy_kl_batch))
    batch["_policy_kl_logps"] = policy_out["completion_logps"]
    return batch


def _kto_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the KTO scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering ``beta``, the desirable / undesirable weights,
        the loss type, KL flag, ``aux_loss_coef``, partition spec, and
        the forward-fn / quantizer identities.
    """
    return scheduled_loss_cache_key(
        call,
        value_fields=(
            "beta",
            "desirable_weight",
            "undesirable_weight",
            "loss_type",
            "calculate_kl",
            "aux_loss_coef",
            "partition_spec",
        ),
        object_fields=("forward_fn", "straight_through_emulator"),
    )


def _make_kto_scheduled_loss(call):
    """Build a SpectraX-scheduled KTO scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` carrying the trainer's
            current configuration.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready for
        :func:`spx.sxvalue_and_grad`.
    """
    forward_fn = call.get("forward_fn")
    beta = call.get("beta")
    desirable_weight = call.get("desirable_weight")
    undesirable_weight = call.get("undesirable_weight")
    loss_type = call.get("loss_type")
    calculate_kl = bool(call.get("calculate_kl", False))
    aux_loss_coef = float(call.get("aux_loss_coef", 0.0))
    partition_spec = call.get("partition_spec")

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar KTO loss inside the SpectraX scheduled VJP.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Minibatch dict carrying ``label``, precomputed
                reference (and optional KL) log-probs, and the
                trainer-supplied forward-friendly fields.

        Returns:
            The scalar KTO loss with optional aux-loss term.
        """
        module = bind_scheduled_module(call, tree)
        call_batch = constrain_scheduled_batch(module, batch, partition_spec)
        policy_out = forward_fn(module, call_batch)
        policy_logps = policy_out["completion_logps"]
        reference_logps = jax.lax.stop_gradient(call_batch["reference_logps"])

        if calculate_kl:
            policy_kl_logps = jax.lax.stop_gradient(call_batch["_policy_kl_logps"])
            reference_kl_logps = jax.lax.stop_gradient(call_batch["_reference_kl_logps"])
        else:
            policy_kl_logps = reference_kl_logps = None

        loss, *_ = kto_objective(
            policy_logps,
            reference_logps,
            call_batch["label"],
            beta=beta,
            desirable_weight=desirable_weight,
            undesirable_weight=undesirable_weight,
            loss_type=loss_type,
            policy_kl_logps=policy_kl_logps,
            reference_kl_logps=reference_kl_logps,
        )

        if aux_loss_coef > 0.0 and "aux_loss" in policy_out:
            loss = loss + aux_loss_coef * policy_out["aux_loss"]
        return loss

    return scheduled_loss


register_scheduled_loss_adapter(
    training_step,
    ScheduledLossAdapter(
        name="kto",
        make_loss=_make_kto_scheduled_loss,
        make_cache_key=_kto_scheduled_loss_cache_key,
        prepare_batch=_prepare_kto_scheduled_batch,
    ),
)


def evaluation_step(
    state: EasyDeLState,
    batch: dict[str, jax.Array],
    reference_state: EasyDeLState,
    forward_fn: tp.Callable[[EasyDeLState | EasyDeLState.model, dict[str, jax.Array]], dict[str, jax.Array]],
    beta: float,
    desirable_weight: float,
    undesirable_weight: float,
    loss_type: str,
    calculate_kl: bool,
    aux_loss_coef: float,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Run one KTO evaluation step (forward only, no parameter update).

    Mirrors :func:`training_step` minus the gradient and optimizer
    plumbing: forwards the policy (and, when ``calculate_kl`` is set,
    the mismatched KL batch) through ``forward_fn``, retrieves
    reference logps from either ``batch`` or by running
    ``reference_state``, and reports the KTO loss alongside per-row
    rewards and the running KL anchor.

    Args:
        state: Policy state to evaluate.
        batch: KTO evaluation minibatch (same structure as the
            training batch).
        reference_state: Frozen reference state used when reference
            logps are not already cached.
        forward_fn: Forward closure returning summed completion
            logps plus optional aux losses.
        beta: KTO inverse-temperature.
        desirable_weight: Loss multiplier for desirable rows.
        undesirable_weight: Loss multiplier for undesirable rows.
        loss_type: ``"kto"`` or ``"apo_zero_unpaired"``.
        calculate_kl: When ``True``, the mismatched KL batch is also
            evaluated.
        aux_loss_coef: Multiplier on any auxiliary load-balance loss.
        partition_spec: Sharding specification.

    Returns:
        Loss metrics.
    """

    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )
    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    policy_out = forward_fn(state.model, batch)
    policy_logps = policy_out["completion_logps"]

    if "reference_logps" in batch:
        reference_logps = batch["reference_logps"]
    else:
        reference_logps = forward_fn(reference_state.model, batch)["completion_logps"]

    if calculate_kl:
        kl_batch = _build_kl_batch(batch)
        policy_kl_logps = forward_fn(state.model, kl_batch)["completion_logps"]
        reference_kl_logps = forward_fn(reference_state.model, kl_batch)["completion_logps"]
    else:
        policy_kl_logps = reference_kl_logps = None

    loss, chosen_rewards, rejected_rewards, kl = kto_objective(
        policy_logps,
        reference_logps,
        batch["label"],
        beta=beta,
        desirable_weight=desirable_weight,
        undesirable_weight=undesirable_weight,
        loss_type=loss_type,
        policy_kl_logps=policy_kl_logps,
        reference_kl_logps=reference_kl_logps,
    )

    if aux_loss_coef > 0.0 and "aux_loss" in policy_out:
        loss = loss + aux_loss_coef * policy_out["aux_loss"]

    return LossMetrics(
        loss=loss,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        other_metrics={"kl": kl},
    )
