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

"""Internal functions for on-policy knowledge distillation training.

This module contains the core computational functions for on-policy distillation,
where the student (or teacher) generates completions from prompts and then
teacher and student logits are compared via KL divergence on the generated tokens.

The key difference from offline distillation is that the sequences being distilled
are generated on-the-fly during training, making the training distribution match
the student's own output distribution (on-policy).
"""

import collections.abc
import functools
import typing as tp

import jax
import optax  # pyright: ignore[reportMissingTypeStubs]
import spectrax as spx
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from spectrax import with_sharding_constraint

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from ..distillation_trainer._fn import (
    _resolve_vocab_parallel_mesh,
    chunked_distillation_loss,
    distillation_loss,
)
from ..training_utils import (
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    sanitize_model_call_kwargs,
    update_metrics,
    update_state_respectfully,
)
from ._streamed_divergences import (
    build_distillation_projections,
    streamed_distillation_loss,
    trust_region_distillation_loss,
)

# Batch key of the per-step trust-region acceptance uniforms (``[B, L]``, sliced per microbatch).
_TRUST_REGION_UNIFORM_KEY = "_trust_region_acceptance_uniform"


def _vocab_parallel_partition_manager(module: tp.Any) -> tp.Any:
    """Return a ``PartitionManager`` for the student when its stage mesh has a real (>1) ``tp`` axis.

    ``partition_manager`` lives on the model config, not on the module, so it cannot be read with
    ``getattr(module, ...)``. Mirrors the offline distillation step: build one from
    ``module.config.partition_axis`` only for genuine tensor parallelism.

    Args:
        module: The merged student module.

    Returns:
        A ``spx.PartitionManager`` when the stage-local mesh has ``tp > 1``, otherwise ``None``.
    """
    stage_mesh = spx.get_current_stage_mesh(getattr(module, "mesh", None), raise_error=False)
    if stage_mesh is not None and "tp" in stage_mesh.axis_names and stage_mesh.shape["tp"] > 1:
        return spx.PartitionManager(paxis=module.config.partition_axis)
    return None


def on_policy_distillation_step(
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
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
    logits_chunk_size: int | None = None,
    beta: float | None = None,
    cakld_gamma: float | None = None,
    loss_top_k: int = 0,
    loss_add_tail: bool = False,
    distillation_mode: str = "divergence",
    trust_region_top_k: int = 64,
    trust_region_seed: int = 0,
    sampling_temperature: float | None = None,
) -> tuple[EasyDeLState, LossMetrics] | LossMetrics:
    """Run one on-policy distillation step (forward, KL loss, optional update).

    Wraps the teacher forward in a hard rematerialisation barrier (no
    residuals saved, no CSE across the boundary, all outputs detached) so
    the teacher's intermediates can be freed as soon as their last
    consumer is done; the student forward then runs through the
    differentiable parameter tree. The KL distillation loss is masked to
    the ``completion_mask`` so only the *generated* tokens contribute,
    matching the on-policy distribution. In training mode the function
    runs gradient accumulation via :func:`minibatch_call` and applies the
    optimizer update; in eval mode it returns metrics only.

    The batch is expected to contain:

    * ``input_ids``: full prompt + completion sequences, shape ``[B, L]``.
    * ``attention_mask``: mask covering the full sequences, shape
      ``[B, L]``.
    * ``completion_mask``: 1s at positions corresponding to generated
      tokens, 0s elsewhere; shape ``[B, L]``.

    Args:
        student_state (EasyDeLState): Current student model state.
        batch (collections.abc.Mapping[str, jax.Array]): Batch of generated
            sequences with masks (see above).
        teacher_state (EasyDeLState): Frozen teacher model state.
        loss_config (LossConfig | None): Optional loss configuration
            consumed by :func:`update_state_respectfully`.
        learning_rate_fn (optax.Schedule): Learning-rate schedule used
            for metric reporting.
        partition_spec (PartitionSpec | None): Sharding specification
            applied to the input batch.
        gradient_accumulation_steps (int): Number of microbatches whose
            gradients are accumulated per optimizer step. Defaults to
            ``1``.
        is_training (bool): When ``True`` differentiate and update; when
            ``False`` only compute eval metrics. Defaults to ``True``.
        temperature (float): Temperature used to soften logits in the KL
            term. Defaults to ``4.0``.
        alpha (float): Weight applied to the distillation loss
            (``1.0`` = pure distillation). Defaults to ``0.9``.
        straight_through_emulator (tp.Callable | None): Optional STE
            wrapper applied to the parameter tree to simulate quantised
            forward passes during training.
        logits_chunk_size (int | None): When positive, evaluate the
            distillation loss in vocabulary chunks driven by the
            (headless) student LM head to bound peak memory; ``None``
            disables chunking. Plain forward KL uses the shared chunked KL
            kernel; the other selectors (``beta``, ``cakld_gamma``,
            ``loss_top_k``) stream the vocabulary in ``logits_chunk_size``
            columns (online logsumexp / running top-k) with identical
            objectives to the dense loss.
        beta: Optional generalized Jensen-Shannon coefficient, including
            forward/reverse KL endpoints. ``None`` selects forward KL.
        cakld_gamma: Optional confidence-aware KL mixture coefficient.
        loss_top_k: Teacher top-k support size; zero selects the full vocabulary.
        loss_add_tail: Include the probability tail in top-k distillation.
        distillation_mode: ``"divergence"`` (default) minimizes the selected
            divergence; ``"trust_region"`` uses speculative-decoding-style
            acceptance of sampled tokens (see
            :func:`trust_region_distillation_loss`): accepted tokens get a
            reverse-KL policy gradient, rejected tokens a teacher-top-k
            forward KL. ``alpha`` and ``temperature`` are not applied in
            this mode; all log-probs use ``sampling_temperature``.
        trust_region_top_k: Teacher support size of the rejected-token
            forward KL in trust-region mode.
        trust_region_seed: Base seed of the per-step acceptance draw. One
            ``[B, L]`` uniform is drawn per step from
            ``fold_in(PRNGKey(seed), step)`` for the *full* batch and sliced
            per gradient-accumulation microbatch, so every row gets an
            independent draw and the acceptance mask does not depend on
            ``gradient_accumulation_steps``.
        sampling_temperature: Temperature the completions were sampled at.
            Required in trust-region mode so the acceptance ratio compares
            the actual sampling distributions.

    Returns:
        tuple[EasyDeLState, LossMetrics] | LossMetrics: In training mode
        returns ``(updated_student_state, metrics)``; in eval mode
        returns only :class:`LossMetrics`.
    """
    scope_root = "easydel/trainer/on_policy_distillation/" + ("train_step" if is_training else "eval_step")
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )
        batch = with_sharding_constraint(batch, partition_spec, mesh=student_state.model.mesh, ignore_mpmd=True)

    if distillation_mode not in ("divergence", "trust_region"):
        raise ValueError(f"`distillation_mode` must be 'divergence' or 'trust_region'; got {distillation_mode!r}.")
    use_trust_region = distillation_mode == "trust_region"
    if use_trust_region:
        if sampling_temperature is None or not float(sampling_temperature) > 0.0:
            raise ValueError(
                "trust-region distillation needs the positive `sampling_temperature` the completions were sampled "
                f"at; got {sampling_temperature!r}."
            )
        # One uniform per (row, position) for the whole batch, sliced per microbatch by ``minibatch_call``.
        input_ids_full = batch["input_ids"]
        acceptance_key = jax.random.fold_in(
            jax.random.PRNGKey(int(trust_region_seed)),
            jnp.asarray(student_state.step, jnp.uint32),
        )
        uniform_full = jax.random.uniform(acceptance_key, input_ids_full.shape[:2], jnp.float32)
        uniform_full = with_sharding_constraint(
            uniform_full, partition_spec, mesh=student_state.model.mesh, ignore_mpmd=True
        )
        batch = {**batch, _TRUST_REGION_UNIFORM_KEY: uniform_full}
    # Plain forward KL keeps the shared chunked KL kernel. Every other selector streams the vocabulary
    # through ``streamed_distillation_loss`` (same objective as the dense loss), so large-vocab models never
    # materialize ``[B, T, V]`` logits once ``logits_chunk_size`` is set.
    use_advanced_vocab_loss = beta is not None or cakld_gamma is not None or loss_top_k > 0 or loss_add_tail
    chunking = logits_chunk_size is not None and logits_chunk_size > 0
    use_chunked = chunking and not use_advanced_vocab_loss and not use_trust_region
    use_streamed = use_trust_region or (chunking and use_advanced_vocab_loss)
    use_hidden = use_chunked or use_streamed
    shift_tokens = bool(getattr(loss_config, "shift_tokens", True)) if loss_config is not None else True

    def loss_fn(tree, minibatch):
        """Compute the on-policy distillation loss for a minibatch.

        Args:
            tree: The student parameter tree (subject to gradient).
            minibatch (collections.abc.Mapping[str, jax.Array]): Minibatch
                with ``input_ids``, ``attention_mask`` and the
                ``completion_mask`` selecting generated tokens.

        Returns:
            tuple[jax.Array, LossMetrics]: Scalar total loss and a
            :class:`LossMetrics` instance with per-component diagnostics.
        """
        if is_training and straight_through_emulator is not None:
            with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                tree = straight_through_emulator(tree)
        with jax.named_scope(scope_root + "/loss_fn/merge_state"):
            module = student_state.merge(tree)

        input_ids = minibatch["input_ids"]
        attention_mask = minibatch["attention_mask"]
        completion_mask = minibatch.get("completion_mask")

        teacher_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if use_hidden:
            teacher_kwargs["apply_lm_head"] = False
        teacher_kwargs = filter_kwargs_for_callable(
            getattr(teacher_state.model, "forward", teacher_state.model), teacher_kwargs
        )
        teacher_kwargs = sanitize_model_call_kwargs(teacher_kwargs)

        _teacher_static_kw = {
            k: teacher_kwargs.pop(k) for k in list(teacher_kwargs) if not hasattr(teacher_kwargs[k], "shape")
        }

        # prevent_cse=True prevents XLA from merging common subexpressions
        # (e.g. shared embedding lookups on the same input_ids) across the
        # checkpoint boundary, preserving it as a hard memory barrier.
        # nothing_saveable tells XLA to save zero residuals for backward —
        # combined with stop_gradient on all outputs, XLA knows there is no
        # backward through the teacher and can free all intermediates as soon
        # as the boundary's outputs are produced.
        @functools.partial(
            jax.checkpoint,
            prevent_cse=True,
            policy=jax.checkpoint_policies.nothing_saveable,
        )
        def _teacher_fwd(kw, t_graphstate):
            """Run the teacher forward pass under a hard rematerialization barrier.

            Args:
                kw: Dynamic keyword arguments (input_ids, attention_mask, ...).
                t_graphstate: Stop-gradient teacher graph state.

            Returns:
                dict: ``{"h": last_hidden_state}`` when chunked KL is enabled,
                otherwise ``{"l": logits}`` -- both detached from gradients.
            """
            teacher_module = teacher_state.merge(t_graphstate)
            out = teacher_module(**kw, **_teacher_static_kw)
            results = {}
            if use_hidden:
                results["h"] = jax.lax.stop_gradient(out.last_hidden_state)
            else:
                results["l"] = jax.lax.stop_gradient(out.logits)
            return results

        with jax.named_scope(scope_root + "/loss_fn/teacher_forward"):
            teacher_out = _teacher_fwd(
                teacher_kwargs,
                jax.lax.stop_gradient(teacher_state.graphstate),
            )
            teacher_hidden_for_kl = teacher_out["h"] if use_hidden else None
            teacher_logits = teacher_out["l"] if not use_hidden else None

        call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if use_hidden:
            call_kwargs["apply_lm_head"] = False
        call_kwargs = filter_kwargs_for_callable(getattr(module, "forward", module), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        with jax.named_scope(scope_root + "/loss_fn/student_forward"):
            student_outputs = module(**call_kwargs)

        # Compute distillation loss on generated tokens only (using completion_mask).
        # No hard labels are used since completions are generated, not from a dataset.
        with jax.named_scope(scope_root + "/loss_fn/distillation_loss"):
            if use_streamed:
                vocab_parallel = _resolve_vocab_parallel_mesh(_vocab_parallel_partition_manager(module)) is not None
                student_proj, teacher_proj, seq_chunk = build_distillation_projections(
                    module,
                    teacher_state.model,
                    int(logits_chunk_size) if chunking else None,
                    prefer_sequence_chunks=vocab_parallel,
                )
                if use_trust_region:
                    uniform = minibatch[_TRUST_REGION_UNIFORM_KEY]
                    if shift_tokens:
                        uniform = uniform[:, 1:]
                    total_loss, loss_components = trust_region_distillation_loss(
                        student_hidden=student_outputs.last_hidden_state,
                        teacher_hidden=teacher_hidden_for_kl,
                        student_projection=student_proj,
                        teacher_projection=teacher_proj,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        loss_mask=completion_mask,
                        temperature=float(sampling_temperature),
                        top_k=int(trust_region_top_k),
                        acceptance_uniform=uniform,
                        shift_tokens=shift_tokens,
                        sequence_chunk_size=seq_chunk,
                    )
                else:
                    total_loss, loss_components = streamed_distillation_loss(
                        student_hidden=student_outputs.last_hidden_state,
                        teacher_hidden=teacher_hidden_for_kl,
                        student_projection=student_proj,
                        teacher_projection=teacher_proj,
                        attention_mask=attention_mask,
                        loss_mask=completion_mask,
                        temperature=temperature,
                        alpha=alpha,
                        beta=beta,
                        cakld_gamma=cakld_gamma,
                        loss_top_k=loss_top_k,
                        loss_add_tail=loss_add_tail,
                        shift_tokens=shift_tokens,
                        sequence_chunk_size=seq_chunk,
                    )
            elif use_chunked:
                total_loss, loss_components = chunked_distillation_loss(
                    student_hidden=student_outputs.last_hidden_state,
                    teacher_hidden=teacher_hidden_for_kl,
                    student_lm_head_fn=module.make_lm_head_fn(),
                    teacher_lm_head_fn=teacher_state.model.make_lm_head_fn(),
                    attention_mask=attention_mask,
                    loss_mask=completion_mask,
                    labels=None,
                    use_hard_labels=False,
                    temperature=temperature,
                    alpha=alpha,
                    chunk_size=int(logits_chunk_size),
                    shift_tokens=shift_tokens,
                )
            else:
                total_loss, loss_components = distillation_loss(
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_logits,
                    attention_mask=attention_mask,
                    loss_mask=completion_mask,
                    labels=None,
                    use_hard_labels=False,
                    temperature=temperature,
                    alpha=alpha,
                    beta=beta,
                    cakld_gamma=cakld_gamma,
                    loss_top_k=loss_top_k,
                    loss_add_tail=loss_add_tail,
                    shift_tokens=shift_tokens,
                    partition_manager=_vocab_parallel_partition_manager(module),
                )

        metrics = LossMetrics(
            loss=total_loss,
            other_metrics={key: jnp.asarray(value) for key, value in loss_components.items()},
        )
        return total_loss, metrics

    if is_training:
        with jax.named_scope(scope_root + "/grad_and_minibatch"):
            gradients, metrics = minibatch_call(
                state=student_state,
                batch=batch,
                minibatch_size=minibatch_size,
                grad_fn=jax.value_and_grad(loss_fn, has_aux=True),
            )
        with jax.named_scope(scope_root + "/update_state"):
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
        with jax.named_scope(scope_root + "/eval_call"):
            _, metrics = loss_fn(tree=student_state.graphstate, minibatch=batch)
        return metrics
