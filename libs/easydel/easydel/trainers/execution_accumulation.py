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

"""Gradient accumulation as separate device executions (``gradient_accumulation_mode="execution"``).

The ``lax.scan`` of :func:`~easydel.trainers.training_utils.minibatch_call` compiles the microbatch loop as a
while-loop body whose temporaries XLA does not share with the optimizer temporaries of the same program, so the
two stack. Here the optimizer step is split into two programs driven from the host:

* a **grad program** ``(state, microbatch, *extra, accumulator, *statics) -> (accumulator, metrics, weight)``:
  one microbatch's fwd/bwd, ``w_k * grads_k`` added into the fp32 accumulator (donated); the state is read only;
* an **apply program** ``(state, None, *extra, (accumulator, metrics), *statics) -> (state, metrics)``: divide by
  the summed weight and run the unchanged optimizer path.

Both recombine microbatches with the same token weights as the scan (the step function's
``microbatch_weight_fn``), so the two modes agree up to floating-point rounding. A step function opts in by
accepting the keyword-only arguments ``accumulate_into`` and ``apply_accumulated`` (see
``distillation_trainer._fn.distillation_step``).
"""

from __future__ import annotations

import typing as tp

import jax
import spectrax as spx
from eformer.loggings import get_logger

from .batch_placement import place_batch, release_batch_slot, replicated_batch_slot_spec
from .training_utils import (
    _infer_batch_size,
    fold_microbatch_metrics,
    slice_host_minibatch,
    zero_gradient_accumulator,
)

logger = get_logger(__name__)


class ExecutionAccumulation:
    """Compiled grad / apply programs of execution-mode gradient accumulation and the per-step driver.

    Args:
        step_fn: Step function accepting ``accumulate_into`` / ``apply_accumulated`` keyword arguments.
        num_steps: Microbatches per optimizer step (``gradient_accumulation_steps``, >= 2).
        in_shardings: The step's ``(state, batch, *extra)`` input shardings.
        state_shardings: Sharding pytree of the state; ``.graphstate`` places the accumulator.
        empty_sharding: Replicated sharding for scalars and metrics.
        static_argnums: Static argument positions in the ``(state, batch, *extra, *statics)`` layout.
        jit_options: Extra ``spx.jit`` keyword arguments (e.g. ``compiler_options``).

    Raises:
        ValueError: When ``num_steps < 2`` or ``in_shardings`` does not cover ``(state, batch)``.
    """

    def __init__(
        self,
        *,
        step_fn: tp.Callable[..., tp.Any],
        num_steps: int,
        in_shardings: tuple[tp.Any, ...],
        state_shardings: tp.Any,
        empty_sharding: tp.Any,
        static_argnums: tp.Sequence[int],
        jit_options: dict[str, tp.Any] | None = None,
        batch_spec: tp.Any = None,
        mesh: tp.Any = None,
    ) -> None:
        if int(num_steps) < 2:
            raise ValueError(f"execution-mode gradient accumulation needs >= 2 microbatches, got {num_steps}.")
        n_extra = len(in_shardings) - 2
        if n_extra < 0:
            raise ValueError("`in_shardings` must cover at least (state, batch).")
        self.num_steps = int(num_steps)
        self.step_fn = step_fn
        self._batch_spec = batch_spec
        self._mesh = mesh
        in_shardings = release_batch_slot(tuple(in_shardings), batch_spec)
        acc_pos = 2 + n_extra
        shifted_statics = tuple(int(i) + 1 for i in static_argnums)
        jit_options = dict(jit_options or {})
        acc_shardings = {"grads": state_shardings.graphstate, "weight": empty_sharding, "count": empty_sharding}
        self._acc_shardings = acc_shardings

        def _grad_phase(state, batch, *rest):
            extra, accumulator, statics = rest[:n_extra], rest[n_extra], rest[n_extra + 1 :]
            return step_fn(state, batch, *extra, *statics, accumulate_into=accumulator)

        def _apply_phase(state, batch, *rest):
            del batch
            extra, packed, statics = rest[:n_extra], rest[n_extra], rest[n_extra + 1 :]
            return step_fn(state, None, *extra, *statics, apply_accumulated=packed)

        self.grad_fn = spx.jit(
            _grad_phase,
            in_shardings=(*in_shardings, acc_shardings),
            out_shardings=(acc_shardings, empty_sharding, empty_sharding),
            donate_argnums=(acc_pos,),
            static_argnums=shifted_statics,
            **jit_options,
        )
        self.apply_fn = spx.jit(
            _apply_phase,
            in_shardings=(in_shardings[0], None, *in_shardings[2:], (acc_shardings, empty_sharding)),
            out_shardings=(in_shardings[0], empty_sharding),
            donate_argnums=(0, acc_pos),
            static_argnums=shifted_statics,
            **jit_options,
        )
        self._zeros_fn = None
        self._init_metrics_fn = jax.jit(lambda m, w: fold_microbatch_metrics(None, m, w))
        self._fold_metrics_fn = jax.jit(fold_microbatch_metrics, donate_argnums=(0,))

    def zeros(self, state) -> dict[str, tp.Any]:
        """A fresh fp32 accumulator on the state's shardings (one compiled program, built on first use)."""
        if self._zeros_fn is None:
            shapes = jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(tuple(x.shape), x.dtype), state.graphstate)
            self._zeros_fn = jax.jit(lambda: zero_gradient_accumulator(shapes), out_shardings=self._acc_shardings)
        return self._zeros_fn()

    def microbatches(self, batch) -> list[tp.Any]:
        """Host-sliced microbatches of ``batch`` (views for numpy leaves).

        Raises:
            ValueError: When the batch does not divide into ``num_steps`` microbatches.
        """
        batch_size = _infer_batch_size(batch)
        if batch_size % self.num_steps:
            raise ValueError(f"batch of {batch_size} rows is not divisible by {self.num_steps} microbatches.")
        size = batch_size // self.num_steps
        parts = [slice_host_minibatch(batch, batch_size, k * size, size) for k in range(self.num_steps)]
        if self._batch_spec is None:
            return parts
        return [place_batch(part, self._batch_spec, self._mesh) for part in parts]

    def run(self, state, batch, extra_args: tuple[tp.Any, ...], static_args: tuple[tp.Any, ...]):
        """One optimizer step: ``num_steps`` grad executions, then the apply execution.

        Returns:
            The ``(state, metrics)`` outputs of the apply program (not yet blocked on).
        """
        accumulator = self.zeros(state)
        acc_metrics = None
        for minibatch in self.microbatches(batch):
            accumulator, step_metrics, weight = self.grad_fn(state, minibatch, *extra_args, accumulator, *static_args)
            if acc_metrics is None:
                acc_metrics = self._init_metrics_fn(step_metrics, weight)
            else:
                acc_metrics = self._fold_metrics_fn(acc_metrics, step_metrics, weight)
        return self.apply_fn(state, None, *extra_args, (accumulator, acc_metrics), *static_args)


def build_execution_accumulation(
    trainer,
    *,
    step_fn: tp.Callable[..., tp.Any],
    in_shardings: tuple[tp.Any, ...],
    static_argnums: tp.Sequence[int],
    empty_sharding: tp.Any,
    jit_options: dict[str, tp.Any] | None = None,
) -> ExecutionAccumulation | None:
    """Build the execution-mode driver when ``trainer.arguments`` asks for it, else ``None``.

    Raises:
        ValueError: With ``mpmd_scheduler`` or training buckets (neither is covered by the driver).
    """
    arguments = trainer.arguments
    mode = getattr(arguments, "gradient_accumulation_mode", "scan") or "scan"
    steps = int(getattr(arguments, "gradient_accumulation_steps", 1) or 1)
    if mode != "execution" or steps <= 1:
        return None
    if getattr(arguments, "mpmd_scheduler", None) is not None:
        raise ValueError("gradient_accumulation_mode='execution' is not supported together with mpmd_scheduler.")
    if getattr(trainer, "_has_buckets", False):
        raise ValueError("gradient_accumulation_mode='execution' is not supported together with training buckets.")
    logger.info("gradient accumulation by separate executions: %d microbatches per optimizer step", steps)
    mesh = getattr(getattr(trainer, "model", None), "mesh", None)
    return ExecutionAccumulation(
        step_fn=step_fn,
        num_steps=steps,
        in_shardings=tuple(in_shardings),
        state_shardings=trainer.state_shardings,
        empty_sharding=empty_sharding,
        static_argnums=static_argnums,
        jit_options=jit_options,
        batch_spec=replicated_batch_slot_spec(arguments, mesh, tuple(in_shardings)),
        mesh=mesh,
    )
