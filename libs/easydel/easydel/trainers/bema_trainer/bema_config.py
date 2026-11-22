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
"""BEMA reference-model update helpers for DPO."""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass, field

import jax
from jax import numpy as jnp

from easydel.infra.base_state import EasyDeLState


@dataclass
class BEMACallback:
    """Maintain BEMA-smoothed graphstates for DPO reference updates.

    The callback stores the initial policy graphstate, an EMA graphstate, and a
    bias-corrected BEMA graphstate. Trainers can call
    :meth:`update_reference_state` after an optimization step to periodically
    replace the reference model with smoothed policy parameters.
    """

    update_freq: int = 400
    ema_power: float = 0.5
    bias_power: float = 0.2
    lag: int = 10
    update_after: int = 0
    multiplier: float = 1.0
    min_ema_multiplier: float = 0.0
    update_ref_model: bool = False
    ref_model_update_freq: int = 400
    ref_model_update_after: int = 0

    def __post_init__(self) -> None:
        """Validate BEMA update cadence and smoothing hyperparameters.

        BEMA updates are scheduled by positive step intervals and optional
        non-negative warmup offsets. Invalid or out-of-range smoothing
        hyperparameters (negative lag/powers, non-finite values, or
        multipliers outside valid bounds) are rejected before the callback is
        attached to a trainer.
        """
        if isinstance(self.update_freq, bool) or self.update_freq <= 0:
            raise ValueError("`update_freq` must be positive.")
        if isinstance(self.ref_model_update_freq, bool) or self.ref_model_update_freq <= 0:
            raise ValueError("`ref_model_update_freq` must be positive.")
        if (
            isinstance(self.update_after, bool)
            or self.update_after < 0
            or isinstance(self.ref_model_update_after, bool)
            or self.ref_model_update_after < 0
        ):
            raise ValueError("BEMA update-after steps must be non-negative.")
        if isinstance(self.lag, bool) or self.lag < 0:
            raise ValueError("`lag` must be non-negative.")
        if isinstance(self.multiplier, bool) or not math.isfinite(float(self.multiplier)) or self.multiplier < 0.0:
            raise ValueError("`multiplier` must be finite and non-negative.")
        if (
            isinstance(self.min_ema_multiplier, bool)
            or not math.isfinite(float(self.min_ema_multiplier))
            or not (0.0 <= float(self.min_ema_multiplier) <= 1.0)
        ):
            raise ValueError("`min_ema_multiplier` must be finite and in [0.0, 1.0].")
        if isinstance(self.ema_power, bool) or not math.isfinite(float(self.ema_power)) or self.ema_power < 0.0:
            raise ValueError("`ema_power` must be finite and non-negative.")
        if isinstance(self.bias_power, bool) or not math.isfinite(float(self.bias_power)) or self.bias_power < 0.0:
            raise ValueError("`bias_power` must be finite and non-negative.")
        if self.lag == 0 and self.multiplier == 0.0:
            raise ValueError("`lag` and `multiplier` cannot both be zero.")

    _theta0_graphstate: tp.Any = field(default=None, init=False, repr=False)
    _ema_graphstate: tp.Any = field(default=None, init=False, repr=False)
    _bema_graphstate: tp.Any = field(default=None, init=False, repr=False)

    def _ema_beta(self, step: int) -> float:
        """Return the per-step EMA interpolation coefficient clamped to ``[min_ema_multiplier, 1.0]``.

        The coefficient decays as training progresses according to ``ema_power``
        and is bounded within ``[min_ema_multiplier, 1.0]`` so ``(1 - beta)``
        never becomes negative and ``lag == 0`` at ``step == 0`` never divides
        by zero.
        """
        base = float(self.lag) + float(self.multiplier) * max(int(step), 0)
        if base <= 0.0:
            beta = 1.0
        else:
            beta = float(base ** (-float(self.ema_power)))
        return float(min(max(beta, float(self.min_ema_multiplier)), 1.0))

    def _bema_alpha(self, step: int) -> float:
        """Return the bias-correction coefficient clamped to ``[0.0, 1.0]``.

        This value decays independently from the EMA coefficient and scales the
        correction between the current graphstate and the initial graphstate.
        """
        base = float(self.lag) + float(self.multiplier) * max(int(step), 0)
        if base <= 0.0:
            alpha = 1.0
        else:
            alpha = float(base ** (-float(self.bias_power)))
        return float(min(max(alpha, 0.0), 1.0))

    @staticmethod
    def _is_floating_array(leaf: object) -> bool:
        """Return True when ``leaf`` is a concrete floating-point array."""
        return (
            hasattr(leaf, "shape")
            and hasattr(leaf, "dtype")
            and not isinstance(leaf, jax.ShapeDtypeStruct)
            and bool(jnp.issubdtype(leaf.dtype, jnp.floating))
        )

    @staticmethod
    def _copy_graphstate(graphstate: tp.Any) -> tp.Any:
        """Copy array leaves in a graphstate while preserving metadata leaves.

        BEMA stores graphstate snapshots across steps, so array leaves must be
        materialized into independent JAX arrays. Non-array leaves are carried
        through unchanged because they represent structure or metadata rather
        than mutable trainable values.
        """
        return jax.tree_util.tree_map(
            lambda leaf: jnp.array(leaf) if hasattr(leaf, "shape") and not isinstance(leaf, jax.ShapeDtypeStruct) else leaf,
            graphstate,
        )

    def update_graphstate(self, graphstate: tp.Any, step: int) -> tp.Any | None:
        """Update and return the BEMA graphstate for a policy pytree.

        Before ``update_after`` the method returns ``None`` to signal that no
        BEMA state should be applied. On the first eligible step it initializes
        the initial, EMA, and BEMA snapshots; on later eligible update steps it
        refreshes EMA/BEMA floating-point leaves while preserving non-floating
        array leaves and non-array metadata untouched.
        """
        if step < self.update_after:
            return None
        if self._theta0_graphstate is None or step == self.update_after:
            copied = self._copy_graphstate(graphstate)
            self._theta0_graphstate = copied
            self._ema_graphstate = self._copy_graphstate(graphstate)
            self._bema_graphstate = copied
            return self._bema_graphstate
        if (step - self.update_after) % self.update_freq != 0:
            return self._bema_graphstate

        beta = self._ema_beta(step)
        alpha = self._bema_alpha(step)
        # No copy of `graphstate`: it is only read as an operand below and both
        # tree maps allocate their outputs, so copying it here would add a
        # transient full parameter set on top of the three BEMA snapshots.
        current_graphstate = graphstate

        def _update(current: object, theta0: object, ema: object) -> object:
            """Compute the bias-corrected BEMA leaf for one pytree position.

            Floating-point array leaves receive the BEMA correction in at least
            float32 and are cast back to ``current.dtype``. Non-floating array
            leaves and metadata leaves are returned unchanged.
            """
            if not self._is_floating_array(current):
                return current
            compute_dtype = jnp.promote_types(current.dtype, jnp.float32)
            updated = ema.astype(compute_dtype) + jnp.asarray(alpha, dtype=compute_dtype) * (
                current.astype(compute_dtype) - theta0.astype(compute_dtype)
            )
            return updated.astype(current.dtype)

        def _update_ema(current: object, ema: object) -> object:
            """Update one EMA leaf while leaving non-floating and metadata leaves unchanged.

            The update is a standard exponential moving average over floating
            array leaves computed in at least float32 and cast back to
            ``current.dtype``.
            """
            if not self._is_floating_array(current):
                return current
            compute_dtype = jnp.promote_types(current.dtype, jnp.float32)
            beta_arr = jnp.asarray(beta, dtype=compute_dtype)
            updated = (jnp.asarray(1.0, dtype=compute_dtype) - beta_arr) * ema.astype(
                compute_dtype
            ) + beta_arr * current.astype(compute_dtype)
            return updated.astype(current.dtype)

        self._ema_graphstate = jax.tree_util.tree_map(_update_ema, current_graphstate, self._ema_graphstate)
        self._bema_graphstate = jax.tree_util.tree_map(
            _update,
            current_graphstate,
            self._theta0_graphstate,
            self._ema_graphstate,
        )
        return self._bema_graphstate

    def update_reference_state(
        self, policy_state: EasyDeLState, reference_state: EasyDeLState, step: int
    ) -> EasyDeLState:
        """Return ``reference_state`` with BEMA-smoothed policy weights when due.

        The reference state is left untouched unless BEMA is enabled, the step
        is past ``ref_model_update_after``, and the reference update frequency
        divides the current step offset. When an update is due, only the
        graphstate is replaced; optimizer state and other reference metadata are
        preserved.

        The reference shares the BEMA snapshot's arrays rather than copying them.
        BEMA never mutates a snapshot in place -- :meth:`update_graphstate`
        rebinds ``_bema_graphstate`` to freshly allocated arrays -- and the
        reference is a non-donated input to the compiled step, so a copy would
        just keep a fourth parameter set resident alongside ``theta0``, ``ema``,
        and ``bema``.
        """
        bema_graphstate = self.update_graphstate(policy_state.graphstate, step)
        if (
            not self.update_ref_model
            or bema_graphstate is None
            or step < self.ref_model_update_after
            or (step - self.ref_model_update_after) % self.ref_model_update_freq != 0
        ):
            return reference_state
        return reference_state.replace(graphstate=bema_graphstate)

    def on_step_end(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Compatibility no-op for external callback lists.

        EasyDeL trainers call :meth:`update_reference_state` directly with
        ``EasyDeLState`` objects instead of routing through Transformers'
        callback handler.
        """
        del args, kwargs


BEMAConfig = BEMACallback
