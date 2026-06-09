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
        non-negative warmup offsets. Negative multipliers would invert the
        smoothing schedule, so they are rejected before the callback is attached
        to a trainer.
        """
        if self.update_freq <= 0:
            raise ValueError("`update_freq` must be positive.")
        if self.ref_model_update_freq <= 0:
            raise ValueError("`ref_model_update_freq` must be positive.")
        if self.update_after < 0 or self.ref_model_update_after < 0:
            raise ValueError("BEMA update-after steps must be non-negative.")
        if self.multiplier < 0.0 or self.min_ema_multiplier < 0.0:
            raise ValueError("BEMA multipliers must be non-negative.")

    _theta0_graphstate: tp.Any = field(default=None, init=False, repr=False)
    _ema_graphstate: tp.Any = field(default=None, init=False, repr=False)
    _bema_graphstate: tp.Any = field(default=None, init=False, repr=False)

    def _ema_beta(self, step: int) -> float:
        """Return the per-step EMA interpolation coefficient.

        The coefficient decays as training progresses according to ``ema_power``
        and is lower-bounded by ``min_ema_multiplier``. It controls how much of
        the current policy graphstate is mixed into the running EMA state.
        """
        beta = (self.lag + self.multiplier * step) ** (-self.ema_power)
        return max(beta, self.min_ema_multiplier)

    def _bema_alpha(self, step: int) -> float:
        """Return the bias-correction coefficient for the BEMA graphstate.

        This value decays independently from the EMA coefficient and scales the
        correction between the current graphstate and the initial graphstate.
        """
        return (self.lag + self.multiplier * step) ** (-self.bias_power)

    @staticmethod
    def _copy_graphstate(graphstate: tp.Any) -> tp.Any:
        """Copy array leaves in a graphstate while preserving metadata leaves.

        BEMA stores graphstate snapshots across steps, so array leaves must be
        materialized into independent JAX arrays. Non-array leaves are carried
        through unchanged because they represent structure or metadata rather
        than mutable trainable values.
        """
        return jax.tree_util.tree_map(lambda leaf: jnp.array(leaf) if hasattr(leaf, "shape") else leaf, graphstate)

    def update_graphstate(self, graphstate: tp.Any, step: int) -> tp.Any | None:
        """Update and return the BEMA graphstate for a policy pytree.

        Before ``update_after`` the method returns ``None`` to signal that no
        BEMA state should be applied. On the first eligible step it initializes
        the initial, EMA, and BEMA snapshots; on later eligible update steps it
        refreshes EMA/BEMA leaves while preserving non-array metadata.
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
        current_graphstate = self._copy_graphstate(graphstate)

        def _update(current: object, theta0: object, ema: object) -> object:
            """Compute the bias-corrected BEMA leaf for one pytree position.

            Array leaves receive the BEMA correction. Non-array leaves are
            returned from the current graphstate so tree metadata remains valid.
            """
            if not hasattr(current, "shape"):
                return current
            return ema + alpha * (current - theta0)

        def _update_ema(current: object, ema: object) -> object:
            """Update one EMA leaf while leaving metadata leaves unchanged.

            The update is a standard exponential moving average over array
            leaves. Non-array leaves are not blended because they do not have
            arithmetic semantics.
            """
            if not hasattr(current, "shape"):
                return current
            return (1.0 - beta) * ema + beta * current

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
        """
        bema_graphstate = self.update_graphstate(policy_state.graphstate, step)
        if (
            not self.update_ref_model
            or bema_graphstate is None
            or step < self.ref_model_update_after
            or (step - self.ref_model_update_after) % self.ref_model_update_freq != 0
        ):
            return reference_state
        return reference_state.replace(graphstate=self._copy_graphstate(bema_graphstate))

    def on_step_end(self, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Compatibility no-op for external callback lists.

        EasyDeL trainers call :meth:`update_reference_state` directly with
        ``EasyDeLState`` objects instead of routing through Transformers'
        callback handler.
        """
        del args, kwargs


BEMAConfig = BEMACallback
