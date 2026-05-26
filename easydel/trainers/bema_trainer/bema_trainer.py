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

import jax.numpy as jnp

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics

from ..direct_preference_optimization_trainer import DPOTrainer
from .bema_config import BEMACallback


class BEMADPOTrainer(DPOTrainer):
    """DPO trainer with native JAX BEMA reference-state updates.

    The trainer delegates all DPO data loading, loss computation, and optimizer
    behavior to :class:`DPOTrainer`. Its only extra responsibility is to call a
    :class:`BEMACallback` after each step and swap the reference state when the
    BEMA schedule says an update is due.
    """

    def __init__(
        self,
        *args: tp.Any,
        bema_callback: BEMACallback | None = None,
        **kwargs: tp.Any,
    ) -> None:
        """Create a DPO trainer with an attached BEMA reference updater.

        Args:
            *args: Positional arguments forwarded unchanged to
                :class:`DPOTrainer`.
            bema_callback: Optional callback controlling EMA/BEMA smoothing and
                reference-model replacement cadence. When omitted, a default
                callback is created with reference updates enabled.
            **kwargs: Keyword arguments forwarded unchanged to
                :class:`DPOTrainer`.

        The constructor only wires the callback and delegates all model,
        dataset, tokenizer, and optimizer setup to the parent DPO trainer.
        """
        self.bema_callback = bema_callback or BEMACallback(update_ref_model=True)
        super().__init__(*args, **kwargs)

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: LossMetrics,
        step: int,
    ) -> tuple[EasyDeLState, LossMetrics]:
        """Run the normal DPO post-step hook and refresh the reference state.

        The policy state returned by the parent trainer is kept unchanged. When
        BEMA decides a reference update is due, the trainer swaps
        ``self.reference_state`` to the smoothed graphstate and emits a metric
        flag for that step.
        """
        state, metrics = super().on_step_end(state, metrics, step)
        if self.reference_state is not None:
            previous_reference = self.reference_state
            self.reference_state = self.bema_callback.update_reference_state(state, self.reference_state, step)
            if self.reference_state is not previous_reference:
                other_metrics = dict(metrics.other_metrics or {})
                other_metrics["bema/ref_model_updated"] = jnp.asarray(1)
                metrics = metrics.replace(other_metrics=other_metrics)
        return state, metrics
