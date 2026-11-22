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

import json
import typing as tp
from dataclasses import fields

import jax
import jax.numpy as jnp
import spectrax as spx
from eformer.paths import ePath
from spectrax.serialization import AsyncCheckpointManager

from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossMetrics
from easydel.utils import Registry

from ..direct_preference_optimization_trainer import DPOConfig, DPOTrainer
from .bema_config import BEMACallback

Registry.register("trainer-arguments", "bema_dpo")(DPOConfig)


@Registry.register("trainer", "bema_dpo")
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
        self._last_bema_ref_update_step: int | None = None
        self._bema_checkpoint_manager: AsyncCheckpointManager | None = None
        super().__init__(*args, **kwargs)

    def _get_bema_checkpoint_manager(self) -> AsyncCheckpointManager:
        """Return a cached ``AsyncCheckpointManager`` without spawning cleanup daemon threads."""
        manager = getattr(self, "_bema_checkpoint_manager", None)
        if manager is None:
            manager = AsyncCheckpointManager()
            self._bema_checkpoint_manager = manager
        return manager

    def _preprocess_batch_input(
        self,
        state: EasyDeLState,
        batch: dict[str, tp.Any],
        is_train: bool,
    ) -> tuple[dict[str, tp.Any], dict[str, float | int | str]]:
        """Drop stale precomputed reference log-probs when BEMA updates ``reference_state``."""
        batch, infos = super()._preprocess_batch_input(state, batch, is_train)
        if getattr(self.bema_callback, "update_ref_model", False) and self.reference_state is not None:
            batch = dict(batch)
            for name in (
                "ref_chosen_logps",
                "ref_rejected_logps",
                "reference_chosen_log_probs",
                "reference_rejected_log_probs",
            ):
                batch.pop(name, None)
        return batch, infos

    def _save_state(self, state: EasyDeLState, save_directory: str | None = None, *args, **kwargs) -> str:
        """Save the policy, reference and all BEMA histories as one resumable checkpoint."""
        if save_directory is None:
            save_directory = self.arguments._get_save_directory_milestone(
                step=self._get_current_step(state), create=False
            )
        directory = str(save_directory)
        auxiliary_path = ePath(directory) / "bema"
        manifest_path = auxiliary_path / "manifest.json"
        callback = self.bema_callback
        histories = {name: getattr(callback, f"_{name}_graphstate") for name in ("theta0", "ema", "bema")}
        manifest = {
            "format": "easydel-bema-v1",
            "complete": False,
            "histories": [name for name, tree in histories.items() if tree is not None],
            "has_reference": self.reference_state is not None,
            "callback": {field.name: getattr(callback, field.name) for field in fields(callback) if field.init},
        }
        write_manifest = jax.process_index() == 0
        if write_manifest:
            auxiliary_path.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest))
        else:
            auxiliary_path.mkdir(parents=True, exist_ok=True)
        # Mark the compound save incomplete before the policy becomes discoverable.
        directory = super()._save_state(state, save_directory, *args, **kwargs)
        manager = self._get_bema_checkpoint_manager()
        for name, graphstate in histories.items():
            if graphstate is not None:
                manager.save_pytree(graphstate.raw(), path=str(auxiliary_path), prefix=name, write_index=True)
        if self.reference_state is not None:
            manager.save_pytree(
                self.reference_state.graphstate.raw(),
                path=str(auxiliary_path),
                prefix="reference",
                write_index=True,
            )
            manager.save_pytree(
                self.reference_state.graphother.raw(),
                path=str(auxiliary_path),
                prefix="reference_other",
                write_index=True,
            )
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            multihost_utils.sync_global_devices("bema.checkpoint.complete")
        if write_manifest:
            manifest["complete"] = True
            manifest_path.write_text(json.dumps(manifest))
        if jax.process_count() > 1:
            from jax.experimental import multihost_utils

            multihost_utils.sync_global_devices("bema.checkpoint.committed")
        return directory

    def _load_auxiliary_states(self, checkpoint_path: str) -> None:
        """Restore BEMA before BaseTrainer can remove the loaded policy checkpoint."""
        super()._load_auxiliary_states(checkpoint_path)
        auxiliary_path = ePath(checkpoint_path) / "bema"
        manifest_path = auxiliary_path / "manifest.json"
        if not manifest_path.exists():
            # Backward compatibility for checkpoints predating BEMA persistence.
            return
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("format") != "easydel-bema-v1" or manifest.get("complete") is not True:
            raise ValueError(f"BEMA checkpoint {checkpoint_path} is incomplete or has an invalid manifest.")
        if not manifest.get("has_reference"):
            if manifest.get("histories"):
                raise ValueError("BEMA histories cannot be restored without a reference-state template.")
            return
        if self.reference_state is None:
            raise ValueError("BEMA checkpoint requires a configured reference model.")
        template_state = self.reference_state
        model = template_state.model
        manager = self._get_bema_checkpoint_manager()

        def load_graphstate(prefix, template):
            raw_template = jax.eval_shape(template.raw)
            raw, _ = manager.load_pytree(
                mesh=model.mesh,
                path=str(auxiliary_path),
                prefix=prefix,
                template=raw_template,
                sharding_rules=model.resolve_shardings_regex(),
            )
            restored = spx.State(raw)
            if set(restored.paths()) != set(template.paths()):
                raise ValueError(f"BEMA {prefix} checkpoint paths do not match the reference template.")
            return restored

        histories = manifest.get("histories", [])
        if histories not in ([], ["theta0", "ema", "bema"]):
            raise ValueError("BEMA checkpoint must contain all three smoothing histories or none.")
        callback = BEMACallback(**manifest["callback"])
        for name in histories:
            setattr(callback, f"_{name}_graphstate", load_graphstate(name, template_state.graphstate))
        reference = template_state.replace(
            graphstate=load_graphstate("reference", template_state.graphstate),
            graphother=load_graphstate("reference_other", template_state.graphother),
        )
        self.bema_callback = callback
        self.reference_state = reference

    def on_step_end(
        self,
        state: EasyDeLState,
        metrics: LossMetrics,
        step: int,
    ) -> tuple[EasyDeLState, LossMetrics]:
        """Run the normal DPO post-step hook and refresh the reference state.

        ``Trainer._train_epoch`` invokes ``on_step_end`` with the post-update
        1-based ``current_step`` (``1`` on the first training step), while
        :class:`BEMACallback` schedules updates on a 0-based index so
        ``update_after=0`` / ``ref_model_update_after=0`` fire on the first
        step. When BEMA swaps ``self.reference_state``, the update flag is both
        attached to ``metrics.other_metrics`` and recorded so
        :meth:`log_metrics` logs ``train/bema/ref_model_updated`` even when
        ``StepMetrics.calculate`` ran before ``on_step_end``.
        """
        state, metrics = super().on_step_end(state, metrics, step)
        if self.reference_state is not None:
            previous_reference = self.reference_state
            bema_step = max(int(step) - 1, 0)
            self.reference_state = self.bema_callback.update_reference_state(state, self.reference_state, bema_step)
            if self.reference_state is not previous_reference:
                other_metrics = dict(metrics.other_metrics or {})
                other_metrics["bema/ref_model_updated"] = jnp.asarray(1)
                metrics = metrics.replace(other_metrics=other_metrics)
                self._last_bema_ref_update_step = step
        return state, metrics

    def log_metrics(
        self,
        metrics: tp.Any,
        pbar: tp.Any,
        step: int,
        mode: str = "train",
        *args: tp.Any,
        **kwargs: tp.Any,
    ):
        """Ensure ``bema/ref_model_updated`` emitted in ``on_step_end`` is logged."""
        pending_step = getattr(self, "_last_bema_ref_update_step", None)
        if mode == "train" and pending_step == step and isinstance(metrics, dict):
            if "train/bema/ref_model_updated" not in metrics and "bema/ref_model_updated" not in metrics:
                metrics = dict(metrics)
                metrics["train/bema/ref_model_updated"] = 1
            self._last_bema_ref_update_step = None
        return super().log_metrics(metrics, pbar, step, mode, *args, **kwargs)
