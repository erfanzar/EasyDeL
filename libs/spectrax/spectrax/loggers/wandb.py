# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Weights & Biases backend for the unified logger."""

from __future__ import annotations

import numpy as np

from .base import ArrayLike, BaseBackend, LogValue, Scalar

try:
    import wandb

    _WANDB_AVAILABLE = True
except Exception:
    wandb = None  # type: ignore[assignment]
    _WANDB_AVAILABLE = False


class WandBBackend(BaseBackend):
    """Backend that writes to Weights & Biases.

    Requires ``wandb`` to be installed and initialized (either externally
    or via the ``init_kwargs`` passed here).

    Args:
        project: W&B project name. If ``None``, assumes ``wandb.init()``
            has already been called elsewhere.
        init_kwargs: Extra keyword arguments forwarded to ``wandb.init()``
            when ``project`` is provided.
    """

    def __init__(
        self,
        project: str | None = None,
        *,
        init_kwargs: dict[str, object] | None = None,
    ):
        """Initialize the WandB backend.

        Args:
            project: WandB project name. If ``None``, assumes ``wandb.init()``
                has already been called elsewhere.
            init_kwargs: Extra keyword arguments forwarded to ``wandb.init()``
                when ``project`` is provided.
        """
        if not _WANDB_AVAILABLE:
            raise RuntimeError("WandBBackend requires wandb. Install it:  pip install wandb")
        if project is not None:
            wandb.init(project=project, **(init_kwargs or {}))
        elif wandb.run is None:
            raise RuntimeError(
                "WandBBackend: wandb is not initialized. Pass ``project=...`` or call ``wandb.init()`` first."
            )

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Log a scalar to W&B.

        Args:
            tag: Metric identifier.
            value: Scalar numeric value.
            step: Training step.
        """
        wandb.log({tag: float(value)}, step=step)

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Log a histogram to W&B.

        Args:
            tag: Metric identifier.
            values: Array of values to histogram.
            step: Training step.
        """
        wandb.log({tag: wandb.Histogram(np.asarray(values))}, step=step)

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Log an image to W&B.

        Args:
            tag: Image identifier.
            image: Image array.
            step: Training step.
        """
        wandb.log({tag: wandb.Image(np.asarray(image))}, step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log HTML text to W&B.

        Args:
            tag: Text identifier.
            text: String content.
            step: Training step.
        """
        wandb.log({tag: wandb.Html(text)}, step=step)

    def log_hparams(self, hparams: dict[str, LogValue]) -> None:
        """Update W&B run config with hyper-parameters.

        Args:
            hparams: Dictionary of hyper-parameters.
        """
        wandb.config.update(hparams)

    def log_summary(self, metrics: dict[str, LogValue]) -> None:
        """Update the W&B run summary with metrics.

        Args:
            metrics: Dictionary of summary metrics.
        """
        if wandb.run is not None:
            wandb.run.summary.update(metrics)

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[LogValue]],
        step: int,
    ) -> None:
        """Log a W&B table.

        Args:
            tag: Table identifier.
            columns: List of column header strings.
            rows: List of row values; each row is a list of cell values.
            step: Training step.
        """
        table = wandb.Table(columns=columns)
        for row in rows:
            table.add_data(*row)
        wandb.log({tag: table}, step=step)

    def flush(self) -> None:
        """No-op — W&B handles its own flushing."""
        pass

    def close(self) -> None:
        """Finish the W&B run."""
        wandb.finish()
