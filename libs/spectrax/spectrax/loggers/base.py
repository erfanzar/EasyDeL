# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Base backend protocol and unified logger implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import jax
import numpy as np

from spectrax._internal.logging import get_logger

_logger = get_logger(__name__)

Scalar = float | int
ArrayLike = np.ndarray | jax.Array
LogValue = Scalar | str | bool | None | ArrayLike | Mapping[str, "LogValue"] | Sequence["LogValue"]


class BaseBackend(ABC):
    """Abstract interface for a logging backend.

    Subclasses must implement the core logging methods. Optional methods
    (``log_summary``, ``log_table``) default to no-ops.
    """

    @abstractmethod
    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Metric identifier, e.g. ``"loss/train"``.
            value: Scalar numeric value.
            step: Training step number.
        """

    @abstractmethod
    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Log a histogram of values.

        Args:
            tag: Metric identifier.
            values: Array of values to histogram.
            step: Training step number.
        """

    @abstractmethod
    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Log an image.

        Args:
            tag: Image identifier.
            image: Image array (format depends on backend).
            step: Training step number.
        """

    @abstractmethod
    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a text string.

        Args:
            tag: Text identifier.
            text: String content.
            step: Training step number.
        """

    @abstractmethod
    def log_hparams(self, hparams: dict[str, LogValue]) -> None:
        """Log hyper-parameters.

        Args:
            hparams: Flat or nested dict of hyper-parameters.
        """

    def log_summary(self, metrics: dict[str, LogValue]) -> None:
        """Log summary-level metrics (optional; default no-op).

        Args:
            metrics: Dictionary of summary metrics.
        """
        return None

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[LogValue]],
        step: int,
    ) -> None:
        """Log a table (optional; default no-op).

        Args:
            tag: Table identifier.
            columns: List of column header strings.
            rows: List of row values; each row is a list of cell values.
            step: Training step number.
        """
        return None

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered writes."""

    @abstractmethod
    def close(self) -> None:
        """Close the backend and release resources."""


class _NullBackend(BaseBackend):
    """No-op backend used when no backends are configured."""

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """No-op.

        Args:
            tag: Tag value consumed by this operation.
            value: Value consumed by the helper.
            step: Step value consumed by this operation.
        """
        pass

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """No-op.

        Args:
            tag: Tag value consumed by this operation.
            values: Values consumed by the helper.
            step: Step value consumed by this operation.
        """
        pass

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """No-op.

        Args:
            tag: Tag value consumed by this operation.
            image: Image value consumed by this operation.
            step: Step value consumed by this operation.
        """
        pass

    def log_text(self, tag: str, text: str, step: int) -> None:
        """No-op.

        Args:
            tag: Tag value consumed by this operation.
            text: Text value consumed by this operation.
            step: Step value consumed by this operation.
        """
        pass

    def log_hparams(self, hparams: dict[str, LogValue]) -> None:
        """No-op.

        Args:
            hparams: Hparams value consumed by this operation.
        """
        pass

    def log_summary(self, metrics: dict[str, LogValue]) -> None:
        """No-op.

        Args:
            metrics: Metrics value consumed by this operation.
        """
        pass

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[LogValue]],
        step: int,
    ) -> None:
        """No-op.

        Args:
            tag: Tag value consumed by this operation.
            columns: Columns value consumed by this operation.
            rows: Rows value consumed by this operation.
            step: Step value consumed by this operation.
        """
        pass

    def flush(self) -> None:
        """No-op."""
        pass

    def close(self) -> None:
        """No-op."""
        pass


class Logger:
    """Unified logger that multiplexes writes to multiple backends.

    In distributed JAX settings only process ``0`` performs actual I/O;
    calls on other ranks are silently dropped.

    This class also exposes EasyDeL-compatible aliases so it can be used
    as a drop-in replacement for ``tensorboardX.SummaryWriter`` and
    ``flax.metrics.tensorboard.SummaryWriter``:

    * ``add_scalar(tag, value, step)`` → ``log_scalar(... )``
    * ``add_histogram(tag, values, step)`` → ``log_histogram(... )``
    * ``scalar(tag, value, step)`` → ``log_scalar(... )``  (Flax style)
    * ``histogram(tag, values, step)`` → ``log_histogram(... )``  (Flax style)

    Args:
        backends: List of :class:`BaseBackend` instances. If empty, a no-op
            null backend is used.
        auto_flush: If ``True`` (default), :meth:`flush` is called after
            every logging operation.

    Example::

        from spectrax.loggers import Logger, TensorBoardBackend, ConsoleBackend

        logger = Logger([
            TensorBoardBackend("./runs"),
            ConsoleBackend(),
        ])
        logger.log_scalar("loss", 0.5, step=100)
        logger.close()
    """

    def __init__(
        self,
        backends: list[BaseBackend] | None = None,
        *,
        auto_flush: bool = True,
    ):
        """Initialize the logger.

        Args:
            backends: List of logging backends to dispatch to. Defaults to
                a single :class:`_NullBackend` if ``None``.
            auto_flush: Whether to flush backends after every log call.
        """
        self._backends = backends or [_NullBackend()]
        self._auto_flush = auto_flush
        self._closed = False
        self._is_main = self._is_main_process()

    @staticmethod
    def _is_main_process() -> bool:
        """Return ``True`` if this is process 0 in a JAX distributed run.

        Falls back to ``True`` when JAX backends are not yet initialized.

        Returns:
            ``True`` for the primary process, ``False`` for all others.
        """
        try:
            return jax.process_index() == 0
        except RuntimeError:
            return True

    def _dispatch(self, method: str, *args: object, **kwargs: object) -> None:
        """Call *method* on every backend, swallowing individual failures.

        If :attr:`_auto_flush` is enabled, :meth:`flush` is called on all
        backends after the dispatch.

        Args:
            method: Backend method name as a string.
            *args: Positional args forwarded to the method.
            **kwargs: Keyword args forwarded to the method.
        """
        if self._closed or not self._is_main:
            return
        for backend in self._backends:
            try:
                getattr(backend, method)(*args, **kwargs)
            except Exception as e:
                _logger.warning_once(f"Logger backend {type(backend).__name__}.{method} failed: {e}")
        if self._auto_flush:
            self.flush()

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Log a scalar value to all backends.

        Args:
            tag: Metric identifier, e.g. ``"loss/train"``.
            value: Scalar number.
            step: Training step.
        """
        self._dispatch("log_scalar", tag, value, step)

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Log a histogram to all backends.

        Args:
            tag: Metric identifier.
            values: Array of values to histogram.
            step: Training step.
        """
        self._dispatch("log_histogram", tag, values, step)

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Log an image to all backends.

        Args:
            tag: Image identifier.
            image: Image array (format depends on backend).
            step: Training step.
        """
        self._dispatch("log_image", tag, image, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text to all backends.

        Args:
            tag: Text identifier.
            text: String content.
            step: Training step.
        """
        self._dispatch("log_text", tag, text, step)

    def log_hparams(self, hparams: dict[str, LogValue]) -> None:
        """Log hyper-parameters to all backends.

        Args:
            hparams: Flat or nested dict of hyper-parameters.
        """
        self._dispatch("log_hparams", hparams)

    def log_summary(self, metrics: dict[str, LogValue]) -> None:
        """Log summary-level metrics (e.g. WandB ``run.summary``).

        Not all backends support this; those that don't silently ignore it.

        Args:
            metrics: Dictionary of summary metrics.
        """
        self._dispatch("log_summary", metrics)

    def log_table(
        self,
        tag: str,
        columns: list[str],
        rows: list[list[LogValue]],
        step: int,
    ) -> None:
        """Log a table to all backends.

        Backends that do not support tables (e.g. TensorBoard) silently
        ignore this call.

        Args:
            tag: Table identifier.
            columns: List of column header strings.
            rows: List of row values; each row is a list of cell values.
            step: Training step.
        """
        self._dispatch("log_table", tag, columns, rows, step)

    def add_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Alias for :meth:`log_scalar`.

        Args:
            tag: Tag value consumed by this operation.
            value: Value consumed by the helper.
            step: Step value consumed by this operation.
        """
        self.log_scalar(tag, value, step)

    def add_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Alias for :meth:`log_histogram`.

        Args:
            tag: Tag value consumed by this operation.
            values: Values consumed by the helper.
            step: Step value consumed by this operation.
        """
        self.log_histogram(tag, values, step)

    def add_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Alias for :meth:`log_image`.

        Args:
            tag: Tag value consumed by this operation.
            image: Image value consumed by this operation.
            step: Step value consumed by this operation.
        """
        self.log_image(tag, image, step)

    def add_text(self, tag: str, text: str, step: int) -> None:
        """Alias for :meth:`log_text`.

        Args:
            tag: Tag value consumed by this operation.
            text: Text value consumed by this operation.
            step: Step value consumed by this operation.
        """
        self.log_text(tag, text, step)

    def add_hparams(self, hparams: dict[str, LogValue]) -> None:
        """Alias for :meth:`log_hparams`.

        Args:
            hparams: Hparams value consumed by this operation.
        """
        self.log_hparams(hparams)

    def scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Flax-style alias for :meth:`log_scalar`.

        Args:
            tag: Tag value consumed by this operation.
            value: Value consumed by the helper.
            step: Step value consumed by this operation.
        """
        self.log_scalar(tag, value, step)

    def histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Flax-style alias for :meth:`log_histogram`.

        Args:
            tag: Tag value consumed by this operation.
            values: Values consumed by the helper.
            step: Step value consumed by this operation.
        """
        self.log_histogram(tag, values, step)

    def image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Flax-style alias for :meth:`log_image`.

        Args:
            tag: Tag value consumed by this operation.
            image: Image value consumed by this operation.
            step: Step value consumed by this operation.
        """
        self.log_image(tag, image, step)

    def text(self, tag: str, textdata: str, step: int) -> None:
        """Flax-style alias for :meth:`log_text`.

        Args:
            tag: Tag value consumed by this operation.
            textdata: Textdata value consumed by this operation.
            step: Step value consumed by this operation.
        """
        self.log_text(tag, textdata, step)

    def hparams(self, hparams: dict[str, LogValue]) -> None:
        """Flax-style alias for :meth:`log_hparams`.

        Args:
            hparams: Hparams value consumed by this operation.
        """
        self.log_hparams(hparams)

    def flush(self) -> None:
        """Flush all backends."""
        if not self._is_main:
            return
        for backend in self._backends:
            try:
                backend.flush()
            except Exception as e:
                _logger.warning_once(f"Logger backend {type(backend).__name__}.flush failed: {e}")

    def close(self) -> None:
        """Close all backends and release resources."""
        if self._closed:
            return
        self._closed = True
        if not self._is_main:
            return
        for backend in self._backends:
            try:
                backend.close()
            except Exception as e:
                _logger.warning_once(f"Logger backend {type(backend).__name__}.close failed: {e}")

    def __enter__(self) -> Logger:
        """Context-manager entry — returns ``self``.

        Returns:
            Result described by this helper.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """Context-manager exit — closes all backends.

        Args:
            *exc: Additional positional arguments forwarded to the wrapped callable or backend.
        """
        self.close()
