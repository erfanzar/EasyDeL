# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Console / stdout backend for the unified logger."""

from __future__ import annotations

import sys
from datetime import datetime

import numpy as np

from spectrax._internal.logging import COLORS

from .base import ArrayLike, BaseBackend, LogValue, Scalar

_KIND_COLORS: dict[str, str] = {
    "scalar": COLORS["GREEN"],
    "image": COLORS["PURPLE"],
    "text": COLORS["YELLOW"],
    "hparams": COLORS["BLUE"],
}


def _fmt_number(v: float) -> str:
    """Format a float compactly for console output.

    Uses scientific notation for very large or very small values;
    otherwise uses ``g`` formatting with up to 6 significant digits.

    Args:
        v: The number to format.

    Returns:
        A compact string representation.
    """
    if abs(v) >= 1e4 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.4e}"
    return f"{v:.6g}"


class ConsoleBackend(BaseBackend):
    """Backend that prints metrics to ``stdout`` with Spectrax-style colours.

    Histograms are silently ignored (they are too noisy for console output).
    Useful for quick debugging or when no other logging infrastructure is
    available.

    Args:
        prefix: Optional string printed before every line. Defaults to ``""``.
    """

    def __init__(self, prefix: str = ""):
        """Initialize the console backend.

        Args:
            prefix: Optional string printed before every line.
        """
        self._prefix = prefix

    def _print(self, kind: str, tag: str, value: str, step: int) -> None:
        """Emit a single coloured log line to ``stdout``.

        Args:
            kind: Metric kind (e.g. ``"scalar"``, ``"image"``).
            tag: Metric identifier.
            value: Pre-formatted value string.
            step: Training step.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        color = _KIND_COLORS.get(kind, COLORS["RESET"])
        reset = COLORS["RESET"]
        bold = COLORS["BOLD"]

        prefix = f"{color}{self._prefix}{reset} " if self._prefix else ""
        header = f"{prefix}{bold}({ts}){reset}"
        step_str = f"{COLORS['BLUE_PURPLE']}step={step:>6}{reset}"
        kind_str = f"{color}{kind:>10}{reset}"
        tag_str = f"{bold}{tag}{reset}"

        line = f"{header}  {step_str}  {kind_str}  {tag_str}: {value}"
        print(line, file=sys.stdout, flush=True)

    def log_scalar(self, tag: str, value: Scalar, step: int) -> None:
        """Print a scalar to stdout with colour.

        Args:
            tag: Metric identifier.
            value: Scalar numeric value.
            step: Training step.
        """
        self._print("scalar", tag, _fmt_number(float(value)), step)

    def log_histogram(self, tag: str, values: ArrayLike, step: int) -> None:
        """Histograms are intentionally skipped in console output.

        Args:
            tag: Tag value consumed by this operation.
            values: Values consumed by the helper.
            step: Step value consumed by this operation.
        """
        pass

    def log_image(self, tag: str, image: ArrayLike, step: int) -> None:
        """Print image shape and dtype to stdout.

        Args:
            tag: Image identifier.
            image: Image array.
            step: Training step.
        """
        arr = np.asarray(image)
        val_str = f"shape={arr.shape}  dtype={arr.dtype}"
        self._print("image", tag, val_str, step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Print text to stdout, clipped to 120 characters.

        Args:
            tag: Text identifier.
            text: String content.
            step: Training step.
        """
        clipped = text[:120] + ("…" if len(text) > 120 else "")
        self._print("text", tag, repr(clipped), step)

    def log_hparams(self, hparams: dict[str, LogValue]) -> None:
        """Print hyper-parameters as a key-value list.

        Args:
            hparams: Dictionary of hyper-parameters.
        """
        ts = datetime.now().strftime("%H:%M:%S")
        color = _KIND_COLORS["hparams"]
        reset = COLORS["RESET"]
        bold = COLORS["BOLD"]
        prefix = f"{color}{self._prefix}{reset} " if self._prefix else ""
        header = f"{prefix}{bold}({ts}){reset}"
        print(f"{header}  {color}   hparams{reset}", file=sys.stdout, flush=True)
        for k, v in hparams.items():
            print(f"  {bold}{k}{reset} = {v}", file=sys.stdout, flush=True)

    def log_table(self, tag: str, columns: list[str], rows: list[list[LogValue]], step: int) -> None:
        """Tables are intentionally skipped in console output.

        Args:
            tag: Tag value consumed by this operation.
            columns: Columns value consumed by this operation.
            rows: Rows value consumed by this operation.
            step: Step value consumed by this operation.
        """
        pass

    def flush(self) -> None:
        """No-op — stdout is already flushed per-line."""
        pass

    def close(self) -> None:
        """No-op."""
        pass
