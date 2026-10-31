# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Activation layers as :class:`~spectrax.Module` subclasses.

Each class here is a parameter-free wrapper around the corresponding
function in :mod:`spectrax.functional.activation`. The module form is
useful when an activation needs to participate in container traversal
(e.g. as a member of a :class:`~spectrax.nn.Sequential`, where
arbitrary callables would not be discovered by
:meth:`~spectrax.Module.tree_flatten`); inside a hand-written
:meth:`~spectrax.Module.forward`, prefer calling the functional form
directly to avoid the trivial wrapper overhead.

Forward signatures all accept ``**_`` so callers can thread auxiliary
keyword arguments (``rngs=``, ``training=``, …) through a generic
container without the activation rejecting them.
"""

from __future__ import annotations

from ..core._typing import Array, ArrayLike
from ..core.module import Module
from ..functional import activation as F


class ReLU(Module):
    """Rectified linear unit: :math:`\\mathrm{ReLU}(x) = \\max(0, x)`.

    Parameter-free pass-through layer. Negative inputs are zeroed and
    positive inputs are forwarded unchanged. Idempotent and dtype
    preserving.
    """

    def __init__(self) -> None:
        """Initialize the layer (no parameters or state).

        Calls :meth:`~spectrax.Module.__init__` so the instance is
        registered as a Spectrax module (and thus participates in
        container traversal).
        """
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.relu` element-wise.

        Args:
            x: Input array of any shape and floating-point dtype.
            **_: Ignored. Accepted so the layer can be placed in a
                container that threads auxiliary kwargs through every
                child.

        Returns:
            ``max(x, 0)`` with the same shape and dtype as ``x``.
        """
        return F.relu(x)


class GELU(Module):
    """Gaussian error linear unit (Hendrycks & Gimpel, 2016).

    Computes :math:`x \\cdot \\Phi(x)` where :math:`\\Phi` is the
    standard normal CDF. The exact form uses :func:`jax.scipy.stats`'s
    error function; the tanh approximation
    (``approximate=True``) trades a small amount of accuracy for a
    cheaper evaluation that matches the original BERT / GPT
    formulation. Both forms agree to within a few ulps for ``|x| < 5``.
    """

    def __init__(self, approximate: bool = False) -> None:
        """Record the ``approximate`` flag as a static module field.

        Args:
            approximate: When ``True``, use the
                :math:`\\tanh`-based approximation popularised by GPT-2
                (faster, slightly less accurate). When ``False``
                (default), use the exact erf-based formulation.
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.gelu` honouring :attr:`approximate`.

        Args:
            x: Input array of any shape and floating-point dtype.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``GELU(x)`` with the same shape and dtype as ``x``.
        """
        return F.gelu(x, approximate=self.approximate)


class SiLU(Module):
    """Sigmoid-weighted linear unit (also known as Swish).

    Computes :math:`x \\cdot \\sigma(x)` where :math:`\\sigma` is the
    logistic sigmoid. Smooth, non-monotonic, and self-gated; widely
    used as the feed-forward activation in modern transformer
    variants.
    """

    def __init__(self) -> None:
        """Initialize the layer (no parameters)."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.silu` element-wise.

        Args:
            x: Input array of any shape and floating-point dtype.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``x * sigmoid(x)`` with the same shape and dtype as ``x``.
        """
        return F.silu(x)


class Tanh(Module):
    """Hyperbolic tangent activation.

    Outputs are bounded in :math:`(-1, 1)` and odd-symmetric; useful as
    a saturating non-linearity in recurrent cells and certain attention
    score functions.
    """

    def __init__(self) -> None:
        """Initialize the layer (no parameters)."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.tanh` element-wise.

        Args:
            x: Input array of any shape and floating-point dtype.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``tanh(x)`` with the same shape and dtype as ``x``.
        """
        return F.tanh(x)


class Sigmoid(Module):
    """Logistic sigmoid activation.

    Computes :math:`\\sigma(x) = 1 / (1 + e^{-x})`. Outputs are bounded
    in :math:`(0, 1)`; commonly used to produce gate values or
    independent-Bernoulli logits.
    """

    def __init__(self) -> None:
        """Initialize the layer (no parameters)."""
        super().__init__()

    def forward(self, x: ArrayLike, **_: object) -> Array:
        """Apply :func:`~spectrax.functional.sigmoid` element-wise.

        Args:
            x: Input array of any shape and floating-point dtype.
            **_: Ignored; accepted for container interoperability.

        Returns:
            ``1 / (1 + exp(-x))`` with the same shape and dtype as ``x``.
        """
        return F.sigmoid(x)
