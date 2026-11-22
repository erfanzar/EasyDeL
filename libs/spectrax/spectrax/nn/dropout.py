# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inverted-dropout layer.

Element-wise dropout with the standard *inverted* scaling — the kept
elements are multiplied by :math:`1/(1-p)` so the expected activation
magnitude is unchanged across train and eval. The mask is sampled
fresh on every call from the ``"dropout"`` stream of the supplied
:class:`~spectrax.Rngs`.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike
from ..core.module import Module
from ..functional.dropout import dropout as F_dropout
from ..rng.rngs import Rngs


class Dropout(Module):
    """Element-wise inverted dropout.

    Each call draws an independent Bernoulli mask from the ``"dropout"``
    stream of the supplied :class:`~spectrax.Rngs` and applies it with
    inverted scaling so :math:`E[\\text{out}] = x`. Dropout is bypassed
    (returns the input as-is) when any of the following is true:

    * the layer is in eval mode (set explicitly via ``module.eval()`` /
      attribute :attr:`~spectrax.Module.training` is ``False``);
    * ``deterministic=True`` is passed explicitly to :meth:`forward`;
    * the configured drop ``rate`` is exactly ``0.0``.

    The :class:`~spectrax.Rngs` argument may be supplied either at
    construction (stored as ``self.rngs``, used as a default) or at
    each :meth:`forward` call. A per-call value always wins.
    """

    def __init__(self, rate: float = 0.5, *, rngs: Rngs | None = None) -> None:
        """Record the drop rate and an optional default :class:`Rngs`.

        Args:
            rate: Drop probability, a Python ``float`` in the
                half-open interval ``[0, 1)``. ``0`` disables dropout
                entirely (forward returns the input unchanged).
            rngs: Optional :class:`~spectrax.Rngs` stashed on the
                module so callers do not need to thread one into every
                :meth:`forward`. Kept for API parity with frameworks
                that wire RNG sources at construction time.

        Raises:
            ValueError: If ``rate`` is outside ``[0, 1)``. Note ``1.0``
                is also rejected (would zero everything).
        """
        super().__init__()
        if not 0.0 <= rate < 1.0:
            raise ValueError(f"dropout rate must be in [0, 1), got {rate}")
        self.rate = rate
        if rngs is not None:
            self.rngs = rngs

    def forward(
        self,
        x: ArrayLike | None = None,
        *,
        inputs: ArrayLike | None = None,
        rngs: Rngs | None = None,
        deterministic: bool | None = None,
        **_: object,
    ) -> Array:
        """Apply dropout to ``x`` (or pass it through in deterministic mode).

        The Bernoulli mask is drawn from ``rngs.key("dropout")`` — a
        fresh key on every call, since :class:`~spectrax.Rngs`
        increments its internal counter per :meth:`~spectrax.Rngs.key`
        invocation.

        Args:
            x: Input tensor of any shape and floating-point dtype.
            inputs: Backwards-compatible alias for ``x``. Exactly one
                of ``x`` or ``inputs`` must be provided.
            rngs: :class:`~spectrax.Rngs` whose ``"dropout"`` stream
                supplies the mask key. When ``None``, falls back to
                the :class:`Rngs` stored on the module at construction
                time (if any).
            deterministic: Explicit override. ``None`` (default) means
                "use ``not self.training``" — i.e. dropout is active
                only in training mode. Pass ``True`` to force a
                pass-through evaluation, or ``False`` to force
                stochastic application even in eval mode.
            **_: Additional kwargs are accepted for forward
                compatibility and silently ignored.

        Returns:
            ``x`` cast through :func:`jax.numpy.asarray` if dropout is
            disabled; otherwise the masked-and-rescaled tensor with
            the same shape and dtype as ``x``.

        Raises:
            TypeError: If both ``x`` and ``inputs`` are passed; if
                neither is passed; or if ``rngs`` is present but is
                not an :class:`Rngs`.
            RuntimeError: If dropout is active (training mode and
                non-zero rate) but no :class:`Rngs` is available
                (neither per-call nor stashed).
        """
        if x is not None and inputs is not None:
            raise TypeError("Dropout.forward() got both 'x' and 'inputs'; pass only one.")
        if x is None and inputs is not None:
            x = inputs
        if x is None:
            raise TypeError("Dropout.forward() missing required argument: 'x'")
        if deterministic is None:
            deterministic = not self.training
        if deterministic or self.rate == 0.0:
            return jnp.asarray(x)
        if rngs is None:
            rngs = getattr(self, "rngs", None)
        if rngs is None:
            raise RuntimeError(
                "Dropout in training mode requires `rngs=...`. Pass rngs through "
                "forward(), or set deterministic=True / call eval()."
            )
        if not isinstance(rngs, Rngs):
            raise TypeError(f"rngs must be an Rngs, got {type(rngs).__name__}")
        return F_dropout(x, self.rate, key=rngs.key("dropout"), deterministic=False)
