# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Two-layer feed-forward MLP block.

Compact :class:`~spectrax.Module` implementing the standard
``Linear -> activation -> Dropout -> Linear`` sandwich found inside
every transformer block. Defaults to ``4 * features`` hidden width
and GELU activation; ``out_features`` defaults back to ``features``
so the block is residual-shape compatible by default.
"""

from __future__ import annotations

from typing import cast

from ..core._typing import Array, ArrayLike, DType
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..functional.activation import gelu, relu, silu
from ..rng.rngs import Rngs
from .dropout import Dropout
from .linear import Linear


class MLPBlock(Module):
    """Two-layer feed-forward block ``Linear -> activation -> Dropout -> Linear``.

    Drop-in feed-forward half of a transformer block. State layout:

    * ``self.fc1`` — :class:`Linear` from ``features`` to
      ``hidden_features``.
    * ``self.fc2`` — :class:`Linear` from ``hidden_features`` to
      ``out_features``.
    * ``self.drop`` — :class:`Dropout` applied to the post-activation
      hidden representation only.

    Both linears use the framework defaults (Kaiming-uniform weight,
    zero bias). Because the activation is selected by a string at
    construction time, the layer is JAX-traceable and pickle-safe.
    """

    fc1: Linear
    fc2: Linear
    drop: Dropout

    def __init__(
        self,
        features: int,
        hidden_features: int | None = None,
        *,
        out_features: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        fc1_sharding: Sharding | AxisNames | None = None,
        fc2_sharding: Sharding | AxisNames | None = None,
        fc1_bias_sharding: Sharding | AxisNames | None = None,
        fc2_bias_sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Initialize the block.

        Args:
            features: Input feature count and default for
                ``out_features``.
            hidden_features: Hidden width. Defaults to
                ``4 * features`` — the standard transformer ratio.
            out_features: Output feature count. Defaults to
                ``features`` so the block can be used as a residual
                update without extra projections.
            dropout: Drop probability for the :class:`Dropout` layer
                between the activation and the second linear.
                ``0.0`` (default) disables it entirely.
            activation: Name of the activation; one of ``"gelu"``,
                ``"relu"``, ``"silu"``. Used as a string so the
                layer remains a static-attribute Spectrax module.
            rngs: PRNG source for both ``fc1`` and ``fc2``.
            dtype: Parameter dtype forwarded to both linears.
            fc1_sharding: Optional sharding for the first linear's
                weight.
            fc2_sharding: Optional sharding for the second linear's
                weight.
            fc1_bias_sharding: Optional sharding for the first
                linear's bias.
            fc2_bias_sharding: Optional sharding for the second
                linear's bias.
        """
        super().__init__()
        hidden = hidden_features if hidden_features is not None else 4 * features
        out = out_features if out_features is not None else features
        self.features = features
        self.hidden_features = hidden
        self.out_features = out
        self.activation = activation
        self.fc1 = Linear(
            features,
            hidden,
            rngs=rngs,
            dtype=dtype,
            sharding=fc1_sharding,
            bias_sharding=fc1_bias_sharding,
        )
        self.fc2 = Linear(
            hidden,
            out,
            rngs=rngs,
            dtype=dtype,
            sharding=fc2_sharding,
            bias_sharding=fc2_bias_sharding,
        )
        self.drop = Dropout(dropout)

    def forward(self, x: ArrayLike, *, rngs: Rngs | None = None, **_: object) -> Array:
        """Thread ``x`` through ``fc1 -> activation -> Dropout -> fc2``.

        Args:
            x: Input tensor whose trailing axis equals
                :attr:`features`.
            rngs: :class:`Rngs` forwarded to the inner
                :class:`Dropout`. Required only when ``dropout > 0``
                and the module is in training mode.
            **_: Ignored; accepted for container interoperability.

        Returns:
            Tensor with shape ``x.shape[:-1] + (out_features,)``.

        Raises:
            ValueError: If :attr:`activation` is not one of
                ``"gelu"``, ``"relu"``, ``"silu"``.
        """
        y = self.fc1(x)
        if self.activation == "gelu":
            y = gelu(y)
        elif self.activation == "relu":
            y = relu(y)
        elif self.activation == "silu":
            y = silu(y)
        else:
            raise ValueError(f"Unknown activation: {self.activation!r}")
        y = self.drop(y, rngs=rngs)
        return cast(Array, self.fc2(y))
