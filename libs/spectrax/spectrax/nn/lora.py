# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Low-Rank Adaptation (LoRA) adapters.

Implements the factorization from *Hu et al. 2021*:

.. math::

    y = W_0 x + \\tfrac{\\alpha}{r} \\cdot (A B) x

where ``W_0`` is a (typically frozen) base weight, ``A \\in R^{d_{in} \\times r}``
and ``B \\in R^{r \\times d_{out}}`` are learned low-rank factors, ``r`` is
the rank, and ``alpha`` is a scaling hyperparameter.

The learnable factors live in the dedicated ``"lora"`` collection so
``spx.grad(wrt="lora")`` (or ``spx.grad(wrt=LoraParameter)``) trains
*only* the adapter while the base layer's ``"parameters"`` collection is
treated as constant. :class:`LoRA` optionally composes with any
callable submodule via ``base_module=``, which is the standard pattern
for fine-tuning a pretrained layer without mutating it. :class:`LoRALinear`
is a convenience for the common case of "a Linear whose output also
gets a LoRA delta".

At inference, folding the adapter into the base weight is a one-line
operation on public :class:`~spectrax.Variable` state::

    base.weight.value += (alpha / rank) * (lora.lora_a.value @ lora.lora_b.value)
    lora.lora_a.value = jnp.zeros_like(lora.lora_a.value)
    lora.lora_b.value = jnp.zeros_like(lora.lora_b.value)

After the fold, forward passes still call the adapter but the delta
is exactly zero — no hidden mode switch, no API to forget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ClassVar

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.variable import Variable
from ..init import kaiming_uniform, zeros
from ..rng.rngs import Rngs, resolve_rngs
from .linear import Linear

__all__ = [
    "LoRA",
    "LoRALinear",
    "LoraParameter",
    "wrap_lora",
]


class LoraParameter(Variable):
    """Variable cell for LoRA factors; defaults to the ``"lora"`` collection.

    Keeping LoRA weights in their own collection means ``spx.grad(wrt="lora")``
    trains only the adapter. The class-based selector
    ``spx.grad(wrt=LoraParameter)`` (via :func:`spectrax.of_type`) is
    equivalent — pick whichever spelling reads better at the call site.
    """

    default_kind: ClassVar[str] = "lora"


class LoRA(Module):
    """Standalone LoRA adapter, optionally wrapping a base module.

    Two usage shapes:

    1. **Additive adapter** — constructed without ``base_module=``, the
       module returns *only* the ``(A, B)`` delta. Useful for
       parameter-efficient insertions (e.g. a residual update on top
       of an existing activation).
    2. **Wrapper over a pretrained layer** — passing ``base_module=``
       makes :meth:`forward` compute ``base(x) + delta(x)``. The
       canonical fine-tuning workflow: load the pretrained ``base``,
       leave its ``"parameters"`` collection frozen, train the adapter.

    Examples::

        >>> lora = spx_nn.LoRA(d_in=768, rank=8, d_out=768, rngs=spx.Rngs(0))
        >>> y = lora(x)                     # just the A·B delta

        >>> base = spx_nn.Linear(768, 768, rngs=spx.Rngs(1))
        >>> lora = spx_nn.LoRA(768, 8, 768, base_module=base, rngs=spx.Rngs(2))
        >>> y = lora(x)                     # base(x) + x·A·B (no alpha set)

    The factors are initialized so that ``lora_b = 0`` — with the
    canonical LoRA zero-init, the adapter is a no-op on the first
    forward pass, meaning ``lora(x) == base(x)`` at step 0 and fine-tuning
    does not drift the pretrained behaviour before training begins.

    Args:
        d_in: Input feature count.
        rank: Rank of the LoRA factorization (``r``). Must be positive.
        d_out: Output feature count.
        base_module: Optional callable whose output is added to the
            LoRA delta. Any callable works; typically a :class:`Linear`
            or another spectrax module.
        alpha: Optional scaling hyperparameter. When provided, the
            delta is multiplied by ``alpha / rank``; when ``None``
            (default), no scaling is applied — equivalent to the
            behaviour where scaling is opt-in.
        rngs: PRNG source for parameter initialization.
        a_init: Initializer for ``lora_a``. Default
            :func:`~spectrax.init.kaiming_uniform` with ``"linear"`` gain.
        b_init: Initializer for ``lora_b``. Default
            :func:`~spectrax.init.zeros` so the adapter is a no-op at init.
        dtype: Parameter dtype (default ``float32``).
    """

    lora_a: LoraParameter
    lora_b: LoraParameter

    def __init__(
        self,
        d_in: int,
        rank: int,
        d_out: int,
        *,
        base_module: Callable[..., Array] | None = None,
        alpha: float | None = None,
        rngs: Rngs | int | None = None,
        a_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the low-rank factors and optionally wire a base module.

        Args:
            d_in: Input feature count (size of the leading axis of
                ``A``).
            rank: Rank ``r`` of the factorization (size of the
                shared inner axis of ``A`` and ``B``). Must be
                positive.
            d_out: Output feature count (size of the trailing axis
                of ``B``).
            base_module: Optional callable whose output is added to
                the LoRA delta. Any callable works; typically a
                :class:`Linear` or another spectrax module. Stored
                on the instance only when not ``None``.
            alpha: Optional scaling hyperparameter. When provided,
                the delta is multiplied by ``alpha / rank``; when
                ``None`` (default), no scaling is applied.
            rngs: PRNG source for parameter initialization.
            a_init: Initializer for ``lora_a``. Defaults to
                Kaiming-uniform with ``"linear"`` gain.
            b_init: Initializer for ``lora_b``. Defaults to
                :func:`~spectrax.init.zeros` so the adapter is a
                no-op at step 0.
            dtype: Storage dtype for the factors; defaults to
                ``float32``.

        Raises:
            ValueError: If ``rank`` is non-positive.
        """
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.d_in = d_in
        self.d_out = d_out
        self.rank = rank
        self.has_alpha = alpha is not None
        self.alpha = float(alpha) if alpha is not None else 0.0
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        a_init = a_init or kaiming_uniform("linear")
        b_init = b_init or zeros
        self.lora_a = LoraParameter(a_init(resolved.parameters, (d_in, rank), dt))
        self.lora_b = LoraParameter(b_init(resolved.parameters, (rank, d_out), dt))
        if base_module is not None:
            self.base_module = base_module

    def _scale(self, out_dtype: DType) -> Array | float:
        """Return the ``alpha / rank`` LoRA scaling factor as ``out_dtype``.

        Args:
            out_dtype: Dtype to cast the result to so it composes with
                the matmul output without surprise upcasts.

        Returns:
            A 0-d :class:`jax.Array` of dtype ``out_dtype`` holding
            ``alpha / rank`` when :attr:`has_alpha` is ``True``;
            otherwise the Python scalar ``1.0`` so the call site
            can skip the broadcast-multiply entirely.
        """
        if not self.has_alpha:
            return 1.0
        return jnp.asarray(self.alpha / self.rank, dtype=out_dtype)

    def forward(self, x: ArrayLike, *args: object, **kwargs: object) -> Array:
        """Return ``base(x) + scale * (x @ A @ B)`` — or just the delta.

        Downcasts the factors to ``x``'s dtype when ``x`` is lower
        precision, matching the policy used in
        :func:`spectrax.functional.linear`. Extra positional / keyword
        arguments are forwarded to ``base_module`` when present, so
        this layer composes transparently with bases that need e.g.
        a mask argument.

        Args:
            x: Input activation.
            *args, **kwargs: Forwarded to ``base_module`` when present.

        Returns:
            ``base(x) + scale * x @ A @ B`` when a base is configured;
            otherwise just the LoRA delta.
        """
        xa = jnp.asarray(x)
        a = self.lora_a.value
        b = self.lora_b.value
        if xa.dtype != a.dtype and jnp.issubdtype(xa.dtype, jnp.floating) and jnp.issubdtype(a.dtype, jnp.floating):
            if jnp.finfo(xa.dtype).bits < jnp.finfo(a.dtype).bits:
                a = a.astype(xa.dtype)
                b = b.astype(xa.dtype)
            else:
                xa = xa.astype(a.dtype)
        delta = (xa @ a) @ b
        if self.has_alpha:
            delta = delta * self._scale(delta.dtype)

        base = getattr(self, "base_module", None)
        if base is not None:
            if not callable(base):
                raise TypeError("LoRA.base_module must be callable")
            base_out = base(x, *args, **kwargs)
            if base_out.dtype != delta.dtype and jnp.issubdtype(base_out.dtype, jnp.floating):
                delta = delta.astype(base_out.dtype)
            return base_out + delta
        return delta


class LoRALinear(Module):
    """Dense layer whose output receives a LoRA adapter.

    State layout is designed for interop: the underlying
    :class:`~spectrax.nn.Linear` is stored as ``self.base``, so its
    ``weight`` / ``bias`` live at canonical paths ``base.weight`` and
    ``base.bias`` in the ``"parameters"`` collection. The LoRA adapter's
    ``lora_a`` / ``lora_b`` live at ``lora.lora_a`` / ``lora.lora_b``
    in the ``"lora"`` collection. Training only the adapter is then a
    one-line selector: ``spx.grad(wrt="lora")``.

    Example::

        >>> ll = spx_nn.LoRALinear(768, 768, rank=8, rngs=spx.Rngs(0))
        >>> y = ll(x)                        # y = x W + b + x A B

    Args:
        in_features: Trailing input feature count.
        out_features: Output feature count.
        rank: LoRA rank.
        alpha: Optional LoRA alpha (scales the delta by ``alpha / rank``).
        use_bias: Whether the base Linear has a bias.
        rngs: PRNG source for both the base and the adapter.
        w_init: Initializer for the base weight.
        b_init: Initializer for the base bias.
        a_init: Initializer for ``lora_a``.
        lora_b_init: Initializer for ``lora_b`` (default zeros).
        dtype: Parameter dtype for base and adapter alike.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float | None = None,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        a_init: Initializer | None = None,
        lora_b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Instantiate the base :class:`Linear` and the side-by-side :class:`LoRA`.

        Both sub-modules draw keys from the same resolved
        :class:`Rngs` instance, so constructing two identically-seeded
        :class:`LoRALinear`\\ s produces identical parameters.

        Args:
            in_features: Trailing input feature count.
            out_features: Output feature count.
            rank: LoRA rank ``r``.
            alpha: Optional LoRA alpha; the delta is scaled by
                ``alpha / rank`` when supplied.
            use_bias: Whether the base :class:`Linear` carries a bias.
            rngs: PRNG source for both the base and the adapter.
            w_init: Initializer for the base weight (forwarded to
                :class:`Linear`).
            b_init: Initializer for the base bias.
            a_init: Initializer for ``lora_a`` (forwarded to
                :class:`LoRA`).
            lora_b_init: Initializer for ``lora_b`` (defaults to
                zeros via :class:`LoRA`'s default).
            dtype: Parameter dtype for both base and adapter.
        """
        super().__init__()
        resolved = resolve_rngs(rngs)
        self.base = Linear(
            in_features,
            out_features,
            use_bias=use_bias,
            rngs=resolved,
            w_init=w_init,
            b_init=b_init,
            dtype=dtype,
        )
        self.lora = LoRA(
            in_features,
            rank,
            out_features,
            rngs=resolved,
            alpha=alpha,
            a_init=a_init,
            b_init=lora_b_init,
            dtype=dtype,
        )

    def forward(self, x: ArrayLike, **kwargs: object) -> Array:
        """Apply the base Linear and add the LoRA delta.

        Args:
            x: Input activation.
            **kwargs: Forwarded to the base layer.

        Returns:
            ``base(x) + lora(x)``.
        """
        return self.base(x, **kwargs) + self.lora(x)


def wrap_lora(
    base: Module,
    rank: int,
    *,
    alpha: float | None = None,
    rngs: Rngs | int | None = None,
    a_init: Initializer | None = None,
    b_init: Initializer | None = None,
    dtype: DType | None = None,
) -> LoRA:
    """Retrofit an existing :class:`Linear`-like module with a LoRA adapter.

    Convenience factory: reads ``in_features`` / ``out_features`` off
    ``base`` and builds a :class:`LoRA` that calls ``base`` inside its
    forward pass. Semantically identical to::

        LoRA(d_in, rank, d_out, base_module=base, ...)

    but saves the caller from re-stating the feature dimensions when
    they are already available on the base layer.

    Args:
        base: Pretrained callable module. Must expose
            ``in_features`` and ``out_features`` attributes —
            satisfied by :class:`~spectrax.nn.Linear`.
        rank: LoRA rank.
        alpha: Optional LoRA alpha.
        rngs: PRNG source for the adapter parameters.
        a_init: Initializer for ``lora_a``.
        b_init: Initializer for ``lora_b`` (defaults to zeros so the
            wrapped module matches ``base`` at step 0).
        dtype: Adapter parameter dtype.

    Returns:
        A :class:`LoRA` instance whose :attr:`base_module` is ``base``.

    Raises:
        TypeError: When ``base`` does not expose both feature attrs.
    """
    d_in = getattr(base, "in_features", None)
    d_out = getattr(base, "out_features", None)
    if d_in is None or d_out is None:
        raise TypeError("wrap_lora requires `base` to expose in_features/out_features")
    return LoRA(
        d_in,
        rank,
        d_out,
        base_module=base,
        alpha=alpha,
        rngs=rngs,
        a_init=a_init,
        b_init=b_init,
        dtype=dtype,
    )
