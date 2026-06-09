# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Embedding lookup table with an output-head (:meth:`Embed.attend`) mode.

The :class:`Embed` layer holds a single learnable ``(vocab, features)``
matrix and exposes two operations on it:

* :meth:`~Embed.lookup` (also the default :meth:`~Embed.forward`) —
  classic gather-by-id used at the input side of language / sequence
  models;
* :meth:`~Embed.attend` — multiplies a query against the *transpose*
  of the table to produce vocabulary logits, which is exactly the
  pattern used to weight-tie input embeddings with the output
  classification head.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.sharding import AxisNames, Sharding
from ..core.variable import DeferredParameter, Parameter
from ..init import normal
from ..rng.rngs import Rngs, resolve_rngs


class Embed(Module):
    """Lookup table mapping integer ids to dense vectors.

    Stores a single :class:`~spectrax.Parameter` named ``weight`` of
    shape ``(num_embeddings, features)`` with logical axis names
    ``("vocab", "embed")`` for sharding resolution. Three call modes:

    * :meth:`lookup` — index the table with integer ids, returning
      ``ids.shape + (features,)``.
    * :meth:`attend` — produce ``q @ W.T``, yielding vocabulary
      logits suitable for a tied output head.
    * :meth:`forward` — alias for :meth:`lookup` so the layer can be
      used as a drop-in replacement inside any container that calls
      its children with a single tensor positional.

    Shape inference: passing ``num_embeddings=None`` defers the table
    allocation until the first :meth:`lookup` call, which materialises
    the table at ``int(ids.max()) + 1`` rows. This is convenient for
    quick prototyping but cannot be used inside a :func:`jax.jit` /
    :func:`spectrax.export` boundary because the resulting shape is
    data-dependent — the implementation guards against that with
    :meth:`~spectrax.Module._spx_guard_not_in_transform`.
    """

    weight: Parameter

    def __init__(
        self,
        num_embeddings: int | None,
        features: int,
        *,
        rngs: Rngs | int | None = None,
        dtype: DType | None = None,
        param_dtype: DType | None = None,
        w_init: Initializer | None = None,
        sharding: Sharding | AxisNames | None = None,
    ) -> None:
        """Allocate the embedding table (or defer it to first use).

        Args:
            num_embeddings: Vocabulary size. Pass ``None`` to defer
                allocation until the first :meth:`lookup` call, which
                will use ``int(ids.max()) + 1`` as the table size; the
                deferred path cannot run inside a JAX transform.
            features: Embedding dimension (trailing axis of the table).
            rngs: Source of PRNG keys used to initialise the table.
                Accepts an :class:`Rngs`, an ``int`` seed, or
                ``None``; resolved via :func:`resolve_rngs`.
            dtype: Storage dtype for the parameter. Defaults to
                ``float32`` when both ``dtype`` and ``param_dtype``
                are ``None``.
            param_dtype: Alias for ``dtype``; takes precedence when
                both are supplied. Provided for parity with frameworks
                that name the storage dtype this way.
            w_init: Weight initializer. Defaults to
                :func:`~spectrax.init.normal` with ``stddev=1`` —
                the standard "small isotropic" init for token
                embeddings.
            sharding: Optional :class:`~spectrax.core.sharding.Sharding`
                or axis-name tuple for the embedding table; the
                logical axis names ``("vocab", "embed")`` are attached
                automatically so a mesh can resolve them.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.features = features
        resolved = resolve_rngs(rngs)
        init = w_init or normal(stddev=1.0)
        weight_dtype = param_dtype or dtype or jnp.float32
        if num_embeddings is None:
            self.weight = DeferredParameter(
                (None, features),
                init,
                resolved.parameters,
                weight_dtype,
                sharding=sharding,
                axis_names=("vocab", "embed"),
            )
        else:
            self.weight = Parameter(
                init(resolved.parameters, (num_embeddings, features), weight_dtype),
                sharding=sharding,
                axis_names=("vocab", "embed"),
            )

    def lookup(self, ids: ArrayLike) -> Array:
        """Gather embedding vectors for the given integer ids.

        On the first call, if ``num_embeddings`` was deferred, this
        materialises the underlying :class:`DeferredParameter` to
        size ``int(ids.max()) + 1`` and stamps :attr:`num_embeddings`
        onto the module so subsequent calls skip the inference path.

        Args:
            ids: Integer-typed array of any shape; values must be
                non-negative and strictly less than the (possibly
                inferred) :attr:`num_embeddings`.

        Returns:
            An array of shape ``ids.shape + (features,)`` whose dtype
            matches the embedding table's parameter dtype.

        Raises:
            RuntimeError: If the deferred allocation path is hit while
                inside a JAX transform (the data-dependent table size
                cannot be traced).
        """
        ids_arr = jnp.asarray(ids)
        if self.num_embeddings is None:
            self._spx_guard_not_in_transform("DeferredParameter materialization")
            num_embeddings = int(ids_arr.max()) + 1
            self._resolve_deferred(self.weight, (num_embeddings, self.features))
            self.num_embeddings = num_embeddings
        return self.weight.value[ids_arr]

    def attend(self, q: ArrayLike) -> Array:
        """Compute logits ``q @ W.T`` against the embedding table.

        The standard "tied head" trick: instead of allocating a
        separate ``(features, vocab)`` projection at the output, reuse
        the input embedding's transpose. The two passes share
        gradients on a single matrix.

        Args:
            q: Query tensor whose trailing axis equals
                :attr:`features`. Any leading shape is preserved.

        Returns:
            ``q @ W.T`` with shape ``q.shape[:-1] + (num_embeddings,)``.
            Dtype follows the standard NumPy promotion rules between
            ``q.dtype`` and the table's parameter dtype.
        """
        return jnp.asarray(q) @ self.weight.value.T

    def forward(self, ids: ArrayLike, **_: object) -> Array:
        """Convenience alias that dispatches to :meth:`lookup`.

        Allows the embedding to be placed inside a generic container
        that calls every child as ``child(x)``.

        Args:
            ids: Integer ids — see :meth:`lookup`.
            **_: Ignored; accepted for container interoperability.

        Returns:
            The result of :meth:`lookup` on ``ids``.
        """
        return self.lookup(ids)
