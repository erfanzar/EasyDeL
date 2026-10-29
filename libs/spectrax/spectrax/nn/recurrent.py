# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Recurrent cells and sequence wrappers.

Cells follow the standard ``(carry, x) -> (new_carry, y)`` shape so
they compose cleanly with :func:`jax.lax.scan`. Each cell exposes an
:meth:`~RNNCellBase.initial_carry` helper that returns the
zero-valued carry for a given batch shape.

Wrappers
:class:`RNN` scans any :class:`RNNCellBase` across a time axis
(sequence-second by default, or sequence-first via
``time_major=True``) with optional reversal and final-carry
return. :class:`Bidirectional` pairs a forward and a
reverse-configured :class:`RNN` and merges their outputs by
concatenation, sum, product, or average.

All weight matrices use the convention
``(in, out)`` with logical axis names ``("in", "out")``; biases use
``("out",)``. Cells default to a Kaiming-uniform initializer for the
input-side matrix and an orthogonal initializer for the
recurrent-side matrix (the common practice for RNNs to keep early
training stable).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import ClassVar, cast

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType, Initializer
from ..core.module import Module
from ..core.variable import Parameter
from ..functional.conv import conv as F_conv
from ..init import kaiming_uniform, orthogonal, zeros
from ..rng.rngs import Rngs, resolve_rngs

Carry = object

__all__ = [
    "RNN",
    "Bidirectional",
    "ConvLSTMCell",
    "GRUCell",
    "LSTMCell",
    "OptimizedLSTMCell",
    "RNNCellBase",
    "SimpleRNNCell",
]


class RNNCellBase(Module):
    """Abstract recurrent cell — single-step interface.

    Subclasses implement :meth:`forward` as a pure step
    ``(carry, x) -> (new_carry, y)`` and :meth:`initial_carry`
    returning a zero-valued carry sized for a given batch shape. The
    single-step contract is what :class:`RNN` (and any other scan
    wrapper) plugs into :func:`jax.lax.scan`.

    The carry type is cell-specific: :class:`SimpleRNNCell` /
    :class:`GRUCell` carry a single hidden tensor; :class:`LSTMCell`,
    :class:`OptimizedLSTMCell`, and :class:`ConvLSTMCell` carry an
    ``(h, c)`` tuple.
    """

    num_feats: ClassVar[int] = 0

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> Carry:
        """Return the zero-valued carry for ``batch_shape``.

        Args:
            batch_shape: Leading dimensions of the recurrent state
                (everything before the trailing feature/channel
                axes).
            dtype: Optional dtype override; defaults to the cell's
                parameter dtype.

        Returns:
            A pytree (single :class:`~jax.Array` or a tuple) shaped
            so it can be threaded as the ``carry`` argument of
            :meth:`forward`.

        Raises:
            NotImplementedError: Always — concrete subclasses must
                override.
        """
        raise NotImplementedError

    def forward(self, carry: Carry, x: ArrayLike) -> tuple[Carry, Array]:
        """Advance the recurrent state by one step.

        Args:
            carry: Recurrent state from the previous step.
            x: Per-step input slice (shape
                ``batch_shape + (in_features,)`` for the dense
                cells).

        Returns:
            ``(new_carry, y)`` where ``new_carry`` has the same
            structure as the input ``carry`` and ``y`` is the
            per-step output.

        Raises:
            NotImplementedError: Always — concrete subclasses must
                override.
        """
        raise NotImplementedError


class SimpleRNNCell(RNNCellBase):
    """Elman recurrent cell.

    Update rule: ``h' = activation(x @ W_xh + h @ W_hh + b)``; the
    output equals the new hidden state. Defaults to a
    :func:`jax.numpy.tanh` activation; the user may pass any
    callable. The recurrent matrix ``W_hh`` defaults to an
    orthogonal initializer (a common choice that avoids early-step
    explosion).
    """

    W_xh: Parameter
    W_hh: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        activation: Callable[[Array], Array] = jnp.tanh,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        h_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the cell parameters.

        Args:
            in_features: Trailing input feature count.
            hidden_features: Hidden state size.
            use_bias: When ``True`` (default), allocate the bias
                ``b`` of shape ``(hidden_features,)``.
            activation: Per-step activation. Stored as ``_activation``
                and called on the pre-activation gate sum. Falls
                back to :func:`jax.numpy.tanh` if the supplied value
                is not callable.
            rngs: PRNG source for parameter initialization.
            w_init: Input-side weight initializer; defaults to
                Kaiming-uniform with the ``"linear"`` gain.
            h_init: Hidden-side weight initializer; defaults to
                :func:`~spectrax.init.orthogonal`.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype for the parameters; defaults to
                ``float32``.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        self._activation = activation if callable(activation) else jnp.tanh
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        h_init = h_init or orthogonal()
        self.W_xh = Parameter(w_init(resolved.parameters, (in_features, hidden_features), dt), axis_names=("in", "out"))
        self.W_hh = Parameter(
            h_init(resolved.parameters, (hidden_features, hidden_features), dt), axis_names=("in", "out")
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (hidden_features,), dt), axis_names=("out",))

    def initial_carry(self, batch_shape: Sequence[int], *, dtype: DType | None = None) -> Array:
        """Allocate a zero hidden state.

        Args:
            batch_shape: Leading dimensions to prepend to the
                hidden axis.
            dtype: Optional dtype override; defaults to the
                ``W_xh`` dtype.

        Returns:
            Zero array of shape ``(*batch_shape, hidden_features)``.
        """
        dt = dtype or self.W_xh.dtype
        return jnp.zeros((*tuple(batch_shape), self.hidden_features), dtype=dt)

    def forward(self, carry: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
        """Advance the cell by one step.

        Args:
            carry: Previous hidden state of shape
                ``(*batch_shape, hidden_features)``.
            x: Per-step input of shape
                ``(*batch_shape, in_features)``.

        Returns:
            ``(h_new, h_new)`` — the new hidden state appears in
            both the carry slot and the output slot, since this
            cell's output equals its hidden state.
        """
        h = jnp.asarray(carry)
        z = jnp.asarray(x) @ self.W_xh.value + h @ self.W_hh.value
        if self.use_bias:
            z = z + self.b.value
        h_new = self._activation(z)
        return h_new, h_new


class LSTMCell(RNNCellBase):
    """Standard LSTM with separate input and hidden weight matrices.

    Update rule: per-step, computes the four gate
    pre-activations as ``z = x @ W_x + h @ W_h + b`` (a
    ``(*, 4*hidden_features)`` tensor), splits into
    ``(i, f, g, o)``, applies :math:`\\sigma` to ``i``, ``f``, ``o``
    and :math:`\\tanh` to ``g``, then ``c' = f * c + i * g`` and
    ``h' = o * tanh(c')``. The carry is the ``(h, c)`` tuple.

    Cheaper to read but slightly more arithmetic than
    :class:`OptimizedLSTMCell`, which fuses the two matmuls.
    """

    W_x: Parameter
    W_h: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        h_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the cell parameters.

        Args:
            in_features: Trailing input feature count.
            hidden_features: Hidden state size.
            use_bias: When ``True`` (default), allocate the bias
                ``b`` of shape ``(4 * hidden_features,)``.
            rngs: PRNG source for parameter initialization.
            w_init: Input-side weight initializer; defaults to
                Kaiming-uniform with the ``"linear"`` gain.
            h_init: Hidden-side weight initializer; defaults to
                :func:`~spectrax.init.orthogonal`.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype; defaults to ``float32``.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        h_init = h_init or orthogonal()
        self.W_x = Parameter(
            w_init(resolved.parameters, (in_features, 4 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        self.W_h = Parameter(
            h_init(resolved.parameters, (hidden_features, 4 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (4 * hidden_features,), dt), axis_names=("out",))

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> tuple[Array, Array]:
        """Allocate a zero ``(h, c)`` tuple.

        Args:
            batch_shape: Leading dimensions to prepend to the
                hidden axis.
            dtype: Optional dtype override; defaults to ``W_x``'s
                dtype.

        Returns:
            ``(h0, c0)`` — both zero arrays of shape
            ``(*batch_shape, hidden_features)``.
        """
        dt = dtype or self.W_x.dtype
        shape = (*tuple(batch_shape), self.hidden_features)
        return jnp.zeros(shape, dtype=dt), jnp.zeros(shape, dtype=dt)

    def forward(self, carry: tuple[Array, Array], x: ArrayLike) -> tuple[tuple[Array, Array], Array]:
        """Advance the LSTM by one step.

        Args:
            carry: Previous ``(h, c)`` tuple, both shaped
                ``(*batch_shape, hidden_features)``.
            x: Per-step input of shape
                ``(*batch_shape, in_features)``.

        Returns:
            ``((h_new, c_new), h_new)`` — the per-step output is the
            new hidden state.
        """
        h, c = carry
        z = jnp.asarray(x) @ self.W_x.value + h @ self.W_h.value
        if self.use_bias:
            z = z + self.b.value
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class OptimizedLSTMCell(RNNCellBase):
    """LSTM cell with a single fused matmul over ``[x, h]``.

    Mathematically identical to :class:`LSTMCell`, but computes the
    pre-activations as ``[x, h] @ W + b`` — a single matmul with a
    ``(in_features + hidden_features, 4 * hidden_features)`` weight.
    On most accelerators this is faster than the two separate matmuls
    used by :class:`LSTMCell`. Note this variant does *not* use a
    separate orthogonal initializer for the recurrent block: a single
    weight is initialized end-to-end with the supplied ``w_init``
    (default Kaiming-uniform).
    """

    W: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the fused weight and bias.

        Args:
            in_features: Trailing input feature count.
            hidden_features: Hidden state size.
            use_bias: When ``True`` (default), allocate the bias.
            rngs: PRNG source for parameter initialization.
            w_init: Weight initializer applied to the full
                ``(in_features + hidden_features, 4 * hidden_features)``
                matrix; defaults to Kaiming-uniform with
                ``"linear"`` gain.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype; defaults to ``float32``.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        self.W = Parameter(
            w_init(resolved.parameters, (in_features + hidden_features, 4 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (4 * hidden_features,), dt), axis_names=("out",))

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> tuple[Array, Array]:
        """Allocate a zero ``(h, c)`` tuple.

        Args:
            batch_shape: Leading dimensions to prepend to the
                hidden axis.
            dtype: Optional dtype override; defaults to the fused
                weight's dtype.

        Returns:
            ``(h0, c0)`` with shape
            ``(*batch_shape, hidden_features)``.
        """
        dt = dtype or self.W.dtype
        shape = (*tuple(batch_shape), self.hidden_features)
        return jnp.zeros(shape, dtype=dt), jnp.zeros(shape, dtype=dt)

    def forward(self, carry: tuple[Array, Array], x: ArrayLike) -> tuple[tuple[Array, Array], Array]:
        """Advance the fused LSTM by one step.

        Args:
            carry: ``(h, c)`` tuple from the previous step.
            x: Per-step input of shape
                ``(*batch_shape, in_features)``.

        Returns:
            ``((h_new, c_new), h_new)``.
        """
        h, c = carry
        xh = jnp.concatenate([jnp.asarray(x), h], axis=-1)
        z = xh @ self.W.value
        if self.use_bias:
            z = z + self.b.value
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class GRUCell(RNNCellBase):
    """Gated recurrent unit (Cho et al., 2014).

    Three gates packed into ``W_x`` (shape
    ``(in_features, 3 * hidden_features)``) and ``W_h`` (shape
    ``(hidden_features, 3 * hidden_features)``); the bias is added to
    the input-side projection only. Slicing convention: the leading
    ``2 * hidden_features`` columns produce the reset/update gates
    ``(r, z)``; the trailing ``hidden_features`` columns produce the
    candidate term.

    Update rule:
    ``rz = sigmoid(rz_x + rz_h)``,
    ``r, z = split(rz, 2)``,
    ``n = tanh(n_x + r * n_h)``,
    ``h' = (1 - z) * n + z * h`` — so ``z = 1`` retains the previous
    hidden state.
    """

    W_x: Parameter
    W_h: Parameter
    b: Parameter

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        *,
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        h_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the GRU cell parameters.

        Args:
            in_features: Trailing input feature count.
            hidden_features: Hidden state size.
            use_bias: When ``True`` (default), allocate the bias of
                shape ``(3 * hidden_features,)``.
            rngs: PRNG source for parameter initialization.
            w_init: Input-side weight initializer; defaults to
                Kaiming-uniform with ``"linear"`` gain.
            h_init: Hidden-side weight initializer; defaults to
                :func:`~spectrax.init.orthogonal`.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype; defaults to ``float32``.
        """
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        h_init = h_init or orthogonal()
        self.W_x = Parameter(
            w_init(resolved.parameters, (in_features, 3 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        self.W_h = Parameter(
            h_init(resolved.parameters, (hidden_features, 3 * hidden_features), dt),
            axis_names=("in", "out"),
        )
        if use_bias:
            b_init = b_init or zeros
            self.b = Parameter(b_init(resolved.parameters, (3 * hidden_features,), dt), axis_names=("out",))

    def initial_carry(self, batch_shape: Sequence[int], *, dtype: DType | None = None) -> Array:
        """Allocate a zero hidden state.

        Args:
            batch_shape: Leading dimensions to prepend to the
                hidden axis.
            dtype: Optional dtype override; defaults to ``W_x``'s
                dtype.

        Returns:
            Zero array of shape ``(*batch_shape, hidden_features)``.
        """
        dt = dtype or self.W_x.dtype
        return jnp.zeros((*tuple(batch_shape), self.hidden_features), dtype=dt)

    def forward(self, carry: ArrayLike, x: ArrayLike) -> tuple[Array, Array]:
        """Advance the GRU by one step.

        Args:
            carry: Previous hidden state.
            x: Per-step input.

        Returns:
            ``(h_new, h_new)`` — the per-step output equals the new
            hidden state.
        """
        h = jnp.asarray(carry)
        x_part = jnp.asarray(x) @ self.W_x.value
        h_part = h @ self.W_h.value
        if self.use_bias:
            x_part = x_part + self.b.value
        rz_x, n_x = x_part[..., : 2 * self.hidden_features], x_part[..., 2 * self.hidden_features :]
        rz_h, n_h = h_part[..., : 2 * self.hidden_features], h_part[..., 2 * self.hidden_features :]
        rz = jax.nn.sigmoid(rz_x + rz_h)
        r, z = jnp.split(rz, 2, axis=-1)
        n = jnp.tanh(n_x + r * n_h)
        h_new = (1.0 - z) * n + z * h
        return h_new, h_new


class ConvLSTMCell(RNNCellBase):
    """2-D convolutional LSTM (Shi et al., 2015).

    Replaces the dense gate matmuls of an LSTM with a single 2-D
    convolution over the channel-concatenation ``[x, h]``. Inputs
    are channels-last ``(N, H, W, C_in)``; the carry is a pair of
    feature maps ``(h, c)`` shaped ``(N, H, W, C_out)`` so the
    spatial dimensions are preserved across time. The kernel has
    shape ``(*kernel_size, C_in + C_out, 4 * C_out)`` and the bias
    has shape ``(4 * C_out,)``.
    """

    weight: Parameter
    bias: Parameter

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
        *,
        padding: str = "SAME",
        use_bias: bool = True,
        rngs: Rngs | int | None = None,
        w_init: Initializer | None = None,
        b_init: Initializer | None = None,
        dtype: DType | None = None,
    ) -> None:
        """Allocate the convolutional kernel and bias.

        Args:
            in_channels: Trailing input channel count.
            out_channels: Output / hidden channel count.
            kernel_size: Convolution kernel size. ``int`` selects a
                square kernel; otherwise a length-2 sequence.
            padding: Padding mode for the gate convolution; default
                ``"SAME"`` so the carry retains its spatial shape.
            use_bias: When ``True`` (default), allocate the
                ``(4 * out_channels,)`` bias.
            rngs: PRNG source for parameter initialization.
            w_init: Kernel initializer; defaults to Kaiming-uniform
                with ``"linear"`` gain.
            b_init: Bias initializer; defaults to
                :func:`~spectrax.init.zeros`.
            dtype: Storage dtype; defaults to ``float32``.

        Raises:
            ValueError: If ``kernel_size`` is a sequence of length
                other than 2.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        ks = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        if len(ks) != 2:
            raise ValueError("ConvLSTMCell kernel_size must be int or length-2 sequence")
        self.kernel_size = ks
        self.padding = padding
        self.use_bias = use_bias
        resolved = resolve_rngs(rngs)
        dt = dtype or jnp.float32
        w_init = w_init or kaiming_uniform("linear")
        kshape = (*ks, in_channels + out_channels, 4 * out_channels)
        self.weight = Parameter(w_init(resolved.parameters, kshape, dt), axis_names=("kh", "kw", "in", "out"))
        if use_bias:
            b_init = b_init or zeros
            self.bias = Parameter(b_init(resolved.parameters, (4 * out_channels,), dt), axis_names=("out",))

    def initial_carry(
        self,
        batch_shape: Sequence[int],
        *,
        dtype: DType | None = None,
    ) -> tuple[Array, Array]:
        """Allocate zero-valued ``(h, c)`` feature maps.

        Args:
            batch_shape: Must be exactly ``(N, H, W)`` —
                channel dim is added internally.
            dtype: Optional dtype override; defaults to the kernel
                dtype.

        Returns:
            ``(h0, c0)`` — both zero arrays of shape
            ``(N, H, W, out_channels)``.

        Raises:
            ValueError: If ``batch_shape`` does not have length 3.
        """
        if len(batch_shape) != 3:
            raise ValueError("ConvLSTMCell batch_shape must be (N, H, W)")
        dt = dtype or self.weight.dtype
        shape = (*tuple(batch_shape), self.out_channels)
        return jnp.zeros(shape, dtype=dt), jnp.zeros(shape, dtype=dt)

    def forward(self, carry: tuple[Array, Array], x: ArrayLike) -> tuple[tuple[Array, Array], Array]:
        """Advance the ConvLSTM by one step.

        Concatenates ``x`` and the previous hidden state along the
        channel axis, runs a single 2-D convolution to compute the
        four LSTM gate pre-activations, then applies the standard
        LSTM update.

        Args:
            carry: ``(h, c)`` tuple, both shaped ``(N, H, W, C_out)``.
            x: ``(N, H, W, C_in)`` channels-last input.

        Returns:
            ``((h_new, c_new), h_new)`` with shapes preserved.
        """
        h, c = carry
        xh = jnp.concatenate([jnp.asarray(x), h], axis=-1)
        z = F_conv(
            xh,
            self.weight.value,
            self.bias.value if self.use_bias else None,
            padding=self.padding,
        )
        i, f, g, o = jnp.split(z, 4, axis=-1)
        i = jax.nn.sigmoid(i)
        f = jax.nn.sigmoid(f)
        g = jnp.tanh(g)
        o = jax.nn.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * jnp.tanh(c_new)
        return (h_new, c_new), h_new


class RNN(Module):
    """Scan an :class:`RNNCellBase` across a time axis.

    A simple wrapper that calls :func:`jax.lax.scan` over a cell,
    preparing the input and unwinding the output to match the
    user-facing layout.

    Time-axis layout: by default sequence-second
    (``time_major=False``, axis ``1`` of the input). The wrapper
    transposes axes 0 and 1 around the scan to keep the user-facing
    shape consistent. Set ``time_major=True`` to skip the transpose.

    Reversal: when ``reverse=True``, the sequence is processed right
    to left (the sliced input is reversed before the scan and the
    output is reversed back afterwards). The instance-level
    :attr:`reverse` is a default — :meth:`forward` accepts a
    per-call override.

    Carry return: when ``return_carry=True``, :meth:`forward`
    returns ``(ys, final_carry)``; the default is ``ys`` only.
    """

    cell: RNNCellBase

    def __init__(
        self,
        cell: RNNCellBase,
        *,
        time_major: bool = False,
        reverse: bool = False,
        return_carry: bool = False,
    ) -> None:
        """Wrap a recurrent cell with scan-based sequence iteration.

        Args:
            cell: The underlying cell whose :meth:`forward` is
                applied at every time step. Must be an
                :class:`RNNCellBase` instance.
            time_major: When ``True`` the sequence axis is axis 0
                (``(T, N, ...)``); when ``False`` (default) it is
                axis 1 (``(N, T, ...)``).
            reverse: Default direction of the scan; per-call
                overridable in :meth:`forward`. ``True`` processes
                the sequence right-to-left.
            return_carry: Default value for the per-call
                ``return_carry`` argument of :meth:`forward`.

        Raises:
            TypeError: If ``cell`` is not an :class:`RNNCellBase`.
        """
        super().__init__()
        if not isinstance(cell, RNNCellBase):
            raise TypeError("RNN requires an RNNCellBase instance as cell")
        self.cell = cell
        self.time_major = time_major
        self.reverse = reverse
        self.return_carry = return_carry

    def _batch_shape(self, xs: Array) -> tuple[int, ...]:
        """Compute the batch shape used to size the cell's initial carry.

        Strips out the time axis (axis 0 or 1 depending on
        :attr:`time_major`) and the trailing feature/channel axis,
        leaving the dimensions that participate in the cell's
        per-step batch.

        Args:
            xs: The full sequence input.

        Returns:
            A tuple of ints to feed
            :meth:`RNNCellBase.initial_carry` as ``batch_shape``.
        """
        if isinstance(self.cell, ConvLSTMCell):
            time_axis = 0 if self.time_major else 1
            shape = list(xs.shape)
            shape.pop(time_axis)
            return tuple(shape[:-1])
        time_axis = 0 if self.time_major else 1
        shape = list(xs.shape)
        shape.pop(time_axis)
        return tuple(shape[:-1])

    def forward(
        self,
        xs: ArrayLike,
        *,
        initial_carry: Carry | None = None,
        reverse: bool | None = None,
        return_carry: bool | None = None,
    ) -> Array | tuple[Array, Carry]:
        """Scan the cell across every step of the time axis.

        Steps performed:

        1. Move the time axis to position 0 (no-op when
           ``time_major=True``).
        2. Optionally reverse along the time axis.
        3. Build (or accept) the initial carry.
        4. Run :func:`jax.lax.scan` calling
           :meth:`RNNCellBase.forward` per step.
        5. Reverse the output back when needed and restore the
           original axis ordering.

        Args:
            xs: Full sequence input. Time axis is determined by
                :attr:`time_major` (axis 0 if ``True``, axis 1
                otherwise).
            initial_carry: Optional starting carry. When ``None``,
                the cell's :meth:`initial_carry` is used with
                ``dtype=xs.dtype``.
            reverse: Per-call override of :attr:`reverse`. Pass
                ``None`` (default) to inherit the instance value.
            return_carry: Per-call override of
                :attr:`return_carry`. Pass ``None`` (default) to
                inherit the instance value.

        Returns:
            Either ``ys`` alone, or ``(ys, final_carry)`` when
            ``return_carry`` resolves to ``True``. ``ys`` matches the
            time-axis layout of ``xs``.
        """
        xs = jnp.asarray(xs)
        time_axis = 0 if self.time_major else 1
        rev = self.reverse if reverse is None else reverse
        ret_carry = self.return_carry if return_carry is None else return_carry
        if not self.time_major:
            xs_t = jnp.swapaxes(xs, 0, 1)
        else:
            xs_t = xs
        if rev:
            xs_t = xs_t[::-1]

        if initial_carry is None:
            batch_shape = self._batch_shape(xs)
            carry = self.cell.initial_carry(batch_shape, dtype=xs.dtype)
        else:
            carry = initial_carry

        cell = self.cell

        def step(c: Carry, x: Array) -> tuple[Carry, Array]:
            """Single :func:`jax.lax.scan` step.

            Delegates to the captured cell's :meth:`forward` so the
            scan signature matches what :func:`jax.lax.scan` expects.

            Args:
                c: Recurrent carry from the previous step.
                x: Per-step input slice.

            Returns:
                ``(new_carry, y)`` from the cell.
            """
            return cell.forward(c, x)

        final_carry, ys = jax.lax.scan(step, carry, xs_t)
        if rev:
            ys = ys[::-1]
        if not self.time_major:
            ys = jnp.swapaxes(ys, 0, 1)
        _ = time_axis
        if ret_carry:
            return ys, final_carry
        return ys


class Bidirectional(Module):
    """Run a forward and a reverse RNN and combine their outputs.

    Holds two :class:`RNN` instances (with their own underlying
    cells) and applies them to the same input — the second is forced
    to ``reverse=True`` at call time. The two output sequences are
    combined per-element according to :attr:`merge_mode`:

    * ``"concat"`` (default) — concatenate along the trailing
      feature axis.
    * ``"sum"`` — element-wise sum.
    * ``"mul"`` — element-wise product.
    * ``"ave"`` — element-wise mean (``0.5 * (ys_f + ys_b)``).
    """

    forward_rnn: RNN
    backward_rnn: RNN

    def __init__(
        self,
        forward_rnn: RNN,
        backward_rnn: RNN,
        *,
        merge_mode: str = "concat",
    ) -> None:
        """Pair two :class:`RNN` instances and pick a merge mode.

        Args:
            forward_rnn: :class:`RNN` to run left-to-right.
            backward_rnn: :class:`RNN` to run right-to-left. Note
                that this layer always passes ``reverse=True`` to
                this RNN's :meth:`forward` regardless of its
                instance default.
            merge_mode: One of ``"concat"``, ``"sum"``, ``"mul"``,
                ``"ave"`` — see the class docstring.

        Raises:
            TypeError: If either argument is not an :class:`RNN`.
            ValueError: If ``merge_mode`` is not recognised.
        """
        super().__init__()
        if not isinstance(forward_rnn, RNN) or not isinstance(backward_rnn, RNN):
            raise TypeError("Bidirectional requires two RNN instances")
        if merge_mode not in {"concat", "sum", "mul", "ave"}:
            raise ValueError(f"Unknown merge_mode {merge_mode!r}")
        self.forward_rnn = forward_rnn
        self.backward_rnn = backward_rnn
        self.merge_mode = merge_mode

    def forward(
        self,
        xs: ArrayLike,
        *,
        initial_carry: tuple[Carry, Carry] | None = None,
    ) -> Array:
        """Run both directions and merge the outputs.

        Args:
            xs: Input sequence; layout follows the inner
                :class:`RNN` instances' :attr:`time_major` setting.
            initial_carry: Optional ``(forward_carry, backward_carry)``
                pair. ``None`` (default) lets each inner RNN allocate
                its own zero-valued carry.

        Returns:
            Merged output. The trailing axis size is doubled when
            ``merge_mode="concat"`` and unchanged otherwise.
        """
        fwd_c = None if initial_carry is None else initial_carry[0]
        bwd_c = None if initial_carry is None else initial_carry[1]
        ys_f = self.forward_rnn.forward(xs, initial_carry=fwd_c, return_carry=False)
        ys_b = self.backward_rnn.forward(xs, initial_carry=bwd_c, reverse=True, return_carry=False)

        if self.merge_mode == "concat":
            return jnp.concatenate([ys_f, ys_b], axis=-1)
        if self.merge_mode == "sum":
            return cast(Array, ys_f + ys_b)
        if self.merge_mode == "mul":
            return cast(Array, ys_f * ys_b)
        return cast(Array, 0.5 * (ys_f + ys_b))
