# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Attention mask implementations for Splash (block-sparse) Attention.

This module provides lazy and dense mask implementations for controlling
attention patterns in Splash Attention kernels. Masks define which query-key
pairs can attend to each other.

Key mask types:
- FullMask: Allows all tokens to attend to all other tokens
- CausalMask: Autoregressive mask (query can only attend to earlier keys)
- LocalMask: Sliding window attention with configurable left/right context
- ChunkedCausalMask: Causal within fixed-size chunks
- MultiHeadMask: Per-head mask specification
- NumpyMask: Dense mask backed by numpy array

Lazy masks (_ComputableMask subclasses) compute mask values on-the-fly
during kernel execution, avoiding memory overhead of materializing the
full [q_seq_len, kv_seq_len] boolean matrix.

Example:
    >>> from ejkernel.kernels._pallas.tpu.blocksparse_attention._masks import (
    ...     CausalMask, LocalMask, MultiHeadMask
    ... )
    >>> # Causal mask for autoregressive attention
    >>> causal = CausalMask((2048, 2048))
    >>>
    >>> # Sliding window attention
    >>> local = LocalMask((2048, 2048), window_size=(128, 0), offset=0)
    >>>
    >>> # Combine with causal
    >>> combined = causal & local
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
from beartype.typing import Callable


class Mask:
    """Base class for block-sparse attention masks.

    Provides the interface for all mask types used in Splash Attention.
    Masks define which query-key pairs can attend to each other. Subclasses
    implement specific masking patterns (causal, local, etc.).

    Masks support composition via bitwise operators:
        - ``mask1 & mask2``: Logical AND (both masks must allow attention)
        - ``mask1 | mask2``: Logical OR (either mask allows attention)
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the mask (q_seq_len, kv_seq_len)."""
        raise NotImplementedError

    def __getitem__(self, idx) -> np.ndarray:
        """Return mask values for the given slice indices."""
        raise NotImplementedError

    def __bool__(self) -> bool:
        raise NotImplementedError(
            "Conversion to bool is unsupported. Could be caused by using logical instead of bitwise operations on masks."
        )

    def __or__(self, other: Mask) -> Mask:
        """Combine masks with logical OR (either mask allows attention)."""
        if self.shape != other.shape:
            raise ValueError(f"Invalid shape for other: {other.shape}, expected: {self.shape}")
        return LogicalOr(self, other)

    def __and__(self, other: Mask) -> Mask:
        """Combine masks with logical AND (both masks must allow attention)."""
        if self.shape != other.shape:
            raise ValueError(f"Invalid shape for other: {other.shape}, expected: {self.shape}")
        return LogicalAnd(self, other)


def make_causal_mask(shape: tuple[int, int], offset: int = 0) -> np.ndarray:
    """Makes a causal attention mask.

    Args:
      shape: Shape of the 2-dim mask: (q_seq_len, kv_seq_len).
      offset: Offset of q start wrt kv. A positive offset shifts the bottom
        triangle upward, a negative one shifts it downward. A negative offset
        makes the first 'offset' rows of the attention matrix all 0s which leads
        to undefined softmax.

    Returns:
      The causal mask.
    """
    q_seq_len, kv_seq_len = shape
    q_idx = np.arange(q_seq_len, dtype=np.int32)
    kv_idx = np.arange(kv_seq_len, dtype=np.int32)
    return (q_idx[:, None] + offset >= kv_idx[None, :]).astype(np.bool_)


def make_local_attention_mask(
    shape: tuple[int, int],
    window_size: tuple[int | None, int | None],
    *,
    offset: int = 0,
) -> np.ndarray:
    """Create a local (sliding window) attention mask.

    Creates a mask where each query position can only attend to key positions
    within a specified window around it. Useful for efficient attention over
    long sequences where full attention is too expensive.

    Args:
        shape: Shape of the mask (q_seq_len, kv_seq_len).
        window_size: Tuple of (left_size, right_size) defining the window.
            None means no limit in that direction.
        offset: Offset of query start relative to key sequence.

    Returns:
        Boolean numpy array where True means attention is allowed.
    """
    q_seq_len, kv_seq_len = shape
    q_idx = np.arange(q_seq_len, dtype=np.int32)
    kv_idx = np.arange(kv_seq_len, dtype=np.int32)
    mask = np.ones((q_seq_len, kv_seq_len), dtype=np.bool_)
    left, right = window_size
    if left is not None:
        mask = mask & (q_idx[:, None] - left + offset <= kv_idx[None, :])
    if right is not None:
        mask = mask & (q_idx[:, None] + right + offset >= kv_idx[None, :])
    return mask.astype(np.bool_)


def make_chunk_attention_mask(shape: tuple[int, int], chunk_size: int) -> np.ndarray:
    """Makes a chunked causal attention mask.

    Args:
      shape: The desired shape of the mask (q_seq_len, kv_seq_len).
      chunk_size: The size of the attention chunks.

    Returns:
      A boolean mask of shape `mask_shape` where True indicates attention is
      allowed according to chunked causal rules, and False otherwise.

    Raises:
      ValueError: If chunk_window_size is None or not positive.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    q_seq_len, kv_seq_len = shape
    q_idx = np.arange(q_seq_len, dtype=np.int32)
    kv_idx = np.arange(kv_seq_len, dtype=np.int32)

    same_chunk = (q_idx[:, None] // chunk_size) == (kv_idx[None, :] // chunk_size)
    mask = same_chunk & (q_idx[:, None] >= kv_idx[None, :])
    return mask


def make_random_mask(shape: tuple[int, int], sparsity: float, seed: int) -> np.ndarray:
    """Create a random attention mask with specified sparsity.

    Useful for testing and experimenting with sparse attention patterns.
    Each position is independently sampled.

    Args:
        shape: Shape of the mask (q_seq_len, kv_seq_len).
        sparsity: Fraction of positions to mask (0.0 = no masking, 1.0 = all masked).
        seed: Random seed for reproducibility.

    Returns:
        Boolean numpy array with approximately (1 - sparsity) fraction of True values.
    """
    np.random.seed(seed)  # noqa: NPY002
    return np.random.binomial(n=1, p=1.0 - sparsity, size=shape).astype(np.bool_)  # noqa: NPY002


@dataclasses.dataclass
class LogicalOr(Mask):
    """Composite mask that combines two masks with logical OR.

    A position is unmasked if either the left or right mask allows it.
    Created automatically when using the ``|`` operator on two Mask instances.

    Attributes:
        left: First mask operand.
        right: Second mask operand.
    """

    left: Mask
    right: Mask

    def __init__(self, left: Mask, right: Mask):
        if left.shape != right.shape:
            raise ValueError("Masks must have the same shape")
        self.left = left
        self.right = right

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the combined mask."""
        return self.left.shape

    def __getitem__(self, idx) -> np.ndarray:
        """Return element-wise OR of both masks at the given indices."""
        return self.left[idx] | self.right[idx]

    def __hash__(self):
        return hash((type(self), self.left, self.right))


@dataclasses.dataclass
class LogicalAnd(Mask):
    """Composite mask that combines two masks with logical AND.

    A position is unmasked only if both the left and right masks allow it.
    Created automatically when using the ``&`` operator on two Mask instances.

    Attributes:
        left: First mask operand.
        right: Second mask operand.
    """

    left: Mask
    right: Mask

    def __init__(self, left: Mask, right: Mask):
        if left.shape != right.shape:
            raise ValueError("Masks must have the same shape")
        self.left = left
        self.right = right

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the combined mask."""
        return self.left.shape

    def __getitem__(self, idx) -> np.ndarray:
        """Return element-wise AND of both masks at the given indices."""
        return self.left[idx] & self.right[idx]

    def __hash__(self):
        return hash((type(self), self.left, self.right))


@dataclasses.dataclass
class MultiHeadMask(Mask):
    """Per-head mask wrapper that combines multiple single-head masks.

    Stores one :class:`Mask` per attention head, allowing each head to have
    a distinct sparsity pattern.  All per-head masks must share the same
    2-D shape ``(q_seq_len, kv_seq_len)``.  Nesting ``MultiHeadMask`` inside
    another ``MultiHeadMask`` is not supported.

    Shape: ``(num_heads, q_seq_len, kv_seq_len)``.

    Attributes:
        masks: Sequence of per-head :class:`Mask` objects.
    """

    masks: Sequence[Mask]

    def __post_init__(self):
        if not self.masks:
            raise ValueError("Unsupported empty tuple of masks")

        shape = self.masks[0].shape
        for mask in self.masks[1:]:
            if shape != mask.shape:
                raise ValueError(f"Unexpected mask shape, got: {mask.shape}, expected: {shape}")

        if not all(isinstance(mask, Mask) for mask in self.masks):
            raise ValueError("masks should be of type Mask")

        if any(isinstance(mask, MultiHeadMask) for mask in self.masks):
            raise ValueError("Nesting MultiHeadMasks is not supported")

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self.masks), *self.masks[0].shape)

    def __getitem__(self, idx) -> np.ndarray:
        if len(idx) != 3:
            raise NotImplementedError(f"Unsupported slice: {idx}")

        head_slice = idx[0]
        if isinstance(head_slice, int):
            assert head_slice >= 0 and head_slice <= len(self.masks)
            return self.masks[head_slice][idx[1:]]
        else:
            slice_masks = [mask[idx[1:]] for mask in self.masks[head_slice]]
            return np.stack(slice_masks)

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.masks == other.masks

    def __hash__(self):
        return hash((type(self), *tuple(hash(mask) for mask in self.masks)))


class _ComputableMask(Mask):
    """Base class for lazily-evaluated masks computed by a callable inside the kernel.

    Designed for use with Splash Attention.  Subclasses supply a
    ``mask_function(q_ids, kv_ids) -> bool[q, kv]`` that the Splash kernel
    calls on-the-fly to avoid materialising the full
    ``(q_seq_len, kv_seq_len)`` boolean matrix, which can be prohibitively
    large for long sequences.

    Attributes:
        _shape: 2-D mask shape ``(q_seq_len, kv_seq_len)``.
        q_sequence: ``int32[q_seq_len]`` index array.  Reused across
            ``__getitem__`` calls to avoid repeated allocation at trace time.
        mask_function: Callable ``(q_ids, kv_ids) -> bool[q, kv]`` supplied
            by the concrete subclass.  Used by the Splash Attention kernel to
            compute mask values without loading pre-stored arrays.
    """

    _shape: tuple[int, int]
    q_sequence: np.ndarray
    mask_function: Callable[..., Any]

    def __init__(
        self,
        shape: tuple[int, int],
        mask_function: Callable[..., Any],
        shard_count: int = 1,
    ):
        self._shape = shape
        self.mask_function = mask_function
        q_seq_len = self.shape[0]

        if q_seq_len % (shard_count * shard_count) != 0:
            raise ValueError(
                f"Shard count squared ({shard_count * shard_count}) must divide Q seq_len ({self.shape[0]}) evenly."
            )

        self.q_sequence = np.arange(q_seq_len, dtype=np.int32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, idx) -> np.ndarray:
        if len(idx) != 2:
            raise NotImplementedError(f"Unsupported slice: {idx}")

        q_slice, kv_slice = idx
        if not isinstance(q_slice, slice) or not isinstance(kv_slice, slice):
            raise NotImplementedError(f"Unsupported slice: {idx}")

        q_slice = _fill_slice(q_slice, self.shape[0])
        kv_slice = _fill_slice(kv_slice, self.shape[1])

        rows = self.q_sequence[q_slice]
        cols = np.arange(kv_slice.start, kv_slice.stop)

        return self.mask_function(rows[:, None], cols[None, :])

    def __eq__(self, other: object):
        raise NotImplementedError()

    def __hash__(self):
        raise NotImplementedError()


class CausalMask(_ComputableMask):
    """Lazy lower-triangular causal mask for autoregressive attention.

    Each query position ``q`` can attend to key positions ``k`` where
    ``k + offset <= q`` (i.e. the lower triangle shifted by ``offset``).

    The mask is not materialised: the Splash Attention kernel calls
    ``mask_function`` on-the-fly to compute mask values for each block.

    Attributes:
        offset: Shift applied to the diagonal.  A positive offset extends the
            triangle upward (earlier keys become visible); a negative offset
            shifts it downward, making the first ``|offset|`` query rows
            all-zero, which produces undefined softmax.  Defaults to 0.
    """

    offset: int

    def __init__(
        self,
        shape: tuple[int, int],
        offset: int = 0,
        shard_count: int = 1,
    ):
        self.offset = offset

        def causal_mask_function(q_ids, kv_ids):
            if self.offset == 0:
                return q_ids >= kv_ids
            else:
                return q_ids + self.offset >= kv_ids

        mask_function = causal_mask_function

        super().__init__(
            shape=shape,
            mask_function=mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.shape == other.shape
            and self.offset == other.offset
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.offset,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


class ChunkedCausalMask(_ComputableMask):
    """Lazy chunked causal mask for block-diagonal attention (Llama4-style).

    Attention is causal within non-overlapping fixed-size chunks.  Tokens in
    chunk ``[i*K, (i+1)*K)`` attend only to earlier tokens within the same
    chunk; no cross-chunk attention is allowed.

    Formally: ``mask[q, k] = (q // chunk_size == k // chunk_size) and (q >= k)``.

    Llama4 models use interleaved chunk attention (this mask) alternating with
    global attention layers.

    Attributes:
        chunk_size: Number of tokens per attention chunk.  Must be positive.
    """

    chunk_size: int

    def __init__(
        self,
        shape: tuple[int, int],
        chunk_size: int,
        shard_count: int = 1,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

        def chunked_causal_mask_function(q_ids, kv_ids):
            """Computes the mask logic for the given slice indices."""

            same_chunk = (q_ids // self.chunk_size) == (kv_ids // self.chunk_size)

            causal = q_ids >= kv_ids

            return same_chunk & causal

        super().__init__(
            shape=shape,
            mask_function=chunked_causal_mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return (
            self.shape == other.shape
            and self.chunk_size == other.chunk_size
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.chunk_size,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


class LocalMask(_ComputableMask):
    """Lazy sliding-window local attention mask.

    Each query position ``q`` attends only to key positions ``k`` satisfying:
        ``q - left_size + offset <= k <= q + right_size + offset``

    Either side may be ``None`` to indicate no limit in that direction.  Combine
    with :class:`CausalMask` (``mask & causal``) for causal sliding-window
    attention.

    Attributes:
        window_size: ``(left_size, right_size)`` where each element is an
            ``int`` (finite window) or ``None`` (unbounded in that direction).
        offset: Shift applied to ``q`` before window comparison.  Positive
            values move the window toward earlier keys.  A large negative
            offset can create all-zero rows, leading to undefined softmax.
    """

    window_size: tuple[int | None, int | None]
    offset: int

    def __init__(
        self,
        shape: tuple[int, int],
        window_size: tuple[int | None, int | None],
        offset: int,
        shard_count: int = 1,
    ):
        self.window_size = window_size
        self.offset = offset

        def local_mask_function(q_ids, kv_ids):
            """Computes the local attention mask for the given slice indices."""
            left_size, right_size = self.window_size

            assert q_ids.ndim == 2
            assert kv_ids.ndim == 2

            if left_size is None and right_size is None:
                return np.ones((q_ids.shape[0], kv_ids.shape[1]), dtype=np.bool_)

            if offset != 0:
                shifted_q_ids = q_ids + self.offset
            else:
                shifted_q_ids = q_ids

            mask = None
            if left_size is not None:
                mask = shifted_q_ids - left_size <= kv_ids
            if right_size is not None:
                if mask is None:
                    mask = shifted_q_ids + right_size >= kv_ids
                else:
                    mask &= shifted_q_ids + right_size >= kv_ids
            return mask

        super().__init__(
            shape=shape,
            mask_function=local_mask_function,
            shard_count=shard_count,
        )

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return False

        return (
            self.shape == other.shape
            and self.window_size == other.window_size
            and self.offset == other.offset
            and np.array_equal(self.q_sequence, other.q_sequence)
        )

    def __hash__(self):
        return hash(
            (
                type(self),
                self.shape,
                self.window_size,
                self.offset,
                self.q_sequence.tobytes() if self.q_sequence is not None else None,
            )
        )


@dataclasses.dataclass
class NumpyMask(Mask):
    """Dense attention mask backed by a numpy boolean array.

    Use when the mask pattern cannot be expressed as a computable function.
    The full ``[q_seq_len, kv_seq_len]`` boolean array is stored in memory
    and sliced on demand.  For long sequences prefer the lazy ``_ComputableMask``
    subclasses (``CausalMask``, ``LocalMask``, etc.) to avoid materialising
    the full matrix.

    Attributes:
        array: 2-D boolean numpy array of shape ``(q_seq_len, kv_seq_len)``.
            Must have dtype ``np.bool_``.
    """

    array: np.ndarray

    def __post_init__(self):
        if self.array.ndim != 2:
            raise ValueError("Expected a 2-dim array")

        if self.array.dtype != np.bool_:
            raise ValueError("Mask must be a boolean array")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.array.shape

    def __getitem__(self, idx) -> np.ndarray:
        return self.array[idx]

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return np.array_equal(self.array, other.array, equal_nan=True)

    def __hash__(self):
        return hash((type(self), self.array.tobytes()))


def _fill_slice(inp_slice: slice, size: int) -> slice:
    """Normalise a slice by filling in ``None`` start/stop with 0 / ``size``.

    Args:
        inp_slice: Input slice; must have ``step`` of ``None`` or ``1``.
        size: Total length of the dimension being sliced.

    Returns:
        An equivalent slice with explicit integer ``start`` and ``stop``.
    """
    assert inp_slice.step is None or inp_slice.step == 1
    start = 0 if inp_slice.start is None else inp_slice.start
    stop = size if inp_slice.stop is None else inp_slice.stop
    assert start >= 0
    assert stop <= size
    return slice(start, stop, None)


@dataclasses.dataclass(frozen=True)
class FullMask(Mask):
    """Lazy dense (all-ones) mask — every token can attend every other token.

    Equivalent to no masking.  Stored lazily: ``__getitem__`` materialises
    only the requested slice as a numpy ones array without allocating the
    full ``[q_seq_len, kv_seq_len]`` matrix.

    Attributes:
        _shape: Tuple ``(q_seq_len, kv_seq_len)`` defining the mask dimensions.
    """

    _shape: tuple[int, int]

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            raise ValueError(f"Unsupported shape type: {type(self.shape)}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def __getitem__(self, idx) -> np.ndarray:
        if len(idx) != 2:
            raise NotImplementedError(f"Unsupported slice: {idx}")
        i, j = idx
        if not isinstance(i, slice) or not isinstance(j, slice):
            raise NotImplementedError(f"Unsupported slice: {idx}")
        i = _fill_slice(i, self.shape[0])
        j = _fill_slice(j, self.shape[1])
        return np.ones((i.stop - i.start, j.stop - j.start), dtype=np.bool_)

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented

        return self.shape == other.shape

    def __hash__(self):
        return hash((type(self), self.shape))
