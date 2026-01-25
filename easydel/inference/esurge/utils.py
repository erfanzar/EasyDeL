# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Utility functions and classes for the eSurge engine.

Provides helper classes and functions for working with immutable lists,
array type checking, and other common operations used throughout the
eSurge inference engine.

Classes:
    ConstantList: Immutable list wrapper that prevents modifications.

Functions:
    is_list_of: Type guard for checking list element types.
    chunk_list: Split a list into chunks of a specified size.
    cdiv: Compute ceiling division.
    next_power_of_2: Find the next power of 2 (inclusive).
    prev_power_of_2: Find the previous power of 2 (inclusive).
    round_up: Round up to the nearest multiple.
    round_down: Round down to the nearest multiple.
    get_dtype_size: Get the size of a data type in bytes.
    truncate_tokens: Truncate a token list to a target length.
    model_uses_mrope: Check if a model uses multi-dimensional RoPE.

Example:
    >>> from easydel.inference.esurge.utils import ConstantList
    >>>
    >>> # Create immutable list
    >>> const_list = ConstantList([1, 2, 3])
    >>> print(const_list[0])  # Works
    >>> const_list.append(4)  # Raises Exception
"""

from collections.abc import Mapping, Sequence
from typing import Any, Generic, Literal, TypeVar, overload

from jax import numpy as jnp
from typing_extensions import TypeIs

T = TypeVar("T")


class ConstantList(Generic[T], Sequence):
    """Immutable list wrapper that prevents modifications.

    Provides read-only access to a list while preventing any
    modification operations. Useful for protecting data structures
    that should not be changed after creation.

    Attributes:
        _x: The underlying list being wrapped.

    Args:
        x: The list to wrap and make immutable.

    Example:
        >>> const_list = ConstantList([1, 2, 3])
        >>> print(const_list[0])  # 1
        >>> print(len(const_list))  # 3
        >>> const_list.append(4)  # Raises Exception
    """

    def __init__(self, x: list[T]) -> None:
        """Initialize with a list to make immutable.

        Args:
            x: List to wrap.
        """
        self._x = x

    def append(self, item):
        """Prevent appending to the list.

        Args:
            item: Item that would be appended.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot append to a constant list")

    def extend(self, item):
        """Prevent extending the list.

        Args:
            item: Items that would be added.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot extend a constant list")

    def insert(self, item):
        """Prevent inserting into the list.

        Args:
            item: Item that would be inserted.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot insert into a constant list")

    def pop(self, item):
        """Prevent popping from the list.

        Args:
            item: Index to pop from.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot pop from a constant list")

    def remove(self, item):
        """Prevent removing from the list.

        Args:
            item: Item that would be removed.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot remove from a constant list")

    def clear(self):
        """Prevent clearing the list.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot clear a constant list")

    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        """Find the index of an item in the list.

        Args:
            item: Item to find.
            start: Starting index for the search.
            stop: Ending index for the search (exclusive).

        Returns:
            Index of the first occurrence of the item.

        Raises:
            ValueError: If item is not found.
        """
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        """Get an item or slice from the list.

        Args:
            item: Integer index or slice object.

        Returns:
            Single item if integer index, list of items if slice.
        """
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...

    @overload
    def __setitem__(self, s: slice, value: T, /): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        """Prevent setting items in the list.

        Args:
            item: Index or slice to set.
            value: Value that would be set.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot set item in a constant list")

    def __delitem__(self, item):
        """Prevent deleting items from the list.

        Args:
            item: Index to delete.

        Raises:
            Exception: Always raised as list is immutable.
        """
        raise Exception("Cannot delete item from a constant list")

    def __iter__(self):
        """Return an iterator over the list.

        Returns:
            Iterator over the underlying list elements.
        """
        return iter(self._x)

    def __contains__(self, item):
        """Check if an item is in the list.

        Args:
            item: Item to check for.

        Returns:
            True if item is in the list, False otherwise.
        """
        return item in self._x

    def __len__(self):
        """Return the length of the list.

        Returns:
            Number of elements in the list.
        """
        return len(self._x)

    def __repr__(self):
        """Return a string representation of the ConstantList.

        Returns:
            String representation showing the wrapped list.
        """
        return f"ConstantList({self._x})"


def is_list_of(
    value: object,
    typ: type[T] | tuple[type[T], ...],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    """Type guard for checking if a value is a list of a specific type.

    Validates that a value is a list and that its elements are of the
    specified type(s).

    Args:
        value: The value to check.
        typ: Type or tuple of types that list elements should match.
        check: Checking strategy - "first" checks only the first element
            (faster), "all" checks every element (thorough). Defaults to "first".

    Returns:
        True if value is a list with elements of the specified type(s),
        False otherwise. Empty lists return True for "first" mode.

    Example:
        >>> is_list_of([1, 2, 3], int)
        True
        >>> is_list_of([1, "a", 3], int, check="all")
        False
        >>> is_list_of([], str)
        True
    """
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)


def chunk_list(lst: list[T], chunk_size: int):
    """Yield successive chunks of a specified size from a list.

    Splits a list into chunks of the specified size. The last chunk
    may be smaller if the list length is not evenly divisible.

    Args:
        lst: The list to split into chunks.
        chunk_size: Maximum size of each chunk. Must be positive.

    Yields:
        Successive sublists of up to chunk_size elements.

    Example:
        >>> list(chunk_list([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
        >>> list(chunk_list([1, 2, 3], 5))
        [[1, 2, 3]]
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def cdiv(a: int, b: int) -> int:
    """Compute ceiling division.

    Divides a by b and rounds up to the nearest integer.
    Equivalent to math.ceil(a / b) but using integer arithmetic.

    Args:
        a: The dividend.
        b: The divisor. Must not be zero.

    Returns:
        The ceiling of a divided by b.

    Example:
        >>> cdiv(7, 3)
        3
        >>> cdiv(6, 3)
        2
        >>> cdiv(5, 3)
        2
    """
    return -(a // -b)


def next_power_of_2(n) -> int:
    """Find the next power of 2 greater than or equal to n.

    Returns the smallest power of 2 that is >= n. For n <= 1,
    returns 1.

    Args:
        n: The input number.

    Returns:
        The next power of 2 (inclusive). Returns 1 for n < 1.

    Example:
        >>> next_power_of_2(5)
        8
        >>> next_power_of_2(8)
        8
        >>> next_power_of_2(0)
        1
    """
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def prev_power_of_2(n: int) -> int:
    """Find the previous power of 2 less than or equal to n.

    Returns the largest power of 2 that is <= n. For n <= 0,
    returns 0.

    Args:
        n: The input number.

    Returns:
        The previous power of 2 (inclusive). Returns 0 for n <= 0.

    Example:
        >>> prev_power_of_2(5)
        4
        >>> prev_power_of_2(8)
        8
        >>> prev_power_of_2(0)
        0
    """
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def round_up(x: int, y: int) -> int:
    """Round x up to the nearest multiple of y.

    Args:
        x: The value to round.
        y: The multiple to round to. Must be positive.

    Returns:
        The smallest multiple of y that is >= x.

    Example:
        >>> round_up(7, 4)
        8
        >>> round_up(8, 4)
        8
        >>> round_up(9, 4)
        12
    """
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    """Round x down to the nearest multiple of y.

    Args:
        x: The value to round.
        y: The multiple to round to. Must be positive.

    Returns:
        The largest multiple of y that is <= x.

    Example:
        >>> round_down(7, 4)
        4
        >>> round_down(8, 4)
        8
        >>> round_down(9, 4)
        8
    """
    return (x // y) * y


def get_dtype_size(dtype: jnp.ndarray) -> int:
    """Get the size of a JAX/NumPy data type in bytes.

    Args:
        dtype: A JAX or NumPy dtype to measure.

    Returns:
        Size of the dtype in bytes.

    Example:
        >>> get_dtype_size(jnp.float32)
        4
        >>> get_dtype_size(jnp.int64)
        8
    """
    return jnp.finfo(dtype).bits // 8 if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).bits // 8


def truncate_tokens(tokens, target_len: int, mode: str = "left"):
    """Truncate a token list to a target length.

    Removes tokens from the list to fit within the target length using
    the specified truncation strategy.

    Args:
        tokens: List of tokens to truncate.
        target_len: Maximum allowed length after truncation.
        mode: Truncation strategy:
            - "left": Remove tokens from the beginning (keeps recent context).
            - "right": Remove tokens from the end (keeps initial context).
            - "middle": Remove tokens from the middle (keeps both ends).

    Returns:
        Tuple of (truncated_tokens, num_dropped) where truncated_tokens
        is the new token list and num_dropped is the count of removed tokens.

    Raises:
        ValueError: If mode is not one of "left", "right", or "middle".

    Example:
        >>> truncate_tokens([1, 2, 3, 4, 5], 3, "left")
        ([3, 4, 5], 2)
        >>> truncate_tokens([1, 2, 3, 4, 5], 3, "right")
        ([1, 2, 3], 2)
        >>> truncate_tokens([1, 2, 3, 4, 5], 3, "middle")
        ([1, 2, 5], 2)
    """
    n = len(tokens)
    if n <= target_len:
        return tokens, 0
    drop = n - target_len
    if mode == "left":
        return tokens[drop:], drop
    elif mode == "right":
        return tokens[:target_len], drop
    elif mode == "middle":
        keep_left = (target_len + 1) // 2
        keep_right = target_len - keep_left
        return tokens[:keep_left] + tokens[n - keep_right :], drop
    else:
        raise ValueError(f"Unknown truncate_mode: {mode}")


def _get_text_config(config: Any) -> Any:
    """Best-effort resolver for text configs on composite models.

    Attempts to extract the text configuration from a model config,
    handling various HuggingFace model configuration patterns.

    Args:
        config: A model configuration object that may contain a text_config.

    Returns:
        The text configuration if found, or the original config if not.
        Returns None if config is None.

    Note:
        Handles models with `get_text_config()` methods (with or without
        decoder parameter) and `text_config` attributes.
    """
    if config is None:
        return None

    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        try:
            return get_text_config()
        except TypeError:
            # Some configs accept `decoder=` or other optional kwargs.
            try:
                return get_text_config(decoder=True)
            except Exception:
                pass
        except Exception:
            pass

    return getattr(config, "text_config", config)


def _rope_scaling_uses_mrope(rope_scaling: Any) -> bool:
    """Check if a rope_scaling config indicates multi-dimensional RoPE.

    Examines a rope_scaling configuration to determine if it uses
    multi-dimensional RoPE (mRoPE) as found in models like Qwen2/3-VL.

    Args:
        rope_scaling: A rope_scaling configuration object or dictionary.

    Returns:
        True if the configuration indicates mRoPE usage, False otherwise.

    Note:
        Checks for mrope_section, rope_type='mrope', and mrope_interleaved
        indicators in the configuration.
    """
    if rope_scaling is None:
        return False

    to_dict = getattr(rope_scaling, "to_dict", None)
    if callable(to_dict):
        try:
            rope_scaling = to_dict()
        except Exception:
            pass

    if not isinstance(rope_scaling, Mapping):
        return False

    # HuggingFace Qwen2/3-VL often uses rope_type='default' with mrope_section present.
    if rope_scaling.get("mrope_section") is not None:
        return True

    rope_type = rope_scaling.get("rope_type")
    if rope_type is None:
        rope_type = rope_scaling.get("type")
    if isinstance(rope_type, str) and rope_type.lower() == "mrope":
        return True

    # Some configs may omit mrope_section but still flag mRoPE behavior.
    if rope_scaling.get("mrope_interleaved") is not None:
        return True

    return False


def model_uses_mrope(model: Any) -> bool:
    """Infer whether a model uses multi-dimensional RoPE (mRoPE).

    Determines if a model is configured to use multi-dimensional rotary
    position embeddings, which is common in vision-language models like
    Qwen2-VL and Qwen3-VL.

    Args:
        model: A model object with a config attribute.

    Returns:
        True if the model uses mRoPE, False otherwise.

    Note:
        Prefers config-based detection (text_config.rope_scaling) and
        falls back to the legacy `_uses_mrope` attribute when unavailable.
    """
    cfg = getattr(model, "config", None)
    text_cfg = _get_text_config(cfg)
    if text_cfg is not None and _rope_scaling_uses_mrope(getattr(text_cfg, "rope_scaling", None)):
        return True
    return bool(getattr(model, "_uses_mrope", False))
