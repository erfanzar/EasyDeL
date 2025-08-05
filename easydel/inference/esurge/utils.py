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

from collections.abc import Sequence
from typing import Generic, Literal, TypeVar, overload

from jax import numpy as jnp
from typing_extensions import TypeIs

T = TypeVar("T")


class ConstantList(Generic[T], Sequence):
    def __init__(self, x: list[T]) -> None:
        self._x = x

    def append(self, item):
        raise Exception("Cannot append to a constant list")

    def extend(self, item):
        raise Exception("Cannot extend a constant list")

    def insert(self, item):
        raise Exception("Cannot insert into a constant list")

    def pop(self, item):
        raise Exception("Cannot pop from a constant list")

    def remove(self, item):
        raise Exception("Cannot remove from a constant list")

    def clear(self):
        raise Exception("Cannot clear a constant list")

    def index(self, item: T, start: int = 0, stop: int | None = None) -> int:
        return self._x.index(item, start, stop if stop is not None else len(self._x))

    @overload
    def __getitem__(self, item: int) -> T: ...

    @overload
    def __getitem__(self, s: slice, /) -> list[T]: ...

    def __getitem__(self, item: int | slice) -> T | list[T]:
        return self._x[item]

    @overload
    def __setitem__(self, item: int, value: T): ...

    @overload
    def __setitem__(self, s: slice, value: T, /): ...

    def __setitem__(self, item: int | slice, value: T | list[T]):
        raise Exception("Cannot set item in a constant list")

    def __delitem__(self, item):
        raise Exception("Cannot delete item from a constant list")

    def __iter__(self):
        return iter(self._x)

    def __contains__(self, item):
        return item in self._x

    def __len__(self):
        return len(self._x)

    def __repr__(self):
        return f"ConstantList({self._x})"


def is_list_of(
    value: object,
    typ: type[T] | tuple[type[T], ...],
    *,
    check: Literal["first", "all"] = "first",
) -> TypeIs[list[T]]:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    elif check == "all":
        return all(isinstance(v, typ) for v in value)


def chunk_list(lst: list[T], chunk_size: int):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)


def next_power_of_2(n) -> int:
    """The next power of 2 (inclusive)"""
    if n < 1:
        return 1
    return 1 << (n - 1).bit_length()


def prev_power_of_2(n: int) -> int:
    """The previous power of 2 (inclusive)"""
    if n <= 0:
        return 0
    return 1 << (n.bit_length() - 1)


def round_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def round_down(x: int, y: int) -> int:
    return (x // y) * y


def get_dtype_size(dtype: jnp.ndarray) -> int:
    """Get the size of the data type in bytes."""
    return jnp.finfo(dtype).bits // 8 if jnp.issubdtype(dtype, jnp.floating) else jnp.iinfo(dtype).bits // 8
