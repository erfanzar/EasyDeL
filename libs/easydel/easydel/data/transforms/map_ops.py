# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Map operations for transforms.

This module provides:
- MapTransform: Apply a function to each example
- MapField: Apply a function to a specific field
"""

from __future__ import annotations

import typing as tp

from .base import Example, Transform


class MapTransform(Transform):
    """General-purpose row mapper that wraps an arbitrary user-supplied function.

    The user function is responsible for whatever transformation it
    needs to do — copy the row, mutate it, replace it entirely. The
    return value becomes the new row. Use this when none of the
    more specific helpers (:class:`MapField`,
    :class:`~easydel.data.transforms.field_ops.RenameFields`, …) fit.

    Example:
        >>> transform = MapTransform(lambda x: {**x, "length": len(x["text"])})
        >>> result = transform({"text": "hello"})
        >>> # {"text": "hello", "length": 5}
    """

    def __init__(self, fn: tp.Callable[[Example], Example]):
        """Capture the user-supplied row mapper function.

        Args:
            fn: Callable taking the row dict and returning the
                replacement row dict. Should be deterministic for
                reproducible iteration.
        """
        self._fn = fn

    def __call__(self, example: Example) -> Example:
        """Run the captured function on the row and return its result.

        Args:
            example: Input row dict; ownership/mutation is at the
                discretion of ``fn``.

        Returns:
            dict: Whatever ``fn(example)`` returned.
        """
        return self._fn(example)

    def __repr__(self) -> str:
        """Concise developer-facing repr identifying the wrapped function.

        Returns:
            str: ``"MapTransform(<name>)"`` using ``fn.__name__``, or
            ``"lambda"`` for anonymous callables.
        """
        fn_name = getattr(self._fn, "__name__", "lambda")
        return f"MapTransform({fn_name})"


class MapField(Transform):
    """Apply a value-level function to a single named field of each row.

    The function operates on the field value, not the whole row, so
    standard ``str``/``len``/``int`` builtins compose without
    boilerplate. The result can be written back to the same field
    (``output_field=None``, the default) or to a different field
    (e.g. computing a length column from a text column).

    Example:
        >>> transform = MapField("text", str.upper)
        >>> result = transform({"text": "hello", "id": 1})
        >>> # {"text": "HELLO", "id": 1}

        >>> # Create new field from existing
        >>> transform = MapField("text", len, output_field="length")
        >>> result = transform({"text": "hello"})
        >>> # {"text": "hello", "length": 5}
    """

    def __init__(
        self,
        field: str,
        fn: tp.Callable[[tp.Any], tp.Any],
        output_field: str | None = None,
    ):
        """Capture the source field, transformation, and (optional) destination.

        Args:
            field: Row key whose value is fed into ``fn``.
            fn: Pure callable mapping the field value to its new
                value (or to a derived value when ``output_field``
                differs).
            output_field: Optional destination key. ``None`` (the
                default) overwrites ``field`` in place; otherwise
                the new value is stored under the named key while
                ``field`` is left untouched.
        """
        self._field = field
        self._fn = fn
        self._output_field = output_field or field

    def __call__(self, example: Example) -> Example:
        """Apply ``fn`` to ``example[field]`` and write the result to ``output_field``.

        When the source field is missing from the row the call is a
        no-op — the original row is returned unchanged so callers
        can safely chain :class:`MapField` after a filter that may
        have removed the column.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: Either ``example`` itself (when the field was
            absent) or a shallow copy with the new value written
            under ``output_field``.
        """
        if self._field not in example:
            return example

        result = example.copy()
        result[self._output_field] = self._fn(example[self._field])
        return result

    def __repr__(self) -> str:
        """Concise developer-facing repr identifying the source/dest and function.

        Returns:
            str: ``"MapField('field', <name>)"`` when source and
            destination match, or
            ``"MapField('src' -> 'dst', <name>)"`` when they differ.
        """
        fn_name = getattr(self._fn, "__name__", "lambda")
        if self._output_field != self._field:
            return f"MapField({self._field!r} -> {self._output_field!r}, {fn_name})"
        return f"MapField({self._field!r}, {fn_name})"
