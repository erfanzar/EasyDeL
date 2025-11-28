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

"""Map operations for transforms.

This module provides:
- MapTransform: Apply a function to each example
- MapField: Apply a function to a specific field
"""

from __future__ import annotations

import typing as tp

from .base import Example, Transform


class MapTransform(Transform):
    """Apply a function to each example.

    Example:
        >>> transform = MapTransform(lambda x: {**x, "length": len(x["text"])})
        >>> result = transform({"text": "hello"})
        >>> # {"text": "hello", "length": 5}
    """

    def __init__(self, fn: tp.Callable[[Example], Example]):
        """Initialize MapTransform.

        Args:
            fn: Function that takes an example dict and returns a transformed dict.
        """
        self._fn = fn

    def __call__(self, example: Example) -> Example:
        """Apply the function to the example."""
        return self._fn(example)

    def __repr__(self) -> str:
        fn_name = getattr(self._fn, "__name__", "lambda")
        return f"MapTransform({fn_name})"


class MapField(Transform):
    """Apply a function to a specific field.

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
        """Initialize MapField.

        Args:
            field: Name of the field to apply the function to.
            fn: Function to apply to the field value.
            output_field: Optional output field name. If not specified,
                overwrites the input field.
        """
        self._field = field
        self._fn = fn
        self._output_field = output_field or field

    def __call__(self, example: Example) -> Example:
        """Apply the function to the specified field."""
        if self._field not in example:
            return example

        result = example.copy()
        result[self._output_field] = self._fn(example[self._field])
        return result

    def __repr__(self) -> str:
        fn_name = getattr(self._fn, "__name__", "lambda")
        if self._output_field != self._field:
            return f"MapField({self._field!r} -> {self._output_field!r}, {fn_name})"
        return f"MapField({self._field!r}, {fn_name})"
