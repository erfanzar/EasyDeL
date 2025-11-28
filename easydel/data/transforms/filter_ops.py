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

"""Filter operations for transforms.

This module provides:
- FilterTransform: Filter examples by a predicate function
- FilterByField: Filter based on a specific field value
- FilterNonEmpty: Filter out examples with empty fields
"""

from __future__ import annotations

import typing as tp

from .base import Example, Transform


class FilterTransform(Transform):
    """Filter examples based on a predicate function.

    Examples that don't match the predicate are filtered out (return None).

    Example:
        >>> transform = FilterTransform(lambda x: len(x["text"]) > 10)
        >>> transform({"text": "hello"})  # Returns None (filtered)
        >>> transform({"text": "hello world!"})  # Returns the example
    """

    def __init__(self, predicate: tp.Callable[[Example], bool]):
        """Initialize FilterTransform.

        Args:
            predicate: Function that returns True for examples to keep.
        """
        self._predicate = predicate

    def __call__(self, example: Example) -> Example | None:
        """Return example if predicate is True, else None."""
        return example if self._predicate(example) else None

    @property
    def is_filter(self) -> bool:
        return True

    def __repr__(self) -> str:
        pred_name = getattr(self._predicate, "__name__", "lambda")
        return f"FilterTransform({pred_name})"


class FilterByField(Transform):
    """Filter examples based on a specific field value.

    Example:
        >>> # Keep only English examples
        >>> transform = FilterByField("lang", lambda x: x == "en")
        >>> transform({"text": "hello", "lang": "en"})  # Returns example
        >>> transform({"text": "bonjour", "lang": "fr"})  # Returns None

        >>> # Keep examples with text longer than 100 chars
        >>> transform = FilterByField("text", lambda x: len(x) > 100)
    """

    def __init__(self, field: str, predicate: tp.Callable[[tp.Any], bool]):
        """Initialize FilterByField.

        Args:
            field: Name of the field to check.
            predicate: Function that takes the field value and returns True to keep.
        """
        self._field = field
        self._predicate = predicate

    def __call__(self, example: Example) -> Example | None:
        """Return example if field matches predicate, else None."""
        if self._field not in example:
            return None
        return example if self._predicate(example[self._field]) else None

    @property
    def is_filter(self) -> bool:
        return True

    def __repr__(self) -> str:
        pred_name = getattr(self._predicate, "__name__", "lambda")
        return f"FilterByField({self._field!r}, {pred_name})"


class FilterNonEmpty(Transform):
    """Filter out examples where specified fields are empty.

    Checks for None, empty string "", and empty list [].

    Example:
        >>> transform = FilterNonEmpty(["text", "messages"])
        >>> transform({"text": "hello", "messages": [{"role": "user"}]})  # Kept
        >>> transform({"text": "", "messages": []})  # Filtered
        >>> transform({"text": "hello"})  # Filtered (missing "messages")
    """

    def __init__(self, fields: list[str]):
        """Initialize FilterNonEmpty.

        Args:
            fields: List of field names that must be non-empty.
        """
        self._fields = fields

    def __call__(self, example: Example) -> Example | None:
        """Return example if all fields are non-empty, else None."""
        for field in self._fields:
            value = example.get(field)
            if value is None or value == "" or value == []:
                return None
        return example

    @property
    def is_filter(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"FilterNonEmpty({self._fields!r})"
