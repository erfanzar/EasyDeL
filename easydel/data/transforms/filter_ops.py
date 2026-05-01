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
    """General-purpose row filter driven by a user-supplied predicate.

    Wraps an arbitrary ``Callable[[dict], bool]`` so it can be plugged
    into the transform DSL via ``>>``. Rows for which the predicate
    returns falsy are dropped (the transform returns ``None``); rows
    for which it returns truthy are forwarded unchanged. The
    predicate must be deterministic so resumed/distributed runs see
    identical filtered streams.

    Example:
        >>> transform = FilterTransform(lambda x: len(x["text"]) > 10)
        >>> transform({"text": "hello"})  # Returns None (filtered)
        >>> transform({"text": "hello world!"})  # Returns the example
    """

    def __init__(self, predicate: tp.Callable[[Example], bool]):
        """Capture the user-supplied predicate.

        Args:
            predicate: Callable receiving the row dict and returning
                truthy to keep, falsy to drop.
        """
        self._predicate = predicate

    def __call__(self, example: Example) -> Example | None:
        """Forward ``example`` if ``predicate(example)`` is truthy, otherwise ``None``.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict | None: ``example`` itself when the predicate is
            truthy, ``None`` to drop the row.
        """
        return example if self._predicate(example) else None

    @property
    def is_filter(self) -> bool:
        """Identifying flag — every :class:`FilterTransform` may drop rows.

        Returns:
            bool: Always ``True`` so chain wrappers know to expect
            ``None`` returns.
        """
        return True

    def __repr__(self) -> str:
        """Concise developer-facing repr identifying the predicate.

        Returns:
            str: ``"FilterTransform(<name>)"`` using the predicate's
            ``__name__`` attribute, or ``"lambda"`` for anonymous
            lambdas.
        """
        pred_name = getattr(self._predicate, "__name__", "lambda")
        return f"FilterTransform({pred_name})"


class FilterByField(Transform):
    """Row filter that runs a predicate against the value of a single named field.

    Convenience wrapper around :class:`FilterTransform` for the very
    common case of "keep rows whose ``foo`` matches some condition".
    Rows lacking the field are always dropped — treated as failing
    the predicate vacuously — so callers don't need to handle
    ``KeyError`` themselves.

    Example:
        >>> # Keep only English examples
        >>> transform = FilterByField("lang", lambda x: x == "en")
        >>> transform({"text": "hello", "lang": "en"})  # Returns example
        >>> transform({"text": "bonjour", "lang": "fr"})  # Returns None

        >>> # Keep examples with text longer than 100 chars
        >>> transform = FilterByField("text", lambda x: len(x) > 100)
    """

    def __init__(self, field: str, predicate: tp.Callable[[tp.Any], bool]):
        """Capture the field name and value-level predicate.

        Args:
            field: Name of the row key whose value is tested.
            predicate: Callable receiving the field value (not the
                whole row) and returning truthy to keep the row.
        """
        self._field = field
        self._predicate = predicate

    def __call__(self, example: Example) -> Example | None:
        """Forward ``example`` if it has ``field`` and the value passes ``predicate``.

        Args:
            example: Input row dict.

        Returns:
            dict | None: ``example`` itself when both conditions are
            true; ``None`` when the field is missing or the
            predicate is falsy.
        """
        if self._field not in example:
            return None
        return example if self._predicate(example[self._field]) else None

    @property
    def is_filter(self) -> bool:
        """Identifying flag — every :class:`FilterByField` may drop rows.

        Returns:
            bool: Always ``True``.
        """
        return True

    def __repr__(self) -> str:
        """Concise developer-facing repr identifying the field and predicate.

        Returns:
            str: ``"FilterByField('field', <name>)"``.
        """
        pred_name = getattr(self._predicate, "__name__", "lambda")
        return f"FilterByField({self._field!r}, {pred_name})"


class FilterNonEmpty(Transform):
    """Drop rows whose required fields are missing or hold empty values.

    Treats ``None``, ``""``, and ``[]`` as empty (the common cases
    for tokenized data: no text, no message list). A missing field
    counts as empty too. Useful as the first transform in a chain to
    purge structurally invalid rows before they reach tokenization
    or packing.

    Example:
        >>> transform = FilterNonEmpty(["text", "messages"])
        >>> transform({"text": "hello", "messages": [{"role": "user"}]})  # Kept
        >>> transform({"text": "", "messages": []})  # Filtered
        >>> transform({"text": "hello"})  # Filtered (missing "messages")
    """

    def __init__(self, fields: list[str]):
        """Capture the list of required-non-empty field names.

        Args:
            fields: Names of the row keys that must be present and
                non-empty for the row to be kept.
        """
        self._fields = fields

    def __call__(self, example: Example) -> Example | None:
        """Drop rows where any required field is missing/None/empty-string/empty-list.

        Args:
            example: Input row dict.

        Returns:
            dict | None: ``example`` when every required field is
            present and non-empty; ``None`` otherwise.
        """
        for field in self._fields:
            value = example.get(field)
            if value is None or value == "" or value == []:
                return None
        return example

    @property
    def is_filter(self) -> bool:
        """Identifying flag — every :class:`FilterNonEmpty` may drop rows.

        Returns:
            bool: Always ``True``.
        """
        return True

    def __repr__(self) -> str:
        """Concise developer-facing repr listing the required fields.

        Returns:
            str: ``"FilterNonEmpty([...])"``.
        """
        return f"FilterNonEmpty({self._fields!r})"
