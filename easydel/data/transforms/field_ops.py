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

"""Field manipulation operations for transforms.

This module provides:
- RenameFields: Rename fields in examples
- SelectFields: Select only specified fields
- DropFields: Drop specified fields
- ExtractField: Extract a nested field to a new top-level field
- CombineFields: Combine multiple fields into one
- AddField: Add a new field with a computed value
"""

from __future__ import annotations

import typing as tp

from .base import Example, Transform


class RenameFields(Transform):
    """Rename fields in examples.

    Example:
        >>> transform = RenameFields({"conversations": "messages", "id": "example_id"})
        >>> result = transform({"conversations": [...], "id": 123})
        >>> # {"messages": [...], "example_id": 123}
    """

    def __init__(self, mapping: dict[str, str]):
        """Initialize RenameFields.

        Args:
            mapping: Dictionary mapping old field names to new field names.
        """
        self._mapping = mapping

    def __call__(self, example: Example) -> Example:
        """Rename fields according to the mapping."""
        result = {}
        for key, value in example.items():
            new_key = self._mapping.get(key, key)
            result[new_key] = value
        return result

    def __repr__(self) -> str:
        return f"RenameFields({self._mapping!r})"


class SelectFields(Transform):
    """Select only specified fields, dropping all others.

    Example:
        >>> transform = SelectFields(["text", "label"])
        >>> result = transform({"text": "hello", "label": 1, "metadata": {...}})
        >>> # {"text": "hello", "label": 1}
    """

    def __init__(self, fields: list[str]):
        """Initialize SelectFields.

        Args:
            fields: List of field names to keep.
        """
        self._fields = set(fields)

    def __call__(self, example: Example) -> Example:
        """Keep only the specified fields."""
        return {k: v for k, v in example.items() if k in self._fields}

    def __repr__(self) -> str:
        return f"SelectFields({list(self._fields)!r})"


class DropFields(Transform):
    """Drop specified fields from examples.

    Example:
        >>> transform = DropFields(["metadata", "source"])
        >>> result = transform({"text": "hello", "metadata": {...}, "source": "web"})
        >>> # {"text": "hello"}
    """

    def __init__(self, fields: list[str]):
        """Initialize DropFields.

        Args:
            fields: List of field names to remove.
        """
        self._fields = set(fields)

    def __call__(self, example: Example) -> Example:
        """Remove the specified fields."""
        return {k: v for k, v in example.items() if k not in self._fields}

    def __repr__(self) -> str:
        return f"DropFields({list(self._fields)!r})"


class ExtractField(Transform):
    """Extract a nested field to a new top-level field.

    Supports dot notation for nested fields and bracket notation for list indices.

    Example:
        >>> transform = ExtractField("metadata.author", "author")
        >>> result = transform({"text": "hello", "metadata": {"author": "John"}})
        >>> # {"text": "hello", "metadata": {...}, "author": "John"}

        >>> # Extract first message content
        >>> transform = ExtractField("messages[0].content", "first_message")
        >>> result = transform({"messages": [{"content": "Hi"}]})
        >>> # {"messages": [...], "first_message": "Hi"}
    """

    def __init__(
        self,
        source_path: str,
        target_field: str,
        default: tp.Any = None,
    ):
        """Initialize ExtractField.

        Args:
            source_path: Path to the nested field (e.g., "metadata.author" or "items[0].name").
            target_field: Name of the new top-level field.
            default: Default value if the path doesn't exist.
        """
        self._source_path = source_path
        self._target_field = target_field
        self._default = default

    def __call__(self, example: Example) -> Example:
        """Extract the nested value to a new field."""
        result = example.copy()
        value = self._extract_path(example, self._source_path)
        result[self._target_field] = value if value is not None else self._default
        return result

    def _extract_path(self, data: tp.Any, path: str) -> tp.Any:
        """Extract value from nested path like 'a.b[0].c'."""
        # Normalize path: replace [n] with .n
        parts = path.replace("]", "").replace("[", ".").split(".")
        current = data

        for part in parts:
            if current is None:
                return None

            if part.isdigit():
                idx = int(part)
                if isinstance(current, list) and 0 <= idx < len(current):
                    current = current[idx]
                else:
                    return None
            elif isinstance(current, dict):
                current = current.get(part)
            else:
                return None

        return current

    def __repr__(self) -> str:
        return f"ExtractField({self._source_path!r} -> {self._target_field!r})"


class CombineFields(Transform):
    """Combine multiple fields into one.

    Example:
        >>> # Concatenate strings with separator
        >>> transform = CombineFields(["first_name", "last_name"], "full_name", separator=" ")
        >>> result = transform({"first_name": "John", "last_name": "Doe"})
        >>> # {"first_name": "John", "last_name": "Doe", "full_name": "John Doe"}

        >>> # Custom combiner function
        >>> transform = CombineFields(
        ...     ["a", "b", "c"],
        ...     "sum",
        ...     combiner=lambda values: sum(v for v in values if v is not None)
        ... )
    """

    def __init__(
        self,
        source_fields: list[str],
        target_field: str,
        combiner: tp.Callable[[list[tp.Any]], tp.Any] | None = None,
        separator: str = " ",
        drop_sources: bool = False,
    ):
        """Initialize CombineFields.

        Args:
            source_fields: List of field names to combine.
            target_field: Name of the combined output field.
            combiner: Optional custom function to combine values.
                If not provided, values are concatenated as strings with separator.
            separator: Separator for string concatenation (used if combiner is None).
            drop_sources: Whether to remove the source fields after combining.
        """
        self._source_fields = source_fields
        self._target_field = target_field
        self._combiner = combiner
        self._separator = separator
        self._drop_sources = drop_sources

    def __call__(self, example: Example) -> Example:
        """Combine the specified fields."""
        values = [example.get(f) for f in self._source_fields]

        if self._combiner:
            combined = self._combiner(values)
        else:
            # Default: concatenate as strings
            combined = self._separator.join(str(v) for v in values if v is not None)

        result = example.copy()
        result[self._target_field] = combined

        if self._drop_sources:
            for field in self._source_fields:
                result.pop(field, None)

        return result

    def __repr__(self) -> str:
        return f"CombineFields({self._source_fields!r} -> {self._target_field!r})"


class AddField(Transform):
    """Add a new field with a constant value or computed value.

    Example:
        >>> # Add constant value
        >>> transform = AddField("source", "web")
        >>> result = transform({"text": "hello"})
        >>> # {"text": "hello", "source": "web"}

        >>> # Add computed value
        >>> transform = AddField("length", lambda x: len(x["text"]))
        >>> result = transform({"text": "hello"})
        >>> # {"text": "hello", "length": 5}

        >>> # Add timestamp
        >>> import time
        >>> transform = AddField("timestamp", lambda _: time.time())
    """

    def __init__(
        self,
        field: str,
        value: tp.Any | tp.Callable[[Example], tp.Any],
    ):
        """Initialize AddField.

        Args:
            field: Name of the field to add.
            value: Either a constant value or a callable that takes the example
                and returns the value.
        """
        self._field = field
        self._value = value

    def __call__(self, example: Example) -> Example:
        """Add the new field to the example."""
        result = example.copy()

        if callable(self._value):
            result[self._field] = self._value(example)
        else:
            result[self._field] = self._value

        return result

    def __repr__(self) -> str:
        if callable(self._value):
            val_name = getattr(self._value, "__name__", "lambda")
            return f"AddField({self._field!r}, {val_name})"
        return f"AddField({self._field!r}, {self._value!r})"
