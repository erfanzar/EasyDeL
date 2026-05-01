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
    """Per-row :class:`Transform` that re-keys top-level dict entries.

    Useful for aligning heterogeneous source schemas onto a single
    canonical schema before downstream stages (tokenization, packing)
    run. Keys not present in the rename map are passed through
    unchanged. Returns a fresh dict — the input is not mutated.

    Example:
        >>> transform = RenameFields({"conversations": "messages", "id": "example_id"})
        >>> result = transform({"conversations": [...], "id": 123})
        >>> # {"messages": [...], "example_id": 123}
    """

    def __init__(self, mapping: dict[str, str]):
        """Capture the rename map.

        Args:
            mapping: ``{old_key: new_key}`` map applied to every row.
                Keys that do not appear in the row are silently
                ignored at call time.
        """
        self._mapping = mapping

    def __call__(self, example: Example) -> Example:
        """Apply the rename map to a row, returning a new dict.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: New row dict with each key replaced by
            ``mapping.get(key, key)``. Values are forwarded by
            reference.
        """
        result = {}
        for key, value in example.items():
            new_key = self._mapping.get(key, key)
            result[new_key] = value
        return result

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"RenameFields({old: new, ...})"``.
        """
        return f"RenameFields({self._mapping!r})"


class SelectFields(Transform):
    """Per-row :class:`Transform` that projects a row down to a whitelist of keys.

    Useful before tokenization to drop large unused columns
    (metadata, raw HTML, embeddings, …) so they don't carry through
    the rest of the pipeline. Keys in the whitelist that don't
    appear in the row produce no entry in the output.

    Example:
        >>> transform = SelectFields(["text", "label"])
        >>> result = transform({"text": "hello", "label": 1, "metadata": {...}})
        >>> # {"text": "hello", "label": 1}
    """

    def __init__(self, fields: list[str]):
        """Capture the whitelist of field names to keep.

        Args:
            fields: Names of the row keys to retain. Stored as a set
                internally for O(1) membership checks during call.
        """
        self._fields = set(fields)

    def __call__(self, example: Example) -> Example:
        """Filter a row down to the configured whitelist.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: New row dict containing only the keys that are
            both in ``example`` and in the configured whitelist.
        """
        return {k: v for k, v in example.items() if k in self._fields}

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"SelectFields([...])"``.
        """
        return f"SelectFields({list(self._fields)!r})"


class DropFields(Transform):
    """Per-row :class:`Transform` that removes a blacklist of keys (complement of :class:`SelectFields`).

    Use when you want to keep most of the row but explicitly remove
    a few large/irrelevant columns. Names in the blacklist that do
    not appear in the row are silently ignored.

    Example:
        >>> transform = DropFields(["metadata", "source"])
        >>> result = transform({"text": "hello", "metadata": {...}, "source": "web"})
        >>> # {"text": "hello"}
    """

    def __init__(self, fields: list[str]):
        """Capture the blacklist of field names to drop.

        Args:
            fields: Names of the row keys to remove. Stored as a set
                internally for O(1) membership checks.
        """
        self._fields = set(fields)

    def __call__(self, example: Example) -> Example:
        """Return a new row dict without the configured blacklist of fields.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: New row dict containing only the keys that are not
            in the blacklist.
        """
        return {k: v for k, v in example.items() if k not in self._fields}

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"DropFields([...])"``.
        """
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
        """Capture the path-expression and target details.

        Args:
            source_path: Path expression resolved by
                :meth:`_extract_path`. Supports ``"."`` for nested
                dict access and ``"[i]"`` for list indexing — for
                example, ``"messages[0].content"`` or
                ``"meta.author"``.
            target_field: Top-level row key under which the extracted
                value is written.
            default: Value used when ``source_path`` cannot be
                resolved (any segment is missing or a type
                mismatches). Defaults to ``None``.
        """
        self._source_path = source_path
        self._target_field = target_field
        self._default = default

    def __call__(self, example: Example) -> Example:
        """Resolve ``source_path`` against the row and write the result to ``target_field``.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: Copy of ``example`` augmented with
            ``{target_field: value}``. ``value`` is the resolved
            path or ``self._default`` when resolution fails.
        """
        result = example.copy()
        value = self._extract_path(example, self._source_path)
        result[self._target_field] = value if value is not None else self._default
        return result

    def _extract_path(self, data: tp.Any, path: str) -> tp.Any:
        """Walk a dotted/bracketed path expression against an arbitrary nested value.

        Implements a small ad-hoc path resolver: bracket indices are
        normalised to dot segments (``"a[0].b"`` -> ``"a.0.b"``) and
        each segment is consumed in turn. Numeric segments index into
        lists, non-numeric segments index into dicts via ``.get``.
        Returns ``None`` instead of raising on any kind of mismatch
        so :meth:`__call__` can substitute the default value.

        Args:
            data: Root value to traverse — typically the row dict.
            path: Pre-normalised path expression
                (e.g. ``"messages[0].content"``).

        Returns:
            Any: The resolved value, or ``None`` if any segment
            cannot be resolved.
        """
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
        """Return a developer-friendly representation.

        Returns:
            ``"ExtractField('source_path' -> 'target_field')"``.
        """
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
        """Capture the source and target fields plus the combination strategy.

        Args:
            source_fields: Row keys whose values are passed (in order)
                into the combiner.
            target_field: Row key under which the combined value is
                stored.
            combiner: Optional callable mapping ``list[value]`` to a
                single combined value. When ``None`` (default),
                the source values are coerced to strings, ``None``
                entries are skipped, and the rest are joined with
                :attr:`separator`.
            separator: String used by the default combiner. Ignored
                when ``combiner`` is supplied.
            drop_sources: When ``True``, the source fields are
                removed from the output row after the combine.
        """
        self._source_fields = source_fields
        self._target_field = target_field
        self._combiner = combiner
        self._separator = separator
        self._drop_sources = drop_sources

    def __call__(self, example: Example) -> Example:
        """Read ``source_fields`` from the row and write the combined value to ``target_field``.

        Missing source fields contribute ``None`` to the combiner's
        argument list (and are silently skipped by the default
        string combiner).

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: Copy of ``example`` with ``target_field`` populated
            and (optionally) the source fields removed.
        """
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
        """Return a developer-friendly representation.

        Returns:
            ``"CombineFields([...] -> 'target_field')"``.
        """
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
        """Capture the destination field and the value source.

        Args:
            field: Row key under which the new value is written. If
                the row already has this key, it is overwritten.
            value: Either a constant value (any type) or a callable
                ``Callable[[dict], Any]`` invoked once per row to
                produce the value. The callable distinction is made
                by ``callable(value)`` at call time.
        """
        self._field = field
        self._value = value

    def __call__(self, example: Example) -> Example:
        """Resolve the configured value and write it to ``field`` on a row copy.

        For callable ``value``, the row is passed in so the new
        column can depend on existing fields (e.g. computing length
        from ``text``); for non-callable values, the value is used
        verbatim.

        Args:
            example: Input row dict; not mutated.

        Returns:
            dict: Copy of ``example`` with ``self._field`` populated.
        """
        result = example.copy()

        if callable(self._value):
            result[self._field] = self._value(example)
        else:
            result[self._field] = self._value

        return result

    def __repr__(self) -> str:
        """Return a developer-friendly representation.

        Returns:
            ``"AddField('name', value_repr)"``.
        """
        if callable(self._value):
            val_name = getattr(self._value, "__name__", "lambda")
            return f"AddField({self._field!r}, {val_name})"
        return f"AddField({self._field!r}, {self._value!r})"
