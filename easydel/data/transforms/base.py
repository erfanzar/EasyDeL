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

"""Base transform protocol and chained transforms.

This module provides:
- Transform: Abstract base class for all transforms
- ExpandTransform: Transform that can produce multiple examples from one input
- ChainedTransform: Chain of transforms applied sequentially
"""

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod
from collections.abc import Iterator

Example = dict[str, tp.Any]


class Transform(ABC):
    """Abstract one-in / one-out transformation primitive of the data DSL.

    A :class:`Transform` is a callable that consumes one example dict
    and returns either a (possibly modified) example dict or ``None``
    to drop the example. Transforms are deliberately small and pure
    so they compose: the ``>>`` operator constructs a
    :class:`ChainedTransform` running the operands in order, sharing
    a single dict between them. Concrete subclasses live in
    :mod:`easydel.data.transforms.field_ops`,
    :mod:`easydel.data.transforms.filter_ops`,
    :mod:`easydel.data.transforms.map_ops`, etc.

    Two flag properties — :attr:`is_filter` and :attr:`is_expand` —
    let downstream code (the source wrappers, packers, …) reason about
    a transform's cardinality without inspecting the implementation:
    a "filter" may return ``None``, an "expand" yields multiple
    examples per input. The base class is non-filtering and
    non-expanding; specialised subclasses override the flags.

    Example:
        >>> transform = RenameFields({"old": "new"}) >> FilterNonEmpty(["text"])
        >>> result = transform({"old": "hello", "text": "world"})
        >>> # {"new": "hello", "text": "world"}
    """

    @abstractmethod
    def __call__(self, example: Example) -> Example | None:
        """Apply this transform to a single row dict.

        Args:
            example: Input row as a plain ``dict``. The transform may
                mutate it in place or return a new dict.

        Returns:
            dict | None: The transformed example to forward downstream,
            or ``None`` to drop the example entirely (useful for
            filter-style transforms).
        """
        ...

    def __rshift__(self, other: "Transform") -> "ChainedTransform":
        """Compose two transforms into a single :class:`ChainedTransform` via ``self >> other``.

        Flattens cleanly: when ``self`` is already a
        :class:`ChainedTransform`, its component list is extended
        rather than nested, so successive ``>>`` applications produce
        a flat chain instead of a tree.

        Args:
            other: The transform to run after ``self`` on every
                example.

        Returns:
            ChainedTransform: A new chain that applies ``self`` then
            ``other``.

        Example:
            >>> transform = RenameFields(...) >> MapTransform(...) >> FilterTransform(...)
        """
        if isinstance(self, ChainedTransform):
            return ChainedTransform([*self._transforms, other])
        return ChainedTransform([self, other])

    @property
    def is_filter(self) -> bool:
        """Hint that this transform may drop examples (return ``None``).

        Used by source wrappers and packers to pre-allocate buffers
        and to decide whether to short-circuit downstream work.

        Returns:
            bool: ``False`` on the base class. Subclasses such as
            :class:`~easydel.data.transforms.filter_ops.FilterTransform`
            override to ``True``.
        """
        return False

    @property
    def is_expand(self) -> bool:
        """Hint that this transform may emit several examples per input.

        :class:`ExpandTransform` is the canonical implementer; the
        regular :class:`Transform` contract is one-in / one-out so
        the base class returns ``False``.

        Returns:
            bool: ``False`` on the base class.
        """
        return False

    def __repr__(self) -> str:
        """Compact developer-facing repr (subclass name with no fields).

        Subclasses with non-trivial state are encouraged to override.

        Returns:
            str: ``"<ClassName>()"``.
        """
        return f"{self.__class__.__name__}()"


class ExpandTransform(ABC):
    """Abstract one-in / many-out transformation primitive.

    Whereas :class:`Transform` returns ``Example | None``,
    :class:`ExpandTransform` yields an arbitrary number of derived
    examples via a generator — including zero (acts as a filter)
    or many (acts as an unroller). Common use cases include
    unpairing preference data (``chosen`` + ``rejected`` -> two rows
    with labels), expanding multiple-choice questions into per-option
    rows, and chunking long documents.

    Note that :class:`ExpandTransform` does **not** subclass
    :class:`Transform` — the call signatures differ — so transforms
    that wish to consume an expand transform should detect via
    :attr:`is_expand` and use a generator-aware iteration loop.

    Example:
        >>> class UnpairTransform(ExpandTransform):
        ...     def __call__(self, example):
        ...         yield {"text": example["chosen"], "label": True}
        ...         yield {"text": example["rejected"], "label": False}
    """

    @abstractmethod
    def __call__(self, example: Example) -> Iterator[Example]:
        """Apply this expand transform, yielding zero or more derived examples.

        Args:
            example: Input row dict.

        Yields:
            Example: Derived rows; the iterator may yield zero (filter
            semantics), one (rare — typically use :class:`Transform`
            for that case), or many examples.
        """
        ...

    @property
    def is_expand(self) -> bool:
        """Identifying flag — every :class:`ExpandTransform` is an expand transform.

        Returns:
            bool: Always ``True``.
        """
        return True

    @property
    def is_filter(self) -> bool:
        """Marks expand transforms as potential filters (zero-yield drops the row).

        Returns:
            bool: Always ``True``.
        """
        return True

    def __repr__(self) -> str:
        """Compact developer-facing repr (subclass name with no fields).

        Returns:
            str: ``"<ClassName>()"``.
        """
        return f"{self.__class__.__name__}()"


class ChainedTransform(Transform):
    """Sequential composition of multiple :class:`Transform` instances.

    Constructed implicitly by the ``>>`` operator on
    :class:`Transform` (or explicitly with a list). Applying the chain
    runs each component in order, threading the row through; the
    first ``None`` short-circuits the rest of the chain so filtered
    rows do not waste work in later transforms. Re-chaining via
    ``>>`` flattens — chained transforms compose into a single flat
    chain rather than nesting.
    """

    def __init__(self, transforms: list[Transform]):
        """Capture the ordered list of transforms that compose this chain.

        Args:
            transforms: Transforms applied in iteration order. Empty
                lists are technically permitted (the chain becomes
                identity) but typically the chain is built by the
                ``>>`` operator with at least two members.
        """
        self._transforms = transforms

    def __call__(self, example: Example) -> Example | None:
        """Run every transform in the chain on ``example``, short-circuiting on ``None``.

        Args:
            example: Input row dict.

        Returns:
            dict | None: The example after all transforms, or ``None``
            if any filter transform in the chain dropped it.
        """
        result = example
        for transform in self._transforms:
            if result is None:
                return None
            result = transform(result)
        return result

    def __rshift__(self, other: Transform) -> "ChainedTransform":
        """Extend the chain with one more transform, returning a flat new chain.

        Overrides :meth:`Transform.__rshift__` to avoid producing a
        nested ``ChainedTransform`` of ``ChainedTransform`` — a long
        chain remains flat regardless of how it was assembled.

        Args:
            other: Transform appended after the existing components.

        Returns:
            ChainedTransform: New chain ``[*self._transforms, other]``.
        """
        return ChainedTransform([*self._transforms, other])

    @property
    def is_filter(self) -> bool:
        """A chain is a filter if any of its components is.

        Returns:
            bool: ``True`` when at least one component reports
            :attr:`Transform.is_filter`.
        """
        return any(t.is_filter for t in self._transforms)

    def __repr__(self) -> str:
        """Developer-facing repr that mirrors the ``>>`` syntax.

        Returns:
            str: ``"ChainedTransform(t1 >> t2 >> ...)"`` with each
            component repr'd individually.
        """
        transform_names = " >> ".join(repr(t) for t in self._transforms)
        return f"ChainedTransform({transform_names})"

    def __len__(self) -> int:
        """Number of components in the chain.

        Returns:
            int: Length of the underlying transform list.
        """
        return len(self._transforms)

    def __iter__(self):
        """Iterate over the chain's component transforms in execution order.

        Useful for tests and for debug tooling that wants to walk the
        chain (e.g. to print a pipeline diagram).

        Returns:
            Iterator[Transform]: Iterator over the underlying list.
        """
        return iter(self._transforms)
