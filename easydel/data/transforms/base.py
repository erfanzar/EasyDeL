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
    """Base class for all transforms.

    Transforms are lazy operations that modify examples during iteration.
    They can be chained together using the >> operator to form transform pipelines.

    Example:
        >>> transform = RenameFields({"old": "new"}) >> FilterNonEmpty(["text"])
        >>> result = transform({"old": "hello", "text": "world"})
        >>> # {"new": "hello", "text": "world"}
    """

    @abstractmethod
    def __call__(self, example: Example) -> Example | None:
        """Apply transform to a single example.

        Args:
            example: Input example dictionary.

        Returns:
            Transformed example, or None to filter out the example.
        """
        ...

    def __rshift__(self, other: "Transform") -> "ChainedTransform":
        """Chain transforms using >> operator.

        Example:
            >>> transform = RenameFields(...) >> MapTransform(...) >> FilterTransform(...)
        """
        if isinstance(self, ChainedTransform):
            return ChainedTransform([*self._transforms, other])
        return ChainedTransform([self, other])

    @property
    def is_filter(self) -> bool:
        """Whether this transform can filter out examples (return None)."""
        return False

    @property
    def is_expand(self) -> bool:
        """Whether this transform can produce multiple examples from one input."""
        return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ExpandTransform(ABC):
    """Transform that can produce multiple examples from a single input.

    Unlike Transform which returns Example | None, ExpandTransform
    yields zero or more examples via a generator. This is useful for
    operations like unpairing preference data (1 pair → 2 examples).

    Example:
        >>> class UnpairTransform(ExpandTransform):
        ...     def __call__(self, example):
        ...         yield {"text": example["chosen"], "label": True}
        ...         yield {"text": example["rejected"], "label": False}
    """

    @abstractmethod
    def __call__(self, example: Example) -> Iterator[Example]:
        """Apply transform, yielding zero or more examples.

        Args:
            example: Input example dictionary.

        Yields:
            Transformed examples. Can yield 0, 1, or many examples.
        """
        ...

    @property
    def is_expand(self) -> bool:
        """Always True for ExpandTransform."""
        return True

    @property
    def is_filter(self) -> bool:
        """ExpandTransform can filter by yielding nothing."""
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class ChainedTransform(Transform):
    """A chain of transforms applied sequentially.

    This is created automatically when using the >> operator between transforms.
    """

    def __init__(self, transforms: list[Transform]):
        """Initialize ChainedTransform.

        Args:
            transforms: List of transforms to apply in order.
        """
        self._transforms = transforms

    def __call__(self, example: Example) -> Example | None:
        """Apply all transforms in sequence.

        Args:
            example: Input example dictionary.

        Returns:
            Transformed example, or None if any filter transform removed it.
        """
        result = example
        for transform in self._transforms:
            if result is None:
                return None
            result = transform(result)
        return result

    def __rshift__(self, other: Transform) -> "ChainedTransform":
        """Append another transform to the chain."""
        return ChainedTransform([*self._transforms, other])

    @property
    def is_filter(self) -> bool:
        """True if any transform in the chain is a filter."""
        return any(t.is_filter for t in self._transforms)

    def __repr__(self) -> str:
        transform_names = " >> ".join(repr(t) for t in self._transforms)
        return f"ChainedTransform({transform_names})"

    def __len__(self) -> int:
        """Number of transforms in the chain."""
        return len(self._transforms)

    def __iter__(self):
        """Iterate over transforms in the chain."""
        return iter(self._transforms)
