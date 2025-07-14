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

from __future__ import annotations

import typing as tp
from abc import ABC, abstractmethod

from eformer.pytree import auto_pytree


@auto_pytree
class BaseCacheMetadata(ABC):
    """
    Abstract base class defining the interface for cache metadata.

    Concrete implementations should provide:
    - Required configuration parameters for cache initialization
    - Validation logic in the create() method
    - Any metadata needed during cache operations
    """

    @classmethod
    @abstractmethod
    def create(cls, *args, **kwargs) -> BaseCacheMetadata:
        """
        Factory method to create validated metadata instance.

        Args:
            *args: Positional arguments for metadata creation
            **kwargs: Keyword arguments for metadata creation

        Returns:
            Instance of concrete metadata implementation

        Raises:
            ValueError: If any validation checks fail
        """
        pass


@auto_pytree
class BaseRunTimeMetadata:
    """
    Abstract base class for optional runtime metadata used during attention computation.

    This can hold dynamic information needed during the forward pass that isn't
    known at cache initialization time.
    """


class BaseCacheView(ABC):
    """
    Abstract base class for a single cache view (typically per layer).

    Responsible for:
    - Storing cached key/value states
    - Tracking current cache position
    - Updating cache with new states
    """

    metadata: BaseCacheMetadata
    layer_index: int | None

    @classmethod
    @abstractmethod
    def init(cls, metadata: BaseCacheMetadata, *args, **kwargs) -> BaseCacheView:
        """
        Initialize a new cache view instance.

        Args:
            metadata: Configuration metadata for the cache
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Initialized cache view instance
        """
        pass

    @abstractmethod
    def concatenate_to_cache(self, *args, **kwargs) -> tp.Any:
        """
        Update cache with new states.

        Args:
            *args: Typically includes new tensors
            **kwargs: Additional parameters for cache update

        Returns:
            Tuple containing:
                                - anything
        """
        pass


class BaseCache(ABC):
    """
    Abstract base class for the main cache container.

    Manages a sequence of cache views (typically one per layer) and provides
    initialization methods.
    """

    views: tp.Sequence[BaseCacheView | None]

    @classmethod
    @abstractmethod
    def init_cache(
        cls,
        metadata: BaseCacheMetadata,
        *args,
        **kwargs,
    ) -> BaseCache:
        """
        Initialize a complete cache with views for all layers.

        Args:
            metadata: Configuration metadata
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Fully initialized cache instance
        """
        pass

    @classmethod
    @abstractmethod
    def init_empty(cls, *args, **kwargs) -> BaseCache:
        """
        Initialize an empty cache container.

        Args:
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Cache instance with uninitialized views
        """
        pass

    def __getitem__(self, index):
        """
        Enable indexing to access cache views.

        Args:
                        index: Index of the cache view to retrieve

        Returns:
                        The cache view at the specified index
        """
        return self.views[index]

    def __setitem__(self, index, value):
        """
        Enable item assignment to update cache views.

        Args:
                        index: Index of the cache view to update
                        value: New cache view to assign
        """
        self.views[index] = value

    def __len__(self) -> int:
        """
        Returns the number of cache views.

        Returns:
                        The number of items in the `views` sequence.

        Raises:
                        AttributeError: If `self.views` has not been initialized by a subclass.
        """
        if not hasattr(self, "views"):
            raise AttributeError(
                "The 'views' attribute has not been initialized. Ensure a concrete subclass initializes it."
            )
        return len(self.views)
