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

import threading
import typing as tp
from collections.abc import Callable, Sequence

from eformer.loggings import get_logger

_T = tp.TypeVar("_T")
_CategoryType = str | tp.Literal["trainer", "trainer-arguments", "serve", "model", "optimizer", "scheduler"]

logger = get_logger(__name__)


class RegistryError(Exception):
    """Base exception for Registry errors."""

    pass


class Registry:
    """
    Thread-safe registry for managing implementations across different categories.

    Example:
        >>> # Direct registration
        >>> Registry.register("trainer", "my_trainer", MyTrainerClass)
        >>>
        >>> # Decorator registration
        >>> @Registry.register_as("model", ["bert", "bert-base"])
        >>> class BertModel:
        >>>     pass
        >>>
        >>> # Get implementation
        >>> model_cls = Registry.get("model", "bert")
        >>> model = model_cls()
    """

    _registry: tp.ClassVar[dict[_CategoryType, dict[str, type]]] = {}
    _lock: tp.ClassVar[threading.RLock] = threading.RLock()
    _metadata: tp.ClassVar[dict[_CategoryType, dict[str, dict[str, tp.Any]]]] = {}

    @classmethod
    def register(
        cls,
        category: _CategoryType,
        impl_names: str | Sequence[str] | None = None,
        metadata: dict[str, tp.Any] | None = None,
        overwrite: bool = False,
    ) -> Callable[[type[_T]], type[_T]]:
        """
        Decorator for registering implementations with the Registry.

        Can be used in two ways:
        1. With explicit implementation names:
            >>> @register("optimizer", "adam")
            >>> class AdamOptimizer:
            >>>     pass

        2. Using the class name (when impl_names is None):
            >>> @register("optimizer")
            >>> class AdamOptimizer:  # Will be registered as "AdamOptimizer"
            >>>     pass

        Args:
            category: Category to register under
            impl_names: Name(s) to register as. If None, uses the class name
            metadata: Optional metadata to associate with the registration
            overwrite: Whether to allow overwriting existing registrations

        Returns:
            Decorator function that registers the class
        """

        def decorator(_cls: type[_T]) -> type[_T]:
            names = impl_names if impl_names is not None else _cls.__name__
            return cls.do_register(category, names, _cls, metadata, overwrite)

        return decorator

    @classmethod
    def do_register(
        cls,
        category: _CategoryType,
        impl_names: str | Sequence[str],
        impl_cls: type[_T],
        metadata: dict[str, tp.Any] | None = None,
        overwrite: bool = False,
    ) -> type[_T]:
        """
        Register an implementation under one or more names in a category.

        Args:
            category: Category to register under
            impl_names: Name(s) to register the implementation as
            impl_cls: Implementation class to register
            metadata: Optional metadata to associate with the registration
            overwrite: Whether to allow overwriting existing registrations

        Returns:
            The registered implementation class

        Raises:
            RegistryError: If registration already exists and overwrite=False
            ValueError: If invalid arguments provided
        """
        if not category:
            raise ValueError("Category cannot be empty")

        if not impl_cls:
            raise ValueError("Implementation class cannot be None")

        if isinstance(impl_names, str):
            impl_names = [impl_names]
        elif not isinstance(impl_names, Sequence):
            raise ValueError(f"impl_names must be str or sequence, got {type(impl_names)}")

        if not impl_names:
            raise ValueError("Must provide at least one implementation name")

        with cls._lock:
            if category not in cls._registry:
                cls._registry[category] = {}
                cls._metadata[category] = {}

            for impl_name in impl_names:
                if not impl_name:
                    raise ValueError("Implementation name cannot be empty")

                if impl_name in cls._registry[category]:
                    if not overwrite:
                        raise RegistryError(
                            f"Implementation '{impl_name}' already registered in category '{category}'. "
                            f"Use overwrite=True to replace."
                        )
                    logger.warning(f"Overwriting implementation '{impl_name}' in category '{category}'")

                cls._registry[category][impl_name] = impl_cls
                cls._metadata[category][impl_name] = metadata or {}
                logger.debug(f"Registered {category} implementation: {impl_name}")

        return impl_cls

    @classmethod
    def register_as(
        cls,
        category: _CategoryType,
        impl_names: str | Sequence[str],
        metadata: dict[str, tp.Any] | None = None,
        overwrite: bool = False,
    ) -> Callable[[type[_T]], type[_T]]:
        """
        Decorator for registering implementations.

        Args:
            category: Category to register under
            impl_names: Name(s) to register the implementation as
            metadata: Optional metadata to associate with the registration
            overwrite: Whether to allow overwriting existing registrations

        Returns:
            Decorator function

        Example:
            >>> @Registry.register_as("model", "gpt2")
            >>> class GPT2Model:
            >>>     pass
        """

        def decorator(impl_cls: type[_T]) -> type[_T]:
            return cls.do_register(category, impl_names, impl_cls, metadata, overwrite)

        return decorator

    @classmethod
    def unregister(
        cls,
        category: _CategoryType,
        impl_name: str,
        raise_if_missing: bool = True,
    ) -> type | None:
        """
        Unregister an implementation.

        Args:
            category: Category to unregister from
            impl_name: Name of implementation to unregister
            raise_if_missing: Whether to raise error if not found

        Returns:
            The unregistered implementation class, or None if not found

        Raises:
            RegistryError: If implementation not found and raise_if_missing=True
        """
        with cls._lock:
            if category not in cls._registry:
                if raise_if_missing:
                    raise RegistryError(f"Category '{category}' not found")
                return None

            if impl_name not in cls._registry[category]:
                if raise_if_missing:
                    raise RegistryError(f"Implementation '{impl_name}' not found in category '{category}'")
                return None

            impl_cls = cls._registry[category].pop(impl_name)
            cls._metadata[category].pop(impl_name, None)
            logger.debug(f"Unregistered {category} implementation: {impl_name}")

            if not cls._registry[category]:
                del cls._registry[category]
                del cls._metadata[category]

            return impl_cls

    @classmethod
    def get(
        cls,
        category: _CategoryType,
        impl_name: str,
        default: type[_T] | None = None,
    ) -> type[_T] | None:
        """
        Get a registered implementation.

        Args:
            category: Category to get from
            impl_name: Name of implementation to get
            default: Default value if not found

        Returns:
            The implementation class, or default if not found
        """
        with cls._lock:
            return cls._registry.get(category, {}).get(impl_name, default)

    @classmethod
    def get_or_raise(cls, category: _CategoryType, impl_name: str, wakeup: bool = True) -> type:
        """
        Get a registered implementation or raise error.

        Args:
            category: Category to get from
            impl_name: Name of implementation to get

        Returns:
            The implementation class

        Raises:
            RegistryError: If implementation not found
        """
        if wakeup:
            try:
                from easydel import (  # noqa
                    inference,
                    infra,
                    kernels,
                    layers,
                    modules,
                    trainers,
                )
            except Exception:
                ...
        impl_cls = cls.get(category, impl_name)
        if impl_cls is None:
            available = cls.list_implementations(category)
            raise RegistryError(
                f"Implementation '{impl_name}' not found in category '{category}'. Available: {available}"
            )
        return impl_cls

    @classmethod
    def exists(cls, category: _CategoryType, impl_name: str) -> bool:
        """Check if an implementation exists."""
        with cls._lock:
            return impl_name in cls._registry.get(category, {})

    @classmethod
    def get_metadata(
        cls,
        category: _CategoryType,
        impl_name: str,
    ) -> dict[str, tp.Any] | None:
        """Get metadata for a registered implementation."""
        with cls._lock:
            return cls._metadata.get(category, {}).get(impl_name)

    @classmethod
    def list_categories(cls) -> list[_CategoryType]:
        """List all registered categories."""
        with cls._lock:
            return list(cls._registry.keys())

    @classmethod
    def list_implementations(cls, category: _CategoryType) -> list[str]:
        """List all implementations in a category."""
        with cls._lock:
            return list(cls._registry.get(category, {}).keys())

    @classmethod
    def get_category_registry(cls, category: _CategoryType, wakeup: bool = True) -> dict[str, type]:
        """Get all implementations in a category."""
        if wakeup:
            try:
                from easydel import (  # noqa
                    inference,
                    infra,
                    kernels,
                    layers,
                    modules,
                    trainers,
                )
            except Exception:
                ...
        with cls._lock:
            return dict(cls._registry.get(category, {}))

    @classmethod
    def clear(cls, category: _CategoryType | None = None) -> None:
        """
        Clear registry.

        Args:
            category: Specific category to clear, or None to clear all
        """
        with cls._lock:
            if category is None:
                cls._registry.clear()
                cls._metadata.clear()
                logger.debug("Cleared entire registry")
            elif category in cls._registry:
                del cls._registry[category]
                cls._metadata.pop(category, None)
                logger.debug(f"Cleared category '{category}' from registry")

    @classmethod
    def create(
        cls,
        category: _CategoryType,
        impl_name: str,
        *args,
        **kwargs,
    ) -> tp.Any:
        """
        Create an instance of a registered implementation.

        Args:
            category: Category to get from
            impl_name: Name of implementation to instantiate
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor

        Returns:
            Instance of the implementation

        Raises:
            RegistryError: If implementation not found
        """
        impl_cls = cls.get_or_raise(category, impl_name)
        return impl_cls(*args, **kwargs)

    @classmethod
    def info(cls) -> dict[str, tp.Any]:
        """Get information about the registry state."""
        with cls._lock:
            return {
                "categories": len(cls._registry),
                "total_implementations": sum(len(impls) for impls in cls._registry.values()),
                "details": {
                    category: {
                        "count": len(cls._registry[category]),
                        "implementations": list(cls._registry[category].keys()),
                    }
                    for category in cls._registry
                },
            }
