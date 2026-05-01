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

"""Thread-safe class registry used by EasyDeL plugins.

This module exposes :class:`Registry`, a process-global, category-based
registry that EasyDeL uses for trainer/model/optimizer/scheduler/serve
plug-ins. Implementations register themselves by category and name,
optionally with metadata, and consumers look them up via :meth:`Registry.get`
or :meth:`Registry.get_or_raise`.

The registry is safe under concurrent access (an internal ``RLock`` guards
every mutation) and supports both decorator-style and direct registration.
"""

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
    """Process-global, category-keyed registry of pluggable EasyDeL implementations.

    EasyDeL pulls trainers, model classes, optimizers, schedulers and serving
    backends from this registry rather than referencing them directly, which
    is what allows users to swap implementations by name (e.g. via a config
    field). Each implementation lives under a ``(category, impl_name)`` pair
    and may carry a free-form ``metadata`` dict for documentation/discovery.

    State is held entirely in class-level attributes and protected by an
    ``RLock`` so that concurrent registration from multiple imports does not
    corrupt the table. Lookups (``get`` / ``get_or_raise``) take the same
    lock for visibility but do not block each other in practice (lookups are
    O(1)).

    Class Attributes:
        _registry: Two-level dict ``{category: {impl_name: cls}}`` that holds
            every registered implementation. Mutated via :meth:`do_register`
            and :meth:`unregister`.
        _lock: Reentrant lock guarding all reads and writes to
            ``_registry``/``_metadata`` — reentrant so that decorator stacks
            calling each other's classmethods do not deadlock.
        _metadata: Parallel dict of the same shape as ``_registry`` that maps
            each registration to an arbitrary metadata dict supplied by the
            caller of :meth:`register`/:meth:`do_register`. Empty when no
            metadata was provided.

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
            category: Bucket under which the class is recorded — typically
                one of ``"trainer"``, ``"model"``, ``"optimizer"``,
                ``"scheduler"``, ``"serve"`` or any custom string. The same
                category is later passed to :meth:`get`/:meth:`get_or_raise`.
            impl_names: One or more lookup keys for the registered class. A
                bare string registers a single alias; a sequence registers
                the same class under several aliases. ``None`` defaults to
                ``cls.__name__`` so a no-argument decorator still works.
            metadata: Optional free-form ``dict`` retrievable later via
                :meth:`get_metadata` — useful for short docstrings,
                category tags, or discovery hints. Stored as-is (not deep
                copied).
            overwrite: When ``True`` the registration silently replaces a
                pre-existing entry (a debug/warning is still logged); when
                ``False`` (default) a duplicate raises :class:`RegistryError`.

        Returns:
            Callable[[type[_T]], type[_T]]: Decorator that records the
            decorated class in the registry and returns it unmodified, so
            normal ``class`` semantics are preserved.
        """

        def decorator(_cls: type[_T]) -> type[_T]:
            """Register ``_cls`` under ``impl_names`` and return it unchanged.

            Args:
                _cls: The class being decorated.

            Returns:
                ``_cls`` unchanged, after recording it in the registry.
            """
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
            category: Bucket under which to install the entry; same set of
                values accepted by :meth:`register`.
            impl_names: A single string or a non-empty sequence of strings;
                each becomes an independent alias mapping to ``impl_cls``.
                Empty strings are rejected with ``ValueError``.
            impl_cls: Concrete implementation class — for trainers, models,
                optimizers, schedulers, etc. Stored verbatim and returned
                from :meth:`get` later.
            metadata: Optional metadata dict associated with every alias in
                ``impl_names``. The same dict reference is shared across
                aliases.
            overwrite: When ``True``, replace any existing registration for
                the same alias; when ``False`` (default), raise
                :class:`RegistryError`.

        Returns:
            type[_T]: ``impl_cls`` itself, returned unchanged so this method
            can be chained from a decorator.

        Raises:
            RegistryError: If a target alias already exists and
                ``overwrite`` is ``False``.
            ValueError: If ``category``, ``impl_cls``, or any name in
                ``impl_names`` is empty/None or has the wrong type.
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
            category: Registry bucket (``"trainer"``, ``"model"``, …).
            impl_names: One alias or a sequence of aliases under which the
                decorated class will be installed.
            metadata: Optional metadata dict stored alongside the entry and
                surfaced via :meth:`get_metadata`.
            overwrite: When ``True``, allow replacing an existing alias.

        Returns:
            Callable[[type[_T]], type[_T]]: Decorator that installs the
            class and returns it unchanged.

        Example:
            >>> @Registry.register_as("model", "gpt2")
            >>> class GPT2Model:
            >>>     pass
        """

        def decorator(impl_cls: type[_T]) -> type[_T]:
            """Register ``impl_cls`` under ``impl_names`` and return it.

            Args:
                impl_cls: The class being decorated.

            Returns:
                ``impl_cls`` unchanged.
            """
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
            category: Registry bucket the alias lives under.
            impl_name: Alias to remove. Removing the last alias from a
                category also drops the category itself, so subsequent
                :meth:`list_categories` calls will not include it.
            raise_if_missing: When ``True`` (default), an unknown
                ``category`` or ``impl_name`` raises
                :class:`RegistryError`; when ``False`` the method silently
                returns ``None``.

        Returns:
            type | None: The class that was previously registered (so the
            caller can re-register or inspect it), or ``None`` if no entry
            existed and ``raise_if_missing`` is ``False``.

        Raises:
            RegistryError: If the category/alias is missing and
                ``raise_if_missing`` is ``True``.
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
            category: Registry bucket to search.
            impl_name: Alias the caller registered the class under.
            default: Sentinel returned when the lookup fails. Defaults to
                ``None`` so callers can use a ``is None`` check; pass a
                placeholder class to keep typed downstream code happy.

        Returns:
            type[_T] | None: The registered class, or ``default`` when no
            entry matches. Lookups are O(1).
        """
        with cls._lock:
            return cls._registry.get(category, {}).get(impl_name, default)

    @classmethod
    def get_or_raise(cls, category: _CategoryType, impl_name: str, wakeup: bool = True) -> type:
        """
        Get a registered implementation or raise error.

        Args:
            category: Registry bucket to search.
            impl_name: Alias to look up.
            wakeup: When ``True`` (default), the method first imports
                ``easydel.inference``/``infra``/``kernels``/``layers``/
                ``modules``/``trainers`` so that any decorator-based
                registrations sitting behind a lazy module get a chance to
                execute. Set to ``False`` for tight inner loops where the
                module is known to already be imported.

        Returns:
            type: The registered implementation class.

        Raises:
            RegistryError: If no entry matches; the message lists the
                currently registered aliases for the category to help the
                caller spot typos.
        """
        if wakeup:
            try:
                from easydel import inference, infra, kernels, layers, modules, trainers  # noqa  # pyright: ignore[reportUnusedImport]
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
        """Check if an implementation exists.

        Args:
            category: Category to look in.
            impl_name: Implementation name.

        Returns:
            ``True`` if ``impl_name`` is registered in ``category``.
        """
        with cls._lock:
            return impl_name in cls._registry.get(category, {})

    @classmethod
    def get_metadata(
        cls,
        category: _CategoryType,
        impl_name: str,
    ) -> dict[str, tp.Any] | None:
        """Get metadata for a registered implementation.

        Args:
            category: Category to look in.
            impl_name: Implementation name.

        Returns:
            The metadata dict registered alongside the implementation, or
            ``None`` when no entry exists.
        """
        with cls._lock:
            return cls._metadata.get(category, {}).get(impl_name)

    @classmethod
    def list_categories(cls) -> list[_CategoryType]:
        """List all registered categories.

        Returns:
            A snapshot list of category names currently present in the registry.
        """
        with cls._lock:
            return list(cls._registry.keys())

    @classmethod
    def list_implementations(cls, category: _CategoryType) -> list[str]:
        """List all implementations in a category.

        Args:
            category: Category whose entries should be listed.

        Returns:
            A snapshot list of implementation names in ``category``; empty list
            when the category does not exist.
        """
        with cls._lock:
            return list(cls._registry.get(category, {}).keys())

    @classmethod
    def get_category_registry(cls, category: _CategoryType, wakeup: bool = True) -> dict[str, type]:
        """Get all implementations in a category.

        Args:
            category: Category to retrieve.
            wakeup: When ``True``, eagerly imports core EasyDeL packages so
                lazy decorator registrations get a chance to run first.

        Returns:
            A shallow copy of the ``{name: implementation}`` mapping for the
            category, or an empty dict when ``category`` is unknown.
        """
        if wakeup:
            try:
                from easydel import inference, infra, kernels, layers, modules, trainers  # noqa  # pyright: ignore[reportUnusedImport]
            except Exception:
                ...
        with cls._lock:
            return dict(cls._registry.get(category, {}))

    @classmethod
    def clear(cls, category: _CategoryType | None = None) -> None:
        """Drop registry entries for one category, or for all of them.

        Intended primarily for test fixtures and interactive sessions where
        you want a clean slate between experiments. Production code should
        rarely need this — registrations are normally permanent for the
        lifetime of the process.

        Args:
            category: When ``None`` (default), every category is wiped along
                with its metadata. When a string, only that bucket is
                removed; missing categories are silently ignored so that
                idempotent teardown doesn't have to guard against absent
                state.
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
            category: Registry bucket to search.
            impl_name: Alias of the implementation to instantiate.
            *args: Forwarded verbatim to the implementation's constructor.
            **kwargs: Forwarded verbatim to the implementation's constructor.

        Returns:
            Any: A freshly constructed instance of the registered class.

        Raises:
            RegistryError: If the alias is not found (propagated from
                :meth:`get_or_raise`).
        """
        impl_cls = cls.get_or_raise(category, impl_name)
        return impl_cls(*args, **kwargs)

    @classmethod
    def info(cls) -> dict[str, tp.Any]:
        """Get information about the registry state.

        Returns:
            A nested dict summarizing the number of categories, total entries
            across all categories, and a per-category list of implementation
            names.
        """
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
