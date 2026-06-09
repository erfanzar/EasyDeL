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

"""Backend-dispatching base class for EasyDeL kernel-style operations.

This module defines :class:`BaseOperation`, the abstract root of every
operation that may have multiple backend-specific implementations (TPU, GPU,
CPU, and the experimental TT backend). The base class wires up:

* A single backend-agnostic implementation (``forward_native``) that every
  subclass must provide.
* Per-backend override hooks (``forward_tpu``, ``forward_gpu``, ``forward_cpu``,
  ``forward_rocm``, ``forward_cuda``, ``forward_tt``) that default to
  ``forward_native`` so subclasses only override what they care about.
* A ``__call__`` dispatcher that picks the right ``forward_*`` based on the
  backend recorded in the instance's :class:`OperationMetadata`, with an
  environment-variable escape hatch (``FORCE_NATIVE_RUNTIME``) for debugging.
* A declarative :class:`OperationRequirements` machinery so callers can ask
  what metadata fields and cache types an operation needs without
  instantiating it.

It also exposes :class:`OperationRegistry`, a class-decorator registry used
by the higher-level attention dispatch in
:mod:`easydel.operations._operation_impl` and the kernels under
``easydel/operations/kernels/`` to look up implementations by name.
"""

import functools
import typing as tp
from abc import ABC, abstractmethod

import jax
from eformer.loggings import get_logger

from easydel.utils.helpers import check_bool_flag

from ._operation_meta import OperationMetadata
from .requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)

__all__ = ["BaseOperation", "OperationRegistry"]

logger = get_logger("EasyDeL-BaseOperation")


class BaseOperation(ABC):
    """Abstract base for kernel-style operations with backend-specific overrides.

    Provides a uniform structure for defining a core operation
    (:meth:`forward_native`) plus optional, optimized implementations for
    different JAX hardware backends (TPU, GPU - CUDA/ROCm, CPU, TT). The
    :meth:`__call__` dispatcher reads ``self.metadata.backend`` and routes the
    call to the matching ``forward_*`` method, falling back to
    :meth:`forward_native` when ``FORCE_NATIVE_RUNTIME`` is set.

    Subclass contract:

    * MUST implement :meth:`get_impl_name`, :meth:`get_impl_metadata`, and
      :meth:`forward_native`.
    * MAY override :meth:`forward_tpu`, :meth:`forward_gpu`, :meth:`forward_cpu`,
      :meth:`forward_rocm`, :meth:`forward_cuda`, and :meth:`forward_tt` to
      provide hardware-specialized variants. Defaults forward to
      :meth:`forward_native` (and ``forward_cuda``/``forward_rocm`` to
      :meth:`forward_gpu`).
    * MAY override :meth:`get_requirements` to declare metadata-field and
      cache-type requirements consumed by the inference engine.

    Attributes:
        metadata (OperationMetadata | None): Runtime configuration carried
            on the instance and consulted by :meth:`__call__` to pick the
            correct backend dispatch arm. Set by subclasses during
            initialization; must be non-``None`` before the operation is
            called.
    """

    metadata: OperationMetadata | None = None

    @classmethod
    @abstractmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """Return the unique name(s) identifying this implementation.

        Used by :class:`OperationRegistry` for registration and lookup. May
        return either a single string or a tuple of strings when the
        implementation should be discoverable under multiple aliases.

        Returns:
            A single name or a tuple of aliases identifying this operation.
        """

    @abstractmethod
    def get_impl_metadata(self) -> OperationMetadata:
        """Return the :class:`OperationMetadata` carried by this instance.

        Returns:
            The metadata supplied during construction. Concrete subclasses
            typically just return ``self.metadata``.
        """

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Declare the metadata fields and cache types this operation needs.

        Subclasses override this to advertise:

        * Required metadata fields (sequence lengths, page tables, segment
          ids, etc.).
        * Supported cache types (transformer, ragged pages, hybrid, etc.).

        The inference engine consults the returned requirements to build only
        the necessary metadata, validate cache compatibility up-front, and
        emit clear errors when an operation cannot be paired with the
        engine's selected cache backend.

        Args:
            mode: Execution mode (``PREFILL``, ``DECODE``, or ``MIXED``).
                Some operations expose different requirements per mode
                (e.g. decode-only paged variants).

        Returns:
            OperationRequirements describing metadata and cache needs. The
            default implementation returns "basic metadata + any cache",
            which is suitable for unconstrained training-time operators.

        Example:
            >>> from easydel.operations.requirements import (
            ...     RequirementsBuilder, MetadataField, CacheType
            ... )
            >>> @classmethod
            ... def get_requirements(cls, mode=ExecutionMode.MIXED):
            ...     return (RequirementsBuilder(cls.get_impl_name())
            ...         .require_metadata(MetadataField.PAGES_TABLES)
            ...         .support_cache(CacheType.RAGGED_PAGES)
            ...         .build())
        """
        # Default implementation: basic metadata, any cache type
        # Subclasses should override with specific requirements
        name = cls.get_impl_name()
        if isinstance(name, tuple):
            name = name[0]
        return OperationRequirements.create(
            name=name,
            required_metadata=MetadataField.basic(),
            supported_cache=CacheType.any(),
        )

    def current_backend(self) -> tp.Literal["tpu", "gpu", "cpu"]:
        """Return the current JAX default backend as a lowercase string.

        Returns:
            One of ``"tpu"``, ``"gpu"``, or ``"cpu"`` as reported by
            :func:`jax.default_backend`. Note: this is *not* what
            :meth:`__call__` dispatches on — dispatch uses
            ``self.metadata.backend`` instead.
        """
        return jax.default_backend()  # type: ignore[return-value]

    @abstractmethod
    def forward_native(self, *args, **kwargs) -> tp.Any:
        """Backend-agnostic implementation of the operation.

        Subclasses MUST implement this method. It is the default that every
        other ``forward_*`` falls back to when no hardware-specialized path
        is provided, and it is also the path forced by
        ``FORCE_NATIVE_RUNTIME=1``.

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation. Concrete type depends on the
            subclass.
        """

    def forward_tpu(self, *args, **kwargs) -> tp.Any:
        """TPU-specific implementation of the operation.

        Defaults to :meth:`forward_native`. Override in subclasses to call
        into a TPU-optimized kernel (typically a Pallas/Mosaic kernel).

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation, potentially optimized for TPU.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tt(self, *args, **kwargs) -> tp.Any:
        """Tenstorrent-specific implementation of the operation.

        Defaults to :meth:`forward_native`. Override in subclasses to call
        into a Tenstorrent-optimized kernel.

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation, potentially optimized for TT.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> tp.Any:
        """CPU-specific implementation of the operation.

        Defaults to :meth:`forward_native`. Most operations have nothing
        CPU-specific to add, so this is rarely overridden.

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation, potentially optimized for CPU.
        """
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> tp.Any:
        """Generic GPU implementation of the operation.

        Defaults to :meth:`forward_native`. Serves as the common fallback
        for both CUDA and ROCm — :meth:`forward_cuda` and
        :meth:`forward_rocm` delegate here unless overridden.

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation, potentially optimized for GPUs.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> tp.Any:
        """ROCm (AMD GPU)-specific implementation of the operation.

        Defaults to :meth:`forward_gpu`. Override only when a ROCm-specific
        code path is needed (rare in EasyDeL today).

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation, potentially optimized for ROCm.
        """
        return self.forward_gpu(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> tp.Any:
        """CUDA (NVIDIA GPU)-specific implementation of the operation.

        Defaults to :meth:`forward_gpu`. Override when a kernel is
        specifically tuned for or depends on CUDA-only features
        (e.g. Triton/CUDA-graphs paths).

        Args:
            *args: Positional arguments forwarded by :meth:`__call__`.
            **kwargs: Keyword arguments forwarded by :meth:`__call__`.

        Returns:
            The result of the operation, potentially optimized for CUDA.
        """
        return self.forward_gpu(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> tp.Any:
        """Dispatch to the backend-appropriate ``forward_*`` method.

        Reads ``self.metadata.backend`` and routes to :meth:`forward_tpu`,
        :meth:`forward_gpu`, :meth:`forward_tt`, or :meth:`forward_native`
        (used for CPU). The ``FORCE_NATIVE_RUNTIME`` environment flag (see
        :func:`easydel.utils.helpers.check_bool_flag`) short-circuits
        dispatch to :meth:`forward_native` for debugging.

        Args:
            *args: Positional arguments forwarded to the resolved
                ``forward_*`` method.
            **kwargs: Keyword arguments forwarded to the resolved
                ``forward_*`` method.

        Returns:
            The value returned by the selected ``forward_*`` method.

        Raises:
            RuntimeError: If ``self.metadata.backend`` is not one of the
                recognised :class:`~easydel.infra.etils.EasyDeLBackends`
                values.
        """

        if check_bool_flag("FORCE_NATIVE_RUNTIME", False):
            return self.forward_native(*args, **kwargs)

        match self.metadata.backend:
            case self.EasyDeLBackends.TPU:
                logger.debug("Calling into TPU exec")
                return self.forward_tpu(*args, **kwargs)
            case self.EasyDeLBackends.GPU:
                logger.debug("Calling into GPU exec")
                return self.forward_gpu(*args, **kwargs)
            case self.EasyDeLBackends.TT:
                logger.debug("Calling into TT exec")
                return self.forward_tt(*args, **kwargs)
            case self.EasyDeLBackends.CPU:
                logger.debug("Calling into CPU exec")
                return self.forward_native(*args, **kwargs)
            case _:
                raise RuntimeError(f"unknown backend at OperationImpl! {self.metadata.backend}")

    @functools.cached_property
    def EasyDeLBackends(self):
        """Lazy import of :class:`easydel.infra.etils.EasyDeLBackends`.

        Cached on the instance to avoid the import cost on every dispatch
        through :meth:`__call__`. Used by ``match`` arms to compare
        ``self.metadata.backend`` against the canonical backend enum.

        Returns:
            type: The ``EasyDeLBackends`` enum class.
        """
        from easydel.infra.etils import EasyDeLBackends

        return EasyDeLBackends


_I = tp.TypeVar("ICa", bound=BaseOperation)


class OperationRegistry:
    """Class-level registry of :class:`BaseOperation` implementations.

    Acts as a name-keyed plugin table populated by the
    :meth:`OperationRegistry.register` decorator and read by the inference
    engine and attention dispatch layers to materialize operations on demand.
    Registration is performed at import time; the registry holds *classes*,
    not instances, and :meth:`create` is the canonical instantiation path.

    Attributes:
        _registry (ClassVar[dict[str, type[BaseOperation]]]): Mapping from
            registered implementation name to the operation class. Populated
            by :meth:`register` and queried by :meth:`get`, :meth:`create`,
            and :meth:`list_implementations`.
    """

    _registry: tp.ClassVar[dict[str, type[BaseOperation]]] = {}

    @classmethod
    def register(cls, impl_cls: type[_I]) -> type[_I]:
        """Class decorator that registers an operation implementation.

        The implementation is registered under each name returned by
        ``impl_cls.get_impl_name()``. If the same name is registered twice
        the second registration wins and a warning is logged.

        Example:
            >>> @OperationRegistry.register
            ... class FlashOperationImpl(OperationImpl):
            ...     @classmethod
            ...     def get_impl_name(cls) -> str:
            ...         return "flash"
            ...     # ... implementation ...

        Args:
            impl_cls: The :class:`BaseOperation` subclass to register.

        Returns:
            The registered class, unchanged (so the decorator is transparent).
        """

        impl_names_raw: str | tuple[str, ...] = impl_cls.get_impl_name()
        impl_names: list[str] | tuple[str, ...]
        if not isinstance(impl_names_raw, list | tuple):
            impl_names = [impl_names_raw]
        else:
            impl_names = impl_names_raw

        impl_name: str
        for impl_name in impl_names:
            already_registered: bool = impl_name in cls._registry
            if already_registered:
                logger.warning(f"Operation implementation '{impl_name}' already registered. Overwriting.")
            cls._registry[impl_name] = impl_cls
            logger.debug(f"Registered attention implementation: {impl_name}")
        return impl_cls

    @classmethod
    def get(cls, impl_name: str) -> type[BaseOperation] | None:
        """Look up an operation implementation class by registered name.

        Args:
            impl_name: Name the implementation was registered under (see
                :meth:`register`).

        Returns:
            The :class:`BaseOperation` subclass registered for ``impl_name``.

        Raises:
            ValueError: If no implementation is registered under
                ``impl_name``. The error message includes the list of all
                available implementations.
        """
        is_registered: bool = impl_name in cls._registry
        if not is_registered:
            available_impls: list[str] = list(cls._registry.keys())
            raise ValueError(
                f"Operation implementation '{impl_name}' not found. Available implementations: {available_impls}"
            )
        impl_class: type[BaseOperation] = cls._registry[impl_name]
        return impl_class

    @classmethod
    def create(
        cls,
        impl_name: str,
        metadata: OperationMetadata,
        requires_cache: bool | None = None,
    ) -> BaseOperation:
        """Look up and instantiate an operation implementation by name.

        Retrieves the class registered for ``impl_name`` via :meth:`get` and
        constructs it with the supplied ``metadata``. When ``requires_cache``
        is given, the metadata's ``requires_cache`` field is mutated to that
        value before construction so the new instance reports the override
        through :meth:`OperationImpl.get_instance_requirements`.

        Args:
            impl_name: Registered name of the implementation to instantiate.
            metadata: Runtime configuration handed to the constructor.
            requires_cache: Optional instance-level override for the
                operation's cache requirement.

                * ``None`` — keep ``metadata.requires_cache`` as-is.
                * ``False`` — disable the cache (e.g., for vision encoders).
                * ``True`` — force the operation to require a cache.

        Returns:
            A freshly constructed :class:`BaseOperation` subclass instance.

        Raises:
            ValueError: If no implementation is registered under
                ``impl_name`` (propagated from :meth:`get`).
        """
        # Apply requires_cache override to metadata if provided
        if requires_cache is not None:
            metadata.requires_cache = requires_cache

        impl_cls: type[BaseOperation] = cls.get(impl_name)
        instance: BaseOperation = impl_cls(metadata)
        return instance

    @classmethod
    def list_implementations(cls) -> list[str]:
        """Return the names of all registered operation implementations.

        Returns:
            A list of registered implementation names. Multiple entries may
            point to the same class when that class declares aliases via
            :meth:`BaseOperation.get_impl_name`.
        """
        return list(cls._registry.keys())
