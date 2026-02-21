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

"""
Defines the base class for operations within EasyDeL that may have
backend-specific implementations (CPU, GPU, TPU).
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

logger = get_logger("EasyDeL-BaseOperation")


class BaseOperation(ABC):
    """
    Abstract Base Class for defining operations with potential backend-specific implementations.

    This class provides a structure for defining a core operation (`forward_native`)
    and allowing optional, optimized implementations for different hardware backends
    supported by JAX (TPU, GPU - CUDA/ROCm, CPU).

    The `__call__` method acts as a dispatcher, detecting the current JAX default
    backend and executing the corresponding `forward_...` method. If a specific
    backend implementation (e.g., `forward_tpu`) is not overridden in a subclass,
    it defaults to calling `forward_native`.

    Subclasses MUST implement the `forward_native` method. They CAN optionally
    override `forward_tpu`, `forward_gpu`, `forward_cpu`, `forward_rocm`, or
    `forward_cuda` to provide backend-specific optimizations.
    """

    metadata: OperationMetadata | None = None

    @classmethod
    @abstractmethod
    def get_impl_name(cls) -> str | tuple[str, ...]:
        """
        Returns the unique name(s) identifying this attention implementation.

        Used by the `OperationRegistry`. Can return a single string or a tuple/list
        of strings if the implementation has multiple aliases.

        Returns:
            A string or tuple/list of strings representing the implementation name(s).
        """

    @abstractmethod
    def get_impl_metadata(self) -> OperationMetadata:
        """
        Returns the `OperationMetadata` associated with this implementation instance.

        Returns:
            The `OperationMetadata` instance passed during initialization.
        """

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """
        Returns the operation requirements for metadata and cache types.

        Operations override this method to declare:
        - Required metadata fields (sequence lengths, page tables, etc.)
        - Supported cache types (transformer, ragged pages, hybrid, etc.)

        The inference engine uses these requirements to:
        - Build only necessary metadata (avoiding unnecessary computation)
        - Validate cache compatibility at initialization
        - Provide clear error messages when requirements aren't met

        Args:
            mode: The execution mode (prefill, decode, or mixed). Some operations
                  may have different requirements based on the mode.

        Returns:
            OperationRequirements declaring metadata and cache needs.

        Example:
            >>> from easydel.layers.operations.requirements import (
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
        """
        Returns the current JAX default backend as a lowercase string literal.

        Returns:
            "tpu", "gpu", or "cpu".
        """
        return jax.default_backend()

    @abstractmethod
    def forward_native(self, *args, **kwargs) -> tp.Any:
        """
        The core, backend-agnostic implementation of the operation.

        This method MUST be implemented by any concrete subclass of `BaseOperation`.
        It serves as the default implementation if no backend-specific override
        is available or applicable.

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation. Type depends on the specific operation.
        """

    def forward_tpu(self, *args, **kwargs) -> tp.Any:
        """
        TPU-specific implementation of the operation.

        Defaults to calling `forward_native`. Subclasses can override this for
        TPU-specific optimizations.

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation, potentially optimized for TPU.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tt(self, *args, **kwargs) -> tp.Any:
        """
        TT-specific implementation of the operation.

        Defaults to calling `forward_native`. Subclasses can override this for
        TT-specific optimizations.

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation, potentially optimized for TT.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> tp.Any:
        """
        CPU-specific implementation of the operation.

        Defaults to calling `forward_native`. Subclasses can override this for
        CPU-specific optimizations (though often `forward_native` is sufficient).

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation, potentially optimized for CPU.
        """
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> tp.Any:
        """
        Generic GPU-specific implementation of the operation.

        Defaults to calling `forward_native`. This method serves as the base
        for CUDA and ROCm backends unless they are specifically overridden.
        Subclasses can override this for general GPU optimizations.

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation, potentially optimized for GPUs.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> tp.Any:
        """
        ROCm (AMD GPU)-specific implementation of the operation.

        Defaults to calling `forward_gpu`. Subclasses can override this for
        optimizations specific to the ROCm platform, if necessary.

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation, potentially optimized for ROCm GPUs.
        """
        return self.forward_gpu(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> tp.Any:
        """
        CUDA (NVIDIA GPU)-specific implementation of the operation.

        Defaults to calling `forward_gpu`. Subclasses can override this for
        optimizations specific to the CUDA platform, if necessary.

        Args:
            *args: Positional arguments for the operation.
            **kwargs: Keyword arguments for the operation.

        Returns:
            The result of the operation, potentially optimized for CUDA GPUs.
        """
        return self.forward_gpu(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> tp.Any:
        """
        Executes the appropriate forward method based on the detected JAX backend.

        This method determines the current `jax.default_backend()` and dispatches
        the call to the corresponding `forward_...` method (e.g., `forward_tpu`
        if the backend is TPU). It logs which execution path is taken at the
        DEBUG level. If the backend is not explicitly recognized, it falls back
        to `forward_native`.

        Args:
            *args: Positional arguments to pass to the forward method.
            **kwargs: Keyword arguments to pass to the forward method.

        Returns:
            The result returned by the executed forward method.
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
        from easydel.infra.etils import EasyDeLBackends

        return EasyDeLBackends


_I = tp.TypeVar("ICa", bound=BaseOperation)


class OperationRegistry:
    """
    Registry for discovering and managing different `OperationImpl` classes.

    Allows registering implementations using a decorator and retrieving or
    instantiating them by name.
    """

    _registry: tp.ClassVar[dict[str, type[BaseOperation]]] = {}

    @classmethod
    def register(cls, impl_cls: type[_I]) -> type[_I]:
        """
        Class method decorator to register an `OperationImpl` subclass.

        The implementation is registered under the name(s) returned by its
        `get_impl_name()` class method.

        Example:
        ```python
        @OperationRegistry.register
        class FlashOperationImpl(OperationImpl):
          @classmethod
          def get_impl_name(cls) -> str:
            return "flash"

          # ... implementation ...
        ```

        Args:
            impl_cls: The `OperationImpl` subclass to register.

        Returns:
            The registered class itself.
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
    def get(cls, impl_name: str) -> type[BaseOperation]:
        """
        Retrieves an attention implementation class by its registered name.

        Args:
            impl_name: The name of the implementation to retrieve.

        Returns:
            The `OperationImpl` subclass registered under the given name.

        Raises:
            ValueError: If no implementation is registered with that name.
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
        """
        Creates an instance of an attention implementation by name.

        Retrieves the class associated with `impl_name` and initializes it
        with the provided `metadata`.

        Args:
            impl_name: The name of the implementation to instantiate.
            metadata: The `OperationMetadata` to pass to the implementation's constructor.
            requires_cache: Optional override for cache requirements. If provided,
                overrides the metadata's requires_cache setting for this instance.
                - None: Use metadata's requires_cache (or class default if metadata is None)
                - False: Disable cache (e.g., for vision encoders)
                - True: Force cache requirement

        Returns:
            An initialized instance of the requested `OperationImpl` subclass.

        Raises:
            ValueError: If no implementation is registered with `impl_name`.
        """
        # Apply requires_cache override to metadata if provided
        if requires_cache is not None:
            metadata.requires_cache = requires_cache

        impl_cls: type[BaseOperation] = cls.get(impl_name)
        instance: BaseOperation = impl_cls(metadata)
        return instance

    @classmethod
    def list_implementations(cls) -> list[str]:
        """
        Returns a list of names of all registered attention implementations.

        Returns:
            A list of strings, where each string is a registered implementation name.
        """
        return list(cls._registry.keys())
