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

import typing as tp
from abc import ABC, abstractmethod

import jax
from eformer.loggings import get_logger

from easydel.infra.etils import EasyDeLBackends

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
        backend = jax.default_backend()
        match backend:
            case EasyDeLBackends.TPU:
                logger.debug("Calling into TPU execution path.")
                return self.forward_tpu(*args, **kwargs)
            case EasyDeLBackends.GPU:
                # Note: JAX identifies GPU generically. We rely on subclasses
                # potentially overriding forward_cuda/forward_rocm if needed,
                # but the primary dispatch here is to forward_gpu.
                logger.debug("Calling into GPU execution path.")
                return self.forward_gpu(*args, **kwargs)
            case EasyDeLBackends.CPU:
                logger.debug("Calling into CPU execution path.")
                # CPU often uses the native implementation directly
                return self.forward_cpu(*args, **kwargs)
            case _:
                # Fallback for unknown or non-standard backends
                logger.debug(f"Calling into Native execution path (Unknown or default backend: {backend}).")
                return self.forward_native(*args, **kwargs)
