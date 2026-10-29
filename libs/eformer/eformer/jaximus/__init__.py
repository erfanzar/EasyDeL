# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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
Implicit Array System for JAX.

This module provides a framework for creating lazy/deferred array representations
that integrate transparently with JAX operations. It enables custom array types
(like quantized arrays) to defer materialization until necessary, allowing for
memory-efficient operations through custom kernels.

Core Components:
    ImplicitArray: Abstract base class for implicit array implementations.
        Subclass this to create custom array types like NF4 or INT8 quantized arrays.

    register: Decorator to register custom handlers for JAX primitives.
        Use this to define how operations like matmul work with your custom arrays.

    use_implicit/implicit: Context manager/decorator to enable implicit array dispatch.
        Wraps functions to automatically route operations to custom handlers.

    ste: Straight-Through Estimator decorator for quantization-aware training.
        Enables gradient flow through non-differentiable quantization operations.

    aux_field: Helper for marking dataclass fields as auxiliary (non-pytree) data.
        Use for static metadata like block_size, dtype info, etc.

Example Usage:
    >>> from eformer.jaximus import ImplicitArray, register, implicit, aux_field
    >>> from dataclasses import dataclass
    >>>
    >>> @dataclass
    >>> class MyQuantizedArray(ImplicitArray):
    ...     packed_data: jax.Array      # Pytree child (traced by JAX)
    ...     scale: jax.Array            # Pytree child
    ...     block_size: int = aux_field()  # Auxiliary (static)
    ...
    ...     def materialize(self):
    ...         return dequantize(self.packed_data, self.scale, self.block_size)
    >>>
    >>> # Register custom matmul handler
    >>> @register("dot_general")
    >>> def quantized_matmul(lhs: jax.Array, rhs: MyQuantizedArray, **params):
    ...     return optimized_quantized_kernel(lhs, rhs)
    >>>
    >>> # Use implicit dispatch
    >>> @implicit
    >>> def forward(x, weights):
    ...     return x @ weights  # Uses quantized_matmul automatically

See Also:
    - eformer.ops.quantization: Quantized array implementations (ArrayNF4, ArrayInt8)
    - JAX transformations: jit, grad, vmap all work with implicit arrays
"""

from ._imus import ImplicitArray, OrginArray, aux_field, implicit, register, ste, use_implicit

__all__ = ("ImplicitArray", "OrginArray", "aux_field", "implicit", "register", "ste", "use_implicit")
