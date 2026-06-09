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

"""Mixed precision policy implementation.

This module provides the Policy dataclass for defining precision configurations
used in mixed precision training and inference.
"""

import dataclasses

import jax.numpy as jnp

from ..dtypes.precision_types import DTYPE_MAPPING, get_platform_default_half


@dataclasses.dataclass(frozen=True)
class Policy:
    """Mixed precision policy defining casting behavior for different operations.

    This immutable dataclass defines the dtypes used for three distinct aspects
    of mixed precision computation:

    - **param_dtype**: The dtype for storing model parameters. Typically float32
      to maintain precision during optimization.
    - **compute_dtype**: The dtype for forward/backward pass computations.
      Lower precision (float16, bfloat16) can speed up computation.
    - **output_dtype**: The dtype for function outputs. Often matches param_dtype
      for loss computation accuracy.

    The policy is frozen (immutable) to ensure consistency during training and
    to allow safe usage as a static argument in JIT-compiled functions.

    Attributes:
        param_dtype: JAX numpy dtype for model parameters.
        compute_dtype: JAX numpy dtype for computations.
        output_dtype: JAX numpy dtype for outputs.

    Example:
        Creating a policy for TPU training with bfloat16 compute::

            policy = Policy(
                param_dtype=jnp.float32,
                compute_dtype=jnp.bfloat16,
                output_dtype=jnp.float32
            )

        Creating from string specification::

            policy = Policy.from_string("p=f32,c=bf16,o=f32")

    Note:
        For TPU training, bfloat16 is typically preferred as compute_dtype
        due to better hardware support. For GPU training, float16 may offer
        better performance with tensor cores.
    """

    param_dtype: jnp.dtype
    compute_dtype: jnp.dtype
    output_dtype: jnp.dtype

    @classmethod
    def from_string(cls, policy_str: str) -> "Policy":
        """Create a Policy from a string specification.

        This factory method parses a string specification to create a Policy
        instance. It supports both simple (single dtype) and detailed
        (per-operation dtype) specifications.

        Args:
            policy_str: A string specifying the precision policy. Supported formats:

                **Simple format** (single dtype for all operations):
                    - "f32", "float32": Use float32 for all operations
                    - "bf16", "bfloat16": Use bfloat16 for all operations
                    - "f16", "float16": Use float16 for all operations
                    - "half": Use platform-specific half precision
                      (bfloat16 on TPU, float16 on GPU/CPU)

                **Detailed format** (comma-separated key=value pairs):
                    - "p=f32,c=bf16,o=f32": Explicit dtypes for each operation
                    - Keys: p/params, c/compute, o/output
                    - Values: Any supported dtype string from DTYPE_MAPPING

        Returns:
            A Policy instance with the specified dtypes.

        Raises:
            ValueError: If an unknown dtype string is provided.

        Example:
            >>> policy = Policy.from_string("p=f32,c=f16,o=f32")
            >>> policy.param_dtype
            dtype('float32')
            >>> policy.compute_dtype
            dtype('float16')

            >>> policy = Policy.from_string("bf16")
            >>> policy.param_dtype == policy.compute_dtype == policy.output_dtype
            True

            >>> policy = Policy.from_string("half")  # Platform-specific
            >>> # On TPU: bfloat16, on GPU: float16

        Note:
            When using the detailed format, if compute_dtype is not specified,
            it defaults to param_dtype. If output_dtype is not specified, it
            defaults to compute_dtype.
        """
        param_dtype = jnp.float32
        compute_dtype = output_dtype = None

        if "=" in policy_str:
            for part in policy_str.split(","):
                key, value = part.strip().split("=", 2)
                target = value.strip().lower()
                if target == "half":
                    dtype = get_platform_default_half()
                else:
                    dtype = DTYPE_MAPPING.get(target)
                if dtype is None:
                    raise ValueError(f"Unknown dtype: {value}")

                if key in ("p", "params"):
                    param_dtype = dtype
                elif key in ("c", "compute"):
                    compute_dtype = dtype
                elif key in ("o", "output"):
                    output_dtype = dtype
        else:
            target = policy_str.strip().lower()
            if target == "half":
                dtype = get_platform_default_half()
            else:
                dtype = DTYPE_MAPPING.get(target)
            if dtype is None:
                raise ValueError(f"Unknown dtype: {policy_str}")
            param_dtype = compute_dtype = output_dtype = dtype

        if compute_dtype is None:
            compute_dtype = param_dtype
        if output_dtype is None:
            output_dtype = compute_dtype

        return cls(param_dtype=param_dtype, compute_dtype=compute_dtype, output_dtype=output_dtype)
