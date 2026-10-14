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

"""Mixed Precision (mpric) module for JAX-based deep learning.

This module provides comprehensive utilities for mixed precision training and inference
in JAX, enabling efficient use of different data types (dtypes) for parameters,
computations, and outputs.

Key Components:
    - **PrecisionHandler**: Main class for managing mixed precision operations,
      including automatic dtype casting and loss scaling.
    - **Policy**: Defines the precision policy specifying dtypes for parameters,
      computations, and outputs.
    - **DynamicLossScale**: Implements dynamic loss scaling to prevent gradient
      underflow/overflow in low-precision training.
    - **NoOpLossScale**: A no-operation loss scaler for full precision training.
    - **LossScaleConfig**: Configuration dataclass for loss scaling parameters.

Dtype Utilities:
    - **DTYPE_MAPPING**: Maps string identifiers to JAX numpy dtypes.
    - **STRING_TO_DTYPE_MAP**: Maps dtype strings to jnp.dtype objects.
    - **DTYPE_TO_STRING_MAP**: Maps jnp.dtype objects to string representations.
    - **put_dtype**: Utility function for converting arrays to specified dtypes.

Example:
    Basic usage with a policy string::

        from eformer.mpric import PrecisionHandler, Policy

        # Create a mixed precision handler with float16 compute and float32 params
        handler = PrecisionHandler(policy="p=f32,c=f16,o=f32")

        # Wrap a training step function
        wrapped_step = handler.training_step_wrapper(my_training_step)

        # Execute with automatic precision handling
        loss, grads, grads_finite = wrapped_step(params, batch)

    Using a Policy object directly::

        from eformer.mpric import Policy, PrecisionHandler

        policy = Policy.from_string("p=f32,c=bf16,o=f32")
        handler = PrecisionHandler(policy=policy, use_dynamic_scale=True)
"""

from .dtypes import DTYPE_MAPPING, DTYPE_TO_STRING_MAP, STRING_TO_DTYPE_MAP, put_dtype
from .handler import PrecisionHandler
from .loss_scaling import DynamicLossScale, LossScaleConfig, NoOpLossScale
from .policy import Policy

__all__ = (
    "DTYPE_MAPPING",
    "DTYPE_MAPPING",
    "DTYPE_TO_STRING_MAP",
    "STRING_TO_DTYPE_MAP",
    "DynamicLossScale",
    "LossScaleConfig",
    "NoOpLossScale",
    "Policy",
    "PrecisionHandler",
    "put_dtype",
)
