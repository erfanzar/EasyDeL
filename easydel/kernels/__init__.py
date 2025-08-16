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

"""Custom optimized kernels for EasyDeL.

Provides hardware-specific optimized kernels using Pallas (TPU),
Triton (GPU), and JAX (CPU) implementations. These kernels offer
significant performance improvements over standard operations.

Submodules:
    cpu_ops: CPU-optimized operations using JAX
    gpu_ops: GPU-optimized operations using Triton
    tpu_ops: TPU-optimized operations using Pallas

Key Operations:
    flash_attention: Optimized attention computation
    matmul: Hardware-optimized matrix multiplication
    ring_attention: Ring-based attention for sequence parallelism
    rms_norm: Optimized RMS normalization
    collective_matmul: Distributed matrix multiplication

Platform Support:
    - TPU: Pallas kernels for v3, v4, v5
    - GPU: Triton kernels for NVIDIA/AMD
    - CPU: JAX-based implementations

Example:
    >>> from easydel.kernels import flash_attention, matmul
    >>> # Use optimized flash attention
    >>> output = flash_attention(
    ...     query, key, value,
    ...     causal=True,
    ...     softmax_scale=1.0/sqrt(head_dim)
    ... )
    >>> # Use optimized matmul
    >>> result = matmul(A, B, precision="high")

Note:
    Kernels automatically select the best implementation
    based on the detected hardware platform.
"""

# Custom Pallas/Triton kernels.
