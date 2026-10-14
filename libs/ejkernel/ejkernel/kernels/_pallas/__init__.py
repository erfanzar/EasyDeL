# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Pallas kernel implementations for TPU and GPU backends.

This package groups JAX Pallas kernel implementations by target hardware.
Pallas is JAX's low-level kernel authoring framework; it compiles to
hardware-specific code via Mosaic (TPU) or Triton (GPU).

Submodules:
    tpu: TPU-optimized kernels compiled through Pallas/Mosaic. Kernels exploit
        the TPU Matrix Multiply Unit (MXU), VMEM/HBM hierarchy, and TPU DMA
        engines.
    gpu: GPU-optimized kernels compiled through Pallas/Triton. Kernels target
        NVIDIA GPUs and may delegate to cuDNN for some attention operations.

Note:
    The ``gpu`` submodule is imported lazily and silently set to ``None`` if
    Triton is not installed, so callers must guard ``_pallas.gpu is not None``
    before use on CPU/TPU-only environments.
"""

from . import tpu

try:
    from . import gpu
except ModuleNotFoundError as err:  # pragma: no cover
    if err.name not in {"triton", "jax.experimental.pallas.triton"} and not (
        isinstance(err.name, str) and err.name.startswith("triton")
    ):
        raise
    gpu = None  # type: ignore[assignment]

__all__ = ("gpu", "tpu")
