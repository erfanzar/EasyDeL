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

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jax
from eformer.ops.quantization import QuantizationConfig, QuantizationType, quantize
from flax import nnx as nn
from jaxtyping import Array
from tqdm.autonotebook import tqdm

from .linear_8bit import Linear8bit
from .linear_nf4 import LinearNF4

DEFAULT_QUANTIZATION_PATTERN = r"^(?!.*(?:embedding|norm)).*$"

_MAP_LINEARS = {QuantizationType.INT8: Linear8bit, QuantizationType.NF4: LinearNF4}


@dataclass
class EasyDeLQuantizationConfig(QuantizationConfig):
    """Extended quantization config with pattern support for layer selection.

    This config extends eformer's QuantizationConfig with an additional `pattern`
    field for selecting which layers to quantize.

    Attributes:
        dtype: The quantization type (NF4, INT8, TERNARY, BINARY).
        block_size: Block size for block-wise quantization.
        simulate: If True, uses STE without actual bit packing (QAT mode).
        use_kernel: If True, uses optimized TPU/GPU kernels when available.
        pattern: Regex pattern for selecting layers to quantize.
                Default excludes embedding and norm layers.

    Example:
        >>> from easydel.layers.quantization import EasyDeLQuantizationConfig, QuantizationType
        >>> config = EasyDeLQuantizationConfig(
        ...     dtype=QuantizationType.NF4,
        ...     block_size=64,
        ...     pattern=r".*proj.*"  # Only quantize projection layers
        ... )
    """

    pattern: str = field(default=DEFAULT_QUANTIZATION_PATTERN)


class EasyQuantizer:
    """Unified quantization interface for EasyDeL models.

    Uses eformer's quantization infrastructure for efficient quantization.
    Supports NF4 (4-bit) and INT8 (8-bit) quantization methods.

    Args:
        quantization_config: The quantization configuration. Pass None to disable quantization.

    Example:
        >>> from easydel.layers.quantization import EasyQuantizer, EasyDeLQuantizationConfig, QuantizationType
        >>> config = EasyDeLQuantizationConfig(dtype=QuantizationType.NF4, block_size=64)
        >>> quantizer = EasyQuantizer(quantization_config=config)
        >>> quantized_model = quantizer.quantize_linears(model)
    """

    def __init__(
        self,
        quantization_config: EasyDeLQuantizationConfig | QuantizationConfig | None = None,
    ) -> None:
        self._config = quantization_config

    @property
    def config(self) -> EasyDeLQuantizationConfig | QuantizationConfig | None:
        """Get the quantization configuration."""
        return self._config

    @property
    def pattern(self) -> str:
        """Get the quantization pattern."""
        if self._config is None:
            return DEFAULT_QUANTIZATION_PATTERN
        if isinstance(self._config, EasyDeLQuantizationConfig):
            return self._config.pattern
        return DEFAULT_QUANTIZATION_PATTERN

    @jax.named_scope("easydel-easyquantize-call")
    def __call__(
        self,
        array: jax.Array,
        path: str | tuple[str] | None = None,
    ) -> Array:
        """Quantize an array using the configured quantization method.

        Args:
            array: The array to quantize.
            path: Optional path for pattern matching to determine if quantization should be applied.

        Returns:
            Quantized array (as ImplicitArray) or original array if quantization is skipped.
        """
        if self._config is None:
            return array

        should_be_quantized = True
        if path is not None:
            if isinstance(path, list):
                path = tuple(path)
            if isinstance(path, tuple):
                path = ".".join(map(str, path))
            if self.pattern is not None:
                should_be_quantized = bool(re.match(self.pattern, path))

        if not should_be_quantized:
            return array

        return quantize(array, self._config)

    def quantize_array(
        self,
        array: jax.Array,
        simulate: bool = False,
    ) -> jax.Array:
        """Quantize an array using eformer's unified quantize function.

        Args:
            array: The array to quantize.
            simulate: If True, uses STE simulation mode (materializes immediately).

        Returns:
            Quantized array.
        """
        if self._config is None:
            return array

        return quantize(
            array,
            config=self._config,
            simulate=simulate,
        )

    def quantize_linears(
        self,
        model: nn.Module,
        /,
        *,
        quantization_pattern: str | None = None,
        verbose: bool = True,
    ) -> nn.Module:
        """Quantize linear layers in a model to the configured precision.

        Args:
            model: The model to quantize.
            quantization_pattern: Regex pattern for layers to be quantized.
                                 Overrides the pattern in config if provided.
            verbose: Whether to use tqdm for progress logging.

        Returns:
            Model with quantized linear layers.
        """
        if self._config is None:
            return model

        from easydel.layers.linear import ParallelLinear
        from easydel.utils.traversals import (
            get_module_from_path,
            iter_module_search,
            set_module_from_path,
        )

        if quantization_pattern is None:
            quantization_pattern = self.pattern

        # Update model config if it exists
        if hasattr(model, "config"):
            model.config.quantization_config = self._config

        pattern = re.compile(quantization_pattern)
        dtype = self._config.dtype
        block_size = self._config.block_size

        linear_class = _MAP_LINEARS.get(dtype)
        if linear_class is None:
            raise NotImplementedError(f"Quantization not supported for dtype: {dtype}")

        dtype_name = dtype.value if isinstance(dtype, QuantizationType) else str(dtype)

        with tqdm(
            total=len([p[0] for p in iter_module_search(model, ParallelLinear)]),
            desc=f"Quantizing to {dtype_name}",
            disable=not verbose,
        ) as pbar:
            for path, _ in iter_module_search(model, ParallelLinear):
                if pattern.search(".".join([str(p) for p in path])):
                    set_module_from_path(
                        model=model,
                        path=path,
                        new_value=linear_class.from_linear(
                            linear=get_module_from_path(model=model, path=path),
                            rngs=None,
                            block_size=block_size,
                        ),
                    )
                pbar.update(1)
        return model

    def __str__(self):
        return self.__class__.__name__ + f"(\n\tconfig = {self._config}\n)"

    __repr__ = __str__
