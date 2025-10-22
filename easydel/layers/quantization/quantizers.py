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


import re

import chex
import jax
from eformer.ops.quantization import Array8B, ArrayNF4
from flax import nnx as nn
from tqdm.autonotebook import tqdm

from easydel.infra.etils import EasyDeLPlatforms, EasyDeLQuantizationMethods
from easydel.layers.linear import ParallelLinear

from .linear_8bit import Linear8bit
from .linear_nf4 import LinearNF4

DEFAULT_QUANTIZATION_PATTERN = (
    "(wo|wq|wk|wv|q_proj|k_proj|v_proj|o_proj|w1|w2|w3|"
    "gate_proj|up_proj|down_proj|dense_4h_to_h|dense_h_to_4h|query_key_value|wqkv|Wqkv|"
    "dense|proj_1|proj_2|out_proj|qkv_proj)"
)

METHOD_TO_LINEAR_MAPPING = {
    EasyDeLQuantizationMethods.NF4: LinearNF4,
    EasyDeLQuantizationMethods.A8BIT: Linear8bit,
}


class EasyQuantizer:
    def __init__(
        self,
        quantization_method: EasyDeLQuantizationMethods = EasyDeLQuantizationMethods.NF4,
        quantization_platform: EasyDeLPlatforms | None = EasyDeLPlatforms.JAX,
        quantization_pattern: str | None = None,
        block_size: int = 256,
        **kwargs,
    ) -> None:
        self.block_size = block_size
        self.quantization_method = quantization_method
        self.quantization_platform = quantization_platform
        if quantization_pattern is None:
            quantization_pattern = r"^(?!.*(?:embedding|norm)).*$"
        self.quantization_pattern = quantization_pattern

    @jax.named_scope("easydel-easyquantize-call")
    def __call__(
        self,
        array,
        path: str | tuple[str] | None = None,
    ) -> chex.Array:
        should_be_quantized = True
        if path is not None:
            if isinstance(path, list):
                path = tuple(path)
            if isinstance(path, tuple):
                path = map(str, path)
                path = ".".join(path)
            if self.quantization_pattern is not None:
                should_be_quantized = bool(re.match(self.quantization_pattern, path))
        if not should_be_quantized:
            return array
        match self.quantization_method:
            case EasyDeLQuantizationMethods.A8BIT:
                return Array8B.quantize(array=array)
            case EasyDeLQuantizationMethods.NF4:
                should_be_quantized = True
                if array.size % self.block_size != 0:
                    should_be_quantized = False
                if should_be_quantized:
                    return ArrayNF4.quantize(array=array, block_size=self.block_size)

                return array
            case EasyDeLQuantizationMethods.NONE:
                return array
            case None:
                return array
            case _:
                raise ValueError(f"unknown quantization_method {self.quantization_method}.")

    def quantize_linears(
        self,
        model: nn.Module,
        /,
        *,
        quantization_pattern: str | None = None,
        verbose: bool = True,
    ) -> nn.Module:
        """
        Quantize parameters to requested precision, excluding specified layers.

        Args:
                        model: The model to quantize.
                        quantization_pattern (str): re pattern for layers to be quantized.
                        verbose (bool): whenever to use tqdm for logging stuff.

        Returns:
                        Quantized parameters in the same structure as the input.
        """
        if self.quantization_method == EasyDeLQuantizationMethods.NONE or self.quantization_method is None:
            return model

        from easydel.utils.traversals import get_module_from_path, iter_module_search, set_module_from_path

        quantizer = METHOD_TO_LINEAR_MAPPING.get(self.quantization_method, None)
        if quantizer is None:
            raise NotImplementedError("Requested Quantizer is not Supported")
        if quantization_pattern is None:
            quantization_pattern = self.quantization_pattern

        if hasattr(model, "config"):
            model.config.quantization_method = self.quantization_method
            model.config.quantization_block_size = self.block_size
            model.config.quantization_pattern = quantization_pattern

        pattern = re.compile(quantization_pattern)

        with tqdm(
            total=len([p[0] for p in iter_module_search(model, ParallelLinear)]),
            desc=f"Quantizing to {self.quantization_method}",
            disable=not verbose,
        ) as pbar:
            for path, _ in iter_module_search(model, ParallelLinear):
                if pattern.search(".".join([str(p) for p in path])):
                    set_module_from_path(
                        model=model,
                        path=path,
                        new_value=quantizer.from_linear(
                            linear=get_module_from_path(model=model, path=path),
                            rngs=None,
                            block_size=self.block_size,
                        ),
                    )
                pbar.update(1)
        return model

    def __str__(self):
        return (
            self.__class__.__name__
            + f"(\n\tblock_size = {self.block_size}"
            + f"\n\tquantization_method = {self.quantization_method}"
            + f"\n\tquantization_platform = {self.quantization_platform}"
            + f"\n\tquantization_pattern = {self.quantization_pattern}\n)"
        )

    __repr__ = __str__
