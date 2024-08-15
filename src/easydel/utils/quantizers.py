from typing import Literal, Optional

import chex
from fjformer.dtypes import Array8Bit, ArrayNF4

DEFAULT_QUANTIZATION_PATTERN = (
    "(wo|wq|wk|wv|q_proj|k_proj|v_proj|o_proj|w1|w2|w3|"
    "gate_proj|up_proj|down_proj|dense_4h_to_h|dense_h_to_4h|query_key_value|wqkv|Wqkv|"
    "dense|proj_1|proj_2|out_proj|qkv_proj)"
)


class EasyQuantizer:
    def __init__(
        self,
        quantization_method: Literal["nf4", "8bit"] = "nf4",
        block_size: int = 256,
        scalar_block_size: Optional[int] = None,
    ) -> None:
        self.scalar_block_size = scalar_block_size
        self.block_size = block_size
        self.quantization_method = quantization_method

    def __call__(self, array) -> chex.Array:
        if self.quantization_method == "8bit":
            return Array8Bit.quantize(array=array)
        elif self.quantization_method == "nf4":
            should_be_quantized = True
            scalar_block_size = self.scalar_block_size
            if scalar_block_size is None:
                scalar_block_size = float(array.size / self.block_size)

                if scalar_block_size.is_integer():
                    scalar_block_size = int(scalar_block_size)
                else:
                    should_be_quantized = True
            if array.ndim <= 2 and should_be_quantized:
                return ArrayNF4.quantize(
                    array=array,
                    block_size=self.block_size,
                    scaler_block_size=self.scalar_block_size,
                )
            return array
        else:
            raise ValueError(f"unknown quantization_method {self.quantization_method}.")
