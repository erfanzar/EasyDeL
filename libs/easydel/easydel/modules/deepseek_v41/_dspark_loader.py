# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Load the DeepSeek-V4.1 DSpark (MTP) draft model from native ``mtp.*`` safetensors shards.

Thin wrapper over the shared :func:`easydel.modules._dspark_loader.load_dspark_model` with the V4.1 draft model and
key normalizer (:func:`native_key_to_easydel_v41_mtp`).
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx

from easydel.modules._dspark_loader import StageShards, load_dspark_model

from ._native_checkpoint import native_key_to_easydel_v41_mtp
from .deepseek_v41_configuration import DeepseekV41Config
from .modeling_deepseek_v41_dspark import DeepseekV41DSpark


def load_deepseek_v41_dspark(
    config: DeepseekV41Config,
    shard_paths: tp.Sequence[str] | None,
    *,
    quantization_config: tp.Mapping[str, tp.Any] | None = None,
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.PrecisionLike = None,
    stage_shards: StageShards | None = None,
) -> DeepseekV41DSpark:
    """Build a :class:`DeepseekV41DSpark` on ``config.mesh`` holding the weights of native ``mtp.*`` shards.

    Args:
        config: The backbone V4.1 config (its mesh and sharding settings are used for placement).
        shard_paths: Safetensors files containing the ``mtp.*`` tensors (other keys are ignored).
        quantization_config: The checkpoint's ``quantization_config`` (e.g. ``{"quant_method": "fp8",
            "weight_block_size": [32, 32], "expert_dtype": "fp4", ...}``); ``None`` for an unquantized checkpoint.
        dtype: Activation dtype of the draft model.
        param_dtype: Parameter storage dtype (fp8 / fp4 values decode exactly into bf16).
        precision: Matmul precision.
        stage_shards: Optional per-stage shard provider used instead of ``shard_paths``.

    Returns:
        The draft model with every parameter loaded and placed on its sharding.

    Raises:
        ValueError: If the shards miss a draft stage or a parameter, carry an unexpected parameter, or a shape
            disagrees with the draft model.
    """
    return load_dspark_model(
        config,
        shard_paths,
        build=lambda: DeepseekV41DSpark(
            config, dtype=dtype, param_dtype=param_dtype, precision=precision, rngs=spx.Rngs(0)
        ),
        key_normalizer=native_key_to_easydel_v41_mtp,
        quantization_config=quantization_config,
        param_dtype=param_dtype,
        stage_shards=stage_shards,
    )


__all__ = ["load_deepseek_v41_dspark"]
