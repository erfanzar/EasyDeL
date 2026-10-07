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

"""Load the DeepSeek-V4 DSpark draft model (``DeepSeek-V4-Flash-0731`` ``mtp.N.*``) from native safetensors shards.

Thin wrapper over :func:`easydel.modules._dspark_loader.load_dspark_model` with :class:`DeepseekV4DSpark` and
:func:`native_key_to_easydel_v4_mtp`. The released ``config.json`` keeps ``expert_dtype: fp4`` outside its
``quantization_config`` (only ``inference/config.json`` carries it), so it is folded in here unless the caller passes it:
without it the MXFP4 experts would decode as fp8 at half their input width.
"""

from __future__ import annotations

import typing as tp

import jax
import jax.numpy as jnp
import spectrax as spx

from easydel.modules._dspark_loader import StageShards, load_dspark_model

from ._native_checkpoint import native_key_to_easydel_v4_mtp
from .deepseek_v4_configuration import DeepseekV4Config
from .modeling_deepseek_v4_dspark import DeepseekV4DSpark


def load_deepseek_v4_dspark(
    config: DeepseekV4Config,
    shard_paths: tp.Sequence[str] | None,
    *,
    num_stages: int,
    quantization_config: tp.Mapping[str, tp.Any] | None = None,
    expert_dtype: str | None = "fp4",
    dtype: jnp.dtype = jnp.bfloat16,
    param_dtype: jnp.dtype = jnp.bfloat16,
    precision: jax.lax.PrecisionLike = None,
    stage_shards: StageShards | None = None,
) -> DeepseekV4DSpark:
    """Build a :class:`DeepseekV4DSpark` on ``config.mesh`` holding the weights of native ``mtp.*`` shards.

    Args:
        config: The backbone V4 config carrying the ``dspark_*`` fields.
        shard_paths: Safetensors files containing the ``mtp.*`` tensors; ``None`` when ``stage_shards`` is given.
        num_stages: Number of ``mtp.N`` stages (``n_mtp_layers`` of ``inference/config.json``; 3 for V4-Flash-0731).
        quantization_config: The checkpoint's ``quantization_config``; ``None`` for an unquantized checkpoint.
        expert_dtype: Routed-expert storage format folded into ``quantization_config`` when absent there.
        dtype: Activation dtype of the draft model.
        param_dtype: Parameter storage dtype.
        precision: Matmul precision.
        stage_shards: Optional per-stage shard provider used instead of ``shard_paths``.

    Returns:
        The draft model with every parameter loaded and placed on its sharding.
    """
    qcfg = None
    if quantization_config is not None:
        qcfg = dict(quantization_config)
        if expert_dtype is not None:
            qcfg.setdefault("expert_dtype", expert_dtype)
    return load_dspark_model(
        config,
        shard_paths,
        build=lambda: DeepseekV4DSpark(
            config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=spx.Rngs(0),
            num_stages=num_stages,
        ),
        key_normalizer=native_key_to_easydel_v4_mtp,
        quantization_config=qcfg,
        param_dtype=param_dtype,
        stage_shards=stage_shards,
    )


__all__ = ["load_deepseek_v4_dspark"]
