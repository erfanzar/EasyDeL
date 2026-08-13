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

"""attn_softmax_dtype must not silently follow a low-precision loader dtype.

Softmax accumulation stays at the model default (float32) unless the config
sets it explicitly; deriving it from ``loader.dtype: bf16`` silently degrades
attention numerics in every bf16 run (see CLAUDE.md dtype conventions).
"""

from easydel.infra.elarge.processing import materialize_base_config
from jax import numpy as jnp


def _cfg(base_values=None):
    cfg = {
        "model": {"name_or_path": "x"},
        "loader": {"dtype": "bf16", "param_dtype": "bf16"},
        "sharding": {"axis_dims": [1, 1, -1, 1, 1, 1]},
    }
    if base_values is not None:
        cfg["base_config"] = {"values": base_values}
    return cfg


def test_softmax_dtype_not_derived_from_loader_dtype():
    base = materialize_base_config(_cfg(), prefer="base")
    assert "attn_softmax_dtype" not in base
    # the compute/kv dtypes still follow the loader dtype
    assert base["attn_dtype"] == jnp.bfloat16
    assert base["kvdtype"] == jnp.bfloat16


def test_softmax_dtype_explicit_override_still_respected():
    base = materialize_base_config(_cfg({"attn_softmax_dtype": "fp32"}), prefer="base")
    assert base["attn_softmax_dtype"] == jnp.float32
    base_bf16 = materialize_base_config(_cfg({"attn_softmax_dtype": "bf16"}), prefer="base")
    assert base_bf16["attn_softmax_dtype"] == jnp.bfloat16
