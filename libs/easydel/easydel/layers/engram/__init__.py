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

"""Engram: gated n-gram hash memory written into hyper-connection streams.

* :class:`GatedEngram` / :class:`GatedEngramConfig` (``_layer``): the layer. Build it from a model config with
  :meth:`GatedEngram.from_config`. Tables: bf16 parameter or released fp8 + e8m0 layout, on device or in host RAM.
* ``_hashing``: host n-gram hashing bit-exact with DeepSeek's reference (:class:`EngramHasher`,
  :class:`EngramWindowHasher`, :func:`build_engram_token_map`, :func:`engram_bucket_primes`,
  :func:`engram_table_rows`).
* ``_loader``: row-chunked fp8 table loader from safetensors (:func:`load_engram_tables`).
* :class:`EngramModelMixin` (``_model``): table management plus the eSurge side-input and trainer host-batch hooks
  for any model that carries engram layers.
"""

from ._hashing import (
    DEAD,
    EngramHasher,
    EngramHashState,
    EngramWindowHasher,
    build_engram_token_map,
    engram_bucket_primes,
    engram_table_rows,
)
from ._layer import FP8_SCALE_BLOCK, GatedEngram, GatedEngramConfig, engram_table_padded_rows
from ._loader import (
    ENGRAM_TABLE_KEY_RE,
    EngramTensorRef,
    engram_table_checkpoint_keys,
    load_engram_tables,
    read_engram_rows,
    scan_engram_tensors,
)
from ._model import EngramModelMixin, EngramSideInputProvider, find_engram_modules

__all__ = (
    "DEAD",
    "ENGRAM_TABLE_KEY_RE",
    "FP8_SCALE_BLOCK",
    "EngramHashState",
    "EngramHasher",
    "EngramModelMixin",
    "EngramSideInputProvider",
    "EngramTensorRef",
    "EngramWindowHasher",
    "GatedEngram",
    "GatedEngramConfig",
    "build_engram_token_map",
    "engram_bucket_primes",
    "engram_table_checkpoint_keys",
    "engram_table_padded_rows",
    "engram_table_rows",
    "find_engram_modules",
    "load_engram_tables",
    "read_engram_rows",
    "scan_engram_tensors",
)
