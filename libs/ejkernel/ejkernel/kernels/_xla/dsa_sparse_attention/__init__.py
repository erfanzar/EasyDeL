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

"""Pool-bitmap block-sparse DSA attention: block tables and the dense XLA reference."""

from ._interface import dsa_sparse_attention
from ._tables import (
    as_pool_bits,
    build_q_tables,
    build_tables,
    expand_open_mask,
    pack_pool_bits,
    parse_sparse_mode,
    unpack_pool_bits,
    visited_open_mask,
)
from ._xla_impl import dsa_sparse_attention_xla

__all__ = (
    "as_pool_bits",
    "build_q_tables",
    "build_tables",
    "dsa_sparse_attention",
    "dsa_sparse_attention_xla",
    "expand_open_mask",
    "pack_pool_bits",
    "parse_sparse_mode",
    "unpack_pool_bits",
    "visited_open_mask",
)
