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

"""TileLang parity tests for multi_latent_ragged_page_attention_v2."""

from __future__ import annotations

from ._helpers import (
    _FP16_FWD_TOL,
    _SEED,
    _make_mla_ragged_inputs,
    _max_abs,
    _tl,
    _xla,
)


def test_multi_latent_ragged_page_attention_v2_native():
    args = _make_mla_ragged_inputs(_SEED + 7)
    (
        queries_nope,
        queries_pe,
        keys_values,
        keys_pe,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
    ) = args[:-2]
    kwargs = {
        "softmax_scale": 0.25,
        "sliding_window": 4,
        "logits_soft_cap": 3.0,
        "q_scale": 0.75,
        "k_scale": 1.25,
        "v_scale": 0.9,
        "num_kv_pages_per_block": (1, 1, 2),
        "num_queries_per_block": (8, 8, 16),
    }
    common = (queries_nope, queries_pe, keys_values, keys_pe, kv_lens, block_tables, query_start_loc, distribution)
    out_xla, cache_xla = _xla("multi_latent_ragged_page_attention_v2")(
        common[0], common[1], common[2], common[3], kv_cache.copy(), *common[4:], **kwargs
    )
    out_tl, cache_tl = _tl("multi_latent_ragged_page_attention_v2")(
        common[0], common[1], common[2], common[3], kv_cache.copy(), *common[4:], **kwargs
    )
    assert _max_abs(out_tl, out_xla) < 8e-2
    assert _max_abs(cache_tl, cache_xla) < _FP16_FWD_TOL
