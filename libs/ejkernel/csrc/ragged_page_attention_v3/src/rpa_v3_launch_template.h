// Copyright 2025 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "rpa_v3.h"
#include "rpa_v3_kernel.h"

namespace rpa_v3 {

template <typename T, int HEAD_DIM>
void run_rpa_v3_update_kv_hdim(RpaV3Params &params, cudaStream_t stream) {
  if (params.total_tokens <= 0) {
    return;
  }
  dim3 grid(static_cast<uint32_t>(params.total_tokens),
            static_cast<uint32_t>(params.num_kv_heads), 1);
  dim3 block(1, 1, 1);

  rpa_v3_update_kv_kernel<T, HEAD_DIM>
      <<<grid, block, 0, stream>>>(
          static_cast<const T *>(params.keys),
          static_cast<const T *>(params.values), params.kv_lens,
          params.block_tables, params.query_start_loc, params.distribution,
          static_cast<T *>(params.kv_cache_out), params.total_tokens,
          params.num_kv_heads, params.head_dim_padded, params.page_size,
          params.pages_per_seq, params.num_kv_heads_x2_per_pack,
          params.kv_packing);
}

template <typename T, int HEAD_DIM>
void run_rpa_v3_attention_hdim(RpaV3Params &params, cudaStream_t stream) {
  if (params.total_tokens <= 0) {
    return;
  }
  dim3 grid(static_cast<uint32_t>(params.total_tokens),
            static_cast<uint32_t>(params.num_q_heads), 1);
  dim3 block(1, 1, 1);

  rpa_v3_attention_kernel<T, T, T, HEAD_DIM>
      <<<grid, block, 0, stream>>>(
          static_cast<const T *>(params.queries),
          static_cast<const T *>(params.kv_cache_out), params.kv_lens,
          params.block_tables, params.query_start_loc, params.distribution,
          params.softmax_aux, static_cast<T *>(params.out), params.total_tokens,
          params.num_q_heads, params.num_kv_heads, params.head_dim_padded,
          params.page_size, params.pages_per_seq, params.num_kv_heads_x2_per_pack,
          params.kv_packing, params.softmax_scale, params.softcap,
          params.use_sinks, params.sliding_window, params.use_q_scale,
          params.use_k_scale, params.use_v_scale, params.q_scale, params.k_scale,
          params.v_scale);
}

}  // namespace rpa_v3
