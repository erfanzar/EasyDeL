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

#include "ua.h"
#include "ua_kernel.h"

namespace ua {

template <typename T, int HEAD_DIM>
void run_unified_attention_hdim(UaParams &params, cudaStream_t stream) {
  if (params.total_tokens <= 0) {
    return;
  }
  dim3 grid(static_cast<uint32_t>(params.total_tokens),
            static_cast<uint32_t>(params.num_q_heads), 1);
  dim3 block(static_cast<uint32_t>(params.block_dim), 1, 1);

  unified_attention_kernel_hdim<T, T, T, HEAD_DIM><<<grid, block, 0, stream>>>(
      static_cast<const T *>(params.queries),
      static_cast<const T *>(params.key_cache),
      static_cast<const T *>(params.value_cache), params.kv_lens,
      params.block_tables, params.query_start_loc, params.softmax_aux,
      static_cast<T *>(params.out), params.total_tokens, params.num_q_heads,
      params.num_kv_heads, params.block_size, params.max_blocks_per_seq,
      params.num_seqs, params.softmax_scale, params.softcap, params.use_sinks,
      params.sliding_window);
}

} // namespace ua
