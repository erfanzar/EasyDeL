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

#include "blocksparse_attention.h"
#include "blocksparse_attention_kernel.h"

namespace blocksparse {

template <typename T, int QK_HEAD_DIM, int V_HEAD_DIM>
void run_blocksparse_fwd_hdim(Blocksparse_fwd_params &params,
                              cudaStream_t stream) {
  dim3 grid(static_cast<uint32_t>(params.batch * params.q_len),
            static_cast<uint32_t>(params.num_heads), 1);
  dim3 block(static_cast<uint32_t>(params.block_dim), 1, 1);

  blocksparse_attention_kernel<T, T, T, QK_HEAD_DIM, V_HEAD_DIM>
      <<<grid, block, 0, stream>>>(
          static_cast<const T *>(params.query),
          static_cast<const T *>(params.key),
          static_cast<const T *>(params.value), params.q_positions,
          params.q_segment_ids, params.kv_positions, params.kv_segment_ids,
          params.lower_bounds, params.upper_bounds, params.softmax_aux,
          static_cast<T *>(params.out), params.batch, params.num_heads,
          params.num_kv_heads, params.q_len, params.kv_len, params.q_blocksize,
          params.kv_blocksize, params.num_q_blocks, params.softmax_scale,
          params.softcap, params.causal, params.window_left, params.window_right,
          params.use_sinks, params.num_sinks);
}

}  // namespace blocksparse
