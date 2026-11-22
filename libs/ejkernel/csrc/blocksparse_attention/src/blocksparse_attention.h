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

#include <cuda_runtime.h>

#include <cstdint>

namespace blocksparse {

constexpr int kMaxHeadDim = 256;
constexpr int kMaxVHeadDim = 256;
constexpr int kMaxBlockDim = 256;
constexpr int kWarpSize = 32;

struct Blocksparse_fwd_params {
  const void *query;
  const void *key;
  const void *value;
  const int32_t *q_positions;
  const int32_t *q_segment_ids;
  const int32_t *kv_positions;
  const int32_t *kv_segment_ids;
  const int32_t *lower_bounds;
  const int32_t *upper_bounds;
  const float *softmax_aux;
  void *out;
  int32_t batch;
  int32_t num_heads;
  int32_t num_kv_heads;
  int32_t q_len;
  int32_t kv_len;
  int32_t q_blocksize;
  int32_t kv_blocksize;
  int32_t num_q_blocks;
  float softmax_scale;
  float softcap;
  int32_t causal;
  int32_t window_left;
  int32_t window_right;
  int32_t use_sinks;
  int32_t num_sinks;
  int32_t block_dim;
};

template <typename T, int QK_HEAD_DIM, int V_HEAD_DIM>
void run_blocksparse_fwd_(Blocksparse_fwd_params &params,
                          cudaStream_t stream);

}  // namespace blocksparse
