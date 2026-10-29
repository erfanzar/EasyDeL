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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "rpa_v3.h"

namespace rpa_v3 {

constexpr float kHalfMax = 65504.0f;
constexpr float kHalfMin = -65504.0f;
constexpr float kBf16Max = 3.38953139e38f;
constexpr float kBf16Min = -3.38953139e38f;

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ half ToHalf(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __nv_bfloat16 ToBf16(float v) {
  return __float2bfloat16(v);
}

template <typename KVType>
__device__ __forceinline__ float QuantizeQ(float q, float q_scale) {
  if constexpr (std::is_same_v<KVType, half>) {
    float v = q / q_scale;
    v = fminf(fmaxf(v, kHalfMin), kHalfMax);
    return __half2float(__float2half_rn(v));
  } else if constexpr (std::is_same_v<KVType, __nv_bfloat16>) {
    float v = q / q_scale;
    v = fminf(fmaxf(v, kBf16Min), kBf16Max);
    return __bfloat162float(__float2bfloat16(v));
  } else {
    return q / q_scale;
  }
}

__device__ __forceinline__ size_t KvIndex(
    int32_t page, int32_t page_offset, int32_t pack_idx, int32_t lane,
    int32_t d, int32_t page_size, int32_t num_kv_heads_x2_per_pack,
    int32_t kv_packing, int32_t head_dim_padded) {
  size_t idx = static_cast<size_t>(page) * page_size + page_offset;
  idx = idx * num_kv_heads_x2_per_pack + pack_idx;
  idx = idx * kv_packing + lane;
  idx = idx * head_dim_padded + d;
  return idx;
}

static __device__ __forceinline__ int find_seq_idx(int token_idx,
                                                   const int32_t *q_start,
                                                   int32_t num_seqs) {
  int32_t left = 0;
  int32_t right = num_seqs;
  while (left < right) {
    int32_t mid = (left + right) >> 1;
    int32_t v = q_start[mid];
    if (v <= token_idx) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

template <typename KVType, int HEAD_DIM>
__global__ void rpa_v3_update_kv_kernel(
    const KVType *keys, const KVType *values, const int32_t *kv_lens,
    const int32_t *block_tables, const int32_t *query_start_loc,
    const int32_t *distribution, KVType *kv_cache_out, int32_t total_tokens,
    int32_t num_kv_heads, int32_t head_dim_padded, int32_t page_size,
    int32_t pages_per_seq, int32_t num_kv_heads_x2_per_pack,
    int32_t kv_packing) {
  static_assert(HEAD_DIM <= kMaxHeadDim, "Unsupported head dim.");

  int token = static_cast<int>(blockIdx.x);
  int kv_head = static_cast<int>(blockIdx.y);
  if (token >= total_tokens || kv_head >= num_kv_heads || threadIdx.x != 0) {
    return;
  }
  int32_t num_seqs = distribution[2];
  if (num_seqs <= 0) {
    return;
  }
  int seq_idx = find_seq_idx(token, query_start_loc, num_seqs);
  if (seq_idx < 0 || seq_idx >= num_seqs) {
    return;
  }
  int32_t q_start = query_start_loc[seq_idx];
  int32_t q_end = query_start_loc[seq_idx + 1];
  int32_t q_len = q_end - q_start;
  if (q_len <= 0) {
    return;
  }
  int32_t q_pos = token - q_start;
  if (q_pos < 0 || q_pos >= q_len) {
    return;
  }
  int32_t kv_len = kv_lens[seq_idx];
  int32_t write_pos = kv_len - q_len + q_pos;
  if (write_pos < 0 || write_pos >= kv_len) {
    return;
  }

  int32_t page_idx = write_pos / page_size;
  int32_t page_off = write_pos - page_idx * page_size;
  int32_t page = block_tables[seq_idx * pages_per_seq + page_idx];

  int32_t head_idx_k = 2 * kv_head;
  int32_t head_idx_v = head_idx_k + 1;
  int32_t pack_k = head_idx_k / kv_packing;
  int32_t lane_k = head_idx_k - pack_k * kv_packing;
  int32_t pack_v = head_idx_v / kv_packing;
  int32_t lane_v = head_idx_v - pack_v * kv_packing;

  size_t base_k = KvIndex(page, page_off, pack_k, lane_k, 0, page_size,
                          num_kv_heads_x2_per_pack, kv_packing, head_dim_padded);
  size_t base_v = KvIndex(page, page_off, pack_v, lane_v, 0, page_size,
                          num_kv_heads_x2_per_pack, kv_packing, head_dim_padded);
  size_t kv_base =
      (static_cast<size_t>(token) * num_kv_heads + kv_head) * HEAD_DIM;

  for (int d = 0; d < HEAD_DIM; ++d) {
    kv_cache_out[base_k + d] = keys[kv_base + d];
    kv_cache_out[base_v + d] = values[kv_base + d];
  }
}

template <typename QType, typename KVType, typename OutType, int HEAD_DIM>
__global__ void rpa_v3_attention_kernel(
    const QType *queries, const KVType *kv_cache, const int32_t *kv_lens,
    const int32_t *block_tables, const int32_t *query_start_loc,
    const int32_t *distribution, const float *softmax_aux, OutType *out,
    int32_t total_tokens, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_dim_padded, int32_t page_size, int32_t pages_per_seq,
    int32_t num_kv_heads_x2_per_pack, int32_t kv_packing, float softmax_scale,
    float softcap, int32_t use_sinks, int32_t sliding_window,
    int32_t use_q_scale, int32_t use_k_scale, int32_t use_v_scale,
    float q_scale, float k_scale, float v_scale) {
  static_assert(HEAD_DIM <= kMaxHeadDim, "Unsupported head dim.");

  int token = static_cast<int>(blockIdx.x);
  int q_head = static_cast<int>(blockIdx.y);
  if (token >= total_tokens || q_head >= num_q_heads || threadIdx.x != 0) {
    return;
  }

  int32_t num_seqs = distribution[2];
  if (num_seqs <= 0) {
    return;
  }
  int32_t num_queries_per_kv = num_q_heads / num_kv_heads;
  int32_t kv_head = q_head / num_queries_per_kv;

  int seq_idx = find_seq_idx(token, query_start_loc, num_seqs);
  if (seq_idx < 0 || seq_idx >= num_seqs) {
    return;
  }
  int32_t q_start = query_start_loc[seq_idx];
  int32_t q_end = query_start_loc[seq_idx + 1];
  int32_t q_len = q_end - q_start;
  if (q_len <= 0) {
    return;
  }
  int32_t q_pos = token - q_start;
  if (q_pos < 0 || q_pos >= q_len) {
    return;
  }

  int32_t kv_len = kv_lens[seq_idx];
  int32_t context_len = kv_len - q_len;
  if (context_len < 0) {
    return;
  }
  int32_t q_abs = context_len + q_pos;

  int32_t left_bound = 0;
  if (sliding_window >= 0) {
    left_bound = q_abs - sliding_window + 1;
    if (left_bound < 0) {
      left_bound = 0;
    }
  }

  float q_buf[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; ++d) {
    size_t q_idx =
        (static_cast<size_t>(token) * num_q_heads + q_head) * HEAD_DIM + d;
    float q_val = ToFloat(queries[q_idx]);
    if (use_q_scale) {
      q_val = QuantizeQ<KVType>(q_val, q_scale);
    }
    q_buf[d] = q_val;
  }

  float acc[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; ++d) {
    acc[d] = 0.0f;
  }

  float m = -INFINITY;
  float l = 0.0f;
  if (use_sinks) {
    m = softmax_aux[q_head];
    l = 1.0f;
  }

  int32_t head_idx_k = 2 * kv_head;
  int32_t head_idx_v = head_idx_k + 1;
  int32_t pack_k = head_idx_k / kv_packing;
  int32_t lane_k = head_idx_k - pack_k * kv_packing;
  int32_t pack_v = head_idx_v / kv_packing;
  int32_t lane_v = head_idx_v - pack_v * kv_packing;

  for (int32_t kv_pos = 0; kv_pos < kv_len; ++kv_pos) {
    if (kv_pos > q_abs) {
      continue;
    }
    if (sliding_window >= 0 && kv_pos < left_bound) {
      continue;
    }

    int32_t page_idx = kv_pos / page_size;
    int32_t page_off = kv_pos - page_idx * page_size;
    int32_t page = block_tables[seq_idx * pages_per_seq + page_idx];

    size_t base_k = KvIndex(page, page_off, pack_k, lane_k, 0, page_size,
                            num_kv_heads_x2_per_pack, kv_packing,
                            head_dim_padded);
    size_t base_v = KvIndex(page, page_off, pack_v, lane_v, 0, page_size,
                            num_kv_heads_x2_per_pack, kv_packing,
                            head_dim_padded);

    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; ++d) {
      float k_val = ToFloat(kv_cache[base_k + d]);
      dot += q_buf[d] * k_val;
    }

    float logit = dot * softmax_scale;
    if (use_k_scale) {
      logit *= k_scale;
    }
    if (use_q_scale) {
      logit *= q_scale;
    }
    if (softcap > 0.0f) {
      logit = softcap * tanhf(logit / softcap);
    }

    float new_m = fmaxf(m, logit);
    float exp_m = expf(m - new_m);
    float exp_l = expf(logit - new_m);
    l = exp_m * l + exp_l;
    float rescale = exp_m;
    for (int d = 0; d < HEAD_DIM; ++d) {
      float v_val = ToFloat(kv_cache[base_v + d]);
      acc[d] = acc[d] * rescale + exp_l * v_val;
    }
    m = new_m;
  }

  if (l < 1e-6f) {
    l = 1e-6f;
  }

  for (int d = 0; d < HEAD_DIM; ++d) {
    float out_val = acc[d] / l;
    if (use_v_scale) {
      out_val *= v_scale;
    }
    size_t out_idx =
        (static_cast<size_t>(token) * num_q_heads + q_head) * HEAD_DIM + d;
    if constexpr (std::is_same_v<OutType, half>) {
      out[out_idx] = ToHalf(out_val);
    } else if constexpr (std::is_same_v<OutType, __nv_bfloat16>) {
      out[out_idx] = ToBf16(out_val);
    } else {
      out[out_idx] = static_cast<float>(out_val);
    }
  }
}

}  // namespace rpa_v3
