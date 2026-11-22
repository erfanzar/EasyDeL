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

namespace ua {

constexpr int kMaxBlockDim = 256;
constexpr int kMaxHeadDim = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kMaxBlockDim / kWarpSize;

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ half ToHalf(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __nv_bfloat16 ToBf16(float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ float WarpReduceSum(float v) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffff, v, offset);
  }
  return v;
}

__device__ __forceinline__ int
find_seq_idx(int token_idx, const int32_t *q_start, int32_t num_seqs) {
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

template <typename QType, typename KVType, typename OutType>
__global__ void unified_attention_kernel_rt(
    const QType *queries, const KVType *key_cache, const KVType *value_cache,
    const int32_t *kv_lens, const int32_t *block_tables,
    const int32_t *query_start_loc, const float *softmax_aux, OutType *out,
    int32_t total_tokens, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t head_dim, int32_t block_size, int32_t max_blocks_per_seq,
    int32_t num_seqs, float softmax_scale, float softcap, int32_t use_sinks,
    int32_t sliding_window) {
  int token = static_cast<int>(blockIdx.x);
  int q_head = static_cast<int>(blockIdx.y);
  int tid = static_cast<int>(threadIdx.x);

  if (token >= total_tokens || q_head >= num_q_heads) {
    return;
  }
  if (head_dim > kMaxHeadDim) {
    return;
  }

  int32_t num_queries_per_kv = num_q_heads / num_kv_heads;
  int32_t kv_head = q_head / num_queries_per_kv;

  int32_t seq_idx = find_seq_idx(token, query_start_loc, num_seqs);
  if (seq_idx < 0) {
    return;
  }
  int32_t q_start = query_start_loc[seq_idx];
  int32_t q_end = query_start_loc[seq_idx + 1];
  int32_t q_len = q_end - q_start;
  int32_t q_pos = token - q_start;
  if (q_pos < 0 || q_pos >= q_len) {
    return;
  }

  int32_t seq_len = kv_lens[seq_idx];
  int32_t context_len = seq_len - q_len;
  if (context_len < 0) {
    return;
  }
  int32_t max_kv = context_len + q_pos;
  if (max_kv < 0) {
    return;
  }

  __shared__ float q_sh[kMaxHeadDim];
  __shared__ float acc_sh[kMaxHeadDim];
  __shared__ float k_sh[kMaxHeadDim];
  __shared__ float v_sh[kMaxHeadDim];
  __shared__ float warp_sums[kMaxWarps];
  __shared__ float m_shared;
  __shared__ float l_shared;
  __shared__ float exp_m_shared;
  __shared__ float exp_s_shared;
  __shared__ float denom_shared;

  int block_dim = blockDim.x;
  if constexpr (std::is_same_v<QType, half>) {
    for (int d = tid * 2; d + 1 < head_dim; d += block_dim * 2) {
      int64_t q_idx =
          (static_cast<int64_t>(token) * num_q_heads + q_head) * head_dim + d;
      const half2 *q2_ptr = reinterpret_cast<const half2 *>(queries + q_idx);
      half2 q2 = *q2_ptr;
      q_sh[d] = __half2float(__low2half(q2));
      q_sh[d + 1] = __half2float(__high2half(q2));
      acc_sh[d] = 0.0f;
      acc_sh[d + 1] = 0.0f;
    }
    if ((head_dim & 1) && tid == 0) {
      int d = head_dim - 1;
      int64_t q_idx =
          (static_cast<int64_t>(token) * num_q_heads + q_head) * head_dim + d;
      q_sh[d] = ToFloat(queries[q_idx]);
      acc_sh[d] = 0.0f;
    }
  } else {
    for (int d = tid; d < head_dim; d += block_dim) {
      int64_t q_idx =
          (static_cast<int64_t>(token) * num_q_heads + q_head) * head_dim + d;
      q_sh[d] = ToFloat(queries[q_idx]);
      acc_sh[d] = 0.0f;
    }
  }
  if (tid == 0) {
    float m_sh = -INFINITY;
    float l_sh = 0.0f;
    if (use_sinks) {
      m_sh = softmax_aux[q_head];
      l_sh = 1.0f;
    }
    m_shared = m_sh;
    l_shared = l_sh;
  }
  __syncthreads();

  int32_t min_kv = 0;
  if (sliding_window > 0) {
    int32_t win_start = max_kv - sliding_window + 1;
    min_kv = win_start > 0 ? win_start : 0;
  }
  for (int32_t kv_pos = min_kv; kv_pos <= max_kv; ++kv_pos) {
    int32_t logical_block = kv_pos / block_size;
    int32_t block_off = kv_pos - logical_block * block_size;
    if (logical_block >= max_blocks_per_seq) {
      break;
    }
    int32_t phys_block =
        block_tables[seq_idx * max_blocks_per_seq + logical_block];
    if (phys_block < 0) {
      continue;
    }

    if constexpr (std::is_same_v<KVType, half>) {
      for (int d = tid * 2; d + 1 < head_dim; d += block_dim * 2) {
        int64_t k_idx =
            (((static_cast<int64_t>(phys_block) * block_size + block_off) *
                  num_kv_heads +
              kv_head) *
                 head_dim +
             d);
        const half2 *k2_ptr =
            reinterpret_cast<const half2 *>(key_cache + k_idx);
        const half2 *v2_ptr =
            reinterpret_cast<const half2 *>(value_cache + k_idx);
        half2 k2 = *k2_ptr;
        half2 v2 = *v2_ptr;
        k_sh[d] = __half2float(__low2half(k2));
        k_sh[d + 1] = __half2float(__high2half(k2));
        v_sh[d] = __half2float(__low2half(v2));
        v_sh[d + 1] = __half2float(__high2half(v2));
      }
      if ((head_dim & 1) && tid == 0) {
        int d = head_dim - 1;
        int64_t k_idx =
            (((static_cast<int64_t>(phys_block) * block_size + block_off) *
                  num_kv_heads +
              kv_head) *
                 head_dim +
             d);
        k_sh[d] = ToFloat(key_cache[k_idx]);
        v_sh[d] = ToFloat(value_cache[k_idx]);
      }
    } else {
      for (int d = tid; d < head_dim; d += block_dim) {
        int64_t k_idx =
            (((static_cast<int64_t>(phys_block) * block_size + block_off) *
                  num_kv_heads +
              kv_head) *
                 head_dim +
             d);
        k_sh[d] = ToFloat(key_cache[k_idx]);
        v_sh[d] = ToFloat(value_cache[k_idx]);
      }
    }
    __syncthreads();

    float partial = 0.0f;
    for (int d = tid; d < head_dim; d += block_dim) {
      partial += q_sh[d] * k_sh[d];
    }
    float warp_sum = WarpReduceSum(partial);
    int lane = tid & (kWarpSize - 1);
    int warp_id = tid / kWarpSize;
    if (lane == 0) {
      warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
      int num_warps = (block_dim + kWarpSize - 1) / kWarpSize;
      float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
      block_sum = WarpReduceSum(block_sum);
      if (lane == 0) {
        float score = block_sum * softmax_scale;
        if (softcap > 0.0f) {
          score = softcap * tanhf(score / softcap);
        }
        float m_prev = m_shared;
        float m_new = fmaxf(m_prev, score);
        float exp_m = expf(m_prev - m_new);
        float exp_s = expf(score - m_new);
        float l_new = l_shared * exp_m + exp_s;
        m_shared = m_new;
        l_shared = l_new;
        exp_m_shared = exp_m;
        exp_s_shared = exp_s;
      }
    }
    __syncthreads();

    float exp_m = exp_m_shared;
    float exp_s = exp_s_shared;
    for (int d = tid; d < head_dim; d += block_dim) {
      acc_sh[d] = acc_sh[d] * exp_m + v_sh[d] * exp_s;
    }
    __syncthreads();
  }

  if (tid == 0) {
    denom_shared = l_shared > 0.0f ? l_shared : 1.0f;
  }
  __syncthreads();

  for (int d = tid; d < head_dim; d += block_dim) {
    float out_val = acc_sh[d] / denom_shared;
    int64_t o_idx =
        (static_cast<int64_t>(token) * num_q_heads + q_head) * head_dim + d;
    if constexpr (std::is_same_v<OutType, half>) {
      out[o_idx] = ToHalf(out_val);
    } else {
      out[o_idx] = ToBf16(out_val);
    }
  }
}

template <typename QType, typename KVType, typename OutType, int HEAD_DIM>
__global__ void unified_attention_kernel_hdim(
    const QType *queries, const KVType *key_cache, const KVType *value_cache,
    const int32_t *kv_lens, const int32_t *block_tables,
    const int32_t *query_start_loc, const float *softmax_aux, OutType *out,
    int32_t total_tokens, int32_t num_q_heads, int32_t num_kv_heads,
    int32_t block_size, int32_t max_blocks_per_seq, int32_t num_seqs,
    float softmax_scale, float softcap, int32_t use_sinks,
    int32_t sliding_window) {
  static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive.");
  static_assert(HEAD_DIM <= kMaxHeadDim, "HEAD_DIM exceeds maximum.");

  int token = static_cast<int>(blockIdx.x);
  int q_head = static_cast<int>(blockIdx.y);
  int tid = static_cast<int>(threadIdx.x);

  if (token >= total_tokens || q_head >= num_q_heads) {
    return;
  }

  int32_t num_queries_per_kv = num_q_heads / num_kv_heads;
  int32_t kv_head = q_head / num_queries_per_kv;

  int32_t seq_idx = find_seq_idx(token, query_start_loc, num_seqs);
  if (seq_idx < 0) {
    return;
  }
  int32_t q_start = query_start_loc[seq_idx];
  int32_t q_end = query_start_loc[seq_idx + 1];
  int32_t q_len = q_end - q_start;
  int32_t q_pos = token - q_start;
  if (q_pos < 0 || q_pos >= q_len) {
    return;
  }

  int32_t seq_len = kv_lens[seq_idx];
  int32_t context_len = seq_len - q_len;
  if (context_len < 0) {
    return;
  }
  int32_t max_kv = context_len + q_pos;
  if (max_kv < 0) {
    return;
  }

  __shared__ float q_sh[HEAD_DIM];
  __shared__ float acc_sh[HEAD_DIM];
  __shared__ float k_sh[HEAD_DIM];
  __shared__ float v_sh[HEAD_DIM];
  __shared__ float warp_sums[kMaxWarps];
  __shared__ float m_shared;
  __shared__ float l_shared;
  __shared__ float exp_m_shared;
  __shared__ float exp_s_shared;
  __shared__ float denom_shared;

  int block_dim = blockDim.x;
  if constexpr (std::is_same_v<QType, half>) {
    for (int d = tid * 2; d + 1 < HEAD_DIM; d += block_dim * 2) {
      int64_t q_idx =
          (static_cast<int64_t>(token) * num_q_heads + q_head) * HEAD_DIM + d;
      const half2 *q2_ptr = reinterpret_cast<const half2 *>(queries + q_idx);
      half2 q2 = *q2_ptr;
      q_sh[d] = __half2float(__low2half(q2));
      q_sh[d + 1] = __half2float(__high2half(q2));
      acc_sh[d] = 0.0f;
      acc_sh[d + 1] = 0.0f;
    }
    if constexpr ((HEAD_DIM & 1) != 0) {
      if (tid == 0) {
        int d = HEAD_DIM - 1;
        int64_t q_idx =
            (static_cast<int64_t>(token) * num_q_heads + q_head) * HEAD_DIM + d;
        q_sh[d] = ToFloat(queries[q_idx]);
        acc_sh[d] = 0.0f;
      }
    }
  } else {
    for (int d = tid; d < HEAD_DIM; d += block_dim) {
      int64_t q_idx =
          (static_cast<int64_t>(token) * num_q_heads + q_head) * HEAD_DIM + d;
      q_sh[d] = ToFloat(queries[q_idx]);
      acc_sh[d] = 0.0f;
    }
  }
  if (tid == 0) {
    float m_sh = -INFINITY;
    float l_sh = 0.0f;
    if (use_sinks) {
      m_sh = softmax_aux[q_head];
      l_sh = 1.0f;
    }
    m_shared = m_sh;
    l_shared = l_sh;
  }
  __syncthreads();

  int32_t min_kv = 0;
  if (sliding_window > 0) {
    int32_t win_start = max_kv - sliding_window + 1;
    min_kv = win_start > 0 ? win_start : 0;
  }
  for (int32_t kv_pos = min_kv; kv_pos <= max_kv; ++kv_pos) {
    int32_t logical_block = kv_pos / block_size;
    int32_t block_off = kv_pos - logical_block * block_size;
    if (logical_block >= max_blocks_per_seq) {
      break;
    }
    int32_t phys_block =
        block_tables[seq_idx * max_blocks_per_seq + logical_block];
    if (phys_block < 0) {
      continue;
    }

    if constexpr (std::is_same_v<KVType, half>) {
      for (int d = tid * 2; d + 1 < HEAD_DIM; d += block_dim * 2) {
        int64_t k_idx =
            (((static_cast<int64_t>(phys_block) * block_size + block_off) *
                  num_kv_heads +
              kv_head) *
                 HEAD_DIM +
             d);
        const half2 *k2_ptr =
            reinterpret_cast<const half2 *>(key_cache + k_idx);
        const half2 *v2_ptr =
            reinterpret_cast<const half2 *>(value_cache + k_idx);
        half2 k2 = *k2_ptr;
        half2 v2 = *v2_ptr;
        k_sh[d] = __half2float(__low2half(k2));
        k_sh[d + 1] = __half2float(__high2half(k2));
        v_sh[d] = __half2float(__low2half(v2));
        v_sh[d + 1] = __half2float(__high2half(v2));
      }
      if constexpr ((HEAD_DIM & 1) != 0) {
        if (tid == 0) {
          int d = HEAD_DIM - 1;
          int64_t k_idx =
              (((static_cast<int64_t>(phys_block) * block_size + block_off) *
                    num_kv_heads +
                kv_head) *
                   HEAD_DIM +
               d);
          k_sh[d] = ToFloat(key_cache[k_idx]);
          v_sh[d] = ToFloat(value_cache[k_idx]);
        }
      }
    } else {
      for (int d = tid; d < HEAD_DIM; d += block_dim) {
        int64_t k_idx =
            (((static_cast<int64_t>(phys_block) * block_size + block_off) *
                  num_kv_heads +
              kv_head) *
                 HEAD_DIM +
             d);
        k_sh[d] = ToFloat(key_cache[k_idx]);
        v_sh[d] = ToFloat(value_cache[k_idx]);
      }
    }
    __syncthreads();

    float partial = 0.0f;
    for (int d = tid; d < HEAD_DIM; d += block_dim) {
      partial += q_sh[d] * k_sh[d];
    }
    float warp_sum = WarpReduceSum(partial);
    int lane = tid & (kWarpSize - 1);
    int warp_id = tid / kWarpSize;
    if (lane == 0) {
      warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    if (warp_id == 0) {
      int num_warps = (block_dim + kWarpSize - 1) / kWarpSize;
      float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
      block_sum = WarpReduceSum(block_sum);
      if (lane == 0) {
        float score = block_sum * softmax_scale;
        if (softcap > 0.0f) {
          score = softcap * tanhf(score / softcap);
        }
        float m_prev = m_shared;
        float m_new = fmaxf(m_prev, score);
        float exp_m = expf(m_prev - m_new);
        float exp_s = expf(score - m_new);
        float l_new = l_shared * exp_m + exp_s;
        m_shared = m_new;
        l_shared = l_new;
        exp_m_shared = exp_m;
        exp_s_shared = exp_s;
      }
    }
    __syncthreads();

    float exp_m = exp_m_shared;
    float exp_s = exp_s_shared;
    for (int d = tid; d < HEAD_DIM; d += block_dim) {
      acc_sh[d] = acc_sh[d] * exp_m + v_sh[d] * exp_s;
    }
    __syncthreads();
  }

  if (tid == 0) {
    denom_shared = l_shared > 0.0f ? l_shared : 1.0f;
  }
  __syncthreads();

  for (int d = tid; d < HEAD_DIM; d += block_dim) {
    float out_val = acc_sh[d] / denom_shared;
    int64_t o_idx =
        (static_cast<int64_t>(token) * num_q_heads + q_head) * HEAD_DIM + d;
    if constexpr (std::is_same_v<OutType, half>) {
      out[o_idx] = ToHalf(out_val);
    } else {
      out[o_idx] = ToBf16(out_val);
    }
  }
}

} // namespace ua
