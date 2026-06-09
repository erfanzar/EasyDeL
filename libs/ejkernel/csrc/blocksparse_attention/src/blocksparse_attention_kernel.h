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

#include "blocksparse_attention.h"

namespace blocksparse {

__device__ __forceinline__ float ToFloat(float v) { return v; }
__device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloat(__nv_bfloat16 v) {
  return __bfloat162float(v);
}

__device__ __forceinline__ half ToHalf(float v) { return __float2half_rn(v); }
__device__ __forceinline__ __nv_bfloat16 ToBf16(float v) {
  return __float2bfloat16(v);
}

template <typename QType, typename KVType, typename OutType, int QK_HEAD_DIM,
          int V_HEAD_DIM>
__global__ void blocksparse_attention_kernel(
    const QType *query, const KVType *key, const KVType *value,
    const int32_t *q_positions, const int32_t *q_segment_ids,
    const int32_t *kv_positions, const int32_t *kv_segment_ids,
    const int32_t *lower_bounds, const int32_t *upper_bounds,
    const float *softmax_aux, OutType *out, int32_t batch, int32_t num_heads,
    int32_t num_kv_heads, int32_t q_len, int32_t kv_len, int32_t q_blocksize,
    int32_t kv_blocksize, int32_t num_q_blocks, float softmax_scale,
    float softcap, int32_t causal, int32_t window_left, int32_t window_right,
    int32_t use_sinks, int32_t num_sinks) {
  static_assert(QK_HEAD_DIM <= kMaxHeadDim, "Unsupported qk head dim.");
  static_assert(V_HEAD_DIM <= kMaxVHeadDim, "Unsupported v head dim.");

  int32_t token = static_cast<int32_t>(blockIdx.x);
  int32_t q_head = static_cast<int32_t>(blockIdx.y);
  int32_t tid = static_cast<int32_t>(threadIdx.x);

  if (tid != 0) {
    return;
  }

  int32_t total_tokens = batch * q_len;
  if (token >= total_tokens || q_head >= num_heads) {
    return;
  }

  int32_t b = token / q_len;
  int32_t q_idx = token - b * q_len;
  if (b >= batch || q_idx >= q_len) {
    return;
  }

  int32_t q_seg = q_segment_ids[b * q_len + q_idx];
  if (q_seg < 0) {
    for (int d = 0; d < V_HEAD_DIM; ++d) {
      int64_t o_idx =
          ((static_cast<int64_t>(b) * num_heads + q_head) * q_len + q_idx) *
              V_HEAD_DIM +
          d;
      if constexpr (std::is_same_v<OutType, half>) {
        out[o_idx] = ToHalf(0.0f);
      } else {
        out[o_idx] = ToBf16(0.0f);
      }
    }
    return;
  }

  int32_t q_pos = q_positions[b * q_len + q_idx];
  int32_t q_block = q_idx / q_blocksize;
  if (q_block < 0 || q_block >= num_q_blocks) {
    return;
  }

  int32_t lb = lower_bounds[b * num_q_blocks + q_block];
  int32_t ub = upper_bounds[b * num_q_blocks + q_block];
  if (lb >= ub) {
    for (int d = 0; d < V_HEAD_DIM; ++d) {
      int64_t o_idx =
          ((static_cast<int64_t>(b) * num_heads + q_head) * q_len + q_idx) *
              V_HEAD_DIM +
          d;
      if constexpr (std::is_same_v<OutType, half>) {
        out[o_idx] = ToHalf(0.0f);
      } else {
        out[o_idx] = ToBf16(0.0f);
      }
    }
    return;
  }

  int32_t kv_start = lb * kv_blocksize;
  int32_t kv_end = ub * kv_blocksize;
  kv_start = kv_start < 0 ? 0 : kv_start;
  kv_end = kv_end > kv_len ? kv_len : kv_end;
  if (kv_start >= kv_end) {
    for (int d = 0; d < V_HEAD_DIM; ++d) {
      int64_t o_idx =
          ((static_cast<int64_t>(b) * num_heads + q_head) * q_len + q_idx) *
              V_HEAD_DIM +
          d;
      if constexpr (std::is_same_v<OutType, half>) {
        out[o_idx] = ToHalf(0.0f);
      } else {
        out[o_idx] = ToBf16(0.0f);
      }
    }
    return;
  }

  int32_t num_queries_per_kv = num_heads / num_kv_heads;
  int32_t kv_head = q_head / num_queries_per_kv;

  float q_buf[kMaxHeadDim];
  float acc[kMaxVHeadDim];

  for (int d = 0; d < QK_HEAD_DIM; ++d) {
    int64_t q_idx_offset =
        ((static_cast<int64_t>(b) * num_heads + q_head) * q_len + q_idx) *
            QK_HEAD_DIM +
        d;
    q_buf[d] = ToFloat(query[q_idx_offset]);
  }
  for (int d = 0; d < V_HEAD_DIM; ++d) {
    acc[d] = 0.0f;
  }

  bool use_window = (window_left >= 0) || (window_right >= 0);
  int32_t wl = window_left < 0 ? 0x7fffffff : window_left;
  int32_t wr = window_right < 0 ? 0x7fffffff : window_right;

  float max_logit = -INFINITY;
  if (use_sinks && num_sinks > 0) {
    for (int s = 0; s < num_sinks; ++s) {
      float aux = softmax_aux[q_head * num_sinks + s];
      max_logit = aux > max_logit ? aux : max_logit;
    }
  }

  for (int32_t kv_idx = kv_start; kv_idx < kv_end; ++kv_idx) {
    int32_t kv_seg = kv_segment_ids[b * kv_len + kv_idx];
    if (kv_seg < 0 || kv_seg != q_seg) {
      continue;
    }
    int32_t kv_pos = kv_positions[b * kv_len + kv_idx];
    if (causal && kv_pos > q_pos) {
      continue;
    }
    if (use_window) {
      if (kv_pos < q_pos - wl || kv_pos > q_pos + wr) {
        continue;
      }
    }

    float dot = 0.0f;
    for (int d = 0; d < QK_HEAD_DIM; ++d) {
      int64_t k_idx_offset =
          ((static_cast<int64_t>(b) * num_kv_heads + kv_head) * kv_len +
           kv_idx) *
              QK_HEAD_DIM +
          d;
      dot += q_buf[d] * ToFloat(key[k_idx_offset]);
    }
    float scaled = dot * softmax_scale;
    if (softcap > 0.0f) {
      scaled = softcap * tanhf(scaled / softcap);
    }
    if (scaled > max_logit) {
      max_logit = scaled;
    }
  }

  if (max_logit == -INFINITY) {
    for (int d = 0; d < V_HEAD_DIM; ++d) {
      int64_t o_idx =
          ((static_cast<int64_t>(b) * num_heads + q_head) * q_len + q_idx) *
              V_HEAD_DIM +
          d;
      if constexpr (std::is_same_v<OutType, half>) {
        out[o_idx] = ToHalf(0.0f);
      } else {
        out[o_idx] = ToBf16(0.0f);
      }
    }
    return;
  }

  float denom = 0.0f;
  if (use_sinks && num_sinks > 0) {
    for (int s = 0; s < num_sinks; ++s) {
      float aux = softmax_aux[q_head * num_sinks + s];
      denom += expf(aux - max_logit);
    }
  }

  for (int32_t kv_idx = kv_start; kv_idx < kv_end; ++kv_idx) {
    int32_t kv_seg = kv_segment_ids[b * kv_len + kv_idx];
    if (kv_seg < 0 || kv_seg != q_seg) {
      continue;
    }
    int32_t kv_pos = kv_positions[b * kv_len + kv_idx];
    if (causal && kv_pos > q_pos) {
      continue;
    }
    if (use_window) {
      if (kv_pos < q_pos - wl || kv_pos > q_pos + wr) {
        continue;
      }
    }

    float dot = 0.0f;
    for (int d = 0; d < QK_HEAD_DIM; ++d) {
      int64_t k_idx_offset =
          ((static_cast<int64_t>(b) * num_kv_heads + kv_head) * kv_len +
           kv_idx) *
              QK_HEAD_DIM +
          d;
      dot += q_buf[d] * ToFloat(key[k_idx_offset]);
    }
    float scaled = dot * softmax_scale;
    if (softcap > 0.0f) {
      scaled = softcap * tanhf(scaled / softcap);
    }
    float w = expf(scaled - max_logit);
    denom += w;

    for (int d = 0; d < V_HEAD_DIM; ++d) {
      int64_t v_idx_offset =
          ((static_cast<int64_t>(b) * num_kv_heads + kv_head) * kv_len +
           kv_idx) *
              V_HEAD_DIM +
          d;
      acc[d] += w * ToFloat(value[v_idx_offset]);
    }
  }

  float inv_denom = denom > 0.0f ? 1.0f / denom : 0.0f;
  for (int d = 0; d < V_HEAD_DIM; ++d) {
    float out_val = acc[d] * inv_denom;
    int64_t o_idx =
        ((static_cast<int64_t>(b) * num_heads + q_head) * q_len + q_idx) *
            V_HEAD_DIM +
        d;
    if constexpr (std::is_same_v<OutType, half>) {
      out[o_idx] = ToHalf(out_val);
    } else {
      out[o_idx] = ToBf16(out_val);
    }
  }
}

}  // namespace blocksparse
