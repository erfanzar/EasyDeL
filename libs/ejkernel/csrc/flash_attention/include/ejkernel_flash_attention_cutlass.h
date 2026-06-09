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

#include <cstdint>
#include <cuda_runtime.h>

extern "C" int ejk_cutlass_fmha_fwd_sm80(
    const void *q, const void *k, const void *v, const void *bias, void *o,
    void *lse, int32_t batch, int32_t seq_len_q, int32_t seq_len_k,
    int32_t num_heads, int32_t num_kv_heads, int32_t head_dim,
    float softmax_scale, int32_t causal, int32_t is_bf16, int32_t block_m,
    int32_t block_n, int32_t force_aligned, cudaStream_t stream);

extern "C" int ejk_cutlass_fmha_fwd_sm90(
    const void *q, const void *k, const void *v, void *o, int32_t batch,
    int32_t seq_len_q, int32_t seq_len_k, int32_t num_heads,
    int32_t num_kv_heads, int32_t head_dim, float softmax_scale, int32_t causal,
    int32_t is_bf16, int32_t block_m, int32_t block_n, cudaStream_t stream);

extern "C" int ejk_cutlass_fmha_fwd_sm100(
    const void *q, const void *k, const void *v, void *o, int32_t batch,
    int32_t seq_len_q, int32_t seq_len_k, int32_t num_heads,
    int32_t num_kv_heads, int32_t head_dim, float softmax_scale, int32_t causal,
    int32_t is_bf16, int32_t block_m, int32_t block_n, cudaStream_t stream);
