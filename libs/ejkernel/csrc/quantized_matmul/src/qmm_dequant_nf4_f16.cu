// Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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
//
// This file is auto-generated. See "code_gen.py".

#include "qmm_dequant_kernels.h"

void LaunchDequantNf4F16(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  dequant_nf4_int<4, half><<<grid, block, 0, stream>>>(
      wq, scales, out, K, N, n_words, group_size, n_groups);
}

void LaunchDequantNf4F16Gs16(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 16, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs32(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 32, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs64(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 64, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs128(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 128, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs256(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 256, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs512(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 512, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}
void LaunchDequantNf4F16Gs1024(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream) {
  (void)group_size;
  dequant_nf4_int_gs<4, 1024, half><<<grid, block, 0, stream>>>(wq, scales, out, K, N, n_words, n_groups);
}

