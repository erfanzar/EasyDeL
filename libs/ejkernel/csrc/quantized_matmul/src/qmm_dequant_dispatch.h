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

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

void LaunchDequantAffineBits1F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits1BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits2BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits3BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits4BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits5BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits6BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits7BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs16(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs32(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs64(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs128(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs256(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs512(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F32Gs1024(const uint32_t *wq, const float *scales, const float *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs16(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs32(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs64(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs128(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs256(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs512(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8F16Gs1024(const uint32_t *wq, const half *scales, const half *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantAffineBits8BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, const __nv_bfloat16 *biases, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs16(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs32(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs64(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs128(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs256(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs512(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F32Gs1024(const uint32_t *wq, const float *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs16(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs32(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs64(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs128(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs256(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs512(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4F16Gs1024(const uint32_t *wq, const half *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs16(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs32(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs64(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs128(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs256(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs512(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNf4BF16Gs1024(const uint32_t *wq, const __nv_bfloat16 *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t group_size, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantMxFp4(const uint32_t *wq, const uint8_t *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantMxFp8(const uint32_t *wq, const uint8_t *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNvFp4(const uint32_t *wq, const uint8_t *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);
void LaunchDequantNvFp8(const uint32_t *wq, const uint8_t *scales, half *out, int64_t K, int64_t N, int64_t n_words, int64_t n_groups, dim3 grid, dim3 block, cudaStream_t stream);

using DequantAffineF32Fn = void (*)(const uint32_t *, const float *, const float *, half *, int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);
using DequantAffineF16Fn = void (*)(const uint32_t *, const half *, const half *, half *, int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);
using DequantAffineBF16Fn = void (*)(const uint32_t *, const __nv_bfloat16 *, const __nv_bfloat16 *, half *, int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);
using DequantNf4F32Fn = void (*)(const uint32_t *, const float *, half *, int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);
using DequantNf4F16Fn = void (*)(const uint32_t *, const half *, half *, int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);
using DequantNf4BF16Fn = void (*)(const uint32_t *, const __nv_bfloat16 *, half *, int64_t, int64_t, int64_t, int64_t, int64_t, dim3, dim3, cudaStream_t);

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits1F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits1F32Gs16;
  case 32: return &LaunchDequantAffineBits1F32Gs32;
  case 64: return &LaunchDequantAffineBits1F32Gs64;
  case 128: return &LaunchDequantAffineBits1F32Gs128;
  case 256: return &LaunchDequantAffineBits1F32Gs256;
  case 512: return &LaunchDequantAffineBits1F32Gs512;
  case 1024: return &LaunchDequantAffineBits1F32Gs1024;
  default: return &LaunchDequantAffineBits1F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits1F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits1F16Gs16;
  case 32: return &LaunchDequantAffineBits1F16Gs32;
  case 64: return &LaunchDequantAffineBits1F16Gs64;
  case 128: return &LaunchDequantAffineBits1F16Gs128;
  case 256: return &LaunchDequantAffineBits1F16Gs256;
  case 512: return &LaunchDequantAffineBits1F16Gs512;
  case 1024: return &LaunchDequantAffineBits1F16Gs1024;
  default: return &LaunchDequantAffineBits1F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits1BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits1BF16Gs16;
  case 32: return &LaunchDequantAffineBits1BF16Gs32;
  case 64: return &LaunchDequantAffineBits1BF16Gs64;
  case 128: return &LaunchDequantAffineBits1BF16Gs128;
  case 256: return &LaunchDequantAffineBits1BF16Gs256;
  case 512: return &LaunchDequantAffineBits1BF16Gs512;
  case 1024: return &LaunchDequantAffineBits1BF16Gs1024;
  default: return &LaunchDequantAffineBits1BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits2F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits2F32Gs16;
  case 32: return &LaunchDequantAffineBits2F32Gs32;
  case 64: return &LaunchDequantAffineBits2F32Gs64;
  case 128: return &LaunchDequantAffineBits2F32Gs128;
  case 256: return &LaunchDequantAffineBits2F32Gs256;
  case 512: return &LaunchDequantAffineBits2F32Gs512;
  case 1024: return &LaunchDequantAffineBits2F32Gs1024;
  default: return &LaunchDequantAffineBits2F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits2F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits2F16Gs16;
  case 32: return &LaunchDequantAffineBits2F16Gs32;
  case 64: return &LaunchDequantAffineBits2F16Gs64;
  case 128: return &LaunchDequantAffineBits2F16Gs128;
  case 256: return &LaunchDequantAffineBits2F16Gs256;
  case 512: return &LaunchDequantAffineBits2F16Gs512;
  case 1024: return &LaunchDequantAffineBits2F16Gs1024;
  default: return &LaunchDequantAffineBits2F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits2BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits2BF16Gs16;
  case 32: return &LaunchDequantAffineBits2BF16Gs32;
  case 64: return &LaunchDequantAffineBits2BF16Gs64;
  case 128: return &LaunchDequantAffineBits2BF16Gs128;
  case 256: return &LaunchDequantAffineBits2BF16Gs256;
  case 512: return &LaunchDequantAffineBits2BF16Gs512;
  case 1024: return &LaunchDequantAffineBits2BF16Gs1024;
  default: return &LaunchDequantAffineBits2BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits3F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits3F32Gs16;
  case 32: return &LaunchDequantAffineBits3F32Gs32;
  case 64: return &LaunchDequantAffineBits3F32Gs64;
  case 128: return &LaunchDequantAffineBits3F32Gs128;
  case 256: return &LaunchDequantAffineBits3F32Gs256;
  case 512: return &LaunchDequantAffineBits3F32Gs512;
  case 1024: return &LaunchDequantAffineBits3F32Gs1024;
  default: return &LaunchDequantAffineBits3F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits3F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits3F16Gs16;
  case 32: return &LaunchDequantAffineBits3F16Gs32;
  case 64: return &LaunchDequantAffineBits3F16Gs64;
  case 128: return &LaunchDequantAffineBits3F16Gs128;
  case 256: return &LaunchDequantAffineBits3F16Gs256;
  case 512: return &LaunchDequantAffineBits3F16Gs512;
  case 1024: return &LaunchDequantAffineBits3F16Gs1024;
  default: return &LaunchDequantAffineBits3F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits3BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits3BF16Gs16;
  case 32: return &LaunchDequantAffineBits3BF16Gs32;
  case 64: return &LaunchDequantAffineBits3BF16Gs64;
  case 128: return &LaunchDequantAffineBits3BF16Gs128;
  case 256: return &LaunchDequantAffineBits3BF16Gs256;
  case 512: return &LaunchDequantAffineBits3BF16Gs512;
  case 1024: return &LaunchDequantAffineBits3BF16Gs1024;
  default: return &LaunchDequantAffineBits3BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits4F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits4F32Gs16;
  case 32: return &LaunchDequantAffineBits4F32Gs32;
  case 64: return &LaunchDequantAffineBits4F32Gs64;
  case 128: return &LaunchDequantAffineBits4F32Gs128;
  case 256: return &LaunchDequantAffineBits4F32Gs256;
  case 512: return &LaunchDequantAffineBits4F32Gs512;
  case 1024: return &LaunchDequantAffineBits4F32Gs1024;
  default: return &LaunchDequantAffineBits4F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits4F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits4F16Gs16;
  case 32: return &LaunchDequantAffineBits4F16Gs32;
  case 64: return &LaunchDequantAffineBits4F16Gs64;
  case 128: return &LaunchDequantAffineBits4F16Gs128;
  case 256: return &LaunchDequantAffineBits4F16Gs256;
  case 512: return &LaunchDequantAffineBits4F16Gs512;
  case 1024: return &LaunchDequantAffineBits4F16Gs1024;
  default: return &LaunchDequantAffineBits4F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits4BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits4BF16Gs16;
  case 32: return &LaunchDequantAffineBits4BF16Gs32;
  case 64: return &LaunchDequantAffineBits4BF16Gs64;
  case 128: return &LaunchDequantAffineBits4BF16Gs128;
  case 256: return &LaunchDequantAffineBits4BF16Gs256;
  case 512: return &LaunchDequantAffineBits4BF16Gs512;
  case 1024: return &LaunchDequantAffineBits4BF16Gs1024;
  default: return &LaunchDequantAffineBits4BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits5F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits5F32Gs16;
  case 32: return &LaunchDequantAffineBits5F32Gs32;
  case 64: return &LaunchDequantAffineBits5F32Gs64;
  case 128: return &LaunchDequantAffineBits5F32Gs128;
  case 256: return &LaunchDequantAffineBits5F32Gs256;
  case 512: return &LaunchDequantAffineBits5F32Gs512;
  case 1024: return &LaunchDequantAffineBits5F32Gs1024;
  default: return &LaunchDequantAffineBits5F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits5F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits5F16Gs16;
  case 32: return &LaunchDequantAffineBits5F16Gs32;
  case 64: return &LaunchDequantAffineBits5F16Gs64;
  case 128: return &LaunchDequantAffineBits5F16Gs128;
  case 256: return &LaunchDequantAffineBits5F16Gs256;
  case 512: return &LaunchDequantAffineBits5F16Gs512;
  case 1024: return &LaunchDequantAffineBits5F16Gs1024;
  default: return &LaunchDequantAffineBits5F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits5BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits5BF16Gs16;
  case 32: return &LaunchDequantAffineBits5BF16Gs32;
  case 64: return &LaunchDequantAffineBits5BF16Gs64;
  case 128: return &LaunchDequantAffineBits5BF16Gs128;
  case 256: return &LaunchDequantAffineBits5BF16Gs256;
  case 512: return &LaunchDequantAffineBits5BF16Gs512;
  case 1024: return &LaunchDequantAffineBits5BF16Gs1024;
  default: return &LaunchDequantAffineBits5BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits6F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits6F32Gs16;
  case 32: return &LaunchDequantAffineBits6F32Gs32;
  case 64: return &LaunchDequantAffineBits6F32Gs64;
  case 128: return &LaunchDequantAffineBits6F32Gs128;
  case 256: return &LaunchDequantAffineBits6F32Gs256;
  case 512: return &LaunchDequantAffineBits6F32Gs512;
  case 1024: return &LaunchDequantAffineBits6F32Gs1024;
  default: return &LaunchDequantAffineBits6F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits6F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits6F16Gs16;
  case 32: return &LaunchDequantAffineBits6F16Gs32;
  case 64: return &LaunchDequantAffineBits6F16Gs64;
  case 128: return &LaunchDequantAffineBits6F16Gs128;
  case 256: return &LaunchDequantAffineBits6F16Gs256;
  case 512: return &LaunchDequantAffineBits6F16Gs512;
  case 1024: return &LaunchDequantAffineBits6F16Gs1024;
  default: return &LaunchDequantAffineBits6F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits6BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits6BF16Gs16;
  case 32: return &LaunchDequantAffineBits6BF16Gs32;
  case 64: return &LaunchDequantAffineBits6BF16Gs64;
  case 128: return &LaunchDequantAffineBits6BF16Gs128;
  case 256: return &LaunchDequantAffineBits6BF16Gs256;
  case 512: return &LaunchDequantAffineBits6BF16Gs512;
  case 1024: return &LaunchDequantAffineBits6BF16Gs1024;
  default: return &LaunchDequantAffineBits6BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits7F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits7F32Gs16;
  case 32: return &LaunchDequantAffineBits7F32Gs32;
  case 64: return &LaunchDequantAffineBits7F32Gs64;
  case 128: return &LaunchDequantAffineBits7F32Gs128;
  case 256: return &LaunchDequantAffineBits7F32Gs256;
  case 512: return &LaunchDequantAffineBits7F32Gs512;
  case 1024: return &LaunchDequantAffineBits7F32Gs1024;
  default: return &LaunchDequantAffineBits7F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits7F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits7F16Gs16;
  case 32: return &LaunchDequantAffineBits7F16Gs32;
  case 64: return &LaunchDequantAffineBits7F16Gs64;
  case 128: return &LaunchDequantAffineBits7F16Gs128;
  case 256: return &LaunchDequantAffineBits7F16Gs256;
  case 512: return &LaunchDequantAffineBits7F16Gs512;
  case 1024: return &LaunchDequantAffineBits7F16Gs1024;
  default: return &LaunchDequantAffineBits7F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits7BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits7BF16Gs16;
  case 32: return &LaunchDequantAffineBits7BF16Gs32;
  case 64: return &LaunchDequantAffineBits7BF16Gs64;
  case 128: return &LaunchDequantAffineBits7BF16Gs128;
  case 256: return &LaunchDequantAffineBits7BF16Gs256;
  case 512: return &LaunchDequantAffineBits7BF16Gs512;
  case 1024: return &LaunchDequantAffineBits7BF16Gs1024;
  default: return &LaunchDequantAffineBits7BF16;
  }
}

inline DequantAffineF32Fn ResolveLaunchDequantAffineBits8F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits8F32Gs16;
  case 32: return &LaunchDequantAffineBits8F32Gs32;
  case 64: return &LaunchDequantAffineBits8F32Gs64;
  case 128: return &LaunchDequantAffineBits8F32Gs128;
  case 256: return &LaunchDequantAffineBits8F32Gs256;
  case 512: return &LaunchDequantAffineBits8F32Gs512;
  case 1024: return &LaunchDequantAffineBits8F32Gs1024;
  default: return &LaunchDequantAffineBits8F32;
  }
}

inline DequantAffineF16Fn ResolveLaunchDequantAffineBits8F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits8F16Gs16;
  case 32: return &LaunchDequantAffineBits8F16Gs32;
  case 64: return &LaunchDequantAffineBits8F16Gs64;
  case 128: return &LaunchDequantAffineBits8F16Gs128;
  case 256: return &LaunchDequantAffineBits8F16Gs256;
  case 512: return &LaunchDequantAffineBits8F16Gs512;
  case 1024: return &LaunchDequantAffineBits8F16Gs1024;
  default: return &LaunchDequantAffineBits8F16;
  }
}

inline DequantAffineBF16Fn ResolveLaunchDequantAffineBits8BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantAffineBits8BF16Gs16;
  case 32: return &LaunchDequantAffineBits8BF16Gs32;
  case 64: return &LaunchDequantAffineBits8BF16Gs64;
  case 128: return &LaunchDequantAffineBits8BF16Gs128;
  case 256: return &LaunchDequantAffineBits8BF16Gs256;
  case 512: return &LaunchDequantAffineBits8BF16Gs512;
  case 1024: return &LaunchDequantAffineBits8BF16Gs1024;
  default: return &LaunchDequantAffineBits8BF16;
  }
}

inline DequantNf4F32Fn ResolveLaunchDequantNf4F32(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantNf4F32Gs16;
  case 32: return &LaunchDequantNf4F32Gs32;
  case 64: return &LaunchDequantNf4F32Gs64;
  case 128: return &LaunchDequantNf4F32Gs128;
  case 256: return &LaunchDequantNf4F32Gs256;
  case 512: return &LaunchDequantNf4F32Gs512;
  case 1024: return &LaunchDequantNf4F32Gs1024;
  default: return &LaunchDequantNf4F32;
  }
}

inline DequantNf4F16Fn ResolveLaunchDequantNf4F16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantNf4F16Gs16;
  case 32: return &LaunchDequantNf4F16Gs32;
  case 64: return &LaunchDequantNf4F16Gs64;
  case 128: return &LaunchDequantNf4F16Gs128;
  case 256: return &LaunchDequantNf4F16Gs256;
  case 512: return &LaunchDequantNf4F16Gs512;
  case 1024: return &LaunchDequantNf4F16Gs1024;
  default: return &LaunchDequantNf4F16;
  }
}

inline DequantNf4BF16Fn ResolveLaunchDequantNf4BF16(int64_t group_size) {
  switch (group_size) {
  case 16: return &LaunchDequantNf4BF16Gs16;
  case 32: return &LaunchDequantNf4BF16Gs32;
  case 64: return &LaunchDequantNf4BF16Gs64;
  case 128: return &LaunchDequantNf4BF16Gs128;
  case 256: return &LaunchDequantNf4BF16Gs256;
  case 512: return &LaunchDequantNf4BF16Gs512;
  case 1024: return &LaunchDequantNf4BF16Gs1024;
  default: return &LaunchDequantNf4BF16;
  }
}

