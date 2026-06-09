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

#pragma once

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "qmm_dequant_dispatch.h"
#include "qmm_dequant_kernels.h"
#include "xla/ffi/api/ffi.h"

namespace {

using xla::ffi::AnyBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Result;
using xla::ffi::ScratchAllocator;
using xla::ffi::Span;

constexpr int kBits4 = 4;
constexpr int kBits8 = 8;

inline const char *CublasErrorString(cublasStatus_t status) {
  switch (status) {
  case CUBLAS_STATUS_SUCCESS:
    return "CUBLAS_STATUS_SUCCESS";
  case CUBLAS_STATUS_NOT_INITIALIZED:
    return "CUBLAS_STATUS_NOT_INITIALIZED";
  case CUBLAS_STATUS_ALLOC_FAILED:
    return "CUBLAS_STATUS_ALLOC_FAILED";
  case CUBLAS_STATUS_INVALID_VALUE:
    return "CUBLAS_STATUS_INVALID_VALUE";
  case CUBLAS_STATUS_ARCH_MISMATCH:
    return "CUBLAS_STATUS_ARCH_MISMATCH";
  case CUBLAS_STATUS_MAPPING_ERROR:
    return "CUBLAS_STATUS_MAPPING_ERROR";
  case CUBLAS_STATUS_EXECUTION_FAILED:
    return "CUBLAS_STATUS_EXECUTION_FAILED";
  case CUBLAS_STATUS_INTERNAL_ERROR:
    return "CUBLAS_STATUS_INTERNAL_ERROR";
  case CUBLAS_STATUS_NOT_SUPPORTED:
    return "CUBLAS_STATUS_NOT_SUPPORTED";
  case CUBLAS_STATUS_LICENSE_ERROR:
    return "CUBLAS_STATUS_LICENSE_ERROR";
  default:
    return "CUBLAS_STATUS_UNKNOWN";
  }
}

__device__ __forceinline__ float ToFloatLocal(float v) { return v; }
__device__ __forceinline__ float ToFloatLocal(half v) { return __half2float(v); }
__device__ __forceinline__ float ToFloatLocal(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ float ToFloatLocal(uint8_t v) {
  return static_cast<float>(v);
}


__global__ void convert_f32_to_f16(const float *in, half *out, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = ToHalf(in[idx]);
  }
}

__global__ void convert_bf16_to_f16(const __nv_bfloat16 *in, half *out,
                                    int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = ToHalf(ToFloatLocal(in[idx]));
  }
}

__global__ void convert_f16_to_bf16(const half *in, __nv_bfloat16 *out,
                                    int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = __float2bfloat16(__half2float(in[idx]));
  }
}

__global__ void convert_f32_to_bf16(const float *in, __nv_bfloat16 *out,
                                    int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = __float2bfloat16(in[idx]);
  }
}

__global__ void convert_f16_to_f32(const half *in, float *out, int64_t size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < size) {
    out[idx] = __half2float(in[idx]);
  }
}

inline bool IsFiniteInt64(int64_t v) {
  return v >= 0 && v <= std::numeric_limits<int32_t>::max();
}

inline Error MakeInvalid(std::string msg) {
  return Error::InvalidArgument(std::move(msg));
}

inline Error MakeInternal(const std::string &msg) {
  return Error::Internal(msg);
}

inline Error CheckCuda(cudaError_t err, const char *what) {
  if (err == cudaSuccess) {
    return Error::Success();
  }
  std::string msg = std::string(what) + ": " + cudaGetErrorString(err);
  return MakeInternal(msg);
}

inline Error CheckCublas(cublasStatus_t status, const char *what) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Error::Success();
  }
  std::string msg = std::string(what) + ": " + CublasErrorString(status);
  return MakeInternal(msg);
}

inline Error CheckCublasLt(cublasStatus_t status, const char *what) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return Error::Success();
  }
  std::string msg = std::string(what) + ": " + CublasErrorString(status);
  return MakeInternal(msg);
}

inline cublasComputeType_t ParseComputeOverride(const char *env) {
  if (!env || !env[0]) {
    return CUBLAS_COMPUTE_32F;
  }
  char c = static_cast<char>(std::tolower(env[0]));
  if (c == 't') {
    return CUBLAS_COMPUTE_32F_FAST_TF32;
  }
  if (c == 'f') {
    return CUBLAS_COMPUTE_32F_FAST_16F;
  }
  if (c == 'b') {
    return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
  if (c == 'h' || c == '1') {
    return CUBLAS_COMPUTE_16F;
  }
  return CUBLAS_COMPUTE_32F;
}

inline cublasComputeType_t ResolveCublasComputeType(
    xla::ffi::DataType x_dtype, xla::ffi::DataType out_dtype) {
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_COMPUTE");
  if (env) {
    return ParseComputeOverride(env);
  }
  if (x_dtype == xla::ffi::DataType::BF16 ||
      out_dtype == xla::ffi::DataType::BF16) {
    return CUBLAS_COMPUTE_32F_FAST_16BF;
  }
  if (x_dtype == xla::ffi::DataType::F16 ||
      out_dtype == xla::ffi::DataType::F16) {
    return CUBLAS_COMPUTE_32F_FAST_16F;
  }
  return CUBLAS_COMPUTE_32F_FAST_TF32;
}

inline cublasMath_t GetCublasMathMode(cublasComputeType_t compute_type) {
  if (compute_type == CUBLAS_COMPUTE_32F_FAST_TF32) {
    return CUBLAS_TF32_TENSOR_OP_MATH;
  }
  return CUBLAS_TENSOR_OP_MATH;
}

inline bool ComputeTypeUsesHalfScalars(cublasComputeType_t compute_type) {
  return compute_type == CUBLAS_COMPUTE_16F ||
         compute_type == CUBLAS_COMPUTE_16F_PEDANTIC;
}

inline cudaDataType_t ToCudaDataType(xla::ffi::DataType dtype) {
  if (dtype == xla::ffi::DataType::F16) {
    return CUDA_R_16F;
  }
  if (dtype == xla::ffi::DataType::BF16) {
    return CUDA_R_16BF;
  }
  return CUDA_R_32F;
}

enum class GemmBackend {
  kCublas = 0,
  kCublasLt = 1,
  kCutlass = 2,
  kCutlassTuned = 3,
};

inline GemmBackend GetGemmBackend() {
  static int cached = -1;
  static GemmBackend cached_backend = GemmBackend::kCublasLt;
  if (cached != -1) {
    return cached_backend;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_GEMM");
  if (!env) {
    cached = 1;
    cached_backend = GemmBackend::kCublasLt;
    return cached_backend;
  }
  std::string mode(env);
  std::transform(mode.begin(), mode.end(), mode.begin(), ::tolower);
  if (mode == "cublaslt") {
    cached_backend = GemmBackend::kCublasLt;
  } else if (mode == "cutlass") {
    cached_backend = GemmBackend::kCutlass;
  } else if (mode == "cutlass_tuned" || mode == "cutlasstuned") {
    cached_backend = GemmBackend::kCutlassTuned;
  } else {
    cached_backend = GemmBackend::kCublas;
  }
  cached = 1;
  return cached_backend;
}

inline bool StrictGemmBackend() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_STRICT");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

inline size_t GetCublasLtWorkspaceLimit() {
  static int cached = -1;
  static size_t cached_bytes = 0;
  if (cached != -1) {
    return cached_bytes;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_CUBLASLT_WS_MB");
  if (!env) {
    cached = 1;
    cached_bytes = static_cast<size_t>(32) * 1024 * 1024;
    return cached_bytes;
  }
  char *end = nullptr;
  long value = std::strtol(env, &end, 10);
  if (end == env || value <= 0) {
    cached = 1;
    cached_bytes = static_cast<size_t>(32) * 1024 * 1024;
    return cached_bytes;
  }
  cached = 1;
  cached_bytes = static_cast<size_t>(value) * 1024 * 1024;
  return cached_bytes;
}

inline bool UseFusedQmm() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_FUSED");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

inline bool UseExperimentalGemvQmm() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_GEMV_EXPERIMENTAL");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

enum class QmmKernelFamily : int64_t {
  kGemm = 0,
  kGemmSplitK = 1,
  kGemvSplitK = 2,
  kGemvRevSplitK = 3,
};

inline bool IsEffective4BitMode(int64_t mode, int64_t bits) {
  if (mode == 0) {
    return bits == kBits4;
  }
  return mode == 1 || mode == 2 || mode == 4;
}

inline Error ResolveQmmKernelFamily(int64_t M, int64_t mode, int64_t bits,
                                    int64_t gemv_mode, int64_t revsplit_k,
                                    int64_t revsplit_k_parts,
                                    QmmKernelFamily *family_out,
                                    int64_t *revsplit_parts_out) {
  if (gemv_mode < 0 || gemv_mode > 2) {
    return MakeInvalid("gemv_mode must be one of {0(auto),1(on),2(off)}.");
  }
  if (revsplit_k < 0 || revsplit_k > 2) {
    return MakeInvalid("revsplit_k must be one of {0(auto),1(on),2(off)}.");
  }
  if (!(revsplit_k_parts == 0 || revsplit_k_parts == 1 || revsplit_k_parts == 2 ||
        revsplit_k_parts == 4 || revsplit_k_parts == 8 || revsplit_k_parts == 16)) {
    return MakeInvalid("revsplit_k_parts must be one of {0,1,2,4,8,16}.");
  }
  if (M <= 0) {
    return MakeInvalid("Input activation leading dimension M must be >= 1.");
  }

  // GemLite parity: keep MXFP M==1 on GEMM-SplitK path.
  if (M == 1 && (mode == 2 || mode == 3)) {
    *family_out = QmmKernelFamily::kGemmSplitK;
    *revsplit_parts_out = 0;
    return Error::Success();
  }

  bool effective_4bit = IsEffective4BitMode(mode, bits);
  bool use_gemv = false;
  if (gemv_mode == 0) {
    use_gemv = (M == 1);
  } else if (gemv_mode == 1) {
    if (M != 1) {
      return MakeInvalid("gemv_mode=on requires M == 1.");
    }
    use_gemv = true;
  } else {
    use_gemv = false;
  }

  int64_t revsplit_parts = 0;
  if (!use_gemv) {
    *family_out = (M > 64) ? QmmKernelFamily::kGemm : QmmKernelFamily::kGemmSplitK;
    *revsplit_parts_out = 0;
    return Error::Success();
  }

  if (revsplit_k == 1) {
    if (!effective_4bit) {
      return MakeInvalid("revsplit_k=on requires an effective 4-bit mode.");
    }
    revsplit_parts = revsplit_k_parts == 0 ? 2 : revsplit_k_parts;
    if (revsplit_parts < 2) {
      return MakeInvalid("revsplit_k=on requires revsplit_k_parts >= 2.");
    }
    *family_out = QmmKernelFamily::kGemvRevSplitK;
    *revsplit_parts_out = revsplit_parts;
    return Error::Success();
  }

  if (revsplit_k == 2) {
    *family_out = QmmKernelFamily::kGemvSplitK;
    *revsplit_parts_out = 0;
    return Error::Success();
  }

  if (effective_4bit) {
    revsplit_parts = revsplit_k_parts == 0 ? 2 : revsplit_k_parts;
    if (revsplit_parts < 2) {
      revsplit_parts = 2;
    }
    *family_out = QmmKernelFamily::kGemvRevSplitK;
    *revsplit_parts_out = revsplit_parts;
    return Error::Success();
  }
  *family_out = QmmKernelFamily::kGemvSplitK;
  *revsplit_parts_out = 0;
  return Error::Success();
}

__device__ __forceinline__ uint32_t LoadPackedQ(const uint32_t *row, int64_t n,
                                                int bits) {
  if (bits == 4) {
    int64_t word_idx = n >> 3;
    int32_t shift = static_cast<int32_t>(n & 7) * 4;
    return (row[word_idx] >> shift) & 0xFu;
  }
  if (bits == 8) {
    int64_t word_idx = n >> 2;
    int32_t shift = static_cast<int32_t>(n & 3) * 8;
    return (row[word_idx] >> shift) & 0xFFu;
  }

  int64_t bit_offset = n * bits;
  int64_t word_idx = bit_offset >> 5;
  int32_t shift = static_cast<int32_t>(bit_offset & 31);
  uint32_t low_word = row[word_idx];
  int32_t low_bits = (shift + bits > 32) ? (32 - shift) : bits;
  int32_t high_bits = bits - low_bits;
  uint32_t low_mask = (uint32_t(1) << low_bits) - 1u;
  uint32_t low = (low_word >> shift) & low_mask;
  uint32_t high = 0;
  if (high_bits > 0) {
    uint32_t high_mask = (uint32_t(1) << high_bits) - 1u;
    high = row[word_idx + 1] & high_mask;
  }
  return low | (high << low_bits);
}

template <int Bits>
__device__ __forceinline__ uint32_t LoadPackedQConst(const uint32_t *row,
                                                     int64_t n,
                                                     int bits_runtime) {
  if constexpr (Bits == 4) {
    int64_t word_idx = n >> 3;
    int32_t shift = static_cast<int32_t>(n & 7) * 4;
    return (row[word_idx] >> shift) & 0xFu;
  } else if constexpr (Bits == 8) {
    int64_t word_idx = n >> 2;
    int32_t shift = static_cast<int32_t>(n & 3) * 8;
    return (row[word_idx] >> shift) & 0xFFu;
  } else if constexpr (Bits > 0) {
    int64_t bit_offset = n * Bits;
    int64_t word_idx = bit_offset >> 5;
    int32_t shift = static_cast<int32_t>(bit_offset & 31);
    uint32_t low_word = row[word_idx];
    int32_t low_bits = (shift + Bits > 32) ? (32 - shift) : Bits;
    int32_t high_bits = Bits - low_bits;
    uint32_t low_mask = (uint32_t(1) << low_bits) - 1u;
    uint32_t low = (low_word >> shift) & low_mask;
    uint32_t high = 0;
    if (high_bits > 0) {
      uint32_t high_mask = (uint32_t(1) << high_bits) - 1u;
      high = row[word_idx + 1] & high_mask;
    }
    return low | (high << low_bits);
  } else {
    return LoadPackedQ(row, n, bits_runtime);
  }
}

template <int Bits>
__device__ __forceinline__ uint32_t LoadPackedQKMajorConst(
    const uint32_t *base, int64_t N, int64_t k, int64_t n, int bits_runtime) {
  if constexpr (Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8) {
    constexpr int kValuesPerWord = 32 / Bits;
    constexpr uint32_t kMask = (uint32_t(1) << Bits) - 1u;
    int64_t word_idx = k / kValuesPerWord;
    int32_t shift = static_cast<int32_t>(k % kValuesPerWord) * Bits;
    return (base[word_idx * N + n] >> shift) & kMask;
  } else {
    int64_t bit_offset = k * bits_runtime;
    int64_t word_idx = bit_offset >> 5;
    int32_t shift = static_cast<int32_t>(bit_offset & 31);
    uint32_t low_word = base[word_idx * N + n];
    int32_t low_bits =
        (shift + bits_runtime > 32) ? (32 - shift) : bits_runtime;
    int32_t high_bits = bits_runtime - low_bits;
    uint32_t low_mask = (uint32_t(1) << low_bits) - 1u;
    uint32_t low = (low_word >> shift) & low_mask;
    uint32_t high = 0;
    if (high_bits > 0) {
      uint32_t high_mask = (uint32_t(1) << high_bits) - 1u;
      high = base[(word_idx + 1) * N + n] & high_mask;
    }
    return low | (high << low_bits);
  }
}

__device__ __forceinline__ float WarpReduceSum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <typename XType>
__device__ __forceinline__ half LoadXAsHalf(const XType *ptr) {
  return ToHalf(ToFloatLocal(*ptr));
}

template <>
__device__ __forceinline__ half LoadXAsHalf<half>(const half *ptr) {
  return *ptr;
}

template <typename OutT>
__device__ __forceinline__ OutT CastOut(float v);

template <>
__device__ __forceinline__ float CastOut<float>(float v) {
  return v;
}

template <>
__device__ __forceinline__ half CastOut<half>(float v) {
  return __float2half_rn(v);
}

template <>
__device__ __forceinline__ __nv_bfloat16 CastOut<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename OutT>
__device__ __forceinline__ void AtomicAddOut(OutT *ptr, float v);

template <>
__device__ __forceinline__ void AtomicAddOut<float>(float *ptr, float v) {
  atomicAdd(ptr, v);
}

template <>
__device__ __forceinline__ void AtomicAddOut<half>(half *ptr, float v) {
  atomicAdd(ptr, __float2half_rn(v));
}

template <>
__device__ __forceinline__ void AtomicAddOut<__nv_bfloat16>(
    __nv_bfloat16 *ptr, float v) {
  atomicAdd(ptr, __float2bfloat16(v));
}

template <typename ScaleT, typename BiasT>
__device__ __forceinline__ float DequantValue(uint32_t q, const ScaleT *scales,
                                              const BiasT *biases, int64_t n,
                                              int64_t k, int64_t g,
                                              int64_t n_groups,
                                              bool transpose, int64_t mode) {
  int64_t idx = transpose ? (n * n_groups + g) : (k * n_groups + g);
  if (mode == 0) {
    float scale = ToFloatLocal(scales[idx]);
    float bias = ToFloatLocal(biases[idx]);
    return static_cast<float>(q) * scale + bias;
  }
  if (mode == 1) {
    float scale = ToFloatLocal(scales[idx]);
    return kNF4Table[q] * scale;
  }
  if (mode == 2) {
    uint8_t exp_u8 = static_cast<uint8_t>(scales[idx]);
    int8_t exp = static_cast<int8_t>(exp_u8);
    float scale = exp2f(static_cast<float>(exp));
    return kE2M1Table[q] * scale;
  }
  if (mode == 3) {
    uint8_t exp_u8 = static_cast<uint8_t>(scales[idx]);
    int8_t exp = static_cast<int8_t>(exp_u8);
    float scale = exp2f(static_cast<float>(exp));
    return kE4M3Table[q] * scale;
  }
  if (mode == 4) {
    uint8_t code = static_cast<uint8_t>(scales[idx]);
    float scale = kE4M3Table[code];
    return kE2M1Table[q] * scale;
  }
  uint8_t code = static_cast<uint8_t>(scales[idx]);
  float scale = kE4M3Table[code];
  return kE4M3Table[q] * scale;
}

template <typename ScaleT, typename BiasT>
__global__ void DequantToBf16Kernel(const uint32_t *wq, const ScaleT *scales,
                                    const BiasT *biases, __nv_bfloat16 *out,
                                    int64_t K, int64_t N, int64_t n_words,
                                    int64_t n_groups, int64_t group_size,
                                    int bits, int64_t mode,
                                    int64_t transpose) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
  bool transposed = transpose != 0;

#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k;
    int64_t n;
    if (transposed) {
      n = idx / K;
      k = idx - n * K;
    } else {
      k = idx / N;
      n = idx - k * N;
    }
    const uint32_t *wq_row =
        transposed ? (wq + n * n_words) : (wq + k * n_words);
    uint32_t q = LoadPackedQ(wq_row, transposed ? k : n, bits);
    int64_t g = transposed ? (k / group_size) : (n / group_size);
    __nv_bfloat16 result;
    if (mode == 0) {
      int64_t s_idx = transposed ? (n * n_groups + g) : (k * n_groups + g);
      result = DequantAffineToBf16(q, scales[s_idx], biases[s_idx]);
    } else {
      float val =
          DequantValue(q, scales, biases, n, k, g, n_groups, transposed, mode);
      result = __float2bfloat16(val);
    }
    out[idx] = result;
  }
}

template <typename ScaleT, typename BiasT>
__global__ void DequantToHalfKernel(const uint32_t *wq, const ScaleT *scales,
                                    const BiasT *biases, half *out, int64_t K,
                                    int64_t N, int64_t n_words, int64_t n_groups,
                                    int64_t group_size, int bits, int64_t mode,
                                    int64_t transpose) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;
  bool transposed = transpose != 0;

#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t k;
    int64_t n;
    if (transposed) {
      n = idx / K;
      k = idx - n * K;
    } else {
      k = idx / N;
      n = idx - k * N;
    }
    const uint32_t *wq_row =
        transposed ? (wq + n * n_words) : (wq + k * n_words);
    uint32_t q = LoadPackedQ(wq_row, transposed ? k : n, bits);
    int64_t g = transposed ? (k / group_size) : (n / group_size);
    float val =
        DequantValue(q, scales, biases, n, k, g, n_groups, transposed, mode);
    out[idx] = __float2half_rn(val);
  }
}

template <typename ScaleT, typename BiasT, typename OutT>
__global__ void DequantKMajorAffineKernel(
    const uint32_t *wq, const ScaleT *scales, const BiasT *biases, OutT *out,
    int64_t K, int64_t N, int64_t n_groups, int64_t group_size, int bits) {
  int64_t total = K * N;
  int64_t base =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) *
      kDequantElemsPerThread;

#pragma unroll
  for (int i = 0; i < kDequantElemsPerThread; ++i) {
    int64_t idx = base + i;
    if (idx >= total) {
      return;
    }
    int64_t n = idx / K;
    int64_t k = idx - n * K;
    int64_t g = k / group_size;
    uint32_t q = LoadPackedQKMajorConst<0>(wq, N, k, n, bits);
    float val = static_cast<float>(q) * ToFloatLocal(scales[g * N + n]) +
                ToFloatLocal(biases[g * N + n]);
    out[idx] = CastOut<OutT>(val);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
__global__ void QmmFusedKernel(const XType *x, const uint32_t *wq,
                               const ScaleT *scales, const BiasT *biases,
                               OutT *out, int64_t M, int64_t N, int64_t K,
                               int64_t n_words, int64_t n_groups,
                               int64_t group_size, int bits, int64_t mode,
                               int64_t transpose) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 32;

  int block_m = static_cast<int>(blockIdx.y) * BM;
  int block_n = static_cast<int>(blockIdx.x) * BN;

  __shared__ half sh_a[BM * BK];
  __shared__ half sh_b[BK * BN];

  int tid = static_cast<int>(threadIdx.y) * blockDim.x + threadIdx.x;
  int num_threads = static_cast<int>(blockDim.x * blockDim.y);

  int base_row = block_m + static_cast<int>(threadIdx.y) * 4;
  int base_col = block_n + static_cast<int>(threadIdx.x) * 4;
  bool transposed = transpose != 0;

  float acc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      acc[i][j] = 0.0f;
    }
  }

  for (int64_t k0 = 0; k0 < K; k0 += BK) {
    for (int idx = tid; idx < BM * BK; idx += num_threads) {
      int a_row = idx / BK;
      int a_col = idx - a_row * BK;
      int64_t g_m = static_cast<int64_t>(block_m + a_row);
      int64_t g_k = k0 + a_col;
      half val = ToHalf(0.0f);
      if (g_m < M && g_k < K) {
        val = LoadXAsHalf(x + g_m * K + g_k);
      }
      sh_a[idx] = val;
    }

    for (int idx = tid; idx < BK * BN; idx += num_threads) {
      int b_row = idx / BN;
      int b_col = idx - b_row * BN;
      int64_t g_k = k0 + b_row;
      int64_t g_n = static_cast<int64_t>(block_n + b_col);
      half val = ToHalf(0.0f);
      if (g_k < K && g_n < N) {
        const uint32_t *wq_row =
            transposed ? (wq + g_n * n_words) : (wq + g_k * n_words);
        uint32_t q =
            LoadPackedQ(wq_row, transposed ? g_k : g_n, bits);
        int64_t g = transposed ? (g_k / group_size) : (g_n / group_size);
        float w_val =
            DequantValue(q, scales, biases, g_n, g_k, g, n_groups, transposed,
                         mode);
        val = ToHalf(w_val);
      }
      sh_b[idx] = val;
    }

    __syncthreads();

#pragma unroll
    for (int k = 0; k < BK; ++k) {
      int a_row0 = static_cast<int>(threadIdx.y) * 4;
      int b_col0 = static_cast<int>(threadIdx.x) * 4;

      float a0 = __half2float(sh_a[(a_row0 + 0) * BK + k]);
      float a1 = __half2float(sh_a[(a_row0 + 1) * BK + k]);
      float a2 = __half2float(sh_a[(a_row0 + 2) * BK + k]);
      float a3 = __half2float(sh_a[(a_row0 + 3) * BK + k]);

      float b0 = __half2float(sh_b[k * BN + (b_col0 + 0)]);
      float b1 = __half2float(sh_b[k * BN + (b_col0 + 1)]);
      float b2 = __half2float(sh_b[k * BN + (b_col0 + 2)]);
      float b3 = __half2float(sh_b[k * BN + (b_col0 + 3)]);

      acc[0][0] += a0 * b0;
      acc[0][1] += a0 * b1;
      acc[0][2] += a0 * b2;
      acc[0][3] += a0 * b3;
      acc[1][0] += a1 * b0;
      acc[1][1] += a1 * b1;
      acc[1][2] += a1 * b2;
      acc[1][3] += a1 * b3;
      acc[2][0] += a2 * b0;
      acc[2][1] += a2 * b1;
      acc[2][2] += a2 * b2;
      acc[2][3] += a2 * b3;
      acc[3][0] += a3 * b0;
      acc[3][1] += a3 * b1;
      acc[3][2] += a3 * b2;
      acc[3][3] += a3 * b3;
    }

    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < 4; ++i) {
    int64_t g_m = static_cast<int64_t>(base_row + i);
    if (g_m >= M) {
      continue;
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int64_t g_n = static_cast<int64_t>(base_col + j);
      if (g_n >= N) {
        continue;
      }
      out[g_m * N + g_n] = CastOut<OutT>(acc[i][j]);
    }
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedKernel(const XType *x, const uint32_t *wq,
                           const ScaleT *scales, const BiasT *biases,
                           OutT *out, int64_t M, int64_t N, int64_t K,
                           int64_t n_words, int64_t n_groups,
                           int64_t group_size, int bits, int64_t mode,
                           int64_t transpose, cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(static_cast<uint32_t>((N + 63) / 64),
            static_cast<uint32_t>((M + 63) / 64));
  QmmFusedKernel<XType, ScaleT, BiasT, OutT><<<grid, block, 0, stream>>>(
      x, wq, scales, biases, out, M, N, K, n_words, n_groups, group_size,
      bits, mode, transpose);
  return CheckCuda(cudaPeekAtLastError(), "fused qmm kernel launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
__global__ void QmmFusedGemvKernel(const XType *x, const uint32_t *wq,
                                   const ScaleT *scales, const BiasT *biases,
                                   OutT *out, int64_t N, int64_t K,
                                   int64_t n_words, int64_t n_groups,
                                   int64_t group_size, int bits, int64_t mode,
                                   int64_t transpose) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  bool transposed = transpose != 0;
  float acc = 0.0f;
  for (int64_t k = 0; k < K; ++k) {
    float x_val = ToFloatLocal(x[k]);
    const uint32_t *wq_row =
        transposed ? (wq + n * n_words) : (wq + k * n_words);
    uint32_t q = LoadPackedQ(wq_row, transposed ? k : n, bits);
    int64_t g = transposed ? (k / group_size) : (n / group_size);
    float w_val =
        DequantValue(q, scales, biases, n, k, g, n_groups, transposed, mode);
    acc += x_val * w_val;
  }
  out[n] = CastOut<OutT>(acc);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
__global__ void QmmFusedGemvRevSplitKKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits, int64_t mode,
    int64_t transpose, int64_t revsplit_k_parts) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  bool transposed = transpose != 0;
  int64_t parts = revsplit_k_parts <= 1 ? 2 : revsplit_k_parts;
  float acc = 0.0f;
  for (int64_t p = 0; p < parts; ++p) {
    int64_t k_start = (K * p) / parts;
    int64_t k_end = (K * (p + 1)) / parts;
    for (int64_t k = k_start; k < k_end; ++k) {
      float x_val = ToFloatLocal(x[k]);
      const uint32_t *wq_row =
          transposed ? (wq + n * n_words) : (wq + k * n_words);
      uint32_t q = LoadPackedQ(wq_row, transposed ? k : n, bits);
      int64_t g = transposed ? (k / group_size) : (n / group_size);
      float w_val =
          DequantValue(q, scales, biases, n, k, g, n_groups, transposed, mode);
      acc += x_val * w_val;
    }
  }
  out[n] = CastOut<OutT>(acc);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int WarpsPerBlock, int WarpsPerOutput>
__global__ void QmmFusedGemvWarpKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits_runtime, int64_t mode,
    int64_t transpose) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  constexpr int kOutputsPerBlock = WarpsPerBlock / WarpsPerOutput;
  int output_slot = warp_id / WarpsPerOutput;
  int output_warp = warp_id - output_slot * WarpsPerOutput;
  int64_t n = static_cast<int64_t>(blockIdx.x) * kOutputsPerBlock +
              output_slot;
  bool valid_n = n < N;

  bool transposed = transpose != 0;
  float acc = 0.0f;
  __shared__ float warp_partials[WarpsPerBlock];
  if (mode == 0 && transposed) {
    float grouped_acc = 0.0f;
    if (valid_n) {
      const uint32_t *wq_row = wq + n * n_words;
      const ScaleT *scale_row = scales + n * n_groups;
      const BiasT *bias_row = biases + n * n_groups;
      for (int64_t g = output_warp; g < n_groups; g += WarpsPerOutput) {
        int64_t k_begin = g * group_size;
        int64_t k_end =
            (k_begin + group_size < K) ? (k_begin + group_size) : K;
        float xq_sum = 0.0f;
        float x_sum = 0.0f;
        for (int64_t k = k_begin + lane; k < k_end; k += 32) {
          float x_val = ToFloatLocal(x[k]);
          uint32_t q = LoadPackedQConst<Bits>(wq_row, k, bits_runtime);
          xq_sum += x_val * static_cast<float>(q);
          x_sum += x_val;
        }
        xq_sum = WarpReduceSum(xq_sum);
        x_sum = WarpReduceSum(x_sum);
        if (lane == 0) {
          grouped_acc += ToFloatLocal(scale_row[g]) * xq_sum +
                         ToFloatLocal(bias_row[g]) * x_sum;
        }
      }
    }
    if (lane == 0) {
      warp_partials[warp_id] = grouped_acc;
    }
    __syncthreads();
    if (valid_n && output_warp == 0 && lane == 0) {
      float total = 0.0f;
#pragma unroll
      for (int i = 0; i < WarpsPerOutput; ++i) {
        total += warp_partials[output_slot * WarpsPerOutput + i];
      }
      out[n] = CastOut<OutT>(total);
    }
    return;
  }

  if (!valid_n) {
    return;
  }

  if (mode == 0) {
    for (int64_t k = lane; k < K; k += 32) {
      const uint32_t *wq_row = wq + k * n_words;
      uint32_t q = LoadPackedQConst<Bits>(wq_row, n, bits_runtime);
      int64_t g = n / group_size;
      int64_t idx = k * n_groups + g;
      float w_val = static_cast<float>(q) * ToFloatLocal(scales[idx]) +
                    ToFloatLocal(biases[idx]);
      acc += ToFloatLocal(x[k]) * w_val;
    }
  } else {
    for (int64_t k = lane; k < K; k += 32) {
      const uint32_t *wq_row =
          transposed ? (wq + n * n_words) : (wq + k * n_words);
      uint32_t q =
          LoadPackedQConst<Bits>(wq_row, transposed ? k : n, bits_runtime);
      int64_t g = transposed ? (k / group_size) : (n / group_size);
      float w_val =
          DequantValue(q, scales, biases, n, k, g, n_groups, transposed, mode);
      acc += ToFloatLocal(x[k]) * w_val;
    }
  }

  acc = WarpReduceSum(acc);
  if (lane == 0) {
    out[n] = CastOut<OutT>(acc);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int BlockN>
__global__ void QmmFusedGemvKMajorAffineKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits_runtime) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * BlockN + threadIdx.x;
  if (n >= N) {
    return;
  }

  float acc = 0.0f;
  if constexpr (Bits == 1 || Bits == 2 || Bits == 4 || Bits == 8) {
    constexpr int kValuesPerWord = 32 / Bits;
    constexpr uint32_t kMask = (uint32_t(1) << Bits) - 1u;
    for (int64_t g = 0; g < n_groups; ++g) {
      float scale = ToFloatLocal(scales[g * N + n]);
      float bias = ToFloatLocal(biases[g * N + n]);
      int64_t k_begin = g * group_size;
      int64_t k_end =
          (k_begin + group_size < K) ? (k_begin + group_size) : K;
      int64_t word_begin = k_begin / kValuesPerWord;
      int64_t word_end = (k_end + kValuesPerWord - 1) / kValuesPerWord;
      for (int64_t word = word_begin; word < word_end; ++word) {
        uint32_t packed = wq[word * N + n];
#pragma unroll
        for (int i = 0; i < kValuesPerWord; ++i) {
          int64_t k_idx = word * kValuesPerWord + i;
          if (k_idx >= k_begin && k_idx < k_end) {
            uint32_t q = (packed >> (i * Bits)) & kMask;
            float w_val = static_cast<float>(q) * scale + bias;
            acc += ToFloatLocal(x[k_idx]) * w_val;
          }
        }
      }
    }
  } else {
    for (int64_t k = 0; k < K; ++k) {
      int64_t g = k / group_size;
      uint32_t q = LoadPackedQKMajorConst<Bits>(wq, N, k, n, bits_runtime);
      float w_val = static_cast<float>(q) * ToFloatLocal(scales[g * N + n]) +
                    ToFloatLocal(biases[g * N + n]);
      acc += ToFloatLocal(x[k]) * w_val;
    }
  }

  out[n] = CastOut<OutT>(acc);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits>
Error LaunchQmmFusedGemvKMajorAffineKernelBits(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits, cudaStream_t stream) {
  constexpr int kBlockN = 256;
  dim3 block(kBlockN);
  dim3 grid(static_cast<uint32_t>((N + kBlockN - 1) / kBlockN));
  QmmFusedGemvKMajorAffineKernel<XType, ScaleT, BiasT, OutT, Bits, kBlockN>
      <<<grid, block, 0, stream>>>(x, wq, scales, biases, out, N, K, n_groups,
                                   group_size, bits);
  return CheckCuda(cudaPeekAtLastError(),
                   "fused qmm k-major affine gemv kernel launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedGemvKMajorAffineKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits, cudaStream_t stream) {
  switch (bits) {
  case 1:
    return LaunchQmmFusedGemvKMajorAffineKernelBits<XType, ScaleT, BiasT, OutT,
                                                    1>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream);
  case 2:
    return LaunchQmmFusedGemvKMajorAffineKernelBits<XType, ScaleT, BiasT, OutT,
                                                    2>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream);
  case 4:
    return LaunchQmmFusedGemvKMajorAffineKernelBits<XType, ScaleT, BiasT, OutT,
                                                    4>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream);
  case 8:
    return LaunchQmmFusedGemvKMajorAffineKernelBits<XType, ScaleT, BiasT, OutT,
                                                    8>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream);
  default:
    return LaunchQmmFusedGemvKMajorAffineKernelBits<XType, ScaleT, BiasT, OutT,
                                                    0>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int BlockN, int BlockK, int Threads>
__global__ void QmmFusedGemvKMajorAffineTileKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t group_size,
    int bits_runtime) {
  __shared__ float tile_acc[BlockN];
  int tid = threadIdx.x;
  int64_t n_base = static_cast<int64_t>(blockIdx.x) * BlockN;
  int64_t k_base = static_cast<int64_t>(blockIdx.y) * BlockK;
  int64_t g = k_base / group_size;

  for (int idx = tid; idx < BlockN; idx += Threads) {
    tile_acc[idx] = 0.0f;
  }
  __syncthreads();

  for (int idx = tid; idx < BlockN * BlockK; idx += Threads) {
    int k_off = idx / BlockN;
    int n_off = idx - k_off * BlockN;
    int64_t n = n_base + n_off;
    int64_t k = k_base + k_off;
    if (n < N && k < K) {
      uint32_t q = LoadPackedQKMajorConst<Bits>(wq, N, k, n, bits_runtime);
      float scale = ToFloatLocal(scales[g * N + n]);
      float bias = ToFloatLocal(biases[g * N + n]);
      float w_val = static_cast<float>(q) * scale + bias;
      atomicAdd(tile_acc + n_off, ToFloatLocal(x[k]) * w_val);
    }
  }
  __syncthreads();

  for (int n_off = tid; n_off < BlockN; n_off += Threads) {
    int64_t n = n_base + n_off;
    if (n < N) {
      AtomicAddOut(out + n, tile_acc[n_off]);
    }
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits>
Error LaunchQmmFusedGemvKMajorAffineTileKernelBits(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t group_size,
    int bits, cudaStream_t stream) {
  size_t out_bytes = static_cast<size_t>(N) * sizeof(OutT);
  if (Error err = CheckCuda(cudaMemsetAsync(out, 0, out_bytes, stream),
                            "K-major tile GEMV output memset");
      err.failure()) {
    return err;
  }

  constexpr int kBlockN = 512;
  constexpr int kBlockK = 16;
  constexpr int kThreads = 128;
  dim3 block(kThreads);
  dim3 grid(static_cast<uint32_t>((N + kBlockN - 1) / kBlockN),
            static_cast<uint32_t>((K + kBlockK - 1) / kBlockK));
  QmmFusedGemvKMajorAffineTileKernel<XType, ScaleT, BiasT, OutT, Bits,
                                     kBlockN, kBlockK, kThreads>
      <<<grid, block, 0, stream>>>(x, wq, scales, biases, out, N, K,
                                   group_size, bits);
  return CheckCuda(cudaPeekAtLastError(),
                   "fused qmm K-major tile GEMV kernel launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedGemvKMajorAffineTileKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t group_size,
    int bits, cudaStream_t stream) {
  switch (bits) {
  case 1:
    return LaunchQmmFusedGemvKMajorAffineTileKernelBits<
        XType, ScaleT, BiasT, OutT, 1>(x, wq, scales, biases, out, N, K,
                                       group_size, bits, stream);
  case 2:
    return LaunchQmmFusedGemvKMajorAffineTileKernelBits<
        XType, ScaleT, BiasT, OutT, 2>(x, wq, scales, biases, out, N, K,
                                       group_size, bits, stream);
  case 4:
    return LaunchQmmFusedGemvKMajorAffineTileKernelBits<
        XType, ScaleT, BiasT, OutT, 4>(x, wq, scales, biases, out, N, K,
                                       group_size, bits, stream);
  case 8:
    return LaunchQmmFusedGemvKMajorAffineTileKernelBits<
        XType, ScaleT, BiasT, OutT, 8>(x, wq, scales, biases, out, N, K,
                                       group_size, bits, stream);
  default:
    return LaunchQmmFusedGemvKMajorAffineTileKernelBits<
        XType, ScaleT, BiasT, OutT, 0>(x, wq, scales, biases, out, N, K,
                                       group_size, bits, stream);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int BlockN, int BlockK>
__global__ void QmmFusedGemvKMajorAffineSplitKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, const float *group_sums, OutT *out, int64_t N,
    int64_t K, int64_t group_size, int bits_runtime) {
  int tid = threadIdx.x;
  int64_t n = static_cast<int64_t>(blockIdx.x) * BlockN + threadIdx.x;
  bool valid_n = n < N;

  int64_t tile_begin = static_cast<int64_t>(blockIdx.y) * BlockK;
  int64_t tile_end = (tile_begin + BlockK < K) ? (tile_begin + BlockK) : K;
  int64_t tile_len = tile_end - tile_begin;
  int64_t g_begin = tile_begin / group_size;
  int64_t g_end = (tile_end + group_size - 1) / group_size;
  __shared__ float x_tile[BlockK];
  for (int off = tid; off < tile_len; off += BlockN) {
    x_tile[off] = ToFloatLocal(x[tile_begin + off]);
  }
  __syncthreads();
  if (!valid_n) {
    return;
  }
  float acc = 0.0f;

  for (int64_t g = g_begin; g < g_end; ++g) {
    int64_t k_begin = g * group_size;
    int64_t k_end = (k_begin + group_size < K) ? (k_begin + group_size) : K;
    k_begin = (k_begin > tile_begin) ? k_begin : tile_begin;
    k_end = (k_end < tile_end) ? k_end : tile_end;
    float scale = ToFloatLocal(scales[g * N + n]);
    float bias = ToFloatLocal(biases[g * N + n]);
    float xq_sum = 0.0f;
    float x_sum = 0.0f;

    if constexpr (Bits == 1) {
      constexpr int kValuesPerWord = 32;
      int64_t word_begin = k_begin / kValuesPerWord;
      int64_t word_end = (k_end + kValuesPerWord - 1) / kValuesPerWord;
      if ((k_begin % kValuesPerWord) == 0 &&
          (k_end % kValuesPerWord) == 0) {
#pragma unroll
        for (int64_t word = word_begin; word < word_end; ++word) {
          uint32_t packed = wq[word * N + n];
          while (packed != 0u) {
            int bit = __ffs(packed) - 1;
            int64_t k_idx = word * kValuesPerWord + bit;
            xq_sum += x_tile[k_idx - tile_begin];
            packed &= packed - 1u;
          }
        }
      } else {
#pragma unroll
        for (int64_t word = word_begin; word < word_end; ++word) {
          uint32_t packed = wq[word * N + n];
          while (packed != 0u) {
            int bit = __ffs(packed) - 1;
            int64_t k_idx = word * kValuesPerWord + bit;
            if (k_idx >= k_begin && k_idx < k_end) {
              xq_sum += x_tile[k_idx - tile_begin];
            }
            packed &= packed - 1u;
          }
        }
      }
    } else if constexpr (Bits == 2 || Bits == 4 || Bits == 8) {
      constexpr int kValuesPerWord = 32 / Bits;
      constexpr uint32_t kMask = (uint32_t(1) << Bits) - 1u;
      int64_t word_begin = k_begin / kValuesPerWord;
      int64_t word_end = (k_end + kValuesPerWord - 1) / kValuesPerWord;
      if ((k_begin % kValuesPerWord) == 0 &&
          (k_end % kValuesPerWord) == 0) {
#pragma unroll
        for (int64_t word = word_begin; word < word_end; ++word) {
          uint32_t packed = wq[word * N + n];
#pragma unroll
          for (int i = 0; i < kValuesPerWord; ++i) {
            int64_t k_idx = word * kValuesPerWord + i;
            float x_val = x_tile[k_idx - tile_begin];
            uint32_t q = (packed >> (i * Bits)) & kMask;
            xq_sum += x_val * static_cast<float>(q);
            if (group_sums == nullptr) {
              x_sum += x_val;
            }
          }
        }
      } else {
#pragma unroll
        for (int64_t word = word_begin; word < word_end; ++word) {
          uint32_t packed = wq[word * N + n];
#pragma unroll
          for (int i = 0; i < kValuesPerWord; ++i) {
            int64_t k_idx = word * kValuesPerWord + i;
            if (k_idx >= k_begin && k_idx < k_end) {
              float x_val = x_tile[k_idx - tile_begin];
              uint32_t q = (packed >> (i * Bits)) & kMask;
              xq_sum += x_val * static_cast<float>(q);
              if (group_sums == nullptr) {
                x_sum += x_val;
              }
            }
          }
        }
      }
    } else {
      for (int64_t k = k_begin; k < k_end; ++k) {
        float x_val = x_tile[k - tile_begin];
        uint32_t q = LoadPackedQKMajorConst<Bits>(wq, N, k, n, bits_runtime);
        xq_sum += x_val * static_cast<float>(q);
        if (group_sums == nullptr) {
          x_sum += x_val;
        }
      }
    }
    float bias_sum = group_sums == nullptr ? x_sum : group_sums[g];
    acc += scale * xq_sum + bias * bias_sum;
  }
  AtomicAddOut(out + n, acc);
}

template <typename XType>
__global__ void QmmGemvGroupSumsKernel(const XType *x, float *group_sums,
                                       int64_t K, int64_t n_groups,
                                       int64_t group_size) {
  int64_t g = static_cast<int64_t>(blockIdx.x);
  int tid = threadIdx.x;
  float acc = 0.0f;
  int64_t k_begin = g * group_size;
  int64_t k_end = (k_begin + group_size < K) ? (k_begin + group_size) : K;
  for (int64_t k = k_begin + tid; k < k_end; k += blockDim.x) {
    acc += ToFloatLocal(x[k]);
  }
  __shared__ float smem[256];
  smem[tid] = acc;
  __syncthreads();
  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      smem[tid] += smem[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0 && g < n_groups) {
    group_sums[g] = smem[0];
  }
}

template <typename XType, typename ScaleT, typename BiasT, int Bits,
          int BlockN, int GroupSize>
__global__ void QmmFusedGemvKMajorAffinePartialKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, float *partials, int64_t N, int64_t K,
    int64_t n_groups, int bits_runtime) {
  int tid = threadIdx.x;
  int64_t n = static_cast<int64_t>(blockIdx.x) * BlockN + tid;
  int64_t g = static_cast<int64_t>(blockIdx.y);
  int64_t k_begin = g * GroupSize;

  __shared__ float x_tile[GroupSize];
  __shared__ float x_sum;
  if (tid < GroupSize) {
    int64_t k_idx = k_begin + tid;
    x_tile[tid] = (k_idx < K) ? ToFloatLocal(x[k_idx]) : 0.0f;
  }
  __syncthreads();

  if (tid == 0) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < GroupSize; ++i) {
      sum += x_tile[i];
    }
    x_sum = sum;
  }
  __syncthreads();

  if (n >= N || g >= n_groups) {
    return;
  }

  float xq_sum = 0.0f;
  if constexpr (Bits == 1) {
    constexpr int kValuesPerWord = 32;
    constexpr int kWordsPerGroup = GroupSize / kValuesPerWord;
    int64_t word_begin = k_begin / kValuesPerWord;
#pragma unroll
    for (int w = 0; w < kWordsPerGroup; ++w) {
      uint32_t packed = wq[(word_begin + w) * N + n];
      while (packed != 0u) {
        int bit = __ffs(packed) - 1;
        xq_sum += x_tile[w * kValuesPerWord + bit];
        packed &= packed - 1u;
      }
    }
  } else if constexpr (Bits == 2 || Bits == 4 || Bits == 8) {
    constexpr int kValuesPerWord = 32 / Bits;
    constexpr int kWordsPerGroup = GroupSize / kValuesPerWord;
    constexpr uint32_t kMask = (uint32_t(1) << Bits) - 1u;
    int64_t word_begin = k_begin / kValuesPerWord;
#pragma unroll
    for (int w = 0; w < kWordsPerGroup; ++w) {
      uint32_t packed = wq[(word_begin + w) * N + n];
#pragma unroll
      for (int i = 0; i < kValuesPerWord; ++i) {
        uint32_t q = (packed >> (i * Bits)) & kMask;
        xq_sum += x_tile[w * kValuesPerWord + i] * static_cast<float>(q);
      }
    }
  } else {
    for (int64_t k = k_begin; k < k_begin + GroupSize && k < K; ++k) {
      uint32_t q = LoadPackedQKMajorConst<Bits>(wq, N, k, n, bits_runtime);
      xq_sum += x_tile[k - k_begin] * static_cast<float>(q);
    }
  }

  float scale = ToFloatLocal(scales[g * N + n]);
  float bias = ToFloatLocal(biases[g * N + n]);
  partials[g * N + n] = scale * xq_sum + bias * x_sum;
}

template <typename OutT, int BlockN>
__global__ void QmmFusedGemvPartialReduceKernel(const float *partials,
                                                OutT *out, int64_t N,
                                                int64_t n_groups) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * BlockN + threadIdx.x;
  if (n >= N) {
    return;
  }
  float acc = 0.0f;
  for (int64_t g = 0; g < n_groups; ++g) {
    acc += partials[g * N + n];
  }
  out[n] = CastOut<OutT>(acc);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int GroupSize>
Error LaunchQmmFusedGemvKMajorAffinePartialKernelGroup(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int bits, cudaStream_t stream, ScratchAllocator &scratch) {
  constexpr int kBlockN = 256;
  size_t partial_bytes =
      static_cast<size_t>(N) * static_cast<size_t>(n_groups) * sizeof(float);
  auto partials_opt = scratch.Allocate(partial_bytes, alignof(float));
  if (!partials_opt.has_value()) {
    return MakeInternal("Failed to allocate partial buffer for K-major GEMV.");
  }
  float *partials = reinterpret_cast<float *>(*partials_opt);
  dim3 block(kBlockN);
  dim3 partial_grid(static_cast<uint32_t>((N + kBlockN - 1) / kBlockN),
                    static_cast<uint32_t>(n_groups));
  QmmFusedGemvKMajorAffinePartialKernel<XType, ScaleT, BiasT, Bits, kBlockN,
                                        GroupSize>
      <<<partial_grid, block, 0, stream>>>(x, wq, scales, biases, partials, N,
                                           K, n_groups, bits);
  if (Error err = CheckCuda(cudaPeekAtLastError(),
                            "fused qmm K-major partial GEMV kernel launch");
      err.failure()) {
    return err;
  }
  dim3 reduce_grid(static_cast<uint32_t>((N + kBlockN - 1) / kBlockN));
  QmmFusedGemvPartialReduceKernel<OutT, kBlockN>
      <<<reduce_grid, block, 0, stream>>>(partials, out, N, n_groups);
  return CheckCuda(cudaPeekAtLastError(),
                   "fused qmm K-major partial GEMV reduce launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits>
Error LaunchQmmFusedGemvKMajorAffinePartialKernelBits(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits, cudaStream_t stream,
    ScratchAllocator &scratch) {
  if (group_size == 32) {
    return LaunchQmmFusedGemvKMajorAffinePartialKernelGroup<
        XType, ScaleT, BiasT, OutT, Bits, 32>(
        x, wq, scales, biases, out, N, K, n_groups, bits, stream, scratch);
  }
  if (group_size == 64) {
    return LaunchQmmFusedGemvKMajorAffinePartialKernelGroup<
        XType, ScaleT, BiasT, OutT, Bits, 64>(
        x, wq, scales, biases, out, N, K, n_groups, bits, stream, scratch);
  }
  if (group_size == 128) {
    return LaunchQmmFusedGemvKMajorAffinePartialKernelGroup<
        XType, ScaleT, BiasT, OutT, Bits, 128>(
        x, wq, scales, biases, out, N, K, n_groups, bits, stream, scratch);
  }
  return MakeInvalid(
      "K-major partial GEMV supports group_size in {32,64,128}.");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedGemvKMajorAffinePartialKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits, cudaStream_t stream,
    ScratchAllocator &scratch) {
  switch (bits) {
  case 1:
    return LaunchQmmFusedGemvKMajorAffinePartialKernelBits<
        XType, ScaleT, BiasT, OutT, 1>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch);
  case 2:
    return LaunchQmmFusedGemvKMajorAffinePartialKernelBits<
        XType, ScaleT, BiasT, OutT, 2>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch);
  case 4:
    return LaunchQmmFusedGemvKMajorAffinePartialKernelBits<
        XType, ScaleT, BiasT, OutT, 4>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch);
  case 8:
    return LaunchQmmFusedGemvKMajorAffinePartialKernelBits<
        XType, ScaleT, BiasT, OutT, 8>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch);
  default:
    return LaunchQmmFusedGemvKMajorAffinePartialKernelBits<
        XType, ScaleT, BiasT, OutT, 0>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename AccumOutT,
          int Bits, int BlockN, int BlockK>
Error LaunchQmmFusedGemvKMajorAffineSplitKernelConfig(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, const float *group_sums, AccumOutT *out, int64_t N,
    int64_t K, int64_t group_size, int bits, cudaStream_t stream) {
  dim3 block(BlockN);
  dim3 grid(static_cast<uint32_t>((N + BlockN - 1) / BlockN),
            static_cast<uint32_t>((K + BlockK - 1) / BlockK));
  QmmFusedGemvKMajorAffineSplitKernel<XType, ScaleT, BiasT, AccumOutT, Bits,
                                      BlockN, BlockK>
      <<<grid, block, 0, stream>>>(x, wq, scales, biases, group_sums, out, N,
                                   K, group_size, bits);
  return CheckCuda(cudaPeekAtLastError(),
                   "fused qmm K-major split GEMV kernel launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename AccumOutT,
          int Bits, int BlockK>
Error LaunchQmmFusedGemvKMajorAffineSplitKernelBlockK(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, const float *group_sums, AccumOutT *out, int64_t N,
    int64_t K, int64_t group_size, int bits, int64_t block_n,
    cudaStream_t stream) {
  if (block_n <= 64) {
    return LaunchQmmFusedGemvKMajorAffineSplitKernelConfig<
        XType, ScaleT, BiasT, AccumOutT, Bits, 64, BlockK>(
        x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
        stream);
  }
  if (block_n <= 128) {
    return LaunchQmmFusedGemvKMajorAffineSplitKernelConfig<
        XType, ScaleT, BiasT, AccumOutT, Bits, 128, BlockK>(
        x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
        stream);
  }
  if (block_n <= 256) {
    return LaunchQmmFusedGemvKMajorAffineSplitKernelConfig<
        XType, ScaleT, BiasT, AccumOutT, Bits, 256, BlockK>(
        x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
        stream);
  }
  return LaunchQmmFusedGemvKMajorAffineSplitKernelConfig<
      XType, ScaleT, BiasT, AccumOutT, Bits, 512, BlockK>(
      x, wq, scales, biases, group_sums, out, N, K, group_size, bits, stream);
}

template <typename XType, typename ScaleT, typename BiasT, typename AccumOutT,
          int Bits>
Error LaunchQmmFusedGemvKMajorAffineSplitKernelOut(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, const float *group_sums, AccumOutT *out, int64_t N,
    int64_t K, int64_t group_size, int bits, int64_t block_n,
    int64_t block_k, cudaStream_t stream) {
  int64_t effective_block_n = block_n > 0 ? block_n : 256;
  if (group_size >= 128 && Bits != 2 && Bits != 8 && block_n <= 0) {
    effective_block_n = 128;
  }
  int64_t effective_block_k = block_k > 0 ? block_k : 128;
  if (effective_block_k <= 64) {
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBlockK<
        XType, ScaleT, BiasT, AccumOutT, Bits, 64>(
        x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
        effective_block_n, stream);
  }
  if (effective_block_k <= 128) {
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBlockK<
        XType, ScaleT, BiasT, AccumOutT, Bits, 128>(
        x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
        effective_block_n, stream);
  }
  if (effective_block_k <= 256) {
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBlockK<
        XType, ScaleT, BiasT, AccumOutT, Bits, 256>(
        x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
        effective_block_n, stream);
  }
  return LaunchQmmFusedGemvKMajorAffineSplitKernelBlockK<
      XType, ScaleT, BiasT, AccumOutT, Bits, 512>(
      x, wq, scales, biases, group_sums, out, N, K, group_size, bits,
      effective_block_n, stream);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits>
Error LaunchQmmFusedGemvKMajorAffineSplitKernelBits(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits, cudaStream_t stream,
    ScratchAllocator &scratch, int64_t block_n, int64_t block_k) {
  const char *gemv_impl = std::getenv("EJKERNEL_QMM_CUDA_KMAJOR_GEMV");
  size_t out_bytes = static_cast<size_t>(N) * sizeof(OutT);
  if (Error err = CheckCuda(cudaMemsetAsync(out, 0, out_bytes, stream),
                            "K-major GEMV output memset");
      err.failure()) {
    return err;
  }
  bool fold_bias_sum =
      gemv_impl != nullptr && std::string(gemv_impl) == "nosum" && Bits != 1;
  float *group_sums = nullptr;
  if (!fold_bias_sum) {
    size_t group_sum_bytes = static_cast<size_t>(n_groups) * sizeof(float);
    auto group_sums_opt = scratch.Allocate(group_sum_bytes, alignof(float));
    if (!group_sums_opt.has_value()) {
      return MakeInternal(
          "Failed to allocate group-sum scratch buffer for K-major GEMV.");
    }
    group_sums = reinterpret_cast<float *>(*group_sums_opt);
    QmmGemvGroupSumsKernel<XType>
        <<<static_cast<uint32_t>(n_groups), 256, 0, stream>>>(
            x, group_sums, K, n_groups, group_size);
    if (Error err = CheckCuda(cudaPeekAtLastError(),
                              "K-major GEMV group-sum kernel launch");
        err.failure()) {
      return err;
    }
  }
  return LaunchQmmFusedGemvKMajorAffineSplitKernelOut<
      XType, ScaleT, BiasT, OutT, Bits>(
      x, wq, scales, biases, group_sums, out, N, K, group_size, bits, block_n,
      block_k, stream);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedGemvKMajorAffineSplitKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_groups,
    int64_t group_size, int bits, cudaStream_t stream,
    ScratchAllocator &scratch, int64_t block_n, int64_t block_k) {
  switch (bits) {
  case 1:
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBits<
        XType, ScaleT, BiasT, OutT, 1>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch, block_n, block_k);
  case 2:
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBits<
        XType, ScaleT, BiasT, OutT, 2>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch, block_n, block_k);
  case 4:
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBits<
        XType, ScaleT, BiasT, OutT, 4>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch, block_n, block_k);
  case 8:
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBits<
        XType, ScaleT, BiasT, OutT, 8>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch, block_n, block_k);
  default:
    return LaunchQmmFusedGemvKMajorAffineSplitKernelBits<
        XType, ScaleT, BiasT, OutT, 0>(
        x, wq, scales, biases, out, N, K, n_groups, group_size, bits, stream,
        scratch, block_n, block_k);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int WarpsPerBlock, int OutputsPerWarp, int GroupSize>
__global__ void QmmFusedGemvAffineColVecKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits_runtime) {
  int warp_id = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int64_t n_base =
      (static_cast<int64_t>(blockIdx.x) * WarpsPerBlock + warp_id) *
      OutputsPerWarp;

  float grouped_acc[OutputsPerWarp];
  const uint32_t *wq_rows[OutputsPerWarp];
  const ScaleT *scale_rows[OutputsPerWarp];
  const BiasT *bias_rows[OutputsPerWarp];
  bool valid[OutputsPerWarp];
#pragma unroll
  for (int j = 0; j < OutputsPerWarp; ++j) {
    int64_t n = n_base + j;
    valid[j] = n < N;
    grouped_acc[j] = 0.0f;
    wq_rows[j] = valid[j] ? (wq + n * n_words) : nullptr;
    scale_rows[j] = valid[j] ? (scales + n * n_groups) : nullptr;
    bias_rows[j] = valid[j] ? (biases + n * n_groups) : nullptr;
  }

  for (int64_t g = 0; g < n_groups; ++g) {
    float scale_vals[OutputsPerWarp];
    float bias_vals[OutputsPerWarp];
#pragma unroll
    for (int j = 0; j < OutputsPerWarp; ++j) {
      scale_vals[j] = valid[j] ? ToFloatLocal(scale_rows[j][g]) : 0.0f;
      bias_vals[j] = valid[j] ? ToFloatLocal(bias_rows[j][g]) : 0.0f;
    }
    if constexpr (GroupSize == 32 || GroupSize == 64 || GroupSize == 128) {
      int64_t k_begin = g * GroupSize;
#pragma unroll
      for (int kk = 0; kk < GroupSize / 32; ++kk) {
        int64_t k_idx = k_begin + kk * 32 + lane;
        if (k_idx >= K) {
          continue;
        }
        float x_val = ToFloatLocal(x[k_idx]);
#pragma unroll
        for (int j = 0; j < OutputsPerWarp; ++j) {
          if (valid[j]) {
            uint32_t q =
                LoadPackedQConst<Bits>(wq_rows[j], k_idx, bits_runtime);
            float w_val =
                static_cast<float>(q) * scale_vals[j] + bias_vals[j];
            grouped_acc[j] += x_val * w_val;
          }
        }
      }
    } else {
      int64_t k_begin = g * group_size;
      int64_t k_end =
          (k_begin + group_size < K) ? (k_begin + group_size) : K;
      for (int64_t k_idx = k_begin + lane; k_idx < k_end; k_idx += 32) {
        float x_val = ToFloatLocal(x[k_idx]);
#pragma unroll
        for (int j = 0; j < OutputsPerWarp; ++j) {
          if (valid[j]) {
            uint32_t q =
                LoadPackedQConst<Bits>(wq_rows[j], k_idx, bits_runtime);
            float w_val =
                static_cast<float>(q) * scale_vals[j] + bias_vals[j];
            grouped_acc[j] += x_val * w_val;
          }
        }
      }
    }
  }

#pragma unroll
  for (int j = 0; j < OutputsPerWarp; ++j) {
    grouped_acc[j] = WarpReduceSum(grouped_acc[j]);
  }
  if (lane == 0) {
#pragma unroll
    for (int j = 0; j < OutputsPerWarp; ++j) {
      if (valid[j]) {
        out[n_base + j] = CastOut<OutT>(grouped_acc[j]);
      }
    }
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int OutputsPerWarp, int GroupSize>
Error LaunchQmmFusedGemvAffineColVecKernelGroup(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits, cudaStream_t stream) {
  constexpr int kWarpsPerBlock = 8;
  dim3 block(kWarpsPerBlock * 32);
  dim3 grid(static_cast<uint32_t>(
      (N + kWarpsPerBlock * OutputsPerWarp - 1) /
      (kWarpsPerBlock * OutputsPerWarp)));
  QmmFusedGemvAffineColVecKernel<XType, ScaleT, BiasT, OutT, Bits,
                                 kWarpsPerBlock, OutputsPerWarp, GroupSize>
      <<<grid, block, 0, stream>>>(x, wq, scales, biases, out, N, K, n_words,
                                   n_groups, group_size, bits);
  return CheckCuda(cudaPeekAtLastError(),
                   "fused qmm affine-col vector gemv kernel launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits, int OutputsPerWarp>
Error LaunchQmmFusedGemvAffineColVecKernelBits(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits, cudaStream_t stream) {
  return LaunchQmmFusedGemvAffineColVecKernelGroup<
      XType, ScaleT, BiasT, OutT, Bits, OutputsPerWarp, 0>(
      x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
      stream);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT,
          int Bits>
Error LaunchQmmFusedGemvWarpKernelBits(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits, int64_t mode,
    int64_t transpose, cudaStream_t stream) {
  if (mode == 0 && transpose != 0) {
    return LaunchQmmFusedGemvAffineColVecKernelBits<
        XType, ScaleT, BiasT, OutT, Bits, 4>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        stream);
  }
  constexpr int kWarpsPerBlock = 8;
  constexpr int kWarpsPerOutput = 1;
  constexpr int kOutputsPerBlock = kWarpsPerBlock / kWarpsPerOutput;
  dim3 block(kWarpsPerBlock * 32);
  dim3 grid(static_cast<uint32_t>((N + kOutputsPerBlock - 1) /
                                  kOutputsPerBlock));
  QmmFusedGemvWarpKernel<XType, ScaleT, BiasT, OutT, Bits, kWarpsPerBlock,
                         kWarpsPerOutput>
      <<<grid, block, 0, stream>>>(x, wq, scales, biases, out, N, K, n_words,
                                   n_groups, group_size, bits, mode,
                                   transpose);
  return CheckCuda(cudaPeekAtLastError(), "fused qmm warp-gemv kernel launch");
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedGemvKernel(const XType *x, const uint32_t *wq,
                               const ScaleT *scales, const BiasT *biases,
                               OutT *out, int64_t N, int64_t K,
                               int64_t n_words, int64_t n_groups,
                               int64_t group_size, int bits, int64_t mode,
                               int64_t transpose, cudaStream_t stream) {
  switch (bits) {
  case 1:
    return LaunchQmmFusedGemvWarpKernelBits<XType, ScaleT, BiasT, OutT, 1>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, stream);
  case 2:
    return LaunchQmmFusedGemvWarpKernelBits<XType, ScaleT, BiasT, OutT, 2>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, stream);
  case 4:
    return LaunchQmmFusedGemvWarpKernelBits<XType, ScaleT, BiasT, OutT, 4>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, stream);
  case 8:
    return LaunchQmmFusedGemvWarpKernelBits<XType, ScaleT, BiasT, OutT, 8>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, stream);
  default:
    return LaunchQmmFusedGemvWarpKernelBits<XType, ScaleT, BiasT, OutT, 0>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, stream);
  }
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedGemvRevSplitKernel(
    const XType *x, const uint32_t *wq, const ScaleT *scales,
    const BiasT *biases, OutT *out, int64_t N, int64_t K, int64_t n_words,
    int64_t n_groups, int64_t group_size, int bits, int64_t mode,
    int64_t transpose, int64_t revsplit_k_parts, cudaStream_t stream) {
  (void)revsplit_k_parts;
  return LaunchQmmFusedGemvKernel<XType, ScaleT, BiasT, OutT>(
      x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
      mode, transpose, stream);
}

template <typename XType, typename ScaleT, typename BiasT, typename OutT>
Error LaunchQmmFusedByFamily(
    QmmKernelFamily family, int64_t revsplit_k_parts, const XType *x,
    const uint32_t *wq, const ScaleT *scales, const BiasT *biases, OutT *out,
    int64_t M, int64_t N, int64_t K, int64_t n_words, int64_t n_groups,
    int64_t group_size, int bits, int64_t mode, int64_t transpose,
    bool packed_k_major, int64_t block_n, int64_t block_k,
    cudaStream_t stream, ScratchAllocator &scratch) {
  if (packed_k_major) {
    if ((family == QmmKernelFamily::kGemvSplitK ||
         family == QmmKernelFamily::kGemvRevSplitK) &&
        mode == 0 && transpose != 0 && M == 1) {
      const char *gemv_impl = std::getenv("EJKERNEL_QMM_CUDA_KMAJOR_GEMV");
      bool use_tile =
          gemv_impl != nullptr && std::string(gemv_impl) == "tile";
      if (use_tile) {
        return LaunchQmmFusedGemvKMajorAffineTileKernel<XType, ScaleT, BiasT,
                                                        OutT>(
            x, wq, scales, biases, out, N, K, group_size, bits, stream);
      }
      bool use_serial =
          gemv_impl != nullptr && std::string(gemv_impl) == "serial";
      if (use_serial) {
        return LaunchQmmFusedGemvKMajorAffineKernel<XType, ScaleT, BiasT, OutT>(
            x, wq, scales, biases, out, N, K, n_groups, group_size, bits,
            stream);
      }
      bool use_partial =
          gemv_impl != nullptr && std::string(gemv_impl) == "partial";
      if (use_partial) {
        return LaunchQmmFusedGemvKMajorAffinePartialKernel<XType, ScaleT, BiasT,
                                                           OutT>(
            x, wq, scales, biases, out, N, K, n_groups, group_size, bits,
            stream, scratch);
      }
      bool use_split =
          gemv_impl != nullptr &&
          (std::string(gemv_impl) == "split" ||
           std::string(gemv_impl) == "nosum");
      if (!use_split && (group_size == 32 || group_size == 64 ||
                         group_size == 128)) {
        return LaunchQmmFusedGemvKMajorAffinePartialKernel<XType, ScaleT, BiasT,
                                                           OutT>(
            x, wq, scales, biases, out, N, K, n_groups, group_size, bits,
            stream, scratch);
      }
      return LaunchQmmFusedGemvKMajorAffineSplitKernel<XType, ScaleT, BiasT,
                                                       OutT>(
          x, wq, scales, biases, out, N, K, n_groups, group_size, bits,
          stream, scratch, block_n, block_k);
    }
    return MakeInvalid(
        "CUDA K-major packed layout currently supports affine transpose=True "
        "GEMV only.");
  }
  if (family == QmmKernelFamily::kGemvSplitK) {
    return LaunchQmmFusedGemvKernel<XType, ScaleT, BiasT, OutT>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, stream);
  }
  if (family == QmmKernelFamily::kGemvRevSplitK) {
    return LaunchQmmFusedGemvRevSplitKernel<XType, ScaleT, BiasT, OutT>(
        x, wq, scales, biases, out, N, K, n_words, n_groups, group_size, bits,
        mode, transpose, revsplit_k_parts, stream);
  }
  return LaunchQmmFusedKernel<XType, ScaleT, BiasT, OutT>(
      x, wq, scales, biases, out, M, N, K, n_words, n_groups, group_size, bits,
      mode, transpose, stream);
}

struct CublasHandleCache {
  cublasHandle_t handle = nullptr;
  int device = -1;

  ~CublasHandleCache() {
    if (handle) {
      cublasDestroy(handle);
    }
  }

  Error Get(cublasHandle_t *out, cudaStream_t stream) {
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return MakeInternal("Failed to query current CUDA device.");
    }
    if (handle == nullptr || device != current_device) {
      if (handle) {
        cublasDestroy(handle);
        handle = nullptr;
      }
      if (Error err = CheckCublas(cublasCreate(&handle), "cublasCreate");
          err.failure()) {
        return err;
      }
      device = current_device;
      if (Error err = CheckCublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH),
                                  "cublasSetMathMode");
          err.failure()) {
        return err;
      }
    }
    if (Error err = CheckCublas(cublasSetStream(handle, stream),
                                "cublasSetStream");
        err.failure()) {
      return err;
    }
    *out = handle;
    return Error::Success();
  }
};

static thread_local CublasHandleCache g_cublas_cache;

struct CublasLtHandleCache {
  cublasLtHandle_t handle = nullptr;
  int device = -1;

  ~CublasLtHandleCache() {
    if (handle) {
      cublasLtDestroy(handle);
    }
  }

  Error Get(cublasLtHandle_t *out) {
    int current_device = -1;
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return MakeInternal("Failed to query current CUDA device.");
    }
    if (handle == nullptr || device != current_device) {
      if (handle) {
        cublasLtDestroy(handle);
        handle = nullptr;
      }
      if (Error err = CheckCublasLt(cublasLtCreate(&handle), "cublasLtCreate");
          err.failure()) {
        return err;
      }
      device = current_device;
    }
    *out = handle;
    return Error::Success();
  }
};

static thread_local CublasLtHandleCache g_cublaslt_cache;

Error RunCublasLtGemm(const void *w_deq, cudaDataType_t a_type, const void *x,
                      cudaDataType_t b_type, void *out,
                      xla::ffi::DataType out_dtype, int64_t M, int64_t N,
                      int64_t K, cudaStream_t stream, ScratchAllocator &scratch,
                      cublasComputeType_t compute_type,
                      bool a_is_row_major_nk) {
  cublasLtHandle_t lt_handle;
  if (Error err = g_cublaslt_cache.Get(&lt_handle); err.failure()) {
    return err;
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  cublasOperation_t transa =
      a_is_row_major_nk ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transb = CUBLAS_OP_N;

  if (Error err = CheckCublasLt(
          cublasLtMatmulDescCreate(&op_desc, compute_type, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
      err.failure()) {
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatmulDescSetAttribute(op_desc,
                                         CUBLASLT_MATMUL_DESC_TRANSA, &transa,
                                         sizeof(transa)),
          "cublasLtMatmulDescSetAttribute(TRANSA)");
      err.failure()) {
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatmulDescSetAttribute(op_desc,
                                         CUBLASLT_MATMUL_DESC_TRANSB, &transb,
                                         sizeof(transb)),
          "cublasLtMatmulDescSetAttribute(TRANSB)");
      err.failure()) {
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatrixLayoutCreate(&layout_a, a_type,
                                     a_is_row_major_nk ? static_cast<int64_t>(K)
                                                       : static_cast<int64_t>(N),
                                     a_is_row_major_nk ? static_cast<int64_t>(N)
                                                       : static_cast<int64_t>(K),
                                     a_is_row_major_nk ? static_cast<int64_t>(K)
                                                       : static_cast<int64_t>(N)),
          "cublasLtMatrixLayoutCreate(A)");
      err.failure()) {
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  if (Error err = CheckCublasLt(
          cublasLtMatrixLayoutCreate(&layout_b, b_type,
                                     static_cast<int64_t>(K),
                                     static_cast<int64_t>(M),
                                     static_cast<int64_t>(K)),
          "cublasLtMatrixLayoutCreate(B)");
      err.failure()) {
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }
  auto out_type = ToCudaDataType(out_dtype);
  if (Error err = CheckCublasLt(
          cublasLtMatrixLayoutCreate(&layout_c, out_type,
                                     static_cast<int64_t>(N),
                                     static_cast<int64_t>(M),
                                     static_cast<int64_t>(N)),
          "cublasLtMatrixLayoutCreate(C)");
      err.failure()) {
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  if (Error err = CheckCublasLt(
          cublasLtMatmulPreferenceCreate(&preference),
          "cublasLtMatmulPreferenceCreate");
      err.failure()) {
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  size_t workspace_limit = GetCublasLtWorkspaceLimit();
  if (Error err = CheckCublasLt(
          cublasLtMatmulPreferenceSetAttribute(
              preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
              &workspace_limit, sizeof(workspace_limit)),
          "cublasLtMatmulPreferenceSetAttribute");
      err.failure()) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  cublasLtMatmulHeuristicResult_t results[4] = {};
  int returned = 0;
  if (Error err = CheckCublasLt(
          cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, layout_a,
                                         layout_b, layout_c, layout_c,
                                         preference, 4, results, &returned),
          "cublasLtMatmulAlgoGetHeuristic");
      err.failure()) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return err;
  }

  if (returned == 0) {
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtMatrixLayoutDestroy(layout_c);
    cublasLtMatrixLayoutDestroy(layout_b);
    cublasLtMatrixLayoutDestroy(layout_a);
    cublasLtMatmulDescDestroy(op_desc);
    return MakeInternal("cublasLtMatmulAlgoGetHeuristic returned no algorithms.");
  }

  const auto &algo = results[0].algo;
  size_t workspace_size = results[0].workspaceSize;
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_opt =
        scratch.Allocate(workspace_size, /*alignment=*/256);
    if (!workspace_opt.has_value()) {
      cublasLtMatmulPreferenceDestroy(preference);
      cublasLtMatrixLayoutDestroy(layout_c);
      cublasLtMatrixLayoutDestroy(layout_b);
      cublasLtMatrixLayoutDestroy(layout_a);
      cublasLtMatmulDescDestroy(op_desc);
      return MakeInternal("Failed to allocate cublasLt workspace.");
    }
    workspace = *workspace_opt;
  }

  float alpha_f = 1.0f;
  float beta_f = 0.0f;
  half alpha_h = __float2half_rn(1.0f);
  half beta_h = __float2half_rn(0.0f);
  const void *alpha_ptr =
      ComputeTypeUsesHalfScalars(compute_type)
          ? static_cast<const void *>(&alpha_h)
          : static_cast<const void *>(&alpha_f);
  const void *beta_ptr =
      ComputeTypeUsesHalfScalars(compute_type)
          ? static_cast<const void *>(&beta_h)
          : static_cast<const void *>(&beta_f);

  cublasStatus_t status = cublasLtMatmul(
      lt_handle, op_desc, alpha_ptr, w_deq, layout_a, x, layout_b, beta_ptr, out,
      layout_c, out, layout_c, &algo, workspace, workspace_size, stream);

  cublasLtMatmulPreferenceDestroy(preference);
  cublasLtMatrixLayoutDestroy(layout_c);
  cublasLtMatrixLayoutDestroy(layout_b);
  cublasLtMatrixLayoutDestroy(layout_a);
  cublasLtMatmulDescDestroy(op_desc);

  return CheckCublasLt(status, "cublasLtMatmul");
}

template <typename ElementC>
Error RunCutlassGemm(const half *w_deq, const half *x_half, ElementC *out,
                     int64_t M, int64_t N, int64_t K, cudaStream_t stream,
                     ScratchAllocator &scratch) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                           ElementC, LayoutC,
                                           ElementAccumulator,
                                           cutlass::arch::OpClassTensorOp,
                                           cutlass::arch::Sm80>;

  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M),
                                        static_cast<int>(N),
                                        static_cast<int>(K));

  typename Gemm::Arguments args(
      problem_size,
      {reinterpret_cast<ElementA const *>(x_half), static_cast<int>(K)},
      {reinterpret_cast<ElementB const *>(w_deq), static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {1.0f, 0.0f});

  cutlass::Status can_impl = gemm_op.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    return MakeInvalid("CUTLASS GEMM configuration not supported.");
  }

  size_t workspace_size = gemm_op.get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_opt =
        scratch.Allocate(workspace_size, /*alignment=*/256);
    if (!workspace_opt.has_value()) {
      return MakeInternal("Failed to allocate CUTLASS workspace.");
    }
    workspace = *workspace_opt;
  }

  cutlass::Status status = gemm_op(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return MakeInternal("CUTLASS GEMM failed.");
  }
  return Error::Success();
}

template <typename Gemm>
Error RunCutlassGemmImpl(const half *w_deq, const half *x_half,
                         typename Gemm::ElementC *out, int64_t M, int64_t N,
                         int64_t K, cudaStream_t stream,
                         ScratchAllocator &scratch) {
  Gemm gemm_op;
  cutlass::gemm::GemmCoord problem_size(static_cast<int>(M),
                                        static_cast<int>(N),
                                        static_cast<int>(K));

  typename Gemm::Arguments args(
      problem_size,
      {reinterpret_cast<typename Gemm::ElementA const *>(x_half),
       static_cast<int>(K)},
      {reinterpret_cast<typename Gemm::ElementB const *>(w_deq),
       static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {out, static_cast<int>(N)},
      {1.0f, 0.0f});

  cutlass::Status can_impl = gemm_op.can_implement(args);
  if (can_impl != cutlass::Status::kSuccess) {
    return MakeInvalid("CUTLASS GEMM configuration not supported.");
  }

  size_t workspace_size = gemm_op.get_workspace_size(args);
  void *workspace = nullptr;
  if (workspace_size > 0) {
    auto workspace_opt =
        scratch.Allocate(workspace_size, /*alignment=*/256);
    if (!workspace_opt.has_value()) {
      return MakeInternal("Failed to allocate CUTLASS workspace.");
    }
    workspace = *workspace_opt;
  }

  cutlass::Status status = gemm_op(args, workspace, stream);
  if (status != cutlass::Status::kSuccess) {
    return MakeInternal("CUTLASS GEMM failed.");
  }
  return Error::Success();
}

template <typename ElementC>
Error RunCutlassGemmTuned(const half *w_deq, const half *x_half, ElementC *out,
                          int64_t M, int64_t N, int64_t K,
                          cudaStream_t stream, ScratchAllocator &scratch) {
  using ElementA = cutlass::half_t;
  using ElementB = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Epilogue =
      cutlass::epilogue::thread::LinearCombination<
          ElementC, 128 / cutlass::sizeof_bits<ElementC>::value,
          ElementAccumulator, ElementAccumulator>;

  using Gemm128x64x64 = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 64, 64>,
      cutlass::gemm::GemmShape<64, 32, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  using Gemm64x128x64 = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<64, 128, 64>,
      cutlass::gemm::GemmShape<32, 64, 64>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  using Gemm128x128x32 = cutlass::gemm::device::Gemm<
      ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
      ElementAccumulator, cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 32>,
      cutlass::gemm::GemmShape<64, 64, 32>,
      cutlass::gemm::GemmShape<16, 8, 16>, Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

  if (N >= 8192 && M >= 8192) {
    return RunCutlassGemmImpl<Gemm128x128x32>(w_deq, x_half, out, M, N, K,
                                             stream, scratch);
  }
  if (N >= M) {
    return RunCutlassGemmImpl<Gemm64x128x64>(w_deq, x_half, out, M, N, K,
                                            stream, scratch);
  }
  return RunCutlassGemmImpl<Gemm128x64x64>(w_deq, x_half, out, M, N, K, stream,
                                          scratch);
}

struct DequantCache {
  const void *wq = nullptr;
  const void *scales = nullptr;
  const void *biases = nullptr;
  int64_t k = 0;
  int64_t n = 0;
  int64_t n_words = 0;
  int64_t n_groups = 0;
  int64_t group_size = 0;
  int bits = 0;
  int mode = -1;
  int64_t transpose = 0;
  bool packed_k_major = false;
  int device = -1;
  size_t bytes = 0;
  half *buffer = nullptr;
  size_t bytes_bf16 = 0;
  __nv_bfloat16 *buffer_bf16 = nullptr;
  bool bf16_valid = false;

  ~DequantCache() {
    if (buffer) {
      cudaFree(buffer);
      buffer = nullptr;
    }
    if (buffer_bf16) {
      cudaFree(buffer_bf16);
      buffer_bf16 = nullptr;
    }
  }

  void Invalidate() {
    wq = nullptr;
    scales = nullptr;
    biases = nullptr;
    k = 0;
    n = 0;
    n_words = 0;
    n_groups = 0;
    group_size = 0;
    bits = 0;
    mode = -1;
    transpose = 0;
    packed_k_major = false;
    device = -1;
    bf16_valid = false;
  }
};

static thread_local DequantCache g_dequant_cache;

inline bool UseDequantCache() {
  static int cached = -1;
  if (cached != -1) {
    return cached == 1;
  }
  const char *env = std::getenv("EJKERNEL_QMM_CUDA_CACHE");
  if (!env) {
    cached = 0;
    return false;
  }
  if (env[0] == '1' || env[0] == 't' || env[0] == 'T' || env[0] == 'y' ||
      env[0] == 'Y') {
    cached = 1;
    return true;
  }
  cached = 0;
  return false;
}

Error QuantizedMatmulCuda(AnyBuffer x, AnyBuffer wq, AnyBuffer scales,
                          std::optional<AnyBuffer> biases,
                          Result<AnyBuffer> out, int64_t group_size,
                          int64_t bits, int64_t mode, int64_t transpose,
                          int64_t gemv_mode, int64_t revsplit_k,
                          int64_t revsplit_k_parts, int64_t block_n,
                          int64_t block_k, cudaStream_t stream,
                          ScratchAllocator scratch) {
  if (bits < 1 || bits > 8) {
    return MakeInvalid("CUDA quantized_matmul supports bits in [1,8].");
  }
  if (mode == 1 && bits != kBits4) {
    return MakeInvalid("nf4 mode requires bits=4 on CUDA.");
  }
  if (mode == 2 && (group_size != 32 || bits != kBits4)) {
    return MakeInvalid("mxfp4 requires group_size=32 and bits=4.");
  }
  if (mode == 3 && (group_size != 32 || bits != kBits8)) {
    return MakeInvalid("mxfp8 requires group_size=32 and bits=8.");
  }
  if (mode == 4 && (group_size != 16 || bits != kBits4)) {
    return MakeInvalid("nvfp4 requires group_size=16 and bits=4.");
  }
  if (mode == 5 && (group_size != 16 || bits != kBits8)) {
    return MakeInvalid("nvfp8 requires group_size=16 and bits=8.");
  }

  Span<const int64_t> x_dims = x.dimensions();
  Span<const int64_t> w_dims = wq.dimensions();
  Span<const int64_t> s_dims = scales.dimensions();
  Span<const int64_t> o_dims = out->dimensions();

  if (x_dims.size() != 2 || w_dims.size() != 2 || s_dims.size() != 2 ||
      o_dims.size() != 2) {
    return MakeInvalid("All inputs/outputs must be rank-2 matrices.");
  }

  int64_t M = x_dims[0];
  int64_t K = x_dims[1];
  int64_t K_w = w_dims[0];
  int64_t N_words = w_dims[1];
  bool use_bf16_gemm = x.element_type() == xla::ffi::DataType::BF16;
  bool transposed = transpose != 0;
  bool packed_k_major = false;

  QmmKernelFamily kernel_family = QmmKernelFamily::kGemm;
  int64_t revsplit_parts_effective = 0;
  if (Error err = ResolveQmmKernelFamily(M, mode, bits, gemv_mode, revsplit_k,
                                         revsplit_k_parts, &kernel_family,
                                         &revsplit_parts_effective);
      err.failure()) {
    return err;
  }

  if (group_size <= 0) {
    return MakeInvalid("group_size must be positive.");
  }

  int64_t n_groups = 0;
  int64_t N = 0;
  if (transposed) {
    if ((K % group_size) != 0) {
      return MakeInvalid("K must be divisible by group_size for transpose=True.");
    }
    n_groups = K / group_size;
    int64_t expected_words = (static_cast<int64_t>(K) * bits + 31) / 32;
    if (s_dims[1] == n_groups && w_dims[1] == expected_words &&
        w_dims[0] == s_dims[0]) {
      // Canonical ejkernel layout: Wq is (N, K/pack), metadata is (N, groups).
      N = w_dims[0];
      N_words = w_dims[1];
    } else if (s_dims[0] == n_groups && w_dims[0] == expected_words &&
               w_dims[1] == s_dims[1]) {
      // GemLite-compatible K-major layout: Wq is (K/pack, N), metadata is
      // (groups, N). This is accepted for native decode kernels.
      packed_k_major = true;
      N = w_dims[1];
      N_words = expected_words;
    } else {
      return MakeInvalid(
          "packed weight/scales shape does not match transpose=True layout. "
          "Expected either Wq=(N,ceil(K*bits/32)), scales=(N,K/group_size) "
          "or Wq=(ceil(K*bits/32),N), scales=(K/group_size,N).");
    }
  } else {
    // Weights stored in KxN (packed), scales/biases stored in (K, N/group_size).
    if (K_w != K) {
      return MakeInvalid("Weight K dimension does not match input K.");
    }
    if (s_dims[0] != K) {
      return MakeInvalid("scales shape must be (K, N/group_size).");
    }
    n_groups = s_dims[1];
    N = n_groups * group_size;
    int64_t expected_words = (static_cast<int64_t>(N) * bits + 31) / 32;
    if (N_words != expected_words) {
      return MakeInvalid("packed weight shape does not match N and bits.");
    }
  }

  if (o_dims[0] != M || o_dims[1] != N) {
    return MakeInvalid("output shape must be (M, N).");
  }

  if (wq.element_type() != xla::ffi::DataType::U32) {
    return MakeInvalid("wq must be uint32 packed codes.");
  }
  auto x_dtype = x.element_type();
  auto out_dtype = out->element_type();
  if (x_dtype != xla::ffi::DataType::F16 &&
      x_dtype != xla::ffi::DataType::BF16 &&
      x_dtype != xla::ffi::DataType::F32) {
    return MakeInvalid("x dtype must be float16/float32/bfloat16.");
  }
  if (out_dtype != xla::ffi::DataType::F16 &&
      out_dtype != xla::ffi::DataType::BF16 &&
      out_dtype != xla::ffi::DataType::F32) {
    return MakeInvalid("output dtype must be float16/float32/bfloat16.");
  }
  if (out_dtype == xla::ffi::DataType::F16 &&
      x_dtype != xla::ffi::DataType::F16) {
    return MakeInvalid("output dtype must match x dtype.");
  }
  if (out_dtype == xla::ffi::DataType::BF16 &&
      x_dtype != xla::ffi::DataType::BF16) {
    return MakeInvalid("output dtype must match x dtype.");
  }

  if (mode == 0) {
    if (!biases.has_value()) {
      return MakeInvalid("affine mode requires biases.");
    }
    Span<const int64_t> b_dims = biases->dimensions();
    bool biases_match = false;
    if (packed_k_major) {
      biases_match = b_dims.size() == 2 && b_dims[0] == n_groups &&
                     b_dims[1] == N;
    } else {
      biases_match = b_dims.size() == 2 && b_dims[0] == s_dims[0] &&
                     b_dims[1] == n_groups;
    }
    if (!biases_match) {
      return MakeInvalid("biases shape must match scales shape.");
    }
  } else if (mode == 1) {
    if (biases.has_value()) {
      return MakeInvalid("nf4 mode does not accept biases.");
    }
  } else if (mode == 2 || mode == 3 || mode == 4 || mode == 5) {
    if (biases.has_value()) {
      return MakeInvalid("mxfp4/mxfp8/nvfp4/nvfp8 modes do not accept biases.");
    }
  } else {
    return MakeInvalid("Unsupported mode for CUDA quantized_matmul.");
  }

  if (!IsFiniteInt64(M) || !IsFiniteInt64(N) || !IsFiniteInt64(K)) {
    return MakeInvalid("M/N/K are too large for CUDA GEMM.");
  }

  bool is_gemv_family = kernel_family == QmmKernelFamily::kGemvSplitK ||
                        kernel_family == QmmKernelFamily::kGemvRevSplitK;
  const char *kmajor_gemv_impl = std::getenv("EJKERNEL_QMM_CUDA_KMAJOR_GEMV");
  bool use_kmajor_dense_gemv =
      packed_k_major && M == 1 && mode == 0 && kmajor_gemv_impl != nullptr &&
      std::string(kmajor_gemv_impl) == "dense";
  bool use_fused_path = false;
  if (is_gemv_family) {
    use_fused_path =
        mode == 0 && bits >= 1 && bits <= 8 ? true : UseExperimentalGemvQmm();
  } else {
    use_fused_path = UseFusedQmm();
  }
  if (use_kmajor_dense_gemv) {
    use_fused_path = false;
  }
  if (packed_k_major && mode != 0) {
    return MakeInvalid(
        "CUDA K-major packed layout currently supports affine mode only.");
  }

  if (use_fused_path) {
    const uint32_t *wq_ptr =
        static_cast<const uint32_t *>(wq.untyped_data());
    auto scales_dtype = scales.element_type();

    auto run_fused = [&](auto *out_ptr) -> Error {
      using OutT = std::remove_pointer_t<decltype(out_ptr)>;
      if (mode == 0 || mode == 1) {
        if (scales_dtype == xla::ffi::DataType::F32) {
          const float *scales_ptr =
              static_cast<const float *>(scales.untyped_data());
          const float *bias_ptr = nullptr;
          if (mode == 0) {
            bias_ptr = static_cast<const float *>(biases->untyped_data());
          }
          if (x_dtype == xla::ffi::DataType::F16) {
            return LaunchQmmFusedByFamily<half, float, float, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
                bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
                mode, transpose, packed_k_major, block_n, block_k, stream,
                scratch);
          }
          if (x_dtype == xla::ffi::DataType::F32) {
            return LaunchQmmFusedByFamily<float, float, float, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
                bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
                mode, transpose, packed_k_major, block_n, block_k, stream,
                scratch);
          }
          if (x_dtype == xla::ffi::DataType::BF16) {
            return LaunchQmmFusedByFamily<__nv_bfloat16, float, float, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
                scales_ptr, bias_ptr, out_ptr, M, N, K, N_words, n_groups,
                group_size, bits, mode, transpose, packed_k_major, block_n,
                block_k, stream, scratch);
          }
          return MakeInvalid("x dtype must be float16/float32/bfloat16.");
        }
        if (scales_dtype == xla::ffi::DataType::F16) {
          const half *scales_ptr =
              static_cast<const half *>(scales.untyped_data());
          const half *bias_ptr = nullptr;
          if (mode == 0) {
            bias_ptr = static_cast<const half *>(biases->untyped_data());
          }
          if (x_dtype == xla::ffi::DataType::F16) {
            return LaunchQmmFusedByFamily<half, half, half, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
                bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
                mode, transpose, packed_k_major, block_n, block_k, stream,
                scratch);
          }
          if (x_dtype == xla::ffi::DataType::F32) {
            return LaunchQmmFusedByFamily<float, half, half, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
                bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
                mode, transpose, packed_k_major, block_n, block_k, stream,
                scratch);
          }
          if (x_dtype == xla::ffi::DataType::BF16) {
            return LaunchQmmFusedByFamily<__nv_bfloat16, half, half, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
                scales_ptr, bias_ptr, out_ptr, M, N, K, N_words, n_groups,
                group_size, bits, mode, transpose, packed_k_major, block_n,
                block_k, stream, scratch);
          }
          return MakeInvalid("x dtype must be float16/float32/bfloat16.");
        }
        if (scales_dtype == xla::ffi::DataType::BF16) {
          const __nv_bfloat16 *scales_ptr =
              static_cast<const __nv_bfloat16 *>(scales.untyped_data());
          const __nv_bfloat16 *bias_ptr = nullptr;
          if (mode == 0) {
            bias_ptr =
                static_cast<const __nv_bfloat16 *>(biases->untyped_data());
          }
          if (x_dtype == xla::ffi::DataType::F16) {
            return LaunchQmmFusedByFamily<half, __nv_bfloat16, __nv_bfloat16,
                                          OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
                bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
                mode, transpose, packed_k_major, block_n, block_k, stream,
                scratch);
          }
          if (x_dtype == xla::ffi::DataType::F32) {
            return LaunchQmmFusedByFamily<float, __nv_bfloat16, __nv_bfloat16,
                                          OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
                bias_ptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
                mode, transpose, packed_k_major, block_n, block_k, stream,
                scratch);
          }
          if (x_dtype == xla::ffi::DataType::BF16) {
            return LaunchQmmFusedByFamily<__nv_bfloat16, __nv_bfloat16,
                                          __nv_bfloat16, OutT>(
                kernel_family, revsplit_parts_effective,
                static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
                scales_ptr, bias_ptr, out_ptr, M, N, K, N_words, n_groups,
                group_size, bits, mode, transpose, packed_k_major, block_n,
                block_k, stream, scratch);
          }
          return MakeInvalid("x dtype must be float16/float32/bfloat16.");
        }
        return MakeInvalid(
            "scales dtype must be float32/float16/bfloat16 for affine/nf4.");
      }

      if (scales_dtype == xla::ffi::DataType::U8) {
        const uint8_t *scales_ptr =
            static_cast<const uint8_t *>(scales.untyped_data());
        if (x_dtype == xla::ffi::DataType::F16) {
          return LaunchQmmFusedByFamily<half, uint8_t, uint8_t, OutT>(
              kernel_family, revsplit_parts_effective,
              static_cast<const half *>(x.untyped_data()), wq_ptr, scales_ptr,
              nullptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, transpose, packed_k_major, block_n, block_k, stream,
              scratch);
        }
        if (x_dtype == xla::ffi::DataType::F32) {
          return LaunchQmmFusedByFamily<float, uint8_t, uint8_t, OutT>(
              kernel_family, revsplit_parts_effective,
              static_cast<const float *>(x.untyped_data()), wq_ptr, scales_ptr,
              nullptr, out_ptr, M, N, K, N_words, n_groups, group_size, bits,
              mode, transpose, packed_k_major, block_n, block_k, stream,
              scratch);
        }
        if (x_dtype == xla::ffi::DataType::BF16) {
          return LaunchQmmFusedByFamily<__nv_bfloat16, uint8_t, uint8_t, OutT>(
              kernel_family, revsplit_parts_effective,
              static_cast<const __nv_bfloat16 *>(x.untyped_data()), wq_ptr,
              scales_ptr, nullptr, out_ptr, M, N, K, N_words, n_groups,
              group_size, bits, mode, transpose, packed_k_major, block_n,
              block_k, stream, scratch);
        }
        return MakeInvalid("x dtype must be float16/float32/bfloat16.");
      }

      return MakeInvalid("scales dtype must be uint8 for mxfp4/mxfp8/nvfp4/nvfp8 modes.");
    };

    if (out_dtype == xla::ffi::DataType::F32) {
      return run_fused(reinterpret_cast<float *>(out->untyped_data()));
    }
    if (out_dtype == xla::ffi::DataType::F16) {
      return run_fused(reinterpret_cast<half *>(out->untyped_data()));
    }
    if (out_dtype == xla::ffi::DataType::BF16) {
      return run_fused(reinterpret_cast<__nv_bfloat16 *>(out->untyped_data()));
    }
    return MakeInvalid("output dtype must be float16/float32/bfloat16.");
  }

  const uint32_t *wq_ptr = static_cast<const uint32_t *>(wq.untyped_data());
  half *w_deq = nullptr;
  bool need_dequantize = true;
  int current_device = -1;
  bool use_cache = UseDequantCache();

  if (use_cache) {
    if (cudaGetDevice(&current_device) != cudaSuccess) {
      return MakeInternal("Failed to query current CUDA device.");
    }
    bool have_cached_weights =
        use_bf16_gemm ? (g_dequant_cache.buffer_bf16 != nullptr &&
                         g_dequant_cache.bf16_valid)
                      : (g_dequant_cache.buffer != nullptr);
    bool match = have_cached_weights &&
                 g_dequant_cache.device == current_device &&
                 g_dequant_cache.wq == wq_ptr &&
                 g_dequant_cache.scales == scales.untyped_data() &&
                 g_dequant_cache.biases ==
                     (biases.has_value() ? biases->untyped_data() : nullptr) &&
                 g_dequant_cache.k == K && g_dequant_cache.n == N &&
                 g_dequant_cache.n_words == N_words &&
                 g_dequant_cache.n_groups == n_groups &&
                 g_dequant_cache.group_size == group_size &&
                 g_dequant_cache.bits == bits && g_dequant_cache.mode == mode &&
                 g_dequant_cache.transpose == transpose &&
                 g_dequant_cache.packed_k_major == packed_k_major;
    if (match) {
      if (!use_bf16_gemm) {
        w_deq = g_dequant_cache.buffer;
      }
      need_dequantize = false;
    } else {
      if (!use_bf16_gemm) {
        size_t w_deq_bytes =
            static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(half);
        if (g_dequant_cache.buffer == nullptr ||
            g_dequant_cache.bytes < w_deq_bytes ||
            g_dequant_cache.device != current_device) {
          if (g_dequant_cache.buffer) {
            cudaFree(g_dequant_cache.buffer);
            g_dequant_cache.buffer = nullptr;
            g_dequant_cache.bytes = 0;
          }
          cudaError_t alloc_err =
              cudaMalloc(&g_dequant_cache.buffer, w_deq_bytes);
          if (alloc_err != cudaSuccess) {
            return MakeInternal("Failed to allocate cached dequant buffer.");
          }
          g_dequant_cache.bytes = w_deq_bytes;
          g_dequant_cache.device = current_device;
        }
      }
      g_dequant_cache.wq = wq_ptr;
      g_dequant_cache.scales = scales.untyped_data();
      g_dequant_cache.biases =
          biases.has_value() ? biases->untyped_data() : nullptr;
      g_dequant_cache.k = K;
      g_dequant_cache.n = N;
      g_dequant_cache.n_words = N_words;
      g_dequant_cache.n_groups = n_groups;
      g_dequant_cache.group_size = group_size;
      g_dequant_cache.bits = bits;
      g_dequant_cache.mode = mode;
      g_dequant_cache.transpose = transpose;
      g_dequant_cache.packed_k_major = packed_k_major;
      g_dequant_cache.device = current_device;
      g_dequant_cache.bf16_valid = false;
      if (!use_bf16_gemm) {
        w_deq = g_dequant_cache.buffer;
      }
      need_dequantize = true;
    }
  } else {
    if (!use_bf16_gemm) {
      size_t w_deq_bytes =
          static_cast<size_t>(K) * static_cast<size_t>(N) * sizeof(half);
      auto w_deq_opt = scratch.Allocate(w_deq_bytes, alignof(half));
      if (!w_deq_opt.has_value()) {
        return MakeInternal(
            "Failed to allocate scratch buffer for dequantized weights.");
      }
      void *w_deq_ptr = *w_deq_opt;
      w_deq = reinterpret_cast<half *>(w_deq_ptr);
    }
    need_dequantize = true;
  }

  dim3 block(256);
  int64_t total = K * N;
  int64_t elems_per_block =
      static_cast<int64_t>(block.x) * kDequantElemsPerThread;
  dim3 grid(static_cast<uint32_t>((total + elems_per_block - 1) /
                                  elems_per_block));

  auto scales_dtype = scales.element_type();
  if (need_dequantize && !use_bf16_gemm) {
    if (transpose != 0) {
      auto launch_dequant_half = [&](auto *scales_ptr, auto *biases_ptr) {
        using ScaleT = std::remove_pointer_t<decltype(scales_ptr)>;
        using BiasT = std::remove_pointer_t<decltype(biases_ptr)>;
        if (packed_k_major) {
          DequantKMajorAffineKernel<ScaleT, BiasT, half>
              <<<grid, block, 0, stream>>>(wq_ptr, scales_ptr, biases_ptr,
                                           w_deq, K, N, n_groups, group_size,
                                           bits);
        } else {
          DequantToHalfKernel<ScaleT, BiasT><<<grid, block, 0, stream>>>(
              wq_ptr, scales_ptr, biases_ptr, w_deq, K, N, N_words, n_groups,
              group_size, bits, mode, transpose);
        }
      };

      if (mode == 0) {
        auto biases_buf = *biases;
        auto bias_dtype = biases_buf.element_type();
        if (bias_dtype != scales_dtype) {
          return MakeInvalid("biases dtype must match scales dtype.");
        }
        if (scales_dtype == xla::ffi::DataType::F32) {
          launch_dequant_half(
              static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()));
        } else if (scales_dtype == xla::ffi::DataType::F16) {
          launch_dequant_half(
              static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()));
        } else if (scales_dtype == xla::ffi::DataType::BF16) {
          launch_dequant_half(
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()));
        } else {
          return MakeInvalid(
              "scales/biases dtype must be float32/float16/bfloat16.");
        }
      } else if (mode == 1) {
        if (scales_dtype == xla::ffi::DataType::F32) {
          launch_dequant_half(static_cast<const float *>(scales.untyped_data()),
                              static_cast<const float *>(nullptr));
        } else if (scales_dtype == xla::ffi::DataType::F16) {
          launch_dequant_half(static_cast<const half *>(scales.untyped_data()),
                              static_cast<const half *>(nullptr));
        } else if (scales_dtype == xla::ffi::DataType::BF16) {
          launch_dequant_half(
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(nullptr));
        } else {
          return MakeInvalid(
              "scales dtype must be float32/float16/bfloat16 for nf4.");
        }
      } else if (mode == 2) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
          return MakeInvalid("mxfp4 requires scales dtype uint8 and bits=4.");
        }
        launch_dequant_half(
            static_cast<const uint8_t *>(scales.untyped_data()),
            static_cast<const uint8_t *>(nullptr));
      } else if (mode == 3) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
          return MakeInvalid("mxfp8 requires scales dtype uint8 and bits=8.");
        }
        launch_dequant_half(
            static_cast<const uint8_t *>(scales.untyped_data()),
            static_cast<const uint8_t *>(nullptr));
      } else if (mode == 4) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
          return MakeInvalid("nvfp4 requires scales dtype uint8 and bits=4.");
        }
        launch_dequant_half(
            static_cast<const uint8_t *>(scales.untyped_data()),
            static_cast<const uint8_t *>(nullptr));
      } else if (mode == 5) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
          return MakeInvalid("nvfp8 requires scales dtype uint8 and bits=8.");
        }
        launch_dequant_half(
            static_cast<const uint8_t *>(scales.untyped_data()),
            static_cast<const uint8_t *>(nullptr));
      } else {
        return MakeInvalid("Unsupported mode for CUDA quantized_matmul.");
      }
    } else if (mode == 0) {
      auto biases_buf = *biases;
      auto bias_dtype = biases_buf.element_type();
      if (bias_dtype != scales_dtype) {
        return MakeInvalid("biases dtype must match scales dtype.");
      }
      if (scales_dtype == xla::ffi::DataType::F32) {
        switch (bits) {
        case 4:
          ResolveLaunchDequantAffineBits4F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 8:
          ResolveLaunchDequantAffineBits8F32(group_size)(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        default:
          DequantToHalfKernel<float, float><<<grid, block, 0, stream>>>(
              wq_ptr, static_cast<const float *>(scales.untyped_data()),
              static_cast<const float *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, n_groups, group_size, bits, mode, transpose);
          break;
        }
      } else if (scales_dtype == xla::ffi::DataType::F16) {
        switch (bits) {
        case 4:
          ResolveLaunchDequantAffineBits4F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 8:
          ResolveLaunchDequantAffineBits8F16(group_size)(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, group_size, n_groups, grid, block, stream);
          break;
        default:
          DequantToHalfKernel<half, half><<<grid, block, 0, stream>>>(
              wq_ptr, static_cast<const half *>(scales.untyped_data()),
              static_cast<const half *>(biases_buf.untyped_data()), w_deq, K,
              N, N_words, n_groups, group_size, bits, mode, transpose);
          break;
        }
      } else if (scales_dtype == xla::ffi::DataType::BF16) {
        switch (bits) {
        case 4:
          ResolveLaunchDequantAffineBits4BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        case 8:
          ResolveLaunchDequantAffineBits8BF16(group_size)(
              wq_ptr,
              static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
              static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()),
              w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
          break;
        default:
          DequantToHalfKernel<__nv_bfloat16, __nv_bfloat16>
              <<<grid, block, 0, stream>>>(
                  wq_ptr,
                  static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
                  static_cast<const __nv_bfloat16 *>(
                      biases_buf.untyped_data()),
                  w_deq, K, N, N_words, n_groups, group_size, bits, mode,
                  transpose);
          break;
        }
      } else {
        return MakeInvalid(
            "scales/biases dtype must be float32/float16/bfloat16.");
      }
    } else if (mode == 1) {
      if (scales_dtype == xla::ffi::DataType::F32) {
        ResolveLaunchDequantNf4F32(group_size)(wq_ptr,
                            static_cast<const float *>(scales.untyped_data()),
                            w_deq, K, N, N_words, group_size, n_groups, grid,
                            block, stream);
      } else if (scales_dtype == xla::ffi::DataType::F16) {
        ResolveLaunchDequantNf4F16(group_size)(wq_ptr,
                            static_cast<const half *>(scales.untyped_data()),
                            w_deq, K, N, N_words, group_size, n_groups, grid,
                            block, stream);
      } else if (scales_dtype == xla::ffi::DataType::BF16) {
        ResolveLaunchDequantNf4BF16(group_size)(
            wq_ptr, static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
            w_deq, K, N, N_words, group_size, n_groups, grid, block, stream);
      } else {
        return MakeInvalid(
            "scales dtype must be float32/float16/bfloat16 for nf4.");
      }
    } else if (mode == 2) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
        return MakeInvalid("mxfp4 requires scales dtype uint8 and bits=4.");
      }
      LaunchDequantMxFp4(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else if (mode == 3) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
        return MakeInvalid("mxfp8 requires scales dtype uint8 and bits=8.");
      }
      LaunchDequantMxFp8(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else if (mode == 4) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
        return MakeInvalid("nvfp4 requires scales dtype uint8 and bits=4.");
      }
      LaunchDequantNvFp4(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else if (mode == 5) {
      if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
        return MakeInvalid("nvfp8 requires scales dtype uint8 and bits=8.");
      }
      LaunchDequantNvFp8(wq_ptr,
                         static_cast<const uint8_t *>(scales.untyped_data()),
                         w_deq, K, N, N_words, n_groups, grid, block, stream);
    } else {
      return MakeInvalid("Unsupported mode for CUDA quantized_matmul.");
    }

    if (Error err = CheckCuda(cudaPeekAtLastError(),
                              "dequant kernel launch");
        err.failure()) {
      return err;
    }
  }

  const half *x_half = nullptr;
  const __nv_bfloat16 *x_bf16 = nullptr;
  const float *x_f32 = nullptr;
  size_t x_elems = static_cast<size_t>(M) * static_cast<size_t>(K);
  // When x is fp32, keep it in fp32 and pair with an fp32 upcast of w_deq for
  // a full-precision fp32 GEMM that matches XLA's dot_general(fp32, fp16, pref=fp32).
  bool x_is_f32 = (!use_bf16_gemm && x.element_type() == xla::ffi::DataType::F32);

  if (use_bf16_gemm) {
    x_bf16 = static_cast<const __nv_bfloat16 *>(x.untyped_data());
  } else if (x.element_type() == xla::ffi::DataType::F16) {
    x_half = static_cast<const half *>(x.untyped_data());
  } else if (x_is_f32) {
    x_f32 = static_cast<const float *>(x.untyped_data());
  } else {
    size_t x_half_bytes = x_elems * sizeof(half);
    auto x_half_opt = scratch.Allocate(x_half_bytes, alignof(half));
    if (!x_half_opt.has_value()) {
      return MakeInternal("Failed to allocate scratch buffer for input cast.");
    }
    void *x_half_ptr = *x_half_opt;
    half *x_half_out = reinterpret_cast<half *>(x_half_ptr);
    dim3 cblock(256);
    dim3 cgrid(static_cast<uint32_t>((x_elems + cblock.x - 1) / cblock.x));
    if (x.element_type() == xla::ffi::DataType::BF16) {
      convert_bf16_to_f16<<<cgrid, cblock, 0, stream>>>(
          static_cast<const __nv_bfloat16 *>(x.untyped_data()), x_half_out,
          static_cast<int64_t>(x_elems));
    } else {
      return MakeInvalid("x dtype must be float16/float32/bfloat16.");
    }
    if (Error err =
            CheckCuda(cudaPeekAtLastError(), "input cast kernel launch");
        err.failure()) {
      return err;
    }
    x_half = x_half_out;
  }

  bool need_bf16_out_cast =
      out_dtype == xla::ffi::DataType::BF16 && !use_bf16_gemm;
  void *gemm_out_ptr = out->untyped_data();
  xla::ffi::DataType gemm_out_dtype = out_dtype;
  void *bf16_cast_src = gemm_out_ptr;
  xla::ffi::DataType bf16_cast_dtype = gemm_out_dtype;
  if (need_bf16_out_cast) {
    size_t out_float_bytes =
        static_cast<size_t>(M) * static_cast<size_t>(N) * sizeof(float);
    auto out_float_opt = scratch.Allocate(out_float_bytes, alignof(float));
    if (!out_float_opt.has_value()) {
      return MakeInternal("Failed to allocate scratch buffer for bf16 output cast.");
    }
    gemm_out_ptr = *out_float_opt;
    gemm_out_dtype = xla::ffi::DataType::F32;
    bf16_cast_src = gemm_out_ptr;
    bf16_cast_dtype = gemm_out_dtype;
  }

  __nv_bfloat16 *w_deq_bf16 = nullptr;
  if (use_bf16_gemm) {
    size_t w_bf16_bytes =
        static_cast<size_t>(K) * static_cast<size_t>(N) *
        sizeof(__nv_bfloat16);
    bool need_bf16_dequantize = need_dequantize;
    if (use_cache) {
      if (g_dequant_cache.buffer_bf16 == nullptr ||
          g_dequant_cache.bytes_bf16 < w_bf16_bytes ||
          g_dequant_cache.device != current_device) {
        if (g_dequant_cache.buffer_bf16) {
          cudaFree(g_dequant_cache.buffer_bf16);
          g_dequant_cache.buffer_bf16 = nullptr;
          g_dequant_cache.bytes_bf16 = 0;
        }
        cudaError_t alloc_err =
            cudaMalloc(&g_dequant_cache.buffer_bf16, w_bf16_bytes);
        if (alloc_err != cudaSuccess) {
          return MakeInternal(
              "Failed to allocate cached bf16 dequant buffer.");
        }
        g_dequant_cache.bytes_bf16 = w_bf16_bytes;
        g_dequant_cache.bf16_valid = false;
      }
      w_deq_bf16 = g_dequant_cache.buffer_bf16;
      need_bf16_dequantize = need_bf16_dequantize || !g_dequant_cache.bf16_valid;
    } else {
      auto w_bf16_opt = scratch.Allocate(w_bf16_bytes, alignof(__nv_bfloat16));
      if (!w_bf16_opt.has_value()) {
        return MakeInternal(
            "Failed to allocate bf16 buffer for dequantized weights.");
      }
      w_deq_bf16 = reinterpret_cast<__nv_bfloat16 *>(*w_bf16_opt);
      need_bf16_dequantize = true;
    }

    auto launch_dequant_bf16 = [&](auto *scales_ptr, auto *biases_ptr) -> Error {
      using ScaleT = std::remove_pointer_t<decltype(scales_ptr)>;
      using BiasT = std::remove_pointer_t<decltype(biases_ptr)>;
      if (packed_k_major) {
        DequantKMajorAffineKernel<ScaleT, BiasT, __nv_bfloat16>
            <<<grid, block, 0, stream>>>(wq_ptr, scales_ptr, biases_ptr,
                                         w_deq_bf16, K, N, n_groups,
                                         group_size, bits);
      } else {
        DequantToBf16Kernel<ScaleT, BiasT><<<grid, block, 0, stream>>>(
            wq_ptr, scales_ptr, biases_ptr, w_deq_bf16, K, N, N_words,
            n_groups, group_size, bits, mode, transpose);
      }
      return CheckCuda(cudaPeekAtLastError(), "bf16 dequant kernel launch");
    };

    if (need_bf16_dequantize) {
      if (mode == 0) {
        auto biases_buf = *biases;
        auto bias_dtype = biases_buf.element_type();
        if (bias_dtype != scales_dtype) {
          return MakeInvalid("biases dtype must match scales dtype.");
        }
        if (scales_dtype == xla::ffi::DataType::F32) {
          if (Error err = launch_dequant_bf16(
                  static_cast<const float *>(scales.untyped_data()),
                  static_cast<const float *>(biases_buf.untyped_data()));
              err.failure()) {
            return err;
          }
        } else if (scales_dtype == xla::ffi::DataType::F16) {
          if (Error err = launch_dequant_bf16(
                  static_cast<const half *>(scales.untyped_data()),
                  static_cast<const half *>(biases_buf.untyped_data()));
              err.failure()) {
            return err;
          }
        } else if (scales_dtype == xla::ffi::DataType::BF16) {
          if (Error err = launch_dequant_bf16(
                  static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
                  static_cast<const __nv_bfloat16 *>(biases_buf.untyped_data()));
              err.failure()) {
            return err;
          }
        } else {
          return MakeInvalid(
              "scales/biases dtype must be float32/float16/bfloat16.");
        }
      } else if (mode == 1) {
        if (scales_dtype == xla::ffi::DataType::F32) {
          if (Error err = launch_dequant_bf16(
                  static_cast<const float *>(scales.untyped_data()),
                  static_cast<const float *>(nullptr));
              err.failure()) {
            return err;
          }
        } else if (scales_dtype == xla::ffi::DataType::F16) {
          if (Error err = launch_dequant_bf16(
                  static_cast<const half *>(scales.untyped_data()),
                  static_cast<const half *>(nullptr));
              err.failure()) {
            return err;
          }
        } else if (scales_dtype == xla::ffi::DataType::BF16) {
          if (Error err = launch_dequant_bf16(
                  static_cast<const __nv_bfloat16 *>(scales.untyped_data()),
                  static_cast<const __nv_bfloat16 *>(nullptr));
              err.failure()) {
            return err;
          }
        } else {
          return MakeInvalid(
              "scales dtype must be float32/float16/bfloat16 for nf4.");
        }
      } else if (mode == 2) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
          return MakeInvalid("mxfp4 requires scales dtype uint8 and bits=4.");
        }
        if (Error err = launch_dequant_bf16(
                static_cast<const uint8_t *>(scales.untyped_data()),
                static_cast<const uint8_t *>(nullptr));
            err.failure()) {
          return err;
        }
      } else if (mode == 3) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
          return MakeInvalid("mxfp8 requires scales dtype uint8 and bits=8.");
        }
        if (Error err = launch_dequant_bf16(
                static_cast<const uint8_t *>(scales.untyped_data()),
                static_cast<const uint8_t *>(nullptr));
            err.failure()) {
          return err;
        }
      } else if (mode == 4) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits4) {
          return MakeInvalid("nvfp4 requires scales dtype uint8 and bits=4.");
        }
        if (Error err = launch_dequant_bf16(
                static_cast<const uint8_t *>(scales.untyped_data()),
                static_cast<const uint8_t *>(nullptr));
            err.failure()) {
          return err;
        }
      } else if (mode == 5) {
        if (scales_dtype != xla::ffi::DataType::U8 || bits != kBits8) {
          return MakeInvalid("nvfp8 requires scales dtype uint8 and bits=8.");
        }
        if (Error err = launch_dequant_bf16(
                static_cast<const uint8_t *>(scales.untyped_data()),
                static_cast<const uint8_t *>(nullptr));
            err.failure()) {
          return err;
        }
      } else {
        return MakeInvalid("Unsupported mode for CUDA quantized_matmul.");
      }
      if (use_cache) {
        g_dequant_cache.bf16_valid = true;
      }
    }
  }

  // For fp32 x: upcast dequanted fp16 w to fp32 so both operands are fp32.
  // This matches XLA's dot_general(fp32, fp16, pref=fp32) which does not
  // truncate the fp32 activations, keeping full precision.
  float *w_deq_f32 = nullptr;
  if (x_is_f32 && w_deq != nullptr) {
    int64_t w_elems = static_cast<int64_t>(K) * static_cast<int64_t>(N);
    size_t w_f32_bytes = static_cast<size_t>(w_elems) * sizeof(float);
    auto w_f32_opt = scratch.Allocate(w_f32_bytes, alignof(float));
    if (!w_f32_opt.has_value()) {
      return MakeInternal("Failed to allocate scratch buffer for fp32 w upcast.");
    }
    w_deq_f32 = reinterpret_cast<float *>(*w_f32_opt);
    dim3 cblock(256);
    dim3 cgrid(static_cast<uint32_t>((w_elems + cblock.x - 1) / cblock.x));
    convert_f16_to_f32<<<cgrid, cblock, 0, stream>>>(w_deq, w_deq_f32, w_elems);
    if (Error err = CheckCuda(cudaPeekAtLastError(), "f16_to_f32 w upcast launch");
        err.failure()) {
      return err;
    }
  }

  GemmBackend backend = GetGemmBackend();
  cublasComputeType_t compute_type = ResolveCublasComputeType(x_dtype, out_dtype);
  if (use_bf16_gemm) {
    compute_type = CUBLAS_COMPUTE_32F_FAST_16BF;
  } else if (x_is_f32) {
    // Pedantic fp32 GEMM: no TF32/fp16 shortcuts, matches XLA's precision for
    // dot_general(fp32, fp16, preferred_element_type=fp32). On Ampere+, standard
    // COMPUTE_32F uses TF32 (10-bit mantissa), same as fp16. PEDANTIC forces true
    // 23-bit fp32, matching how XLA upcast fp16 w to fp32 for the GEMM.
    compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
  } else if (compute_type == CUBLAS_COMPUTE_32F_FAST_16BF) {
    // Inputs are dequantized/cast to FP16 for GEMM, so use a FP16 compute path.
    compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  }
  const void *gemm_a_ptr =
      use_bf16_gemm ? static_cast<const void *>(w_deq_bf16)
      : x_is_f32    ? static_cast<const void *>(w_deq_f32)
                    : static_cast<const void *>(w_deq);
  const void *gemm_b_ptr =
      use_bf16_gemm ? static_cast<const void *>(x_bf16)
      : x_is_f32    ? static_cast<const void *>(x_f32)
                    : static_cast<const void *>(x_half);
  cudaDataType_t gemm_a_type =
      use_bf16_gemm ? CUDA_R_16BF : (x_is_f32 ? CUDA_R_32F : CUDA_R_16F);
  cudaDataType_t gemm_b_type =
      use_bf16_gemm ? CUDA_R_16BF : (x_is_f32 ? CUDA_R_32F : CUDA_R_16F);

  if (use_bf16_gemm &&
      (backend == GemmBackend::kCutlass ||
       backend == GemmBackend::kCutlassTuned)) {
    if (StrictGemmBackend()) {
      return MakeInvalid("CUTLASS BF16 GEMM is not supported.");
    }
    backend = GemmBackend::kCublasLt;
  }

  auto dispatch_cutlass = [&](auto func) -> Error {
    if (use_bf16_gemm) {
      return MakeInvalid("CUTLASS BF16 GEMM is not supported in this path.");
    }
    if (gemm_out_dtype == xla::ffi::DataType::F32) {
      return func(reinterpret_cast<float *>(gemm_out_ptr));
    }
    if (gemm_out_dtype == xla::ffi::DataType::F16) {
      return func(reinterpret_cast<cutlass::half_t *>(gemm_out_ptr));
    }
    if (gemm_out_dtype == xla::ffi::DataType::BF16) {
      return func(reinterpret_cast<cutlass::bfloat16_t *>(gemm_out_ptr));
    }
    return MakeInvalid("output dtype must be float16/float32/bfloat16.");
  };

  auto finalize_bf16 = [&]() -> Error {
    if (!need_bf16_out_cast) {
      return Error::Success();
    }
    dim3 cblock(256);
    int64_t out_elems = static_cast<int64_t>(M) * static_cast<int64_t>(N);
    dim3 cgrid(static_cast<uint32_t>((out_elems + cblock.x - 1) / cblock.x));
    if (bf16_cast_dtype == xla::ffi::DataType::F32) {
      convert_f32_to_bf16<<<cgrid, cblock, 0, stream>>>(
          static_cast<const float *>(bf16_cast_src),
          static_cast<__nv_bfloat16 *>(out->untyped_data()), out_elems);
    } else {
      convert_f16_to_bf16<<<cgrid, cblock, 0, stream>>>(
          static_cast<const half *>(bf16_cast_src),
          static_cast<__nv_bfloat16 *>(out->untyped_data()), out_elems);
    }
    return CheckCuda(cudaPeekAtLastError(), "bf16 output cast kernel launch");
  };

  auto try_fast_backends = [&]() -> Error {
    // Try cublasLt then CUTLASS for non-cublas fallback paths.
    Error err = RunCublasLtGemm(gemm_a_ptr, gemm_a_type, gemm_b_ptr, gemm_b_type,
                                gemm_out_ptr, gemm_out_dtype, M, N, K, stream,
                                scratch, compute_type, transposed);
    if (!err.failure()) {
      return finalize_bf16();
    }
    if (!use_bf16_gemm && !x_is_f32 && !transposed) {
      Error cutlass_err = dispatch_cutlass([&](auto *out_ptr) {
        return RunCutlassGemm(w_deq, x_half, out_ptr, M, N, K, stream, scratch);
      });
      if (!cutlass_err.failure()) {
        return finalize_bf16();
      }
    }
    return err;
  };

  if (backend == GemmBackend::kCublasLt) {
    Error err = RunCublasLtGemm(gemm_a_ptr, gemm_a_type, gemm_b_ptr, gemm_b_type,
                                gemm_out_ptr, gemm_out_dtype, M, N, K, stream,
                                scratch, compute_type, transposed);
    if (!err.failure()) {
      return finalize_bf16();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  } else if (backend == GemmBackend::kCutlass && !x_is_f32 && !transposed) {
    Error err = dispatch_cutlass([&](auto *out_ptr) {
      return RunCutlassGemm(w_deq, x_half, out_ptr, M, N, K, stream, scratch);
    });
    if (!err.failure()) {
      return finalize_bf16();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  } else if (backend == GemmBackend::kCutlassTuned && !x_is_f32 && !transposed) {
    Error err = dispatch_cutlass([&](auto *out_ptr) {
      return RunCutlassGemmTuned(w_deq, x_half, out_ptr, M, N, K, stream,
                                 scratch);
    });
    if (!err.failure()) {
      return finalize_bf16();
    }
    if (StrictGemmBackend()) {
      return err;
    }
  }

  cublasHandle_t handle;
  if (Error err = g_cublas_cache.Get(&handle, stream); err.failure()) {
    return err;
  }
  if (Error err =
          CheckCublas(cublasSetMathMode(handle, GetCublasMathMode(compute_type)),
                      "cublasSetMathMode");
      err.failure()) {
    return err;
  }

  if (backend == GemmBackend::kCublas &&
      gemm_out_dtype != xla::ffi::DataType::F32 && !StrictGemmBackend()) {
    Error fast_err = try_fast_backends();
    if (!fast_err.failure()) {
      return finalize_bf16();
    }
  }

  float alpha_f = 1.0f;
  float beta_f = 0.0f;
  half alpha_h = __float2half_rn(1.0f);
  half beta_h = __float2half_rn(0.0f);
  const void *alpha_ptr =
      ComputeTypeUsesHalfScalars(compute_type)
          ? static_cast<const void *>(&alpha_h)
          : static_cast<const void *>(&alpha_f);
  const void *beta_ptr =
      ComputeTypeUsesHalfScalars(compute_type)
          ? static_cast<const void *>(&beta_h)
          : static_cast<const void *>(&beta_f);

  cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

  void *out_ptr = gemm_out_ptr;
  cudaDataType_t out_type = ToCudaDataType(gemm_out_dtype);
  cublasOperation_t gemm_transa =
      transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  int gemm_lda = static_cast<int>(transposed ? K : N);

  cublasStatus_t gemm_status =
      cublasGemmEx(handle, gemm_transa, CUBLAS_OP_N, static_cast<int>(N),
                   static_cast<int>(M), static_cast<int>(K), alpha_ptr,
                   gemm_a_ptr, gemm_a_type, gemm_lda, gemm_b_ptr,
                   gemm_b_type, static_cast<int>(K), beta_ptr, out_ptr,
                   out_type,
                   static_cast<int>(N), compute_type, algo);

  if (gemm_status == CUBLAS_STATUS_NOT_SUPPORTED && !StrictGemmBackend()) {
    Error fast_err = try_fast_backends();
    if (!fast_err.failure()) {
      return finalize_bf16();
    }
  }

  if (Error err = CheckCublas(gemm_status, "cublasGemmEx"); err.failure()) {
    return err;
  }

  if (Error err = finalize_bf16(); err.failure()) {
    return err;
  }

  return Error::Success();
}

} // namespace

extern "C" XLA_FFI_Error *ejk_qmm_cuda(XLA_FFI_CallFrame *call_frame) {
  static auto handler = xla::ffi::Ffi::Bind()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .OptionalArg<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Attr<int64_t>("group_size")
                            .Attr<int64_t>("bits")
                            .Attr<int64_t>("mode")
                            .Attr<int64_t>("transpose")
                            .Attr<int64_t>("gemv_mode")
                            .Attr<int64_t>("revsplit_k")
                            .Attr<int64_t>("revsplit_k_parts")
                            .Attr<int64_t>("block_n")
                            .Attr<int64_t>("block_k")
                            .Ctx<PlatformStream<cudaStream_t>>()
                            .Ctx<ScratchAllocator>()
                            .To(QuantizedMatmulCuda);
  return handler->Call(call_frame);
}
