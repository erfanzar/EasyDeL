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

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>

#include "xla/ffi/api/ffi.h"

#include "ua.h"
#include "ua_kernel.h"

namespace {

using xla::ffi::AnyBuffer;
using xla::ffi::Error;
using xla::ffi::PlatformStream;
using xla::ffi::Result;
using xla::ffi::Span;

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

inline bool IsDevicePointer(const void *ptr) {
  cudaPointerAttributes attr;
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
#if CUDART_VERSION >= 10000
  if (err == cudaSuccess) {
    return attr.type == cudaMemoryTypeDevice ||
           attr.type == cudaMemoryTypeManaged;
  }
#else
  if (err == cudaSuccess) {
    return attr.memoryType == cudaMemoryTypeDevice;
  }
#endif
  cudaGetLastError();
  return false;
}

template <typename T>
void LaunchUnifiedAttentionRuntime(ua::UaParams &params, cudaStream_t stream) {
  if (params.total_tokens <= 0) {
    return;
  }
  dim3 grid(static_cast<uint32_t>(params.total_tokens),
            static_cast<uint32_t>(params.num_q_heads), 1);
  dim3 block(static_cast<uint32_t>(params.block_dim), 1, 1);

  ua::unified_attention_kernel_rt<T, T, T><<<grid, block, 0, stream>>>(
      static_cast<const T *>(params.queries),
      static_cast<const T *>(params.key_cache),
      static_cast<const T *>(params.value_cache), params.kv_lens,
      params.block_tables, params.query_start_loc, params.softmax_aux,
      static_cast<T *>(params.out), params.total_tokens, params.num_q_heads,
      params.num_kv_heads, params.head_dim, params.block_size,
      params.max_blocks_per_seq, params.num_seqs, params.softmax_scale,
      params.softcap, params.use_sinks, params.sliding_window);
}

template <typename T>
bool DispatchUnifiedAttentionHeadDim(ua::UaParams &params,
                                     cudaStream_t stream) {
#define UA_DISPATCH(HDIM)                                                      \
  if (params.head_dim == HDIM) {                                               \
    ua::run_unified_attention_<T, HDIM>(params, stream);                       \
    return true;                                                               \
  }
  UA_DISPATCH(32);
  UA_DISPATCH(64);
  UA_DISPATCH(96);
  UA_DISPATCH(128);
  UA_DISPATCH(192);
  UA_DISPATCH(256);
#undef UA_DISPATCH
  return false;
}

Error UnifiedAttentionCuda(AnyBuffer queries, AnyBuffer key_cache,
                           AnyBuffer value_cache, AnyBuffer kv_lens,
                           AnyBuffer block_tables, AnyBuffer query_start_loc,
                           AnyBuffer softmax_aux_buf, Result<AnyBuffer> out,
                           double softmax_scale_attr,
                           double logits_soft_cap_attr, int64_t causal,
                           int64_t sliding_window, int64_t use_sinks,
                           int64_t block_dim_attr, cudaStream_t stream) {
  if (causal == 0) {
    return MakeInvalid("CUDA unified_attention requires causal=True.");
  }
  if (sliding_window < -1) {
    return MakeInvalid("sliding_window must be -1 or positive.");
  }

  if (queries.dimensions().size() != 3 || key_cache.dimensions().size() != 4 ||
      value_cache.dimensions().size() != 4) {
    return MakeInvalid("queries must be rank-3, key_cache/value_cache rank-4.");
  }
  if (key_cache.dimensions().size() != value_cache.dimensions().size()) {
    return MakeInvalid("key_cache/value_cache shape mismatch.");
  }
  for (size_t i = 0; i < key_cache.dimensions().size(); ++i) {
    if (key_cache.dimensions()[i] != value_cache.dimensions()[i]) {
      return MakeInvalid("key_cache/value_cache shape mismatch.");
    }
  }
  if (kv_lens.element_type() != xla::ffi::DataType::S32 ||
      block_tables.element_type() != xla::ffi::DataType::S32 ||
      query_start_loc.element_type() != xla::ffi::DataType::S32) {
    return MakeInvalid("kv_lens/block_tables/query_start_loc must be int32.");
  }

  Span<const int64_t> q_dims = queries.dimensions();
  Span<const int64_t> k_dims = key_cache.dimensions();
  Span<const int64_t> b_dims = block_tables.dimensions();
  Span<const int64_t> qstart_dims = query_start_loc.dimensions();

  int32_t total_tokens = static_cast<int32_t>(q_dims[0]);
  int32_t num_q_heads = static_cast<int32_t>(q_dims[1]);
  int32_t head_dim = static_cast<int32_t>(q_dims[2]);

  int32_t block_size = static_cast<int32_t>(k_dims[1]);
  int32_t num_kv_heads = static_cast<int32_t>(k_dims[2]);
  int32_t head_dim_kv = static_cast<int32_t>(k_dims[3]);

  int32_t num_seqs = static_cast<int32_t>(b_dims[0]);
  int32_t max_blocks_per_seq = static_cast<int32_t>(b_dims[1]);

  if (head_dim_kv != head_dim) {
    return MakeInvalid("head_dim mismatch between queries and KV cache.");
  }
  if (num_q_heads % num_kv_heads != 0) {
    return MakeInvalid("num_q_heads must be divisible by num_kv_heads.");
  }
  if (qstart_dims.size() != 1 || qstart_dims[0] != num_seqs + 1) {
    return MakeInvalid("query_start_loc must be shape [num_seqs + 1].");
  }
  if (head_dim > ua::kMaxHeadDim) {
    return MakeInvalid("CUDA unified_attention supports head_dim <= 256.");
  }

  if (softmax_aux_buf.element_type() != xla::ffi::DataType::F32) {
    return MakeInvalid("softmax_aux must be float32 buffer.");
  }
  float softmax_scale = static_cast<float>(softmax_scale_attr);
  float softcap = logits_soft_cap_attr > 0.0
                      ? static_cast<float>(logits_soft_cap_attr)
                      : 0.0f;

  int32_t block_dim = static_cast<int32_t>(block_dim_attr);
  if (block_dim < 32 || block_dim > ua::kMaxBlockDim ||
      (block_dim % ua::kWarpSize) != 0) {
    return MakeInvalid("block_dim must be a multiple of 32 within [32, 256].");
  }

  if (!IsDevicePointer(kv_lens.untyped_data()) ||
      !IsDevicePointer(block_tables.untyped_data()) ||
      !IsDevicePointer(query_start_loc.untyped_data())) {
    return MakeInvalid(
        "kv_lens/block_tables/query_start_loc must be device buffers.");
  }

  ua::UaParams params{};
  params.queries = queries.untyped_data();
  params.key_cache = key_cache.untyped_data();
  params.value_cache = value_cache.untyped_data();
  params.kv_lens = static_cast<const int32_t *>(kv_lens.untyped_data());
  params.block_tables =
      static_cast<const int32_t *>(block_tables.untyped_data());
  params.query_start_loc =
      static_cast<const int32_t *>(query_start_loc.untyped_data());
  params.softmax_aux =
      static_cast<const float *>(softmax_aux_buf.untyped_data());
  params.out = out->untyped_data();
  params.total_tokens = total_tokens;
  params.num_q_heads = num_q_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim = head_dim;
  params.block_size = block_size;
  params.max_blocks_per_seq = max_blocks_per_seq;
  params.num_seqs = num_seqs;
  params.softmax_scale = softmax_scale;
  params.softcap = softcap;
  params.use_sinks = static_cast<int32_t>(use_sinks);
  params.sliding_window = static_cast<int32_t>(sliding_window);
  params.block_dim = block_dim;

  auto dtype = queries.element_type();
  auto out_dtype = out->element_type();

  bool launched = false;
  if (dtype == xla::ffi::DataType::F16 &&
      out_dtype == xla::ffi::DataType::F16) {
    launched = DispatchUnifiedAttentionHeadDim<half>(params, stream);
    if (!launched) {
      LaunchUnifiedAttentionRuntime<half>(params, stream);
    }
  } else if (dtype == xla::ffi::DataType::BF16 &&
             out_dtype == xla::ffi::DataType::BF16) {
    launched = DispatchUnifiedAttentionHeadDim<__nv_bfloat16>(params, stream);
    if (!launched) {
      LaunchUnifiedAttentionRuntime<__nv_bfloat16>(params, stream);
    }
  } else {
    return MakeInvalid("queries/output must be float16 or bfloat16.");
  }

  if (Error err =
          CheckCuda(cudaPeekAtLastError(), "unified_attention kernel launch");
      err.failure()) {
    return err;
  }
  if (Error err =
          CheckCuda(cudaStreamSynchronize(stream), "unified_attention sync");
      err.failure()) {
    return err;
  }

  return Error::Success();
}

} // namespace

extern "C" XLA_FFI_Error *
ejk_unified_attention_cuda(XLA_FFI_CallFrame *call_frame) {
  static auto handler = xla::ffi::Ffi::Bind()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Attr<double>("softmax_scale")
                            .Attr<double>("logits_soft_cap")
                            .Attr<int64_t>("causal")
                            .Attr<int64_t>("sliding_window")
                            .Attr<int64_t>("use_sinks")
                            .Attr<int64_t>("block_dim")
                            .Ctx<PlatformStream<cudaStream_t>>()
                            .To(UnifiedAttentionCuda);
  return handler->Call(call_frame);
}
