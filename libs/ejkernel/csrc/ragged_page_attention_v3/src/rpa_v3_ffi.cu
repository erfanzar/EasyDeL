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

#include "xla/ffi/api/ffi.h"

#include "rpa_v3.h"

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

inline size_t DataTypeBytes(xla::ffi::DataType dtype) {
  switch (dtype) {
    case xla::ffi::DataType::F16:
      return 2;
    case xla::ffi::DataType::BF16:
      return 2;
    case xla::ffi::DataType::F32:
      return 4;
    default:
      return 0;
  }
}

inline size_t NumElements(Span<const int64_t> dims) {
  size_t n = 1;
  for (int64_t d : dims) {
    n *= static_cast<size_t>(d);
  }
  return n;
}

template <typename T>
Error DispatchRpaV3HeadDim(rpa_v3::RpaV3Params &params, int head_dim,
                           cudaStream_t stream) {
#define RPA_DISPATCH(HDIM)                                                     \
  if (head_dim == HDIM) {                                                      \
    rpa_v3::run_rpa_v3_update_kv_<T, HDIM>(params, stream);                     \
    rpa_v3::run_rpa_v3_attention_<T, HDIM>(params, stream);                     \
    return Error::Success();                                                   \
  }
  RPA_DISPATCH(32);
  RPA_DISPATCH(64);
  RPA_DISPATCH(96);
  RPA_DISPATCH(128);
  RPA_DISPATCH(192);
  RPA_DISPATCH(256);
#undef RPA_DISPATCH
  return MakeInvalid("Unsupported head_dim for ragged_page_attention_v3.");
}

Error RaggedPageAttentionV3Cuda(
    AnyBuffer queries, AnyBuffer keys, AnyBuffer values, AnyBuffer kv_cache,
    AnyBuffer kv_lens, AnyBuffer block_tables, AnyBuffer query_start_loc,
    AnyBuffer distribution, AnyBuffer softmax_aux_buf, Result<AnyBuffer> out,
    Result<AnyBuffer> kv_cache_out, double softmax_scale_attr,
    double logits_soft_cap_attr, int64_t sliding_window, int64_t use_sinks,
    int64_t use_q_scale, int64_t use_k_scale, int64_t use_v_scale,
    double q_scale_attr, double k_scale_attr, double v_scale_attr,
    cudaStream_t stream) {
  if (queries.dimensions().size() != 3 || keys.dimensions().size() != 3 ||
      values.dimensions().size() != 3) {
    return MakeInvalid("queries/keys/values must be rank-3.");
  }
  if (kv_cache.dimensions().size() != 5) {
    return MakeInvalid("kv_cache must be rank-5.");
  }
  if (kv_lens.element_type() != xla::ffi::DataType::S32 ||
      block_tables.element_type() != xla::ffi::DataType::S32 ||
      query_start_loc.element_type() != xla::ffi::DataType::S32 ||
      distribution.element_type() != xla::ffi::DataType::S32) {
    return MakeInvalid(
        "kv_lens/block_tables/query_start_loc/distribution must be int32.");
  }

  Span<const int64_t> q_dims = queries.dimensions();
  Span<const int64_t> k_dims = keys.dimensions();
  Span<const int64_t> v_dims = values.dimensions();
  Span<const int64_t> cache_dims = kv_cache.dimensions();
  Span<const int64_t> cache_out_dims = kv_cache_out->dimensions();
  Span<const int64_t> kv_lens_dims = kv_lens.dimensions();
  Span<const int64_t> qstart_dims = query_start_loc.dimensions();
  Span<const int64_t> bt_dims = block_tables.dimensions();
  Span<const int64_t> dist_dims = distribution.dimensions();
  Span<const int64_t> sink_dims = softmax_aux_buf.dimensions();

  int32_t total_tokens = static_cast<int32_t>(q_dims[0]);
  int32_t num_q_heads = static_cast<int32_t>(q_dims[1]);
  int32_t head_dim = static_cast<int32_t>(q_dims[2]);

  int32_t num_kv_heads = static_cast<int32_t>(k_dims[1]);
  int32_t head_dim_kv = static_cast<int32_t>(k_dims[2]);

  if (k_dims[0] != v_dims[0] || k_dims[1] != v_dims[1] ||
      k_dims[2] != v_dims[2]) {
    return MakeInvalid("keys/values shape mismatch.");
  }
  if (k_dims[0] != q_dims[0] || head_dim_kv != head_dim) {
    return MakeInvalid("queries/keys/values shape mismatch.");
  }
  if (num_q_heads % num_kv_heads != 0) {
    return MakeInvalid("num_q_heads must be divisible by num_kv_heads.");
  }

  int32_t num_pages = static_cast<int32_t>(cache_dims[0]);
  int32_t page_size = static_cast<int32_t>(cache_dims[1]);
  int32_t num_kv_heads_x2_per_pack = static_cast<int32_t>(cache_dims[2]);
  int32_t kv_packing = static_cast<int32_t>(cache_dims[3]);
  int32_t head_dim_padded = static_cast<int32_t>(cache_dims[4]);

  if (head_dim_padded < head_dim) {
    return MakeInvalid("kv_cache head_dim_padded must be >= head_dim.");
  }
  if (head_dim > rpa_v3::kMaxHeadDim ||
      head_dim_padded > rpa_v3::kMaxHeadDim) {
    return MakeInvalid("head_dim/head_dim_padded must be <= 256 for CUDA.");
  }
  if (kv_packing != 1 && kv_packing != 2) {
    return MakeInvalid("kv_packing must be 1 or 2.");
  }
  if (page_size <= 0) {
    return MakeInvalid("page_size must be > 0.");
  }
  if (page_size % kv_packing != 0) {
    return MakeInvalid("page_size must be divisible by kv_packing.");
  }
  if (num_kv_heads_x2_per_pack * kv_packing < num_kv_heads * 2) {
    return MakeInvalid("kv_cache head packing incompatible with num_kv_heads.");
  }
  (void)num_pages;

  if (cache_out_dims.size() != cache_dims.size()) {
    return MakeInvalid("kv_cache_out shape mismatch.");
  }
  for (size_t i = 0; i < cache_dims.size(); ++i) {
    if (cache_out_dims[i] != cache_dims[i]) {
      return MakeInvalid("kv_cache_out shape mismatch.");
    }
  }

  if (qstart_dims.size() != 1 || kv_lens_dims.size() != 1) {
    return MakeInvalid("kv_lens/query_start_loc must be rank-1.");
  }
  if (qstart_dims[0] != kv_lens_dims[0] + 1) {
    return MakeInvalid("query_start_loc must be length kv_lens + 1.");
  }
  int32_t max_num_seqs = static_cast<int32_t>(kv_lens_dims[0]);

  int32_t pages_per_seq = 0;
  if (bt_dims.size() == 1) {
    if (bt_dims[0] % max_num_seqs != 0) {
      return MakeInvalid(
          "block_tables length must be divisible by max_num_seqs.");
    }
    pages_per_seq = static_cast<int32_t>(bt_dims[0] / max_num_seqs);
  } else if (bt_dims.size() == 2) {
    if (bt_dims[0] != max_num_seqs) {
      return MakeInvalid("block_tables first dim must match max_num_seqs.");
    }
    pages_per_seq = static_cast<int32_t>(bt_dims[1]);
  } else {
    return MakeInvalid("block_tables must be rank-1 or rank-2.");
  }

  if (dist_dims.size() != 1 || dist_dims[0] != 3) {
    return MakeInvalid("distribution must be shape [3].");
  }
  if (sink_dims.size() != 1 || sink_dims[0] != num_q_heads) {
    return MakeInvalid("softmax_aux must be shape [num_q_heads].");
  }

  auto dtype = queries.element_type();
  auto kv_dtype = kv_cache.element_type();
  auto out_dtype = out->element_type();
  if (dtype != kv_dtype || kv_dtype != keys.element_type() ||
      kv_dtype != values.element_type()) {
    return MakeInvalid("queries/keys/values/kv_cache must share dtype.");
  }
  if (out_dtype != dtype) {
    return MakeInvalid("output dtype must match queries dtype.");
  }
  if (kv_cache_out->element_type() != kv_dtype) {
    return MakeInvalid("kv_cache_out dtype must match kv_cache dtype.");
  }
  if (dtype != xla::ffi::DataType::F16 && dtype != xla::ffi::DataType::BF16 &&
      dtype != xla::ffi::DataType::F32) {
    return MakeInvalid(
        "CUDA ragged_page_attention_v3 supports f16/bf16/f32.");
  }

  if (sliding_window == 0) {
    return MakeInvalid("sliding_window must be -1 or positive.");
  }
  if (use_q_scale && q_scale_attr == 0.0) {
    return MakeInvalid("q_scale must be non-zero.");
  }

  if (!IsDevicePointer(queries.untyped_data()) ||
      !IsDevicePointer(keys.untyped_data()) ||
      !IsDevicePointer(values.untyped_data()) ||
      !IsDevicePointer(kv_cache.untyped_data()) ||
      !IsDevicePointer(kv_lens.untyped_data()) ||
      !IsDevicePointer(block_tables.untyped_data()) ||
      !IsDevicePointer(query_start_loc.untyped_data()) ||
      !IsDevicePointer(distribution.untyped_data()) ||
      !IsDevicePointer(softmax_aux_buf.untyped_data())) {
    return MakeInvalid("all inputs must be device buffers.");
  }

  size_t elem_bytes = DataTypeBytes(kv_dtype);
  if (elem_bytes == 0) {
    return MakeInvalid("unsupported kv_cache dtype.");
  }
  size_t kv_elems = NumElements(cache_dims);
  size_t kv_bytes = kv_elems * elem_bytes;

  if (kv_cache.untyped_data() != kv_cache_out->untyped_data()) {
    if (Error err = CheckCuda(
            cudaMemcpyAsync(kv_cache_out->untyped_data(),
                            kv_cache.untyped_data(), kv_bytes,
                            cudaMemcpyDeviceToDevice, stream),
            "kv_cache memcpy");
        err.failure()) {
      return err;
    }
  }

  float softmax_scale = static_cast<float>(softmax_scale_attr);
  float softcap =
      logits_soft_cap_attr > 0.0 ? static_cast<float>(logits_soft_cap_attr)
                                 : 0.0f;
  float q_scale = static_cast<float>(q_scale_attr);
  float k_scale = static_cast<float>(k_scale_attr);
  float v_scale = static_cast<float>(v_scale_attr);

  rpa_v3::RpaV3Params params{};
  params.queries = queries.untyped_data();
  params.keys = keys.untyped_data();
  params.values = values.untyped_data();
  params.kv_cache = kv_cache.untyped_data();
  params.kv_lens = static_cast<const int32_t *>(kv_lens.untyped_data());
  params.block_tables = static_cast<const int32_t *>(block_tables.untyped_data());
  params.query_start_loc =
      static_cast<const int32_t *>(query_start_loc.untyped_data());
  params.distribution =
      static_cast<const int32_t *>(distribution.untyped_data());
  params.softmax_aux =
      static_cast<const float *>(softmax_aux_buf.untyped_data());
  params.out = out->untyped_data();
  params.kv_cache_out = kv_cache_out->untyped_data();
  params.total_tokens = total_tokens;
  params.num_q_heads = num_q_heads;
  params.num_kv_heads = num_kv_heads;
  params.head_dim_padded = head_dim_padded;
  params.page_size = page_size;
  params.pages_per_seq = pages_per_seq;
  params.num_kv_heads_x2_per_pack = num_kv_heads_x2_per_pack;
  params.kv_packing = kv_packing;
  params.softmax_scale = softmax_scale;
  params.softcap = softcap;
  params.sliding_window = static_cast<int32_t>(sliding_window);
  params.use_sinks = static_cast<int32_t>(use_sinks);
  params.use_q_scale = static_cast<int32_t>(use_q_scale);
  params.use_k_scale = static_cast<int32_t>(use_k_scale);
  params.use_v_scale = static_cast<int32_t>(use_v_scale);
  params.q_scale = q_scale;
  params.k_scale = k_scale;
  params.v_scale = v_scale;

  Error dispatch_err = Error::Success();
  if (dtype == xla::ffi::DataType::F16) {
    dispatch_err = DispatchRpaV3HeadDim<half>(params, head_dim, stream);
  } else if (dtype == xla::ffi::DataType::BF16) {
    dispatch_err =
        DispatchRpaV3HeadDim<__nv_bfloat16>(params, head_dim, stream);
  } else {
    dispatch_err = DispatchRpaV3HeadDim<float>(params, head_dim, stream);
  }

  if (dispatch_err.failure()) {
    return dispatch_err;
  }

  if (Error err =
          CheckCuda(cudaPeekAtLastError(), "rpa_v3 kernel launch");
      err.failure()) {
    return err;
  }
  if (Error err = CheckCuda(cudaStreamSynchronize(stream), "rpa_v3 sync");
      err.failure()) {
    return err;
  }

  return Error::Success();
}

}  // namespace

extern "C" XLA_FFI_Error *ejk_ragged_page_attention_v3_cuda(
    XLA_FFI_CallFrame *call_frame) {
  static auto handler = xla::ffi::Ffi::Bind()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Arg<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Attr<double>("softmax_scale")
                            .Attr<double>("logits_soft_cap")
                            .Attr<int64_t>("sliding_window")
                            .Attr<int64_t>("use_sinks")
                            .Attr<int64_t>("use_q_scale")
                            .Attr<int64_t>("use_k_scale")
                            .Attr<int64_t>("use_v_scale")
                            .Attr<double>("q_scale")
                            .Attr<double>("k_scale")
                            .Attr<double>("v_scale")
                            .Ctx<PlatformStream<cudaStream_t>>()
                            .To(RaggedPageAttentionV3Cuda);
  return handler->Call(call_frame);
}
