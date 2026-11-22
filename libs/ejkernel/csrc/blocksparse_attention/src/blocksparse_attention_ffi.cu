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

#include "blocksparse_attention.h"

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

inline int32_t CeilDiv(int32_t a, int32_t b) { return (a + b - 1) / b; }

template <typename T>
Error DispatchBlocksparseHeadDim(blocksparse::Blocksparse_fwd_params &params,
                                 int qk_head_dim, int v_head_dim,
                                 cudaStream_t stream) {
#define BS_DISPATCH(HDIM)                                                     \
  if (qk_head_dim == HDIM && v_head_dim == HDIM) {                             \
    blocksparse::run_blocksparse_fwd_<T, HDIM, HDIM>(params, stream);          \
    return Error::Success();                                                  \
  }
  BS_DISPATCH(32);
  BS_DISPATCH(64);
  BS_DISPATCH(96);
  BS_DISPATCH(128);
  BS_DISPATCH(192);
  BS_DISPATCH(256);
#undef BS_DISPATCH
  return MakeInvalid("Unsupported head_dim for blocksparse_attention.");
}

Error BlocksparseAttentionCuda(
    AnyBuffer query, AnyBuffer key, AnyBuffer value, AnyBuffer q_positions,
    AnyBuffer q_segment_ids, AnyBuffer kv_positions, AnyBuffer kv_segment_ids,
    AnyBuffer lower_bounds, AnyBuffer upper_bounds, AnyBuffer softmax_aux,
    Result<AnyBuffer> out, double softmax_scale_attr,
    double logits_soft_cap_attr, int64_t causal_attr, int64_t window_left,
    int64_t window_right, int64_t use_sinks, int64_t num_sinks,
    int64_t q_blocksize, int64_t kv_blocksize, int64_t block_dim_attr,
    cudaStream_t stream) {
  if (query.dimensions().size() != 4 || key.dimensions().size() != 4 ||
      value.dimensions().size() != 4) {
    return MakeInvalid("query/key/value must be rank-4 tensors.");
  }

  if (q_positions.dimensions().size() != 2 ||
      q_segment_ids.dimensions().size() != 2 ||
      kv_positions.dimensions().size() != 2 ||
      kv_segment_ids.dimensions().size() != 2) {
    return MakeInvalid("q/kv positions and segment ids must be rank-2.");
  }

  if (lower_bounds.dimensions().size() != 3 ||
      upper_bounds.dimensions().size() != 3) {
    return MakeInvalid("lower/upper bounds must be rank-3.");
  }

  if (q_positions.element_type() != xla::ffi::DataType::S32 ||
      q_segment_ids.element_type() != xla::ffi::DataType::S32 ||
      kv_positions.element_type() != xla::ffi::DataType::S32 ||
      kv_segment_ids.element_type() != xla::ffi::DataType::S32 ||
      lower_bounds.element_type() != xla::ffi::DataType::S32 ||
      upper_bounds.element_type() != xla::ffi::DataType::S32) {
    return MakeInvalid("positions/segment_ids/bounds must be int32.");
  }

  if (softmax_aux.element_type() != xla::ffi::DataType::F32) {
    return MakeInvalid("softmax_aux must be float32.");
  }

  Span<const int64_t> q_dims = query.dimensions();
  Span<const int64_t> k_dims = key.dimensions();
  Span<const int64_t> v_dims = value.dimensions();
  Span<const int64_t> aux_dims = softmax_aux.dimensions();

  int32_t batch = static_cast<int32_t>(q_dims[0]);
  int32_t num_heads = static_cast<int32_t>(q_dims[1]);
  int32_t q_len = static_cast<int32_t>(q_dims[2]);
  int32_t qk_head_dim = static_cast<int32_t>(q_dims[3]);

  int32_t batch_k = static_cast<int32_t>(k_dims[0]);
  int32_t num_kv_heads = static_cast<int32_t>(k_dims[1]);
  int32_t kv_len = static_cast<int32_t>(k_dims[2]);
  int32_t k_head_dim = static_cast<int32_t>(k_dims[3]);

  int32_t batch_v = static_cast<int32_t>(v_dims[0]);
  int32_t num_kv_heads_v = static_cast<int32_t>(v_dims[1]);
  int32_t kv_len_v = static_cast<int32_t>(v_dims[2]);
  int32_t v_head_dim = static_cast<int32_t>(v_dims[3]);

  if (batch_k != batch || batch_v != batch) {
    return MakeInvalid("batch mismatch between query/key/value.");
  }
  if (num_kv_heads_v != num_kv_heads || kv_len_v != kv_len) {
    return MakeInvalid("key/value shapes must match.");
  }
  if (k_head_dim != qk_head_dim) {
    return MakeInvalid("key head_dim must match query head_dim.");
  }
  if (num_heads % num_kv_heads != 0) {
    return MakeInvalid("num_heads must be divisible by num_kv_heads.");
  }
  if (qk_head_dim != v_head_dim) {
    return MakeInvalid("q/k head_dim must match value head_dim.");
  }
  if (qk_head_dim > blocksparse::kMaxHeadDim ||
      v_head_dim > blocksparse::kMaxVHeadDim) {
    return MakeInvalid("head_dim exceeds CUDA kernel limits (<=256).");
  }
  if (aux_dims.size() != 2 || aux_dims[0] != num_heads) {
    return MakeInvalid("softmax_aux must have shape [num_heads, num_sinks].");
  }
  if (use_sinks && num_sinks > 0 && aux_dims[1] < num_sinks) {
    return MakeInvalid("softmax_aux second dim is smaller than num_sinks.");
  }

  if (q_positions.dimensions()[0] != batch ||
      q_positions.dimensions()[1] != q_len ||
      q_segment_ids.dimensions()[0] != batch ||
      q_segment_ids.dimensions()[1] != q_len) {
    return MakeInvalid("q_positions/q_segment_ids shape mismatch.");
  }
  if (kv_positions.dimensions()[0] != batch ||
      kv_positions.dimensions()[1] != kv_len ||
      kv_segment_ids.dimensions()[0] != batch ||
      kv_segment_ids.dimensions()[1] != kv_len) {
    return MakeInvalid("kv_positions/kv_segment_ids shape mismatch.");
  }

  if (q_blocksize <= 0 || kv_blocksize <= 0) {
    return MakeInvalid("q_blocksize/kv_blocksize must be positive.");
  }

  int32_t num_q_blocks = CeilDiv(q_len, static_cast<int32_t>(q_blocksize));
  if (lower_bounds.dimensions()[0] != batch ||
      lower_bounds.dimensions()[1] != 1 ||
      lower_bounds.dimensions()[2] != num_q_blocks ||
      upper_bounds.dimensions()[0] != batch ||
      upper_bounds.dimensions()[1] != 1 ||
      upper_bounds.dimensions()[2] != num_q_blocks) {
    return MakeInvalid("lower/upper bounds shape mismatch.");
  }

  int32_t block_dim = static_cast<int32_t>(block_dim_attr);
  if (block_dim < 32 || block_dim > blocksparse::kMaxBlockDim ||
      (block_dim % blocksparse::kWarpSize) != 0) {
    return MakeInvalid("block_dim must be a multiple of 32 within [32, 256].");
  }

  if (!IsDevicePointer(q_positions.untyped_data()) ||
      !IsDevicePointer(q_segment_ids.untyped_data()) ||
      !IsDevicePointer(kv_positions.untyped_data()) ||
      !IsDevicePointer(kv_segment_ids.untyped_data()) ||
      !IsDevicePointer(lower_bounds.untyped_data()) ||
      !IsDevicePointer(upper_bounds.untyped_data()) ||
      !IsDevicePointer(softmax_aux.untyped_data())) {
    return MakeInvalid("All auxiliary buffers must be device buffers.");
  }

  float softmax_scale = static_cast<float>(softmax_scale_attr);
  float softcap = logits_soft_cap_attr > 0.0
                      ? static_cast<float>(logits_soft_cap_attr)
                      : 0.0f;
  int32_t causal_flag = causal_attr ? 1 : 0;

  blocksparse::Blocksparse_fwd_params params{};
  params.query = query.untyped_data();
  params.key = key.untyped_data();
  params.value = value.untyped_data();
  params.q_positions = static_cast<const int32_t *>(q_positions.untyped_data());
  params.q_segment_ids =
      static_cast<const int32_t *>(q_segment_ids.untyped_data());
  params.kv_positions =
      static_cast<const int32_t *>(kv_positions.untyped_data());
  params.kv_segment_ids =
      static_cast<const int32_t *>(kv_segment_ids.untyped_data());
  params.lower_bounds =
      static_cast<const int32_t *>(lower_bounds.untyped_data());
  params.upper_bounds =
      static_cast<const int32_t *>(upper_bounds.untyped_data());
  params.softmax_aux = static_cast<const float *>(softmax_aux.untyped_data());
  params.out = out->untyped_data();
  params.batch = batch;
  params.num_heads = num_heads;
  params.num_kv_heads = num_kv_heads;
  params.q_len = q_len;
  params.kv_len = kv_len;
  params.q_blocksize = static_cast<int32_t>(q_blocksize);
  params.kv_blocksize = static_cast<int32_t>(kv_blocksize);
  params.num_q_blocks = num_q_blocks;
  params.softmax_scale = softmax_scale;
  params.softcap = softcap;
  params.causal = causal_flag;
  params.window_left = static_cast<int32_t>(window_left);
  params.window_right = static_cast<int32_t>(window_right);
  params.use_sinks = static_cast<int32_t>(use_sinks);
  params.num_sinks = static_cast<int32_t>(num_sinks);
  params.block_dim = block_dim;

  auto dtype = query.element_type();
  auto out_dtype = out->element_type();

  Error dispatch_err = Error::Success();
  if (dtype == xla::ffi::DataType::F16 &&
      out_dtype == xla::ffi::DataType::F16) {
    dispatch_err =
        DispatchBlocksparseHeadDim<half>(params, qk_head_dim, v_head_dim, stream);
  } else if (dtype == xla::ffi::DataType::BF16 &&
             out_dtype == xla::ffi::DataType::BF16) {
    dispatch_err = DispatchBlocksparseHeadDim<__nv_bfloat16>(
        params, qk_head_dim, v_head_dim, stream);
  } else {
    return MakeInvalid("query/output must be float16 or bfloat16.");
  }

  if (dispatch_err.failure()) {
    return dispatch_err;
  }

  if (Error err =
          CheckCuda(cudaPeekAtLastError(), "blocksparse_attention launch");
      err.failure()) {
    return err;
  }
  if (Error err = CheckCuda(cudaStreamSynchronize(stream),
                            "blocksparse_attention sync");
      err.failure()) {
    return err;
  }

  return Error::Success();
}

}  // namespace

extern "C" XLA_FFI_Error *ejk_blocksparse_attention_cuda(
    XLA_FFI_CallFrame *call_frame) {
  static auto handler =
      xla::ffi::Ffi::Bind()
          .Arg<AnyBuffer>()
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
          .Attr<double>("softmax_scale")
          .Attr<double>("logits_soft_cap")
          .Attr<int64_t>("causal")
          .Attr<int64_t>("window_left")
          .Attr<int64_t>("window_right")
          .Attr<int64_t>("use_sinks")
          .Attr<int64_t>("num_sinks")
          .Attr<int64_t>("q_blocksize")
          .Attr<int64_t>("kv_blocksize")
          .Attr<int64_t>("block_dim")
          .Ctx<PlatformStream<cudaStream_t>>()
          .To(BlocksparseAttentionCuda);
  return handler->Call(call_frame);
}
