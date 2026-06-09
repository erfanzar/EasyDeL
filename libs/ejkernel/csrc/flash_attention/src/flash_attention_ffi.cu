// Copyright 2025 The EasyDeL/ejKernel Author.
// CUTLASS-based FlashAttention FFI for JAX.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

#include <cutlass/numeric_types.h>

#include "flash.h"
#include "namespace_config.h"

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

constexpr float kLog2E = 1.4426950408889634f;

inline int RoundUp(int x, int m) { return (x + m - 1) / m * m; }

struct RngStateDevice {
  uint64_t *ptr = nullptr;
  bool uses_u32 = false;
};

inline Error PrepareRngState(const AnyBuffer &rng_state, cudaStream_t stream,
                             bool for_bwd, RngStateDevice &out) {
  auto dtype = rng_state.element_type();
  if (dtype == xla::ffi::DataType::U64) {
    out.ptr = reinterpret_cast<uint64_t *>(rng_state.untyped_data());
    out.uses_u32 = false;
    return Error::Success();
  }
  if (dtype != xla::ffi::DataType::U32) {
    return MakeInvalid("rng_state must be uint32 or uint64.");
  }
  out.uses_u32 = true;
  if (cudaError_t err = cudaMalloc(&out.ptr, sizeof(uint64_t) * 2);
      err != cudaSuccess) {
    return MakeInternal("Failed to allocate rng_state buffer.");
  }
  if (for_bwd) {
    std::array<uint32_t, 2> host_u32{};
    if (cudaError_t err = cudaMemcpyAsync(
            host_u32.data(), rng_state.untyped_data(), sizeof(uint32_t) * 2,
            cudaMemcpyDeviceToHost, stream);
        err != cudaSuccess) {
      return CheckCuda(err, "rng_state copy d2h");
    }
    if (cudaError_t err = cudaStreamSynchronize(stream); err != cudaSuccess) {
      return CheckCuda(err, "rng_state sync");
    }
    std::array<uint64_t, 2> host_u64{static_cast<uint64_t>(host_u32[0]),
                                     static_cast<uint64_t>(host_u32[1])};
    if (cudaError_t err = cudaMemcpyAsync(out.ptr, host_u64.data(),
                                          sizeof(uint64_t) * 2,
                                          cudaMemcpyHostToDevice, stream);
        err != cudaSuccess) {
      return CheckCuda(err, "rng_state copy h2d");
    }
  }
  return Error::Success();
}

inline void CleanupRngState(RngStateDevice &state) {
  if (state.uses_u32 && state.ptr) {
    cudaFree(state.ptr);
  }
  state.ptr = nullptr;
  state.uses_u32 = false;
}

inline Error SyncBackRngStateU32(const AnyBuffer &rng_state,
                                 const RngStateDevice &state,
                                 cudaStream_t stream) {
  if (!state.uses_u32 || state.ptr == nullptr) {
    return Error::Success();
  }
  std::array<uint64_t, 2> host_u64{};
  if (cudaError_t err = cudaMemcpyAsync(host_u64.data(), state.ptr,
                                        sizeof(uint64_t) * 2,
                                        cudaMemcpyDeviceToHost, stream);
      err != cudaSuccess) {
    return CheckCuda(err, "rng_state copy back d2h");
  }
  if (cudaError_t err = cudaStreamSynchronize(stream); err != cudaSuccess) {
    return CheckCuda(err, "rng_state sync back");
  }
  std::array<uint32_t, 2> host_u32{
      static_cast<uint32_t>(host_u64[0]),
      static_cast<uint32_t>(host_u64[1]),
  };
  if (cudaError_t err = cudaMemcpyAsync(
          rng_state.untyped_data(), host_u32.data(), sizeof(uint32_t) * 2,
          cudaMemcpyHostToDevice, stream);
      err != cudaSuccess) {
    return CheckCuda(err, "rng_state copy back h2d");
  }
  return Error::Success();
}

struct ShapeInfo {
  int64_t batch = 0;
  int64_t seqlen = 0;
  int64_t heads = 0;
  int64_t head_dim = 0;
  int64_t total = 0; // for packed varlen
};

inline Error ParseQkvShape(const AnyBuffer &buf, bool varlen, ShapeInfo &out) {
  Span<const int64_t> dims = buf.dimensions();
  if (!varlen) {
    if (dims.size() != 4) {
      return MakeInvalid("q/k/v must be rank-4 for non-varlen.");
    }
    out.batch = dims[0];
    out.seqlen = dims[1];
    out.heads = dims[2];
    out.head_dim = dims[3];
    out.total = out.batch * out.seqlen;
    return Error::Success();
  }
  if (dims.size() != 3) {
    return MakeInvalid("q/k/v must be rank-3 for varlen (packed).");
  }
  out.total = dims[0];
  out.heads = dims[1];
  out.head_dim = dims[2];
  return Error::Success();
}

inline Error
ValidateDeviceBuffers(std::initializer_list<const AnyBuffer *> bufs) {
  for (const AnyBuffer *buf : bufs) {
    if (buf == nullptr) {
      continue;
    }
    if (!IsDevicePointer(buf->untyped_data())) {
      return MakeInvalid("All inputs must be device buffers.");
    }
  }
  return Error::Success();
}

inline Error DispatchFwdSm80(FLASH_NAMESPACE::Flash_fwd_params &params,
                             xla::ffi::DataType dtype, bool is_causal,
                             int head_dim, cudaStream_t stream) {
  if (dtype == xla::ffi::DataType::F16) {
    using T = cutlass::half_t;
    if (is_causal) {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_fwd_<T, 32, true>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_fwd_<T, 64, true>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_fwd_<T, 96, true>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_fwd_<T, 128, true>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_fwd_<T, 192, true>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_fwd_<T, 256, true>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for F16.");
      }
    } else {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_fwd_<T, 32, false>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_fwd_<T, 64, false>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_fwd_<T, 96, false>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_fwd_<T, 128, false>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_fwd_<T, 192, false>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_fwd_<T, 256, false>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for F16.");
      }
    }
    return Error::Success();
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (dtype == xla::ffi::DataType::BF16) {
    using T = cutlass::bfloat16_t;
    if (is_causal) {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_fwd_<T, 32, true>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_fwd_<T, 64, true>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_fwd_<T, 96, true>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_fwd_<T, 128, true>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_fwd_<T, 192, true>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_fwd_<T, 256, true>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for BF16.");
      }
    } else {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_fwd_<T, 32, false>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_fwd_<T, 64, false>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_fwd_<T, 96, false>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_fwd_<T, 128, false>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_fwd_<T, 192, false>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_fwd_<T, 256, false>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for BF16.");
      }
    }
    return Error::Success();
  }
#endif

  return MakeInvalid("Unsupported dtype for SM80 flash attention.");
}

inline Error DispatchBwdSm80(FLASH_NAMESPACE::Flash_bwd_params &params,
                             xla::ffi::DataType dtype, bool is_causal,
                             int head_dim, cudaStream_t stream) {
  if (dtype == xla::ffi::DataType::F16) {
    using T = cutlass::half_t;
    if (is_causal) {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_bwd_<T, 32, true>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_bwd_<T, 64, true>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_bwd_<T, 96, true>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_bwd_<T, 128, true>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_bwd_<T, 192, true>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_bwd_<T, 256, true>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for F16.");
      }
    } else {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_bwd_<T, 32, false>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_bwd_<T, 64, false>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_bwd_<T, 96, false>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_bwd_<T, 128, false>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_bwd_<T, 192, false>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_bwd_<T, 256, false>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for F16.");
      }
    }
    return Error::Success();
  }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (dtype == xla::ffi::DataType::BF16) {
    using T = cutlass::bfloat16_t;
    if (is_causal) {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_bwd_<T, 32, true>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_bwd_<T, 64, true>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_bwd_<T, 96, true>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_bwd_<T, 128, true>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_bwd_<T, 192, true>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_bwd_<T, 256, true>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for BF16.");
      }
    } else {
      switch (head_dim) {
      case 32:
        FLASH_NAMESPACE::run_mha_bwd_<T, 32, false>(params, stream);
        break;
      case 64:
        FLASH_NAMESPACE::run_mha_bwd_<T, 64, false>(params, stream);
        break;
      case 96:
        FLASH_NAMESPACE::run_mha_bwd_<T, 96, false>(params, stream);
        break;
      case 128:
        FLASH_NAMESPACE::run_mha_bwd_<T, 128, false>(params, stream);
        break;
      case 192:
        FLASH_NAMESPACE::run_mha_bwd_<T, 192, false>(params, stream);
        break;
      case 256:
        FLASH_NAMESPACE::run_mha_bwd_<T, 256, false>(params, stream);
        break;
      default:
        return MakeInvalid("Unsupported head_dim for BF16.");
      }
    }
    return Error::Success();
  }
#endif

  return MakeInvalid("Unsupported dtype for SM80 flash attention.");
}

Error FlashAttentionFwd(AnyBuffer q, AnyBuffer k, AnyBuffer v,
                        AnyBuffer alibi_slopes, AnyBuffer cu_seqlens_q,
                        AnyBuffer cu_seqlens_k, AnyBuffer block_table,
                        AnyBuffer rng_state, Result<AnyBuffer> out,
                        Result<AnyBuffer> softmax_lse,
                        double softmax_scale_attr, double dropout_prob_attr,
                        int64_t dropout_seed_attr, int64_t causal_attr,
                        int64_t window_left_attr, int64_t window_right_attr,
                        double softcap_attr, int64_t use_alibi_attr,
                        int64_t use_paged_kv_attr, int64_t max_seqlen_q_attr,
                        int64_t max_seqlen_k_attr, int64_t is_varlen_attr,
                        int64_t normalize_output_attr, cudaStream_t stream) {
  if (normalize_output_attr == 0) {
    return MakeInvalid(
        "normalize_output must be True for CUDA flash attention.");
  }
  if (dropout_prob_attr < 0.0 || dropout_prob_attr >= 1.0) {
    return MakeInvalid("dropout_prob must be in [0, 1).");
  }

  bool is_varlen = is_varlen_attr != 0;
  bool use_paged_kv = use_paged_kv_attr != 0;
  ShapeInfo q_info{};
  ShapeInfo k_info{};
  ShapeInfo v_info{};
  if (Error err = ParseQkvShape(q, is_varlen, q_info); err.failure())
    return err;
  if (Error err = ParseQkvShape(k, is_varlen, k_info); err.failure())
    return err;
  if (Error err = ParseQkvShape(v, is_varlen, v_info); err.failure())
    return err;

  if (q_info.head_dim != k_info.head_dim ||
      q_info.head_dim != v_info.head_dim) {
    return MakeInvalid("q/k/v head_dim mismatch.");
  }
  if (q_info.heads % k_info.heads != 0) {
    return MakeInvalid("num_heads must be divisible by num_kv_heads.");
  }

  if (k_info.seqlen != v_info.seqlen) {
    return MakeInvalid("k/v seqlen mismatch.");
  }
  if (use_paged_kv && is_varlen) {
    return MakeInvalid("paged_kv does not support varlen inputs.");
  }
  if (!is_varlen) {
    if (!use_paged_kv) {
      if (q_info.batch != k_info.batch || q_info.batch != v_info.batch) {
        return MakeInvalid("q/k/v batch mismatch.");
      }
    }
  } else {
    if (max_seqlen_q_attr <= 0 || max_seqlen_k_attr <= 0) {
      return MakeInvalid("max_seqlen_q/max_seqlen_k must be set for varlen.");
    }
  }

  if (q.element_type() != k.element_type() ||
      q.element_type() != v.element_type()) {
    return MakeInvalid("q/k/v dtype mismatch.");
  }

  if (softmax_lse->element_type() != xla::ffi::DataType::F32) {
    return MakeInvalid("softmax_lse must be float32.");
  }

  if (Error err =
          ValidateDeviceBuffers({&q, &k, &v, &alibi_slopes, &cu_seqlens_q,
                                 &cu_seqlens_k, &block_table, &rng_state});
      err.failure()) {
    return err;
  }

  RngStateDevice rng_state_dev;
  if (Error err = PrepareRngState(rng_state, stream, /*for_bwd=*/false,
                                  rng_state_dev);
      err.failure()) {
    CleanupRngState(rng_state_dev);
    return err;
  }

  int64_t batch = is_varlen ? cu_seqlens_q.dimensions().size() == 1
                                  ? cu_seqlens_q.dimensions()[0] - 1
                                  : 0
                            : q_info.batch;
  if (is_varlen && batch <= 0) {
    return MakeInvalid("cu_seqlens_q must have shape [batch+1].");
  }

  int64_t seqlen_q = is_varlen ? max_seqlen_q_attr : q_info.seqlen;
  int64_t page_block_size = k_info.seqlen;
  int64_t seqlen_k =
      is_varlen ? max_seqlen_k_attr
                : (use_paged_kv ? max_seqlen_k_attr : k_info.seqlen);

  int head_dim = static_cast<int>(q_info.head_dim);
  int h = static_cast<int>(q_info.heads);
  int h_k = static_cast<int>(k_info.heads);

  if (use_paged_kv) {
    if (block_table.element_type() != xla::ffi::DataType::S32) {
      return MakeInvalid("block_table must be int32 for paged_kv.");
    }
    Span<const int64_t> bt_dims = block_table.dimensions();
    if (bt_dims.size() != 2) {
      return MakeInvalid("block_table must be rank-2 for paged_kv.");
    }
    if (bt_dims[0] != q_info.batch) {
      return MakeInvalid("block_table batch dimension must match query batch.");
    }
    if (bt_dims[1] <= 0) {
      return MakeInvalid("block_table must have at least one block.");
    }
    if (page_block_size <= 0) {
      return MakeInvalid("paged_kv requires non-empty page_block_size.");
    }
    if (max_seqlen_k_attr <= 0) {
      return MakeInvalid("max_seqlen_k must be set for paged_kv.");
    }
    int64_t max_pages = bt_dims[1];
    if (max_seqlen_k_attr > max_pages * page_block_size) {
      return MakeInvalid("max_seqlen_k exceeds block_table capacity.");
    }
  }

  FLASH_NAMESPACE::Flash_fwd_params params{};
  params.q_ptr = const_cast<void *>(q.untyped_data());
  params.k_ptr = const_cast<void *>(k.untyped_data());
  params.v_ptr = const_cast<void *>(v.untyped_data());
  params.o_ptr = out->untyped_data();
  params.oaccum_ptr = nullptr;
  params.p_ptr = nullptr;
  params.softmax_lse_ptr = softmax_lse->untyped_data();
  params.softmax_lseaccum_ptr = nullptr;

  params.b = static_cast<int>(batch);
  params.seqlen_q = static_cast<int>(seqlen_q);
  params.seqlen_k = static_cast<int>(seqlen_k);
  params.seqlen_knew = 0;
  params.d = head_dim;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;

  params.q_row_stride = h * head_dim;
  params.k_row_stride = h_k * head_dim;
  params.v_row_stride = h_k * head_dim;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.q_batch_stride = params.q_row_stride * params.seqlen_q;
  params.k_batch_stride =
      params.k_row_stride * (use_paged_kv ? page_block_size : params.seqlen_k);
  params.v_batch_stride =
      params.v_row_stride * (use_paged_kv ? page_block_size : params.seqlen_k);

  params.o_row_stride = h * head_dim;
  params.o_head_stride = head_dim;
  params.o_batch_stride = params.o_row_stride * params.seqlen_q;

  params.seqlen_q_rounded = RoundUp(static_cast<int>(seqlen_q), 128);
  params.seqlen_k_rounded = RoundUp(static_cast<int>(seqlen_k), 128);
  params.d_rounded = RoundUp(head_dim, 8);
  params.rotary_dim = 0;
  params.total_q = static_cast<int>(is_varlen ? q_info.total : q_info.total);

  params.scale_softmax = static_cast<float>(softmax_scale_attr);
  params.scale_softmax_log2 = static_cast<float>(softmax_scale_attr * kLog2E);

  float keep_prob = 1.0f - static_cast<float>(dropout_prob_attr);
  params.p_dropout = keep_prob;
  params.p_dropout_in_uint8_t =
      static_cast<uint8_t>(std::floor(keep_prob * 255.0f));
  params.rp_dropout = keep_prob > 0.0f ? 1.0f / keep_prob : 0.0f;
  params.scale_softmax_rp_dropout = params.scale_softmax * params.rp_dropout;

  params.window_size_left = static_cast<int>(window_left_attr);
  params.window_size_right = static_cast<int>(window_right_attr);
  params.softcap = softcap_attr > 0.0
                       ? static_cast<float>(softcap_attr / params.scale_softmax)
                       : 0.0f;

  params.cu_seqlens_q =
      is_varlen ? static_cast<int *>(cu_seqlens_q.untyped_data()) : nullptr;
  params.cu_seqlens_k =
      is_varlen ? static_cast<int *>(cu_seqlens_k.untyped_data()) : nullptr;
  params.is_seqlens_k_cumulative = is_varlen;

  params.leftpad_k = nullptr;
  params.seqused_k = nullptr;
  params.blockmask = nullptr;

  params.knew_ptr = nullptr;
  params.vnew_ptr = nullptr;
  params.knew_batch_stride = 0;
  params.vnew_batch_stride = 0;
  params.knew_row_stride = 0;
  params.vnew_row_stride = 0;
  params.knew_head_stride = 0;
  params.vnew_head_stride = 0;

  params.rotary_cos_ptr = nullptr;
  params.rotary_sin_ptr = nullptr;

  params.cache_batch_idx = nullptr;
  if (use_paged_kv) {
    params.block_table =
        static_cast<int *>(const_cast<void *>(block_table.untyped_data()));
    params.block_table_batch_stride =
        static_cast<int>(block_table.dimensions()[1]);
    params.page_block_size = static_cast<int>(page_block_size);
  } else {
    params.block_table = nullptr;
    params.block_table_batch_stride = 0;
    params.page_block_size = 0;
  }

  params.philox_args.seed = static_cast<uint64_t>(dropout_seed_attr);
  params.philox_args.offset = 0;
  params.rng_state = rng_state_dev.ptr;

  params.is_bf16 = q.element_type() == xla::ffi::DataType::BF16;
  params.is_causal = causal_attr != 0;
  params.is_seqlens_k_cumulative = is_varlen;
  params.is_rotary_interleaved = false;
  params.num_splits = 1;
  params.alibi_slopes_ptr =
      (use_alibi_attr != 0) ? alibi_slopes.untyped_data() : nullptr;
  params.alibi_slopes_batch_stride = 0;
  params.unpadded_lse = is_varlen;
  params.seqlenq_ngroups_swapped = false;

  if (use_alibi_attr != 0) {
    if (alibi_slopes.element_type() != xla::ffi::DataType::F32) {
      return MakeInvalid("alibi_slopes must be float32.");
    }
    Span<const int64_t> adims = alibi_slopes.dimensions();
    if (adims.size() == 1) {
      if (adims[0] != h) {
        return MakeInvalid(
            "alibi_slopes must have shape [num_heads] or [batch, num_heads].");
      }
      params.alibi_slopes_batch_stride = 0;
    } else if (adims.size() == 2) {
      if (adims[0] != batch || adims[1] != h) {
        return MakeInvalid("alibi_slopes must have shape [batch, num_heads].");
      }
      params.alibi_slopes_batch_stride = static_cast<int64_t>(adims[1]);
    } else {
      return MakeInvalid("alibi_slopes must be rank 1 or 2.");
    }
  }

  Error err = DispatchFwdSm80(params, q.element_type(), params.is_causal,
                              head_dim, stream);
  if (err.failure()) {
    CleanupRngState(rng_state_dev);
    return err;
  }

  if (Error e = CheckCuda(cudaPeekAtLastError(), "flash_attention_fwd launch");
      e.failure()) {
    CleanupRngState(rng_state_dev);
    return e;
  }
  if (Error e = SyncBackRngStateU32(rng_state, rng_state_dev, stream);
      e.failure()) {
    CleanupRngState(rng_state_dev);
    return e;
  }
  CleanupRngState(rng_state_dev);
  return Error::Success();
}

Error FlashAttentionBwd(AnyBuffer q, AnyBuffer k, AnyBuffer v,
                        AnyBuffer out_buf, AnyBuffer dout,
                        AnyBuffer softmax_lse, AnyBuffer alibi_slopes,
                        AnyBuffer cu_seqlens_q, AnyBuffer cu_seqlens_k,
                        AnyBuffer block_table, AnyBuffer rng_state,
                        Result<AnyBuffer> dq, Result<AnyBuffer> dk,
                        Result<AnyBuffer> dv, double softmax_scale_attr,
                        double dropout_prob_attr, int64_t causal_attr,
                        int64_t window_left_attr, int64_t window_right_attr,
                        double softcap_attr, int64_t use_alibi_attr,
                        int64_t use_paged_kv_attr, int64_t max_seqlen_q_attr,
                        int64_t max_seqlen_k_attr, int64_t is_varlen_attr,
                        int64_t normalize_output_attr, cudaStream_t stream) {
  if (use_paged_kv_attr != 0) {
    return MakeInvalid("paged_kv is not supported for flash attention backward.");
  }
  if (normalize_output_attr == 0) {
    return MakeInvalid(
        "normalize_output must be True for CUDA flash attention.");
  }
  if (dropout_prob_attr < 0.0 || dropout_prob_attr >= 1.0) {
    return MakeInvalid("dropout_prob must be in [0, 1).");
  }

  bool is_varlen = is_varlen_attr != 0;
  ShapeInfo q_info{};
  ShapeInfo k_info{};
  ShapeInfo v_info{};
  if (Error err = ParseQkvShape(q, is_varlen, q_info); err.failure())
    return err;
  if (Error err = ParseQkvShape(k, is_varlen, k_info); err.failure())
    return err;
  if (Error err = ParseQkvShape(v, is_varlen, v_info); err.failure())
    return err;

  if (q_info.head_dim != k_info.head_dim ||
      q_info.head_dim != v_info.head_dim) {
    return MakeInvalid("q/k/v head_dim mismatch.");
  }
  if (q_info.heads % k_info.heads != 0) {
    return MakeInvalid("num_heads must be divisible by num_kv_heads.");
  }

  if (q.element_type() != k.element_type() ||
      q.element_type() != v.element_type()) {
    return MakeInvalid("q/k/v dtype mismatch.");
  }
  if (dout.element_type() != q.element_type()) {
    return MakeInvalid("dout dtype mismatch.");
  }

  if (softmax_lse.element_type() != xla::ffi::DataType::F32) {
    return MakeInvalid("softmax_lse must be float32.");
  }

  if (Error err = ValidateDeviceBuffers(
          {&q, &k, &v, &out_buf, &dout, &softmax_lse, &alibi_slopes,
           &cu_seqlens_q, &cu_seqlens_k, &block_table, &rng_state});
      err.failure()) {
    return err;
  }

  RngStateDevice rng_state_dev;
  if (Error err = PrepareRngState(rng_state, stream, /*for_bwd=*/true,
                                  rng_state_dev);
      err.failure()) {
    CleanupRngState(rng_state_dev);
    return err;
  }

  int64_t batch = is_varlen ? cu_seqlens_q.dimensions().size() == 1
                                  ? cu_seqlens_q.dimensions()[0] - 1
                                  : 0
                            : q_info.batch;
  if (is_varlen && batch <= 0) {
    return MakeInvalid("cu_seqlens_q must have shape [batch+1].");
  }

  int64_t seqlen_q = is_varlen ? max_seqlen_q_attr : q_info.seqlen;
  int64_t seqlen_k = is_varlen ? max_seqlen_k_attr : k_info.seqlen;

  int head_dim = static_cast<int>(q_info.head_dim);
  int h = static_cast<int>(q_info.heads);
  int h_k = static_cast<int>(k_info.heads);

  int seqlen_q_rounded = RoundUp(static_cast<int>(seqlen_q), 128);
  int seqlen_k_rounded = RoundUp(static_cast<int>(seqlen_k), 128);
  int d_rounded = RoundUp(head_dim, 8);

  size_t dq_accum_len = static_cast<size_t>(h) * d_rounded;
  size_t dq_rows = is_varlen ? static_cast<size_t>(q_info.total + 128 * batch)
                             : static_cast<size_t>(batch * seqlen_q_rounded);
  dq_accum_len *= dq_rows;

  size_t dkv_rows = static_cast<size_t>(batch) * seqlen_k_rounded;
  size_t dkv_accum_len = dkv_rows * static_cast<size_t>(h_k) * d_rounded;

  size_t dpsum_len =
      static_cast<size_t>(h) *
      (is_varlen ? static_cast<size_t>(q_info.total + 128 * batch)
                 : static_cast<size_t>(batch * seqlen_q_rounded));

  void *dq_accum = nullptr;
  void *dk_accum = nullptr;
  void *dv_accum = nullptr;
  void *dsoftmax_sum = nullptr;

  if (Error err = CheckCuda(cudaMalloc(&dq_accum, dq_accum_len * sizeof(float)),
                            "cudaMalloc dq_accum");
      err.failure())
    return err;
  if (Error err =
          CheckCuda(cudaMalloc(&dk_accum, dkv_accum_len * sizeof(float)),
                    "cudaMalloc dk_accum");
      err.failure())
    return err;
  if (Error err =
          CheckCuda(cudaMalloc(&dv_accum, dkv_accum_len * sizeof(float)),
                    "cudaMalloc dv_accum");
      err.failure())
    return err;
  if (Error err =
          CheckCuda(cudaMalloc(&dsoftmax_sum, dpsum_len * sizeof(float)),
                    "cudaMalloc dsoftmax_sum");
      err.failure())
    return err;

  FLASH_NAMESPACE::Flash_bwd_params params{};
  params.q_ptr = const_cast<void *>(q.untyped_data());
  params.k_ptr = const_cast<void *>(k.untyped_data());
  params.v_ptr = const_cast<void *>(v.untyped_data());
  params.o_ptr = const_cast<void *>(out_buf.untyped_data());
  params.do_ptr = const_cast<void *>(dout.untyped_data());

  params.dq_ptr = dq->untyped_data();
  params.dk_ptr = dk->untyped_data();
  params.dv_ptr = dv->untyped_data();

  params.dq_accum_ptr = dq_accum;
  params.dk_accum_ptr = dk_accum;
  params.dv_accum_ptr = dv_accum;
  params.dsoftmax_sum = dsoftmax_sum;

  params.b = static_cast<int>(batch);
  params.seqlen_q = static_cast<int>(seqlen_q);
  params.seqlen_k = static_cast<int>(seqlen_k);
  params.seqlen_knew = 0;
  params.d = head_dim;
  params.h = h;
  params.h_k = h_k;
  params.h_h_k_ratio = h / h_k;

  params.q_row_stride = h * head_dim;
  params.k_row_stride = h_k * head_dim;
  params.v_row_stride = h_k * head_dim;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.q_batch_stride = params.q_row_stride * params.seqlen_q;
  params.k_batch_stride = params.k_row_stride * params.seqlen_k;
  params.v_batch_stride = params.v_row_stride * params.seqlen_k;

  params.o_row_stride = h * head_dim;
  params.o_head_stride = head_dim;
  params.o_batch_stride = params.o_row_stride * params.seqlen_q;

  params.do_row_stride = params.o_row_stride;
  params.do_head_stride = params.o_head_stride;
  params.do_batch_stride = params.o_batch_stride;

  params.dq_row_stride = params.q_row_stride;
  params.dk_row_stride = params.k_row_stride;
  params.dv_row_stride = params.v_row_stride;
  params.dq_head_stride = head_dim;
  params.dk_head_stride = head_dim;
  params.dv_head_stride = head_dim;
  params.dq_batch_stride = params.dq_row_stride * params.seqlen_q;
  params.dk_batch_stride = params.dk_row_stride * params.seqlen_k;
  params.dv_batch_stride = params.dv_row_stride * params.seqlen_k;

  params.seqlen_q_rounded = seqlen_q_rounded;
  params.seqlen_k_rounded = seqlen_k_rounded;
  params.d_rounded = d_rounded;
  params.rotary_dim = 0;
  params.total_q = static_cast<int>(is_varlen ? q_info.total : q_info.total);

  params.scale_softmax = static_cast<float>(softmax_scale_attr);
  params.scale_softmax_log2 = static_cast<float>(softmax_scale_attr * kLog2E);

  float keep_prob = 1.0f - static_cast<float>(dropout_prob_attr);
  params.p_dropout = keep_prob;
  params.p_dropout_in_uint8_t =
      static_cast<uint8_t>(std::floor(keep_prob * 255.0f));
  params.rp_dropout = keep_prob > 0.0f ? 1.0f / keep_prob : 0.0f;
  params.scale_softmax_rp_dropout = params.scale_softmax * params.rp_dropout;

  params.window_size_left = static_cast<int>(window_left_attr);
  params.window_size_right = static_cast<int>(window_right_attr);
  params.softcap = softcap_attr > 0.0
                       ? static_cast<float>(softcap_attr / params.scale_softmax)
                       : 0.0f;

  params.cu_seqlens_q =
      is_varlen ? static_cast<int *>(cu_seqlens_q.untyped_data()) : nullptr;
  params.cu_seqlens_k =
      is_varlen ? static_cast<int *>(cu_seqlens_k.untyped_data()) : nullptr;
  params.is_seqlens_k_cumulative = is_varlen;

  params.leftpad_k = nullptr;
  params.seqused_k = nullptr;
  params.blockmask = nullptr;

  params.knew_ptr = nullptr;
  params.vnew_ptr = nullptr;
  params.knew_batch_stride = 0;
  params.vnew_batch_stride = 0;
  params.knew_row_stride = 0;
  params.vnew_row_stride = 0;
  params.knew_head_stride = 0;
  params.vnew_head_stride = 0;

  params.rotary_cos_ptr = nullptr;
  params.rotary_sin_ptr = nullptr;

  params.cache_batch_idx = nullptr;
  params.block_table = nullptr;
  params.block_table_batch_stride = 0;
  params.page_block_size = 0;

  params.rng_state = rng_state_dev.ptr;
  params.softmax_lse_ptr = const_cast<void *>(softmax_lse.untyped_data());
  params.oaccum_ptr = nullptr;
  params.softmax_lseaccum_ptr = nullptr;

  params.is_bf16 = q.element_type() == xla::ffi::DataType::BF16;
  params.is_causal = causal_attr != 0;
  params.is_rotary_interleaved = false;
  params.num_splits = 1;
  params.alibi_slopes_ptr =
      (use_alibi_attr != 0) ? alibi_slopes.untyped_data() : nullptr;
  params.alibi_slopes_batch_stride = 0;
  params.unpadded_lse = is_varlen;
  params.seqlenq_ngroups_swapped = false;

  if (use_alibi_attr != 0) {
    if (alibi_slopes.element_type() != xla::ffi::DataType::F32) {
      return MakeInvalid("alibi_slopes must be float32.");
    }
    Span<const int64_t> adims = alibi_slopes.dimensions();
    if (adims.size() == 1) {
      if (adims[0] != h) {
        return MakeInvalid(
            "alibi_slopes must have shape [num_heads] or [batch, num_heads].");
      }
      params.alibi_slopes_batch_stride = 0;
    } else if (adims.size() == 2) {
      if (adims[0] != batch || adims[1] != h) {
        return MakeInvalid("alibi_slopes must have shape [batch, num_heads].");
      }
      params.alibi_slopes_batch_stride = static_cast<int64_t>(adims[1]);
    } else {
      return MakeInvalid("alibi_slopes must be rank 1 or 2.");
    }
  }

  Error err = DispatchBwdSm80(params, q.element_type(), params.is_causal,
                              head_dim, stream);
  if (err.failure()) {
    cudaFree(dq_accum);
    cudaFree(dk_accum);
    cudaFree(dv_accum);
    cudaFree(dsoftmax_sum);
    CleanupRngState(rng_state_dev);
    return err;
  }

  if (Error e = CheckCuda(cudaPeekAtLastError(), "flash_attention_bwd launch");
      e.failure()) {
    cudaFree(dq_accum);
    cudaFree(dk_accum);
    cudaFree(dv_accum);
    cudaFree(dsoftmax_sum);
    CleanupRngState(rng_state_dev);
    return e;
  }

  cudaFree(dq_accum);
  cudaFree(dk_accum);
  cudaFree(dv_accum);
  cudaFree(dsoftmax_sum);
  CleanupRngState(rng_state_dev);
  return Error::Success();
}

} // namespace

extern "C" XLA_FFI_Error *
ejk_flash_attention_cuda_fwd(XLA_FFI_CallFrame *call_frame) {
  static auto handler = xla::ffi::Ffi::Bind()
                            .Arg<AnyBuffer>() // q
                            .Arg<AnyBuffer>() // k
                            .Arg<AnyBuffer>() // v
                            .Arg<AnyBuffer>() // alibi_slopes
                            .Arg<AnyBuffer>() // cu_seqlens_q
                            .Arg<AnyBuffer>() // cu_seqlens_k
                            .Arg<AnyBuffer>() // block_table
                            .Arg<AnyBuffer>() // rng_state
                            .Ret<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Attr<double>("softmax_scale")
                            .Attr<double>("dropout_prob")
                            .Attr<int64_t>("dropout_seed")
                            .Attr<int64_t>("causal")
                            .Attr<int64_t>("window_left")
                            .Attr<int64_t>("window_right")
                            .Attr<double>("softcap")
                            .Attr<int64_t>("use_alibi")
                            .Attr<int64_t>("use_paged_kv")
                            .Attr<int64_t>("max_seqlen_q")
                            .Attr<int64_t>("max_seqlen_k")
                            .Attr<int64_t>("is_varlen")
                            .Attr<int64_t>("normalize_output")
                            .Ctx<PlatformStream<cudaStream_t>>()
                            .To(FlashAttentionFwd);
  return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error *
ejk_flash_attention_cuda_bwd(XLA_FFI_CallFrame *call_frame) {
  static auto handler = xla::ffi::Ffi::Bind()
                            .Arg<AnyBuffer>() // q
                            .Arg<AnyBuffer>() // k
                            .Arg<AnyBuffer>() // v
                            .Arg<AnyBuffer>() // out
                            .Arg<AnyBuffer>() // dout
                            .Arg<AnyBuffer>() // softmax_lse
                            .Arg<AnyBuffer>() // alibi_slopes
                            .Arg<AnyBuffer>() // cu_seqlens_q
                            .Arg<AnyBuffer>() // cu_seqlens_k
                            .Arg<AnyBuffer>() // block_table
                            .Arg<AnyBuffer>() // rng_state
                            .Ret<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Ret<AnyBuffer>()
                            .Attr<double>("softmax_scale")
                            .Attr<double>("dropout_prob")
                            .Attr<int64_t>("causal")
                            .Attr<int64_t>("window_left")
                            .Attr<int64_t>("window_right")
                            .Attr<double>("softcap")
                            .Attr<int64_t>("use_alibi")
                            .Attr<int64_t>("use_paged_kv")
                            .Attr<int64_t>("max_seqlen_q")
                            .Attr<int64_t>("max_seqlen_k")
                            .Attr<int64_t>("is_varlen")
                            .Attr<int64_t>("normalize_output")
                            .Ctx<PlatformStream<cudaStream_t>>()
                            .To(FlashAttentionBwd);
  return handler->Call(call_frame);
}
