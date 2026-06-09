// Minimal ATen/CUDA shim for flash-attention kernels (JAX FFI integration).
// Provides PhiloxCudaState and unpack compatible with upstream code.

#pragma once

#include <cstdint>
#include <tuple>

namespace at {

struct PhiloxCudaState {
  uint64_t seed = 0;
  uint64_t offset = 0;
};

namespace cuda {
namespace philox {

inline __host__ __device__ std::tuple<uint64_t, uint64_t>
unpack(PhiloxCudaState state) {
  return {state.seed, state.offset};
}

} // namespace philox
} // namespace cuda
} // namespace at
