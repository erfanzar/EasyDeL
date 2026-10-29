// Minimal C10 CUDA exception shim for FlashAttention kernels.
#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#ifndef C10_CUDA_CHECK
#define C10_CUDA_CHECK(expr)                                                   \
  do {                                                                         \
    cudaError_t _err = (expr);                                                 \
    if (_err != cudaSuccess) {                                                 \
      std::printf("C10_CUDA_CHECK failed: %s (%s:%d)\n",                       \
                  cudaGetErrorString(_err), __FILE__, __LINE__);               \
    }                                                                          \
  } while (0)
#endif

#ifndef C10_CUDA_KERNEL_LAUNCH_CHECK
#define C10_CUDA_KERNEL_LAUNCH_CHECK()                                         \
  do {                                                                         \
    cudaError_t _err = cudaGetLastError();                                     \
    if (_err != cudaSuccess) {                                                 \
      std::printf("C10_CUDA_KERNEL_LAUNCH_CHECK failed: %s (%s:%d)\n",         \
                  cudaGetErrorString(_err), __FILE__, __LINE__);               \
    }                                                                          \
  } while (0)
#endif
