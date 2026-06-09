
// Copyright (c) 2025, erfanzar.
//
// Original work by Tri Dao (Copyright (c) 2024).
// Special thanks to Tri Dao for the foundational implementation.
// We have only modified the code structure for better integration with minimal changes.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "namespace_config.h"
#include "flash_fwd_launch_template.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_<cutlass::half_t, 64, true>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64<cutlass::half_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE