
// Copyright (c) 2025, erfanzar.
//
// Original work by Tri Dao (Copyright (c) 2024).
// Special thanks to Tri Dao for the foundational implementation.
// We have only modified the code structure for better integration with minimal changes.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "flash_bwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_SM8x
#ifndef FLASHATTENTION_DISABLE_HDIM128
template<>
void run_mha_bwd_<80, cutlass::bfloat16_t, 128, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<80, cutlass::bfloat16_t, true>(params, stream);
}
template<>
void run_mha_bwd_<86, cutlass::bfloat16_t, 128, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_hdim128<86, cutlass::bfloat16_t, true>(params, stream);
}
#endif
#endif
