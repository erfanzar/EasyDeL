
// Copyright (c) 2025, erfanzar.
//
// Original work by Tri Dao (Copyright (c) 2024).
// Special thanks to Tri Dao for the foundational implementation.
// We have only modified the code structure for better integration with minimal changes.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "flash_fwd_launch_template.h"

#ifndef FLASHATTENTION_DISABLE_HDIM96
template void run_mha_fwd_<90, cutlass::bfloat16_t, 96, 96, false, false, false, true>(Flash_fwd_params &params, cudaStream_t stream);
#endif
