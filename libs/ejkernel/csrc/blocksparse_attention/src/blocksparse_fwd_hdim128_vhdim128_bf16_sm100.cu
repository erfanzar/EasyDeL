// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "blocksparse_attention_launch_template.h"

namespace blocksparse {

template<>
void run_blocksparse_fwd_<__nv_bfloat16, 128, 128>(Blocksparse_fwd_params &params, cudaStream_t stream) {
    run_blocksparse_fwd_hdim<__nv_bfloat16, 128, 128>(params, stream);
}

} // namespace blocksparse
