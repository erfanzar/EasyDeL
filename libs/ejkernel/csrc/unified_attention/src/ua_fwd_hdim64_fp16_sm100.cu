// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "ua_launch_template.h"

namespace ua {

template<>
void run_unified_attention_<half, 64>(UaParams &params, cudaStream_t stream) {
    run_unified_attention_hdim<half, 64>(params, stream);
}

}  // namespace ua
