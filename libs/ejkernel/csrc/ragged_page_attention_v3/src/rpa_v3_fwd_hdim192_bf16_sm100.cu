// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "rpa_v3_launch_template.h"

namespace rpa_v3 {

template<>
void run_rpa_v3_update_kv_<__nv_bfloat16, 192>(RpaV3Params &params, cudaStream_t stream) {
    run_rpa_v3_update_kv_hdim<__nv_bfloat16, 192>(params, stream);
}

template<>
void run_rpa_v3_attention_<__nv_bfloat16, 192>(RpaV3Params &params, cudaStream_t stream) {
    run_rpa_v3_attention_hdim<__nv_bfloat16, 192>(params, stream);
}

} // namespace rpa_v3
