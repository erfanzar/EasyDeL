// Copyright (c) 2025, erfanzar.
//
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "code_gen.py"

#include "rpa_v3_launch_template.h"

namespace rpa_v3 {

template<>
void run_rpa_v3_update_kv_<half, 128>(RpaV3Params &params, cudaStream_t stream) {
    run_rpa_v3_update_kv_hdim<half, 128>(params, stream);
}

template<>
void run_rpa_v3_attention_<half, 128>(RpaV3Params &params, cudaStream_t stream) {
    run_rpa_v3_attention_hdim<half, 128>(params, stream);
}

} // namespace rpa_v3
