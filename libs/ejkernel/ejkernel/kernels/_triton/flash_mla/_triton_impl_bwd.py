# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Backward pass Triton kernels for Flash Multi-Latent Attention.

This module provides gradient computation kernels for Multi-Latent Attention (MLA),
enabling end-to-end training of models that use latent-projected attention.

MLA Backward Algorithm:
----------------------
The backward pass must compute gradients with respect to:
1. Queries (dQ): Gradient of loss w.r.t. query tensor
2. Keys (dK): Gradient of loss w.r.t. key tensor
3. Values (dV): Gradient of loss w.r.t. value tensor
4. Latent projections (dLatent_K, dLatent_V): Gradients for projection matrices

The latent projection gradients require special handling since they participate
in both the key-query dot product and value aggregation.

Expected Kernels (not yet implemented):
- _mla_attn_bwd_preprocess: Compute delta = sum(O * dO) for stable softmax gradients
- _mla_attn_bwd_dq: Compute query gradients
- _mla_attn_bwd_dkv: Compute key/value gradients
- _bwd_mla_attention_kernel_call: Python wrapper for backward pass

Note:
    This module is a placeholder for future MLA backward pass implementation.
    The forward pass is defined in _triton_impl_fwd.py.
"""
