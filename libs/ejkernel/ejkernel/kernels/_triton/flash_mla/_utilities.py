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

"""Utility functions for Flash Multi-Latent Attention.

This module provides helper functions for Multi-Latent Attention (MLA),
including latent projection preparation and block size selection.

Expected Utilities (not yet implemented):
- compute_latent_projections: Prepare low-rank key/value projections
- select_block_sizes: Choose optimal block dimensions for MLA kernels
- validate_mla_inputs: Input shape and dtype validation
- create_mla_mask: Generate attention masks compatible with MLA

Multi-Latent Attention Overview:
-------------------------------
MLA reduces memory and computation by projecting keys and values
to lower-dimensional latent spaces before attention computation:

1. Project K and V to latent spaces: K_latent = K @ W_k, V_latent = V @ W_v
2. Compute attention in latent space
3. Project output back to original dimension

This reduces complexity from O(N * D) to O(N * L) where L << D.

Note:
    This module is a placeholder for future MLA utility implementation.
"""
