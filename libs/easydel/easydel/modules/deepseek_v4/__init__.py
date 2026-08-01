# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""DeepSeek-V4 model family for EasyDeL.

DeepSeek-V4 combines manifold-constrained hyper-connections (mHC parallel
residual streams with Sinkhorn-projected mixing), shared-KV multi-query
attention with per-head sinks and partial interleaved RoPE, per-layer
sliding / compressed-sparse (CSA + Lightning Indexer) / heavily-compressed
(HCA) attention regimes, and a 256-expert MoE with ``sqrtsoftplus`` scoring,
clamped SwiGLU experts, a shared expert, and hash routing on the first
layers.

This port implements the numerically-faithful *stateless* forward
(training / prefill). Decode-time caching of the compressor branches and
eSurge serving are not supported — see
``.claude/projects/port_hyv3_dsv4_mistral4.md``.
"""

from .deepseek_v4_configuration import DeepseekV4Config
from .modeling_deepseek_v4 import DeepseekV4ForCausalLM, DeepseekV4Model

__all__ = ["DeepseekV4Config", "DeepseekV4ForCausalLM", "DeepseekV4Model"]
