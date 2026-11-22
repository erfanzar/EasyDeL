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

"""CuTe DSL Flash Attention implementation.

Exposes :func:`flash_attention`, the registry entry point for the CuTe
platform.  Both forward and backward passes are supported through a
``jax.custom_vjp`` rule backed by CuTe DSL GPU kernels.  Supports
dense and paged-KV inputs, GQA/MQA, causal masking, sliding-window
attention, explicit attention masks, additive bias, logit soft-capping,
and attention-sink auxiliary logits.
"""

from ._interface import flash_attention

__all__ = ["flash_attention"]
