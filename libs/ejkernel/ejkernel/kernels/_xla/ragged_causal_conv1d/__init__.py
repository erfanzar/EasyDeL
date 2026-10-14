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

"""XLA ragged causal conv1d kernel package.

Provides a pure-JAX/XLA causal depthwise conv1d that operates over a packed
("ragged") batch of variable-length requests while maintaining a per-slot rolling
convolution state, as used by Qwen3-Next / GatedDeltaNet style mixers during
mixed prefill+decode inference. The public entry point is
:func:`ragged_causal_conv1d`, registered with the kernel registry for
``Platform.XLA``; the interface shim lives in ``_interface`` and the XLA
implementation in ``_xla_impl_fwd``.
"""

from ._interface import ragged_causal_conv1d

__all__ = ("ragged_causal_conv1d",)
