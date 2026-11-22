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

"""XLA backend package for the grouped gated-delta-rule (GDR) single-step decode kernel.

This package provides the portable XLA/JAX reference path for one autoregressive decode
step of grouped gated delta-rule linear attention (used by Qwen3-Next / Qwen3.5 style
models). "Grouped" means several value heads share each key head via an ``expand_ratio``
expansion, and "decode" means exactly one new token is processed per call. The linear
recurrent state is read as a read-only input and a new, updated state is returned (JAX
arrays are immutable, so it is never mutated in place).

Contents:
    - ``gated_delta_rule_grouped_decode``: the registry-facing entry point (re-exported from
      :mod:`._interface`), runtime-typechecked and registered for ``Platform.XLA`` so the
      kernel registry can dispatch to it on any backend that lacks a fused implementation.

The pure-JAX numerics live in :mod:`._xla_impl_fwd`; this package is selected by the kernel
registry as the XLA fallback for the ``"gated_delta_rule_grouped_decode"`` algorithm.
"""

from ._interface import gated_delta_rule_grouped_decode

__all__ = ("gated_delta_rule_grouped_decode",)
