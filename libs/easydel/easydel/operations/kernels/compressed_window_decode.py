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

"""EasyDeL adapter for DeepSeek-V4's compressed-window decode attention.

Thin ``OperationImpl`` over the ejkernel ``compressed_window_decode`` public
operation. DeepSeek-V4's decode step attends a single query per request slot
over that slot's sliding-window token ring concatenated with its visible
compressed entries, with per-head learnable attention sinks. The model
materialises the shared-KV axis (``K == V``) and the additive bias — the bias
already folds in ring-slot validity, compressed-entry causal thresholds, and
the Lightning-Indexer top-k gating — so this adapter is a pure attention
primitive: it maps EasyDeL dtype/backend/sharding onto the ejkernel op and
returns an :class:`AttentionOutput`.

The single shared KV head is replicated across the tensor-parallel axis while
the query/output heads are sharded (the natural head-parallel layout), so each
device runs the kernel over its local heads without in-kernel collectives.
"""

from __future__ import annotations

import os
import typing as tp

import jax
from ejkernel.modules import compressed_window_decode as _ejkernel_compressed_window_decode
from jax import numpy as jnp
from jaxtyping import Array, Float

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationRegistry
from ..requirements import ExecutionMode, MetadataField, OperationRequirements, RequirementsBuilder

# Platform override for A/B testing the Pallas kernel against the XLA reference
# in-model (see debug-ejkernel-ops). Default "auto" prefers Pallas on TPU.
_PLATFORM: tp.Final = os.getenv("EASYDEL_COMPRESSED_WINDOW_DECODE_PLATFORM", "auto")


@OperationRegistry.register
class CompressedWindowDecodeAttn(OperationImpl):
    """Shared-KV sink-augmented decode attention (DeepSeek-V4 compressed window).

    Delegates to the ejkernel ``compressed_window_decode`` operation (Pallas on
    TPU, XLA reference elsewhere). Cache-agnostic: the caller passes a
    materialised KV axis and additive bias rather than a cache view.
    """

    @classmethod
    def get_impl_name(cls) -> str:
        """Return the operation registry key."""
        return "compressed_window_decode"

    @classmethod
    def get_requirements(cls, mode: ExecutionMode = ExecutionMode.DECODE) -> OperationRequirements:
        """Declare requirements: basic metadata, no engine-managed cache view.

        DeepSeek-V4 owns its :class:`~easydel.caching.CompressedWindowCache`
        and hands this operation an already-materialised KV axis plus bias, so
        no engine cache view is required.

        Args:
            mode: Execution mode (decode).

        Returns:
            OperationRequirements: The declared requirements.
        """
        return (
            RequirementsBuilder("compressed_window_decode")
            .require_metadata(MetadataField.NONE)
            .no_cache_required()
            .build()
        )

    @jax.named_scope("easydel-compressed_window_decode")
    def forward_native(
        self,
        query: Float[Array, "batch num_heads q_len head_dim"],
        kv: Float[Array, "batch kv_len head_dim"],
        bias: Float[Array, "batch q_len kv_len"],
        softmax_aux: Float[Array, "num_heads"] | None = None,  # noqa: F821
        softmax_scale: float | None = None,
        **ignores,
    ) -> AttentionOutput:
        """Run compressed-window decode attention via the ejkernel operation.

        Casts the query/KV to the runtime compute dtype (the additive bias and
        the sink logits stay float32 to preserve the softmax accumulation
        precision), then delegates to the ejkernel op. Sharding of the
        head-parallel query/output and the replicated KV is left to GSPMD (both
        the XLA and Pallas paths partition over the head batch axis).

        Args:
            query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
            kv: Shared key/value axis ``[batch, kv_len, head_dim]`` (``K == V``).
            bias: Additive fp32 mask ``[batch, q_len, kv_len]`` (head-broadcast).
            softmax_aux: Optional per-head sink logits ``[num_heads]``.
            softmax_scale: Logit scale; defaults to ``head_dim ** -0.5``.
            **ignores: Ignored keyword arguments (interface parity).

        Returns:
            AttentionOutput: ``attention_outputs`` of shape
            ``[batch, num_heads, q_len, head_dim]``; no weights.
        """
        del ignores
        runtime_dtype = self.metadata.runtime_dtype
        out = _ejkernel_compressed_window_decode(
            query.astype(runtime_dtype),
            kv.astype(runtime_dtype),
            bias.astype(jnp.float32),
            None if softmax_aux is None else softmax_aux.astype(jnp.float32),
            softmax_scale=softmax_scale,
            platform=None if _PLATFORM == "auto" else _PLATFORM,
        )
        return AttentionOutput(attention_weights=None, attention_outputs=out)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """TPU forward pass (Pallas kernel); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass; delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """CPU forward pass (XLA reference); delegates to :meth:`forward_native`."""
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """CUDA forward pass; delegates to :meth:`forward_gpu`."""
        return self.forward_gpu(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """ROCm forward pass; delegates to :meth:`forward_gpu`."""
        return self.forward_gpu(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch num_heads q_len head_dim"],
        kv: Float[Array, "batch kv_len head_dim"],
        bias: Float[Array, "batch q_len kv_len"],
        softmax_aux: Float[Array, "num_heads"] | None = None,  # noqa: F821
        softmax_scale: float | None = None,
        **ignores,
    ) -> AttentionOutput:
        """Dispatch to the backend-specific forward.

        Args:
            query: Rotated queries ``[batch, num_heads, q_len, head_dim]``.
            kv: Shared key/value axis ``[batch, kv_len, head_dim]``.
            bias: Additive fp32 mask ``[batch, q_len, kv_len]``.
            softmax_aux: Optional per-head sink logits ``[num_heads]``.
            softmax_scale: Logit scale.
            **ignores: Ignored keyword arguments.

        Returns:
            AttentionOutput: The attention result.
        """
        return super().__call__(
            query=query,
            kv=kv,
            bias=bias,
            softmax_aux=softmax_aux,
            softmax_scale=softmax_scale,
            **ignores,
        )
