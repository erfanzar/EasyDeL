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

"""Splash Attention implementation for TPU acceleration.

This module provides the Splash Attention implementation, a TPU-optimized
attention mechanism that leverages the Pallas framework for maximum performance.
Splash Attention is specifically designed to take advantage of TPU's matrix
multiplication units and memory hierarchy.

Key features:
- TPU-specific optimization using Pallas kernels
- Support for Multi-Query Attention (MQA) and Grouped-Query Attention (GQA)
- Efficient handling of causal masks
- Automatic fallback to vanilla attention for unsupported configurations
- Optimized for sequences with lengths divisible by 128

Implementation details:
- Uses `make_splash_mqa_single_device` primitive from JAX experimental
- Requires specific block sizes for optimal TPU utilization
- Falls back to vanilla attention for:
  * Single token generation (seq_len = 1)
  * Non-causal attention patterns
  * Sequences not divisible by 128

Example:
    >>> from easydel.layers.attention_operator import OperationMetadata
    >>> from easydel.layers.attention_operator.modules import BlockSparseAttn
    >>>
    >>> # Configure for TPU execution
    >>> metadata = OperationMetadata(
    ...     runtime_dtype=jnp.bfloat16,
    ...     softmax_scale=1.0 / math.sqrt(head_dim),
    ...     blocksize_q=256,
    ...     blocksize_k=512
    ... )
    >>> splash_attn = BlockSparseAttn(metadata)
    >>>
    >>> # Use with sequences divisible by 128
    >>> output = splash_attn(query, key, value, causal=True)

Note:
    Splash Attention is only available on TPU devices and will raise
    NotImplementedError on CPU or GPU backends.

References:
    - JAX Pallas documentation for TPU kernels
    - Google Research papers on TPU-optimized attention
"""

from __future__ import annotations

import typing as tp

import jax
from eformer import common_types
from eformer.loggings import get_logger
from ejkernel.modules import blocksparse_attention  # pyright: ignore[reportMissingTypeStubs]
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from jax import numpy as jnp
from jax import random as jr
from jax.sharding import PartitionSpec
from jaxtyping import Array, Float

from easydel.caching import TransformerCacheView
from easydel.caching.transformer import TransformerMetadata

from .._attention_outputs import AttentionOutput
from .._operation_impl import OperationImpl, OperationMetadata, OperationRegistry
from ..requirements import (
    CacheType,
    ExecutionMode,
    MetadataField,
    OperationRequirements,
)
from .vanilla_attention import VanillaAttn

if tp.TYPE_CHECKING:
    pass

logger = get_logger("EasyDeL-BlockSparseAttn")


@OperationRegistry.register
class BlockSparseAttn(OperationImpl):
    """
    An attention implementation using the Pallas Splash Attention kernel for TPUs.

    Splash Attention is an optimized attention mechanism designed for TPUs.
    This implementation provides a wrapper around the `make_splash_mqa_single_device`
    primitive.

    Note:
        - This implementation is primarily intended for TPUs.
        - It falls back to `VanillaAttn` under certain conditions:
            - Query sequence length is 1 (generation mode).
            - `causal` is False.
            - Query sequence length is not divisible by 128 (kernel constraint).
        - Non-TPU forward methods (`forward_native`, `forward_gpu`, etc.) are not
          implemented and will raise `NotImplementedError`.

    Registered under the name "blocksparse".
    """

    @classmethod
    def get_impl_name(cls) -> str | tuple[str]:
        """
        Returns the registered name of this attention implementation.

        Returns:
            The string "blocksparse".
        """
        return "blocksparse"

    def get_impl_metadata(self) -> OperationMetadata:
        """
        Returns the metadata associated with this attention implementation instance.

        Returns:
            The `OperationMetadata` provided during initialization.
        """
        assert self.metadata is not None
        return self.metadata

    @classmethod
    def get_requirements(
        cls,
        mode: ExecutionMode = ExecutionMode.MIXED,
    ) -> OperationRequirements:
        """Returns requirements for BlockSparseAttn (Splash Attention).

        BlockSparse/Splash attention requires basic metadata and uses
        TransformerCacheView for KV-cache management.
        """
        return OperationRequirements.create(
            name="splash",
            required_metadata=MetadataField.basic(),
            supported_cache=CacheType.TRANSFORMER | CacheType.HYBRID,
            cache_view_class=TransformerCacheView,
        )

    def forward_native(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_info: MaskInfo | None = None,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        cache_metadata: TransformerMetadata | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Performs Splash Attention on TPU/GPU using the Pallas/Triton kernel.

        Handles fallback logic, attention_mask processing, block size configuration, and
        sharding via `shard_map`. Expects inputs potentially in BTHD format and
        transposes them to BHTD for the kernel.

        Args:
            query: Query tensor (B, T, Hq, D).
            key: Key tensor (B, S, Hkv, D).
            value: Value tensor (B, S, Hkv, Dv).
            attention_mask: Optional boolean attention attention_mask (broadcastable to B, 1, T, S).
                Used to generate segment IDs if provided.
            causal: If True, applies causal masking via the kernel's attention_mask configuration.
                If False, falls back to VanillaAttn.
            **ignore: Ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention outputs. Attention weights
            are not computed or returned by Splash Attention.
        """

        def _run_vanilla_fallback() -> AttentionOutput:
            vanilla_attn = VanillaAttn(self.metadata)
            return vanilla_attn(
                query=query,
                key=key,
                value=value,
                softmax_aux=softmax_aux,
                mask_info=mask_info,
                logits_soft_cap=logits_soft_cap,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                causal=causal,
                cache_metadata=cache_metadata,
                **ignore,
            )

        def _extract_block_size(
            cfg_obj: tp.Any,
            *field_names: str,
        ) -> int | None:
            if cfg_obj is None:
                return None
            for field_name in field_names:
                value = getattr(cfg_obj, field_name, None)
                if value is None:
                    continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
            return None

        query_length: int = query.shape[1]
        key_length: int = key.shape[1]

        # Check dimension compatibility for MLA-style attention
        query_dim: int = query.shape[-1]
        key_dim: int = key.shape[-1]
        value_dim: int = value.shape[-1]
        dims_incompatible: bool = not (query_dim == key_dim == value_dim)

        if dims_incompatible:
            return _run_vanilla_fallback()

        # Check hardware-specific constraints for block sparse attention
        current_backend: str = jax.default_backend()
        is_tpu: bool = current_backend == "tpu"
        is_gpu: bool = current_backend == "gpu"

        tpu_constraints_failed: bool = query_length == 1

        query_dim_mod_16: int = query_dim % 16
        value_dim_mod_16: int = value_dim % 16
        gpu_constraints_failed: bool = query_dim_mod_16 != 0 or value_dim_mod_16 != 0

        blocksparse_cfg = self.metadata.get_operation_config("blocksparse")
        base_cfg = self.metadata.base_config
        q_block_size = _extract_block_size(
            blocksparse_cfg,
            "q_block_size",
            "block_size_q",
            "blocksize_q",
        )
        if q_block_size is None:
            q_block_size = _extract_block_size(
                base_cfg,
                "q_block_size",
                "block_size_q",
                "blocksize_q",
            )
        k_block_size = _extract_block_size(
            blocksparse_cfg,
            "k_block_size",
            "block_size_k",
            "blocksize_k",
        )
        if k_block_size is None:
            k_block_size = _extract_block_size(
                base_cfg,
                "k_block_size",
                "block_size_k",
                "blocksize_k",
            )
        invalid_q_block = q_block_size is not None and q_block_size > 0 and query_length % q_block_size != 0
        invalid_k_block = k_block_size is not None and k_block_size > 0 and key_length % k_block_size != 0
        blockshape_constraints_failed = invalid_q_block or invalid_k_block

        should_fallback: bool = (
            (tpu_constraints_failed and is_tpu) or (gpu_constraints_failed and is_gpu) or blockshape_constraints_failed
        )

        if should_fallback:
            if blockshape_constraints_failed:
                logger.warning(
                    "Falling back to vanilla attention due to incompatible block-sparse shape "
                    "(q_seq_len=%s, k_seq_len=%s, q_block_size=%s, k_block_size=%s).",
                    query_length,
                    key_length,
                    q_block_size,
                    k_block_size,
                )
            return _run_vanilla_fallback()

        head_dim: int = query.shape[-1]
        softmax_scale_computed: float = softmax_scale if softmax_scale is not None else head_dim**-0.5
        dtype: jnp.dtype = self.metadata.runtime_dtype
        model_mode = self.get_mode(query=query, BTHD=False)
        is_decode_mode = model_mode == common_types.MODE_DECODE
        causal_computed: bool = causal if not is_decode_mode else False
        if softmax_aux is not None:
            softmax_aux = softmax_aux.reshape(-1)
        shardings = self.metadata.get_shardings(
            mode=model_mode,
            layout="bhtd",
            qkv_mni_sharding=True,
            softmax_aux=softmax_aux,
        )

        # Transpose from BTHD to BHTD format and cast to runtime dtype
        query_transposed: Float[Array, "batch num_heads seq_len head_dim"] = query.transpose(0, 2, 1, 3).astype(dtype)
        key_transposed: Float[Array, "batch kv_num_heads kv_len head_dim"] = key.transpose(0, 2, 1, 3).astype(dtype)
        value_transposed: Float[Array, "batch kv_num_heads kv_len vhead_dim"] = value.transpose(0, 2, 1, 3).astype(dtype)

        # Create sharding specs
        query_sharding = self.create_stable_sharding(
            shardings.query,
            dep=query_transposed,
            tensor=query_transposed,
            preserved_indices=[0, 1],
        )
        key_sharding = self.create_stable_sharding(
            shardings.key,
            dep=key_transposed,
            tensor=key_transposed,
            preserved_indices=[0, 1],
        )
        value_sharding = self.create_stable_sharding(
            shardings.value,
            dep=value_transposed,
            tensor=value_transposed,
            preserved_indices=[0, 1],
        )
        softmax_aux_sharding = self.create_stable_sharding(
            shardings.softmax_aux,
            dep=softmax_aux,
            tensor=softmax_aux,
        )
        output_sharding = self.create_stable_sharding(
            shardings.output,
            tensor=query_transposed,
            preserved_indices=[0, 1],
        )

        try:
            outputs_bhtd: Float[Array, "batch num_heads seq_len head_dim"] = blocksparse_attention(
                query_transposed,
                key_transposed,
                value_transposed,
                softmax_aux,
                None,
                mask_info=mask_info,
                logits_soft_cap=logits_soft_cap,
                softmax_scale=softmax_scale_computed,
                sliding_window=sliding_window,
                causal=causal_computed,
                fused_backward=fused_backward,
                cfg=self.metadata.get_operation_config("blocksparse"),
                mesh=self.metadata.mesh,
                out_specs=output_sharding,
                in_specs=(
                    query_sharding,
                    key_sharding,
                    value_sharding,
                    softmax_aux_sharding,
                    PartitionSpec(None),
                ),
            )
        except ValueError as exc:
            msg = str(exc)
            if "should divide" in msg and ("q_block_size" in msg or "k_block_size" in msg):
                logger.warning("Falling back to vanilla attention after block-sparse validation error: %s", msg)
                return _run_vanilla_fallback()
            raise

        # Transpose back from BHTD to BTHD format
        outputs: Float[Array, "batch seq_len num_heads head_dim"] = outputs_bhtd.transpose(0, 2, 1, 3)

        result: AttentionOutput = AttentionOutput(attention_weights=None, attention_outputs=outputs)
        return result

    def forward_gpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_tpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cpu(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def forward_rocm(self, *args, **kwargs) -> AttentionOutput:
        """GPU forward pass. Not implemented for Splash Attention.

        Splash Attention is TPU-specific and has no GPU implementation.

        Args:
            *args: Ignored arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            NotImplementedError: Always raised as GPU execution is not supported.
        """
        return self.forward_native(*args, **kwargs)

    def __call__(
        self,
        query: Float[Array, "batch num_heads seq_len head_dim"],
        key: Float[Array, "batch kv_num_heads kv_len head_dim"],
        value: Float[Array, "batch kv_num_heads kv_len vhead_dim"],
        softmax_aux: Float[Array, "num_kv_heads num_sinks"] | Float[Array, "num_sinks"] | None = None,  # noqa
        mask_info: MaskInfo | None = None,
        logits_soft_cap: float | None = None,
        softmax_scale: float | None = None,
        sliding_window: int | tuple[int, int] | None = None,
        causal: bool = True,
        fused_backward: bool = False,
        cache_metadata: TransformerMetadata | None = None,
        **ignore,
    ) -> AttentionOutput:
        """
        Executes the Splash Attention computation or falls back to Vanilla Attention.

        Calls the appropriate backend-specific forward method (`forward_tpu`) via
        `super().__call__`. If the backend is not TPU or fallback conditions are met,
        it relies on the fallback mechanism within `forward_tpu`.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            attention_mask: Optional attention attention_mask.
            causal: If True, applies causal masking. Affects fallback logic and
                kernel configuration.
                        cache_metadata: cache view for current layer.
            **ignore: Additional ignored keyword arguments.

        Returns:
            An `AttentionOutput` object containing the attention results.
        """
        return super().__call__(
            query=query,
            key=key,
            value=value,
            causal=causal,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            softmax_aux=softmax_aux,
            sliding_window=sliding_window,
            logits_soft_cap=logits_soft_cap,
            softmax_scale=softmax_scale,
            fused_backward=fused_backward,
            **ignore,
        )


if __name__ == "__main__":
    from easydel.infra import EasyDeLBaseConfig

    test_cases = [
        # (batch_size, q_seq_len, k_seq_len, q_heads, k_heads)
        (1, 2048, 2048, 8, 8),
        # (2, 2**13, 2**13, 32, 8),
        # (4, 2**14, 2**14, 16, 8),
        # (4, 2**13, 2**14, 16, 4),
    ]

    metadata = OperationMetadata(
        runtime_dtype=jnp.bfloat16,
        base_config=EasyDeLBaseConfig(sharding_axis_dims=(1, 1, 1, 1, -1)),
    )

    splash_attn = BlockSparseAttn(metadata)
    vanilla_attn = VanillaAttn(metadata)

    for idx, (b, qs, ks, qh, kh) in enumerate(test_cases):
        d, vd = 128, 128
        print(
            f"Running test case {idx + 1}/{len(test_cases)}: b={b}, qs={qs}, ks={ks}, qh={qh}, kh={kh}, d={d}, vd={vd}"
        )
        key_q, key_k, key_v = jr.split(jr.PRNGKey(0), 3)

        query = jr.normal(key_q, (b, qs, qh, d), dtype=jnp.bfloat16)
        key = jr.normal(key_k, (b, ks, kh, d), dtype=jnp.bfloat16)
        value = jr.normal(key_v, (b, ks, kh, vd), dtype=jnp.bfloat16)
        mask_info = MaskInfo.from_random(b, qs, ks)
        splash_out = splash_attn(
            query=query,
            key=key,
            value=value,
            mask_info=mask_info,
            causal=False,
            sliding_window=None,
        ).attention_outputs
        vanilla_out = vanilla_attn(
            query=query,
            key=key,
            value=value,
            mask_info=mask_info,
            causal=False,
            sliding_window=None,
        ).attention_outputs
        is_close = jnp.allclose(splash_out, vanilla_out, atol=0.125)
        max_diff = jnp.max(jnp.abs(splash_out - vanilla_out))

        print(f"Test case {idx + 1} result: {'PASS' if is_close else 'FAIL'}")
        print(f"Maximum absolute difference: {max_diff}\n")
