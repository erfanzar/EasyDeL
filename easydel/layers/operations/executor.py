# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Mode-bound operation executor for dynamic discovery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from easydel.layers.operations.requirements import ExecutionMode, OperationRequirements

if TYPE_CHECKING:
    from easydel.layers.operations._base_operation import BaseOperation

__all__ = ["OperationExecutor"]


@dataclass
class OperationExecutor:
    """Mode-bound operation executor for dynamic discovery.

    This class wraps prefill and decode operations, making it easy to:
    1. Discover operations via iter_module_search
    2. Get the right operation for a given execution mode
    3. Combine requirements from both operations

    Args:
        prefill_impl: Operation for prefill mode (required if decode_impl is None)
        decode_impl: Operation for decode mode (falls back to prefill_impl if None)
        mixin_impl: Shared operation for both modes (used if prefill/decode are None)

    Logic:
        - If prefill_impl is set and decode_impl is None: decode uses prefill_impl
        - If both prefill_impl and decode_impl are None but mixin_impl is set: both use mixin_impl
        - prefill_impl and decode_impl take precedence over mixin_impl

    Example:
        >>> # Create from FlexibleAttentionModule
        >>> executor = OperationExecutor.from_flexible_attention(flex_attn)
        >>> # Get operation for specific mode
        >>> prefill_op = executor.get_operation(ExecutionMode.PREFILL)
        >>> decode_op = executor.get_operation(ExecutionMode.DECODE)
        >>> # Get combined requirements
        >>> reqs = executor.get_combined_requirements()
    """

    prefill_impl: BaseOperation | None = None
    decode_impl: BaseOperation | None = None
    mixin_impl: BaseOperation | None = None

    @property
    def prefill_operation(self) -> BaseOperation | None:
        """Get the operation for prefill mode."""
        if self.prefill_impl is not None:
            return self.prefill_impl
        return self.mixin_impl

    @property
    def decode_operation(self) -> BaseOperation | None:
        """Get the operation for decode mode."""
        if self.decode_impl is not None:
            return self.decode_impl
        if self.prefill_impl is not None:
            return self.prefill_impl  # Fallback to prefill if decode not set
        return self.mixin_impl

    def get_operation(self, mode: ExecutionMode) -> BaseOperation | None:
        """Get operation for a specific execution mode.

        Args:
            mode: The execution mode (PREFILL, DECODE, or MIXED).

        Returns:
            The operation for that mode, or None if not available.
        """
        if mode == ExecutionMode.PREFILL:
            return self.prefill_operation
        elif mode == ExecutionMode.DECODE:
            return self.decode_operation
        else:  # MIXED
            return self.prefill_operation  # Default to prefill for mixed

    def get_requirements(self, mode: ExecutionMode = ExecutionMode.MIXED) -> OperationRequirements:
        """Get requirements for the specified mode.

        Uses get_instance_requirements() to respect instance-level overrides
        (e.g., requires_cache=False for vision encoders).

        Args:
            mode: The execution mode.

        Returns:
            Requirements for the operation in that mode.
        """
        op = self.get_operation(mode)
        if op is not None:
            # Use instance requirements to respect metadata overrides
            if hasattr(op, "get_instance_requirements"):
                return op.get_instance_requirements(mode)
            return op.get_requirements(mode)
        return OperationRequirements.default()

    def get_combined_requirements(self) -> OperationRequirements:
        """Get combined requirements from both prefill and decode operations.

        Uses get_instance_requirements() to respect instance-level overrides
        (e.g., requires_cache=False for vision encoders).

        Returns:
            Combined requirements (intersection of cache support, union of metadata).
        """
        prefill_reqs = None
        decode_reqs = None

        prefill_op = self.prefill_operation
        decode_op = self.decode_operation

        if prefill_op is not None:
            # Use instance requirements to respect metadata overrides
            if hasattr(prefill_op, "get_instance_requirements"):
                prefill_reqs = prefill_op.get_instance_requirements(ExecutionMode.PREFILL)
            else:
                prefill_reqs = prefill_op.get_requirements(ExecutionMode.PREFILL)
        if decode_op is not None:
            # Use instance requirements to respect metadata overrides
            if hasattr(decode_op, "get_instance_requirements"):
                decode_reqs = decode_op.get_instance_requirements(ExecutionMode.DECODE)
            else:
                decode_reqs = decode_op.get_requirements(ExecutionMode.DECODE)

        if prefill_reqs is None and decode_reqs is None:
            return OperationRequirements.default()
        if prefill_reqs is None:
            return decode_reqs
        if decode_reqs is None:
            return prefill_reqs

        # Combine requirements (intersection of cache, union of metadata)
        return prefill_reqs | decode_reqs

    @property
    def requires_cache(self) -> bool:
        """Whether any operation requires cache."""
        reqs = self.get_combined_requirements()
        return reqs.cache.requires_cache

    @property
    def has_separate_decode(self) -> bool:
        """Whether decode uses a different operation than prefill."""
        return (
            self.decode_impl is not None and self.prefill_impl is not None and self.decode_impl is not self.prefill_impl
        )

    @property
    def is_valid(self) -> bool:
        """Whether at least one operation is available."""
        return self.prefill_impl is not None or self.decode_impl is not None or self.mixin_impl is not None

    def get_operation_name(self, mode: ExecutionMode = ExecutionMode.MIXED) -> str | None:
        """Get the name of the operation for a specific mode.

        Args:
            mode: The execution mode.

        Returns:
            The operation name, or None if no operation available.
        """
        op = self.get_operation(mode)
        if op is not None:
            name = op.get_impl_name()
            if isinstance(name, tuple):
                return name[0]
            return name
        return None

    @classmethod
    def from_flexible_attention(cls, flex_attn) -> OperationExecutor:
        """Create from a FlexibleAttentionModule instance.

        Args:
            flex_attn: A FlexibleAttentionModule instance.

        Returns:
            An OperationExecutor wrapping the module's operations.
        """
        return cls(
            prefill_impl=getattr(flex_attn, "impl", None),
            decode_impl=getattr(flex_attn, "impl_decode", None),
            mixin_impl=None,
        )

    @classmethod
    def from_operations(
        cls,
        prefill: BaseOperation | None = None,
        decode: BaseOperation | None = None,
        mixin: BaseOperation | None = None,
    ) -> OperationExecutor:
        """Create from individual operation instances.

        Args:
            prefill: Operation for prefill mode.
            decode: Operation for decode mode.
            mixin: Shared operation for both modes.

        Returns:
            An OperationExecutor wrapping the operations.
        """
        return cls(
            prefill_impl=prefill,
            decode_impl=decode,
            mixin_impl=mixin,
        )
