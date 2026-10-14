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


"""Global registry for tracking kernel invocations across devices.

This module provides a simple in-memory registry that tracks all kernel
invocations organized by device, operation, and call signature. This
information is used for batch autotuning and cross-run optimization analysis.

Registry Structure:
    device_fingerprint -> operation_id -> call_key -> (kernel, args, kwargs)

The registry enables:
    - Batch autotuning of all recorded operations
    - Analysis of operation patterns and frequencies
    - Cross-device optimization comparison
    - Debugging and profiling of kernel usage

Example Usage:
    >>>
    >>> record_invocation('gpu|cuda_12.0', 'matmul@v1', 'abc123', kernel, args, kwargs)
    >>>
    >>>
    >>> invocations = get_invocations('gpu|cuda_12.0')
    >>> for op_id, calls in invocations.items():
    ...     print(f"Operation {op_id} has {len(calls)} recorded calls")

Note:
    The registry is stored in memory and does not persist across program runs.
    For persistent optimization data, use the PersistentCache system.
"""

from __future__ import annotations

from typing import Any

_REG: dict[str, dict[str, dict[str, tuple[Any, tuple[Any, ...], dict[str, Any]]]]] = {}


def record_invocation(dev: str, op_id_v: str, call_key: str, kernel, args, kwargs):
    """Record a kernel invocation in the global registry.

    Stores the kernel instance and its arguments for later analysis or
    batch autotuning. The invocation is indexed by device, operation ID,
    and call signature hash.

    Args:
        dev: Device fingerprint (e.g., 'gpu|cuda_12.0')
        op_id_v: Operation identifier with version (e.g., 'matmul@v1')
        call_key: Call signature hash from Invocation.call_key
        kernel: Kernel instance that was invoked
        args: Positional arguments used in the call
        kwargs: Keyword arguments used in the call

    Note:
        Arguments are copied (tuple/dict) to prevent mutation of stored data.
        This function is typically called automatically by the execution system.
    """
    _REG.setdefault(dev, {}).setdefault(op_id_v, {})[call_key] = (kernel, tuple(args), dict(kwargs))


def get_invocations(dev: str):
    """Retrieve all recorded invocations for a specific device.

    Returns all kernel invocations that have been recorded for the given
    device, organized by operation ID and call signature.

    Args:
        dev: Device fingerprint to query

    Returns:
        Dictionary mapping operation_id -> call_key -> (kernel, args, kwargs)
        Returns empty dict if no invocations recorded for this device

    Example:
        >>> invocations = get_invocations('gpu|cuda_12.0')
        >>> for op_id, calls in invocations.items():
        ...     for call_key, (kernel, args, kwargs) in calls.items():
        ...         print(f"Found {op_id} call {call_key[:8]}...")
    """
    return _REG.get(dev, {})
