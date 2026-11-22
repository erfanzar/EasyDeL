# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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


"""Sentinel types and reference wrappers.

This module provides a wrapper around Ray ObjectRefs to prevent automatic
dereferencing, and a sentinel class for signaling completion states.

Attributes:
    DONE: Global singleton instance of DoneSentinel used to signal completion.
"""

from __future__ import annotations

from dataclasses import dataclass

import ray


@dataclass
class RefBox:
    """Wrapper to prevent automatic ObjectRef dereferencing in Ray.

    Ray automatically dereferences ObjectRefs when they are passed as arguments
    to remote functions, but this doesn't happen when they're nested inside other
    objects. RefBox takes advantage of this behavior to control when dereferencing
    occurs, which can be useful for lazy evaluation or passing references between
    actors without triggering computation.

    Attributes:
        ref: The Ray ObjectRef to be wrapped.

    Example:
        >>>
        >>> result_ref = expensive_computation.remote()
        >>> boxed = RefBox(result_ref)
        >>> another_task.remote(boxed)
        >>>
        >>>
        >>> actual_result = boxed.get()

    See Also:
        Ray documentation on object passing:
        https://docs.ray.io/en/latest/ray-core/objects.html
    """

    ref: ray.ObjectRef

    def get(self):
        """Dereference the wrapped ObjectRef and return its value.

        Returns:
            The actual value stored in the ObjectRef.

        Raises:
            Any exception that occurred during the computation of the ObjectRef.

        Example:
            >>> computation_ref = expensive_task.remote()
            >>> box = RefBox(computation_ref)
            >>> result = box.get()
        """
        return ray.get(self.ref)


class DoneSentinel:
    """Sentinel class to indicate completion or termination state.

    This class serves as a unique marker object to signal that a process,
    computation, or data stream has reached its end. Using a sentinel class
    instead of None or other values prevents ambiguity when None might be
    a valid result.

    Example:
        >>> def process_items(items):
        ...     for item in items:
        ...         if item is DONE:
        ...             break
        ...         yield process_item(item)
    """

    pass


DONE = DoneSentinel()
