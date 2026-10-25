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

"""Process-wide lock serialising HF's Jinja ``AssistantTracker``.

``transformers`` compiles each chat template through an ``lru_cache`` keyed on the template string, and
the Jinja ``AssistantTracker`` extension belongs to that compiled template. Its mutable state is shared by
every render of the same template in the process and is not thread-safe. It is engaged only by
``return_assistant_tokens_mask=True``. Concurrent renders fail in one of three ways:

- ``ValueError: AssistantTracker should not be reused before closed`` (tracker entered twice),
- ``AttributeError: 'NoneType' object has no attribute 'append'`` (tracker never opened), or
- a silent, empty assistant mask with no exception at all.

One lock covers all templates. Renders of different templates hold different trackers and cannot race, so
this serialises slightly more than strictly required, but keying a lock on the template string would have to
track ``lru_cache`` eviction to stay correct. Renders that do not request the assistant mask never take it.

The lock is re-entrant so that a caller already holding it may call a render site that takes it again on the
same thread. It is not picklable, so never store it on an object that is sent to worker processes: call
:func:`assistant_tracker_lock` at the render site instead.
"""

from __future__ import annotations

import threading

__all__ = ("assistant_tracker_lock",)

_ASSISTANT_TRACKER_LOCK = threading.RLock()


def assistant_tracker_lock() -> threading.RLock:
    """Return the lock guarding assistant-mask chat-template renders in this process.

    Returns:
        The single module-level re-entrant lock.
    """
    return _ASSISTANT_TRACKER_LOCK
