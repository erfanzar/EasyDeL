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

"""Utility functions for the scheduler module.

This module provides helper functions used by the scheduler for checking
stop conditions and managing request state transitions.

Functions:
    check_stop: Check if a request should stop generating tokens.

Example:
    >>> from easydel.inference.esurge.scheduler.utils import check_stop
    >>> stopped = check_stop(request, max_model_len=8192)
    >>> if stopped:
    ...     print(f"Request stopped with status: {request.status}")
"""

from ..request import EngineRequest, EngineRequestStatus


def check_stop(request: EngineRequest, max_model_len: int) -> bool:
    """Check if a request should stop generating tokens.

    This function evaluates various stopping conditions for token generation
    including length limits, EOS tokens, and custom stop tokens.

    The function checks conditions in the following order:
        1. Maximum model length exceeded
        2. Maximum output tokens exceeded
        3. EOS token generated (if not ignored)
        4. Custom stop token generated

    Args:
        request: The engine request to check for stop conditions.
        max_model_len: The maximum model sequence length allowed.

    Returns:
        bool: True if the request should stop generating tokens, False otherwise.

    Side Effects:
        - Updates ``request.status`` to the appropriate finished status:
            - ``FINISHED_LENGTH_CAPPED``: If length limits are exceeded
            - ``FINISHED_STOPPED``: If EOS or stop token is generated
        - Sets ``request.stop_reason`` to the stop token ID if a custom
          stop token triggered the stop.

    Example:
        >>> request = EngineRequest(...)
        >>> # After generating some tokens
        >>> stopped = check_stop(request, max_model_len=8192)
        >>> if stopped:
        ...     print(f"Request finished: {request.status}")

    Note:
        The request must have ``sampling_params`` set and at least one
        output token generated before calling this function.
    """
    if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
        request.status = EngineRequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    assert sampling_params is not None
    last_token_id = request.output_token_ids[-1]
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = EngineRequestStatus.FINISHED_STOPPED
        return True

    # Respect ignore_eos even when EOS ids were injected into stop_token_ids.
    is_ignored_eos = bool(
        sampling_params.ignore_eos and request.eos_token_id is not None and last_token_id == request.eos_token_id
    )
    if not is_ignored_eos and last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = EngineRequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False
