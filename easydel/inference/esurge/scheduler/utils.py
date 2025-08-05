# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from ..request import EngineRequest, EngineRequestStatus


def check_stop(request: EngineRequest, max_model_len: int) -> bool:
    if request.num_tokens >= max_model_len or request.num_output_tokens >= request.max_tokens:
        request.status = EngineRequestStatus.FINISHED_LENGTH_CAPPED
        return True

    sampling_params = request.sampling_params
    assert sampling_params is not None
    last_token_id = request.output_token_ids[-1]
    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        request.status = EngineRequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        request.status = EngineRequestStatus.FINISHED_STOPPED
        request.stop_reason = last_token_id
        return True
    return False
