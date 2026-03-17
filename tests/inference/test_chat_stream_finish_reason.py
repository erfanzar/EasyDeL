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

from easydel.inference.openai_api_modules import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    DeltaMessage,
    UsageInfo,
)


def test_chat_stream_finish_reason_tool_calls_is_accepted():
    chunk = ChatCompletionStreamResponse(
        model="test-model",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(role="assistant", content=""),
                finish_reason="tool_calls",
            )
        ],
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )
    payload = chunk.model_dump(exclude_none=True)
    assert payload["choices"][0]["finish_reason"] == "tool_calls"
