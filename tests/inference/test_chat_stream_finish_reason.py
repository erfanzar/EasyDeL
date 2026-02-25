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
