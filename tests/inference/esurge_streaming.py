"""Test eSurge streaming responses."""

import json

from easydel.inference.esurge.server.api_server import (
    ChatCompletionStreamResponse,
    ChatCompletionStreamResponseChoice,
    CompletionStreamResponse,
    CompletionStreamResponseChoice,
    DeltaMessage,
    UsageInfo,
)


def test_chat_streaming_format():
    """Test that chat streaming responses serialize correctly."""
    print("Testing Chat Streaming Format")
    print("-" * 40)

    # Test initial chunk with role (with empty usage)
    initial_chunk = ChatCompletionStreamResponse(
        model="test-model",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )

    # This should not raise validation errors
    initial_json = initial_chunk.model_dump_json(exclude_unset=True, exclude_none=True)
    print("Initial chunk (with role):")
    print(f"  {initial_json}")

    # Test content chunk (with empty usage)
    content_chunk = ChatCompletionStreamResponse(
        model="test-model",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(content="Hello"),
                finish_reason=None,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )

    content_json = content_chunk.model_dump_json(exclude_unset=True, exclude_none=True)
    print("\nContent chunk:")
    print(f"  {content_json}")

    # Test final chunk with usage
    usage = UsageInfo(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        tokens_per_second=2.5,
        processing_time=2.0,
    )

    final_chunk = ChatCompletionStreamResponse(
        model="test-model",
        choices=[
            ChatCompletionStreamResponseChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    final_json = final_chunk.model_dump_json(exclude_unset=True)
    print("\nFinal chunk (with usage):")
    print(f"  {final_json}")

    # Verify JSON is valid
    try:
        json.loads(initial_json)
        json.loads(content_json)
        json.loads(final_json)
        print("\n✅ All chat streaming chunks are valid JSON")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON validation failed: {e}")
        return False

    return True


def test_completion_streaming_format():
    """Test that completion streaming responses serialize correctly."""
    print("\nTesting Completion Streaming Format")
    print("-" * 40)

    # Test content chunk (with empty usage)
    content_chunk = CompletionStreamResponse(
        model="test-model",
        choices=[
            CompletionStreamResponseChoice(
                index=0,
                text="Hello world",
                finish_reason=None,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
        ),
    )

    content_json = content_chunk.model_dump_json(exclude_unset=True, exclude_none=True)
    print("Content chunk:")
    print(f"  {content_json}")

    # Test final chunk with usage
    usage = UsageInfo(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        tokens_per_second=2.5,
        processing_time=2.0,
    )

    final_chunk = CompletionStreamResponse(
        model="test-model",
        choices=[
            CompletionStreamResponseChoice(
                index=0,
                text="",
                finish_reason="stop",
            )
        ],
        usage=usage,
    )

    final_json = final_chunk.model_dump_json(exclude_unset=True)
    print("\nFinal chunk (with usage):")
    print(f"  {final_json}")

    # Verify JSON is valid
    try:
        json.loads(content_json)
        json.loads(final_json)
        print("\n✅ All completion streaming chunks are valid JSON")
    except json.JSONDecodeError as e:
        print(f"\n❌ JSON validation failed: {e}")
        return False

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("eSurge Streaming Response Format Tests")
    print("=" * 60)

    chat_passed = test_chat_streaming_format()
    completion_passed = test_completion_streaming_format()

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    if chat_passed and completion_passed:
        print("✅ All streaming format tests passed!")
        print("\nThe streaming responses now properly use Pydantic models")
        print("and match the vSurge/OAI server pattern.")
    else:
        print("❌ Some tests failed")

    return chat_passed and completion_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
