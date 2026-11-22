"""Pure-Python compatibility contracts for relocating the shared chat converter."""

import copy
from types import SimpleNamespace

import pytest
from easydel.trainers import prompt_utils as trainer_prompt_utils
from easydel.trainers.prompt_utils import convert_to_openai_format, reverse_openai_format
from easydel.utils import prompt_utils as shared_prompt_utils

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    "name",
    [
        "InputDict",
        "InputListDict",
        "InputListListDict",
        "InputType",
        "OpenAIMessageContentPart",
        "OpenAIMessage",
        "OpenAIMessageList",
        "_is_valid_openai_message_list",
        "_convert_single_dict",
        "convert_to_openai_format",
    ],
)
def test_trainer_exports_are_shared_compatibility_aliases(name):
    assert getattr(trainer_prompt_utils, name) is getattr(shared_prompt_utils, name)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (
            {"ROLE": "Assistant", "TEXT": "hello"},
            [{"role": "assistant", "content": [{"type": "text", "text": "hello"}]}],
        ),
        (
            [[{"content": "first"}], [{"role": "assistant", "message": "second"}]],
            [
                {"role": "user", "content": [{"type": "text", "text": "first"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
            ],
        ),
        (
            {"message": "third", "text": "second", "content": "first"},
            [{"role": "user", "content": [{"type": "text", "text": "first"}]}],
        ),
        (
            [{"text": "first"}, [{"role": "assistant", "message": "second"}, None], "ignored"],
            [
                {"role": "user", "content": [{"type": "text", "text": "first"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
            ],
        ),
        (None, []),
        ("unsupported", []),
        ([], []),
    ],
)
def test_public_conversion_outputs_and_input_preservation(payload, expected):
    original = copy.deepcopy(payload)
    assert convert_to_openai_format(payload) == expected
    assert payload == original


def test_already_formatted_multimodal_messages_preserve_parts_and_metadata():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,example"}},
            ],
            "name": "viewer",
        }
    ]
    original = copy.deepcopy(messages)
    assert convert_to_openai_format(messages) is messages
    assert messages == original


def test_reverse_conversion_remains_available_with_custom_content_key():
    messages = convert_to_openai_format({"role": "assistant", "text": "hello"})
    assert reverse_openai_format(messages, content_key_name="text") == {"role": "assistant", "text": "hello"}


@pytest.mark.parametrize(
    ("use_processor", "oai_like_processor", "content", "expected_content"),
    [
        (True, True, "hello", [{"type": "text", "text": "hello"}]),
        (True, False, "hello", "hello"),
        (False, True, "hello", "hello"),
        (
            True,
            True,
            [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,example"}},
            ],
            [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,example"}},
            ],
        ),
    ],
)
def test_server_processor_conversion_preserves_compatibility(
    use_processor, oai_like_processor, content, expected_content
):
    from easydel.inference.esurge.server.api_server import eSurgeApiServer
    from easydel.inference.openai_api_modules import ChatCompletionRequest
    from transformers import ProcessorMixin

    # Only message preparation is exercised: no engine, model, device or server startup.
    server = object.__new__(eSurgeApiServer)
    server.oai_like_processor = oai_like_processor
    processor = object.__new__(ProcessorMixin) if use_processor else object()
    request = ChatCompletionRequest(model="test", messages=[{"role": "user", "content": content}])
    original = request.model_dump()

    messages = server._prepare_messages_for_engine(request, SimpleNamespace(tokenizer=processor))

    assert messages == [{"role": "user", "content": expected_content}]
    assert request.model_dump() == original
