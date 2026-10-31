"""Processor-path message conversion must not import training code."""

import builtins
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import ChatCompletionRequest
from transformers import ProcessorMixin

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("content", ["hello", [{"type": "text", "text": "hello"}, {"type": "image", "image": "inline"}]])
def test_processor_path_uses_neutral_converter_without_training_import(monkeypatch, content):
    server = object.__new__(eSurgeApiServer)
    server.oai_like_processor = True
    processor = object.__new__(ProcessorMixin)
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": content}])
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("easydel.trainers"):
            raise AssertionError("Serving processor conversion must not depend on trainers")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    actual = server._prepare_messages_for_engine(request, SimpleNamespace(tokenizer=processor))
    expected_content = [{"type": "text", "text": content}] if isinstance(content, str) else content
    assert actual == [{"role": "user", "content": expected_content}]
    assert request.messages[0].content == content


@pytest.mark.parametrize("processor,enabled", [(object.__new__(ProcessorMixin), False), (object(), True)])
def test_conversion_opt_in_preserves_plain_tokenizer_and_disabled_paths(processor, enabled):
    server = object.__new__(eSurgeApiServer)
    server.oai_like_processor = enabled
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hello"}])
    assert server._prepare_messages_for_engine(request, SimpleNamespace(tokenizer=processor)) == [
        {"role": "user", "content": "hello"}
    ]
