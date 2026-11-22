"""Metadata-only regressions for mixed messages, raw export layouts and context isolation."""

import copy
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jax
import pytest
import spectrax as spx
from easydel.utils.jit_context import (
    CompilationContext,
    clear_jit_context,
    get_jit_context,
    jit_context,
    peek_jit_context,
)
from easydel.utils.parameters_transformation import StateDictConverter, TensorConverter
from easydel.utils.prompt_utils import convert_to_openai_format
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


def mixed_conversation():
    return [
        {"role": "system", "content": "You are helpful."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe these."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,eA=="}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,eA=="}},
            ],
            "name": "viewer",
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
            "reasoning_content": "Need a lookup.",
        },
        {"role": "tool", "content": "result", "tool_call_id": "call_1", "name": "lookup"},
    ]


def test_mixed_messages_preserve_media_and_tool_association_without_mutation():
    source = mixed_conversation()
    original = copy.deepcopy(source)
    converted = convert_to_openai_format(source)
    assert converted[0] == {"role": "system", "content": [{"type": "text", "text": "You are helpful."}]}
    assert converted[1:3] == source[1:3]
    assert converted[3] == {**source[3], "content": [{"type": "text", "text": "result"}]}
    assert source == original
    converted[1]["content"][1]["image_url"]["url"] = "changed"
    assert source == original


def test_tool_call_only_message_preserves_metadata_when_content_is_omitted():
    source = mixed_conversation()[2]
    source.pop("content")
    converted = convert_to_openai_format(source)[0]
    assert converted["tool_calls"] == source["tool_calls"]
    assert converted["reasoning_content"] == source["reasoning_content"]
    assert converted["content"] is None


def test_processor_serving_conversion_preserves_mixed_conversation():
    from easydel.inference.esurge.server.api_server import eSurgeApiServer
    from easydel.inference.openai_api_modules import ChatCompletionRequest
    from transformers import ProcessorMixin

    # No processor/model initialization or serving runtime is needed for message preparation.
    processor = object.__new__(ProcessorMixin)
    request = ChatCompletionRequest(model="test", messages=mixed_conversation())
    server = SimpleNamespace(oai_like_processor=True)
    converted = eSurgeApiServer._prepare_messages_for_engine(server, request, SimpleNamespace(tokenizer=processor))
    assert converted[1]["content"] == mixed_conversation()[1]["content"]
    assert converted[2]["tool_calls"] == mixed_conversation()[2]["tool_calls"]
    assert converted[3]["tool_call_id"] == "call_1"
    assert converted[3]["name"] == "lookup"


class ShapeTensor:
    """Tensor metadata only: layout transformations never execute array operations."""

    dtype = jnp.float32

    def __init__(self, shape):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)

    def permute(self, *axes):
        return ShapeTensor(tuple(self.shape[axis] for axis in axes))


def test_internvl_raw_parameter_import_export_keeps_hf_layout(monkeypatch):
    from easydel.modules.internvl.modeling_internvl import InternVLVisionEmbeddings

    source = {
        "cls_token": ShapeTensor((1, 1, 8)),
        "mask_token": ShapeTensor((1, 1, 8)),
        "position_embeddings": ShapeTensor((1, 5, 8)),
        "projection.weight": ShapeTensor((7, 8)),
        "projection.bias": ShapeTensor((2, 3)),
    }
    monkeypatch.setattr(jax, "local_device_count", lambda: 0)
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(TensorConverter, "convert_pytorch_to_jnp", lambda value, dtype: value)
    monkeypatch.setattr(TensorConverter, "jax_to_pytorch", lambda value: value)
    converted = StateDictConverter.huggingface_to_easydel(source, dtype=jnp.float32, verbose=False)

    class MetadataEmbeddings(InternVLVisionEmbeddings):
        def __init__(self):
            for name in ("cls_token", "mask_token", "position_embeddings"):
                setattr(self, name, spx.Parameter(converted[name]))
            self.projection = spx.Module()
            self.projection.weight = spx.Parameter(converted["projection"]["weight"])
            self.projection.bias = spx.Parameter(converted["projection"]["bias"])

        @property
        def parameters(self):
            return spx.export(self)[1]

    exported = StateDictConverter.easydel_to_torch(MetadataEmbeddings(), dtype=jnp.float32)
    assert set(exported) == set(source)
    assert {key: tensor.shape for key, tensor in exported.items()} == {
        key: tensor.shape for key, tensor in source.items()
    }
    target_shapes = {key: tensor.shape for key, tensor in source.items()}
    reconciled = StateDictConverter.reconcile_to_target_shapes(exported, target_shapes)
    assert {key: tensor.shape for key, tensor in reconciled.items()} == target_shapes


def test_other_thread_clear_does_not_invalidate_nested_context_restoration():
    before = peek_jit_context()
    outer = CompilationContext("outer")
    inner = CompilationContext("inner")
    with jit_context(outer):
        with jit_context(inner):
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(clear_jit_context).result(timeout=10)
            assert get_jit_context() == inner
        assert get_jit_context() == outer
    assert peek_jit_context() == before


def test_same_context_clear_prevents_enclosing_scope_restoration():
    with jit_context(CompilationContext("outer")):
        with jit_context(CompilationContext("inner")):
            clear_jit_context()
            assert peek_jit_context() is None
        assert peek_jit_context() is None
    assert peek_jit_context() is None
