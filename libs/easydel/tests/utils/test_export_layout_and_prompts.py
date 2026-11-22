"""Non-computation regressions for precise export paths and shared prompt conversion."""

import copy

import jax
import pytest
import spectrax as spx
from easydel.layers import Embed
from easydel.utils.parameters_transformation import StateDictConverter, TensorConverter
from easydel.utils.prompt_utils import convert_to_openai_format
from jax import numpy as jnp

from .test_transfer_layout_compat import DeclaredNorm, LayoutTensor, PlainLeaf

pytestmark = pytest.mark.cpu_ok


class DeclaredEmbed(Embed):
    def __init__(self, tensor):
        self.weight = spx.Parameter(tensor)


class ExportState:
    @property
    def parameters(self):
        return spx.export(self)[1]


@pytest.mark.parametrize("preserved_type", [DeclaredNorm, DeclaredEmbed])
@pytest.mark.parametrize("root", [False, True])
@pytest.mark.parametrize("shape", [(2, 3), (2, 2)])
def test_discovered_layout_paths_do_not_exempt_dense_neighbors(monkeypatch, preserved_type, root, shape):
    source = LayoutTensor([[f"r{row}c{col}" for col in range(shape[1])] for row in range(shape[0])])
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(TensorConverter, "jax_to_pytorch", lambda value: value)

    class RootPreserved(ExportState, preserved_type):
        def __init__(self):
            super().__init__(source)
            self.projection = PlainLeaf(source)

    class Container(ExportState, spx.Module):
        def __init__(self):
            self.norm = preserved_type(source)
            self.norm_projection = PlainLeaf(source)
            self.renorm = PlainLeaf(source)
            # The same path suffix on a separate branch is not the same owner.
            self.nested = spx.Module()
            self.nested.norm = PlainLeaf(source)

    model = RootPreserved() if root else Container()
    exported = StateDictConverter.easydel_to_torch(model, dtype=jnp.float32)
    preserved_key = "weight" if root else "norm.weight"
    assert exported[preserved_key].rows == source.rows
    dense_keys = {"projection.weight"} if root else {"norm_projection.weight", "renorm.weight", "nested.norm.weight"}
    assert set(exported) == dense_keys | {preserved_key}
    for key in dense_keys:
        assert exported[key].rows == tuple(zip(*source.rows, strict=True))


def test_explicit_norm_declarations_keep_substring_contract(monkeypatch):
    source = LayoutTensor([["a", "b", "c"], ["d", "e", "f"]])
    monkeypatch.setattr(jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(TensorConverter, "jax_to_pytorch", lambda value: value)

    class Container(ExportState, spx.Module):
        def __init__(self):
            self.custom_norm_matrix = PlainLeaf(source)
            self.projection = PlainLeaf(source)

    exported = StateDictConverter.easydel_to_torch(Container(), dtype=jnp.float32, layernorm_names=["norm"])
    assert exported["custom_norm_matrix.weight"].rows == source.rows
    assert exported["projection.weight"].rows == tuple(zip(*source.rows, strict=True))


@pytest.mark.parametrize(
    ("source", "role", "text"),
    [
        ({"ROLE": "ASSISTANT", "MESSAGE": "last", "TEXT": "middle", "CONTENT": "first"}, "assistant", "first"),
        ({"content": None, "Text": "fallback"}, "user", "fallback"),
        ({"role": 17, "message": "hello"}, "user", "hello"),
        ({"role": "CUSTOM", "content": ""}, "custom", ""),
        ({}, "user", ""),
    ],
)
def test_shared_prompt_conversion_preserves_role_and_content_precedence(source, role, text):
    original = copy.deepcopy(source)
    assert convert_to_openai_format(source) == [{"role": role, "content": [{"type": "text", "text": text}]}]
    assert source == original


@pytest.mark.parametrize("nested", [False, True])
def test_shared_prompt_conversion_flattens_supported_nested_inputs(nested):
    first = {"text": "first"}
    second = {"role": "assistant", "message": "second"}
    source = [[first], [None, second]] if nested else [first, [second], 3]
    assert convert_to_openai_format(source) == [
        {"role": "user", "content": [{"type": "text", "text": "first"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "second"}]},
    ]


@pytest.mark.parametrize("source", [None, 3, "hello", (), [None]])
def test_shared_prompt_conversion_unsupported_inputs_are_empty(source):
    assert convert_to_openai_format(source) == []


def test_shared_prompt_conversion_passes_multimodal_messages_through_by_identity():
    source = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
            ],
            "name": "viewer",
        }
    ]
    assert convert_to_openai_format(source) is source
    empty = []
    assert convert_to_openai_format(empty) is empty
