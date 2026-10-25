"""CPU-only compatibility and dependency-direction checks for host side inputs."""

import ast
import inspect
import pickle
import typing as tp
from pathlib import Path
from types import SimpleNamespace

import easydel
import numpy as np
import pytest
from easydel.inference.esurge.multimodal import side_inputs as legacy
from easydel.infra import side_inputs as canonical
from easydel.layers.engram import EngramModelMixin


@pytest.mark.cpu_ok
@pytest.mark.parametrize("name", ["SideInputSpec", "SideInputWindow", "SideInputProvider"])
def test_legacy_exports_are_identical(name):
    assert getattr(legacy, name) is getattr(canonical, name)
    assert name in legacy.__all__
    assert name in canonical.__all__


@pytest.mark.cpu_ok
def test_named_tuple_and_protocol_contracts():
    spec = canonical.SideInputSpec(per_token_shape=(2, 3), dtype=np.int32)
    assert isinstance(spec, tuple)
    assert spec._asdict() == {"per_token_shape": (2, 3), "dtype": np.int32, "fill": 0}
    assert spec._replace(fill=-1) == ((2, 3), np.int32, -1)
    assert pickle.loads(pickle.dumps(spec)) == spec

    token_ids = np.asarray([3, 5], dtype=np.int32)
    window = canonical.SideInputWindow(token_ids=token_ids, token_types=None, start=0, end=2)
    assert isinstance(window, tuple)
    assert window._fields == ("token_ids", "token_types", "start", "end", "num_tokens")
    assert window.token_ids is token_ids
    assert window.token_types is None
    assert (window.start, window.end, window.num_tokens) == (0, 2, -1)
    assert window._replace(num_tokens=2).num_tokens == 2

    assert tp.Protocol in canonical.SideInputProvider.__bases__
    assert tuple(inspect.signature(canonical.SideInputProvider.build).parameters) == ("self", "windows", "num_tokens")
    assert tp.get_type_hints(canonical.SideInputProvider.build) == {
        "windows": tp.Sequence[canonical.SideInputWindow],
        "num_tokens": int,
        "return": dict[str, np.ndarray],
    }


@pytest.mark.cpu_ok
def test_engram_declares_canonical_specs():
    model = SimpleNamespace(
        config=SimpleNamespace(engram_layer_ids=[1, 3], engram_max_ngram_size=3, engram_n_heads=4),
        engram_uses_host_tables=lambda: False,
    )
    specs = EngramModelMixin.esurge_side_input_specs(model)
    assert specs == {"engram_hash_ids": canonical.SideInputSpec((2, 8), np.int32)}
    assert type(specs["engram_hash_ids"]) is canonical.SideInputSpec


@pytest.mark.cpu_ok
def test_layers_do_not_import_inference_side_input_contracts():
    layers = Path(easydel.__file__).parent / "layers"
    contracts = {"SideInputSpec", "SideInputWindow", "SideInputProvider"}
    violations = []
    for path in sorted(layers.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Inspect TYPE_CHECKING and local imports too, including relative imports back to inference.
                if "inference" in module.split(".") and (
                    module.endswith("side_inputs") or any(alias.name in contracts | {"*"} for alias in node.names)
                ):
                    violations.append(f"{path.relative_to(layers)}:{node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if "inference" in alias.name.split(".") and alias.name.endswith("side_inputs"):
                        violations.append(f"{path.relative_to(layers)}:{node.lineno}")
    assert not violations, f"Layers depend on serving side-input contracts: {violations}"
