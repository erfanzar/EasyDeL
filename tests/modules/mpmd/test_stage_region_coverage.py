"""Lightweight MPMD coverage checks for module registrations and regions."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from jax.sharding import Mesh
from spectrax.runtime.mpmd import sxjit
from spectrax.runtime.mpmd.markers import stage_region_specs
from spectrax.runtime.types import MpMdMesh

import easydel as ed
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.modules.llama.llama_configuration import LlamaConfig

_SPMD_ROOT = Path(__file__).resolve().parents[1] / "spmd"
_DEFAULT_TASK = "CAUSAL_LM"


@dataclass(frozen=True)
class ModuleCase:
    path: Path
    module_name: str
    task_name: str

    @property
    def id(self) -> str:
        return f"{self.path.relative_to(_SPMD_ROOT)}::{self.module_name}:{self.task_name}"


class _RunCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.cases: list[tuple[str, str]] = []

    def visit_Call(self, node: ast.Call) -> None:
        module_name: str | None = None
        task_name: str | None = None
        for keyword in node.keywords:
            if keyword.arg == "module_name" and isinstance(keyword.value, ast.Constant):
                if isinstance(keyword.value.value, str):
                    module_name = keyword.value.value
            elif keyword.arg == "task":
                task_name = _task_name_from_ast(keyword.value)
        if module_name is not None:
            self.cases.append((module_name, task_name or _DEFAULT_TASK))
        self.generic_visit(node)


def _task_name_from_ast(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        for task in ed.TaskType:
            if node.value == task.value:
                return task.name
    return None


def _spmd_test_files() -> list[Path]:
    return sorted(path for path in _SPMD_ROOT.rglob("test_*.py") if path.is_file())


def _module_cases() -> list[ModuleCase]:
    cases: list[ModuleCase] = []
    seen: set[tuple[Path, str, str]] = set()
    for path in _spmd_test_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        visitor = _RunCallVisitor()
        visitor.visit(tree)
        for module_name, task_name in visitor.cases:
            key = (path, module_name, task_name)
            if key not in seen:
                seen.add(key)
                cases.append(ModuleCase(path=path, module_name=module_name, task_name=task_name))
    return cases


@pytest.mark.parametrize("case", _module_cases(), ids=lambda case: case.id)
def test_spmd_module_case_resolves_to_mpmd_capable_module(case: ModuleCase) -> None:
    config_cls, module_cls = ed.get_modules_by_type(
        model_type=case.module_name,
        task_type=getattr(ed.TaskType, case.task_name),
    )

    assert issubclass(config_cls, EasyDeLBaseConfig)
    assert issubclass(module_cls, EasyDeLBaseModule)


def test_every_spmd_module_test_with_runtime_call_has_mpmd_case() -> None:
    cases = _module_cases()
    covered_files = {case.path for case in cases}
    missing = [
        path.relative_to(_SPMD_ROOT).as_posix()
        for path in _spmd_test_files()
        if path.read_text().count("module_name=") and path not in covered_files
    ]
    assert not missing
    assert cases


class _DummyRegionModule(EasyDeLBaseModule):
    config_class = LlamaConfig
    base_model_prefix = "dummy"

    def _pipeline_stage_count(self) -> int:
        return 2

    def forward(self, x):
        return x + 1


def test_easydel_module_call_emits_sxstage_region_markers() -> None:
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        pipeline_stage_regions=True,
    )
    module = _DummyRegionModule(config, jnp.float32, jnp.float32, None, spx.Rngs(0))

    closed = jax.make_jaxpr(lambda x: module(x))(jnp.ones((2,), dtype=jnp.float32))
    specs = stage_region_specs(closed.jaxpr)

    assert {spec.name for spec in specs} == {"llama"}


def test_scheduled_sxjit_rejects_stage_regions_loudly() -> None:
    devices = np.asarray(jax.devices()[:1], dtype=object).reshape(1)
    mesh = MpMdMesh(Mesh(devices, axis_names=("pp",)), "pp")
    config = LlamaConfig(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
        pipeline_stage_regions=True,
    )
    module = _DummyRegionModule(config, jnp.float32, jnp.float32, None, spx.Rngs(0))

    @sxjit(mesh=mesh, schedule=spx.GPipe(microbatches=1))
    def scheduled(x):
        return module(x).sum()

    with pytest.raises(NotImplementedError, match="sxstage_region markers were found"):
        scheduled(jnp.ones((2,), dtype=jnp.float32))
