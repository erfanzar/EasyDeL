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

import ast
from pathlib import Path

from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES_ROOT = REPO_ROOT / "easydel" / "modules"
WRAPPER_MODELS_WITHOUT_LOCAL_CHECKPOINTS = {
    "easydel/modules/dflash/modeling_dflash.py",
    "easydel/modules/dspark/modeling_dspark.py",
    "easydel/modules/eagle3/modeling_eagle3.py",
    "easydel/modules/glm46v/modeling_glm46v.py",
    "easydel/modules/glm4v_moe/modeling_glm4v_moe.py",
    "easydel/modules/qwen3_5_moe/modeling_qwen3_5_moe.py",
}

#: Shared layer helpers that emit ``checkpoint_name`` labels on the caller's
#: behalf. A model file that routes its forward through one of these still
#: participates in name-based remat policies even though the literal call does
#: not appear in the model file — see :func:`easydel.layers.gated_mlp_forward`,
#: which tags ``mlp_gate_up`` / ``mlp_gate`` / ``mlp_down`` / ``mlp_output``.
LABEL_EMITTING_HELPERS = ("gated_mlp_forward",)


def _collect_checkpoint_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            fn = node.func
            if not isinstance(fn, ast.Name) or fn.id != "checkpoint_name":
                self.generic_visit(node)
                return

            name = None
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and isinstance(node.args[1].value, str):
                name = node.args[1].value
            else:
                for keyword in node.keywords:
                    if (
                        keyword.arg == "name"
                        and isinstance(keyword.value, ast.Constant)
                        and isinstance(keyword.value.value, str)
                    ):
                        name = keyword.value.value
                        break

            if name is not None:
                names.add(name)
            self.generic_visit(node)

    Visitor().visit(tree)
    return names


def test_all_checkpoint_name_labels_are_registered():
    actual_names: set[str] = set()
    for path in MODULES_ROOT.rglob("modeling_*.py"):
        actual_names.update(_collect_checkpoint_names(path))

    missing_names = actual_names.difference(GRADIENT_CHECKPOINT_TARGETS)
    assert missing_names == set()


def test_only_wrapper_model_files_skip_checkpoint_names():
    """Every model family must be reachable by name-based remat policies.

    A family satisfies this either by tagging its own activations with
    ``checkpoint_name`` or by routing its forward through a shared helper that
    tags them (:data:`LABEL_EMITTING_HELPERS`). Only true wrappers — families
    that define no local blocks and reuse another family's modules — are
    allowed to do neither; those are enumerated explicitly so a new family
    cannot silently lose ``EasyDeLGradientCheckPointers`` coverage.
    """
    files_without_checkpoint_names = set()
    for path in MODULES_ROOT.rglob("modeling_*.py"):
        source = path.read_text()
        if "checkpoint_name(" in source:
            continue
        if any(helper in source for helper in LABEL_EMITTING_HELPERS):
            continue
        files_without_checkpoint_names.add(str(path.relative_to(REPO_ROOT)))

    assert files_without_checkpoint_names == WRAPPER_MODELS_WITHOUT_LOCAL_CHECKPOINTS
