import ast
from pathlib import Path

from easydel.infra.etils import GRADIENT_CHECKPOINT_TARGETS

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULES_ROOT = REPO_ROOT / "easydel" / "modules"
WRAPPER_MODELS_WITHOUT_LOCAL_CHECKPOINTS = {
    "easydel/modules/qwen3_5/modeling_qwen3_5.py",
    "easydel/modules/qwen3_5_moe/modeling_qwen3_5_moe.py",
}


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
    files_without_checkpoint_names = {
        str(path.relative_to(REPO_ROOT))
        for path in MODULES_ROOT.rglob("modeling_*.py")
        if "checkpoint_name(" not in path.read_text()
    }

    assert files_without_checkpoint_names == WRAPPER_MODELS_WITHOUT_LOCAL_CHECKPOINTS
