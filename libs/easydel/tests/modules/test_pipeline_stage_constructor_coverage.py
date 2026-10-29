"""Static coverage for pipeline-stage-aware module construction."""

from __future__ import annotations

from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parents[2] / "easydel" / "modules"
FINAL_STAGE_ATTRS = (
    "self.norm =",
    "self.ln_f =",
    "self.final_layer_norm =",
    "self.layer_norm =",
)


def _modeling_files() -> list[Path]:
    return sorted(MODULE_ROOT.glob("*/modeling_*.py"))


def test_module_constructors_use_config_aware_stage_assignment():
    """Decoder/encoder stacks should not bypass EasyDeL's PP stage mapper."""
    offenders = []
    for path in _modeling_files():
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("with spx.assign_stage("):
                offenders.append(f"{path.relative_to(MODULE_ROOT.parents[1])}:{line_no}")

    assert not offenders, "raw spx.assign_stage calls found:\n" + "\n".join(offenders)


def test_final_stack_norms_are_created_on_last_layer_stage():
    """Final stack-owned norms must carry last-stage parameter metadata."""
    offenders = []
    for path in _modeling_files():
        lines = path.read_text().splitlines()
        active = False
        total_layers_expr: str | None = None
        constructor_indent = 0

        for line_no, line in enumerate(lines, start=1):
            stripped = line.strip()
            indent = len(line) - len(line.lstrip())

            if stripped.startswith("self.") and "= nn.ModuleList([])" in stripped:
                active = True
                total_layers_expr = None
                constructor_indent = indent
                continue

            if active and "with self.assign_layer_stage(" in line and "total_layers=" in line:
                total_layers_expr = line.split("total_layers=", 1)[1].split("):", 1)[0].strip()
                continue

            if active and stripped.startswith("def forward") and indent <= constructor_indent:
                active = False
                continue

            if not active or total_layers_expr is None or indent != constructor_indent:
                continue

            if not any(stripped.startswith(prefix) for prefix in FINAL_STAGE_ATTRS):
                continue

            previous = next((prev.strip() for prev in reversed(lines[: line_no - 1]) if prev.strip()), "")
            if not previous.startswith("with self.assign_layer_stage"):
                offenders.append(f"{path.relative_to(MODULE_ROOT.parents[1])}:{line_no}")

    assert not offenders, "final stack norms outside last-stage context:\n" + "\n".join(offenders)
