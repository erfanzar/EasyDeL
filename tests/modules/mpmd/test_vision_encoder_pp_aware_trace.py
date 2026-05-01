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

"""Regression: vision-encoder ``.layers.scan`` calls must force ``trace=True`` when ``pp>1``.

When the encoder's ``trace=`` expression evaluates to ``False`` and the body
runs as a JAX scan, the carried ``idx`` becomes a *traced* int32 array. The
mixin's ``_mark_layer_stage_boundary`` then catches the resulting
``TypeError`` from ``int(layer_idx)`` and silently no-ops -- meaning
``spx.sxstage_iter`` never fires and the pscan compiler doesn't see stage
boundaries.

The fix appends ``or self._pipeline_stage_count() > 1`` to each affected
encoder's trace expression. This test guards each of the 5 known sites
against a regression that strips the clause. We do a static source-level
check because the alternative (running pp>1 forward) requires multi-host
machinery that single-host CI cannot exercise.

Affected sites (all in ``EasyDeLLayerStackMixin``-using vision encoders):
  - ``CLIPEncoder.forward``                 (clip/modeling_clip.py)
  - ``Llama4VisionEncoder.forward``         (llama4/modeling_llama4.py)
  - ``PixtralTransformer.forward``          (pixtral/modeling_pixtral.py)
  - ``SiglipEncoder.forward``               (siglip/modeling_siglip.py)
  - ``Gemma4VisionEncoder.forward``         (gemma4/modeling_gemma4.py)

Whisper and Roberta hardcode ``trace=True`` and are not affected.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest


def _trace_clauses(forward_src: str) -> list[str]:
    """Return the source-text of every ``trace=`` keyword argument in ``.scan(...)`` calls.

    Uses the AST so multi-line expressions with nested calls (e.g.
    ``_pipeline_stage_count() > 1``) are extracted verbatim. We look for
    ``Call`` nodes whose function ends in ``.scan`` -- that covers
    ``self.layers.scan(...)`` and ``self.blocks.scan(...)`` alike.
    """
    src = textwrap.dedent(forward_src)
    tree = ast.parse(src)
    clauses: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute) and func.attr == "scan"):
            continue
        for kw in node.keywords:
            if kw.arg == "trace":
                clauses.append(ast.unparse(kw.value))
    return clauses


def _has_pp_aware_clause(expr: str) -> bool:
    """True if the trace expression honors ``pp>1`` either directly or via the helper."""

    if "_pipeline_stage_count()" in expr and ">" in expr:
        return True

    if "_layer_scan_trace" in expr:
        return True

    stripped = expr.strip()
    if stripped == "True":
        return True
    return False


@pytest.mark.parametrize(
    "encoder_class",
    [
        pytest.param(
            "easydel.modules.clip.modeling_clip:CLIPEncoder",
            id="clip-CLIPEncoder",
        ),
        pytest.param(
            "easydel.modules.llama4.modeling_llama4:Llama4VisionEncoder",
            id="llama4-Llama4VisionEncoder",
        ),
        pytest.param(
            "easydel.modules.pixtral.modeling_pixtral:PixtralTransformer",
            id="pixtral-PixtralTransformer",
        ),
        pytest.param(
            "easydel.modules.siglip.modeling_siglip:SiglipEncoder",
            id="siglip-SiglipEncoder",
        ),
        pytest.param(
            "easydel.modules.gemma4.modeling_gemma4:Gemma4VisionEncoder",
            id="gemma4-Gemma4VisionEncoder",
        ),
    ],
)
def test_vision_encoder_trace_expression_honors_pp(encoder_class: str):
    module_path, class_name = encoder_class.split(":")
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)

    candidate_methods = []
    for name in ("forward", "__call__"):
        if hasattr(cls, name):
            candidate_methods.append((name, getattr(cls, name)))

    found_any_scan = False
    for name, fn in candidate_methods:
        try:
            src = inspect.getsource(fn)
        except (OSError, TypeError):
            continue

        clauses = _trace_clauses(src)
        if not clauses:
            continue
        found_any_scan = True

        for expr in clauses:
            assert _has_pp_aware_clause(expr), (
                f"{class_name}.{name} has a .layers.scan(trace=...) expression that "
                f"does not honor pp>1:\n  {expr!r}\n"
                f"Expected one of: '_pipeline_stage_count() > 1', "
                f"'_layer_scan_trace(...)', or 'True' literal."
            )

    assert found_any_scan, (
        f"{class_name}: no .scan(trace=...) call found in forward/__call__ -- test parametrization may be stale."
    )


def test_trace_clauses_helper_picks_up_multiline_expressions():
    """Sanity-check the source-grep helper handles the multiline pattern this fix uses."""
    sample = """
    hidden_states, _, all_hidden_states, _ = self.layers.scan(
        _layer_loop,
        (hidden_states, all_hidden_states, all_attentions, 0),
        trace=output_hidden_states
        or output_attentions
        or not self.config.scan_layers
        or self._pipeline_stage_count() > 1,
    )
    """
    clauses = _trace_clauses(sample)
    assert len(clauses) == 1
    assert "_pipeline_stage_count()" in clauses[0]
    assert _has_pp_aware_clause(clauses[0])


def test_trace_clauses_helper_rejects_buggy_expression():
    """Confirm the regression check would catch the pre-fix expression."""
    sample = """
    self.layers.scan(
        _layer_loop,
        carry,
        trace=output_hidden_states or output_attentions or not self.config.scan_layers,
    )
    """
    clauses = _trace_clauses(sample)
    assert len(clauses) == 1
    assert not _has_pp_aware_clause(clauses[0]), (
        "Helper failed to flag the buggy pre-fix trace expression; regression guard is broken."
    )
