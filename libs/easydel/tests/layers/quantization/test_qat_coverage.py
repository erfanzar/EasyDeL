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

"""Guard: no weight matmul may quietly escape the quantization rules.

Quantization here is applied at the layers that consult a stamped rule,
which covers a model only insofar as its weight contractions go through
those layers. A model that multiplies a parameter by hand — with
``jnp.einsum``, ``jnp.matmul`` or ``@`` — bypasses the mechanism, and it
does so *silently*: the module is stamped like every other one, reports
itself as quantized, and ignores its plan.

So the bypasses are enumerated rather than assumed. Every site where a
module contracts its own parameter directly is listed below with the
reason it is exempt, and the test fails when a new one appears. The three
categories that legitimately stay in full precision:

* **Router and gate projections.** They produce routing *decisions*, not
  features. A rounding error there changes which expert a token is sent
  to, which is a discrete change to the computation rather than a small
  numerical perturbation, and they are tiny enough that the memory saved
  is irrelevant.
* **Mixing coefficient projections.** Same argument: the output feeds a
  sigmoid and becomes a blend weight.
* **Everything else** — which must be empty.
"""

from __future__ import annotations

import pathlib
import re

import easydel

_CONTRACTION = re.compile(
    r"(jnp\.einsum|jnp\.matmul|jax\.numpy\.einsum|lax\.dot_general|jax\.lax\.dot_general|@)"
)
_OWN_PARAMETER = re.compile(r"self\.([A-Za-z_][A-Za-z0-9_]*)\.value|self\.(weight|kernel)\b")

_EXEMPT: dict[tuple[str, str], str] = {
    ("modules/glm4_moe/modeling_glm4_moe.py", "weight"): (
        "MoE router gate: produces expert routing decisions rather than features, so a rounding error "
        "changes which expert a token reaches. Kept in float32."
    ),
    ("modules/glm4_moe_lite/modeling_glm4_moe_lite.py", "weight"): (
        "MoE router gate: routing decisions, kept in float32."
    ),
    ("modules/glm_moe_dsa/modeling_glm_moe_dsa.py", "weight"): (
        "MoE router gate: routing decisions, kept in float32."
    ),
    ("modules/deepseek_v4/modeling_deepseek_v4.py", "hc_fn"): (
        "Hyper-connection mixing matrix: a tiny float32 projection whose output feeds a sigmoid to become "
        "blend weights, not features."
    ),
}
"""Reviewed exemptions, keyed by ``(package-relative path, parameter name)``.

Keyed by name rather than line number so the guard survives edits above
it, and by name rather than file so a second bypass in an already-listed
file is still caught.
"""


def _source_root() -> pathlib.Path:
    """Return the root of the installed easydel package.

    Returns:
        The package directory.
    """
    return pathlib.Path(easydel.__file__).parent


def _direct_parameter_contractions() -> dict[tuple[str, str], list[int]]:
    """Find every site where a module contracts one of its own parameters.

    Returns:
        Mapping from ``(package-relative path, parameter name)`` to the
        1-based line numbers where it happens.
    """
    root = _source_root()
    found: dict[tuple[str, str], list[int]] = {}
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("layers/quantization/"):
            continue  # the quantization implementation itself
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("*"):
                continue
            if not _CONTRACTION.search(line):
                continue
            for match in _OWN_PARAMETER.finditer(line):
                name = match.group(1) or match.group(2)
                found.setdefault((relative, name), []).append(number)
    return found


def test_no_unreviewed_weight_matmul_bypasses_the_rules():
    """Every direct parameter contraction must be a reviewed, documented exemption.

    A new one is not necessarily a bug — but it is always a decision,
    because such a site cannot be reached by a quantization rule and will
    silently stay in full precision while reporting itself as quantized.
    """
    unreviewed = {key: lines for key, lines in _direct_parameter_contractions().items() if key not in _EXEMPT}
    assert not unreviewed, (
        "These modules contract a parameter directly, so a quantization rule cannot reach them and they "
        "will silently train in full precision:\n"
        + "\n".join(f"  {path} :: self.{name} at line(s) {lines}" for (path, name), lines in sorted(unreviewed.items()))
        + "\n\nEither route the contraction through spx.quantization (see DeepseekV4GroupedLinear for the "
        "pattern) or add it to _EXEMPT with the reason it must stay full precision."
    )


def test_the_exemption_list_has_not_gone_stale():
    """An exemption for a site that no longer exists hides a future bypass."""
    stale = sorted(set(_EXEMPT) - set(_direct_parameter_contractions()))
    assert not stale, (
        f"These entries are exempted but no longer contain a direct parameter contraction: {stale}. "
        f"Remove them from _EXEMPT so the guard keeps its teeth."
    )


_CONV_EXEMPT: dict[str, tuple[int, str]] = {
    "modules/mamba/modeling_mamba.py": (1, "depthwise causal conv1d"),
    "modules/mamba2/modeling_mamba2.py": (1, "depthwise causal conv1d"),
    "modules/falcon_mamba/modeling_falcon_mamba.py": (1, "depthwise causal conv1d"),
    "modules/falcon_h1/modeling_falcon_h1.py": (1, "depthwise causal conv1d"),
    "modules/qwen3_next/modeling_qwen3_next.py": (2, "depthwise causal conv1d"),
}
"""Convolutions that deliberately stay in full precision, with their call counts.

Every convolution in the model zoo is the depthwise causal ``conv1d`` of a
state-space or linear-attention layer. Three things make them a poor
quantization target and a needless one:

* they are depthwise with a kernel width of about four, so the weight is a
  few thousand values against billions in the projections — quantizing
  them saves nothing measurable;
* they are computed in float32 on purpose, because the recurrence that
  consumes them is numerically delicate;
* the ones that matter for performance do not go through
  ``jax.lax.conv_general_dilated`` at all in production, but through
  ejkernel Pallas kernels, which op-level interception cannot reach.

So a quantized convolution is not implemented, and this records the
decision rather than leaving it implicit. The counts are pinned so that a
*new* convolution — one that might not fit the reasoning above — fails
here and gets a fresh decision.
"""


def _convolution_sites() -> dict[str, int]:
    """Count convolution calls per file in the model zoo.

    Returns:
        Mapping from package-relative path to the number of
        ``conv_general_dilated`` call sites.
    """
    root = _source_root()
    found: dict[str, int] = {}
    for path in sorted(root.rglob("*.py")):
        count = path.read_text(encoding="utf-8").count("conv_general_dilated(")
        if count:
            found[path.relative_to(root).as_posix()] = count
    return found


def test_no_unreviewed_convolution_escapes_the_decision():
    """A new convolution must be a fresh decision, not an inherited exemption."""
    found = _convolution_sites()
    problems = []
    for path, count in sorted(found.items()):
        if path not in _CONV_EXEMPT:
            problems.append(f"  {path}: {count} convolution(s), not reviewed")
        elif _CONV_EXEMPT[path][0] != count:
            problems.append(f"  {path}: {count} convolution(s), but {_CONV_EXEMPT[path][0]} were reviewed")
    assert not problems, (
        "Convolutions are deliberately left in full precision (see _CONV_EXEMPT), but this set has changed:\n"
        + "\n".join(problems)
        + "\n\nConfirm the new convolution is also a tiny float32 depthwise kernel, or implement quantized "
        "convolution support, then update _CONV_EXEMPT."
    )


def test_the_convolution_exemption_list_has_not_gone_stale():
    """An exemption for a file with no convolutions hides a future one."""
    stale = sorted(set(_CONV_EXEMPT) - set(_convolution_sites()))
    assert not stale, f"These paths are exempted but contain no convolution any more: {stale}."
