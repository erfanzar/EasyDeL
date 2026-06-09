# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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

"""Static guardrails for the TileLang backend.

These tests do not certify an algorithm as native. They prevent the backend
from getting less honest while the known JAX orchestration backlog is replaced
with real ``@T.prim_func`` kernels.
"""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TILELANG_ROOT = _REPO_ROOT / "ejkernel" / "kernels" / "_tilelang"

_BANNED_JAX_CALLS = frozenset(
    {
        "jax.lax.all_gather",
        "jax.lax.psum_scatter",
        "jax.lax.scan",
        "jax.lax.top_k",
        "jax.nn.softmax",
        "jax.random.bernoulli",
        "jax.random.key_data",
        "jnp.all",
        "jnp.arange",
        "jnp.broadcast_to",
        "jnp.concatenate",
        "jnp.cumsum",
        "jnp.einsum",
        "jnp.exp",
        "jnp.flip",
        "jnp.full",
        "jnp.isclose",
        "jnp.max",
        "jnp.ones",
        "jnp.ones_like",
        "jnp.repeat",
        "jnp.sum",
        "jnp.take",
        "jnp.where",
        "jnp.zeros",
        "jnp.zeros_like",
    }
)

_KNOWN_JAX_COMPUTE = frozenset(
    {
        ("fused_cross_entropy/_impl.py", "jnp.exp", "jnp.exp(local_max - global_max)"),
        ("fused_cross_entropy/_impl.py", "jnp.full", "jnp.full(flat_logits.shape[0], -1.0, dtype=jnp.float32)"),
        ("fused_cross_entropy/_impl.py", "jnp.ones", "jnp.ones(flat_logits.shape[0], dtype=jnp.float32)"),
        ("fused_cross_entropy/_impl.py", "jnp.sum", "jnp.sum(flat_weights)"),
        ("fused_cross_entropy/_impl.py", "jnp.sum", "jnp.sum(per_row_loss)"),
        ("fused_kl_divergence/_impl.py", "jnp.exp", "jnp.exp(local_max_s - global_max_s)"),
        ("fused_kl_divergence/_impl.py", "jnp.exp", "jnp.exp(local_max_t - global_max_t)"),
        ("fused_kl_divergence/_impl.py", "jnp.exp", "jnp.exp(log_p_s)"),
        ("fused_kl_divergence/_impl.py", "jnp.exp", "jnp.exp(log_p_t)"),
        ("fused_kl_divergence/_impl.py", "jnp.ones", "jnp.ones(flat_student.shape[0], dtype=jnp.float32)"),
        ("fused_kl_divergence/_impl.py", "jnp.sum", "jnp.sum(flat_weights)"),
        ("fused_kl_divergence/_impl.py", "jnp.sum", "jnp.sum(per_row)"),
        ("fused_kl_divergence/_impl.py", "jnp.sum", "jnp.sum(per_token, axis=-1)"),
        ("fused_kl_divergence/_impl.py", "jnp.zeros_like", "jnp.zeros_like(teacher_2d)"),
    }
)


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _iter_tilelang_sources() -> list[Path]:
    return sorted(_TILELANG_ROOT.glob("**/_interface.py")) + sorted(_TILELANG_ROOT.glob("**/_impl.py"))


def test_tilelang_public_interfaces_do_not_delete_arguments() -> None:
    offenders: list[str] = []
    for path in sorted(_TILELANG_ROOT.glob("**/_interface.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = path.relative_to(_TILELANG_ROOT)
        for node in ast.walk(tree):
            if isinstance(node, ast.Delete):
                targets = ", ".join(ast.unparse(target) for target in node.targets)
                offenders.append(f"{rel}:{node.lineno}: del {targets}")

    assert not offenders, "TileLang public interfaces must honor args or raise explicitly:\n" + "\n".join(offenders)


def test_tilelang_has_no_new_jax_compute_or_orchestration() -> None:
    hits: set[tuple[str, str, str]] = set()
    lines: dict[tuple[str, str, str], int] = {}

    for path in _iter_tilelang_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        rel = str(path.relative_to(_TILELANG_ROOT))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _qualified_name(node.func)
                if name in _BANNED_JAX_CALLS:
                    hit = (rel, name, ast.unparse(node))
                    hits.add(hit)
                    lines[hit] = node.lineno
            elif isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "at":
                hit = (rel, ".at[]", ast.unparse(node))
                hits.add(hit)
                lines[hit] = node.lineno

    unexpected = hits - _KNOWN_JAX_COMPUTE
    retired = _KNOWN_JAX_COMPUTE - hits

    unexpected_msg = "\n".join(
        f"{file}:{lines[item]}: {func}: {expr}" for item in sorted(unexpected) for file, func, expr in [item]
    )
    retired_msg = "\n".join(f"{file}: {func}: {expr}" for file, func, expr in sorted(retired))

    assert not unexpected, "New JAX-side TileLang compute/orchestration found:\n" + unexpected_msg
    assert not retired, "Known JAX-side TileLang backlog changed; update this allowlist:\n" + retired_msg
