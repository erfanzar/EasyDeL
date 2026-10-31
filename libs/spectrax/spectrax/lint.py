# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Optional lint checks for :class:`~spectrax.Module` instances.

These checks are advisory: nothing in spectrax's runtime depends on
running them. They exist as escape hatches for catching common authoring
mistakes (e.g. accidentally aliased parameters) before they show up as
divergent gradients during training.
"""

from __future__ import annotations

from .core.graph import export
from .core.module import Module

__all__ = ["check_unintentional_sharing"]


def check_unintentional_sharing(module: Module) -> list[tuple[str, str]]:
    """Flag variables shared across paths without an explicit ``tie_group``.

    Weight tying is legitimate when deliberate (input/output embedding
    sharing, for instance). Accidental sharing — two parameters that
    reference the same :class:`~spectrax.Variable` by mistake — shows
    up here as a list of ``(alias_path, canonical_path)`` pairs whose
    variable lacks a ``"tie_group"`` metadata entry.

    Args:
        module: The module to audit.

    Returns:
        Every suspected unintentional alias. An empty list means no
        untagged sharing was detected.
    """
    gdef, _ = export(module)
    out: list[tuple[str, str]] = []
    rid_metadata: dict[int, dict[str, object]] = {}
    for node_idx, rid in gdef.var_refs:
        vn = gdef.nodes[node_idx]
        rid_metadata[rid] = dict(vn.metadata) if hasattr(vn, "metadata") else {}

    canonical_paths = {p for _, p in gdef.var_canonical}

    def untagged_variable_alias(_alias: str, canonical: str) -> bool:
        """Decide whether the variable at ``canonical`` lacks a ``tie_group`` tag.

        Looks up the variable's metadata via :data:`gdef.var_canonical`
        and the ``rid_metadata`` map captured in the enclosing scope.

        Args:
            _alias: The aliased path (unused here; kept so the
                signature mirrors the iterator producing pairs).
            canonical: The canonical variable path being checked.

        Returns:
            ``True`` when the variable is shared without an explicit
            ``"tie_group"`` metadata key (i.e. the alias is suspect),
            ``False`` otherwise.
        """
        for r, p in gdef.var_canonical:
            if p == canonical:
                return "tie_group" not in rid_metadata.get(r, {})
        return False

    for alias, canonical in gdef.shared_paths:
        if canonical in canonical_paths:
            if untagged_variable_alias(alias, canonical):
                out.append((alias, canonical))
        else:
            for _, var_path in gdef.var_canonical:
                if canonical == "":
                    var_alias = f"{alias}.{var_path}" if alias else var_path
                elif var_path.startswith(canonical + "."):
                    var_alias = f"{alias}{var_path[len(canonical) :]}"
                else:
                    continue
                if untagged_variable_alias(var_alias, var_path):
                    out.append((var_alias, var_path))
    return out
