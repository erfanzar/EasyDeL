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
"""Helpers shared by the HuggingFace download / conversion scripts.

Shared by the two downloaders and the batch converter: the comment-stripping
list reader, the glob matcher and the ``owner/name`` output-directory layout.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import os
from pathlib import Path


def strip_comment(line: str) -> str:
    """Strip a trailing ``# ...`` comment and surrounding whitespace from one line.

    Args:
        line: Raw line from a list file.

    Returns:
        The comment-free, stripped line; empty for blank or comment-only input.
    """
    return line.split("#", 1)[0].strip()


def read_id_list(path: str | os.PathLike) -> list[str]:
    """Read one identifier per line from a text file (``#`` comments allowed).

    Args:
        path: Path to the list file (UTF-8).

    Returns:
        The non-empty, comment-free lines in file order.
    """
    text = Path(path).read_text(encoding="utf-8")
    ids: list[str] = []
    for raw_line in text.splitlines():
        line = strip_comment(raw_line)
        if line:
            ids.append(line)
    return ids


def matches_any_glob(name: str, globs: tuple[str, ...]) -> bool:
    """Return True when ``name`` matches at least one fnmatch pattern in ``globs``.

    Args:
        name: Filename or repo-relative path to test.
        globs: Tuple of fnmatch patterns.

    Returns:
        True if any pattern matches ``name``.
    """
    return any(fnmatch.fnmatch(name, pattern) for pattern in globs)


def repo_out_dir(out_root: Path, repo_id: str) -> Path:
    """Compute the local destination directory for a repository.

    Args:
        out_root: Configured output root.
        repo_id: HuggingFace repo id (``owner/name`` or a single segment).

    Returns:
        ``out_root/owner/name`` when ``repo_id`` contains a slash, otherwise ``out_root/repo_id``.
    """
    if "/" in repo_id:
        owner, name = repo_id.split("/", 1)
        return out_root / owner / name
    return out_root / repo_id


def enable_hf_fast_transfer() -> str | None:
    """Turn on the accelerated HuggingFace download path for the running process.

    ``huggingface_hub`` snapshots its transfer settings into ``huggingface_hub.constants`` at import time.
    Update the in-process backend setting directly, rather than adding environment variables that would arrive
    too late. Legacy releases use ``hf_transfer``; current releases use the accelerated Xet backend for Xet-hosted
    files. Enabling Xet also works when it was disabled before the hub was imported.

    Returns:
        A warning message when the optional accelerator package is missing, else ``None``.
    """
    from huggingface_hub import constants

    if hasattr(constants, "HF_HUB_ENABLE_HF_TRANSFER"):
        if importlib.util.find_spec("hf_transfer") is None:
            return "`hf_transfer` is not installed; accelerated downloads require the optional hf_transfer package."
        constants.HF_HUB_ENABLE_HF_TRANSFER = True
        return None

    if importlib.util.find_spec("hf_xet") is None:
        return "`hf_xet` is not installed; accelerated downloads require the optional hf_xet package."
    constants.HF_HUB_DISABLE_XET = False
    return None
