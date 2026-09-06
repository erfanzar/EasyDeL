# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Persisted pool specs (``~/.eray/capacity.json``).

Only the *desired state* is persisted — which pools exist and what they
want. Observed state always comes from the cloud (pool-label rediscovery),
so this file is tiny and losing it costs nothing but re-running
``eray tpu provision``. The kill-test case is why desired state must be
persisted at all: a QR deleted out from under the pool leaves nothing in
the cloud to infer the target count from.

Same file conventions as the fleet registry's local backend: atomic
replace, forward-compatible ``extra`` bag, versioned document.
"""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from pathlib import Path

from .pool import PoolSpec
from .types import CapacityType

STATE_PATH = Path("~/.eray/capacity.json").expanduser()

_DOC_VERSION = 1


def spec_to_dict(spec: PoolSpec) -> dict:
    """Serialize a PoolSpec to a JSON-safe dict.

    Args:
        spec: The spec.

    Returns:
        Plain dict with lists/strings only.
    """
    data = dataclasses.asdict(spec)
    data["zones"] = list(spec.zones)
    data["capacity"] = str(spec.capacity)
    return data


def spec_from_dict(data: dict) -> PoolSpec:
    """Deserialize a PoolSpec, ignoring unknown keys (forward compat).

    Args:
        data: Dict as produced by :func:`spec_to_dict` (possibly newer).

    Returns:
        The spec.
    """
    known = {f.name for f in dataclasses.fields(PoolSpec)}
    kwargs = {k: v for k, v in data.items() if k in known}
    kwargs["zones"] = tuple(kwargs.get("zones", ()))
    kwargs["capacity"] = CapacityType(kwargs.get("capacity", "spot"))
    return PoolSpec(**kwargs)


def _load_doc(path: Path) -> dict:
    """Load the state document, returning an empty one when absent.

    Raises:
        RuntimeError: With an actionable message when the file exists but
            is not valid JSON (e.g. a bad hand edit) — every capacity
            command reads this file, so a raw traceback here bricks them
            all without saying why.
    """
    if not path.exists():
        return {"version": _DOC_VERSION, "pools": {}}
    with open(path) as f:
        try:
            doc = json.load(f)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{path} is not valid JSON ({exc}); fix or remove it, then re-run `eray tpu provision`"
            ) from exc
    doc.setdefault("pools", {})
    return doc


def _save_doc(doc: dict, path: Path) -> None:
    """Atomically write the state document.

    Uses a unique temp file (not a shared ``.tmp`` name) so two concurrent
    writers cannot replace the target with each other's half-written file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(doc, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def load_pool_specs(path: Path | None = None) -> dict[str, PoolSpec]:
    """Load every saved pool spec.

    Args:
        path: Override the state file (tests).

    Returns:
        Mapping of pool name → spec.
    """
    doc = _load_doc(path or STATE_PATH)
    return {name: spec_from_dict(data) for name, data in doc["pools"].items()}


def save_pool_spec(spec: PoolSpec, path: Path | None = None) -> None:
    """Insert or update one pool spec.

    Args:
        spec: The spec to persist.
        path: Override the state file (tests).
    """
    target = path or STATE_PATH
    doc = _load_doc(target)
    doc["pools"][spec.name] = spec_to_dict(spec)
    _save_doc(doc, target)


def remove_pool_spec(name: str, path: Path | None = None) -> bool:
    """Remove one pool spec.

    Args:
        name: Pool name.
        path: Override the state file (tests).

    Returns:
        True when a spec was removed, False when it wasn't saved.
    """
    target = path or STATE_PATH
    doc = _load_doc(target)
    removed = doc["pools"].pop(name, None) is not None
    if removed:
        _save_doc(doc, target)
    return removed
