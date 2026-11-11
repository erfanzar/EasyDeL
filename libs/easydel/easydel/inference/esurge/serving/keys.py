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

"""The serving API key: one stable key held in a file, which is the only source of truth.

The file is read when it exists and created (mode 0600) when it does not, so relaunching the server keeps the same
key and clients and dashboards never need updating. Rotating the key means replacing the file and restarting. The
auth manager runs without persistence, so keys from earlier launches cannot accumulate and stay valid.
"""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

MIN_KEY_LENGTH = 32
"""Shortest accepted key; generated keys are 64 URL-safe characters after the ``sk-`` prefix."""


class ApiKeyFileError(RuntimeError):
    """The API-key file is unusable: unreadable, empty, too short or readable by other users."""


def _check_private(path: Path) -> None:
    mode = path.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ApiKeyFileError(f"{path} is not a regular file")
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ApiKeyFileError(f"{path} is accessible to other users (mode {stat.S_IMODE(mode):o}); chmod 600 it")


def load_or_create_api_key(path: str | Path) -> tuple[str, bool]:
    """Return the serving API key from ``path``, creating the file with a fresh key when it is absent.

    Creation is exclusive (``O_EXCL``), so two processes racing on a missing file cannot end up with different
    keys: the loser reads the winner's key.

    Args:
        path: Key file location (``~`` is expanded). Parent directories are created with mode 0700.

    Returns:
        ``(key, created)``.

    Raises:
        ApiKeyFileError: When the file is not a private regular file, cannot be read, or holds no usable key.
    """
    path = Path(path).expanduser()
    created = False
    if not path.exists():
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = f"sk-{secrets.token_urlsafe(48)}"
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(key + "\n")
            created = True
    try:
        _check_private(path)
        key = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ApiKeyFileError(f"cannot read API key file {path}: {exc}") from exc
    if len(key) < MIN_KEY_LENGTH or any(ch.isspace() for ch in key):
        raise ApiKeyFileError(f"{path} must hold one key of at least {MIN_KEY_LENGTH} non-space characters")
    return key, created
