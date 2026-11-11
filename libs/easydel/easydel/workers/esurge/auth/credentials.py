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

"""Where an HTTP request's API key is read from.

Keys are accepted only in ``Authorization: Bearer <key>`` or ``X-API-Key: <key>``. Query strings end up in
access logs, proxy logs and browser history, and request-body fields such as OpenAI's ``user`` are not secrets,
so neither is ever read as a credential.
"""

from __future__ import annotations

from collections.abc import Mapping


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """Value of header ``name`` (case-insensitive), or ``None``."""
    for key, value in headers.items():
        if key.lower() == name:
            return value
    return None


def api_key_from_headers(headers: Mapping[str, str]) -> str | None:
    """Return the API key a request presents, or ``None`` when it presents none.

    Header names and the ``Bearer`` scheme are case-insensitive; the key itself is returned exactly, only
    stripped of surrounding whitespace. ``Authorization: Bearer`` wins over ``X-API-Key``.

    Args:
        headers: Request headers (a Starlette ``Headers`` or any string mapping).

    Returns:
        The presented key, or ``None``.
    """
    authorization = _header(headers, "authorization")
    if authorization:
        scheme, _, token = authorization.strip().partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    api_key = _header(headers, "x-api-key")
    if api_key and api_key.strip():
        return api_key.strip()
    return None
