# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Shared, lightweight correlation-identity rules for records and transports."""

import hashlib
import re

CORRELATION_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}"
CORRELATION_ID_WIRE_PATTERN = "^" + CORRELATION_ID_PATTERN + "$"
_VALID = re.compile(CORRELATION_ID_PATTERN)


def canonical_correlation_id(identifier: str) -> str:
    """Keep ordinary server IDs; hash legacy IDs without changing their records."""

    if _VALID.fullmatch(identifier):
        return identifier
    return "cid_" + hashlib.sha256(identifier.encode("utf-8", errors="surrogatepass")).hexdigest()


def valid_correlation_id(identifier: str | None) -> bool:
    return identifier is not None and _VALID.fullmatch(identifier) is not None
