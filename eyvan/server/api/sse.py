# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Server-sent event plumbing shared by every streaming route.

Two things live here, and neither is about what is being streamed.

The **framing**: an SSE frame is ``id:``/``event:``/``data:`` terminated by a
blank line, and a keep-alive is a comment. Getting the blank line wrong produces
a stream that looks fine in a terminal and never fires an event in a browser,
because ``EventSource`` waits for the terminator.

The **headers**: ``Cache-Control: no-cache`` and ``X-Accel-Buffering: no``. The
second is not decoration -- an nginx or ingress buffer in front of the control
plane holds a streaming response until its buffer fills, which for a quiet log
turns "live" into "several minutes late" with nothing in any log to explain it.

Extracted so a second streaming route does not restate them slightly
differently. The audit stream and a log follower stream completely different
things, on different cursors, with different authorization -- and exactly the
same transport.
"""

from __future__ import annotations

import json
import typing as tp

from fastapi.responses import StreamingResponse

#: Sent when a poll finds nothing, so an idle stream is distinguishable from a
#: dead one by every intermediary between here and the client. A comment rather
#: than an event: ``EventSource`` ignores it, so a client never sees a synthetic
#: record it would have to filter out.
KEEPALIVE_FRAME = ": keep-alive\n\n"

_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_frame(*, data: object, event: str | None = None, identifier: str | None = None) -> str:
    """Render one server-sent event.

    ``data`` is JSON-encoded compactly on a single line. A newline inside a
    ``data:`` field silently starts a second field, so a multi-line payload
    would arrive as a truncated one.
    """

    payload = json.dumps(data, separators=(",", ":"))
    lines = []
    if identifier is not None:
        lines.append(f"id: {identifier}")
    if event is not None:
        lines.append(f"event: {event}")
    lines.append(f"data: {payload}")
    return "\n".join(lines) + "\n\n"


def sse_response(source: tp.AsyncIterator[str]) -> StreamingResponse:
    """Wrap a frame generator in a correctly-headed streaming response.

    The generator is passed already constructed, which matters: authorization
    must happen *before* it is built. A check inside the generator runs after
    the response has been committed, so an unauthorized caller sees ``200``
    followed by nothing -- indistinguishable from a stream with no events.
    """

    return StreamingResponse(source, media_type="text/event-stream", headers=dict(_STREAM_HEADERS))


__all__ = ("KEEPALIVE_FRAME", "sse_frame", "sse_response")
