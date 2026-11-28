// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it, vi } from "vitest";

import { followEventStream, parseFrame, splitFrames, type StreamEvent } from "./eventStream";

/** Build a response whose body streams the given chunks, then ends. */
function streaming(chunks: string[], status = 200): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(status === 200 ? body : null, { status });
}

const frame = (id: number, action: string) => `id: ${id}\nevent: ${action}\ndata: {"sequence":${id}}\n\n`;

describe("splitFrames", () => {
  it("holds back a partial frame rather than emitting truncated JSON", () => {
    // The failure this prevents: a chunk boundary mid-frame delivers half a
    // JSON document, which parses as malformed and is indistinguishable from a
    // server bug.
    const { frames, rest } = splitFrames(`${frame(1, "a")}id: 2\ndata: {"seq`);

    expect(frames).toHaveLength(1);
    expect(rest).toBe(`id: 2\ndata: {"seq`);
  });

  it("accepts CRLF line endings", () => {
    const { frames } = splitFrames("id: 1\r\ndata: {}\r\n\r\n");

    expect(frames).toHaveLength(1);
  });
});

describe("parseFrame", () => {
  it("decodes id, event and data", () => {
    expect(parseFrame(`id: 7\nevent: workload.submitted\ndata: {"a":1}`)).toEqual({
      id: "7",
      event: "workload.submitted",
      data: { a: 1 },
    });
  });

  it("ignores the server's keep-alive frame, byte for byte", () => {
    // Keep-alives hold the connection open through proxies, and the server emits
    // one every 15 seconds when nothing has happened. Treating one as an event
    // would refresh the whole dashboard on a timer -- worse than the fixed poll
    // this replaces. The literal is `KEEPALIVE_FRAME` from eyvan/server/api/sse.py;
    // matching the real bytes is what makes this a contract rather than a guess.
    const { frames, rest } = splitFrames(": keep-alive\n\n");

    expect(frames.map(parseFrame)).toEqual([undefined]);
    expect(rest).toBe("");
  });

  it("drops an unparseable frame instead of throwing", () => {
    // Throwing here would kill the connection, trading one missed refresh for
    // no further refreshes at all.
    expect(parseFrame("data: {not json")).toBeUndefined();
  });
});

describe("followEventStream", () => {
  const options = (fetchImpl: typeof fetch, onEvent: (event: StreamEvent) => void, signal: AbortSignal) => ({
    url: "/api/v1/events/stream",
    headers: { Authorization: "Bearer t" },
    signal,
    onEvent,
    fetchImpl,
    delay: async () => {},
  });

  it("carries the auth header the browser EventSource cannot send", async () => {
    // The reason this module exists rather than `new EventSource(url)`.
    const controller = new AbortController();
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      controller.abort();
      return streaming([frame(1, "workload.submitted")]);
    });

    await followEventStream(options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal));

    const init = fetchImpl.mock.calls[0][1] as RequestInit;
    expect((init.headers as Record<string, string>).Authorization).toBe("Bearer t");
    expect((init.headers as Record<string, string>).Accept).toBe("text/event-stream");
  });

  it("delivers each event once, across chunk boundaries", async () => {
    const controller = new AbortController();
    const seen: StreamEvent[] = [];
    const whole = frame(1, "a") + frame(2, "b");
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      controller.abort();
      return streaming([whole.slice(0, 20), whole.slice(20)]);
    });

    await followEventStream(
      options(fetchImpl as unknown as typeof fetch, (event) => seen.push(event), controller.signal),
    );

    expect(seen.map((event) => event.id)).toEqual(["1", "2"]);
  });

  it("resumes from the last id it delivered", async () => {
    // Without this a reconnect replays the tenant's whole audit history, which
    // on a busy plane is both a stampede and a refresh storm.
    const controller = new AbortController();
    let call = 0;
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      call += 1;
      if (call === 1) return streaming([frame(41, "a"), frame(42, "b")]);
      controller.abort();
      return streaming([]);
    });

    await followEventStream(options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal));

    const second = fetchImpl.mock.calls[1][1] as RequestInit;
    expect((second.headers as Record<string, string>)["Last-Event-ID"]).toBe("42");
  });

  it("does not send Last-Event-ID before anything has been delivered", () => {
    // Sending an empty header would make the server parse "" as an integer and
    // reject the very first connection.
    const controller = new AbortController();
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      controller.abort();
      return streaming([]);
    });

    return followEventStream(options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal)).then(() => {
      const init = fetchImpl.mock.calls[0][1] as RequestInit;
      expect((init.headers as Record<string, string>)["Last-Event-ID"]).toBeUndefined();
    });
  });

  it("reconnects after the control plane refuses, rather than giving up", async () => {
    // A restarting control plane is ordinary. An implementation that exited on
    // the first non-200 would leave the page frozen until someone reloaded it.
    const controller = new AbortController();
    let call = 0;
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      call += 1;
      if (call === 1) return streaming([], 503);
      controller.abort();
      return streaming([frame(1, "a")]);
    });

    await followEventStream(options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal));

    expect(fetchImpl).toHaveBeenCalledTimes(2);
  });

  it.each([401, 403])(
    "surfaces terminal authorization status %s instead of retrying a forbidden read",
    async (status) => {
      const controller = new AbortController();
      const fetchImpl = vi.fn(async () => streaming([], status));

      await expect(
        followEventStream(options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal)),
      ).rejects.toMatchObject({ status, name: "EventStreamHttpError" });
      expect(fetchImpl).toHaveBeenCalledTimes(1);
    },
  );

  it("reports the connection dropping and coming back", async () => {
    // The indicator exists because a dead stream and an idle fleet look the
    // same on the page.
    const controller = new AbortController();
    const states: boolean[] = [];
    let call = 0;
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      call += 1;
      if (call === 1) return streaming([frame(1, "a")]);
      controller.abort();
      return streaming([]);
    });

    await followEventStream({
      ...options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal),
      onConnectionChange: (connected) => states.push(connected),
    });

    expect(states.slice(0, 3)).toEqual([true, false, true]);
    expect(states.at(-1)).toBe(false);
  });

  it("stops when aborted instead of reconnecting forever", async () => {
    const controller = new AbortController();
    const fetchImpl = vi.fn(async (_url: string, _init?: RequestInit) => {
      controller.abort();
      throw new Error("aborted");
    });

    await followEventStream(options(fetchImpl as unknown as typeof fetch, () => {}, controller.signal));

    expect(fetchImpl).toHaveBeenCalledTimes(1);
  });
});
