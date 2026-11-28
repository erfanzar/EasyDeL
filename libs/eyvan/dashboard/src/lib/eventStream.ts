// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

export class EventStreamHttpError extends Error {
  constructor(
    message: string,
    public readonly status: number,
  ) {
    super(message);
    this.name = "EventStreamHttpError";
  }
}

/** One decoded server-sent event. */
export interface StreamEvent {
  id?: string;
  event: string;
  data: unknown;
}

export interface EventStreamOptions {
  url: string;
  headers: Record<string, string>;
  signal: AbortSignal;
  onEvent: (event: StreamEvent) => void;
  /** Called with true once connected and false when the connection drops. */
  onConnectionChange?: (connected: boolean) => void;
  /** Injected in tests; defaults to `globalThis.fetch`. */
  fetchImpl?: typeof fetch;
  /** Injected in tests so backoff does not make the suite wait. */
  delay?: (ms: number) => Promise<void>;
}

/** Reconnect backoff, in milliseconds. Capped so a long outage still recovers promptly. */
const BACKOFF_MS = [1_000, 2_000, 5_000, 10_000, 30_000];

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => {
    globalThis.setTimeout(resolve, ms);
  });

/**
 * Split a raw SSE chunk buffer into complete frames, returning the remainder.
 *
 * A frame ends at a blank line. Anything after the last blank line is a partial
 * frame: emitting it would deliver truncated JSON, which is indistinguishable
 * from a malformed event.
 */
export function splitFrames(buffer: string): { frames: string[]; rest: string } {
  const normalized = buffer.replace(/\r\n/g, "\n");
  const parts = normalized.split("\n\n");
  const rest = parts.pop() ?? "";
  return { frames: parts.filter((part) => part.trim() !== ""), rest };
}

/**
 * Decode one frame, or return undefined when it carries no data.
 *
 * Keep-alive frames are comments (`: keepalive`) and carry no `data:` field;
 * they exist to hold the connection open through proxies and are not events.
 */
export function parseFrame(frame: string): StreamEvent | undefined {
  let id: string | undefined;
  let event = "message";
  const dataLines: string[] = [];
  for (const line of frame.split("\n")) {
    // Spec conformance, and deliberately not covered by a test: a comment's
    // field name parses as "" and matches none of the three below, so removing
    // this line changes nothing observable today. It is kept because the
    // implication is an accident of the dispatch, and one `else` clause added
    // later would turn every keep-alive into an event.
    if (line.startsWith(":")) continue;
    const separator = line.indexOf(":");
    const field = separator === -1 ? line : line.slice(0, separator);
    const value = separator === -1 ? "" : line.slice(separator + 1).replace(/^ /, "");
    if (field === "id") id = value;
    else if (field === "event") event = value;
    else if (field === "data") dataLines.push(value);
  }
  // No explicit "has no data field" branch: `JSON.parse("")` throws, so a
  // data-less frame already lands in the catch below. A separate guard read as
  // load-bearing while nothing could observe its absence, which is a comment
  // pretending to be code.
  try {
    return { id, event, data: JSON.parse(dataLines.join("\n")) };
  } catch {
    // A frame we cannot parse is dropped rather than thrown. The stream is a
    // change notification; killing the connection over one bad frame would
    // trade a missed refresh for no refreshes at all.
    return undefined;
  }
}

/**
 * Follow an SSE endpoint until `signal` aborts, reconnecting on failure.
 *
 * Resolves only when aborted. Every other exit reconnects, because the states
 * this survives -- the control plane restarting, a proxy closing an idle
 * connection, a laptop waking up -- are ordinary rather than exceptional.
 */
export async function followEventStream(options: EventStreamOptions): Promise<void> {
  const { url, headers, signal, onEvent, onConnectionChange } = options;
  const doFetch = options.fetchImpl ?? globalThis.fetch;
  const pause = options.delay ?? sleep;
  let lastEventId: string | undefined;
  let failures = 0;

  while (!signal.aborted) {
    try {
      const response = await doFetch(url, {
        headers: {
          ...headers,
          Accept: "text/event-stream",
          ...(lastEventId === undefined ? {} : { "Last-Event-ID": lastEventId }),
        },
        signal,
      });
      if (!response.ok) throw new EventStreamHttpError(`event stream returned ${response.status}`, response.status);
      if (response.body === null) throw new Error("event stream returned no response body");
      failures = 0;
      onConnectionChange?.(true);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const { frames, rest } = splitFrames(buffer);
        buffer = rest;
        for (const frame of frames) {
          const decoded = parseFrame(frame);
          if (decoded === undefined) continue;
          // Recorded even when the consumer throws, so a failing handler cannot
          // make the next reconnect replay the whole stream.
          if (decoded.id !== undefined) lastEventId = decoded.id;
          onEvent(decoded);
        }
      }
    } catch (error) {
      if (signal.aborted) break;
      if (error instanceof EventStreamHttpError && [401, 403].includes(error.status)) {
        // Retrying an authorization denial forever is both a forbidden-read
        // loop and a stale-capability bug. Surface it so App can resolve
        // /session and either resubscribe with current authority or stop.
        onConnectionChange?.(false);
        throw error;
      }
    }
    onConnectionChange?.(false);
    if (signal.aborted) break;
    await pause(BACKOFF_MS[Math.min(failures, BACKOFF_MS.length - 1)]);
    failures += 1;
  }
  onConnectionChange?.(false);
}
