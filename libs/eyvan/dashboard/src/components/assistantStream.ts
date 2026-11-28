// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import type { AssistantConversationMessage, AssistantStreamEvent } from "../types";

/*
 * How one streamed assistant turn becomes a message that grows in place. Pure
 * functions, so the order rules -- text before a tool call was narration, text
 * after the last step is the answer -- are testable without a DOM.
 */

export type ToolStatus = "running" | "ok" | "failed" | "interrupted";

export type TraceEntry =
  | { kind: "thinking"; text: string }
  | { kind: "narration"; text: string }
  | {
      kind: "tool";
      id: string;
      name: string;
      arguments: Record<string, unknown> | null;
      status: ToolStatus;
      summary?: string;
    };

export type AnswerStatus = "streaming" | "done" | "stopped" | "error";

export interface AssistantAnswer {
  kind: "assistant";
  id: string;
  /** The current round's text: the answer once the turn is done. */
  content: string;
  trace: TraceEntry[];
  status: AnswerStatus;
  step?: number;
  maxSteps?: number;
  /** Rounds the turn took, once known. */
  steps?: number;
  error?: string;
  model?: string;
}

export const streamingAnswer = (id: string, model?: string): AssistantAnswer => ({
  kind: "assistant",
  id,
  content: "",
  trace: [],
  status: "streaming",
  model,
});

const interruptTools = (trace: TraceEntry[]): TraceEntry[] =>
  trace.map(
    (entry): TraceEntry =>
      entry.kind === "tool" && entry.status === "running" ? { ...entry, status: "interrupted" } : entry,
  );

/** Fold one stream event into the message it belongs to. */
export function applyStreamEvent(answer: AssistantAnswer, event: AssistantStreamEvent): AssistantAnswer {
  switch (event.type) {
    case "start":
      return { ...answer, model: event.model ?? answer.model };
    case "step":
      return { ...answer, step: event.step, maxSteps: event.max_steps };
    case "thinking": {
      const last = answer.trace[answer.trace.length - 1];
      if (last?.kind === "thinking") {
        return {
          ...answer,
          trace: [...answer.trace.slice(0, -1), { kind: "thinking", text: last.text + event.delta }],
        };
      }
      return { ...answer, trace: [...answer.trace, { kind: "thinking", text: event.delta }] };
    }
    case "text":
      return { ...answer, content: answer.content + event.delta };
    case "tool_call": {
      // Text of a round that ends in tool calls was narration, not the answer.
      const trace: TraceEntry[] = answer.content.trim()
        ? [...answer.trace, { kind: "narration", text: answer.content.trim() }]
        : [...answer.trace];
      const tool: TraceEntry = {
        kind: "tool",
        id: event.id,
        name: event.name,
        arguments: event.arguments ?? null,
        status: "running",
      };
      const known = trace.findIndex((entry) => entry.kind === "tool" && entry.id === event.id);
      if (known >= 0) trace[known] = tool;
      else trace.push(tool);
      return { ...answer, content: "", trace };
    }
    case "tool_result": {
      const trace = [...answer.trace];
      let index = trace.findIndex((entry) => entry.kind === "tool" && entry.id === event.id);
      if (index < 0) {
        index = trace.findIndex(
          (entry) => entry.kind === "tool" && entry.name === event.name && entry.status === "running",
        );
      }
      const status: ToolStatus = event.ok ? "ok" : "failed";
      if (index >= 0) {
        const entry = trace[index] as Extract<TraceEntry, { kind: "tool" }>;
        trace[index] = { ...entry, status, summary: event.summary };
      } else {
        trace.push({ kind: "tool", id: event.id, name: event.name, arguments: null, status, summary: event.summary });
      }
      return { ...answer, trace };
    }
    case "done":
      return {
        ...answer,
        status: "done",
        content: event.reply || answer.content,
        steps: event.steps,
        trace: interruptTools(answer.trace),
      };
    case "error":
      return finishAnswer(answer, "error", event.detail || "The assistant could not answer");
    default:
      return answer;
  }
}

/** End a turn that will receive no more events: stopped by the user, or failed. */
export function finishAnswer(answer: AssistantAnswer, status: "stopped" | "error", error?: string): AssistantAnswer {
  return { ...answer, status, error: status === "error" ? error : undefined, trace: interruptTools(answer.trace) };
}

const clip = (text: string, limit: number) => (text.length > limit ? `${text.slice(0, limit - 1)}…` : text);

/** `key=value` pairs for a tool row: short enough for one line, never the whole payload. */
export function argumentSummary(args: Record<string, unknown> | null | undefined, limit = 80): string {
  if (!args) return "";
  const parts = Object.entries(args).map(([key, value]) => {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    return `${key}=${clip(text ?? "null", 32)}`;
  });
  return clip(parts.join(" "), limit);
}

export const toolCount = (answer: AssistantAnswer) => answer.trace.filter((entry) => entry.kind === "tool").length;

const plural = (count: number, word: string) => `${count} ${word}${count === 1 ? "" : "s"}`;

/** The one-line expander a finished turn keeps above its answer: "Used 3 tools · 2 steps". */
export function traceLabel(answer: AssistantAnswer): string {
  const tools = toolCount(answer);
  const head = tools ? `Used ${plural(tools, "tool")}` : "Show reasoning";
  return answer.steps ? `${head} · ${plural(answer.steps, "step")}` : head;
}

/** A saved assistant message, rendered as a finished turn. */
export function answerFromSaved(message: AssistantConversationMessage): AssistantAnswer {
  return {
    kind: "assistant",
    id: message.id,
    content: message.content,
    status: "done",
    model: message.model ?? undefined,
    trace: (message.trace ?? [])
      .filter((entry) => entry.type === "tool")
      .map(
        (entry, index): TraceEntry => ({
          kind: "tool",
          id: `${message.id}-tool-${index}`,
          name: entry.name,
          arguments: entry.arguments ?? null,
          status: entry.ok ? "ok" : "failed",
          summary: entry.summary,
        }),
      ),
  };
}
