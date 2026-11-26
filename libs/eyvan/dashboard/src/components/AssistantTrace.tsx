// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, Brain, ChevronRight, LoaderCircle, Wrench } from "lucide-react";
import { useState } from "react";

import { AssistantMarkdown } from "./AssistantMarkdown";
import { argumentSummary, traceLabel, type AssistantAnswer, type ToolStatus, type TraceEntry } from "./assistantStream";

/*
 * One assistant turn as it happens: which step it is on, each tool it runs and
 * whether that tool answered, thinking and narration between rounds, then the
 * answer as it arrives. Once the turn ends the trace folds into one line above
 * the answer, so a finished conversation reads as answers rather than logs.
 */

const STATUS_WORD: Record<ToolStatus, string> = {
  running: "running",
  ok: "succeeded",
  failed: "failed",
  interrupted: "interrupted",
};

const STATUS_MARK: Record<Exclude<ToolStatus, "running">, string> = {
  ok: "✓",
  failed: "✗",
  interrupted: "–",
};

type ToolEntry = Extract<TraceEntry, { kind: "tool" }>;

function ToolRow({ entry }: { entry: ToolEntry }) {
  const [open, setOpen] = useState(false);
  const args = argumentSummary(entry.arguments);
  return (
    <li className={`dock-tool-row dock-tool-row-${entry.status}`}>
      <button
        className="dock-tool-head"
        type="button"
        aria-expanded={open}
        aria-label={`Tool ${entry.name}: ${STATUS_WORD[entry.status]}`}
        title={args || entry.name}
        onClick={() => setOpen((value) => !value)}
      >
        <ChevronRight className="dock-tool-chevron" size={11} aria-hidden="true" />
        <Wrench size={11} aria-hidden="true" />
        <span className="dock-tool-name">{entry.name}</span>
        {args && <span className="dock-tool-args">{args}</span>}
        <span className="dock-tool-status" aria-hidden="true">
          {entry.status === "running" ? <LoaderCircle className="spin" size={11} /> : STATUS_MARK[entry.status]}
        </span>
      </button>
      {open && (
        <div className="dock-tool-detail">
          <span className="dock-tool-detail-label">Arguments</span>
          <pre>{entry.arguments ? JSON.stringify(entry.arguments, null, 2) : "none"}</pre>
          <span className="dock-tool-detail-label">Result</span>
          <p>{entry.summary || (entry.status === "running" ? "Running…" : "No result was reported.")}</p>
        </div>
      )}
    </li>
  );
}

function ThinkingBlock({ text }: { text: string }) {
  // Always starts collapsed, live or not: reasoning is there for whoever wants
  // it, not something every reader has to scroll past.
  const [open, setOpen] = useState(false);
  return (
    <li className="dock-thinking">
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        <ChevronRight className="dock-tool-chevron" size={11} aria-hidden="true" />
        <Brain size={11} aria-hidden="true" />
        Thinking
      </button>
      {open && <p>{text}</p>}
    </li>
  );
}

function TraceList({ answer }: { answer: AssistantAnswer }) {
  return (
    <ul className="dock-trace" aria-label="Assistant steps">
      {answer.trace.map((entry, index) => {
        if (entry.kind === "tool") return <ToolRow key={`tool-${entry.id}-${index}`} entry={entry} />;
        if (entry.kind === "thinking") return <ThinkingBlock key={`thinking-${index}`} text={entry.text} />;
        return (
          <li key={`narration-${index}`} className="dock-narration">
            <AssistantMarkdown text={entry.text} />
          </li>
        );
      })}
    </ul>
  );
}

export function AnswerView({ answer }: { answer: AssistantAnswer }) {
  const [expanded, setExpanded] = useState(false);
  const streaming = answer.status === "streaming";
  const hasTrace = answer.trace.length > 0;
  const content = answer.content || (answer.status === "done" ? "(no answer)" : "");
  return (
    <div
      className={`dock-message dock-message-assistant ${streaming ? "dock-answer-streaming" : ""}`}
      data-testid="assistant-answer"
      aria-busy={streaming}
    >
      {streaming && (
        <div className="dock-step" role="status">
          <LoaderCircle className="spin" size={11} aria-hidden="true" />
          <span>{answer.step ? `Step ${answer.step}/${answer.maxSteps ?? "?"}` : "Working…"}</span>
        </div>
      )}
      {hasTrace &&
        (streaming ? (
          <TraceList answer={answer} />
        ) : (
          <>
            <button
              className="dock-trace-toggle"
              type="button"
              aria-expanded={expanded}
              onClick={() => setExpanded((value) => !value)}
            >
              <ChevronRight className="dock-tool-chevron" size={11} aria-hidden="true" />
              {traceLabel(answer)}
            </button>
            {expanded && <TraceList answer={answer} />}
          </>
        ))}
      {content && <AssistantMarkdown text={content} />}
      {answer.status === "stopped" && <span className="dock-answer-note">Stopped</span>}
      {answer.status === "error" && (
        <div className="dock-inline-error" role="alert">
          <AlertTriangle size={13} aria-hidden="true" />
          <span>{answer.error}</span>
        </div>
      )}
    </div>
  );
}
