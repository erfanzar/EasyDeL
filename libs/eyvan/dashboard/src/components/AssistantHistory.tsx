// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { AlertTriangle, MessageSquare, Pencil, Trash2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import type { AssistantConversationSummary } from "../types";

/** Methods the history list calls -- always as `client.method(...)`. */
export interface AssistantHistoryClient {
  listAssistantConversations(signal?: AbortSignal): Promise<AssistantConversationSummary[]>;
  renameAssistantConversation(id: string, title: string): Promise<AssistantConversationSummary>;
  deleteAssistantConversation(id: string): Promise<{ deleted: string }>;
}

export type HistoryGroup = "Today" | "Yesterday" | "Earlier";
const GROUPS: HistoryGroup[] = ["Today", "Yesterday", "Earlier"];
const DAY_MS = 86_400_000;

/** Which heading a conversation sits under, by its last change in local time. */
export function historyGroup(iso: string, now: Date = new Date()): HistoryGroup {
  const when = Date.parse(iso);
  if (!Number.isFinite(when)) return "Earlier";
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  if (when >= today) return "Today";
  if (when >= today - DAY_MS) return "Yesterday";
  return "Earlier";
}

export function relativeTime(iso: string, now: Date = new Date()): string {
  const when = Date.parse(iso);
  if (!Number.isFinite(when)) return "";
  const seconds = Math.max(0, Math.round((now.getTime() - when) / 1000));
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86_400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 7 * 86_400) return `${Math.floor(seconds / 86_400)} d ago`;
  return new Date(when).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export const conversationTitle = (conversation: Pick<AssistantConversationSummary, "title">) =>
  conversation.title.trim() || "Untitled chat";

const message = (error: unknown, fallback: string) => (error instanceof Error ? error.message : fallback);

export function AssistantHistory({
  client,
  activeId,
  revision,
  onOpen,
  onDeleted,
}: {
  client: AssistantHistoryClient;
  activeId: string | null;
  /** Bumped when a turn saves, so a reopened list is current. */
  revision: number;
  onOpen: (id: string) => void;
  onDeleted: (id: string) => void;
}) {
  const [conversations, setConversations] = useState<AssistantConversationSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    Promise.resolve()
      .then(() => client.listAssistantConversations(controller.signal))
      .then((value) => {
        if (!controller.signal.aborted) {
          setConversations(value);
          setError(null);
        }
      })
      .catch((failure: unknown) => {
        if (!controller.signal.aborted) {
          setConversations((current) => current ?? []);
          setError(message(failure, "Could not read your chats"));
        }
      });
    return () => controller.abort();
  }, [client, revision]);

  const grouped = useMemo(() => {
    const now = new Date();
    return GROUPS.map((group) => ({
      group,
      items: (conversations ?? []).filter((item) => historyGroup(item.updated_at, now) === group),
    })).filter((entry) => entry.items.length > 0);
  }, [conversations]);

  const rename = async () => {
    if (!renaming) return;
    const title = renaming.title.trim();
    if (!title) return setRenaming(null);
    setBusyId(renaming.id);
    setError(null);
    try {
      const saved = await client.renameAssistantConversation(renaming.id, title);
      setConversations(
        (current) => current?.map((item) => (item.id === saved.id ? { ...item, ...saved } : item)) ?? null,
      );
      setRenaming(null);
    } catch (failure) {
      setError(message(failure, "The chat was not renamed"));
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (id: string) => {
    setBusyId(id);
    setError(null);
    try {
      await client.deleteAssistantConversation(id);
      setConversations((current) => current?.filter((item) => item.id !== id) ?? null);
      setConfirming(null);
      onDeleted(id);
    } catch (failure) {
      setError(message(failure, "The chat was not deleted"));
    } finally {
      setBusyId(null);
    }
  };

  const now = new Date();
  return (
    <div className="dock-history" role="region" aria-label="Saved chats">
      {error && (
        <div className="dock-alert" role="alert">
          <AlertTriangle size={14} aria-hidden="true" />
          <span>{error}</span>
        </div>
      )}
      {conversations === null && <p className="dock-hint">Reading your chats…</p>}
      {conversations?.length === 0 && !error && (
        <div className="dock-empty">
          <MessageSquare size={18} aria-hidden="true" />
          <span>No saved chats yet.</span>
        </div>
      )}
      {grouped.map(({ group, items }) => (
        <div key={group} className="dock-history-group" role="group" aria-label={group}>
          <span className="dock-history-heading" aria-hidden="true">
            {group}
          </span>
          <ul className="dock-history-list">
            {items.map((item) => {
              const title = conversationTitle(item);
              const busy = busyId === item.id;
              const active = item.id === activeId;
              return (
                <li
                  key={item.id}
                  className={`dock-history-item ${active ? "dock-history-item-active" : ""}`}
                  aria-label={title}
                >
                  {renaming?.id === item.id ? (
                    <form
                      className="dock-history-rename"
                      onSubmit={(event) => {
                        event.preventDefault();
                        void rename();
                      }}
                    >
                      <input
                        aria-label="Chat title"
                        autoFocus
                        maxLength={200}
                        value={renaming.title}
                        disabled={busy}
                        onChange={(event) => setRenaming({ id: item.id, title: event.target.value })}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            // The rename owns this Escape; the dock stays open.
                            event.preventDefault();
                            event.stopPropagation();
                            setRenaming(null);
                          }
                        }}
                      />
                      <button className="button button-primary" type="submit" disabled={busy}>
                        Save title
                      </button>
                      <button
                        className="button button-secondary"
                        type="button"
                        disabled={busy}
                        onClick={() => setRenaming(null)}
                      >
                        Cancel rename
                      </button>
                    </form>
                  ) : confirming === item.id ? (
                    <div className="dock-history-confirm">
                      <span>Delete this chat?</span>
                      <button
                        className="button button-primary"
                        type="button"
                        disabled={busy}
                        onClick={() => void remove(item.id)}
                      >
                        Confirm delete
                      </button>
                      <button
                        className="button button-secondary"
                        type="button"
                        disabled={busy}
                        onClick={() => setConfirming(null)}
                      >
                        Cancel
                      </button>
                    </div>
                  ) : (
                    <>
                      <button
                        className="dock-history-open"
                        type="button"
                        aria-label={`Open chat ${title}`}
                        aria-current={active ? "true" : undefined}
                        title={title}
                        onClick={() => onOpen(item.id)}
                      >
                        <span className="dock-history-title">{title}</span>
                        <time className="dock-history-time" dateTime={item.updated_at}>
                          {relativeTime(item.updated_at, now)}
                        </time>
                      </button>
                      <span className="dock-history-actions">
                        <button
                          className="icon-button icon-button-small"
                          type="button"
                          aria-label={`Rename ${title}`}
                          title="Rename"
                          onClick={() => {
                            setConfirming(null);
                            setRenaming({ id: item.id, title: item.title });
                          }}
                        >
                          <Pencil size={12} aria-hidden="true" />
                        </button>
                        <button
                          className="icon-button icon-button-small"
                          type="button"
                          aria-label={`Delete ${title}`}
                          title="Delete"
                          onClick={() => {
                            setRenaming(null);
                            setConfirming(item.id);
                          }}
                        >
                          <Trash2 size={12} aria-hidden="true" />
                        </button>
                      </span>
                    </>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </div>
  );
}
