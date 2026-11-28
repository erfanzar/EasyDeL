// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

export interface MetricSubgroup {
  id: string;
  label: string;
  keys: string[];
}

export interface MetricGroup {
  id: string;
  label: string;
  priority: number;
  defaultExpanded: boolean;
  noisy: boolean;
  keys: string[];
  subgroups: MetricSubgroup[];
}

export const DEFAULT_EXPANDED_MAX_KEYS = 24;
export const INITIAL_GROUP_PAGE_SIZE = 12;
export const GROUP_PAGE_INCREMENT = 24;
export const SUBGROUP_THRESHOLD = 12;

const TERMINAL_RUN_STATUSES = new Set(["finished", "failed", "crashed", "cancelled", "killed", "aborted"]);

/** True when a run has reached a terminal lifecycle state and will not emit new points. */
export function isTerminalRunStatus(status: string | null | undefined): boolean {
  if (!status) return false;
  return TERMINAL_RUN_STATUSES.has(status.toLowerCase());
}

/** Numeric-aware comparator so layer 2 sorts before layer 10. */
export function naturalCompare(a: string, b: string): number {
  return a.localeCompare(b, undefined, { numeric: true, sensitivity: "base" });
}

interface GroupRule {
  id: string;
  label: string;
  priority: number;
  noisy: boolean;
}

const KNOWN_GROUPS: Record<string, GroupRule> = {
  rl: { id: "rl", label: "RL Health & IcePop", priority: 10, noisy: false },
  agent_flow: { id: "agent_flow", label: "AgentFlow", priority: 20, noisy: false },
  reward: { id: "reward", label: "Rewards", priority: 25, noisy: false },
  loss: { id: "loss", label: "Loss", priority: 28, noisy: false },
  clip_ratio: { id: "clip_ratio", label: "Policy Clipping", priority: 30, noisy: false },
  train: { id: "train", label: "Train", priority: 40, noisy: false },
  eval: { id: "eval", label: "Eval", priority: 50, noisy: false },
  esurge: { id: "esurge", label: "eSurge", priority: 60, noisy: false },
  policy: { id: "policy", label: "Policy", priority: 65, noisy: false },
  groups: { id: "groups", label: "Completion Groups", priority: 66, noisy: false },
  performance: { id: "performance", label: "Performance", priority: 800, noisy: true },
  grad_norm: { id: "grad_norm", label: "Gradient Norms", priority: 900, noisy: true },
  parameters: { id: "parameters", label: "Weights & Parameters", priority: 950, noisy: true },
};

function formatCustomGroupLabel(prefix: string): string {
  if (!prefix || prefix === "metrics") return "Metrics";
  return prefix
    .split(/[_-]+/)
    .filter(Boolean)
    .map((part) => part[0].toUpperCase() + part.slice(1))
    .join(" ");
}

/** Classify a scalar metric key into its semantic group. */
export function classifyMetricKey(key: string): GroupRule {
  const lower = key.toLowerCase();

  // High-signal RL & AgentFlow namespaces (both top-level and under train/)
  if (lower.startsWith("train/rl/") || lower.startsWith("rl/")) return KNOWN_GROUPS.rl;
  if (lower.startsWith("train/agent_flow/") || lower.startsWith("agent_flow/")) return KNOWN_GROUPS.agent_flow;
  if (lower.startsWith("train/clip_ratio/") || lower.startsWith("clip_ratio/")) return KNOWN_GROUPS.clip_ratio;

  // Massive / noisy namespaces (both top-level and nested under train/ or eval/)
  if (
    lower.startsWith("grad_norm/") ||
    lower.startsWith("grad_norms/") ||
    lower.startsWith("train/grad_norm/") ||
    lower.startsWith("eval/grad_norm/")
  ) {
    return KNOWN_GROUPS.grad_norm;
  }
  if (
    lower.startsWith("weight_distribution/") ||
    lower.startsWith("parameters/") ||
    lower.startsWith("weights/") ||
    lower.startsWith("params/") ||
    lower.startsWith("train/weight_distribution/") ||
    lower.startsWith("train/parameters/")
  ) {
    return KNOWN_GROUPS.parameters;
  }
  if (
    lower.startsWith("performance/") ||
    lower.startsWith("train/performance/") ||
    lower.startsWith("eval/performance/")
  ) {
    return KNOWN_GROUPS.performance;
  }

  // Serving / inference
  if (lower.startsWith("esurge/") || lower.startsWith("train/esurge/")) return KNOWN_GROUPS.esurge;

  // Core train / eval / reward / loss / policy / groups
  if (lower.startsWith("train/")) return KNOWN_GROUPS.train;
  if (lower.startsWith("eval/") || lower.startsWith("validation/")) return KNOWN_GROUPS.eval;
  if (lower.startsWith("reward/") || lower.startsWith("rewards/")) return KNOWN_GROUPS.reward;
  if (lower.startsWith("loss/")) return KNOWN_GROUPS.loss;
  if (lower.startsWith("policy/")) return KNOWN_GROUPS.policy;
  if (lower.startsWith("groups/")) return KNOWN_GROUPS.groups;

  // Generic top-level prefix
  const slash = key.indexOf("/");
  if (slash > 0) {
    const prefix = key.slice(0, slash);
    const id = prefix.toLowerCase();
    return {
      id,
      label: formatCustomGroupLabel(prefix),
      priority: 100,
      noisy: false,
    };
  }
  return { id: "metrics", label: "Metrics", priority: 90, noisy: false };
}

/** Classify a histogram key so RL distributions always surface ahead of per-layer parameter histograms. */
export function classifyHistogramKey(key: string): GroupRule {
  const lower = key.toLowerCase();
  if (
    lower.startsWith("rl/") ||
    lower.startsWith("train/rl/") ||
    lower === "advantages" ||
    lower.endsWith("/advantages") ||
    lower.includes("reward_dist") ||
    lower.includes("advantage_dist") ||
    lower.includes("icepop_ratio") ||
    lower.includes("episode_turns") ||
    lower.includes("completion_length")
  ) {
    return { id: "rl", label: "RL Distributions", priority: 10, noisy: false };
  }
  if (lower.startsWith("agent_flow/") || lower.startsWith("train/agent_flow/")) {
    return { id: "agent_flow", label: "AgentFlow Distributions", priority: 20, noisy: false };
  }
  if (lower.startsWith("policy/")) {
    return { id: "policy", label: "Policy", priority: 30, noisy: false };
  }
  if (
    lower.startsWith("parameters/") ||
    lower.startsWith("weight_distribution/") ||
    lower.startsWith("weights/") ||
    lower.startsWith("params/")
  ) {
    return KNOWN_GROUPS.parameters;
  }
  if (lower.startsWith("grad_norm/") || lower.startsWith("grad_norms/")) {
    return KNOWN_GROUPS.grad_norm;
  }
  return classifyMetricKey(key);
}

const LAYER_INDEX_RE = /(?:^|[./_-])(?:layers?|blocks?|h)[./](\d+)(?:[./]|$)/i;

/** Build sub-groups for large metric groups (layer buckets for transformer params/grads, or sub-namespace). */
function buildSubgroups(groupId: string, keys: string[]): MetricSubgroup[] {
  if (keys.length <= SUBGROUP_THRESHOLD) return [];

  const layerMatches = new Map<number, string[]>();
  const nonLayerKeys: string[] = [];

  for (const key of keys) {
    const match = LAYER_INDEX_RE.exec(key);
    if (match) {
      const idx = Number.parseInt(match[1], 10);
      const bucket = layerMatches.get(idx);
      if (bucket) bucket.push(key);
      else layerMatches.set(idx, [key]);
    } else {
      nonLayerKeys.push(key);
    }
  }

  // If keys map cleanly to transformer layers/blocks, bucket them into crisp ranges.
  if (layerMatches.size >= 2) {
    const indices = [...layerMatches.keys()].sort((a, b) => a - b);
    const maxLayer = indices[indices.length - 1];
    const bucketSize = maxLayer >= 16 ? 8 : maxLayer >= 8 ? 4 : 1;
    const buckets = new Map<number, { start: number; end: number; keys: string[] }>();

    for (const idx of indices) {
      const bucketIdx = Math.floor(idx / bucketSize);
      const start = bucketIdx * bucketSize;
      const end = Math.min(maxLayer, start + bucketSize - 1);
      const existing = buckets.get(bucketIdx);
      const layerKeys = layerMatches.get(idx)!;
      if (existing) existing.keys.push(...layerKeys);
      else buckets.set(bucketIdx, { start, end, keys: [...layerKeys] });
    }

    const result: MetricSubgroup[] = [];
    if (nonLayerKeys.length) {
      result.push({
        id: "non-layer",
        label: "Embeddings & Head",
        keys: nonLayerKeys,
      });
    }
    for (const [bucketIdx, bucket] of [...buckets.entries()].sort((a, b) => a[0] - b[0])) {
      result.push({
        id: `layers-${bucketIdx}`,
        label: bucket.start === bucket.end ? `Layer ${bucket.start}` : `Layers ${bucket.start}–${bucket.end}`,
        keys: bucket.keys,
      });
    }
    return result.length >= 2 ? result : [];
  }

  // Otherwise group by secondary namespace segment after the primary prefix.
  const segments = new Map<string, string[]>();
  for (const key of keys) {
    const parts = key.split("/");
    let sub = "general";
    if (parts.length >= 3) {
      sub = parts[0].toLowerCase() === groupId ? parts[1] : parts[1] === groupId ? parts[2] : parts[1];
    } else if (parts.length === 2) {
      const dot = parts[1].indexOf(".");
      if (dot > 0) sub = parts[1].slice(0, dot);
    }
    const list = segments.get(sub);
    if (list) list.push(key);
    else segments.set(sub, [key]);
  }

  if (segments.size < 2 || segments.size > 16) return [];
  return [...segments.entries()]
    .sort(([a], [b]) => naturalCompare(a, b))
    .map(([sub, subKeys]) => ({
      id: sub.toLowerCase(),
      label: formatCustomGroupLabel(sub),
      keys: subKeys,
    }));
}

function assembleGroups(keys: readonly string[], classify: (key: string) => GroupRule): MetricGroup[] {
  const map = new Map<string, { rule: GroupRule; keys: string[] }>();
  const seen = new Set<string>();

  for (const key of keys) {
    if (seen.has(key)) continue;
    seen.add(key);
    const rule = classify(key);
    const existing = map.get(rule.id);
    if (existing) {
      existing.keys.push(key);
    } else {
      map.set(rule.id, { rule, keys: [key] });
    }
  }

  return [...map.values()]
    .sort((a, b) => {
      if (a.rule.priority !== b.rule.priority) return a.rule.priority - b.rule.priority;
      return naturalCompare(a.rule.label, b.rule.label);
    })
    .map(({ rule, keys: groupKeys }) => {
      const sortedKeys = [...groupKeys].sort(naturalCompare);
      return {
        id: rule.id,
        label: rule.label,
        priority: rule.priority,
        noisy: rule.noisy,
        defaultExpanded: !rule.noisy && sortedKeys.length <= DEFAULT_EXPANDED_MAX_KEYS,
        keys: sortedKeys,
        subgroups: buildSubgroups(rule.id, sortedKeys),
      };
    });
}

/** Group and priority-order scalar metric keys for the Charts tab. */
export function groupMetricKeys(keys: readonly string[]): MetricGroup[] {
  return assembleGroups(keys, classifyMetricKey);
}

/** Group and priority-order histogram keys for the Histograms tab. */
export function groupHistogramKeys(keys: readonly string[]): MetricGroup[] {
  return assembleGroups(keys, classifyHistogramKey);
}

/**
 * Filter grouped metrics by a free-text search query (matches group label, group id, or key path).
 * Space-separated terms are AND-matched so queries like "layer 0 q_proj" narrow precisely.
 * When keys within a group match all query terms directly (e.g. "icepop"), prefer those exact
 * key matches instead of pulling in unrelated siblings that only matched the group's display title.
 */
export function filterMetricGroups(groups: readonly MetricGroup[], rawQuery: string): MetricGroup[] {
  const terms = rawQuery.trim().toLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return [...groups];

  const matched: MetricGroup[] = [];
  for (const group of groups) {
    const directKeyMatches = group.keys.filter((key) => {
      const lowerKey = key.toLowerCase();
      return terms.every((term) => lowerKey.includes(term));
    });
    const groupHaystack = `${group.label} ${group.id}`.toLowerCase();
    const matchingKeys = directKeyMatches.length
      ? directKeyMatches
      : group.keys.filter((key) => {
          const combined = `${groupHaystack} ${key.toLowerCase()}`;
          return terms.every((term) => combined.includes(term));
        });
    if (!matchingKeys.length) continue;
    matched.push({
      ...group,
      keys: matchingKeys,
      subgroups: buildSubgroups(group.id, matchingKeys),
    });
  }
  return matched;
}

/** Format a numeric metric value cleanly for chart headers, axes, and hover readouts. */
export function formatChartValue(value: number | null | undefined, compact = false): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs < 1e-3 || abs >= 1e6) return value.toExponential(compact ? 1 : 2);
  if (abs >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: compact ? 0 : 2 });
  if (abs >= 1) return value.toLocaleString(undefined, { maximumFractionDigits: compact ? 2 : 4 });
  return value.toLocaleString(undefined, { maximumFractionDigits: compact ? 3 : 5 });
}

/** Split a metric key into muted namespace prefix and bold leaf label for scanability. */
export function splitMetricKey(key: string): { prefix: string; leaf: string } {
  const lastSlash = key.lastIndexOf("/");
  if (lastSlash <= 0) return { prefix: "", leaf: key };
  return { prefix: key.slice(0, lastSlash + 1), leaf: key.slice(lastSlash + 1) };
}
