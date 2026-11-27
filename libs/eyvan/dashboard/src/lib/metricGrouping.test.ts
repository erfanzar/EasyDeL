// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

import { describe, expect, it } from "vitest";
import {
  classifyHistogramKey,
  classifyMetricKey,
  filterMetricGroups,
  groupHistogramKeys,
  groupMetricKeys,
  isTerminalRunStatus,
  splitMetricKey,
} from "./metricGrouping";

describe("metricGrouping", () => {
  it("groups nested namespaces hierarchically and orders high-signal groups ahead of noisy groups", () => {
    const keys = [
      "grad_norm/params/model.layers.10.mlp.gate_proj.kernel",
      "grad_norm/params/model.layers.2.mlp.gate_proj.kernel",
      "parameters/model.embed_tokens.embedding",
      "weight_distribution/model.layers.0.self_attn.q_proj",
      "performance/step_time",
      "train/performance/tokens_per_sec",
      "train/loss",
      "train/learning_rate",
      "train/rl/icepop_ratio_mean",
      "rl/pass_rate",
      "train/agent_flow/mean_reward",
      "agent_flow/mean_turns",
      "train/clip_ratio/high_mean",
      "eval/accuracy",
      "esurge/ttft_ms",
      "custom/foo",
    ];

    const groups = groupMetricKeys(keys);
    expect(groups.map((g) => g.id)).toEqual([
      "rl",
      "agent_flow",
      "clip_ratio",
      "train",
      "eval",
      "esurge",
      "custom",
      "performance",
      "grad_norm",
      "parameters",
    ]);

    // High-signal groups default expanded; noisy groups default collapsed.
    const byId = new Map(groups.map((g) => [g.id, g]));
    expect(byId.get("rl")?.defaultExpanded).toBe(true);
    expect(byId.get("agent_flow")?.defaultExpanded).toBe(true);
    expect(byId.get("train")?.defaultExpanded).toBe(true);
    expect(byId.get("eval")?.defaultExpanded).toBe(true);
    expect(byId.get("performance")?.defaultExpanded).toBe(false);
    expect(byId.get("grad_norm")?.defaultExpanded).toBe(false);
    expect(byId.get("parameters")?.defaultExpanded).toBe(false);

    // Natural numeric sorting puts layer 2 before layer 10.
    expect(byId.get("grad_norm")?.keys).toEqual([
      "grad_norm/params/model.layers.2.mlp.gate_proj.kernel",
      "grad_norm/params/model.layers.10.mlp.gate_proj.kernel",
    ]);
  });

  it("collapses non-noisy groups that exceed 24 keys by default", () => {
    const keys = Array.from({ length: 26 }, (_, i) => `custom/metric_${i}`);
    const [group] = groupMetricKeys(keys);
    expect(group.defaultExpanded).toBe(false);
    expect(group.keys).toHaveLength(26);
  });

  it("sub-groups large per-layer grad_norm namespaces into layer blocks plus embeddings/head", () => {
    const keys = [
      "grad_norm/model.embed_tokens.embedding",
      "grad_norm/lm_head.kernel",
      ...Array.from({ length: 24 }, (_, layer) => [
        `grad_norm/model.layers.${layer}.self_attn.q_proj.kernel`,
        `grad_norm/model.layers.${layer}.mlp.gate_proj.kernel`,
      ]).flat(),
    ];
    const [group] = groupMetricKeys(keys);
    expect(group.id).toBe("grad_norm");
    expect(group.subgroups.map((s) => s.label)).toEqual([
      "Embeddings & Head",
      "Layers 0–7",
      "Layers 8–15",
      "Layers 16–23",
    ]);
    expect(group.subgroups[0].keys).toHaveLength(2);
    expect(group.subgroups[1].keys).toHaveLength(16);
  });

  it("filters groups by multi-term queries matching group names and key paths", () => {
    const groups = groupMetricKeys([
      "train/rl/icepop_ratio_mean",
      "train/rl/pass_rate",
      "grad_norm/model.layers.0.self_attn.q_proj.kernel",
      "grad_norm/model.layers.1.mlp.down_proj.kernel",
    ]);

    const icepop = filterMetricGroups(groups, "icepop");
    expect(icepop).toHaveLength(1);
    expect(icepop[0].keys).toEqual(["train/rl/icepop_ratio_mean"]);

    const byGroupLabel = filterMetricGroups(groups, "gradient q_proj");
    expect(byGroupLabel).toHaveLength(1);
    expect(byGroupLabel[0].id).toBe("grad_norm");
    expect(byGroupLabel[0].keys).toEqual(["grad_norm/model.layers.0.self_attn.q_proj.kernel"]);
  });

  it("orders RL histograms ahead of per-layer parameter histograms", () => {
    const keys = [
      "parameters/model.embed_tokens.embedding",
      "parameters/model.layers.0.self_attn.q_proj",
      "rl/reward_dist",
      "rl/advantage_dist",
      "rl/icepop_ratio_dist",
    ];
    const groups = groupHistogramKeys(keys);
    expect(groups.map((g) => g.id)).toEqual(["rl", "parameters"]);
    expect(groups[0].defaultExpanded).toBe(true);
    expect(groups[1].defaultExpanded).toBe(false);
    expect(classifyHistogramKey("advantages").id).toBe("rl");
    expect(classifyMetricKey("train/grad_norm/layer0").id).toBe("grad_norm");
  });

  it("identifies terminal run statuses and splits metric keys cleanly", () => {
    for (const status of ["finished", "failed", "crashed", "cancelled", "killed", "aborted"]) {
      expect(isTerminalRunStatus(status)).toBe(true);
    }
    for (const status of ["running", "unknown", "", null, undefined]) {
      expect(isTerminalRunStatus(status)).toBe(false);
    }
    expect(splitMetricKey("train/rl/icepop_ratio_mean")).toEqual({
      prefix: "train/rl/",
      leaf: "icepop_ratio_mean",
    });
    expect(splitMetricKey("loss")).toEqual({ prefix: "", leaf: "loss" });
  });
});
