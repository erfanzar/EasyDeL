// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

export type Section =
  | "overview"
  | "workloads"
  | "runs"
  | "experiments"
  | "research"
  | "fleet"
  | "capacity"
  | "billing"
  | "budgets"
  | "audit"
  | "assistant"
  | "settings";

export type RunTab = "charts" | "episodes" | "histograms" | "rollouts" | "completions" | "media";

export type Route =
  | { section: Section }
  | { section: "workloads"; workloadId: string }
  | { section: "runs"; compareIds: string[] }
  | { section: "runs"; runId: string; tab?: RunTab }
  | { section: "experiments"; experimentId: string }
  | { section: "research"; planId: string };

export type SectionAvailability = Partial<Record<Section, boolean>>;

/** Keep a forbidden deep link from mounting panels that issue fleet-only reads. */
export function accessibleRoute(route: Route, availability: SectionAvailability): Route {
  return availability[route.section] === false ? { section: "overview" } : route;
}

export function parseRoute(hash = globalThis.location?.hash ?? ""): Route {
  // A query after the route (`#/settings?assistant=...`) is a notice for the
  // page, never part of the route itself, except `#/runs/compare?ids=...`.
  const raw = hash.replace(/^#\/?/, "");
  const queryIdx = raw.indexOf("?");
  const path = queryIdx === -1 ? raw : raw.slice(0, queryIdx);
  const query = queryIdx === -1 ? "" : raw.slice(queryIdx + 1);
  const [section, id, tab] = path.split("/");
  if (section === "runs" && id === "compare") {
    const params = new URLSearchParams(query);
    const idsParam = params.get("ids") ?? "";
    const compareIds = [
      ...new Set(
        idsParam
          .split(",")
          .map((item) => {
            const trimmed = item.trim();
            if (!trimmed) return "";
            try {
              return decodeURIComponent(trimmed);
            } catch {
              return trimmed;
            }
          })
          .filter(Boolean),
      ),
    ];
    return { section: "runs", compareIds };
  }
  if (section === "experiments" && id) {
    try {
      return { section: "experiments", experimentId: decodeURIComponent(id) };
    } catch (error) {
      if (!(error instanceof URIError)) throw error;
      return { section: "experiments" };
    }
  }
  if (section === "research" && id) {
    try {
      return { section: "research", planId: decodeURIComponent(id) };
    } catch (error) {
      if (!(error instanceof URIError)) throw error;
      return { section: "research" };
    }
  }
  if ((section === "workloads" || section === "runs") && id) {
    try {
      return section === "runs"
        ? {
            section: "runs",
            runId: decodeURIComponent(id),
            ...(tab === "episodes" ||
            tab === "histograms" ||
            tab === "rollouts" ||
            tab === "completions" ||
            tab === "media"
              ? { tab }
              : {}),
          }
        : { section: "workloads", workloadId: decodeURIComponent(id) };
    } catch (error) {
      if (!(error instanceof URIError)) throw error;
      // Browser hashes may come from malformed external links, not routeHash.
      // Keep the list usable without requesting an invalid workload detail.
      return { section };
    }
  }
  // Every section the rail offers, or the rail has an item that navigates to
  // the overview. `billing` was missing here while it had a nav entry, a page,
  // and an availability flag -- so pressing Billing set the hash, the hash
  // parsed back to overview, and the section could only be reached by not
  // reloading.
  if (
    [
      "workloads",
      "runs",
      "experiments",
      "research",
      "fleet",
      "capacity",
      "billing",
      "budgets",
      "audit",
      "assistant",
      "settings",
    ].includes(section)
  ) {
    return { section: section as Section };
  }
  // Team accounts first shipped as their own `#/users` section; keep those links working.
  if (section === "users") return { section: "settings" };
  return { section: "overview" };
}

export function routeHash(route: Route): string {
  if ("compareIds" in route) {
    const ids = route.compareIds.map((id) => encodeURIComponent(id)).join(",");
    return ids ? `#/runs/compare?ids=${ids}` : "#/runs/compare";
  }
  if ("runId" in route) {
    const suffix = route.tab && route.tab !== "charts" ? `/${route.tab}` : "";
    return `#/runs/${encodeURIComponent(route.runId)}${suffix}`;
  }
  if ("experimentId" in route) {
    return `#/experiments/${encodeURIComponent(route.experimentId)}`;
  }
  if ("workloadId" in route) {
    return `#/workloads/${encodeURIComponent(route.workloadId)}`;
  }
  if ("planId" in route) {
    return `#/research/${encodeURIComponent(route.planId)}`;
  }
  return `#/${route.section}`;
}

export function navigate(route: Route): void {
  globalThis.location.hash = routeHash(route);
}
