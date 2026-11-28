// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

/** Colorblind-safe, high-contrast palette tuned for dark and light console surfaces. */
export const COMPARE_RUN_PALETTE_DARK = [
  "#fbbf24",
  "#7aa2f7",
  "#9ece6a",
  "#f7768e",
  "#bb9af7",
  "#2ac3de",
  "#ff9e64",
  "#73daca",
];

export const COMPARE_RUN_PALETTE_LIGHT = [
  "#b45309",
  "#0969da",
  "#116329",
  "#cf222e",
  "#8250df",
  "#0e7490",
  "#c2410c",
  "#047857",
];

export function compareRunColor(index: number, theme: "dark" | "light" = "dark"): string {
  const palette = theme === "light" ? COMPARE_RUN_PALETTE_LIGHT : COMPARE_RUN_PALETTE_DARK;
  return palette[index % palette.length];
}
