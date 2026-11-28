// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { SubmitWorkloadModal } from "./SubmitWorkloadModal";
import type { AvailableShape, WorkloadDetail } from "../types";

const SHAPES: AvailableShape[] = [
  {
    fleet: "research",
    pool: "research-v6e-8-us-east5-a",
    shape: "v6e-8",
    capacityClass: "spot",
    region: "us-east5",
    zone: "us-east5-a",
  },
];

const onSubmit = async (): Promise<WorkloadDetail> => {
  throw new Error("This focus test must not submit a workload.");
};

afterEach(cleanup);

describe("submit dialog focus ownership", () => {
  it("keeps the caret and draft through a parent callback refresh, using the latest Escape handler", async () => {
    const previous = document.createElement("button");
    document.body.append(previous);
    previous.focus();
    const firstClose = vi.fn();
    const latestClose = vi.fn();
    const page = render(
      <SubmitWorkloadModal open project="research" shapes={SHAPES} onClose={firstClose} onSubmit={onSubmit} />,
    );
    try {
      const shape = screen.getByRole("combobox", { name: "Shape" });
      await waitFor(() => expect(document.activeElement).toBe(shape));
      const command = screen.getByRole("textbox", { name: "Command" }) as HTMLTextAreaElement;
      fireEvent.change(command, { target: { value: "python train.py" } });
      command.focus();
      command.setSelectionRange(7, 12);

      page.rerender(
        <SubmitWorkloadModal open project="research" shapes={[...SHAPES]} onClose={latestClose} onSubmit={onSubmit} />,
      );
      expect(document.activeElement).toBe(command);
      expect(command.value).toBe("python train.py");
      expect([command.selectionStart, command.selectionEnd]).toEqual([7, 12]);
      expect(document.body.classList.contains("modal-open")).toBe(true);

      fireEvent.keyDown(window, { key: "Escape" });
      expect(firstClose).not.toHaveBeenCalled();
      expect(latestClose).toHaveBeenCalledTimes(1);
      page.rerender(
        <SubmitWorkloadModal
          open={false}
          project="research"
          shapes={SHAPES}
          onClose={latestClose}
          onSubmit={onSubmit}
        />,
      );
      expect(document.activeElement).toBe(previous);
      expect(document.body.classList.contains("modal-open")).toBe(false);
      fireEvent.keyDown(window, { key: "Escape" });
      expect(latestClose).toHaveBeenCalledTimes(1);
    } finally {
      page.unmount();
      previous.remove();
    }
  });

  it("traps Tab after callback refresh and restores focus only on unmount", async () => {
    const previous = document.createElement("button");
    document.body.append(previous);
    previous.focus();
    const page = render(
      <SubmitWorkloadModal open project="research" shapes={SHAPES} onClose={() => {}} onSubmit={onSubmit} />,
    );
    try {
      await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("combobox", { name: "Shape" })));
      page.rerender(
        <SubmitWorkloadModal open project="research" shapes={[...SHAPES]} onClose={() => {}} onSubmit={onSubmit} />,
      );
      const first = screen.getByRole("button", { name: "Close dialog" });
      first.focus();
      const backwards = new KeyboardEvent("keydown", { key: "Tab", shiftKey: true, cancelable: true });
      window.dispatchEvent(backwards);
      expect(backwards.defaultPrevented).toBe(true);
      const last = document.activeElement;
      expect(last).not.toBe(first);
      const forwards = new KeyboardEvent("keydown", { key: "Tab", cancelable: true });
      window.dispatchEvent(forwards);
      expect(forwards.defaultPrevented).toBe(true);
      expect(document.activeElement).toBe(first);
      page.unmount();
      expect(document.activeElement).toBe(previous);
    } finally {
      page.unmount();
      previous.remove();
    }
  });
});
