// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AvailableShape, SubmitWorkloadInput } from "../types";
import { SubmitWorkloadModal } from "./SubmitWorkloadModal";

afterEach(cleanup);

const v6e = (zone: string): AvailableShape => ({
  fleet: "research",
  pool: `research-v6e-8-${zone}`,
  shape: "v6e-8",
  capacityClass: "spot",
  region: "us-east5",
  zone,
});

const cpu: AvailableShape = {
  fleet: "research",
  pool: "research-cpu-16",
  shape: "cpu-16",
  capacityClass: "on_demand",
  region: "us-east5",
  zone: "us-east5-a",
};

function renderForm(options: { shapes?: AvailableShape[]; loading?: boolean; error?: string | null } = {}) {
  const submitted: SubmitWorkloadInput[] = [];
  const onSubmit = vi.fn(async (input: SubmitWorkloadInput) => {
    submitted.push(input);
    return { id: "wl-1" } as never;
  });
  render(
    <SubmitWorkloadModal
      open
      project="research"
      shapes={options.shapes ?? [v6e("us-east5-a"), v6e("us-east5-b"), cpu]}
      loadingShapes={options.loading ?? false}
      shapesError={options.error ?? null}
      onClose={() => {}}
      onSubmit={onSubmit}
    />,
  );
  return { submitted, onSubmit };
}

const dialog = () => screen.getByRole("dialog");
const field = (name: string) => within(dialog()).getByLabelText(name) as HTMLInputElement & HTMLSelectElement;
const run = () => fireEvent.click(within(dialog()).getByRole("button", { name: "Run" }));

describe("choosing what to run on", () => {
  it("offers the shapes the fleet has, not a list of hardware", async () => {
    renderForm();

    const options = within(field("Shape"))
      .getAllByRole("option")
      .map((option) => option.textContent);

    // Two zones of one shape are one choice: they are alternates, and picking
    // between them is the scheduler's job.
    expect(options).toEqual(["Choose hardware", "research/cpu-16 · 1 pool", "research/v6e-8 · 2 pools"]);
  });

  it("preselects the only thing on offer", async () => {
    renderForm({ shapes: [v6e("us-east5-a")] });

    await waitFor(() => expect(field("Shape").value).toBe("research/v6e-8"));
  });

  it("offers the chosen shape's zones as a constraint, defaulting to unpinned", async () => {
    // Pinning withholds the cross-zone alternates, which is the entire reason
    // a shape exists in more than one zone.
    renderForm();

    fireEvent.change(field("Shape"), { target: { value: "research/v6e-8" } });

    expect(field("Zone").value).toBe("");
    const zones = within(field("Zone"))
      .getAllByRole("option")
      .map((option) => option.textContent);
    expect(zones).toEqual(["Wherever there is capacity", "Pin to us-east5-a", "Pin to us-east5-b"]);
  });

  it("drops a zone that belonged to the previous shape", () => {
    // A leftover pin is a constraint nothing can satisfy, and the refusal would
    // name a zone the submitter never knowingly chose.
    renderForm({ shapes: [v6e("us-east5-a"), { ...cpu, zone: "us-central2-b" }] });

    fireEvent.change(field("Shape"), { target: { value: "research/v6e-8" } });
    fireEvent.change(field("Zone"), { target: { value: "us-east5-a" } });
    fireEvent.change(field("Shape"), { target: { value: "research/cpu-16" } });

    expect(field("Zone").value).toBe("");
  });

  it("says nothing is applied only when the read succeeded and found nothing", async () => {
    renderForm({ error: "Eyvan API request failed with 503" });

    const alerts = within(dialog())
      .getAllByRole("alert")
      .map((alert) => alert.textContent);
    expect(alerts.join(" ")).toContain("503");
    // An empty fleet and a failed read look identical on screen unless one of
    // them is named, and only one of them is fixed by running `eyvan up`.
    expect(alerts.join(" ")).not.toContain("No fleet is applied");
  });

  it("does not report a fleet still being read as an empty one", () => {
    renderForm({ shapes: [], loading: true });

    expect(within(dialog()).queryByText(/No fleet is applied/)).toBeNull();
  });
});

describe("what the submission carries", () => {
  it("sends the shape and pod count, and nothing the pool owns", async () => {
    const { submitted } = renderForm();

    fireEvent.change(field("Shape"), { target: { value: "research/v6e-8" } });
    fireEvent.change(field("Pods"), { target: { value: "2" } });
    fireEvent.change(field("Command"), { target: { value: "python train.py" } });
    fireEvent.change(field("Name"), { target: { value: "sft-run" } });
    run();

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0]).toEqual({
      fleet: "research",
      shape: "v6e-8",
      pods: 2,
      zone: undefined,
      name: "sft-run",
      priority: "interactive",
      command: "python train.py",
      estimatedDurationSeconds: 28_800,
      checkpointable: false,
      checkpointHook: undefined,
      checkpointUri: undefined,
      environment: {},
      labels: {},
    });
    // The four the pool owns. A browser that names any of them is guessing, and
    // two of the guesses used to make the job unplaceable.
    expect(Object.keys(submitted[0])).not.toContain("provider");
    expect(Object.keys(submitted[0])).not.toContain("resources");
  });

  it("carries a pinned zone through as the constraint it is", async () => {
    const { submitted } = renderForm();

    fireEvent.change(field("Shape"), { target: { value: "research/v6e-8" } });
    fireEvent.change(field("Zone"), { target: { value: "us-east5-b" } });
    fireEvent.change(field("Command"), { target: { value: "python train.py" } });
    run();

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0].zone).toBe("us-east5-b");
  });

  it("refuses a checkpointable job with nowhere to write", () => {
    // Both, or neither has any effect: the scheduler checkpoints on preemption
    // only when the job says it can and says how, so half of this pair is a
    // spot job that silently restarts from zero -- which looks like a job that
    // is simply slow.
    const { onSubmit } = renderForm();

    fireEvent.change(field("Shape"), { target: { value: "research/v6e-8" } });
    fireEvent.change(field("Command"), { target: { value: "python train.py" } });
    fireEvent.click(within(dialog()).getByRole("checkbox"));
    fireEvent.change(field("Checkpoint hook"), { target: { value: "python checkpoint.py" } });
    run();

    expect(onSubmit).not.toHaveBeenCalled();
    expect(field("Checkpoint output").checkValidity()).toBe(false);
  });

  it("cannot be submitted against a fleet that has nothing", () => {
    renderForm({ shapes: [] });

    expect(within(dialog()).getByRole("button", { name: "Run" }).hasAttribute("disabled")).toBe(true);
  });
});
