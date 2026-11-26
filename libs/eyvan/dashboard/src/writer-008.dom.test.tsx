// Copyright (C) 2026 Erfan Zare Chavoshi
// This file is part of Eyvan.
//
// SPDX-License-Identifier: AGPL-3.0-or-later

// @vitest-environment jsdom

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";

import { SubmitWorkloadModal } from "./components/SubmitWorkloadModal";
import type { AvailableShape, SubmitWorkloadInput } from "./types";

afterEach(cleanup);

const shape: AvailableShape = {
  fleet: "research",
  pool: "research-cpu-16",
  shape: "cpu-16",
  capacityClass: "on_demand",
  region: "us-east5",
  zone: "us-east5-a",
};

async function renderForm() {
  const submitted: SubmitWorkloadInput[] = [];
  render(
    <SubmitWorkloadModal
      open
      project="research"
      shapes={[shape]}
      onClose={() => {}}
      onSubmit={async (input) => {
        submitted.push(input);
        return { id: "wl-1" } as never;
      }}
    />,
  );
  await waitFor(() => expect((screen.getByLabelText("Shape") as HTMLSelectElement).value).toBe("research/cpu-16"));
  fireEvent.change(screen.getByLabelText("Command"), { target: { value: "python train.py" } });
  return submitted;
}

const run = () => fireEvent.click(screen.getByRole("button", { name: "Run" }));

describe("submission pair values", () => {
  it("preserves everything after the first equals sign in environment and labels", async () => {
    const submitted = await renderForm();
    fireEvent.change(screen.getByLabelText("Environment"), {
      target: { value: " PADDED=abc==, ARGS=--name=trial, EQUALS== " },
    });
    fireEvent.change(screen.getByLabelText("Labels"), { target: { value: "config=a=b, team=research" } });
    run();

    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0].environment).toEqual({ PADDED: "abc==", ARGS: "--name=trial", EQUALS: "=" });
    expect(submitted[0].labels).toEqual({ config: "a=b", team: "research" });
  });

  it.each(["Environment", "Labels"])("keeps an entirely blank %s map optional", async (field) => {
    const submitted = await renderForm();
    fireEvent.change(screen.getByLabelText(field), { target: { value: "   " } });
    run();
    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0].environment).toEqual({});
    expect(submitted[0].labels).toEqual({});
  });

  it.each(
    ["Environment", "Labels"].flatMap((field) =>
      ["VALID=value,MISSING", "VALID=value,EMPTY=", "VALID=value,EMPTY=   ", "VALID=value,=hidden", "VALID=value,"].map(
        (value) => [field, value],
      ),
    ),
  )("visibly refuses malformed %s pairs (%s), then accepts a correction", async (field, value) => {
    const submitted = await renderForm();
    fireEvent.change(screen.getByLabelText(field), { target: { value } });
    run();

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain(
      `${field} must contain comma-separated key=value pairs with nonempty keys and values.`,
    );
    expect(alert.textContent).not.toContain(value);
    expect(submitted).toHaveLength(0);
    expect((screen.getByLabelText(field) as HTMLInputElement).value).toBe(value);

    fireEvent.change(screen.getByLabelText(field), { target: { value: "VALID=value=tail" } });
    run();
    await waitFor(() => expect(submitted).toHaveLength(1));
    expect(submitted[0][field === "Environment" ? "environment" : "labels"]).toEqual({ VALID: "value=tail" });
  });
});
