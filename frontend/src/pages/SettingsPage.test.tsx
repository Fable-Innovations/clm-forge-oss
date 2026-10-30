import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { SettingsPage } from "./SettingsPage";
import type { Settings } from "../api/types";
import { stubFetch } from "../testing/stubFetch";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const SETTINGS: Settings = {
  provider: "ollama",
  model: "qwen3:8b",
  ollama_url: "http://127.0.0.1:11434",
  max_attempts: 3,
};

describe("SettingsPage", () => {
  it("has no app-wide encoder: no encoder section, and a save sends no encoder field", async () => {
    const calls = stubFetch((c) => {
      if (c.url.pathname === "/api/settings" && c.method === "PUT") return [200, { ...SETTINGS, ...(c.body as object) }];
      if (c.url.pathname === "/api/settings") return [200, SETTINGS];
      if (c.url.pathname === "/api/health") return [200, { ok: true, encoders: [], providers: { claude: false, ollama: true, ollama_models: [] } }];
      if (c.url.pathname === "/api/providers") return [200, []];
      return undefined;
    });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter>
          <SettingsPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await waitFor(() => expect(screen.getByRole("button", { name: "Save settings" })).toBeTruthy());
    expect(screen.queryByLabelText("Default encoder")).toBeNull();
    expect(screen.queryByText("Encoder")).toBeNull();
    expect(screen.getByText(/no app-wide encoder/)).toBeTruthy();
    fireEvent.change(screen.getByDisplayValue("3"), { target: { value: "4" } });
    fireEvent.click(screen.getByRole("button", { name: "Save settings" }));
    await waitFor(() => expect(calls.some((c) => c.method === "PUT")).toBe(true));
    const body = calls.find((c) => c.method === "PUT")!.body as Record<string, unknown>;
    expect(body.max_attempts).toBe(4);
    expect(Object.keys(body).filter((k) => /encoder|base_models/.test(k))).toEqual([]);
  });
});
