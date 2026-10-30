import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter } from "react-router-dom";
import { NewProjectDialog } from "./NewProjectDialog";
import { ENCODERS, EXPERIMENT_RECOMMENDATION } from "../testing/fixtures";
import { stubFetch } from "../testing/stubFetch";

// jsdom has no <dialog> modal API
HTMLDialogElement.prototype.showModal = function () {
  this.open = true;
};
HTMLDialogElement.prototype.close = function () {
  this.open = false;
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("NewProjectDialog encoder", () => {
  it("preselects Forge's recommendation, says why, and sends the picked key (required)", async () => {
    const calls = stubFetch((c) => {
      if (c.url.pathname === "/api/settings") return [200, { provider: "ollama", model: "qwen3:8b", ollama_url: "", max_attempts: 3 }];
      if (c.url.pathname === "/api/providers")
        return [200, [{ id: "ollama", label: "Ollama", available: true, detail: "", models: [{ id: "qwen3:8b", label: "qwen3:8b" }] }]];
      if (c.url.pathname === "/api/encoders/recommendation") return [200, EXPERIMENT_RECOMMENDATION];
      if (c.url.pathname === "/api/encoders") return [200, { encoders: ENCODERS.map((e) => ({ ...e, used_by: { projects: [], targets: [] } })) }];
      if (c.url.pathname === "/api/projects" && c.method === "POST") return [201, { id: "p9", name: "x", state: "analyzing" }];
      return undefined;
    });
    const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={qc}>
        <MemoryRouter>
          <NewProjectDialog open onClose={() => {}} />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    await waitFor(() => expect((screen.getByLabelText("Encoder") as HTMLSelectElement).value).toBe("qwen3-4b-gguf-q4km"));
    expect(screen.getByText(/Recommended: Qwen3-4B \(GGUF Q4_K_M\): 2560-d is wide enough/)).toBeTruthy();
    fireEvent.click(screen.getByText(/Describe a task/));
    fireEvent.change(screen.getByPlaceholderText(/Classify inbound support tickets/), { target: { value: "Route tickets to billing, tech or account" } });
    fireEvent.change(screen.getByLabelText("Encoder"), { target: { value: "qwen3-4b" } });
    expect(screen.getByText(/Forge recommends Qwen3-4B \(GGUF Q4_K_M\)/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Start analysis" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST" && c.url.pathname === "/api/projects")).toBe(true));
    const post = calls.find((c) => c.method === "POST" && c.url.pathname === "/api/projects")!;
    expect((post.body as { encoder: string }).encoder).toBe("qwen3-4b");
  });
});
