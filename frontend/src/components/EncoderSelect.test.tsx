import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { EncoderSelect, encoderLabel } from "./EncoderSelect";
import { ENCODERS } from "../testing/fixtures";
import type { EncodersView } from "../api/types";
import { stubFetch } from "../testing/stubFetch";

const VIEW: EncodersView = {
  encoders: ENCODERS.map((e) =>
    e.key === "qwen3-0.6b-bf16" ? { ...e, used_by: { projects: [{ id: "p1", name: "Tickets" }], targets: [{ id: "t1", name: "Ticket heads" }] } } : e,
  ),
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

function renderSelect(props: Partial<Parameters<typeof EncoderSelect>[0]> = {}) {
  stubFetch((c) => (c.url.pathname === "/api/encoders" ? [200, VIEW] : undefined));
  const onChange = vi.fn();
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={qc}>
      <EncoderSelect aria-label="Encoder" value="qwen3-0.6b-bf16" onChange={onChange} {...props} />
    </QueryClientProvider>,
  );
  return onChange;
}

describe("EncoderSelect", () => {
  it("shows name, backend, quant, size, download state and users per encoder", async () => {
    renderSelect();
    await waitFor(() => expect(screen.getAllByRole("option").length).toBe(ENCODERS.length));
    const texts = screen.getAllByRole("option").map((o) => o.textContent);
    expect(texts).toContain("Qwen3-0.6B (MLX bf16) · mlx · bf16 · ~3.0 GB · Downloaded · used by 2");
    expect(texts).toContain("Example-12B · llamacpp · Q8_0 · size not measured · Not downloaded");
    expect((screen.getByLabelText("Encoder") as HTMLSelectElement).value).toBe("qwen3-0.6b-bf16");
  });

  it("reports the picked key and disables what is not allowed", async () => {
    const onChange = renderSelect({ allowed: ["qwen3-0.6b", "qwen3-0.6b-bf16"], disabledReason: "MLX only" });
    await waitFor(() => expect(screen.getAllByRole("option").length).toBe(ENCODERS.length));
    const gguf = screen.getAllByRole("option").find((o) => o.textContent?.startsWith("Qwen3-4B (GGUF")) as HTMLOptionElement;
    expect(gguf.disabled).toBe(true);
    expect(gguf.textContent).toContain("(MLX only)");
    fireEvent.change(screen.getByLabelText("Encoder"), { target: { value: "qwen3-0.6b" } });
    expect(onChange).toHaveBeenCalledWith("qwen3-0.6b");
  });

  it("labels one encoder in words", () => {
    const e = { ...ENCODERS[0], cached: false, used_by: { projects: [], targets: [] } };
    expect(encoderLabel(e)).toBe("Qwen3-4B (MLX 4-bit) · mlx · 4-bit · ~4.2 GB · Not downloaded");
  });
});
