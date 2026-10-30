/* api/client.ts: changes carry the Forge session from the same-origin bootstrap; a server restart (old session
 * refused) fetches the new one once; reads carry none. */
import { afterEach, describe, expect, it, vi } from "vitest";
import { api, resetSession } from "./client";

afterEach(() => {
  vi.unstubAllGlobals();
  resetSession();
});

function stub(tokens: string[], accept: string) {
  const seen: { url: string; method: string; session: string | null }[] = [];
  vi.stubGlobal("fetch", async (input: string, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    seen.push({ url: input, method: (init?.method ?? "GET").toUpperCase(), session: headers.get("X-Forge-Session") });
    if (input === "/api/session") return new Response(JSON.stringify({ token: tokens.shift() }), { status: 200 });
    if ((init?.method ?? "GET") !== "GET" && headers.get("X-Forge-Session") !== accept)
      return new Response(JSON.stringify({ detail: "this request's Forge session is not the running server's" }), { status: 403 });
    return new Response(JSON.stringify({ row: {}, counts: {} }), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  return seen;
}

describe("Forge session", () => {
  it("sends the session on changes, not on reads, and fetches it once", async () => {
    const seen = stub(["s1"], "s1");
    await api.labelRow("h:x", "r1", { label: "a" });
    await api.labelRow("h:x", "r2", { label: "b" });
    await api.labelsets();
    expect(seen.filter((c) => c.url === "/api/session")).toHaveLength(1);
    expect(seen.filter((c) => c.method === "POST").map((c) => c.session)).toEqual(["s1", "s1"]);
    expect(seen.find((c) => c.url === "/api/labelsets")?.session).toBeNull();
  });

  it("after a server restart takes the new session once and retries", async () => {
    const seen = stub(["old", "new"], "new");
    await api.labelRow("h:x", "r1", { label: "a" });
    expect(seen.filter((c) => c.method === "POST").map((c) => c.session)).toEqual(["old", "new"]);
  });
});
